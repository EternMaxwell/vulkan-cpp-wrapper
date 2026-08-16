"""Tests for the middle-layer IR built from registry XML."""

from __future__ import annotations

import json
from pathlib import Path

from vulkan_wrapper_gen.ir import IrRegistry, build_ir
from vulkan_wrapper_gen.ir.builder import _receiver_member_name
from vulkan_wrapper_gen.config import GeneratorConfig

FIXTURE = Path(__file__).parent / "fixtures" / "mini_vk.xml"


def _build() -> IrRegistry:
    return build_ir([FIXTURE])


def test_processed_type_names_versus_c_type_names():
    registry = _build()
    submits = registry.commands["queueSubmit"].param("pSubmits")
    assert submits.type == "SubmitInfo"
    assert submits.c_type == "VkSubmitInfo"
    assert submits.c_declaration == "const VkSubmitInfo* pSubmits"
    # Reconstruction from processed fields reproduces the exact C declarator.
    assert submits.c_signature_piece == "const VkSubmitInfo* pSubmits"
    geometries = registry.structs["AccelerationStructureBuildGeometryInfoKHR"].member("ppGeometries")
    assert geometries.type == "AccelerationStructureGeometryKHR"
    assert geometries.c_type == "VkAccelerationStructureGeometryKHR"
    # Inner pointer-level constness is preserved, not a stored declaration.
    # `const T* const*` -> const pointee, const inner pointer.
    assert geometries.const
    assert geometries.pointer_consts == (True, False)
    assert geometries.c_declaration == "const VkAccelerationStructureGeometryKHR* const* ppGeometries"
    assert registry.commands["createBuffer"].c_signature == (
        "VkResult vkCreateBuffer(VkDevice device, const VkBufferCreateInfo* pCreateInfo, "
        "const VkAllocationCallbacks* pAllocator, VkBuffer* pBuffer)"
    )


def test_entities_are_keyed_by_general_name_with_c_name():
    registry = _build()
    # Handles: general name keys, C name kept alongside.
    assert registry.handles["Buffer"].name == "Buffer"
    assert registry.handles["Buffer"].c_name == "VkBuffer"
    assert "VkBuffer" not in registry.handles
    # Structs and enums behave the same way.
    assert registry.structs["VideoProfileListInfoKHR"].c_name == "VkVideoProfileListInfoKHR"
    assert registry.enums["Result"].c_name == "VkResult"
    assert registry.enums["Result"].value("VK_SUCCESS") is not None  # values keep C spelling
    # Commands: vk prefix dropped and decapitalized; return type is split
    # into processed + C spellings like a parameter type.
    assert registry.commands["createBuffer"].name == "createBuffer"
    assert registry.commands["createBuffer"].c_name == "vkCreateBuffer"
    assert registry.commands["createBuffer"].return_type == "Result"
    assert registry.commands["createBuffer"].c_return_type == "VkResult"
    assert "vkCreateBuffer" not in registry.commands
    # Bitmask references use general names; the C typedef still reconstructs.
    assert registry.bitmasks["BufferUsageFlags"].c_name == "VkBufferUsageFlags"
    assert registry.bitmasks["BufferUsageFlags"].base == "Flags"
    assert registry.bitmasks["BufferUsageFlags"].bits == "BufferUsageFlagBits"
    assert registry.bitmasks["BufferUsageFlags"].c_declaration == "typedef VkFlags VkBufferUsageFlags;"
    assert registry.basetypes["Flags"].c_name == "VkFlags"
    assert registry.basetypes["Flags"].c_declaration == "typedef uint32_t VkFlags;"
    # C declarations keep exact C names.
    assert registry.handles["Buffer"].c_declaration == "VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkBuffer)"
    assert registry.structs["VideoProfileListInfoKHR"].c_declaration.startswith(
        "typedef struct VkVideoProfileListInfoKHR {"
    )


def test_docs_from_comments_and_requirement_blocks():
    registry = _build()
    assert registry.enums["Result"].doc == "API result codes"
    code = registry.structs["ShaderModuleCreateInfo"].member("pCode")
    assert code.doc == "SPIR-V bytecode"
    assert registry.commands["createBuffer"].availability.doc == "Core 1.0 functionality"


def test_struct_array_and_count_relationships():
    registry = _build()
    profile_list = registry.structs["VideoProfileListInfoKHR"]
    profile_count = profile_list.member("profileCount")
    profiles = profile_list.member("pProfiles")
    assert profile_count is not None and profiles is not None
    assert profile_count.counts_for == ("pProfiles",)
    assert profiles.lengths[0].text == "profileCount"
    assert profiles.is_array
    assert profiles.public_name == "profiles"
    assert profiles.direction == "input"
    assert profile_count.direction == "input"


def test_latex_lengths_and_byte_arrays_are_preserved():
    registry = _build()
    shader = registry.structs["ShaderModuleCreateInfo"]
    code = shader.member("pCode")
    assert code is not None
    assert code.alt_length == "codeSize / 4"
    assert code.is_byte_array
    # LaTeX control words are filtered; the real identifier remains.
    assert "codeSize" in code.length_names
    assert "textrm" not in code.length_names
    assert code.lengths[0].latex is not None


def test_double_pointer_geometry_member_keeps_dimensions():
    registry = _build()
    geometry = registry.structs["AccelerationStructureBuildGeometryInfoKHR"]
    geometries = geometry.member("ppGeometries")
    assert geometries is not None
    assert geometries.pointer_depth == 2
    assert [length.text for length in geometries.lengths] == ["geometryCount", "1"]
    assert geometries.optional == ("true", "false")


def test_command_receivers_member_name_and_c_signature():
    registry = _build()
    submit = registry.commands["queueSubmit"]
    assert submit.dispatch == "Queue"
    # Optional scalar handles (the fence) are synchronization arguments, not
    # receivers, unless configuration explicitly adds them.
    assert submit.receivers == ("Queue",)
    assert submit.member_name == "submit"
    assert submit.c_signature == (
        "VkResult vkQueueSubmit(VkQueue queue, uint32_t submitCount, "
        "const VkSubmitInfo* pSubmits, VkFence fence)"
    )
    submits = submit.param("pSubmits")
    count = submit.param("submitCount")
    assert submits is not None and count is not None
    assert submits.is_array and submits.direction == "input"
    assert count.counts_for == ("pSubmits",)
    assert submit.outputs == ()

    wait_idle = registry.commands["deviceWaitIdle"]
    assert wait_idle.receivers == ("Device",)
    assert wait_idle.member_name == "waitIdle"

    create = registry.commands["createBuffer"]
    assert create.c_signature == (
        "VkResult vkCreateBuffer(VkDevice device, const VkBufferCreateInfo* pCreateInfo, "
        "const VkAllocationCallbacks* pAllocator, VkBuffer* pBuffer)"
    )
    buffer = create.param("pBuffer")
    assert buffer is not None and buffer.direction == "output"
    assert create.outputs == ("pBuffer",)
    assert create.owned_outputs == ("pBuffer",)

    pipelines = registry.commands["createGraphicsPipelines"]
    assert pipelines.param("pipelineCache").is_optional
    # The by-value count sizes spans; it is not the two-call enumeration shape.
    assert pipelines.count_param is None
    assert pipelines.outputs == ("pPipelines",)


def test_enumeration_shape_and_status_alternatives():
    registry = _build()
    enumerate_buffers = registry.commands["enumerateBuffersEXT"]
    assert enumerate_buffers.count_param == "pBufferCount"
    assert enumerate_buffers.vector_output == "pBuffers"
    assert enumerate_buffers.count_name == "bufferCount"
    assert enumerate_buffers.status_alternatives
    assert enumerate_buffers.owned_outputs == ()
    assert enumerate_buffers.dispatch == "Device"
    assert enumerate_buffers.receivers == ("Device",)


def test_handle_relationships_and_releasers():
    registry = _build()
    buffer = registry.handles["Buffer"]
    assert buffer.parent == "Device"
    assert not buffer.dispatchable
    assert buffer.object_type_enum == "VK_OBJECT_TYPE_BUFFER"
    assert buffer.create_infos == ("BufferCreateInfo",)
    assert buffer.create_info == "BufferCreateInfo"
    assert buffer.releaser == "destroyBuffer"
    instance = registry.handles["Instance"]
    assert instance.parents == () and instance.dispatchable
    pipeline = registry.handles["Pipeline"]
    # The fixture has a single pipeline producer, so the concrete record is
    # the create-info struct itself (full registries synthesize the variant).
    assert pipeline.create_info == "GraphicsPipelineCreateInfo"


def test_aliases_resolve_and_keep_raw_data():
    registry = _build()
    alias = registry.aliases["BufferEXT"]
    assert alias.target == "Buffer"
    assert alias.c_name == "VkBufferEXT"
    assert alias.resolved_category == "handle"
    assert registry.type_category("BufferEXT") == "handle"
    assert registry.resolve("BufferEXT") is registry.handles["Buffer"]


def test_docs_and_raw_attributes_are_kept():
    registry = _build()
    value = registry.enums["Result"].value("VK_ERROR_UNKNOWN")
    assert value is not None
    assert value.doc == "An unknown error has occurred"
    assert value.value == "-13"
    instance = registry.handles["Instance"]
    assert instance.c_declaration == "VK_DEFINE_HANDLE(VkInstance)"
    assert instance.dispatchable
    buffer = registry.handles["Buffer"]
    assert buffer.c_declaration == "VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkBuffer)"
    profile_count = registry.structs["VideoProfileListInfoKHR"].member("profileCount")
    assert profile_count.optional == ("true",)


def test_json_roundtrip_is_lossless():
    registry = _build()
    text = registry.to_json(indent=2)
    restored = IrRegistry.from_json(text)
    assert restored.to_dict() == registry.to_dict()
    # Sanity: the round-tripped IR still answers the same questions.
    assert restored.commands["queueSubmit"].member_name == "submit"
    assert restored.commands["createBuffer"].c_signature == registry.commands["createBuffer"].c_signature
    assert restored.handles["Buffer"].releaser == "destroyBuffer"


def test_cli_emits_ir_json(tmp_path):
    from vulkan_wrapper_gen.cli import run

    output = tmp_path / "ir.json"
    assert run(["--registry", str(FIXTURE), "--emit-ir", str(output)]) == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["api"] == "vulkan"
    assert "Buffer" in data["handles"]
    assert data["handles"]["Buffer"]["name"] == "Buffer"
    assert data["handles"]["Buffer"]["c_name"] == "VkBuffer"
    assert data["handles"]["Buffer"]["releaser"] == "destroyBuffer"
    assert data["commands"]["queueSubmit"]["receivers"] == ["Queue"]


def test_full_registry_builds_and_survives_roundtrip():
    import os

    headers = Path(os.environ.get("LOCALAPPDATA", "")) / "Temp" / "Vulkan-Headers" / "registry"
    if not (headers / "vk.xml").is_file():
        import pytest

        pytest.skip("real vk.xml not available locally")
    registry = build_ir([headers / "vk.xml", headers / "video.xml"])
    assert registry.handles
    assert registry.commands
    assert registry.commands["createGraphicsPipelines"].param("pipelineCache").is_optional
    assert registry.commands["queueSubmit"].member_name == "submit"
    # Full-registry general names reconstruct their C spellings exactly.
    assert registry.structs["DeviceGroupPresentCapabilitiesKHR"].c_name == "VkDeviceGroupPresentCapabilitiesKHR"
    assert registry.commands["createBuffer"].c_name == "vkCreateBuffer"
    assert registry.handles["Buffer"].c_name == "VkBuffer"
    restored = IrRegistry.from_json(registry.to_json())
    assert restored.to_dict() == registry.to_dict()


def test_receiver_member_name_strips_receiver_and_plural_s():
    config = GeneratorConfig()
    assert _receiver_member_name("getPipelineCacheData", "PipelineCache", config) == "getData"
    assert _receiver_member_name("getQueryPoolResults", "QueryPool", config) == "getResults"
    assert _receiver_member_name("getFenceStatus", "Fence", config) == "getStatus"
    assert _receiver_member_name("bindBufferMemory", "Buffer", config) == "bindMemory"
    assert _receiver_member_name("mergePipelineCaches", "PipelineCache", config) == "merge"
