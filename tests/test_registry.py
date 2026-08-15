from pathlib import Path

from vulkan_wrapper_gen.analysis import analyze
from vulkan_wrapper_gen.config import GeneratorConfig, ReceiverOverride
from vulkan_wrapper_gen.registry import parse_registries


FIXTURE = Path(__file__).parent / "fixtures" / "mini_vk.xml"


def test_registry_retains_relationships_and_availability():
    registry = parse_registries([FIXTURE])
    assert registry.types["VkBuffer"].parent == "VkDevice"
    assert registry.types["VkExtraInfo"].struct_extends == ("VkBufferCreateInfo",)
    assert registry.commands["vkUseBufferEXT"].params[1].externsync
    assert registry.commands["vkEnumerateBuffersEXT"].success_codes == ("VK_SUCCESS", "VK_INCOMPLETE")
    assert registry.commands["vkEnumerateBuffersEXT"].availability.extensions == ("VK_EXT_test",)
    assert registry.platforms["win32"] == "VK_USE_PLATFORM_WIN32_KHR"
    assert registry.constants["VK_MAX_TEST"].value == "16U"


def test_analysis_infers_non_templated_create_info_and_receivers():
    registry = parse_registries([FIXTURE])
    config = GeneratorConfig(receivers={
        "vkEnumerateBuffersEXT": ReceiverOverride(add=("VkBuffer",), rename="enumerateRelatedEXT")
    })
    result = analyze(registry, config)
    assert result.handles["VkBuffer"].create_info == "VkBufferCreateInfo"
    assert result.handles["VkFence"].create_info is None
    enumerate_command = next(item for item in result.commands if item.command.name == "vkEnumerateBuffersEXT")
    assert enumerate_command.receivers == ("VkDevice", "VkBuffer")
    assert enumerate_command.cpp_name == "enumerateRelatedEXT"
    assert enumerate_command.output.count.name == "pBufferCount"
    assert enumerate_command.output.vector.name == "pBuffers"
    assert enumerate_command.output.status_value
    queue_submit = next(item for item in result.commands if item.command.name == "vkQueueSubmit")
    assert queue_submit.receivers == ("VkQueue",)
    graphics = next(item for item in result.commands if item.command.name == "vkCreateGraphicsPipelines")
    assert graphics.receivers == ("VkDevice",)


def test_extension_filters_remove_excluded_requirements():
    registry = parse_registries([FIXTURE], exclude_extensions=("VK_EXT_test",))
    assert "vkCreateBuffer" in registry.commands
    assert "vkEnumerateBuffersEXT" not in registry.commands
    assert "vkUseBufferEXT" not in registry.commands


def test_supplemental_registry_replaces_forward_type_with_complete_struct(tmp_path):
    primary = tmp_path / "primary.xml"
    primary.write_text("""<registry><types><type name="StdThing" requires="external.h"/></types></registry>""", encoding="utf-8")
    supplemental = tmp_path / "supplemental.xml"
    supplemental.write_text("""<registry><types><type category="struct" name="StdThing"><member><type>uint32_t</type><name>value</name></member></type></types></registry>""", encoding="utf-8")
    registry = parse_registries([primary, supplemental])
    assert registry.types["StdThing"].category == "struct"
    assert [member.name for member in registry.types["StdThing"].members] == ["value"]
