from pathlib import Path

import pytest

from vulkan_wrapper_gen.emitter import _vma_resource_types, _vma_sections
from vulkan_wrapper_gen.vma import VmaError, VmaFunction, VmaModel, parse_vma_header


def vma_model(*names: str) -> VmaModel:
    return VmaModel(functions={name: VmaFunction(name, "void") for name in names})


def test_vma_header_is_parsed_with_libclang(tmp_path: Path):
    pytest.importorskip("clang.cindex")
    header = tmp_path / "vk_mem_alloc.h"
    header.write_text(
        """
typedef struct VmaAllocator_T* VmaAllocator;
typedef struct VmaAllocation_T* VmaAllocation;
typedef struct VmaAllocationInfo { unsigned memoryType; } VmaAllocationInfo;
int vmaMapMemory(VmaAllocator allocator, VmaAllocation allocation, void** data);
void vmaUnmapMemory(VmaAllocator allocator, VmaAllocation allocation);
""",
        encoding="utf-8",
    )
    model = parse_vma_header(header, selected=("vmaMapMemory", "vmaUnmapMemory"))
    assert tuple(model.functions) == ("vmaMapMemory", "vmaUnmapMemory")
    assert model.functions["vmaMapMemory"].parameters[-1].type == "void **"


def test_vma_reports_missing_selection(tmp_path: Path):
    pytest.importorskip("clang.cindex")
    header = tmp_path / "vk_mem_alloc.h"
    header.write_text("void vmaPresent(void);", encoding="utf-8")
    with pytest.raises(VmaError, match="vmaMissing"):
        parse_vma_header(header, selected=("vmaMissing",))


def test_vma_lifetimes_do_not_need_ownership_flags_or_double_cleanup():
    declarations, implementations = _vma_sections(vma_model(
        "vmaCreateAllocator", "vmaDestroyAllocator", "vmaAllocateMemory",
        "vmaFreeMemory", "vmaMapMemory", "vmaUnmapMemory",
        "vmaFlushAllocation", "vmaInvalidateAllocation", "vmaGetAllocationInfo",
        "vmaCreateBuffer", "vmaDestroyBuffer", "vmaCreateImage", "vmaDestroyImage",
    ))
    assert "bool owning" not in declarations
    assert "std::make_shared<State>(allocator, device, false)" not in declarations
    assert "return Allocator(allocator, device);" in declarations
    assert "std::shared_ptr<void> allocator_lifetime" in declarations
    assert "std::make_shared<Allocation::State>(lifetime(), raw(), value)" in implementations
    create_buffer = implementations[implementations.index("Result<Buffer> Allocator::createBuffer"):]
    create_buffer = create_buffer[:create_buffer.index("Result<Image> Allocator::createImage")]
    assert create_buffer.count("vmaDestroyBuffer") == 1
    assert "if (!wrapped) vmaDestroyBuffer" not in create_buffer
    assert "const BufferCreateInfo& bufferInfo" in create_buffer
    assert "bufferInfo.to_cstruct(&bufferNative)" in create_buffer
    assert "std::make_shared<const BufferCreateInfo>(bufferInfo)" in create_buffer
    assert "flush(DeviceSize offset" in declarations
    assert "flush(VkDeviceSize" not in declarations
    assert "Result<void> AllocationView::flush(DeviceSize offset" in implementations
    assert "vmaCreateBuffer(" not in declarations


def test_vma_sections_only_reference_selected_capabilities():
    declarations, _ = _vma_sections(vma_model("vmaMapMemory", "vmaUnmapMemory"))
    assert "Result<void*> map()" in declarations
    assert "void unmap()" in declarations
    assert "vmaGetAllocationInfo" not in declarations
    assert "class Allocation {" not in declarations
    assert "Result<Allocator> create" not in declarations
    assert "Result<Buffer> createBuffer" not in declarations
    assert "Result<Image> createImage" not in declarations


def test_vma_resource_creation_requires_matching_destroy_function():
    model = vma_model("vmaCreateBuffer")
    declarations, _ = _vma_sections(model)
    assert "Result<Buffer> createBuffer" not in declarations
    assert _vma_resource_types(model) == frozenset()
    assert _vma_resource_types(vma_model("vmaCreateBuffer", "vmaDestroyBuffer")) == frozenset({"VkBuffer"})
