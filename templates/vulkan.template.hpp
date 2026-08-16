#pragma once

{{generated_notice}}
{{includes}}

#ifndef VMA_STATIC_VULKAN_FUNCTIONS
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#endif
#ifndef VMA_DYNAMIC_VULKAN_FUNCTIONS
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#endif
#include <vk_mem_alloc.h>

{{begin_inject}}
typename Device:
    // VulkanMemoryAllocator convenience (template-provided; the generator stays
    // Vulkan-only). allocator() creates the per-device VmaAllocator on first use
    // and caches it in this handle's user data so it is destroyed with the
    // device. createAllocatedBuffer/createAllocatedImage route through it and
    // hand the returned handles a destroyer that releases the VMA allocation.
    // Definitions live in the paired .cpp, where every Vulkan type is complete.
    [[nodiscard]] VmaAllocator allocator() const;
    [[nodiscard]] Result<Buffer> createAllocatedBuffer(const BufferCreateInfo& bufferInfo, const VmaAllocationCreateInfo& allocationInfo) const;
    [[nodiscard]] Result<Image> createAllocatedImage(const ImageCreateInfo& imageInfo, const VmaAllocationCreateInfo& allocationInfo) const;
{{end_inject}}

namespace {{namespace}} {

{{result_code}}

{{runtime_declarations}}

{{forward_declarations}}

{{constants}}

{{enums}}

{{aliases}}

{{structure_extensions}}

{{handles}}

{{structs}}

{{context}}

{{command_declarations}}

{{struct_template_implementations}}

{{handle_template_implementations}}

{{command_template_implementations}}

} // namespace {{namespace}}
