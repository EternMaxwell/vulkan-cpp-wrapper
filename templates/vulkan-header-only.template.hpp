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
    // Definitions are emitted after the wrapper body, where every Vulkan type is
    // complete.
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
{{struct_implementations}}
{{handle_implementations}}
{{command_implementations}}

} // namespace {{namespace}}

// ---------------------------------------------------------------------------
// VulkanMemoryAllocator convenience definitions (template-provided). These live
// after the generated wrapper so every Vulkan handle/struct type is complete.
// ---------------------------------------------------------------------------

namespace {{namespace}} {

namespace detail {
struct VmaAllocatorLifetime {
    VmaAllocator allocator{};
    VmaAllocatorLifetime(VmaAllocator value) noexcept : allocator(value) {}
    VmaAllocatorLifetime(const VmaAllocatorLifetime&) = delete;
    VmaAllocatorLifetime& operator=(const VmaAllocatorLifetime&) = delete;
    ~VmaAllocatorLifetime() {
        if (allocator) vmaDestroyAllocator(allocator);
    }
};
} // namespace detail

inline VmaAllocator Device::allocator() const {
    if (auto cached = getUserData<detail::VmaAllocatorLifetime>()) return cached->allocator;
    VmaAllocatorCreateInfo info{};
    info.instance = parent().parent().raw();
    info.physicalDevice = parent().raw();
    info.device = raw();
    info.vulkanApiVersion = VK_API_VERSION_1_3;
    VmaVulkanFunctions functions{};
    functions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    functions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
    info.pVulkanFunctions = &functions;
    VmaAllocator value{};
    if (vmaCreateAllocator(&info, &value) != VK_SUCCESS || !value) return nullptr;
    if (!setUserData<detail::VmaAllocatorLifetime>(
            std::make_shared<const detail::VmaAllocatorLifetime>(value))) {
        return nullptr;  // the shared_ptr just created is released and destroys the allocator
    }
    return value;
}

inline Result<Buffer> Device::createAllocatedBuffer(
    const BufferCreateInfo& bufferInfo,
    const VmaAllocationCreateInfo& allocationInfo) const {
    VmaAllocator vmaAllocator = allocator();
    if (!vmaAllocator) return std::unexpected(ResultCode::ErrorOutOfHostMemory);
    BufferCreateInfo::CStruct native{};
    bufferInfo.to_cstruct(&native);
    VkBuffer value{};
    VmaAllocation allocation{};
    const auto result = vmaCreateBuffer(vmaAllocator, &native.value, &allocationInfo,
                                        &value, &allocation, nullptr);
    if (result != VK_SUCCESS || value == VkBuffer{}) {
        return std::unexpected(static_cast<ResultCode>(result));
    }
    return Buffer::adopt(
        value, *this,
        [vmaAllocator, allocation](VkBuffer handle) {
            vmaDestroyBuffer(vmaAllocator, handle, allocation);
        },
        std::make_shared<const BufferCreateInfo>(bufferInfo));
}

inline Result<Image> Device::createAllocatedImage(
    const ImageCreateInfo& imageInfo,
    const VmaAllocationCreateInfo& allocationInfo) const {
    VmaAllocator vmaAllocator = allocator();
    if (!vmaAllocator) return std::unexpected(ResultCode::ErrorOutOfHostMemory);
    ImageCreateInfo::CStruct native{};
    imageInfo.to_cstruct(&native);
    VkImage value{};
    VmaAllocation allocation{};
    const auto result = vmaCreateImage(vmaAllocator, &native.value, &allocationInfo,
                                       &value, &allocation, nullptr);
    if (result != VK_SUCCESS || value == VkImage{}) {
        return std::unexpected(static_cast<ResultCode>(result));
    }
    return Image::adopt(
        value, *this,
        [vmaAllocator, allocation](VkImage handle) {
            vmaDestroyImage(vmaAllocator, handle, allocation);
        },
        std::make_shared<const ImageCreateInfo>(imageInfo));
}

} // namespace {{namespace}}
