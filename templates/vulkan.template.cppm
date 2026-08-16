module;

{{generated_notice}}
{{includes}}

#ifndef VMA_STATIC_VULKAN_FUNCTIONS
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#endif
#ifndef VMA_DYNAMIC_VULKAN_FUNCTIONS
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#endif
#include <vk_mem_alloc.h>

export module {{module_name}};

{{begin_inject}}
typename Device:
    // VulkanMemoryAllocator convenience (template-provided; the generator stays
    // Vulkan-only). allocator() creates the per-device VmaAllocator on first use
    // and caches it in this handle's user data so it is destroyed with the
    // device. createAllocatedBuffer/createAllocatedImage route through it and
    // hand the returned handles a destroyer that releases the VMA allocation.
    // Definitions live in the module implementation section below.
    [[nodiscard]] VmaAllocator allocator() const;
    [[nodiscard]] Result<Buffer> createAllocatedBuffer(const BufferCreateInfo& bufferInfo, const VmaAllocationCreateInfo& allocationInfo) const;
    [[nodiscard]] Result<Image> createAllocatedImage(const ImageCreateInfo& imageInfo, const VmaAllocationCreateInfo& allocationInfo) const;
{{end_inject}}

export namespace {{namespace}} {

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

namespace {{namespace}} {
{{struct_implementations}}
{{handle_implementations}}
{{command_implementations}}

// ---------------------------------------------------------------------------
// VulkanMemoryAllocator convenience definitions (template-provided). These live
// in the module implementation section so every Vulkan handle/struct type is
// complete.
// ---------------------------------------------------------------------------

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
    auto holder = getOrSetUserData<detail::VmaAllocatorLifetime>(
        [this]() -> std::shared_ptr<const detail::VmaAllocatorLifetime> {
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
            return std::make_shared<const detail::VmaAllocatorLifetime>(value);
        });
    return holder ? holder->allocator : nullptr;
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
