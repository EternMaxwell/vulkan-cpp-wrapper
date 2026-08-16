{{generated_notice}}
#include "vulkan_wrapper.hpp"

namespace {{namespace}} {

{{runtime_implementations}}

{{struct_implementations}}

{{handle_implementations}}

{{command_implementations}}

// ---------------------------------------------------------------------------
// VulkanMemoryAllocator convenience definitions (template-provided). These are
// ordinary (non-inline) definitions in the paired translation unit.
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

VmaAllocator Device::allocator() const {
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

Result<Buffer> Device::createAllocatedBuffer(
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

Result<Image> Device::createAllocatedImage(
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
