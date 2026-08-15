// 07-vma: verifies the wrapper's VulkanMemoryAllocator integration.
//
// Directionally exercises Allocator::create, VMA-backed buffer/image creation
// (with their creation records and allocation metadata), and raw allocation +
// AllocationView::map/unmap. VMA is compiled in this TU (VMA_IMPLEMENTATION)
// and driven through volk's vkGetInstanceProcAddr / vkGetDeviceProcAddr.
#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <vulkan_wrapper_vma.hpp>
#include <validation.hpp>

#include <cstdint>
#include <cstring>
#include <print>

static std::uint32_t find_graphics_family(const vk::PhysicalDevice& physical) {
    auto families = physical.getQueueFamilyProperties();
    for (std::uint32_t i = 0; i < families.size(); ++i)
        if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) return i;
    return 0;
}

int main() {
    if (volkInitialize() != VK_SUCCESS) { std::println(stderr, "volkInitialize failed"); return 1; }
    auto context = vk::Context::create();
    if (!context) { std::println(stderr, "Context::create failed"); return 1; }

    vk::ApplicationInfo app{};
    app.setApplicationName("vulkan-wrapper-vma");
    app.setApiVersion(VK_API_VERSION_1_3);
    vk::InstanceCreateInfo instanceInfo{};
    instanceInfo.setApplicationInfo(app);
    sample::ValidationReporter validation{};
    vk::DebugUtilsMessengerCreateInfoEXT messengerInfo{};
    const bool validationEnabled =
        sample::enableValidationIfAvailable(*context, instanceInfo, messengerInfo, validation);
    auto instance = context->createInstance(instanceInfo, std::nullopt);
    if (!instance) { std::println(stderr, "createInstance failed"); return 1; }
    vk::DebugUtilsMessengerEXT messenger{};
    if (validationEnabled) {
        auto created = instance->createDebugUtilsMessengerEXT(messengerInfo, std::nullopt);
        if (!created) std::println(stderr, "warning: createDebugUtilsMessengerEXT failed");
        else messenger = std::move(*created);
    }

    auto devices = instance->enumeratePhysicalDevices();
    if (devices.value.empty()) { std::println(stderr, "no physical devices"); return 1; }
    auto physical = std::move(devices.value[0]);

    std::uint32_t family = find_graphics_family(physical);
    vk::DeviceQueueCreateInfo queueInfo{};
    queueInfo.setQueueFamilyIndex(family).setQueuePriorities({1.0f});
    vk::DeviceCreateInfo deviceInfo{};
    deviceInfo.setQueueCreateInfos({queueInfo});
    auto device = physical.createDevice(deviceInfo, std::nullopt);
    if (!device) { std::println(stderr, "createDevice failed"); return 1; }

    // Wire VMA to volk's loader/device proc-address functions.
    VmaVulkanFunctions vmaFunctions{};
    vmaFunctions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vmaFunctions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
    VmaAllocatorCreateInfo allocatorInfo{};
    allocatorInfo.physicalDevice = physical.raw();
    allocatorInfo.device = device->raw();
    allocatorInfo.instance = instance->raw();
    allocatorInfo.pVulkanFunctions = &vmaFunctions;
    allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_3;

    auto allocator = vk::Allocator::create(*device, allocatorInfo);
    if (!allocator) { std::println(stderr, "Allocator::create failed"); return 1; }

    // VMA-backed buffer (with creation record + allocation metadata).
    VmaAllocationCreateInfo bufferAllocInfo{};
    bufferAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    bufferAllocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    auto vmaBuffer = allocator->createBuffer(
        vk::BufferCreateInfo{}.setSize(64).setUsage(vk::BufferUsageFlagBits::VertexBuffer),
        bufferAllocInfo);
    bool bufferOk = vmaBuffer && vmaBuffer->createInfo() &&
                    vmaBuffer->createInfo()->size == 64 &&
                    vmaBuffer->allocation() != VmaAllocation{} &&
                    vmaBuffer->allocationInfo() && vmaBuffer->allocationInfo()->size >= 64 &&
                    vmaBuffer->allocationCreateInfo() &&
                    vmaBuffer->allocationCreateInfo()->usage == VMA_MEMORY_USAGE_AUTO;

    // VMA-backed image.
    VmaAllocationCreateInfo imageAllocInfo{};
    imageAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    auto vmaImage = allocator->createImage(
        vk::ImageCreateInfo{}.setImageType(vk::ImageType::Value2d)
            .setFormat(vk::Format::R8g8b8a8Unorm).setExtent(vk::Extent3D{16, 16, 1})
            .setMipLevels(1).setArrayLayers(1).setSamples(vk::SampleCountFlagBits::Value1)
            .setTiling(vk::ImageTiling::Optimal)
            .setUsage(vk::ImageUsageFlagBits::Sampled),
        imageAllocInfo);
    bool imageOk = vmaImage && vmaImage->allocation() != VmaAllocation{};

    // Raw allocation + map/unmap round-trip.
    vk::MemoryRequirements reqs{};
    reqs.setSize(64).setAlignment(16).setMemoryTypeBits(0xFFFFFFFF);
    VmaAllocationCreateInfo rawAllocInfo{};
    rawAllocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;  // host-visible, mappable
    auto allocation = allocator->allocate(reqs, rawAllocInfo);
    bool mapOk = false;
    if (allocation) {
        auto mapped = allocation->view().map();
        if (mapped) {
            std::memset(*mapped, 0xAB, 64);
            auto verify = allocation->view().map();
            if (verify) {
                mapOk = std::memcmp(*mapped, *verify, 64) == 0;
                const auto* bytes = static_cast<const std::uint8_t*>(*verify);
                mapOk = mapOk && bytes[0] == 0xAB && bytes[63] == 0xAB;
            }
            allocation->view().unmap();
        }
    }

    std::println("VMA: allocator={} buffer={} image={} map={}",
                 allocator->use_count() > 0 ? "ok" : "missing",
                 bufferOk ? "ok" : "missing", imageOk ? "ok" : "missing", mapOk ? "ok" : "failed");
    if (!bufferOk || !imageOk || !mapOk) {
        std::println(stderr, "FAIL: VMA verification failed");
        return 1;
    }
    std::println("PASS: VMA integration verified");
    if (!sample::reportValidation(validation)) return 1;
    return 0;
}
