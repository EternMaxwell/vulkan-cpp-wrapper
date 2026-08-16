// 07-vma: verifies the wrapper's VulkanMemoryAllocator convenience.
//
// The generator stays Vulkan-only; VMA integration lives entirely in the
// wrapper template as three Device conveniences: allocator() (created on first
// use and cached in the device's user data, so it is destroyed with the
// device) plus createAllocatedBuffer/createAllocatedImage. VMA is compiled in
// this TU (VMA_IMPLEMENTATION) and driven through volk's proc addresses, which
// the template wires up for us.
#define VMA_IMPLEMENTATION
#include <vulkan_wrapper.hpp>
#include <validation.hpp>

#include <cstdint>
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

    // The allocator is created on first use and cached in the device's user
    // data (destroyed with the device).
    VmaAllocator allocator = device->allocator();
    bool allocatorOk = allocator != VmaAllocator{};

    // VMA-backed buffer (its creation record is retained).
    VmaAllocationCreateInfo bufferAllocInfo{};
    bufferAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    bufferAllocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    auto vmaBuffer = device->createAllocatedBuffer(
        vk::BufferCreateInfo{}.setSize(64).setUsage(vk::BufferUsageFlagBits::VertexBuffer),
        bufferAllocInfo);
    bool bufferOk = vmaBuffer && vmaBuffer->createInfo() &&
                    vmaBuffer->createInfo()->size == 64 &&
                    vmaBuffer->raw() != VkBuffer{};

    // VMA-backed image.
    VmaAllocationCreateInfo imageAllocInfo{};
    imageAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    auto vmaImage = device->createAllocatedImage(
        vk::ImageCreateInfo{}.setImageType(vk::ImageType::Value2d)
            .setFormat(vk::Format::R8g8b8a8Unorm).setExtent(vk::Extent3D{16, 16, 1})
            .setMipLevels(1).setArrayLayers(1).setSamples(vk::SampleCountFlagBits::Value1)
            .setTiling(vk::ImageTiling::Optimal)
            .setUsage(vk::ImageUsageFlagBits::Sampled),
        imageAllocInfo);
    bool imageOk = vmaImage && vmaImage->raw() != VkImage{};

    std::println("VMA: allocator={} buffer={} image={}",
                 allocatorOk ? "ok" : "missing",
                 bufferOk ? "ok" : "missing", imageOk ? "ok" : "missing");
    if (!allocatorOk || !bufferOk || !imageOk) {
        std::println(stderr, "FAIL: VMA verification failed");
        return 1;
    }
    std::println("PASS: VMA integration verified");
    if (!sample::reportValidation(validation)) return 1;
    return 0;
}
