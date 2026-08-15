// 06-pnext: verifies the wrapper's pNext ExtensionChain usability.
//
// Directionally exercises setNextInChain / nextInChain.get<T>() with both the
// ergonomic nested setter (a.setNextInChain(std::move(b.setNextInChain(c))))
// and the mutable node accessor, plus single-node query round-trips. The chain
// is the wrapper's replacement for hand-building VkBaseOutStructure lists.
#include <volk.h>
#include <vulkan_wrapper.hpp>

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
    app.setApplicationName("vulkan-wrapper-pnext");
    app.setApiVersion(VK_API_VERSION_1_3);
    vk::InstanceCreateInfo instanceInfo{};
    instanceInfo.setApplicationInfo(app);
    auto instance = context->createInstance(instanceInfo, std::nullopt);
    if (!instance) { std::println(stderr, "createInstance failed"); return 1; }

    auto devices = instance->enumeratePhysicalDevices();
    if (devices.value.empty()) { std::println(stderr, "no physical devices"); return 1; }
    auto physical = std::move(devices.value[0]);

    // 1. An empty query has no chained struct (the chain is real, not always
    //    populated).
    auto empty = physical.getFeatures2();
    bool emptyChain = empty.nextInChain.get<vk::PhysicalDeviceVulkan13Features>() == nullptr;

    // 2. Chain a feature struct, query, and read it back through the chain.
    vk::PhysicalDeviceFeatures2 query{};
    vk::PhysicalDeviceVulkan13Features v13{};
    v13.setDynamicRendering(true);
    query.setNextInChain(v13);
    physical.getFeatures2(&query);
    auto* got13 = query.nextInChain.get<vk::PhysicalDeviceVulkan13Features>();
    bool roundtrip = got13 != nullptr && got13->dynamicRendering != 0;

    // 3. A two-node chain built with the nested setter.
    vk::PhysicalDeviceFeatures2 query2{};
    vk::PhysicalDeviceVulkan12Features v12{};
    vk::PhysicalDeviceVulkan13Features v13b{};
    v13b.setDynamicRendering(true);
    query2.setNextInChain(std::move(v12.setNextInChain(v13b)));
    auto* got12 = query2.nextInChain.get<vk::PhysicalDeviceVulkan12Features>();
    bool nested = got12 != nullptr &&
                  got12->nextInChain.get<vk::PhysicalDeviceVulkan13Features>() != nullptr;

    // 4. The same chain built through the mutable node accessor.
    vk::PhysicalDeviceFeatures2 query3{};
    query3.setNextInChain(vk::PhysicalDeviceVulkan12Features{});
    query3.nextInChain.get<vk::PhysicalDeviceVulkan12Features>()
          ->setNextInChain(vk::PhysicalDeviceVulkan13Features{});
    auto* got12b = query3.nextInChain.get<vk::PhysicalDeviceVulkan12Features>();
    bool recursive = got12b != nullptr &&
                     got12b->nextInChain.get<vk::PhysicalDeviceVulkan13Features>() != nullptr;

    // 5. Enable a feature at device creation through a chained struct.
    vk::PhysicalDeviceVulkan13Features enable13{};
    enable13.setDynamicRendering(true);
    std::uint32_t family = find_graphics_family(physical);
    vk::DeviceQueueCreateInfo queueInfo{};
    queueInfo.setQueueFamilyIndex(family).setQueuePriorities({1.0f});
    vk::DeviceCreateInfo deviceInfo{};
    deviceInfo.setQueueCreateInfos({queueInfo}).setNextInChain(enable13);
    auto device = physical.createDevice(deviceInfo, std::nullopt);

    std::println("pNext: empty={} roundtrip={} nested={} recursive={} device={}",
                 emptyChain ? "null" : "present", roundtrip ? "ok" : "missing",
                 nested ? "ok" : "missing", recursive ? "ok" : "missing",
                 device ? "yes" : "no");
    if (!emptyChain || !roundtrip || !nested || !recursive || !device) {
        std::println(stderr, "FAIL: pNext chain verification failed");
        return 1;
    }
    std::println("PASS: pNext chain verified");
    return 0;
}
