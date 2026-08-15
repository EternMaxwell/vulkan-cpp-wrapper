// 01-info: enumerate the Vulkan instance and physical devices through the
// wrapper. This sample has no windowing requirement and exercises the basic
// RAII handle + counted-query convenience APIs.
#include <volk.h>
#include <vulkan_wrapper.hpp>
#include <validation.hpp>

#include <cstdio>
#include <cstring>
#include <string>

int main() {
    if (volkInitialize() != VK_SUCCESS) {
        std::fprintf(stderr, "error: failed to initialize volk (no Vulkan loader?)\n");
        return 1;
    }

    auto context = vk::Context::create();
    if (!context) {
        std::fprintf(stderr, "error: failed to create context: %d\n",
                     static_cast<int>(context.error()));
        return 1;
    }

    vk::ApplicationInfo app{};
    app.setApplicationName("vulkan-wrapper-info");
    app.setApplicationVersion(VK_MAKE_API_VERSION(0, 1, 0, 0));
    app.setEngineName("vulkan-wrapper");
    app.setApiVersion(VK_API_VERSION_1_3);

    vk::InstanceCreateInfo instanceInfo{};
    instanceInfo.setApplicationInfo(app);

    sample::ValidationReporter validation{};
    vk::DebugUtilsMessengerCreateInfoEXT messengerInfo{};
    const bool validationEnabled =
        sample::enableValidationIfAvailable(*context, instanceInfo, messengerInfo, validation);

    auto instance = context->createInstance(instanceInfo, std::nullopt);
    if (!instance) {
        std::fprintf(stderr, "error: failed to create instance: %d\n",
                     static_cast<int>(instance.error()));
        return 1;
    }
    vk::DebugUtilsMessengerEXT messenger{};
    if (validationEnabled) {
        auto created = instance->createDebugUtilsMessengerEXT(messengerInfo, std::nullopt);
        if (!created) {
            std::fprintf(stderr, "warning: createDebugUtilsMessengerEXT failed: %d\n",
                         static_cast<int>(created.error()));
        } else {
            messenger = std::move(*created);
        }
    }

    auto devices = instance->enumeratePhysicalDevices();
    if (devices.status != vk::ResultCode::Success &&
        devices.status != vk::ResultCode::Incomplete) {
        std::fprintf(stderr, "error: enumeratePhysicalDevices: %d\n",
                     static_cast<int>(devices.status));
        return 1;
    }

    std::printf("Physical devices: %zu\n", devices.value.size());
    for (const auto& physical : devices.value) {
        auto props = physical.getProperties();
        // deviceName is a fixed-size array; Vulkan guarantees NUL termination.
        std::printf("  - %s (type %d, vendor 0x%08x)\n",
                    props.deviceName.data(),
                    static_cast<int>(props.deviceType),
                    props.vendorID);

        auto families = physical.getQueueFamilyProperties();
        std::printf("    queue families: %zu\n", families.size());
        for (std::size_t i = 0; i < families.size(); ++i) {
            const char* flags = (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
                                    ? "graphics" : "other";
            std::printf("      family %zu: %s (%u queues)\n",
                        i, flags, families[i].queueCount);
        }
    }

    std::printf("done\n");
    if (!sample::reportValidation(validation)) return 1;
    return 0;
}
