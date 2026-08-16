// Optional validation-layer + debug-messenger wiring shared by all samples.
//
// The samples exist to prove the wrapper is usable against a real driver, so
// they turn on VK_LAYER_KHRONOS_validation whenever the loader exposes it and
// surface every validation message. Any ERROR-level message means the sample
// (or the wrapper) misused the API, so reportValidation() treats errors as a
// failure; warnings are printed but non-fatal.
//
// Include this AFTER your wrapper header (vulkan_wrapper.hpp); it relies on
// the vk:: types and the Vulkan constants that header already pulls in through
// volk.
#pragma once

#include <cstdio>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sample {

inline constexpr std::string_view kValidationLayerName = "VK_LAYER_KHRONOS_validation";

struct ValidationReporter {
    int errors = 0;
    int warnings = 0;
    bool enabled = false;
};

// Enables VK_LAYER_KHRONOS_validation when the loader reports it and wires a
// debug-utils messenger into the instance pNext so create/destroy-time
// messages are captured too. Returns true when validation is active; the
// caller then passes `messengerInfo` to createDebugUtilsMessengerEXT after the
// instance exists to capture everything in between.
inline bool enableValidationIfAvailable(const vk::Context& context,
                                        vk::InstanceCreateInfo& info,
                                        vk::DebugUtilsMessengerCreateInfoEXT& messengerInfo,
                                        ValidationReporter& reporter) {
    auto layers = context.enumerateInstanceLayerProperties();
    if (layers.status != vk::ResultCode::Success) return false;
    bool available = false;
    for (const auto& layer : layers.value) {
        if (std::string_view(layer.layerName.data()) == kValidationLayerName) {
            available = true;
            break;
        }
    }
    if (!available) {
        std::fprintf(stderr, "validation layer not found; running without validation\n");
        return false;
    }

    info.setEnabledLayerNames({std::string(kValidationLayerName)});

    // Preserve any extensions the caller already requested (e.g. surface), and
    // add the debug-utils extension the messenger needs.
    auto extensions = info.enabledExtensionNames;
    extensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    info.setEnabledExtensionNames(std::move(extensions));

    messengerInfo
        .setMessageSeverity(vk::DebugUtilsMessageSeverityFlagBitsEXT::Warning |
                            vk::DebugUtilsMessageSeverityFlagBitsEXT::Error)
        .setMessageType(vk::DebugUtilsMessageTypeFlagBitsEXT::General |
                        vk::DebugUtilsMessageTypeFlagBitsEXT::Validation |
                        vk::DebugUtilsMessageTypeFlagBitsEXT::Performance)
        .setUserCallback([reporter_ptr = &reporter](
                             VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                             VkDebugUtilsMessageTypeFlagsEXT /*types*/,
                             const VkDebugUtilsMessengerCallbackDataEXT* data) -> VkBool32 {
            const char* level = "info";
            if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
                ++reporter_ptr->errors;
                level = "error";
            } else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
                ++reporter_ptr->warnings;
                level = "warning";
            }
            const char* id = data && data->pMessageIdName ? data->pMessageIdName : "";
            const char* message = data && data->pMessage ? data->pMessage : "";
            std::fprintf(stderr, "[validation:%s] %s%s%s\n", level, id, *id ? ": " : "", message);
            return VK_FALSE;
        });
    info.setNextInChain(messengerInfo);
    reporter.enabled = true;
    return true;
}

// Prints a validation summary and fails the sample on any API-misuse error.
inline bool reportValidation(const ValidationReporter& reporter) {
    if (reporter.enabled) {
        std::fprintf(stderr, "validation: %d error(s), %d warning(s)\n",
                     reporter.errors, reporter.warnings);
    }
    return reporter.errors == 0;
}

}  // namespace sample
