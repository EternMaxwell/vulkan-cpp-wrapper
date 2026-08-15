// 08-allocator: verifies custom allocation callbacks round-trip through the
// wrapper. VkAllocationCallbacks is the multi-callback case: five function
// pointers share one pUserData. The wrapper turns them into refcounted
// callables; here they are capturing lambdas wrapped around a correct
// aligned allocator, and the sample asserts the driver actually invokes them.
#include <volk.h>
#include <vulkan_wrapper.hpp>
#include <validation.hpp>

#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <print>

static std::size_t g_allocations = 0;
static std::size_t g_frees = 0;

// A minimal aligned allocator: overallocate, align, and store the raw pointer
// plus the usable size in a header just before the aligned block so reallocation
// can copy the old contents.
struct AllocationHeader {
    void* base;
    std::size_t size;
};

static void* aligned_allocate(std::size_t size, std::size_t alignment, bool zero) {
    std::size_t total = size + alignment + sizeof(AllocationHeader);
    void* raw = std::malloc(total);
    if (!raw) return nullptr;
    std::uintptr_t base = reinterpret_cast<std::uintptr_t>(raw) + sizeof(AllocationHeader);
    std::uintptr_t aligned = (base + alignment - 1) & ~(alignment - 1);
    auto* header = reinterpret_cast<AllocationHeader*>(aligned) - 1;
    header->base = raw;
    header->size = size;
    void* ptr = reinterpret_cast<void*>(aligned);
    if (zero) std::memset(ptr, 0, size);
    return ptr;
}

static void aligned_deallocate(void* ptr) {
    if (!ptr) return;
    auto* header = reinterpret_cast<AllocationHeader*>(ptr) - 1;
    std::free(header->base);
}

static void* aligned_reallocate(void* original, std::size_t size, std::size_t alignment, bool zero) {
    if (!original) return aligned_allocate(size, alignment, zero);
    auto* header = reinterpret_cast<AllocationHeader*>(original) - 1;
    std::size_t old_size = header->size;
    void* ptr = aligned_allocate(size, alignment, false);
    if (!ptr) return nullptr;
    std::memcpy(ptr, original, old_size < size ? old_size : size);
    aligned_deallocate(original);
    return ptr;
}

int main() {
    if (volkInitialize() != VK_SUCCESS) { std::println(stderr, "volkInitialize failed"); return 1; }
    auto context = vk::Context::create();
    if (!context) { std::println(stderr, "Context::create failed"); return 1; }

    vk::ApplicationInfo app{};
    app.setApplicationName("vulkan-wrapper-allocator");
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

    std::uint32_t family = 0;
    {
        auto families = physical.getQueueFamilyProperties();
        for (std::uint32_t i = 0; i < families.size(); ++i)
            if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { family = i; break; }
    }
    vk::DeviceQueueCreateInfo queueInfo{};
    queueInfo.setQueueFamilyIndex(family).setQueuePriorities({1.0f});
    vk::DeviceCreateInfo deviceInfo{};
    deviceInfo.setQueueCreateInfos({queueInfo});
    auto device = physical.createDevice(deviceInfo, std::nullopt);
    if (!device) { std::println(stderr, "createDevice failed"); return 1; }

    // Custom allocation callbacks. pfnAllocation/pfnReallocation/pfnFree must be
    // provided together; the wrapper only emits a non-null native pointer for
    // callables that are actually set.
    vk::AllocationCallbacks callbacks{};
    callbacks.setAllocation([&](std::size_t size, std::size_t alignment, VkSystemAllocationScope scope) -> void* {
        ++g_allocations;
        return aligned_allocate(size, alignment, scope == VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
    });
    callbacks.setReallocation([&](void* original, std::size_t size, std::size_t alignment, VkSystemAllocationScope scope) -> void* {
        return aligned_reallocate(original, size, alignment, scope == VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
    });
    callbacks.setFree([&](void* memory) {
        ++g_frees;
        aligned_deallocate(memory);
    });

    auto buffer = device->createBuffer(
        vk::BufferCreateInfo{}.setSize(64).setUsage(vk::BufferUsageFlagBits::VertexBuffer),
        std::cref(callbacks));
    if (!buffer) { std::println(stderr, "createBuffer with custom allocator failed"); return 1; }
    std::size_t allocations_after_create = g_allocations;
    buffer->reset();
    std::size_t frees_after_destroy = g_frees;

    std::println("allocator: allocations={} frees={}", allocations_after_create, frees_after_destroy);
    bool ok = allocations_after_create > 0 && frees_after_destroy > 0;
    if (!ok) { std::println(stderr, "FAIL: allocation callbacks were not invoked"); return 1; }
    std::println("PASS: allocation callbacks round-tripped");
    if (!sample::reportValidation(validation)) return 1;
    return 0;
}
