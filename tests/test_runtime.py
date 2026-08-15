from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest

from vulkan_wrapper_gen.cli import run

ROOT = Path(__file__).parents[1]


def _dependency(name: str) -> Path | None:
    candidates = [ROOT / ".ci" / name]
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidates.append(Path(local) / "Temp" / name)
    return next((path for path in candidates if path.is_dir()), None)


def _compiler() -> str | None:
    found = shutil.which("clang++") or shutil.which("g++")
    if found:
        return found
    llvm = Path(r"D:\Program Files\LLVM\bin\clang++.exe")
    return str(llvm) if llvm.is_file() else None


def test_generated_handle_lifetimes_execute_against_fake_volk(tmp_path: Path):
    compiler = _compiler()
    headers = _dependency("Vulkan-Headers")
    volk = _dependency("volk")
    vma = _dependency("VulkanMemoryAllocator")
    if not compiler or not headers or not volk or not vma:
        pytest.skip("runtime C++ test needs a compiler, Vulkan-Headers, Volk, and VMA")

    generated = tmp_path / "vulkan_wrapper.hpp"
    assert (
        run(
            [
                "--registry",
                str(headers / "registry" / "vk.xml"),
                "--vma-header",
                str(vma / "include" / "vk_mem_alloc.h"),
                "--clang-arg=-I" + str(headers / "include"),
                "--emit", str(ROOT / "templates" / "vulkan-header-only.template.hpp") + ":" + str(generated),
            ]
        )
        == 0
    )

    source = tmp_path / "runtime.cpp"
    source.write_text(
        r"""
#define VOLK_IMPLEMENTATION
#include <volk.h>
#include "vulkan_wrapper.hpp"

#include <cassert>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <type_traits>
#include <thread>
#include <vector>

template <typename T> T fake_handle(std::uintptr_t value) {
    if constexpr (std::is_pointer_v<T>) return reinterpret_cast<T>(value);
    else return static_cast<T>(value);
}
template <typename T> std::uint64_t fake_key(T value) {
    if constexpr (std::is_pointer_v<T>) return reinterpret_cast<std::uintptr_t>(value);
    else return static_cast<std::uint64_t>(value);
}

using PrivateKey = std::tuple<std::uint64_t, std::int32_t, std::uint64_t, std::uint64_t>;
static std::map<PrivateKey, std::uint64_t> private_values;
static std::mutex private_mutex;
static bool reject_private_data = false;
static int slot_creates = 0;
static int slot_destroys = 0;
static int private_sets = 0;
static bool private_feature_supported = true;
static int create_device_calls = 0;
static int generated_device_destroys = 0;
static bool create_device_saw_private_feature = false;
static bool create_device_saw_slot_request = false;
static bool create_device_saw_vulkan13_private = false;
static bool create_device_saw_private_data_features = false;
static bool enumerate_force_retry = false;
static int enumerate_calls = 0;
static std::atomic_int queue_submit_active{0};
static std::atomic_int queue_submit_calls{0};
static std::atomic_bool queue_submit_overlap{false};
static std::atomic_bool debug_name_entered{false};
static std::atomic_bool release_debug_name{false};
static int pipeline_destroys = 0;
static int descriptor_set_frees = 0;
static int destruction_errors = 0;
static vk::ResultCode destruction_error_code{};
static std::string_view destruction_error_type{};
static std::uint64_t destruction_error_identity = 0;
static int vma_allocator_destroys = 0;
static int vma_allocation_frees = 0;
static int vma_buffer_destroys = 0;
static int vma_image_destroys = 0;
static int vma_maps = 0;
static int vma_unmaps = 0;
static int vma_flushes = 0;
static int vma_invalidates = 0;
static std::byte mapped_bytes[32]{};

static void capture_destruction_error(
    vk::ResultCode code, std::string_view type, std::uint64_t identity) noexcept {
    ++destruction_errors;
    destruction_error_code = code;
    destruction_error_type = type;
    destruction_error_identity = identity;
}

VKAPI_ATTR VkResult VKAPI_CALL fake_create_slot(
    VkDevice, const VkPrivateDataSlotCreateInfo*, const VkAllocationCallbacks*, VkPrivateDataSlot* value) {
    ++slot_creates;
    *value = fake_handle<VkPrivateDataSlot>(0x9000u + static_cast<unsigned>(slot_creates));
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL fake_destroy_slot(
    VkDevice, VkPrivateDataSlot slot, const VkAllocationCallbacks*) {
    ++slot_destroys;
    std::lock_guard lock(private_mutex);
    for (auto it = private_values.begin(); it != private_values.end();) {
        if (std::get<3>(it->first) == fake_key(slot)) it = private_values.erase(it);
        else ++it;
    }
}
VKAPI_ATTR VkResult VKAPI_CALL fake_set_private(
    VkDevice device, VkObjectType type, std::uint64_t object, VkPrivateDataSlot slot, std::uint64_t value) {
    ++private_sets;
    if (reject_private_data) return VK_ERROR_UNKNOWN;
    PrivateKey key{fake_key(device), static_cast<std::int32_t>(type), object, fake_key(slot)};
    std::lock_guard lock(private_mutex);
    if (value) private_values[key] = value;
    else private_values.erase(key);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL fake_get_private(
    VkDevice device, VkObjectType type, std::uint64_t object, VkPrivateDataSlot slot, std::uint64_t* value) {
    PrivateKey key{fake_key(device), static_cast<std::int32_t>(type), object, fake_key(slot)};
    std::lock_guard lock(private_mutex);
    auto found = private_values.find(key);
    *value = found == private_values.end() ? 0 : found->second;
}
VKAPI_ATTR void VKAPI_CALL fake_get_physical_device_features2(
    VkPhysicalDevice, VkPhysicalDeviceFeatures2* features) {
    for (auto* node = static_cast<VkBaseOutStructure*>(features->pNext); node; node = node->pNext) {
        if (node->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIVATE_DATA_FEATURES) {
            reinterpret_cast<VkPhysicalDevicePrivateDataFeatures*>(node)->privateData =
                private_feature_supported ? VK_TRUE : VK_FALSE;
        }
    }
}
VKAPI_ATTR VkResult VKAPI_CALL fake_create_device(
    VkPhysicalDevice, const VkDeviceCreateInfo* info, const VkAllocationCallbacks*, VkDevice* value) {
    ++create_device_calls;
    create_device_saw_private_feature = false;
    create_device_saw_slot_request = false;
    create_device_saw_vulkan13_private = false;
    create_device_saw_private_data_features = false;
    for (auto* node = static_cast<const VkBaseInStructure*>(info->pNext); node; node = node->pNext) {
        if (node->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIVATE_DATA_FEATURES) {
            create_device_saw_private_feature =
                reinterpret_cast<const VkPhysicalDevicePrivateDataFeatures*>(node)->privateData == VK_TRUE;
            create_device_saw_private_data_features = true;
        }
        if (node->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES) {
            create_device_saw_vulkan13_private =
                reinterpret_cast<const VkPhysicalDeviceVulkan13Features*>(node)->privateData == VK_TRUE;
        }
        if (node->sType == VK_STRUCTURE_TYPE_DEVICE_PRIVATE_DATA_CREATE_INFO) {
            create_device_saw_slot_request =
                reinterpret_cast<const VkDevicePrivateDataCreateInfo*>(node)->privateDataSlotRequestCount == 4;
        }
    }
    *value = fake_handle<VkDevice>(0x3500);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL fake_destroy_generated_device(VkDevice, const VkAllocationCallbacks*) {
    ++generated_device_destroys;
}
VKAPI_ATTR void VKAPI_CALL fake_get_device_queue(VkDevice, uint32_t family, uint32_t index, VkQueue* queue) {
    *queue = fake_handle<VkQueue>(0x3800 + family * 16 + index);
}
VKAPI_ATTR VkResult VKAPI_CALL fake_enumerate_physical_devices(
    VkInstance, std::uint32_t* count, VkPhysicalDevice* values) {
    ++enumerate_calls;
    if (!values) {
        *count = enumerate_force_retry && enumerate_calls >= 3 ? 2u : 1u;
        return VK_SUCCESS;
    }
    values[0] = fake_handle<VkPhysicalDevice>(0x2000);
    if (*count >= 2) {
        values[1] = fake_handle<VkPhysicalDevice>(0x2001);
        *count = 2;
        return VK_SUCCESS;
    }
    *count = 1;
    return VK_INCOMPLETE;
}
VKAPI_ATTR VkResult VKAPI_CALL fake_queue_submit(
    VkQueue, std::uint32_t, const VkSubmitInfo*, VkFence) {
    if (queue_submit_active.fetch_add(1, std::memory_order_acq_rel) != 0) {
        queue_submit_overlap.store(true, std::memory_order_release);
    }
    ++queue_submit_calls;
    for (int i = 0; i != 2000; ++i) std::this_thread::yield();
    queue_submit_active.fetch_sub(1, std::memory_order_acq_rel);
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL fake_set_debug_object_name(
    VkDevice, const VkDebugUtilsObjectNameInfoEXT*) {
    debug_name_entered.store(true, std::memory_order_release);
    while (!release_debug_name.load(std::memory_order_acquire)) std::this_thread::yield();
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL fake_create_graphics_pipelines(
    VkDevice, VkPipelineCache, std::uint32_t count,
    const VkGraphicsPipelineCreateInfo*, const VkAllocationCallbacks*, VkPipeline* pipelines) {
    for (std::uint32_t i = 0; i < count; ++i) pipelines[i] = fake_handle<VkPipeline>(0xb000 + i);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL fake_destroy_pipeline(
    VkDevice, VkPipeline, const VkAllocationCallbacks*) {
    ++pipeline_destroys;
}
VKAPI_ATTR VkResult VKAPI_CALL fake_allocate_descriptor_sets(
    VkDevice, const VkDescriptorSetAllocateInfo* info, VkDescriptorSet* sets) {
    for (std::uint32_t i = 0; i < info->descriptorSetCount; ++i) {
        sets[i] = fake_handle<VkDescriptorSet>(
            0xc000 + fake_key(info->descriptorPool) * 0x10 + i);
    }
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL fake_free_descriptor_sets(
    VkDevice, VkDescriptorPool, std::uint32_t count, const VkDescriptorSet*) {
    descriptor_set_frees += static_cast<int>(count);
    return VK_SUCCESS;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL fake_get_device_proc(VkDevice, const char* name) {
    if (std::strcmp(name, "vkCreatePrivateDataSlot") == 0) return reinterpret_cast<PFN_vkVoidFunction>(fake_create_slot);
    if (std::strcmp(name, "vkDestroyPrivateDataSlot") == 0) return reinterpret_cast<PFN_vkVoidFunction>(fake_destroy_slot);
    if (std::strcmp(name, "vkSetPrivateData") == 0) return reinterpret_cast<PFN_vkVoidFunction>(fake_set_private);
    if (std::strcmp(name, "vkGetPrivateData") == 0) return reinterpret_cast<PFN_vkVoidFunction>(fake_get_private);
    if (std::strcmp(name, "vkDestroyDevice") == 0) return reinterpret_cast<PFN_vkVoidFunction>(fake_destroy_generated_device);
    if (std::strcmp(name, "vkGetDeviceQueue") == 0) return reinterpret_cast<PFN_vkVoidFunction>(fake_get_device_queue);
    if (std::strcmp(name, "vkQueueSubmit") == 0) return reinterpret_cast<PFN_vkVoidFunction>(fake_queue_submit);
    if (std::strcmp(name, "vkCreateGraphicsPipelines") == 0) return reinterpret_cast<PFN_vkVoidFunction>(fake_create_graphics_pipelines);
    if (std::strcmp(name, "vkDestroyPipeline") == 0) return reinterpret_cast<PFN_vkVoidFunction>(fake_destroy_pipeline);
    if (std::strcmp(name, "vkAllocateDescriptorSets") == 0) return reinterpret_cast<PFN_vkVoidFunction>(fake_allocate_descriptor_sets);
    if (std::strcmp(name, "vkFreeDescriptorSets") == 0) return reinterpret_cast<PFN_vkVoidFunction>(fake_free_descriptor_sets);
    return nullptr;
}
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL fake_get_instance_proc(VkInstance, const char* name) {
    if (std::strcmp(name, "vkGetDeviceProcAddr") == 0) return reinterpret_cast<PFN_vkVoidFunction>(fake_get_device_proc);
    if (std::strcmp(name, "vkGetPhysicalDeviceFeatures2") == 0) return reinterpret_cast<PFN_vkVoidFunction>(fake_get_physical_device_features2);
    if (std::strcmp(name, "vkCreateDevice") == 0) return reinterpret_cast<PFN_vkVoidFunction>(fake_create_device);
    if (std::strcmp(name, "vkEnumeratePhysicalDevices") == 0) return reinterpret_cast<PFN_vkVoidFunction>(fake_enumerate_physical_devices);
    if (std::strcmp(name, "vkSetDebugUtilsObjectNameEXT") == 0) return reinterpret_cast<PFN_vkVoidFunction>(fake_set_debug_object_name);
    return nullptr;
}

VkResult vmaCreateAllocator(const VmaAllocatorCreateInfo*, VmaAllocator* allocator) {
    *allocator = fake_handle<VmaAllocator>(0xa000);
    return VK_SUCCESS;
}
void vmaDestroyAllocator(VmaAllocator) { ++vma_allocator_destroys; }
VkResult vmaAllocateMemory(VmaAllocator, const VkMemoryRequirements*, const VmaAllocationCreateInfo*, VmaAllocation* allocation, VmaAllocationInfo* info) {
    *allocation = fake_handle<VmaAllocation>(0xa100);
    if (info) { *info = {}; info->size = 32; }
    return VK_SUCCESS;
}
void vmaFreeMemory(VmaAllocator, VmaAllocation) { ++vma_allocation_frees; }
VkResult vmaMapMemory(VmaAllocator, VmaAllocation, void** data) {
    ++vma_maps;
    *data = mapped_bytes;
    return VK_SUCCESS;
}
void vmaUnmapMemory(VmaAllocator, VmaAllocation) { ++vma_unmaps; }
VkResult vmaFlushAllocation(VmaAllocator, VmaAllocation, VkDeviceSize, VkDeviceSize) {
    ++vma_flushes;
    return VK_SUCCESS;
}
VkResult vmaInvalidateAllocation(VmaAllocator, VmaAllocation, VkDeviceSize, VkDeviceSize) {
    ++vma_invalidates;
    return VK_SUCCESS;
}
void vmaGetAllocationInfo(VmaAllocator, VmaAllocation, VmaAllocationInfo* info) {
    *info = {};
    info->size = 32;
}
VkResult vmaCreateBuffer(VmaAllocator, const VkBufferCreateInfo*, const VmaAllocationCreateInfo*, VkBuffer* buffer, VmaAllocation* allocation, VmaAllocationInfo* info) {
    *buffer = fake_handle<VkBuffer>(0xa200);
    *allocation = fake_handle<VmaAllocation>(0xa201);
    if (info) { *info = {}; info->size = 64; }
    return VK_SUCCESS;
}
void vmaDestroyBuffer(VmaAllocator, VkBuffer, VmaAllocation) { ++vma_buffer_destroys; }
VkResult vmaCreateImage(VmaAllocator, const VkImageCreateInfo*, const VmaAllocationCreateInfo*, VkImage* image, VmaAllocation* allocation, VmaAllocationInfo* info) {
    *image = fake_handle<VkImage>(0xa300);
    *allocation = fake_handle<VmaAllocation>(0xa301);
    if (info) { *info = {}; info->size = 128; }
    return VK_SUCCESS;
}
void vmaDestroyImage(VmaAllocator, VkImage, VmaAllocation) { ++vma_image_destroys; }

int main() {
    volkInitializeCustom(fake_get_instance_proc);

    int instance_destroys = 0;
    int device_destroys = 0;
    int buffer_destroys = 0;
    int replacement_destroys = 0;

    auto instance = vk::Instance::adopt(
        fake_handle<VkInstance>(0x1000),
        [&](VkInstance) noexcept { ++instance_destroys; });
    assert(instance);
    auto physical = vk::PhysicalDevice::borrow(fake_handle<VkPhysicalDevice>(0x2000), *instance);
    assert(physical && physical->use_count() == 0);

    // Writable pNext nodes round-trip through the owned chain and remain
    // independently retrievable after copying the top-level structure.
    private_feature_supported = true;
    vk::PhysicalDeviceFeatures2 queried_features{};
    queried_features.setNextInChain(vk::PhysicalDevicePrivateDataFeatures{});
    physical->getFeatures2(&queried_features);
    auto* queried_private = queried_features.nextInChain.get<vk::PhysicalDevicePrivateDataFeatures>();
    assert(queried_private && queried_private->privateData == VK_TRUE);
    vk::PhysicalDeviceFeatures2 copied_features = queried_features;
    private_feature_supported = false;
    physical->getFeatures2(&copied_features);
    auto* copied_private = copied_features.nextInChain.get<vk::PhysicalDevicePrivateDataFeatures>();
    assert(copied_private && copied_private->privateData == VK_FALSE);
    assert(queried_private->privateData == VK_TRUE);

    enumerate_calls = 0;
    enumerate_force_retry = true;
    auto all_physical = instance->enumeratePhysicalDevices();
    assert(all_physical.status == vk::ResultCode::Success);
    assert(all_physical.value.size() == 2 && enumerate_calls == 4);
    // parent()/dispatchState()/deviceAssociation() are private plumbing now;
    // parent linkage is exercised through the borrow/adopt and teardown paths.
    enumerate_calls = 0;
    enumerate_force_retry = false;
    auto limited_physical = instance->enumeratePhysicalDevices(1);
    assert(limited_physical.status == vk::ResultCode::Incomplete);
    assert(limited_physical.value.size() == 1 && enumerate_calls == 1);
    all_physical.value.clear();
    limited_physical.value.clear();

    // Device creation requires privateData support, enables it in a cloned
    // native feature chain, and reserves one wrapper slot without modifying
    // the caller's owned structure record.
    vk::PhysicalDevicePrivateDataFeatures requested_private_data{};
    requested_private_data.setPrivateData(VK_FALSE);
    vk::DevicePrivateDataCreateInfo requested_private_slots{};
    requested_private_slots.setPrivateDataSlotRequestCount(3);
    vk::DeviceCreateInfo generated_device_info{};
    generated_device_info.setNextInChain(requested_private_data);
    generated_device_info.nextInChain.get<vk::PhysicalDevicePrivateDataFeatures>()
        ->setNextInChain(requested_private_slots);
    vk::DeviceQueueCreateInfo generated_queue_info{};
    generated_queue_info.setQueueFamilyIndex(0).setQueuePriorities({1.0f});
    generated_device_info.setQueueCreateInfos({generated_queue_info});
    vk::DeviceCreateInfo copied_device_info = generated_device_info;
    vk::DeviceCreateInfo::CStruct copied_native{};
    copied_device_info.to_cstruct(&copied_native);
    auto* copied_feature = reinterpret_cast<const VkPhysicalDevicePrivateDataFeatures*>(copied_native.value.pNext);
    assert(copied_feature && copied_feature->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIVATE_DATA_FEATURES);
    auto* copied_slots = reinterpret_cast<const VkDevicePrivateDataCreateInfo*>(copied_feature->pNext);
    assert(copied_slots && copied_slots->sType == VK_STRUCTURE_TYPE_DEVICE_PRIVATE_DATA_CREATE_INFO);
    assert(copied_slots->privateDataSlotRequestCount == 3 && copied_slots->pNext == nullptr);
    private_feature_supported = false;
    auto unsupported_device = physical->createDevice(generated_device_info, {});
    assert(!unsupported_device && unsupported_device.error() == vk::ResultCode::ErrorFeatureNotPresent);
    assert(create_device_calls == 0);
    private_feature_supported = true;
    auto generated_device = physical->createDevice(generated_device_info, {});
    assert(generated_device && create_device_calls == 1);
    assert(create_device_saw_private_feature && create_device_saw_slot_request);
    vk::DeviceCreateInfo::CStruct unchanged_native{};
    generated_device_info.to_cstruct(&unchanged_native);
    auto* unchanged_feature = reinterpret_cast<const VkPhysicalDevicePrivateDataFeatures*>(unchanged_native.value.pNext);
    assert(unchanged_feature && unchanged_feature->privateData == VK_FALSE);
    // Queues are device-owned: the device creation pre-registers a control
    // block per queue, so getQueue returns a tracked handle with a stable id
    // (not a bare borrowed handle with use_count() == 0).
    auto generated_queue = generated_device->getQueue(0, 0);
    assert(generated_queue && generated_queue.use_count() > 0);
    auto generated_queue_again = generated_device->getQueue(0, 0);
    assert(generated_queue_again && generated_queue_again.id() == generated_queue.id());
    generated_queue.reset();
    generated_queue_again.reset();
    generated_device->reset();
    assert(generated_device_destroys == 1);

    // Chaining the promoted 1.3 feature struct must enable privateData on it
    // directly instead of injecting a conflicting VkPhysicalDevicePrivateDataFeatures
    // node (VUID-VkDeviceCreateInfo-pNext-06532).
    vk::PhysicalDeviceVulkan13Features v13_features{};
    v13_features.setPrivateData(VK_FALSE);
    vk::DeviceCreateInfo v13_device_info{};
    v13_device_info.setNextInChain(v13_features);
    auto v13_device = physical->createDevice(v13_device_info, {});
    assert(v13_device && create_device_calls == 2);
    assert(create_device_saw_vulkan13_private);
    assert(!create_device_saw_private_data_features);
    v13_device->reset();
    assert(generated_device_destroys == 2);
    const int initial_slot_creates = slot_creates;
    const int initial_slot_destroys = slot_destroys;

    // A borrowed device remains allocation-free until metadata or ownership
    // actually needs a control block and private-data slot.
    auto lightweight_device = vk::Device::borrow(fake_handle<VkDevice>(0x3000), *physical);
    assert(lightweight_device && lightweight_device->use_count() == 0);
    assert(slot_creates == initial_slot_creates);

    // A borrowed device cannot publish child associations; adopt the device
    // to own its private-data slot before adopting a child under it.
    int lightweight_device_destroys = 0;
    auto owned_device = vk::Device::adopt(
        fake_handle<VkDevice>(0x3000), *physical,
        [&](VkDevice) noexcept { ++lightweight_device_destroys; });
    assert(owned_device && owned_device->use_count() == 1);
    assert(slot_creates == initial_slot_creates + 1);
    int lightweight_child_destroys = 0;
    auto lightweight_child = vk::Buffer::adopt(
        fake_handle<VkBuffer>(0x3050), *owned_device,
        [&](VkBuffer) noexcept { ++lightweight_child_destroys; }, {});
    assert(lightweight_child);
    lightweight_child->reset();
    assert(lightweight_child_destroys == 1);
    owned_device->reset();
    assert(lightweight_device_destroys == 1);
    assert(slot_destroys == initial_slot_destroys + 1);
    lightweight_device->reset();

    auto device = vk::Device::adopt(
        fake_handle<VkDevice>(0x3000), *physical,
        [&](VkDevice) noexcept { ++device_destroys; });
    assert(device && device->use_count() == 1);
    assert(slot_creates == initial_slot_creates + 2);

    // A command whose loader slot was never populated (the extension is not
    // "enabled" in the fake loader) must report ErrorExtensionNotPresent
    // instead of calling through a null dispatch slot.
    auto missing_device_command = device->createBuffer(vk::BufferCreateInfo{}, std::nullopt);
    assert(!missing_device_command);
    assert(missing_device_command.error() == vk::ResultCode::ErrorExtensionNotPresent);

    int queue_finalizes = 0;
    auto queue = vk::Queue::adopt(fake_handle<VkQueue>(0x3800), *device,
        [&](VkQueue) noexcept { ++queue_finalizes; });
    assert(queue && queue->use_count() == 1);
    std::vector<std::thread> submitters;
    for (int worker = 0; worker != 8; ++worker) {
        submitters.emplace_back([queue = *queue] {
            for (int call = 0; call != 8; ++call) assert(queue.submit({}, {}));
        });
    }
    for (auto& submitter : submitters) submitter.join();
    submitters.clear();
    assert(queue_submit_calls.load() == 64);
    assert(!queue_submit_overlap.load());
    queue->reset();
    assert(queue_finalizes == 1);

    // Borrowed children are allocation-free and cannot hold metadata. Only
    // owned handles store typed user data in their single control block.
    auto pure_borrow = vk::Buffer::borrow(fake_handle<VkBuffer>(0x4100), *device);
    assert(pure_borrow && pure_borrow->use_count() == 0);
    assert(!pure_borrow->setData<int>(std::make_shared<const int>(9)));
    int observer_destroys = 0;
    auto observer = vk::Buffer::adopt(
        fake_handle<VkBuffer>(0x4100), *device,
        [&](VkBuffer) noexcept { ++observer_destroys; }, {});
    assert(observer && observer->use_count() == 1);
    assert(observer->setData<int>(std::make_shared<const int>(7)));
    assert(*observer->getData<int>() == 7);
    observer->reset();
    assert(observer_destroys == 1);
    pure_borrow->reset();

    // Type-erased externsync lookup must pin the tracked object's mutex before
    // final release clears private data and destroys the native handle.
    std::atomic_int debug_buffer_destroys{0};
    auto debug_buffer = vk::Buffer::adopt(
        fake_handle<VkBuffer>(0x4150), *device,
        [&](VkBuffer) noexcept { ++debug_buffer_destroys; }, {});
    assert(debug_buffer);
    vk::DebugUtilsObjectNameInfoEXT debug_name{};
    debug_name.setObjectType(vk::ObjectType::Buffer)
              .setObjectHandle(fake_key(debug_buffer->raw()))
              .setObjectName("tracked");
    debug_name_entered.store(false, std::memory_order_release);
    release_debug_name.store(false, std::memory_order_release);
    std::thread naming([&] { assert(device->setDebugUtilsObjectNameEXT(debug_name)); });
    while (!debug_name_entered.load(std::memory_order_acquire)) std::this_thread::yield();
    std::atomic_bool reset_finished{false};
    std::thread debug_reset([&] {
        debug_buffer->reset();
        reset_finished.store(true, std::memory_order_release);
    });
    for (int spin = 0; spin != 10000; ++spin) std::this_thread::yield();
    assert(!reset_finished.load(std::memory_order_acquire));
    assert(debug_buffer_destroys.load(std::memory_order_acquire) == 0);
    release_debug_name.store(true, std::memory_order_release);
    naming.join();
    debug_reset.join();
    assert(reset_finished.load(std::memory_order_acquire));
    assert(debug_buffer_destroys.load(std::memory_order_acquire) == 1);

    // Heterogeneous handle creation records retain the exact concrete
    // producer alternative for each indexed output.
    std::vector<vk::GraphicsPipelineCreateInfo> graphics_infos{
        vk::GraphicsPipelineCreateInfo{}.setFlags(vk::PipelineCreateFlags{}),
        vk::GraphicsPipelineCreateInfo{}.setFlags(vk::PipelineCreateFlags{})};
    auto graphics = device->createGraphicsPipelines({}, graphics_infos, {});
    assert(graphics.status == vk::ResultCode::Success && graphics.value.size() == 2);
    for (const auto& pipeline : graphics.value) {
        assert(pipeline.createInfo());
        assert(std::get_if<vk::GraphicsPipelineCreateInfo>(&pipeline.createInfo()->value));
    }
    graphics.value.clear();
    assert(pipeline_destroys == 2);

    // Descriptor sets are individually owned only when their pool permits
    // vkFreeDescriptorSets. Otherwise the lightweight wrapper retains the
    // pool and lets pool destruction release the native set.
    int descriptor_pool_destroys = 0;
    auto unflagged_pool_info = std::make_shared<const vk::DescriptorPoolCreateInfo>(
        vk::DescriptorPoolCreateInfo{}.setMaxSets(1));
    auto unflagged_pool = vk::DescriptorPool::adopt(
        fake_handle<VkDescriptorPool>(0xc100), *device,
        [&](VkDescriptorPool) noexcept { ++descriptor_pool_destroys; },
        unflagged_pool_info);
    auto descriptor_layout = vk::DescriptorSetLayout::borrow(
        fake_handle<VkDescriptorSetLayout>(0xc200), *device);
    assert(unflagged_pool && descriptor_layout);
    vk::DescriptorSetAllocateInfo unflagged_allocate{};
    unflagged_allocate.setDescriptorPool(*unflagged_pool)
                      .setSetLayouts(std::vector<vk::DescriptorSetLayout>{*descriptor_layout});
    auto unflagged_sets = device->allocateDescriptorSets(unflagged_allocate);
    assert(unflagged_sets && unflagged_sets->size() == 1);
    assert((*unflagged_sets)[0].use_count() == 0);
    unflagged_sets->clear();
    assert(descriptor_set_frees == 0);
    unflagged_allocate = {};
    unflagged_pool->reset();
    assert(descriptor_pool_destroys == 1);

    auto flagged_pool_info = std::make_shared<const vk::DescriptorPoolCreateInfo>(
        vk::DescriptorPoolCreateInfo{}
            .setFlags(vk::DescriptorPoolCreateFlagBits::FreeDescriptorSet)
            .setMaxSets(1));
    auto flagged_pool = vk::DescriptorPool::adopt(
        fake_handle<VkDescriptorPool>(0xc101), *device,
        [&](VkDescriptorPool) noexcept { ++descriptor_pool_destroys; },
        flagged_pool_info);
    assert(flagged_pool);
    vk::DescriptorSetAllocateInfo flagged_allocate{};
    flagged_allocate.setDescriptorPool(*flagged_pool)
                    .setSetLayouts(std::vector<vk::DescriptorSetLayout>{*descriptor_layout});
    auto flagged_sets = device->allocateDescriptorSets(flagged_allocate);
    assert(flagged_sets && flagged_sets->size() == 1);
    assert((*flagged_sets)[0].use_count() == 1);
    flagged_sets->clear();
    assert(descriptor_set_frees == 1);
    flagged_allocate = {};
    flagged_pool->reset();
    assert(descriptor_pool_destroys == 2);
    descriptor_layout->reset();

    vk::setDestructionErrorSink(capture_destruction_error);
    auto throwing_buffer = vk::Buffer::adopt(
        fake_handle<VkBuffer>(0x4200), *device,
        [](VkBuffer) { throw 7; }, {});
    assert(throwing_buffer);
    throwing_buffer->reset();
    assert(destruction_errors == 1);
    assert(destruction_error_code == vk::ResultCode::ErrorUnknown);
    assert(destruction_error_type == "Buffer");
    assert(destruction_error_identity == fake_key(fake_handle<VkBuffer>(0x4200)));
    vk::setDestructionErrorSink(nullptr);

    auto record = std::make_shared<const vk::BufferCreateInfo>(
        vk::BufferCreateInfo{}.setSize(64));
    auto buffer = vk::Buffer::adopt(
        fake_handle<VkBuffer>(0x4000), *device,
        [&](VkBuffer) noexcept { ++buffer_destroys; }, record);
    assert(buffer && buffer->createInfo() && buffer->createInfo()->size == 64);
    record.reset();

    auto copy = *buffer;
    auto borrowed = vk::Buffer::borrow(fake_handle<VkBuffer>(0x4000), *device);
    auto adopted_again = vk::Buffer::adopt(
        fake_handle<VkBuffer>(0x4000), *device,
        [&](VkBuffer) noexcept { ++replacement_destroys; }, {});
    assert(borrowed && adopted_again);
    assert(buffer->use_count() == 4);
    assert(*buffer == *borrowed);
    assert(*buffer == *adopted_again);

    instance->reset();
    device->reset();
    assert(instance_destroys == 0 && device_destroys == 0);
    buffer->reset();
    copy.reset();
    borrowed->reset();
    assert(buffer_destroys == 0);
    adopted_again->reset();
    assert(buffer_destroys == 1 && replacement_destroys == 0);
    assert(device_destroys == 1 && slot_destroys == initial_slot_destroys + 2);
    physical->reset();
    assert(instance_destroys == 1);
    assert(private_values.empty());

    // Host-tracked non-dispatchable values are scoped by their complete
    // parent lineage. Equal raw surface values from distinct instances must
    // never reuse one another's ownership block.
    int scoped_instance_destroys = 0;
    int first_surface_destroys = 0;
    int second_surface_destroys = 0;
    auto first_instance = vk::Instance::adopt(
        fake_handle<VkInstance>(0x7000),
        [&](VkInstance) noexcept { ++scoped_instance_destroys; });
    auto second_instance = vk::Instance::adopt(
        fake_handle<VkInstance>(0x7001),
        [&](VkInstance) noexcept { ++scoped_instance_destroys; });
    assert(first_instance && second_instance);
    const auto repeated_surface = fake_handle<VkSurfaceKHR>(0x7100);
    auto first_surface = vk::SurfaceKHR::adopt(
        repeated_surface, *first_instance,
        [&](VkSurfaceKHR) noexcept { ++first_surface_destroys; });
    auto second_surface = vk::SurfaceKHR::adopt(
        repeated_surface, *second_instance,
        [&](VkSurfaceKHR) noexcept { ++second_surface_destroys; });
    assert(first_surface && second_surface);
    assert(*first_surface != *second_surface);
    auto first_surface_again = vk::SurfaceKHR::borrow(repeated_surface, *first_instance);
    auto second_surface_again = vk::SurfaceKHR::borrow(repeated_surface, *second_instance);
    assert(first_surface_again && second_surface_again);
    assert(*first_surface_again == *first_surface);
    assert(*second_surface_again == *second_surface);
    first_instance->reset();
    second_instance->reset();
    first_surface->reset();
    second_surface->reset();
    assert(first_surface_destroys == 0 && second_surface_destroys == 0);
    first_surface_again->reset();
    second_surface_again->reset();
    assert(first_surface_destroys == 1 && second_surface_destroys == 1);
    assert(scoped_instance_destroys == 2);

    // Device dispatchable values are normally unique, but the host lookup
    // must still include the PhysicalDevice lineage.  This also protects fake
    // loaders and handle-value reuse after destruction.
    int device_scope_instance_destroys = 0;
    int first_scoped_device_destroys = 0;
    int second_scoped_device_destroys = 0;
    auto device_scope_first_instance = vk::Instance::adopt(
        fake_handle<VkInstance>(0x7200),
        [&](VkInstance) noexcept { ++device_scope_instance_destroys; });
    auto device_scope_second_instance = vk::Instance::adopt(
        fake_handle<VkInstance>(0x7201),
        [&](VkInstance) noexcept { ++device_scope_instance_destroys; });
    auto device_scope_first_physical = vk::PhysicalDevice::borrow(
        fake_handle<VkPhysicalDevice>(0x7210), *device_scope_first_instance);
    auto device_scope_second_physical = vk::PhysicalDevice::borrow(
        fake_handle<VkPhysicalDevice>(0x7211), *device_scope_second_instance);
    const auto repeated_device = fake_handle<VkDevice>(0x7220);
    auto first_scoped_device = vk::Device::adopt(
        repeated_device, *device_scope_first_physical,
        [&](VkDevice) noexcept { ++first_scoped_device_destroys; });
    auto second_scoped_device = vk::Device::adopt(
        repeated_device, *device_scope_second_physical,
        [&](VkDevice) noexcept { ++second_scoped_device_destroys; });
    assert(first_scoped_device && second_scoped_device);
    assert(*first_scoped_device != *second_scoped_device);
    auto first_scoped_device_again = vk::Device::borrow(
        repeated_device, *device_scope_first_physical);
    auto second_scoped_device_again = vk::Device::borrow(
        repeated_device, *device_scope_second_physical);
    assert(first_scoped_device_again && second_scoped_device_again);
    assert(*first_scoped_device_again == *first_scoped_device);
    assert(*second_scoped_device_again == *second_scoped_device);
    device_scope_first_instance->reset();
    device_scope_second_instance->reset();
    device_scope_first_physical->reset();
    device_scope_second_physical->reset();
    first_scoped_device->reset();
    second_scoped_device->reset();
    assert(first_scoped_device_destroys == 0 && second_scoped_device_destroys == 0);
    first_scoped_device_again->reset();
    second_scoped_device_again->reset();
    assert(first_scoped_device_destroys == 1 && second_scoped_device_destroys == 1);
    assert(device_scope_instance_destroys == 2);

    // Concurrent private-data lookup/copy/reset must never retain a block
    // after its final detach and must still destroy exactly once.
    auto race_instance = vk::Instance::adopt(
        fake_handle<VkInstance>(0x6000), [](VkInstance) noexcept {});
    auto race_physical = vk::PhysicalDevice::borrow(
        fake_handle<VkPhysicalDevice>(0x6100), *race_instance);
    auto race_device = vk::Device::adopt(
        fake_handle<VkDevice>(0x6200), *race_physical,
        [](VkDevice) noexcept {});
    std::atomic_int race_buffer_destroys{0};
    auto race_buffer = vk::Buffer::adopt(
        fake_handle<VkBuffer>(0x6300), *race_device,
        [&](VkBuffer) noexcept { ++race_buffer_destroys; }, {});
    assert(race_buffer);

    // Standalone allocations and VMA resources retain their allocator. Their
    // metadata remains separate from the concrete Vulkan createInfo getter,
    // and each path chooses exactly one matching VMA destructor.
    VmaAllocatorCreateInfo allocator_info{};
    auto allocator = vk::Allocator::create(*race_device, allocator_info);
    assert(allocator && allocator->use_count() == 1);
    auto allocation = allocator->allocate(
        vk::MemoryRequirements{}.setSize(32).setAlignment(8).setMemoryTypeBits(1),
        VmaAllocationCreateInfo{});
    assert(allocation);
    auto mapped = allocation->view().map();
    assert(mapped && *mapped == mapped_bytes && vma_maps == 1);
    assert(allocation->view().flush() && allocation->view().invalidate());
    allocation->view().unmap();
    assert(vma_flushes == 1 && vma_invalidates == 1 && vma_unmaps == 1);
    assert(allocation->view().information().size == 32);

    VmaAllocationCreateInfo resource_allocation_info{};
    resource_allocation_info.usage = VMA_MEMORY_USAGE_AUTO;
    auto vma_buffer = allocator->createBuffer(
        vk::BufferCreateInfo{}.setSize(64).setUsage(vk::BufferUsageFlagBits::TransferSrc),
        resource_allocation_info);
    auto vma_image = allocator->createImage(vk::ImageCreateInfo{}, resource_allocation_info);
    assert(vma_buffer && vma_image);
    assert(vma_buffer->createInfo() && vma_buffer->createInfo()->size == 64);
    assert(vma_buffer->allocation() != VmaAllocation{} && vma_buffer->allocationInfo()->size == 64);
    assert(vma_buffer->allocationCreateInfo()->usage == VMA_MEMORY_USAGE_AUTO);
    allocator->reset();
    assert(vma_allocator_destroys == 0);
    allocation->reset();
    assert(vma_allocation_frees == 1 && vma_allocator_destroys == 0);
    vma_buffer->reset();
    assert(vma_buffer_destroys == 1 && vma_allocator_destroys == 0);
    vma_image->reset();
    assert(vma_image_destroys == 1 && vma_allocator_destroys == 1);
    std::atomic_int ready{0};
    std::atomic_bool release_workers{false};
    std::vector<std::thread> workers;
    for (int worker = 0; worker != 8; ++worker) {
        workers.emplace_back([&] {
            auto held = vk::Buffer::borrow(fake_handle<VkBuffer>(0x6300), *race_device);
            assert(held && held->use_count() > 0);
            ready.fetch_add(1, std::memory_order_release);
            while (!release_workers.load(std::memory_order_acquire)) std::this_thread::yield();
            held->reset();
            for (int iteration = 0; iteration != 2000; ++iteration) {
                auto found = vk::Buffer::borrow(fake_handle<VkBuffer>(0x6300), *race_device);
                assert(found);
                auto copy = *found;
                assert(copy.raw() == fake_handle<VkBuffer>(0x6300));
            }
        });
    }
    while (ready.load(std::memory_order_acquire) != 8) std::this_thread::yield();
    race_buffer->reset();
    release_workers.store(true, std::memory_order_release);
    for (auto& worker : workers) worker.join();
    assert(race_buffer_destroys.load() == 1);
    auto after_detach = vk::Buffer::borrow(fake_handle<VkBuffer>(0x6300), *race_device);
    assert(after_detach && after_detach->use_count() == 0);
    after_detach->reset();
    race_device->reset();
    race_physical->reset();
    race_instance->reset();

    // Device setup rollback destroys its newly-created slot and native device
    // exactly once when publishing the association fails.
    auto rollback_instance = vk::Instance::adopt(
        fake_handle<VkInstance>(0x5000), [](VkInstance) noexcept {});
    auto rollback_physical = vk::PhysicalDevice::borrow(
        fake_handle<VkPhysicalDevice>(0x5100), *rollback_instance);
    int rollback_device_destroys = 0;
    const int slot_creates_before_device_rollback = slot_creates;
    const int slot_destroys_before_device_rollback = slot_destroys;
    reject_private_data = true;
    auto failed_device = vk::Device::adopt(
        fake_handle<VkDevice>(0x5200), *rollback_physical,
        [&](VkDevice) noexcept { ++rollback_device_destroys; });
    reject_private_data = false;
    assert(!failed_device);
    assert(rollback_device_destroys == 1);
    assert(slot_creates == slot_creates_before_device_rollback + 1);
    assert(slot_destroys == slot_destroys_before_device_rollback + 1);

    // Child publication failure likewise calls only the offered deleter and
    // leaves no private-data entry behind.
    auto rollback_device = vk::Device::adopt(
        fake_handle<VkDevice>(0x5300), *rollback_physical,
        [](VkDevice) noexcept {});
    assert(rollback_device);
    int rollback_buffer_destroys = 0;
    reject_private_data = true;
    auto failed_buffer = vk::Buffer::adopt(
        fake_handle<VkBuffer>(0x5400), *rollback_device,
        [&](VkBuffer) noexcept { ++rollback_buffer_destroys; }, {});
    reject_private_data = false;
    assert(!failed_buffer && rollback_buffer_destroys == 1);
    assert(private_values.size() == 1); // rollback device association only
    rollback_device->reset();
    rollback_physical->reset();
    rollback_instance->reset();
    assert(private_values.empty());

    // Callback struct fields serialize to a trampoline + bundle userdata and
    // round-trip the captured callable through the raw function pointer.
    vk::DebugUtilsMessengerCreateInfoEXT callback_info{};
    int callback_calls = 0;
    callback_info.setUserCallback([&callback_calls](
        VkDebugUtilsMessageSeverityFlagBitsEXT severity,
        VkDebugUtilsMessageTypeFlagsEXT,
        const VkDebugUtilsMessengerCallbackDataEXT* data) -> VkBool32 {
        ++callback_calls;
        return severity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT
                   ? VK_FALSE : VK_TRUE;
    });
    vk::DebugUtilsMessengerCreateInfoEXT::CStruct callback_native{};
    callback_info.to_cstruct(&callback_native);
    assert(callback_native.value.pfnUserCallback != nullptr);
    assert(callback_native.value.pUserData != nullptr);
    auto result = callback_native.value.pfnUserCallback(
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT,
        nullptr, callback_native.value.pUserData);
    assert(callback_calls == 1 && result == VK_FALSE);
    // Vulkan callbacks are persistent, not oneshot: a second invocation fires
    // the same captured callable again.
    result = callback_native.value.pfnUserCallback(
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT,
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT,
        nullptr, callback_native.value.pUserData);
    assert(callback_calls == 2 && result == VK_TRUE);

    // Multi-callback struct sharing one pUserData (VkAllocationCallbacks):
    // userdata sits first in the native signature, returns differ, and every
    // trampoline routes to its own callable in the shared bundle.
    vk::AllocationCallbacks alloc_callbacks{};
    int alloc_calls = 0;
    int free_calls = 0;
    auto* alloc_result = reinterpret_cast<void*>(0xdead);
    alloc_callbacks.setAllocation([&](size_t size, size_t alignment, VkSystemAllocationScope scope) -> void* {
        assert(size == 64 && alignment == 8 && scope == VK_SYSTEM_ALLOCATION_SCOPE_COMMAND);
        ++alloc_calls;
        return alloc_result;
    });
    alloc_callbacks.setFree([&](void* memory) {
        assert(memory == alloc_result);
        ++free_calls;
    });
    vk::AllocationCallbacks::CStruct alloc_native{};
    alloc_callbacks.to_cstruct(&alloc_native);
    assert(alloc_native.value.pUserData != nullptr);
    assert(alloc_native.value.pfnAllocation != nullptr && alloc_native.value.pfnFree != nullptr);
    auto* memory = alloc_native.value.pfnAllocation(
        alloc_native.value.pUserData, 64, 8, VK_SYSTEM_ALLOCATION_SCOPE_COMMAND);
    assert(memory == alloc_result && alloc_calls == 1);
    alloc_native.value.pfnFree(alloc_native.value.pUserData, alloc_result);
    assert(free_calls == 1);
}
""",
        encoding="utf-8",
    )

    executable = tmp_path / ("runtime.exe" if os.name == "nt" else "runtime")
    command = [
        compiler,
        "-std=c++23",
        "-O0",
        "-I",
        str(tmp_path),
        "-I",
        str(volk),
        "-I",
        str(vma / "include"),
        "-I",
        str(headers / "include"),
        str(source),
        "-o",
        str(executable),
    ]
    if os.name != "nt":
        command.append("-ldl")
    compiled = subprocess.run(command, text=True, capture_output=True)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    executed = subprocess.run([str(executable)], text=True, capture_output=True)
    assert executed.returncode == 0, executed.stdout + executed.stderr


def test_externsync_disabled_generates_a_working_wrapper(tmp_path: Path):
    compiler = _compiler()
    headers = _dependency("Vulkan-Headers")
    volk = _dependency("volk")
    if not compiler or not headers or not volk:
        pytest.skip("runtime C++ test needs a compiler, Vulkan-Headers, and Volk")

    generated = tmp_path / "vulkan_wrapper.hpp"
    assert (
        run(
            [
                "--registry",
                str(headers / "registry" / "vk.xml"),
                "--no-externsync",
                "--emit", str(ROOT / "templates" / "vulkan-header-only.template.hpp") + ":" + str(generated),
            ]
        )
        == 0
    )
    text = generated.read_text(encoding="utf-8")
    assert "externsync_states" not in text

    source = tmp_path / "no_sync.cpp"
    source.write_text(
        r"""
#define VOLK_IMPLEMENTATION
#include <volk.h>
#include "vulkan_wrapper.hpp"
#include <cassert>

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL fake_get_instance_proc(VkInstance, const char*) {
    return nullptr;
}

int main() {
    volkInitializeCustom(fake_get_instance_proc);
    int destroyed = 0;
    auto instance = vk::Instance::adopt(
        reinterpret_cast<VkInstance>(0x1000),
        [&](VkInstance) noexcept { ++destroyed; });
    assert(instance && destroyed == 0);
    auto copy = *instance;
    assert(instance->use_count() == 2 && destroyed == 0);
    instance->reset();
    assert(destroyed == 0);
    copy.reset();
    assert(destroyed == 1);
    return 0;
}
""",
        encoding="utf-8",
    )

    executable = tmp_path / ("no_sync.exe" if os.name == "nt" else "no_sync")
    command = [
        compiler,
        "-std=c++23",
        "-O0",
        "-I",
        str(tmp_path),
        "-I",
        str(volk),
        "-I",
        str(headers / "include"),
        str(source),
        "-o",
        str(executable),
    ]
    if os.name != "nt":
        command.append("-ldl")
    compiled = subprocess.run(command, text=True, capture_output=True)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    executed = subprocess.run([str(executable)], text=True, capture_output=True)
    assert executed.returncode == 0, executed.stdout + executed.stderr
