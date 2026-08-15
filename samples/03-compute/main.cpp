// 03-compute: a data-parallel compute kernel mapped onto the wrapper.
//
// This mirrors the official compute samples (compute-only, no window): it
// uploads an input buffer, dispatches a shader that writes output[i] =
// input[i] * 2, then reads the output back on the CPU and verifies every
// element. It exercises descriptor sets/pools/layouts, storage buffers,
// compute pipelines, dispatch, and buffer barriers.
#include <volk.h>
#include <vulkan_wrapper.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <print>
#include <vector>

static constexpr std::uint32_t kCount = 1024;
static constexpr std::uint32_t kLocalSize = 64;

static std::vector<std::uint32_t> read_spirv(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return {};
    std::fseek(f, 0, SEEK_END);
    long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<std::uint32_t> code(size / sizeof(std::uint32_t));
    std::fread(code.data(), 1, size, f);
    std::fclose(f);
    return code;
}

static std::uint32_t find_memory_type(const vk::PhysicalDevice& physical,
                                      vk::MemoryPropertyFlags required) {
    auto props = physical.getMemoryProperties();
    for (std::uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if ((props.memoryTypes[i].propertyFlags & required) == required) return i;
    }
    return 0;
}

static std::uint32_t find_compute_family(const vk::PhysicalDevice& physical) {
    auto families = physical.getQueueFamilyProperties();
    for (std::uint32_t i = 0; i < families.size(); ++i)
        if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) return i;
    return 0;
}

int main() {
    if (volkInitialize() != VK_SUCCESS) { std::println(stderr, "volkInitialize failed"); return 1; }
    auto context = vk::Context::create();
    if (!context) { std::println(stderr, "Context::create failed"); return 1; }

    vk::ApplicationInfo app{};
    app.setApplicationName("vulkan-wrapper-compute");
    app.setApiVersion(VK_API_VERSION_1_3);
    vk::InstanceCreateInfo instanceInfo{};
    instanceInfo.setApplicationInfo(app);
    auto instance = context->createInstance(instanceInfo, std::nullopt);
    if (!instance) { std::println(stderr, "createInstance failed"); return 1; }

    auto devices = instance->enumeratePhysicalDevices();
    if (devices.value.empty()) { std::println(stderr, "no physical devices"); return 1; }
    auto physical = std::move(devices.value[0]);
    auto props = physical.getProperties();
    std::println("Using GPU: {}", props.deviceName.data());

    std::uint32_t queueFamily = find_compute_family(physical);
    vk::DeviceQueueCreateInfo queueInfo{};
    queueInfo.setQueueFamilyIndex(queueFamily).setQueuePriorities({1.0f});
    vk::DeviceCreateInfo deviceInfo{};
    deviceInfo.setQueueCreateInfos({queueInfo});
    auto device = physical.createDevice(deviceInfo, std::nullopt);
    if (!device) { std::println(stderr, "createDevice failed"); return 1; }
    auto queue = device->getQueue(queueFamily, 0);

    auto code = read_spirv(SHADER_COMP);
    vk::ShaderModuleCreateInfo moduleInfo{};
    moduleInfo.setCode(std::move(code));
    auto shader = device->createShaderModule(moduleInfo, std::nullopt);
    if (!shader) { std::println(stderr, "createShaderModule failed"); return 1; }

    // Descriptor set layout: two storage buffers (read-only in, writable out).
    vk::DescriptorSetLayoutBinding inBinding{};
    inBinding.setBinding(0).setDescriptorType(vk::DescriptorType::StorageBuffer)
              .setDescriptorCount(1).setStageFlags(vk::ShaderStageFlagBits::Compute);
    vk::DescriptorSetLayoutBinding outBinding{};
    outBinding.setBinding(1).setDescriptorType(vk::DescriptorType::StorageBuffer)
               .setDescriptorCount(1).setStageFlags(vk::ShaderStageFlagBits::Compute);
    vk::DescriptorSetLayoutCreateInfo setLayoutInfo{};
    setLayoutInfo.setBindings({inBinding, outBinding});
    auto setLayout = device->createDescriptorSetLayout(setLayoutInfo, std::nullopt);
    if (!setLayout) { std::println(stderr, "createDescriptorSetLayout failed"); return 1; }

    vk::PipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.setSetLayouts({*setLayout});
    auto layout = device->createPipelineLayout(layoutInfo, std::nullopt);
    if (!layout) { std::println(stderr, "createPipelineLayout failed"); return 1; }

    vk::PipelineShaderStageCreateInfo stage{};
    stage.setStage(vk::ShaderStageFlagBits::Compute).setModule(*shader).setName("main");
    vk::ComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.setStage(stage).setLayout(*layout);
    auto pipelines = device->createComputePipelines(
        vk::PipelineCache{}, std::array{pipelineInfo}, std::nullopt);
    if (pipelines.status != vk::ResultCode::Success || pipelines.value.empty()) {
        std::println(stderr, "createComputePipelines failed: {}", static_cast<int>(pipelines.status));
        return 1;
    }
    auto pipeline = std::move(pipelines.value[0]);

    // Storage buffers (host-visible for both upload and readback).
    const vk::DeviceSize size = static_cast<vk::DeviceSize>(kCount * sizeof(float));
    const auto hostFlags = vk::MemoryPropertyFlagBits::HostVisible | vk::MemoryPropertyFlagBits::HostCoherent;
    auto inputBuffer = device->createBuffer(
        vk::BufferCreateInfo{}.setSize(size).setUsage(vk::BufferUsageFlagBits::StorageBuffer)
            .setSharingMode(vk::SharingMode::Exclusive), std::nullopt);
    auto outputBuffer = device->createBuffer(
        vk::BufferCreateInfo{}.setSize(size).setUsage(vk::BufferUsageFlagBits::StorageBuffer)
            .setSharingMode(vk::SharingMode::Exclusive), std::nullopt);
    if (!inputBuffer || !outputBuffer) { std::println(stderr, "createBuffer failed"); return 1; }

    auto inputReqs = device->getBufferMemoryRequirements(*inputBuffer);
    auto outputReqs = device->getBufferMemoryRequirements(*outputBuffer);
    auto inputMemory = device->allocateMemory(
        vk::MemoryAllocateInfo{}.setAllocationSize(inputReqs.size)
            .setMemoryTypeIndex(find_memory_type(physical, hostFlags)), std::nullopt);
    auto outputMemory = device->allocateMemory(
        vk::MemoryAllocateInfo{}.setAllocationSize(outputReqs.size)
            .setMemoryTypeIndex(find_memory_type(physical, hostFlags)), std::nullopt);
    if (!inputMemory || !outputMemory ||
        !device->bindBufferMemory(*inputBuffer, *inputMemory, 0) ||
        !device->bindBufferMemory(*outputBuffer, *outputMemory, 0)) {
        std::println(stderr, "buffer memory setup failed"); return 1;
    }

    void* mapped = nullptr;
    if (!device->mapMemory(*inputMemory, 0, size, vk::MemoryMapFlags{}, &mapped)) {
        std::println(stderr, "mapMemory (input) failed"); return 1;
    }
    auto* input = static_cast<float*>(mapped);
    for (std::uint32_t i = 0; i < kCount; ++i) input[i] = static_cast<float>(i);
    device->unmapMemory(*inputMemory);

    // Descriptor pool + set.
    vk::DescriptorPoolSize poolSize{};
    poolSize.setType(vk::DescriptorType::StorageBuffer).setDescriptorCount(2);
    vk::DescriptorPoolCreateInfo poolInfo{};
    poolInfo.setPoolSizes({poolSize}).setMaxSets(1);
    auto pool = device->createDescriptorPool(poolInfo, std::nullopt);
    if (!pool) { std::println(stderr, "createDescriptorPool failed"); return 1; }
    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.setDescriptorPool(*pool).setSetLayouts({*setLayout});
    auto sets = device->allocateDescriptorSets(allocInfo);
    if (!sets || sets.value().empty()) { std::println(stderr, "allocateDescriptorSets failed"); return 1; }
    auto descriptorSet = sets.value()[0];

    vk::DescriptorBufferInfo inInfo{};
    inInfo.setBuffer(*inputBuffer).setOffset(0).setRange(size);
    vk::DescriptorBufferInfo outInfo{};
    outInfo.setBuffer(*outputBuffer).setOffset(0).setRange(size);
    vk::WriteDescriptorSet writeIn{};
    writeIn.setDstSet(descriptorSet).setDstBinding(0)
            .setDescriptorType(vk::DescriptorType::StorageBuffer)
            .setBufferInfo({inInfo});
    vk::WriteDescriptorSet writeOut{};
    writeOut.setDstSet(descriptorSet).setDstBinding(1)
             .setDescriptorType(vk::DescriptorType::StorageBuffer)
             .setBufferInfo({outInfo});
    device->updateDescriptorSets(std::array{writeIn, writeOut}, {});

    // Record + dispatch.
    auto commandPool = device->createCommandPool(
        vk::CommandPoolCreateInfo{}.setQueueFamilyIndex(queueFamily), std::nullopt);
    if (!commandPool) { std::println(stderr, "createCommandPool failed"); return 1; }
    auto commandBuffers = device->allocateCommandBuffers(
        vk::CommandBufferAllocateInfo{}.setCommandPool(*commandPool)
            .setLevel(vk::CommandBufferLevel::Primary).setCommandBufferCount(1));
    if (!commandBuffers || commandBuffers.value().empty()) {
        std::println(stderr, "allocateCommandBuffers failed"); return 1;
    }
    auto cmd = commandBuffers.value()[0];

    vk::CommandBufferBeginInfo beginInfo{};
    cmd.begin(beginInfo);
    vk::BufferMemoryBarrier preBarrier{};
    preBarrier.setSrcAccessMask(vk::AccessFlagBits::HostWrite)
               .setDstAccessMask(vk::AccessFlagBits::ShaderRead)
               .setBuffer(*inputBuffer).setOffset(0).setSize(size);
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::Host, vk::PipelineStageFlagBits::ComputeShader,
                        vk::DependencyFlags{}, {}, std::array{preBarrier}, {});
    cmd.bindPipeline(vk::PipelineBindPoint::Compute, pipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::Compute, *layout, 0, std::array{descriptorSet}, {});
    cmd.dispatch(kCount / kLocalSize, 1, 1);
    vk::BufferMemoryBarrier postBarrier{};
    postBarrier.setSrcAccessMask(vk::AccessFlagBits::ShaderWrite)
                .setDstAccessMask(vk::AccessFlagBits::HostRead)
                .setBuffer(*outputBuffer).setOffset(0).setSize(size);
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::ComputeShader, vk::PipelineStageFlagBits::Host,
                        vk::DependencyFlags{}, {}, std::array{postBarrier}, {});
    cmd.end();

    vk::SubmitInfo submit{};
    submit.setCommandBuffers({cmd});
    if (!queue.submit(std::array{submit}, vk::Fence{}) || !device->waitIdle()) {
        std::println(stderr, "compute submit failed"); return 1;
    }

    // Read back and verify.
    if (!device->mapMemory(*outputMemory, 0, size, vk::MemoryMapFlags{}, &mapped)) {
        std::println(stderr, "mapMemory (output) failed"); return 1;
    }
    const auto* output = static_cast<const float*>(mapped);
    bool ok = true;
    int firstBad = -1;
    for (std::uint32_t i = 0; i < kCount; ++i) {
        float expected = static_cast<float>(i) * 2.0f;
        if (std::fabs(output[i] - expected) > 1e-4f) { ok = false; firstBad = static_cast<int>(i); break; }
    }
    device->unmapMemory(*outputMemory);

    if (!ok) {
        std::println(stderr, "FAIL: compute verification failed at index {}: {} != {}",
                     firstBad, output[firstBad], static_cast<float>(firstBad) * 2.0f);
        return 1;
    }
    std::println("PASS: compute verified {} elements", kCount);
    return 0;
}
