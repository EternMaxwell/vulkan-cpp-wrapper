// 04-textured: a textured, indexed quad rendered offscreen and CPU-verified.
//
// Directionally exercises the wrapper's sampled-image path end to end:
// staging-buffer upload + copyBufferToImage + image layout barriers, a
// Sampler + CombinedImageSampler descriptor, a uniform buffer (UBO), push
// constants, interleaved vertex attributes, and an index buffer + drawIndexed.
// The 2x2 checkerboard texture is rendered full-screen and each quadrant is
// checked against its texel color.
#include <volk.h>
#include <vulkan_wrapper.hpp>
#include <validation.hpp>

#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <print>
#include <vector>

static constexpr int kWidth = 800;
static constexpr int kHeight = 600;

struct Vertex {
    float x, y, u, v;
};

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

static std::uint32_t find_graphics_family(const vk::PhysicalDevice& physical) {
    auto families = physical.getQueueFamilyProperties();
    for (std::uint32_t i = 0; i < families.size(); ++i)
        if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) return i;
    return 0;
}

static vk::ShaderModule load_shader(const vk::Device& device, const char* path) {
    auto code = read_spirv(path);
    vk::ShaderModuleCreateInfo info{};
    info.setCode(std::move(code));
    auto module = device.createShaderModule(info, std::nullopt);
    if (!module) std::println(stderr, "error: createShaderModule: {}", path);
    return *module;
}

int main() {
    if (volkInitialize() != VK_SUCCESS) { std::println(stderr, "volkInitialize failed"); return 1; }
    auto context = vk::Context::create();
    if (!context) { std::println(stderr, "Context::create failed"); return 1; }

    vk::ApplicationInfo app{};
    app.setApplicationName("vulkan-wrapper-textured");
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
    auto props = physical.getProperties();
    std::println("Using GPU: {}", props.deviceName.data());

    std::uint32_t queueFamily = find_graphics_family(physical);
    vk::DeviceQueueCreateInfo queueInfo{};
    queueInfo.setQueueFamilyIndex(queueFamily).setQueuePriorities({1.0f});
    vk::DeviceCreateInfo deviceInfo{};
    deviceInfo.setQueueCreateInfos({queueInfo});
    auto device = physical.createDevice(deviceInfo, std::nullopt);
    if (!device) { std::println(stderr, "createDevice failed"); return 1; }
    auto queue = device->getQueue(queueFamily, 0);

    auto vert = load_shader(*device, SHADER_VERT);
    auto frag = load_shader(*device, SHADER_FRAG);

    const auto hostFlags = vk::MemoryPropertyFlagBits::HostVisible | vk::MemoryPropertyFlagBits::HostCoherent;
    const auto deviceFlags = vk::MemoryPropertyFlagBits::DeviceLocal;

    // ------------------------------------------------------------------
    // 2x2 checkerboard texture (R8G8B8A8), uploaded via a staging buffer.
    // ------------------------------------------------------------------
    const std::array<std::uint8_t, 16> texels = {
        255, 0, 0, 255,        // (0,0) red
        0, 255, 0, 255,        // (1,0) green
        0, 0, 255, 255,        // (0,1) blue
        255, 255, 255, 255,    // (1,1) white
    };
    vk::ImageCreateInfo imageInfo{};
    imageInfo.setImageType(vk::ImageType::Value2d)
             .setFormat(vk::Format::R8g8b8a8Unorm)
             .setExtent(vk::Extent3D{2, 2, 1})
             .setMipLevels(1).setArrayLayers(1)
             .setSamples(vk::SampleCountFlagBits::Value1)
             .setTiling(vk::ImageTiling::Optimal)
             .setUsage(vk::ImageUsageFlagBits::TransferDst | vk::ImageUsageFlagBits::Sampled)
             .setSharingMode(vk::SharingMode::Exclusive)
             .setInitialLayout(vk::ImageLayout::Undefined);
    auto textureImage = device->createImage(imageInfo, std::nullopt);
    auto textureReqs = textureImage->getMemoryRequirements();
    auto textureMemory = device->allocateMemory(
        vk::MemoryAllocateInfo{}.setAllocationSize(textureReqs.size)
            .setMemoryTypeIndex(find_memory_type(physical, deviceFlags)), std::nullopt);
    if (!textureImage || !textureMemory ||
        !textureImage->bindMemory(*textureMemory, 0)) {
        std::println(stderr, "texture image setup failed"); return 1;
    }

    auto staging = device->createBuffer(
        vk::BufferCreateInfo{}.setSize(texels.size())
            .setUsage(vk::BufferUsageFlagBits::TransferSrc)
            .setSharingMode(vk::SharingMode::Exclusive), std::nullopt);
    auto stagingReqs = staging->getMemoryRequirements();
    auto stagingMemory = device->allocateMemory(
        vk::MemoryAllocateInfo{}.setAllocationSize(stagingReqs.size)
            .setMemoryTypeIndex(find_memory_type(physical, hostFlags)), std::nullopt);
    if (!staging || !stagingMemory || !staging->bindMemory(*stagingMemory, 0)) {
        std::println(stderr, "staging buffer setup failed"); return 1;
    }
    void* mapped = nullptr;
    if (!stagingMemory->mapMemory(0, texels.size(), vk::MemoryMapFlags{}, &mapped)) {
        std::println(stderr, "mapMemory failed"); return 1;
    }
    std::memcpy(mapped, texels.data(), texels.size());
    stagingMemory->unmapMemory();

    // Sampler + image view for sampling.
    vk::SamplerCreateInfo samplerInfo{};
    samplerInfo.setMagFilter(vk::Filter::Nearest).setMinFilter(vk::Filter::Nearest)
               .setMipmapMode(vk::SamplerMipmapMode::Nearest)
               .setAddressModeU(vk::SamplerAddressMode::ClampToEdge)
               .setAddressModeV(vk::SamplerAddressMode::ClampToEdge)
               .setAddressModeW(vk::SamplerAddressMode::ClampToEdge)
               .setMipLodBias(0.0f).setAnisotropyEnable(false).setMaxAnisotropy(1.0f)
               .setCompareEnable(false).setMinLod(0.0f).setMaxLod(0.0f);
    auto sampler = device->createSampler(samplerInfo, std::nullopt);

    vk::ImageSubresourceRange subresource{};
    subresource.setAspectMask(vk::ImageAspectFlagBits::Color)
               .setBaseMipLevel(0).setLevelCount(1)
               .setBaseArrayLayer(0).setLayerCount(1);
    auto textureView = device->createImageView(
        vk::ImageViewCreateInfo{}.setImage(*textureImage)
            .setViewType(vk::ImageViewType::Value2d)
            .setFormat(vk::Format::R8g8b8a8Unorm)
            .setSubresourceRange(subresource), std::nullopt);
    if (!sampler || !textureView) { std::println(stderr, "sampler/view failed"); return 1; }

    // ------------------------------------------------------------------
    // Uniform buffer (vec2 offset) + descriptor set + push-constant range.
    // ------------------------------------------------------------------
    auto ubo = device->createBuffer(
        vk::BufferCreateInfo{}.setSize(8)
            .setUsage(vk::BufferUsageFlagBits::UniformBuffer)
            .setSharingMode(vk::SharingMode::Exclusive), std::nullopt);
    auto uboReqs = ubo->getMemoryRequirements();
    auto uboMemory = device->allocateMemory(
        vk::MemoryAllocateInfo{}.setAllocationSize(uboReqs.size)
            .setMemoryTypeIndex(find_memory_type(physical, hostFlags)), std::nullopt);
    if (!ubo || !uboMemory || !ubo->bindMemory(*uboMemory, 0)) {
        std::println(stderr, "ubo setup failed"); return 1;
    }
    float offset[2] = {0.0f, 0.0f};
    if (!uboMemory->mapMemory(0, 8, vk::MemoryMapFlags{}, &mapped)) {
        std::println(stderr, "ubo map failed"); return 1;
    }
    std::memcpy(mapped, offset, sizeof(offset));
    uboMemory->unmapMemory();

    vk::DescriptorSetLayoutBinding uboBinding{};
    uboBinding.setBinding(0).setDescriptorType(vk::DescriptorType::UniformBuffer)
               .setDescriptorCount(1).setStageFlags(vk::ShaderStageFlagBits::Vertex);
    vk::DescriptorSetLayoutBinding texBinding{};
    texBinding.setBinding(1).setDescriptorType(vk::DescriptorType::CombinedImageSampler)
               .setDescriptorCount(1).setStageFlags(vk::ShaderStageFlagBits::Fragment);
    vk::DescriptorSetLayoutCreateInfo setLayoutInfo{};
    setLayoutInfo.setBindings({uboBinding, texBinding});
    auto setLayout = device->createDescriptorSetLayout(setLayoutInfo, std::nullopt);

    vk::PushConstantRange pushRange{};
    pushRange.setStageFlags(vk::ShaderStageFlagBits::Fragment).setOffset(0).setSize(sizeof(float));
    vk::PipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.setSetLayouts({*setLayout}).setPushConstantRanges({pushRange});
    auto layout = device->createPipelineLayout(layoutInfo, std::nullopt);
    if (!setLayout || !layout) { std::println(stderr, "layout setup failed"); return 1; }

    vk::DescriptorPoolSize poolSizes[] = {
        vk::DescriptorPoolSize{}.setType(vk::DescriptorType::UniformBuffer).setDescriptorCount(1),
        vk::DescriptorPoolSize{}.setType(vk::DescriptorType::CombinedImageSampler).setDescriptorCount(1),
    };
    vk::DescriptorPoolCreateInfo poolInfo{};
    poolInfo.setPoolSizes({poolSizes[0], poolSizes[1]}).setMaxSets(1);
    auto pool = device->createDescriptorPool(poolInfo, std::nullopt);
    auto sets = device->allocateDescriptorSets(
        vk::DescriptorSetAllocateInfo{}.setDescriptorPool(*pool).setSetLayouts({*setLayout}));
    if (!pool || !sets || sets.value().empty()) { std::println(stderr, "descriptor pool/set failed"); return 1; }
    auto descriptorSet = sets.value()[0];

    vk::DescriptorBufferInfo uboInfo{};
    uboInfo.setBuffer(*ubo).setOffset(0).setRange(8);
    vk::DescriptorImageInfo texInfo{};
    texInfo.setSampler(*sampler).setImageView(*textureView)
            .setImageLayout(vk::ImageLayout::ShaderReadOnlyOptimal);
    vk::WriteDescriptorSet writeUbo{};
    writeUbo.setDstSet(descriptorSet).setDstBinding(0)
            .setDescriptorType(vk::DescriptorType::UniformBuffer).setBufferInfo({uboInfo});
    vk::WriteDescriptorSet writeTex{};
    writeTex.setDstSet(descriptorSet).setDstBinding(1)
            .setDescriptorType(vk::DescriptorType::CombinedImageSampler).setImageInfo({texInfo});
    device->updateDescriptorSets(std::array{writeUbo, writeTex}, {});

    // ------------------------------------------------------------------
    // Vertex + index buffers (interleaved position + UV, uint16 indices).
    // ------------------------------------------------------------------
    const std::vector<Vertex> vertices = {
        {-1.0f, -1.0f, 0.0f, 0.0f},  // top-left    -> red
        { 1.0f, -1.0f, 1.0f, 0.0f},  // top-right   -> green
        { 1.0f,  1.0f, 1.0f, 1.0f},  // bottom-right -> white
        {-1.0f,  1.0f, 0.0f, 1.0f},  // bottom-left -> blue
    };
    const std::vector<std::uint16_t> indices = {0, 1, 2, 0, 2, 3};

    auto vertexBuffer = device->createBuffer(
        vk::BufferCreateInfo{}.setSize(vertices.size() * sizeof(Vertex))
            .setUsage(vk::BufferUsageFlagBits::VertexBuffer), std::nullopt);
    auto vertexReqs = vertexBuffer->getMemoryRequirements();
    auto vertexMemory = device->allocateMemory(
        vk::MemoryAllocateInfo{}.setAllocationSize(vertexReqs.size)
            .setMemoryTypeIndex(find_memory_type(physical, hostFlags)), std::nullopt);
    if (!vertexBuffer || !vertexMemory || !vertexBuffer->bindMemory(*vertexMemory, 0)) {
        std::println(stderr, "vertex buffer setup failed"); return 1;
    }
    if (!vertexMemory->mapMemory(0, vertices.size() * sizeof(Vertex), vk::MemoryMapFlags{}, &mapped)) {
        std::println(stderr, "vertex map failed"); return 1;
    }
    std::memcpy(mapped, vertices.data(), vertices.size() * sizeof(Vertex));
    vertexMemory->unmapMemory();

    auto indexBuffer = device->createBuffer(
        vk::BufferCreateInfo{}.setSize(indices.size() * sizeof(std::uint16_t))
            .setUsage(vk::BufferUsageFlagBits::IndexBuffer), std::nullopt);
    auto indexReqs = indexBuffer->getMemoryRequirements();
    auto indexMemory = device->allocateMemory(
        vk::MemoryAllocateInfo{}.setAllocationSize(indexReqs.size)
            .setMemoryTypeIndex(find_memory_type(physical, hostFlags)), std::nullopt);
    if (!indexBuffer || !indexMemory || !indexBuffer->bindMemory(*indexMemory, 0)) {
        std::println(stderr, "index buffer setup failed"); return 1;
    }
    if (!indexMemory->mapMemory(0, indices.size() * sizeof(std::uint16_t), vk::MemoryMapFlags{}, &mapped)) {
        std::println(stderr, "index map failed"); return 1;
    }
    std::memcpy(mapped, indices.data(), indices.size() * sizeof(std::uint16_t));
    indexMemory->unmapMemory();

    // ------------------------------------------------------------------
    // Offscreen color target + pipeline + command buffer.
    // ------------------------------------------------------------------
    vk::ImageCreateInfo targetInfo{};
    targetInfo.setImageType(vk::ImageType::Value2d)
              .setFormat(vk::Format::R8g8b8a8Unorm)
              .setExtent(vk::Extent3D{kWidth, kHeight, 1})
              .setMipLevels(1).setArrayLayers(1)
              .setSamples(vk::SampleCountFlagBits::Value1)
              .setTiling(vk::ImageTiling::Optimal)
              .setUsage(vk::ImageUsageFlagBits::ColorAttachment | vk::ImageUsageFlagBits::TransferSrc)
              .setSharingMode(vk::SharingMode::Exclusive)
              .setInitialLayout(vk::ImageLayout::Undefined);
    auto targetImage = device->createImage(targetInfo, std::nullopt);
    auto targetReqs = targetImage->getMemoryRequirements();
    auto targetMemory = device->allocateMemory(
        vk::MemoryAllocateInfo{}.setAllocationSize(targetReqs.size)
            .setMemoryTypeIndex(find_memory_type(physical, deviceFlags)), std::nullopt);
    if (!targetImage || !targetMemory || !targetImage->bindMemory(*targetMemory, 0)) {
        std::println(stderr, "target image setup failed"); return 1;
    }
    auto targetView = device->createImageView(
        vk::ImageViewCreateInfo{}.setImage(*targetImage)
            .setViewType(vk::ImageViewType::Value2d)
            .setFormat(vk::Format::R8g8b8a8Unorm)
            .setSubresourceRange(subresource), std::nullopt);

    vk::AttachmentDescription colorAttachment{};
    colorAttachment.setFormat(vk::Format::R8g8b8a8Unorm)
                   .setSamples(vk::SampleCountFlagBits::Value1)
                   .setLoadOp(vk::AttachmentLoadOp::Clear)
                   .setStoreOp(vk::AttachmentStoreOp::Store)
                   .setStencilLoadOp(vk::AttachmentLoadOp::DontCare)
                   .setStencilStoreOp(vk::AttachmentStoreOp::DontCare)
                   .setInitialLayout(vk::ImageLayout::Undefined)
                   .setFinalLayout(vk::ImageLayout::TransferSrcOptimal);
    vk::AttachmentReference colorRef{};
    colorRef.setAttachment(0).setLayout(vk::ImageLayout::ColorAttachmentOptimal);
    vk::SubpassDescription subpass{};
    subpass.setPipelineBindPoint(vk::PipelineBindPoint::Graphics).setColorAttachments({colorRef});
    vk::RenderPassCreateInfo renderPassInfo{};
    renderPassInfo.setAttachments({colorAttachment}).setSubpasses({subpass});
    auto renderPass = device->createRenderPass(renderPassInfo, std::nullopt);
    auto framebuffer = device->createFramebuffer(
        vk::FramebufferCreateInfo{}.setRenderPass(*renderPass)
            .setAttachments({*targetView}).setWidth(kWidth).setHeight(kHeight).setLayers(1), std::nullopt);
    if (!renderPass || !framebuffer) { std::println(stderr, "render pass/framebuffer failed"); return 1; }

    vk::PipelineShaderStageCreateInfo vertStage{};
    vertStage.setStage(vk::ShaderStageFlagBits::Vertex).setModule(vert).setName("main");
    vk::PipelineShaderStageCreateInfo fragStage{};
    fragStage.setStage(vk::ShaderStageFlagBits::Fragment).setModule(frag).setName("main");
    vk::VertexInputBindingDescription binding{};
    binding.setBinding(0).setStride(sizeof(Vertex)).setInputRate(vk::VertexInputRate::Vertex);
    vk::VertexInputAttributeDescription posAttr{};
    posAttr.setLocation(0).setBinding(0).setFormat(vk::Format::R32g32Sfloat).setOffset(0);
    vk::VertexInputAttributeDescription uvAttr{};
    uvAttr.setLocation(1).setBinding(0).setFormat(vk::Format::R32g32Sfloat).setOffset(2 * sizeof(float));
    vk::PipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.setVertexBindingDescriptions({binding})
               .setVertexAttributeDescriptions({posAttr, uvAttr});
    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.setTopology(vk::PrimitiveTopology::TriangleList).setPrimitiveRestartEnable(false);
    vk::Viewport viewport{};
    viewport.setX(0.0f).setY(0.0f).setWidth(static_cast<float>(kWidth)).setHeight(static_cast<float>(kHeight))
            .setMinDepth(0.0f).setMaxDepth(1.0f);
    vk::Rect2D scissor{};
    scissor.setOffset(vk::Offset2D{0, 0}).setExtent(vk::Extent2D{kWidth, kHeight});
    vk::PipelineViewportStateCreateInfo viewportState{};
    viewportState.setViewports({viewport}).setScissors({scissor});
    vk::PipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.setDepthClampEnable(false).setRasterizerDiscardEnable(false)
              .setPolygonMode(vk::PolygonMode::Fill).setCullMode(vk::CullModeFlagBits::None)
              .setFrontFace(vk::FrontFace::CounterClockwise).setLineWidth(1.0f);
    vk::PipelineMultisampleStateCreateInfo multisampling{};
    multisampling.setRasterizationSamples(vk::SampleCountFlagBits::Value1).setSampleShadingEnable(false);
    vk::PipelineColorBlendAttachmentState colorBlend{};
    colorBlend.setBlendEnable(false).setColorWriteMask(
        vk::ColorComponentFlagBits::R | vk::ColorComponentFlagBits::G |
        vk::ColorComponentFlagBits::B | vk::ColorComponentFlagBits::A);
    vk::PipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.setLogicOpEnable(false).setAttachments({colorBlend});
    vk::GraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.setStages({vertStage, fragStage})
                .setVertexInputState(vertexInput)
                .setInputAssemblyState(inputAssembly)
                .setViewportState(viewportState)
                .setRasterizationState(rasterizer)
                .setMultisampleState(multisampling)
                .setColorBlendState(colorBlending)
                .setLayout(*layout)
                .setRenderPass(*renderPass)
                .setSubpass(0);
    auto pipelines = device->createGraphicsPipelines(
        vk::PipelineCache{}, std::array{pipelineInfo}, std::nullopt);
    if (pipelines.status != vk::ResultCode::Success || pipelines.value.empty()) {
        std::println(stderr, "createGraphicsPipelines failed: {}", static_cast<int>(pipelines.status));
        return 1;
    }
    auto pipeline = std::move(pipelines.value[0]);

    auto commandPool = device->createCommandPool(
        vk::CommandPoolCreateInfo{}.setQueueFamilyIndex(queueFamily)
            .setFlags(vk::CommandPoolCreateFlagBits::ResetCommandBuffer), std::nullopt);
    auto commandBuffers = device->allocateCommandBuffers(
        vk::CommandBufferAllocateInfo{}.setCommandPool(*commandPool)
            .setLevel(vk::CommandBufferLevel::Primary).setCommandBufferCount(1));
    if (!commandBuffers || commandBuffers.value().empty()) { std::println(stderr, "command buffer failed"); return 1; }
    auto cmd = commandBuffers.value()[0];

    vk::CommandBufferBeginInfo beginInfo{};
    cmd.begin(beginInfo);

    // Transition the texture: UNDEFINED -> TRANSFER_DST, copy, then
    // TRANSFER_DST -> SHADER_READ_ONLY.
    vk::ImageMemoryBarrier toDst{};
    toDst.setSrcAccessMask(vk::AccessFlagBits{}).setDstAccessMask(vk::AccessFlagBits::TransferWrite)
         .setOldLayout(vk::ImageLayout::Undefined).setNewLayout(vk::ImageLayout::TransferDstOptimal)
         .setImage(*textureImage).setSubresourceRange(subresource);
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::TopOfPipe, vk::PipelineStageFlagBits::Transfer,
                        vk::DependencyFlags{}, {}, {}, std::array{toDst});
    vk::BufferImageCopy copyRegion{};
    copyRegion.setBufferOffset(0).setBufferRowLength(0).setBufferImageHeight(0)
              .setImageSubresource(vk::ImageSubresourceLayers{}
                  .setAspectMask(vk::ImageAspectFlagBits::Color)
                  .setMipLevel(0).setBaseArrayLayer(0).setLayerCount(1))
              .setImageOffset(vk::Offset3D{0, 0, 0}).setImageExtent(vk::Extent3D{2, 2, 1});
    cmd.copyBufferToImage(*staging, *textureImage, vk::ImageLayout::TransferDstOptimal, std::array{copyRegion});
    vk::ImageMemoryBarrier toRead{};
    toRead.setSrcAccessMask(vk::AccessFlagBits::TransferWrite).setDstAccessMask(vk::AccessFlagBits::ShaderRead)
          .setOldLayout(vk::ImageLayout::TransferDstOptimal).setNewLayout(vk::ImageLayout::ShaderReadOnlyOptimal)
          .setImage(*textureImage).setSubresourceRange(subresource);
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::Transfer, vk::PipelineStageFlagBits::FragmentShader,
                        vk::DependencyFlags{}, {}, {}, std::array{toRead});

    // Render the textured quad with an identity UBO and tint=1.0.
    vk::RenderPassBeginInfo rpBegin{};
    vk::ClearValue clear{};
    clear.color.float32[0] = 0.0f; clear.color.float32[1] = 0.0f;
    clear.color.float32[2] = 0.0f; clear.color.float32[3] = 1.0f;
    rpBegin.setRenderPass(*renderPass).setFramebuffer(*framebuffer)
           .setRenderArea(vk::Rect2D{{0, 0}, {kWidth, kHeight}}).setClearValues({clear});
    cmd.beginRenderPass(rpBegin, vk::SubpassContents::Inline);
    cmd.bindPipeline(vk::PipelineBindPoint::Graphics, pipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::Graphics, *layout, 0, std::array{descriptorSet}, {});
    float tint = 1.0f;
    cmd.pushConstants(*layout, vk::ShaderStageFlagBits::Fragment, 0, sizeof(float),
                      std::bit_cast<std::array<std::byte, sizeof(float)>>(tint));
    cmd.bindVertexBuffers(0, std::array{*vertexBuffer}, std::array{vk::DeviceSize{0}});
    cmd.bindIndexBuffer(*indexBuffer, 0, vk::IndexType::Uint16);
    cmd.drawIndexed(static_cast<std::uint32_t>(indices.size()), 1, 0, 0, 0);
    cmd.endRenderPass();
    cmd.end();

    vk::SubmitInfo submit{};
    submit.setCommandBuffers({cmd});
    if (!queue.submit(std::array{submit}, vk::Fence{}) || !device->waitIdle()) {
        std::println(stderr, "submit failed"); return 1;
    }

    // Readback and verify the four quadrants.
    const vk::DeviceSize readbackSize = static_cast<vk::DeviceSize>(kWidth) * kHeight * 4;
    auto readbackBuffer = device->createBuffer(
        vk::BufferCreateInfo{}.setSize(readbackSize)
            .setUsage(vk::BufferUsageFlagBits::TransferDst), std::nullopt);
    auto readbackReqs = readbackBuffer->getMemoryRequirements();
    auto readbackMemory = device->allocateMemory(
        vk::MemoryAllocateInfo{}.setAllocationSize(readbackReqs.size)
            .setMemoryTypeIndex(find_memory_type(physical, hostFlags)), std::nullopt);
    if (!readbackBuffer || !readbackMemory || !readbackBuffer->bindMemory(*readbackMemory, 0)) {
        std::println(stderr, "readback buffer failed"); return 1;
    }
    auto copyCmd = commandBuffers.value()[0];
    copyCmd.reset(vk::CommandBufferResetFlags{});
    vk::CommandBufferBeginInfo copyBegin{};
    copyCmd.begin(copyBegin);
    vk::BufferImageCopy targetRegion{};
    targetRegion.setBufferOffset(0).setBufferRowLength(0).setBufferImageHeight(0)
                .setImageSubresource(vk::ImageSubresourceLayers{}
                    .setAspectMask(vk::ImageAspectFlagBits::Color)
                    .setMipLevel(0).setBaseArrayLayer(0).setLayerCount(1))
                .setImageOffset(vk::Offset3D{0, 0, 0}).setImageExtent(vk::Extent3D{kWidth, kHeight, 1});
    copyCmd.copyImageToBuffer(*targetImage, vk::ImageLayout::TransferSrcOptimal,
                              *readbackBuffer, std::array{targetRegion});
    copyCmd.end();
    vk::SubmitInfo copySubmit{};
    copySubmit.setCommandBuffers({copyCmd});
    if (!queue.submit(std::array{copySubmit}, vk::Fence{}) || !device->waitIdle()) {
        std::println(stderr, "readback submit failed"); return 1;
    }
    if (!readbackMemory->mapMemory(0, readbackSize, vk::MemoryMapFlags{}, &mapped)) {
        std::println(stderr, "readback map failed"); return 1;
    }
    const auto* bytes = static_cast<const std::uint8_t*>(mapped);
    auto pixel = [&](int x, int y) {
        const std::uint8_t* p = bytes + (y * kWidth + x) * 4;
        return std::array<std::uint8_t, 4>{p[0], p[1], p[2], p[3]};
    };
    auto tl = pixel(200, 150);
    auto tr = pixel(600, 150);
    auto bl = pixel(200, 450);
    auto br = pixel(600, 450);
    readbackMemory->unmapMemory();

    auto closeTo = [](const std::array<std::uint8_t, 4>& p, std::array<int, 4> want) {
        for (int i = 0; i < 4; ++i) if (std::abs(static_cast<int>(p[i]) - want[i]) > 24) return false;
        return true;
    };
    bool ok = closeTo(tl, {255, 0, 0, 255}) && closeTo(tr, {0, 255, 0, 255}) &&
              closeTo(bl, {0, 0, 255, 255}) && closeTo(br, {255, 255, 255, 255});
    std::println("texture quadrants (RGBA): TL={} TR={} BL={} BR={}",
                 std::array<int, 4>{tl[0], tl[1], tl[2], tl[3]},
                 std::array<int, 4>{tr[0], tr[1], tr[2], tr[3]},
                 std::array<int, 4>{bl[0], bl[1], bl[2], bl[3]},
                 std::array<int, 4>{br[0], br[1], br[2], br[3]});
    if (!ok) { std::println(stderr, "FAIL: texture verification failed"); return 1; }
    std::println("PASS: textured quad verified");
    if (!sample::reportValidation(validation)) return 1;
    return 0;
}
