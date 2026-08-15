// 05-depth: a depth-tested scene rendered offscreen and CPU-verified.
//
// Directionally exercises the wrapper's depth path: a depth image (D32_SFLOAT),
// a depth attachment + depth/stencil state, and per-vertex color attributes.
// A far red triangle is drawn first, then a nearer blue triangle over it; the
// depth test (LESS) must keep the blue triangle on top in the overlap region.
#include <volk.h>
#include <vulkan_wrapper.hpp>
#include <validation.hpp>

#include <array>
#include <cstdint>
#include <cstring>
#include <print>
#include <vector>

static constexpr int kWidth = 800;
static constexpr int kHeight = 600;

struct Vertex {
    float x, y, z, r, g, b;
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

int main() {
    if (volkInitialize() != VK_SUCCESS) { std::println(stderr, "volkInitialize failed"); return 1; }
    auto context = vk::Context::create();
    if (!context) { std::println(stderr, "Context::create failed"); return 1; }

    vk::ApplicationInfo app{};
    app.setApplicationName("vulkan-wrapper-depth");
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

    auto code = read_spirv(SHADER_VERT);
    vk::ShaderModuleCreateInfo vertInfo{};
    vertInfo.setCode(std::move(code));
    auto vert = device->createShaderModule(vertInfo, std::nullopt);
    code = read_spirv(SHADER_FRAG);
    vk::ShaderModuleCreateInfo fragInfo{};
    fragInfo.setCode(std::move(code));
    auto frag = device->createShaderModule(fragInfo, std::nullopt);
    if (!vert || !frag) { std::println(stderr, "createShaderModule failed"); return 1; }

    const auto hostFlags = vk::MemoryPropertyFlagBits::HostVisible | vk::MemoryPropertyFlagBits::HostCoherent;
    const auto deviceFlags = vk::MemoryPropertyFlagBits::DeviceLocal;

    // Two triangles: a far red full-screen triangle, then a near blue one.
    const std::vector<Vertex> vertices = {
        {-1.0f, -1.0f, 0.9f, 1.0f, 0.0f, 0.0f},   // A (red, far)
        { 3.0f, -1.0f, 0.9f, 1.0f, 0.0f, 0.0f},
        {-1.0f,  3.0f, 0.9f, 1.0f, 0.0f, 0.0f},
        {-0.5f, -0.5f, 0.1f, 0.0f, 0.0f, 1.0f},   // B (blue, near)
        { 0.5f, -0.5f, 0.1f, 0.0f, 0.0f, 1.0f},
        { 0.0f,  0.5f, 0.1f, 0.0f, 0.0f, 1.0f},
    };

    auto vertexBuffer = device->createBuffer(
        vk::BufferCreateInfo{}.setSize(vertices.size() * sizeof(Vertex))
            .setUsage(vk::BufferUsageFlagBits::VertexBuffer), std::nullopt);
    auto vertexReqs = device->getBufferMemoryRequirements(*vertexBuffer);
    auto vertexMemory = device->allocateMemory(
        vk::MemoryAllocateInfo{}.setAllocationSize(vertexReqs.size)
            .setMemoryTypeIndex(find_memory_type(physical, hostFlags)), std::nullopt);
    if (!vertexBuffer || !vertexMemory || !device->bindBufferMemory(*vertexBuffer, *vertexMemory, 0)) {
        std::println(stderr, "vertex buffer failed"); return 1;
    }
    void* mapped = nullptr;
    if (!device->mapMemory(*vertexMemory, 0, vertices.size() * sizeof(Vertex), vk::MemoryMapFlags{}, &mapped)) {
        std::println(stderr, "vertex map failed"); return 1;
    }
    std::memcpy(mapped, vertices.data(), vertices.size() * sizeof(Vertex));
    device->unmapMemory(*vertexMemory);

    // Offscreen color target + depth buffer.
    vk::ImageSubresourceRange colorRange{};
    colorRange.setAspectMask(vk::ImageAspectFlagBits::Color)
               .setBaseMipLevel(0).setLevelCount(1).setBaseArrayLayer(0).setLayerCount(1);
    vk::ImageCreateInfo colorInfo{};
    colorInfo.setImageType(vk::ImageType::Value2d).setFormat(vk::Format::R8g8b8a8Unorm)
             .setExtent(vk::Extent3D{kWidth, kHeight, 1}).setMipLevels(1).setArrayLayers(1)
             .setSamples(vk::SampleCountFlagBits::Value1).setTiling(vk::ImageTiling::Optimal)
             .setUsage(vk::ImageUsageFlagBits::ColorAttachment | vk::ImageUsageFlagBits::TransferSrc)
             .setSharingMode(vk::SharingMode::Exclusive).setInitialLayout(vk::ImageLayout::Undefined);
    auto colorImage = device->createImage(colorInfo, std::nullopt);
    auto colorReqs = device->getImageMemoryRequirements(*colorImage);
    auto colorMemory = device->allocateMemory(
        vk::MemoryAllocateInfo{}.setAllocationSize(colorReqs.size)
            .setMemoryTypeIndex(find_memory_type(physical, deviceFlags)), std::nullopt);
    if (!colorImage || !colorMemory || !device->bindImageMemory(*colorImage, *colorMemory, 0)) {
        std::println(stderr, "color image failed"); return 1;
    }
    auto colorView = device->createImageView(
        vk::ImageViewCreateInfo{}.setImage(*colorImage).setViewType(vk::ImageViewType::Value2d)
            .setFormat(vk::Format::R8g8b8a8Unorm).setSubresourceRange(colorRange), std::nullopt);

    vk::ImageSubresourceRange depthRange{};
    depthRange.setAspectMask(vk::ImageAspectFlagBits::Depth)
               .setBaseMipLevel(0).setLevelCount(1).setBaseArrayLayer(0).setLayerCount(1);
    vk::ImageCreateInfo depthInfo{};
    depthInfo.setImageType(vk::ImageType::Value2d).setFormat(vk::Format::D32Sfloat)
             .setExtent(vk::Extent3D{kWidth, kHeight, 1}).setMipLevels(1).setArrayLayers(1)
             .setSamples(vk::SampleCountFlagBits::Value1).setTiling(vk::ImageTiling::Optimal)
             .setUsage(vk::ImageUsageFlagBits::DepthStencilAttachment)
             .setSharingMode(vk::SharingMode::Exclusive).setInitialLayout(vk::ImageLayout::Undefined);
    auto depthImage = device->createImage(depthInfo, std::nullopt);
    auto depthReqs = device->getImageMemoryRequirements(*depthImage);
    auto depthMemory = device->allocateMemory(
        vk::MemoryAllocateInfo{}.setAllocationSize(depthReqs.size)
            .setMemoryTypeIndex(find_memory_type(physical, deviceFlags)), std::nullopt);
    if (!depthImage || !depthMemory || !device->bindImageMemory(*depthImage, *depthMemory, 0)) {
        std::println(stderr, "depth image failed"); return 1;
    }
    auto depthView = device->createImageView(
        vk::ImageViewCreateInfo{}.setImage(*depthImage).setViewType(vk::ImageViewType::Value2d)
            .setFormat(vk::Format::D32Sfloat).setSubresourceRange(depthRange), std::nullopt);

    vk::AttachmentDescription colorAttachment{};
    colorAttachment.setFormat(vk::Format::R8g8b8a8Unorm).setSamples(vk::SampleCountFlagBits::Value1)
                   .setLoadOp(vk::AttachmentLoadOp::Clear).setStoreOp(vk::AttachmentStoreOp::Store)
                   .setStencilLoadOp(vk::AttachmentLoadOp::DontCare).setStencilStoreOp(vk::AttachmentStoreOp::DontCare)
                   .setInitialLayout(vk::ImageLayout::Undefined).setFinalLayout(vk::ImageLayout::TransferSrcOptimal);
    vk::AttachmentDescription depthAttachment{};
    depthAttachment.setFormat(vk::Format::D32Sfloat).setSamples(vk::SampleCountFlagBits::Value1)
                   .setLoadOp(vk::AttachmentLoadOp::Clear).setStoreOp(vk::AttachmentStoreOp::DontCare)
                   .setStencilLoadOp(vk::AttachmentLoadOp::DontCare).setStencilStoreOp(vk::AttachmentStoreOp::DontCare)
                   .setInitialLayout(vk::ImageLayout::Undefined).setFinalLayout(vk::ImageLayout::DepthStencilAttachmentOptimal);
    vk::AttachmentReference colorRef{};
    colorRef.setAttachment(0).setLayout(vk::ImageLayout::ColorAttachmentOptimal);
    vk::AttachmentReference depthRef{};
    depthRef.setAttachment(1).setLayout(vk::ImageLayout::DepthStencilAttachmentOptimal);
    vk::SubpassDescription subpass{};
    subpass.setPipelineBindPoint(vk::PipelineBindPoint::Graphics)
           .setColorAttachments({colorRef}).setDepthStencilAttachment(depthRef);
    vk::RenderPassCreateInfo renderPassInfo{};
    renderPassInfo.setAttachments({colorAttachment, depthAttachment}).setSubpasses({subpass});
    auto renderPass = device->createRenderPass(renderPassInfo, std::nullopt);
    auto framebuffer = device->createFramebuffer(
        vk::FramebufferCreateInfo{}.setRenderPass(*renderPass)
            .setAttachments({*colorView, *depthView}).setWidth(kWidth).setHeight(kHeight).setLayers(1), std::nullopt);
    if (!renderPass || !framebuffer) { std::println(stderr, "render pass/framebuffer failed"); return 1; }

    vk::PipelineLayoutCreateInfo layoutInfo{};
    auto layout = device->createPipelineLayout(layoutInfo, std::nullopt);
    vk::PipelineShaderStageCreateInfo vertStage{};
    vertStage.setStage(vk::ShaderStageFlagBits::Vertex).setModule(*vert).setName("main");
    vk::PipelineShaderStageCreateInfo fragStage{};
    fragStage.setStage(vk::ShaderStageFlagBits::Fragment).setModule(*frag).setName("main");
    vk::VertexInputBindingDescription binding{};
    binding.setBinding(0).setStride(sizeof(Vertex)).setInputRate(vk::VertexInputRate::Vertex);
    vk::VertexInputAttributeDescription posAttr{};
    posAttr.setLocation(0).setBinding(0).setFormat(vk::Format::R32g32b32Sfloat).setOffset(0);
    vk::VertexInputAttributeDescription colorAttr{};
    colorAttr.setLocation(1).setBinding(0).setFormat(vk::Format::R32g32b32Sfloat).setOffset(3 * sizeof(float));
    vk::PipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.setVertexBindingDescriptions({binding}).setVertexAttributeDescriptions({posAttr, colorAttr});
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
    vk::PipelineDepthStencilStateCreateInfo depthState{};
    depthState.setDepthTestEnable(true).setDepthWriteEnable(true)
              .setDepthCompareOp(vk::CompareOp::Less);
    vk::PipelineColorBlendAttachmentState colorBlend{};
    colorBlend.setBlendEnable(false).setColorWriteMask(
        vk::ColorComponentFlagBits::R | vk::ColorComponentFlagBits::G |
        vk::ColorComponentFlagBits::B | vk::ColorComponentFlagBits::A);
    vk::PipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.setLogicOpEnable(false).setAttachments({colorBlend});
    vk::GraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.setStages({vertStage, fragStage})
                .setVertexInputState(vertexInput).setInputAssemblyState(inputAssembly)
                .setViewportState(viewportState).setRasterizationState(rasterizer)
                .setMultisampleState(multisampling).setDepthStencilState(depthState)
                .setColorBlendState(colorBlending).setLayout(*layout)
                .setRenderPass(*renderPass).setSubpass(0);
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
    vk::ClearValue colorClear{};
    colorClear.color.float32[0] = 0.0f; colorClear.color.float32[1] = 0.0f;
    colorClear.color.float32[2] = 0.0f; colorClear.color.float32[3] = 1.0f;
    vk::ClearValue depthClear{};
    depthClear.depthStencil.depth = 1.0f;
    depthClear.depthStencil.stencil = 0;
    vk::RenderPassBeginInfo rpBegin{};
    rpBegin.setRenderPass(*renderPass).setFramebuffer(*framebuffer)
           .setRenderArea(vk::Rect2D{{0, 0}, {kWidth, kHeight}})
           .setClearValues({colorClear, depthClear});
    cmd.beginRenderPass(rpBegin, vk::SubpassContents::Inline);
    cmd.bindPipeline(vk::PipelineBindPoint::Graphics, pipeline);
    cmd.bindVertexBuffers(0, std::array{*vertexBuffer}, std::array{vk::DeviceSize{0}});
    cmd.draw(3, 1, 0, 0);   // far red triangle
    cmd.draw(3, 1, 3, 0);   // near blue triangle
    cmd.endRenderPass();
    cmd.end();

    vk::SubmitInfo submit{};
    submit.setCommandBuffers({cmd});
    if (!queue.submit(std::array{submit}, vk::Fence{}) || !device->waitIdle()) {
        std::println(stderr, "submit failed"); return 1;
    }

    // Readback and verify: center = blue (near wins), corner = red (far).
    const vk::DeviceSize readbackSize = static_cast<vk::DeviceSize>(kWidth) * kHeight * 4;
    auto readbackBuffer = device->createBuffer(
        vk::BufferCreateInfo{}.setSize(readbackSize).setUsage(vk::BufferUsageFlagBits::TransferDst), std::nullopt);
    auto readbackReqs = device->getBufferMemoryRequirements(*readbackBuffer);
    auto readbackMemory = device->allocateMemory(
        vk::MemoryAllocateInfo{}.setAllocationSize(readbackReqs.size)
            .setMemoryTypeIndex(find_memory_type(physical, hostFlags)), std::nullopt);
    if (!readbackBuffer || !readbackMemory || !device->bindBufferMemory(*readbackBuffer, *readbackMemory, 0)) {
        std::println(stderr, "readback buffer failed"); return 1;
    }
    auto copyCmd = commandBuffers.value()[0];
    copyCmd.reset(vk::CommandBufferResetFlags{});
    vk::CommandBufferBeginInfo copyBegin{};
    copyCmd.begin(copyBegin);
    vk::BufferImageCopy region{};
    region.setBufferOffset(0).setBufferRowLength(0).setBufferImageHeight(0)
          .setImageSubresource(vk::ImageSubresourceLayers{}
              .setAspectMask(vk::ImageAspectFlagBits::Color)
              .setMipLevel(0).setBaseArrayLayer(0).setLayerCount(1))
          .setImageOffset(vk::Offset3D{0, 0, 0}).setImageExtent(vk::Extent3D{kWidth, kHeight, 1});
    copyCmd.copyImageToBuffer(*colorImage, vk::ImageLayout::TransferSrcOptimal, *readbackBuffer, std::array{region});
    copyCmd.end();
    vk::SubmitInfo copySubmit{};
    copySubmit.setCommandBuffers({copyCmd});
    if (!queue.submit(std::array{copySubmit}, vk::Fence{}) || !device->waitIdle()) {
        std::println(stderr, "readback submit failed"); return 1;
    }
    if (!device->mapMemory(*readbackMemory, 0, readbackSize, vk::MemoryMapFlags{}, &mapped)) {
        std::println(stderr, "readback map failed"); return 1;
    }
    const auto* bytes = static_cast<const std::uint8_t*>(mapped);
    auto pixel = [&](int x, int y) {
        const std::uint8_t* p = bytes + (y * kWidth + x) * 4;
        return std::array<std::uint8_t, 4>{p[0], p[1], p[2], p[3]};
    };
    auto center = pixel(kWidth / 2, kHeight / 2);
    auto corner = pixel(100, 100);
    device->unmapMemory(*readbackMemory);

    bool centerBlue = center[0] < 60 && center[2] > 200;   // blue
    bool cornerRed = corner[0] > 200 && corner[2] < 60;    // red
    std::println("depth readback: center={} ({}) corner={} ({})",
                 std::array<int, 4>{center[0], center[1], center[2], center[3]}, centerBlue ? "blue" : "wrong",
                 std::array<int, 4>{corner[0], corner[1], corner[2], corner[3]}, cornerRed ? "red" : "wrong");
    if (!centerBlue || !cornerRed) { std::println(stderr, "FAIL: depth verification failed"); return 1; }
    std::println("PASS: depth test verified");
    if (!sample::reportValidation(validation)) return 1;
    return 0;
}
