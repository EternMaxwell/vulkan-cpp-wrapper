// 02-triangle: the official Vulkan "triangle" sample mapped onto the wrapper.
//
// Unlike a bare setup/teardown, this sample actually renders and presents in
// a bounded loop, and then verifies the output by rendering offscreen and
// reading pixels back to the CPU. It exercises image views, render passes,
// framebuffers, command buffers, a graphics pipeline, a vertex buffer,
// device-memory binding, swapchain acquire/submit/present, and readback.
#include <volk.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan_wrapper.hpp>
#include <validation.hpp>

#include <array>
#include <cstdint>
#include <cstring>
#include <print>
#include <vector>

static constexpr int kWidth = 800;
static constexpr int kHeight = 600;
static constexpr int kMaxFrames = 120;  // bounded render loop (not infinite)

struct Vertex {
    float x, y;
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

static vk::ShaderModule load_shader(const vk::Device& device, const char* path) {
    auto code = read_spirv(path);
    vk::ShaderModuleCreateInfo info{};
    info.setCode(std::move(code));
    auto module = device.createShaderModule(info, std::nullopt);
    if (!module) std::println(stderr, "error: createShaderModule: {}", path);
    return *module;
}

// Find a memory type with the requested property flags.
static std::uint32_t find_memory_type(const vk::PhysicalDevice& physical,
                                      vk::MemoryPropertyFlags required) {
    auto props = physical.getMemoryProperties();
    for (std::uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if ((props.memoryTypes[i].propertyFlags & required) == required) return i;
    }
    return 0;
}

// Pick a queue family that supports graphics and presentation.
static std::uint32_t find_queue_family(const vk::PhysicalDevice& physical,
                                       const vk::SurfaceKHR& surface) {
    auto families = physical.getQueueFamilyProperties();
    for (std::uint32_t i = 0; i < families.size(); ++i) {
        if (!(families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) continue;
        auto supported = physical.getSurfaceSupportKHR(i, surface);
        if (supported && *supported) return i;
    }
    for (std::uint32_t i = 0; i < families.size(); ++i)
        if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) return i;
    return 0;
}

static vk::Pipeline create_pipeline(const vk::Device& device,
                                    const vk::RenderPass& renderPass,
                                    const vk::PipelineLayout& layout,
                                    const vk::ShaderModule& vert,
                                    const vk::ShaderModule& frag) {
    vk::PipelineShaderStageCreateInfo vertStage{};
    vertStage.setStage(vk::ShaderStageFlagBits::Vertex).setModule(vert).setName("main");
    vk::PipelineShaderStageCreateInfo fragStage{};
    fragStage.setStage(vk::ShaderStageFlagBits::Fragment).setModule(frag).setName("main");

    vk::VertexInputBindingDescription binding{};
    binding.setBinding(0).setStride(sizeof(Vertex)).setInputRate(vk::VertexInputRate::Vertex);
    vk::VertexInputAttributeDescription attribute{};
    attribute.setLocation(0).setBinding(0).setFormat(vk::Format::R32g32Sfloat).setOffset(0);

    vk::PipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.setVertexBindingDescriptions({binding})
               .setVertexAttributeDescriptions({attribute});
    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.setTopology(vk::PrimitiveTopology::TriangleList)
                 .setPrimitiveRestartEnable(false);
    vk::Viewport viewport{};
    viewport.setX(0.0f).setY(0.0f)
            .setWidth(static_cast<float>(kWidth)).setHeight(static_cast<float>(kHeight))
            .setMinDepth(0.0f).setMaxDepth(1.0f);
    vk::Rect2D scissor{};
    scissor.setOffset(vk::Offset2D{0, 0}).setExtent(vk::Extent2D{kWidth, kHeight});
    vk::PipelineViewportStateCreateInfo viewportState{};
    viewportState.setViewports({viewport}).setScissors({scissor});
    vk::PipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.setDepthClampEnable(false)
              .setRasterizerDiscardEnable(false)
              .setPolygonMode(vk::PolygonMode::Fill)
              .setCullMode(vk::CullModeFlagBits::None)
              .setFrontFace(vk::FrontFace::CounterClockwise)
              .setLineWidth(1.0f);
    vk::PipelineMultisampleStateCreateInfo multisampling{};
    multisampling.setRasterizationSamples(vk::SampleCountFlagBits::Value1)
                 .setSampleShadingEnable(false);
    vk::PipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.setBlendEnable(false)
                        .setColorWriteMask(
                            vk::ColorComponentFlagBits::R | vk::ColorComponentFlagBits::G |
                            vk::ColorComponentFlagBits::B | vk::ColorComponentFlagBits::A);
    vk::PipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.setLogicOpEnable(false).setAttachments({colorBlendAttachment});

    vk::GraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.setStages({vertStage, fragStage})
                .setVertexInputState(vertexInput)
                .setInputAssemblyState(inputAssembly)
                .setViewportState(viewportState)
                .setRasterizationState(rasterizer)
                .setMultisampleState(multisampling)
                .setColorBlendState(colorBlending)
                .setLayout(layout)
                .setRenderPass(renderPass)
                .setSubpass(0);

    auto result = device.createGraphicsPipelines(vk::PipelineCache{}, std::array{pipelineInfo}, std::nullopt);
    if (result.status != vk::ResultCode::Success || result.value.empty()) {
        std::println(stderr, "error: createGraphicsPipelines: {}", static_cast<int>(result.status));
        return {};
    }
    return std::move(result.value[0]);
}

// Records just the draw (render pass + pipeline + vertex buffer + draw) into
// an already-begun command buffer; the caller owns begin()/end().
static void record_draw(const vk::CommandBuffer& cmd,
                        const vk::Pipeline& pipeline,
                        const vk::RenderPass& renderPass,
                        const vk::Framebuffer& framebuffer,
                        const vk::Buffer& vertexBuffer) {
    vk::RenderPassBeginInfo renderPassInfo{};
    renderPassInfo.setRenderPass(renderPass)
                  .setFramebuffer(framebuffer)
                  .setRenderArea(vk::Rect2D{{0, 0}, {kWidth, kHeight}});
    vk::ClearValue clear{};
    clear.color.float32[0] = 0.0f;  // R
    clear.color.float32[1] = 0.0f;  // G
    clear.color.float32[2] = 1.0f;  // B
    clear.color.float32[3] = 1.0f;  // A
    renderPassInfo.setClearValues({clear});

    cmd.beginRenderPass(renderPassInfo, vk::SubpassContents::Inline);
    cmd.bindPipeline(vk::PipelineBindPoint::Graphics, pipeline);
    cmd.bindVertexBuffers(0, std::array{vertexBuffer}, std::array{vk::DeviceSize{0}});
    cmd.draw(3, 1, 0, 0);
    cmd.endRenderPass();
}

int main() {
    if (!glfwInit()) { std::println(stderr, "glfwInit failed"); return 1; }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(kWidth, kHeight, "vulkan-wrapper triangle", nullptr, nullptr);
    if (!window) { std::println(stderr, "glfwCreateWindow failed"); return 1; }

    if (volkInitialize() != VK_SUCCESS) { std::println(stderr, "volkInitialize failed"); return 1; }
    auto context = vk::Context::create();
    if (!context) { std::println(stderr, "Context::create failed"); return 1; }

    vk::ApplicationInfo app{};
    app.setApplicationName("vulkan-wrapper-triangle");
    app.setApiVersion(VK_API_VERSION_1_3);
    vk::InstanceCreateInfo instanceInfo{};
    instanceInfo.setApplicationInfo(app);
    instanceInfo.setEnabledExtensionNames(
        {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME});
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

    VkSurfaceKHR rawSurface{};
    if (glfwCreateWindowSurface(instance->raw(), window, nullptr, &rawSurface) != VK_SUCCESS) {
        std::println(stderr, "glfwCreateWindowSurface failed"); return 1;
    }
    auto surface = vk::SurfaceKHR::adopt(
        rawSurface, *instance,
        [inst = *instance](VkSurfaceKHR s) noexcept {
            VolkInstanceTable table{};
            volkLoadInstanceTable(&table, inst.raw());
            table.vkDestroySurfaceKHR(inst.raw(), s, nullptr);
        });
    if (!surface) { std::println(stderr, "SurfaceKHR::adopt failed"); return 1; }

    auto devices = instance->enumeratePhysicalDevices();
    if (devices.value.empty()) { std::println(stderr, "no physical devices"); return 1; }
    auto physical = std::move(devices.value[0]);
    auto props = physical.getProperties();
    std::println("Using GPU: {}", props.deviceName.data());

    std::uint32_t queueFamily = find_queue_family(physical, *surface);

    vk::DeviceQueueCreateInfo queueInfo{};
    queueInfo.setQueueFamilyIndex(queueFamily).setQueuePriorities({1.0f});
    vk::DeviceCreateInfo deviceInfo{};
    deviceInfo.setQueueCreateInfos({queueInfo});
    deviceInfo.setEnabledExtensionNames({VK_KHR_SWAPCHAIN_EXTENSION_NAME});
    auto device = physical.createDevice(deviceInfo, std::nullopt);
    if (!device) { std::println(stderr, "createDevice failed"); return 1; }
    auto queue = device->getQueue(queueFamily, 0);

    vk::SwapchainCreateInfoKHR swapInfo{};
    swapInfo.setSurface(*surface).setMinImageCount(2)
           .setImageFormat(vk::Format::B8g8r8a8Unorm)
           .setImageColorSpace(vk::ColorSpaceKHR::SrgbNonlinear)
           .setImageExtent(vk::Extent2D{kWidth, kHeight})
           .setImageArrayLayers(1)
           .setImageUsage(vk::ImageUsageFlagBits::ColorAttachment)
           .setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::Opaque)
           .setPreTransform(vk::SurfaceTransformFlagBitsKHR::Identity)
           .setPresentMode(vk::PresentModeKHR::Fifo);
    auto swapchain = device->createSwapchainKHR(swapInfo, std::nullopt);
    if (!swapchain) { std::println(stderr, "createSwapchainKHR failed"); return 1; }
    auto swapImages = device->getSwapchainImagesKHR(*swapchain);
    if (swapImages.status != vk::ResultCode::Success) {
        std::println(stderr, "getSwapchainImagesKHR failed"); return 1;
    }

    // Image views + framebuffers for the swapchain.
    std::vector<vk::ImageView> swapViews;
    std::vector<vk::Framebuffer> framebuffers;
    vk::ImageSubresourceRange subresource{};
    subresource.setAspectMask(vk::ImageAspectFlagBits::Color)
               .setBaseMipLevel(0).setLevelCount(1)
               .setBaseArrayLayer(0).setLayerCount(1);
    for (const auto& image : swapImages.value) {
        vk::ImageViewCreateInfo viewInfo{};
        viewInfo.setImage(image).setViewType(vk::ImageViewType::Value2d)
                .setFormat(vk::Format::B8g8r8a8Unorm)
                .setSubresourceRange(subresource);
        auto view = device->createImageView(viewInfo, std::nullopt);
        if (!view) { std::println(stderr, "createImageView failed"); return 1; }
        swapViews.push_back(std::move(*view));
    }

    vk::AttachmentDescription colorAttachment{};
    colorAttachment.setFormat(vk::Format::B8g8r8a8Unorm)
                   .setSamples(vk::SampleCountFlagBits::Value1)
                   .setLoadOp(vk::AttachmentLoadOp::Clear)
                   .setStoreOp(vk::AttachmentStoreOp::Store)
                   .setStencilLoadOp(vk::AttachmentLoadOp::DontCare)
                   .setStencilStoreOp(vk::AttachmentStoreOp::DontCare)
                   .setInitialLayout(vk::ImageLayout::Undefined)
                   .setFinalLayout(vk::ImageLayout::PresentSrcKhr);
    vk::AttachmentReference colorRef{};
    colorRef.setAttachment(0).setLayout(vk::ImageLayout::ColorAttachmentOptimal);
    vk::SubpassDescription subpass{};
    subpass.setPipelineBindPoint(vk::PipelineBindPoint::Graphics).setColorAttachments({colorRef});
    vk::RenderPassCreateInfo renderPassInfo{};
    renderPassInfo.setAttachments({colorAttachment}).setSubpasses({subpass});
    auto renderPass = device->createRenderPass(renderPassInfo, std::nullopt);
    if (!renderPass) { std::println(stderr, "createRenderPass failed"); return 1; }

    for (const auto& view : swapViews) {
        vk::FramebufferCreateInfo fbInfo{};
        fbInfo.setRenderPass(*renderPass).setAttachments({view})
              .setWidth(kWidth).setHeight(kHeight).setLayers(1);
        auto fb = device->createFramebuffer(fbInfo, std::nullopt);
        if (!fb) { std::println(stderr, "createFramebuffer failed"); return 1; }
        framebuffers.push_back(std::move(*fb));
    }

    // Shaders, pipeline layout, and graphics pipeline.
    auto vert = load_shader(*device, SHADER_VERT);
    auto frag = load_shader(*device, SHADER_FRAG);
    vk::PipelineLayoutCreateInfo layoutInfo{};
    auto layout = device->createPipelineLayout(layoutInfo, std::nullopt);
    if (!layout) { std::println(stderr, "createPipelineLayout failed"); return 1; }
    auto pipeline = create_pipeline(*device, *renderPass, *layout, vert, frag);
    if (!pipeline) return 1;

    // Vertex buffer (host-visible; GPU reads over PCIe for this simple case).
    const std::vector<Vertex> vertices = {{-0.5f, -0.5f}, {0.5f, -0.5f}, {0.0f, 0.5f}};
    const vk::DeviceSize vbSize = sizeof(Vertex) * vertices.size();
    auto vertexBuffer = device->createBuffer(
        vk::BufferCreateInfo{}.setSize(vbSize)
            .setUsage(vk::BufferUsageFlagBits::VertexBuffer)
            .setSharingMode(vk::SharingMode::Exclusive), std::nullopt);
    if (!vertexBuffer) { std::println(stderr, "createBuffer failed"); return 1; }
    auto vbReqs = device->getBufferMemoryRequirements(*vertexBuffer);
    auto vertexMemory = device->allocateMemory(
        vk::MemoryAllocateInfo{}.setAllocationSize(vbReqs.size)
            .setMemoryTypeIndex(find_memory_type(physical,
                vk::MemoryPropertyFlagBits::HostVisible | vk::MemoryPropertyFlagBits::HostCoherent)), std::nullopt);
    if (!vertexMemory) { std::println(stderr, "allocateMemory failed"); return 1; }
    if (!device->bindBufferMemory(*vertexBuffer, *vertexMemory, 0)) {
        std::println(stderr, "bindBufferMemory failed"); return 1;
    }
    void* mapped = nullptr;
    if (!device->mapMemory(*vertexMemory, 0, vbSize, vk::MemoryMapFlags{}, &mapped)) {
        std::println(stderr, "mapMemory failed"); return 1;
    }
    std::memcpy(mapped, vertices.data(), static_cast<std::size_t>(vbSize));
    device->unmapMemory(*vertexMemory);

    // Command pool + per-frame command buffers.
    auto commandPool = device->createCommandPool(
        vk::CommandPoolCreateInfo{}.setQueueFamilyIndex(queueFamily)
            .setFlags(vk::CommandPoolCreateFlagBits::ResetCommandBuffer), std::nullopt);
    if (!commandPool) { std::println(stderr, "createCommandPool failed"); return 1; }
    auto commandBuffers = device->allocateCommandBuffers(
        vk::CommandBufferAllocateInfo{}.setCommandPool(*commandPool)
            .setLevel(vk::CommandBufferLevel::Primary)
            .setCommandBufferCount(static_cast<std::uint32_t>(framebuffers.size())));
    if (!commandBuffers) { std::println(stderr, "allocateCommandBuffers failed"); return 1; }

    // Synchronization. renderFinished is per-swapchain-image: a present waits
    // on it asynchronously, so reusing a single semaphore for every frame would
    // re-signal it while the previous present is still consuming it.
    auto imageAvailable = device->createSemaphore(vk::SemaphoreCreateInfo{}, std::nullopt);
    std::vector<vk::Semaphore> renderFinished;
    for (std::size_t i = 0; i < swapImages.value.size(); ++i) {
        auto rf = device->createSemaphore(vk::SemaphoreCreateInfo{}, std::nullopt);
        if (!rf) { std::println(stderr, "sync object creation failed"); return 1; }
        renderFinished.push_back(std::move(*rf));
    }
    auto inFlight = device->createFence(
        vk::FenceCreateInfo{}.setFlags(vk::FenceCreateFlagBits::Signaled), std::nullopt);
    if (!imageAvailable || !inFlight) {
        std::println(stderr, "sync object creation failed"); return 1;
    }

    // ---- Render + present loop (bounded). ----
    std::println("rendering {} frames", kMaxFrames);
    int rendered = 0;
    while (rendered < kMaxFrames && !glfwWindowShouldClose(window)) {
        glfwPollEvents();
        if (!device->waitForFences(std::array{*inFlight}, true, UINT64_MAX)) {
            std::println(stderr, "waitForFences failed"); return 1;
        }
        if (!device->resetFences(std::array{*inFlight})) {
            std::println(stderr, "resetFences failed"); return 1;
        }
        auto acquired = device->acquireNextImageKHR(
            *swapchain, UINT64_MAX, *imageAvailable, vk::Fence{});
        if (acquired.status == vk::ResultCode::ErrorOutOfDateKhr) {
            std::println(stderr, "acquire out-of-date (resize not implemented)"); break;
        }
        if (acquired.status != vk::ResultCode::Success &&
            acquired.status != vk::ResultCode::SuboptimalKhr) {
            std::println(stderr, "acquireNextImageKHR: {}", static_cast<int>(acquired.status));
            return 1;
        }
        std::uint32_t imageIndex = acquired.value;

        auto& cmd = commandBuffers.value()[imageIndex];
        cmd.reset(vk::CommandBufferResetFlags{});
        vk::CommandBufferBeginInfo beginInfo{};
        cmd.begin(beginInfo);
        record_draw(cmd, pipeline, *renderPass, framebuffers[imageIndex], *vertexBuffer);
        cmd.end();

        vk::SubmitInfo submit{};
        submit.setWaitSemaphores({*imageAvailable})
              .setWaitDstStageMask({vk::PipelineStageFlagBits::ColorAttachmentOutput})
              .setCommandBuffers({cmd})
              .setSignalSemaphores({renderFinished[imageIndex]});
        if (!queue.submit(std::array{submit}, *inFlight)) {
            std::println(stderr, "queueSubmit failed"); return 1;
        }

        vk::PresentInfoKHR present{};
        present.setWaitSemaphores({renderFinished[imageIndex]})
               .setSwapchains({*swapchain})
               .setImageIndices({imageIndex});
        auto presentResult = queue.presentKHR(present);
        if (!presentResult) {
            std::println(stderr, "queuePresentKHR: {}", static_cast<int>(presentResult.error()));
            return 1;
        }
        ++rendered;
    }
    device->waitIdle();
    std::println("rendered {} frames", rendered);

    // ---- Offscreen readback: prove pixels are actually produced. ----
    // Render the same triangle into a dedicated color image, copy it to a
    // host-visible buffer, and check the center pixel (triangle = red) and a
    // corner pixel (clear color = blue).
    vk::ImageCreateInfo imageInfo{};
    imageInfo.setImageType(vk::ImageType::Value2d)
             .setFormat(vk::Format::B8g8r8a8Unorm)
             .setExtent(vk::Extent3D{kWidth, kHeight, 1})
             .setMipLevels(1).setArrayLayers(1)
             .setSamples(vk::SampleCountFlagBits::Value1)
             .setTiling(vk::ImageTiling::Optimal)
             .setUsage(vk::ImageUsageFlagBits::ColorAttachment | vk::ImageUsageFlagBits::TransferSrc)
             .setSharingMode(vk::SharingMode::Exclusive)
             .setInitialLayout(vk::ImageLayout::Undefined);
    auto offscreenImage = device->createImage(imageInfo, std::nullopt);
    if (!offscreenImage) { std::println(stderr, "offscreen createImage failed"); return 1; }
    auto imageReqs = device->getImageMemoryRequirements(*offscreenImage);
    auto offscreenMemory = device->allocateMemory(
        vk::MemoryAllocateInfo{}.setAllocationSize(imageReqs.size)
            .setMemoryTypeIndex(find_memory_type(physical,
                vk::MemoryPropertyFlagBits::DeviceLocal)), std::nullopt);
    if (!offscreenMemory || !device->bindImageMemory(*offscreenImage, *offscreenMemory, 0)) {
        std::println(stderr, "offscreen memory bind failed"); return 1;
    }
    auto offscreenView = device->createImageView(
        vk::ImageViewCreateInfo{}.setImage(*offscreenImage)
            .setViewType(vk::ImageViewType::Value2d)
            .setFormat(vk::Format::B8g8r8a8Unorm)
            .setSubresourceRange(subresource), std::nullopt);
    if (!offscreenView) { std::println(stderr, "offscreen createImageView failed"); return 1; }

    // A render pass whose final layout is TRANSFER_SRC so copyImageToBuffer
    // needs no explicit barrier.
    vk::AttachmentDescription readbackAttachment{};
    readbackAttachment.setFormat(vk::Format::B8g8r8a8Unorm)
                      .setSamples(vk::SampleCountFlagBits::Value1)
                      .setLoadOp(vk::AttachmentLoadOp::Clear)
                      .setStoreOp(vk::AttachmentStoreOp::Store)
                      .setStencilLoadOp(vk::AttachmentLoadOp::DontCare)
                      .setStencilStoreOp(vk::AttachmentStoreOp::DontCare)
                      .setInitialLayout(vk::ImageLayout::Undefined)
                      .setFinalLayout(vk::ImageLayout::TransferSrcOptimal);
    vk::RenderPassCreateInfo readbackPassInfo{};
    readbackPassInfo.setAttachments({readbackAttachment}).setSubpasses({subpass});
    auto readbackPass = device->createRenderPass(readbackPassInfo, std::nullopt);
    if (!readbackPass) { std::println(stderr, "readback createRenderPass failed"); return 1; }
    auto readbackPipeline = create_pipeline(*device, *readbackPass, *layout, vert, frag);
    if (!readbackPipeline) return 1;
    auto readbackFramebuffer = device->createFramebuffer(
        vk::FramebufferCreateInfo{}.setRenderPass(*readbackPass)
            .setAttachments({*offscreenView})
            .setWidth(kWidth).setHeight(kHeight).setLayers(1), std::nullopt);
    if (!readbackFramebuffer) { std::println(stderr, "readback createFramebuffer failed"); return 1; }

    const vk::DeviceSize readbackSize = static_cast<vk::DeviceSize>(kWidth) * kHeight * 4;
    auto readbackBuffer = device->createBuffer(
        vk::BufferCreateInfo{}.setSize(readbackSize)
            .setUsage(vk::BufferUsageFlagBits::TransferDst)
            .setSharingMode(vk::SharingMode::Exclusive), std::nullopt);
    auto readbackReqs = device->getBufferMemoryRequirements(*readbackBuffer);
    auto readbackMemory = device->allocateMemory(
        vk::MemoryAllocateInfo{}.setAllocationSize(readbackReqs.size)
            .setMemoryTypeIndex(find_memory_type(physical,
                vk::MemoryPropertyFlagBits::HostVisible | vk::MemoryPropertyFlagBits::HostCoherent)), std::nullopt);
    if (!readbackBuffer || !readbackMemory ||
        !device->bindBufferMemory(*readbackBuffer, *readbackMemory, 0)) {
        std::println(stderr, "readback buffer setup failed"); return 1;
    }

    auto readbackCmd = commandBuffers.value()[0];
    readbackCmd.reset(vk::CommandBufferResetFlags{});
    vk::CommandBufferBeginInfo rbBeginInfo{};
    readbackCmd.begin(rbBeginInfo);
    record_draw(readbackCmd, readbackPipeline, *readbackPass, *readbackFramebuffer, *vertexBuffer);
    vk::BufferImageCopy region{};
    region.setBufferOffset(0).setBufferRowLength(0).setBufferImageHeight(0)
          .setImageSubresource(vk::ImageSubresourceLayers{}
              .setAspectMask(vk::ImageAspectFlagBits::Color)
              .setMipLevel(0).setBaseArrayLayer(0).setLayerCount(1))
          .setImageOffset(vk::Offset3D{0, 0, 0})
          .setImageExtent(vk::Extent3D{kWidth, kHeight, 1});
    readbackCmd.copyImageToBuffer(*offscreenImage, vk::ImageLayout::TransferSrcOptimal,
                                  *readbackBuffer, std::array{region});
    readbackCmd.end();
    vk::SubmitInfo readbackSubmit{};
    readbackSubmit.setCommandBuffers({readbackCmd});
    if (!queue.submit(std::array{readbackSubmit}, vk::Fence{}) || !device->waitIdle()) {
        std::println(stderr, "readback submit failed"); return 1;
    }

    void* pixels = nullptr;
    if (!device->mapMemory(*readbackMemory, 0, readbackSize, vk::MemoryMapFlags{}, &pixels)) {
        std::println(stderr, "readback mapMemory failed"); return 1;
    }
    const auto* bytes = static_cast<const std::uint8_t*>(pixels);
    // B8G8R8A8: blue, green, red, alpha.
    auto pixel = [&](int x, int y) {
        const std::uint8_t* p = bytes + (y * kWidth + x) * 4;
        return std::array<std::uint8_t, 4>{p[0], p[1], p[2], p[3]};
    };
    auto center = pixel(kWidth / 2, kHeight / 2);
    auto corner = pixel(4, 4);
    device->unmapMemory(*readbackMemory);

    // Triangle interior is red (1,0,0,1) -> B8G8R8A8 {0,0,255,255}.
    // Clear color is blue (0,0,1,1) -> B8G8R8A8 {255,0,0,255}.
    bool centerRed = center[2] > 200 && center[0] < 60;
    bool cornerBlue = corner[0] > 200 && corner[2] < 60;
    std::println("readback center B8G8R8A8 = [{},{},{},{}] ({}red)",
                 center[0], center[1], center[2], center[3], centerRed ? "" : "not ");
    std::println("readback corner B8G8R8A8 = [{},{},{},{}] ({}blue)",
                 corner[0], corner[1], corner[2], corner[3], cornerBlue ? "" : "not ");

    glfwDestroyWindow(window);
    glfwTerminate();
    if (!centerRed || !cornerBlue) {
        std::println(stderr, "FAIL: render verification failed");
        return 1;
    }
    std::println("PASS: triangle rendered and verified");
    if (!sample::reportValidation(validation)) return 1;
    return 0;
}
