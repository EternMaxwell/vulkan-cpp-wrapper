# Samples

Real, runnable examples that use the generated Vulkan wrapper. Dependencies
(GLFW, GLM, volk, Vulkan-Headers) are pulled in with CMake `FetchContent`; the
wrapper header itself is generated at build time by `vulkan-wrapper-gen`.

## Build

```console
# Install the generator first (from the repository root)
python -m pip install -e .

cmake -S samples -B samples/build
cmake --build samples/build
```

## Samples

- `01-info` — enumerates the instance and physical devices/queue families.
  No windowing or GPU surface required.
- `02-triangle` — the classic "hello triangle" mapped onto the full pipeline:
  image views, render pass, framebuffers, command buffers, a graphics pipeline,
  a vertex buffer, swapchain acquire/submit/present, and synchronization. It
  renders a bounded frame loop and then proves the output by rendering
  offscreen and reading pixels back (center = triangle color, corner = clear
  color). Requires `glfw` and `glslc`.
- `03-compute` — a compute-only kernel (`out[i] = in[i] * 2`) that exercises
  descriptor sets/pools/layouts, storage buffers, a compute pipeline,
  `dispatch`, and buffer barriers, then verifies every element on the CPU.
  Requires `glslc`.
- `04-textured` — an offscreen textured, indexed quad that exercises the
  sampled-image path end to end: staging upload + `copyBufferToImage`, image
  layout barriers, a `Sampler` + combined-image-sampler descriptor, a uniform
  buffer, push constants, interleaved vertex attributes, and `drawIndexed`.
  The 2x2 checkerboard's four quadrants are checked against their texels.
  Requires `glslc`.
- `05-depth` — an offscreen depth-tested scene (far red triangle under a near
  blue triangle) exercising a D32 depth buffer, a depth attachment, depth/stencil
  state, and per-vertex color. The overlap region is checked to be blue (near
  wins) and the background red (far). Requires `glslc`.
- `06-pnext` — the wrapper's pNext `ExtensionChain` (a single linked list): a
  core Vulkan 1.3 feature query round-trip via `nextInChain.get<T>()`, multi-node
  chains built with either the nested setter
  (`a.setNextInChain(std::move(b.setNextInChain(c)))`) or the mutable node
  accessor, and feature enablement at device creation. No shaders.
- `07-vma` — the VulkanMemoryAllocator integration: `Allocator::create`,
  VMA-backed buffer/image creation (with creation records and allocation
  metadata), and a raw allocation + `AllocationView::map`/`unmap` round-trip.
  Uses a separate VMA-enabled wrapper (`vulkan_wrapper_vma.hpp`) so the other
  samples carry no VMA dependency. No shaders.

## Notes / known limitations

- Every sample enables `VK_LAYER_KHRONOS_validation` when the loader exposes it
  (see `common/validation.hpp`) and registers a debug-utils messenger, then
  fails if any ERROR-level validation message was emitted. This turned the
  samples into an API-misuse checker: it caught a zero `compositeAlpha` on the
  swapchain, a reused presentation semaphore in `02-triangle`, and command
  buffers reset from pools lacking `VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT`.
- Samples enable the extensions whose commands they call
  (`VK_KHR_surface` + `VK_KHR_win32_surface` on the instance,
  `VK_KHR_swapchain` on the device). Command dispatch is null-guarded: calling
  an extension command whose extension was not enabled returns
  `ErrorExtensionNotPresent` instead of crashing.
- `02-triangle` creates the window surface via `glfwCreateWindowSurface` and
  adopts it into a `SurfaceKHR`; because the wrapper keeps its dispatch tables
  private, the sample loads a local `VolkInstanceTable` in the surface's
  destroyer. A future generator improvement is to auto-derive the deleter for
  `adopt` from the handle's known releaser command.
- These samples surfaced and drove fixes for several generator bugs, including
  the `descriptorCount` field of `VkDescriptorSetLayoutBinding` being wrongly
  derived from `pImmutableSamplers.size()` (it is now an explicit field).
- Device creation enables the wrapper's private-data handle tracking without
  tripping `VUID-VkDeviceCreateInfo-pNext-06532`: when the user chains a
  `VkPhysicalDeviceVulkan13Features` node, the wrapper sets its `privateData`
  member directly instead of also injecting the older
  `VkPhysicalDevicePrivateDataFeatures` extension struct.
