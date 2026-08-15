cmake_minimum_required(VERSION 3.24)
project(vulkan-wrapper-samples LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(REPO_ROOT ${CMAKE_CURRENT_LIST_DIR}/..)

include(FetchContent)

# ---------------------------------------------------------------------------
# Dependencies (FetchContent).  Vulkan-Headers is made available first because
# both volk and GLFW need its include directory.
# ---------------------------------------------------------------------------
FetchContent_Declare(vulkan_headers
    GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
    GIT_TAG        vulkan-sdk-1.4.357.0
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(vulkan_headers)

set(VULKAN_HEADERS_INSTALL_DIR ${vulkan_headers_SOURCE_DIR} CACHE PATH "" FORCE)
set(VOLK_PULL_IN_VULKAN ON CACHE BOOL "" FORCE)

FetchContent_Declare(volk
    GIT_REPOSITORY https://github.com/zeux/volk.git
    GIT_TAG        vulkan-sdk-1.4.357.0
    GIT_SHALLOW    TRUE)

FetchContent_Declare(glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.4
    GIT_SHALLOW    TRUE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL        OFF CACHE BOOL "" FORCE)

FetchContent_Declare(glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        1.0.1
    GIT_SHALLOW    TRUE)

FetchContent_Declare(vma
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
    GIT_TAG        v3.4.0
    GIT_SHALLOW    TRUE)

FetchContent_MakeAvailable(volk glfw glm vma)

# GLFW's vulkan.c needs the Vulkan headers to build glfwCreateWindowSurface,
# and the platform define to compile the real (non-stub) surface creation.
if (TARGET glfw)
    target_include_directories(glfw PRIVATE ${vulkan_headers_SOURCE_DIR}/include)
    if (WIN32)
        target_compile_definitions(glfw PRIVATE VK_USE_PLATFORM_WIN32_KHR)
    endif()
endif()

# ---------------------------------------------------------------------------
# Generate the wrapper header into the build tree.
# ---------------------------------------------------------------------------
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(WRAPPER ${CMAKE_BINARY_DIR}/generated/vulkan_wrapper.hpp)
file(GLOB_RECURSE WRAPPER_GEN_SOURCES
    ${REPO_ROOT}/src/vulkan_wrapper_gen/*.py
    ${REPO_ROOT}/templates/*.template.hpp)
add_custom_command(
    OUTPUT ${WRAPPER}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/generated
    COMMAND ${Python3_EXECUTABLE} -m vulkan_wrapper_gen
        --registry ${vulkan_headers_SOURCE_DIR}/registry/vk.xml
        --registry ${vulkan_headers_SOURCE_DIR}/registry/video.xml
        --emit ${REPO_ROOT}/templates/vulkan-header-only.template.hpp:${WRAPPER}
    WORKING_DIRECTORY ${REPO_ROOT}
    DEPENDS ${WRAPPER_GEN_SOURCES}
    COMMENT "Generating vulkan_wrapper.hpp"
    VERBATIM)
add_custom_target(generate_wrapper DEPENDS ${WRAPPER})

# A second wrapper with VMA support, used only by the 07-vma sample so the
# other samples do not carry an unconditional VMA dependency.
set(WRAPPER_VMA ${CMAKE_BINARY_DIR}/generated/vulkan_wrapper_vma.hpp)
add_custom_command(
    OUTPUT ${WRAPPER_VMA}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/generated
    COMMAND ${Python3_EXECUTABLE} -m vulkan_wrapper_gen
        --registry ${vulkan_headers_SOURCE_DIR}/registry/vk.xml
        --registry ${vulkan_headers_SOURCE_DIR}/registry/video.xml
        --vma-header ${vma_SOURCE_DIR}/include/vk_mem_alloc.h
        --clang-arg=-I${vulkan_headers_SOURCE_DIR}/include
        --emit ${REPO_ROOT}/templates/vulkan-header-only.template.hpp:${WRAPPER_VMA}
    WORKING_DIRECTORY ${REPO_ROOT}
    DEPENDS ${WRAPPER_GEN_SOURCES}
    COMMENT "Generating vulkan_wrapper_vma.hpp"
    VERBATIM)
add_custom_target(generate_wrapper_vma DEPENDS ${WRAPPER_VMA})

function(wrapper_sample target)
    target_link_libraries(${target} PRIVATE volk::volk_headers volk::volk)
    target_include_directories(${target} PRIVATE
        ${CMAKE_BINARY_DIR}/generated
        ${vulkan_headers_SOURCE_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/common)
    if (WIN32)
        target_compile_definitions(${target} PRIVATE VK_USE_PLATFORM_WIN32_KHR NOMINMAX)
    endif()
    add_dependencies(${target} generate_wrapper)
endfunction()

# ---------------------------------------------------------------------------
# 01-info: no windowing; always builds.
# ---------------------------------------------------------------------------
add_executable(info 01-info/main.cpp)
wrapper_sample(info)

# ---------------------------------------------------------------------------
# 02-triangle: GLFW window + glslc-compiled shaders.
# ---------------------------------------------------------------------------
find_program(GLSLC glslc)
if (TARGET glfw AND GLSLC)
    foreach(shader vert frag)
        add_custom_command(
            OUTPUT ${CMAKE_BINARY_DIR}/shader.${shader}.spv
            COMMAND ${GLSLC} ${CMAKE_CURRENT_LIST_DIR}/02-triangle/shaders/shader.${shader}
                    -o ${CMAKE_BINARY_DIR}/shader.${shader}.spv
            DEPENDS ${CMAKE_CURRENT_LIST_DIR}/02-triangle/shaders/shader.${shader}
            COMMENT "Compiling shader.${shader} to SPIR-V")
    endforeach()
    add_custom_target(triangle_shaders ALL
        DEPENDS ${CMAKE_BINARY_DIR}/shader.vert.spv ${CMAKE_BINARY_DIR}/shader.frag.spv)

    add_executable(triangle 02-triangle/main.cpp)
    wrapper_sample(triangle)
    target_link_libraries(triangle PRIVATE glfw)
    add_dependencies(triangle triangle_shaders)
    target_compile_definitions(triangle PRIVATE
        SHADER_VERT="${CMAKE_BINARY_DIR}/shader.vert.spv"
        SHADER_FRAG="${CMAKE_BINARY_DIR}/shader.frag.spv")
else()
    message(STATUS "02-triangle skipped: requires GLFW and glslc")
endif()

# ---------------------------------------------------------------------------
# 03-compute: compute-only sample; needs glslc.
# ---------------------------------------------------------------------------
if (GLSLC)
    add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/compute.spv
        COMMAND ${GLSLC} ${CMAKE_CURRENT_LIST_DIR}/03-compute/shaders/compute.comp
                -o ${CMAKE_BINARY_DIR}/compute.spv
        DEPENDS ${CMAKE_CURRENT_LIST_DIR}/03-compute/shaders/compute.comp
        COMMENT "Compiling compute.comp to SPIR-V")
    add_custom_target(compute_shaders ALL DEPENDS ${CMAKE_BINARY_DIR}/compute.spv)

    add_executable(compute 03-compute/main.cpp)
    wrapper_sample(compute)
    add_dependencies(compute compute_shaders)
    target_compile_definitions(compute PRIVATE
        SHADER_COMP="${CMAKE_BINARY_DIR}/compute.spv")
else()
    message(STATUS "03-compute skipped: requires glslc")
endif()

# ---------------------------------------------------------------------------
# 04-textured: offscreen textured quad; needs glslc.
# ---------------------------------------------------------------------------
if (GLSLC)
    foreach(shader vert frag)
        add_custom_command(
            OUTPUT ${CMAKE_BINARY_DIR}/texture.${shader}.spv
            COMMAND ${GLSLC} ${CMAKE_CURRENT_LIST_DIR}/04-textured/shaders/texture.${shader}
                    -o ${CMAKE_BINARY_DIR}/texture.${shader}.spv
            DEPENDS ${CMAKE_CURRENT_LIST_DIR}/04-textured/shaders/texture.${shader}
            COMMENT "Compiling texture.${shader} to SPIR-V")
    endforeach()
    add_custom_target(textured_shaders ALL
        DEPENDS ${CMAKE_BINARY_DIR}/texture.vert.spv ${CMAKE_BINARY_DIR}/texture.frag.spv)

    add_executable(textured 04-textured/main.cpp)
    wrapper_sample(textured)
    add_dependencies(textured textured_shaders)
    target_compile_definitions(textured PRIVATE
        SHADER_VERT="${CMAKE_BINARY_DIR}/texture.vert.spv"
        SHADER_FRAG="${CMAKE_BINARY_DIR}/texture.frag.spv")
else()
    message(STATUS "04-textured skipped: requires glslc")
endif()

# ---------------------------------------------------------------------------
# 05-depth: offscreen depth-tested scene; needs glslc.
# ---------------------------------------------------------------------------
if (GLSLC)
    foreach(shader vert frag)
        add_custom_command(
            OUTPUT ${CMAKE_BINARY_DIR}/depth.${shader}.spv
            COMMAND ${GLSLC} ${CMAKE_CURRENT_LIST_DIR}/05-depth/shaders/depth.${shader}
                    -o ${CMAKE_BINARY_DIR}/depth.${shader}.spv
            DEPENDS ${CMAKE_CURRENT_LIST_DIR}/05-depth/shaders/depth.${shader}
            COMMENT "Compiling depth.${shader} to SPIR-V")
    endforeach()
    add_custom_target(depth_shaders ALL
        DEPENDS ${CMAKE_BINARY_DIR}/depth.vert.spv ${CMAKE_BINARY_DIR}/depth.frag.spv)

    add_executable(depth 05-depth/main.cpp)
    wrapper_sample(depth)
    add_dependencies(depth depth_shaders)
    target_compile_definitions(depth PRIVATE
        SHADER_VERT="${CMAKE_BINARY_DIR}/depth.vert.spv"
        SHADER_FRAG="${CMAKE_BINARY_DIR}/depth.frag.spv")
else()
    message(STATUS "05-depth skipped: requires glslc")
endif()

# ---------------------------------------------------------------------------
# 06-pnext: pNext ExtensionChain round-trip; no shaders, always builds.
# ---------------------------------------------------------------------------
add_executable(pnext 06-pnext/main.cpp)
wrapper_sample(pnext)

# ---------------------------------------------------------------------------
# 07-vma: VMA allocator integration; no shaders, always builds.
# ---------------------------------------------------------------------------
add_executable(vma 07-vma/main.cpp)
wrapper_sample(vma)
target_include_directories(vma PRIVATE ${vma_SOURCE_DIR}/include)
add_dependencies(vma generate_wrapper_vma)
