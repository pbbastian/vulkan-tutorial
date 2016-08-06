cmake_minimum_required(VERSION 3.5)

project(vulkan-tutorial)

option(ASSET_PATH
        "Relative or absolute path to asset location, trailing slash."
        "${CMAKE_SOURCE_DIR}/assets/")

LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake/modules/")

find_package(Vulkan REQUIRED)
find_package(GLFW REQUIRED)
find_package(GLM REQUIRED)

add_executable(vulkan-tutorial src/main.cpp)
target_sources(vulkan-tutorial PUBLIC
        src/HelloTriangleApplication.hpp
        src/QueueFamilyIndices.hpp
        src/SwapChainSupportDetails.hpp
        src/UniformBufferObject.hpp
        src/VDeleter.hpp
        src/Vertex.hpp
        src/vkFunctions.hpp
        src/VMappedMemory.hpp)
target_include_directories(vulkan-tutorial PUBLIC "${CMAKE_SOURCE_DIR}/src/" ${VULKAN_INCLUDE_DIR} ${GLFW_INCLUDE_DIR} ${GLM_INCLUDE_DIR})
target_link_libraries(vulkan-tutorial PUBLIC ${VULKAN_LIBRARY} ${GLFW_LIBRARIES})
target_compile_definitions(vulkan-tutorial
        PUBLIC
        GLFW_INCLUDE_VULKAN
        GLM_FORCE_RADIANS
        GLM_FORCE_DEPTH_ZERO_TO_ONE
        ASSET_PATH="${CMAKE_SOURCE_DIR}/assets/")
