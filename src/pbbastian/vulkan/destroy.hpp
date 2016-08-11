#pragma once

#include "functions.hpp"
#include <functional>
#include <utility>
#include <vulkan/vulkan.hpp>

namespace pbbastian {
namespace vulkan {

inline void destroy(vk::Instance instance,
                    vk::AllocationCallbacks *pAllocator) {
  instance.destroy(pAllocator);
}

inline void destroy(vk::Device device, vk::AllocationCallbacks *pAllocator) {
  device.destroy(pAllocator);
}

inline void destroy(const vk::Instance instance,
                    vk::DebugReportCallbackEXT callback,
                    const vk::AllocationCallbacks *pAllocator) {
  DestroyDebugReportCallbackEXT(
      instance, callback,
      reinterpret_cast<const VkAllocationCallbacks *>(pAllocator));
}

inline void destroy(const vk::Instance instance, vk::SurfaceKHR surface,
                    const vk::AllocationCallbacks *pAllocator) {
  instance.destroySurfaceKHR(surface, pAllocator);
}

inline void destroy(const vk::Device device, vk::SwapchainKHR swapchain,
                    const vk::AllocationCallbacks *pAllocator) {
  device.destroySwapchainKHR(swapchain, pAllocator);
}

inline void destroy(const vk::Device device, vk::RenderPass renderPass,
                    const vk::AllocationCallbacks *pAllocator) {
  device.destroyRenderPass(renderPass, pAllocator);
}

inline void destroy(const vk::Device device, vk::PipelineLayout pipelineLayout,
                    const vk::AllocationCallbacks *pAllocator) {
  device.destroyPipelineLayout(pipelineLayout, pAllocator);
}

inline void destroy(const vk::Device device, vk::Pipeline pipeline,
                    const vk::AllocationCallbacks *pAllocator) {
  device.destroyPipeline(pipeline, pAllocator);
}

inline void destroy(const vk::Device device, vk::CommandPool commandPool,
                    const vk::AllocationCallbacks *pAllocator) {
  device.destroyCommandPool(commandPool, pAllocator);
}

inline void destroy(const vk::Device device, vk::Image image,
                    const vk::AllocationCallbacks *pAllocator) {
  device.destroyImage(image, pAllocator);
}

inline void destroy(const vk::Device device, vk::DeviceMemory deviceMemory,
                    const vk::AllocationCallbacks *pAllocator) {
  device.freeMemory(deviceMemory, pAllocator);
}

inline void destroy(const vk::Device device, vk::ImageView imageView,
                    const vk::AllocationCallbacks *pAllocator) {
  device.destroyImageView(imageView, pAllocator);
}

inline void destroy(const vk::Device device, vk::Buffer buffer,
                    const vk::AllocationCallbacks *pAllocator) {
  device.destroyBuffer(buffer, pAllocator);
}

inline void destroy(const vk::Device device, vk::DescriptorPool descriptorPool,
                    const vk::AllocationCallbacks *pAllocator) {
  device.destroyDescriptorPool(descriptorPool, pAllocator);
}

inline void destroy(const vk::Device device,
                    vk::DescriptorSetLayout descriptorSetLayout,
                    const vk::AllocationCallbacks *pAllocator) {
  device.destroyDescriptorSetLayout(descriptorSetLayout, pAllocator);
}

inline void destroy(const vk::Device device, vk::Semaphore semaphore,
                    const vk::AllocationCallbacks *pAllocator) {
  device.destroySemaphore(semaphore, pAllocator);
}

inline void destroy(const vk::Device &device, vk::ShaderModule &shaderModule,
                    const vk::AllocationCallbacks *pAllocator) {
  device.destroyShaderModule(shaderModule, pAllocator);
}

inline void destroy(const vk::Device device, vk::Framebuffer framebuffer,
                    const vk::AllocationCallbacks *pAllocator) {
  device.destroyFramebuffer(framebuffer, pAllocator);
}
}
}
