#pragma once

#include <vulkan/vulkan.hpp>

namespace pbbastian {
namespace vulkan {

struct VoidObjectOwner {
  using Type = void;
};

struct InstanceObjectOwner {
  using Type = vk::Instance;
};

struct DeviceObjectOwner {
  using Type = vk::Device;
};

template <typename ObjectT> struct ObjectOwner {};

template <> struct ObjectOwner<vk::Instance> : VoidObjectOwner {};

template <> struct ObjectOwner<vk::Device> : VoidObjectOwner {};

template <>
struct ObjectOwner<vk::DebugReportCallbackEXT> : InstanceObjectOwner {};

template <> struct ObjectOwner<vk::SurfaceKHR> : InstanceObjectOwner {};

template <> struct ObjectOwner<vk::SwapchainKHR> : DeviceObjectOwner {};

template <> struct ObjectOwner<vk::ImageView> : DeviceObjectOwner {};

template <> struct ObjectOwner<vk::Framebuffer> : DeviceObjectOwner {};

template <> struct ObjectOwner<vk::RenderPass> : DeviceObjectOwner {};

template <> struct ObjectOwner<vk::DescriptorSetLayout> : DeviceObjectOwner {};

template <> struct ObjectOwner<vk::PipelineLayout> : DeviceObjectOwner {};

template <> struct ObjectOwner<vk::Pipeline> : DeviceObjectOwner {};

template <> struct ObjectOwner<vk::CommandPool> : DeviceObjectOwner {};

template <> struct ObjectOwner<vk::Image> : DeviceObjectOwner {};

template <> struct ObjectOwner<vk::DeviceMemory> : DeviceObjectOwner {};

template <> struct ObjectOwner<vk::Buffer> : DeviceObjectOwner {};

template <> struct ObjectOwner<vk::DescriptorPool> : DeviceObjectOwner {};

template <> struct ObjectOwner<vk::Semaphore> : DeviceObjectOwner {};

template <> struct ObjectOwner<vk::ShaderModule> : DeviceObjectOwner {};
}
}
