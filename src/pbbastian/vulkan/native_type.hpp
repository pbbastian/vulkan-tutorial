#pragma once

#include <vulkan/vulkan.hpp>

namespace pbbastian {
namespace vulkan {

template <typename T> struct native_type {};

template <> struct native_type<vk::Instance> { using value = VkInstance; };

template <> struct native_type<vk::Device> { using value = VkDevice; };

template <> struct native_type<vk::DebugReportCallbackEXT> {
  using value = VkDebugReportCallbackEXT;
};

template <> struct native_type<vk::SurfaceKHR> { using value = VkSurfaceKHR; };
}
}
