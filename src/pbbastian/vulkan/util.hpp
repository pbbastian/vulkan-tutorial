#pragma once

#include <pbbastian/vulkan/UniqueObject.hpp>
#include <vulkan/vulkan.hpp>

namespace pbbastian {
namespace vulkan {
template <typename T>
bool is_success(const vk::ResultValue<T> &resultValue, T &value) {
  if (resultValue.result != vk::Result::eSuccess) {
    return false;
  }

  value = resultValue.value;
  return true;
}

template <typename T>
bool is_success(const vk::ResultValue<T> &resultValue, UniqueObject<T> &value) {
  return is_success(resultValue, static_cast<T>(value));
}
}
}
