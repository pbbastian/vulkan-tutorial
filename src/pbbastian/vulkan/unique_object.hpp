#pragma once

#include "deleter_function.hpp"
#include <vulkan/vulkan.hpp>

namespace pbbastian {
namespace vulkan {
template <typename ObjectT> class UniqueObject {
public:
  UniqueObject() : deleter([=](ObjectT obj) { destroy(obj, nullptr); }) {}

  UniqueObject(const vk::Instance &instance)
      : deleter([&instance](ObjectT obj) { destroy(instance, obj, nullptr); }) {
  }

  UniqueObject(const vk::Device &device)
      : deleter([&device](ObjectT obj) { destroy(device, obj, nullptr); }) {}

  // Destructor
  ~UniqueObject() { cleanup(); }

  // Copy constructor
  UniqueObject(const UniqueObject<ObjectT> &) = delete;

  // Move constructor
  UniqueObject(UniqueObject<ObjectT> &&other)
      : object(other.object), deleter(std::move(other.deleter)) {
    other.object = ObjectT();
  }

  // Copy assignment
  UniqueObject &operator=(const UniqueObject<ObjectT> &) = delete;

  // Move assignment
  UniqueObject &operator=(UniqueObject &&other) {
    cleanup();
    object = other.object;
    other.object = ObjectT();
    deleter = std::move(other.deleter);
    return *this;
  }

  UniqueObject<ObjectT> &operator=(ObjectT other) {
    cleanup();
    object = other;
    return *this;
  }

  UniqueObject<ObjectT> &operator=(typename ObjectT::NativeType other) {
    cleanup();
    object = static_cast<ObjectT>(other);
    return *this;
  }

  ObjectT *operator&() {
    cleanup();
    return &object;
  }

  operator ObjectT() const { return object; }

  explicit operator typename ObjectT::NativeType() const {
    return static_cast<typename ObjectT::NativeType>(object);
  }

  ObjectT *operator->() { return &object; }

private:
  ObjectT object;
  std::function<void(ObjectT)> deleter;

  void cleanup() {
    if (object) {
      deleter(object);
      object = ObjectT();
    }
  }
};
}
}
