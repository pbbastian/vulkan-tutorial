#pragma once

#include "ObjectOwner.hpp"
#include "destroy.hpp"
#include <vulkan/vulkan.hpp>

namespace pbbastian {
namespace vulkan {

template <typename ObjectT, typename OwnerT> class BaseUniqueObject {
public:
  BaseUniqueObject(
      const BaseUniqueObject<OwnerT, typename ObjectOwner<OwnerT>::Type> &owner,
      ObjectT object = ObjectT())
      : owner(owner), object(object) {}

  // Destructor
  ~BaseUniqueObject() { cleanup(); }

  // Copy constructor
  BaseUniqueObject(const BaseUniqueObject &) = delete;

  // Move constructor
  BaseUniqueObject(BaseUniqueObject &&other)
      : object(other.object), owner(other.owner) {
    other.object = ObjectT();
  }

  // Copy assignment
  BaseUniqueObject &operator=(const BaseUniqueObject &) = delete;

  // Move assignment
  BaseUniqueObject &operator=(BaseUniqueObject &&other) {
    cleanup();
    object = other.object;
    other.object = ObjectT();
    owner = other.owner;
    return *this;
  }

  BaseUniqueObject &operator=(ObjectT other) {
    cleanup();
    object = other;
    return *this;
  }

  BaseUniqueObject &operator=(typename ObjectT::NativeType other) {
    cleanup();
    object = static_cast<ObjectT>(other);
    return *this;
  }

  ObjectT *operator&() {
    cleanup();
    return &object;
  }

  operator const ObjectT &() const { return object; }

  explicit operator typename ObjectT::NativeType() const {
    return static_cast<typename ObjectT::NativeType>(object);
  }

  ObjectT *operator->() { return &object; }

private:
  const BaseUniqueObject<OwnerT, typename ObjectOwner<OwnerT>::Type> &owner;
  ObjectT object;

  void cleanup() {
    if (object) {
      destroy(owner, object, nullptr);
      object = ObjectT();
    }
  }
};

template <typename ObjectT> class BaseUniqueObject<ObjectT, void> {
public:
  BaseUniqueObject(ObjectT object = ObjectT()) : object(object) {}

  // Destructor
  ~BaseUniqueObject() { cleanup(); }

  // Copy constructor
  BaseUniqueObject(const BaseUniqueObject &) = delete;

  // Move constructor
  BaseUniqueObject(BaseUniqueObject &&other) : object(other.object) {
    other.object = ObjectT();
  }

  // Copy assignment
  BaseUniqueObject &operator=(const BaseUniqueObject &) = delete;

  // Move assignment
  BaseUniqueObject &operator=(BaseUniqueObject &&other) {
    cleanup();
    object = other.object;
    other.object = ObjectT();
    return *this;
  }

  BaseUniqueObject &operator=(ObjectT other) {
    cleanup();
    object = other;
    return *this;
  }

  BaseUniqueObject &operator=(typename ObjectT::NativeType other) {
    cleanup();
    object = static_cast<ObjectT>(other);
    return *this;
  }

  ObjectT *operator&() {
    cleanup();
    return &object;
  }

  operator const ObjectT &() const { return object; }

  explicit operator typename ObjectT::NativeType() const {
    return static_cast<typename ObjectT::NativeType>(object);
  }

  ObjectT *operator->() { return &object; }

private:
  ObjectT object;

  void cleanup() {
    if (object) {
      destroy(object, nullptr);
      object = ObjectT();
    }
  }
};

template <typename ObjectT>
using UniqueObject =
    BaseUniqueObject<ObjectT, typename ObjectOwner<ObjectT>::Type>;
}
}
