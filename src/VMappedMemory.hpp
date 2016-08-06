#pragma once

#include "VDeleter.hpp"

class VMappedMemory {
public:
  VMappedMemory(VDeleter<VkDevice> &device, VDeleter<VkDeviceMemory> &memory,
                VkDeviceSize offset, VkDeviceSize size, VkMemoryMapFlags flags)
      : device(device), memory(memory) {
    vkMapMemory(device, memory, offset, size, flags, &data);
  }

  void *operator&() const { return data; }

  ~VMappedMemory() { vkUnmapMemory(device, memory); }

private:
  void *data;
  VDeleter<VkDevice> &device;
  VDeleter<VkDeviceMemory> &memory;
};
