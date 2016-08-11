# Vulkan Tutorial
This repository contains the result of me following excellent [Vulkan Tutorial](https://vulkan-tutorial.com/).

While the tutorial uses "native" Vulkan, I've decided to use the [Vulkan-Hpp](https://github.com/KhronosGroup/Vulkan-Hpp) wrapper.
I've made some changes to the Vulkan-Hpp generator, in order to make it emit which native type is being wrapped and a constexpr string literal containing the name of the object type.