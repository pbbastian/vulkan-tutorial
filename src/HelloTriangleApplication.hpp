#pragma once

#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <tiny_obj_loader.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <pbbastian/stbi.hpp>
#include <pbbastian/vulkan/UniqueObject.hpp>
#include <pbbastian/vulkan/util.hpp>

#include "QueueFamilyIndices.hpp"
#include "SwapChainSupportDetails.hpp"
#include "UniformBufferObject.hpp"
#include "VDeleter.hpp"
#include "VMappedMemory.hpp"
#include "Vertex.hpp"
#include "settings.hpp"
#include "vkFunctions.hpp"

const int WIDTH = 800;
const int HEIGHT = 600;

const std::vector<const char *> validationLayers = {
    "VK_LAYER_LUNARG_standard_validation"};

const std::vector<const char *> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME};

const auto shaderPath = settings::assetPath + "shaders/";
const auto modelPath = settings::assetPath + "models/";
const auto texturePath = settings::assetPath + "textures/";

const auto modelFilePath = modelPath + "chalet.obj";
const auto textureFilePath = texturePath + "chalet.jpg";

namespace vkp = pbbastian::vulkan;

class HelloTriangleApplication {
public:
  void run() {
    initWindow();
    initVulkan();
    mainLoop();
  }

private:
  GLFWwindow *window;

  vkp::UniqueObject<vk::Instance> instance;
  vkp::UniqueObject<vk::DebugReportCallbackEXT> callback{instance};
  vkp::UniqueObject<vk::SurfaceKHR> surface{instance};

  vk::PhysicalDevice physicalDevice;
  vkp::UniqueObject<vk::Device> device;

  vk::Queue graphicsQueue;
  vk::Queue presentQueue;

  vkp::UniqueObject<vk::SwapchainKHR> swapChain{device};
  std::vector<vk::Image> swapChainImages;
  vk::Format swapChainImageFormat;
  vk::Extent2D swapChainExtent;
  std::vector<vkp::UniqueObject<vk::ImageView>> swapChainImageViews;
  std::vector<vkp::UniqueObject<vk::Framebuffer>> swapChainFramebuffers;

  vkp::UniqueObject<vk::RenderPass> renderPass{device};
  vkp::UniqueObject<vk::DescriptorSetLayout> descriptorSetLayout{device};
  vkp::UniqueObject<vk::PipelineLayout> pipelineLayout{device};
  vkp::UniqueObject<vk::Pipeline> graphicsPipeline{device};

  vkp::UniqueObject<vk::CommandPool> commandPool{device};

  vkp::UniqueObject<vk::Image> depthImage{device};
  vkp::UniqueObject<vk::DeviceMemory> depthImageMemory{device};
  vkp::UniqueObject<vk::ImageView> depthImageView{device};

  vkp::UniqueObject<vk::Image> textureImage{device};
  vkp::UniqueObject<vk::DeviceMemory> textureImageMemory{device};
  vkp::UniqueObject<vk::ImageView> textureImageView{device};
  vkp::UniqueObject<vk::Sampler> textureSampler{device};

  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;
  vkp::UniqueObject<vk::Buffer> vertexBuffer{device};
  vkp::UniqueObject<vk::DeviceMemory> vertexBufferMemory{device};

  vkp::UniqueObject<vk::Buffer> indexBuffer{device};
  vkp::UniqueObject<vk::DeviceMemory> indexBufferMemory{device};

  vkp::UniqueObject<vk::Buffer> uniformStagingBuffer{device};
  vkp::UniqueObject<vk::DeviceMemory> uniformStagingBufferMemory{device};
  vkp::UniqueObject<vk::Buffer> uniformBuffer{device};
  vkp::UniqueObject<vk::DeviceMemory> uniformBufferMemory{device};

  vkp::UniqueObject<vk::DescriptorPool> descriptorPool{device};
  vk::DescriptorSet descriptorSet;

  std::vector<vk::CommandBuffer> commandBuffers;

  vkp::UniqueObject<vk::Semaphore> imageAvailableSemaphore{device};
  vkp::UniqueObject<vk::Semaphore> renderFinishedSemaphore{device};

  void initWindow() {
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);

    glfwSetWindowUserPointer(window, this);
    glfwSetWindowSizeCallback(window, onWindowResized);
  }

  void initVulkan() {
    createInstance();
    setupDebugCall();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapChain();
    createImageViews();
    createRenderPass();
    createDescriptorSetLayout();
    createGraphicsPipeline();
    createCommandPool();
    createDepthResources();
    createFramebuffers();
    createTextureImage();
    createTextureImageView();
    createTextureSampler();
    loadModel();
    createVertexBuffer();
    createIndexBuffer();
    createUniformBuffer();
    createDescriptorPool();
    createDescriptorSet();
    createCommandBuffers();
    createSemaphores();
  }

  void mainLoop() {
    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();

      updateUniformBuffer();
      drawFrame();
    }

    device->waitIdle();
  }

  static void onWindowResized(GLFWwindow *window, int width, int height) {
    if (width == 0 || height == 0) {
      return;
    }

    HelloTriangleApplication *app =
        reinterpret_cast<HelloTriangleApplication *>(
            glfwGetWindowUserPointer(window));
    app->recreateSwapChain();
  }

  void recreateSwapChain() {
    device->waitIdle();

    createSwapChain();
    createImageViews();
    createRenderPass();
    createGraphicsPipeline();
    createDepthResources();
    createFramebuffers();
    createCommandBuffers();
  }

  void createInstance() {
    if (settings::enableValidationLayers && !checkValidationLayerSupport()) {
      throw std::runtime_error(
          "Validation layers requested, but not available!");
    }

    vk::ApplicationInfo appInfo;
    appInfo.pApplicationName = "Hello Triangle";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    vk::InstanceCreateInfo createInfo;
    createInfo.pApplicationInfo = &appInfo;

    auto extensions = getRequiredExtensions();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    if (settings::enableValidationLayers) {
      createInfo.enabledLayerCount =
          static_cast<uint32_t>(validationLayers.size());
      createInfo.ppEnabledLayerNames = validationLayers.data();
    } else {
      createInfo.enabledLayerCount = 0;
    }

    auto result = vk::createInstance(createInfo);
    if (result.result != vk::Result::eSuccess) {
      throw std::runtime_error("Failed to create instance!");
    }
    instance = result.value;
  }

  void setupDebugCall() {
    if (!settings::enableValidationLayers) {
      return;
    }

    vk::DebugReportCallbackCreateInfoEXT createInfo;
    createInfo.flags = vk::DebugReportFlagBitsEXT::eError |
                       vk::DebugReportFlagBitsEXT::eWarning;
    createInfo.pfnCallback =
        reinterpret_cast<PFN_vkDebugReportCallbackEXT>(debugCallback);

    if (vkp::CreateDebugReportCallbackEXT(
            static_cast<VkInstance>(instance),
            reinterpret_cast<VkDebugReportCallbackCreateInfoEXT *>(&createInfo),
            nullptr, reinterpret_cast<VkDebugReportCallbackEXT *>(&callback)) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to set up debug callback!");
    }
  }

  void createSurface() {
    if (glfwCreateWindowSurface(
            static_cast<VkInstance>(instance), window, nullptr,
            reinterpret_cast<VkSurfaceKHR *>(&surface)) != VK_SUCCESS) {
      throw std::runtime_error("failed to create window surface!");
    }
  }

  void pickPhysicalDevice() {
    namespace vkp = pbbastian::vulkan;

    std::vector<vk::PhysicalDevice> devices;
    if (!vkp::is_success(instance->enumeratePhysicalDevices(), devices)) {
      throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }

    for (const auto &device : devices) {
      if (isDeviceSuitable(device)) {
        physicalDevice = device;
        break;
      }
    }

    if (!physicalDevice) {
      throw std::runtime_error("failed to find a suitable GPU!");
    }
  }

  void createLogicalDevice() {
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
    std::set<int> uniqueQueueFamilies = {indices.graphicsFamily,
                                         indices.presentFamily};

    for (int queueFamily : uniqueQueueFamilies) {
      float queuePriority = 1.0f;

      vk::DeviceQueueCreateInfo queueCreateInfo;
      queueCreateInfo.queueFamilyIndex = static_cast<uint32_t>(queueFamily);
      queueCreateInfo.queueCount = 1;
      queueCreateInfo.pQueuePriorities = &queuePriority;

      queueCreateInfos.push_back(queueCreateInfo);
    }

    vk::PhysicalDeviceFeatures deviceFeatures;

    vk::DeviceCreateInfo createInfo;

    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.queueCreateInfoCount =
        static_cast<uint32_t>(queueCreateInfos.size());

    createInfo.pEnabledFeatures = &deviceFeatures;

    createInfo.enabledExtensionCount =
        static_cast<uint32_t>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();

    if (settings::enableValidationLayers) {
      createInfo.enabledLayerCount =
          static_cast<uint32_t>(validationLayers.size());
      createInfo.ppEnabledLayerNames = validationLayers.data();
    } else {
      createInfo.enabledLayerCount = 0;
    }

    if (physicalDevice.createDevice(&createInfo, nullptr, &device) !=
        vk::Result::eSuccess) {
      throw std::runtime_error("failed to create logical device!");
    }

    device->getQueue(static_cast<uint32_t>(indices.graphicsFamily), 0,
                     &graphicsQueue);
    device->getQueue(static_cast<uint32_t>(indices.presentFamily), 0,
                     &presentQueue);
  }

  void createSwapChain() {
    SwapChainSupportDetails swapChainSupport =
        querySwapChainSupport(physicalDevice);

    vk::SurfaceFormatKHR surfaceFormat =
        chooseSwapSurfaceFormat(swapChainSupport.formats);
    vk::PresentModeKHR presentMode =
        chooseSwapPresentMode(swapChainSupport.presentModes);
    vk::Extent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
    if (swapChainSupport.capabilities.maxImageCount > 0 &&
        imageCount > swapChainSupport.capabilities.maxImageCount) {
      imageCount = swapChainSupport.capabilities.maxImageCount;
    }

    vk::SwapchainCreateInfoKHR createInfo;
    createInfo.surface = surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;

    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
    uint32_t queueFamilyIndices[] = {
        static_cast<uint32_t>(indices.graphicsFamily),
        static_cast<uint32_t>(indices.presentFamily)};

    if (indices.graphicsFamily != indices.presentFamily) {
      createInfo.imageSharingMode = vk::SharingMode::eConcurrent;
      createInfo.queueFamilyIndexCount = 2;
      createInfo.pQueueFamilyIndices = queueFamilyIndices;
    } else {
      createInfo.imageSharingMode = vk::SharingMode::eExclusive;
      createInfo.queueFamilyIndexCount = 0;
      createInfo.pQueueFamilyIndices = nullptr;
    }

    createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
    createInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
    createInfo.presentMode = presentMode;
    createInfo.clipped = true;

    vk::SwapchainKHR oldSwapChain = swapChain;
    createInfo.oldSwapchain = oldSwapChain;

    vk::SwapchainKHR newSwapChain;
    if (device->createSwapchainKHR(&createInfo, nullptr, &newSwapChain) !=
        vk::Result::eSuccess) {
      throw std::runtime_error("failed to create swap chain!");
    }

    swapChain = newSwapChain;

    device->getSwapchainImagesKHR(swapChain, &imageCount, nullptr);
    swapChainImages.resize(imageCount);
    device->getSwapchainImagesKHR(swapChain, &imageCount,
                                  swapChainImages.data());

    swapChainImageFormat = surfaceFormat.format;
    swapChainExtent = extent;
  }

  void createImageViews() {
    swapChainImageViews.clear();

    for (uint32_t i = 0; i < swapChainImages.size(); i++) {
      swapChainImageViews.emplace_back(
          vkp::UniqueObject<vk::ImageView>{device});
      createImageView(swapChainImages[i], swapChainImageFormat,
                      vk::ImageAspectFlagBits::eColor, swapChainImageViews[i]);
    }
  }

  void createRenderPass() {
    auto colorAttachment = vk::AttachmentDescription{};
    colorAttachment.format = swapChainImageFormat;
    colorAttachment.samples = vk::SampleCountFlagBits::e1;
    colorAttachment.loadOp = vk::AttachmentLoadOp::eClear;
    colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;
    colorAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
    colorAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
    colorAttachment.initialLayout = vk::ImageLayout::eUndefined;
    colorAttachment.finalLayout = vk::ImageLayout::ePresentSrcKHR;

    auto depthAttachment = vk::AttachmentDescription{};
    depthAttachment.format = findDepthFormat();
    depthAttachment.samples = vk::SampleCountFlagBits::e1;
    depthAttachment.loadOp = vk::AttachmentLoadOp::eClear;
    depthAttachment.storeOp = vk::AttachmentStoreOp::eDontCare;
    depthAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
    depthAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
    depthAttachment.initialLayout =
        vk::ImageLayout::eDepthStencilAttachmentOptimal;
    depthAttachment.finalLayout =
        vk::ImageLayout::eDepthStencilAttachmentOptimal;

    auto colorAttachmentRef = vk::AttachmentReference{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = vk::ImageLayout::eColorAttachmentOptimal;

    auto depthAttachmentRef = vk::AttachmentReference{};
    depthAttachmentRef.attachment = 1;
    depthAttachmentRef.layout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

    auto subPass = vk::SubpassDescription{};
    subPass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
    subPass.colorAttachmentCount = 1;
    subPass.pColorAttachments = &colorAttachmentRef;
    subPass.pDepthStencilAttachment = &depthAttachmentRef;

    auto dependency = vk::SubpassDependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = vk::PipelineStageFlagBits::eBottomOfPipe;
    dependency.srcAccessMask = vk::AccessFlagBits::eMemoryRead;
    dependency.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    dependency.dstAccessMask = vk::AccessFlagBits::eColorAttachmentRead |
                               vk::AccessFlagBits::eColorAttachmentWrite;

    auto attachments = std::array<vk::AttachmentDescription, 2>{
        {colorAttachment, depthAttachment}};
    auto renderPassInfo = vk::RenderPassCreateInfo{};
    renderPassInfo.attachmentCount = attachments.size();
    renderPassInfo.pAttachments = attachments.data();
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subPass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    if (device->createRenderPass(&renderPassInfo, nullptr, &renderPass) !=
        vk::Result::eSuccess) {
      throw std::runtime_error("failed to create render pass!");
    }
  }

  void createDescriptorSetLayout() {
    auto uboLayoutBinding = vk::DescriptorSetLayoutBinding{};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.descriptorType = vk::DescriptorType::eUniformBuffer;
    uboLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eVertex;

    auto samplerLayoutBinding = vk::DescriptorSetLayoutBinding{};
    samplerLayoutBinding.binding = 1;
    samplerLayoutBinding.descriptorCount = 1;
    samplerLayoutBinding.descriptorType =
        vk::DescriptorType::eCombinedImageSampler;
    samplerLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;

    auto bindings = std::array<vk::DescriptorSetLayoutBinding, 2>{
        {uboLayoutBinding, samplerLayoutBinding}};
    auto layoutCreateInfo = vk::DescriptorSetLayoutCreateInfo{};
    layoutCreateInfo.bindingCount = bindings.size();
    layoutCreateInfo.pBindings = bindings.data();

    if (device->createDescriptorSetLayout(&layoutCreateInfo, nullptr,
                                          &descriptorSetLayout) !=
        vk::Result::eSuccess) {
      throw std::runtime_error("failed to create descriptor set layout!");
    }
  }

  void createGraphicsPipeline() {
    auto vertShaderCode = readFile(shaderPath + "vert.spv");
    auto fragShaderCode = readFile(shaderPath + "frag.spv");

    auto vertShaderModule = vkp::UniqueObject<vk::ShaderModule>{device};
    auto fragShaderModule = vkp::UniqueObject<vk::ShaderModule>{device};

    createShaderModule(vertShaderCode, vertShaderModule);
    createShaderModule(fragShaderCode, fragShaderModule);

    auto vertShaderStageInfo = vk::PipelineShaderStageCreateInfo{};
    vertShaderStageInfo.stage = vk::ShaderStageFlagBits::eVertex;
    vertShaderStageInfo.module = vertShaderModule;
    vertShaderStageInfo.pName = "main";

    auto fragShaderStageInfo = vk::PipelineShaderStageCreateInfo{};
    fragShaderStageInfo.stage = vk::ShaderStageFlagBits::eFragment;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";

    auto shaderStages = std::array<vk::PipelineShaderStageCreateInfo, 2>{
        {vertShaderStageInfo, fragShaderStageInfo}};

    auto vertexInputInfo = vk::PipelineVertexInputStateCreateInfo{};

    auto bindingDescription = Vertex::getBindingDescription();
    auto attributeDescriptions = Vertex::getAttributeDescriptions();

    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.vertexAttributeDescriptionCount =
        static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    auto inputAssembly = vk::PipelineInputAssemblyStateCreateInfo{};
    inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;
    inputAssembly.primitiveRestartEnable = false;

    auto viewport = vk::Viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(swapChainExtent.width);
    viewport.height = static_cast<float>(swapChainExtent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    auto scissor = vk::Rect2D{};
    scissor.offset = vk::Offset2D(0, 0);
    scissor.extent = swapChainExtent;

    auto viewportState = vk::PipelineViewportStateCreateInfo{};
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    auto rasterizer = vk::PipelineRasterizationStateCreateInfo{};
    rasterizer.depthClampEnable = false;
    rasterizer.rasterizerDiscardEnable = false;
    rasterizer.polygonMode = vk::PolygonMode::eFill;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = vk::CullModeFlagBits::eBack;
    rasterizer.frontFace = vk::FrontFace::eCounterClockwise;
    rasterizer.depthBiasEnable = false;
    rasterizer.depthBiasConstantFactor = 0.0f;
    rasterizer.depthBiasClamp = 0.0f;
    rasterizer.depthBiasSlopeFactor = 0.0f;

    auto multisampling = vk::PipelineMultisampleStateCreateInfo{};
    multisampling.sampleShadingEnable = false;
    multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;
    multisampling.minSampleShading = 1.0f;
    multisampling.pSampleMask = nullptr;
    multisampling.alphaToCoverageEnable = false;
    multisampling.alphaToOneEnable = false;

    auto depthStencil = vk::PipelineDepthStencilStateCreateInfo{};
    depthStencil.depthTestEnable = true;
    depthStencil.depthWriteEnable = true;
    depthStencil.depthCompareOp = vk::CompareOp::eLess;
    depthStencil.depthBoundsTestEnable = false;
    depthStencil.stencilTestEnable = false;

    auto colorBlendAttachment = vk::PipelineColorBlendAttachmentState{};
    colorBlendAttachment.colorWriteMask =
        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
        vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
    colorBlendAttachment.blendEnable = false;

    auto colorBlending = vk::PipelineColorBlendStateCreateInfo{};
    colorBlending.logicOpEnable = false;
    colorBlending.logicOp = vk::LogicOp::eCopy;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    colorBlending.blendConstants[0] = 0.0f;
    colorBlending.blendConstants[1] = 0.0f;
    colorBlending.blendConstants[2] = 0.0f;
    colorBlending.blendConstants[3] = 0.0f;

    auto setLayouts =
        std::array<vk::DescriptorSetLayout, 1>{{descriptorSetLayout}};
    auto pipelineLayoutInfo = vk::PipelineLayoutCreateInfo{};
    pipelineLayoutInfo.setLayoutCount = setLayouts.size();
    pipelineLayoutInfo.pSetLayouts = setLayouts.data();
    pipelineLayoutInfo.pushConstantRangeCount = 0;
    pipelineLayoutInfo.pPushConstantRanges = nullptr;

    if (device->createPipelineLayout(&pipelineLayoutInfo, nullptr,
                                     &pipelineLayout) != vk::Result::eSuccess) {
      throw std::runtime_error("failed to create pipeline layout!");
    }

    auto pipelineInfo = vk::GraphicsPipelineCreateInfo{};
    pipelineInfo.stageCount = shaderStages.size();
    pipelineInfo.pStages = shaderStages.data();
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = nullptr;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = vk::Pipeline();
    pipelineInfo.basePipelineIndex = -1;

    if (device->createGraphicsPipelines(vk::PipelineCache(), 1, &pipelineInfo,
                                        nullptr, &graphicsPipeline) !=
        vk::Result::eSuccess) {
      throw std::runtime_error("failed to create graphics pipeline!");
    }
  }

  void createFramebuffers() {
    swapChainFramebuffers.clear();

    for (size_t i = 0; i < swapChainImageViews.size(); i++) {
      swapChainFramebuffers.emplace_back(
          vkp::UniqueObject<vk::Framebuffer>{device});

      auto attachments = std::array<vk::ImageView, 2>{
          {swapChainImageViews[i], depthImageView}};

      vk::FramebufferCreateInfo framebufferInfo;
      framebufferInfo.renderPass = renderPass;
      framebufferInfo.attachmentCount = attachments.size();
      framebufferInfo.pAttachments = attachments.data();
      framebufferInfo.width = swapChainExtent.width;
      framebufferInfo.height = swapChainExtent.height;
      framebufferInfo.layers = 1;

      if (device->createFramebuffer(&framebufferInfo, nullptr,
                                    &swapChainFramebuffers[i]) !=
          vk::Result::eSuccess) {
        throw std::runtime_error("failed to create framebuffer!");
      }
    }
  }

  void createTextureImage() {
    namespace stbi = pbbastian::stbi;

    int texWidth = 0, texHeight = 0, texChannels;
    auto pixels = stbi::load(textureFilePath.c_str(), texWidth, texHeight,
                             texChannels, stbi::comp_type::rgb_alpha);

    vk::DeviceSize imageSize = static_cast<vk::DeviceSize>(texWidth) *
                               static_cast<vk::DeviceSize>(texHeight) * 4;

    if (!pixels) {
      throw std::runtime_error("failed to load texture image!");
    }

    vkp::UniqueObject<vk::Image> stagingImage{device};
    vkp::UniqueObject<vk::DeviceMemory> stagingImageMemory{device};
    createImage(static_cast<uint32_t>(texWidth),
                static_cast<uint32_t>(texHeight), vk::Format::eR8G8B8A8Unorm,
                vk::ImageTiling::eLinear, vk::ImageUsageFlagBits::eTransferSrc,
                vk::MemoryPropertyFlagBits::eHostVisible |
                    vk::MemoryPropertyFlagBits::eHostCoherent,
                stagingImage, stagingImageMemory);

    auto data = device->mapMemory(stagingImageMemory, 0, imageSize);
    memcpy(data.value, pixels.get(), static_cast<size_t>(imageSize));
    device->unmapMemory(stagingImageMemory);

    createImage(
        static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight),
        vk::Format::eR8G8B8A8Unorm, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal, textureImage,
        textureImageMemory);

    transitionImageLayout(stagingImage, vk::ImageLayout::ePreinitialized,
                          vk::ImageLayout::eTransferSrcOptimal);
    transitionImageLayout(textureImage, vk::ImageLayout::ePreinitialized,
                          vk::ImageLayout::eTransferDstOptimal);
    copyImage(stagingImage, textureImage, static_cast<uint32_t>(texWidth),
              static_cast<uint32_t>(texHeight));
    transitionImageLayout(textureImage, vk::ImageLayout::eTransferDstOptimal,
                          vk::ImageLayout::eShaderReadOnlyOptimal);
  }

  void createTextureImageView() {
    createImageView(textureImage, vk::Format::eR8G8B8A8Unorm,
                    vk::ImageAspectFlagBits::eColor, textureImageView);
  }

  void createTextureSampler() {
    auto samplerInfo = vk::SamplerCreateInfo{};
    samplerInfo.magFilter = vk::Filter::eLinear;
    samplerInfo.minFilter = vk::Filter::eLinear;
    samplerInfo.addressModeU = vk::SamplerAddressMode::eRepeat;
    samplerInfo.addressModeV = vk::SamplerAddressMode::eRepeat;
    samplerInfo.addressModeW = vk::SamplerAddressMode::eRepeat;
    samplerInfo.anisotropyEnable = true;
    samplerInfo.maxAnisotropy = 16;
    samplerInfo.borderColor = vk::BorderColor::eIntOpaqueBlack;
    samplerInfo.unnormalizedCoordinates = false;
    samplerInfo.compareEnable = false;
    samplerInfo.compareOp = vk::CompareOp::eAlways;
    samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f;

    if (device->createSampler(&samplerInfo, nullptr, &textureSampler) !=
        vk::Result::eSuccess) {
      throw std::runtime_error("failed to create texture sampler!");
    }
  }

  void createImageView(vk::Image image, vk::Format format,
                       vk::ImageAspectFlags aspectFlags,
                       vkp::UniqueObject<vk::ImageView> &imageView) {
    auto viewInfo = vk::ImageViewCreateInfo{};
    viewInfo.image = image;
    viewInfo.viewType = vk::ImageViewType::e2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = aspectFlags;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    if (device->createImageView(&viewInfo, nullptr, &imageView) !=
        vk::Result::eSuccess) {
      throw std::runtime_error("failed to create texture image view!");
    }
  }

  void createImage(uint32_t width, uint32_t height, vk::Format format,
                   vk::ImageTiling tiling, vk::ImageUsageFlags usage,
                   vk::MemoryPropertyFlags properties,
                   vkp::UniqueObject<vk::Image> &image,
                   vkp::UniqueObject<vk::DeviceMemory> &imageMemory) {
    vk::ImageCreateInfo imageInfo = {};
    imageInfo.imageType = vk::ImageType::e2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = vk::ImageLayout::ePreinitialized;
    imageInfo.usage = usage;
    imageInfo.samples = vk::SampleCountFlagBits::e1;
    imageInfo.sharingMode = vk::SharingMode::eExclusive;

    if (device->createImage(&imageInfo, nullptr, &image) !=
        vk::Result::eSuccess) {
      throw std::runtime_error("failed to create image!");
    }

    vk::MemoryRequirements memRequirements =
        device->getImageMemoryRequirements(image);

    vk::MemoryAllocateInfo allocInfo;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex =
        findMemoryType(memRequirements.memoryTypeBits, properties);

    if (device->allocateMemory(&allocInfo, nullptr, &imageMemory) !=
        vk::Result::eSuccess) {
      throw std::runtime_error("failed to allocate image memory!");
    }

    device->bindImageMemory(image, imageMemory, 0);
  }

  void transitionImageLayout(vk::Image image, vk::ImageLayout oldLayout,
                             vk::ImageLayout newLayout) {
    vk::CommandBuffer commandBuffer = beginSingleTimeCommands();

    vk::ImageMemoryBarrier barrier;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

    barrier.image = image;
    if (newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal) {
      barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth;
    } else {
      barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    }

    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    if (oldLayout == vk::ImageLayout::ePreinitialized &&
        newLayout == vk::ImageLayout::eTransferSrcOptimal) {
      barrier.srcAccessMask = vk::AccessFlagBits::eHostWrite;
      barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;
    } else if (oldLayout == vk::ImageLayout::ePreinitialized &&
               newLayout == vk::ImageLayout::eTransferDstOptimal) {
      barrier.srcAccessMask = vk::AccessFlagBits::eHostWrite;
      barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
    } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal &&
               newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
      barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
      barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
    } else if (oldLayout == vk::ImageLayout::eUndefined &&
               newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal) {
      barrier.srcAccessMask = vk::AccessFlags();
      barrier.dstAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentRead |
                              vk::AccessFlagBits::eDepthStencilAttachmentWrite;
    } else {
      throw std::invalid_argument("unsupported layout transition");
    }

    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                                  vk::PipelineStageFlagBits::eTopOfPipe,
                                  vk::DependencyFlags(), {}, {}, {barrier});

    endSingleTimeCommands(commandBuffer);
  }

  void copyImage(vk::Image srcImage, vk::Image dstImage, uint32_t width,
                 uint32_t height) {
    vk::CommandBuffer commandBuffer = beginSingleTimeCommands();

    vk::ImageSubresourceLayers subResource;
    subResource.aspectMask = vk::ImageAspectFlagBits::eColor;
    subResource.baseArrayLayer = 0;
    subResource.mipLevel = 0;
    subResource.layerCount = 1;

    vk::ImageCopy region;
    region.srcSubresource = subResource;
    region.dstSubresource = subResource;
    region.srcOffset = vk::Offset3D(0, 0, 0);
    region.dstOffset = vk::Offset3D(0, 0, 0);
    region.extent.width = width;
    region.extent.height = height;
    region.extent.depth = 1;

    commandBuffer.copyImage(srcImage, vk::ImageLayout::eTransferSrcOptimal,
                            dstImage, vk::ImageLayout::eTransferDstOptimal,
                            {region});

    endSingleTimeCommands(commandBuffer);
  }

  void createCommandPool() {
    QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

    vk::CommandPoolCreateInfo poolInfo;
    poolInfo.queueFamilyIndex =
        static_cast<uint32_t>(queueFamilyIndices.graphicsFamily);
    poolInfo.flags = vk::CommandPoolCreateFlags();

    if (device->createCommandPool(&poolInfo, nullptr, &commandPool) !=
        vk::Result::eSuccess) {
      throw std::runtime_error("failed to create command pool!");
    }
  }

  void createDepthResources() {
    auto depthFormat = findDepthFormat();
    createImage(swapChainExtent.width, swapChainExtent.height, depthFormat,
                vk::ImageTiling::eOptimal,
                vk::ImageUsageFlagBits::eDepthStencilAttachment,
                vk::MemoryPropertyFlagBits::eDeviceLocal, depthImage,
                depthImageMemory);
    createImageView(depthImage, depthFormat, vk::ImageAspectFlagBits::eDepth,
                    depthImageView);
    transitionImageLayout(depthImage, vk::ImageLayout::eUndefined,
                          vk::ImageLayout::eDepthStencilAttachmentOptimal);
  }

  vk::Format findDepthFormat() {
    return findSupportedFormat(
        {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint,
         vk::Format::eD24UnormS8Uint},
        vk::ImageTiling::eOptimal,
        vk::FormatFeatureFlagBits::eDepthStencilAttachment);
  }

  vk::Format findSupportedFormat(const std::vector<vk::Format> &candidates,
                                 vk::ImageTiling tiling,
                                 vk::FormatFeatureFlags features) {
    for (auto format : candidates) {
      auto props = physicalDevice.getFormatProperties(format);

      if (tiling == vk::ImageTiling::eLinear &&
          (props.linearTilingFeatures & features) == features) {
        return format;
      } else if (tiling == vk::ImageTiling::eOptimal &&
                 (props.optimalTilingFeatures & features) == features) {
        return format;
      }
    }

    throw std::runtime_error("failed to find supported format!");
  }

  void loadModel() {
    auto attrib = tinyobj::attrib_t{};
    auto shapes = std::vector<tinyobj::shape_t>{};
    auto materials = std::vector<tinyobj::material_t>{};
    auto err = std::string{};

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &err,
                          modelFilePath.c_str())) {
      throw std::runtime_error(err);
    }

    auto uniqueVertices = std::unordered_map<Vertex, int>{};

    for (const auto &shape : shapes) {
      for (const auto &index : shape.mesh.indices) {
        auto vertex = Vertex{};

        vertex.pos = {
            attrib.vertices[static_cast<size_t>(3 * index.vertex_index + 0)],
            attrib.vertices[static_cast<size_t>(3 * index.vertex_index + 1)],
            attrib.vertices[static_cast<size_t>(3 * index.vertex_index + 2)]};

        vertex.texCoord = {
            attrib.texcoords[static_cast<size_t>(2 * index.texcoord_index + 0)],
            1.0f -
                attrib.texcoords[static_cast<size_t>(2 * index.texcoord_index +
                                                     1)]};

        if (uniqueVertices.count(vertex) == 0) {
          uniqueVertices[vertex] = static_cast<int>(vertices.size());
          vertices.push_back(vertex);
        }

        indices.push_back(static_cast<uint32_t>(uniqueVertices[vertex]));
      }
    }
  }

  void createVertexBuffer() {
    vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

    vkp::UniqueObject<vk::Buffer> stagingBuffer{device};
    vkp::UniqueObject<vk::DeviceMemory> stagingBufferMemory{device};
    createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                 vk::MemoryPropertyFlagBits::eHostVisible |
                     vk::MemoryPropertyFlagBits::eHostCoherent,
                 stagingBuffer, stagingBufferMemory);

    auto data = device->mapMemory(stagingBufferMemory, 0, bufferSize);
    memcpy(data.value, vertices.data(), static_cast<size_t>(bufferSize));
    device->unmapMemory(stagingBufferMemory);

    createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst |
                                 vk::BufferUsageFlagBits::eVertexBuffer,
                 vk::MemoryPropertyFlagBits::eDeviceLocal, vertexBuffer,
                 vertexBufferMemory);

    copyBuffer(stagingBuffer, vertexBuffer, bufferSize);
  }

  void createIndexBuffer() {
    vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();

    vkp::UniqueObject<vk::Buffer> stagingBuffer{device};
    vkp::UniqueObject<vk::DeviceMemory> stagingBufferMemory{device};
    createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                 vk::MemoryPropertyFlagBits::eHostVisible |
                     vk::MemoryPropertyFlagBits::eHostCoherent,
                 stagingBuffer, stagingBufferMemory);

    auto data = device->mapMemory(stagingBufferMemory, 0, bufferSize);
    memcpy(data.value, indices.data(), static_cast<size_t>(bufferSize));
    device->unmapMemory(stagingBufferMemory);

    createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst |
                                 vk::BufferUsageFlagBits::eIndexBuffer,
                 vk::MemoryPropertyFlagBits::eDeviceLocal, indexBuffer,
                 indexBufferMemory);

    copyBuffer(stagingBuffer, indexBuffer, bufferSize);
  }

  void createUniformBuffer() {
    VkDeviceSize bufferSize = sizeof(UniformBufferObject);

    createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                 vk::MemoryPropertyFlagBits::eHostVisible |
                     vk::MemoryPropertyFlagBits::eHostCoherent,
                 uniformStagingBuffer, uniformStagingBufferMemory);

    createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst |
                                 vk::BufferUsageFlagBits::eUniformBuffer,
                 vk::MemoryPropertyFlagBits::eDeviceLocal, uniformBuffer,
                 uniformBufferMemory);
  }

  void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
                    vk::MemoryPropertyFlags properties,
                    vkp::UniqueObject<vk::Buffer> &buffer,
                    vkp::UniqueObject<vk::DeviceMemory> &bufferMemory) {
    vk::BufferCreateInfo bufferInfo;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = vk::SharingMode::eExclusive;

    if (device->createBuffer(&bufferInfo, nullptr, &buffer) !=
        vk::Result::eSuccess) {
      throw std::runtime_error("failed to create buffer!");
    }

    vk::MemoryRequirements memoryRequirements =
        device->getBufferMemoryRequirements(buffer);

    vk::MemoryAllocateInfo allocInfo;
    allocInfo.allocationSize = memoryRequirements.size;
    allocInfo.memoryTypeIndex =
        findMemoryType(memoryRequirements.memoryTypeBits, properties);

    if (device->allocateMemory(&allocInfo, nullptr, &bufferMemory) !=
        vk::Result::eSuccess) {
      throw std::runtime_error("failed to allocate buffer memory!");
    }

    device->bindBufferMemory(buffer, bufferMemory, 0);
  }

  VkCommandBuffer beginSingleTimeCommands() {
    vk::CommandBufferAllocateInfo allocInfo;
    allocInfo.level = vk::CommandBufferLevel::ePrimary;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    vk::CommandBuffer commandBuffer;
    device->allocateCommandBuffers(&allocInfo, &commandBuffer);

    vk::CommandBufferBeginInfo beginInfo;
    beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

    commandBuffer.begin(&beginInfo);

    return commandBuffer;
  }

  void endSingleTimeCommands(vk::CommandBuffer commandBuffer) {
    vkEndCommandBuffer(commandBuffer);

    vk::SubmitInfo submitInfo;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    graphicsQueue.submit(1, &submitInfo, vk::Fence());
    graphicsQueue.waitIdle();

    device->freeCommandBuffers(commandPool, 1, &commandBuffer);
  }

  void copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer,
                  vk::DeviceSize size) {
    vk::CommandBuffer commandBuffer = beginSingleTimeCommands();

    vk::BufferCopy copyRegion = {};
    copyRegion.size = size;
    commandBuffer.copyBuffer(srcBuffer, dstBuffer, 1, &copyRegion);

    endSingleTimeCommands(commandBuffer);
  }

  uint32_t findMemoryType(uint32_t typeFilter,
                          vk::MemoryPropertyFlags properties) {
    auto memProperties = physicalDevice.getMemoryProperties();

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
      if ((typeFilter & (1 << i)) &&
          (memProperties.memoryTypes[i].propertyFlags & properties) ==
              properties) {
        return i;
      }
    }

    throw std::runtime_error("failed to find suitable memory type!");
  }

  void createDescriptorPool() {
    auto poolSizes = std::array<vk::DescriptorPoolSize, 2>{};
    poolSizes[0].type = vk::DescriptorType::eUniformBuffer;
    poolSizes[0].descriptorCount = 1;
    poolSizes[1].type = vk::DescriptorType::eCombinedImageSampler;
    poolSizes[1].descriptorCount = 1;

    auto poolInfo = vk::DescriptorPoolCreateInfo{};
    poolInfo.poolSizeCount = poolSizes.size();
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 1;

    if (device->createDescriptorPool(&poolInfo, nullptr, &descriptorPool) !=
        vk::Result::eSuccess) {
      throw std::runtime_error("failed to create descriptor pool!");
    }
  }

  void createDescriptorSet() {
    vk::DescriptorSetLayout layouts[] = {descriptorSetLayout};
    vk::DescriptorSetAllocateInfo allocateInfo;
    allocateInfo.descriptorPool = descriptorPool;
    allocateInfo.descriptorSetCount = 1;
    allocateInfo.pSetLayouts = layouts;

    if (device->allocateDescriptorSets(&allocateInfo, &descriptorSet) !=
        vk::Result::eSuccess) {
      throw std::runtime_error("failed to allocate descriptor set!");
    }

    vk::DescriptorBufferInfo bufferInfo;
    bufferInfo.buffer = uniformBuffer;
    bufferInfo.offset = 0;
    bufferInfo.range = sizeof(UniformBufferObject);

    auto imageInfo = vk::DescriptorImageInfo{};
    imageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    imageInfo.imageView = textureImageView;
    imageInfo.sampler = textureSampler;

    auto descriptorWrites = std::array<vk::WriteDescriptorSet, 2>{};

    descriptorWrites[0].dstSet = descriptorSet;
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].dstArrayElement = 0;
    descriptorWrites[0].descriptorType = vk::DescriptorType::eUniformBuffer;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].pBufferInfo = &bufferInfo;

    descriptorWrites[1].dstSet = descriptorSet;
    descriptorWrites[1].dstBinding = 1;
    descriptorWrites[1].dstArrayElement = 0;
    descriptorWrites[1].descriptorType =
        vk::DescriptorType::eCombinedImageSampler;
    descriptorWrites[1].descriptorCount = 1;
    descriptorWrites[1].pImageInfo = &imageInfo;

    device->updateDescriptorSets(descriptorWrites.size(),
                                 descriptorWrites.data(), 0, nullptr);
  }

  void createCommandBuffers() {
    if (commandBuffers.size() > 0) {
      device->freeCommandBuffers(commandPool,
                                 static_cast<uint32_t>(commandBuffers.size()),
                                 commandBuffers.data());
    }

    commandBuffers.resize(swapChainFramebuffers.size());

    auto allocInfo = vk::CommandBufferAllocateInfo{};
    allocInfo.commandPool = commandPool;
    allocInfo.level = vk::CommandBufferLevel::ePrimary;
    allocInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());

    if (device->allocateCommandBuffers(&allocInfo, commandBuffers.data()) !=
        vk::Result::eSuccess) {
      throw std::runtime_error("failed to allocate command buffers!");
    }

    for (size_t i = 0; i < commandBuffers.size(); i++) {
      auto beginInfo = vk::CommandBufferBeginInfo{};
      beginInfo.flags = vk::CommandBufferUsageFlagBits::eSimultaneousUse;
      beginInfo.pInheritanceInfo = nullptr;

      commandBuffers[i].begin(&beginInfo);

      auto renderPassInfo = vk::RenderPassBeginInfo{};
      renderPassInfo.renderPass = renderPass;
      renderPassInfo.framebuffer = swapChainFramebuffers[i];
      renderPassInfo.renderArea.offset = vk::Offset2D(0, 0);
      renderPassInfo.renderArea.extent = swapChainExtent;

      auto clearValues = std::array<vk::ClearValue, 2>{
          {vk::ClearColorValue{std::array<float, 4>{{0.0f, 0.0f, 0.0f, 1.0f}}},
           vk::ClearDepthStencilValue{1.0f, 0}}};

      renderPassInfo.clearValueCount = clearValues.size();
      renderPassInfo.pClearValues = clearValues.data();

      commandBuffers[i].beginRenderPass(&renderPassInfo,
                                        vk::SubpassContents::eInline);
      commandBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics,
                                     graphicsPipeline);

      vk::Buffer vertexBuffers[] = {vertexBuffer};
      VkDeviceSize offsets[] = {0};
      commandBuffers[i].bindVertexBuffers(0, 1, vertexBuffers, offsets);

      commandBuffers[i].bindIndexBuffer(indexBuffer, 0, vk::IndexType::eUint32);

      commandBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                           pipelineLayout, 0, 1, &descriptorSet,
                                           0, nullptr);

      commandBuffers[i].drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0,
                                    0, 0);

      commandBuffers[i].endRenderPass();

      if (commandBuffers[i].end() != vk::Result::eSuccess) {
        throw std::runtime_error("failed to record comman buffer!");
      }
    }
  }

  void createSemaphores() {
    vk::SemaphoreCreateInfo semaphoreInfo;

    if (device->createSemaphore(&semaphoreInfo, nullptr,
                                &imageAvailableSemaphore) !=
            vk::Result::eSuccess ||
        device->createSemaphore(&semaphoreInfo, nullptr,
                                &renderFinishedSemaphore) !=
            vk::Result::eSuccess) {
      throw std::runtime_error("failed to create semaphores!");
    }
  }

  void updateUniformBuffer() {
    namespace chrono = std::chrono;

    static auto startTime = chrono::high_resolution_clock::now();

    auto currentTime = chrono::high_resolution_clock::now();
    float time =
        chrono::duration_cast<chrono::milliseconds>(currentTime - startTime)
            .count() /
        1000.0f;

    UniformBufferObject ubo = {};
    ubo.model = glm::rotate(glm::mat4(), time * glm::radians(45.0f),
                            glm::vec3(0.0f, 0.0f, 1.0f));
    ubo.view =
        glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f),
                    glm::vec3(0.0f, 0.0f, 1.0f));
    ubo.proj = glm::perspective(glm::radians(45.0f),
                                swapChainExtent.width /
                                    static_cast<float>(swapChainExtent.height),
                                0.1f, 10.0f);
    ubo.proj[1][1] *= -1;

    auto data = device->mapMemory(uniformStagingBufferMemory, 0, sizeof(ubo));
    memcpy(data.value, &ubo, sizeof(ubo));
    device->unmapMemory(uniformStagingBufferMemory);

    copyBuffer(uniformStagingBuffer, uniformBuffer, sizeof(ubo));
  }

  void drawFrame() {
    uint32_t imageIndex;
    vk::Result result = device->acquireNextImageKHR(
        swapChain, std::numeric_limits<uint64_t>::max(),
        imageAvailableSemaphore, vk::Fence(), &imageIndex);

    if (result == vk::Result::eErrorOutOfDateKHR) {
      recreateSwapChain();
      return;
    } else if (result != vk::Result::eSuccess &&
               result != vk::Result::eSuboptimalKHR) {
      throw std::runtime_error("failed to acquire swap chain image!");
    }

    vk::SubmitInfo submitInfo;

    vk::Semaphore waitSemaphores[] = {imageAvailableSemaphore};
    vk::PipelineStageFlags waitStages[] = {
        vk::PipelineStageFlagBits::eColorAttachmentOutput};
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;

    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffers[imageIndex];

    vk::Semaphore signalSemaphores[] = {renderFinishedSemaphore};
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    if (graphicsQueue.submit(1, &submitInfo, vk::Fence()) !=
        vk::Result::eSuccess) {
      throw std::runtime_error("failed to submit draw command buffer!");
    }

    vk::PresentInfoKHR presentInfo;

    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;

    vk::SwapchainKHR swapChains[] = {swapChain};
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;
    presentInfo.pResults = nullptr;

    result = presentQueue.presentKHR(&presentInfo);

    if (result == vk::Result::eErrorOutOfDateKHR ||
        result == vk::Result::eSuboptimalKHR) {
      recreateSwapChain();
    } else if (result != vk::Result::eSuccess) {
      throw std::runtime_error("failed to present swap chain image!");
    }
  }

  void createShaderModule(const std::vector<char> &code,
                          vkp::UniqueObject<vk::ShaderModule> &shaderModule) {
    auto createInfo = vk::ShaderModuleCreateInfo{};
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<uint32_t const *>(code.data());

    if (device->createShaderModule(&createInfo, nullptr, &shaderModule) !=
        vk::Result::eSuccess) {
      throw std::runtime_error("failed to create shader module!");
    }
  }

  vk::SurfaceFormatKHR chooseSwapSurfaceFormat(
      const std::vector<vk::SurfaceFormatKHR> &availableFormats) {
    if (availableFormats.size() == 1 &&
        availableFormats[0].format == vk::Format::eUndefined) {
      return {vk::Format::eB8G8R8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear};
    }

    for (const auto &availableFormat : availableFormats) {
      if (availableFormat.format == vk::Format::eB8G8R8Unorm &&
          availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
        return availableFormat;
      }
    }

    return availableFormats[0];
  }

  vk::PresentModeKHR chooseSwapPresentMode(
      const std::vector<vk::PresentModeKHR> availablePresentModes) {
    for (const auto &availablePresentMode : availablePresentModes) {
      if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
        return availablePresentMode;
      }
    }

    return vk::PresentModeKHR::eFifo;
  }

  template <typename T> T clamp(T value, T min, T max) {
    return std::max(min, std::min(max, value));
  }

  vk::Extent2D
  chooseSwapExtent(const vk::SurfaceCapabilitiesKHR &capabilities) {
    if (capabilities.currentExtent.width !=
        std::numeric_limits<uint32_t>::max()) {
      return capabilities.currentExtent;
    }

    vk::Extent2D actualExtent = {WIDTH, HEIGHT};

    actualExtent.width =
        clamp(actualExtent.width, capabilities.minImageExtent.width,
              capabilities.maxImageExtent.width);
    actualExtent.height =
        clamp(actualExtent.height, capabilities.minImageExtent.height,
              capabilities.maxImageExtent.height);

    return actualExtent;
  }

  SwapChainSupportDetails
  querySwapChainSupport(vk::PhysicalDevice physicalDevice) {
    SwapChainSupportDetails details;

    physicalDevice.getSurfaceCapabilitiesKHR(surface, &details.capabilities);

    uint32_t formatCount;
    physicalDevice.getSurfaceFormatsKHR(surface, &formatCount, nullptr);

    if (formatCount != 0) {
      details.formats.resize(formatCount);
      physicalDevice.getSurfaceFormatsKHR(surface, &formatCount,
                                          details.formats.data());
    }

    uint32_t presentModeCount;
    physicalDevice.getSurfacePresentModesKHR(surface, &presentModeCount,
                                             nullptr);

    if (presentModeCount != 0) {
      details.presentModes.resize(presentModeCount);
      physicalDevice.getSurfacePresentModesKHR(surface, &presentModeCount,
                                               details.presentModes.data());
    }

    return details;
  }

  bool isDeviceSuitable(vk::PhysicalDevice physicalDevice) {
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

    bool extensionsSupported = checkDeviceExtensionSupport(physicalDevice);

    bool swapChainAdequate = false;
    if (extensionsSupported) {
      SwapChainSupportDetails swapChainSupport =
          querySwapChainSupport(physicalDevice);
      swapChainAdequate = !swapChainSupport.formats.empty() &&
                          !swapChainSupport.presentModes.empty();
    }

    return indices.isComplete() && extensionsSupported && swapChainAdequate;
  }

  bool checkDeviceExtensionSupport(vk::PhysicalDevice physicalDevice) {
    uint32_t extensionCount;
    physicalDevice.enumerateDeviceExtensionProperties(nullptr, &extensionCount,
                                                      nullptr);

    std::vector<vk::ExtensionProperties> availableExtensions(extensionCount);
    physicalDevice.enumerateDeviceExtensionProperties(
        nullptr, &extensionCount, availableExtensions.data());

    std::set<std::string> requiredExtensions(deviceExtensions.begin(),
                                             deviceExtensions.end());

    for (const auto &extension : availableExtensions) {
      requiredExtensions.erase(extension.extensionName);
    }

    return requiredExtensions.empty();
  }

  QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice physicalDevice) {
    QueueFamilyIndices indices;

    uint32_t queueFamilyCount = 0;
    physicalDevice.getQueueFamilyProperties(&queueFamilyCount, nullptr);

    std::vector<vk::QueueFamilyProperties> queueFamilies(queueFamilyCount);
    physicalDevice.getQueueFamilyProperties(&queueFamilyCount,
                                            queueFamilies.data());

    int i = 0;
    for (const auto &queueFamily : queueFamilies) {
      VkBool32 presentSupport = false;
      physicalDevice.getSurfaceSupportKHR(static_cast<uint32_t>(i), surface,
                                          &presentSupport);

      if (queueFamily.queueCount > 0) {
        if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
          indices.graphicsFamily = i;
        }
        if (presentSupport) {
          indices.presentFamily = i;
        }
      }

      if (indices.isComplete()) {
        break;
      }

      i++;
    }

    return indices;
  }

  std::vector<const char *> getRequiredExtensions() {
    std::vector<const char *> extensions;

    unsigned int glfwExtensionCount = 0;
    const auto glfwExtensions =
        glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    for (size_t i = 0; i < glfwExtensionCount; i++) {
      extensions.push_back(glfwExtensions[i]);
    }

    if (settings::enableValidationLayers) {
      extensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
    }

    return extensions;
  }

  bool checkValidationLayerSupport() {
    uint32_t layerCount;
    vk::enumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<vk::LayerProperties> availableLayers(layerCount);
    vk::enumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const char *layerName : validationLayers) {
      bool layerFound = false;

      for (const auto &layerProperties : availableLayers) {
        if (strcmp(layerName, layerProperties.layerName) == 0) {
          layerFound = true;
          break;
        }
      }

      if (!layerFound) {
        return false;
      }
    }

    return true;
  }

  static std::vector<char> readFile(const std::string &filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
      throw std::runtime_error("failed to open file at " + filename);
    }

    auto fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), static_cast<std::streamsize>(fileSize));
    file.close();

    return buffer;
  }

  static VkBool32 debugCallback(VkDebugReportFlagsEXT flags,
                                VkDebugReportObjectTypeEXT objType,
                                uint64_t obj, size_t location, int32_t code,
                                const char *layerPrefix, const char *msg,
                                void *userData) {
    std::cerr << "Validation layer: " << msg << std::endl;

    return VK_FALSE;
  }
};
