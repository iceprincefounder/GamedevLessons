// Copyright LearnVulkan-06: Draw with PBR, @xukai. All Rights Reserved.
#define GLFW_INCLUDE_VULKAN
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE // 深度缓存区，OpenGL默认是（-1.0， 1.0）Vulakn为（0.0， 1.0）
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <iostream>
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
#include <filesystem>
#include <fstream>
#include <regex>
#include <limits>
#include <optional>
#include <vector>
#include <set>
#include <array>
#include <chrono>
#include <unordered_map>


const uint32_t VIEWPORT_WIDTH = 1080;
const uint32_t VIEWPORT_HEIGHT = 720;
#define SHADOWMAP_DIM 1024


/** 同时渲染多帧的最大帧数*/
const int MAX_FRAMES_IN_FLIGHT = 2;

struct FVertex {
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec3 color;
	glm::vec2 texCoord;

	static VkVertexInputBindingDescription getBindingDescription() {
		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(FVertex);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		return bindingDescription;
	}

	static std::array<VkVertexInputAttributeDescription, 4> getAttributeDescriptions() {
		std::array<VkVertexInputAttributeDescription, 4> attributeDescriptions{};

		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(FVertex, position);

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(FVertex, normal);

		attributeDescriptions[2].binding = 0;
		attributeDescriptions[2].location = 2;
		attributeDescriptions[2].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[2].offset = offsetof(FVertex, color);

		attributeDescriptions[3].binding = 0;
		attributeDescriptions[3].location = 3;
		attributeDescriptions[3].format = VK_FORMAT_R32G32_SFLOAT;
		attributeDescriptions[3].offset = offsetof(FVertex, texCoord);

		return attributeDescriptions;
	}

	bool operator==(const FVertex& other) const {
		return position == other.position && normal == other.normal && color == other.color && texCoord == other.texCoord;
	}
};

namespace std {
	template<> struct hash<FVertex> {
		size_t operator()(FVertex const& vertex) const {
			return ((hash<glm::vec3>()(vertex.position) ^ (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^ (hash<glm::vec2>()(vertex.texCoord) << 1);
		}
	};
}


/** 物体的MVP矩阵信息*/
struct FUniformBufferBase {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
};


struct FLight
{
	glm::vec4 position;
	glm::vec4 color; // rgb for color, a for intensity
	glm::vec4 direction;
	glm::vec4 info;
};


/** 场景灯光信息*/
struct FUniformBufferView {
	glm::mat4 shadowmap_space;
	glm::mat4 local_to_world;
	glm::vec4 camera_info;
	FLight directional_lights[4];
	FLight point_lights[4];
	FLight spot_lights[4];
	// lights_count:
	// [0] for number of directional_lights, [1] for number of point_lights, 
	// [2] for number of spot_lights, [3] for number of cube map max miplevels.
	glm::ivec4 lights_count;
	glm::float32 zNear;
	glm::float32 zFar;
};


const std::vector<const char*> validationLayers = { "VK_LAYER_KHRONOS_validation" }; // VK_LAYER_KHRONOS_validation这个是固定的，不能重命名
const std::vector<const char*> deviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };


#ifdef NDEBUG
const bool enableValidationLayers = false;  // Build Configuration: Release
#else
const bool enableValidationLayers = true;   // Build Configuration: Debug
#endif


static VkResult createDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
	auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
	if (func != nullptr)
	{
		return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
	}
	else
	{
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}


static void destroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator)
{
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
	if (func != nullptr)
	{
		func(instance, debugMessenger, pAllocator);
	}
}


/** 所有的硬件信息*/
struct FQueueFamilyIndices
{
	std::optional<uint32_t> graphicsFamily;
	std::optional<uint32_t> presentFamily;

	bool isComplete()
	{
		return graphicsFamily.has_value() && presentFamily.has_value();
	}
};


/** 支持的硬件细节信息*/
struct FSwapChainSupportDetails
{
	VkSurfaceCapabilitiesKHR capabilities;
	std::vector<VkSurfaceFormatKHR> formats;
	std::vector<VkPresentModeKHR> presentModes;
};


class FVulkanRendererApp
{
	struct FInput {
		glm::vec3 cameraPos;
		glm::vec3 cameraLookat;
		float cameraSpeed;
		float cameraFOV;
		float zNear;
		float zFar;

		float cameraArm;
		float cameraYaw;
		float cameraPitch;

		bool focusCamera;
		bool updateCamera;

		float currentTime;
		float deltaTime;
		bool firstInit;
		double lastMouseX, lastMouseY;
		bool playStageRoll;
		float rollStage;
		bool playLightRoll;
		float rollLight;

		void resetToFocus()
		{
			cameraPos = glm::vec3(2.0, 2.0, 2.0);
			cameraLookat = glm::vec3(0.0, 0.0, 0.0);
			cameraSpeed = 2.5;
			cameraFOV = 45.0;
			zNear = 0.1f;
			zFar = 15.0f;

			glm::mat4 transform = glm::lookAt(cameraPos, cameraLookat, glm::vec3(0.0f, 0.0f, 1.0f));
			glm::quat rotation(transform);
			glm::vec3 direction = glm::normalize(cameraLookat - cameraPos);
			cameraYaw = 30.0; //glm::degrees(glm::atan(direction.x, direction.y));
			cameraPitch = 30.0; //glm::degrees(glm::asin(direction.z));
			cameraArm = 4.0;
			updateCamera = false;
			focusCamera = true;

			cameraPos.x = cos(glm::radians(cameraYaw)) * cos(glm::radians(cameraPitch)) * cameraArm;
			cameraPos.y = sin(glm::radians(cameraYaw)) * cos(glm::radians(cameraPitch)) * cameraArm;
			cameraPos.z = sin(glm::radians(cameraPitch)) * cameraArm;
		}
		void resetToTravel()
		{
			cameraPos = glm::vec3(2.0, 0.0, 0.0);
			cameraLookat = glm::vec3(0.0, 0.0, 0.0);
			cameraSpeed = 2.5;
			cameraFOV = 45.0;
			zNear = 0.1f;
			zFar = 15.0f;

			cameraArm = (float)(cameraLookat - cameraPos).length();
			glm::mat4 transform = glm::lookAt(cameraPos, cameraLookat, glm::vec3(0.0f, 0.0f, 1.0f));
			glm::quat rotation(transform);
			cameraYaw = -90.0f;;
			cameraPitch = 0.0;
			updateCamera = false;
			focusCamera = false;
		}
		void resetAnimation()
		{
			playStageRoll = false;
			rollStage = 0.0;
			playLightRoll = false;
			rollLight = 0.0;
		}
	} gloablInput;

	/** 构建 ShadowmapPass 需要的 Vulkan 资源*/
	struct FShadowmapPass {
		float zNear, zFar;
		int32_t width, height;
		VkFormat format;
		VkFramebuffer frameBuffer;
		VkRenderPass renderPass;
		VkImage image;
		VkDeviceMemory memory;
		VkImageView imageView;
		VkSampler sampler;
		VkDescriptorSetLayout descriptorSetLayout;
		VkDescriptorPool descriptorPool;
		std::vector<VkDescriptorSet> descriptorSets;
		VkPipelineLayout pipelineLayout;
		VkPipeline pipeline;
		std::vector<VkBuffer> uniformBuffers;
		std::vector<VkDeviceMemory> uniformBuffersMemory;
	} shadowmapPass;
	
	/** 构建 BackgroundPass 需要的 Vulkan 资源*/
	struct FBackgroundPass {
		VkImage image;
		VkDeviceMemory memory;
		VkImageView imageView;
		VkSampler sampler;
		VkRenderPass renderPass;
		VkDescriptorSetLayout descriptorSetLayout;
		VkDescriptorPool descriptorPool;
		std::vector<VkDescriptorSet> descriptorSets;
		VkPipelineLayout pipelineLayout;
		std::vector<VkPipeline> pipelines;
	} backgroundPass;

	/** 构建 RenderObject 需要的 Vulkan 资源*/
	struct FRenderObject {
		std::vector<FVertex> vertices;						// 顶点
		std::vector<uint32_t> indices;						// 点序
		VkBuffer vertexBuffer;								// 顶点缓存
		VkDeviceMemory vertexBufferMemory;					// 顶点缓存内存
		VkBuffer indexBuffer;								// 点序缓存
		VkDeviceMemory indexBufferMemory;					// 点序缓存内存

		std::vector<VkImage> textureImages;					// 贴图
		std::vector<VkDeviceMemory> textureImageMemorys;	// 贴图内存
		std::vector<VkImageView> textureImageViews;			// 贴图视口
		std::vector<VkSampler> textureSamplers;				// 贴图采样器

		VkDescriptorPool descriptorPool;					// 描述符池
		std::vector<VkDescriptorSet> descriptorSets;		// 描述符集合
	};

	/** 构建 BaseScenePass 需要的 Vulkan 资源*/
	struct FBaseScenePass {
		std::vector<FRenderObject> renderObjects;	// 待渲染的物体
		VkDescriptorSetLayout descriptorSetLayout;  // 描述符集合布局
		VkPipelineLayout pipelineLayout;            // 渲染管线布局
		std::vector<VkPipeline> pipelines;			// 渲染管线
	} baseScenePass;

	/** 全局常量*/
	struct FGlobalConstants {
		float time;
		float metallic;
		float roughness;
		uint32_t specConstants;
		uint32_t specConstantsCount;				// 特殊常数，用于优化着色器变体

		void resetConstants()
		{
			time = 0.0f;
			metallic = 0.0f;
			roughness = 1.0;
			specConstants = 0;
			specConstantsCount = 9;
		}
	} globalConstants;

	GLFWwindow* window;										// Window 渲染桌面

	VkInstance instance;									// 链接程序的Vulkan实例
	VkDebugUtilsMessengerEXT debugMessenger;
	VkSurfaceKHR surface;									// 链接桌面和Vulkan的实例

	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;		// 物理显卡硬件
	VkDevice device;										// 逻辑硬件，对接物理硬件

	VkQueue graphicsQueue;									// 显卡的队列
	VkQueue presentQueue;									// 显示器的队列

	VkSwapchainKHR swapChain;								// 缓存渲染图像队列，同步到显示器
	std::vector<VkImage> swapChainImages;					// 渲染图像队列
	VkFormat swapChainImageFormat;							// 渲染图像格式
	VkExtent2D swapChainExtent;								// 渲染图像范围
	std::vector<VkImageView> swapChainImageViews;			// 渲染图像队列对应的视图队列
	std::vector<VkFramebuffer> swapChainFramebuffers;		// 渲染图像队列对应的帧缓存队列
	VkRenderPass renderPass;								// 渲染层，保存Framebuffer和采样信息

	VkImage depthImage;										// 深度纹理资源
	VkDeviceMemory depthImageMemory;						// 深度纹理内存
	VkImageView depthImageView;								// 深度纹理图像视口

	uint32_t cubemapMaxMips;								// 环境反射纹理最大Mips数
	VkImage cubemapImage;									// 环境反射纹理资源
	VkDeviceMemory cubemapImageMemory;						// 环境反射纹理内存
	VkImageView cubemapImageView;							// 环境反射纹理图像视口
	VkSampler cubemapSampler;								// 环境反射纹理采样器

	std::vector<VkBuffer> baseUniformBuffers;				// 统一缓存区
	std::vector<VkDeviceMemory> baseUniformBuffersMemory;	// 统一缓存区内存地址

	std::vector<VkBuffer> viewUniformBuffers;				// 统一缓存区
	std::vector<VkDeviceMemory> viewUniformBuffersMemory;	// 统一缓存区内存地址

	VkCommandPool commandPool;								// 指令池
	VkCommandBuffer commandBuffer;							// 指令缓存

	VkSemaphore imageAvailableSemaphore;					// 图像是否完成的信号
	VkSemaphore renderFinishedSemaphore;					// 渲染是否结束的信号
	VkFence inFlightFence;									// 围栏，下一帧渲染前等待上一帧全部渲染完成

	std::vector<VkCommandBuffer> commandBuffers;			// 指令缓存
	std::vector<VkSemaphore> imageAvailableSemaphores;		// 图像是否完成的信号
	std::vector<VkSemaphore> renderFinishedSemaphores;		// 渲染是否结束的信号
	std::vector<VkFence> inFlightFences;					// 围栏，下一帧渲染前等待上一帧全部渲染完成
	uint32_t currentFrame = 0;								// 当前渲染帧序号

	bool framebufferResized = false;
public:
	/** 主函数调用接口*/
	void mainTask()
	{
		initWindow();		// 使用传统的GLFW，初始化窗口
		initVulkan();		// 初始化Vulkan，创建资源
		mainTick();			// 每帧循环调用，执行渲染指令
		clearWindow();		// 渲染窗口关闭时，删除创建的资源
	}

public:
	/** 初始化GUI渲染窗口*/
	void initWindow()
	{
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE); // 打开窗口的Resize功能

		window = glfwCreateWindow(VIEWPORT_WIDTH, VIEWPORT_HEIGHT, "LearnVulkan-07: Draw with Shadow", nullptr /* glfwGetPrimaryMonitor() 全屏模式*/, nullptr);
		glfwSetWindowUserPointer(window, this);
		glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
		GLFWimage iconImages[2];
		iconImages[0].pixels = stbi_load("Resources/AppIcons/vulkan_renderer.png", &iconImages[0].width, &iconImages[0].height, 0, STBI_rgb_alpha);
		iconImages[1].pixels = stbi_load("Resources/AppIcons/vulkan_renderer_small.png", &iconImages[1].width, &iconImages[1].height, 0, STBI_rgb_alpha);
		glfwSetWindowIcon(window, 2, iconImages);
		stbi_image_free(iconImages[0].pixels);
		stbi_image_free(iconImages[1].pixels);
		
		gloablInput.resetToFocus();
		gloablInput.resetAnimation();
		globalConstants.resetConstants();

		glfwSetKeyCallback(window, keyboardCallback);
		glfwSetMouseButtonCallback(window, mouseButtonCallback);
		glfwSetCursorPosCallback(window, mousePositionCallback);
		glfwSetScrollCallback(window, mouseScrollCallback);
	}

	/** 初始化Vulkan的渲染管线*/
	void initVulkan()
	{
		createInstance();			// 连接此程序和Vulkan，一般由显卡驱动实现
		createDebugMessenger();		// 创建调试打印信息
		createWindowsSurface();		// 连接此程序的窗口和Vulkan，渲染Vulkan输出
		selectPhysicalDevice();		// 找到此电脑的物理显卡硬件
		createLogicalDevice();		// 创建逻辑硬件，对应物理硬件
		createSwapChain();			// 创建交换链，用于渲染数据和图像显示的中间交换
		createSwapChainImageViews();// 创建图像显示，包含在SwapChain中
		createRenderPass();			// 创建渲染通道
		createCommandPool();		// 创建指令池，存储所有的渲染指令
		createFramebuffers();		// 创建帧缓存，包含在SwaoChain中
		createUniformBuffers();		// 创建UnifromBuffer统一缓存区
		createShadowmapPass();		// 创建阴影贴图渲染通道
		createBaseScenePass();		// 创建基础物体渲染通道
		createBackgroundPass();		// 创建背景渲染通道
		createCommandBuffer();		// 创建指令缓存，指令发送前变成指令缓存
		createSyncObjects();		// 创建同步围栏，确保下一帧渲染前，上一帧全部渲染完成
	}

	/** 主循环，执行每帧渲染*/
	void mainTick()
	{
		while (!glfwWindowShouldClose(window))
		{
			glfwPollEvents();
			updateInputs();
			drawFrame(); // 绘制一帧
		}

		vkDeviceWaitIdle(device);
	}

	/** 清除Vulkan的渲染管线*/
	void clearWindow()
	{
		cleanupSwapChain(); // 清理FrameBuffer相关的资源
		destroyVulkan(); // 删除Vulkan对象

		glfwDestroyWindow(window);
		glfwTerminate();
	}

	static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
		auto app = reinterpret_cast<FVulkanRendererApp*>(glfwGetWindowUserPointer(window));
		app->framebufferResized = true;
	}

	static void keyboardCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
	{
		FVulkanRendererApp* app = reinterpret_cast<FVulkanRendererApp*>(glfwGetWindowUserPointer(window));
		FInput* input = &app->gloablInput;
		FGlobalConstants* constants = &app->globalConstants;

		if (action == GLFW_PRESS && key == GLFW_KEY_F)
		{
			input->resetToFocus();
		}
		if (action == GLFW_PRESS && key == GLFW_KEY_G)
		{
			input->resetToTravel();
		}
		if (action == GLFW_PRESS && key == GLFW_KEY_R)
		{
			input->resetAnimation();
		}
		if (action == GLFW_PRESS && key == GLFW_KEY_M)
		{
			input->playStageRoll = !input->playStageRoll;
		}
		if (action == GLFW_PRESS && key == GLFW_KEY_L)
		{
			input->playLightRoll = !input->playLightRoll;
		}
		if (action == GLFW_PRESS && key == GLFW_KEY_0)
		{
			constants->specConstants = 0;
		}
		if (action == GLFW_PRESS && key == GLFW_KEY_1)
		{
			constants->specConstants = 1;
		}
		if (action == GLFW_PRESS && key == GLFW_KEY_2)
		{
			constants->specConstants = 2;
		}
		if (action == GLFW_PRESS && key == GLFW_KEY_3)
		{
			constants->specConstants = 3;
		}
		if (action == GLFW_PRESS && key == GLFW_KEY_4)
		{
			constants->specConstants = 4;
		}
		if (action == GLFW_PRESS && key == GLFW_KEY_5)
		{
			constants->specConstants = 5;
		}
		if (action == GLFW_PRESS && key == GLFW_KEY_6)
		{
			constants->specConstants = 6;
		}
		if (action == GLFW_PRESS && key == GLFW_KEY_P)
		{
			constants->specConstants = (constants->specConstants == 7) ? 8 : 7;
		}
	}

	static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
	{
		FVulkanRendererApp* app = reinterpret_cast<FVulkanRendererApp*>(glfwGetWindowUserPointer(window));
		FInput* input = &app->gloablInput;

		if (action == GLFW_PRESS && button == GLFW_MOUSE_BUTTON_RIGHT)
		{
			input->updateCamera = true;
			input->firstInit = true;
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		}
		if (action == GLFW_RELEASE && button == GLFW_MOUSE_BUTTON_RIGHT)
		{
			input->updateCamera = false;
			input->firstInit = true;
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		}
	}

	static void mousePositionCallback(GLFWwindow* window, double xpos, double ypos)
	{
		FVulkanRendererApp* app = reinterpret_cast<FVulkanRendererApp*>(glfwGetWindowUserPointer(window));
		FInput* input = &app->gloablInput;

		if (!input->updateCamera)
		{
			return;
		}

		if (input->firstInit)
		{
			input->lastMouseX = xpos;
			input->lastMouseY = ypos;
			input->firstInit = false;
		}

		float cameraYaw = input->cameraYaw;
		float cameraPitch = input->cameraPitch;

		float xoffset = (float)(xpos - input->lastMouseX);
		float yoffset = (float)(ypos - input->lastMouseY);
		input->lastMouseX = xpos;
		input->lastMouseY = ypos;

		float sensitivityX = 1.0;
		float sensitivityY = 0.5;
		xoffset *= sensitivityX;
		yoffset *= sensitivityY;

		cameraYaw -= xoffset;
		cameraPitch += yoffset;
		if (cameraPitch > 89.0)
			cameraPitch = 89.0;
		if (cameraPitch < -89.0)
			cameraPitch = -89.0;

		if (input->focusCamera)
		{
			glm::vec3 cameraPos = input->cameraPos;
			float cameraArm = input->cameraArm;
			cameraPos.x = cos(glm::radians(cameraYaw)) * cos(glm::radians(cameraPitch)) * cameraArm;
			cameraPos.y = sin(glm::radians(cameraYaw)) * cos(glm::radians(cameraPitch)) * cameraArm;
			cameraPos.z = sin(glm::radians(cameraPitch)) * cameraArm;
			input->cameraPos = cameraPos;
			input->cameraYaw = cameraYaw;
			input->cameraPitch = cameraPitch;
		}
		else
		{
			// @TODO: 鼠标右键控制相机的旋转
		}
	}

	static void mouseScrollCallback(GLFWwindow* window, double xoffset, double yoffset)
	{
		FVulkanRendererApp* app = reinterpret_cast<FVulkanRendererApp*>(glfwGetWindowUserPointer(window));
		FInput* input = &app->gloablInput;

		if (input->focusCamera)
		{
			glm::vec3 cameraPos = input->cameraPos;
			glm::vec3 cameraLookat = input->cameraLookat;
			glm::vec3 lookatToPos = cameraLookat - cameraPos;
			glm::vec3 direction = glm::normalize(lookatToPos);
			float cameraDeltaMove = (float)yoffset * 0.5f;
			float camerArm = input->cameraArm;
			camerArm += cameraDeltaMove;
			camerArm = glm::max(camerArm, 1.0f);
			cameraPos = cameraLookat - camerArm * direction;
			input->cameraPos = cameraPos;
			input->cameraArm = camerArm;
		}
	}

	void updateInputs()
	{
		float deltaTime = (float)gloablInput.deltaTime;    // Time between current frame and last frame
		float lastFrame = (float)gloablInput.currentTime;  // Time of last frame

		float currentFrame = (float)glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		
		bool focusCamera = gloablInput.focusCamera;
		glm::vec3 cameraPos = gloablInput.cameraPos;
		glm::vec3 cameraLookat = gloablInput.cameraLookat;
		glm::vec3 cameraForward = glm::vec3(-1.0, 0.0, 0.0);
		glm::vec3 cameraUp = glm::vec3(0.0, 0.0, 1.0);
		float cameraSpeed = gloablInput.cameraSpeed;
		float cameraYaw = gloablInput.cameraYaw;
		float cameraPitch = gloablInput.cameraPitch;
		glm::vec3 cameraDirection = glm::normalize(cameraLookat - cameraPos);
		const float cameraDeltaMove = 2.5f * deltaTime; // adjust accordingly

		if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		{
			if (focusCamera)
			{
				float cameraArm = gloablInput.cameraArm;
				cameraArm -= cameraDeltaMove;
				cameraArm = glm::max(cameraArm, 1.0f);
				cameraPos = cameraLookat - cameraArm * cameraDirection;
				gloablInput.cameraPos = cameraPos;
				gloablInput.cameraArm = cameraArm;
			}
			else
			{
				gloablInput.cameraPos += cameraDeltaMove * cameraDirection;
			}
			gloablInput.cameraLookat = focusCamera ? cameraLookat : (cameraPos + cameraDirection);
		}
		if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		{
			if (focusCamera)
			{
				float cameraArm = gloablInput.cameraArm;
				cameraArm += cameraDeltaMove;
				cameraArm = glm::max(cameraArm, 1.0f);
				cameraPos = cameraLookat - cameraArm * cameraDirection;
				gloablInput.cameraPos = cameraPos;
				gloablInput.cameraArm = cameraArm;
			}
			else
			{
				gloablInput.cameraPos -= cameraDeltaMove * cameraDirection;
			}
			gloablInput.cameraLookat = focusCamera ? cameraLookat : (cameraPos + cameraDirection);
		}
		if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		{
			if (focusCamera)
			{
				cameraYaw += cameraDeltaMove * 45.0f;
				float cameraArm = gloablInput.cameraArm;
				cameraPos.x = cos(glm::radians(cameraYaw)) * cos(glm::radians(cameraPitch)) * cameraArm;
				cameraPos.y = sin(glm::radians(cameraYaw)) * cos(glm::radians(cameraPitch)) * cameraArm;
				gloablInput.cameraYaw = cameraYaw;
			}
			else
			{
				cameraPos -= glm::normalize(glm::cross(cameraForward, cameraUp)) * cameraDeltaMove;
				cameraLookat = cameraPos + cameraForward;
			}
			gloablInput.cameraPos = cameraPos;
			gloablInput.cameraLookat = cameraLookat;
		}
		if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		{
			if (focusCamera)
			{
				cameraYaw -= cameraDeltaMove * 45.0f;
				float cameraArm = gloablInput.cameraArm;
				cameraPos.x = cos(glm::radians(cameraYaw)) * cos(glm::radians(cameraPitch)) * cameraArm;
				cameraPos.y = sin(glm::radians(cameraYaw)) * cos(glm::radians(cameraPitch)) * cameraArm;
				gloablInput.cameraYaw = cameraYaw;
			}
			else
			{
				cameraPos += glm::normalize(glm::cross(cameraForward, cameraUp)) * cameraDeltaMove;
				cameraLookat = cameraPos + cameraForward;
			}
			gloablInput.cameraPos = cameraPos;
			gloablInput.cameraLookat = cameraLookat;
		}
		if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
		{
			if (!focusCamera)
			{
				cameraPos.z -= cameraDeltaMove;
				cameraLookat = cameraPos + cameraForward;
				gloablInput.cameraPos = cameraPos;
				gloablInput.cameraLookat = cameraLookat;
			}
		}
		if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
		{
			if (!focusCamera)
			{
				cameraPos.z += cameraDeltaMove;
				cameraLookat = cameraPos + cameraForward;
				gloablInput.cameraPos = cameraPos;
				gloablInput.cameraLookat = cameraLookat;
			}
		}
		gloablInput.currentTime = currentFrame;
		gloablInput.deltaTime = deltaTime;
	}

	/** 在创建好一切必要资源后，执行绘制操作*/
	void drawFrame()
	{
		// 等待上一帧绘制完成
		vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

		uint32_t imageIndex;
		VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

		// 当窗口过期时（窗口尺寸改变或者窗口最小化后又重新显示），需要重新创建SwapChain并且停止这一帧的绘制
		if (result == VK_ERROR_OUT_OF_DATE_KHR) {
			recreateSwapChain();
			return;
		}
		else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
			throw std::runtime_error("failed to acquire swap chain image!");
		}

		// 更新统一缓存区（UBO）
		updateUniformBuffer(currentFrame);
		vkResetFences(device, 1, &inFlightFences[currentFrame]);

		// 清除渲染指令缓存
		vkResetCommandBuffer(commandBuffers[currentFrame], /*VkCommandBufferResetFlagBits*/ 0);
		// 记录新的所有的渲染指令缓存
		recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = waitSemaphores;
		submitInfo.pWaitDstStageMask = waitStages;

		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffers[currentFrame];

		VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = signalSemaphores;

		// 提交渲染指令
		if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
			throw std::runtime_error("failed to submit draw command buffer!");
		}

		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = signalSemaphores;

		VkSwapchainKHR swapChains[] = { swapChain };
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = swapChains;

		presentInfo.pImageIndices = &imageIndex;

		vkQueuePresentKHR(presentQueue, &presentInfo);

		currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
	}

protected:
	/** 创建程序和Vulkan之间的连接，涉及程序和显卡驱动之间特殊细节*/
	void createInstance()
	{
		if (enableValidationLayers && !checkValidationLayerSupport())
		{
			throw std::runtime_error("validation layers requested, but not available!");
		}

		VkApplicationInfo appInfo{};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = "Render the Scene";
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.pEngineName = "Vulkan Engine";
		appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.apiVersion = VK_API_VERSION_1_0;

		VkInstanceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		createInfo.pApplicationInfo = &appInfo;

		// 获取需要的glfw拓展名
		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensions;
		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

		if (enableValidationLayers)
		{
			extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		}
		createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
		createInfo.ppEnabledExtensionNames = extensions.data();

		VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
		if (enableValidationLayers)
		{
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();

			populateDebugMessengerCreateInfo(debugCreateInfo);
			createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
		}
		else
		{
			createInfo.enabledLayerCount = 0;

			createInfo.pNext = nullptr;
		}

		if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create instance!");
		}
	}

	/** 合法性监测层 Validation Layers
	 *	- 检查参数规范，检测是否使用
	 *	- 最终对象创建和销毁，找到资源泄漏
	 *	- 通过追踪线程原始调用，检查线程安全性
	 *	- 打印输出每次调用
	 *	- 为优化和重现追踪Vulkan调用
	*/
	void createDebugMessenger()
	{
		if (!enableValidationLayers) return;

		VkDebugUtilsMessengerCreateInfoEXT createInfo;
		populateDebugMessengerCreateInfo(createInfo);

		if (createDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to set up debug messenger!");
		}
	}

	/** WSI (Window System Integration) 链接Vulkan和Window系统，渲染Vulkan到桌面*/
	void createWindowsSurface()
	{
		if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
			throw std::runtime_error("failed to create window surface!");
		}
	}

	/** 选择支持Vulkan的显卡硬件*/
	void selectPhysicalDevice()
	{
		uint32_t deviceCount = 0;
		vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

		if (deviceCount == 0)
		{
			throw std::runtime_error("failed to find GPUs with Vulkan support!");
		}

		std::vector<VkPhysicalDevice> devices(deviceCount);
		vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

		for (const auto& device : devices)
		{
			if (isDeviceSuitable(device))
			{
				physicalDevice = device;
				break;
			}
		}

		if (physicalDevice == VK_NULL_HANDLE)
		{
			throw std::runtime_error("failed to find a suitable GPU!");
		}
	}

	/** 创建逻辑硬件对接物理硬件，相同物理硬件可以对应多个逻辑硬件*/
	void createLogicalDevice()
	{
		FQueueFamilyIndices queue_family_indices = findQueueFamilies(physicalDevice);

		std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
		std::set<uint32_t> uniqueQueueFamilies = { queue_family_indices.graphicsFamily.value(), queue_family_indices.presentFamily.value() };

		float queuePriority = 1.0f;
		for (uint32_t queueFamily : uniqueQueueFamilies)
		{
			VkDeviceQueueCreateInfo queueCreateInfo{};
			queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queueCreateInfo.queueFamilyIndex = queueFamily;
			queueCreateInfo.queueCount = 1;
			queueCreateInfo.pQueuePriorities = &queuePriority;
			queueCreateInfos.push_back(queueCreateInfo);
		}

		VkPhysicalDeviceFeatures deviceFeatures{};
		deviceFeatures.samplerAnisotropy = VK_TRUE;

		VkDeviceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

		createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
		createInfo.pQueueCreateInfos = queueCreateInfos.data();

		createInfo.pEnabledFeatures = &deviceFeatures;

		createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
		createInfo.ppEnabledExtensionNames = deviceExtensions.data();

		if (enableValidationLayers)
		{
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();
		}
		else
		{
			createInfo.enabledLayerCount = 0;
		}

		if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create logical device!");
		}

		vkGetDeviceQueue(device, queue_family_indices.graphicsFamily.value(), 0, &graphicsQueue);
		vkGetDeviceQueue(device, queue_family_indices.presentFamily.value(), 0, &presentQueue);
	}

	/** 交换链 Swap Chain
	 * Vulkan一种基础结构，持有帧缓存FrameBuffer
	 * SwapChain持有显示到窗口的图像队列
	 * 通常Vulkan获取图像，渲染到图像上，然后将图像推入SwapChain的图像队列
	 * SwapChain显示图像，通常和屏幕刷新率保持同步
	*/
	void createSwapChain()
	{
		FSwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

		VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
		VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
		VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

		uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
		if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
		{
			imageCount = swapChainSupport.capabilities.maxImageCount;
		}

		VkSwapchainCreateInfoKHR createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		createInfo.surface = surface;

		createInfo.minImageCount = imageCount;
		createInfo.imageFormat = surfaceFormat.format;
		createInfo.imageColorSpace = surfaceFormat.colorSpace;
		createInfo.imageExtent = extent;
		createInfo.imageArrayLayers = 1;
		createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

		FQueueFamilyIndices queue_family_indices = findQueueFamilies(physicalDevice);
		uint32_t queueFamilyIndices[] = { queue_family_indices.graphicsFamily.value(), queue_family_indices.presentFamily.value() };

		if (queue_family_indices.graphicsFamily != queue_family_indices.presentFamily)
		{
			createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
			createInfo.queueFamilyIndexCount = 2;
			createInfo.pQueueFamilyIndices = queueFamilyIndices;
		}
		else
		{
			createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		}

		createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
		createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		createInfo.presentMode = presentMode;
		createInfo.clipped = VK_TRUE;

		createInfo.oldSwapchain = VK_NULL_HANDLE;

		if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create swap chain!");
		}

		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
		swapChainImages.resize(imageCount);
		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

		swapChainImageFormat = surfaceFormat.format;
		swapChainExtent = extent;
	}

	/** 重新创建SwapChain*/
	void recreateSwapChain()
	{
		// 当窗口长宽都是零时，说明窗口被最小化了，这时需要等待
		int width = 0, height = 0;
		glfwGetFramebufferSize(window, &width, &height);
		while (width == 0 || height == 0) {
			glfwGetFramebufferSize(window, &width, &height);
			glfwWaitEvents();
		}

		vkDeviceWaitIdle(device);

		cleanupSwapChain();

		createSwapChain();
		createSwapChainImageViews();
		createFramebuffers();
	}

	/** 清理旧的SwapChain*/
	void cleanupSwapChain() {
		vkDestroyImageView(device, depthImageView, nullptr);
		vkDestroyImage(device, depthImage, nullptr);
		vkFreeMemory(device, depthImageMemory, nullptr);

		for (auto framebuffer : swapChainFramebuffers) {
			vkDestroyFramebuffer(device, framebuffer, nullptr);
		}

		for (auto imageView : swapChainImageViews) {
			vkDestroyImageView(device, imageView, nullptr);
		}

		vkDestroySwapchainKHR(device, swapChain, nullptr);
	}

	/** 图像视图 Image View
	 * 将视图显示为图像
	 * ImageView定义了SwapChain里定义的图像是什么样的
	 * 比如，带深度信息的RGB图像
	*/
	void createSwapChainImageViews()
	{
		swapChainImageViews.resize(swapChainImages.size());

		for (size_t i = 0; i < swapChainImages.size(); i++)
		{
			createImageView(swapChainImageViews[i], swapChainImages[i], swapChainImageFormat);
		}
	}

	/** 渲染层 RenderPass
	 * 创建渲染管线之前，需要先创建渲染层，告诉Vulkan渲染时使用的帧缓存FrameBuffer
	 * 我们需要指定渲染中使用的颜色缓存和深度缓存的数量，以及采样信息
	*/
	void createRenderPass()
	{
		VkAttachmentDescription colorAttachment{};
		colorAttachment.format = swapChainImageFormat;
		colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
		colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentDescription depthAttachment{};
		depthAttachment.format = findDepthFormat();
		depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
		depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkAttachmentReference colorAttachmentRef{};
		colorAttachmentRef.attachment = 0;
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference depthAttachmentRef{};
		depthAttachmentRef.attachment = 1;
		depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		// 渲染子通道 SubPass
		// SubPass是RenderPass的下属任务，和RenderPass共享Framebuffer等渲染资源
		// 某些渲染操作，比如后处理的Blooming，当前渲染需要依赖上一个渲染结果，但是渲染资源不变，这是SubPass可以优化性能
		VkSubpassDescription subpass{};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &colorAttachmentRef;
		subpass.pDepthStencilAttachment = &depthAttachmentRef;

		VkSubpassDependency dependency{};
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		dependency.dstSubpass = 0;
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		dependency.srcAccessMask = 0;
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

		// 这里将渲染三角形的操作，简化成一个SubPass提交
		std::array<VkAttachmentDescription, 2> attachments = { colorAttachment, depthAttachment };
		VkRenderPassCreateInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
		renderPassInfo.pAttachments = attachments.data();
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subpass;
		renderPassInfo.dependencyCount = 1;
		renderPassInfo.pDependencies = &dependency;

		if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
			throw std::runtime_error("failed to create render pass!");
		}
	}

	/** 创建指令池，管理所有的指令，比如DrawCall或者内存交换等*/
	void createCommandPool()
	{
		FQueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

		VkCommandPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

		if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create command pool!");
		}
	}

	/** 创建帧缓存，即每帧图像对应的渲染数据*/
	void createFramebuffers()
	{
		// 创建深度纹理资源
		VkFormat depthFormat = findDepthFormat();
		createImage(depthImage, depthImageMemory, swapChainExtent.width, swapChainExtent.height, depthFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		createImageView(depthImageView, depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);

		swapChainFramebuffers.resize(swapChainImageViews.size());

		for (size_t i = 0; i < swapChainImageViews.size(); i++) {
			std::array<VkImageView, 2> attachments = {
				swapChainImageViews[i],
				depthImageView
			};

			VkFramebufferCreateInfo framebufferInfo{};
			framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			framebufferInfo.renderPass = renderPass;
			framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
			framebufferInfo.pAttachments = attachments.data();
			framebufferInfo.width = swapChainExtent.width;
			framebufferInfo.height = swapChainExtent.height;
			framebufferInfo.layers = 1;

			if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create framebuffer!");
			}
		}
	}

	/**
	 * 创建阴影贴图资源 Shadow map
	*/
	void createShadowmapPass()
	{
		shadowmapPass.width = SHADOWMAP_DIM;
		shadowmapPass.height = SHADOWMAP_DIM;
		shadowmapPass.format = findDepthFormat();
		createImage(
			shadowmapPass.image, 
			shadowmapPass.memory, 
			shadowmapPass.width, shadowmapPass.height, shadowmapPass.format, 
			VK_IMAGE_TILING_OPTIMAL, 
			VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, 
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		createImageView(shadowmapPass.imageView, shadowmapPass.image, shadowmapPass.format, VK_IMAGE_ASPECT_DEPTH_BIT);
		createSampler(shadowmapPass.sampler, 
			VK_FILTER_LINEAR,
			VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE); // @TODO: 是否应该将 sample filter 改成 VK_FILTER_NEAREST

		//////////////////////////////////////////////////////////
		// 创建 uniformBuffers 和 uniformBuffersMemory
		VkDeviceSize bufferSize = sizeof(FUniformBufferBase);
		shadowmapPass.uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
		shadowmapPass.uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			createBuffer(bufferSize,
				VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				shadowmapPass.uniformBuffers[i],
				shadowmapPass.uniformBuffersMemory[i]);
		}

		//////////////////////////////////////////////////////////
		// 创建 descriptorSetLayout
		VkDescriptorSetLayoutBinding uboLayoutBinding{};
		uboLayoutBinding.binding = 0;
		uboLayoutBinding.descriptorCount = 1;
		uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		uboLayoutBinding.pImmutableSamplers = nullptr;
		uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
		std::vector<VkDescriptorSetLayoutBinding> bindings;
		bindings.resize(1);
		bindings[0] = uboLayoutBinding;
		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
		layoutInfo.pBindings = bindings.data();
		if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &shadowmapPass.descriptorSetLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor set layout!");
		}

		//////////////////////////////////////////////////////////
		// 创建 descriptorPool
		std::vector<VkDescriptorPoolSize> poolSizes;
		poolSizes.resize(1);
		poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
		poolInfo.pPoolSizes = poolSizes.data();
		poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
		if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &shadowmapPass.descriptorPool) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor pool!");
		}

		//////////////////////////////////////////////////////////
		// 创建 descriptorSets
		std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, shadowmapPass.descriptorSetLayout);
		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = shadowmapPass.descriptorPool;
		allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
		allocInfo.pSetLayouts = layouts.data();
		shadowmapPass.descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
		if (vkAllocateDescriptorSets(device, &allocInfo, shadowmapPass.descriptorSets.data()) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate descriptor sets!");
		}

		//////////////////////////////////////////////////////////
		// 绑定 descriptorSets
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) 
		{
			std::vector<VkWriteDescriptorSet> descriptorWrites{};
			descriptorWrites.resize(1);
			VkDescriptorBufferInfo bufferInfo{};
			bufferInfo.buffer = shadowmapPass.uniformBuffers[i];
			bufferInfo.offset = 0;
			bufferInfo.range = sizeof(FUniformBufferBase);
			descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[0].dstSet = shadowmapPass.descriptorSets[i];
			descriptorWrites[0].dstBinding = 0;
			descriptorWrites[0].dstArrayElement = 0;
			descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			descriptorWrites[0].descriptorCount = 1;
			descriptorWrites[0].pBufferInfo = &bufferInfo;
			vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
		}

		VkAttachmentDescription attachmentDescription{};
		attachmentDescription.format = shadowmapPass.format;
		attachmentDescription.samples = VK_SAMPLE_COUNT_1_BIT;
		// Clear depth at beginning of the render pass
		attachmentDescription.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		// We will read from depth, so it's important to store the depth attachment results
		attachmentDescription.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachmentDescription.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachmentDescription.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		// We don't care about initial layout of the attachment
		attachmentDescription.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		// Attachment will be transitioned to shader read at render pass end
		attachmentDescription.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

		VkAttachmentReference depthReference = {};
		depthReference.attachment = 0;
		// Attachment will be used as depth/stencil during render pass
		depthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpass = {};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		// No color attachments
		subpass.colorAttachmentCount = 0;
		// Reference to our depth attachment
		subpass.pDepthStencilAttachment = &depthReference;

		// Use subpass dependencies for layout transitions
		std::array<VkSubpassDependency, 2> dependencies;

		dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[0].dstSubpass = 0;
		dependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		dependencies[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		dependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
		dependencies[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		dependencies[1].srcSubpass = 0;
		dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[1].srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
		dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		dependencies[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
		
		VkRenderPassCreateInfo renderPassCreateInfo{};
		renderPassCreateInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassCreateInfo.attachmentCount = 1;
		renderPassCreateInfo.pAttachments = &attachmentDescription;
		renderPassCreateInfo.subpassCount = 1;
		renderPassCreateInfo.pSubpasses = &subpass;
		renderPassCreateInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());;
		renderPassCreateInfo.pDependencies = dependencies.data();

		if (vkCreateRenderPass(device, &renderPassCreateInfo, nullptr, &shadowmapPass.renderPass)) {
			throw std::runtime_error("failed to create shadow map render pass!");
		}

		// Create frame buffer
		VkFramebufferCreateInfo fbufCreateInfo{};
		fbufCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		fbufCreateInfo.renderPass = shadowmapPass.renderPass;
		fbufCreateInfo.attachmentCount = 1;
		fbufCreateInfo.pAttachments = &shadowmapPass.imageView;
		fbufCreateInfo.width = shadowmapPass.width;
		fbufCreateInfo.height = shadowmapPass.height;
		fbufCreateInfo.layers = 1;

		if (vkCreateFramebuffer(device, &fbufCreateInfo, nullptr, &shadowmapPass.frameBuffer)) {
			throw std::runtime_error("failed to create shadow map frame buffer!");
		}

		VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCI{};
		inputAssemblyStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssemblyStateCI.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		inputAssemblyStateCI.primitiveRestartEnable = VK_FALSE;
		VkPipelineRasterizationStateCreateInfo rasterizationStateCI{};
		rasterizationStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizationStateCI.depthClampEnable = VK_FALSE;
		rasterizationStateCI.rasterizerDiscardEnable = VK_FALSE;
		rasterizationStateCI.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizationStateCI.lineWidth = 1.0f;
		rasterizationStateCI.cullMode = VK_CULL_MODE_BACK_BIT /*VK_CULL_MODE_BACK_BIT*/;
		rasterizationStateCI.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterizationStateCI.depthBiasEnable = VK_FALSE;
		VkPipelineColorBlendAttachmentState blendAttachmentState{};
		blendAttachmentState.colorWriteMask = 0xf;
		blendAttachmentState.blendEnable = VK_FALSE;
		VkPipelineColorBlendStateCreateInfo colorBlendStateCI{};
		colorBlendStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlendStateCI.logicOpEnable = VK_FALSE;
		colorBlendStateCI.logicOp = VK_LOGIC_OP_COPY;
		colorBlendStateCI.attachmentCount = 1;
		colorBlendStateCI.pAttachments = &blendAttachmentState;
		colorBlendStateCI.blendConstants[0] = 0.0f;
		colorBlendStateCI.blendConstants[1] = 0.0f;
		colorBlendStateCI.blendConstants[2] = 0.0f;
		colorBlendStateCI.blendConstants[3] = 0.0f;
		VkPipelineDepthStencilStateCreateInfo depthStencilStateCI{};
		depthStencilStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthStencilStateCI.depthTestEnable = VK_TRUE;
		depthStencilStateCI.depthWriteEnable = VK_TRUE;
		depthStencilStateCI.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthStencilStateCI.depthBoundsTestEnable = VK_FALSE;
		depthStencilStateCI.minDepthBounds = 0.0f; // Optional
		depthStencilStateCI.maxDepthBounds = 1.0f; // Optional
		depthStencilStateCI.stencilTestEnable = VK_FALSE; // 没有写轮廓信息，所以跳过轮廓测试
		depthStencilStateCI.front = {}; // Optional
		depthStencilStateCI.back = {}; // Optional
		VkPipelineViewportStateCreateInfo viewportStateCI{};
		viewportStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportStateCI.viewportCount = 1;
		viewportStateCI.scissorCount = 1;
		VkPipelineMultisampleStateCreateInfo multisampleStateCI{};
		multisampleStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampleStateCI.sampleShadingEnable = VK_FALSE;
		multisampleStateCI.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		std::vector<VkDynamicState> dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
		VkPipelineDynamicStateCreateInfo dynamicStateCI{};
		dynamicStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicStateCI.dynamicStateCount = static_cast<uint32_t>(dynamicStateEnables.size());
		dynamicStateCI.pDynamicStates = dynamicStateEnables.data();
		
		auto vertShaderCode = readShaderSource("Resources/Shaders/draw_with_shadow_sm_vert.spv");
		auto fragShaderCode = readShaderSource("Resources/Shaders/draw_with_shadow_sm_frag.spv");
		VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
		VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);
		VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderStageInfo.module = vertShaderModule;
		vertShaderStageInfo.pName = "main";
		VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageInfo.module = fragShaderModule;
		fragShaderStageInfo.pName = "main";
		VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

		VkPipelineVertexInputStateCreateInfo vertexInputCI{};
		auto bindingDescription = FVertex::getBindingDescription();
		auto attributeDescriptions = FVertex::getAttributeDescriptions();
		vertexInputCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertexInputCI.vertexBindingDescriptionCount = 1;
		vertexInputCI.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
		vertexInputCI.pVertexBindingDescriptions = &bindingDescription;
		vertexInputCI.pVertexAttributeDescriptions = attributeDescriptions.data();

		VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 1;
		pipelineLayoutInfo.pSetLayouts = &shadowmapPass.descriptorSetLayout;
		pipelineLayoutInfo.pushConstantRangeCount = 0;

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &shadowmapPass.pipelineLayout) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create pipeline layout!");
		}

		VkGraphicsPipelineCreateInfo pipelineCI{};
		pipelineCI.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineCI.stageCount = 2;
		pipelineCI.pStages = shaderStages;
		pipelineCI.pVertexInputState = &vertexInputCI;
		pipelineCI.pInputAssemblyState = &inputAssemblyStateCI;
		pipelineCI.pViewportState = &viewportStateCI;
		pipelineCI.pRasterizationState = &rasterizationStateCI;
		pipelineCI.pMultisampleState = &multisampleStateCI;
		pipelineCI.pDepthStencilState = &depthStencilStateCI; // 加上深度测试
		pipelineCI.pColorBlendState = &colorBlendStateCI;
		pipelineCI.pDynamicState = &dynamicStateCI;
		pipelineCI.layout = shadowmapPass.pipelineLayout;
		pipelineCI.renderPass = renderPass;
		pipelineCI.subpass = 0;
		pipelineCI.basePipelineHandle = VK_NULL_HANDLE;

		// Offscreen pipeline (vertex shader only)
		//shaderStages[0] = loadShader(getShadersPath() + "shadowmapping/offscreen.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
		pipelineCI.stageCount = 1;
		// No blend attachment states (no color attachments used)
		colorBlendStateCI.attachmentCount = 0;
		// Disable culling, so all faces contribute to shadows
		rasterizationStateCI.cullMode = VK_CULL_MODE_NONE;
		depthStencilStateCI.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		// Enable depth bias
		rasterizationStateCI.depthBiasEnable = VK_TRUE;
		// Add depth bias to dynamic state, so we can change it at runtime
		dynamicStateEnables.push_back(VK_DYNAMIC_STATE_DEPTH_BIAS);
		dynamicStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicStateCI.dynamicStateCount = static_cast<uint32_t>(dynamicStateEnables.size());
		dynamicStateCI.pDynamicStates = dynamicStateEnables.data();

		pipelineCI.renderPass = shadowmapPass.renderPass;

		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineCI, nullptr, &shadowmapPass.pipeline) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		vkDestroyShaderModule(device, fragShaderModule, nullptr);
		vkDestroyShaderModule(device, vertShaderModule, nullptr);
	}


	/** Cube map Faces Rules:
	 *       Y3
	 *       ||
	 * X1 == Z4 == X0 == Z5
	 *       ||
	 *       Y4
	 * | https://matheowis.github.io/HDRI-to-CubeMap/    |
	 * | X0(p-x) | X1(n-x) | Y2(p-z) | Y3(n-z) | Z4(n-y) | Z5(p-y)|
	 * |    90   |   -90   |    0    |   180   |    0    |   180  |
	 */
	void createCubemapResources()
	{
		createImageHDRContext(cubemapImage, cubemapImageMemory, cubemapImageView, cubemapSampler, cubemapMaxMips, {
			"Resources/Textures/cubemap_X0.png",
			"Resources/Textures/cubemap_X1.png",
			"Resources/Textures/cubemap_Y2.png",
			"Resources/Textures/cubemap_Y3.png",
			"Resources/Textures/cubemap_Z4.png",
			"Resources/Textures/cubemap_Z5.png" });
	}

	/** 创建统一缓存区（UBO）*/
	void createUniformBuffers()
	{
		VkDeviceSize baseBufferSize = sizeof(FUniformBufferBase);
		baseUniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
		baseUniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			createBuffer(baseBufferSize, 
				VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, 
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
				baseUniformBuffers[i], 
				baseUniformBuffersMemory[i]);

			// 这里会导致 memory stack overflow ，不应该在这里 vkMapMemory
			//vkMapMemory(device, baseUniformBuffersMemory[i], 0, bufferSize, 0, &uniformBuffersMapped[i]);
		}

		VkDeviceSize bufferSizeOfView = sizeof(FUniformBufferView);
		viewUniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
		viewUniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			createBuffer(
				bufferSizeOfView, 
				VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, 
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
				viewUniformBuffers[i], 
				viewUniformBuffersMemory[i]);
		}
	}

	void createBaseScenePass()
	{
		auto createRenderResource = [this](FRenderObject& outObject, const std::string& objfile, const std::vector<std::string>& pngfiles) -> void
		{
			createGeometry(outObject.vertices, outObject.indices, objfile);
			outObject.textureImages.resize(pngfiles.size());
			outObject.textureImageMemorys.resize(pngfiles.size());
			outObject.textureImageViews.resize(pngfiles.size());
			outObject.textureSamplers.resize(pngfiles.size());
			for (size_t i = 0; i < pngfiles.size(); i++)
			{
				// 一个便捷函数，创建图像，视口和采样器
				bool sRGB = (i == 0);
				createImageContext(outObject.textureImages[i], outObject.textureImageMemorys[i], outObject.textureImageViews[i], outObject.textureSamplers[i], pngfiles[i], sRGB);
			}

			createVertexBuffer(outObject.vertexBuffer, outObject.vertexBufferMemory, outObject.vertices);
			createIndexBuffer(outObject.indexBuffer, outObject.indexBufferMemory, outObject.indices);
			createDescriptorPool(outObject.descriptorPool, static_cast<uint32_t>(pngfiles.size() + 1));
			createDescriptorSets(outObject.descriptorSets, outObject.descriptorPool, baseScenePass.descriptorSetLayout, outObject.textureImageViews, outObject.textureSamplers);
		};

		// 创建环境反射纹理资源
		createCubemapResources();

		// layout_size直接定义了DescriptorSets的大小，如，有一个UBO和两张贴图，那么布局的大小就是 1+2=3
		uint32_t sampler_number = 5;
		// 创建场景渲染流水线和着色器
		createDescriptorSetLayout(baseScenePass.descriptorSetLayout, sampler_number);
		uint32_t specConstantsCount = globalConstants.specConstantsCount;
		baseScenePass.pipelines.resize(specConstantsCount);
		CreateGraphicsPipelines(
			baseScenePass.pipelineLayout, 
			baseScenePass.pipelines, specConstantsCount,
			baseScenePass.descriptorSetLayout, 
			"Resources/Shaders/draw_with_shadow_base_vert.spv", 
			"Resources/Shaders/draw_with_shadow_base_frag.spv");

		{
			std::string asset_set_dir = "Resources/AssetSets";
			for (const auto & folder : std::filesystem::directory_iterator(asset_set_dir))
			{
				std::string asset_set = folder.path().generic_string();
				std::string models_dir = asset_set + std::string("/models/");
				std::string textures_dir = asset_set + std::string("/textures/");
				if (!std::filesystem::is_directory(models_dir) ||
					!std::filesystem::is_directory(textures_dir))
				{
					continue;
				}
				for (const auto & model : std::filesystem::directory_iterator(models_dir))
				{
					std::string model_file = model.path().generic_string();
					std::string model_file_name = model_file.substr(model_file.find_last_of("/\\") + 1);
					std::string::size_type const p(model_file_name.find_last_of('.'));
					std::string model_name = model_file_name.substr(0, p);
					
					std::string texture_bc = textures_dir + model_name + std::string("_bc.png");
					if (!std::filesystem::exists(texture_bc)) {
						texture_bc = std::string("Resources/Textures/default_grey.png");
					}
					std::string texture_m = textures_dir + model_name + std::string("_m.png");
					if (!std::filesystem::exists(texture_m)) {
						texture_m = std::string("Resources/Textures/default_black.png");
					}
					std::string texture_r = textures_dir + model_name + std::string("_r.png");
					if (!std::filesystem::exists(texture_r)) {
						texture_r = std::string("Resources/Textures/default_white.png");
					}
					std::string texture_n = textures_dir + model_name + std::string("_n.png");
					if (!std::filesystem::exists(texture_n)) {
						texture_n = std::string("Resources/Textures/default_normal.png");
					}
					std::string texture_ao = textures_dir + model_name + std::string("_ao.png");
					if (!std::filesystem::exists(texture_ao)) {
						texture_ao = std::string("Resources/Textures/default_white.png");
					}
					
					FRenderObject asset;
					std::string asset_obj = model_file;
					std::vector<std::string> asset_imgs = {
						texture_bc,
						texture_m,
						texture_r,
						texture_n,
						texture_ao};
					createRenderResource(asset, asset_obj, asset_imgs);
					baseScenePass.renderObjects.push_back(asset);
				}
			}
		}
		//assert(house_element_num == 0);
		
		if (baseScenePass.renderObjects.size() == 0) {
			FRenderObject sphere;
			std::string sphere_obj = "Resources/Models/sphere.obj";
			std::vector<std::string> sphere_imgs = {
				"Resources/Textures/default_grey.png",
				"Resources/Textures/default_black.png",
				"Resources/Textures/default_white.png",
				"Resources/Textures/default_normal.png",
				"Resources/Textures/default_white.png" };
			createRenderResource(sphere, sphere_obj, sphere_imgs);
			baseScenePass.renderObjects.push_back(sphere);
		}
		FRenderObject stage;
		std::string stage_obj = "Resources/Models/stage.obj";
		std::vector<std::string> stage_imgs = {
			"Resources/Textures/default_grey.png",
			"Resources/Textures/default_black.png",
			"Resources/Textures/default_white.png",
			"Resources/Textures/default_normal.png",
			"Resources/Textures/default_white.png" };
		createRenderResource(stage, stage_obj, stage_imgs);
		baseScenePass.renderObjects.push_back(stage);
		//~ 结束 创建场景，包括VBO，UBO，贴图等
	}


	void createBackgroundPass()
	{
		// 创建背景贴图
		createImageContext(
			backgroundPass.image,
			backgroundPass.memory,
			backgroundPass.imageView,
			backgroundPass.sampler,
			"Resources/Textures/background.png");
		createDescriptorSetLayout(backgroundPass.descriptorSetLayout);
		createDescriptorPool(backgroundPass.descriptorPool);
		createDescriptorSets(
			backgroundPass.descriptorSets,
			backgroundPass.descriptorPool,
			backgroundPass.descriptorSetLayout,
			backgroundPass.imageView,
			backgroundPass.sampler);
		backgroundPass.pipelines.resize(1);
		CreateGraphicsPipelines(
			backgroundPass.pipelineLayout,
			backgroundPass.pipelines, 1,
			backgroundPass.descriptorSetLayout, 
			"Resources/Shaders/draw_with_shadow_bg_vert.spv", 
			"Resources/Shaders/draw_with_shadow_bg_frag.spv",
			VK_FALSE, VK_CULL_MODE_NONE);
	}

	/** 创建指令缓存，多个CPU Core可以并行的往CommandBuffer中发送指令，可以充分利用CPU的多核性能*/
	void createCommandBuffer()
	{
		commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = commandPool;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

		if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate command buffers!");
		}
	}

	/** 把需要执行的指令写入指令缓存，对应每一个SwapChain的图像*/
	void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex)
	{
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

		// 开始记录指令
		if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to begin recording command buffer!");
		}

		// 渲染阴影
		{
			VkRenderPassBeginInfo renderPassInfo{};
			renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			renderPassInfo.renderPass = shadowmapPass.renderPass;
			renderPassInfo.framebuffer = shadowmapPass.frameBuffer;
			renderPassInfo.renderArea.extent.width = shadowmapPass.width;
			renderPassInfo.renderArea.extent.height = shadowmapPass.height;
			std::array<VkClearValue, 1> clearValues{};
			clearValues[0].depthStencil = { 1.0f, 0 };
			renderPassInfo.clearValueCount = 1;
			renderPassInfo.pClearValues = clearValues.data();

			// 【阴影】开始 RenderPass
			vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

			// 【阴影】视口信息
			VkViewport viewport{};
			viewport.x = 0.0f;
			viewport.y = 0.0f;
			viewport.width = (float)shadowmapPass.width;
			viewport.height = (float)shadowmapPass.height;
			viewport.minDepth = 0.0f;
			viewport.maxDepth = 1.0f;

			// 【阴影】视口剪切信息
			VkRect2D scissor{};
			scissor.offset = { 0, 0 };
			scissor.extent.width = shadowmapPass.width;
			scissor.extent.height = shadowmapPass.height;

			// 【阴影】设置渲染视口
			vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

			// 【阴影】设置视口剪切，是否可以通过这个函数来实现 Tiled-Based Rendering ？
			vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

			// Set depth bias (aka "Polygon offset")
			// Required to avoid shadow mapping artifacts
			// Depth bias (and slope) are used to avoid shadowing artifacts
			// Constant depth bias factor (always applied)
			float depthBiasConstant = 1.25f;
			// Slope depth bias factor, applied depending on polygon's slope
			float depthBiasSlope = 7.5; // change from 1.75f to fix PCF artifact
			vkCmdSetDepthBias(
				commandBuffer,
				depthBiasConstant,
				0.0f,
				depthBiasSlope);

			// 【阴影】渲染场景
			for (size_t i = 0; i < baseScenePass.renderObjects.size(); i++)
			{
				vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowmapPass.pipeline);
				FRenderObject renderObject = baseScenePass.renderObjects[i];
				VkBuffer objectVertexBuffers[] = { renderObject.vertexBuffer };
				VkDeviceSize objectOffsets[] = { 0 };
				vkCmdBindVertexBuffers(commandBuffer, 0, 1, objectVertexBuffers, objectOffsets);
				vkCmdBindIndexBuffer(commandBuffer, renderObject.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
				vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
					shadowmapPass.pipelineLayout, 0, 1, &shadowmapPass.descriptorSets[currentFrame], 0, nullptr);
				vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(renderObject.indices.size()), 1, 0, 0, 0);
			}

			// 【阴影】结束 RenderPass
			vkCmdEndRenderPass(commandBuffer);
		}

		// 渲染场景
		{
			VkRenderPassBeginInfo renderPassInfo{};
			renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			renderPassInfo.renderPass = renderPass;
			renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
			renderPassInfo.renderArea.offset = { 0, 0 };
			renderPassInfo.renderArea.extent = swapChainExtent;

			std::array<VkClearValue, 2> clearValues{};
			clearValues[0].color = { {0.0f, 0.0f, 0.0f, 1.0f} };
			clearValues[1].depthStencil = { 1.0f, 0 };

			renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
			renderPassInfo.pClearValues = clearValues.data();

			// 【主场景】开始 RenderPass
			vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

			// 【主场景】渲染视口信息
			VkViewport viewport{};
			viewport.x = 0.0f;
			viewport.y = 0.0f;
			viewport.width = (float)swapChainExtent.width;
			viewport.height = (float)swapChainExtent.height;
			viewport.minDepth = 0.0f;
			viewport.maxDepth = 1.0f;

			// 【主场景】视口剪切信息
			VkRect2D scissor{};
			scissor.offset = { 0, 0 };
			scissor.extent = swapChainExtent;

			// 【主场景】设置渲染视口
			vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

			// 【主场景】设置视口剪切，是否可以通过这个函数来实现 Tiled-Based Rendering ？
			vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

			// 【主场景】渲染背景面片
			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, backgroundPass.pipelines[0]);
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, backgroundPass.pipelineLayout, 0, 1, &backgroundPass.descriptorSets[currentFrame], 0, nullptr);
			vkCmdPushConstants(commandBuffer, backgroundPass.pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(FGlobalConstants), &globalConstants);
			vkCmdDraw(commandBuffer, 6, 1, 0, 0);

			// 【主场景】渲染场景
			for (size_t i = 0; i < baseScenePass.renderObjects.size(); i++)
			{
				uint32_t specConstants = globalConstants.specConstants;
				VkPipeline baseScenePassPipeline = baseScenePass.pipelines[specConstants];
				vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, baseScenePassPipeline);
				FRenderObject renderObject = baseScenePass.renderObjects[i];
				VkBuffer objectVertexBuffers[] = { renderObject.vertexBuffer };
				VkDeviceSize objectOffsets[] = { 0 };
				vkCmdBindVertexBuffers(commandBuffer, 0, 1, objectVertexBuffers, objectOffsets);
				vkCmdBindIndexBuffer(commandBuffer, renderObject.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
				vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, baseScenePass.pipelineLayout, 0, 1, &renderObject.descriptorSets[currentFrame], 0, nullptr);
				vkCmdPushConstants(commandBuffer, baseScenePass.pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(FGlobalConstants), &globalConstants);
				vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(renderObject.indices.size()), 1, 0, 0, 0);
			}

			// 【主场景】结束RenderPass
			vkCmdEndRenderPass(commandBuffer);
		}

		// 结束记录指令
		if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to record command buffer!");
		}
	}

	/** 创建同步物体，同步显示当前渲染*/
	void createSyncObjects()
	{
		imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

		VkSemaphoreCreateInfo semaphoreInfo{};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkFenceCreateInfo fenceInfo{};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
				vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
				vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {

				throw std::runtime_error("failed to create synchronization objects for a frame!");
			}
		}
	}

	/** 删除函数 initVulkan 中创建的元素*/
	void destroyVulkan()
	{
		// commandBuffer 不需要释放
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
			vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
			vkDestroyFence(device, inFlightFences[i], nullptr);
		}

		vkDestroyRenderPass(device, renderPass, nullptr);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			vkDestroyBuffer(device, baseUniformBuffers[i], nullptr);
			vkFreeMemory(device, baseUniformBuffersMemory[i], nullptr);
		}

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			vkDestroyBuffer(device, viewUniformBuffers[i], nullptr);
			vkFreeMemory(device, viewUniformBuffersMemory[i], nullptr);
		}

		vkDestroyImageView(device, cubemapImageView, nullptr);
		vkDestroySampler(device, cubemapSampler, nullptr);
		vkDestroyImage(device, cubemapImage, nullptr);
		vkFreeMemory(device, cubemapImageMemory, nullptr);

		// 清理 shadowmapPass
		vkDestroyRenderPass(device, shadowmapPass.renderPass, nullptr);
		vkDestroyFramebuffer(device, shadowmapPass.frameBuffer, nullptr);
		vkDestroyDescriptorPool(device, shadowmapPass.descriptorPool, nullptr);
		vkDestroyDescriptorSetLayout(device, shadowmapPass.descriptorSetLayout, nullptr);
		vkDestroyPipelineLayout(device, shadowmapPass.pipelineLayout, nullptr);
		vkDestroyPipeline(device, shadowmapPass.pipeline, nullptr);
		vkDestroyImageView(device, shadowmapPass.imageView, nullptr);
		vkDestroySampler(device, shadowmapPass.sampler, nullptr);
		vkDestroyImage(device, shadowmapPass.image, nullptr);
		vkFreeMemory(device, shadowmapPass.memory, nullptr);
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			vkDestroyBuffer(device, shadowmapPass.uniformBuffers[i], nullptr);
			vkFreeMemory(device, shadowmapPass.uniformBuffersMemory[i], nullptr);
		}

		// 清理 backgroundPass
		vkDestroyDescriptorSetLayout(device, backgroundPass.descriptorSetLayout, nullptr);
		vkDestroyPipelineLayout(device, backgroundPass.pipelineLayout, nullptr);
		vkDestroyPipeline(device, backgroundPass.pipelines[0], nullptr);
		vkDestroyDescriptorPool(device, backgroundPass.descriptorPool, nullptr);
		vkDestroyImageView(device, backgroundPass.imageView, nullptr);
		vkDestroySampler(device, backgroundPass.sampler, nullptr);
		vkDestroyImage(device, backgroundPass.image, nullptr);
		vkFreeMemory(device, backgroundPass.memory, nullptr);

		// 清理 baseScenePass
		vkDestroyDescriptorSetLayout(device, baseScenePass.descriptorSetLayout, nullptr);
		vkDestroyPipelineLayout(device, baseScenePass.pipelineLayout, nullptr);
		for (uint32_t i = 0; i < globalConstants.specConstantsCount; i++)
		{
			vkDestroyPipeline(device, baseScenePass.pipelines[i], nullptr);
		}
		for (size_t i = 0; i < baseScenePass.renderObjects.size(); i++)
		{
			FRenderObject renderObject = baseScenePass.renderObjects[i];

			vkDestroyDescriptorPool(device, renderObject.descriptorPool, nullptr);

			for (size_t j = 0; j < renderObject.textureImages.size(); j++)
			{
				vkDestroyImageView(device, renderObject.textureImageViews[j], nullptr);
				vkDestroySampler(device, renderObject.textureSamplers[j], nullptr);
				vkDestroyImage(device, renderObject.textureImages[j], nullptr);
				vkFreeMemory(device, renderObject.textureImageMemorys[j], nullptr);
			}

			vkDestroyBuffer(device, renderObject.vertexBuffer, nullptr);
			vkFreeMemory(device, renderObject.vertexBufferMemory, nullptr);
			vkDestroyBuffer(device, renderObject.indexBuffer, nullptr);
			vkFreeMemory(device, renderObject.indexBufferMemory, nullptr);
		}

		vkDestroyCommandPool(device, commandPool, nullptr);

		vkDestroyDevice(device, nullptr);

		if (enableValidationLayers)
		{
			destroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
		}

		vkDestroySurfaceKHR(instance, surface, nullptr);
		vkDestroyInstance(instance, nullptr);
	}

protected:
	/** 选择SwapChain渲染到视图的图像的格式*/
	VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats)
	{
		// 找到合适的图像格式
		// VK_FORMAT_B8G8R8A8_SRGB 将图像BGRA存储在 unsigned normalized format 中，使用SRGB 非线性编码，颜色空间为非线性空间，不用Gamma矫正最终结果
		// VK_FORMAT_B8G8R8A8_UNORM 将图像BGRA存储在 unsigned normalized format 中，颜色空间为线性空间，像素的最终输出颜色需要Gamma矫正
		for (const auto& availableFormat : availableFormats)
		{
			// 将 FrameBuffer Image 设置为线性空间，方便 PBR 的工作流以及颜色矫正（ColorCorrection）
			//if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
			if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM && availableFormat.colorSpace == VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT)
			{
				return availableFormat;
			}
		}

		return availableFormats[0];
	}

	/** 选择SwapChain的显示方式
	 * VK_PRESENT_MODE_IMMEDIATE_KHR 图形立即显示在屏幕上，会出现图像撕裂
	 * VK_PRESENT_MODE_FIFO_KHR 图像会被推入一个队列，先入后出显示到屏幕，如果队列满了，程序会等待，和垂直同步相似
	 * VK_PRESENT_MODE_FIFO_RELAXED_KHR 基于第二个Mode，当队列满了，程序不会等待，而是直接渲染到屏幕，会出现图像撕裂
	 * VK_PRESENT_MODE_MAILBOX_KHR 基于第二个Mode，当队列满了，程序不会等待，而是直接替换队列中的图像，
	*/
	VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes)
	{
		for (const auto& availablePresentMode : availablePresentModes)
		{
			if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
			{
				return availablePresentMode;
			}
		}

		return VK_PRESENT_MODE_FIFO_KHR;
	}

	VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities)
	{
		if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
		{
			return capabilities.currentExtent;
		}
		else
		{
			int width, height;
			glfwGetFramebufferSize(window, &width, &height);

			VkExtent2D actualExtent = {
				static_cast<uint32_t>(width),
				static_cast<uint32_t>(height)
			};

			actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
			actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

			return actualExtent;
		}
	}

	FSwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device)
	{
		FSwapChainSupportDetails details;

		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

		uint32_t formatCount;
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

		if (formatCount != 0)
		{
			details.formats.resize(formatCount);
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
		}

		uint32_t presentModeCount;
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

		if (presentModeCount != 0)
		{
			details.presentModes.resize(presentModeCount);
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
		}

		return details;
	}

	/** 检测硬件是否合适*/
	bool isDeviceSuitable(VkPhysicalDevice device)
	{
		FQueueFamilyIndices queue_family_indices = findQueueFamilies(device);

		uint32_t extensionCount;
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

		std::vector<VkExtensionProperties> availableExtensions(extensionCount);
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

		std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

		for (const auto& extension : availableExtensions)
		{
			requiredExtensions.erase(extension.extensionName);
		}

		bool extensionsSupported = requiredExtensions.empty();

		bool swapChainAdequate = false;
		if (extensionsSupported)
		{
			FSwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
			swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
		}

		VkPhysicalDeviceFeatures supportedFeatures;
		vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

		return queue_family_indices.isComplete() && extensionsSupported && swapChainAdequate && supportedFeatures.samplerAnisotropy;
	}

	/** 队列家族 Queue Family
	 * 找到所有支持Vulkan的显卡硬件
	*/
	FQueueFamilyIndices findQueueFamilies(VkPhysicalDevice device)
	{
		FQueueFamilyIndices queue_family_indices;

		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

		int i = 0;
		for (const auto& queueFamily : queueFamilies)
		{
			if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
			{
				queue_family_indices.graphicsFamily = i;
			}

			VkBool32 presentSupport = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

			if (presentSupport)
			{
				queue_family_indices.presentFamily = i;
			}

			if (queue_family_indices.isComplete())
			{
				break;
			}

			i++;
		}

		return queue_family_indices;
	}

	/** 找到物理硬件支持的图片格式*/
	VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {
		for (VkFormat format : candidates) {
			VkFormatProperties props;
			vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);

			if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
				return format;
			}
			else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
				return format;
			}
		}

		throw std::runtime_error("failed to find supported format!");
	}

	/** 找到支持深度贴图的格式*/
	VkFormat findDepthFormat() {
		return findSupportedFormat(
			{ VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
			VK_IMAGE_TILING_OPTIMAL,
			VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
		);
	}

	/** 查找内存类型*/
	uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
	{
		VkPhysicalDeviceMemoryProperties memProperties;
		vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

		// 自动寻找适合的内存类型
		for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
			if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
				return i;
			}
		}

		throw std::runtime_error("failed to find suitable memory type!");
	}

	VkCommandBuffer beginSingleTimeCommands()
	{
		// 和渲染一样，使用CommandBuffer拷贝缓存
		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandPool = commandPool;
		allocInfo.commandBufferCount = 1;

		VkCommandBuffer commandBuffer;
		vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		vkBeginCommandBuffer(commandBuffer, &beginInfo);

		return commandBuffer;
	}

	void endSingleTimeCommands(VkCommandBuffer commandBuffer)
	{
		vkEndCommandBuffer(commandBuffer);

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;

		vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
		vkQueueWaitIdle(graphicsQueue);

		vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
	}

	/** 通用函数用来创建Buffer*/
	void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
		VkBufferCreateInfo bufferInfo{};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = size;
		bufferInfo.usage = usage;
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to create buffer!");
		}

		// 为VertexBuffer创建内存，并赋予
		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		// 自动找到适合的内存类型
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);
		// 关联分配的内存地址
		if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate buffer memory!");
		}
		// 绑定VertexBuffer和它的内存地址
		vkBindBufferMemory(device, buffer, bufferMemory, 0);
	}

	/** 通用函数用来拷贝Buffer*/
	void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size)
	{
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkBufferCopy copyRegion{};
		copyRegion.size = size;
		vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

		endSingleTimeCommands(commandBuffer);
	}

	/** 从文件中读取顶点和点序*/
	void createGeometry(std::vector<FVertex>& outVertices, std::vector<uint32_t>& outIndices, const std::string& filename)
	{
		readModelAsset(filename, outVertices, outIndices);
	}

	/** 创建顶点缓存区VBO*/
	void createVertexBuffer(VkBuffer& outBuffer, VkDeviceMemory& outMemory, const std::vector<FVertex>& inVertices)
	{
		// 根据vertices大小创建VertexBuffer
		VkDeviceSize bufferSize = sizeof(inVertices[0]) * inVertices.size();

		// 为什么需要stagingBuffer，因为直接创建VertexBuffer，CPU端可以直接通过vertexBufferMemory范围GPU使用的内存，这样太危险了，
		// 所以我们先创建一个临时的Buffer写入数据，然后将这个Buffer拷贝给最终的VertexBuffer，
		// VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT标签，使得最终的VertexBuffer位于硬件本地内存中，比如显卡的显存。
		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		// 通用函数用来创建VertexBuffer，这样可以方便创建StagingBuffer和真正的VertexBuffer
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		// 把数据拷贝到顶点缓存区中
		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, inVertices.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, outBuffer, outMemory);

		copyBuffer(stagingBuffer, outBuffer, bufferSize);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

	/** 创建点序缓存区IBO*/
	void createIndexBuffer(VkBuffer& outBuffer, VkDeviceMemory& outMemory, const std::vector<uint32_t>& inIndices)
	{
		VkDeviceSize bufferSize = sizeof(inIndices[0]) * inIndices.size();

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, inIndices.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, outBuffer, outMemory);

		copyBuffer(stagingBuffer, outBuffer, bufferSize);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

	/** 更新统一缓存区（UBO）*/
	void updateUniformBuffer(const uint32_t currentImageIdx)
	{
		glm::vec3 cameraPos = gloablInput.cameraPos;
		glm::vec3 cameraLookat = gloablInput.cameraLookat;
		glm::vec3 cameraUp = glm::vec3(0.0, 0.0, 1.0);
		float cameraFOV = gloablInput.cameraFOV;
		float zNear = gloablInput.zNear;
		float zFar = gloablInput.zFar;

		static auto startTime = std::chrono::high_resolution_clock::now();
		auto currentTime = std::chrono::high_resolution_clock::now();
		float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();
		globalConstants.time = time;

		shadowmapPass.zNear = zNear;
		shadowmapPass.zFar = zFar;

		FLight light;
		glm::vec3 center = glm::vec3(0.0f);
		glm::vec3 lightPos = glm::vec3(4.0f, 0.0f, 4.0f);
		float rollLight = gloablInput.playLightRoll ? 
			(gloablInput.rollLight + gloablInput.deltaTime * 45.0f) : gloablInput.rollLight;
		gloablInput.rollLight = rollLight;
		lightPos.x = cos(glm::radians(rollLight)) * 4.0f;
		lightPos.y = sin(glm::radians(rollLight)) * 4.0f;
		glm::vec3 lightDir = glm::normalize(lightPos - center);

		light.position = glm::vec4(lightPos, 0.0);
		light.color = glm::vec4(1.0, 1.0, 1.0, 3.0);
		light.direction = glm::vec4(lightDir, 0.0);
		light.info = glm::vec4(0.0, 0.0, 0.0, 0.0);

		float rollStage = gloablInput.playStageRoll ?
			(gloablInput.rollStage + gloablInput.deltaTime * glm::radians(15.0f)) : gloablInput.rollStage;
		gloablInput.rollStage = rollStage;
		glm::mat4 localToWorld = glm::rotate(glm::mat4(1.0f), rollStage, glm::vec3(0.0f, 0.0f, 1.0f));
		glm::mat4 shadowView = glm::lookAt(lightPos, glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		glm::mat4 shadowProjection = glm::perspective(glm::radians(cameraFOV), 1.0f, shadowmapPass.zNear, shadowmapPass.zFar);
		shadowProjection[1][1] *= -1;

		FUniformBufferBase UBOBaseData{};
		UBOBaseData.model = localToWorld;
		UBOBaseData.view = glm::lookAt(cameraPos, cameraLookat, cameraUp);
		UBOBaseData.proj = glm::perspective(glm::radians(cameraFOV), swapChainExtent.width / (float)swapChainExtent.height, zNear, zFar);
		UBOBaseData.proj[1][1] *= -1;

		void* data_base_ubo;
		vkMapMemory(device, baseUniformBuffersMemory[currentImageIdx], 0, sizeof(UBOBaseData), 0, &data_base_ubo);
		memcpy(data_base_ubo, &UBOBaseData, sizeof(UBOBaseData));
		vkUnmapMemory(device, baseUniformBuffersMemory[currentImageIdx]);

		FUniformBufferView UBOViewData{};
		// shadowmap_space 的 MVP 矩阵中，M矩阵在FS中计算，所以传入 localToWorld 进入FS
		UBOViewData.shadowmap_space = shadowProjection * shadowView;
		UBOViewData.local_to_world = localToWorld;
		UBOViewData.camera_info = glm::vec4(cameraPos, cameraFOV);
		UBOViewData.lights_count = glm::ivec4(1, 0, 0, cubemapMaxMips);
		UBOViewData.directional_lights[0] = light;
		UBOViewData.zNear = shadowmapPass.zNear;
		UBOViewData.zFar = shadowmapPass.zFar;

		void* data_view;
		vkMapMemory(device, viewUniformBuffersMemory[currentImageIdx], 0, sizeof(UBOViewData), 0, &data_view);
		memcpy(data_view, &UBOViewData, sizeof(UBOViewData));
		vkUnmapMemory(device, viewUniformBuffersMemory[currentImageIdx]);

		FUniformBufferBase UBOShadowData{};
		UBOShadowData.model = localToWorld;
		UBOShadowData.view = shadowView;
		UBOShadowData.proj = shadowProjection;

		void* data_shadow_ubo;
		vkMapMemory(device, shadowmapPass.uniformBuffersMemory[currentImageIdx], 0, sizeof(UBOShadowData), 0, &data_shadow_ubo);
		memcpy(data_shadow_ubo, &UBOShadowData, sizeof(UBOShadowData));
		vkUnmapMemory(device, shadowmapPass.uniformBuffersMemory[currentImageIdx]);
	}

	/** 创建Shader模块*/
	VkShaderModule createShaderModule(const std::vector<char>& code)
	{
		VkShaderModuleCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		createInfo.codeSize = code.size();
		createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

		VkShaderModule shaderModule;
		if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create shader module!");
		}

		return shaderModule;
	}

	/**创建图形渲染管线*/
	void CreateGraphicsPipelines(
		VkPipelineLayout& outPipelineLayout,
		std::vector<VkPipeline>& outPipelines,
		const uint32_t specConstantsCount,
		const VkDescriptorSetLayout& inDescriptorSetLayout,
		const std::string& vert_filename,
		const std::string& frag_filename,
		const VkBool32 bDepthTest = VK_TRUE, 
		const VkCullModeFlags CullMode = VK_CULL_MODE_BACK_BIT)
	{
		auto vertShaderCode = readShaderSource(vert_filename);
		auto fragShaderCode = readShaderSource(frag_filename);

		VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
		VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

		VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderStageInfo.module = vertShaderModule;
		vertShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageInfo.module = fragShaderModule;
		fragShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

		// 顶点缓存绑定的描述，定义了顶点都需要绑定什么数据，比如第一个位置绑定Position，第二个位置绑定Color，第三个位置绑定UV等
		auto bindingDescription = FVertex::getBindingDescription();
		auto attributeDescriptions = FVertex::getAttributeDescriptions();

		// 渲染管线VertexBuffer输入
		VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
		if (bDepthTest == VK_FALSE && CullMode == VK_CULL_MODE_NONE) // 如果不做深度检测并且不做背面剔除，那么认为是渲染背景，不需要绑定VBO
		{
			vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
			vertexInputInfo.vertexBindingDescriptionCount = 0;
			vertexInputInfo.vertexAttributeDescriptionCount = 0;
		}
		else // 正常VBO渲染绑定
		{
			vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
			vertexInputInfo.vertexBindingDescriptionCount = 1;
			vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
			vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
			vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();
		}

		VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		inputAssembly.primitiveRestartEnable = VK_FALSE;

		VkPipelineViewportStateCreateInfo viewportState{};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.scissorCount = 1;

		VkPipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth = 1.0f;
		// 关闭背面剔除，使得材质TwoSide渲染
		rasterizer.cullMode = CullMode /*VK_CULL_MODE_BACK_BIT*/;
		rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterizer.depthBiasEnable = VK_FALSE;

		VkPipelineMultisampleStateCreateInfo multisampling{};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		// 打开深度测试
		VkPipelineDepthStencilStateCreateInfo depthStencil{};
		depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthStencil.depthTestEnable = bDepthTest; //VK_TRUE;
		depthStencil.depthWriteEnable = bDepthTest; //VK_TRUE;
		depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
		depthStencil.depthBoundsTestEnable = VK_FALSE;
		depthStencil.minDepthBounds = 0.0f; // Optional
		depthStencil.maxDepthBounds = 1.0f; // Optional
		depthStencil.stencilTestEnable = VK_FALSE; // 没有写轮廓信息，所以跳过轮廓测试
		depthStencil.front = {}; // Optional
		depthStencil.back = {}; // Optional

		VkPipelineColorBlendAttachmentState colorBlendAttachment{};
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_FALSE;

		VkPipelineColorBlendStateCreateInfo colorBlending{};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlending.logicOpEnable = VK_FALSE;
		colorBlending.logicOp = VK_LOGIC_OP_COPY;
		colorBlending.attachmentCount = 1;
		colorBlending.pAttachments = &colorBlendAttachment;
		colorBlending.blendConstants[0] = 0.0f;
		colorBlending.blendConstants[1] = 0.0f;
		colorBlending.blendConstants[2] = 0.0f;
		colorBlending.blendConstants[3] = 0.0f;

		std::vector<VkDynamicState> dynamicStates = {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR
		};
		VkPipelineDynamicStateCreateInfo dynamicState{};
		dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
		dynamicState.pDynamicStates = dynamicStates.data();

		// 设置 push constants
		VkPushConstantRange pushConstant;
		// 这个PushConstant的范围从头开始
		pushConstant.offset = 0;
		pushConstant.size = sizeof(FGlobalConstants);
		// 这是个全局PushConstant，所以希望各个着色器都能访问到
		pushConstant.stageFlags = VK_SHADER_STAGE_ALL;

		// 在渲染管线创建时，指定DescriptorSetLayout，用来传UniformBuffer
		VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 1;
		pipelineLayoutInfo.pSetLayouts = &inDescriptorSetLayout;
		pipelineLayoutInfo.pPushConstantRanges = &pushConstant;
		pipelineLayoutInfo.pushConstantRangeCount = 1;

		// PipelineLayout可以用来创建和绑定VertexBuffer和UniformBuffer，这样可以往着色器中传递参数
		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &outPipelineLayout) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create pipeline layout!");
		}

		VkGraphicsPipelineCreateInfo pipelineInfo{};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.stageCount = 2;
		pipelineInfo.pStages = shaderStages;
		pipelineInfo.pVertexInputState = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssembly;
		pipelineInfo.pViewportState = &viewportState;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState = &multisampling;
		pipelineInfo.pDepthStencilState = &depthStencil; // 加上深度测试
		pipelineInfo.pColorBlendState = &colorBlending;
		pipelineInfo.pDynamicState = &dynamicState;
		pipelineInfo.layout = outPipelineLayout;
		pipelineInfo.renderPass = renderPass;
		pipelineInfo.subpass = 0;
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

		// Use specialization constants 优化着色器变体
		for (uint32_t i = 0; i < specConstantsCount; i++)
		{
			uint32_t specConstants = i;
			VkSpecializationMapEntry specializationMapEntry = VkSpecializationMapEntry{};
			specializationMapEntry.constantID = 0;
			specializationMapEntry.offset = 0;
			specializationMapEntry.size = sizeof(uint32_t);
			VkSpecializationInfo specializationInfo = VkSpecializationInfo();
			specializationInfo.mapEntryCount = 1;
			specializationInfo.pMapEntries = &specializationMapEntry;
			specializationInfo.dataSize = sizeof(uint32_t);
			specializationInfo.pData = &specConstants;
			shaderStages[1].pSpecializationInfo = &specializationInfo;

			if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &outPipelines[i]) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create graphics pipeline!");
			}
		}

		vkDestroyShaderModule(device, fragShaderModule, nullptr);
		vkDestroyShaderModule(device, vertShaderModule, nullptr);
	}

	/** 读取一个贴图路径，然后创建图像、视口和采样器等资源*/
	void createImageContext(
		VkImage& outImage, 
		VkDeviceMemory& outMemory, 
		VkImageView& outImageView, 
		VkSampler& outSampler, 
		const std::string& filename, bool sRGB = true)
	{
		int texWidth, texHeight, texChannels, mipLevels;
		stbi_uc* pixels = readTextureAsset(filename, texWidth, texHeight, texChannels, mipLevels);

		VkDeviceSize imageSize = texWidth * texHeight * 4;

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
		memcpy(data, pixels, static_cast<size_t>(imageSize));
		vkUnmapMemory(device, stagingBufferMemory);

		// 清理pixels数据结构
		stbi_image_free(pixels);

		VkFormat format = sRGB ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
		// VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT 告诉Vulkan这张贴图即要被读也要被写
		createImage(
			outImage, 
			outMemory, 
			texWidth, texHeight, format, 
			VK_IMAGE_TILING_OPTIMAL, 
			VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, 
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, mipLevels);

		transitionImageLayout(outImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mipLevels);
		copyBufferToImage(stagingBuffer, outImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
		//transitionImageLayout(outImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);

		generateMipmaps(outImage, format, texWidth, texHeight, mipLevels);

		createImageView(outImageView, outImage, format, VK_IMAGE_ASPECT_COLOR_BIT, mipLevels);
		createSampler(outSampler, 
			VK_FILTER_LINEAR, 
			VK_SAMPLER_ADDRESS_MODE_REPEAT, 
			VK_SAMPLER_ADDRESS_MODE_REPEAT, 
			VK_SAMPLER_ADDRESS_MODE_REPEAT,
			VK_BORDER_COLOR_INT_OPAQUE_BLACK,
			mipLevels);
	}

	/** 读取一个HDR贴图路径，然后创建CUBEMAP图像资源*/
	void createImageHDRContext(
		VkImage& outImage, 
		VkDeviceMemory& outMemory, 
		VkImageView& outImageView, 
		VkSampler& outSampler, 
		uint32_t& outMaxMipLevels, 
		const std::vector<std::string>& filenames)
	{
		// https://matheowis.github.io/HDRI-to-CubeMap/
		int texWidth, texHeight, texChannels, mipLevels;
		stbi_uc* pixels[6];
		for (int i = 0; i < 6; ++i) {
			pixels[i] = readTextureAsset(filenames[i], texWidth, texHeight, texChannels, mipLevels);
		}
		VkDeviceSize imageSize = texWidth * texHeight * 4;

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(imageSize * 6, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		for (int i = 0; i < 6; ++i) {
			void* writeLocation;
			vkMapMemory(device, stagingBufferMemory, imageSize * i, imageSize, 0, &writeLocation);
			memcpy(writeLocation, pixels[i], imageSize);
			vkUnmapMemory(device, stagingBufferMemory);
			// 清理pixels数据结构
			stbi_image_free(pixels[i]);
		}

		// createImage
		{
			VkImageCreateInfo imageInfo{};
			imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
			imageInfo.imageType = VK_IMAGE_TYPE_2D;
			imageInfo.extent.width = texWidth;
			imageInfo.extent.height = texHeight;
			imageInfo.extent.depth = 1;
			imageInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
			imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
			imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
			imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
			imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
			imageInfo.mipLevels = mipLevels;
			imageInfo.arrayLayers = 6;
			imageInfo.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;

			if (vkCreateImage(device, &imageInfo, nullptr, &outImage) != VK_SUCCESS) {
				throw std::runtime_error("failed to create image!");
			}

			VkMemoryRequirements memRequirements;
			vkGetImageMemoryRequirements(device, outImage, &memRequirements);

			VkMemoryAllocateInfo allocInfo{};
			allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			allocInfo.allocationSize = memRequirements.size;
			allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

			if (vkAllocateMemory(device, &allocInfo, nullptr, &outMemory) != VK_SUCCESS) {
				throw std::runtime_error("failed to allocate image memory!");
			}

			vkBindImageMemory(device, outImage, outMemory, 0);
		}
		// transitionImageLayout
		{
			VkCommandBuffer commandBuffer = beginSingleTimeCommands();

			VkImageMemoryBarrier barrier{};
			barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.image = outImage;
			VkImageSubresourceRange subresourceRange;
			subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			subresourceRange.baseMipLevel = 0;
			subresourceRange.levelCount = mipLevels;
			subresourceRange.baseArrayLayer = 0;
			subresourceRange.layerCount = 6;
			barrier.subresourceRange = subresourceRange;

			VkPipelineStageFlags sourceStage;
			VkPipelineStageFlags destinationStage;
			sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
			vkCmdPipelineBarrier(
				commandBuffer,
				sourceStage, destinationStage,
				0,
				0, nullptr,
				0, nullptr,
				1, &barrier
			);
			endSingleTimeCommands(commandBuffer);
		}
		// copyBufferToImage
		{
			VkCommandBuffer commandBuffer = beginSingleTimeCommands();

			VkBufferImageCopy region{};
			region.bufferOffset = 0;
			region.bufferRowLength = 0;
			region.bufferImageHeight = 0;
			VkImageSubresourceLayers imageSubresource;
			imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			imageSubresource.mipLevel = 0;
			imageSubresource.baseArrayLayer = 0;
			imageSubresource.layerCount = 6;
			region.imageSubresource = imageSubresource;
			region.imageOffset = { 0, 0, 0 };
			region.imageExtent = {
				static_cast<uint32_t>(texWidth),
				static_cast<uint32_t>(texHeight),
				1
			};
			vkCmdCopyBufferToImage(commandBuffer, stagingBuffer, outImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
			endSingleTimeCommands(commandBuffer);
		}
		// generateMipmaps
		{
			VkFormatProperties formatProperties;
			vkGetPhysicalDeviceFormatProperties(physicalDevice, VK_FORMAT_R8G8B8A8_SRGB, &formatProperties);

			if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) {
				throw std::runtime_error("texture image format does not support linear blitting!");
			}

			VkCommandBuffer commandBuffer = beginSingleTimeCommands();

			VkImageMemoryBarrier barrier{};
			barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.image = outImage;
			VkImageSubresourceRange subresourceRange;
			subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			subresourceRange.baseMipLevel = 0;
			subresourceRange.levelCount = 1; // 注意，这里传入的CUBEMAP只有一级Mips，不是mipLevels
			subresourceRange.baseArrayLayer = 0;
			subresourceRange.layerCount = 6;
			barrier.subresourceRange = subresourceRange;

			// 当layerCount = 6时，vkCmd执行时会 loop每个faces，所以不用在外部写循环去逐个执行Cubemap的faces ： for(uint32_t face = 0; face < 6; face++) {...}
			int32_t mipWidth = texWidth;
			int32_t mipHeight = texHeight;
			for (uint32_t i = 1; i < static_cast<uint32_t>(mipLevels); i++)
			{
				barrier.subresourceRange.baseMipLevel = i - 1;
				barrier.subresourceRange.levelCount = 1; // 注意，这里传入的CUBEMAP只有一级Mips，不是mipLevels
				barrier.subresourceRange.baseArrayLayer = 0;
				barrier.subresourceRange.layerCount = 6;
				barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
				barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
				barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
				barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

				vkCmdPipelineBarrier(commandBuffer,
					VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
					0, nullptr,
					0, nullptr,
					1, &barrier);

				VkImageBlit blit{};
				blit.srcOffsets[0] = { 0, 0, 0 };
				blit.srcOffsets[1] = { mipWidth, mipHeight, 1 };
				blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
				blit.srcSubresource.mipLevel = i - 1;
				blit.srcSubresource.baseArrayLayer = 0;
				blit.srcSubresource.layerCount = 6;
				blit.dstOffsets[0] = { 0, 0, 0 };
				blit.dstOffsets[1] = { mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1 };
				blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
				blit.dstSubresource.mipLevel = i;
				blit.dstSubresource.baseArrayLayer = 0;
				blit.dstSubresource.layerCount = 6;

				vkCmdBlitImage(commandBuffer,
					outImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
					outImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
					1, &blit,
					VK_FILTER_LINEAR);

				barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
				barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
				barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

				vkCmdPipelineBarrier(commandBuffer,
					VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
					0, nullptr,
					0, nullptr,
					1, &barrier);

				if (mipWidth > 1) mipWidth /= 2;
				if (mipHeight > 1) mipHeight /= 2;
			}

			barrier.subresourceRange.baseMipLevel = mipLevels - 1;
			barrier.subresourceRange.baseArrayLayer = 0;
			barrier.subresourceRange.layerCount = 6;
			barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

			vkCmdPipelineBarrier(commandBuffer,
				VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
				0, nullptr,
				0, nullptr,
				1, &barrier);

			endSingleTimeCommands(commandBuffer);
		}
		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);

		// createImageView
		{
			VkImageViewCreateInfo viewInfo{};
			viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			viewInfo.image = outImage;
			viewInfo.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
			viewInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
			viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			viewInfo.subresourceRange.baseMipLevel = 0;
			viewInfo.subresourceRange.levelCount = mipLevels;
			viewInfo.subresourceRange.baseArrayLayer = 0;
			viewInfo.subresourceRange.layerCount = 6;
			if (vkCreateImageView(device, &viewInfo, nullptr, &outImageView) != VK_SUCCESS) {
				throw std::runtime_error("failed to create texture image view!");
			}
		}
		createSampler(outSampler,
			VK_FILTER_LINEAR,
			VK_SAMPLER_ADDRESS_MODE_REPEAT,
			VK_SAMPLER_ADDRESS_MODE_REPEAT,
			VK_SAMPLER_ADDRESS_MODE_REPEAT,
			VK_BORDER_COLOR_INT_OPAQUE_BLACK,
			mipLevels);
		outMaxMipLevels = mipLevels;
	}

	/** 使用ImageMemoryBarrier，可以同步的访问贴图资源，避免一张贴图被读取时正在被写入*/
	void transitionImageLayout(VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t miplevels = 1)
	{
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkImageMemoryBarrier barrier{};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.oldLayout = oldLayout;
		barrier.newLayout = newLayout;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image = image;
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		barrier.subresourceRange.baseMipLevel = 0;
		barrier.subresourceRange.levelCount = miplevels;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;

		VkPipelineStageFlags sourceStage;
		VkPipelineStageFlags destinationStage;

		if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

			sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

			sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
			destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		}
		else {
			throw std::invalid_argument("unsupported layout transition!");
		}

		vkCmdPipelineBarrier(
			commandBuffer,
			sourceStage, destinationStage,
			0,
			0, nullptr,
			0, nullptr,
			1, &barrier
		);

		endSingleTimeCommands(commandBuffer);
	}

	void generateMipmaps(VkImage image, VkFormat imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels)
	{
		// 检查图像格式是否支持 linear blitting
		VkFormatProperties formatProperties;
		vkGetPhysicalDeviceFormatProperties(physicalDevice, imageFormat, &formatProperties);

		if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) {
			throw std::runtime_error("texture image format does not support linear blitting!");
		}

		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkImageMemoryBarrier barrier{};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.image = image;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;
		barrier.subresourceRange.levelCount = 1;

		int32_t mipWidth = texWidth;
		int32_t mipHeight = texHeight;

		for (uint32_t i = 1; i < mipLevels; i++) {
			barrier.subresourceRange.baseMipLevel = i - 1;
			barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

			vkCmdPipelineBarrier(commandBuffer,
				VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
				0, nullptr,
				0, nullptr,
				1, &barrier);

			VkImageBlit blit{};
			blit.srcOffsets[0] = { 0, 0, 0 };
			blit.srcOffsets[1] = { mipWidth, mipHeight, 1 };
			blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			blit.srcSubresource.mipLevel = i - 1;
			blit.srcSubresource.baseArrayLayer = 0;
			blit.srcSubresource.layerCount = 1;
			blit.dstOffsets[0] = { 0, 0, 0 };
			blit.dstOffsets[1] = { mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1 };
			blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			blit.dstSubresource.mipLevel = i;
			blit.dstSubresource.baseArrayLayer = 0;
			blit.dstSubresource.layerCount = 1;

			vkCmdBlitImage(commandBuffer,
				image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				1, &blit,
				VK_FILTER_LINEAR);

			barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

			vkCmdPipelineBarrier(commandBuffer,
				VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
				0, nullptr,
				0, nullptr,
				1, &barrier);

			if (mipWidth > 1) mipWidth /= 2;
			if (mipHeight > 1) mipHeight /= 2;
		}

		barrier.subresourceRange.baseMipLevel = mipLevels - 1;
		barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		vkCmdPipelineBarrier(commandBuffer,
			VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
			0, nullptr,
			0, nullptr,
			1, &barrier);

		endSingleTimeCommands(commandBuffer);
	}

	/** 将缓存拷贝到图片对象中*/
	void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height)
	{
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkBufferImageCopy region{};
		region.bufferOffset = 0;
		region.bufferRowLength = 0;
		region.bufferImageHeight = 0;
		region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		region.imageSubresource.mipLevel = 0;
		region.imageSubresource.baseArrayLayer = 0;
		region.imageSubresource.layerCount = 1;
		region.imageOffset = { 0, 0, 0 };
		region.imageExtent = {
			width,
			height,
			1
		};

		vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

		endSingleTimeCommands(commandBuffer);
	}

	/** 创建图像资源*/
	void createImage(
		VkImage& outImage, 
		VkDeviceMemory& OutImageMemory, 
		const uint32_t width, const uint32_t height, const VkFormat format,
		const VkImageTiling tiling, const VkImageUsageFlags usage, 
		const VkMemoryPropertyFlags properties, const uint32_t miplevels = 1)
	{
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.extent.width = width;
		imageInfo.extent.height = height;
		imageInfo.extent.depth = 1;
		imageInfo.format = format;
		imageInfo.tiling = tiling;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage = usage;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageInfo.arrayLayers = 1;
		imageInfo.mipLevels = miplevels;

		if (vkCreateImage(device, &imageInfo, nullptr, &outImage) != VK_SUCCESS) {
			throw std::runtime_error("failed to create image!");
		}

		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(device, outImage, &memRequirements);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

		if (vkAllocateMemory(device, &allocInfo, nullptr, &OutImageMemory) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate image memory!");
		}

		vkBindImageMemory(device, outImage, OutImageMemory, 0);
	}

	/** 创建图像视口*/
	void createImageView(
		VkImageView& outImageView, 
		const VkImage& inImage, 
		const VkFormat inFormat, 
		const VkImageAspectFlags aspectFlags = VK_IMAGE_ASPECT_COLOR_BIT,
		const uint32_t levelCount = 1)
	{
		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = inImage;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = inFormat;
		viewInfo.subresourceRange.aspectMask = aspectFlags; // VK_IMAGE_ASPECT_COLOR_BIT 颜色 VK_IMAGE_ASPECT_DEPTH_BIT 深度
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = levelCount;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 1;

		if (vkCreateImageView(device, &viewInfo, nullptr, &outImageView) != VK_SUCCESS) {
			throw std::runtime_error("failed to create texture image view!");
		}
	}

	/** 创建采样器*/
	void createSampler(
		VkSampler& outSampler,
		const VkFilter filter = VK_FILTER_LINEAR,
		const VkSamplerAddressMode addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		const VkSamplerAddressMode addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		const VkSamplerAddressMode addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		const VkBorderColor borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
		const uint32_t miplevels = 1)
	{
		VkPhysicalDeviceProperties properties{};
		vkGetPhysicalDeviceProperties(physicalDevice, &properties);

		VkSamplerCreateInfo samplerInfo{};
		samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerInfo.magFilter = filter;
		samplerInfo.minFilter = filter;
		samplerInfo.addressModeU = addressModeU;
		samplerInfo.addressModeV = addressModeV;
		samplerInfo.addressModeW = addressModeW;
		// 可在此处关闭各项异性，有些硬件可能不支持
		samplerInfo.anisotropyEnable = VK_TRUE;
		samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
		samplerInfo.borderColor = borderColor;
		samplerInfo.unnormalizedCoordinates = VK_FALSE;
		samplerInfo.compareEnable = VK_FALSE;
		samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
		samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		samplerInfo.minLod = 0.0f;
		samplerInfo.maxLod = static_cast<float>(miplevels);
		samplerInfo.mipLodBias = 0.0f;

		if (vkCreateSampler(device, &samplerInfo, nullptr, &outSampler) != VK_SUCCESS) {
			throw std::runtime_error("failed to create texture sampler!");
		}
	}

	/** 通用函数用来创建DescriptorSetLayout*/
	void createDescriptorSetLayout(VkDescriptorSetLayout& outDescriptorSetLayout, uint32_t sampler_number = 1)
	{
		// UnifromBufferObject（ubo）绑定
		VkDescriptorSetLayoutBinding baseUBOLayoutBinding{};
		baseUBOLayoutBinding.binding = 0;
		baseUBOLayoutBinding.descriptorCount = 1;
		baseUBOLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		baseUBOLayoutBinding.pImmutableSamplers = nullptr;
		baseUBOLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

		// UnifromBufferObject（ubo）绑定
		VkDescriptorSetLayoutBinding viewUBOLayoutBinding{};
		viewUBOLayoutBinding.binding = 1;
		viewUBOLayoutBinding.descriptorCount = 1;
		viewUBOLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		viewUBOLayoutBinding.pImmutableSamplers = nullptr;
		viewUBOLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT; // view ubo 主要信息用于 fragment shader

		// 环境反射Cubemap贴图绑定
		VkDescriptorSetLayoutBinding cubemapLayoutBinding{};
		cubemapLayoutBinding.binding = 2;
		cubemapLayoutBinding.descriptorCount = 1;
		cubemapLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		cubemapLayoutBinding.pImmutableSamplers = nullptr;
		cubemapLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutBinding shadowmapLayoutBinding{};
		shadowmapLayoutBinding.binding = 3;
		shadowmapLayoutBinding.descriptorCount = 1;
		shadowmapLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		shadowmapLayoutBinding.pImmutableSamplers = nullptr;
		shadowmapLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		// 将UnifromBufferObject和贴图采样器绑定到DescriptorSetLayout上
		std::vector<VkDescriptorSetLayoutBinding> bindings;
		bindings.resize(sampler_number + 4); // 这里3是2个UniformBuffer和1个环境Cubemap贴图
		bindings[0] = baseUBOLayoutBinding;
		bindings[1] = viewUBOLayoutBinding;
		bindings[2] = cubemapLayoutBinding;
		bindings[3] = shadowmapLayoutBinding;
		for (size_t i = 0; i < sampler_number; i++)
		{
			VkDescriptorSetLayoutBinding samplerLayoutBinding{};
			samplerLayoutBinding.binding = static_cast<uint32_t>(i + 4);
			samplerLayoutBinding.descriptorCount = 1;
			samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			samplerLayoutBinding.pImmutableSamplers = nullptr;
			samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

			bindings[i + 4] = samplerLayoutBinding;
		}
		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
		layoutInfo.pBindings = bindings.data();
		if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &outDescriptorSetLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor set layout!");
		}
	}

	/** 通用函数用来创建DescriptorPool
	 * outDescriptorPool ：输出的DescriptorPool
	 * sampler_num ：贴图采样器的数量
	 */
	void createDescriptorPool(VkDescriptorPool& outDescriptorPool, uint32_t sampler_num = 1)
	{
		std::vector<VkDescriptorPoolSize> poolSizes;
		poolSizes.resize(sampler_num + 4); // 这里3是2个UniformBuffer和一个环境Cubemap贴图
		poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
		poolSizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
		poolSizes[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[2].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
		poolSizes[3].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[3].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
		for (size_t i = 0; i < sampler_num; i++)
		{
			poolSizes[i + 4].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			poolSizes[i + 4].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
		}

		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
		poolInfo.pPoolSizes = poolSizes.data();
		poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

		if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &outDescriptorPool) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor pool!");
		}
	}

	/** 函数用来创建默认的只有一份贴图的DescriptorSets*/
	void createDescriptorSets(std::vector<VkDescriptorSet>& outDescriptorSets, const VkDescriptorPool& inDescriptorPool, const VkDescriptorSetLayout& inDescriptorSetLayout, const VkImageView& inImageView, const VkSampler& inSampler)
	{
		std::vector<VkImageView> imageViews;
		imageViews.push_back(inImageView);
		std::vector<VkSampler> samplers;
		samplers.push_back(inSampler);
		createDescriptorSets(outDescriptorSets, inDescriptorPool, inDescriptorSetLayout, imageViews, samplers);
	}

	/** 通用函数用来创建DescriptorSets*/
	void createDescriptorSets(std::vector<VkDescriptorSet>& outDescriptorSets, const VkDescriptorPool& inDescriptorPool, const VkDescriptorSetLayout& inDescriptorSetLayout, const std::vector<VkImageView>& inImageViews, const std::vector<VkSampler>& inSamplers)
	{
		std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, inDescriptorSetLayout);
		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = inDescriptorPool;
		allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
		allocInfo.pSetLayouts = layouts.data();

		outDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
		if (vkAllocateDescriptorSets(device, &allocInfo, outDescriptorSets.data()) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate descriptor sets!");
		}

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {

			uint32_t write_size = static_cast<uint32_t>(inImageViews.size()) + 4; // 这里加2为 UniformBuffer 的个数
			std::vector<VkWriteDescriptorSet> descriptorWrites{};
			descriptorWrites.resize(write_size);

			// 绑定 UnifromBuffer
			VkDescriptorBufferInfo baseBufferInfo{};
			baseBufferInfo.buffer = baseUniformBuffers[i];
			baseBufferInfo.offset = 0;
			baseBufferInfo.range = sizeof(FUniformBufferBase);

			descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[0].dstSet = outDescriptorSets[i];
			descriptorWrites[0].dstBinding = 0;
			descriptorWrites[0].dstArrayElement = 0;
			descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			descriptorWrites[0].descriptorCount = 1;
			descriptorWrites[0].pBufferInfo = &baseBufferInfo;

			// 绑定 UnifromBuffer
			VkDescriptorBufferInfo viewBufferInfo{};
			viewBufferInfo.buffer = viewUniformBuffers[i];
			viewBufferInfo.offset = 0;
			viewBufferInfo.range = sizeof(FUniformBufferView);

			descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[1].dstSet = outDescriptorSets[i];
			descriptorWrites[1].dstBinding = 1;
			descriptorWrites[1].dstArrayElement = 0;
			descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			descriptorWrites[1].descriptorCount = 1;
			descriptorWrites[1].pBufferInfo = &viewBufferInfo;

			VkDescriptorImageInfo imageInfo{};
			imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			imageInfo.imageView = cubemapImageView;
			imageInfo.sampler = cubemapSampler;

			descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[2].dstSet = outDescriptorSets[i];
			descriptorWrites[2].dstBinding = 2;
			descriptorWrites[2].dstArrayElement = 0;
			descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descriptorWrites[2].descriptorCount = 1;
			descriptorWrites[2].pImageInfo = &imageInfo;

			VkDescriptorImageInfo shadowmapImageInfo{};
			shadowmapImageInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
			shadowmapImageInfo.imageView = shadowmapPass.imageView;
			shadowmapImageInfo.sampler = shadowmapPass.sampler;

			descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[3].dstSet = outDescriptorSets[i];
			descriptorWrites[3].dstBinding = 3;
			descriptorWrites[3].dstArrayElement = 0;
			descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descriptorWrites[3].descriptorCount = 1;
			descriptorWrites[3].pImageInfo = &shadowmapImageInfo;

			// 绑定 Textures
			// descriptorWrites会引用每一个创建的VkDescriptorImageInfo，所以需要用一个数组把它们存储起来
			std::vector<VkDescriptorImageInfo> imageInfos;
			imageInfos.resize(inImageViews.size());
			for (size_t j = 0; j < inImageViews.size(); j++)
			{
				VkDescriptorImageInfo imageInfo{};
				imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				imageInfo.imageView = inImageViews[j];
				imageInfo.sampler = inSamplers[j];
				imageInfos[j] = imageInfo;

				descriptorWrites[j + 4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				descriptorWrites[j + 4].dstSet = outDescriptorSets[i];
				descriptorWrites[j + 4].dstBinding = static_cast<uint32_t>(j + 4);
				descriptorWrites[j + 4].dstArrayElement = 0;
				descriptorWrites[j + 4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				descriptorWrites[j + 4].descriptorCount = 1;
				descriptorWrites[j + 4].pImageInfo = &imageInfos[j]; // 注意，这里是引用了VkDescriptorImageInfo，所有需要创建imageInfos这个数组，存储所有的imageInfo而不是使用局部变量imageInfo
			}

			vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
		}
	}

private:
	/** 将编译的着色器二进制SPV文件，读入内存Buffer中*/
	static std::vector<char> readShaderSource(const std::string& filename)
	{
		std::ifstream file(filename, std::ios::ate | std::ios::binary);

		if (!file.is_open())
		{
			throw std::runtime_error("failed to open shader file!");
		}

		size_t fileSize = (size_t)file.tellg();
		std::vector<char> buffer(fileSize);

		file.seekg(0);
		file.read(buffer.data(), fileSize);

		file.close();

		return buffer;
	}

	/** 从图片文件中读取贴像素信息*/
	static stbi_uc* readTextureAsset(const std::string& filename, int& texWidth, int& texHeight, int& texChannels, int& mipLevels)
	{
		stbi_hdr_to_ldr_scale(2.2f);
		stbi_uc* pixels = stbi_load(filename.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
		stbi_hdr_to_ldr_scale(1.0f);
		mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;
		if (!pixels) {
			throw std::runtime_error("failed to load texture image!");
			assert(true);
		}
		return pixels;
	}

	/** 从HDR图片文件中读取贴像素信息*/
	static float* readTextureHDRAsset(const std::string& filename, int& texWidth, int& texHeight, int& texChannels, int& mipLevels)
	{
		float* pixels = stbi_loadf(filename.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
		mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;
		if (!pixels) {
			throw std::runtime_error("failed to load texture image!");
			assert(true);
		}
		return pixels;
	}

	/** 从模型文件中读取贴顶点信息*/
	static void readModelAsset(const std::string& filename, std::vector<FVertex>& vertices, std::vector<uint32_t>& indices)
	{
		tinyobj::attrib_t attrib;
		std::vector<tinyobj::shape_t> shapes;
		std::vector<tinyobj::material_t> materials;
		std::string warn, err;

		if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename.c_str())) {
			throw std::runtime_error(warn + err);
			assert(true);
		}

		std::unordered_map<FVertex, uint32_t> uniqueVertices{};

		for (const auto& shape : shapes) {
			for (const auto& index : shape.mesh.indices) {
				FVertex vertex{};

				vertex.position = {
					attrib.vertices[3 * index.vertex_index + 0],
					attrib.vertices[3 * index.vertex_index + 1],
					attrib.vertices[3 * index.vertex_index + 2]
				};

				vertex.normal = {
					attrib.normals[3 * index.vertex_index + 0],
					attrib.normals[3 * index.vertex_index + 1],
					attrib.normals[3 * index.vertex_index + 2]
				};

				vertex.color = { 1.0f, 1.0f, 1.0f };

				vertex.texCoord = {
					attrib.texcoords[2 * index.texcoord_index + 0],
					1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
				};

				if (uniqueVertices.count(vertex) == 0) {
					uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
					vertices.push_back(vertex);
				}

				indices.push_back(uniqueVertices[vertex]);
			}
		}
	}

	/** 选择打印Debug信息的内容*/
	void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo)
	{
		createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		createInfo.pfnUserCallback = debugCallback;
	}

	/** 检查是否支持合法性检测*/
	bool checkValidationLayerSupport()
	{
		uint32_t layerCount;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

		std::vector<VkLayerProperties> availableLayers(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

		for (const char* layerName : validationLayers)
		{
			bool layerFound = false;

			for (const auto& layerProperties : availableLayers)
			{
				if (strcmp(layerName, layerProperties.layerName) == 0)
				{
					layerFound = true;
					break;
				}
			}

			if (!layerFound)
			{
				return false;
			}
		}

		return true;
	}

	/** 打印调试信息时的回调函数，可以用来处理调试信息*/
	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData)
	{
		std::cerr << "[LOG]: " << pCallbackData->pMessage << std::endl;

		return VK_FALSE;
	}
};


/** 主函数*/
int main()
{
	FVulkanRendererApp app;

	try {
		app.mainTask();
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
