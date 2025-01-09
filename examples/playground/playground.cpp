/*
* Vulkan Demo Scene
*
* Don't take this a an example, it's more of a personal playground
*
* Copyright (C) 2016-2023 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/


/*
# Physics
Accurate voxel needed on : 
- volume collision test
- connectivity test

damaged pixel walk :
- Raycast : mesh in/out -> damanged pixel walk


volume collision :
- Generate contact points using regular convex volume
- Refine contact point in damaged voxel
  - ignore refinment for hich velocities
  - x8 voxel scale
*/

#include "vulkanexamplebase.h"
#include "VulkanglTFModel.h"

#define VOID 0x00000000
#define LEAF 0xC0000000
#define BRAN 0x80000000

namespace Octree{
	struct Layer{
		Layer* cells[8];
	};
	const auto ZERO = (Layer*)0;
	const auto ONE = (Layer*)1;
	inline bool isLeaf(void* cell) {
		return cell <= ONE;
	}

/*
	void set(Layer *root, int scale, glm::ivec3 pos){
		auto ptr = (void*)root;
		while(scale > 0){
			int idx = (((pos.x >> scale) & 1) << 0) | (((pos.y >> scale) & 1) << 1) | (((pos.z >> scale) & 1) << 2);
			auto cell = static_cast<Layer*>(ptr)->cells[idx];
			if(isLeaf(cell)){
				cell = new Layer();
				static_cast<Layer*>(ptr)->cells[idx] = cell;
			}
			scale--;
		}

		int idx = (((pos.x >> 0) & 1) << 0) | (((pos.y >> 0) & 1) << 1) | (((pos.z >> 0) & 1) << 2);
		static_cast<Layer*>(ptr)->cells[idx] = ONE;
	}


	void set(Layer **root, int scale, glm::ivec3 pos){
		auto ptr = (void**)root;
		while(scale > 0){
			auto cell = *ptr;
			if(isLeaf(cell)){
				cell = new Layer();
				*ptr = cell;
			}
			int idx = (((pos.x >> scale) & 1) << 0) | (((pos.y >> scale) & 1) << 1) | (((pos.z >> scale) & 1) << 2);
			ptr = (&((static_cast<Layer*>(cell))->cells[idx]));

			scale--;
		}

		*ptr = ONE;
	}*/
	void set(Layer **root, int scale, glm::ivec3 pos){
		auto ptr = root;
		while(scale >= 0){
			auto cell = *ptr;
			if(isLeaf(cell)){
				cell = new Layer();
				*ptr = cell;
			}
			int idx = (((pos.x >> scale) & 1) << 0) | (((pos.y >> scale) & 1) << 1) | (((pos.z >> scale) & 1) << 2);
			ptr = (&(((cell))->cells[idx]));

			scale--;
		}

		*ptr = ONE;
	}

	
	void symplify(Layer **cell){
		auto value = *cell;
		if(isLeaf(value)) return;

		bool isOne = true, isZero = true;
		for(int i = 0;i < 8;i++){
			symplify(&value->cells[i]);
			auto newVal = value->cells[i];
			isOne &= newVal == ONE;
			isZero &= newVal == ZERO;
		}

		if(isOne) {
			delete *cell;
			*cell = ONE;
		}
		else if(isZero) {
			delete *cell;
			*cell = ZERO;
		}
	}

	int allocPtr = 0;
	void uploadRec(Layer *layer, std::vector<uint32_t> &ssboData, int ptr){
		if(isLeaf(layer)){
			ssboData[ptr] = layer == ONE ? VOID : LEAF;
			return;
		}
		int allocated = allocPtr;
		allocPtr+=8;
		ssboData[ptr] = BRAN | allocated;
		for(int i = 0;i < 8;i++){
			uploadRec(layer->cells[i], ssboData, allocated + i);
		}
	}
	void upload(Layer *root, std::vector<uint32_t> &ssboData){
		int allocated = allocPtr;
		allocPtr+=8;
		for(int i = 0;i < 8;i++){
			uploadRec(root->cells[i], ssboData, allocated + i);
		}
	}

	
};




struct VoxelGrid{
	glm::ivec3 dim;
	glm::u8 *data;
	void init(glm::ivec3 dim){
		this->dim = dim;
		//if(data) delete [] data;
		//data = new glm::u8[dim[0]*dim[1]*dim[2]];
	}
	~VoxelGrid(){
		//delete [] data;
	}
};

class VulkanExample : public VulkanExampleBase
{
public:
	struct DemoModel {
		vkglTF::Model* glTF;
		VkPipeline *pipeline;
		VoxelGrid vox;
	};
	std::vector<DemoModel> demoModels;
	vks::TextureCubeMap skybox;


	VkBuffer ssboBuffer;
	VkDeviceMemory ssboMemory;

	struct UniformData {
		glm::mat4 projection;
		glm::mat4 model;
		glm::mat4 normal;
		glm::mat4 view;
		glm::vec4 lightPos;
		glm::mat4 modelInv;
	} uniformData;
	vks::Buffer uniformBuffer;

	struct {
		VkPipeline logos{ VK_NULL_HANDLE };
		VkPipeline models{ VK_NULL_HANDLE };
		VkPipeline skybox{ VK_NULL_HANDLE };
	} pipelines;

	VkPipelineLayout pipelineLayout{ VK_NULL_HANDLE };
	VkDescriptorSet descriptorSet{ VK_NULL_HANDLE };
	VkDescriptorSetLayout descriptorSetLayout{ VK_NULL_HANDLE };

	glm::vec4 lightPos = glm::vec4(1.0f, 4.0f, 0.0f, 0.0f);
	//glm::vec4 lightPos = glm::vec4(-4.0f, -4.0f, -4.0f, 0.0f);

	VulkanExample() : VulkanExampleBase()
	{
		title = "Vulkan Demo Scene (c) by Sascha Willems";
		camera.type = Camera::CameraType::lookat;
		//camera.flipY = true;
		camera.setPosition(glm::vec3(0.0f, 0.0f, -3.75f));
		camera.setRotation(glm::vec3(0.0f, 0.0f, 0.0f));
		camera.setRotationSpeed(0.5f);
		camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 256.0f);
	}

	~VulkanExample()
	{
		if (device) {
			vkDestroyPipeline(device, pipelines.logos, nullptr);
			vkDestroyPipeline(device, pipelines.models, nullptr);
			vkDestroyPipeline(device, pipelines.skybox, nullptr);
			vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
			vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
			vkDestroyBuffer(device, ssboBuffer, nullptr);
			vkFreeMemory(device, ssboMemory, nullptr);
			for (auto demoModel : demoModels) {
				delete demoModel.glTF;
			}
			uniformBuffer.destroy();
			skybox.destroy();
		}
	}

	

	void loadAssets()
	{
		// Models
		//std::vector<std::string> modelFiles = { "cube.gltf", "vulkanscenelogos.gltf", "vulkanscenebackground.gltf", "vulkanscenemodels.gltf" };
		//std::vector<VkPipeline*> modelPipelines = { &pipelines.skybox, &pipelines.logos, &pipelines.models, &pipelines.models };
		std::vector<std::string> modelFiles = { "vulkanscenemodels.gltf" };
		//std::vector<std::string> modelFiles = { "cube.gltf" };
		//std::vector<std::string> modelFiles = { "cube_blender.gltf" };

		
		std::vector<VkPipeline*> modelPipelines = { &pipelines.models };
		for (auto i = 0; i < modelFiles.size(); i++) {
			DemoModel model;
			const uint32_t glTFLoadingFlags = vkglTF::FileLoadingFlags::PreTransformVertices | vkglTF::FileLoadingFlags::PreMultiplyVertexColors | vkglTF::FileLoadingFlags::FlipY;
			model.pipeline = modelPipelines[i];
			model.glTF = new vkglTF::Model();
			model.glTF->loadFromFile(getAssetPath() + "models/" + modelFiles[i], vulkanDevice, queue, glTFLoadingFlags);

			model.vox.init(glm::ivec3(10000, 0, 0));
			for(int i = 0;i++;i < model.vox.dim[0]) model.vox.data[i] = 0x40;

			demoModels.push_back(model);
		}
		// Textures
		skybox.loadFromFile(getAssetPath() + "textures/cubemap_vulkan.ktx", VK_FORMAT_R8G8B8A8_UNORM, vulkanDevice, queue);
	}

	void setupDescriptors()
	{
		// Pool
		std::vector<VkDescriptorPoolSize> poolSizes = {
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2),
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1),
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1)
		};
		VkDescriptorPoolCreateInfo descriptorPoolInfo = vks::initializers::descriptorPoolCreateInfo(poolSizes, 2);
		VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));

		// Layout
		std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
			// Binding 0 : Vertex shader uniform buffer
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0),
			// Binding 1 : Fragment shader color map image sampler
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1),
			// Binding 2 : Fragment voxels
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 2)
		};
		VkDescriptorSetLayoutCreateInfo descriptorLayout = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &descriptorSetLayout));

		// Set
		VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayout, 1);
		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet));

		VkDescriptorBufferInfo miaou{ ssboBuffer, 0, VK_WHOLE_SIZE };

		std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
			// Binding 0 : Vertex shader uniform buffer
			vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &uniformBuffer.descriptor),
			// Binding 1 : Fragment shader image sampler
			vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, &skybox.descriptor),
			// Binding 2 : Fragment voxels
			vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2, &miaou),
		};
		vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
	}

	void preparePipelines()
	{
		// Layout
		VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayout, 1);
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout));

		// Pipelines
		VkPipelineInputAssemblyStateCreateInfo inputAssemblyState = vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
		VkPipelineRasterizationStateCreateInfo rasterizationState = vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE,0);
		VkPipelineColorBlendAttachmentState blendAttachmentState = vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
		VkPipelineColorBlendStateCreateInfo colorBlendState = vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);
		VkPipelineDepthStencilStateCreateInfo depthStencilState = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL);
		VkPipelineViewportStateCreateInfo viewportState = vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);
		VkPipelineMultisampleStateCreateInfo multisampleState = vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT, 0);
		std::vector<VkDynamicState> dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
		VkPipelineDynamicStateCreateInfo dynamicState = vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);
		std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages;
		VkGraphicsPipelineCreateInfo pipelineCI = vks::initializers::pipelineCreateInfo(pipelineLayout, renderPass, 0);
		pipelineCI.pInputAssemblyState = &inputAssemblyState;
		pipelineCI.pRasterizationState = &rasterizationState;
		pipelineCI.pColorBlendState = &colorBlendState;
		pipelineCI.pMultisampleState = &multisampleState;
		pipelineCI.pViewportState = &viewportState;
		pipelineCI.pDepthStencilState = &depthStencilState;
		pipelineCI.pDynamicState = &dynamicState;
		pipelineCI.stageCount = static_cast<uint32_t>(shaderStages.size());
		pipelineCI.pStages = shaderStages.data();
		pipelineCI.pVertexInputState = vkglTF::Vertex::getPipelineVertexInputState({ vkglTF::VertexComponent::Position, vkglTF::VertexComponent::Normal, vkglTF::VertexComponent::UV, vkglTF::VertexComponent::Color });;

		// Default mesh rendering pipeline
		shaderStages[0] = loadShader(getShadersPath() + "playground/mesh.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
		shaderStages[1] = loadShader(getShadersPath() + "playground/mesh.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipelines.models));

		// Pipeline for the logos
		shaderStages[0] = loadShader(getShadersPath() + "playground/logo.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
		shaderStages[1] = loadShader(getShadersPath() + "playground/logo.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipelines.logos));

		// Pipeline for the skybox
		rasterizationState.cullMode = VK_CULL_MODE_FRONT_BIT;
		depthStencilState.depthWriteEnable = VK_FALSE;
		shaderStages[0] = loadShader(getShadersPath() + "playground/skybox.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
		shaderStages[1] = loadShader(getShadersPath() + "playground/skybox.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipelines.skybox));
	}

	// Prepare and initialize uniform buffer containing shader uniforms
	void prepareUniformBuffers()
	{
		vulkanDevice->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,&uniformBuffer, sizeof(uniformData));
		VK_CHECK_RESULT(uniformBuffer.map());
	}

	void updateUniformBuffers()
	{
		uniformData.projection = camera.matrices.perspective;
		uniformData.view = camera.matrices.view;
		uniformData.model = glm::mat4(1.0f);
		//uniformData.model = glm::scale(uniformData.model, glm::vec3(0.25f, 0.25f, 0.25f));
		uniformData.normal = glm::inverseTranspose(uniformData.view * uniformData.model);
		uniformData.lightPos = lightPos;
		uniformData.modelInv = inverse(uniformData.model) * inverse(uniformData.view);
		memcpy(uniformBuffer.mapped, &uniformData, sizeof(uniformData));
	}

	void buildCommandBuffers()
	{
		VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		VkClearValue clearValues[2];
		clearValues[0].color = defaultClearColor;
		clearValues[1].depthStencil = { 1.0f, 0 };

		VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
		renderPassBeginInfo.renderPass = renderPass;
		renderPassBeginInfo.renderArea.offset.x = 0;
		renderPassBeginInfo.renderArea.offset.y = 0;
		renderPassBeginInfo.renderArea.extent.width = width;
		renderPassBeginInfo.renderArea.extent.height = height;
		renderPassBeginInfo.clearValueCount = 2;
		renderPassBeginInfo.pClearValues = clearValues;

		for (int32_t i = 0; i < drawCmdBuffers.size(); ++i)
		{
			renderPassBeginInfo.framebuffer = frameBuffers[i];

			VK_CHECK_RESULT(vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo));

			vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

			VkViewport viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
			vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);

			VkRect2D scissor = vks::initializers::rect2D(width, height, 0, 0);
			vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

			vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

			for (auto &model : demoModels) {
				vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, *model.pipeline);
				model.glTF->draw(drawCmdBuffers[i]);
			}

			drawUI(drawCmdBuffers[i]);

			vkCmdEndRenderPass(drawCmdBuffers[i]);

			VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
		}
	}

	void draw()
	{
		VulkanExampleBase::prepareFrame();
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];
		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
		VulkanExampleBase::submitFrame();
	}

	void prepareVoxelBuffer(){
		// Define buffer creation info
		VkBufferCreateInfo bufferCreateInfo = {};
		bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		#define DIM 512
		bufferCreateInfo.size = DIM*DIM*DIM*4;  // Size of the buffer in bytes
		bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; // Usage as a storage buffer
		bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // No sharing between queues

		// Create the buffer
		vkCreateBuffer(device, &bufferCreateInfo, nullptr, &ssboBuffer);

		// Allocate memory for the buffer
		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(device, ssboBuffer, &memRequirements);

		// Define memory allocation info
		VkMemoryAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = vulkanDevice->getMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

		// Allocate memory
		vkAllocateMemory(device, &allocInfo, nullptr, &ssboMemory);

		// Bind the memory to the buffer
		vkBindBufferMemory(device, ssboBuffer, ssboMemory, 0);


		//std::vector<uint32_t> ssboData = {255, 128, 0, 4, 5, 6, 7, 8};
		std::vector<uint32_t> ssboData(DIM*DIM*DIM);
		for(int z = 0;z < DIM;z++){
			for(int y = 0;y < DIM;y++){
				for(int x = 0;x < DIM;x++){
					ssboData[x+DIM*y+DIM*DIM*z] = (x-DIM/2)*(x-DIM/2)+(y-DIM/2)*(y-DIM/2) > (DIM/4)*(DIM/4) ? 0 : 1;
				}
			}
		}


		int n = 0;
		for(int i = 0;i < 8;i++) ssboData[n+i] = VOID + n;
		ssboData[n+0] = BRAN + n + 8;
		n+=8;
		for(int i = 0;i < 8;i++) ssboData[n+i] = LEAF;


		n = 0;
		for(int i = 0;i < 8;i++) ssboData[n+i] = BRAN + n + 8;
		//n+=8;
		for(int i = 0;i < 8;i++) ssboData[n+i] = VOID;
		n+=8;
	

		ssboData[0] = BRAN + n;
		std::vector<int> miaou = {7,4,5,6};
		for(int p : miaou){
			if(n != 0) for(int i = 0;i < 8;i++) ssboData[n+i] = rand() < RAND_MAX/3 ? LEAF : VOID;
			ssboData[n+p] = BRAN + n + 8;
			n+=8;
		}
		for(int i = 0;i < 8;i++) ssboData[n+i] = LEAF;
			n+=8;

		ssboData[1] = BRAN + n;
		miaou = {7,7,7,7};
		for(int p : miaou){
			if(n != 0) for(int i = 0;i < 8;i++) ssboData[n+i] = rand() < RAND_MAX/3 ? LEAF : VOID;
			ssboData[n+p] = BRAN + n + 8;
			n+=8;
		}
		for(int i = 0;i < 8;i++) ssboData[n+i] = LEAF;
		n+=8;

		ssboData[2] = BRAN + n;
		miaou = {7,0,7,0};
		for(int p : miaou){
			if(n != 0) for(int i = 0;i < 8;i++) ssboData[n+i] = rand() < RAND_MAX/5 ? LEAF : VOID;
			ssboData[n+p] = BRAN + n + 8;
			n+=8;
		}
		for(int i = 0;i < 8;i++) ssboData[n+i] = LEAF;
		n+=8;

		ssboData[3] = BRAN + n;
		miaou = {0,7,0,7};
		for(int p : miaou){
			if(n != 0) for(int i = 0;i < 8;i++) ssboData[n+i] = rand() < RAND_MAX/3 ? LEAF : VOID;
			ssboData[n+p] = BRAN + n + 8;
			n+=8;
		}
		for(int i = 0;i < 8;i++) ssboData[n+i] = LEAF;
		n+=8;













		n = 0;
		for(int i = 0;i < 8;i++) ssboData[n+i] = LEAF;
		n+=8;
	

		ssboData[0] = BRAN + n;
		miaou = {7,4,5,6};
		for(int p : miaou){
			if(n != 0) for(int i = 0;i < 8;i++) ssboData[n+i] = rand() < RAND_MAX/3 ? LEAF : VOID;
			ssboData[n+p] = BRAN + n + 8;
			n+=8;
		}
		for(int i = 0;i < 8;i++) ssboData[n+i] = LEAF;
			n+=8;

		ssboData[1] = BRAN + n;
		miaou = {7,7,7,7};
		for(int p : miaou){
			if(n != 0) for(int i = 0;i < 8;i++) ssboData[n+i] = rand() < RAND_MAX/3 ? LEAF : VOID;
			ssboData[n+p] = BRAN + n + 8;
			n+=8;
		}
		for(int i = 0;i < 8;i++) ssboData[n+i] = LEAF;
		n+=8;

		ssboData[2] = BRAN + n;
		miaou = {7,0,7,0};
		for(int p : miaou){
			if(n != 0) for(int i = 0;i < 8;i++) ssboData[n+i] = rand() < RAND_MAX/5 ? LEAF : VOID;
			ssboData[n+p] = BRAN + n + 8;
			n+=8;
		}
		for(int i = 0;i < 8;i++) ssboData[n+i] = LEAF;
		n+=8;

		ssboData[3] = BRAN + n;
		miaou = {0,7,0,7,6,5,4,2,1,0};
		for(int p : miaou){
			if(n != 0) for(int i = 0;i < 8;i++) ssboData[n+i] = rand() < RAND_MAX/3 ? LEAF : VOID;
			ssboData[n+p] = BRAN + n + 8;
			n+=8;
		}
		for(int i = 0;i < 8;i++) ssboData[n+i] = LEAF;
		n+=8;



		Octree::Layer *root = (Octree::Layer*)Octree::ZERO;
		/*for(int i = 0;i < 10000;i++){
			Octree::set(&root, rand()%3+5, glm::ivec3(rand(),rand(),rand()));
		}*/
	/*	Octree::set(&root, 0, glm::ivec3(0,0,0));
		Octree::set(&root, 0, glm::ivec3(1,1,1));
		Octree::set(&root, 2, glm::ivec3(4,2,2));*/
		int depth = 5; //1199928/2097152
		int center = (2 << depth)/2;
		for(int z = 0;z < 2 << depth;z++){
			for(int y = 0;y < 2 << depth;y++){
				for(int x = 0;x < 2 << depth;x++){
					//if(x+y+z > 1 << depth){
					if((x-center)*(x-center)+(y-center)*(y-center)+(z-center)*(z-center) > center*center){
						Octree::set(&root, depth, glm::ivec3(x,y,z));
					}
				}		
			}	
		}

		/*for(int z = 0;z < 2;z++){
			for(int y = 0;y < 2;y++){
				for(int x = 0;x < 2;x++){
					Octree::set(&root, 1, glm::ivec3(x,y,z));
				}		
			}	
		}*/
		Octree::symplify(&root);
		Octree::upload(root, ssboData);
		std::cout << Octree::allocPtr << " / " << center*center*center*8/Octree::allocPtr << std::endl;



/*
		miaou = {2,7,4};
		for(int p : miaou){
			if(n != 0) for(int i = 0;i < 8;i++) ssboData[n+i] = VOID;
			ssboData[n+p] = BRAN + n + 8;
			n+=8;
		}
		for(int i = 0;i < 8;i++) ssboData[n+i] = LEAF;*/

		//for(int i = 0;i < 8;i++) ssboData[n+i] = BRAN + n + 8;

		//ssboData[n+0] = VOID;
		//for(int i = 0;i < 8;i++) ssboData[n+i] = LEAF;


		//for(int i = 0;i < 8;i++) ssboData[n+i] = VOID;
		//for(int i = 0;i < 8;i++) ssboData[n+i] = LEAF;

		void* mappedMemory;

		// Step 1: Map the buffer memory
		vkMapMemory(device, ssboMemory, 0, ssboData.size() * sizeof(uint32_t), 0, &mappedMemory);

		// Step 2: Copy data to the mapped memory
		memcpy(mappedMemory, ssboData.data(), ssboData.size() * sizeof(uint32_t));

		// Step 3: Flush the memory if not coherent
		VkMappedMemoryRange memoryRange = {};
		memoryRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
		memoryRange.memory = ssboMemory;
		memoryRange.offset = 0;
		memoryRange.size = VK_WHOLE_SIZE;  // Flush the entire memory

		//if (!(memoryProperties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
		//	vkFlushMappedMemoryRanges(device, 1, &memoryRange);
		//}

		// Step 4: Unmap the memory
		vkUnmapMemory(device, ssboMemory);
	}

	void prepare()
	{
		VulkanExampleBase::prepare();
		loadAssets();
		prepareUniformBuffers();
		prepareVoxelBuffer();
		setupDescriptors();
		preparePipelines();
		buildCommandBuffers();
		prepared = true;
	}

	virtual void render()
	{
		if (!prepared)
			return;
		updateUniformBuffers();
		draw();
	}

};

VULKAN_EXAMPLE_MAIN()