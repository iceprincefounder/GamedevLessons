#version 450

// push constants block
layout( push_constant ) uniform constants
{
	float time;
	float roughness;
	float metallic;
	uint specConstants;
	uint specConstantsCount;
} global;

layout(set = 0, binding = 0) uniform uniformbuffer
{
	mat4 model;
	mat4 view;
	mat4 proj;
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;
layout(location = 3) in vec2 inTexCoord;

void main()
{
	vec3 position = inPosition;
	gl_Position = ubo.proj * ubo.view * ubo.model * vec4(position, 1.0);
}