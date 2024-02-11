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

// Instanced attributes
layout (location = 4) in vec3 inInstancePosition;
layout (location = 5) in vec3 inInstanceRotation;
layout (location = 6) in float inInstancePScale;
layout (location = 7) in uint inInstanceTexIndex;

mat4 MakeRotMatrix(vec3 R)
{
	mat4 mx, my, mz;
	// rotate around x
	float s = sin(R.x);
	float c = cos(R.x);
	mx[0] = vec4(c, 0.0, s, 0.0);
	mx[1] = vec4(0.0, 1.0, 0.0, 0.0);
	mx[2] = vec4(-s, 0.0, c, 0.0);
	mx[3] = vec4(0.0, 0.0, 0.0, 1.0);	
	// rotate around y
	s = sin(R.y);
	c = cos(R.y);
	my[0] = vec4(c, s, 0.0, 0.0);
	my[1] = vec4(-s, c, 0.0, 0.0);
	my[2] = vec4(0.0, 0.0, 1.0, 0.0);
	my[3] = vec4(0.0, 0.0, 0.0, 1.0);
	// rot around z
	s = sin(R.z);
	c = cos(R.z);
	mz[0] = vec4(1.0, 0.0, 0.0, 0.0);
	mz[1] = vec4(0.0, c, s, 0.0);
	mz[2] = vec4(0.0, -s, c, 0.0);
	mz[3] = vec4(0.0, 0.0, 0.0, 1.0);

	mat4 rotMat = mz * my * mx;
	return rotMat;
}

void main()
{
	mat4 rotMat = MakeRotMatrix(inInstanceRotation);
	vec3 position = (inPosition * inInstancePScale) * mat3(rotMat) + inInstancePosition;
	gl_Position = ubo.proj * ubo.view * ubo.model * vec4(position, 1.0);
}