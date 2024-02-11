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

// Vertex attributes
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;
layout(location = 3) in vec2 inTexCoord;

// Instanced attributes
layout (location = 4) in vec3 inInstancePosition;
layout (location = 5) in vec3 inInstanceRotation;
layout (location = 6) in float inInstancePScale;
layout (location = 7) in uint inInstanceTexIndex;

layout(location = 0) out vec3 outPosition;
layout(location = 1) out vec3 outPositionWS;
layout(location = 2) out vec3 outNormal;
layout(location = 3) out vec3 outColor;
layout(location = 4) out vec2 outTexCoord;

// https://www.ronja-tutorials.com/post/041-hsv-colorspace/
vec3 Hue2RGB(float hue) {
    hue = fract(hue); //only use fractional part of hue, making it loop
    float r = abs(hue * 6 - 3) - 1; //red
    float g = 2 - abs(hue * 6 - 2); //green
    float b = 2 - abs(hue * 6 - 4); //blue
    vec3 rgb = vec3(r,g,b); //combine components
    rgb = clamp(rgb,0.0,1.0); //clamp between 0 and 1
    return rgb;
}

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
	outPosition = position;
	outPositionWS = (ubo.model * vec4(position, 1.0)).rgb;
	outNormal = (ubo.model * vec4(normalize(inNormal), 1.0)).rgb * mat3(rotMat);
	outColor = Hue2RGB(inInstanceTexIndex / 256.0f);
	outTexCoord = inTexCoord;
}