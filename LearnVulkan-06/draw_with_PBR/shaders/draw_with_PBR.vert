#version 450

// push constants block
layout( push_constant ) uniform constants
{
	float time;
} global;

layout(set = 0, binding = 0) uniform UniformBufferObject
{
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;
layout(location = 3) in vec2 inTexCoord;

layout(location = 0) out vec3 fragPosition;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragColor;
layout(location = 3) out vec2 fragTexCoord;

void main()
{
    //vec3 newPosition = vec3(inPosition.x, inPosition.y, inPosition.z + sin(global.time) * 0.25);
    // Render object with MVP
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);
    fragPosition = (ubo.model * vec4(inPosition, 1.0)).rgb;
    fragNormal = (ubo.model * vec4(normalize(inNormal), 1.0)).rgb;
    fragColor = inColor;
    fragTexCoord = inTexCoord;
}
