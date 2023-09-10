#version 450

layout(set = 0, binding = 1) uniform sampler2D sampler1;
layout(set = 0, binding = 2) uniform sampler2D sampler2;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

void main() 
{
    vec3 c = texture(sampler1, fragTexCoord).rgb;
    vec3 o = texture(sampler2, fragTexCoord).rgb;
    outColor = vec4(fragColor * (c * o * 2.0), 1.0);
}
