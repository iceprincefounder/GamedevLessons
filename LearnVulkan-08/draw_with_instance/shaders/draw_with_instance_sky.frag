#version 450

layout(binding = 4) uniform sampler2D skydomeSampler;

layout(location = 0) in vec3 fragPosition;
layout(location = 1) in vec3 fragPositionWS;
layout(location = 2) in vec3 fragNormal;
layout(location = 3) in vec3 fragColor;
layout(location = 4) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

void main() {
	vec3 color = texture(skydomeSampler, fragTexCoord).rgb;

	// Gamma correct
	color = pow(color, vec3(0.4545));

	outColor = vec4(color, 1.0);
}