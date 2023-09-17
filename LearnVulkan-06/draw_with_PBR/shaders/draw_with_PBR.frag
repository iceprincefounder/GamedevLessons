#version 450

// push constants block
layout( push_constant ) uniform constants
{
	float time;
} global;

struct Light
{
	vec4 position;  // position.w represents type of light
	vec4 color;     // color.w represents light intensity
	vec4 direction; // direction.w represents range
	vec4 info;      // (only used for spot lights) info.x represents light inner cone angle, info.y represents light outer cone angle
};

layout(set = 0, binding = 1) uniform UniformBufferObjectView
{
	Light directional_lights[4];
	Light point_lights[4];
	Light spot_lights[4];
    ivec4 lights_count; // [0] for directional_lights, [1] for point_lights, [2] for spot_lights
	vec4 camera_position;
} view;

uint DIRECTIONAL_LIGHTS = view.lights_count[0];
uint POINT_LIGHTS = view.lights_count[1];
uint SPOT_LIGHTS = view.lights_count[2];
uint AREA_LIGHTS = view.lights_count[3];

layout(set = 0, binding = 2) uniform sampler2D sampler1; // c
layout(set = 0, binding = 3) uniform sampler2D sampler2; // m
layout(set = 0, binding = 4) uniform sampler2D sampler3; // r
layout(set = 0, binding = 5) uniform sampler2D sampler4; // n
layout(set = 0, binding = 6) uniform sampler2D sampler5; // o

layout(location = 0) in vec3 fragPosition;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragColor;
layout(location = 3) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;


const float PI = 3.14159265359;

vec3 F0 = vec3(0.04);

// [0] Frensel Schlick
vec3 F_Schlick(vec3 f0, float f90, float u)
{
	return f0 + (f90 - f0) * pow(1.0 - u, 5.0);
}

// [1] IBL Defuse Irradiance
vec3 F_Schlick_Roughness(vec3 F0, float cos_theta, float roughness)
{
	return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(1.0 - cos_theta, 5.0);
}

// [0] Diffuse Term
float Fr_DisneyDiffuse(float NdotV, float NdotL, float LdotH, float roughness)
{
	float E_bias        = 0.0 * (1.0 - roughness) + 0.5 * roughness;
	float E_factor      = 1.0 * (1.0 - roughness) + (1.0 / 1.51) * roughness;
	float fd90          = E_bias + 2.0 * LdotH * LdotH * roughness;
	vec3  f0            = vec3(1.0);
	float light_scatter = F_Schlick(f0, fd90, NdotL).r;
	float view_scatter  = F_Schlick(f0, fd90, NdotV).r;
	return light_scatter * view_scatter * E_factor;
}

// [0] Specular Microfacet Model
float V_SmithGGXCorrelated(float NdotV, float NdotL, float roughness)
{
	float alphaRoughnessSq = roughness * roughness;

	float GGXV = NdotL * sqrt(NdotV * NdotV * (1.0 - alphaRoughnessSq) + alphaRoughnessSq);
	float GGXL = NdotV * sqrt(NdotL * NdotL * (1.0 - alphaRoughnessSq) + alphaRoughnessSq);

	float GGX = GGXV + GGXL;
	if (GGX > 0.0)
	{
		return 0.5 / GGX;
	}
	return 0.0;
}

// [0] GGX Normal Distribution Function
float D_GGX(float NdotH, float roughness)
{
	float alphaRoughnessSq = roughness * roughness;
	float f                = (NdotH * alphaRoughnessSq - NdotH) * NdotH + 1.0;
	return alphaRoughnessSq / (PI * f * f);
}

float saturate(float t)
{
	return clamp(t, 0.0, 1.0);
}

vec3 saturate(vec3 t)
{
	return clamp(t, 0.0, 1.0);
}

vec3 calcNormal()
{
    vec3 pos_dx = dFdx(fragPosition);
    vec3 pos_dy = dFdy(fragPosition);
    vec3 st1    = dFdx(vec3(fragTexCoord, 0.0));
    vec3 st2    = dFdy(vec3(fragTexCoord, 0.0));
    vec3 T      = (st2.t * pos_dx - st1.t * pos_dy) / (st1.s * st2.t - st2.s * st1.t);
    vec3 N      = normalize(fragNormal);
    T           = normalize(T - N * dot(N, T));
    vec3 B      = normalize(cross(N, T));
    mat3 TBN    = mat3(T, B, N);
    
    return normalize(TBN[2].xyz);
}

vec3 calcNormal(vec3 n)
{
    vec3 pos_dx = dFdx(fragPosition);
    vec3 pos_dy = dFdy(fragPosition);
    vec3 st1    = dFdx(vec3(fragTexCoord, 0.0));
    vec3 st2    = dFdy(vec3(fragTexCoord, 0.0));
    vec3 T      = (st2.t * pos_dx - st1.t * pos_dy) / (st1.s * st2.t - st2.s * st1.t);
    vec3 N      = normalize(fragNormal);
    T           = normalize(T - N * dot(N, T));
    vec3 B      = normalize(cross(N, T));
    mat3 TBN    = mat3(T, B, N);

    return normalize(TBN * (2.0 * n - 1.0));
}

vec3 get_directional_light_direction(uint index)
{
	return -view.directional_lights[index].direction.xyz;
}

vec3 apply_directional_light(uint index, vec3 normal)
{
	vec3 world_to_light = -view.directional_lights[index].direction.xyz;

	world_to_light = normalize(world_to_light);

	float ndotl = clamp(dot(normal, world_to_light), 0.0, 1.0);

    return ndotl * view.directional_lights[index].color.w * view.directional_lights[index].color.rgb;
}

void main()
{
    vec3 base_color = vec3(0.01);
    float metallic = 1.0;
    float roughness = 0.1;
    vec3 normal = calcNormal();
    vec3 ambient_occlution = vec3(1.0);

    //vec3 base_color = texture(sampler1, fragTexCoord).rgb;
    //float metallic = saturate(texture(sampler2, fragTexCoord).r);
    //float roughness = saturate(texture(sampler3, fragTexCoord).r);
    //vec3 normal = calcNormal(texture(sampler4, fragTexCoord).rgb);
    //vec3 ambient_occlution = texture(sampler5, fragTexCoord).rgb;

	vec3 N = normal;
	vec3 V = normalize(view.camera_position.xyz - fragPosition);
	float NdotV = saturate(dot(N, V));

	vec3 LightContribution = vec3(0.0);
	//vec3 diffuse_color = base_color.rgb * (1.0 - metallic);
    vec3 diffuse_color = base_color.rgb / PI;

    for (uint i = 0U; i < DIRECTIONAL_LIGHTS; ++i)
    {
        vec3 L = get_directional_light_direction(i);
        vec3 H = normalize(V + L);

        float LdotH = saturate(dot(L, H));
        float NdotH = saturate(dot(N, H));
        float NdotL = saturate(dot(N, L));

        float F90 = saturate(50.0 * F0.r);
        vec3  F   = F_Schlick(F0, F90, LdotH);
        float Vis = V_SmithGGXCorrelated(NdotV, NdotL, roughness);
        float D   = D_GGX(NdotH, roughness);
        vec3  Fr  = F * D * Vis;

        float Fd = Fr_DisneyDiffuse(NdotV, NdotL, LdotH, roughness);

        LightContribution += apply_directional_light(i, N) * (diffuse_color * (vec3(1.0) - F) * Fd + Fr);
    }

    // [1] Tempory irradiance to fix dark metals
	// TODO: add specular irradiance for realistic metals
	vec3 irradiance  = vec3(0.5);
	vec3 F           = F_Schlick_Roughness(F0, max(dot(N, V), 0.0), roughness * roughness * roughness * roughness);
	vec3 ibl_diffuse = irradiance * base_color.rgb;

	vec3 ambient_color = ibl_diffuse;

	outColor = vec4(0.3 * ambient_color + LightContribution, 1.0);
}
