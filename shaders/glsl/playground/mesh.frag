#version 450

layout (binding = 1) uniform sampler2D tex;
struct VoxelsNode {
    uint miaou;
};

layout(binding = 2) buffer readonly VoxelsTree {
    VoxelsNode nodes[];
};

layout (location = 0) in vec2 inUV;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec3 inColor;
layout (location = 3) in vec3 inEyePos;
layout (location = 4) in vec3 inLightVec;
layout (location = 5) in vec3 fragModelPos;

layout (location = 0) out vec4 outFragColor;

float specpart(vec3 L, vec3 N, vec3 H)
{
	if (dot(N, L) > 0.0)
	{
		return pow(clamp(dot(H, N), 0.0, 1.0), 64.0);
	}
	return 0.0;
}

#define DIM 100
#define HALF DIM/2
#define GAIN 50

void main() 
{
	vec3 Eye = normalize(-inEyePos);
	vec3 Reflected = normalize(reflect(-inLightVec, inNormal)); 

	vec3 halfVec = normalize(inLightVec + inEyePos);
	float diff = clamp(dot(inLightVec, inNormal), 0.0, 1.0);
	float spec = specpart(inLightVec, inNormal, halfVec);
	float intensity = 0.1 + diff + spec;
 
	vec4 IAmbient = vec4(0.2, 0.2, 0.2, 1.0);
	vec4 IDiffuse = vec4(0.5, 0.5, 0.5, 0.5) * max(dot(inNormal, inLightVec), 0.0);
	float shininess = 0.75;
	vec4 ISpecular = vec4(0.5, 0.5, 0.5, 1.0) * pow(max(dot(Reflected, Eye), 0.0), 2.0) * shininess; 

	outFragColor = vec4((IAmbient + IDiffuse) * vec4(inColor, 1.0) + ISpecular);
 
	// Some manual saturation
	if (intensity > 0.95)
		outFragColor *= 2.25;
	if (intensity < 0.15)
		outFragColor = vec4(0.1);
	VoxelsNode node = nodes[1];
	outFragColor = vec4(intensity);
	//outFragColor += node.miaou/255.0;
	ivec3 i = ivec3(fragModelPos*GAIN)+HALF;
	int idx = i.x+DIM*i.y+DIM*DIM*i.z;
	outFragColor.xyz *= nodes[idx].miaou != 0 ? 1.0 : 0.3;
	if(nodes[idx].miaou == 0){
		//discard;
	}
	if(i.x < 0 || i.x > DIM) outFragColor.xyz = vec3(1.0, 0.0, 0.0);
	if(i.y < 0 || i.y > DIM) outFragColor.xyz = vec3(0.0, 1.0, 0.0);
	if(i.z < 0 || i.z > DIM) outFragColor.xyz = vec3(0.0, 0.0, 1.0);
}