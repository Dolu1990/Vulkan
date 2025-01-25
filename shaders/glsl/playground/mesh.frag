#version 450

layout (binding = 1) uniform sampler2D tex;
layout (binding = 2) readonly buffer Voxels {
    uint voxels[];
};

//layout (binding = 1) uniform sampler2D depth;

layout (binding = 0) uniform UBO 
{
    mat4 projection;  // Projection matrix for perspective/orthographic projection
    mat4 model;       // Model matrix to transform object space to world space
    mat4 normal;      // Normal matrix (typically the inverse transpose of the model matrix)
    mat4 view;        // View matrix to transform world space to camera (view) space
    vec3 lightpos;    // Position of the light source in world space
    mat4 modelInv;     
} ubo;

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

//#extension GL_EXT_shader_explicit_arithmetic_types : require

#define STACK_SIZE 23
#define EPS 3.552713678800501e-15

bool raytrace(vec3 start, vec3 dir, out vec3 o_pos, out vec3 o_color, out vec3 o_normal, out uint o_iter) {

    //TODO make sure the ray hits the bounding box of the root octree node

/*
	ivec3 posInt = start * (1 << 24); //TODO
	while(true){
		var idx;
		
	}


	d.x = abs(d.x) > EPS ? d.x : (d.x >= 0 ? EPS : -EPS);
	d.y = abs(d.y) > EPS ? d.y : (d.y >= 0 ? EPS : -EPS);
	d.z = abs(d.z) > EPS ? d.z : (d.z >= 0 ? EPS : -EPS);

	vec3 t_coef = 1.0f / -abs(d);


	uint stack[STACK_SIZE];
	uint iter = 0;
	while(true){

	}*/

	return false;
}




bool Octree_RayMarchLeaf2(vec3 o, vec3 d, out vec3 o_pos, out vec3 o_color, out vec3 o_normal, out uint o_iter, out bool o_direct) {
	uint iter = 0;

	//o *= 0.5;
	o += 1.5;
	/*o_color = vec3(0.0,1.0,1.0);
	if(o.x < 1.0 || o.x >= 2.0) return false;
	if(o.y < 1.0 || o.y >= 2.0) return false;
	if(o.z < 1.0 || o.z >= 2.0) return false;*/
	o_color = vec3(0.0,0.0,0.0);
	o_direct = false;

	d.x = abs(d.x) > EPS ? d.x : (d.x >= 0 ? EPS : -EPS);
	d.y = abs(d.y) > EPS ? d.y : (d.y >= 0 ? EPS : -EPS);
	d.z = abs(d.z) > EPS ? d.z : (d.z >= 0 ? EPS : -EPS);

	// Precompute the coefficients of tx(x), ty(y), and tz(z).
	// The octree is assumed to reside at coordinates [1, 2].
	vec3 t_coef = 1.0f / -abs(d);
	vec3 t_bias = t_coef * o;

	uint oct_mask = 0u;
	if (d.x > 0.0f)
		oct_mask ^= 1u, t_bias.x = 3.0f * t_coef.x - t_bias.x;
	if (d.y > 0.0f)
		oct_mask ^= 2u, t_bias.y = 3.0f * t_coef.y - t_bias.y;
	if (d.z > 0.0f)
		oct_mask ^= 4u, t_bias.z = 3.0f * t_coef.z - t_bias.z;

	// Initialize the active span of t-values.
	float t_min = max(max(2.0f * t_coef.x - t_bias.x, 2.0f * t_coef.y - t_bias.y), 2.0f * t_coef.z - t_bias.z);
	float t_max = min(min(t_coef.x - t_bias.x, t_coef.y - t_bias.y), t_coef.z - t_bias.z);
	t_min = max(t_min, 0.0f);
	float h = t_max;

	uint parent = 0u;
	uint cur = 0u;
	vec3 pos = vec3(1.0f);
	uint idx = 0u;
	if (1.5f * t_coef.x - t_bias.x > t_min)
		idx ^= 1u, pos.x = 1.5f;
	if (1.5f * t_coef.y - t_bias.y > t_min)
		idx ^= 2u, pos.y = 1.5f;
	if (1.5f * t_coef.z - t_bias.z > t_min)
		idx ^= 4u, pos.z = 1.5f;

	uint scale = STACK_SIZE - 1;
	float scale_exp2 = 0.5f; // exp2( scale - STACK_SIZE )
	uint stack[STACK_SIZE];
	bool firstHit = true;
	while (scale < STACK_SIZE) {
		++iter;
		if (cur == 0u)
			cur = voxels[parent + (idx ^ oct_mask)];
		// Determine maximum t-value of the cube by evaluating
		// tx(), ty(), and tz() at its corner.

		vec3 t_corner = pos * t_coef - t_bias;
		float tc_max = min(min(t_corner.x, t_corner.y), t_corner.z);

		if ((cur & 0x80000000u) != 0 && t_min <= t_max) {
			//if(iter == 1) o_color += vec3(0.0,0.3,0.0);
			// INTERSECT
			float half_scale_exp2 = scale_exp2 * 0.5f;
			vec3 t_center = half_scale_exp2 * t_coef + t_corner;

			if ((cur & 0x40000000u) != 0){ // leaf node
				if(firstHit){
					o_color.z += 0.5;
					o_direct = true;
				}
				break;
			}

			// PUSH
			if (tc_max < h)
				stack[scale] = parent;
			h = tc_max;

			parent = cur & 0x3fffffffu;

			idx = 0u;
			--scale;
			scale_exp2 = half_scale_exp2;
			if (t_center.x > t_min)
				idx ^= 1u, pos.x += scale_exp2;
			if (t_center.y > t_min)
				idx ^= 2u, pos.y += scale_exp2;
			if (t_center.z > t_min)
				idx ^= 4u, pos.z += scale_exp2;

			cur = 0;

			continue;
		}

		firstHit = false;


		// ADVANCE
		uint step_mask = 0u;
		if (t_corner.x <= tc_max)
			step_mask ^= 1u, pos.x -= scale_exp2;
		if (t_corner.y <= tc_max)
			step_mask ^= 2u, pos.y -= scale_exp2;
		if (t_corner.z <= tc_max)
			step_mask ^= 4u, pos.z -= scale_exp2;

		// Update active t-span and flip bits of the child slot index.
		t_min = tc_max;
		idx ^= step_mask;

		// Proceed with pop if the bit flips disagree with the ray direction.
		if ((idx & step_mask) != 0) {
			// POP
			// Find the highest differing bit between the two positions.
			uint differing_bits = 0;
			if ((step_mask & 1u) != 0)
				differing_bits |= floatBitsToUint(pos.x) ^ floatBitsToUint(pos.x + scale_exp2);
			if ((step_mask & 2u) != 0)
				differing_bits |= floatBitsToUint(pos.y) ^ floatBitsToUint(pos.y + scale_exp2);
			if ((step_mask & 4u) != 0)
				differing_bits |= floatBitsToUint(pos.z) ^ floatBitsToUint(pos.z + scale_exp2);
			scale = findMSB(differing_bits);
			if (scale >= STACK_SIZE)
				break;
			scale_exp2 = uintBitsToFloat((scale - STACK_SIZE + 127u) << 23u); // exp2f(scale - s_max)

			// Restore parent voxel from the stack.
			parent = stack[scale];

			// Round cube position and extract child slot index.
			uint shx = floatBitsToUint(pos.x) >> scale;
			uint shy = floatBitsToUint(pos.y) >> scale;
			uint shz = floatBitsToUint(pos.z) >> scale;
			pos.x = uintBitsToFloat(shx << scale);
			pos.y = uintBitsToFloat(shy << scale);
			pos.z = uintBitsToFloat(shz << scale);
			idx = (shx & 1u) | ((shy & 1u) << 1u) | ((shz & 1u) << 2u);

			// Prevent same parent from being stored again and invalidate cached
			// child descriptor.
			h = 0.0f;
			cur = 0;
		}
	}

	vec3 t_corner = t_coef * (pos + scale_exp2) - t_bias;

	vec3 norm = (t_corner.x > t_corner.y && t_corner.x > t_corner.z)
	                ? vec3(-1, 0, 0)
	                : (t_corner.y > t_corner.z ? vec3(0, -1, 0) : vec3(0, 0, -1));
	if ((oct_mask & 1u) == 0u)
		norm.x = -norm.x;
	if ((oct_mask & 2u) == 0u)
		norm.y = -norm.y;
	if ((oct_mask & 4u) == 0u)
		norm.z = -norm.z;

	// Undo mirroring of the coordinate system.
	if ((oct_mask & 1u) != 0u)
		pos.x = 3.0f - scale_exp2 - pos.x;
	if ((oct_mask & 2u) != 0u)
		pos.y = 3.0f - scale_exp2 - pos.y;
	if ((oct_mask & 4u) != 0u)
		pos.z = 3.0f - scale_exp2 - pos.z;

	// Output results.
	o_pos = clamp(o + t_min * d, pos, pos + scale_exp2);
	if (norm.x != 0)
		o_pos.x = norm.x > 0 ? pos.x + scale_exp2 + EPS * 2 : pos.x - EPS;
	if (norm.y != 0)
		o_pos.y = norm.y > 0 ? pos.y + scale_exp2 + EPS * 2 : pos.y - EPS;
	if (norm.z != 0)
		o_pos.z = norm.z > 0 ? pos.z + scale_exp2 + EPS * 2 : pos.z - EPS;
	o_normal = norm;
	//o_color = unpackUnorm4x8(cur).xyz;
	o_iter = iter;

	o_pos -= 1.5;
	return scale < STACK_SIZE && t_min <= t_max;
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
	float intensity = 0.3 + diff + spec;
 
	vec4 IAmbient = vec4(0.5, 0.5, 0.5, 1.0);
	vec4 IDiffuse = vec4(0.5, 0.5, 0.5, 0.5) * max(dot(inNormal, inLightVec), 0.0);
	float shininess = 0.75;
	vec4 ISpecular = vec4(0.5, 0.5, 0.5, 1.0) * pow(max(dot(Reflected, Eye), 0.0), 2.0) * shininess; 

	outFragColor = vec4((IAmbient + IDiffuse) * vec4(inColor, 1.0) + ISpecular);
 
	// Some manual saturation
	if (intensity > 0.95)
		outFragColor *= 2.25;
	if (intensity < 0.15)
		outFragColor = vec4(0.1);
	outFragColor = vec4(intensity);
	//outFragColor.xyz += 0.1;

	ivec3 i = ivec3(fragModelPos*GAIN)+HALF;
	int idx = i.x+DIM*i.y+DIM*DIM*i.z;
	//if(i.x < 0 || i.x > DIM) outFragColor.xyz = vec3(1.0, 0.0, 0.0);
	//if(i.y < 0 || i.y > DIM) outFragColor.xyz = vec3(0.0, 1.0, 0.0);
	//if(i.z < 0 || i.z > DIM) outFragColor.xyz = vec3(0.0, 0.0, 1.0);

	uint xxx = 0x3fc00000;
	float yyy = uintBitsToFloat(xxx);
	xxx = floatBitsToUint(yyy);
	
	vec3 pos, color, normal;
	uint iter;
	//vec4 ray = ubo.modelInv  * vec4(inEyePos, 1.0);
	vec3 ray = fragModelPos - (ubo.modelInv * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
	//outFragColor.xyz = vec3(0.0);
	bool direct;
	bool hit = Octree_RayMarchLeaf2(fragModelPos, normalize(ray.xyz), pos, color, normal, iter, direct);
	//fragModelPos*0.5
	//outFragColor.xyz += color;
	//if(hit) outFragColor.x += 0.5;
	
	if(!hit) discard;
	if(!direct)  outFragColor.xyz = normal + 0.3;

	vec4 hitPos = ubo.view * ubo.model * vec4(pos, 1.0);
	vec4 miaou = gl_FragCoord;

    // Custom Z in view space (your custom computation)
    float zViewCustom = inEyePos.z;
    // Compute Z in clip space
    float zClip = ubo.projection[2][2] * zViewCustom + ubo.projection[2][3];
    // Compute w_clip (assuming projectionMatrix[3][2] scales zViewCustom to w_clip)
    float wClip = ubo.projection[3][2] * zViewCustom + ubo.projection[3][3];
    // Compute Z in NDC
    float zNDC = zClip / wClip;
    // Map to [0, 1] depth range
    float zDepth = 0.5 * zNDC + 0.5;


    vec4 clipPosRef = ubo.projection * vec4(inEyePos, 1.0);
    // Perspective divide to get NDC
    float ndcDepthRef = clipPosRef.z / clipPosRef.w;
    // Normalize to [0, 1]
    float normalizedDepthRef = 0.5 * ndcDepthRef + 0.5;

	//hitPos.z *=20;
    vec4 clipPos = ubo.projection * hitPos;
    // Perspective divide to get NDC
    float ndcDepth = clipPos.z / clipPos.w;
    // Normalize to [0, 1]
    float normalizedDepth = 0.5 * ndcDepth + 0.5;

	vec4 fragCoord = gl_FragCoord;
   // if (gl_FragCoord.z > clipPos.z) {
       // discard; // Fragment is behind another object, discard it
    //}

    // Write to depth buffer
    gl_FragDepth = direct ? fragCoord.z : ndcDepth;

	int x = 0;
	//gl_FragDepth = 0.5;
	//outFragColor.xyz = abs(inEyePos);

	// outFragColor.xyz *= voxels[idx] != 0 ? 1.0 : 0.3;
	//outFragColor.xyz = 1.0-abs(inEyePos);

}