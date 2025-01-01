#version 450

// Input vertex attributes from the vertex buffer
layout (location = 0) in vec4 inPos;       // Position of the vertex in object space
layout (location = 1) in vec3 inNormal;    // Normal vector of the vertex in object space
layout (location = 2) in vec2 inTexCoord;  // Texture coordinates of the vertex
layout (location = 3) in vec3 inColor;     // Color information for the vertex (e.g., vertex color)

// Uniform buffer object containing transformation matrices and lighting data
layout (binding = 0) uniform UBO 
{
    mat4 projection;  // Projection matrix for perspective/orthographic projection
    mat4 model;       // Model matrix to transform object space to world space
    mat4 normal;      // Normal matrix (typically the inverse transpose of the model matrix)
    mat4 view;        // View matrix to transform world space to camera (view) space
    vec3 lightpos;    // Position of the light source in world space
    mat4 modelInv;     
} ubo;

// Outputs passed to the fragment shader
layout (location = 0) out vec2 outUV;         // Pass-through texture coordinates
layout (location = 1) out vec3 outNormal;     // Normal vector in view space
layout (location = 2) out vec3 outColor;      // Pass-through color from the vertex
layout (location = 3) out vec3 outEyePos;     // Position of the vertex in view space
layout (location = 4) out vec3 outLightVec;   // Direction from the vertex to the light in view space
layout (location = 5) out vec3 outModelPos;   // Add an output for the model-space position

//#extension GL_EXT_debug_printf : enable

void main() 
{
    // Pass texture coordinates to the fragment shader
    outUV = inTexCoord.st;

	// Pass the model-space position directly to the fragment shader
    outModelPos = inPos.xyz;

    // Transform the normal vector from object space to view space using the normal matrix
    // mat3 extracts the top-left 3x3 part of the matrix for transforming the normal
    outNormal = normalize(mat3(ubo.normal) * inNormal);

    // Pass vertex color directly to the fragment shader
    outColor = inColor;

    // Compute the model-view matrix (combines model and view transformations)
    mat4 modelView = ubo.view * ubo.model; //TODO precompute in UBO

    // Transform the vertex position from object space to clip space
    // - First, apply model-view transformation to get the position in view space
    // - Then apply the projection matrix to get the final clip space position
    vec4 pos = modelView * inPos;    // Position in view space
    gl_Position = ubo.projection * pos; // Position in clip space

    // Pass the vertex position in view space to the fragment shader
    outEyePos = vec3(modelView * inPos); //TODO reuse pos above

    // Compute the light position in view space
    // - Transform the light position from world space to view space using the model-view matrix
    vec4 lightPos = ubo.view * vec4(ubo.lightpos, 1.0); //swapped modelView

    // Compute the normalized vector pointing from the vertex position to the light source in view space
    // - This will be used for lighting calculations in the fragment shader
    outLightVec = normalize(lightPos.xyz - outEyePos);

	//float myfloat = 3.1415f;
    //debugPrintfEXT("My float is %f", myfloat);
}
