#version 430

uniform mat4 worldViewProjMatrix;
uniform mat4 worldViewMatrix;

in vec4 sPos;
in vec3 sNormal;
in float sId;

out vec3 normal;
out vec3 viewNormal;
flat out int trianId;

void main()
{
    normal = sNormal;
    viewNormal = (worldViewMatrix*vec4(sNormal, 0.0)).xyz;
    trianId = int(sId);
    gl_Position = worldViewProjMatrix*sPos;
}
