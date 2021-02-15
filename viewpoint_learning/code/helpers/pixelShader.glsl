#version 430

in vec3 normal;
in vec3 viewNormal;
flat in int trianId;

out ivec4 outId;
out vec4 outNormal;
out vec4 outColor;

void main()
{
    outId = ivec4(trianId);
    outNormal = vec4(normalize(normal), 1.0);
    outColor = vec4(max(dot(normalize(viewNormal), normalize(vec3(0.2, -0.2, 1.0))), 0.0)*0.8+0.2);
}