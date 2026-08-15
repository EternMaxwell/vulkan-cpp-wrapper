#version 450

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec2 inUV;

layout(set = 0, binding = 0) uniform UBO {
    vec2 offset;
} ubo;

layout(location = 0) out vec2 fragUV;

void main() {
    gl_Position = vec4(inPosition + ubo.offset, 0.0, 1.0);
    fragUV = inUV;
}
