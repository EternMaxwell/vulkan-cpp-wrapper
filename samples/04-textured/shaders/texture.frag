#version 450

layout(location = 0) in vec2 fragUV;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 1) uniform sampler2D tex;
layout(push_constant) uniform PC {
    float tint;
} pc;

void main() {
    outColor = texture(tex, fragUV) * pc.tint;
}
