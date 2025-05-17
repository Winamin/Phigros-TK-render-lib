#version 100
precision highp float;

varying vec2 uv;
uniform sampler2D screenTexture;
uniform float progress;
uniform vec4 burnColor;
uniform float time;
uniform vec2 screenSize;

float random(vec2 co) {
    return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

float noise(vec2 uv) {
    vec2 p = floor(uv);
    vec2 f = fract(uv);
    float a = random(p);
    float b = random(p + vec2(1.0, 0.0));
    float c = random(p + vec2(0.0, 1.0));
    float d = random(p + vec2(1.0, 1.0));
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

void main() {
    vec2 uv = uv;
    vec4 color = texture2D(screenTexture, uv); // 替换 CC_Texture0 为 screenTexture

    float burnNoise = noise(uv * 10.0 + vec2(time * 0.5, time * 0.5));

    float burnFactor = progress - burnNoise;

    if (burnFactor < 0.0) {
        gl_FragColor = color;
        gl_FragColor.a = 1.0;
    } else {
        vec4 burnedColor = mix(color, burnColor, burnFactor);
        burnedColor.a = mix(1.0, 0.0, burnFactor);
        gl_FragColor = burnedColor;
    }
}
