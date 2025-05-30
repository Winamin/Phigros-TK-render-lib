#version 100
precision highp float;

varying highp vec2 uv;
uniform vec2 screenSize;
uniform sampler2D screenTexture;

#define FXAA_EDGE_THRESHOLD      (1.0/16.0)
#define FXAA_EDGE_THRESHOLD_MIN  (1.0/24.0)
#define FXAA_SUBPIX_QUALITY      0.75

float luminance(vec3 rgb) {
    return dot(rgb, vec3(0.299, 0.587, 0.114));
}

vec4 textureBilinear(sampler2D tex, vec2 coord, vec2 texelSize) {
    vec2 f = fract(coord * screenSize);
    vec4 tl = texture2D(tex, coord);
    vec4 tr = texture2D(tex, coord + vec2(texelSize.x, 0.0));
    vec4 bl = texture2D(tex, coord + vec2(0.0, texelSize.y));
    vec4 br = texture2D(tex, coord + texelSize);
    return mix(mix(tl, tr, f.x), mix(bl, br, f.x), f.y);
}

void computeSamples(vec2 fragCoord, vec2 invRes,
                   out vec2[5] samples, out vec2 texelSize) {
    texelSize = invRes;
    vec2 center = fragCoord * invRes;

    samples[0] = center;                              // M
    samples[1] = center + vec2(-1.0, -1.0) * texelSize; // NW
    samples[2] = center + vec2( 1.0, -1.0) * texelSize; // NE
    samples[3] = center + vec2(-1.0,  1.0) * texelSize; // SW
    samples[4] = center + vec2( 1.0,  1.0) * texelSize; // SE
}

vec4 enhancedFXAA(sampler2D tex, vec2 fragCoord, vec2 resolution) {
    vec2 invRes = 1.0 / resolution;
    vec2 texelSize;
    vec2[5] samples;
    computeSamples(fragCoord, invRes, samples, texelSize);

    float luma[5];
    for(int i = 0; i < 5; i++) {
        luma[i] = luminance(texture2D(tex, samples[i]).rgb);
    }

    float edgeHorz = abs(luma[3] - luma[4]) * 2.0 +
                    abs(luma[0] - luma[1]) * 2.0 +
                    abs(luma[2] - luma[0]);
    float edgeVert = abs(luma[1] - luma[2]) * 2.0 +
                    abs(luma[0] - luma[3]) * 2.0 +
                    abs(luma[4] - luma[0]);

    float edgeThreshold = max(FXAA_EDGE_THRESHOLD,
                             min(luma[0], min(min(luma[1], luma[2]), min(luma[3], luma[4])) * FXAA_EDGE_THRESHOLD_MIN);

    if(edgeHorz < edgeThreshold && edgeVert < edgeThreshold) {
        return texture2D(tex, samples[0]);
    }

    float dirReduce = max((luma[1] + luma[2] + luma[3] + luma[4]) * 0.03125, 0.0078125);
    vec2 dir = vec2(-((luma[1] + luma[2]) - (luma[3] + luma[4])),
                   ((luma[1] + luma[3]) - (luma[2] + luma[4])));
    dir = normalize(dir / (min(abs(dir.x), abs(dir.y)) + dirReduce));

    vec2 samplePos1 = samples[0] + dir * (0.5 - FXAA_SUBPIX_QUALITY) * texelSize;
    vec2 samplePos2 = samples[0] + dir * (0.5 + FXAA_SUBPIX_QUALITY) * texelSize;

    vec4 color1 = textureBilinear(tex, samplePos1, texelSize);
    vec4 color2 = textureBilinear(tex, samplePos2, texelSize);

    float blendFactor = abs(luma[0] - (luma[1]+luma[2]+luma[3]+luma[4])*0.25) / max(luma[0], 0.05);
    return mix(color1, color2, blendFactor);
}

void main() {
    vec2 fragCoord = uv * screenSize;
    gl_FragColor = enhancedFXAA(screenTexture, fragCoord, screenSize);
}