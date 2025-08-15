@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> biases: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let output_idx = id.x;
    let input_size = 16u;
    let output_size = 128u;
    
    if (output_idx >= output_size) { return; }
    
    var sum = 0.0;
    for (var i = 0u; i < input_size; i = i + 1u) {
        sum += input[i] * weights[i * output_size + output_idx];
    }
    output[output_idx] = sum + biases[output_idx];
}