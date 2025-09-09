struct BatchSize {
    size: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> biases: array<f32>;
@group(0) @binding(4) var<storage, read> batch_size: BatchSize;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let output_size = arrayLength(&biases);
    let input_size = arrayLength(&input) / batch_size.size;
    let global_id = id.x;

    if (global_id >= output_size * batch_size.size) {
        return;
    }

    let batch_idx = global_id / output_size;
    let output_idx = global_id % output_size;

    var sum = biases[output_idx];
    for (var i = 0u; i < input_size; i++) {
        let input_val = input[batch_idx * input_size + i];
        let weight_idx = output_idx * input_size + i;
        sum += weights[weight_idx] * input_val;
    }

    output[batch_idx * output_size + output_idx] = sum;
}