@group(0) @binding(0) var<storage, read> gradients: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> weight_gradients: array<f32>;
@group(0) @binding(3) var<storage, read_write> bias_gradients: array<f32>;
@group(0) @binding(4) var<storage, read_write> prev_gradients: array<f32>;
@group(0) @binding(5) var<uniform> batch_size: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let output_size = arrayLength(&gradients) / batch_size;
    let input_size = arrayLength(&weights) / output_size;

    if (index >= input_size * output_size) {
        return;
    }

    let output_idx = index / input_size;
    let input_idx = index % input_size;

    // 计算权重梯度
    var total_weight_grad = 0.0;
    for (var b = 0u; b < batch_size; b = b + 1) {
        let grad = gradients[b * output_size + output_idx];
        let input = prev_activations[b * input_size + input_idx];
        total_weight_grad = total_weight_grad + grad * input;
    }
    weight_gradients[index] = total_weight_grad;

    // 计算偏置梯度
    var total_bias_grad = 0.0;
    for (var b = 0u; b < batch_size; b = b + 1) {
        total_bias_grad = total_bias_grad + gradients[b * output_size + output_idx];
    }
    bias_gradients[output_idx] = total_bias_grad;

    // 计算前一层的梯度
    for (var b = 0u; b < batch_size; b = b + 1) {
        let grad = gradients[b * output_size + output_idx];
        for (var i = 0u; i < input_size; i = i + 1) {
            let weight_idx = output_idx * input_size + i;
            atomicAdd(&prev_gradients[b * input_size + i], grad * weights[weight_idx]);
        }
    }
}