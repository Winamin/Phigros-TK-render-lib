// backward.wgsl 修改后的内容
struct BatchSize {
    size: u32,
}

@group(0) @binding(0) var<storage, read> gradients: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> weight_gradients: array<f32>;
@group(0) @binding(3) var<storage, read_write> bias_gradients: array<f32>;
@group(0) @binding(4) var<storage, read_write> prev_gradients: array<f32>;
@group(0) @binding(5) var<storage, read> batch_size: BatchSize;
@group(0) @binding(6) var<storage, read> prev_activations: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let output_size = arrayLength(&gradients) / batch_size.size; // 访问 batch_size.size
    let input_size = arrayLength(&weights) / output_size;

    if (index >= input_size * output_size) {
        return;
    }

    let output_idx = index / input_size;
    let input_idx = index % input_size;

    // 计算权重梯度
    var total_weight_grad = 0.0;
    for (var b = 0u; b < batch_size.size; b = b + 1) { // 使用 batch_size.size
        let grad = gradients[b * output_size + output_idx];
        let input = prev_activations[b * input_size + input_idx];
        total_weight_grad = total_weight_grad + grad * input;
    }
    weight_gradients[index] = total_weight_grad;

    // 计算偏置梯度
    var total_bias_grad = 0.0;
    for (var b = 0u; b < batch_size.size; b = b + 1) { // 使用 batch_size.size
        total_bias_grad = total_bias_grad + gradients[b * output_size + output_idx];
    }
    bias_gradients[output_idx] = total_bias_grad;

    // 计算前一层的梯度（注意：这里简化了原子操作，实际可能需要调整）
    for (var b = 0u; b < batch_size.size; b = b + 1) { // 使用 batch_size.size
        let grad = gradients[b * output_size + output_idx];
        for (var i = 0u; i < input_size; i = i + 1) {
            let weight_idx = output_idx * input_size + i;
            let value_to_add = grad * weights[weight_idx];
            // 注意：这里需要原子操作，但为简化先直接赋值
            // 实际代码中应根据需要实现原子操作
            prev_gradients[b * input_size + i] = prev_gradients[b * input_size + i] + value_to_add;
        }
    }
}