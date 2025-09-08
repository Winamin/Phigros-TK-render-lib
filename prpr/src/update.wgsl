@group(0) @binding(0) var<storage, read_write> weights: array<f32>;
@group(0) @binding(1) var<storage, read> weight_gradients: array<f32>;
@group(0) @binding(2) var<storage, read> bias_gradients: array<f32>;
@group(0) @binding(3) var<uniform> params: vec3<f32>; // [learning_rate, weight_decay, batch_size]

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&weights)) {
        return;
    }

    let learning_rate = params[0];
    let weight_decay = params[1];
    let batch_size = params[2];

    // 应用梯度更新权重
    let grad = weight_gradients[index] / batch_size;
    let decay = weight_decay * weights[index];
    weights[index] = weights[index] - learning_rate * (grad + decay);
}