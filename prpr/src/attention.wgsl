struct AttentionParams {
    input_size: u32,
    output_size: u32,
    batch_size: u32,
};

// 输入数据: [batch_size, input_size]
// 权重: [3 * output_size, input_size] (Q, K, V投影矩阵)
// 偏置: [3 * output_size]
// 输出: [batch_size, output_size]

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> biases: array<f32>;
@group(0) @binding(4) var<storage, read> params: AttentionParams;

fn softmax_single(value: f32) -> f32 {
    // 对于单个值的softmax，结果总是1.0
    return 1.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let global_id = id.x;
    let total_output_size = params.output_size * params.batch_size;
    
    if (global_id >= total_output_size) {
        return;
    }
    
    let batch_idx = global_id / params.output_size;
    let output_idx = global_id % params.output_size;
    
    // 计算Q, K, V
    var q = 0.0;
    var k = 0.0;
    var v = 0.0;
    
    // 计算Q
    var sum_q = biases[output_idx];
    for (var i = 0u; i < params.input_size; i++) {
        let weight_idx = output_idx * params.input_size + i;
        sum_q += weights[weight_idx] * input[batch_idx * params.input_size + i];
    }
    q = sum_q;
    
    // 计算K
    var sum_k = biases[params.output_size + output_idx];
    for (var i = 0u; i < params.input_size; i++) {
        let weight_idx = (params.output_size + output_idx) * params.input_size + i;
        sum_k += weights[weight_idx] * input[batch_idx * params.input_size + i];
    }
    k = sum_k;
    
    // 计算V
    var sum_v = biases[2u * params.output_size + output_idx];
    for (var i = 0u; i < params.input_size; i++) {
        let weight_idx = (2u * params.output_size + output_idx) * params.input_size + i;
        sum_v += weights[weight_idx] * input[batch_idx * params.input_size + i];
    }
    v = sum_v;
    
    // Scaled Dot-Product Attention
    let dk = f32(params.output_size);
    let score = (q * k) / sqrt(dk);
    
    // 应用softmax (简化版)
    let weight = softmax_single(score);
    
    // 输出结果
    output[global_id] = weight * v;
}