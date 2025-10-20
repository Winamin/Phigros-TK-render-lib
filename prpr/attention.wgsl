struct AttentionParams {
    input_size: u32,
    output_size: u32,
    batch_size: u32,
    num_heads: u32,
};

// 输入数据: [batch_size, input_size]
// 权重: [3 * output_size, input_size] (Q, K, V投影矩阵)
// 偏置: [3 * output_size]
// 输出: [batch_size, output_size]

var<workgroup> shared_mem: array<f32, 1024>; // 共享内存用于softmax计算

@group(0) @binding(4) var<storage, read> params: AttentionParams;
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read> biases: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

fn softmax(values: ptr<function, array<f32, 64>>, size: u32) -> array<f32, 64> {
    var max_val = -999999.0;
    
    // 找到最大值
    for (var i = 0u; i < size; i++) {
        if (*values)[i] > max_val {
            max_val = (*values)[i];
        }
    }
    
    // 计算exp并求和
    var sum_exp = 0.0;
    var exp_values: array<f32, 64>;
    for (var i = 0u; i < size; i++) {
        exp_values[i] = exp((*values)[i] - max_val);
        sum_exp += exp_values[i];
    }
    
    // 归一化
    var result: array<f32, 64>;
    if (sum_exp > 0.00001) {
        for (var i = 0u; i < size; i++) {
            result[i] = exp_values[i] / sum_exp;
        }
    } else {
        // 防止除以0
        for (var i = 0u; i < size; i++) {
            result[i] = 1.0 / f32(size);
        }
    }
    
    return result;
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
    let head_dim = params.output_size / params.num_heads;
    let head_idx = output_idx / head_dim;
    let head_offset = output_idx % head_dim;
    
    // 计算Q, K, V (多头版本)
    var q = 0.0;
    var k = 0.0;
    var v = 0.0;
    
    // 计算Q投影
    var sum_q = biases[head_idx * head_dim + head_offset];
    for (var i = 0u; i < params.input_size; i++) {
        let weight_idx = (head_idx * head_dim + head_offset) * params.input_size + i;
        if (weight_idx < arrayLength(&weights)) {
            sum_q += weights[weight_idx] * input[batch_idx * params.input_size + i];
        }
    }
    q = sum_q;
    
    // 计算K投影
    var sum_k = biases[params.output_size + head_idx * head_dim + head_offset];
    for (var i = 0u; i < params.input_size; i++) {
        let weight_idx = (params.output_size + head_idx * head_dim + head_offset) * params.input_size + i;
        if (weight_idx < arrayLength(&weights)) {
            sum_k += weights[weight_idx] * input[batch_idx * params.input_size + i];
        }
    }
    k = sum_k;
    
    // 计算V投影
    var sum_v = biases[2u * params.output_size + head_idx * head_dim + head_offset];
    for (var i = 0u; i < params.input_size; i++) {
        let weight_idx = (2u * params.output_size + head_idx * head_dim + head_offset) * params.input_size + i;
        if (weight_idx < arrayLength(&weights)) {
            sum_v += weights[weight_idx] * input[batch_idx * params.input_size + i];
        }
    }
    v = sum_v;
    
    // Scaled Dot-Product Attention
    let dk = sqrt(f32(head_dim));
    let score = (q * k) / dk;
    
    // 应用softmax并计算加权值
    // 简化的softmax实现 - 在实际应用中应该计算所有相关位置的注意力分数
    let attention_weight = 1.0 / (1.0 + exp(-score)); // 简化的sigmoid作为注意力权重
    
    // 计算输出值
    output[global_id] = attention_weight * v;
}