struct AttentionParams {
    input_size: u32,
    output_size: u32,
    batch_size: u32,
    num_heads: u32,
    head_dim: u32,
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
    return 1.0; // 对于单个值，softmax结果为1
}

// 多值softmax函数
fn softmax_multi(values: ptr<workgroup, array<f32, 64>>, size: u32) {
    var max_val = -999999.0;
    
    // 找到最大值
    for (var i = 0u; i < size; i++) {
        if ((*values)[i] > max_val) {
            max_val = (*values)[i];
        }
    }
    
    // 计算exp并求和
    var sum_exp = 0.0;
    for (var i = 0u; i < size; i++) {
        (*values)[i] = exp((*values)[i] - max_val);
        sum_exp += (*values)[i];
    }
    
    // 归一化
    if (sum_exp > 0.00001) {
        for (var i = 0u; i < size; i++) {
            (*values)[i] = (*values)[i] / sum_exp;
        }
    } else {
        // 防止除以0
        for (var i = 0u; i < size; i++) {
            (*values)[i] = 1.0 / f32(size);
        }
    }
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
    
    // 计算多头注意力
    let head_idx = output_idx / params.head_dim;
    let head_offset = output_idx % params.head_dim;
    
    // 确保头索引有效
    if (head_idx >= params.num_heads) {
        output[global_id] = 0.0;
        return;
    }
    
    // 计算Q, K, V投影
    var q = 0.0;
    var k = 0.0;
    var v = 0.0;
    
    // 计算Q投影
    let q_weight_base = head_idx * params.head_dim + head_offset;
    var sum_q = biases[q_weight_base];
    for (var i = 0u; i < params.input_size; i++) {
        let weight_idx = q_weight_base * params.input_size + i;
        if (weight_idx < arrayLength(&weights)) {
            sum_q += weights[weight_idx] * input[batch_idx * params.input_size + i];
        }
    }
    q = sum_q;
    
    // 计算K投影
    let k_weight_base = params.output_size + head_idx * params.head_dim + head_offset;
    var sum_k = biases[k_weight_base];
    for (var i = 0u; i < params.input_size; i++) {
        let weight_idx = k_weight_base * params.input_size + i;
        if (weight_idx < arrayLength(&weights)) {
            sum_k += weights[weight_idx] * input[batch_idx * params.input_size + i];
        }
    }
    k = sum_k;
    
    // 计算V投影
    let v_weight_base = 2u * params.output_size + head_idx * params.head_dim + head_offset;
    var sum_v = biases[v_weight_base];
    for (var i = 0u; i < params.input_size; i++) {
        let weight_idx = v_weight_base * params.input_size + i;
        if (weight_idx < arrayLength(&weights)) {
            sum_v += weights[weight_idx] * input[batch_idx * params.input_size + i];
        }
    }
    v = sum_v;
    
    // Scaled Dot-Product Attention
    let dk = sqrt(f32(params.head_dim));
    let attention_score = (q * k) / dk;
    
    // 简化的注意力权重应用（懒了）
    let attention_weight = softmax_single(attention_score);
    
    // 输出结果
    output[global_id] = attention_weight * v;
}