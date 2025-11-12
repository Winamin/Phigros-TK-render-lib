struct AttentionParams {
    batch_size: u32,   // 批大小
    seq_len: u32,      // 序列长度 (必须添加)
    input_size: u32,   // 输入特征维度 (d_model)
    num_heads: u32,    // 注意力头数量
    head_dim: u32,     // 每个头的维度 (通常 = input_size / num_heads)
};

// 输入数据: [batch_size, seq_len, input_size]
// 权重布局:
//   Q权重: [num_heads, head_dim, input_size]
//   K权重: [num_heads, head_dim, input_size]
//   V权重: [num_heads, head_dim, input_size]
// 偏置布局:
//   Q偏置: [num_heads, head_dim]
//   K偏置: [num_heads, head_dim]
//   V偏置: [num_heads, head_dim]
// 输出: [batch_size, seq_len, input_size]

// 共享内存大小 = 2 * seq_len (用于Softmax的归约)
const SHARED_MEM_SIZE: u32 = 128; // 支持最大seq_len=64
var<workgroup> shared_mem: array<f32, SHARED_MEM_SIZE>;

@group(0) @binding(0) var<storage, read> params: AttentionParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> q_weights: array<f32>; // Q权重和偏置
@group(0) @binding(3) var<storage, read> k_weights: array<f32>; // K权重和偏置  
@group(0) @binding(4) var<storage, read> v_weights: array<f32>; // V权重和偏置
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

// 计算Q/K/V投影 - qkv_type: 0=Q, 1=K, 2=V
fn compute_projection(
    batch_idx: u32,
    seq_idx: u32,
    head_idx: u32,
    qkv_type: u32
) -> array<f32, 64> { // 返回head_dim维向量
    var result: array<f32, 64>;
    let base_input_idx = (batch_idx * params.seq_len + seq_idx) * params.input_size;

    // 计算偏置区域的起始位置（在合并缓冲区中）
    let weights_size = params.num_heads * params.head_dim * params.input_size;

    for (var d = 0u; d < params.head_dim; d++) {
        var sum = 0.0;
        
        // 根据qkv_type选择正确的权重缓冲区
        if (qkv_type == 0u) { // Q
            // 读取Q偏置
            let bias_idx = weights_size + head_idx * params.head_dim + d;
            if (bias_idx < arrayLength(&q_weights)) {
                sum = q_weights[bias_idx];
            }
            
            // 读取Q权重并计算
            for (var i = 0u; i < params.input_size; i++) {
                let weight_idx = head_idx * params.head_dim * params.input_size + d * params.input_size + i;
                if (weight_idx < arrayLength(&q_weights)) {
                    sum += q_weights[weight_idx] * input[base_input_idx + i];
                }
            }
        } else if (qkv_type == 1u) { // K  
            // 读取K偏置
            let bias_idx = weights_size + head_idx * params.head_dim + d;
            if (bias_idx < arrayLength(&k_weights)) {
                sum = k_weights[bias_idx];
            }
            
            // 读取K权重并计算
            for (var i = 0u; i < params.input_size; i++) {
                let weight_idx = head_idx * params.head_dim * params.input_size + d * params.input_size + i;
                if (weight_idx < arrayLength(&k_weights)) {
                    sum += k_weights[weight_idx] * input[base_input_idx + i];
                }
            }
        } else { // V
            // 读取V偏置
            let bias_idx = weights_size + head_idx * params.head_dim + d;
            if (bias_idx < arrayLength(&v_weights)) {
                sum = v_weights[bias_idx];
            }
            
            // 读取V权重并计算
            for (var i = 0u; i < params.input_size; i++) {
                let weight_idx = head_idx * params.head_dim * params.input_size + d * params.input_size + i;
                if (weight_idx < arrayLength(&v_weights)) {
                    sum += v_weights[weight_idx] * input[base_input_idx + i];
                }
            }
        }
        result[d] = sum;
    }
    return result;
}

// Softmax使用共享内存优化
fn softmax_shared(local_id: u32, values: ptr<function, array<f32, 64>>) -> array<f32, 64> {
    let size = params.seq_len;
    var max_val = -3.4028235e38; // -FLT_MAX

    // 1. 找到最大值 (使用共享内存归约)
    shared_mem[local_id] = (*values)[local_id];
    workgroupBarrier();

    for (var offset = 1u; offset < size; offset = offset * 2u) {
        if (local_id % (2u * offset) == 0u && local_id + offset < size) {
            shared_mem[local_id] = max(shared_mem[local_id], shared_mem[local_id + offset]);
        }
        workgroupBarrier();
    }
    max_val = shared_mem[0];

    // 2. 计算exp并求和
    var exp_val = exp((*values)[local_id] - max_val);
    shared_mem[local_id] = exp_val;
    workgroupBarrier();

    var sum_exp = 0.0;
    for (var i = 0u; i < size; i++) {
        sum_exp += shared_mem[i];
    }

    // 3. 归一化
    var result: array<f32, 64>;
    if (sum_exp > 1e-8) {
        result[local_id] = exp_val / sum_exp;
    } else {
        result[local_id] = 1.0 / f32(size);
    }
    return result;
}

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    // 每个工作组处理: 一个batch项, 一个注意力头, 一个目标token
    let batch_idx = wg_id.x;
    let head_idx = wg_id.y;
    let target_seq_idx = wg_id.z;

    // 验证索引
    if (batch_idx >= params.batch_size ||
        head_idx >= params.num_heads ||
        target_seq_idx >= params.seq_len) {
        return;
    }

    // 计算当前目标token的Q向量
    let q = compute_projection(
        batch_idx,
        target_seq_idx,
        head_idx,
        0u // Q类型
    );

    // 为所有源token计算K和V
    var scores: array<f32, 64>;
    var v_vectors: array<array<f32, 64>, 64>; // [seq_len][head_dim]

    for (var src_seq_idx = 0u; src_seq_idx < params.seq_len; src_seq_idx++) {
        // 计算K
        let k = compute_projection(
            batch_idx,
            src_seq_idx,
            head_idx,
            1u // K类型
        );

        // 计算点积 Q·K
        var dot_product = 0.0;
        for (var d = 0u; d < params.head_dim; d++) {
            dot_product += q[d] * k[d];
        }

        // 缩放
        scores[src_seq_idx] = dot_product / sqrt(f32(params.head_dim));

        // 计算V (稍后用于加权和)
        v_vectors[src_seq_idx] = compute_projection(
            batch_idx,
            src_seq_idx,
            head_idx,
            2u // V类型
        );
    }

    // 应用Softmax (使用共享内存)
    let attention_weights = softmax_shared(local_id.x, &scores);

    // 计算加权和: Σ(attention_weight * V)
    var head_output: array<f32, 64>;
    for (var d = 0u; d < params.head_dim; d++) {
        var weighted_sum = 0.0;
        for (var src_seq_idx = 0u; src_seq_idx < params.seq_len; src_seq_idx++) {
            weighted_sum += attention_weights[src_seq_idx] * v_vectors[src_seq_idx][d];
        }
        head_output[d] = weighted_sum;
    }

    // 写入输出 (多头拼接)
    let output_base = (batch_idx * params.seq_len + target_seq_idx) * params.input_size;
    let head_offset = head_idx * params.head_dim;

    for (var d = 0u; d < params.head_dim; d++) {
        let out_idx = output_base + head_offset + d;
        if (out_idx < arrayLength(&output)) {
            output[out_idx] = head_output[d];
        }
    }
}