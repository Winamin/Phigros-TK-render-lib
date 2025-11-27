struct ResidualParams {
    input_size: u32,
    output_size: u32,
    batch_size: u32,
    optimization_flags: u32,  // 优化标志
};

// 输入数据: [batch_size, input_size]
// 残差输入: [batch_size, output_size] (或input_size，如果维度匹配)
// 权重: [output_size, input_size]
// 偏置: [output_size]
// 输出: [batch_size, output_size]

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> biases: array<f32>;
@group(0) @binding(4) var<storage, read> residual_input: array<f32>;
@group(0) @binding(5) var<storage, read> params: ResidualParams;

fn activate(val: f32) -> f32 {
    // 使用GELU激活函数作为默认
    let sqrt_2_over_pi = 0.7978845608;
    let x3 = val * val * val;
    let a = sqrt_2_over_pi * (val + 0.044715 * x3);
    let tanh_a = tanh(a);
    return 0.5 * val * (1.0 + tanh_a);
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
    
    // 计算主路径输出
    var sum = biases[output_idx];
    for (var i = 0u; i < params.input_size; i++) {
        let weight_idx = output_idx * params.input_size + i;
        if (weight_idx < arrayLength(&weights)) {
            sum += weights[weight_idx] * input[batch_idx * params.input_size + i];
        }
    }
    
    // 应用激活函数
    var main_output = activate(sum);
    
    // 添加残差连接
    var residual_value = 0.0;
    if (params.input_size == params.output_size) {
        // 维度匹配，直接添加残差
        residual_value = residual_input[global_id];
    } else {
        // 维度不匹配，可能需要投影或其他处理
        // 这里简化处理，只使用主路径输出
        residual_value = 0.0;
    }
    
    // 最终输出 = 主路径输出 + 残差输入
    output[global_id] = main_output + residual_value;
}