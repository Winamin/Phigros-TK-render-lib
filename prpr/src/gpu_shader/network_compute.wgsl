// 通用神经网络计算着色器 - 高度优化版本
struct NetworkParams {
    input_size: u32,
    output_size: u32,
    batch_size: u32,
    seq_len: u32,
    layer_index: u32,
    optimization_flags: u32,
};

// 共享内存用于缓存输入数据，提高内存访问效率
var<workgroup> shared_input: array<f32, 2048>; // 增大共享内存到2048
var<workgroup> shared_weights: array<f32, 2048>; // 增大共享内存到2048

@group(0) @binding(0) var<uniform> params: NetworkParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;
@group(0) @binding(4) var<storage, read> biases: array<f32>;

// 优化的激活函数
fn activate(val: f32, activation_type: u32) -> f32 {
    switch (activation_type) {
        case 0u: { // ReLU
            return max(val, 0.0);
        }
        case 1u: { // Sigmoid
            return 1.0 / (1.0 + exp(-val));
        }
        case 2u: { // Tanh
            return tanh(val);
        }
        case 3u: { // Swish (x * sigmoid(beta * x), 这里 beta = 1)
            return val * (1.0 / (1.0 + exp(-val)));
        }
        case 4u: { // GELU
            let sqrt_2_over_pi = 0.7978845608;
            let x3 = val * val * val;
            let a = sqrt_2_over_pi * (val + 0.044715 * x3);
            let tanh_a = tanh(a);
            return 0.5 * val * (1.0 + tanh_a);
        }
        case 5u: { // Linear
            return val;
        }
        default: { // 默认返回 Linear
            return val;
        }
    }
}

// 向量点积优化函数（内联）- 增强版
fn dot_product_optimized_inline(
    input_offset: u32,
    weight_offset: u32,
    size: u32
) -> f32 {
    var sum = 0.0;
    
    // 使用更大的循环展开因子提高效率
    let unroll_factor = 8u;
    let unrolled_size = (size / unroll_factor) * unroll_factor;
    
    // 展开主循环 - 8路展开
    for (var i = 0u; i < unrolled_size; i += unroll_factor) {
        sum += input[input_offset + i] * weights[weight_offset + i];
        sum += input[input_offset + i + 1u] * weights[weight_offset + i + 1u];
        sum += input[input_offset + i + 2u] * weights[weight_offset + i + 2u];
        sum += input[input_offset + i + 3u] * weights[weight_offset + i + 3u];
        sum += input[input_offset + i + 4u] * weights[weight_offset + i + 4u];
        sum += input[input_offset + i + 5u] * weights[weight_offset + i + 5u];
        sum += input[input_offset + i + 6u] * weights[weight_offset + i + 6u];
        sum += input[input_offset + i + 7u] * weights[weight_offset + i + 7u];
    }
    
    // 处理剩余元素
    for (var i = unrolled_size; i < size; i++) {
        sum += input[input_offset + i] * weights[weight_offset + i];
    }
    
    return sum;
}

@compute @workgroup_size(256) // 增大工作组大小以提高并行度
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let gid = global_id.x;
    let lid = local_id.x;
    let total_elements = params.output_size * params.batch_size;
    
    if (gid >= total_elements) {
        return;
    }

    let batch_idx = gid / params.output_size;
    let output_idx = gid % params.output_size;
    
    // 计算输入和权重的偏移量
    let input_offset = batch_idx * params.input_size;
    let weight_offset = output_idx * params.input_size;
    
    // 根据优化标志选择计算策略
    let use_shared_memory = (params.optimization_flags & 0x01u) != 0u;
    let use_vectorization = (params.optimization_flags & 0x02u) != 0u;
    
    var sum = biases[output_idx];
    
    if (use_shared_memory && params.input_size <= 2048u) {
        // 使用共享内存优化小型矩阵乘法 - 增大tile大小
        let tile_size = 64u; // 增大tile大小
        let tiles = (params.input_size + tile_size - 1u) / tile_size;
        
        for (var tile = 0u; tile < tiles; tile++) {
            let tile_start = tile * tile_size;
            let tile_end = min(tile_start + tile_size, params.input_size);
            let tile_size_actual = tile_end - tile_start;
            
            // 协作加载到共享内存 - 使用更大的工作组
            for (var i = lid; i < tile_size_actual; i += 256u) {
                shared_input[i] = input[input_offset + tile_start + i];
                shared_weights[i] = weights[weight_offset + tile_start + i];
            }
            
            workgroupBarrier();
            
            // 计算部分点积 - 使用向量化访问
            let vectorized_size = (tile_size_actual / 4u) * 4u;
            for (var i = 0u; i < vectorized_size; i += 4u) {
                sum += shared_input[i] * shared_weights[i];
                sum += shared_input[i + 1u] * shared_weights[i + 1u];
                sum += shared_input[i + 2u] * shared_weights[i + 2u];
                sum += shared_input[i + 3u] * shared_weights[i + 3u];
            }
            
            // 处理剩余元素
            for (var i = vectorized_size; i < tile_size_actual; i++) {
                sum += shared_input[i] * shared_weights[i];
            }
            
            workgroupBarrier();
        }
    } else if (use_vectorization) {
        // 使用增强的向量化点积计算
        sum += dot_product_optimized_inline(input_offset, weight_offset, params.input_size);
    } else {
        // 标准计算路径 - 也使用一定的循环展开
        let unroll_factor = 4u;
        let unrolled_size = (params.input_size / unroll_factor) * unroll_factor;
        
        for (var i = 0u; i < unrolled_size; i += unroll_factor) {
            sum += input[input_offset + i] * weights[weight_offset + i];
            sum += input[input_offset + i + 1u] * weights[weight_offset + i + 1u];
            sum += input[input_offset + i + 2u] * weights[weight_offset + i + 2u];
            sum += input[input_offset + i + 3u] * weights[weight_offset + i + 3u];
        }
        
        for (var i = unrolled_size; i < params.input_size; i++) {
            sum += input[input_offset + i] * weights[weight_offset + i];
        }
    }
    
    // 应用激活函数
    let activation_type = (params.optimization_flags >> 8u) & 0x0Fu;
    let result = activate(sum, activation_type);
    
    output[batch_idx * params.output_size + output_idx] = result;
}