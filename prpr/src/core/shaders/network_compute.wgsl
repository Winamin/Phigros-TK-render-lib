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
var<workgroup> shared_input: array<f32, 1024>; // 支持最大输入大小1024
var<workgroup> shared_weights: array<f32, 1024>; // 支持最大权重大小1024

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
        case 1u: { // GELU
            let sqrt_2_over_pi = 0.7978845608;
            let x3 = val * val * val;
            let a = sqrt_2_over_pi * (val + 0.044715 * x3);
            let tanh_a = tanh(a);
            return 0.5 * val * (1.0 + tanh_a);
        }
        case 2u: { // Sigmoid
            return 1.0 / (1.0 + exp(-val));
        }
        default: { // Linear
            return val;
        }
    }
}

// 向量点积优化函数（内联）
fn dot_product_optimized_inline(
    input_offset: u32,
    weight_offset: u32,
    size: u32
) -> f32 {
    var sum = 0.0;
    
    // 使用循环展开优化
    let unroll_factor = 4u;
    let unrolled_size = (size / unroll_factor) * unroll_factor;
    
    // 展开主循环
    for (var i = 0u; i < unrolled_size; i += unroll_factor) {
        sum += input[input_offset + i] * weights[weight_offset + i];
        sum += input[input_offset + i + 1u] * weights[weight_offset + i + 1u];
        sum += input[input_offset + i + 2u] * weights[weight_offset + i + 2u];
        sum += input[input_offset + i + 3u] * weights[weight_offset + i + 3u];
    }
    
    // 处理剩余元素
    for (var i = unrolled_size; i < size; i++) {
        sum += input[input_offset + i] * weights[weight_offset + i];
    }
    
    return sum;
}

@compute @workgroup_size(256)
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
    
    if (use_shared_memory && params.input_size <= 1024u) {
        // 使用共享内存优化小型矩阵乘法
        // 预加载输入数据到共享内存
        let tile_size = 32u;
        let tiles = (params.input_size + tile_size - 1u) / tile_size;
        
        for (var tile = 0u; tile < tiles; tile++) {
            let tile_start = tile * tile_size;
            let tile_end = min(tile_start + tile_size, params.input_size);
            let tile_size_actual = tile_end - tile_start;
            
            // 协作加载到共享内存
            for (var i = lid; i < tile_size_actual; i += 256u) {
                shared_input[i] = input[input_offset + tile_start + i];
                shared_weights[i] = weights[weight_offset + tile_start + i];
            }
            
            workgroupBarrier();
            
            // 计算部分点积
            for (var i = 0u; i < tile_size_actual; i++) {
                sum += shared_input[i] * shared_weights[i];
            }
            
            workgroupBarrier();
        }
    } else if (use_vectorization) {
        // 使用向量化的点积计算
        sum += dot_product_optimized_inline(input_offset, weight_offset, params.input_size);
    } else {
        // 标准计算路径
        for (var i = 0u; i < params.input_size; i++) {
            sum += input[input_offset + i] * weights[weight_offset + i];
        }
    }
    
    // 应用激活函数
    let activation_type = (params.optimization_flags >> 8u) & 0x0Fu;
    let result = activate(sum, activation_type);
    
    output[batch_idx * params.output_size + output_idx] = result;
}