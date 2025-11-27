struct ConcatParams {
    input_size: u32,      // 每个输入的大小
    output_size: u32,     // 总输出大小
    batch_size: u32,      // 批大小
    num_inputs: u32,      // 输入数量
};

// 输入数据布局：
// 所有输入数据都按顺序存储在input缓冲区中
// [input1, input2, input3, ...]
// output: 拼接后的输出

@group(0) @binding(0) var<storage, read> params: ConcatParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let global_id = id.x;
    
    // 确保不越界
    if (global_id >= params.output_size * params.batch_size) {
        return;
    }
    
    // 计算在拼接数据中的位置
    let input_offset = global_id % params.output_size;
    let batch_offset = (global_id / params.output_size) * params.input_size * params.num_inputs;
    
    // 从对应的输入位置读取数据
    var value = 0.0;
    let src_idx = batch_offset + input_offset;
    if (src_idx < arrayLength(&input)) {
        value = input[src_idx];
    }
    
    // 写入输出
    output[global_id] = value;
}
