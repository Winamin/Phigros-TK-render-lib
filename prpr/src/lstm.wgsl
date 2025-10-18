struct LSTMParams {
    input_size: u32,
    hidden_size: u32,
    seq_len: u32,
    batch_size: u32,
    bidirectional: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> biases: array<f32>;
@group(0) @binding(4) var<storage, read_write> hidden_states: array<f32>;
@group(0) @binding(5) var<storage, read_write> cell_states: array<f32>;
@group(0) @binding(6) var<storage, read> params: LSTMParams;

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

fn tanh_approx(x: f32) -> f32 {
    return tanh(x);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let global_id = id.x;
    let total_output_size = params.hidden_size * params.seq_len * params.batch_size * (params.bidirectional + 1u);

    if (global_id >= total_output_size) {
        return;
    }

    let batch_idx = global_id / (params.hidden_size * params.seq_len * (params.bidirectional + 1u));
    let remaining = global_id % (params.hidden_size * params.seq_len * (params.bidirectional + 1u));
    let seq_dir_idx = remaining / params.hidden_size;
    let hidden_idx = remaining % params.hidden_size;

    let seq_idx = seq_dir_idx % params.seq_len;
    let is_backward = seq_dir_idx >= params.seq_len;

    // 计算输入索引
    let input_base_idx = batch_idx * params.seq_len * params.input_size;

    // 计算权重索引
    let weight_output_size = 4u * params.hidden_size;
    let weight_input_size = params.input_size + params.hidden_size;

    // 初始化隐藏状态和细胞状态
    var h_prev = 0.0;
    var c_prev = 0.0;

    // 计算时间步 - 使用select函数
    let actual_seq_idx = select(
        seq_idx,
        params.seq_len - 1u - seq_idx,
        is_backward
    );

    // 获取当前时间步的输入
    let x_t_idx = input_base_idx + actual_seq_idx * params.input_size;

    // 计算四个门的线性组合
    var ifgo = array<f32, 4>(0.0, 0.0, 0.0, 0.0);

    // 计算输入门、遗忘门、候选值、输出门
    for (var gate = 0u; gate < 4u; gate++) {
        var sum = biases[gate * params.hidden_size + hidden_idx];

        // 输入到隐藏的连接 (W_ih)
        for (var i = 0u; i < params.input_size; i++) {
            let weight_idx = (gate * params.hidden_size + hidden_idx) * weight_input_size + i;
            sum += weights[weight_idx] * input[x_t_idx + i];
        }

        // 隐藏到隐藏的连接 (W_hh)
        if (actual_seq_idx > 0u) {
            let prev_seq_idx = select(
                actual_seq_idx - 1u,
                actual_seq_idx + 1u,
                is_backward
            );
            let prev_h_idx = batch_idx * params.seq_len * params.hidden_size + prev_seq_idx * params.hidden_size;
            for (var i = 0u; i < params.hidden_size; i++) {
                let weight_idx = (gate * params.hidden_size + hidden_idx) * weight_input_size + params.input_size + i;
                sum += weights[weight_idx] * hidden_states[prev_h_idx + i];
            }
        }

        ifgo[gate] = sum;
    }

    // 应用激活函数
    let i_gate = sigmoid(ifgo[0]);
    let f_gate = sigmoid(ifgo[1]);
    let g_candidate = tanh_approx(ifgo[2]);
    let o_gate = sigmoid(ifgo[3]);

    // 获取前一时间步的细胞状态
    if (actual_seq_idx > 0u) {
        let prev_seq_idx = select(
            actual_seq_idx - 1u,
            actual_seq_idx + 1u,
            is_backward
        );
        let prev_c_idx = batch_idx * params.seq_len * params.hidden_size + prev_seq_idx * params.hidden_size;
        c_prev = cell_states[prev_c_idx + hidden_idx];
    }
    
    // 更新细胞状态
    let c_t = f_gate * c_prev + i_gate * g_candidate;
    
    // 更新隐藏状态
    let h_t = o_gate * tanh_approx(c_t);
    
    // 存储隐藏状态和细胞状态
    let h_idx = batch_idx * params.seq_len * params.hidden_size + actual_seq_idx * params.hidden_size + hidden_idx;
    let c_idx = batch_idx * params.seq_len * params.hidden_size + actual_seq_idx * params.hidden_size + hidden_idx;
    hidden_states[h_idx] = h_t;
    cell_states[c_idx] = c_t;
    
    // 输出结果
    let output_idx = batch_idx * params.seq_len * params.hidden_size * (params.bidirectional + 1u) + 
                     seq_dir_idx * params.hidden_size + hidden_idx;
    output[output_idx] = h_t;
}