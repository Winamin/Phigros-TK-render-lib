struct LSTMParams {
    input_size: u32,
    hidden_size: u32,
    seq_len: u32,
    batch_size: u32,
    bidirectional: u32,  // 0: unidirectional, 1: bidirectional
}

struct DispatchParams {
    t: u32,             // Current timestep (0 to seq_len-1)
    direction: u32,     // 0: forward, 1: backward
}

// Buffer bindings
@group(0) @binding(0) var<storage, read> input: array<f32>;               // [batch, seq_len, input_size]
@group(0) @binding(1) var<storage, read> weights: array<f32>;             // [num_directions, 4*hidden, input+hidden]
@group(0) @binding(2) var<storage, read_write> output: array<f32>;       // [batch, seq_len, num_dirs, hidden]
@group(0) @binding(3) var<storage, read> biases: array<f32>;             // [num_directions, 4*hidden]
@group(0) @binding(4) var<storage, read_write> hidden_states: array<f32>; // [batch, seq_len, num_dirs, hidden]
@group(0) @binding(5) var<storage, read_write> cell_states: array<f32>;   // [batch, seq_len, num_dirs, hidden]
@group(0) @binding(6) var<storage, read> params: LSTMParams;
@group(0) @binding(7) var<storage, read> dispatch: DispatchParams;

// 激活函数
fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let global_id = id.x;
    let num_directions = params.bidirectional + 1u;

    // 验证方向有效性
    if (params.bidirectional == 0u && dispatch.direction != 0u) {
        return; // 单向模式只处理方向0
    }

    // 验证batch索引
    if (global_id >= params.batch_size) {
        return;
    }
    let batch_idx = global_id;
    let dir = dispatch.direction;
    let is_backward = (dir == 1u);

    // 计算实际时间步索引
    let actual_t = select(
        dispatch.t,               // forward: t
        params.seq_len - 1u - dispatch.t, // backward: seq_len-1-t
        is_backward
    );

    // 添加时间步越界检查
    if (actual_t >= params.seq_len) {
        return;
    }

    // 1. 获取当前输入 x_t [input_size]
    let x_base = batch_idx * params.seq_len * params.input_size + actual_t * params.input_size;

    // 2. 获取前一时刻状态 (h_prev, c_prev) [hidden_size]
    var h_prev: array<f32, 1024>; // 支持最大 hidden_size=1024
    var c_prev: array<f32, 1024>;

    // 初始化为0
    for (var i = 0u; i < params.hidden_size; i++) {
        h_prev[i] = 0.0;
        c_prev[i] = 0.0;
    }

    // 处理非首时间步
    if (dispatch.t > 0u) {
        var prev_actual_t: u32;
        if (is_backward) {
            // 反向LSTM：前一时间步是当前+1
            prev_actual_t = actual_t + 1u;
        } else {
            // 正向LSTM：前一时间步是当前-1
            prev_actual_t = actual_t - 1u;
        }

        // 读取前一状态
        let state_base = batch_idx * params.seq_len * num_directions * params.hidden_size
                       + prev_actual_t * num_directions * params.hidden_size
                       + dir * params.hidden_size;

        for (var i = 0u; i < params.hidden_size; i++) {
            h_prev[i] = hidden_states[state_base + i];
            c_prev[i] = cell_states[state_base + i];
        }
    }

    // 3. 计算门控
    let weight_dir_offset = dir * (4u * params.hidden_size * params.input_size +
                                 4u * params.hidden_size * params.hidden_size);
    let bias_dir_offset = dir * 4u * params.hidden_size;

    var h_t: array<f32, 1024>;
    var c_t: array<f32, 1024>;

    // 为每个隐藏单元计算
    for (var h_idx = 0u; h_idx < params.hidden_size; h_idx++) {
        var gates = array<f32, 4>(0.0, 0.0, 0.0, 0.0);

        // 计算四个门 (input, forget, cell, output)
        for (var gate = 0u; gate < 4u; gate++) {
            // 从偏置开始
            var sum = biases[bias_dir_offset + gate * params.hidden_size + h_idx];

            // W_ih * x_t (输入权重部分)
            let w_ih_offset = weight_dir_offset;
            for (var i = 0u; i < params.input_size; i++) {
                let w_idx = w_ih_offset
                          + (gate * params.hidden_size + h_idx) * params.input_size
                          + i;
                sum += weights[w_idx] * input[x_base + i];
            }

            // W_hh * h_prev (隐藏层权重部分)
            let w_hh_offset = weight_dir_offset + 4u * params.hidden_size * params.input_size;
            for (var j = 0u; j < params.hidden_size; j++) {
                let w_idx = w_hh_offset
                          + (gate * params.hidden_size + h_idx) * params.hidden_size
                          + j;
                sum += weights[w_idx] * h_prev[j];
            }

            gates[gate] = sum;
        }

        // 应用激活函数
        let i_gate = sigmoid(gates[0]); // input gate
        let f_gate = sigmoid(gates[1]); // forget gate
        let g_val  = tanh(gates[2]);    // cell candidate
        let o_gate = sigmoid(gates[3]); // output gate

        // 更新细胞状态
        c_t[h_idx] = f_gate * c_prev[h_idx] + i_gate * g_val;

        // 更新隐藏状态
        h_t[h_idx] = o_gate * tanh(c_t[h_idx]);
    }

    // 4. 写回结果
    let write_base = batch_idx * params.seq_len * num_directions * params.hidden_size
                   + actual_t * num_directions * params.hidden_size
                   + dir * params.hidden_size;

    for (var i = 0u; i < params.hidden_size; i++) {
        hidden_states[write_base + i] = h_t[i];
        cell_states[write_base + i] = c_t[i];
        output[write_base + i] = h_t[i];
    }
}