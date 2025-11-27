// 自定义 is_finite 函数 (WGSL 标准库不提供此函数)
fn is_finite(x: f32) -> bool {
    // 检查 NaN: NaN != NaN 总是成立
    let is_nan = (x != x);
    // 检查无穷大: 使用安全阈值 (f32 最大值 ~3.4e38)
    let is_inf = (abs(x) > 1e30);
    return !is_nan && !is_inf;
}

// 数值稳定的激活函数
fn sigmoid_stable(x: f32) -> f32 {
    // 对输入进行裁剪以防止数值溢出
    let clipped_x = clamp(x, -20.0, 20.0);
    return 1.0 / (1.0 + exp(-clipped_x));
}

fn tanh_stable(x: f32) -> f32 {
    // 使用稳定的tanh实现
    let clipped_x = clamp(x, -10.0, 10.0);
    return tanh(clipped_x);
}

struct LSTMParams {
    input_size: u32,
    hidden_size: u32,
    seq_len: u32,
    batch_size: u32,
    bidirectional: u32,
}

struct DispatchParams {
    t: u32,
    direction: u32,
}

// Buffer bindings
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> biases: array<f32>;
@group(0) @binding(4) var<storage, read_write> hidden_states: array<f32>;
@group(0) @binding(5) var<storage, read_write> cell_states: array<f32>;
@group(0) @binding(6) var<storage, read> params: LSTMParams;
@group(0) @binding(7) var<storage, read> dispatch: DispatchParams;
@group(0) @binding(8) var<storage, read> initial_hidden: array<f32>;
@group(0) @binding(9) var<storage, read> initial_cell: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let global_id = id.x;
    let num_directions = select(1u, 2u, params.bidirectional != 0u); // 安全的双向判断

    // 更严格的边界检查
    if (global_id >= params.batch_size ||
        dispatch.t >= params.seq_len ||
        dispatch.direction >= num_directions ||
        params.hidden_size > 512u) { // 保护固定大小数组
        return;
    }

    let batch_idx = global_id;
    let dir = dispatch.direction;
    let is_backward = (dir == 1u);

    // 计算实际时间步索引
    let actual_t = select(dispatch.t, params.seq_len - 1u - dispatch.t, is_backward);

    // 1. 获取当前输入
    let x_base = batch_idx * params.seq_len * params.input_size + actual_t * params.input_size;

    // 2. 获取前一时刻状态
    var h_prev: array<f32, 512>; // 固定大小数组 (最大512)
    var c_prev: array<f32, 512>;
    let state_size = params.seq_len * num_directions * params.hidden_size;
    let batch_state_offset = batch_idx * state_size;

    // 检查是否为序列起始点
    let is_sequence_start = (is_backward && (actual_t == params.seq_len - 1u))
                          || (!is_backward && (actual_t == 0u));

    if (is_sequence_start) {
        // 使用较小的初始值以提高数值稳定性
        let init_base = dir * params.batch_size * params.hidden_size + batch_idx * params.hidden_size;
        for (var i: u32 = 0u; i < params.hidden_size; i++) {
            // 使用较小的随机值初始化，而不是0
            let init_val = fract(sin(f32(i + batch_idx * 1000u)) * 43758.5453) * 0.01 - 0.005;
            h_prev[i] = initial_hidden[init_base + i] + init_val;
            c_prev[i] = initial_cell[init_base + i] + init_val * 0.1;
        }
    } else {
        // 获取前一时间步的实际索引
        let prev_actual_t = select(actual_t - 1u, actual_t + 1u, is_backward);

        // 读取前一状态，添加数值检查
        let state_base = batch_state_offset +
                        prev_actual_t * num_directions * params.hidden_size +
                        dir * params.hidden_size;

        for (var i: u32 = 0u; i < params.hidden_size; i++) {
            let h_val = hidden_states[state_base + i];
            let c_val = cell_states[state_base + i];

            // 使用自定义 is_finite 检查
            h_prev[i] = select(h_val, 0.0, !is_finite(h_val));
            c_prev[i] = select(c_val, 0.0, !is_finite(c_val));

            // 额外的范围限制
            h_prev[i] = clamp(h_prev[i], -10.0, 10.0);
            c_prev[i] = clamp(c_prev[i], -10.0, 10.0);
        }
    }

    // 计算门控
    let weight_dir_offset = dir * 4u * params.hidden_size * (params.input_size + params.hidden_size);
    let bias_dir_offset = dir * 4u * params.hidden_size;

    var h_t: array<f32, 512>;
    var c_t: array<f32, 512>;

    for (var h_idx: u32 = 0u; h_idx < params.hidden_size; h_idx++) {
        var gates = array<f32, 4>(0.0, 0.0, 0.0, 0.0);

        // 计算四个门，对每个门进行数值检查
        for (var gate: u32 = 0u; gate < 4u; gate++) {
            var sum = biases[bias_dir_offset + gate * params.hidden_size + h_idx];

            // 检查偏置是否为有限值
            if (!is_finite(sum)) {
                sum = 0.0;
            }

            // W_ih * x_t 部分
            let w_ih_offset = weight_dir_offset + gate * params.hidden_size * params.input_size;
            for (var i: u32 = 0u; i < params.input_size; i++) {
                let w_idx = w_ih_offset + h_idx * params.input_size + i;
                let weight_val = weights[w_idx];
                let input_val = input[x_base + i];

                // 检查权重和输入值
                if (is_finite(weight_val) && is_finite(input_val)) {
                    sum += weight_val * input_val;
                }
            }

            // W_hh * h_prev 部分
            let w_hh_base = weight_dir_offset + 4u * params.hidden_size * params.input_size;
            let w_hh_offset = w_hh_base + gate * params.hidden_size * params.hidden_size;
            for (var j: u32 = 0u; j < params.hidden_size; j++) {
                let w_idx = w_hh_offset + h_idx * params.hidden_size + j;
                let weight_val = weights[w_idx];
                let h_val = h_prev[j];

                // 检查权重和隐藏状态值
                if (is_finite(weight_val) && is_finite(h_val)) {
                    sum += weight_val * h_val;
                }
            }

            // 限制中间结果的范围
            sum = clamp(sum, -50.0, 50.0);
            gates[gate] = sum;
        }

        // 应用激活函数
        let i_gate = sigmoid_stable(gates[0]);
        let f_gate = sigmoid_stable(gates[1]);
        let g_val  = tanh_stable(gates[2]);
        let o_gate = sigmoid_stable(gates[3]);

        // 更新细胞状态，增加数值保护
        let prev_c = c_prev[h_idx];

        // 安全值选择
        let f_gate_safe = select(f_gate, 0.5, !is_finite(f_gate));
        let i_gate_safe = select(i_gate, 0.0, !is_finite(i_gate));
        let g_val_safe = select(g_val, 0.0, !is_finite(g_val));
        let prev_c_safe = select(prev_c, 0.0, !is_finite(prev_c));

        c_t[h_idx] = f_gate_safe * prev_c_safe + i_gate_safe * g_val_safe;

        // 细胞状态范围限制 (放宽范围)
        c_t[h_idx] = clamp(c_t[h_idx], -50.0, 50.0);

        // 检查细胞状态是否为有限值
        if (!is_finite(c_t[h_idx])) {
            c_t[h_idx] = 0.0;
        }

        // 更新隐藏状态
        let o_gate_safe = select(o_gate, 0.5, !is_finite(o_gate));
        let tanh_c = tanh_stable(c_t[h_idx]);
        h_t[h_idx] = o_gate_safe * tanh_c;

        // 隐藏状态范围限制
        h_t[h_idx] = clamp(h_t[h_idx], -10.0, 10.0);

        // 检查隐藏状态是否为有限值
        if (!is_finite(h_t[h_idx])) {
            h_t[h_idx] = 0.0;
        }
    }

    // 4. 写回结果，最终检查
    let write_base = batch_state_offset +
                    actual_t * num_directions * params.hidden_size +
                    dir * params.hidden_size;

    for (var i: u32 = 0u; i < params.hidden_size; i++) {
        // 最终数值检查
        let h_final = select(h_t[i], 0.0, !is_finite(h_t[i]));
        let c_final = select(c_t[i], 0.0, !is_finite(c_t[i]));
        
        hidden_states[write_base + i] = h_final;
        cell_states[write_base + i] = c_final;
        output[write_base + i] = h_final;
    }
}