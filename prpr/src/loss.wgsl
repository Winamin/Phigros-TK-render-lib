// loss.wgsl - 用 storage buffer 传入参数（只需替换此文件）
@group(0) @binding(0) var<storage, read> outputs: array<f32>;
@group(0) @binding(1) var<storage, read> targets: array<f32>;
@group(0) @binding(2) var<storage, read_write> losses: array<f32>;

// 把 params 当成 storage 的 float 数组传进来
// Rust 端如果传的是 [batch_size as f32, input_size as f32, output_size as f32]
// 那么这里按索引读取即可
@group(0) @binding(3) var<storage, read> params_buf: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;

    // 从 params_buf 中读取参数（注意 Rust 传的是 f32）
    let batchSizeF: f32 = params_buf[0];
    let outputSizeF: f32 = params_buf[2];

    let batchSize: u32 = u32(batchSizeF);
    let outputSize: u32 = u32(outputSizeF);

    if (index >= batchSize) {
        return;
    }

    var total_loss: f32 = 0.0;
    let start: u32 = index * outputSize;

    for (var i: u32 = 0u; i < outputSize; i = i + 1u) {
        let output = outputs[start + i];
        let target_val = targets[start + i];
        let diff = output - target_val;
        total_loss = total_loss + diff * diff;
    }

    losses[index] = total_loss;
}
