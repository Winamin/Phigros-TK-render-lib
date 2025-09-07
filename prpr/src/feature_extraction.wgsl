// feature_extraction.wgsl
struct Note {
    position_x: f32,
    position_y: f32,
    time: f32,
    kind: u32,      // NoteKind
    judge: u32,     // JudgeStatus
    duration: f32,
    difficulty: f32,
};

struct BpmPoint {
    time: f32,
    bpm: f32,
};

struct InputParams {
    num_notes: u32,
    num_bpm_points: u32,
    context_window: u32,
    total_features: u32,  // 28
};

@group(0) @binding(0) var<storage, read> notes: array<Note>;
@group(0) @binding(1) var<storage, read> bpm_points: array<BpmPoint>;
@group(0) @binding(2) var<storage, read> params: InputParams;
@group(0) @binding(3) var<storage, read_write> features: array<f32>;

// 计算两点之间的距离
fn distance(a: vec2<f32>, b: vec2<f32>) -> f32 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

// 获取时间点的BPM
fn get_bpm_at_time(time: f32) -> f32 {
    if (params.num_bpm_points == 0) {
        return 120.0;  // 默认BPM
    }

    // 简单线性查找 - 实际应用中可能需要二分查找
    var bpm = 120.0;
    for (var i = 0u; i < params.num_bpm_points - 1u; i = i + 1) {
        if (time >= bpm_points[i].time && time < bpm_points[i + 1].time) {
            return bpm_points[i].bpm;
        }
    }
    return bpm_points[params.num_bpm_points - 1u].bpm;
}

// 位置特征: [left_ratio, right_ratio, center_distance, spread]
fn extract_position_features(start_idx: u32, end_idx: u32) -> array<f32, 4> {
    var left_count = 0u;
    var right_count = 0u;
    var total_x = 0.0;
    var min_x = 100.0;
    var max_x = -100.0;

    for (var i = start_idx; i < end_idx; i = i + 1) {
        let x = notes[i].position_x;
        total_x = total_x + x;

        if (x < 0.0) {
            left_count = left_count + 1u;
        } else {
            right_count = right_count + 1u;
        }

        min_x = min(min_x, x);
        max_x = max(max_x, x);
    }

    let total = end_idx - start_idx;
    if (total == 0u) {
        return array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    }

    let left_ratio = f32(left_count) / f32(total);
    let right_ratio = f32(right_count) / f32(total);
    let center_distance = abs(total_x / f32(total));  // 距离中心的距离
    let spread = max_x - min_x;

    return array<f32, 4>(left_ratio, right_ratio, center_distance, spread);
}

// 时间特征: [tempo, density, variance, acceleration, jerk, rhythm_consistency]
fn extract_temporal_features(start_idx: u32, end_idx: u32) -> array<f32, 6> {
    if (end_idx - start_idx < 2u) {
        return array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }

    var total_time_diff = 0.0;
    var min_time_diff = 100.0;
    var max_time_diff = -100.0;
    var total_acceleration = 0.0;
    var rhythm_score = 0.0;
    var prev_time_diff = 0.0;

    for (var i = start_idx + 1u; i < end_idx; i = i + 1) {
        let time_diff = notes[i].time - notes[i - 1u].time;
        total_time_diff = total_time_diff + time_diff;

        min_time_diff = min(min_time_diff, time_diff);
        max_time_diff = max(max_time_diff, time_diff);

        if (i > start_idx + 1u) {
            let acceleration = time_diff - prev_time_diff;
            total_acceleration = total_acceleration + abs(acceleration);

            // 节奏一致性评分 (越接近等间距，分数越高)
            if (abs(acceleration) < 0.02) {
                rhythm_score = rhythm_score + 1.0;
            }
        }

        prev_time_diff = time_diff;
    }

    let total = f32(end_idx - start_idx - 1u);
    let avg_time_diff = total_time_diff / total;
    let density = 1.0 / max(avg_time_diff, 0.001);
    let variance = (max_time_diff - min_time_diff) / max(avg_time_diff, 0.001);
    let acceleration = total_acceleration / total;

    return array<f32, 6>(
        60.0 / max(avg_time_diff, 0.001),  // BPM
        density,
        variance,
        acceleration,
        0.0,  // 可以添加jerk计算
        rhythm_score / total
    );
}

// 模式特征: [alternating, stream, chord, jack]
fn extract_pattern_features(start_idx: u32, end_idx: u32) -> array<f32, 4> {
    if (end_idx - start_idx < 2u) {
        return array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    }

    var alternating_score = 0.0;
    var stream_score = 0.0;
    var chord_score = 0.0;
    var jack_score = 0.0;
    var consecutive_direction = 0.0;
    var consecutive_same_x = 0u;
    var max_simultaneous = 1u;

    // 检测交替模式
    for (var i = start_idx + 1u; i < end_idx; i = i + 1) {
        let prev_x = notes[i - 1u].position_x;
        let curr_x = notes[i].position_x;

        if ((prev_x < 0.0 && curr_x > 0.0) || (prev_x > 0.0 && curr_x < 0.0)) {
            consecutive_direction = consecutive_direction + 1.0;
            alternating_score = max(alternating_score, consecutive_direction);
        } else {
            consecutive_direction = 0.0;
        }
    }

    // 检测流模式
    consecutive_direction = 0.0;
    for (var i = start_idx + 1u; i < end_idx; i = i + 1) {
        let prev_x = notes[i - 1u].position_x;
        let curr_x = notes[i].position_x;

        if ((prev_x < 0.0 && curr_x < 0.0) || (prev_x > 0.0 && curr_x > 0.0)) {
            consecutive_direction = consecutive_direction + 1.0;
            stream_score = max(stream_score, consecutive_direction);
        } else {
            consecutive_direction = 0.0;
        }
    }

    // 检测和弦模式
    for (var i = start_idx; i < end_idx; i = i + 1) {
        var simultaneous_count = 1u;
        for (var j = i + 1u; j < end_idx; j = j + 1) {
            if (abs(notes[j].time - notes[i].time) < 0.05) {
                simultaneous_count = simultaneous_count + 1u;
            } else {
                break;
            }
        }
        max_simultaneous = max(max_simultaneous, simultaneous_count);
    }

    if (max_simultaneous >= 2u) {
        chord_score = f32(max_simultaneous);
    }

    // 检测Jack模式
    for (var i = start_idx; i < end_idx - 2u; i = i + 1) {
        if (abs(notes[i].time - notes[i + 1u].time) < 0.15 &&
            abs(notes[i + 1u].time - notes[i + 2u].time) < 0.15) {
            jack_score = jack_score + 1.0;
        }
    }

    // 归一化
    alternating_score = min(alternating_score / 5.0, 1.0);
    stream_score = min(stream_score / 8.0, 1.0);
    chord_score = min(chord_score / 4.0, 1.0);
    jack_score = min(jack_score / 3.0, 1.0);

    return array<f32, 4>(alternating_score, stream_score, chord_score, jack_score);
}

// 速度特征: [avg_velocity, max_velocity, avg_acceleration, max_acceleration]
fn extract_velocity_features(start_idx: u32, end_idx: u32) -> array<f32, 4> {
    if (end_idx - start_idx < 2u) {
        return array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    }

    var total_velocity = 0.0;
    var max_velocity = 0.0;
    var total_acceleration = 0.0;
    var max_acceleration = 0.0;
    var prev_velocity = 0.0;

    for (var i = start_idx + 1u; i < end_idx; i = i + 1) {
        let time_diff = notes[i].time - notes[i - 1u].time;
        if (time_diff < 0.01) {  // 防止除以接近0
            continue;
        }

        let distance = distance(
            vec2<f32>(notes[i - 1u].position_x, notes[i - 1u].position_y),
            vec2<f32>(notes[i].position_x, notes[i].position_y)
        );

        let velocity = distance / time_diff;
        total_velocity = total_velocity + velocity;

        if (velocity > max_velocity) {
            max_velocity = velocity;
        }

        if (i > start_idx + 1u) {
            let acceleration = velocity - prev_velocity;
            total_acceleration = total_acceleration + abs(acceleration);

            if (abs(acceleration) > max_acceleration) {
                max_acceleration = abs(acceleration);
            }
        }

        prev_velocity = velocity;
    }

    let total = f32(end_idx - start_idx - 1u);
    let avg_velocity = total_velocity / total;
    let avg_acceleration = total_acceleration / total;

    // 限制最大值
    let MAX_VELOCITY = 40.0;
    let MAX_ACCELERATION = 100.0;

    return array<f32, 4>(
        min(avg_velocity, MAX_VELOCITY),
        min(max_velocity, MAX_VELOCITY),
        min(avg_acceleration, MAX_ACCELERATION),
        min(max_acceleration, MAX_ACCELERATION)
    );
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let note_idx = global_id.x;
    if (note_idx >= params.num_notes) {
        return;
    }

    // 计算上下文窗口
    let start_idx = max(0u, note_idx - params.context_window / 2u);
    let end_idx = min(params.num_notes, note_idx + params.context_window / 2u + 1u);

    // 提取所有特征
    let position_features = extract_position_features(start_idx, end_idx);
    let temporal_features = extract_temporal_features(start_idx, end_idx);
    let pattern_features = extract_pattern_features(start_idx, end_idx);
    let velocity_features = extract_velocity_features(start_idx, end_idx);

    // 合并特征 (28维)
    let feature_idx = note_idx * params.total_features;

    // 位置特征 (0-3)
    for (var i = 0u; i < 4u; i = i + 1) {
        features[feature_idx + i] = position_features[i];
    }

    // 时间特征 (4-9)
    for (var i = 0u; i < 6u; i = i + 1) {
        features[feature_idx + 4u + i] = temporal_features[i];
    }

    // 模式特征 (10-13)
    for (var i = 0u; i < 4u; i = i + 1) {
        features[feature_idx + 10u + i] = pattern_features[i];
    }

    // 速度特征 (18-21)
    for (var i = 0u; i < 4u; i = i + 1) {
        features[feature_idx + 18u + i] = velocity_features[i];
    }

    // 空间特征 (22-27) - 为简化，这里用位置特征代替
    features[feature_idx + 22u] = position_features[3];  // spread
    features[feature_idx + 23u] = 0.0;  // 可以添加更多空间特征
    features[feature_idx + 24u] = 0.0;
    features[feature_idx + 25u] = 0.0;
    features[feature_idx + 26u] = 0.0;
    features[feature_idx + 27u] = 0.0;

    // 特征归一化 - 在GPU上完成
    for (var i = 0u; i < params.total_features; i = i + 1) {
        let idx = feature_idx + i;
        switch (i) {
            case 0u, 1u, 2u, 3u: {  // 位置特征
                features[idx] = clamp(features[idx], -2.0, 2.0);
                break;
            }
            case 4u, 5u, 6u, 7u, 8u, 9u: {  // 时间特征
                features[idx] = clamp(features[idx], -10.0, 10.0);
                break;
            }
            case 10u, 11u, 12u, 13u: {  // 模式特征
                features[idx] = clamp(features[idx], -5.0, 5.0);
                break;
            }
            case 18u, 19u, 20u, 21u: {  // 速度特征
                features[idx] = clamp(features[idx], -50.0, 50.0);
                break;
            }
            case 22u, 23u, 24u, 25u, 26u, 27u: {  // 空间特征
                features[idx] = clamp(features[idx], -10.0, 10.0);
                break;
            }
            default: {
                features[idx] = clamp(features[idx], -5.0, 5.0);
            }
        }
    }
}