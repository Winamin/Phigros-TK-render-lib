use std::collections::HashMap;
use crate::core::Note;
use crate::core::note::Hand;
use crate::core::NoteKind;

pub fn assign_hands(notes: &mut [Note], rotation: f32) {
    // 边界检查 - 防止空数组崩溃
    if notes.is_empty() {
        return;
    }

    // === 核心参数配置 ===
    const COMFORT_ZONE_RADIUS: f32 = 0.25;
    const MAX_COMFORTABLE_REACH: f32 = 0.45;
    const FATIGUE_RECOVERY_RATE: f32 = 0.08;
    const HAND_BALANCE_WEIGHT: f32 = 0.3;
    const FLOW_CONTINUITY_BONUS: f32 = 0.4;
    const SIMULTANEOUS_NOTE_TOLERANCE: f32 = 0.03;
    const CROSS_HAND_PENALTY: f32 = 0.35;
    const SAME_FINGER_PENALTY: f32 = 0.4;

    // 手指灵活度配置
    const FINGER_AGILITY: [f32; 4] = [0.15, 0.45, 0.60, 0.35]; // 中指,食指,食指,中指
    const FINGER_COMFORT_ZONES: [f32; 4] = [0.28, 0.15, 0.15, 0.28];
    const FINGER_MAX_REACH: [f32; 4] = [0.4, 0.5, 0.5, 0.4];

    /// 手指状态
    #[derive(Clone)]
    struct FingerState {
        id: usize,
        hand: Hand,
        position: Vector2,
        comfort_center: Vector2,
        max_reach: f32,
        agility: f32,
        current_fatigue: f32,
        last_action_time: f32,
        success_rate: f32,
        recent_workload: f32,
    }

    /// 音符分析数据
    #[derive(Clone)]
    struct ProcessedNote {
        index: usize,
        time: f32,
        position: Vector2,
        velocity: Vector2,
        difficulty: f32,
        natural_hand: Hand,
        density_score: f32,
        is_simultaneous: bool,
        pattern_complexity: f32,
    }

    /// 分配结果
    struct Assignment {
        note_index: usize,
        finger_id: usize,
        cost: f32,
        confidence: f32,
    }

    /// 场景检测器 - 自动识别最佳指法模式
    struct ScenarioAnalyzer {
        note_count: usize,
        max_simultaneous: usize,
        density_peaks: Vec<f32>,
        complexity_score: f32,
        hand_separation_ratio: f32,
    }

    impl ScenarioAnalyzer {
        fn analyze(notes: &[Note]) -> Self {
            let mut analyzer = Self {
                note_count: notes.len(),
                max_simultaneous: 1,
                density_peaks: Vec::new(),
                complexity_score: 0.0,
                hand_separation_ratio: 0.0,
            };

            analyzer.calculate_metrics(notes);
            analyzer
        }

        fn calculate_metrics(&mut self, notes: &[Note]) {
            if notes.is_empty() { return; }

            // 计算密度峰值
            let mut time_groups: Vec<Vec<usize>> = Vec::new();
            let mut current_group = Vec::new();
            let mut last_time = notes[0].time - 1.0;

            for (i, note) in notes.iter().enumerate() {
                if (note.time - last_time).abs() <= SIMULTANEOUS_NOTE_TOLERANCE {
                    current_group.push(i);
                } else {
                    if !current_group.is_empty() {
                        time_groups.push(current_group.clone());
                    }
                    current_group = vec![i];
                }
                last_time = note.time;
            }
            if !current_group.is_empty() {
                time_groups.push(current_group);
            }

            // 分析同时音符
            self.max_simultaneous = time_groups.iter()
                .map(|group| group.len())
                .max()
                .unwrap_or(1);

            // 计算复杂度
            let mut total_complexity = 0.0;
            for note in notes {
                total_complexity += match note.kind {
                    NoteKind::Click => 1.0,
                    NoteKind::Drag => 1.4,
                    NoteKind::Flick => 1.6,
                    NoteKind::Hold { .. } => 1.8,
                };
            }
            self.complexity_score = total_complexity / notes.len() as f32;

            // 分析左右分布
            let mut left_count = 0;
            let mut right_count = 0;
            for note in notes {
                let x = note.object.translation.0.now();
                if x < 0.0 { left_count += 1; } else { right_count += 1; }
            }
            let total = (left_count + right_count) as f32;
            if total > 0.0 {
                self.hand_separation_ratio = (left_count.min(right_count) as f32 * 2.0) / total;
            }
        }

        fn recommend_finger_mode(&self) -> FingerMode {
            // 智能推荐指法模式
            if self.complexity_score > 1.4 || self.max_simultaneous >= 3 {
                FingerMode::FourFinger
            } else if self.hand_separation_ratio > 0.6 {
                FingerMode::TwoFingerBalanced
            } else {
                FingerMode::TwoFingerDominant
            }
        }
    }

    #[derive(Clone, Copy, Debug)]
    enum FingerMode {
        TwoFingerDominant,   // 主手为主的双指
        TwoFingerBalanced,   // 平衡双指
        FourFinger,          // 四指模式
    }

    // === 主算法流程 ===

    // 场景分析
    let scenario = ScenarioAnalyzer::analyze(notes);
    let finger_mode = scenario.recommend_finger_mode();

    // 手指配置
    let mut fingers = initialize_fingers(finger_mode, rotation);

    // 音符预处理
    let processed_notes = preprocess_notes(notes, rotation);

    // 主分配循环
    let assignments = assign_notes_optimally(&processed_notes, &mut fingers, finger_mode);

    // 应用结果到原始音符数组
    apply_assignments_safely(notes, &assignments, &fingers);

    // === 实现函数 ===

    fn initialize_fingers(mode: FingerMode, rotation: f32) -> Vec<FingerState> {
        let rad = rotation.to_radians();
        let cos_r = rad.cos();
        let sin_r = rad.sin();

        match mode {
            FingerMode::TwoFingerDominant | FingerMode::TwoFingerBalanced => {
                vec![
                    // 左手食指
                    FingerState {
                        id: 0,
                        hand: Hand::Left,
                        position: Vector2::new(-0.25, 0.0),
                        comfort_center: Vector2::new(-0.3, 0.0),
                        max_reach: FINGER_MAX_REACH[1],
                        agility: FINGER_AGILITY[1],
                        current_fatigue: 0.0,
                        last_action_time: -1.0,
                        success_rate: 1.0,
                        recent_workload: 0.0,
                    },
                    // 右手食指
                    FingerState {
                        id: 1,
                        hand: Hand::Right,
                        position: Vector2::new(0.25, 0.0),
                        comfort_center: Vector2::new(0.3, 0.0),
                        max_reach: FINGER_MAX_REACH[2],
                        agility: FINGER_AGILITY[2],
                        current_fatigue: 0.0,
                        last_action_time: -1.0,
                        success_rate: 1.0,
                        recent_workload: 0.0,
                    },
                ]
            },
            FingerMode::FourFinger => {
                vec![
                    // 左中指
                    FingerState {
                        id: 0,
                        hand: Hand::Left,
                        position: Vector2::new(-0.35, -0.05),
                        comfort_center: Vector2::new(-0.4, 0.0),
                        max_reach: FINGER_MAX_REACH[0],
                        agility: FINGER_AGILITY[0],
                        current_fatigue: 0.0,
                        last_action_time: -1.0,
                        success_rate: 0.95,
                        recent_workload: 0.0,
                    },
                    // 左食指
                    FingerState {
                        id: 1,
                        hand: Hand::Left,
                        position: Vector2::new(-0.15, 0.05),
                        comfort_center: Vector2::new(-0.2, 0.0),
                        max_reach: FINGER_MAX_REACH[1],
                        agility: FINGER_AGILITY[1],
                        current_fatigue: 0.0,
                        last_action_time: -1.0,
                        success_rate: 1.0,
                        recent_workload: 0.0,
                    },
                    // 右食指
                    FingerState {
                        id: 2,
                        hand: Hand::Right,
                        position: Vector2::new(0.15, -0.05),
                        comfort_center: Vector2::new(0.2, 0.0),
                        max_reach: FINGER_MAX_REACH[2],
                        agility: FINGER_AGILITY[2],
                        current_fatigue: 0.0,
                        last_action_time: -1.0,
                        success_rate: 1.0,
                        recent_workload: 0.0,
                    },
                    // 右中指
                    FingerState {
                        id: 3,
                        hand: Hand::Right,
                        position: Vector2::new(0.35, 0.05),
                        comfort_center: Vector2::new(0.4, 0.0),
                        max_reach: FINGER_MAX_REACH[3],
                        agility: FINGER_AGILITY[3],
                        current_fatigue: 0.0,
                        last_action_time: -1.0,
                        success_rate: 0.95,
                        recent_workload: 0.0,
                    },
                ]
            }
        }
    }

    fn preprocess_notes(notes: &[Note], rotation: f32) -> Vec<ProcessedNote> {
        let rad = rotation.to_radians();
        let cos_r = rad.cos();
        let sin_r = rad.sin();

        let mut processed: Vec<ProcessedNote> = notes.iter()
            .enumerate()
            .map(|(i, note)| {
                // 位置获取
                let raw_x = note.object.translation.0.now();
                let raw_y = note.object.translation.1.now();

                // 旋转变换
                let rotated_pos = Vector2::new(
                    raw_x * cos_r - raw_y * sin_r,
                    raw_x * sin_r + raw_y * cos_r
                );

                // 速度计算
                let velocity = calculate_velocity_safely(note, cos_r, sin_r);

                // 难度评估
                let difficulty = match note.kind {
                    NoteKind::Click => 1.0,
                    NoteKind::Drag => 1.4,
                    NoteKind::Flick => 1.6,
                    NoteKind::Hold { .. } => 1.8,
                };

                ProcessedNote {
                    index: i,
                    time: note.time,
                    position: rotated_pos,
                    velocity,
                    difficulty,
                    natural_hand: if rotated_pos.x < 0.0 { Hand::Left } else { Hand::Right },
                    density_score: 1.0, // 后续计算
                    is_simultaneous: false, // 后续标记
                    pattern_complexity: difficulty,
                }
            })
            .collect();

        // 按时间排序
        processed.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap_or(std::cmp::Ordering::Equal));

        // 计算密度分数
        for i in 0..processed.len() {
            let mut nearby_count = 0;
            let current_time = processed[i].time;

            for j in 0..processed.len() {
                if i != j && (processed[j].time - current_time).abs() < 0.3 {
                    nearby_count += 1;
                }
            }

            processed[i].density_score = (nearby_count as f32 * 0.15 + 1.0).min(2.5);
        }

        // 标记同时音符
        for i in 0..processed.len() {
            for j in (i+1)..processed.len() {
                if (processed[j].time - processed[i].time).abs() <= SIMULTANEOUS_NOTE_TOLERANCE {
                    processed[i].is_simultaneous = true;
                    processed[j].is_simultaneous = true;
                } else {
                    break;
                }
            }
        }

        processed
    }

    fn calculate_velocity_safely(note: &Note, cos_r: f32, sin_r: f32) -> Vector2 {
        const EPS: f32 = 0.001;
        let t = note.time;

        // 创建临时拷贝来计算导数
        let mut temp_x = note.object.translation.0.clone();
        let mut temp_y = note.object.translation.1.clone();

        // 安全的导数计算
        temp_x.set_time(t + EPS);
        let x_plus = temp_x.now();
        temp_x.set_time(t.max(EPS) - EPS); // 防止负时间
        let x_minus = temp_x.now();
        let dx = (x_plus - x_minus) / (2.0 * EPS);

        temp_y.set_time(t + EPS);
        let y_plus = temp_y.now();
        temp_y.set_time(t.max(EPS) - EPS);
        let y_minus = temp_y.now();
        let dy = (y_plus - y_minus) / (2.0 * EPS);

        // 旋转速度向量
        Vector2::new(
            dx * cos_r - dy * sin_r,
            dx * sin_r + dy * cos_r
        )
    }

    fn assign_notes_optimally(
        processed_notes: &[ProcessedNote],
        fingers: &mut [FingerState],
        mode: FingerMode
    ) -> Vec<Assignment> {
        let mut assignments = Vec::new();
        let mut hand_usage = [0.0f32; 2];

        for pnote in processed_notes {
            let mut best_finger = 0;
            let mut best_cost = f32::INFINITY;

            for (finger_idx, finger) in fingers.iter().enumerate() {
                let cost = calculate_assignment_cost(pnote, finger, &hand_usage, mode);

                if cost < best_cost {
                    best_cost = cost;
                    best_finger = finger_idx;
                }
            }

            assignments.push(Assignment {
                note_index: pnote.index,
                finger_id: best_finger,
                cost: best_cost,
                confidence: calculate_confidence(best_cost),
            });

            // 更新手指状态
            if best_finger < fingers.len() { // 边界检查
                let finger = &mut fingers[best_finger];
                let movement_distance = (pnote.position - finger.position).magnitude();

                finger.position = pnote.position;
                finger.last_action_time = pnote.time;
                finger.current_fatigue += 0.05 * pnote.difficulty;
                finger.recent_workload = (finger.recent_workload * 0.8 + pnote.difficulty * 0.2).min(2.0);

                // 更新手部使用统计
                let hand_idx = finger.hand as usize;
                if hand_idx < hand_usage.len() { // 边界检查
                    hand_usage[hand_idx] += pnote.difficulty;
                }
            }
        }

        assignments
    }

    fn calculate_assignment_cost(
        note: &ProcessedNote,
        finger: &FingerState,
        hand_usage: &[f32; 2],
        _mode: FingerMode
    ) -> f32 {
        let mut cost = 0.0;

        // 基础距离成本
        let distance = (note.position - finger.position).magnitude();
        let normalized_distance = (distance / finger.max_reach).min(2.0);
        cost += normalized_distance * normalized_distance * 1.5;

        // 舒适区奖励
        let comfort_distance = (note.position - finger.comfort_center).magnitude();
        if comfort_distance <= COMFORT_ZONE_RADIUS {
            cost *= 0.7; // 舒适区内操作奖励
        }

        // 疲劳惩罚
        cost += finger.current_fatigue * (1.0 + note.difficulty * 0.2);

        // 同指快速连击惩罚
        let time_since_last = note.time - finger.last_action_time;
        if time_since_last > 0.0 && time_since_last < 0.15 {
            let penalty_factor = (0.15 - time_since_last) / 0.15;
            cost += SAME_FINGER_PENALTY * penalty_factor * note.difficulty;
        }

        // 交叉手惩罚
        if finger.hand != note.natural_hand {
            cost += CROSS_HAND_PENALTY * (1.0 + distance * 0.4);
        }

        // 手部平衡考虑
        let total_usage = hand_usage[0] + hand_usage[1];
        if total_usage > 0.1 {
            let hand_idx = finger.hand as usize;
            let current_usage = hand_usage[hand_idx] / total_usage;
            if current_usage > 0.65 {
                cost *= 1.0 + (current_usage - 0.5) * HAND_BALANCE_WEIGHT;
            }
        }

        // 手指灵活度调整
        cost /= finger.agility;
        cost /= finger.success_rate;

        if note.density_score > 1.5 {
            cost *= 0.85; // 密集区段中稍微降低成本，鼓励连续操作
        }

        cost
    }

    fn calculate_confidence(cost: f32) -> f32 {
        // 将成本转换为置信度 (0.0 - 1.0)
        (1.0 / (1.0 + cost * 0.5)).min(1.0).max(0.0)
    }

    fn apply_assignments_safely(notes: &mut [Note], assignments: &[Assignment], fingers: &[FingerState]) {
        for assignment in assignments {
            // 严格边界检查
            if assignment.note_index < notes.len() && assignment.finger_id < fingers.len() {
                notes[assignment.note_index].hand = fingers[assignment.finger_id].hand;
            }
        }

        // 同时音符检测和标记
        mark_simultaneous_notes(notes);
    }

    fn mark_simultaneous_notes(notes: &mut [Note]) {
        let mut time_groups: Vec<Vec<usize>> = Vec::new();
        let mut current_group = Vec::new();
        let mut last_time = if notes.is_empty() { 0.0 } else { notes[0].time - 1.0 };

        for (i, note) in notes.iter().enumerate() {
            if current_group.is_empty() || (note.time - last_time).abs() <= SIMULTANEOUS_NOTE_TOLERANCE {
                current_group.push(i);
            } else {
                if current_group.len() > 1 {
                    time_groups.push(current_group.clone());
                }
                current_group = vec![i];
            }
            last_time = note.time;
        }

        if current_group.len() > 1 {
            time_groups.push(current_group);
        }

        // 标记同时音符
        for group in time_groups {
            if group.len() >= 2 {
                for &note_idx in &group {
                    if note_idx < notes.len() {
                        notes[note_idx].multiple_hint = true;
                    }
                }
            }
        }
    }


    #[derive(Clone, Copy, Debug)]
    struct Vector2 {
        x: f32,
        y: f32,
    }

    impl Vector2 {
        fn new(x: f32, y: f32) -> Self { Self { x, y } }
        fn magnitude(&self) -> f32 { (self.x * self.x + self.y * self.y).sqrt() }
    }

    impl std::ops::Sub for Vector2 {
        type Output = Vector2;
        fn sub(self, other: Vector2) -> Vector2 {
            Vector2::new(self.x - other.x, self.y - other.y)
        }
    }

    impl std::ops::Add for Vector2 {
        type Output = Vector2;
        fn add(self, other: Vector2) -> Vector2 {
            Vector2::new(self.x + other.x, self.y + other.y)
        }
    }

    impl std::ops::Mul<f32> for Vector2 {
        type Output = Vector2;
        fn mul(self, scalar: f32) -> Vector2 {
            Vector2::new(self.x * scalar, self.y * scalar)
        }
    }
}