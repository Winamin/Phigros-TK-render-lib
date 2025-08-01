use crate::core::{note::Hand, Note, Vector, NoteKind};
pub fn assign_hands(notes: &mut [Note], rotation: f32) {
    let config = crate::config::Config::default();

    /*
    如果 config 中的 hand_split 为 false，直接返回
    byd这里不写个边界检查直接panic了
    这里 notes 为空，直接返回
     */

    if !config.hand_split || notes.is_empty() {
        return;
    }

    const LANE_SPLIT: f32 = 0.0;
    const MAX_STRETCH_RADIUS: f32 = 0.38;
    const FATIGUE_DECAY_RATE: f32 = 0.03;
    const CROSS_HAND_PENALTY: f32 = 0.35; // 适中的交叉手惩罚
    const SAME_FINGER_PENALTY: f32 = 0.4;  // 降低同指惩罚
    const DENSITY_BONUS: f32 = 0.3;        // 密集区段奖励
    const COOPERATION_BONUS: f32 = 0.25;   // 协作奖励

    const FINGER_COMFORT_ZONES: [f32; 4] = [0.28, 0.12, 0.12, 0.28];
    const FINGER_DEXTERITY: [f32; 4] = [0.9, 1.0, 1.0, 0.9];
    const FINGER_COOPERATION: [f32; 4] = [0.65, 0.80, 1.0, 0.85]; // 协作系数

    #[derive(Clone, Copy)]
    struct FingerState {
        id: usize,
        hand: Hand,
        position: Vector,
        last_used_time: f32,
        fatigue: f32,
        travel_distance: f32,
        activity_level: f32,
        preferred_zone: f32,
        comfort_radius: f32,
        cooperation_score: f32,
    }

    #[derive(Clone)]
    struct NoteAssignment {
        note_idx: usize,
        finger_idx: usize,
        cost: f32,
        is_cross_hand: bool,
        in_dense_section: bool,
    }

    struct ProcessedNote {
        idx: usize,
        time: f32,
        position: Vector,
        natural_hand: Hand,
        velocity: Vector,
        complexity: f32,
        density_score: f32,
    }

    let rad = rotation.to_radians();
    let cos = rad.cos();
    let sin = rad.sin();

    let mut fingers = vec![
        // 左手食指
        FingerState {
            id: 0,
            hand: Hand::Left,
            position: Vector::new(-0.35, 0.0),
            last_used_time: -1.0,
            fatigue: 0.0,
            travel_distance: 0.0,
            activity_level: 0.0,
            preferred_zone: -0.35,
            comfort_radius: FINGER_COMFORT_ZONES[0],
            cooperation_score: 1.0,
        },
        // 左手中指
        FingerState {
            id: 1,
            hand: Hand::Left,
            position: Vector::new(-0.15, 0.02),
            last_used_time: -1.0,
            fatigue: 0.0,
            travel_distance: 0.0,
            activity_level: 0.0,
            preferred_zone: -0.15,
            comfort_radius: FINGER_COMFORT_ZONES[1],
            cooperation_score: 1.0,
        },
        // 右手食指
        FingerState {
            id: 2,
            hand: Hand::Right,
            position: Vector::new(0.15, -0.02),
            last_used_time: -1.0,
            fatigue: 0.0,
            travel_distance: 0.0,
            activity_level: 0.0,
            preferred_zone: 0.15,
            comfort_radius: FINGER_COMFORT_ZONES[2],
            cooperation_score: 1.0,
        },
        // 右手中指
        FingerState {
            id: 3,
            hand: Hand::Right,
            position: Vector::new(0.35, 0.0),
            last_used_time: -1.0,
            fatigue: 0.0,
            travel_distance: 0.0,
            activity_level: 0.0,
            preferred_zone: 0.35,
            comfort_radius: FINGER_COMFORT_ZONES[3],
            cooperation_score: 1.0,
        },
    ];

    let mut processed_notes: Vec<ProcessedNote> = notes.iter()
        .enumerate()
        .map(|(i, note)| {
            let raw_x = note.object.translation.0.now();
            let raw_y = note.object.translation.1.now();
            let rotated_x = raw_x * cos - raw_y * sin;
            let rotated_y = raw_x * sin + raw_y * cos;

            let natural_hand = if rotated_x < LANE_SPLIT {
                Hand::Left
            } else {
                Hand::Right
            };

            const EPS: f32 = 0.001;
            let t = note.time;
            let mut temp_trans_x = note.object.translation.0.clone();
            let mut temp_trans_y = note.object.translation.1.clone();

            temp_trans_x.set_time(t + EPS);
            let x_plus = temp_trans_x.now();
            temp_trans_x.set_time(t - EPS);
            let x_minus = temp_trans_x.now();
            let dx = (x_plus - x_minus) / (2.0 * EPS);

            temp_trans_y.set_time(t + EPS);
            let y_plus = temp_trans_y.now();
            temp_trans_y.set_time(t - EPS);
            let y_minus = temp_trans_y.now();
            let dy = (y_plus - y_minus) / (2.0 * EPS);

            let velocity = Vector::new(
                dx * cos - dy * sin,
                dx * sin + dy * cos
            );

            let complexity = match note.kind {
                NoteKind::Click => 1.0,
                NoteKind::Drag => 1.3,
                NoteKind::Flick => 1.5,
                NoteKind::Hold { .. } => 1.7,
            };

            ProcessedNote {
                idx: i,
                time: note.time,
                position: Vector::new(rotated_x, rotated_y),
                natural_hand,
                velocity,
                complexity,
                density_score: 1.0,
            }
        })
        .collect();

    processed_notes.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());

    for i in 0..processed_notes.len() {
        let mut nearby_count = 0;
        let current_time = processed_notes[i].time;

        for j in 0..processed_notes.len() {
            if i != j && (processed_notes[j].time - current_time).abs() < 0.2 {
                nearby_count += 1;
            }
        }

        processed_notes[i].density_score = (nearby_count as f32 * 0.1 + 1.0).min(2.0);
    }

    let mut dense_sections = Vec::new();
    let mut current_section_start = None;

    for i in 0..processed_notes.len() {
        if processed_notes[i].density_score > 1.3 {
            if current_section_start.is_none() {
                current_section_start = Some(i);
            }
        } else if let Some(start) = current_section_start {
            if i - start >= 3 { // 至少3个音符
                dense_sections.push((start, i - 1));
            }
            current_section_start = None;
        }
    }

    if let Some(start) = current_section_start {
        if processed_notes.len() - start >= 3 {
            dense_sections.push((start, processed_notes.len() - 1));
        }
    }

    let mut assignments: Vec<NoteAssignment> = Vec::new();
    let mut hand_usage = [0.0f32; 2];

    for pnote in &processed_notes {
        let mut candidate_fingers = Vec::new();

        let in_dense_section = dense_sections.iter().any(|&(start, end)| {
            let note_idx = processed_notes.iter().position(|p| p.idx == pnote.idx).unwrap();
            note_idx >= start && note_idx <= end
        });

        for (finger_idx, finger) in fingers.iter().enumerate() {
            let delta_pos = pnote.position - finger.position;
            let distance = delta_pos.magnitude();

            if distance > MAX_STRETCH_RADIUS * 1.5 {
                continue;
            }

            let prediction_time = if pnote.velocity.magnitude() > 0.1 { 0.1 } else { 0.05 };
            let predicted_position = pnote.position + pnote.velocity * prediction_time;
            let predicted_distance = (predicted_position - finger.position).magnitude();

            let mut cost = predicted_distance * 1.0;

            let time_since_last = pnote.time - finger.last_used_time;
            if time_since_last > 0.0 && time_since_last < 0.15 {
                let penalty_factor = (0.15 - time_since_last) / 0.15;
                cost += SAME_FINGER_PENALTY * penalty_factor * pnote.complexity * 0.8;
            }

            cost += finger.fatigue * (0.4 + finger.activity_level * 0.2) / finger.cooperation_score;

            if finger.hand != pnote.natural_hand {
                let cross_penalty = CROSS_HAND_PENALTY;
                let density_reduction = if in_dense_section { 0.6 } else { 1.0 };
                cost += cross_penalty * (1.0 + distance * 0.3) * density_reduction;
            }

            let comfort_factor = (predicted_distance / finger.comfort_radius).min(3.0);
            if comfort_factor > 1.0 {
                cost += (comfort_factor - 1.0).powi(2) * 0.6;
            }

            let zone_diff = (predicted_position.x - finger.preferred_zone).abs();
            if zone_diff < finger.comfort_radius * 0.7 {
                cost *= 0.8;
            }

            if in_dense_section {
                cost *= 0.7;

                let same_hand_fingers: Vec<usize> = fingers.iter()
                    .enumerate()
                    .filter(|(_, f)| f.hand == finger.hand)
                    .map(|(i, _)| i)
                    .collect();

                if same_hand_fingers.len() >= 2 {
                    cost *= 1.0 - COOPERATION_BONUS * finger.cooperation_score;
                }
            }

            cost *= 1.1 - FINGER_DEXTERITY[finger.id] * 0.1 - FINGER_COOPERATION[finger.id] * 0.05;

            candidate_fingers.push((finger_idx, cost));
        }

        if candidate_fingers.is_empty() {
            candidate_fingers = fingers.iter()
                .enumerate()
                .map(|(i, _)| (i, f32::MAX))
                .collect();
        }

        candidate_fingers.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let (best_finger_idx, best_cost) = candidate_fingers[0];

        let is_cross_hand = fingers[best_finger_idx].hand != pnote.natural_hand;
        assignments.push(NoteAssignment {
            note_idx: pnote.idx,
            finger_idx: best_finger_idx,
            cost: best_cost,
            is_cross_hand,
            in_dense_section,
        });

        let finger = &mut fingers[best_finger_idx];
        let movement_distance = (pnote.position - finger.position).magnitude();

        finger.position = pnote.position;
        finger.last_used_time = pnote.time;
        finger.travel_distance += movement_distance;
        finger.activity_level = (finger.activity_level * 0.7 + 0.3).min(1.0);
        finger.fatigue += 0.06 * pnote.complexity;

        if in_dense_section {
            finger.cooperation_score = (finger.cooperation_score * 0.9 + 1.2 * 0.1).min(1.5);
        }

        let hand_idx = finger.hand as usize;
        hand_usage[hand_idx] += pnote.complexity;
    }

    const MAX_OPTIMIZATION_PASSES: usize = 3;
    let mut improved = true;
    let mut pass_count = 0;

    while improved && pass_count < MAX_OPTIMIZATION_PASSES {
        improved = false;
        pass_count += 1;

        for i in 0..assignments.len() {
            let original_assignment = &assignments[i];
            if original_assignment.cost < 1.2 {
                continue;
            }

            let pnote = &processed_notes[original_assignment.note_idx];
            let mut best_swap: Option<(usize, f32)> = None;

            for j in 0..assignments.len() {
                if i == j { continue; }
                let other_note = &processed_notes[assignments[j].note_idx];
                if (pnote.time - other_note.time).abs() > 0.2 {
                    continue;
                }

                let current_finger_i = assignments[i].finger_idx;
                let current_finger_j = assignments[j].finger_idx;
                let current_cost = assignments[i].cost + assignments[j].cost;

                let cost_i = calculate_note_finger_cost(pnote, &fingers[current_finger_j]);
                let cost_j = calculate_note_finger_cost(other_note, &fingers[current_finger_i]);
                let new_cost = cost_i + cost_j;

                if new_cost < current_cost * 0.9 {
                    let improvement = current_cost - new_cost;
                    if best_swap.map_or(true, |(_, delta)| improvement > delta) {
                        best_swap = Some((j, improvement));
                    }
                }
            }

            if let Some((swap_idx, _)) = best_swap {
                assignments.swap(i, swap_idx);
                improved = true;
            }
        }
    }

    for assignment in &assignments {
        let note_idx = assignment.note_idx;
        let finger_idx = assignment.finger_idx;
        notes[note_idx].hand = fingers[finger_idx].hand;
    }

    let mut time_groups: Vec<Vec<usize>> = Vec::new();
    let mut current_group: Vec<usize> = Vec::new();
    let mut last_time = -1.0;

    for (i, pnote) in processed_notes.iter().enumerate() {
        if current_group.is_empty() || (pnote.time - last_time) <= 0.02 {
            current_group.push(i);
        } else {
            if current_group.len() > 1 {
                time_groups.push(current_group.clone());
            }
            current_group = vec![i];
        }
        last_time = pnote.time;
    }

    if current_group.len() > 1 {
        time_groups.push(current_group);
    }

    for group in &time_groups {
        let mut finger_used = [false; 4];
        let mut hands_used = [false; 2];

        for &note_idx in group {
            let assignment_idx = assignments.iter()
                .position(|a| a.note_idx == note_idx)
                .unwrap();
            let finger_idx = assignments[assignment_idx].finger_idx;
            finger_used[finger_idx] = true;
            hands_used[fingers[finger_idx].hand as usize] = true;
        }

        let is_simultaneous = group.len() > 1;
        let is_alternating = hands_used[0] && hands_used[1];
        let is_chord = finger_used.iter().filter(|&&used| used).count() >= 2;

        if is_simultaneous || is_alternating || is_chord {
            for &note_idx in group {
                notes[processed_notes[note_idx].idx].multiple_hint = true;
            }
        }
    }

    let total_hand_usage = hand_usage[0] + hand_usage[1];
    let balance_ratio = if total_hand_usage > 0.0 {
        (hand_usage[0] - hand_usage[1]).abs() / total_hand_usage
    } else {
        0.0
    };
    let dynamic_decay = FATIGUE_DECAY_RATE * (1.0 + balance_ratio * 0.3);

    for finger in &mut fingers {
        finger.fatigue = (finger.fatigue - dynamic_decay).max(0.0);
        finger.activity_level *= 0.85;
        finger.cooperation_score = (finger.cooperation_score * 0.95 + 1.0 * 0.05).max(0.8);
    }

    fn calculate_note_finger_cost(note: &ProcessedNote, finger: &FingerState) -> f32 {
        let delta_pos = note.position - finger.position;
        let distance = delta_pos.magnitude();

        let time_since_last = note.time - finger.last_used_time;
        let is_same_finger_recent = time_since_last > 0.0 && time_since_last < 0.15;

        let mut cost = distance * 1.0;

        if is_same_finger_recent {
            let penalty_factor = (0.15 - time_since_last) / 0.15;
            cost += SAME_FINGER_PENALTY * penalty_factor * note.complexity;
        }

        cost += finger.fatigue * (0.4 + finger.activity_level * 0.2);

        if finger.hand != note.natural_hand {
            cost += CROSS_HAND_PENALTY * (1.0 + distance * 0.3);
        }

        let comfort_factor = (distance / finger.comfort_radius).min(3.0);
        if comfort_factor > 1.0 {
            cost += (comfort_factor - 1.0).powi(2) * 0.6;
        }

        let zone_diff = (note.position.x - finger.preferred_zone).abs();
        if zone_diff < finger.comfort_radius * 0.7 {
            cost *= 0.8;
        }

        cost *= 1.1 - FINGER_DEXTERITY[finger.id] * 0.1;

        cost
    }
}