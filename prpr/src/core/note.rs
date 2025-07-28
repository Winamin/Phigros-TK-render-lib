use super::{chart::ChartSettings, BpmList, CtrlObject, JudgeLine, Matrix, Object, Point, Resource};
use crate::{
    judge::JudgeStatus, 
    parse::RPE_HEIGHT,
    core::HEIGHT_RATIO,
};

use macroquad::prelude::*;
//use ::rand::{thread_rng, Rng};
use crate::core::Vector;
//const HOLD_PARTICLE_INTERVAL: f32 = 0.15;
const FADEOUT_TIME: f32 = 0.16;
const BAD_TIME: f32 = 0.5;
const RPE_HEIGHT_SCALE: f32 = RPE_HEIGHT * (1.0 / 720.0);

#[derive(Clone, Debug)]
pub enum NoteKind {
    Click,
    Hold { end_time: f32, end_height: f32 },
    Flick,
    Drag,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hand {
    Left,
    Right,
}

impl NoteKind {
    #[inline]
    pub fn order(&self) -> i8 {
        match self {
            Self::Hold { .. } => 0,
            Self::Drag => 1,
            Self::Click => 2,
            Self::Flick => 3,
        }
    }
}

pub struct Note {
    pub time: f32,
    pub kind: NoteKind,
    pub height: f32,
    pub object: Object,
    pub speed: f32,
    pub end_speed: f32,
    pub start_height: f32,
    pub hand: Hand,

    pub above: bool,
    pub multiple_hint: bool,
    pub fake: bool,
    pub judge: JudgeStatus,
    pub format: bool,
}

pub struct RenderConfig<'a> {
    pub settings: &'a ChartSettings,
    pub ctrl_obj: &'a mut CtrlObject,
    pub line_height: f32,
    pub appear_before: f32,
    pub invisible_time: f32,
    pub draw_below: bool,
    pub incline_sin: f32,
}

#[inline(always)]
fn draw_tex(res: &Resource, texture: Texture2D, order: i8, x: f32, y: f32, color: Color, mut params: DrawTextureParams, clip: bool) {
    let Vec2 { x: w, y: h } = params.dest_size.unwrap();
    if h <= 0. || (clip && y + h <= 0.) {
        return;
    }
    let mut source = params.source.unwrap_or(Rect::new(0., 0., 1., 1.));
    if clip && y < 0. {
        let visible_height = y + h;
        if visible_height <= 0. {
            return;
        }
        let visible_ratio = (-y) / visible_height;
        source.y += source.h * visible_ratio;
        source.h *= 1. - visible_ratio;
    }
    const INIT_POINTS: [Point; 4] = [
        Point::new(0., 0.),
        Point::new(1., 0.),
        Point::new(1., 1.),
        Point::new(0., 1.),
    ];

    let p = INIT_POINTS.map(|pt| Point::new(x + pt.x * w, y + pt.y * h));

    params.flip_y = true;
    draw_tex_pts(res, texture, order, p, color, DrawTextureParams {
        source: Some(source),
        ..params
    });
}

#[inline(always)]
fn draw_tex_pts(res: &Resource, texture: Texture2D, order: i8, p: [Point; 4], color: Color, params: DrawTextureParams) {
    let p_screen = p.map(|pt| res.world_to_screen(pt));

    let (min_x, max_x) = p_screen.iter()
        .fold((f32::MAX, f32::MIN), |(min, max), pt|
            (min.min(pt.x), max.max(pt.x)));

    let (min_y, max_y) = p_screen.iter()
        .fold((f32::MAX, f32::MIN), |(min, max), pt|
            (min.min(pt.y), max.max(pt.y)));

    let chart_ratio_inv = 1.0 / res.config.chart_ratio;
    if min_x > chart_ratio_inv ||
        max_x < -chart_ratio_inv ||
        min_y > chart_ratio_inv ||
        max_y < -chart_ratio_inv
    {
        return;
    }

    let mut p = p_screen;
    if params.flip_x {
        p.swap(1, 0);
        p.swap(2, 3);
    }
    if params.flip_y {
        p.swap(0, 3);
        p.swap(1, 2);
    }

    let Rect { x: sx, y: sy, w: sw, h: sh } = params.source.unwrap_or(Rect::new(0., 0., 1., 1.));
    let sx1 = sx + sw;
    let sy1 = sy + sh;

    let vertices = [
        Vertex::new(p[0].x, p[0].y, 0., sx,  sy,  color),
        Vertex::new(p[1].x, p[1].y, 0., sx1, sy,  color),
        Vertex::new(p[2].x, p[2].y, 0., sx1, sy1, color),
        Vertex::new(p[3].x, p[3].y, 0., sx,  sy1, color),
    ];

    res.note_buffer.borrow_mut().push(
        (order, texture.raw_miniquad_texture_handle().gl_internal_id()),
        vertices
    );
}

fn random_rotate() -> f32 {
    // good good good good good good
    static mut SEED: u32 = 0x12345678;
    unsafe {
        SEED = SEED.wrapping_mul(1664525).wrapping_add(1013904223);
        (SEED % 4) as f32 * 90.0
    }
}

fn draw_center(res: &Resource, tex: Texture2D, order: i8, scale: f32, color: Color) {
    let hf = vec2(scale, tex.height() * scale / tex.width());
    draw_tex(
        res,
        tex,
        order,
        -hf.x,
        -hf.y,
        color,
        DrawTextureParams {
            dest_size: Some(hf * 2.),
            ..Default::default()
        },
        false,
    );
}

impl Note {
    pub fn rotation(&self, line: &JudgeLine) -> f32 {
        line.object.rotation.now() + if self.above { 0. } else { 180. }
    }

    #[inline]
    pub fn plain(&self) -> bool {
        !self.fake && !matches!(self.kind, NoteKind::Hold { .. }) && self.object.translation.1.keyframes.len() <= 1
    }

    pub fn dead(&self) -> bool {
        (!matches!(self.kind, NoteKind::Hold { .. }) || matches!(self.judge, JudgeStatus::Judged))
            && self.object.dead()
    }

    pub fn update(&mut self, res: &mut Resource, parent_rot: f32, parent_tr: &Matrix, ctrl_obj: &mut CtrlObject, line_height: f32, bpm_list: &mut BpmList, index: usize) {
        self.object.set_time(res.time);
        let color = match &mut self.judge {
            JudgeStatus::Hold(perfect, ref mut at, ..) if res.time >= *at => {
                let bpm_index = if self.format { index as f32 } else { self.time };
                let now_bpm = bpm_list.now_bpm(bpm_index);
                let beat_duration = 30.0 / (now_bpm * res.config.speed);
                *at = res.time + beat_duration;
                let colors = [res.res_pack.info.fx_good(), res.res_pack.info.fx_perfect()];
                Some(colors[*perfect as usize])
            }
            _ => None
        };
        if let Some(color) = color {
            self.init_ctrl_obj(ctrl_obj, line_height);
            let rotation = if res.config.chart_debug {
                if self.above { 0. } else { 180. }
            } else {
                random_rotate()
            };
            let transform = *parent_tr * self.now_transform(res, ctrl_obj, 0., 0.);
            res.with_model(transform, |res| {
                res.emit_at_origin(parent_rot + rotation, color)
            });
        }
    }

    fn init_ctrl_obj(&self, ctrl_obj: &mut CtrlObject, line_height: f32) {
        ctrl_obj.set_height((self.height - line_height + self.object.translation.1.now() / self.speed) * RPE_HEIGHT / 2.);
    }

    #[inline(always)]
    pub fn now_transform(&self, res: &Resource, ctrl_obj: &CtrlObject, base: f32, incline_sin: f32) -> Matrix {
        let translation_y = self.object.translation.1.now();
        let aspect_base = base * res.aspect_ratio;
        let incline_val = 1.0 - incline_sin * (aspect_base + translation_y) * RPE_HEIGHT_SCALE;
        let ctrl_pos = ctrl_obj.pos.now_opt().unwrap_or(1.0);
        let mut tr = self.object.now_translation(res);
        tr.x *= incline_val * ctrl_pos;
        tr.y += base;
        self.object.now_rotation()
            .append_nonuniform_scaling(&self.object.scale.now_with_def(1.0, 1.0))
            .append_translation(&tr)
    }

    pub fn assign_hands(notes: &mut [Note], rotation: f32) {
        // 使用传入的 hand_split 参数
        const LANE_SPLIT: f32 = 0.0;
        //const MAX_COMFORT_RADIUS: f32 = 0.25;
        const MAX_STRETCH_RADIUS: f32 = 0.38;
        //const BASE_FINGER_SPEED: f32 = 1.2;
        const FATIGUE_DECAY_RATE: f32 = 0.03;
        //const HAND_BALANCE_FACTOR: f32 = 0.25;
        const CROSS_HAND_PENALTY: f32 = 0.4;
        const SAME_FINGER_PENALTY: f32 = 0.35;

        const FINGER_COMFORT_ZONES: [f32; 4] = [0.28, 0.12, 0.12, 0.28];
        const FINGER_DEXTERITY: [f32; 4] = [0.9, 1.0, 1.0, 0.9];

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
        }

        #[derive(Clone)]
        struct NoteAssignment {
            note_idx: usize,
            finger_idx: usize,
            cost: f32,
            is_cross_hand: bool,
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
            },
        ];

        struct ProcessedNote {
            idx: usize,
            time: f32,
            position: Vector,
            natural_hand: Hand,
            velocity: Vector,
            complexity: f32,
        }

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
                //let x_now = note.object.translation.0.now();
                //let y_now = note.object.translation.1.now();

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
                }
            })
            .collect();

        processed_notes.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());

        let mut assignments: Vec<NoteAssignment> = Vec::new();
        let mut hand_usage = [0.0f32; 2]; // [left, right]

        for pnote in &processed_notes {
            let mut candidate_fingers = Vec::new();

            for (finger_idx, finger) in fingers.iter().enumerate() {
                let delta_pos = pnote.position - finger.position;
                let distance = delta_pos.magnitude();

                if distance > MAX_STRETCH_RADIUS * 1.5 {
                    continue;
                }

                let time_since_last = pnote.time - finger.last_used_time;
                let is_same_finger_recent = time_since_last > 0.0 && time_since_last < 0.18;

                let predicted_position = pnote.position + pnote.velocity * (pnote.time - pnote.velocity.magnitude().max(0.05));

                let predicted_delta = predicted_position - finger.position;
                let predicted_distance = predicted_delta.magnitude();

                let mut cost = 0.0;
                cost += predicted_distance * 1.2;

                if is_same_finger_recent {
                    let penalty_factor = (0.18 - time_since_last) / 0.18;
                    cost += SAME_FINGER_PENALTY * penalty_factor * pnote.complexity;
                }

                cost += finger.fatigue * (0.5 + finger.activity_level * 0.3);

                if finger.hand != pnote.natural_hand {
                    cost += CROSS_HAND_PENALTY * (1.0 + distance * 0.5);
                }

                let comfort_factor = (predicted_distance / finger.comfort_radius).min(3.0);
                if comfort_factor > 1.0 {
                    cost += (comfort_factor - 1.0).powi(2) * 0.8;
                }

                let zone_diff = (predicted_position.x - finger.preferred_zone).abs();
                if zone_diff < finger.comfort_radius * 0.6 {
                    cost *= 0.85;
                }

                cost *= 1.1 - FINGER_DEXTERITY[finger.id] * 0.1;

                candidate_fingers.push((finger_idx, cost));
            }

            if candidate_fingers.is_empty() {
                candidate_fingers = fingers.iter()
                    .enumerate()
                    .map(|(i, _)| (i, f32::MAX))
                    .collect();
            }

            candidate_fingers.sort_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let (best_finger_idx, best_cost) = candidate_fingers[0];

            let is_cross_hand = fingers[best_finger_idx].hand != pnote.natural_hand;
            assignments.push(NoteAssignment {
                note_idx: pnote.idx,
                finger_idx: best_finger_idx,
                cost: best_cost,
                is_cross_hand,
            });

            let finger = &mut fingers[best_finger_idx];
            finger.position = pnote.position;
            finger.last_used_time = pnote.time;
            finger.travel_distance += (pnote.position - finger.position).magnitude();
            finger.activity_level = (finger.activity_level * 0.7 + 0.3).min(1.0);

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
                if original_assignment.cost < 1.5 {
                    continue;
                }

                let pnote = &processed_notes[original_assignment.note_idx];
                let mut best_swap: Option<(usize, f32)> = None; // (swap_with, delta)

                for j in 0..assignments.len() {
                    if i == j { continue; }
                    let other_note = &processed_notes[assignments[j].note_idx];
                    if (pnote.time - other_note.time).abs() > 0.15 {
                        continue;
                    }
                    let current_finger_i = assignments[i].finger_idx;
                    let current_finger_j = assignments[j].finger_idx;
                    let current_cost = assignments[i].cost + assignments[j].cost;
                    let cost_i = calculate_note_finger_cost(pnote, &fingers[current_finger_j]);
                    let cost_j = calculate_note_finger_cost(other_note, &fingers[current_finger_i]);
                    let new_cost = cost_i + cost_j;
                    if new_cost < current_cost * 0.85 {
                        let improvement = current_cost - new_cost;
                        if best_swap.map_or(true, |(_, delta)| improvement > delta) {
                            best_swap = Some((j, improvement));
                        }
                    }
                }
                if let Some((swap_idx, _)) = best_swap {
                    assignments.swap(i, swap_idx);
                    improved = true;
                    let finger_i = assignments[i].finger_idx;
                    let finger_j = assignments[swap_idx].finger_idx;
                    fingers[finger_i].position = pnote.position;
                    fingers[finger_j].position = processed_notes[assignments[swap_idx].note_idx].position;
                }
            }
        }
        let mut finger_usage_count = [0usize; 4];

        for assignment in &assignments {
            let note_idx = assignment.note_idx;
            let finger_idx = assignment.finger_idx;
            notes[note_idx].hand = fingers[finger_idx].hand;
            let finger = &mut fingers[finger_idx];
            finger.last_used_time = processed_notes[note_idx].time;
            finger.fatigue += 0.08 * processed_notes[note_idx].complexity;
            finger.travel_distance += (processed_notes[note_idx].position - finger.position).magnitude();
            finger.activity_level = (finger.activity_level * 0.6 + 0.4).min(1.0);
            finger.position = processed_notes[note_idx].position;
            finger_usage_count[finger_idx] += 1;
            hand_usage[finger.hand as usize] += processed_notes[note_idx].complexity;
        }
        let mut time_groups: Vec<Vec<usize>> = Vec::new();
        let mut current_group: Vec<usize> = Vec::new();
        let mut last_time = -1.0;
        for (i, pnote) in processed_notes.iter().enumerate() {
            if current_group.is_empty() || (pnote.time - last_time) <= 0.015 {
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
        let dynamic_decay = FATIGUE_DECAY_RATE * (1.0 + balance_ratio * 0.5);

        for finger in &mut fingers {
            let usage_factor = finger_usage_count[finger.id] as f32 / processed_notes.len() as f32;
            finger.fatigue = (finger.fatigue - dynamic_decay * (1.0 + usage_factor * 0.5)).max(0.0);
            finger.activity_level *= 0.8;
        }

        let avg_left_x: f32 = fingers.iter()
            .filter(|f| f.hand == Hand::Left)
            .map(|f| f.position.x)
            .sum::<f32>() / 2.0;

        let avg_right_x: f32 = fingers.iter()
            .filter(|f| f.hand == Hand::Right)
            .map(|f| f.position.x)
            .sum::<f32>() / 2.0;

        for finger in &mut fingers {
            if finger.hand == Hand::Left {
                finger.preferred_zone = (finger.preferred_zone * 0.7 + avg_left_x * 0.3).clamp(-0.5, 0.0);
            } else {
                finger.preferred_zone = (finger.preferred_zone * 0.7 + avg_right_x * 0.3).clamp(0.0, 0.5);
            }

            finger.comfort_radius = FINGER_COMFORT_ZONES[finger.id] *
                (1.0 - finger.activity_level * 0.2 + finger.fatigue * 0.1);
        }

        fn calculate_note_finger_cost(note: &ProcessedNote, finger: &FingerState) -> f32 {
            let delta_pos = note.position - finger.position;
            let distance = delta_pos.magnitude();

            let time_since_last = note.time - finger.last_used_time;
            let is_same_finger_recent = time_since_last > 0.0 && time_since_last < 0.18;

            let mut cost = distance * 1.2;

            if is_same_finger_recent {
                let penalty_factor = (0.18 - time_since_last) / 0.18;
                cost += SAME_FINGER_PENALTY * penalty_factor * note.complexity;
            }

            cost += finger.fatigue * (0.5 + finger.activity_level * 0.3);

            if finger.hand != note.natural_hand {
                cost += CROSS_HAND_PENALTY * (1.0 + distance * 0.5);
            }

            let comfort_factor = (distance / finger.comfort_radius).min(3.0);
            if comfort_factor > 1.0 {
                cost += (comfort_factor - 1.0).powi(2) * 0.8;
            }

            let zone_diff = (note.position.x - finger.preferred_zone).abs();
            if zone_diff < finger.comfort_radius * 0.6 {
                cost *= 0.85;
            }

            cost *= 1.1 - FINGER_DEXTERITY[finger.id] * 0.1;

            cost
        }
    }

    pub fn render(&self, res: &mut Resource, config: &mut RenderConfig, bpm_list: &mut BpmList) {
        if matches!(self.judge, JudgeStatus::Judged) && !matches!(self.kind, NoteKind::Hold { .. }) {
            return;
        }

        if config.appear_before.is_finite() {
            //if config.appear_before.is_finite() && !matches!(self.kind, NoteKind::Hold { .. }) {
            let beat = bpm_list.beat(self.time);
            let time = bpm_list.time_beats(beat - config.appear_before);
            if time > res.time {
                return;
            }
        }

        if config.invisible_time.is_finite() && self.time - config.invisible_time < res.time {
            return;
        }
        let scale = res.note_width * if self.multiple_hint {
            res.res_pack.note_style_mh.click.width() / res.res_pack.note_style.click.width()
        } else {
            1.0
        };
        let ctrl_obj = &mut config.ctrl_obj;
        self.init_ctrl_obj(ctrl_obj, config.line_height);
        let mut color = self.object.now_color();

        if res.config.hand_split {
            match self.hand {
                Hand::Left => {
                    color.r = 1.0;
                    color.g = 0.6;
                    color.b = 0.7;
                }
                Hand::Right => {
                    color.r = 0.2;
                    color.g = 0.5;
                    color.b = 1.0;
                }
            }
            const BASE_LUMINANCE: f32 = 0.7;
            let luminance = color.r * 0.299 + color.g * 0.587 + color.b * 0.114;
            let adjust_factor = BASE_LUMINANCE / luminance.max(0.001);
            color.r = (color.r * adjust_factor).min(1.0);
            color.g = (color.g * adjust_factor).min(1.0);
            color.b = (color.b * adjust_factor).min(1.0);
        }

        color.a *= res.alpha * ctrl_obj.alpha.now_opt().unwrap_or(1.);
        let y_factor = ctrl_obj.y.now_opt().unwrap_or(1.);
        let spd = self.speed * y_factor;
        let end_spd = self.end_speed * y_factor;

        let inv_aspect = 1.0 / res.aspect_ratio;
        let line_height = config.line_height * inv_aspect * spd;
        let height = self.height * inv_aspect * spd;
        let base = height - line_height;
        //let base = (self.height - config.line_height) / res.aspect_ratio * spd;

        if res.config.aggressive && matches!(self.kind, NoteKind::Hold { .. }) {
            let h = if self.time <= res.time { line_height } else { height };
            let bottom = h + self.object.translation.1.now() - line_height;
            if bottom - line_height > 1. / res.config.chart_ratio {
                return;
            }
        }

        // 无分支渲染决策
        let should_skip = !config.draw_below && (
            (res.time - FADEOUT_TIME >= self.time && !matches!(self.kind, NoteKind::Hold { .. })) ||
                (self.time > res.time && base <= -0.0075)
        ) && self.speed != 0.;

        if should_skip {
            if res.config.chart_debug {
                color.a *= 0.2;
                //println!("{}", base);
            } else {
                return;
            }
        }
        let order = self.kind.order();
        let style = if res.config.double_hint && self.multiple_hint {
            &res.res_pack.note_style_mh
        } else {
            &res.res_pack.note_style
        };

        let draw = |res: &mut Resource, tex: Texture2D| {
            let mut color = color;
            if !config.draw_below {
                let fade_factor = (self.time - res.time).min(0.0) / FADEOUT_TIME + 1.0;
                color.a *= fade_factor;
            }
            res.with_model(self.now_transform(res, ctrl_obj, base, config.incline_sin), |res| {
                draw_center(res, tex, order, scale, color);
            });
        };

        match self.kind {
            NoteKind::Click => {
                if self.fake && res.time >= self.time {return};
                draw(res, *style.click);
            }
            NoteKind::Hold { end_time, end_height } => {
                if self.fake && res.time >= end_time {return};
                res.with_model(self.now_transform(res, ctrl_obj, 0., 0.), |res| {
                    let style = if res.config.double_hint && self.multiple_hint {
                        &res.res_pack.note_style_mh
                    } else {
                        &res.res_pack.note_style
                    };
                    if matches!(self.judge, JudgeStatus::Judged) {
                        // miss
                        color.a *= 0.5;
                    }
                    if res.time >= end_time {
                        return;
                    }
                    let end_height = end_height / res.aspect_ratio * spd;
                    let start_height = self.start_height / res.aspect_ratio * spd;
                    let hold_height = end_height - start_height;
                    let time = if res.time >= self.time {res.time} else {self.time};
                    let hold_line_height = (time - self.time) * end_spd / res.aspect_ratio / HEIGHT_RATIO;

                    let clip = !config.draw_below && config.settings.hold_partial_cover;


                    let h = if self.time <= res.time { line_height } else { height };
                    let bottom = h - line_height; //StartY
                    let top = if self.format {
                        bottom + hold_height - hold_line_height
                    } else {
                        end_height - line_height
                    };

                    //let max_hold_height = 3. / res.config.chart_ratio / res.aspect_ratio;
                    //let top = if res.config.aggressive && hold_height - hold_line_height >= max_hold_height { bottom + max_hold_height } else { top };

                    if self.format && end_spd == 0. {
                        if res.config.chart_debug {
                            color.a *= 0.2;
                        } else {
                            return;
                        }
                    }


                    if res.time < self.time && bottom < -1e-6 && (!config.settings.hold_partial_cover && !self.format) {
                        return;
                    }
                    let tex = &style.hold;
                    let ratio = style.hold_ratio();
                    // body
                    // TODO (end_height - height) is not always total height
                    draw_tex(
                        res,
                        **(if res.res_pack.info.hold_repeat {
                            style.hold_body.as_ref().unwrap()
                        } else {
                            tex
                        }),
                        order,
                        -scale,
                        bottom,
                        color,
                        DrawTextureParams {
                            source: Some({
                                if res.res_pack.info.hold_repeat {
                                    let hold_body = style.hold_body.as_ref().unwrap();
                                    let width = hold_body.width();
                                    let height = hold_body.height();
                                    Rect::new(0., 0., 1., (top - bottom) / scale / 2. * width / height)
                                } else {
                                    style.hold_body_rect()
                                }
                            }),
                            dest_size: Some(vec2(scale * 2., top - bottom)),
                            ..Default::default()
                        },
                        clip,
                    );
                    // head
                    if res.time < self.time || res.res_pack.info.hold_keep_head {
                        let r = style.hold_head_rect();
                        let hf = vec2(scale, r.h / r.w * scale * ratio);
                        draw_tex(
                            res,
                            **tex,
                            order,
                            -scale,
                            bottom - if res.res_pack.info.hold_compact { hf.y } else { hf.y * 2. },
                            color,
                            DrawTextureParams {
                                source: Some(r),
                                dest_size: Some(hf * 2.),
                                ..Default::default()
                            },
                            clip,
                        );
                    }
                    // tail
                    let r = style.hold_tail_rect();
                    let hf = vec2(scale, r.h / r.w * scale * ratio);
                    draw_tex(
                        res,
                        **tex,
                        order,
                        -scale,
                        top - if res.res_pack.info.hold_compact { hf.y } else { 0. },
                        color,
                        DrawTextureParams {
                            source: Some(r),
                            dest_size: Some(hf * 2.),
                            ..Default::default()
                        },
                        clip,
                    );
                });
            }
            NoteKind::Flick => {
                if self.fake && res.time >= self.time {return};
                draw(res, *style.flick);
            }
            NoteKind::Drag => {
                if self.fake && res.time >= self.time {return};
                draw(res, *style.drag);
            }
        }
    }
}

pub struct BadNote {
    pub time: f32,
    pub kind: NoteKind,
    pub matrix: Matrix,
}

impl BadNote {
    pub fn render(&self, res: &mut Resource) -> bool {
        if res.time > self.time + BAD_TIME {
            return false;
        }
        res.with_model(self.matrix, |res| {
            let style = &res.res_pack.note_style;
            draw_center(
                res,
                match &self.kind {
                    NoteKind::Click => *style.click,
                    NoteKind::Drag => *style.drag,
                    NoteKind::Flick => *style.flick,
                    _ => unreachable!(),
                },
                self.kind.order(),
                res.note_width,
                Color::new(0.423529, 0.262745, 0.262745, (self.time - res.time).max(-1.) / BAD_TIME + 1.),
            );
        });
        true
    }
}