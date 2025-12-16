crate::tl_file!("parser" ptl);

use super::process_lines;
    
use crate::{
    core::{
        Anim, AnimFloat, AnimVector, BpmList, Chart, ChartExtra, ChartSettings, JudgeLine, JudgeLineCache, JudgeLineKind, Keyframe, Note, NoteKind,
        Object, HEIGHT_RATIO,
    },
    ext::{NotNanExt, ChunkedChart},
    judge::JudgeStatus,
};
use anyhow::{Context, Result};
use serde::{Deserialize};
use tracing::warn;
use anyhow::bail;
use crate::core::note::Hand;
use crate::hand::assign_hands;
use crate::config::Config;
use std::sync::{Mutex, Arc};
use crate::core::CtrlObject;

// 智能手部分配函数
fn smart_assign_hand(
    position_x: f32,
    time: f32,
    previous_notes: &[(f32, f32, Hand)],
    _switch_threshold: f32,
) -> Hand {
    // 时间窗口（秒）内，认为是连续音符
    const TEMPORAL_WINDOW: f32 = 1.0;
    // 位置切换阈值
    const POSITION_THRESHOLD: f32 = 0.2;
    
    // 查找时间窗口内最近的音符
    let mut best_match = None;
    let mut min_time_diff = f32::INFINITY;
    
    for &(note_time, note_pos, note_hand) in previous_notes {
        let time_diff = (time - note_time).abs();
        if time_diff < TEMPORAL_WINDOW && time_diff < min_time_diff {
            min_time_diff = time_diff;
            best_match = Some((note_hand, note_pos, time_diff));
        }
    }
    
    if let Some((best_hand, best_pos, time_diff)) = best_match {
        // 如果位置变化不大，保持同一只手
        let pos_diff = (position_x - best_pos).abs();
        if pos_diff < POSITION_THRESHOLD {
            return best_hand;
        }
        
        // 位置变化大但时间很近，避免频繁切换
        if time_diff < TEMPORAL_WINDOW * 0.3 {
            return best_hand;
        }
    }
    
    // 基于位置的智能分配
    if position_x < 0.5 {
        Hand::Left
    } else {
        Hand::Right
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PgrEvent {
    pub start_time: f32,
    pub end_time: f32,
    pub start: f32,
    pub end: f32,
    #[serde(default)]
    pub start2: f32,
    #[serde(default)]
    pub end2: f32,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PgrSpeedEvent {
    pub start_time: f32,
    pub end_time: f32,
    pub value: f32,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PgrNote {
    #[serde(rename = "type")]
    kind: u8,
    time: f32,
    position_x: f32,
    hold_time: f32,
    speed: f32,
    floor_position: f32,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PgrJudgeLine {
    bpm: f32,
    #[serde(rename = "judgeLineDisappearEvents")]
    alpha_events: Vec<PgrEvent>,
    #[serde(rename = "judgeLineRotateEvents")]
    rotate_events: Vec<PgrEvent>,
    #[serde(rename = "judgeLineMoveEvents")]
    move_events: Vec<PgrEvent>,
    speed_events: Vec<PgrSpeedEvent>,
    notes_above: Vec<PgrNote>,
    notes_below: Vec<PgrNote>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PgrChart {
    offset: f32,
    judge_line_list: Vec<PgrJudgeLine>,
}

macro_rules! validate_events {
    ($pgr:expr) => {
        // 保留有效的事件
        $pgr.retain(|it| {
            if it.start_time > it.end_time {
                warn!("invalid time range, ignoring");
                false
            } else {
                true
            }
        });
        
        /* Uncomment if needed
        Official music should be continuous, so it is useless
        for i in 0..($pgr.len() - 1) {
            if $pgr[i].end_time != $pgr[i + 1].start_time {
                ptl!(bail "event-not-contiguous");
            }
        }
        */

        // Uncomment this check if needed
        // if $pgr.last().unwrap().end_time <= 900000000.0 {
        //     bail!("End time is not great enough ({})", $pgr.last().unwrap().end_time);
        // }
    };
}

fn parse_speed_events(r: f32, mut pgr: Vec<PgrSpeedEvent>, max_time: f32) -> Result<(AnimFloat, AnimFloat)> {
    validate_events!(pgr);
    assert_eq!(pgr[0].start_time, 0.0);
    let mut kfs = Vec::with_capacity(pgr.len() + 1); // 预分配 kfs 的内存容量
    let mut pos = 0.0;
    for it in pgr.iter().take(pgr.len() - 1) {
        let from_pos = pos;
        pos += (it.end_time - it.start_time) * r * it.value;
        kfs.push(Keyframe::new(it.start_time * r, from_pos, 2));
    }
    let last = pgr.last().unwrap();
    kfs.push(Keyframe::new(last.start_time * r, pos, 2));
    kfs.push(Keyframe::new(max_time, pos + (max_time - last.start_time * r) * last.value, 0));
    for kf in &mut kfs {
        kf.value /= HEIGHT_RATIO;
    }
    Ok((AnimFloat::new(pgr.iter().map(|it| Keyframe::new(it.start_time * r, it.value, 0)).collect()), AnimFloat::new(kfs)))
}

fn parse_float_events(r: f32, mut pgr: Vec<PgrEvent>) -> Result<AnimFloat> {
    validate_events!(pgr);
    let mut kfs = Vec::<Keyframe<f32>>::new();
    for e in pgr {
        let st = (e.start_time * r).max(0.); // 处理开始时间
        let en = (e.end_time * r).max(0.); // 处理结束时间
        if kfs.is_empty() || kfs.last().unwrap().value != e.start {
            kfs.push(Keyframe::new(st, e.start, 2));
        }
        kfs.push(Keyframe::new(en, e.end, 2));
    }
    // 只在非空情况下移除最后一个关键帧
    //if !kfs.is_empty() {
    //    kfs.pop();
    //}
    Ok(AnimFloat::new(kfs))
}

fn parse_move_events(r: f32, mut pgr: Vec<PgrEvent>) -> Result<AnimVector> {
    validate_events!(pgr);
    let mut kf1 = Vec::<Keyframe<f32>>::new();
    let mut kf2 = Vec::<Keyframe<f32>>::new();
    for e in pgr {
        let st = (e.start_time * r).max(0.);
        let en = e.end_time * r;
        if !kf1.last().map_or(false, |it| it.value == e.start) {
            kf1.push(Keyframe::new(st, e.start, 2));
        }
        if !kf2.last().map_or(false, |it| it.value == e.start2) {
            kf2.push(Keyframe::new(st, e.start2, 2));
        }
        kf1.push(Keyframe::new(en, e.end, 2));
        kf2.push(Keyframe::new(en, e.end2, 2));
    }
    //kf1.pop();
    //kf2.pop();
    for kf in &mut kf1 {
        kf.value = -1. + kf.value * 2.;
    }
    for kf in &mut kf2 {
        kf.value = -1. + kf.value * 2.;
    }
    Ok(AnimVector(AnimFloat::new(kf1), AnimFloat::new(kf2)))
}

fn parse_notes(r: f32, mut pgr: Vec<PgrNote>, speed: &mut AnimFloat, height: &mut AnimFloat, above: bool) -> Result<Vec<Note>> {
    // is_sorted is unstable...
    if pgr.is_empty() {
        return Ok(Vec::new());
    }
    pgr.sort_by(|a, b| a.time.partial_cmp(&b.time).expect("Invalid note time"));
    
    // 用于智能手部分配的追踪
    let previous_notes: Vec<(f32, f32, Hand)> = Vec::new();
    
    pgr.into_iter()
        .map(|pgr| {
            let time = pgr.time * r;
            Ok(Note {
                object: Object {
                    translation: AnimVector(AnimFloat::fixed(pgr.position_x * (2. * 9. / 160.)), AnimFloat::default()),
                    ..Default::default()
                },
                kind: match pgr.kind {
                    1 => NoteKind::Click,
                    2 => NoteKind::Drag,
                    3 => {
                        let end_time = (pgr.time + pgr.hold_time) * r;
                        height.set_time(time);
                        let start_height = height.now();
                        let end_height = start_height + (pgr.hold_time * pgr.speed * r / HEIGHT_RATIO);
                        NoteKind::Hold { end_time, end_height }
                    }
                    4 => NoteKind::Flick,
                    _ => ptl!(bail "unknown-note-type", "type" => pgr.kind),
                },
                time,
                speed: if pgr.kind == 3 {
                    speed.set_time(time);
                    1.
                } else {
                    pgr.speed
                },
                end_speed: pgr.speed,
                height: pgr.floor_position / HEIGHT_RATIO,
                start_height: {
                    height.set_time(time);
                    height.now()
                },
                hand: smart_assign_hand(pgr.position_x, time, &previous_notes, 0.3),
                above,
                multiple_hint: false,
                fake: false,
                judge: JudgeStatus::NotJudged,
                format: true
            })
        })
        .collect::<Result<Vec<_>>>()
}

fn parse_notes_chunked(r: f32, mut pgr: Vec<PgrNote>, speed: &mut AnimFloat, height: &mut AnimFloat, above: bool, chunked_chart: &mut ChunkedChart, line_id: usize) -> Result<Vec<Note>> {
    if pgr.is_empty() {
        return Ok(Vec::new());
    }
    pgr.sort_by(|a, b| a.time.partial_cmp(&b.time).expect("Invalid note time"));
    let mut previous_notes: Vec<(f32, f32, Hand)> = Vec::new();
    for note in &pgr {
        let time = note.time * r;
        let chunk_id = chunked_chart.get_chunk_for_time(time);
        
        if chunk_id < chunked_chart.chunks.len() {
            if !chunked_chart.chunks[chunk_id].line_ids.contains(&line_id) {
                chunked_chart.chunks[chunk_id].line_ids.push(line_id);
            }
        }
    }

    let first_chunk_notes: Vec<_> = pgr.into_iter()
        .filter(|note| {
            let time = note.time * r;
            let chunk_id = chunked_chart.get_chunk_for_time(time);
            chunk_id == 0
        })
        .collect();
    
    first_chunk_notes.into_iter()
        .map(|pgr| {
            let time = pgr.time * r;
            Ok(Note {
                object: Object {
                    translation: AnimVector(AnimFloat::fixed(pgr.position_x * (2. * 9. / 160.)), AnimFloat::default()),
                    ..Default::default()
                },
                kind: match pgr.kind {
                    1 => NoteKind::Click,
                    2 => NoteKind::Drag,
                    3 => {
                        let end_time = (pgr.time + pgr.hold_time) * r;
                        height.set_time(time);
                        let start_height = height.now();
                        let end_height = start_height + (pgr.hold_time * pgr.speed * r / HEIGHT_RATIO);
                        NoteKind::Hold { end_time, end_height }
                    }
                    4 => NoteKind::Flick,
                    _ => ptl!(bail "unknown-note-type", "type" => pgr.kind),
                },
                time,
                speed: if pgr.kind == 3 {
                    speed.set_time(time);
                    1.
                } else {
                    pgr.speed
                },
                end_speed: pgr.speed,
                height: pgr.floor_position / HEIGHT_RATIO,
                start_height: {
                    height.set_time(time);
                    height.now()
                },
                hand: smart_assign_hand(pgr.position_x, time, &previous_notes, 0.3),
                above,
                multiple_hint: false,
                fake: false,
                judge: JudgeStatus::NotJudged,
                format: true
            })
        })
        .collect::<Result<Vec<_>>>()
}

fn parse_judge_line(pgr: PgrJudgeLine, max_time: f32, bpm_list: &BpmList, id: usize) -> Result<JudgeLine> {
    if pgr.bpm <= 0.0 {
        bail!("Invalid BPM: {}", pgr.bpm);
    }
    let r = 60. / pgr.bpm / 32.;
    let (mut speed, mut height) = parse_speed_events(r, pgr.speed_events, max_time).context("Failed to parse speed events")?;
    let notes_above = parse_notes(r, pgr.notes_above, &mut speed, &mut height, true).context("Failed to parse notes above")?;
    let mut notes_below = parse_notes(r, pgr.notes_below, &mut speed, &mut height, false).context("Failed to parse notes below")?;
    let mut notes = notes_above;
    let config = Config::default();
    notes.append(&mut notes_below);
    let initial_rotation = pgr.rotate_events.first().map(|e| e.start).unwrap_or(0.0);
    assign_hands(&mut notes, &config, id, initial_rotation, bpm_list);
    let cache = JudgeLineCache::new(&mut notes);
    Ok(JudgeLine {
        object: Object {
            alpha: parse_float_events(r, pgr.alpha_events).with_context(|| ptl!("alpha-events-parse-failed"))?,
            rotation: parse_float_events(r, pgr.rotate_events).with_context(|| ptl!("rotate-events-parse-failed"))?,
            translation: parse_move_events(r, pgr.move_events).with_context(|| ptl!("move-events-parse-failed"))?,
            ..Default::default()
        },
        ctrl_obj: Arc::new(Mutex::new(CtrlObject::default())),
        kind: JudgeLineKind::Normal,
        height,
        incline: AnimFloat::default(),
        notes,
        color: Anim::default(),
        parent: None,
        z_index: 0,
        show_below: false,
        attach_ui: None,

        cache,
    })
}

fn parse_judge_line_chunked(pgr: PgrJudgeLine, max_time: f32, bpm_list: &BpmList, id: usize, chunked_chart: &mut ChunkedChart) -> Result<JudgeLine> {
    if pgr.bpm <= 0.0 {
        bail!("Invalid BPM: {}", pgr.bpm);
    }
    let r = 60. / pgr.bpm / 32.;
    let (mut speed, mut height) = parse_speed_events(r, pgr.speed_events, max_time).context("Failed to parse speed events")?;
    let notes_above = parse_notes_chunked(r, pgr.notes_above, &mut speed, &mut height, true, chunked_chart, id).context("Failed to parse notes above")?;
    let mut notes_below = parse_notes_chunked(r, pgr.notes_below, &mut speed, &mut height, false, chunked_chart, id).context("Failed to parse notes below")?;
    let mut notes = notes_above;
    let config = Config::default();
    notes.append(&mut notes_below);
    let initial_rotation = pgr.rotate_events.first().map(|e| e.start).unwrap_or(0.0);
    assign_hands(&mut notes, &config, id, initial_rotation, bpm_list);
    let cache = JudgeLineCache::new(&mut notes);
    Ok(JudgeLine {
        object: Object {
            alpha: parse_float_events(r, pgr.alpha_events).with_context(|| ptl!("alpha-events-parse-failed"))?,
            rotation: parse_float_events(r, pgr.rotate_events).with_context(|| ptl!("rotate-events-parse-failed"))?,
            translation: parse_move_events(r, pgr.move_events).with_context(|| ptl!("move-events-parse-failed"))?,
            ..Default::default()
        },
        ctrl_obj: Arc::new(Mutex::new(CtrlObject::default())),
        kind: JudgeLineKind::Normal,
        height,
        incline: AnimFloat::default(),
        notes,
        color: Anim::default(),
        parent: None,
        z_index: 0,
        show_below: false,
        attach_ui: None,

        cache,
    })
}

pub fn parse_phigros(source: &str, extra: ChartExtra) -> Result<Chart> {
    let pgr: PgrChart = serde_json::from_str(source).with_context(|| ptl!("json-parse-failed"))?;
    let mut bpm_values = Vec::new();
    let _indices: Vec<usize> = (0..pgr.judge_line_list.len()).collect();
    for (index, judge_line) in pgr.judge_line_list.iter().enumerate() {
        bpm_values.push((index as f32, judge_line.bpm));
    }
    let _r = BpmList::new(bpm_values.clone());

    let max_time = *pgr
        .judge_line_list
        .iter()
        .map(|line| {
            line.notes_above
                .iter()
                .chain(line.notes_below.iter())
                .map(|note| note.time.not_nan())
                .max()
                .unwrap_or_default()
                * (60. / line.bpm / 32.)
        })
        .max()
        .unwrap_or_default()
        + 1.;
    let mut lines = pgr
        .judge_line_list
        .into_iter()
        .enumerate()
        .map(|(id, pgr)| parse_judge_line(pgr, max_time, &_r, id).with_context(|| ptl!("judge-line-location", "jlid" => id)))
        .collect::<Result<Vec<_>>>()?;

    process_lines(&mut lines);
    Ok(Chart::new(pgr.offset, lines, BpmList::new_time(bpm_values), ChartSettings::default(), extra))
}

pub fn parse_phigros_chunked(source: &str, extra: ChartExtra) -> Result<Chart> {
    let pgr: PgrChart = serde_json::from_str(source).with_context(|| ptl!("json-parse-failed"))?;
    let mut bpm_values = Vec::new();
    let _indices: Vec<usize> = (0..pgr.judge_line_list.len()).collect();
    for (index, judge_line) in pgr.judge_line_list.iter().enumerate() {
        bpm_values.push((index as f32, judge_line.bpm));
    }
    let _r = BpmList::new(bpm_values.clone());

    let max_time = *pgr
        .judge_line_list
        .iter()
        .map(|line| {
            line.notes_above
                .iter()
                .chain(line.notes_below.iter())
                .map(|note| note.time.not_nan())
                .max()
                .unwrap_or_default()
                * (60. / line.bpm / 32.)
        })
        .max()
        .unwrap_or_default()
        + 1.;

    let mut chunked_chart = ChunkedChart::new(max_time, 5);
    
    let mut lines = pgr
        .judge_line_list
        .into_iter()
        .enumerate()
        .map(|(id, pgr)| parse_judge_line_chunked(pgr, max_time, &_r, id, &mut chunked_chart).with_context(|| ptl!("judge-line-location", "jlid" => id)))
        .collect::<Result<Vec<_>>>()?;

    process_lines(&mut lines);
    let mut chart = Chart::new(pgr.offset, lines, BpmList::new_time(bpm_values), ChartSettings::default(), extra);
    chart.enable_chunked_loading();
    Ok(chart)
}
