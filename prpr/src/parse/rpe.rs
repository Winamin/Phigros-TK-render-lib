crate::tl_file!("parser" ptl);

use super::{process_lines, RPE_TWEEN_MAP};
use crate::{
    core::{
        Anim, AnimFloat, AnimVector, BezierTween, BpmList, Chart, ChartExtra, ChartSettings, ClampedTween, CtrlObject, GifFrames,
        JudgeLine, JudgeLineCache, JudgeLineKind, Keyframe, Note, NoteKind, Object, StaticTween, Triple, TweenFunction, Tweenable, UIElement, EPS,
        HEIGHT_RATIO,
    },
    ext::NotNanExt,
    fs::FileSystem,
    judge::JudgeStatus,
};
use anyhow::{Context, Result};
use macroquad::prelude::Color;
use serde::Deserialize;
use image::{codecs::gif, AnimationDecoder, DynamicImage};
use std::{collections::HashMap, time::Duration};
use crate::ext::SafeTexture;
use crate::core::note::Hand;
use crate::hand::assign_hands;
use crate::config::Config;
use std::sync::{Arc, Mutex};
use std::rc::Rc;
use std::cell::RefCell;

// 智能手部分配函数
fn smart_assign_hand(
    position_x: f32,
    time: f32,
    previous_notes: &[(f32, f32, Hand)],
    _switch_threshold: f32,
) -> Hand {
    // 时间窗口（秒）内，认为是连续音符
    const TEMPORAL_WINDOW: f32 = 2.0;
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
    if position_x < 0.0 {
        Hand::Left
    } else {
        Hand::Right
    }
}

pub const RPE_WIDTH: f32 = 1350.;
pub const RPE_HEIGHT: f32 = 900.;
const SPEED_RATIO: f32 = 10. / 45. / HEIGHT_RATIO;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RPEBpmItem {
    bpm: f32,
    start_time: Triple,
}

// serde is weird...
fn f32_zero() -> f32 {
    0.
}

fn f32_one() -> f32 {
    1.
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RPEEvent<T = f32> {
    // TODO linkgroup
    #[serde(default = "f32_zero")]
    easing_left: f32,
    #[serde(default = "f32_one")]
    easing_right: f32,
    #[serde(default)]
    bezier: u8,
    #[serde(default)]
    bezier_points: [f32; 4],
    easing_type: i32,
    start: T,
    end: T,
    start_time: Triple,
    end_time: Triple,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RPECtrlEvent {
    easing: u8,
    x: f32,
    #[serde(flatten)]
    value: HashMap<String, f32>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RPESpeedEvent {
    // TODO linkgroup
    start_time: Triple,
    end_time: Triple,
    start: f32,
    end: f32,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RPEEventLayer {
    alpha_events: Option<Vec<RPEEvent>>,
    move_x_events: Option<Vec<RPEEvent>>,
    move_y_events: Option<Vec<RPEEvent>>,
    rotate_events: Option<Vec<RPEEvent>>,
    speed_events: Option<Vec<RPESpeedEvent>>,
}

#[derive(Clone, Deserialize)]
struct RGBColor(u8, u8, u8);
impl From<RGBColor> for Color {
    fn from(RGBColor(r, g, b): RGBColor) -> Self {
        Self::from_rgba(r, g, b, 255)
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RPEExtendedEvents {
    color_events: Option<Vec<RPEEvent<RGBColor>>>,
    text_events: Option<Vec<RPEEvent<String>>>,
    scale_x_events: Option<Vec<RPEEvent>>,
    scale_y_events: Option<Vec<RPEEvent>>,
    incline_events: Option<Vec<RPEEvent>>,
    paint_events: Option<Vec<RPEEvent>>,
    gif_events: Option<Vec<RPEEvent>>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RPENote {
    // TODO above == 0? what does that even mean?
    #[serde(rename = "type")]
    kind: u8,
    above: u8,
    start_time: Triple,
    end_time: Triple,
    position_x: f32,
    y_offset: f32,
    alpha: u16, // some alpha has 256...
    size: f32,
    speed: f32,
    is_fake: u8,
    visible_time: f32,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RPEJudgeLine {
    // TODO group
    // TODO bpmfactor
    #[serde(rename = "Name")]
    name: String,
    #[serde(rename = "Texture")]
    texture: String,
    #[serde(rename = "father")]
    parent: Option<isize>,
    event_layers: Vec<Option<RPEEventLayer>>,
    extended: Option<RPEExtendedEvents>,
    notes: Option<Vec<RPENote>>,
    is_cover: u8,
    #[serde(default)]
    z_order: i32,
    #[serde(rename = "attachUI")]
    attach_ui: Option<UIElement>,

    #[serde(default)]
    pos_control: Vec<RPECtrlEvent>,
    #[serde(default)]
    size_control: Vec<RPECtrlEvent>,
    #[serde(default)]
    alpha_control: Vec<RPECtrlEvent>,
    #[serde(default)]
    y_control: Vec<RPECtrlEvent>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RPEMetadata {
    offset: i32,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RPEChart {
    #[serde(rename = "META")]
    meta: RPEMetadata,
    #[serde(rename = "BPMList")]
    bpm_list: Vec<RPEBpmItem>,
    judge_line_list: Vec<RPEJudgeLine>,
}

type BezierMap = HashMap<(u16, i16, i16), Arc<dyn TweenFunction>>;

fn bezier_key<T>(event: &RPEEvent<T>) -> (u16, i16, i16) {
    let p = &event.bezier_points;
    let int = |p: f32| (p * 100.).round() as i16;
    ((int(p[0]) * 100 + int(p[1])) as u16, int(p[2]), int(p[3]))
}

fn parse_events<T: Tweenable, V: Clone + Into<T>>(
    r: &mut BpmList,
    rpe: &[RPEEvent<V>],
    default: Option<T>,
    bezier_map: &BezierMap,
) -> Result<Anim<T>> {
    let mut kfs = Vec::new();
    if rpe.len() > 0{
        if let Some(default) = default {
            if rpe[0].start_time.beats() != 0.0 {
                kfs.push(Keyframe::new(0.0, default, 0));
            }
        }
    }
    for e in rpe {
        let tween = RPE_TWEEN_MAP.get(e.easing_type.max(1) as usize)
            .copied()
            .unwrap_or(RPE_TWEEN_MAP[0]);
        kfs.push(Keyframe {
            time: r.time(&e.start_time),
            value: e.start.clone().into(),
            tween: {
                if e.bezier != 0 {
                    Arc::clone(&bezier_map[&bezier_key(e)])
                } else if e.easing_left.abs() < EPS && (e.easing_right - 1.0).abs() < EPS {
                    StaticTween::get_arc(tween)
                } else {
                    Arc::new(ClampedTween::new(tween, e.easing_left..e.easing_right))
                }
            },
        });
        kfs.push(Keyframe::new(r.time(&e.end_time), e.end.clone().into(), 0));
    }
    Ok(Anim::new(kfs))
}

fn parse_speed_events(r: &mut BpmList, rpe: &[RPEEventLayer], max_time: f32) -> Result<AnimFloat> {
    let rpe: Vec<_> = rpe.iter().filter_map(|it| it.speed_events.as_ref()).collect();
    if rpe.is_empty() {
        return Ok(AnimFloat::default());
    };
    let anis: Vec<_> = rpe
        .into_iter()
        .map(|it| {
            let mut kfs = Vec::new();
            for e in it {
                kfs.push(Keyframe::new(r.time(&e.start_time), e.start, 2));
                kfs.push(Keyframe::new(r.time(&e.end_time), e.end, 0));
            }
            AnimFloat::new(kfs)
        })
        .collect();
    let mut pts: Vec<_> = anis.iter().flat_map(|it| it.keyframes.iter().map(|it| it.time.not_nan())).collect();
    pts.push(max_time.not_nan());
    pts.sort();
    pts.dedup();
    let mut sani = AnimFloat::chain(anis);
    sani.map_value(|v| v * SPEED_RATIO);
    for i in 0..(pts.len() - 1) {
        let now_time = *pts[i]; // 解引用为 f32
        let end_time = *pts[i + 1]; // 解引用为 f32
        sani.set_time(now_time);
        let start_speed = sani.now();
        sani.set_time(end_time - 1e-4);
        let end_speed = sani.now();

        // 检查速度是否线性变化（加速度恒定）
        let duration = end_time - now_time;
        if duration > EPS && (start_speed - end_speed).abs() > EPS {
            // 计算加速度 (a = Δv/Δt)
            let acceleration = (end_speed - start_speed) / duration;

            // 如果加速度为0（匀速），不需要额外分割
            if acceleration.abs() > EPS {
                // 计算速度过零点的时间（如果存在）
                if start_speed.signum() != end_speed.signum() {
                    let zero_time = now_time - start_speed / acceleration;
                    if zero_time > now_time && zero_time < end_time {
                        pts.push(zero_time.not_nan());
                    }
                }

                // 在区间中点添加关键帧以获得更好的二次曲线近似
                let mid_time = (now_time + end_time) / 2.0;
                pts.push(mid_time.not_nan());
            }
        }
    }
    pts.sort();
    pts.dedup();
    let mut kfs = Vec::new();
    let mut height = 0.0;

    // 确保时间0处有初始关键帧
    if *pts[0] > 0.0 {
        kfs.push(Keyframe::new(0.0, height, 2));
    }

    for i in 0..(pts.len() - 1) {
        let now_time = *pts[i]; // 解引用为 f32
        let end_time = *pts[i + 1]; // 解引用为 f32
        let duration = end_time - now_time;

        sani.set_time(now_time);
        let start_speed = sani.now();
        sani.set_time(end_time - 1e-4);
        let end_speed = sani.now();

        // 精确计算位移增量（使用积分公式）
        let delta_height = if duration < EPS {
            0.0
        } else if (start_speed - end_speed).abs() < EPS {
            // 匀速运动：Δs = v * t
            start_speed * duration
        } else {
            // 匀加速运动：Δs = (v0 + v1)/2 * t
            // 使用精确的二次积分公式
            (start_speed + end_speed) * duration / 2.0
        };

        // 添加当前时间点的关键帧
        kfs.push(Keyframe::new(now_time, height, 2));
        height += delta_height;
    }

    // 添加最终关键帧
    kfs.push(Keyframe::new(max_time, height, 0));

    Ok(AnimFloat::new(kfs))
}

fn parse_notes(r: &mut BpmList, rpe: Vec<RPENote>, height: &mut AnimFloat) -> Result<Vec<Note>> {
    // 用于智能手部分配的追踪
    let mut previous_notes: Vec<(f32, f32, Hand)> = Vec::new();

    let notes = rpe.into_iter()
        .map(|note| {
            let time = r.time(&note.start_time);
            let y_offset = note.y_offset * 2. / RPE_HEIGHT * note.speed;

            // Set time and get height after calculating y_offset
            height.set_time(time);
            let note_height = height.now() + y_offset;
            let normalized_x = note.position_x / (RPE_WIDTH / 2.0) - 1.0;

            Ok(Note {
                object: Object {
                    alpha: if note.visible_time >= time {
                        if note.alpha >= 255 {
                            AnimFloat::default()
                        } else {
                            AnimFloat::fixed(note.alpha as f32 / 255.)
                        }
                    } else {
                        let alpha = note.alpha.min(255) as f32 / 255.;
                        AnimFloat::new(vec![Keyframe::new(0.0, 0.0, 0), Keyframe::new(time - note.visible_time, alpha, 0)])
                    },
                    translation: AnimVector(AnimFloat::fixed(note.position_x / (RPE_WIDTH / 2.)), AnimFloat::fixed(y_offset)),
                    scale: AnimVector(
                        if note.size == 1.0 {
                            AnimFloat::default()
                        } else {
                            AnimFloat::fixed(note.size)
                        },
                        AnimFloat::default(),
                    ),
                    ..Default::default()
                },
                kind: match note.kind {
                    1 => NoteKind::Click,
                    2 => {
                        let end_time = r.time(&note.end_time);
                        height.set_time(end_time);
                        NoteKind::Hold {
                            end_time,
                            end_height: height.now() + y_offset,
                        }
                    }
                    3 => NoteKind::Flick,
                    4 => NoteKind::Drag,
                    _ => ptl!(bail "unknown-note-type", "type" => note.kind),
                },
                time,
                height: note_height,
                speed: note.speed,
                end_speed: note.speed,
                start_height: {
                    height.set_time(r.time(&note.start_time));
                    height.now() + y_offset
                },
                hand: smart_assign_hand(normalized_x, time, &previous_notes, 0.3),
                above: note.above == 1,
                multiple_hint: false,
                fake: note.is_fake != 0,
                judge: JudgeStatus::NotJudged,
                format: false,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    // 更新追踪列表
    for note in &notes {
        let pos = note.object.translation.0.now();
        previous_notes.push((note.time, pos, note.hand));
    }

    Ok(notes)
}

fn parse_ctrl_events(rpe: &[RPECtrlEvent], key: &str) -> AnimFloat {
    let vals: Vec<_> = rpe
        .iter()
        .filter_map(|it| it.value.get(key).copied())
        .collect();

    if rpe.is_empty() || (rpe.len() == 2 && rpe[0].easing == 1 && (vals[0] - 1.).abs() < 1e-4) {
        return AnimFloat::default();
    }

    // 修复：声明并初始化 kfs 向量
    let mut kfs = Vec::new(); // <-- 添加这行声明 kfs

    // 修复：替换未定义的 default 变量
    // 原代码中的 default 可能是想用第一个值作为默认值？
    if let Some(&first_val) = vals.first() { // <-- 使用 vals 的第一个值
        kfs.push(Keyframe::new(0.0, first_val, 0)); // 注意：这里使用的缓动类型0需要确认是否合适
    }

    // 添加其他关键帧
    for (it, val) in rpe.iter().zip(vals.iter().copied()) {
        kfs.push(Keyframe::new(
            it.x,
            val,
            RPE_TWEEN_MAP.get(it.easing.max(1) as usize).copied().unwrap_or(RPE_TWEEN_MAP[0])
        ));
    }

    AnimFloat::new(kfs)
}

async fn parse_judge_line(
    r: &mut BpmList,
    rpe: RPEJudgeLine,
    max_time: f32,
    fs: &mut dyn FileSystem,
    bezier_map: &BezierMap,
    texture_cache: &mut std::collections::HashMap<String, SafeTexture>,
    id: usize
) -> Result<JudgeLine> {
    let event_layers: Vec<_> = rpe.event_layers.into_iter().flatten().collect();

    fn events_with_factor(
        r: &mut BpmList,
        event_layers: &[RPEEventLayer],
        get: impl Fn(&RPEEventLayer) -> &Option<Vec<RPEEvent>>,
        factor: f32,
        desc: &str,
        bezier_map: &BezierMap,
    ) -> Result<AnimFloat> {
        let anis: Vec<_> = event_layers
            .iter()
            .filter_map(|it| get(it).as_ref().map(|es| parse_events(r, es, None, bezier_map)))
            .collect::<Result<_>>()
            .with_context(|| ptl!("type-events-parse-failed", "type" => desc))?;
        let mut res = AnimFloat::chain(anis);
        res.map_value(|v| v * factor);
        Ok(res)
    }

    let mut height = parse_speed_events(r, &event_layers, max_time)?;
    let mut notes = parse_notes(r, rpe.notes.unwrap_or_default(), &mut height)?;
    let config = Config::default();
    let mut rotation = events_with_factor(r, &event_layers, |it| &it.rotate_events, -1., "rotate", bezier_map)?;
    rotation.set_time(0.0);
    let rotation_angle = rotation.now();
    assign_hands(&mut notes, &config, id, rotation_angle, r);
    let cache = JudgeLineCache::new(&mut notes);

    Ok(JudgeLine {
        object: Object {
            alpha: events_with_factor(r, &event_layers, |it| &it.alpha_events, 1. / 255., "alpha", bezier_map)?,
            rotation: events_with_factor(r, &event_layers, |it| &it.rotate_events, -1., "rotate", bezier_map)?,
            translation: AnimVector(
                events_with_factor(r, &event_layers, |it| &it.move_x_events, 2. / RPE_WIDTH, "move X", bezier_map)?,
                events_with_factor(r, &event_layers, |it| &it.move_y_events, 2. / RPE_HEIGHT, "move Y", bezier_map)?,
            ),
            scale: {
                fn parse(r: &mut BpmList, opt: &Option<Vec<RPEEvent>>, factor: f32, bezier_map: &BezierMap) -> Result<AnimFloat> {
                    let mut res = opt
                        .as_ref()
                        .map(|it| parse_events(r, it, None, bezier_map))
                        .transpose()?
                        .unwrap_or_default();
                    res.map_value(|v| v * factor);
                    Ok(res)
                }
                let factor = if rpe.texture == "line.png" {
                    1.
                } else {
                    2. / RPE_WIDTH /*TODO tweak*/
                };
                rpe.extended
                    .as_ref()
                    .map(|e| -> Result<_> {
                        Ok(AnimVector(
                            parse(
                                r,
                                &e.scale_x_events,
                                factor
                                    * if rpe.texture == "line.png"
                                    && rpe
                                    .extended
                                    .as_ref()
                                    .map_or(true, |it| it.text_events.as_ref().map_or(true, |it| it.is_empty()))
                                    && rpe.attach_ui.is_none()
                                {
                                    0.5
                                } else {
                                    1.
                                },
                                bezier_map,
                            )?,
                            parse(r, &e.scale_y_events, factor, bezier_map)?,
                        ))
                    })
                    .transpose()?
                    .unwrap_or_default()
            },
        },
        ctrl_obj: Rc::new(RefCell::new(CtrlObject {
            alpha: parse_ctrl_events(&rpe.alpha_control, "alpha"),
            size: parse_ctrl_events(&rpe.size_control, "size"),
            pos: parse_ctrl_events(&rpe.pos_control, "pos"),
            y: parse_ctrl_events(&rpe.y_control, "y"),
        })),
        height,
        incline: if let Some(events) = rpe.extended.as_ref().and_then(|e| e.incline_events.as_ref()) {
            parse_events(r, events, Some(0.), bezier_map).with_context(|| ptl!("incline-events-parse-failed"))?
        } else {
            AnimFloat::default()
        },
        notes,
        kind: if rpe.texture == "line.png" {
            if let Some(events) = rpe.extended.as_ref().and_then(|e| e.paint_events.as_ref()) {
                JudgeLineKind::Paint(
                    parse_events(r, events, Some(-1.), bezier_map).with_context(|| ptl!("paint-events-parse-failed"))?,
                    Arc::new(Mutex::new((None, false))),
                )
            } else if let Some(extended) = rpe.extended.as_ref() {
                if let Some(events) = extended.gif_events.as_ref() {
                    let data = fs
                        .load_file(&rpe.texture)
                        .await
                        .with_context(|| ptl!("gif-load-failed", "path" => rpe.texture.clone()))?;
                    let decoder = gif::GifDecoder::new(&data[..])?;
                    let frames = GifFrames::new(
                        decoder
                            .into_frames()
                            .map(|frame| -> (u128, SafeTexture) {
                                let frame = frame.unwrap();
                                let delay: Duration = frame.delay().into();
                                (delay.as_millis(), SafeTexture::from(DynamicImage::ImageRgba8(frame.into_buffer())))
                            })
                            .collect(),
                    );
                    // TODO: process events
                    let events = parse_events(r, events, Some(0.), bezier_map).with_context(|| ptl!("gif-events-parse-failed"))?;
                    JudgeLineKind::TextureGif(events, frames, rpe.texture.clone())
                } else if let Some(events) = extended.text_events.as_ref() {
                    JudgeLineKind::Text(parse_events(r, events, Some(String::new()), bezier_map).with_context(|| ptl!("text-events-parse-failed"))?)
                } else {
                    JudgeLineKind::Normal
                }
            } else {
                JudgeLineKind::Normal
            }
        } else {
            match texture_cache.get(&rpe.texture) {
                Some(texture) => JudgeLineKind::Texture(texture.clone(), rpe.texture.clone()),
                None => {
                    let img_data = fs
                        .load_file(&rpe.texture)
                        .await
                        .with_context(|| ptl!("illustration-load-failed", "path" => rpe.texture.clone()))?;
                    let img = image::load_from_memory(&img_data)?;
                    let texture = SafeTexture::from_image(&img).with_mipmap();
                    texture_cache.insert(rpe.texture.clone(), texture.clone());

                    JudgeLineKind::Texture(texture, rpe.texture.clone())
                }
            }
        },
        color: if let Some(events) = rpe.extended.as_ref().and_then(|e| e.color_events.as_ref()) {
            parse_events(r, events, Some(Color::new(0.0, 0.0, 0.0, 0.0)), bezier_map).with_context(|| ptl!("color-events-parse-failed"))?
        } else {
            Anim::default()
        },
        parent: {
            let parent = rpe.parent.unwrap_or(-1);
            if parent == -1 {
                None
            } else {
                Some(parent as usize)
            }
        },
        z_index: rpe.z_order,
        show_below: rpe.is_cover != 1,
        attach_ui: rpe.attach_ui,
        cache,
        cached_world_pos: None,
    })
}

fn add_bezier<T>(map: &mut BezierMap, event: &RPEEvent<T>) {
    if event.bezier != 0 {
        let p = &event.bezier_points;
        let int = |p: f32| (p * 100.).round() as i16;
        map.entry(((int(p[0]) * 100 + int(p[1])) as u16, int(p[2]), int(p[3])))
            .or_insert_with(|| Arc::new(BezierTween::new((p[0], p[1]), (p[2], p[3]))));
    }
}

fn get_bezier_map(rpe: &RPEChart) -> BezierMap {
    let mut map = HashMap::new();
    for line in &rpe.judge_line_list {
        for event_layer in line.event_layers.iter().flatten() {
            for event in event_layer
                .alpha_events
                .iter()
                .chain(event_layer.move_x_events.iter())
                .chain(event_layer.move_y_events.iter())
                .chain(event_layer.rotate_events.iter())
                .flatten()
            {
                add_bezier(&mut map, event);
            }
        }
    }
    map
}

pub async fn parse_rpe(source: &str, fs: &mut dyn FileSystem, extra: ChartExtra) -> Result<Chart> {
    let rpe: RPEChart = serde_json::from_str(source).with_context(|| ptl!("json-parse-failed"))?;
    let bezier_map = get_bezier_map(&rpe);
    let mut r = BpmList::new(rpe.bpm_list.into_iter().map(|it| (it.start_time.beats(), it.bpm)).collect());
    let mut texture_cache = std::collections::HashMap::new();
    fn vec<T>(v: &Option<Vec<T>>) -> impl Iterator<Item = &T> {
        v.iter().flat_map(|it| it.iter())
    }

    #[rustfmt::skip]
    let max_time = *rpe
        .judge_line_list
        .iter()
        .map(|line| {
            line.notes.as_ref().map(|notes| {
                notes
                    .iter()
                    .map(|note| {
                        // 修复：在这里处理Hold音符的结束时间
                        let time = if note.kind == 2 { // Hold类型
                            r.time(&note.end_time)
                        } else {
                            r.time(&note.start_time)
                        };
                        time.not_nan()
                    })
                    .max()
                    .unwrap_or_default()
            }).unwrap_or_default().max(
                line.event_layers.iter().filter_map(|it| it.as_ref().map(|layer| {
                    vec(&layer.alpha_events)
                        .chain(vec(&layer.move_x_events))
                        .chain(vec(&layer.move_y_events))
                        .chain(vec(&layer.rotate_events))
                        .map(|it| r.time(&it.end_time).not_nan())
                        .max().unwrap_or_default()
                })).max().unwrap_or_default()
            ).max(
                line.extended.as_ref().map(|e| {
                    vec(&e.scale_x_events)
                        .chain(vec(&e.scale_y_events))
                        .map(|it| r.time(&it.end_time).not_nan())
                        .max().unwrap_or_default()
                        .max(vec(&e.text_events).map(|it| r.time(&it.end_time).not_nan()).max().unwrap_or_default())
                }).unwrap_or_default()
            )
        })
        .max().unwrap_or_default() + 1.;

    let mut lines = Vec::new();
    for (id, rpe) in rpe.judge_line_list.into_iter().enumerate() {
        let name = rpe.name.clone();
        lines.push(
            parse_judge_line(&mut r, rpe, max_time, fs, &bezier_map, &mut texture_cache, id)
                .await
                .with_context(move || ptl!("judge-line-location-name", "jlid" => id, "name" => name))?,
        );
    }
    process_lines(&mut lines);
    Ok(Chart::new(rpe.meta.offset as f32 / 1000.0, lines, r, ChartSettings::default(), extra))
}