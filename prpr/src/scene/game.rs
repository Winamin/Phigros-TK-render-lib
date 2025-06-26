#![allow(unused)]

crate::tl_file!("game");

use super::{
    draw_background,
    ending::RecordUpdateState,
    loading::{BasicPlayer, UpdateFn, UploadFn},
    request_input, return_input, show_message, take_input, EndingScene, NextScene, Scene,
};
use crate::core::NoteKind;
use crate::{
    bin::{BinaryReader, BinaryWriter},
    config::{Config, Mods},
    core::{copy_fbo, BadNote, Chart, ChartExtra, Effect, Point, Resource, UIElement, Vector},
    ext::{parse_time, screen_aspect, semi_white, RectExt, SafeTexture},
    fs::FileSystem,
    info::{ChartFormat, ChartInfo},
    judge::Judge,
    parse::{parse_extra, parse_pec, parse_phigros, parse_rpe},
    task::Task,
    time::TimeManager,
    ui::{RectButton, Ui},
};
use anyhow::{bail, Context, Result};
use concat_string::concat_string;
use lyon::path::Path;
use macroquad::{prelude::*, window::InternalGlContext};
use sasa::{Music, MusicParams};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use std::{
    any::Any,
    cell::RefCell,
    fs::File,
    io::{Cursor, ErrorKind},
    ops::{DerefMut, Range},
    path::PathBuf,
    process::{Command, Stdio},
    rc::Rc,
    sync::{Arc, Mutex},
};
use tracing::{debug, warn};
use crate::judge::JudgeStatus;

const PAUSE_CLICK_INTERVAL: f32 = 0.7;

#[cfg(feature = "closed")]
mod inner;
#[cfg(feature = "closed")]
use inner::*;

const WAIT_TIME: f32 = 0.5;
const AFTER_TIME: f32 = 0.7;

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SimpleRecord {
    pub score: i32,
    pub accuracy: f32,
    pub full_combo: bool,
}

impl SimpleRecord {
    pub fn update(&mut self, other: &SimpleRecord) -> bool {
        let mut changed = false;
        if other.score > self.score {
            self.score = other.score;
            changed = true;
        }
        if other.accuracy > self.accuracy {
            self.accuracy = other.accuracy;
            changed = true;
        }
        if other.full_combo & !self.full_combo {
            self.full_combo = other.full_combo;
            changed = true;
        }
        changed
    }
}

fn fmt_time(t: f32) -> String {
    let f = t < 0.;
    let t = t.abs();
    let secs = t % 60.;
    let mut t = (t / 60.) as u64;
    let mins = t % 60;
    t /= 60;
    let hrs = t % 100;
    format!("{}{mins:02}:{secs:02.0}", if f { "-" } else { "" })
    //format!("{}{hrs:02}:{mins:02}:{secs:05.2}", if f { "-" } else { "" })
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
extern "C" {
    fn on_game_start();
}

#[derive(PartialEq, Eq)]
pub enum GameMode {
    Normal,
    TweakOffset,
    Exercise,
    NoRetry,
    View,
}

#[derive(Clone)]
enum State {
    Starting,
    BeforeMusic,
    Playing,
    Ending,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum NoteType {
    Click,
    Drag,
    Flick,
    Hold,
}

// 判定计量器，保存统计数据及动画信息
#[derive(Clone)]
struct JudgementCounter {
    note_type: NoteType,
    count: u32,
    last_update: f64,
    current_x: f32,
    current_y: f32,
    current_alpha: f32,
    multiplier: u32,
    interval: f32,
}

impl JudgementCounter {
    fn new(note_type: NoteType, chart_ratio: f32, initial_y: f32) -> Self {
        Self {
            note_type,
            count: 0,
            last_update: 0.0,
            current_x: 1.2, // 初始横向偏移
            current_y: initial_y,// 初始垂直位置
            current_alpha: 0.0,
            multiplier: 1,
            interval: 0.0,
        }
    }
    fn target_x(&self, current_time: f64) -> f32 {
        if current_time - self.last_update <= 1.0 {
            0.0
        } else {
            -0.2
        }
    }
    fn target_alpha(&self, current_time: f64) -> f32 {
        if current_time - self.last_update <= 1.0 {
            0.92
        } else {
            0.0
        }
    }
    fn update_position(&mut self, current_time: f64, dt: f32) {
        let target = self.target_x(current_time);
        self.current_x += (target - self.current_x) * dt * 5.0;
    }
    fn update_alpha(&mut self, current_time: f64, dt: f32) {
        let target = self.target_alpha(current_time);
        self.current_alpha += (target - self.current_alpha) * dt * 5.0;
    }
    fn update_vertical(&mut self, target_y: f32, dt: f32) {
        let duration = 0.05;
        //dt/duration 表示本帧占比
        self.current_y += (target_y - self.current_y) * (dt / duration).min(1.0);
    }
    // 统一更新
    fn update(&mut self, current_time: f64, dt: f32, target_y: f32) {
        self.update_position(current_time, dt);
        self.update_alpha(current_time, dt);
        self.update_vertical(target_y, dt);
    }
    fn display_text(&self) -> String {
        let type_str = match self.note_type {
            NoteType::Click => "Tap",
            NoteType::Drag  => "Drag",
            NoteType::Flick => "Flick",
            NoteType::Hold  => "Hold",
        };
        if self.multiplier > 1 {
            //format!("X{} {}:{} [{:.2}s]", self.multiplier, type_str, self.count, self.interval)
            format!("X{} {}:{}", self.multiplier, type_str, self.count)
        } else {
            format!("{}:{} [{:.2}s]", type_str, self.count, self.interval)
        }
    }
    fn reset(&mut self, chart_ratio: f32, initial_y: f32) {
        self.count = 0;
        self.multiplier = 1;
        self.interval = 0.0;
        self.last_update = 0.0;
        self.current_alpha = 0.0;
        self.current_x = 1.2;
        self.current_y = initial_y;
    }
}

pub struct GameScene {
    should_exit: bool,
    next_scene: Option<NextScene>,

    pub mode: GameMode,
    pub res: Resource,
    pub chart: Chart,
    pub judge: Judge,
    pub gl: InternalGlContext<'static>,
    player: Option<BasicPlayer>,
    chart_bytes: Vec<u8>,
    chart_format: ChartFormat,
    info_offset: f32,
    effects: Vec<Effect>,

    first_in: bool,
    exercise_range: Range<f32>,
    exercise_press: Option<(i8, u64)>,
    exercise_btns: (RectButton, RectButton),

    pub music: Music,

    state: State,
    pub last_update_time: f64,
    pause_rewind: Option<f64>,
    pause_first_time: f32,

    pub bad_notes: Vec<BadNote>,

    upload_fn: Option<UploadFn>,
    update_fn: Option<UpdateFn>,

    pub touch_points: Vec<(f32, f32)>,
    pub is_fast_forwarding: bool,

    judgement_counters: Vec<JudgementCounter>,
    judgement_reset_done: bool,

    pub target_chart_ratio: f32,  // 目标缩放比例
    pub current_chart_ratio: f32, // 当前缩放比例
}

macro_rules! reset {
    ($self:ident, $res:expr, $tm:ident) => {{
        $self.bad_notes.clear();
        $self.judge.reset();
        $self.chart.reset();
        $res.judge_line_color = Color::from_hex($res.res_pack.info.color_perfect_line);
        $self.music.pause()?;
        $self.music.seek_to(0.)?;
        $tm.speed = $res.config.speed as _;
        $tm.reset();
        $self.last_update_time = $tm.now();
        for counter in $self.judgement_counters.iter_mut() {
            counter.reset($res.config.chart_ratio, 0.0);
        }
        $self.state = State::Starting;
    }};
}

macro_rules! reset_speed {
    ($self:ident, $res:expr, $tm:ident) => {{
        $self.bad_notes.clear();
        $self.judge.reset();
        $self.chart.reset();
        $res.judge_line_color = Color::from_hex($res.res_pack.info.color_perfect_line);
        $self.music.pause();
        $self.music.seek_to(0.);
        $tm.speed = $res.config.speed as _;
        $tm.reset();
        $self.last_update_time = $tm.now();
        $self.state = State::Starting;
    }};
}

impl GameScene {
    pub const BEFORE_TIME: f32 = 0.7;
    pub const BEFORE_DURATION: f32 = 1.2;
    pub const WAIT_AFTER_TIME: f32 = AFTER_TIME + 0.3;
    pub const FADEOUT_TIME: f32 = WAIT_TIME + AFTER_TIME + 0.3;

    pub async fn load_chart_bytes(fs: &mut dyn FileSystem, info: &ChartInfo) -> Result<Vec<u8>> {
        if let Ok(bytes) = fs.load_file(&info.chart).await {
            return Ok(bytes);
        }
        if let Some(name) = info.chart.strip_suffix(".pec") {
            if let Ok(bytes) = fs.load_file(&concat_string!(name, ".json")).await {
                return Ok(bytes);
            }
        }
        bail!("Cannot find chart file")
    }

    pub async fn load_chart(fs: &mut dyn FileSystem, info: &ChartInfo) -> Result<(Chart, Vec<u8>, ChartFormat)> {
        let extra = fs.load_file("extra.json").await.ok().map(String::from_utf8).transpose()?;
        let extra = if let Some(extra) = extra {
            parse_extra(&extra, fs).await.context("Failed to parse extra")?
        } else {
            ChartExtra::default()
        };
        let bytes = Self::load_chart_bytes(fs, info).await.context("Failed to load chart")?;
        let format = info.format.clone().unwrap_or_else(|| {
            if let Ok(text) = String::from_utf8(bytes.clone()) {
                if text.starts_with('{') {
                    if text.contains("\"META\"") {
                        ChartFormat::Rpe
                    } else {
                        ChartFormat::Pgr
                    }
                } else {
                    ChartFormat::Pec
                }
            } else {
                ChartFormat::Pbc
            }
        });
        let mut chart = match format {
            ChartFormat::Rpe => parse_rpe(&String::from_utf8_lossy(&bytes), fs, extra).await,
            ChartFormat::Pgr => parse_phigros(&String::from_utf8_lossy(&bytes), extra),
            ChartFormat::Pec => parse_pec(&String::from_utf8_lossy(&bytes), extra),
            ChartFormat::Pbc => {
                let mut r = BinaryReader::new(Cursor::new(&bytes));
                r.read()
            }
        }?;
        chart.load_textures(fs).await?;
        chart.settings.hold_partial_cover = info.hold_partial_cover;
        Ok((chart, bytes, format))
    }

    fn draw_judgement_counters(&self, ui: &mut Ui, tm: &TimeManager, _base_x: f32, base_y: f32) {
        if !self.res.config.chart_debug {
            return;
        }
        let chart_ratio = self.res.config.chart_ratio;
        let margin = -0.77 / chart_ratio;
        let fixed_x = margin;
        let base_spacing = 0.2;
        let spacing = (base_spacing / chart_ratio).max(0.13);
        let gap = -1.0;
        let extra_offset = 0.8;
        let target_base_y = base_y - gap + extra_offset;
    
        let mut counters = self.judgement_counters.clone();
        counters.sort_by(|a, b| a.last_update.partial_cmp(&b.last_update).unwrap());
    
        for (i, counter) in counters.iter().enumerate() {
            let target_y = target_base_y - (i as f32 * spacing);
            let pos_y = counter.current_y;
            ui.text(&counter.display_text())
                .pos(fixed_x - 0.078, pos_y - 0.18)
                .anchor(0.5, 0.5)
                .size((0.32 / chart_ratio))
                .color(Color::new(1.0, 1.0, 1.0, counter.current_alpha))
            .draw();
        }
    }

    pub async fn new(
        mode: GameMode,
        info: ChartInfo,
        mut config: Config,
        mut fs: Box<dyn FileSystem>,
        player: Option<BasicPlayer>,
        background: SafeTexture,
        illustration: SafeTexture,
        upload_fn: Option<UploadFn>,
        update_fn: Option<UpdateFn>,
    ) -> Result<Self> {
        match mode {
            GameMode::TweakOffset => {
                config.mods.insert(Mods::AUTOPLAY);
            }
            GameMode::Exercise => {
                config.mods.remove(Mods::AUTOPLAY);
            }
            _ => {}
        }
        let (mut chart, chart_bytes, chart_format) = Self::load_chart(fs.deref_mut(), &info).await?;
        let effects = std::mem::take(&mut chart.extra.global_effects);
        if config.fxaa {
            chart
                .extra
                .effects
                .push(Effect::new(0.0..f32::INFINITY, include_str!("fxaa.glsl"), Vec::new(), false).unwrap());
        }

        let info_offset = info.offset;
        let mut res = Resource::new(
            config.clone(),
            info,
            fs,
            player.as_ref().and_then(|it| it.avatar.clone()),
            background,
            illustration,
            chart.extra.effects.is_empty() && effects.is_empty(),
        )
        .await
        .context("Failed to load resources")?;
        let exercise_range = (chart.offset + info_offset + res.config.offset)..res.track_length;

        let judge = Judge::new(&chart);

        let music = Self::new_music(&mut res)?;

        let chart_ratio = res.config.chart_ratio;
        let judgement_counters = vec![
            JudgementCounter::new(NoteType::Click, chart_ratio, 0.0),
            JudgementCounter::new(NoteType::Drag, chart_ratio, 0.0),
            JudgementCounter::new(NoteType::Flick, chart_ratio, 0.0),
            JudgementCounter::new(NoteType::Hold, chart_ratio, 0.0),
        ];
        let target_chart_ratio = config.chart_ratio;

        Ok(Self {
            should_exit: false,
            next_scene: None,

            mode,
            res,
            chart,
            judge,
            gl: unsafe { get_internal_gl() },
            player,
            chart_bytes,
            chart_format,
            effects,
            info_offset,

            first_in: false,
            exercise_range,
            exercise_press: None,
            exercise_btns: (RectButton::new(), RectButton::new()),

            music,

            state: State::Starting,
            last_update_time: 0.,
            pause_rewind: None,
            pause_first_time: f32::NEG_INFINITY,

            bad_notes: Vec::new(),

            upload_fn,
            update_fn,

            touch_points: Vec::new(),
            is_fast_forwarding: false,

            judgement_counters,

            judgement_reset_done: false,

            target_chart_ratio,
            current_chart_ratio: 1.0,
        })
    }

    fn new_music(res: &mut Resource) -> Result<Music> {
        res.audio.create_music(
            res.music.clone(),
            MusicParams {
                amplifier: res.config.volume_music as _,
                playback_rate: res.config.speed as _,
                ..Default::default()
            },
        )
    }

    fn touch_scale(&self) -> f32 {
        (screen_width() / screen_height()) / self.res.aspect_ratio
    }

    fn ui(&mut self, ui: &mut Ui, tm: &mut TimeManager) -> Result<()> {
        let time = tm.now() as f32;
        let p = match self.state {
            State::Starting => {
                if time <= Self::BEFORE_TIME {
                    1. - (1. - time / Self::BEFORE_TIME).powi(3)
                } else {
                    1.
                }
            }
            State::BeforeMusic => 1.,
            State::Playing => 1.,
            State::Ending => {
                let t = time - self.res.track_length - WAIT_TIME;
                1. - (t / (AFTER_TIME + 0.3)).min(1.).powi(2)
            }
        };
        let c = Color::new(1., 1., 1., self.res.alpha);
        let res = &mut self.res;
        let eps = 2e-2 / res.aspect_ratio;
        let top = -1. / res.aspect_ratio;
        let pause_w = 0.011;
        let pause_h = pause_w * 3.4;
        let pause_center = Point::new(pause_w * 4.4 - 1., top + eps * 3.6454 - (1. - p) * 0.4 + pause_h / 2.);
        if res.config.ui_pause {
            if res.config.interactive
                && !tm.paused()
                && self.pause_rewind.is_none()
                && Judge::get_touches().iter().any(|touch| {
                touch.phase == TouchPhase::Started && {
                    let p = touch.position;
                    let p = Point::new(p.x, p.y);
                    (pause_center - p).norm() < 0.05
                }
            })
            {
                let t = tm.now() as f32;
                if t - self.pause_first_time > PAUSE_CLICK_INTERVAL && res.config.double_click_to_pause {
                    self.pause_first_time = t;
                } else {
                    self.pause_first_time = f32::NEG_INFINITY;
                    if !self.music.paused() {
                        self.music.pause()?;
                    }
                    tm.pause();
                }
            }
        }
        if tm.now() as f32 - self.pause_first_time <= PAUSE_CLICK_INTERVAL {
            ui.fill_circle(pause_center.x, pause_center.y, 0.05, Color::new(1., 1., 1., 0.5));
        }
        let score = format!("{:07}", self.judge.score());
        let margin = 0.046;
        let score_top = top + eps * 2.2 - (1. - p) * 0.4;
        let ct = ui.text(&score).size(0.8).center();
        if res.config.ui_score {
            self.chart.with_element(ui, res, UIElement::Score, Some((-ct.x + 1. - margin, ct.y + score_top)), Some((1. - margin + 0.001, top + eps * 2.8125)), |ui, color| {
                let mut text_size = 0.70867;
                let mut text = ui.text(&score).size(text_size);
                let max_width = 0.55;
                let text_width = text.measure().w;
                if text_width > max_width {
                    text_size *= max_width / text_width
                }
                drop(text);
                    ui.text(format!("{:07}", self.judge.score()))
                        .pos(1. - margin + 0.001, top + eps * 2.8125 - (1. - p) * 0.4)
                        .anchor(1., 0.)
                        .size(0.70867)
                        .color(Color { a: color.a * c.a, ..color })
                        .draw();
            });
        }
        if res.config.show_acc {
            ui.text(format!("{:05.2}%", self.judge.real_time_accuracy() * 100.))
                .pos(1. - margin, top + eps * 2.2 - (1. - p) * 0.4 + 0.07)
                .anchor(1., 0.)
                .size(0.4)
                .color(semi_white(0.7))
                .draw();
        }
        if res.config.ui_pause {
            self.chart.with_element(ui, res, UIElement::Pause, Some((pause_center.x, pause_center.y)), Some((pause_center.x - pause_w * 1.2, pause_center.y - pause_h / 2.2)), |ui, color| {
                let mut r = Rect::new(pause_center.x - pause_w * 1.2, pause_center.y - pause_h / 2.2, pause_w, pause_h);
                let c = Color { a: color.a * c.a, ..color };
                ui.fill_rect(r, c);
                r.x += pause_w * 2.;
                ui.fill_rect(r, c);
            });
        }
        let unit_h = ui.text("0").measure().h;
        let combo_top = top + eps * 1.346 - (1. - p) * 0.4;
        if res.config.ui_combo {
            if self.judge.combo() >= 3 {
                let btm = self.chart.with_element(ui, res, UIElement::ComboNumber, Some((0., combo_top + unit_h / 2.)), Some((0., combo_top + unit_h / 2.)), |ui, color| {
                    let mut text_size = 1.;
                    let max_width = 0.55;
                    let mut text = ui.text(&res.config.combo)
                        .pos(0., top + eps * 1.346 - (1. - p) * 0.4)
                        .anchor(0.5, 0.)
                        .color(Color::new(0., 0., 0., 0.));
                    let text_width = text.measure().w;
                    let text_btm = text.draw().bottom();
                    if text_width > max_width {
                        text_size *= max_width / text_width
                    }
                    ui.text(self.judge.combo().to_string())
                        .pos(0., top + eps * 1.346 - (1. - p) * 0.4)
                        .anchor(0.5, 0.)
                        .color(Color { a: color.a * c.a, ..color })
                        .size(text_size)
                        .draw();
                    text_btm
                });
                self.chart.with_element(ui, res, UIElement::Combo, Some((0., btm + 0.007777 + unit_h * 0.325 / 2.)), Some((0., btm + 0.007777 + unit_h * 0.325 / 2.)), |ui, color| {
                    ui.text(&res.config.combo)
                        .pos(0., btm + 0.007777)
                        .anchor(0.5, 0.)
                        .size(0.325)
                        .color(Color { a: color.a * c.a, ..color })
                        .draw();
                });
            }
        }
        let lf = -1. + margin;
        let bt = -top - eps * 3.64;
        if res.config.ui_name {
            self.chart.with_element(ui, res, UIElement::Name, Some((lf + ct.x, bt - ct.y)), Some((-1. + margin * 0.7, -top - eps * 2.)), |ui, color| {
                let mut text_size = 0.5;
                let mut text = ui.text(&res.info.name).size(text_size);
                let max_width = 0.9;
                let text_width = text.measure().w;
                if text_width > max_width {
                    text_size *= max_width / text_width
                }
                drop(text);
                ui.text(&res.info.name)
                    .pos(lf, bt + (1. - p) * 0.4)
                    .anchor(0., 1.)
                    .size(text_size)
                    .color(Color { a: color.a * c.a, ..color })
                    .draw();
            });
        }
        if res.config.ui_level {
            self.chart.with_element(ui, res, UIElement::Level, Some((-lf - ct.x, bt - ct.y)), Some((1. - margin * 0.7, -top - eps * 2.)), |ui, color| {
                ui.text(&res.info.level)
                    .pos(-lf, bt + (1. - p) * 0.4)
                    .anchor(1., 1.)
                    .size(0.5)
                    .color(Color { a: color.a * c.a, ..color })
                    .draw();
            });
        }
        {
            let watermark = res.config.watermark.clone();
            if res.config.chart_ratio >= 0.95 {
                ui.text(&watermark)
                    .pos(0., -top * 0.98 + (1. - p) * 0.4)
                    .anchor(0.5, 1.)
                    .size(0.25)
                    .color(Color::new(1., 1., 1., 0.5 * c.a))
                    .draw();
            } else {
                ui.text(&watermark)
                    .pos(0., (-top * 0.98 + (1. - p) * 0.4) / res.config.chart_ratio)
                    .anchor(0.5, 1.)
                    .size(0.25 / res.config.chart_ratio)
                    .color(Color::new(1., 1., 1., 0.5 * c.a))
                    .draw();
            }
        };
        let hw = 0.0015;
        let height = eps * 1.1;
        let mut dest = (2. * res.time / res.track_length).min(2.0);
        let mut bar_y = top;
        let mut bar_alpha = 1.0;
        if matches!(self.state, State::Ending) {
            let t = time - res.track_length - WAIT_TIME;
            let progress = (t / (AFTER_TIME + 0.3)).min(1.0);
            bar_alpha = 1.0 - progress.powi(2);
            bar_y = top - progress * height * 2.5;
        }
        if res.config.ui_pb {
            self.chart.with_element(ui, res, UIElement::Bar, Some((-1., top + height / 2.)), Some((-1., top + height / 2.)), |ui, color| {
                ui.fill_rect(
                    Rect::new(-1., bar_y, dest, height),
                    Color::new(0.565, 0.565, 0.565, color.a * c.a * bar_alpha),
                );
                ui.fill_rect(Rect::new(-1. + dest - hw, bar_y, hw * 2., height), Color::new(1., 1., 1., color.a * c.a * bar_alpha));
            });
            self.chart.with_element(ui, res, UIElement::Bar, Some((-1., top + height / 2.)), Some((-1., top + height / 2.)), |ui, color| {
                let ct = Vector::new(0., top + height / 2.);
                ui.fill_rect(
                    Rect::new(-1., bar_y, dest, height),
                    Color::new(0.45, 0.45, 0.45, bar_alpha),
                );
                ui.fill_rect(Rect::new(-1. + dest - hw, bar_y, hw * 2., height), Color { a: color.a * c.a * bar_alpha, ..color });
            });
        }
        self.chart.with_element(ui, res, UIElement::Bar, Some((-1., top + height / 2.)), Some((-1., top + height / 2.)), |ui, color| {
            let progress = res.time / res.track_length;
            let corrected_progress = if progress >= 0.9999 { 1.0 } else { progress };
            let progress_percentage = (corrected_progress * 100.).min(100.);
            let truncated_percentage = ((progress_percentage * 10000.0).floor() / 10000.0).min(100.0);
            let progress_text = format!("{:.4}%", truncated_percentage);
            let parts: Vec<&str> = progress_text.split('.').collect();
            let current_time_text = fmt_time(res.time);
            let total_time_text = fmt_time(res.track_length);
            let time_text = format!("{}", current_time_text);

            if res.config.show_progress_text {
                ui.text(progress_text)
                    .pos(1. - margin, top + eps * 2.2 - (1. - p) * 0.4 + 0.07 + 0.01)
                    .anchor(1., 0.)
                    .size(0.4)
                    .color(semi_white(0.7))
                    .draw();
            }
            if res.config.show_time_text {
                ui.text(time_text)
                    .pos(-1. + dest - 0.01, top + height / 2. - 0.0008)
                    .anchor(1., 0.5)
                    .size(0.17867)
                    .color(Color::new(1.0, 1.0, 1.0, color.a * c.a))
                    .draw();
            }
        });
        self.draw_judgement_counters(ui, tm, score_top, 2.3);
        Ok(())
    }

    fn overlay_ui(&mut self, ui: &mut Ui, tm: &mut TimeManager) -> Result<()> {
        let c = semi_white(self.res.alpha);
        let res = &mut self.res;
        if tm.paused() {
            let h = 1. / res.aspect_ratio;
            draw_rectangle(-1., -h, 2., h * 2., Color::new(0., 0., 0., 0.6));
            let o = if self.mode == GameMode::Exercise { -0.3 } else { 0. };
            let s = 0.06;
            let w = 0.05;
            let no_retry = self.mode == GameMode::NoRetry;
            draw_texture_ex(
                *res.icon_back,
                -s * 3. - w,
                -s + o,
                c,
                DrawTextureParams {
                    dest_size: Some(vec2(s * 2., s * 2.)),
                    ..Default::default()
                },
            );
            draw_texture_ex(
                *res.icon_retry,
                -s,
                -s + o,
                if no_retry { semi_white(res.alpha * 0.6) } else { c },
                DrawTextureParams {
                    dest_size: Some(vec2(s * 2., s * 2.)),
                    ..Default::default()
                },
            );
            draw_texture_ex(
                *res.icon_resume,
                s + w,
                -s + o,
                c,
                DrawTextureParams {
                    dest_size: Some(vec2(s * 2., s * 2.)),
                    ..Default::default()
                },
            );
            if res.config.ui_pause {
                if res.config.interactive {
                    let mut clicked = None;
                    for touch in Judge::get_touches() {
                        if touch.phase != TouchPhase::Started {
                            continue;
                        }
                        let p = touch.position;
                        let p = Point::new(p.x, p.y);
                        for i in -1..=1 {
                            let ct = Point::new((s * 2. + w) * i as f32, o);
                            let d = p - ct;
                            if d.x.abs() <= s && d.y.abs() <= s {
                                clicked = Some(i);
                                break;
                            }
                        }
                    }
                    if no_retry && clicked == Some(0) {
                        clicked = None;
                    }
                    let mut pos = self.music.position();
                    if clicked.map_or(false, |it| it != -1) && (tm.speed - res.config.speed as f64).abs() > 0.01 {
                        debug!("recreating music");
                        self.music = res.audio.create_music(
                            res.music.clone(),
                            MusicParams {
                                amplifier: res.config.volume_music as _,
                                playback_rate: res.config.speed as _,
                                ..Default::default()
                            },
                        )?;
                    }
                    match clicked {
                        Some(-1) => {
                            self.should_exit = true;
                        }
                        Some(0) => {
                            reset!(self, res, tm);
                        }
                        Some(1) => {
                            if self.mode == GameMode::Exercise && tm.now() > self.exercise_range.end as f64 {
                                tm.seek_to(self.exercise_range.start as f64);
                                self.music.seek_to(self.exercise_range.start)?;
                                pos = self.exercise_range.start;
                            }
                            self.music.play()?;
                            res.time -= 3.;
                            let dst = pos - 3.;
                            if dst < 0. {
                                self.music.pause()?;
                                self.state = State::BeforeMusic;
                            } else {
                                self.music.seek_to(dst)?;
                            }
                            let now = tm.now();
                            tm.speed = res.config.speed as _;
                            tm.resume();
                            tm.seek_to(now - 3.);
                            self.pause_rewind = Some(tm.now() - 0.2);
                        }
                        _ => {}
                    }
                }
            }
            if self.mode == GameMode::Exercise {
                let asp = self.touch_scale();
                for touch in ui.ensure_touches() {
                    touch.position *= asp;
                }
                ui.scope(|ui| {
                    ui.dx(0.3);
                    ui.dy(-0.3);
                    ui.slider(tl!("speed"), 0.5..2.0, 0.05, &mut self.res.config.speed, Some(0.5));
                });
                ui.dy(0.06);
                let hw = 0.7;
                let h = 0.06;
                let eh = 0.12;
                let rad = 0.03;
                let sp = self.offset().min(0.);
                ui.fill_rect(Rect::new(-hw, -h, hw * 2., h * 2.), GRAY);
                let st = -hw + (self.exercise_range.start - sp) / (self.res.track_length - sp) * hw * 2.;
                let en = -hw + (self.exercise_range.end - sp) / (self.res.track_length - sp) * hw * 2.;
                let t = tm.now() as f32;
                let cur = -hw + (t - sp) / (self.res.track_length - sp) * hw * 2.;
                ui.fill_rect(Rect::new(st, -h, en - st, h * 2.), WHITE);
                ui.fill_rect(Rect::new(st, -eh, 0., eh + h).feather(0.005), BLUE);
                ui.fill_circle(st, -eh, rad, BLUE);
                if self.exercise_press.is_none() {
                    let r = ui.rect_to_global(Rect::new(st, -eh, 0., 0.).feather(rad));
                    self.exercise_press = Judge::get_touches()
                        .iter()
                        .find(|it| it.phase == TouchPhase::Started && r.contains(it.position))
                        .map(|it| (-1, it.id));
                }
                ui.fill_rect(Rect::new(en, -h, 0., eh + h).feather(0.005), RED);
                ui.fill_circle(en, eh, rad, RED);
                if self.exercise_press.is_none() {
                    let r = ui.rect_to_global(Rect::new(en, eh, 0., 0.).feather(rad));
                    self.exercise_press = Judge::get_touches()
                        .iter()
                        .find(|it| it.phase == TouchPhase::Started && r.contains(it.position))
                        .map(|it| (1, it.id));
                }
                ui.fill_rect(Rect::new(cur, -h, 0., h * 2.).feather(0.005), GREEN);
                ui.fill_circle(cur, 0., rad, GREEN);
                if self.exercise_press.is_none() {
                    let r = ui.rect_to_global(Rect::new(cur, 0., 0., 0.).feather(rad));
                    self.exercise_press = Judge::get_touches()
                        .iter()
                        .find(|it| it.phase == TouchPhase::Started && r.contains(it.position))
                        .map(|it| (0, it.id));
                }
                ui.text(fmt_time(t)).pos(0., -0.23).anchor(0.5, 0.).size(0.8).draw();
                if let Some((ctrl, id)) = &self.exercise_press {
                    if let Some(touch) = Judge::get_touches().iter().rfind(|it| it.id == *id) {
                        let x = touch.position.x;
                        let p = (x + hw) / (hw * 2.) * (self.res.track_length - sp) + sp;
                        let p = if self.res.track_length - sp <= 3. || *ctrl == 0 {
                            p.clamp(sp, self.res.track_length)
                        } else {
                            p.clamp(
                                if *ctrl == -1 { sp } else { self.exercise_range.start + 3. },
                                if *ctrl == -1 {
                                    self.exercise_range.end - 3.
                                } else {
                                    self.res.track_length
                                },
                            )
                        };
                        if *ctrl == 0 {
                            tm.seek_to(p as f64);
                            self.music.seek_to(p)?;
                        } else {
                            *(if *ctrl == -1 {
                                &mut self.exercise_range.start
                            } else {
                                &mut self.exercise_range.end
                            }) = p;
                        }
                        if matches!(touch.phase, TouchPhase::Cancelled | TouchPhase::Ended) {
                            self.exercise_press = None;
                        }
                    }
                }
                ui.dy(0.2);
                let r = ui.text(tl!("to")).size(0.8).anchor(0.5, 0.).draw();
                let mut tx = ui
                    .text(fmt_time(self.exercise_range.start))
                    .pos(r.x - 0.02, 0.)
                    .anchor(1., 0.)
                    .size(0.8)
                    .color(BLACK);
                let re = tx.measure();
                self.exercise_btns.0.set(tx.ui, re);
                tx.ui
                    .fill_rect(re.feather(0.01), Color::new(1., 1., 1., if self.exercise_btns.0.touching() { 0.5 } else { 1. }));
                tx.draw();

                let mut tx = ui
                    .text(fmt_time(self.exercise_range.end))
                    .pos(r.right() + 0.02, 0.)
                    .size(0.8)
                    .color(BLACK);
                let re = tx.measure();
                self.exercise_btns.1.set(tx.ui, re);
                tx.ui
                    .fill_rect(re.feather(0.01), Color::new(1., 1., 1., if self.exercise_btns.1.touching() { 0.5 } else { 1. }));
                tx.draw();
                for touch in ui.ensure_touches() {
                    touch.position /= asp;
                }
            }
        }
        if let Some(time) = self.pause_rewind {
            let dt = tm.now() - time;
            let t = 3 - dt.floor() as i32;
            if t <= 0 {
                self.pause_rewind = None;
            } else {
                let a = (1. - dt as f32 / 3.) * 1.;
                let h = 1. / self.res.aspect_ratio;
                draw_rectangle(-1., -h, 2., h * 2., Color::new(0., 0., 0., a));
                ui.text(t.to_string()).anchor(0.5, 0.5).size(1.).color(c).draw();
            }
        }
        if self.res.config.touch_debug {
            for touch in Judge::get_touches() {
                ui.fill_circle(touch.position.x, touch.position.y, 0.04, Color { a: 0.4, ..RED });
            }
        }
        for pos in &self.touch_points {
            ui.fill_circle(pos.0, pos.1, 0.04, Color { a: 0.4, ..BLUE });
        }
        Ok(())
    }
    
    fn interactive(res: &Resource, state: &State) -> bool {
        res.config.interactive && matches!(state, State::Playing)
    }

    fn offset(&self) -> f32 {
        self.chart.offset + self.res.config.offset + self.info_offset
    }

    fn process_judgements(&mut self, tm: &TimeManager) {
        if !self.res.config.chart_debug {
            return;
        }
        let combo_threshold = 0.02;
        let mut judgements = self.judge.judgements.borrow_mut();
        judgements.sort_by(|(t1, _, _, _), (t2, _, _, _)| t1.partial_cmp(t2).unwrap());

        let note_types = [
            NoteType::Click,
            NoteType::Drag,
            NoteType::Flick,
            NoteType::Hold,
        ];
        for note_type in note_types {
            if !self.judgement_counters.iter().any(|c| c.note_type == note_type) {
                self.judgement_counters.push(JudgementCounter::new(
                    note_type,
                    self.res.config.chart_ratio,
                    0.0,
                ));
            }
        }

        for &(t, line_id, note_id, _) in judgements.iter() {
            if let Some(line) = self.chart.lines.get(line_id as usize) {
                if let Some(note) = line.notes.get(note_id as usize) {
                    let note_type = match note.kind {
                        NoteKind::Click => NoteType::Click,
                        NoteKind::Drag => NoteType::Drag,
                        NoteKind::Flick => NoteType::Flick,
                        NoteKind::Hold { .. } => {
                            // 仅当是Hold尾部时处理
                            if let JudgeStatus::Hold(_, _, _, _, up_time) = note.judge {
                                if (t as f32) < up_time {
                                    continue; // 跳过头部事件
                                }
                            }
                            NoteType::Hold
                        }
                        _ => continue,
                    };

                    if let Some(counter) = self.judgement_counters
                        .iter_mut()
                        .find(|c| c.note_type == note_type)
                    {
                        let current_time = t as f64;
                        let is_same_time = (current_time - counter.last_update).abs() <= f64::EPSILON * 2.0;

                        //if note_type == NoteType::Hold && is_same_time {
                        //    continue; // 跳过重复时间点 [ 爱修不修 ]
                        //}

                        let effective_interval = if is_same_time {
                            0.0
                        } else {
                            current_time - counter.last_update
                        };

                        if effective_interval <= combo_threshold {
                            counter.multiplier += 1;
                        } else {
                            counter.multiplier = 1;
                        }

                        counter.last_update = current_time;
                        counter.count += 1;
                        counter.interval = effective_interval as f32;
                    }
                }
            }
        }
        judgements.clear();
    }

    fn tweak_offset(&mut self, ui: &mut Ui, ita: bool, tm: &mut TimeManager) {
        let width = 0.55;
        let height = 0.3;
        ui.scope(|ui| {
            let width = 0.55;
            let height = 0.3;
            ui.dx(1. - width - 0.02);
            ui.dy(ui.top - height - 0.02);
            ui.fill_rect(Rect::new(0., 0., width, height), GRAY);
            ui.dy(0.02);
            ui.text(tl!("adjust-offset")).pos(width / 2., 0.).anchor(0.5, 0.).size(0.7).draw();
            ui.dx(width / 1.22);
            if ui.button("cancel", Rect::new(0.02, 0., 0.06, 0.06), "×") {
                self.next_scene = Some(NextScene::PopWithResult(Box::new(None::<f32>)));
            }
            ui.dx(-width / 1.22);
            ui.dy(0.16);
            let r = ui
                .text(format!("{}ms", (self.info_offset * 1000.).round() as i32))
                .pos(width / 2., 0.)
                .anchor(0.5, 0.)
                .size(0.6)
                .no_baseline()
                .draw();
            let d = 0.14;
            let mut bpm_list = self.chart.bpm_list.borrow_mut();
            let beat = 15. / bpm_list.now_bpm(tm.now() as f32);
            if ui.button("lg_sub", Rect::new(d, r.center().y, 0., 0.).feather(0.026), "-") && ita {
                self.info_offset -= beat;
            }
            if ui.button("lg_add", Rect::new(width - d, r.center().y, 0., 0.).feather(0.026), "+") && ita {
                self.info_offset -= beat;
            }
            let d = 0.08;
            if ui.button("sm_sub", Rect::new(d, r.center().y, 0., 0.).feather(0.022), "-") && ita {
                self.info_offset -= 0.01;
            }
            if ui.button("sm_add", Rect::new(width - d, r.center().y, 0., 0.).feather(0.022), "+") && ita {
                self.info_offset -= 0.01;
            }
            let d = 0.03;
            if ui.button("ti_sub", Rect::new(d, r.center().y, 0., 0.).feather(0.017), "-") && ita {
                self.info_offset -= 0.001;
            }
            if ui.button("ti_add", Rect::new(width - d, r.center().y, 0., 0.).feather(0.017), "+") && ita {
                self.info_offset += 0.001;
            }
        });
        ui.scope(|ui| {
            ui.dx(1. - width * 0.97);
            ui.dy(ui.top - height * 0.75);
            ui.slider(tl!("speed"), 0.1..2.0, 0.05, &mut self.res.config.speed, Some(0.3));
            if ui.button("save-speed", Rect::new(0.44, 0.033, 0.05, 0.05), "=") && (tm.speed - self.res.config.speed as f64).abs() > 0.01 {
                debug!("recreating music");
                self.music = self.res.audio.create_music(
                    self.res.music.clone(),
                    MusicParams {
                        amplifier: self.res.config.volume_music as _,
                        playback_rate: self.res.config.speed as _,
                        ..Default::default()
                    },
                ).expect("failed to create music");
                reset_speed!(self, self.res, tm);
            }
        });
    }
}

impl Scene for GameScene {
    fn enter(&mut self, tm: &mut TimeManager, target: Option<RenderTarget>) -> Result<()> {
        #[cfg(target_arch = "wasm32")]
        on_game_start();
        self.music = Self::new_music(&mut self.res)?;
        self.res.camera.render_target = target;
        tm.speed = self.res.config.speed as _;
        tm.adjust_time = self.res.config.adjust_time;
        reset!(self, self.res, tm);
        set_camera(&self.res.camera);
        self.first_in = true;
        Ok(())
    }

    fn pause(&mut self, tm: &mut TimeManager) -> Result<()> {
        if !tm.paused() {
            self.pause_rewind = None;
            self.music.pause()?;
            tm.pause();
        }
        Ok(())
    }

    fn resume(&mut self, tm: &mut TimeManager) -> Result<()> {
        if !matches!(self.state, State::Playing) {
            tm.resume();
        }
        Ok(())
    }

    fn update(&mut self, tm: &mut TimeManager) -> Result<()> {
        self.res.audio.recover_if_needed()?;
        let time = tm.now() as f32;
        let p = match self.state {
            State::Starting => {
                if time <= Self::BEFORE_TIME {
                    1. - (1. - time / Self::BEFORE_TIME).powi(3)
                } else {
                    1.
                }
            }
            State::BeforeMusic => 1.,
            State::Playing => 1.,
            State::Ending => {
                let t = time - self.res.track_length - WAIT_TIME;
                1. - (t / (AFTER_TIME + 0.3)).min(1.).powi(2)
            }
        };

        // 更新当前缩放比例（如果启用了加载动画）
        if !self.res.config.disable_loading {
            match self.state {
                State::Starting => {
                    // 从 1.0 动画到目标值
                    self.current_chart_ratio = 1.0 + (self.target_chart_ratio - 1.0) * p;
                }
                State::Ending => {
                    // 从当前值动画回 1.0
                    self.current_chart_ratio = self.target_chart_ratio + (1.0 - self.target_chart_ratio) * (1.0 - p);
                }
                _ => {
                    // 其他状态使用目标值
                    self.current_chart_ratio = self.target_chart_ratio;
                }
            }
        } else {
            // 禁用加载动画时直接使用目标值
            self.current_chart_ratio = self.target_chart_ratio;
        }
        if matches!(self.state, State::Playing) {
            tm.update(self.music.position() as f64);
        }
        if self.mode == GameMode::Exercise && tm.now() > self.exercise_range.end as f64 && !tm.paused() {
            let state = self.state.clone();
            reset!(self, self.res, tm);
            self.state = state;
            tm.seek_to(self.exercise_range.start as f64);
            tm.pause();
            self.music.pause()?;
        }
        let offset = self.offset();
        let time = tm.now() as f32;
        let time = match self.state {
            State::Starting => {
                if time >= Self::BEFORE_TIME {
                    self.res.alpha = 1.;
                    self.state = State::BeforeMusic;
                    tm.reset();
                    tm.seek_to(if self.mode == GameMode::Exercise {
                        self.exercise_range.start as f64
                    } else {
                        offset.min(0.) as f64
                    });
                    self.last_update_time = tm.real_time();
                    if self.first_in && self.mode == GameMode::Exercise {
                        tm.pause();
                        self.first_in = false;
                    }
                    tm.now() as f32
                } else {
                    self.res.alpha = 1. - (1. - time / Self::BEFORE_TIME).powi(3);
                    if self.mode == GameMode::Exercise {
                        self.exercise_range.start
                    } else {
                        offset
                    }
                }
            }
            State::BeforeMusic => {
                if time >= 0.0 {
                    self.music.seek_to(time)?;
                    if !tm.paused() {
                        self.music.play()?;
                    }
                    self.state = State::Playing;
                }
                time
            }
            State::Playing => {
                if time > self.res.track_length + WAIT_TIME {
                    self.state = State::Ending;
                }
                time
            }
            State::Ending => {
                // 确保只在第一次进入 Ending 状态时重置
                if !self.judgement_reset_done {
                    // 遍历所有 JudgementCounter 并重置
                    for counter in self.judgement_counters.iter_mut() {
                        counter.reset(self.res.config.chart_ratio, 0.0);
                    }
                    self.judgement_reset_done = true; // 标记为已重置
                }

                // 计算 Ending 状态的时间
                let t = time - self.res.track_length - WAIT_TIME;

                // 如果 Ending 状态持续时间超过 AFTER_TIME + 0.3，则进入下一步逻辑
                if t >= AFTER_TIME + 0.3 {
                    let mut record_data = None;
                    #[cfg(feature = "closed")]
                    if let Some(upload_fn) = &self.upload_fn {
                        if !self.res.config.offline_mode
                            && !self.res.config.autoplay()
                            && self.res.config.speed >= 1.0 - 1e-3
                        {
                            if let Some(player) = &self.player {
                                if let Some(chart) = &self.res.info.id {
                                    record_data = Some(encode_record(self, player.id, *chart));
                                }
                            }
                        }
                    }
                    let result = self.judge.result();
                    let record = if self.res.config.autoplay() || self.res.config.speed < 1.0 - 1e-3 {
                        None
                    } else {
                        Some(SimpleRecord {
                            score: result.score as _,
                            accuracy: result.accuracy as _,
                            full_combo: result.max_combo == result.num_of_notes,
                        })
                    };
                    self.next_scene = match self.mode {
                        GameMode::Normal | GameMode::NoRetry | GameMode::View => Some(NextScene::Overlay(Box::new(EndingScene::new(
                            self.res.background.clone(),
                            self.res.illustration.clone(),
                            self.res.player.clone(),
                            self.res.icons.clone(),
                            self.res.icon_retry.clone(),
                            self.res.icon_proceed.clone(),
                            self.res.info.clone(),
                            self.judge.result(),
                            self.res.challenge_icons[self.res.config.challenge_color.clone() as usize].clone(),
                            &self.res.config,
                            self.res.res_pack.ending.clone(),
                            self.upload_fn.as_ref().map(Arc::clone),
                            self.player.as_ref().map(|it| it.rks),
                            record_data,
                            record,
                        )?))),
                        GameMode::TweakOffset => Some(NextScene::PopWithResult(Box::new(None::<f32>))),
                        GameMode::Exercise => None,
                    };
                }
                self.res.alpha = 1. - (t / AFTER_TIME).min(1.).powi(2);
                self.res.track_length
            }
        };
        let time = (time - offset).max(0.);
        self.res.time = time;
        if !tm.paused() && self.pause_rewind.is_none() && self.mode != GameMode::View {
            self.gl.quad_gl.viewport(self.res.camera.viewport);
            self.judge.update(&mut self.res, &mut self.chart, &mut self.bad_notes, self.is_fast_forwarding);
            self.gl.quad_gl.viewport(None);
        }

        self.process_judgements(tm);

        let dt = 0.016_f32;
        {
            // 先排序，获得目标位置
            let chart_ratio = self.current_chart_ratio;
            let base_spacing = 0.1;
            let spacing = base_spacing / chart_ratio;
            let gap = 0.05 * chart_ratio;
            let target_base_y = gap;
            
            let mut counters = self.judgement_counters.clone();
            counters.sort_by(|a, b| a.last_update.partial_cmp(&b.last_update).unwrap());
            // 遍历排序后的索引，为每个计数器计算目标垂直位置
            for (i, target_counter) in counters.iter().enumerate() {
                let target_y = target_base_y - (i as f32 * spacing);
                // 找到原始集合中对应的计数器并更新：
                if let Some(counter) = self.judgement_counters.iter_mut().find(|c| c.note_type == target_counter.note_type) {
                    counter.update(tm.now(), dt, target_y);
                }
            }
        }

        if let Some(update) = &mut self.update_fn {
            update(self.res.time, &mut self.res, &mut self.judge);
        }
        let counts = self.judge.counts();
        self.res.judge_line_color = if counts[2] + counts[3] == 0 {
            Color::from_hex(if counts[1] == 0 {
                self.res.res_pack.info.color_perfect_line
            } else {
                self.res.res_pack.info.color_good_line
            })
        } else {
            WHITE
        };
        self.res.judge_line_color.a *= self.res.alpha;
        self.chart.update(&mut self.res);
        let res = &mut self.res;
        if res.config.interactive && is_key_pressed(KeyCode::Space) {
            if tm.paused() {
                if matches!(self.state, State::Playing) {
                    self.music.play()?;
                    tm.resume();
                }
            } else if matches!(self.state, State::Playing | State::BeforeMusic) {
                if !self.music.paused() {
                    self.music.pause()?;
                }
                tm.pause();
            }
        }
        self.is_fast_forwarding = false;
        if Self::interactive(res, &self.state) {
            if is_key_pressed(KeyCode::Left) {
                res.time -= 1.;
                let dst = (self.music.position() - 1.).max(0.);
                self.music.seek_to(dst)?;
                tm.seek_to(dst as f64);
            }
            if is_key_pressed(KeyCode::Right) {
                self.is_fast_forwarding = true;
                res.time += 5.;
                let dst = (self.music.position() + 5.).min(res.track_length);
                self.music.seek_to(dst)?;
                tm.seek_to(dst as f64);
            }
            if is_key_pressed(KeyCode::Q) {
                self.should_exit = true;
            }
        }
        for e in &mut self.effects {
            e.update(&self.res);
        }
        if let Some((id, text)) = take_input() {
            let offset = self.offset().min(0.);
            match id.as_str() {
                "exercise_start" => {
                    if let Some(t) = parse_time(&text) {
                        if !(offset..self.res.track_length.min(self.exercise_range.end - 3.).max(offset)).contains(&t) {
                            show_message(tl!("ex-time-out-of-range")).error();
                        } else {
                            self.exercise_range.start = t;
                            show_message(tl!("ex-time-set")).ok();
                        }
                    } else {
                        show_message(tl!("ex-invalid-format")).error();
                    }
                }
                "exercise_end" => {
                    if let Some(t) = parse_time(&text) {
                        if !((self.exercise_range.start + 3.).max(offset).min(self.res.track_length)..self.res.track_length).contains(&t) {
                            show_message(tl!("ex-time-out-of-range")).error();
                        } else {
                            self.exercise_range.end = t;
                            show_message(tl!("ex-time-set")).ok();
                        }
                    } else {
                        show_message(tl!("ex-invalid-format")).error();
                    }
                }
                _ => return_input(id, text),
            }
        }
        Ok(())
    }

    fn touch(&mut self, tm: &mut TimeManager, touch: &Touch) -> Result<bool> {
        if self.mode == GameMode::Exercise && tm.paused() {
            let touch = Touch {
                position: touch.position * self.touch_scale(),
                ..touch.clone()
            };
            if self.exercise_btns.0.touch(&touch) {
                request_input("exercise_start", &fmt_time(self.exercise_range.start));
                return Ok(true);
            }
            if self.exercise_btns.1.touch(&touch) {
                request_input("exercise_end", &fmt_time(self.exercise_range.end));
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn render(&mut self, tm: &mut TimeManager, ui: &mut Ui) -> Result<()> {
        let res = &mut self.res;
        let asp = ui.viewport.2 as f32 / ui.viewport.3 as f32;
        
        let vp = res.camera.viewport.unwrap_or(ui.viewport);
        let asp2 = vp.2 as f32 / vp.3 as f32;
        let vec2_asp = vec2(1. * self.current_chart_ratio, -asp2 * self.current_chart_ratio);
        if res.update_size(ui.viewport) || self.mode == GameMode::View {
            set_camera(&res.camera);
        }

        let msaa = res.config.sample_count > 1;

        let chart_onto = res
            .chart_target
            .as_ref()
            .map(|it| if msaa { it.input() } else { it.output() })
            .or(res.camera.render_target);
        set_camera(&Camera2D {
            zoom: vec2(1., -asp),
            viewport: if res.chart_target.is_some() { None } else { Some(ui.viewport) },
            render_target: chart_onto,
            ..Default::default()
        });
        clear_background(BLACK);
        draw_background(*res.background);

        let chart_target_vp = if res.chart_target.is_some() {
            let vp = res.camera.viewport.unwrap();
            Some((vp.0 - ui.viewport.0, vp.1 - ui.viewport.1, vp.2, vp.3))
        } else {
            res.camera.viewport
        };
        let chart_ratio = self.current_chart_ratio;
        let h = 1. / res.aspect_ratio;
        if chart_ratio >= 1.0 {
            let dim_alpha = 0.7;
            let dim = Color::new(0.1, 0.1, 0.1, dim_alpha * res.alpha);
            let x_range = vp.0 as f32 / ui.viewport.2 as f32;

            draw_rectangle(-1., -h, x_range * 2., h * 2., dim);
            draw_rectangle(1., -h, -x_range * 2., h * 2., dim);
            draw_rectangle(
                x_range * 2. - 1.,
                -h,
                (1. - x_range * 2.) * 2.,
                h * 2.,
                Color::new(0., 0., 0., res.alpha * res.info.background_dim)
            );
        }
        
        set_camera( &Camera2D {
            zoom: vec2_asp,
            viewport: chart_target_vp,
            ..Default::default()
        });
        
        self.gl.quad_gl.render_pass(chart_onto.map(|it| it.render_pass));
        draw_rectangle(-1., -h, 2., h * 2., Color::new(0., 0., 0., res.alpha * res.info.background_dim));
        self.chart.render(ui, res);

        set_camera( &Camera2D {
            zoom: vec2_asp,
            viewport: chart_target_vp,
            ..Default::default()
        });

        self.gl.quad_gl.render_pass(
            res.chart_target
                .as_ref()
                .map(|it| it.output().render_pass)
                .or_else(|| res.camera.render_pass()),
        );

        self.bad_notes.retain(|dummy| dummy.render(res));
        let t = tm.real_time();
        let dt = (t - std::mem::replace(&mut self.last_update_time, t)) as f32;
        if res.config.particle {
            res.emitter.draw(dt);
        }
        
        self.ui(ui, tm)?;

        if !self.res.no_effect && !self.effects.is_empty() {
            set_camera(&Camera2D {
                zoom: vec2(1., asp),
                ..Default::default()
            });
            for e in &self.effects {
                e.render(&mut self.res);
            }
        }
        
        {
            set_camera(&Camera2D {
                zoom: vec2(1., -asp2),
                viewport: chart_target_vp,
                render_target: self.res.chart_target.as_ref().map(|it| it.output()).or(self.res.camera.render_target),
                ..Default::default()
            });
            self.overlay_ui(ui, tm)?;
        }

        if self.mode == GameMode::TweakOffset {
            //push_camera_state();
            self.gl.quad_gl.viewport(None);
            set_camera(&Camera2D {
                zoom: vec2(1., asp),
                render_target: self.res.chart_target.as_ref().map(|it| it.output()).or(self.res.camera.render_target),
                ..Default::default()
            });
            self.tweak_offset(ui, Self::interactive(&self.res, &self.state), tm);
            //pop_camera_state();
        }

        if msaa || !self.res.no_effect {
            if let Some(target) = &self.res.chart_target {
                self.gl.flush();
                self.gl.quad_gl.viewport(None);
                set_camera(&Camera2D {
                    zoom: vec2(1., asp),
                    render_target: self.res.camera.render_target,
                    viewport: Some(ui.viewport),
                    ..Default::default()
                });
                draw_texture_ex(
                    target.output().texture,
                    -1.,
                    -ui.top,
                    WHITE,
                    DrawTextureParams {
                        dest_size: Some(vec2(2., ui.top * 2.)),
                        ..Default::default()
                    },
                );
            }
        } else {
            self.gl.flush();
        }
        Ok(())
    }

    fn next_scene(&mut self, tm: &mut TimeManager) -> NextScene {
        if self.should_exit {
            if tm.paused() {
                tm.resume();
            }
            tm.speed = 1.0;
            tm.adjust_time = false;
            match self.mode {
                GameMode::Normal | GameMode::Exercise | GameMode::NoRetry | GameMode::View => NextScene::Pop,
                GameMode::TweakOffset => NextScene::PopWithResult(Box::new(None::<f32>)),
            }
        } else if let Some(next_scene) = self.next_scene.take() {
            tm.speed = 1.0;
            tm.adjust_time = false;
            next_scene
        } else {
            NextScene::None
        }
    }
}