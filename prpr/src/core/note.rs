use super::{chart::ChartSettings, BpmList, CtrlObject, JudgeLine, Matrix, Object, Point, Resource};
use crate::{
    judge::JudgeStatus, 
    parse::RPE_HEIGHT,
    core::HEIGHT_RATIO,
};
use serde::Serialize;
use serde::Deserialize;
use macroquad::prelude::*;
use macroquad::miniquad::gl::GLuint;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use once_cell::sync::Lazy;

//const HOLD_PARTICLE_INTERVAL: f32 = 0.15;
const FADEOUT_TIME: f32 = 0.16;
const BAD_TIME: f32 = 0.5;
const RPE_HEIGHT_SCALE: f32 = RPE_HEIGHT * (1.0 / 720.0);

// 懒加载静态资源
static INIT_POINTS: Lazy<[Point; 4]> = Lazy::new(|| [
    Point::new(0., 0.),
    Point::new(1., 0.),
    Point::new(1., 1.),
    Point::new(0., 1.),
]);
static RANDOM_SEED: AtomicU32 = AtomicU32::new(0x12345678);
static HAND_COLORS: Lazy<[Color; 2]> = Lazy::new(|| [
    Color::new(0.2, 0.5, 1.0, 1.0),  // Left
    Color::new(1.0, 0.6, 0.7, 1.0),  // Right
]);

// Cache texture GL internal IDs by raw pointer to avoid repeated FFI calls
thread_local! {
    static TEXTURE_GL_CACHE: RefCell<HashMap<u32, GLuint>> = RefCell::new(HashMap::new());
}

#[inline(always)]
fn get_texture_gl_id(texture: &Texture2D) -> GLuint {
    let tex = texture.raw_miniquad_texture_handle();
    let key = tex.gl_internal_id();
    TEXTURE_GL_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let id = *cache.entry(key).or_insert(key);
        id
    })
}

// 基础亮度常量
const BASE_LUMINANCE: f32 = 0.9;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum NoteKind {
    Click,
    Hold { end_time: f32, end_height: f32 },
    Flick,
    Drag,
}

impl From<NoteKind> for u32 {
    fn from(kind: NoteKind) -> u32 {
        match kind {
            NoteKind::Click => 0,
            NoteKind::Hold { .. } => 1,
            NoteKind::Flick => 2,
            NoteKind::Drag => 3,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hand {
    Left,
    Right,
}

impl Hand {
    pub fn sign(&self) -> f32 {
        match self {
            Hand::Left => -1.0,
            Hand::Right => 1.0,
        }
    }
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

#[derive(Clone,Copy)]
pub struct NoteInstance {
    pub texture_id: u32,      // texture.gl_internal_id()
    pub order: i8,
    pub vertices: [Vertex; 4],
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
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

pub struct RenderConfig<'a>
{
    pub settings: &'a ChartSettings,
    pub ctrl_obj: &'a mut CtrlObject,
    pub line_height: f32,
    pub appear_before: f32,
    pub invisible_time: f32,
    pub draw_below: bool,
    pub incline_sin: f32,
    pub global_speed_factor: f32, //speed
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

    let (min_x, max_x, min_y, max_y) = p_screen.iter().fold(
        (f32::MAX, f32::MIN, f32::MAX, f32::MIN),
        |(min_x, max_x, min_y, max_y), pt| {
            (min_x.min(pt.x), max_x.max(pt.x), min_y.min(pt.y), max_y.max(pt.y))
        }
    );

    let chart_ratio_inv = res.chart_ratio_inv;
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
        (order, get_texture_gl_id(&texture)),
        vertices
    );
}

fn random_rotate() -> f32
{
    let seed = RANDOM_SEED.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |x| {
        Some(x.wrapping_mul(1664525).wrapping_add(1013904223))
    }).unwrap_or(0x12345678);
    (seed % 4) as f32 * 90.0
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

    pub fn update(&mut self, res: &mut Resource, parent_rot: f32, parent_tr: &Matrix, ctrl_obj: &mut CtrlObject, line_height: f32, bpm_list: &BpmList, index: usize) {
        self.object.set_time(res.time);
        //if matches!(self.judge, JudgeStatus::Hold(..)) {
        //    self.height = line_height;
        //}
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

    pub fn render(&self, res: &mut Resource, config: &mut RenderConfig, bpm_list: &BpmList) {
    if matches!(self.judge, JudgeStatus::Judged) && !matches!(self.kind, NoteKind::Hold { .. }) {
        return;
    }

    if config.appear_before.is_finite() {
        let beat = bpm_list.beat(self.time);
        let time = bpm_list.time_beats(beat - config.appear_before);
        if time > res.time {
            return;
        }
    }

    if config.invisible_time.is_finite() && self.time - config.invisible_time < res.time {
        return;
    }

    let y_factor = config.ctrl_obj.y.now_opt().unwrap_or(1.);
    let spd = self.speed * y_factor * config.global_speed_factor;
    let inv_aspect = res.inv_aspect_ratio; // 使用缓存的值
    let line_height = config.line_height * inv_aspect * spd;
    let height = self.height * inv_aspect * spd;
    let base = height - line_height;

    if res.config.aggressive && matches!(self.kind, NoteKind::Hold { .. }) {
        let h = if self.time <= res.time { line_height } else { height };
        let bottom = h + self.object.translation.1.now() - line_height;
        if bottom - line_height > res.chart_ratio_inv {
            return;
        }
    }

    let should_skip = !config.draw_below && (
        (res.time - FADEOUT_TIME >= self.time && !matches!(self.kind, NoteKind::Hold { .. })) ||
        (self.time > res.time && base <= -0.001)
    ) && self.speed != 0.;

    if should_skip && !res.config.chart_debug {
        return;
    }

    let scale = res.note_width * if self.multiple_hint {
        res.res_pack.note_style_mh.click.width() / res.res_pack.note_style.click.width()
    } else {
        1.0
    };

    self.init_ctrl_obj(&mut config.ctrl_obj, config.line_height);
    let mut color = self.object.now_color();

    if res.config.hand_split {
        let hand_color = match self.hand {
            Hand::Left => &HAND_COLORS[0],
            Hand::Right => &HAND_COLORS[1],
        };
        color.r = hand_color.r;
        color.g = hand_color.g;
        color.b = hand_color.b;
        let luminance = color.r * 0.299 + color.g * 0.587 + color.b * 0.114;
        let adjust = BASE_LUMINANCE / luminance.max(0.001);
        color.r = (color.r * adjust).min(1.0);
        color.g = (color.g * adjust).min(1.0);
        color.b = (color.b * adjust).min(1.0);
    }

    // Alpha
    let ctrl_alpha = config.ctrl_obj.alpha.now_opt().unwrap_or(1.0);
    color.a *= res.alpha * ctrl_alpha;

    if should_skip && res.config.chart_debug {
        color.a *= 0.2;
    }

    let order = self.kind.order();
    let style = if res.config.double_hint && self.multiple_hint {
        &res.res_pack.note_style_mh
    } else {
        &res.res_pack.note_style
    };

    // Fade out for non-hold
    if !config.draw_below && !matches!(self.kind, NoteKind::Hold { .. }) {
        let fade = (self.time - res.time).min(0.0) / FADEOUT_TIME + 1.0;
        color.a *= fade;
    }

    match &self.kind {
        NoteKind::Click => {
            if self.fake && res.time >= self.time { return; }
            self.render_quad(
                res, config, base, scale, color, order,
                *style.click, Rect::new(0., 0., 1., 1.)
            );
        }
        NoteKind::Flick => {
            if self.fake && res.time >= self.time { return; }
            self.render_quad(
                res, config, base, scale, color, order,
                *style.flick, Rect::new(0., 0., 1., 1.)
            );
        }
        NoteKind::Drag => {
            if self.fake && res.time >= self.time { return; }
            self.render_quad(
                res, config, base, scale, color, order,
                *style.drag, Rect::new(0., 0., 1., 1.)
            );
        }
        NoteKind::Hold { end_time, end_height } => {
            if self.fake && res.time >= *end_time { return; }
            if res.time >= *end_time { return; }

            let tex = &style.hold;
            let ratio = style.hold_ratio();
            let clip = !config.draw_below && config.settings.hold_partial_cover;
            let end_h = *end_height / res.aspect_ratio * spd;
            let start_h = self.start_height / res.aspect_ratio * spd;
            let hold_h = end_h - start_h;
            let current_time = self.time.max(res.time);
            let hold_line_h = (current_time - self.time) * (self.end_speed * y_factor * config.global_speed_factor)
                / res.aspect_ratio / HEIGHT_RATIO;

            let h = if self.time <= res.time { line_height } else { height };
            let bottom = h - line_height;
            let top = if self.format {
                bottom + hold_h - hold_line_h
            } else {
                end_h - line_height
            };

            if self.format && self.end_speed == 0. {
                if !res.config.chart_debug { return; }
                let mut debug_color = color;
                debug_color.a *= 0.2;
                color = debug_color;
            }

            if res.time < self.time && bottom < -1e-6 && (!config.settings.hold_partial_cover && !self.format) {
                return;
            }

            if matches!(self.judge, JudgeStatus::Judged) {
                color.a *= 0.5;
            }

            let body_tex = if res.res_pack.info.hold_repeat {
                style.hold_body.as_ref().unwrap()
            } else {
                tex
            };
            let body_source = if res.res_pack.info.hold_repeat {
                let w = body_tex.width();
                let h = body_tex.height();
                Rect::new(0., 0., 1., (top - bottom) / scale / 2. * w / h)
            } else {
                style.hold_body_rect()
            };

            // Compute transform once for all hold parts
            let transform = self.now_transform(res, &config.ctrl_obj, 0., config.incline_sin);

            self.render_quad_raw(
                res, config, &transform, scale, color, order,
                **body_tex, body_source,
                vec2(scale * 2., top - bottom), clip, bottom
            );

            if res.time < self.time || res.res_pack.info.hold_keep_head {
                let r = style.hold_head_rect();
                let hf = vec2(scale, r.h / r.w * scale * ratio);
                let head_y = bottom - if res.res_pack.info.hold_compact { hf.y } else { hf.y * 2. };
                self.render_quad_raw(
                    res, config, &transform, scale, color, order,
                    **tex, r, hf * 2., clip, head_y
                );
            }

            let r = style.hold_tail_rect();
            let hf = vec2(scale, r.h / r.w * scale * ratio);
            let tail_y = top - if res.res_pack.info.hold_compact { hf.y } else { 0. };
            self.render_quad_raw(
                res, config, &transform, scale, color, order,
                **tex, r, hf * 2., clip, tail_y
            );
        }
    }
}

    // 渲染居中 quad
    fn render_quad(&self, res: &Resource, config: &RenderConfig, base: f32, scale: f32, color: Color, order: i8, tex: Texture2D, source: Rect) {
        let hf = vec2(scale, tex.height() * scale / tex.width());
        let transform = self.now_transform(res, &config.ctrl_obj, base, config.incline_sin);
        self.render_quad_raw(res, config, &transform, scale, color, order, tex, source, hf * 2., false, -hf.y);
    }

    // 底层 quad 提交
    fn render_quad_raw(
        &self,
        res: &Resource,
        config: &RenderConfig,
        transform: &Matrix,
        scale: f32,
        color: Color,
        order: i8,
        tex: Texture2D,
        source: Rect,
        dest_size: Vec2,
        clip: bool,
        y_offset: f32,
    ) {
        let x = -scale;
        let y = y_offset;
        let w = dest_size.x;
        let h = dest_size.y;

        if h <= 0. || (clip && y + h <= 0.) {
            return;
        }

        let mut final_source = source;
        if clip && y < 0. {
            let visible = y + h;
            if visible <= 0. { return; }
            let ratio = (-y) / visible;
            final_source.y += final_source.h * ratio;
            final_source.h *= 1.0 - ratio;
        }

        // Compute world positions using pre-computed transform
        let p = [
            Point::new(x, y),
            Point::new(x + w, y),
            Point::new(x + w, y + h),
            Point::new(x, y + h),
        ];

        let p_world = p.map(|pt| transform.transform_point(&pt));
        let p_screen = p_world.map(|pt| res.world_to_screen(pt));
        let chart_ratio_inv = res.chart_ratio_inv; // 使用缓存的值
        let (min_x, max_x, min_y, max_y) = p_screen.iter().fold(
            (f32::MAX, f32::MIN, f32::MAX, f32::MIN),
            |(min_x, max_x, min_y, max_y), pt| {
                (min_x.min(pt.x), max_x.max(pt.x), min_y.min(pt.y), max_y.max(pt.y))
            }
        );
        if min_x > chart_ratio_inv || max_x < -chart_ratio_inv || min_y > chart_ratio_inv || max_y < -chart_ratio_inv {
            return;
        }

        // Flip if needed (your original logic)
        let mut p_final = p_screen;
        // flip_y is always true in your code
        p_final.swap(0, 3);
        p_final.swap(1, 2);

        // Build vertices
        let sx1 = final_source.x + final_source.w;
        let sy1 = final_source.y + final_source.h;
        let vertices = [
            Vertex::new(p_final[0].x, p_final[0].y, 0., final_source.x, final_source.y, color),
            Vertex::new(p_final[1].x, p_final[1].y, 0., sx1, final_source.y, color),
            Vertex::new(p_final[2].x, p_final[2].y, 0., sx1, sy1, color),
            Vertex::new(p_final[3].x, p_final[3].y, 0., final_source.x, sy1, color),
        ];

        // Submit to batch buffer
        res.note_buffer.borrow_mut().push(
            (order, get_texture_gl_id(&tex)),
            vertices
        );
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