use super::{chart::ChartSettings, BpmList, CtrlObject, JudgeLine, Matrix, Object, Point, Resource};
use crate::{
    judge::JudgeStatus, 
    parse::RPE_HEIGHT,
    core::HEIGHT_RATIO,
};

use macroquad::prelude::*;
//use ::rand::{thread_rng, Rng};

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
                    // 红色
                    color.r = 1.0;
                    color.g *= 0.5;
                    color.b *= 0.5;
                }
                Hand::Right => {
                    //蓝色
                    color.b = 1.0;
                    color.r *= 0.5;
                    color.g *= 0.5;
                }
            }
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