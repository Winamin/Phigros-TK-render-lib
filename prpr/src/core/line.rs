use super::{chart::ChartSettings, object::CtrlObject, Anim, AnimFloat, BpmList, Matrix, Note, Object, Point, RenderConfig, Resource, Vector};
use crate::{
    config::Mods,
    ext::{draw_text_aligned, get_viewport, NotNanExt, SafeTexture},
    judge::{JudgeStatus, LIMIT_BAD},
    ui::Ui,
};
use macroquad::prelude::*;
use miniquad::{RenderPass, Texture, TextureParams, TextureWrap, FilterMode};
use nalgebra::Rotation2;
use serde::Deserialize;
use std::{cell::RefCell, rc::Rc};

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
#[repr(u8)]
pub enum UIElement {
    Pause = 1,
    ComboNumber = 2,
    Combo = 3,
    Score = 4,
    Bar = 5,
    Name = 6,
    Level = 7,
}

impl UIElement {
    pub fn from_u8(val: u8) -> Option<Self> {
        Some(match val {
            1 => Self::Bar,
            2 => Self::Pause,
            3 => Self::ComboNumber,
            4 => Self::Combo,
            5 => Self::Score,
            6 => Self::Name,
            7 => Self::Level,
            _ => return None,
        })
    }
}

#[derive(Default)]
pub enum JudgeLineKind {
    #[default]
    Normal,
    Texture(SafeTexture, String),
    Text(Anim<String>),
    Paint(Anim<f32>, RefCell<(Option<RenderPass>, bool)>),
    TextureGif(Anim<f32>, GifFrames, String), //add gif frames
}

#[derive(Clone)]
pub struct JudgeLineCache {
    update_order: Vec<u32>,
    not_plain_count: usize,
    above_indices: Vec<usize>,
    below_indices: Vec<usize>,
}

impl JudgeLineCache {
    pub fn new(notes: &mut Vec<Note>) -> Self {
        notes.sort_by_key(|it| (it.plain(), !it.above, it.speed.not_nan(), ((it.height + it.object.translation.1.now()) * it.speed).not_nan()));
        let mut res = Self {
            update_order: Vec::new(),
            not_plain_count: 0,
            above_indices: Vec::new(),
            below_indices: Vec::new(),
        };
        res.reset(notes);
        res
    }

    pub(crate) fn reset(&mut self, notes: &mut Vec<Note>) {
        self.update_order = (0..notes.len() as u32).collect();
        self.above_indices.clear();
        self.below_indices.clear();
        let mut index = notes.iter().position(|it| it.plain()).unwrap_or(notes.len());
        self.not_plain_count = index;
        while notes.get(index).map_or(false, |it| it.above) {
            self.above_indices.push(index);
            let speed = notes[index].speed;
            loop {
                index += 1;
                if !notes.get(index).map_or(false, |it| it.above && it.speed == speed) {
                    break;
                }
            }
        }
        while index != notes.len() {
            self.below_indices.push(index);
            let speed = notes[index].speed;
            loop {
                index += 1;
                if !notes.get(index).map_or(false, |it| it.speed == speed) {
                    break;
                }
            }
        }
    }
}

pub struct GifFrames {
    /// time of each frame in milliseconds
    frames: Vec<(u128, SafeTexture)>,
    /// milliseconds
    total_time: u128,
}

impl GifFrames {
    pub fn new(frames: Vec<(u128, SafeTexture)>) -> Self {
        let total_time = frames.iter().map(|(time, _)| *time).sum();
        Self { frames, total_time }
    }

    pub fn get_time_frame(&self, time: u128) -> &SafeTexture {
        let mut time = time % self.total_time;
        for (t, frame) in &self.frames {
            if time < *t {
                return frame;
            }
            time -= t;
        }
        &self.frames.last().unwrap().1
    }

    pub fn get_prog_frame(&self, prog: f32) -> &SafeTexture {
        let time = (prog * self.total_time as f32) as u128;
        self.get_time_frame(time)
    }

    pub fn total_time(&self) -> u128 {
        self.total_time
    }
}

pub struct JudgeLine {
    pub object: Object,
    pub ctrl_obj: RefCell<CtrlObject>,
    pub kind: JudgeLineKind,
    pub height: AnimFloat,
    pub incline: AnimFloat,
    pub notes: Vec<Note>,
    pub color: Anim<Color>,
    pub parent: Option<usize>,
    pub z_index: i32,
    pub show_below: bool,
    pub attach_ui: Option<UIElement>,

    pub cache: JudgeLineCache,
}

struct Painter {
    pass: RenderPass,
    viewport: (i32, i32, i32, i32),
    cleared: bool,
}

impl Painter {
    fn new() -> Self {
        let mut gl = unsafe { get_internal_gl() };
        let vp = get_viewport();
        let tex = Texture::new_render_texture(
            &mut gl.quad_context,
            TextureParams {
                width: vp.2 as _,
                height: vp.3 as _,
                format: miniquad::TextureFormat::RGBA8,
                filter: FilterMode::Linear,
                wrap: TextureWrap::Clamp,
            },
        );
        let pass = RenderPass::new(&mut gl.quad_context, tex, None);
        Painter { pass, viewport: vp, cleared: false }
    }

    fn paint(&mut self, ui: &mut Ui, size: f32, alpha: f32, mut color: Color) {
        let gl = unsafe { get_internal_gl() };
        let old_pass = gl.quad_gl.get_active_render_pass();
        gl.quad_gl.render_pass(Some(self.pass));
        gl.quad_gl.viewport(Some(self.viewport));

        color.a = alpha.max(0.0) * 2.55;
        if size <= 0.0 {
            if self.cleared {
                clear_background(Color::default());
                self.cleared = false;
            }
        } else {
            let radius = size / self.viewport.2 as f32 * 2.0;
            ui.fill_circle(0., 0., radius, color);
            self.cleared = true;
        }

        gl.quad_gl.render_pass(old_pass);
        gl.quad_gl.viewport(Some(self.viewport));
    }
}



impl JudgeLine {
    pub fn update(&mut self, res: &mut Resource, tr: Matrix, bpm_list: &mut BpmList, index: usize) {
        let rot = self.object.rotation.now();
        self.height.set_time(res.time);
        let line_height = self.height.now();
        let mut ctrl_obj = self.ctrl_obj.borrow_mut();
        self.cache.update_order.retain(|id| {
            let note = &mut self.notes[*id as usize];
            note.update(res, rot, &tr, &mut ctrl_obj, line_height, bpm_list, index);
            !note.dead()
        });
        drop(ctrl_obj);
        match &mut self.kind {
            JudgeLineKind::Text(anim) => {
                anim.set_time(res.time);
            }
            JudgeLineKind::Paint(anim, ..) => {
                anim.set_time(res.time);
            }
            JudgeLineKind::TextureGif(anim, ..) => {
                anim.set_time(res.time);
            }
            _ => {}
        }
        self.color.set_time(res.time);
        self.cache.above_indices.retain_mut(|index| {
            while matches!(self.notes[*index].judge, JudgeStatus::Judged) {
                if self
                    .notes
                    .get(*index + 1)
                    .map_or(false, |it| it.above && it.speed == self.notes[*index].speed)
                {
                    *index += 1;
                } else {
                    return false;
                }
            }
            true
        });
        self.cache.below_indices.retain_mut(|index| {
            while matches!(self.notes[*index].judge, JudgeStatus::Judged) {
                if self.notes.get(*index + 1).map_or(false, |it| it.speed == self.notes[*index].speed) {
                    *index += 1;
                } else {
                    return false;
                }
            }
            true
        });
    }

    pub fn fetch_pos(line: &JudgeLine, res: &Resource, lines: &[JudgeLine]) -> Vector {
        if let Some(parent) = line.parent {
            let parent = &lines[parent];
            let mut parent_translation = Self::fetch_pos(parent, res, lines);
            parent_translation += Rotation2::new(parent.object.rotation.now().to_radians()) * line.object.now_translation(res);
            return parent_translation;
        }
        line.object.now_translation(res)
    }


    pub fn now_transform(&self, res: &Resource, lines: &[JudgeLine]) -> Matrix {
        /*if let Some(parent) = self.parent {
            let po = &lines[parent].object;
            let mut tr = Rotation2::new(po.rotation.now().to_radians()) * self.object.now_translation(res);
            tr += po.now_translation(res);
            self.object.now_rotation().append_translation(&tr)
        } else {
            self.object.now(res)
        }*/
            self.object.now_rotation().append_translation(&Self::fetch_pos(self, res, lines))
    }

    pub fn render(&self, mut ui: &mut Ui, res: &mut Resource, lines: &[JudgeLine], bpm_list: &mut BpmList, settings: &ChartSettings, id: usize) {
        let alpha = self.object.alpha.now_opt().unwrap_or(1.0) * res.alpha;
        let color = self.color.now_opt();
        let painter_state: Rc<RefCell<Option<Painter>>> = Rc::new(RefCell::new(None));
        res.with_model(self.now_transform(res, lines), |res| {
            res.with_model(self.object.now_scale(), |res| {
                res.apply_model(|res| {
                    match &self.kind {
                        JudgeLineKind::Normal => {
                            if res.config.ui_line {
                                let mut color = color.unwrap_or(res.judge_line_color);
                                let len = res.info.line_length;
                                color.a *= alpha.max(0.0);
                                if res.config.chart_debug {
                                    color.a = 0.10 + 0.90 * color.a;
                                } else if color.a == 0.0 {
                                    return;
                                }

                                if res.config.disable_loading {
                                    draw_line(-len, 0., len, 0., 0.01, color);
                                } else {
                                    let full_len = res.info.line_length;
                                    let duration = 4.03;
                                    let t_norm = (res.time / duration).min(1.0);
                                    let threshold = 0.2;
                                    let exp_factor = if t_norm < threshold {
                                        0.5 * (t_norm / threshold).powi(2)
                                    } else {
                                        0.5 + 0.5 * (1.0 - ((1.0 - t_norm) / (1.0 - threshold)).powi(2))
                                    };
                                    let current_len = full_len * exp_factor;
                                    draw_line(-current_len, 0., current_len, 0., 0.0075, color);
                                }
                            }
                        }
                        JudgeLineKind::Texture(texture, _) => {
                            let texture_ref = **texture;
                            let alpha = alpha.max(0.0);
                            if alpha == 0.0 && !res.config.chart_debug {
                                return;
                            }
                            let mut final_color = color.unwrap_or(WHITE);
                            final_color.a = if res.config.chart_debug {
                                0.10 + 0.90 * alpha
                            } else {
                                alpha
                            };
                            let (w, h) = (texture_ref.width() as f32, texture_ref.height() as f32);
                            let half_size = vec2(w * 0.5, h * 0.5);
                            draw_texture_ex(
                                texture_ref,
                                -half_size.x,
                                -half_size.y,
                                final_color,
                                DrawTextureParams {
                                    dest_size: Some(vec2(w, h)),
                                    flip_y: true,
                                    ..Default::default()
                                },
                            );
                        }
                        JudgeLineKind::TextureGif(anim, frames, _) => {
                            let t = anim.now_opt().unwrap_or(0.0);
                            let frame = frames.get_prog_frame(t);
                            let mut color = color.unwrap_or(WHITE);
                            color.a = alpha.max(0.0);
                            let hf = vec2(frame.width(), frame.height());
                            draw_texture_ex(
                                **frame,
                                -hf.x / 2.,
                                -hf.y / 2.,
                                color,
                                DrawTextureParams {
                                    dest_size: Some(hf),
                                    flip_y: true,
                                    ..Default::default()
                                },
                            );
                        }
                        JudgeLineKind::Text(anim) => {
                            let mut base_color = color.unwrap_or(WHITE);
                            if base_color.r == 0.0 && base_color.g == 0.0 && base_color.b == 0.0 && base_color.a == 0.0 {
                                base_color = WHITE;
                            }
                            let mut final_color = base_color;
                            final_color.a = alpha.max(0.0);
                            if res.config.chart_debug {
                                final_color.a = 0.10 + 0.90 * final_color.a;
                            } else if final_color.a == 0.0 {
                                return;
                            }
                            let now = anim.now();
                            res.apply_model_of(&Matrix::identity().append_nonuniform_scaling(&Vector::new(1., -1.)), |_| {
                                draw_text_aligned(ui, &now, 0., 0., (0.5, 0.5), 1., final_color);
                            });
                        }
                        JudgeLineKind::Paint(anim, _state) => {
                            let size  = anim.now();
                            let color = color.unwrap_or(WHITE);
                            let alpha = alpha;
                            let mut opt = painter_state.borrow_mut();
                            let painter = opt.get_or_insert_with(|| Painter::new());
                            painter.paint(&mut ui, size, alpha, color);
                        }
                    }
                })
            });
            if let JudgeLineKind::Paint(_, state) = &self.kind {
                let guard = state.borrow_mut();
                if guard.1 {
                    let ctx = unsafe { get_internal_gl() }.quad_context;
                    let tex = guard.0.as_ref().unwrap().texture(ctx);
                    let top = 1. / res.aspect_ratio;
                    draw_texture_ex(
                        Texture2D::from_miniquad_texture(tex),
                        -1.,
                        -top,
                        WHITE,
                        DrawTextureParams {
                            dest_size: Some(vec2(2., top * 2.)),
                            ..Default::default()
                        },
                    );
                }
            }
            let mut config = RenderConfig {
                settings,
                ctrl_obj: &mut self.ctrl_obj.borrow_mut(),
                line_height: self.height.now(),
                appear_before: f32::INFINITY,
                invisible_time: f32::INFINITY,
                draw_below: self.show_below,
                incline_sin: self.incline.now_opt().map(|it| it.to_radians().sin()).unwrap_or_default(),
            };
            if res.config.has_mod(Mods::FADE_OUT) {
                config.invisible_time = LIMIT_BAD;
            }
            if alpha < 0.0 {
                if !settings.pe_alpha_extension {
                    return;
                }
                let w = (-alpha).floor() as u32;
                match w {
                    1 => {
                        return;
                    }
                    2 => {
                        config.draw_below = false;
                    }
                    w if (100..1000).contains(&w) => {
                        config.appear_before = (w as f32 - 100.) / 10.;
                    }
                    w if (1000..2000).contains(&w) => {
                        // TODO unsupported
                        //Link: The implementation is to set the invisible time of the note, which is calculated as (w - 1000) / 10 seconds
                        config.invisible_time = (w as f32 - 1000.) / 10.;
                    }
                    _ => {}
                }
            }
            let (vw, vh) = (1.1, 1.);
            let p = [
                res.screen_to_world(Point::new(-vw, -vh)),
                res.screen_to_world(Point::new(-vw, vh)),
                res.screen_to_world(Point::new(vw, -vh)),
                res.screen_to_world(Point::new(vw, vh)),
            ];
            let height_above = p[0].y.max(p[1].y.max(p[2].y.max(p[3].y))) * res.aspect_ratio;
            let height_below = -p[0].y.min(p[1].y.min(p[2].y.min(p[3].y))) * res.aspect_ratio;
            let agg = res.config.aggressive;
            for note in self.notes.iter().take(self.cache.not_plain_count).filter(|it| it.above) {
                note.render(res, &mut config, bpm_list);
            }
            for index in &self.cache.above_indices {
                let speed = self.notes[*index].speed;
                let limit = height_above / speed;
                for note in self.notes[*index..].iter() {
                    if !note.above || speed != note.speed {
                        break;
                    }
                    if agg && note.height - config.line_height + note.object.translation.1.now() > limit {
                        break;
                    }
                    note.render(res, &mut config, bpm_list);
                }
            }
            res.with_model(Matrix::identity().append_nonuniform_scaling(&Vector::new(1.0, -1.0)), |res| {
                for note in self.notes.iter().take(self.cache.not_plain_count).filter(|it| !it.above) {
                    note.render(res, &mut config, bpm_list);
                }
                for index in &self.cache.below_indices {
                    let speed = self.notes[*index].speed;
                    let limit = height_below / speed;
                    for note in self.notes[*index..].iter() {
                        if speed != note.speed {
                            break;
                        }
                        if agg && note.height - config.line_height + note.object.translation.1.now() > limit {
                            break;
                        }
                        note.render(res, &mut config, bpm_list);
                    }
                }
            });
            if res.config.chart_debug {
                let pos = Self::fetch_pos(self, res, lines);
                let rotation = self.object.rotation.now();
                let text_alpha = {
                    let mut alpha_val = alpha;
                    if res.config.chart_debug {
                        alpha_val = 0.10 + 0.90 * alpha_val;
                    }
                    alpha_val.max(0.4)
                };

                let text_color = Color::new(1.0, 1.0, 1.0, text_alpha);
                //let height = self.height.now();
                let judged_count = self.notes.iter().filter(|n| matches!(n.judge, JudgeStatus::Judged)).count();
                let total_notes = self.notes.len();

                let mut parent_info = String::new();
                let mut current_parent = self.parent;
                let mut valid_parents = Vec::new(); // 存储有效的父线索引

                // 使用循环收集父线信息
                while let Some(parent_index) = current_parent {
                    // 检查索引是否有效
                    if parent_index < lines.len() {
                        valid_parents.push(parent_index);
                        current_parent = lines[parent_index].parent;
                    } else {
                        break;
                    }
                }

                if !valid_parents.is_empty() {
                    parent_info = "Parents: ".to_string();
                    parent_info += &valid_parents
                        .iter()
                        .map(|id| id.to_string())
                        .collect::<Vec<_>>()
                        .join(" -> ");
                }
                res.with_model(Matrix::identity().append_nonuniform_scaling(&Vector::new(1.0, -1.0)), |res| {
                res.apply_model(|_| {
                    ui.text(id.to_string()).pos(0., -0.01).anchor(0.5, 1.).color(text_color).size(0.5).draw();
                    let state_str = format!(
                        "P({:.3},{:.3})   R{:.1}°  N{}/{}   {}",
                        pos.x, pos.y,
                        rotation,
                        //height,
                        judged_count,
                        total_notes,
                        parent_info
                    );

                    ui.text(&state_str)
                        .pos(0., -0.05)
                        .anchor(0.5, 1.)
                        .size(0.35)
                        .color(text_color)
                        .draw();
                });
                });
            }
        });
    }
}
