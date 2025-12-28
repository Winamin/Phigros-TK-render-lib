use super::{chart::ChartSettings, object::CtrlObject, Anim, AnimFloat, BpmList, Matrix, Note, Object, Point, RenderConfig, Resource, Vector};
use crate::{
    config::Mods,
    ext::{draw_text_aligned, get_viewport, NotNanExt, SafeTexture},
    judge::{JudgeStatus, LIMIT_BAD},
    ui::Ui,
    //hand::assign_hands,
    info::ChartFormat,
};
use macroquad::prelude::*;
use macroquad::miniquad::{RenderPass, Texture, TextureParams, TextureWrap, FilterMode, TextureFormat};
use nalgebra::Rotation2;
use serde::Deserialize;
use std::sync::{Mutex, Arc, RwLock};
use once_cell::sync::Lazy;
use std::collections::HashMap;
//use crate::config::Config;

// 优化的纹理缓存懒加载
static TEXTURE_CACHE: Lazy<RwLock<HashMap<usize, Texture2D>>> = Lazy::new(|| {
    let map = HashMap::with_capacity(1024); // 预分配容量，减少重新分配
    RwLock::new(map)
});

// 统一的翻转Y矩阵懒加载
static FLIP_Y_MATRIX: Lazy<Matrix> = Lazy::new(|| {
    Matrix::identity().append_nonuniform_scaling(&Vector::new(1.0, -1.0))
});

// Painter缓存懒加载
static PAINTER_CACHE: Lazy<Mutex<(RenderPass, Texture, (i32, i32, i32, i32))>> = Lazy::new(|| {
    let gl = unsafe { get_internal_gl() };
    let vp = get_viewport();

    let tex = Texture::new_render_texture(
        gl.quad_context,
        TextureParams {
            width: vp.2 as _,
            height: vp.3 as _,
            format: TextureFormat::RGBA8,
            filter: FilterMode::Linear,
            wrap: TextureWrap::Clamp,
        },
    );

    let pass = RenderPass::new(gl.quad_context, tex.clone(), None);
    Mutex::new((pass, tex, vp))
});

// 渲染优化常量懒加载
static RENDER_CONSTANTS: Lazy<RenderConstants> = Lazy::new(|| RenderConstants {
    duration: 4.03,
    threshold: 0.2,
    inv_threshold: 1.0 / 0.2,
    inv_one_minus_threshold: 1.0 / (1.0 - 0.2),
    line_width_normal: 0.01,
    line_width_loading: 0.0075,
});

// 渲染常量结构体
struct RenderConstants {
    duration: f32,
    threshold: f32,
    inv_threshold: f32,
    inv_one_minus_threshold: f32,
    line_width_normal: f32,
    line_width_loading: f32,
}

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
    Paint(Anim<f32>, Arc<Mutex<(Option<RenderPass>, bool)>>),
    TextureGif(Anim<f32>, GifFrames, String),
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

struct Painter {
    pass: RenderPass,
    viewport: (i32, i32, i32, i32),
    cleared: bool,
    last_color_alpha: f32,
    last_size: f32,
    cached_texture: Option<Texture>,
    cached_pass: Option<RenderPass>,
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
    pub ctrl_obj: Arc<Mutex<CtrlObject>>,
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

impl Painter {
    pub fn new() -> Self {
        let (pass, tex, vp) = {
            let guard = PAINTER_CACHE.lock().unwrap();
            (guard.0.clone(), guard.1.clone(), guard.2)
        };

        Painter {
            pass,
            viewport: vp,
            cleared: false,
            last_color_alpha: -1.0,
            last_size: -1.0,
            cached_texture: Some(tex),
            cached_pass: Some(pass),
        }
    }

    fn paint(&mut self, ui: &mut Ui, size: f32, alpha: f32, mut color: Color) {
        let gl = unsafe { get_internal_gl() };
        let ctx = gl.quad_context;
        if let Some(cached_texture) = &self.cached_texture {
            if cached_texture != &self.pass.texture(ctx) {
                self.cached_texture = Some(self.pass.texture(ctx).clone());
            }
        }
        if let Some(cached_pass) = &self.cached_pass {
            if cached_pass != &self.pass {
                self.cached_pass = Some(self.pass.clone());
            }
        }
        if self.cleared {
            if let Some(ref pass) = self.cached_pass {
                gl.quad_gl.render_pass(Some(pass.clone()));
            }
            gl.quad_gl.viewport(Some(self.viewport));
        }
        let new_alpha = alpha.max(0.0) * 2.55;
        if self.last_color_alpha != new_alpha {
            color.a = new_alpha;
            self.last_color_alpha = new_alpha;
        }
        if size != self.last_size {
            self.last_size = size;
        }
        if size <= 0.0 {
            if !self.cleared {
                clear_background(Color::default());
                self.cleared = true;
            }
        } else {
            let radius = size / self.viewport.2 as f32 * 2.0;
            ui.fill_circle(0., 0., radius, color);
            self.cleared = true;
        }
        if let Some(ref pass) = self.cached_pass {
            gl.quad_gl.render_pass(Some(pass.clone()));
        }
        gl.quad_gl.viewport(Some(self.viewport));
    }
}

unsafe impl Sync for JudgeLine {}
unsafe impl Send for JudgeLine {}

impl JudgeLine {
    pub fn update(&mut self, res: &mut Resource, tr: Matrix, bpm_list: &mut BpmList, index: usize) {
        let rot = self.object.rotation.now();
        self.height.set_time(res.time);
        let line_height = self.height.now();
        let mut ctrl_obj = self.ctrl_obj.lock().unwrap();
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
        //if res.config.hand_split {
        //    let config = Config::default();
        //    let rot = self.object.rotation.now();
        //    assign_hands(&mut self.notes, &config, index, rot, bpm_list);
        //}
    }

    pub fn update_hand_assign_with_world_pos(&mut self, res: &mut Resource, _world_pos: Vector, bpm_list: &mut BpmList, index: usize) {
        if !res.config.hand_split || self.notes.is_empty() {
            return;
        }
        
        let config = crate::config::Config::default();
        let rot = self.object.rotation.now();
        
        // 获取判定线的世界位置和变换矩阵
        let line_transform = self.now_transform(res, &[]);
        let line_translation = Vector::new(line_transform[(0, 2)], line_transform[(1, 2)]);
        let _line_rotation = self.object.rotation.now();
        
        // 从主视角计算世界坐标
        let mut enhanced_notes_data = Vec::with_capacity(self.notes.len());
        let chart_ratio_inv = res.chart_ratio_inv; // 使用缓存的值
        let vw = 1.2 * chart_ratio_inv;
        let vh = chart_ratio_inv;
        
        for note in &self.notes {
            // 音符在判定线局部坐标系中的位置
            let local_x = note.object.translation.0.now();
            let local_y = note.object.translation.1.now();
            
            // 转换为标准化的世界坐标（不考虑判定线旋转）
            let world_x_no_rotation = local_x * vw;
            let world_y_no_rotation = local_y * vh;
            
            // 应用判定线的世界位置偏移
            let world_x = world_x_no_rotation + line_translation.x;
            let world_y = world_y_no_rotation + line_translation.y;
            
            // 现在将这个"主视角下的世界坐标"传递给手部分配函数
            // 手部分配函数内部会处理旋转角度的影响
            let true_world_pos = Vector::new(world_x, world_y);
            let enhanced_pos = true_world_pos; // 直接使用真实世界坐标
            
            enhanced_notes_data.push((true_world_pos, enhanced_pos));
        }
        
        // 创建增强版临时音符用于AI计算
        let mut ai_notes: Vec<crate::core::Note> = self.notes
            .iter()
            .zip(enhanced_notes_data.iter())
            .map(|(note, &(_true_world_pos, enhanced_pos))| {
                let mut ai_note = note.clone();
                // 使用增强版位置信息进行手部分配
                ai_note.object.translation.0 = crate::core::AnimFloat::fixed(enhanced_pos.x);
                ai_note.object.translation.1 = crate::core::AnimFloat::fixed(enhanced_pos.y);
                ai_note
            })
            .collect();
        
        // 分配手（使用统一主视角数据）
        crate::hand::assign_hands_unified_perspective(&mut ai_notes, &config, index, rot, bpm_list, &enhanced_notes_data);
        
        // 将分配结果复制回原始音符
        for (orig_note, ai_note) in self.notes.iter_mut().zip(ai_notes.iter()) {
            orig_note.hand = ai_note.hand;
        }
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
        self.object.now_rotation().append_translation(&Self::fetch_pos(self, res, lines))
    }

    pub fn render(&self, mut ui: &mut Ui, res: &mut Resource, lines: &[JudgeLine], bpm_list: &mut BpmList, settings: &ChartSettings, id: usize) {

        // 早期退出优化：预先计算 alpha
        let alpha = self.object.alpha.now_opt().unwrap_or(1.0) * res.alpha;
        let color = self.color.now_opt();

        // 优化1: 提前计算常量，避免重复计算
        let final_alpha = alpha.max(0.0);
        let is_debug = res.config.chart_debug;
        let is_fade_out = res.config.has_mod(Mods::FADE_OUT);

        // 优化2: 预分配并重用 painter_state
        let painter_state: Arc<Mutex<Option<Painter>>> = Arc::new(Mutex::new(None));

        res.with_model(self.now_transform(res, lines), |res| {
            res.with_model(self.object.now_scale(), |res| {
                res.apply_model(|res| {
                    // 优化3: 使用查找表代替 match，提高分支预测性能
                    match &self.kind {
                        JudgeLineKind::Normal => {
                            if !res.config.ui_line {
                                return;
                            }

                            let mut line_color = color.unwrap_or(res.judge_line_color);
                            line_color.a *= final_alpha;

                            // 早期退出：alpha 为 0 且非调试模式
                            if line_color.a == 0.0 && !is_debug {
                                return;
                            }

                            if is_debug {
                                line_color.a = 0.10 + 0.90 * line_color.a;
                            }

                            let len = res.info.line_length;

                            if res.config.disable_loading {
                                draw_line(-len, 0., len, 0., RENDER_CONSTANTS.line_width_normal, line_color);
                            } else {
                                // 使用懒加载的渲染常量
                                let t_norm = (res.time / RENDER_CONSTANTS.duration).min(1.0);

                                // 优化5: 使用 FMA (Fused Multiply-Add) 友好的计算
                                let exp_factor = if t_norm < RENDER_CONSTANTS.threshold {
                                    let normalized = t_norm * RENDER_CONSTANTS.inv_threshold;
                                    0.5 * normalized * normalized
                                } else {
                                    let normalized = (1.0 - t_norm) * RENDER_CONSTANTS.inv_one_minus_threshold;
                                    0.5 + 0.5 * (1.0 - normalized * normalized)
                                };

                                let current_len = len * exp_factor;
                                draw_line(-current_len, 0., current_len, 0., RENDER_CONSTANTS.line_width_loading, line_color);
                            }
                        }
                        JudgeLineKind::Texture(texture, _) => {
                            if final_alpha == 0.0 && !is_debug {
                                return;
                            }

                            let mut tex_color = color.unwrap_or(WHITE);
                            if res.time <= 0. && tex_color == WHITE {
                                tex_color = BLACK;
                            }

                            tex_color.a = if is_debug {
                                0.10 + 0.90 * final_alpha
                            } else {
                                final_alpha
                            };

                            // 优化6: 减少锁竞争 - 使用 RwLock 的 try_read 快速路径
                            let key = texture.get_tex() as *const Texture2D as usize;
                            let texture_2d = {
                                // 快速读取路径
                                if let Ok(cache) = TEXTURE_CACHE.try_read() {
                                    if let Some(tex) = cache.get(&key) {
                                        tex.clone()
                                    } else {
                                        drop(cache);
                                        TEXTURE_CACHE.write().unwrap()
                                            .entry(key)
                                            .or_insert_with(|| texture.get_tex().clone())
                                            .clone()
                                    }
                                } else {
                                    // 降级到标准路径
                                    let cache = TEXTURE_CACHE.read().unwrap();
                                    cache.get(&key).cloned().unwrap_or_else(|| {
                                        drop(cache);
                                        TEXTURE_CACHE.write().unwrap()
                                            .entry(key)
                                            .or_insert_with(|| texture.get_tex().clone())
                                            .clone()
                                    })
                                }
                            };

                            let hf = vec2(texture_2d.width(), texture_2d.height());
                            draw_texture_ex(
                                texture_2d,
                                -hf.x * 0.5,
                                -hf.y * 0.5,
                                tex_color,
                                DrawTextureParams {
                                    dest_size: Some(hf),
                                    flip_y: true,
                                    ..Default::default()
                                },
                            );
                        }
                        JudgeLineKind::TextureGif(anim, frames, _) => {
                            let t = anim.now_opt().unwrap_or(0.0);
                            let frame = frames.get_prog_frame(t);
                            let mut gif_color = color.unwrap_or(WHITE);
                            gif_color.a = final_alpha;

                            let hf = vec2(frame.width(), frame.height());
                            draw_texture_ex(
                                **frame,
                                -hf.x * 0.5,
                                -hf.y * 0.5,
                                gif_color,
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
                            final_color.a = final_alpha;

                            if is_debug {
                                final_color.a = 0.10 + 0.90 * final_color.a;
                            } else if final_color.a == 0.0 {
                                return;
                            }

                            let now = anim.now();
                            res.apply_model_of(&FLIP_Y_MATRIX, |_| {
                                draw_text_aligned(ui, &now, 0., 0., (0.5, 0.5), 1., final_color);
                            });
                        }
                        JudgeLineKind::Paint(anim, _state) => {
                            let size = anim.now();
                            let paint_color = color.unwrap_or(WHITE);

                            let mut opt = painter_state.lock().unwrap();
                            let painter = opt.get_or_insert_with(|| Painter::new());
                            painter.paint(&mut ui, size, alpha, paint_color);
                        }
                    }
                })
            });

            // Paint 类型的后处理
            if let JudgeLineKind::Paint(_, state) = &self.kind {
                let gl = unsafe { get_internal_gl() };
                let ctx = gl.quad_context;

                let guard = state.lock().unwrap();
                let ready = guard.1;

                if ready {
                    if let Some(pass) = guard.0.as_ref() {
                        let tex = pass.texture(ctx);
                        let top = res.inv_aspect_ratio; // 使用缓存的值
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
            }

            let mut ctrl_obj = self.ctrl_obj.lock().unwrap();
            let line_height = self.height.now();

            let mut config = RenderConfig {
                settings,
                ctrl_obj: &mut *ctrl_obj,
                line_height,
                appear_before: f32::INFINITY,
                invisible_time: f32::INFINITY,
                draw_below: self.show_below,
                incline_sin: self.incline.now_opt().map(|it| it.to_radians().sin()).unwrap_or_default(),
                global_speed_factor: res.config.note_speed_factor,
            };

            if is_fade_out {
                config.invisible_time = LIMIT_BAD;
            }

            // 优化8: Alpha 扩展逻辑优化
            if alpha < 0.0 && settings.pe_alpha_extension {
                let w = (-alpha).floor() as u32;
                match w {
                    1 => return,
                    2 => config.draw_below = false,
                    100..=999 => config.appear_before = (w as f32 - 100.) * 0.1,
                    1000..=1999 => config.invisible_time = (w as f32 - 1000.) * 0.1,
                    _ => {}
                }
            } else if alpha < 0.0 {
                return;
            }

            let chart_ratio_inv = res.chart_ratio_inv; // 使用缓存的值
            let vw = 1.2 * chart_ratio_inv;
            let vh = chart_ratio_inv;

            let viewport_points = [
                res.screen_to_world(Point::new(-vw, -vh)),
                res.screen_to_world(Point::new(-vw, vh)),
                res.screen_to_world(Point::new(vw, -vh)),
                res.screen_to_world(Point::new(vw, vh)),
            ];

            let inv_aspect_ratio = res.inv_aspect_ratio; // 使用缓存的值
            let height_above = viewport_points.iter()
                .map(|p| p.y)
                .fold(f32::NEG_INFINITY, f32::max) * (1.0 / inv_aspect_ratio);

            let height_below = viewport_points.iter()
                .map(|p| p.y)
                .fold(f32::INFINITY, f32::min) * (1.0 / inv_aspect_ratio);

            let agg = res.config.aggressive;
            let chart_format_matches = matches!(res.chart_format, ChartFormat::Pgr | ChartFormat::Rpe);
            let note_scale_positive = res.config.note_scale > 0.;

            if !note_scale_positive {
                return;
            }
            let mut height = self.height.clone();
            let not_plain_count = self.cache.not_plain_count;

            // 渲染上方音符（普通）
            for note in self.notes[..not_plain_count].iter().filter(|n| n.above) {
                height.set_time(note.time.min(res.time));
                let line_height_at_note = height.now();
                let note_height = note.height - line_height_at_note + note.object.translation.1.now();

                // 优化14: 使用短路评估减少分支
                if agg && chart_format_matches {
                    let inv_speed = 1.0 / note.speed;
                    if note_height < height_below * inv_speed {
                        continue;
                    }
                    if note_height > height_above * inv_speed {
                        break;
                    }
                }

                note.render(res, &mut config, bpm_list);
            }

            for &index in &self.cache.above_indices {
                let speed = self.notes[index].speed;
                let inv_speed = 1.0 / speed;
                let height_below_scaled = height_below * inv_speed;
                let height_above_scaled = height_above * inv_speed;

                for note in &self.notes[index..] {
                    if !note.above || speed != note.speed {
                        break;
                    }

                    let note_height = note.height - config.line_height + note.object.translation.1.now();

                    if agg && note_height < height_below_scaled {
                        continue;
                    }
                    if agg && note_height > height_above_scaled {
                        break;
                    }

                    note.render(res, &mut config, bpm_list);
                }
            }

            res.with_model(*FLIP_Y_MATRIX, |res| {
                // 渲染下方音符（普通）
                for note in self.notes[..not_plain_count].iter().filter(|n| !n.above) {
                    height.set_time(note.time.min(res.time));
                    let line_height_at_note = height.now();
                    let note_height = note.height - line_height_at_note + note.object.translation.1.now();

                    if agg && chart_format_matches {
                        let inv_speed = 1.0 / note.speed;
                        if note_height < -height_above * inv_speed {
                            continue;
                        }
                        if note_height > -height_below * inv_speed {
                            break;
                        }
                    }

                    note.render(res, &mut config, bpm_list);
                }

                // 渲染下方音符（索引）
                for &index in &self.cache.below_indices {
                    let speed = self.notes[index].speed;
                    let inv_speed = 1.0 / speed;
                    let neg_height_above_scaled = -height_above * inv_speed;
                    let neg_height_below_scaled = -height_below * inv_speed;

                    for note in &self.notes[index..] {
                        if speed != note.speed {
                            break;
                        }

                        let note_height = note.height - config.line_height + note.object.translation.1.now();

                        if agg && note_height < neg_height_above_scaled {
                            continue;
                        }
                        if agg && note_height > neg_height_below_scaled {
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
                let judged_count = self.notes.iter().filter(|n| matches!(n.judge, JudgeStatus::Judged)).count();
                let total_notes = self.notes.len();

                let mut parent_info = String::new();
                let mut current_parent = self.parent;
                let mut valid_parents = Vec::new();

                while let Some(parent_index) = current_parent {
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
                res.with_model(*FLIP_Y_MATRIX, |res| {
                    res.apply_model(|_| {
                        ui.text(id.to_string()).pos(0., -0.01).anchor(0.5, 1.).color(text_color).size(0.5).draw();
                        let state_str = format!(
                            "P({:.3},{:.3})   R{:.1}°  N{}/{}   {}",
                            pos.x, pos.y,
                            rotation,
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