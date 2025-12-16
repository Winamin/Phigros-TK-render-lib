use super::{BpmList, Effect, JudgeLine, JudgeLineKind, Matrix, Resource, UIElement, Vector, Video};
use crate::{ext::ChunkLoader, fs::FileSystem, judge::JudgeStatus, ui::Ui};
use anyhow::{Context, Result};
use macroquad::prelude::*;
use std::cell::RefCell;
use tracing::warn;

//use rayon::prelude::*;

#[derive(Default)]
pub struct ChartExtra {
    pub effects: Vec<Effect>,
    pub global_effects: Vec<Effect>,
    pub videos: Vec<Video>,
}

#[derive(Default)]
pub struct ChartSettings {
    pub pe_alpha_extension: bool,
    pub hold_partial_cover: bool,
}

pub struct Chart {
    pub offset: f32,
    pub lines: Vec<JudgeLine>,
    pub bpm_list: RefCell<BpmList>,
    pub settings: ChartSettings,
    pub extra: ChartExtra,
    pub order: Vec<usize>,
    pub attach_ui: [Option<usize>; 7],
    pub chunk_loader: Option<ChunkLoader>,
    pub max_time: f32,
}

impl Chart {
    pub fn new(offset: f32, lines: Vec<JudgeLine>, bpm_list: BpmList, settings: ChartSettings, extra: ChartExtra) -> Self {
        let mut attach_ui = [None; 7];
        let mut order = (0..lines.len())
            .filter(|it| {
                if let Some(element) = lines[*it].attach_ui {
                    attach_ui[element as usize - 1] = Some(*it);
                    false
                } else {
                    true
                }
            })
            .collect::<Vec<_>>();
        order.sort_by_key(|it| (lines[*it].z_index, *it));
        let max_time = lines
            .iter()
            .flat_map(|line| line.notes.iter())
            .map(|note| {
                match note.kind {
                    crate::core::NoteKind::Hold { end_time, .. } => end_time,
                    _ => note.time,
                }
            })
            .fold(0.0_f32, |acc, time| acc.max(time));
            
        Self {
            offset,
            lines,
            bpm_list: RefCell::new(bpm_list),
            settings,
            extra,

            order,
            attach_ui,
            chunk_loader: None,
            max_time,
        }
    }

    pub fn enable_chunked_loading(&mut self) {
        self.chunk_loader = Some(ChunkLoader::new(self.max_time));
    }

    pub fn update_chunk_loading(&mut self, current_time: f32) -> Vec<usize> {
        if let Some(loader) = &mut self.chunk_loader {
            loader.update(current_time)
        } else {
            Vec::new()
        }
    }

    pub fn should_render_at_time(&self, time: f32) -> bool {
        if let Some(loader) = &self.chunk_loader {
            let chunk_id = loader.chunked_chart.get_chunk_for_time(time);
            chunk_id < loader.chunked_chart.chunks.len() && loader.chunked_chart.chunks[chunk_id].loaded
        } else {
            true
        }
    }

    #[inline]
    pub fn with_element<R>(&self, ui: &mut Ui, res: &Resource, element: UIElement, ct: Option<(f32, f32)>, pt: Option<(f32, f32)>, f: impl FnOnce(&mut Ui, Color) -> R) -> R {
        if let Some(id) = self.attach_ui[element as usize - 1] {
            let lines = &self.lines;
            let line = &lines[id];
            let obj = &line.object;
            let mut tr = JudgeLine::fetch_pos(line, res, lines);
            tr.y = -tr.y;
            let mut color = self.lines[id].color.now_opt().unwrap_or(WHITE);
            color.a *= obj.now_alpha().max(0.);
            let scale = obj.now_scale_fix(ct.map_or_else(|| Vector::default(), |(x, y)| Vector::new(x, y)));
            let ro = obj.new_rotation_wrt_point(-obj.rotation.now().to_radians(), pt.map_or_else(|| Vector::default(), |(x, y)| Vector::new(x, y)));
            ui.with(Matrix::new_translation(&tr) * ro * scale, |ui| f(ui, color))
        } else {
            f(ui, WHITE)
        }
    }

    pub fn with_element_noscale<R>(&self, ui: &mut Ui, res: &Resource, element: UIElement, ct: Option<(f32, f32)>, f: impl FnOnce(&mut Ui, Color) -> R) -> R {
        if let Some(id) = self.attach_ui[element as usize - 1] {
            let obj = &self.lines[id].object;
            let mut tr = obj.now_translation(res);
            tr.y = -tr.y;
            let mut color = self.lines[id].color.now_opt().unwrap_or(WHITE);
            color.a *= obj.now_alpha().max(0.);
            let mut scale = obj.now_scale_fix(ct.map_or_else(|| Vector::default(), |(x, y)| Vector::new(x , y)));
            scale.m11 = 1.0;
            ui.with(obj.now_rotation().append_translation(&tr) * scale, |ui| f(ui, color))
        } else {
            f(ui, WHITE)
        }
    }

    pub async fn load_textures(&mut self, fs: &mut dyn FileSystem) -> Result<()> {
        for line in &mut self.lines {
            if let JudgeLineKind::Texture(tex, path) = &mut line.kind {
                *tex = image::load_from_memory(&fs.load_file(path).await.with_context(|| format!("failed to load illustration {path}"))?)?.into();
            }
        }
        Ok(())
    }

    pub fn reset(&mut self) {
        self.lines
            .iter_mut()
            .flat_map(|it| it.notes.iter_mut())
            .for_each(|note| note.judge = JudgeStatus::NotJudged);
        for line in &mut self.lines {
            line.cache.reset(&mut line.notes);
        }
        for video in &mut self.extra.videos {
            video.next_frame = 0;
        }
    }

    pub fn update(&mut self, res: &mut Resource) {
        //TODO: 优化
        for line in &mut self.lines {
            line.object.set_time(res.time);
        }

        let world_positions: Vec<Vector> = (0..self.lines.len())
            .map(|i| JudgeLine::fetch_pos(&self.lines[i], res, &self.lines))
            .collect();

        let trs: Vec<Matrix> = self.lines
            .iter()
            .map(|line| line.now_transform(res, &self.lines))
            .collect();

        let mut guard = self.bpm_list.borrow_mut();
        for (index, (line, tr)) in self.lines.iter_mut().zip(trs).enumerate() {
            line.update(res, tr, &mut guard, index);

            if res.config.hand_split {
                line.update_hand_assign_with_world_pos(res, world_positions[index], &mut guard, index);
            }
        }
        drop(guard);

        for effect in &mut self.extra.effects {
            effect.update(res);
        }
        for video in &mut self.extra.videos {
            if let Err(err) = video.update(res.time) {
                warn!("video error: {err:?}");
            }
        }
    }

    pub fn render(&self, ui: &mut Ui, res: &mut Resource) {
        res.apply_model_of(&Matrix::identity().append_nonuniform_scaling(&Vector::new(if res.config.flip_x() { -1. } else { 1. }, 1.)), |res| {
            for video in &self.extra.videos {
                video.render(res);
            }
        });
        res.apply_model_of(&Matrix::identity().append_nonuniform_scaling(&Vector::new(if res.config.flip_x() { -1. } else { 1. }, -1.)), |res| {
            let mut guard = self.bpm_list.borrow_mut();
            for id in &self.order {
                self.lines[*id].render(ui, res, &self.lines, &mut guard, &self.settings, *id);
            }
            drop(guard);
            res.note_buffer.borrow_mut().draw_all();
            if res.config.sample_count > 1 {
                unsafe { get_internal_gl() }.flush();
                if let Some(target) = &res.chart_target {
                    target.blit();
                }
            }
        });
    }
}