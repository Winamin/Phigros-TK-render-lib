use super::{BpmList, Effect, JudgeLine, JudgeLineKind, Matrix, Resource, UIElement, Vector, Video};
use crate::{fs::FileSystem, judge::JudgeStatus, ui::Ui};
use anyhow::{Context, Result};
use macroquad::prelude::*;
use std::cell::RefCell;
use tracing::warn;

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
}

impl Chart {
    pub fn new(offset: f32, lines: Vec<JudgeLine>, bpm_list: BpmList, settings: ChartSettings, extra: ChartExtra) -> Self {
        let (attach_ui, order) = Self::init_attach_ui_and_order(&lines);
        Self {
            offset,
            lines,
            bpm_list: RefCell::new(bpm_list),
            settings,
            extra,
            order,
            attach_ui,
        }
    }

    fn init_attach_ui_and_order(lines: &[JudgeLine]) -> ([Option<usize>; 7], Vec<usize>) {
        let mut attach_ui = [None; 7];
        let mut unassigned = Vec::new();

        for (idx, line) in lines.iter().enumerate() {
            if let Some(element) = line.attach_ui {
                attach_ui[element as usize - 1] = Some(idx);
            } else {
                unassigned.push(idx);
            }
        }

        unassigned.sort_by_key(|&idx| (lines[idx].z_index, idx));
        (attach_ui, unassigned)
    }

    fn get_element_properties(
        &self,
        element: UIElement,
        res: &Resource,
        ct: Option<(f32, f32)>,
        pt: Option<(f32, f32)>,
    ) -> Option<(Matrix, Color)> {
        let id = self.attach_ui[element as usize - 1]?;
        let line = &self.lines[id];
        let obj = &line.object;

        let base_color = line.color.now_opt().unwrap_or(WHITE);
        let alpha = obj.now_alpha().max(0.0);
        let color = Color::new(base_color.r, base_color.g, base_color.b, base_color.a * alpha);

        Some((self.calculate_transform(line, res, ct, pt), color))
    }

    fn calculate_transform(&self, line: &JudgeLine, res: &Resource, ct: Option<(f32, f32)>, pt: Option<(f32, f32)>) -> Matrix {
        let mut translation = JudgeLine::fetch_pos(line, res, &self.lines);
        translation.y = -translation.y;

        let scale = line.object.now_scale_fix(
            ct.map_or_else(Vector::default, |(x, y)| Vector::new(x, y))
        );

        let rotation = line.object.new_rotation_wrt_point(
            -line.object.rotation.now().to_radians(),
            pt.map_or_else(Vector::default, |(x, y)| Vector::new(x, y))
        );

        Matrix::new_translation(&translation) * rotation * scale
    }

    #[inline]
    pub fn with_element<R>(
        &self,
        ui: &mut Ui,
        res: &Resource,
        element: UIElement,
        ct: Option<(f32, f32)>,
        pt: Option<(f32, f32)>,
        f: impl FnOnce(&mut Ui, Color) -> R,
    ) -> R {
        match self.get_element_properties(element, res, ct, pt) {
            Some((transform, color)) => ui.with(transform, |ui| f(ui, color)),
            None => f(ui, WHITE),
        }
    }

    pub fn with_element_noscale<R>(
        &self,
        ui: &mut Ui,
        res: &Resource,
        element: UIElement,
        ct: Option<(f32, f32)>,
        f: impl FnOnce(&mut Ui, Color) -> R,
    ) -> R {
        if let Some(id) = self.attach_ui[element as usize - 1] {
            let line = &self.lines[id];
            let obj = &line.object;
            
            let mut translation = obj.now_translation(res);
            translation.y = -translation.y;
            
            let mut scale = obj.now_scale_fix(ct.map_or_else(Vector::default, |(x, y)| Vector::new(x, y)));
            scale.m11 = 1.0;

            let transform = obj.now_rotation().append_translation(&translation) * scale;
            
            let base_color = line.color.now_opt().unwrap_or(WHITE);
            let alpha = obj.now_alpha().max(0.0);
            let color = Color::new(base_color.r, base_color.g, base_color.b, base_color.a * alpha);

            ui.with(transform, |ui| f(ui, color))
        } else {
            f(ui, WHITE)
        }
    }

    pub async fn load_textures(&mut self, fs: &mut dyn FileSystem) -> Result<()> {
        for line in &mut self.lines {
            if let JudgeLineKind::Texture(tex, path) = &mut line.kind {
                let data = fs.load_file(path)
                    .await
                    .with_context(|| format!("Failed to load texture: {}", path))?;
                *tex = image::load_from_memory(&data)?.into();
            }
        }
        Ok(())
    }

    pub fn reset(&mut self) {
        for line in &mut self.lines {
            line.cache.reset(&mut line.notes);
            for note in &mut line.notes {
                note.judge = JudgeStatus::NotJudged;
            }
        }
        for video in &mut self.extra.videos {
            video.next_frame = 0;
        }
    }

    pub fn update(&mut self, res: &mut Resource) {
        // Pre-calculate all transforms
        let transforms: Vec<_> = self.lines
            .iter()
            .map(|line| line.now_transform(res, &self.lines))
            .collect();
        
        {
            let mut bpm_guard = self.bpm_list.borrow_mut();
            self.lines.iter_mut()
                .zip(transforms)
                .enumerate()
                .for_each(|(idx, (line, tr))| {
                    line.update(res, tr, &mut bpm_guard, idx);
                });
        }

        self.extra.effects.iter_mut().for_each(|effect| effect.update(res));
        self.update_videos(res);
    }

    fn update_videos(&mut self, res: &mut Resource) {
        for video in &mut self.extra.videos {
            if let Err(e) = video.update(res.time) {
                warn!("Video update error: {:?}", e);
            }
        }
    }

    pub fn render(&self, ui: &mut Ui, res: &mut Resource) {
        self.render_videos(res);
        
        res.apply_model_of(
            &Matrix::identity().append_nonuniform_scaling(&Vector::new(
                if res.config.flip_x() { -1.0 } else { 1.0 },
                -1.0
            )),
            |res| {
                let mut bpm_guard = self.bpm_list.borrow_mut();
                self.render_lines(ui, res, &mut bpm_guard);
                res.note_buffer.borrow_mut().draw_all();
                self.finalize_rendering(res);
            }
        );
    }

    fn render_videos(&self, res: &mut Resource) {
        self.extra.videos.iter().for_each(|video| video.render(res));
    }

    fn render_lines(&self, ui: &mut Ui, res: &mut Resource, bpm_guard: &mut BpmList) {
        self.order.iter().for_each(|&idx| {
            self.lines[idx].render(ui, res, &self.lines, bpm_guard, &self.settings, idx);
        });
    }

    fn finalize_rendering(&self, res: &mut Resource) {
        if res.config.sample_count > 1 {
            unsafe { get_internal_gl() }.flush();
            if let Some(target) = &res.chart_target {
                target.blit();
            }
        }
    }
}