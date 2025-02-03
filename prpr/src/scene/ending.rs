crate::tl_file!("ending");

use super::{draw_background, game::SimpleRecord, loading::UploadFn, NextScene, Scene};
use crate::{
    config::Config,
    ext::{
        create_audio_manger, draw_illustration, draw_parallelogram, draw_parallelogram_ex, draw_text_aligned, SafeTexture, ScaleType,
        PARALLELOGRAM_SLOPE,
    },
    info::ChartInfo,
    judge::{icon_index, PlayResult},
    scene::show_message,
    task::Task,
    time::TimeManager,
    ui::{Dialog, MessageHandle, RectButton, Ui},
};
use anyhow::Result;
use macroquad::prelude::*;
use sasa::{AudioClip, AudioManager, Music, MusicParams};
use serde::Deserialize;
use std::{cell::RefCell, ops::DerefMut};

#[derive(Deserialize)]
pub struct RecordUpdateState {
    pub best: bool,
    pub improvement: u32,
    pub gain_exp: f32,
    pub new_rks: f32,
}

pub struct EndingScene {
    background: SafeTexture,
    illustration: SafeTexture,
    player: SafeTexture,
    icons: [SafeTexture; 8],
    icon_retry: SafeTexture,
    icon_proceed: SafeTexture,
    target: Option<RenderTarget>,
    audio: AudioManager,
    bgm: Music,

    info: ChartInfo,
    result: PlayResult,
    player_name: String,
    player_rks: Option<f32>,
    challenge_texture: SafeTexture,
    challenge_rank: u32,
    autoplay: bool,
    speed: f32,
    next: u8, // 0 -> none, 1 -> pop, 2 -> exit
    update_state: Option<RecordUpdateState>,
    rated: bool,

    upload_fn: Option<UploadFn>,
    upload_task: Option<(Task<Result<RecordUpdateState>>, MessageHandle)>,
    record_data: Option<Vec<u8>>,
    record: Option<SimpleRecord>,

    btn_retry: RectButton,
    btn_proceed: RectButton,
}

impl EndingScene {
    pub fn new(
        background: SafeTexture,
        illustration: SafeTexture,
        player: SafeTexture,
        icons: [SafeTexture; 8],
        icon_retry: SafeTexture,
        icon_proceed: SafeTexture,
        info: ChartInfo,
        result: PlayResult,
        challenge_texture: SafeTexture,
        config: &Config,
        bgm: AudioClip,
        upload_fn: Option<UploadFn>,
        player_rks: Option<f32>,
        record_data: Option<Vec<u8>>,
        record: Option<SimpleRecord>,
    ) -> Result<Self> {
        let mut audio = create_audio_manger(config)?;
        let bgm = audio.create_music(
            bgm,
            MusicParams {
                amplifier: config.volume_music,
                loop_mix_time: 0.,
                ..Default::default()
            },
        )?;
        let upload_task = upload_fn
            .as_ref()
            .and_then(|f| record_data.clone().map(|data| (f(data), show_message(tl!("uploading")).handle())));
        Ok(Self {
            background,
            illustration,
            player,
            icons,
            icon_retry,
            icon_proceed,
            target: None,
            audio,
            bgm,
            update_state: if upload_task.is_some() {
                None
            } else {
                Some(RecordUpdateState {
                    best: true,
                    improvement: result.score,
                    gain_exp: 0.,
                    new_rks: 0.,
                })
            },
            rated: upload_task.is_some(),

            info,
            result,
            player_name: config.player_name.clone(),
            player_rks,
            challenge_texture,
            challenge_rank: config.challenge_rank,
            autoplay: config.autoplay(),
            speed: config.speed,
            next: 0,

            upload_fn,
            upload_task,
            record_data,
            record,

            btn_retry: RectButton::new(),
            btn_proceed: RectButton::new(),
        })
    }
}

thread_local! {
    static RE_UPLOAD: RefCell<bool> = RefCell::default();
}

impl Scene for EndingScene {
    fn enter(&mut self, tm: &mut TimeManager, target: Option<RenderTarget>) -> Result<()> {
        tm.reset();
        tm.seek_to(-0.4);
        self.target = target;
        Ok(())
    }

    fn pause(&mut self, tm: &mut TimeManager) -> Result<()> {
        self.bgm.pause()?;
        tm.pause();
        Ok(())
    }

    fn resume(&mut self, tm: &mut TimeManager) -> Result<()> {
        self.bgm.play()?;
        tm.resume();
        Ok(())
    }

    fn touch(&mut self, _tm: &mut TimeManager, touch: &Touch) -> Result<bool> {
        if self.btn_retry.touch(touch) {
            if self.upload_task.is_some() {
                show_message(tl!("still-uploading"));
            } else {
                self.next = 1;
            }
            return Ok(true);
        }
        if self.btn_proceed.touch(touch) {
            if self.upload_task.is_some() {
                show_message(tl!("still-uploading"));
            } else {
                self.next = 2;
            }
            return Ok(true);
        }
        Ok(false)
    }

    fn update(&mut self, tm: &mut TimeManager) -> Result<()> {
        self.audio.recover_if_needed()?;
        if tm.now() >= 0.24 && self.target.is_none() && self.bgm.paused() {
            self.bgm.play()?;
        }
        if RE_UPLOAD.with(|it| std::mem::replace(it.borrow_mut().deref_mut(), false)) && self.upload_task.is_none() {
            self.upload_task = self
                .record_data
                .clone()
                .map(|data| ((self.upload_fn.as_ref().unwrap())(data), show_message(tl!("uploading")).handle()));
        }
        if let Some((task, handle)) = &mut self.upload_task {
            if let Some(result) = task.take() {
                handle.cancel();
                match result {
                    Err(err) => {
                        let error = format!("{:?}", err.context(tl!("upload-failed")));
                        Dialog::plain(tl!("upload-failed"), error)
                            .buttons(vec![tl!("upload-cancel").to_string(), tl!("upload-retry").to_string()])
                            .listener(move |pos| {
                                if pos == 1 {
                                    RE_UPLOAD.with(|it| *it.borrow_mut() = true);
                                }
                            })
                            .show();
                    }
                    Ok(state) => {
                        self.update_state = Some(state);
                        show_message(tl!("uploaded")).ok();
                    }
                }
                self.upload_task = None;
            }
        }
        Ok(())
    }

    fn render(&mut self, tm: &mut TimeManager, ui: &mut Ui) -> Result<()> {

        const START0: f32 = 0.0;
        const END0: f32 = 1.25;
        const START1: f32 = 0.00;
        const END1: f32 = 0.55;
        const START2: f32 = 0.00;
        const END2: f32 = 0.85;
        const START3: f32 = 0.00;
        const END3: f32 = 1.20;

        let mut cam = ui.camera();
        let asp = -cam.zoom.y;
        let top = 1. / asp;
        let t = tm.now() as f32;
        let gl = unsafe { get_internal_gl() }.quad_gl;
        let res = &self.result;
        cam.render_target = self.target;
        set_camera(&cam);
        draw_background(*self.background);

        fn ran(t: f32, l: f32, r: f32) -> f32 {
            ((t - l) / (r - l)).clamp(0., 1.)
        }
        fn tran(gl: &mut QuadGl, x: f32) {
            gl.push_model_matrix(Mat4::from_translation(vec3(x * 2., 0., 0.)));
        }

        let p_main = (1. - ran(t, START0, END0)).powi(6);
        tran(gl, p_main);
        let r = draw_illustration(*self.illustration, -0.37, 0., 1.03, 1.2, WHITE, true);
        let slope = PARALLELOGRAM_SLOPE;
        let ratio = 0.2;
        draw_parallelogram_ex(
            Rect::new(r.x, r.y + r.h * (1. - ratio), r.w - r.h * (1. - ratio) * slope, r.h * ratio),
            None,
            Color::default(),
            Color::new(0., 0., 0., 0.6),
            true,
        );
        let rr = draw_text_aligned(ui, &self.info.level, r.right() - r.h / 7. * 13. * 0.13 - 0.02, r.bottom() - top / 20., (1., 1.), 0.40, WHITE);
        let p = (r.x + 0.05, r.bottom() - top / 16.);
        let mw = rr.x - 0.02 - p.0;
        let mut text_size = 0.92;
        let mut text = ui.text(&self.info.name).pos(p.0, p.1).anchor(0., 1.).size(text_size);
        let max_width = mw;
        let text_width = text.measure().w;
        if text_width > max_width {
            text_size *= max_width / text_width
        }
        //if text.measure().w <= mw {
        //    text.draw();
        //} else {
            drop(text);
            ui.text(&self.info.name)
            .pos(p.0, p.1)
            .anchor(0., 1.)
            .size(text_size)
            //.max_width(mw)
            .draw();
        //}
        gl.pop_model_matrix();

        let dx = 0.07;
        let c = Color::new(0., 0., 0., 1.0);
        let c2 = Color::new(0., 0., 0., 0.6);

        tran(gl, (1. - ran(t, START1, END1)).powi(4) + p_main);
        let main = Rect::new(r.right() - 0.05, r.y, r.w * 0.80, r.h / 2.);
        draw_parallelogram(main, None, c2, true);
        {
            let spd = if (self.speed - 1.).abs() <= 1e-4 {
                format!(" ")//String::new()
            } else {
                format!(" {:.2}x", self.speed)
            };
            let text = if self.autoplay {
                format!("AUTOPLAY {spd}")
            } else if !self.rated {
                format!("{spd}")
            } else if let Some(state) = &self.update_state {
                format!(
                    "{spd}  {}",
                    if state.best {
                        format!("NEW BEST +{:07}", state.improvement)
                    } else {
                        format!(" ")//String::new()
                    }
                )
            } else {
                "Uploading…".to_owned()
            };
            let pa = ran(t, 0.2, 0.6).powi(5);
            let r = draw_text_aligned(ui, &text, main.x + dx + 0.01, main.bottom() - 0.040, (0., 1.), 0.34, Color::new(1., 1., 1., pa));
            let r = draw_text_aligned(ui, &format!("{:07}", res.score), r.x - 0.005, r.y - 0.022, (0., 1.), 1.05, Color::new(1., 1., 1., pa));
            let icon = icon_index(res.score, res.num_of_notes == res.max_combo);
            let p = ran(t, 1.2, 1.6).powi(5);
            let p2 = ran(t, 1.65, 1.9).powi(3);
            let s = main.h * 0.72;
            let ct = (main.right() + 0.01 - main.h * slope - s / 2., r.bottom() + 0.03 - s / 2.);
            let s = s + s * (1. - p2) * 0.3;
            draw_texture_ex(
                *self.icons[icon],
                ct.0 - s / 2.,
                ct.1 - s / 2.,
                Color::new(1., 1., 1., p),
                DrawTextureParams {
                    dest_size: Some(vec2(s, s)),
                    ..Default::default()
                },
            );
        }
        gl.pop_model_matrix();

        tran(gl, (1. - ran(t, START2, END2)).powi(2) + p_main);
        let d = r.h / 15.5;
        let pa = ran(t, 0.6, 1.0).powi(5);
        let s1 = Rect::new(main.x - d * 4. * slope, main.bottom() + d, main.w - d * 5. * slope, d * 2.85);
        draw_parallelogram(s1, None, c2, true);
        {
            let dy = 0.025;
            let r = draw_text_aligned(ui, "Max Combo", s1.x + dx - 0.01, s1.bottom() - dy, (0., 1.), 0.32, Color::new(1., 1., 1., pa));
            draw_text_aligned(ui, &res.max_combo.to_string(), r.x, r.y - 0.008, (0., 1.), 0.65, Color::new(1., 1., 1., pa));
            let r = draw_text_aligned(ui, "Accuracy", s1.right() - dx + 0.02, s1.bottom() - dy, (1., 1.), 0.32, Color::new(1., 1., 1., pa));
            draw_text_aligned(ui, &format!("{:.2}%", res.accuracy * 100.), r.right(), r.y - 0.008, (1., 1.), 0.63, Color::new(1., 1., 1., pa));
        }
        gl.pop_model_matrix();

        tran(gl, (1. - ran(t, START3, END3)).powi(2) + p_main);
        let s2 = Rect::new(s1.x - d * 4. * slope, s1.bottom() + d, s1.w, s1.h);
        draw_parallelogram(s2, None, c2, true);
        {
            let dy = 0.022;
            let dy2 = 0.014;
            let bg = 0.55;
            let sm = 0.24;
            let pa = ran(t, 1.1, 1.4).powi(5);
            let draw_count = |ui: &mut Ui, ratio: f32, name: &str, count: u32| {
                let r = draw_text_aligned(ui, name, s2.x + s2.w * ratio, s2.bottom() - dy, (0.5, 1.), sm, Color::new(1., 1., 1., pa));
                draw_text_aligned(ui, &count.to_string(), r.center().x, r.y - dy2, (0.5, 1.), bg, Color::new(1., 1., 1., pa));
            };
            draw_count(ui, 0.13, "Perfect", res.counts[0]);
            draw_count(ui, 0.31, "Good", res.counts[1]);
            draw_count(ui, 0.45, "Bad", res.counts[2]);
            draw_count(ui, 0.59, "Miss", res.counts[3]);

            let sm = 0.3;
            let l = s2.x + s2.w * 0.70;
            let rt = s2.x + s2.w * 0.92;
            let cy = s2.center().y;
            let r = draw_text_aligned(ui, "Early", l, cy - dy2 / 2.3, (0., 1.), sm, Color::new(1., 1., 1., pa));
            draw_text_aligned(ui, &res.early.to_string(), rt, r.bottom(), (1., 1.), sm, Color::new(1., 1., 1., pa));
            let r = draw_text_aligned(ui, "Late", l, cy + dy2 / 2.3, (0., 0.), 0.3, Color::new(1., 1., 1., pa));
            draw_text_aligned(ui, &res.late.to_string(), rt, r.y, (1., 0.), sm, Color::new(1., 1., 1., pa));
        }
        gl.pop_model_matrix();

        let dy = 0.010;
        let w = 0.202;
        let p = (1. - ran(t, 0.7, 1.8)).powi(7); // retry
        let p2 = (1. - ran(t, 0.7, 1.8)).powi(5); // next
        let h = 0.117;
        let s = 0.10;
        let hs = h * 0.28;
        let params = DrawTextureParams {
            dest_size: Some(vec2(hs * 2., hs * 2.)),
            ..Default::default()
        };
        tran(gl, -p * 0.1);
        let r = Rect::new(-1. - h * slope, -top + dy, w, h);
        draw_parallelogram(r, None, c, true);
        draw_parallelogram(Rect::new(r.x + r.w * (1. - s), r.y, r.w * s, r.h), None, WHITE, false);
        let ct = r.center();
        draw_texture_ex(*self.icon_retry, ct.x - hs * 0.9, ct.y - hs, WHITE, params.clone());
        gl.pop_model_matrix();
        if p <= 0. {
            self.btn_retry.set(ui, r);
        }

        tran(gl, p2 * 0.1);
        let r = Rect::new(1. + h * slope - w, top - dy - h, w, h);
        draw_parallelogram(r, None, c, true);
        draw_parallelogram(Rect::new(r.x, r.y, r.w * s, r.h), None, WHITE, false);
        let ct = r.center();
        draw_texture_ex(*self.icon_proceed, ct.x - hs * 0.8 - r.w * s / 2., ct.y - hs, WHITE, params);
        gl.pop_model_matrix();
        if p <= 0. {
            self.btn_proceed.set(ui, r);
        }

        let alpha = ran(t, 0.65, 1.15); // rks
        let main = Rect::new(1. - 0.27, -top + dy * 3.2, 0.35, 0.11);
        draw_parallelogram(main, None, Color::new(0., 0., 0., c.a * alpha), false);
        let sub = Rect::new(1. - 0.125, main.center().y + 0.015, 0.12, 0.03);
        let color = Color::new(1., 1., 1., alpha);
        draw_parallelogram(sub, None, color, false);
        draw_text_aligned(
            ui,
            /*&if let Some(state) = &self.update_state {
                format!("{:.2}", state.new_rks)
            } else if let Some(rks) = &self.player_rks {
                format!("{rks:.2}")
            } else {
                "".to_owned()
            }*/
            &if let Some(rks) = &self.player_rks {
                format!("{rks:.2}")
            } else {
                "16.00".to_owned()
            }
            ,
            sub.center().x,
            sub.center().y - 0.003,
            (0.5, 0.5),
            0.37,
            Color::new(0., 0., 0., alpha),
        );
        let r = draw_illustration(*self.player, 1. - 0.21, main.center().y, 0.12 / (0.076 * 7.), 0.12 / (0.076 * 7.), color, true);
        let text = draw_text_aligned(ui, &self.player_name, r.x - 0.005, r.center().y, (1., 0.5), 0.54, color);
        draw_parallelogram(
            Rect::new(text.x - main.h * slope - 0.02, main.y, r.x - text.x + main.h * slope * 2. + 0.0220, main.h),
            None,
            Color::new(0., 0., 0., c.a * alpha),
            false,
        );
        draw_text_aligned(ui, &self.player_name, r.x - 0.01, r.center().y, (1., 0.5), 0.54, color);

        let ct = (1. - 0.1 + 0.043, main.center().y - 0.034 + 0.02);
        let (w, h) = (0.09 * self.challenge_texture.width() / 78., 0.04 * self.challenge_texture.height() / 38.);
        let r = Rect::new(ct.0 - w / 2., ct.1 - h / 2., w, h);
        ui.fill_rect(r, (*self.challenge_texture, r, ScaleType::Fit, color));
        let ct = r.center();
        ui.text(self.challenge_rank.to_string())
            .pos(ct.x, ct.y)
            .anchor(0.5, 1.)
            .size(0.46)
            .color(color)
            .draw();

        Ok(())
    }

    fn next_scene(&mut self, _tm: &mut TimeManager) -> NextScene {
        if self.next != 0 {
            let _ = self.bgm.pause();
        }
        match self.next {
            0 => NextScene::None,
            1 => NextScene::Pop,
            2 => {
                if let Some(rec) = &self.record {
                    NextScene::PopNWithResult(2, Box::new(rec.clone()))
                } else {
                    NextScene::PopN(2)
                }
            }
            _ => unreachable!(),
        }
    }
}