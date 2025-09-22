use super::{Anim, Resource};
use crate::ext::{source_of_image, ScaleType};
use anyhow::Result;
use macroquad::prelude::*;
use miniquad::{Texture, TextureFormat, TextureParams, TextureWrap};
use prpr_avc::AVPixelFormat;
use std::collections::VecDeque;
use std::io::Write;
use tempfile::NamedTempFile;

#[derive(Debug, Clone)]
struct FrameData {
    frame_index: usize,
    buf_y: Vec<u8>,
    buf_u: Vec<u8>,
    buf_v: Vec<u8>,
}

pub struct Video {
    video: prpr_avc::Video,
    _video_file: NamedTempFile,

    material: Material,
    tex_y: Texture2D,
    tex_u: Texture2D,
    tex_v: Texture2D,

    start_time: f32,
    scale_type: ScaleType,
    alpha: Anim<f32>,
    dim: Anim<f32>,
    frame_delta: f64,

    // 帧缓存系统
    frame_cache: VecDeque<FrameData>,
    current_video_frame: usize,
    pub next_frame: usize,
    ended: bool,

    // 异步解码配置
    max_cache_size: usize,
    decode_ahead_frames: usize,

    // 视频信息
    width: u32,
    height: u32,

    // 缓冲区重用
    buf_y: Vec<u8>,
    buf_u: Vec<u8>,
    buf_v: Vec<u8>,
}

fn new_tex(w: u32, h: u32) -> Texture2D {
    Texture2D::from_miniquad_texture(Texture::new_render_texture(
        unsafe { get_internal_gl() }.quad_context,
        TextureParams {
            width: w,
            height: h,
            format: TextureFormat::Alpha,
            filter: FilterMode::Linear,
            wrap: TextureWrap::Clamp,
        },
    ))
}

impl Video {
    pub fn new(
        data: Vec<u8>,
        start_time: f32,
        scale_type: ScaleType,
        alpha: Anim<f32>,
        dim: Anim<f32>,
    ) -> Result<Self> {
        let mut video_file = NamedTempFile::new()?;
        video_file.write_all(&data)?;
        drop(data);

        let video = prpr_avc::Video::open(
            video_file.path().as_os_str().to_str().unwrap(),
            AVPixelFormat::YUV420P,
        )?;

        let frame_delta = video.frame_rate().to_f64_inv();
        let format = video.stream_format();
        let w = format.width as u32;
        let h = format.height as u32;

        let buf_size_y = (w * h) as usize;
        let buf_size_uv = ((w / 2) * (h / 2)) as usize;
        let buf_y = vec![0; buf_size_y];
        let buf_u = vec![0; buf_size_uv];
        let buf_v = vec![0; buf_size_uv];

        let material = load_material(
            shader::VERTEX,
            shader::FRAGMENT,
            MaterialParams {
                pipeline_params: PipelineParams::default(),
                uniforms: Vec::new(),
                textures: vec!["tex_y".to_owned(), "tex_u".to_owned(), "tex_v".to_owned()],
            },
        )?;
        let tex_y = new_tex(w, h);
        let tex_u = new_tex(w / 2, h / 2);
        let tex_v = new_tex(w / 2, h / 2);
        material.set_texture("tex_y", tex_y);
        material.set_texture("tex_u", tex_u);
        material.set_texture("tex_v", tex_v);

        Ok(Self {
            video,
            _video_file: video_file,
            material,
            tex_y,
            tex_u,
            tex_v,
            start_time,
            scale_type,
            alpha,
            dim,
            frame_delta,
            frame_cache: VecDeque::new(),
            current_video_frame: 0,
            next_frame: 0,
            ended: false,
            max_cache_size: 10, // 最多缓存10帧
            decode_ahead_frames: 3, // 提前解码3帧
            width: w,
            height: h,
            buf_y,
            buf_u,
            buf_v,
        })
    }

    fn decode_frame_to_cache(&mut self) -> Result<bool> {
        if self.ended {
            return Ok(false);
        }

        let frame_data = self.video.with_frame(|frame| {
            self.buf_y.copy_from_slice(frame.data(0));
            self.buf_u.copy_from_slice(frame.data_half(1));
            self.buf_v.copy_from_slice(frame.data_half(2));

            FrameData {
                frame_index: self.current_video_frame,
                buf_y: self.buf_y.clone(),
                buf_u: self.buf_u.clone(),
                buf_v: self.buf_v.clone(),
            }
        });

        if let Some(data) = frame_data {
            self.frame_cache.push_back(data);
            self.current_video_frame += 1;
            Ok(true)
        } else {
            self.ended = true;
            Ok(false)
        }
    }

    fn ensure_frame_available(&mut self, target_frame: usize) -> Result<()> {
        // 清理过期的缓存帧
        while let Some(front) = self.frame_cache.front() {
            if front.frame_index < target_frame.saturating_sub(2) {
                self.frame_cache.pop_front();
            } else {
                break;
            }
        }

        // 跳过不需要的帧（如果目标帧远超当前帧）
        while self.current_video_frame < target_frame && self.frame_cache.is_empty() {
            if !self.decode_frame_to_cache()? {
                return Ok(());
            }
            // 如果跳跃太大，直接丢弃中间帧
            if target_frame > self.current_video_frame + 5 {
                self.frame_cache.clear();
            }
        }

        // 预解码帧以保持流畅播放
        let decode_target = target_frame + self.decode_ahead_frames;
        while self.current_video_frame <= decode_target
            && self.frame_cache.len() < self.max_cache_size
            && !self.ended {
            if !self.decode_frame_to_cache()? {
                break;
            }
        }

        Ok(())
    }

    pub fn update(&mut self, t: f32) -> Result<()> {
        if t < self.start_time || self.ended {
            return Ok(());
        }

        self.alpha.set_time(t);
        self.dim.set_time(t);

        let target_frame = ((t - self.start_time) as f64 / self.frame_delta) as usize;

        // 确保目标帧可用
        self.ensure_frame_available(target_frame)?;

        // 查找并应用目标帧
        if let Some(frame_data) = self.frame_cache.iter().find(|f| f.frame_index <= target_frame).cloned() {
            if frame_data.frame_index >= self.next_frame {
                let ctx = unsafe { get_internal_gl() }.quad_context;
                self.tex_y.raw_miniquad_texture_handle().update(ctx, &frame_data.buf_y);
                self.tex_u.raw_miniquad_texture_handle().update(ctx, &frame_data.buf_u);
                self.tex_v.raw_miniquad_texture_handle().update(ctx, &frame_data.buf_v);
                self.next_frame = frame_data.frame_index + 1;
            }
        }

        Ok(())
    }

    pub fn render(&self, res: &Resource) {
        if res.time < self.start_time || self.ended {
            return;
        }

        let alpha = self.alpha.now_opt().unwrap_or(1.0);
        if alpha <= 0.0 {
            return;
        }

        let top = 1. / res.aspect_ratio;
        let r = Rect::new(-1., -top, 2., top * 2.);

        gl_use_material(self.material);
        let s = source_of_image(&self.tex_y, r, self.scale_type).unwrap_or_else(|| Rect::new(0., 0., 1., 1.));
        let dim = 1. - self.dim.now();
        let color = Color::new(dim, dim, dim, alpha);
        let vertices = [
            Vertex::new(r.x, r.y, 0., s.x, s.y, color),
            Vertex::new(r.right(), r.y, 0., s.right(), s.y, color),
            Vertex::new(r.x, r.bottom(), 0., s.x, s.bottom(), color),
            Vertex::new(r.right(), r.bottom(), 0., s.right(), s.bottom(), color),
        ];
        let gl = unsafe { get_internal_gl() }.quad_gl;
        gl.draw_mode(DrawMode::Triangles);
        gl.geometry(&vertices, &[0, 2, 3, 0, 1, 3]);
        gl_use_default_material();
    }

    // 添加一些性能调优方法
    pub fn set_cache_size(&mut self, size: usize) {
        self.max_cache_size = size;
    }

    pub fn set_decode_ahead_frames(&mut self, frames: usize) {
        self.decode_ahead_frames = frames;
    }
}

mod shader {
    pub const VERTEX: &str = r#"#version 300 es
in vec3 position;
in vec2 texcoord;
in vec4 color0;

out vec2 uv;
out vec4 color;

uniform mat4 Model;
uniform mat4 Projection;

void main() {
    gl_Position = Projection * Model * vec4(position, 1.0);
    color = color0 / 255.0;
    uv = texcoord;
}"#;

    pub const FRAGMENT: &str = r#"#version 300 es
precision mediump float;

in vec4 color;
in vec2 uv;

out vec4 FragColor;

uniform sampler2D tex_y;
uniform sampler2D tex_u;
uniform sampler2D tex_v;

void main() {
    vec3 yuv = vec3(
        texture(tex_y, uv).a,
        texture(tex_u, uv).a - 0.5,
        texture(tex_v, uv).a - 0.5
    );
    yuv.x = 1.1643 * (yuv.x - 0.0625);
    mat3 color_matrix = mat3(
        vec3(1.0,   0.0,     1.402),
        vec3(1.0,  -0.344,  -0.714),
        vec3(1.0,   1.772,   0.0  )
    );

    FragColor = vec4(yuv * color_matrix, 1.0) * color;
}"#;
}