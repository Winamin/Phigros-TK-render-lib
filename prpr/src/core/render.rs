use macroquad::{
    texture::{RenderTarget, Texture2D},
    window::get_internal_gl,
    miniquad::{gl::GLuint, RenderPass, Texture, TextureFormat},
};

pub struct MSRenderTarget {
    dim: (u32, u32),
    fbo: GLuint,
    rbo: GLuint,
    dummy: RenderTarget,
    output: [RenderTarget; 2],
}

pub fn copy_fbo(src: GLuint, dst: GLuint, dim: (u32, u32)) -> bool {
    unsafe {
                use macroquad::miniquad::gl::*;
                glBindFramebuffer(GL_READ_FRAMEBUFFER, src);
                glBindFramebuffer(GL_DRAW_FRAMEBUFFER, dst);
                let (w, h) = (dim.0 as i32, dim.1 as i32);
                glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, GL_COLOR_BUFFER_BIT, GL_NEAREST);
                glGetError() == GL_NO_ERROR
            }}

pub fn internal_id(target: &RenderTarget) -> GLuint {
    target.render_pass.gl_internal_id(unsafe { get_internal_gl() }.quad_context)
}

impl MSRenderTarget {
    pub fn new(dim: (u32, u32), samples: u32) -> Self {
        let mut fbo = 0;
        let mut rbo = 0;
        unsafe {
            use macroquad::miniquad::gl::*;
            glGenRenderbuffers(1, &mut rbo);
            glBindRenderbuffer(GL_RENDERBUFFER, rbo);
            glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples as i32, GL_RGB8, dim.0 as i32, dim.1 as i32);
            glGenFramebuffers(1, &mut fbo);
            glBindFramebuffer(GL_FRAMEBUFFER, fbo);
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rbo);
        }

        let gl = unsafe { get_internal_gl() };

        // 创建输出目标
        let mut create_target = || {
            let texture = Texture::new_render_texture(
                gl.quad_context,
                macroquad::miniquad::TextureParams {
                    width: dim.0,
                    height: dim.1,
                    format: TextureFormat::RGB8,
                    ..Default::default()
                },
            );
            RenderTarget {
                texture: Texture2D::from_miniquad_texture(texture),
                render_pass: RenderPass::new(gl.quad_context, texture, None),
            }
        };

        let output1 = create_target();
        let output2 = create_target();

        // 创建dummy纹理
        let dummy_texture = Texture::new_render_texture(
            gl.quad_context,
            macroquad::miniquad::TextureParams {
                width: dim.0,
                height: dim.1,
                format: TextureFormat::RGB8,
                ..Default::default()
            },
        );

        Self {
            dim,
            fbo,
            rbo,
            dummy: RenderTarget {
                texture: Texture2D::from_miniquad_texture(dummy_texture),
                render_pass: RenderPass::from_raw(gl.quad_context, fbo, dummy_texture),
            },
            output: [output1, output2],
        }
    }

    pub fn blit(&self) {
        copy_fbo(self.fbo, internal_id(&self.output[0]), self.dim);
    }

    pub fn swap(&mut self) {
        self.output.swap(0, 1);
    }

    pub fn input(&self) -> RenderTarget {
        self.dummy
    }

    pub fn output(&self) -> RenderTarget {
        self.output[0]
    }

    pub fn old(&self) -> RenderTarget {
        self.output[1]
    }
}

impl Drop for MSRenderTarget {
    fn drop(&mut self) {
        unsafe {
            use miniquad::gl::*;
            glDeleteRenderbuffers(1, &self.rbo);
            glDeleteFramebuffers(1, &self.fbo);
        }
    }
}