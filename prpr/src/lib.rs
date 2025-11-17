pub mod bin;
pub mod config;
pub mod core;
pub mod dir;
pub mod ext;
pub mod fs;
pub mod info;
pub mod judge;
pub mod l10n;
pub mod parse;
pub mod particle;
pub mod scene;
pub mod task;
pub mod time;
pub mod ui;

pub mod hand;
pub mod hand_model;
pub mod gpu_utils;

#[cfg(feature = "log")]
pub mod log;

#[cfg(feature = "closed")]
pub mod inner;

#[cfg(target_os = "ios")]
pub mod objc;

pub use scene::Main;

pub fn build_conf() -> macroquad::window::Conf {
    macroquad::window::Conf {
        window_title: "Phi TK".to_string(),
        window_width: 600,
        window_height: 800,
        ..Default::default()
    }
}
