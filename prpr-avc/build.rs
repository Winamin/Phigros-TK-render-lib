use std::path::Path;

fn main() {
    pkg_config::Config::new()
        .atleast_version("58.76.100")
        .probe("libavformat")
        .unwrap();
    println!("cargo:rustc-link-lib=avformat");
    println!("cargo:rustc-link-lib=swresample");
    println!("cargo:rustc-link-lib=avfilter");
    println!("cargo:rustc-link-lib=avdevice");
    println!("cargo:rustc-link-lib=avcodec");
    println!("cargo:rustc-link-lib=avutil");
    println!("cargo:rustc-link-lib=swscale");
    println!("cargo:rustc-link-lib=bz2");
    println!("cargo:rustc-link-lib=lzma");
    let libs_dir = std::env::var("PRPR_AVC_LIBS").unwrap_or_else(|_| format!("{}/static-lib", std::env::var("CARGO_MANIFEST_DIR").unwrap()));
    let libs_path = Path::new(&libs_dir).join(std::env::var("TARGET").unwrap());
    let libs_path = libs_path.display();
    println!("cargo:rustc-link-search={libs_path}");
    println!("cargo:rustc-link-lib=z");
    println!("cargo:rerun-if-changed={libs_path}");
}
