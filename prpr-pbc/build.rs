use std::path::Path;

fn main() {
  println!("cargo:rustc-link-lib=X11");
  println!("cargo:rustc-link-lib=Xext");
  println!("cargo:rustc-link-lib=Xv");
}
