use std::path::Path;

//extern crate cc;
//extern crate libc;

fn main() {
  println!("cargo:rustc-link-search=/usr/lib/x86_64-linux-gnu");
  println!("cargo:rustc-link-lib=lzma");
  println!("cargo:rustc-link-lib=X11");
  println!("cargo:rustc-link-lib=Xext");
  println!("cargo:rustc-link-lib=Xv");
}
