extern crate cc;

fn main() {
    cc::Build::new()
        .cuda(true)
        .cpp(true)
        .flag("-cudart=shared")
        .flag("-ccbin=clang")
        .files(&["./src/mandelbrot.cpp", "./src/mandelbrot_gpu.cu"])
        .compile("mandelbrot.a");
    println!("cargo:rustc-link-search=native=/opt/cuda");
    println!("cargo:rustc-link-search=/opt/cuda");
    println!("cargo:rustc-env=LD_LIBRARY_PATH=/opt/cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
}
