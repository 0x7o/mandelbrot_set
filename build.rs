use cc;

fn main() {
    cc::Build::new()
        .cuda(true)
        .cpp(true)
        .flag("-cudart=shared")
        .flag("-ccbin=gcc")
        .files(&["./src/mandelbrot.cpp", "./src/mandelbrot_gpu.cu"])
        .compile("mandelbrot.a");
}
