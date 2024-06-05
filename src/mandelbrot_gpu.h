#ifndef MANDELBROT_GPU_H
#define MANDELBROT_GPU_H

unsigned int* gpu_mandelbrot(double* cx, double* cy, size_t num_points, unsigned int max_iters);

#endif // MANDELBROT_GPU_H
