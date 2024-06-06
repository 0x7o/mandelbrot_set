#include <iostream>
#include "mandelbrot_gpu.h"

extern "C" {
    void calculate_mandelbrot(const double* cx, const double* cy, int num_points, unsigned int max_iters, unsigned int* output) {
        double* non_const_cx = const_cast<double*>(cx);
        double* non_const_cy = const_cast<double*>(cy);

        unsigned int* gpu_results = gpu_mandelbrot(non_const_cx, non_const_cy, num_points, max_iters);

        for (int i = 0; i < num_points; ++i) {
            output[i] = gpu_results[i];
        }

        delete[] gpu_results;
    }
}