#include <thrust/complex.h>
#include <cuda_runtime.h>
#include "mandelbrot_gpu.h"

__device__ unsigned int num_iters(double cx, double cy, unsigned int max_iters) {
    thrust::complex<double> z(0.0, 0.0);
    thrust::complex<double> c(cx, cy);

    for (unsigned int i = 0; i <= max_iters; ++i) {
        if (thrust::abs(z) > 2.0) {
            return i;
        }
        z = z * z + c;
    }

    return max_iters;
}

__global__ void mandelbrot_kernel(unsigned int* results, double* cx, double* cy, unsigned int max_iters, int num_points) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < num_points) {
        results[idx] = num_iters(cx[idx], cy[idx], max_iters);
    }
}

unsigned int* gpu_mandelbrot(double* cx, double* cy, size_t num_points, unsigned int max_iters) {
    unsigned int *dev_results, *results;
    double *dev_cx, *dev_cy;

    results = new unsigned int[num_points];

    cudaMalloc((void**)&dev_cx, num_points * sizeof(double));
    cudaMalloc((void**)&dev_cy, num_points * sizeof(double));
    cudaMalloc((void**)&dev_results, num_points * sizeof(unsigned int));

    cudaMemcpy(dev_cx, cx, num_points * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_cy, cy, num_points * sizeof(double), cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks_per_grid = (num_points + threads_per_block - 1) / threads_per_block;

    mandelbrot_kernel<<<blocks_per_grid, threads_per_block>>>(dev_results, dev_cx, dev_cy, max_iters, num_points);

    cudaMemcpy(results, dev_results, num_points * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudaFree(dev_cx);
    cudaFree(dev_cy);
    cudaFree(dev_results);

    return results;
}
