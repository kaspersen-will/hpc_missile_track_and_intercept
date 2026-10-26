// Minimal CUDA test: SAXPY with explicit host<->device transfers.
// Purpose: check that nvcc, nsys and ncu work on the GPU nodes.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void saxpy(int n, float a, const float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
    const int n = 1 << 24;                       // ~16.8M elements
    const size_t bytes = n * sizeof(float);

    float *h_x = (float *)malloc(bytes), *h_y = (float *)malloc(bytes);
    for (int i = 0; i < n; ++i) { h_x[i] = 1.0f; h_y[i] = 2.0f; }

    float *d_x, *d_y;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);
    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);   // dst first
    cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice);

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;       // ceiling
    saxpy<<<blocks, threads>>>(n, 3.0f, d_x, d_y);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost);

    printf("y[0] = %.1f, y[n-1] = %.1f (expected 5.0)\n", h_y[0], h_y[n - 1]);

    cudaFree(d_x); cudaFree(d_y);
    free(h_x); free(h_y);
    return 0;
}
