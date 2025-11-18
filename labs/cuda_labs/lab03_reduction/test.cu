/**
 * Lab 03: Reduction - Test Suite
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "Error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define BLOCK_SIZE 256

__global__ void reduceWarp(const float *input, float *output, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float sum = 0.0f;
    if (i < n) sum += input[i];
    if (i + blockDim.x < n) sum += input[i + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid < 32) {
        float warpSum = sdata[tid];
        for (int offset = 16; offset > 0; offset >>= 1) {
            warpSum += __shfl_down_sync(0xffffffff, warpSum, offset);
        }
        if (tid == 0) output[blockIdx.x] = warpSum;
    }
}

int main() {
    printf("Reduction Test Suite\n");

    int sizes[] = {1024, 65536, 1<<20, 1<<24};
    int passed = 0;

    for (int t = 0; t < 4; t++) {
        int n = sizes[t];
        float *h_in = (float*)malloc(n * sizeof(float));
        for (int i = 0; i < n; i++) h_in[i] = 1.0f;

        float *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, 1024 * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice));

        int grid = (n + 2*BLOCK_SIZE - 1) / (2*BLOCK_SIZE);
        reduceWarp<<<grid, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_in, d_out, n);

        float *h_out = (float*)malloc(grid * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_out, d_out, grid * sizeof(float), cudaMemcpyDeviceToHost));

        float gpu_sum = 0.0f;
        for (int i = 0; i < grid; i++) gpu_sum += h_out[i];

        bool ok = fabs(gpu_sum - n) < 1.0f;
        printf("Size %d: %s (%.0f vs %d)\n", n, ok ? "PASS" : "FAIL", gpu_sum, n);
        if (ok) passed++;

        free(h_in); free(h_out);
        CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_out));
    }

    printf("\nResults: %d/4 tests passed\n", passed);
    return passed == 4 ? 0 : 1;
}
