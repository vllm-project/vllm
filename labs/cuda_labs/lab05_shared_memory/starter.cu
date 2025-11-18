/**
 * Lab 05: Shared Memory - Starter Code
 * 1D Stencil operation with radius
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "Error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define RADIUS 3
#define BLOCK_SIZE 256

// TODO 1: Implement naive stencil (global memory only)
__global__ void stencilNaive(const float *input, float *output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        float sum = 0.0f;
        for (int offset = -RADIUS; offset <= RADIUS; offset++) {
            int idx = i + offset;
            if (idx >= 0 && idx < n) {
                sum += input[idx];
            }
        }
        output[i] = sum;
    }
}

// TODO 2: Implement stencil with shared memory
__global__ void stencilShared(const float *input, float *output, int n) {
    // Allocate shared memory: BLOCK_SIZE + 2*RADIUS
    __shared__ float sdata[BLOCK_SIZE + 2 * RADIUS];

    int gindex = blockIdx.x * blockDim.x + threadIdx.x;
    int lindex = threadIdx.x + RADIUS;

    // TODO: Load main data into shared memory

    // TODO: Load halo elements (left and right boundaries)

    // TODO: Synchronize

    // TODO: Compute stencil using shared memory

    // Your code here:

}

void stencilCPU(const float *input, float *output, int n) {
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int offset = -RADIUS; offset <= RADIUS; offset++) {
            int idx = i + offset;
            if (idx >= 0 && idx < n) {
                sum += input[idx];
            }
        }
        output[i] = sum;
    }
}

int main() {
    int n = 1 << 24;  // 16M elements
    size_t size = n * sizeof(float);

    float *h_in = (float*)malloc(size);
    float *h_out = (float*)malloc(size);
    float *h_ref = (float*)malloc(size);
    for (int i = 0; i < n; i++) h_in[i] = (float)(i % 100);

    stencilCPU(h_in, h_ref, n);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, size));
    CUDA_CHECK(cudaMalloc(&d_out, size));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Naive
    CUDA_CHECK(cudaEventRecord(start));
    stencilNaive<<<gridSize, BLOCK_SIZE>>>(d_in, d_out, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float naive_time;
    CUDA_CHECK(cudaEventElapsedTime(&naive_time, start, stop));
    printf("Naive: %.3f ms\n", naive_time);

    // Shared
    CUDA_CHECK(cudaEventRecord(start));
    stencilShared<<<gridSize, BLOCK_SIZE>>>(d_in, d_out, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float shared_time;
    CUDA_CHECK(cudaEventElapsedTime(&shared_time, start, stop));
    printf("Shared: %.3f ms (%.2fx speedup)\n", shared_time, naive_time/shared_time);

    CUDA_CHECK(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));

    // Verify
    bool ok = true;
    for (int i = 0; i < n && ok; i++) {
        if (fabs(h_out[i] - h_ref[i]) > 0.01f) ok = false;
    }
    printf("Verification: %s\n", ok ? "PASS" : "FAIL");

    free(h_in); free(h_out); free(h_ref);
    CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_out));

    return 0;
}
