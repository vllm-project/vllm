/**
 * Lab 03: Parallel Reduction - Complete Solution
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define BLOCK_SIZE 256

// Naive reduction - interleaved addressing
__global__ void reduceNaive(const float *input, float *output, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    // Interleaved addressing (causes bank conflicts)
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// Optimized reduction - sequential addressing
__global__ void reduceOptimized(const float *input, float *output, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    // Sequential addressing (no bank conflicts)
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// Warp-level reduction
__global__ void reduceWarp(const float *input, float *output, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Load two elements per thread
    float sum = 0.0f;
    if (i < n) sum += input[i];
    if (i + blockDim.x < n) sum += input[i + blockDim.x];

    sdata[tid] = sum;
    __syncthreads();

    // Reduce within block
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Warp-level reduction using shuffle
    if (tid < 32) {
        float warpSum = sdata[tid];
        for (int offset = 16; offset > 0; offset >>= 1) {
            warpSum += __shfl_down_sync(0xffffffff, warpSum, offset);
        }
        if (tid == 0) output[blockIdx.x] = warpSum;
    }
}

float reduceCPU(const float *data, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += data[i];
    return sum;
}

int main() {
    int n = 1 << 24;
    size_t size = n * sizeof(float);

    float *h_input = (float*)malloc(size);
    for (int i = 0; i < n; i++) h_input[i] = 1.0f;

    float cpu_result = reduceCPU(h_input, n);
    printf("CPU result: %f\n", cpu_result);

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float) * 1024));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Naive
    CUDA_CHECK(cudaEventRecord(start));
    reduceNaive<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_input, d_output, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float naive_time;
    CUDA_CHECK(cudaEventElapsedTime(&naive_time, start, stop));
    printf("Naive: %.3f ms\n", naive_time);

    // Optimized
    CUDA_CHECK(cudaEventRecord(start));
    reduceOptimized<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_input, d_output, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float opt_time;
    CUDA_CHECK(cudaEventElapsedTime(&opt_time, start, stop));
    printf("Optimized: %.3f ms (%.2fx speedup)\n", opt_time, naive_time/opt_time);

    // Warp
    CUDA_CHECK(cudaEventRecord(start));
    reduceWarp<<<(n + 2*BLOCK_SIZE - 1) / (2*BLOCK_SIZE), BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_input, d_output, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float warp_time;
    CUDA_CHECK(cudaEventElapsedTime(&warp_time, start, stop));
    printf("Warp: %.3f ms (%.2fx speedup)\n", warp_time, naive_time/warp_time);

    free(h_input);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
