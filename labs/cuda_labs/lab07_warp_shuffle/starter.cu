/**
 * Lab 07: Warp Shuffle Operations - Starter Code
 * Warp-level reduction using shuffle primitives
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

#define WARP_SIZE 32
#define BLOCK_SIZE 256

// TODO 1: Implement warp reduction using __shfl_down_sync
__device__ float warpReduceSum(float val) {
    // TODO: Use __shfl_down_sync to reduce within warp
    // Hint: Start with offset=16, then 8, 4, 2, 1
    // Each iteration halves the active threads

    // Your code here:


    return val;
}

// TODO 2: Implement block reduction using warp primitives
__global__ void reduceShfl(const float *input, float *output, int n) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data
    float val = (i < n) ? input[i] : 0.0f;

    // TODO: Reduce within warp using warpReduceSum

    // Your code here:


    // TODO: Reduce across warps using shared memory
    // Only need shared memory for warp-level results
    __shared__ float warp_sums[BLOCK_SIZE / WARP_SIZE];

    // Your code here:


    // First thread writes result
    if (tid == 0) {
        output[blockIdx.x] = 0.0f;  // Replace with actual sum
    }
}

// TODO 3: Implement warp scan (prefix sum) using shuffle
__device__ float warpScan(float val) {
    // TODO: Implement inclusive prefix sum within warp
    // Hint: Use __shfl_up_sync
    // offset = 1, 2, 4, 8, 16

    // Your code here:


    return val;
}

// Reference implementation using shared memory
__global__ void reduceShared(const float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

int main() {
    int n = 1 << 24;  // 16M elements
    size_t size = n * sizeof(float);

    float *h_input = (float*)malloc(size);
    for (int i = 0; i < n; i++) h_input[i] = 1.0f;

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float) * 1024));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Shared memory version
    CUDA_CHECK(cudaEventRecord(start));
    reduceShared<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float shared_time;
    CUDA_CHECK(cudaEventElapsedTime(&shared_time, start, stop));
    printf("Shared memory: %.3f ms\n", shared_time);

    // Shuffle version
    CUDA_CHECK(cudaEventRecord(start));
    reduceShfl<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float shfl_time;
    CUDA_CHECK(cudaEventElapsedTime(&shfl_time, start, stop));
    printf("Warp shuffle:  %.3f ms (%.2fx speedup)\n", shfl_time, shared_time/shfl_time);

    free(h_input);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
