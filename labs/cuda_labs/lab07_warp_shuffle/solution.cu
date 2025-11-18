/**
 * Lab 07: Warp Shuffle Operations - Complete Solution
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

__device__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void reduceShfl(const float *input, float *output, int n) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float val = (i < n) ? input[i] : 0.0f;

    // Reduce within warp
    val = warpReduceSum(val);

    // First thread in each warp writes to shared memory
    __shared__ float warp_sums[BLOCK_SIZE / WARP_SIZE];
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    if (lane_id == 0) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();

    // Final reduction by first warp
    if (tid < BLOCK_SIZE / WARP_SIZE) {
        val = warp_sums[tid];
    } else {
        val = 0.0f;
    }

    if (warp_id == 0) {
        val = warpReduceSum(val);
        if (lane_id == 0) {
            output[blockIdx.x] = val;
        }
    }
}

__device__ float warpScan(float val) {
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        float temp = __shfl_up_sync(0xffffffff, val, offset);
        if (threadIdx.x % WARP_SIZE >= offset) {
            val += temp;
        }
    }
    return val;
}

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
    int n = 1 << 24;
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

    CUDA_CHECK(cudaEventRecord(start));
    reduceShared<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float shared_time;
    CUDA_CHECK(cudaEventElapsedTime(&shared_time, start, stop));
    printf("Shared memory: %.3f ms\n", shared_time);

    CUDA_CHECK(cudaEventRecord(start));
    reduceShfl<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float shfl_time;
    CUDA_CHECK(cudaEventElapsedTime(&shfl_time, start, stop));
    printf("Warp shuffle: %.3f ms (%.2fx speedup)\n", shfl_time, shared_time/shfl_time);

    free(h_input);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
