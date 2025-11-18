/**
 * Lab 08: Occupancy Optimization - Complete Solution
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

// High register usage - reduces occupancy
__global__ void kernelHighRegisters(const float *input, float *output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        // Use many local variables to increase register pressure
        float r0 = input[i];
        float r1 = sinf(r0);
        float r2 = cosf(r0);
        float r3 = expf(r0 * 0.01f);
        float r4 = logf(r0 + 1.0f);
        float r5 = sqrtf(r0);
        float r6 = r1 * r2;
        float r7 = r3 + r4;
        float r8 = r5 - r6;
        float result = r7 + r8;

        output[i] = result;
    }
}

// Large shared memory - reduces occupancy
__global__ void kernelLargeSharedMem(const float *input, float *output, int n) {
    __shared__ float sdata[4096];  // 16KB
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (i < n && tid < 4096) {
        sdata[tid] = input[i];
    }
    __syncthreads();

    if (i < n && tid < 4096) {
        float val = sdata[tid];
        val = sinf(val) + cosf(val);
        output[i] = val;
    }
}

// Optimized with launch bounds
__launch_bounds__(256, 4)
__global__ void kernelOptimized(const float *input, float *output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        float val = input[i];
        val = sinf(val) + cosf(val);
        output[i] = val;
    }
}

void printOccupancy(const char *kernelName, const void *kernel, int blockSize) {
    int maxActiveBlocks;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks, kernel, blockSize, 0));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    int maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;
    float occupancy = (maxActiveBlocks * blockSize) / (float)maxThreadsPerSM;

    printf("%s:\n", kernelName);
    printf("  Block size: %d threads\n", blockSize);
    printf("  Max active blocks per SM: %d\n", maxActiveBlocks);
    printf("  Theoretical occupancy: %.1f%%\n", occupancy * 100.0f);
    printf("  Active warps per SM: %d (max: %d)\n",
           (maxActiveBlocks * blockSize) / 32, maxThreadsPerSM / 32);
}

int main() {
    int n = 1 << 24;
    size_t size = n * sizeof(float);

    float *h_input = (float*)malloc(size);
    for (int i = 0; i < n; i++) h_input[i] = (float)(i % 100);

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    printf("Occupancy Optimization Analysis\n\n");

    int blockSizes[] = {128, 256, 512};

    for (int bs = 0; bs < 3; bs++) {
        int blockSize = blockSizes[bs];
        int gridSize = (n + blockSize - 1) / blockSize;

        printf("--- Block Size: %d ---\n\n", blockSize);

        printOccupancy("High Registers", (void*)kernelHighRegisters, blockSize);
        printOccupancy("Large Shared Mem", (void*)kernelLargeSharedMem, blockSize);
        printOccupancy("Optimized", (void*)kernelOptimized, blockSize);

        printf("\nPerformance:\n");

        CUDA_CHECK(cudaEventRecord(start));
        kernelHighRegisters<<<gridSize, blockSize>>>(d_input, d_output, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float time1;
        CUDA_CHECK(cudaEventElapsedTime(&time1, start, stop));
        printf("  High Registers:   %.3f ms\n", time1);

        CUDA_CHECK(cudaEventRecord(start));
        kernelLargeSharedMem<<<gridSize, blockSize>>>(d_input, d_output, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float time2;
        CUDA_CHECK(cudaEventElapsedTime(&time2, start, stop));
        printf("  Large Shared Mem: %.3f ms\n", time2);

        CUDA_CHECK(cudaEventRecord(start));
        kernelOptimized<<<gridSize, blockSize>>>(d_input, d_output, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float time3;
        CUDA_CHECK(cudaEventElapsedTime(&time3, start, stop));
        printf("  Optimized:        %.3f ms\n\n", time3);
    }

    free(h_input);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
