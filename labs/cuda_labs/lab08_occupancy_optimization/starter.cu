/**
 * Lab 08: Occupancy Optimization - Starter Code
 * Compare kernels with different resource usage patterns
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

// TODO 1: Implement kernel with high register usage (low occupancy)
__global__ void kernelHighRegisters(const float *input, float *output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        // TODO: Use many local variables to increase register pressure
        // This will reduce occupancy but might improve per-thread performance
        // Try using arrays or multiple intermediate calculations

        float result = input[i];

        // Your code here (use many registers):


        output[i] = result;
    }
}

// TODO 2: Implement kernel with large shared memory (medium occupancy)
__global__ void kernelLargeSharedMem(const float *input, float *output, int n) {
    // Large shared memory limits blocks per SM
    __shared__ float sdata[4096];  // 16KB

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // TODO: Use shared memory for some computation
    // This reduces occupancy due to shared memory limits

    if (i < n) {
        sdata[tid] = input[i];
        __syncthreads();

        // Your code here:


        output[i] = sdata[tid];
    }
}

// TODO 3: Implement optimized kernel with launch bounds
// __launch_bounds__ guides compiler for target occupancy
__launch_bounds__(256, 4)  // 256 threads/block, min 4 blocks/SM
__global__ void kernelOptimized(const float *input, float *output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        // TODO: Implement balanced computation
        // Moderate register usage, minimal shared memory
        // Compiler will optimize for specified launch bounds

        float result = input[i];

        // Your code here:


        output[i] = result;
    }
}

// Helper function to compute theoretical occupancy
void printOccupancy(const char *kernelName, const void *kernel, int blockSize) {
    int maxActiveBlocks;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks, kernel, blockSize, 0));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    int maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;
    int maxBlocksPerSM = prop.maxBlocksPerMultiProcessor;

    float occupancy = (maxActiveBlocks * blockSize) / (float)maxThreadsPerSM;

    printf("%s:\n", kernelName);
    printf("  Block size: %d threads\n", blockSize);
    printf("  Max active blocks per SM: %d (limit: %d)\n",
           maxActiveBlocks, maxBlocksPerSM);
    printf("  Theoretical occupancy: %.1f%%\n", occupancy * 100.0f);
    printf("  Active warps per SM: %d (max: %d)\n",
           (maxActiveBlocks * blockSize) / 32,
           maxThreadsPerSM / 32);
}

int main() {
    int n = 1 << 24;  // 16M elements
    size_t size = n * sizeof(float);

    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc(size);
    for (int i = 0; i < n; i++) {
        h_input[i] = (float)(i % 100);
    }

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    printf("========================================\n");
    printf("Occupancy Optimization Analysis\n");
    printf("========================================\n\n");

    // Test different block sizes
    int blockSizes[] = {128, 256, 512};

    for (int bs = 0; bs < 3; bs++) {
        int blockSize = blockSizes[bs];
        int gridSize = (n + blockSize - 1) / blockSize;

        printf("--- Block Size: %d ---\n\n", blockSize);

        // Analyze occupancy
        printOccupancy("High Registers", (void*)kernelHighRegisters, blockSize);
        printOccupancy("Large Shared Mem", (void*)kernelLargeSharedMem, blockSize);
        printOccupancy("Optimized", (void*)kernelOptimized, blockSize);

        printf("\nPerformance:\n");

        // High registers kernel
        CUDA_CHECK(cudaEventRecord(start));
        kernelHighRegisters<<<gridSize, blockSize>>>(d_input, d_output, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float time1;
        CUDA_CHECK(cudaEventElapsedTime(&time1, start, stop));
        printf("  High Registers:   %.3f ms\n", time1);

        // Large shared memory kernel
        CUDA_CHECK(cudaEventRecord(start));
        kernelLargeSharedMem<<<gridSize, blockSize>>>(d_input, d_output, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float time2;
        CUDA_CHECK(cudaEventElapsedTime(&time2, start, stop));
        printf("  Large Shared Mem: %.3f ms\n", time2);

        // Optimized kernel
        CUDA_CHECK(cudaEventRecord(start));
        kernelOptimized<<<gridSize, blockSize>>>(d_input, d_output, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float time3;
        CUDA_CHECK(cudaEventElapsedTime(&time3, start, stop));
        printf("  Optimized:        %.3f ms\n", time3);

        printf("\n");
    }

    printf("========================================\n");
    printf("Key Insights:\n");
    printf("- Higher occupancy enables better latency hiding\n");
    printf("- But occupancy alone doesn't guarantee performance\n");
    printf("- Balance resource usage with parallelism\n");
    printf("- Use profiling tools to measure achieved occupancy\n");
    printf("========================================\n");

    free(h_input);
    free(h_output);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
