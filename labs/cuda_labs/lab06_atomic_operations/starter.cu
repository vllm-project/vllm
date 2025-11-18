/**
 * Lab 06: Atomic Operations - Starter Code
 * Histogram computation
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

#define NUM_BINS 256
#define BLOCK_SIZE 256

// TODO 1: Implement naive histogram using global atomics
__global__ void histogramNaive(const unsigned char *input, unsigned int *bins, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        unsigned char value = input[i];
        // TODO: Use atomicAdd to increment bins[value]

        // Your code here:

    }
}

// TODO 2: Implement optimized histogram with shared memory privatization
__global__ void histogramOptimized(const unsigned char *input, unsigned int *bins, int n) {
    // Create private histogram in shared memory
    __shared__ unsigned int s_bins[NUM_BINS];

    // TODO: Initialize shared memory bins to zero
    // Hint: Each thread initializes multiple bins

    // Your code here:


    __syncthreads();

    // TODO: Compute histogram in shared memory
    // Each thread processes one element using shared memory atomics

    // Your code here:


    __syncthreads();

    // TODO: Merge shared memory bins into global memory
    // Each thread handles multiple bins

    // Your code here:

}

void histogramCPU(const unsigned char *input, unsigned int *bins, int n) {
    for (int i = 0; i < NUM_BINS; i++) bins[i] = 0;
    for (int i = 0; i < n; i++) {
        bins[input[i]]++;
    }
}

int main() {
    int n = 1 << 24;  // 16M elements
    size_t size = n * sizeof(unsigned char);

    unsigned char *h_input = (unsigned char*)malloc(size);
    unsigned int *h_bins = (unsigned int*)malloc(NUM_BINS * sizeof(unsigned int));
    unsigned int *h_ref = (unsigned int*)malloc(NUM_BINS * sizeof(unsigned int));

    // Initialize with random values
    for (int i = 0; i < n; i++) {
        h_input[i] = (unsigned char)(i % NUM_BINS);
    }

    histogramCPU(h_input, h_ref, n);

    unsigned char *d_input;
    unsigned int *d_bins;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_bins, NUM_BINS * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Naive
    CUDA_CHECK(cudaMemset(d_bins, 0, NUM_BINS * sizeof(unsigned int)));
    CUDA_CHECK(cudaEventRecord(start));
    histogramNaive<<<gridSize, BLOCK_SIZE>>>(d_input, d_bins, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float naive_time;
    CUDA_CHECK(cudaEventElapsedTime(&naive_time, start, stop));
    printf("Naive: %.3f ms\n", naive_time);

    // Optimized
    CUDA_CHECK(cudaMemset(d_bins, 0, NUM_BINS * sizeof(unsigned int)));
    CUDA_CHECK(cudaEventRecord(start));
    histogramOptimized<<<gridSize, BLOCK_SIZE>>>(d_input, d_bins, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float opt_time;
    CUDA_CHECK(cudaEventElapsedTime(&opt_time, start, stop));
    printf("Optimized: %.3f ms (%.2fx speedup)\n", opt_time, naive_time/opt_time);

    CUDA_CHECK(cudaMemcpy(h_bins, d_bins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    // Verify
    bool ok = true;
    for (int i = 0; i < NUM_BINS && ok; i++) {
        if (h_bins[i] != h_ref[i]) ok = false;
    }
    printf("Verification: %s\n", ok ? "PASS" : "FAIL");

    free(h_input); free(h_bins); free(h_ref);
    CUDA_CHECK(cudaFree(d_input)); CUDA_CHECK(cudaFree(d_bins));

    return 0;
}
