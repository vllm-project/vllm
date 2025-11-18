/**
 * Lab 06: Atomic Operations - Complete Solution
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

__global__ void histogramNaive(const unsigned char *input, unsigned int *bins, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        unsigned char value = input[i];
        atomicAdd(&bins[value], 1);
    }
}

__global__ void histogramOptimized(const unsigned char *input, unsigned int *bins, int n) {
    __shared__ unsigned int s_bins[NUM_BINS];

    // Initialize shared memory
    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        s_bins[i] = 0;
    }
    __syncthreads();

    // Compute histogram in shared memory
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned char value = input[i];
        atomicAdd(&s_bins[value], 1);
    }
    __syncthreads();

    // Merge to global memory
    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        if (s_bins[i] > 0) {
            atomicAdd(&bins[i], s_bins[i]);
        }
    }
}

void histogramCPU(const unsigned char *input, unsigned int *bins, int n) {
    for (int i = 0; i < NUM_BINS; i++) bins[i] = 0;
    for (int i = 0; i < n; i++) {
        bins[input[i]]++;
    }
}

int main() {
    int n = 1 << 24;
    size_t size = n * sizeof(unsigned char);

    unsigned char *h_input = (unsigned char*)malloc(size);
    unsigned int *h_bins = (unsigned int*)malloc(NUM_BINS * sizeof(unsigned int));

    for (int i = 0; i < n; i++) {
        h_input[i] = (unsigned char)(i % NUM_BINS);
    }

    unsigned char *d_input;
    unsigned int *d_bins;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_bins, NUM_BINS * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CUDA_CHECK(cudaMemset(d_bins, 0, NUM_BINS * sizeof(unsigned int)));
    CUDA_CHECK(cudaEventRecord(start));
    histogramNaive<<<gridSize, BLOCK_SIZE>>>(d_input, d_bins, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float naive_time;
    CUDA_CHECK(cudaEventElapsedTime(&naive_time, start, stop));
    printf("Naive: %.3f ms\n", naive_time);

    CUDA_CHECK(cudaMemset(d_bins, 0, NUM_BINS * sizeof(unsigned int)));
    CUDA_CHECK(cudaEventRecord(start));
    histogramOptimized<<<gridSize, BLOCK_SIZE>>>(d_input, d_bins, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float opt_time;
    CUDA_CHECK(cudaEventElapsedTime(&opt_time, start, stop));
    printf("Optimized: %.3f ms (%.2fx speedup)\n", opt_time, naive_time/opt_time);

    free(h_input); free(h_bins);
    CUDA_CHECK(cudaFree(d_input)); CUDA_CHECK(cudaFree(d_bins));

    return 0;
}
