/**
 * Lab 06: Atomic Operations - Test Suite
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { cudaError_t err = call; if (err != cudaSuccess) { \
    fprintf(stderr, "Error: %s\n", cudaGetErrorString(err)); exit(1); } } while(0)

#define NUM_BINS 256
#define BLOCK_SIZE 256

__global__ void histogramOptimized(const unsigned char *input, unsigned int *bins, int n) {
    __shared__ unsigned int s_bins[NUM_BINS];

    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        s_bins[i] = 0;
    }
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(&s_bins[input[i]], 1);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        if (s_bins[i] > 0) {
            atomicAdd(&bins[i], s_bins[i]);
        }
    }
}

int main() {
    printf("Histogram Test Suite\n");

    int sizes[] = {1024, 65536, 1<<20};
    int passed = 0;

    for (int t = 0; t < 3; t++) {
        int n = sizes[t];
        unsigned char *h_in = (unsigned char*)malloc(n);
        unsigned int *h_bins = (unsigned int*)malloc(NUM_BINS * sizeof(unsigned int));
        for (int i = 0; i < n; i++) h_in[i] = (unsigned char)(i % NUM_BINS);

        unsigned char *d_in;
        unsigned int *d_bins;
        CUDA_CHECK(cudaMalloc(&d_in, n));
        CUDA_CHECK(cudaMalloc(&d_bins, NUM_BINS * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in, n, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_bins, 0, NUM_BINS * sizeof(unsigned int)));

        histogramOptimized<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_in, d_bins, n);
        CUDA_CHECK(cudaMemcpy(h_bins, d_bins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));

        bool ok = true;
        int expected = n / NUM_BINS;
        for (int i = 0; i < NUM_BINS && ok; i++) {
            if (h_bins[i] != expected) ok = false;
        }

        printf("Size %d: %s\n", n, ok ? "PASS" : "FAIL");
        if (ok) passed++;

        free(h_in); free(h_bins);
        CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_bins));
    }

    printf("\nResults: %d/3 tests passed\n", passed);
    return passed == 3 ? 0 : 1;
}
