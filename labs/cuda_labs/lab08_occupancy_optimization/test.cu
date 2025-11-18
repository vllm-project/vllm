/**
 * Lab 08: Occupancy Optimization - Test Suite
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { cudaError_t err = call; if (err != cudaSuccess) { \
    fprintf(stderr, "Error: %s\n", cudaGetErrorString(err)); exit(1); } } while(0)

__launch_bounds__(256, 4)
__global__ void testKernel(const float *input, float *output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = input[i] * 2.0f;
    }
}

int main() {
    printf("Occupancy Optimization Test Suite\n\n");

    int n = 1 << 20;
    float *h_in = (float*)malloc(n * sizeof(float));
    float *h_out = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) h_in[i] = (float)i;

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice));

    // Test occupancy API
    int maxActiveBlocks;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks, testKernel, 256, 0));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    float occupancy = (maxActiveBlocks * 256) / (float)prop.maxThreadsPerMultiProcessor;

    printf("Test: Occupancy API\n");
    printf("  Max active blocks: %d\n", maxActiveBlocks);
    printf("  Theoretical occupancy: %.1f%%\n", occupancy * 100.0f);
    printf("  Status: %s\n\n", (occupancy > 0.5f) ? "PASS" : "FAIL");

    // Test kernel execution
    testKernel<<<(n + 255) / 256, 256>>>(d_in, d_out, n);
    CUDA_CHECK(cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < n && ok; i++) {
        if (h_out[i] != h_in[i] * 2.0f) ok = false;
    }

    printf("Test: Kernel Execution\n");
    printf("  Status: %s\n", ok ? "PASS" : "FAIL");

    free(h_in); free(h_out);
    CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_out));

    return (occupancy > 0.5f && ok) ? 0 : 1;
}
