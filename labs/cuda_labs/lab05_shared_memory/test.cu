/**
 * Lab 05: Shared Memory - Test Suite
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { cudaError_t err = call; if (err != cudaSuccess) { \
    fprintf(stderr, "Error: %s\n", cudaGetErrorString(err)); exit(1); } } while(0)

#define RADIUS 3
#define BLOCK_SIZE 256

__global__ void stencilShared(const float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE + 2 * RADIUS];
    int gindex = blockIdx.x * blockDim.x + threadIdx.x;
    int lindex = threadIdx.x + RADIUS;

    if (gindex < n) sdata[lindex] = input[gindex];
    if (threadIdx.x < RADIUS) {
        sdata[lindex - RADIUS] = (gindex >= RADIUS) ? input[gindex - RADIUS] : 0.0f;
        sdata[lindex + BLOCK_SIZE] = (gindex + BLOCK_SIZE < n) ? input[gindex + BLOCK_SIZE] : 0.0f;
    }
    __syncthreads();

    if (gindex < n) {
        float sum = 0.0f;
        for (int offset = -RADIUS; offset <= RADIUS; offset++) {
            sum += sdata[lindex + offset];
        }
        output[gindex] = sum;
    }
}

int main() {
    printf("Stencil Test Suite\n");
    int sizes[] = {1024, 65536, 1<<20};
    int passed = 0;

    for (int t = 0; t < 3; t++) {
        int n = sizes[t];
        float *h_in = (float*)malloc(n * sizeof(float));
        float *h_out = (float*)malloc(n * sizeof(float));
        for (int i = 0; i < n; i++) h_in[i] = 1.0f;

        float *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice));

        stencilShared<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_in, d_out, n);
        CUDA_CHECK(cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < n && ok; i++) {
            int expected = (RADIUS * 2 + 1);
            if (i < RADIUS || i >= n - RADIUS) expected = (RADIUS + 1 + (i < RADIUS ? i : n - 1 - i));
            if (fabs(h_out[i] - expected) > 0.01f) ok = false;
        }

        printf("Size %d: %s\n", n, ok ? "PASS" : "FAIL");
        if (ok) passed++;

        free(h_in); free(h_out);
        CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_out));
    }

    printf("\nResults: %d/3 tests passed\n", passed);
    return passed == 3 ? 0 : 1;
}
