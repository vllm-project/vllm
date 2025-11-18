/**
 * Lab 05: Shared Memory - Complete Solution
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

#define RADIUS 3
#define BLOCK_SIZE 256

__global__ void stencilNaive(const float *input, float *output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        float sum = 0.0f;
        for (int offset = -RADIUS; offset <= RADIUS; offset++) {
            int idx = i + offset;
            if (idx >= 0 && idx < n) {
                sum += input[idx];
            }
        }
        output[i] = sum;
    }
}

__global__ void stencilShared(const float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE + 2 * RADIUS];

    int gindex = blockIdx.x * blockDim.x + threadIdx.x;
    int lindex = threadIdx.x + RADIUS;

    // Load main data
    if (gindex < n) {
        sdata[lindex] = input[gindex];
    }

    // Load halo elements
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

void stencilCPU(const float *input, float *output, int n) {
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int offset = -RADIUS; offset <= RADIUS; offset++) {
            int idx = i + offset;
            if (idx >= 0 && idx < n) {
                sum += input[idx];
            }
        }
        output[i] = sum;
    }
}

int main() {
    int n = 1 << 24;
    size_t size = n * sizeof(float);

    float *h_in = (float*)malloc(size);
    float *h_out = (float*)malloc(size);
    for (int i = 0; i < n; i++) h_in[i] = (float)(i % 100);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, size));
    CUDA_CHECK(cudaMalloc(&d_out, size));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CUDA_CHECK(cudaEventRecord(start));
    stencilNaive<<<gridSize, BLOCK_SIZE>>>(d_in, d_out, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float naive_time;
    CUDA_CHECK(cudaEventElapsedTime(&naive_time, start, stop));
    printf("Naive: %.3f ms\n", naive_time);

    CUDA_CHECK(cudaEventRecord(start));
    stencilShared<<<gridSize, BLOCK_SIZE>>>(d_in, d_out, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float shared_time;
    CUDA_CHECK(cudaEventElapsedTime(&shared_time, start, stop));
    printf("Shared: %.3f ms (%.2fx speedup)\n", shared_time, naive_time/shared_time);

    free(h_in); free(h_out);
    CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_out));

    return 0;
}
