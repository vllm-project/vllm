/**
 * Lab 04: Memory Coalescing - Starter Code
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

#define TILE_SIZE 32

// TODO 1: Implement naive transpose (non-coalesced writes)
__global__ void transposeNaive(const float *input, float *output, int N) {
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    if (x < N && y < N) {
        // TODO: Read from input[y][x], write to output[x][y]
        // Problem: Column-wise writes are non-coalesced

        // Your code here:

    }
}

// TODO 2: Implement coalesced transpose using shared memory
__global__ void transposeCoalesced(const float *input, float *output, int N) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE];

    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    // TODO: Load tile from global memory (coalesced)
    // Store to shared memory
    // Sync
    // Read from shared memory (transposed)
    // Write to global memory (coalesced)

    // Your code here:

}

// TODO 3: Implement optimized transpose with padding
__global__ void transposeOptimized(const float *input, float *output, int N) {
    // Add +1 padding to avoid bank conflicts
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];

    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    // TODO: Same as coalesced but with padded shared memory

    // Your code here:

}

void transposeCPU(const float *input, float *output, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            output[j * N + i] = input[i * N + j];
        }
    }
}

int main() {
    int N = 4096;
    size_t size = N * N * sizeof(float);

    float *h_in = (float*)malloc(size);
    float *h_out = (float*)malloc(size);
    for (int i = 0; i < N * N; i++) h_in[i] = (float)i;

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, size));
    CUDA_CHECK(cudaMalloc(&d_out, size));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Test naive
    CUDA_CHECK(cudaEventRecord(start));
    transposeNaive<<<grid, block>>>(d_in, d_out, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float naive_time;
    CUDA_CHECK(cudaEventElapsedTime(&naive_time, start, stop));
    printf("Naive: %.3f ms\n", naive_time);

    // Test coalesced
    CUDA_CHECK(cudaEventRecord(start));
    transposeCoalesced<<<grid, block>>>(d_in, d_out, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float coal_time;
    CUDA_CHECK(cudaEventElapsedTime(&coal_time, start, stop));
    printf("Coalesced: %.3f ms (%.2fx)\n", coal_time, naive_time/coal_time);

    // Test optimized
    CUDA_CHECK(cudaEventRecord(start));
    transposeOptimized<<<grid, block>>>(d_in, d_out, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float opt_time;
    CUDA_CHECK(cudaEventElapsedTime(&opt_time, start, stop));
    printf("Optimized: %.3f ms (%.2fx)\n", opt_time, naive_time/opt_time);

    free(h_in); free(h_out);
    CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_out));

    return 0;
}
