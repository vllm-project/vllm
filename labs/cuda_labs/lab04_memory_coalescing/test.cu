/**
 * Lab 04: Memory Coalescing - Test Suite
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

__global__ void transposeOptimized(const float *input, float *output, int N) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    if (x < N && y < N) {
        tile[threadIdx.y][threadIdx.x] = input[y * N + x];
    }
    __syncthreads();

    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;

    if (x < N && y < N) {
        output[y * N + x] = tile[threadIdx.x][threadIdx.y];
    }
}

int main() {
    printf("Transpose Test Suite\n");

    int sizes[] = {32, 128, 1024, 2048};
    int passed = 0;

    for (int t = 0; t < 4; t++) {
        int N = sizes[t];
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

        transposeOptimized<<<grid, block>>>(d_in, d_out, N);
        CUDA_CHECK(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N && ok; i++) {
            for (int j = 0; j < N && ok; j++) {
                if (h_out[i * N + j] != h_in[j * N + i]) ok = false;
            }
        }

        printf("Size %dx%d: %s\n", N, N, ok ? "PASS" : "FAIL");
        if (ok) passed++;

        free(h_in); free(h_out);
        CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_out));
    }

    printf("\nResults: %d/4 tests passed\n", passed);
    return passed == 4 ? 0 : 1;
}
