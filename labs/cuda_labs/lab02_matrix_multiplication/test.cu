/**
 * Lab 02: Matrix Multiplication - Test Suite
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define TILE_SIZE 16

__global__ void matrixMulTiled(const float *A, const float *B, float *C,
                                int M, int K, int N) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_SIZE + tx;
        As[ty][tx] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;

        int bRow = t * TILE_SIZE + ty;
        Bs[ty][tx] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void matrixMulCPU(const float *A, const float *B, float *C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main() {
    printf("Matrix Multiplication Test Suite\n");

    typedef struct { int M, K, N; const char *name; } TestCase;
    TestCase tests[] = {
        {32, 32, 32, "Small square"},
        {64, 128, 256, "Rectangular"},
        {1024, 1024, 1024, "Medium square"},
        {1000, 1000, 1000, "Non-power-of-2"},
    };

    int passed = 0;
    for (int t = 0; t < 4; t++) {
        TestCase tc = tests[t];
        size_t size_A = tc.M * tc.K * sizeof(float);
        size_t size_B = tc.K * tc.N * sizeof(float);
        size_t size_C = tc.M * tc.N * sizeof(float);

        float *h_A = (float*)malloc(size_A);
        float *h_B = (float*)malloc(size_B);
        float *h_C = (float*)malloc(size_C);
        float *h_C_ref = (float*)malloc(size_C);

        for (int i = 0; i < tc.M * tc.K; i++) h_A[i] = (float)(rand() % 10);
        for (int i = 0; i < tc.K * tc.N; i++) h_B[i] = (float)(rand() % 10);

        matrixMulCPU(h_A, h_B, h_C_ref, tc.M, tc.K, tc.N);

        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, size_A));
        CUDA_CHECK(cudaMalloc(&d_B, size_B));
        CUDA_CHECK(cudaMalloc(&d_C, size_C));

        CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((tc.N + TILE_SIZE - 1) / TILE_SIZE,
                  (tc.M + TILE_SIZE - 1) / TILE_SIZE);

        matrixMulTiled<<<grid, block>>>(d_A, d_B, d_C, tc.M, tc.K, tc.N);
        CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < tc.M * tc.N && ok; i++) {
            if (fabs(h_C[i] - h_C_ref[i]) > 0.1f) ok = false;
        }

        printf("%s [%dx%dx%d]: %s\n", tc.name, tc.M, tc.K, tc.N, ok ? "PASS" : "FAIL");
        if (ok) passed++;

        free(h_A); free(h_B); free(h_C); free(h_C_ref);
        CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    }

    printf("\nResults: %d/4 tests passed\n", passed);
    return passed == 4 ? 0 : 1;
}
