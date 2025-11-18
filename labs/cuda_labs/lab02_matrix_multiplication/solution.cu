/**
 * Lab 02: Matrix Multiplication - Complete Solution
 *
 * Three progressive implementations demonstrating optimization techniques
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define TILE_SIZE 16

// ============================================================================
// Naive Implementation - Global Memory Only
// ============================================================================
__global__ void matrixMulNaive(const float *A, const float *B, float *C,
                                int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ============================================================================
// Tiled Implementation - Shared Memory Optimization
// ============================================================================
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
        // Load tile of A
        int aCol = t * TILE_SIZE + tx;
        if (row < M && aCol < K) {
            As[ty][tx] = A[row * K + aCol];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load tile of B
        int bRow = t * TILE_SIZE + ty;
        if (bRow < K && col < N) {
            Bs[ty][tx] = B[bRow * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ============================================================================
// Optimized Implementation - Register Tiling
// ============================================================================
#define BLOCK_SIZE 16
#define REG_TILE 4

__global__ void matrixMulOptimized(const float *A, const float *B, float *C,
                                    int M, int K, int N) {
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x * (BLOCK_SIZE * REG_TILE);
    int by = blockIdx.y * (BLOCK_SIZE * REG_TILE);

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    float c[REG_TILE][REG_TILE] = {0.0f};

    int numTiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // Collaborative loading with register tiling
        for (int i = 0; i < REG_TILE; i++) {
            int row = by + ty + i * BLOCK_SIZE;
            int col = t * BLOCK_SIZE + tx;
            As[ty + i * BLOCK_SIZE / REG_TILE][tx] =
                (row < M && col < K) ? A[row * K + col] : 0.0f;
        }

        for (int i = 0; i < REG_TILE; i++) {
            int row = t * BLOCK_SIZE + ty;
            int col = bx + tx + i * BLOCK_SIZE;
            Bs[ty][tx + i * BLOCK_SIZE / REG_TILE] =
                (row < K && col < N) ? B[row * N + col] : 0.0f;
        }

        __syncthreads();

        // Compute using registers
        for (int k = 0; k < BLOCK_SIZE; k++) {
            float a[REG_TILE], b[REG_TILE];
            for (int i = 0; i < REG_TILE; i++) {
                a[i] = As[ty + i * BLOCK_SIZE / REG_TILE][k];
                b[i] = Bs[k][tx + i * BLOCK_SIZE / REG_TILE];
            }
            for (int i = 0; i < REG_TILE; i++) {
                for (int j = 0; j < REG_TILE; j++) {
                    c[i][j] += a[i] * b[j];
                }
            }
        }

        __syncthreads();
    }

    // Write results
    for (int i = 0; i < REG_TILE; i++) {
        for (int j = 0; j < REG_TILE; j++) {
            int row = by + ty + i * BLOCK_SIZE / REG_TILE;
            int col = bx + tx + j * BLOCK_SIZE / REG_TILE;
            if (row < M && col < N) {
                C[row * N + col] = c[i][j];
            }
        }
    }
}

// CPU reference
void matrixMulCPU(const float *A, const float *B, float *C,
                   int M, int K, int N) {
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

void initMatrix(float *mat, int rows, int cols, int seed) {
    srand(seed);
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)(rand() % 100) / 10.0f;
    }
}

bool verifyResults(const float *expected, const float *actual,
                   int rows, int cols, float epsilon = 1e-2) {
    int errors = 0;
    for (int i = 0; i < rows * cols; i++) {
        if (fabs(expected[i] - actual[i]) > epsilon) {
            if (errors < 10) {
                fprintf(stderr, "Error at %d: expected=%f, got=%f\n",
                        i, expected[i], actual[i]);
            }
            errors++;
        }
    }
    return errors == 0;
}

int main(int argc, char **argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 1024;
    int K = (argc > 2) ? atoi(argv[2]) : 1024;
    int N = (argc > 3) ? atoi(argv[3]) : 1024;

    printf("Matrix Multiplication: C[%d][%d] = A[%d][%d] × B[%d][%d]\n",
           M, N, M, K, K, N);

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);
    float *h_C_ref = (float*)malloc(size_C);

    initMatrix(h_A, M, K, 42);
    initMatrix(h_B, K, N, 43);

    if (M * N * K < 100000000) {
        printf("\nCPU computation...\n");
        clock_t start = clock();
        matrixMulCPU(h_A, h_B, h_C_ref, M, K, N);
        double cpu_time = ((double)(clock() - start)) / CLOCKS_PER_SEC * 1000.0;
        printf("CPU: %.3f ms\n", cpu_time);
    }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                 (M + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Naive
    printf("\n--- Naive ---\n");
    CUDA_CHECK(cudaEventRecord(start));
    matrixMulNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float naive_time;
    CUDA_CHECK(cudaEventElapsedTime(&naive_time, start, stop));
    printf("Time: %.3f ms\n", naive_time);

    // Tiled
    printf("\n--- Tiled ---\n");
    CUDA_CHECK(cudaEventRecord(start));
    matrixMulTiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float tiled_time;
    CUDA_CHECK(cudaEventElapsedTime(&tiled_time, start, stop));
    printf("Time: %.3f ms (%.2fx speedup)\n", tiled_time, naive_time/tiled_time);

    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
    if (M * N * K < 100000000) {
        printf("Verification: %s\n", verifyResults(h_C_ref, h_C, M, N) ? "PASS" : "FAIL");
    }

    // Performance
    double gflops = (2.0 * M * N * K / 1e9) / (tiled_time / 1000.0);
    printf("\n=== Performance ===\n");
    printf("TFLOPS: %.3f\n", gflops / 1000.0);

    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
