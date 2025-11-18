/**
 * Lab 02: Matrix Multiplication - Starter Code
 *
 * Three progressive implementations:
 * 1. Naive - Global memory only
 * 2. Tiled - Shared memory optimization
 * 3. Optimized - Advanced optimizations
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define TILE_SIZE 16  // Tile dimension for shared memory

// ============================================================================
// TODO 1-3: Implement Naive Matrix Multiplication
// ============================================================================
/**
 * Naive matrix multiplication kernel
 * C[M][N] = A[M][K] × B[K][N]
 *
 * Each thread computes one element of C
 * All accesses go to global memory (slow)
 *
 * Hints:
 * - Use 2D grid and block dimensions
 * - Calculate row and col from thread indices
 * - Compute dot product of A[row][:] and B[:][col]
 * - Add boundary checks
 */
__global__ void matrixMulNaive(const float *A, const float *B, float *C,
                                int M, int K, int N) {
    // TODO 1: Calculate row and column indices
    // row = blockIdx.y * blockDim.y + threadIdx.y
    // col = blockIdx.x * blockDim.x + threadIdx.x
    int row = 0;  // Replace
    int col = 0;  // Replace

    // TODO 2: Add boundary check
    // if (row < M && col < N) {

        // TODO 3: Compute dot product
        // Initialize sum = 0
        // Loop k from 0 to K-1:
        //   sum += A[row * K + k] * B[k * N + col]
        // C[row * N + col] = sum

        // Your code here:


    // }
}

// ============================================================================
// TODO 4-8: Implement Tiled Matrix Multiplication with Shared Memory
// ============================================================================
/**
 * Tiled matrix multiplication using shared memory
 *
 * Strategy:
 * 1. Divide matrices into TILE_SIZE × TILE_SIZE tiles
 * 2. Load one tile of A and B into shared memory collaboratively
 * 3. Compute partial dot product using shared data
 * 4. Repeat for all tiles, accumulating results
 *
 * Benefits:
 * - Each element loaded from global memory is reused TILE_SIZE times
 * - Significantly reduces global memory bandwidth
 */
__global__ void matrixMulTiled(const float *A, const float *B, float *C,
                                int M, int K, int N) {
    // TODO 4: Calculate thread indices
    int row = 0;  // Global row index
    int col = 0;  // Global column index
    int tx = 0;   // Local thread x index
    int ty = 0;   // Local thread y index

    // Shared memory for tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Accumulator for output element
    float sum = 0.0f;

    // TODO 5: Calculate number of tiles
    int numTiles = 0;  // (K + TILE_SIZE - 1) / TILE_SIZE

    // TODO 6-8: Loop over tiles
    for (int t = 0; t < numTiles; t++) {
        // TODO 6: Load tile of A into shared memory
        // Calculate global indices for this tile
        // Each thread loads one element
        // Handle boundary conditions (may need to pad with zeros)

        // Your code here:


        // TODO 7: Load tile of B into shared memory
        // Similar to A, but different indexing

        // Your code here:


        // TODO: Synchronize to ensure tile is fully loaded
        // __syncthreads();


        // TODO 8: Compute partial dot product using shared memory
        // for (int k = 0; k < TILE_SIZE; k++) {
        //     sum += As[ty][k] * Bs[k][tx];
        // }

        // Your code here:


        // TODO: Synchronize before loading next tile
        // __syncthreads();
    }

    // TODO: Write result to global memory (with boundary check)
    // if (row < M && col < N) {
    //     C[row * N + col] = sum;
    // }

    // Your code here:

}

// ============================================================================
// TODO 9-10: Implement Optimized Version (Optional Challenge)
// ============================================================================
/**
 * Optimized matrix multiplication
 *
 * Advanced optimizations:
 * - Each thread computes multiple output elements (register tiling)
 * - Vectorized memory access where possible
 * - Better instruction-level parallelism
 */
__global__ void matrixMulOptimized(const float *A, const float *B, float *C,
                                    int M, int K, int N) {
    // This is an advanced challenge - implement if you complete the tiled version
    // Hints:
    // - Each thread computes a small block of output (e.g., 4x4 or 8x8)
    // - Use registers to hold partial results
    // - Load data more efficiently

    // Your code here (optional):

}

// ============================================================================
// CPU Reference Implementation
// ============================================================================
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

// ============================================================================
// Utility Functions
// ============================================================================
void initMatrix(float *mat, int rows, int cols, int seed) {
    srand(seed);
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)(rand() % 100) / 10.0f;
    }
}

bool verifyResults(const float *expected, const float *actual,
                   int rows, int cols, float epsilon = 1e-2) {
    for (int i = 0; i < rows * cols; i++) {
        if (fabs(expected[i] - actual[i]) > epsilon) {
            int row = i / cols;
            int col = i % cols;
            fprintf(stderr, "Mismatch at [%d][%d]: CPU=%f, GPU=%f, diff=%f\n",
                    row, col, expected[i], actual[i],
                    fabs(expected[i] - actual[i]));
            return false;
        }
    }
    return true;
}

// ============================================================================
// Main Function
// ============================================================================
int main(int argc, char **argv) {
    // Parse matrix dimensions
    int M = (argc > 1) ? atoi(argv[1]) : 1024;
    int K = (argc > 2) ? atoi(argv[2]) : 1024;
    int N = (argc > 3) ? atoi(argv[3]) : 1024;

    printf("Matrix Multiplication: C[%d][%d] = A[%d][%d] × B[%d][%d]\n",
           M, N, M, K, K, N);

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    printf("Memory: A=%.2fMB, B=%.2fMB, C=%.2fMB\n",
           size_A/1e6, size_B/1e6, size_C/1e6);

    // Allocate host memory
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);
    float *h_C_ref = (float*)malloc(size_C);

    // Initialize matrices
    initMatrix(h_A, M, K, 42);
    initMatrix(h_B, K, N, 43);

    // CPU computation (for small matrices)
    if (M * N * K < 100000000) {  // Skip for very large matrices
        printf("\nRunning CPU version...\n");
        clock_t start = clock();
        matrixMulCPU(h_A, h_B, h_C_ref, M, K, N);
        clock_t end = clock();
        double cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
        printf("CPU Time: %.3f ms\n", cpu_time);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    // Configure kernel launch
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                 (M + TILE_SIZE - 1) / TILE_SIZE);

    printf("\nGrid: (%d, %d), Block: (%d, %d)\n",
           gridDim.x, gridDim.y, blockDim.x, blockDim.y);

    // Create events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Test naive kernel
    printf("\n--- Testing Naive Kernel ---\n");
    CUDA_CHECK(cudaEventRecord(start));
    matrixMulNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float naive_time;
    CUDA_CHECK(cudaEventElapsedTime(&naive_time, start, stop));
    printf("Naive Time: %.3f ms\n", naive_time);

    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

    if (M * N * K < 100000000) {
        if (verifyResults(h_C_ref, h_C, M, N)) {
            printf("✓ Verification PASSED\n");
        } else {
            printf("✗ Verification FAILED\n");
        }
    }

    // Test tiled kernel
    printf("\n--- Testing Tiled Kernel ---\n");
    CUDA_CHECK(cudaMemset(d_C, 0, size_C));  // Clear output
    CUDA_CHECK(cudaEventRecord(start));
    matrixMulTiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float tiled_time;
    CUDA_CHECK(cudaEventElapsedTime(&tiled_time, start, stop));
    printf("Tiled Time: %.3f ms\n", tiled_time);

    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

    if (M * N * K < 100000000) {
        if (verifyResults(h_C_ref, h_C, M, N)) {
            printf("✓ Verification PASSED\n");
        } else {
            printf("✗ Verification FAILED\n");
        }
    }

    // Performance summary
    printf("\n=== Performance Summary ===\n");
    printf("Naive:  %.3f ms\n", naive_time);
    printf("Tiled:  %.3f ms (%.2fx speedup)\n", tiled_time, naive_time/tiled_time);

    double gflops = (2.0 * M * N * K / 1e9) / (tiled_time / 1000.0);
    printf("GFLOPS: %.2f\n", gflops);

    // Cleanup
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaDeviceReset());

    printf("\nLab 02 Complete!\n");
    return 0;
}
