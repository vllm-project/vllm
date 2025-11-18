/**
 * Lab 01: Vector Addition - Test Suite
 *
 * Comprehensive testing and benchmarking for vector addition kernel
 *
 * Tests include:
 * - Correctness validation (multiple sizes)
 * - Edge cases (small sizes, power-of-2, non-power-of-2)
 * - Performance benchmarking (different block sizes)
 * - Memory bandwidth analysis
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Vector addition kernel
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// CPU reference
void vectorAddCPU(const float *A, const float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

// Initialize vector with pattern for easier debugging
void initVector(float *vec, int N, int seed) {
    srand(seed);
    for (int i = 0; i < N; i++) {
        vec[i] = (float)(rand() % 100) / 10.0f;  // 0.0 to 10.0
    }
}

// Verify results with detailed reporting
bool verifyResults(const float *expected, const float *actual, int N,
                   const char *testName, float epsilon = 1e-4) {
    int errors = 0;
    int maxErrors = 10;  // Report first 10 errors

    for (int i = 0; i < N; i++) {
        float diff = fabs(expected[i] - actual[i]);
        if (diff > epsilon) {
            if (errors < maxErrors) {
                fprintf(stderr, "  [%s] Error at index %d: expected=%f, got=%f, diff=%f\n",
                        testName, i, expected[i], actual[i], diff);
            }
            errors++;
        }
    }

    if (errors > 0) {
        fprintf(stderr, "  [%s] Total errors: %d / %d (%.2f%%)\n",
                testName, errors, N, (errors * 100.0f) / N);
        return false;
    }

    return true;
}

// Benchmark kernel with specific configuration
float benchmarkKernel(const float *d_A, const float *d_B, float *d_C, int N,
                      int blockSize, int numRuns = 10) {
    int numBlocks = (N + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    vectorAdd<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < numRuns; i++) {
        vectorAdd<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float totalTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&totalTime, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return totalTime / numRuns;  // Average time
}

// Test case structure
typedef struct {
    const char *name;
    int size;
    int blockSize;
} TestCase;

int main() {
    printf("========================================\n");
    printf("Vector Addition - Comprehensive Test Suite\n");
    printf("========================================\n\n");

    // Get device properties
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Memory Clock Rate: %.2f GHz\n", prop.memoryClockRate / 1e6);
    printf("Memory Bus Width: %d bits\n", prop.memoryBusWidth);
    double theoreticalBW = (prop.memoryClockRate * 1e3) *
                          (prop.memoryBusWidth / 8.0) * 2.0 / 1e9;
    printf("Theoretical Bandwidth: %.2f GB/s\n\n", theoreticalBW);

    // ========================================================================
    // Test 1: Correctness Tests (Various Sizes)
    // ========================================================================
    printf("========================================\n");
    printf("Test 1: Correctness Validation\n");
    printf("========================================\n");

    TestCase correctnessTests[] = {
        {"Small (1K)", 1024, 256},
        {"Medium (1M)", 1000000, 256},
        {"Large (10M)", 10000000, 256},
        {"Non-power-of-2 (1000001)", 1000001, 256},
        {"Prime (999983)", 999983, 256},
        {"Edge case (1)", 1, 256},
        {"Edge case (31)", 31, 256},
        {"Edge case (32)", 32, 256},
        {"Edge case (33)", 33, 256},
    };

    int numTests = sizeof(correctnessTests) / sizeof(TestCase);
    int passed = 0;

    for (int t = 0; t < numTests; t++) {
        TestCase tc = correctnessTests[t];
        int N = tc.size;
        size_t size = N * sizeof(float);

        // Allocate host memory
        float *h_A = (float*)malloc(size);
        float *h_B = (float*)malloc(size);
        float *h_C = (float*)malloc(size);
        float *h_C_ref = (float*)malloc(size);

        // Initialize
        initVector(h_A, N, 42);
        initVector(h_B, N, 43);

        // CPU computation
        vectorAddCPU(h_A, h_B, h_C_ref, N);

        // GPU computation
        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, size));
        CUDA_CHECK(cudaMalloc(&d_B, size));
        CUDA_CHECK(cudaMalloc(&d_C, size));

        CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

        int numBlocks = (N + tc.blockSize - 1) / tc.blockSize;
        vectorAdd<<<numBlocks, tc.blockSize>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

        // Verify
        bool testPassed = verifyResults(h_C_ref, h_C, N, tc.name);
        if (testPassed) {
            printf("✓ PASS: %s\n", tc.name);
            passed++;
        } else {
            printf("✗ FAIL: %s\n", tc.name);
        }

        // Cleanup
        free(h_A); free(h_B); free(h_C); free(h_C_ref);
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
    }

    printf("\nCorrectness: %d/%d tests passed\n", passed, numTests);

    // ========================================================================
    // Test 2: Performance Benchmarking (Block Size Sweep)
    // ========================================================================
    printf("\n========================================\n");
    printf("Test 2: Block Size Performance Analysis\n");
    printf("========================================\n");

    int N = 10000000;  // 10M elements
    size_t size = N * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    initVector(h_A, N, 42);
    initVector(h_B, N, 43);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    int blockSizes[] = {32, 64, 128, 256, 512, 1024};
    int numBlockSizes = sizeof(blockSizes) / sizeof(int);

    printf("\n%-12s %-12s %-15s %-15s %-12s\n",
           "Block Size", "Time (ms)", "Bandwidth(GB/s)", "Efficiency(%)", "Blocks");
    printf("─────────────────────────────────────────────────────────────────────\n");

    float bestTime = 1e9;
    int bestBlockSize = 0;

    for (int i = 0; i < numBlockSizes; i++) {
        int blockSize = blockSizes[i];
        float avgTime = benchmarkKernel(d_A, d_B, d_C, N, blockSize, 20);

        double totalData_GB = (3.0 * N * sizeof(float)) / (1024.0 * 1024.0 * 1024.0);
        double bandwidth = totalData_GB / (avgTime / 1000.0);
        double efficiency = (bandwidth / theoreticalBW) * 100.0;
        int numBlocks = (N + blockSize - 1) / blockSize;

        printf("%-12d %-12.3f %-15.2f %-15.1f %-12d",
               blockSize, avgTime, bandwidth, efficiency, numBlocks);

        if (avgTime < bestTime) {
            bestTime = avgTime;
            bestBlockSize = blockSize;
            printf(" ← Best");
        }
        printf("\n");
    }

    printf("\nOptimal block size: %d threads (%.3f ms)\n", bestBlockSize, bestTime);

    // Cleanup
    free(h_A); free(h_B);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // ========================================================================
    // Test 3: Scalability Analysis
    // ========================================================================
    printf("\n========================================\n");
    printf("Test 3: Scalability Analysis\n");
    printf("========================================\n");

    int sizes[] = {1000, 10000, 100000, 1000000, 10000000, 100000000};
    int numSizes = sizeof(sizes) / sizeof(int);

    printf("\n%-15s %-12s %-15s %-12s\n",
           "Size", "Time (ms)", "Bandwidth(GB/s)", "GFLOPS");
    printf("────────────────────────────────────────────────────────────\n");

    for (int i = 0; i < numSizes; i++) {
        N = sizes[i];
        size = N * sizeof(float);

        float *h_A = (float*)malloc(size);
        float *h_B = (float*)malloc(size);
        initVector(h_A, N, 42);
        initVector(h_B, N, 43);

        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, size));
        CUDA_CHECK(cudaMalloc(&d_B, size));
        CUDA_CHECK(cudaMalloc(&d_C, size));

        CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

        float avgTime = benchmarkKernel(d_A, d_B, d_C, N, 256, 10);

        double totalData_GB = (3.0 * N * sizeof(float)) / (1024.0 * 1024.0 * 1024.0);
        double bandwidth = totalData_GB / (avgTime / 1000.0);
        double gflops = (N / 1e9) / (avgTime / 1000.0);

        printf("%-15d %-12.3f %-15.2f %-12.2f\n",
               N, avgTime, bandwidth, gflops);

        free(h_A); free(h_B);
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
    }

    // ========================================================================
    // Summary
    // ========================================================================
    printf("\n========================================\n");
    printf("Test Summary\n");
    printf("========================================\n");
    printf("Correctness Tests: %d/%d passed\n", passed, numTests);
    printf("Performance Tests: Complete\n");
    printf("Scalability Tests: Complete\n");

    if (passed == numTests) {
        printf("\n✓ All tests PASSED!\n");
        printf("\nNext Steps:\n");
        printf("1. Review solution.cu for optimization details\n");
        printf("2. Profile with: nsys profile ./test\n");
        printf("3. Try optimization challenges in README.md\n");
        printf("4. Proceed to Lab 02: Matrix Multiplication\n");
    } else {
        printf("\n✗ Some tests FAILED - review implementation\n");
    }

    printf("========================================\n");

    CUDA_CHECK(cudaDeviceReset());
    return (passed == numTests) ? 0 : 1;
}
