/**
 * Lab 01: Vector Addition - CUDA Fundamentals
 *
 * Starter Code with TODOs
 *
 * Learning Goals:
 * - Understand CUDA kernel syntax and launch configuration
 * - Master 1D thread indexing
 * - Learn device memory management
 * - Implement error checking
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// TODO 1: Implement the vector addition kernel
// ============================================================================
/**
 * CUDA kernel for vector addition
 * Computes C[i] = A[i] + B[i] for each element i
 *
 * Hints:
 * - Calculate global thread index using blockIdx, blockDim, and threadIdx
 * - Add boundary check to prevent out-of-bounds access
 * - Each thread processes exactly one element
 *
 * @param A     Input vector A
 * @param B     Input vector B
 * @param C     Output vector C
 * @param N     Number of elements in vectors
 */
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    // TODO: Calculate the global thread index
    // Formula: i = blockIdx.x * blockDim.x + threadIdx.x
    int i = 0;  // Replace with correct calculation

    // TODO: Add boundary check and perform addition
    // Only process if i < N to avoid out-of-bounds access

    // Your code here:

}

// ============================================================================
// CPU reference implementation for correctness checking
// ============================================================================
void vectorAddCPU(const float *A, const float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

// ============================================================================
// Utility function to initialize vector with random values
// ============================================================================
void initVector(float *vec, int N) {
    for (int i = 0; i < N; i++) {
        vec[i] = (float)rand() / (float)RAND_MAX;
    }
}

// ============================================================================
// Utility function to verify results
// ============================================================================
bool verifyResults(const float *A, const float *B, int N, float epsilon = 1e-5) {
    for (int i = 0; i < N; i++) {
        if (fabs(A[i] - B[i]) > epsilon) {
            fprintf(stderr, "Mismatch at index %d: CPU=%f, GPU=%f\n",
                    i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

// ============================================================================
// Main function
// ============================================================================
int main(int argc, char **argv) {
    // Parse vector size from command line or use default
    int N = (argc > 1) ? atoi(argv[1]) : 1000000;
    size_t size = N * sizeof(float);

    printf("Vector Addition with %d elements\n", N);
    printf("Memory per vector: %.2f MB\n", size / (1024.0 * 1024.0));

    // Allocate host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    float *h_C_ref = (float*)malloc(size);  // CPU reference result

    if (!h_A || !h_B || !h_C || !h_C_ref) {
        fprintf(stderr, "Failed to allocate host memory\n");
        exit(EXIT_FAILURE);
    }

    // Initialize input vectors with random values
    srand(time(NULL));
    initVector(h_A, N);
    initVector(h_B, N);

    // ========================================================================
    // CPU computation for verification
    // ========================================================================
    printf("\nRunning CPU version...\n");
    clock_t cpu_start = clock();
    vectorAddCPU(h_A, h_B, h_C_ref, N);
    clock_t cpu_end = clock();
    double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000.0;
    printf("CPU Time: %.3f ms\n", cpu_time);

    // ========================================================================
    // TODO 2: Allocate device memory
    // ========================================================================
    printf("\nSetting up GPU...\n");
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    // TODO: Allocate memory on device for A, B, and C
    // Use cudaMalloc and CUDA_CHECK macro
    // Example: CUDA_CHECK(cudaMalloc(&d_A, size));

    // Your code here:


    // ========================================================================
    // TODO 3: Copy input vectors from host to device
    // ========================================================================
    // TODO: Copy h_A and h_B to device
    // Use cudaMemcpy with cudaMemcpyHostToDevice

    // Your code here:


    // ========================================================================
    // TODO 4: Configure and launch the kernel
    // ========================================================================
    // TODO: Calculate grid and block dimensions
    // Hints:
    // - Common block sizes: 128, 256, 512
    // - Grid size should cover all elements: (N + blockSize - 1) / blockSize

    int blockSize = 256;  // Number of threads per block
    int numBlocks = 0;    // TODO: Calculate number of blocks needed

    // Your code here:


    printf("Launching kernel with %d blocks of %d threads\n", numBlocks, blockSize);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Record start event
    CUDA_CHECK(cudaEventRecord(start));

    // TODO: Launch the kernel
    // Syntax: kernelName<<<numBlocks, blockSize>>>(args...);

    // Your code here:


    // Record stop event and synchronize
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Calculate elapsed time
    float gpu_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time, start, stop));
    printf("GPU Kernel Time: %.3f ms\n", gpu_time);

    // ========================================================================
    // TODO 5: Copy result from device to host
    // ========================================================================
    // TODO: Copy d_C back to h_C
    // Use cudaMemcpy with cudaMemcpyDeviceToHost

    // Your code here:


    // ========================================================================
    // Verify results
    // ========================================================================
    printf("\nVerifying results...\n");
    if (verifyResults(h_C_ref, h_C, N)) {
        printf("✓ Results match! Verification PASSED.\n");
    } else {
        printf("✗ Results mismatch! Verification FAILED.\n");
    }

    // ========================================================================
    // Performance analysis
    // ========================================================================
    printf("\n=== Performance Summary ===\n");
    printf("CPU Time:        %.3f ms\n", cpu_time);
    printf("GPU Kernel Time: %.3f ms\n", gpu_time);
    printf("Speedup:         %.2fx\n", cpu_time / gpu_time);

    // Calculate bandwidth
    // Total data: Read A, Read B, Write C = 3N * sizeof(float)
    double totalData_GB = (3.0 * N * sizeof(float)) / (1024.0 * 1024.0 * 1024.0);
    double bandwidth_GBps = totalData_GB / (gpu_time / 1000.0);
    printf("Effective Bandwidth: %.2f GB/s\n", bandwidth_GBps);

    // ========================================================================
    // TODO 6: Free device memory
    // ========================================================================
    // TODO: Free all device memory allocations
    // Use cudaFree

    // Your code here:


    // Cleanup host memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);

    // Cleanup CUDA events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Reset device
    CUDA_CHECK(cudaDeviceReset());

    printf("\nLab 01 Complete!\n");
    return 0;
}

// ============================================================================
// HINTS FOR IMPLEMENTATION
// ============================================================================
/*

TODO 1: Vector Addition Kernel
-------------------------------
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

TODO 2: Allocate Device Memory
-------------------------------
CUDA_CHECK(cudaMalloc(&d_A, size));
CUDA_CHECK(cudaMalloc(&d_B, size));
CUDA_CHECK(cudaMalloc(&d_C, size));

TODO 3: Copy to Device
----------------------
CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

TODO 4: Launch Kernel
---------------------
int numBlocks = (N + blockSize - 1) / blockSize;
vectorAdd<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);
CUDA_CHECK(cudaGetLastError());

TODO 5: Copy from Device
------------------------
CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

TODO 6: Free Device Memory
--------------------------
CUDA_CHECK(cudaFree(d_A));
CUDA_CHECK(cudaFree(d_B));
CUDA_CHECK(cudaFree(d_C));

*/
