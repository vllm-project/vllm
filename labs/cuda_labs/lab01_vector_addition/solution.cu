/**
 * Lab 01: Vector Addition - CUDA Fundamentals
 *
 * Complete Solution with Detailed Comments
 *
 * This solution demonstrates:
 * - Proper CUDA kernel implementation
 * - Thread indexing and boundary checking
 * - Memory management patterns
 * - Error handling best practices
 * - Performance measurement
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

// Error checking macro - wraps CUDA calls and reports errors with context
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
// SOLUTION: Vector Addition Kernel
// ============================================================================
/**
 * CUDA kernel for vector addition
 *
 * Key Concepts:
 * 1. Thread Indexing: Each thread calculates its global index
 * 2. Boundary Check: Prevents out-of-bounds memory access
 * 3. Memory Access Pattern: Coalesced (consecutive threads access consecutive memory)
 *
 * Performance Characteristics:
 * - Memory-bound operation (limited by DRAM bandwidth)
 * - Compute intensity: Very low (1 FLOP per 3 memory accesses)
 * - Expected bandwidth: 60-80% of theoretical peak
 *
 * @param A     Input vector A (device memory)
 * @param B     Input vector B (device memory)
 * @param C     Output vector C (device memory)
 * @param N     Number of elements
 */
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    // Calculate global thread index
    // blockIdx.x: which block this thread belongs to
    // blockDim.x: number of threads per block
    // threadIdx.x: thread index within the block
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check: necessary when N is not a multiple of blockDim.x
    // Without this, threads in the last block might access invalid memory
    if (i < N) {
        // Perform the addition
        // This is a coalesced memory access pattern:
        // - Thread 0 accesses A[0], B[0], writes C[0]
        // - Thread 1 accesses A[1], B[1], writes C[1]
        // - Consecutive threads access consecutive memory locations
        C[i] = A[i] + B[i];
    }
}

// ============================================================================
// CPU reference implementation
// ============================================================================
void vectorAddCPU(const float *A, const float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

// ============================================================================
// Utility functions
// ============================================================================
void initVector(float *vec, int N) {
    for (int i = 0; i < N; i++) {
        vec[i] = (float)rand() / (float)RAND_MAX;
    }
}

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
    // Parse vector size from command line
    int N = (argc > 1) ? atoi(argv[1]) : 1000000;
    size_t size = N * sizeof(float);

    printf("========================================\n");
    printf("Vector Addition - CUDA Solution\n");
    printf("========================================\n");
    printf("Vector size: %d elements\n", N);
    printf("Memory per vector: %.2f MB\n", size / (1024.0 * 1024.0));
    printf("Total memory: %.2f MB\n", 3 * size / (1024.0 * 1024.0));

    // ========================================================================
    // Host memory allocation
    // ========================================================================
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    float *h_C_ref = (float*)malloc(size);

    if (!h_A || !h_B || !h_C || !h_C_ref) {
        fprintf(stderr, "Failed to allocate host memory\n");
        exit(EXIT_FAILURE);
    }

    // Initialize input vectors
    srand(time(NULL));
    initVector(h_A, N);
    initVector(h_B, N);

    // ========================================================================
    // CPU computation for verification
    // ========================================================================
    printf("\n--- CPU Computation ---\n");
    clock_t cpu_start = clock();
    vectorAddCPU(h_A, h_B, h_C_ref, N);
    clock_t cpu_end = clock();
    double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000.0;
    printf("CPU Time: %.3f ms\n", cpu_time);

    // ========================================================================
    // Device memory allocation
    // ========================================================================
    printf("\n--- GPU Setup ---\n");
    float *d_A, *d_B, *d_C;

    // Allocate device memory
    // cudaMalloc returns a pointer to GPU memory
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));
    printf("Device memory allocated\n");

    // ========================================================================
    // Copy data from host to device
    // ========================================================================
    // cudaMemcpy performs synchronous data transfer
    // cudaMemcpyHostToDevice: CPU → GPU
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    printf("Input data copied to device\n");

    // ========================================================================
    // Kernel configuration and launch
    // ========================================================================
    // Block size selection:
    // - Must be multiple of 32 (warp size)
    // - Common choices: 128, 256, 512
    // - 256 is a good default for many kernels
    int blockSize = 256;

    // Grid size calculation:
    // - Need enough blocks to cover all N elements
    // - Ceiling division: (N + blockSize - 1) / blockSize
    // - Example: N=1000, blockSize=256 → numBlocks = (1000+255)/256 = 4
    int numBlocks = (N + blockSize - 1) / blockSize;

    printf("Kernel config: %d blocks × %d threads = %d threads total\n",
           numBlocks, blockSize, numBlocks * blockSize);

    // Create CUDA events for accurate kernel timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Record start event on the default stream
    CUDA_CHECK(cudaEventRecord(start));

    // Launch kernel
    // Syntax: kernelName<<<gridDim, blockDim, sharedMemSize, stream>>>(args)
    // - gridDim: number of blocks (1D in this case)
    // - blockDim: threads per block (1D in this case)
    // - sharedMemSize: 0 (not using shared memory)
    // - stream: default (0)
    vectorAdd<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);

    // Check for kernel launch errors
    // This catches configuration errors (e.g., invalid grid/block size)
    CUDA_CHECK(cudaGetLastError());

    // Record stop event
    CUDA_CHECK(cudaEventRecord(stop));

    // Wait for event to complete
    // This ensures the kernel has finished execution
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Calculate elapsed time in milliseconds
    float gpu_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time, start, stop));
    printf("GPU Kernel Time: %.3f ms\n", gpu_time);

    // ========================================================================
    // Copy result back to host
    // ========================================================================
    // cudaMemcpyDeviceToHost: GPU → CPU
    // This call is synchronous and implicitly waits for kernel completion
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
    printf("Result copied back to host\n");

    // ========================================================================
    // Verify results
    // ========================================================================
    printf("\n--- Verification ---\n");
    if (verifyResults(h_C_ref, h_C, N)) {
        printf("✓ Results match! Verification PASSED.\n");
    } else {
        printf("✗ Results mismatch! Verification FAILED.\n");
    }

    // ========================================================================
    // Performance analysis
    // ========================================================================
    printf("\n========================================\n");
    printf("Performance Summary\n");
    printf("========================================\n");
    printf("CPU Time:          %.3f ms\n", cpu_time);
    printf("GPU Kernel Time:   %.3f ms\n", gpu_time);
    printf("Speedup (kernel):  %.2fx\n", cpu_time / gpu_time);

    // Calculate effective memory bandwidth
    // Vector addition performs:
    // - 2 reads: A[i], B[i]
    // - 1 write: C[i]
    // - Total: 3 * N * sizeof(float) bytes
    double totalData_GB = (3.0 * N * sizeof(float)) / (1024.0 * 1024.0 * 1024.0);
    double bandwidth_GBps = totalData_GB / (gpu_time / 1000.0);
    printf("Effective Bandwidth: %.2f GB/s\n", bandwidth_GBps);

    // Get device properties to show theoretical peak
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    double theoreticalBW = (prop.memoryClockRate * 1e3) * (prop.memoryBusWidth / 8.0) * 2.0 / 1e9;
    printf("Theoretical Peak BW: %.2f GB/s\n", theoreticalBW);
    printf("Bandwidth Efficiency: %.1f%%\n", (bandwidth_GBps / theoreticalBW) * 100.0);

    // Compute intensity analysis
    double flops = N;  // One addition per element
    double gflops = (flops / 1e9) / (gpu_time / 1000.0);
    printf("\nCompute Analysis:\n");
    printf("GFLOPS: %.2f\n", gflops);
    printf("Arithmetic Intensity: %.3f FLOP/byte (memory-bound)\n",
           flops / (3.0 * N * sizeof(float)));

    // ========================================================================
    // Cleanup
    // ========================================================================
    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);

    // Destroy events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Reset device (optional, but good practice)
    CUDA_CHECK(cudaDeviceReset());

    printf("\n========================================\n");
    printf("Lab 01 Complete!\n");
    printf("========================================\n");

    return 0;
}

// ============================================================================
// OPTIMIZATION NOTES
// ============================================================================
/*

1. Memory Access Pattern:
   - This kernel has perfect coalescing: consecutive threads access consecutive
     memory locations
   - Each warp (32 threads) issues a single memory transaction for aligned data
   - This is the ideal access pattern for global memory

2. Block Size Selection:
   - 256 threads is a good default
   - For this memory-bound kernel, block size has minimal impact on performance
   - Must be multiple of 32 (warp size)
   - Maximum: depends on GPU (usually 1024)

3. Grid-Stride Loop (Alternative Pattern):
   Instead of launching exactly N threads, you could launch fewer threads
   and have each thread process multiple elements:

   __global__ void vectorAddGridStride(float *A, float *B, float *C, int N) {
       for (int i = blockIdx.x * blockDim.x + threadIdx.x;
            i < N;
            i += blockDim.x * gridDim.x) {
           C[i] = A[i] + B[i];
       }
   }

   Advantages:
   - No need to worry about grid size limits
   - Can tune occupancy independently of N
   - Better for very large arrays

4. Unified Memory (cudaMallocManaged):
   Could simplify memory management but may have different performance:
   - No explicit transfers needed
   - Page migration overhead
   - Use cudaMemPrefetchAsync for control

5. Streams for Overlap:
   For large vectors, could overlap transfer and compute:
   - Divide data into chunks
   - Use multiple streams
   - Transfer chunk i while computing chunk i-1

6. Kernel Fusion:
   If you have multiple vector operations, fuse them:
   - D = A + B * C in one kernel instead of two
   - Reduces memory traffic (only write D once)
   - Higher arithmetic intensity

7. Performance Bottlenecks:
   - Memory bandwidth limited (typical for simple kernels)
   - PCIe transfer overhead dominates for small arrays
   - Launch overhead negligible for modern GPUs

8. When to Use GPU:
   - Large arrays (> 1M elements for this operation)
   - When data already on GPU
   - Part of larger GPU pipeline
   - Not worth it for small, one-off computations

*/
