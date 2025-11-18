#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * PARALLEL REDUCTION - OPTIMIZED SOLUTION
 *
 * Time Complexity: O(log n) per element
 * Space Complexity: O(n/BLOCK_SIZE) for partial sums
 *
 * Key optimizations:
 * 1. Sequential addressing to avoid bank conflicts
 * 2. First-level reduction during load to shared memory
 * 3. Unrolled last warp for better performance
 * 4. Multiple elements per thread
 */

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/**
 * Warp-level reduction using shuffle instructions
 * This is the most efficient way to reduce within a warp
 */
__device__ __forceinline__ float warpReduce(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * Main reduction kernel with optimizations
 *
 * Strategy:
 * 1. Each thread loads multiple elements and reduces them
 * 2. Store to shared memory
 * 3. Reduce in shared memory using sequential addressing
 * 4. Use warp primitives for final warp reduction
 *
 * @param input: Input array
 * @param output: Partial sums (one per block)
 * @param n: Number of elements
 */
__global__ void reductionKernel(float* input, float* output, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    unsigned int gridSize = blockDim.x * 2 * gridDim.x;

    // First level reduction: each thread sums multiple elements
    // This reduces the number of blocks needed and improves memory coalescing
    float sum = 0.0f;
    while (idx < n) {
        sum += input[idx];
        // Also load the element at idx + blockDim.x if it exists
        if (idx + blockDim.x < n) {
            sum += input[idx + blockDim.x];
        }
        idx += gridSize;
    }

    // Store to shared memory
    sdata[tid] = sum;
    __syncthreads();

    // Reduction in shared memory using sequential addressing
    // This avoids bank conflicts and reduces warp divergence
    for (unsigned int s = blockDim.x / 2; s > WARP_SIZE; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Final reduction using warp primitives (no sync needed)
    if (tid < WARP_SIZE) {
        float val = sdata[tid];

        // Unroll the last warp
        if (blockDim.x >= 64) val += sdata[tid + 32];
        val = warpReduce(val);

        // First thread writes result
        if (tid == 0) {
            output[blockIdx.x] = val;
        }
    }
}

/**
 * Host function to compute parallel sum
 * Handles kernel launch and recursive reduction if needed
 *
 * @param h_input: Host input array
 * @param n: Number of elements
 * @return: Sum of all elements
 */
float parallelSum(float* h_input, int n) {
    if (n == 0) return 0.0f;
    if (n == 1) return h_input[0];

    // Allocate device memory
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, n * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Calculate grid dimensions
    int threadsPerBlock = BLOCK_SIZE;
    int numBlocks = (n + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);
    numBlocks = min(numBlocks, 1024); // Limit number of blocks

    // Allocate output for partial sums
    CUDA_CHECK(cudaMalloc(&d_output, numBlocks * sizeof(float)));

    // Shared memory size
    size_t sharedMemSize = threadsPerBlock * sizeof(float);

    // Launch kernel
    reductionKernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
        d_input, d_output, n);
    CUDA_CHECK(cudaGetLastError());

    // Copy partial sums back to host
    float* h_output = (float*)malloc(numBlocks * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_output, d_output, numBlocks * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Final reduction on CPU (small enough)
    float result = 0.0f;
    for (int i = 0; i < numBlocks; i++) {
        result += h_output[i];
    }

    // Cleanup
    free(h_output);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return result;
}

/**
 * Alternative: Recursive GPU reduction for very large arrays
 * This keeps everything on GPU until final result
 */
float parallelSumRecursive(float* h_input, int n) {
    if (n == 0) return 0.0f;
    if (n == 1) return h_input[0];

    float *d_input, *d_temp;
    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, n * sizeof(float),
                          cudaMemcpyHostToDevice));

    int currentSize = n;
    float *d_in = d_input;
    float *d_out = nullptr;

    int threadsPerBlock = BLOCK_SIZE;

    // Reduce until we have a single block result
    while (currentSize > 1) {
        int numBlocks = (currentSize + threadsPerBlock * 2 - 1) /
                        (threadsPerBlock * 2);
        numBlocks = min(numBlocks, 1024);

        CUDA_CHECK(cudaMalloc(&d_out, numBlocks * sizeof(float)));

        size_t sharedMemSize = threadsPerBlock * sizeof(float);
        reductionKernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
            d_in, d_out, currentSize);
        CUDA_CHECK(cudaGetLastError());

        // Free previous input if it's not the original
        if (d_in != d_input) {
            CUDA_CHECK(cudaFree(d_in));
        }

        d_in = d_out;
        currentSize = numBlocks;
    }

    // Copy final result
    float result;
    CUDA_CHECK(cudaMemcpy(&result, d_out, sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_out));

    return result;
}

// Main function for demonstration
int main() {
    const int N = 1000000;
    float *h_input = (float*)malloc(N * sizeof(float));

    // Initialize array
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;
    }

    // Compute sum
    float result = parallelSum(h_input, N);
    printf("Sum of %d elements: %f\n", N, result);
    printf("Expected: %d\n", N);
    printf("Error: %f\n", fabs(result - N));

    free(h_input);
    return 0;
}
