#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 256
#define CUDA_CHECK(call) \
    do { cudaError_t err = call; \
         if (err != cudaSuccess) { \
             fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
             exit(1); \
         } \
    } while(0)

/**
 * Work-efficient Blelloch scan within a block
 * Performs both up-sweep and down-sweep phases
 */
__global__ void scanBlockKernel(float* input, float* output, float* blockSums,
                                int n, bool inclusive) {
    __shared__ float temp[BLOCK_SIZE * 2];

    int tid = threadIdx.x;
    int idx = blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x;

    // Load input into shared memory
    temp[tid] = (idx < n) ? input[idx] : 0.0f;
    temp[tid + BLOCK_SIZE] = (idx + BLOCK_SIZE < n) ? input[idx + BLOCK_SIZE] : 0.0f;
    __syncthreads();

    // Up-sweep (reduce) phase
    int offset = 1;
    for (int d = BLOCK_SIZE; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset <<= 1;
    }

    // Store block sum for later
    if (tid == 0) {
        if (blockSums != nullptr) {
            blockSums[blockIdx.x] = temp[BLOCK_SIZE * 2 - 1];
        }
        temp[BLOCK_SIZE * 2 - 1] = 0; // Set last element to 0 for exclusive scan
    }

    // Down-sweep phase
    for (int d = 1; d < BLOCK_SIZE * 2; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    // Write results
    if (inclusive) {
        if (idx < n) output[idx] = temp[tid] + ((idx < n) ? input[idx] : 0.0f);
        if (idx + BLOCK_SIZE < n) output[idx + BLOCK_SIZE] = temp[tid + BLOCK_SIZE] + input[idx + BLOCK_SIZE];
    } else {
        if (idx < n) output[idx] = temp[tid];
        if (idx + BLOCK_SIZE < n) output[idx + BLOCK_SIZE] = temp[tid + BLOCK_SIZE];
    }
}

/**
 * Add block sums to scanned blocks
 */
__global__ void addBlockSums(float* output, float* blockSums, int n) {
    int idx = blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x;
    if (blockIdx.x > 0) {
        float sum = blockSums[blockIdx.x - 1];
        if (idx < n) output[idx] += sum;
        if (idx + BLOCK_SIZE < n) output[idx + BLOCK_SIZE] += sum;
    }
}

/**
 * Host function for prefix sum with multi-level approach
 */
void prefixSum(float* h_input, float* h_output, int n, bool inclusive) {
    float *d_input, *d_output, *d_blockSums, *h_blockSums;

    size_t size = n * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    int numBlocks = (n + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);

    if (numBlocks > 1) {
        CUDA_CHECK(cudaMalloc(&d_blockSums, numBlocks * sizeof(float)));
        h_blockSums = new float[numBlocks];
    } else {
        d_blockSums = nullptr;
    }

    // Scan each block
    scanBlockKernel<<<numBlocks, BLOCK_SIZE>>>(d_input, d_output, d_blockSums, n, inclusive);
    CUDA_CHECK(cudaGetLastError());

    // If multiple blocks, scan block sums and add them
    if (numBlocks > 1) {
        CUDA_CHECK(cudaMemcpy(h_blockSums, d_blockSums, numBlocks * sizeof(float),
                              cudaMemcpyDeviceToHost));

        // CPU scan of block sums (small enough)
        float sum = 0;
        for (int i = 0; i < numBlocks; i++) {
            float temp = h_blockSums[i];
            h_blockSums[i] = sum;
            sum += temp;
        }

        CUDA_CHECK(cudaMemcpy(d_blockSums, h_blockSums, numBlocks * sizeof(float),
                              cudaMemcpyHostToDevice));

        // Add block sums to each block's results
        addBlockSums<<<numBlocks, BLOCK_SIZE>>>(d_output, d_blockSums, n);
        CUDA_CHECK(cudaGetLastError());

        delete[] h_blockSums;
        CUDA_CHECK(cudaFree(d_blockSums));
    }

    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

int main() {
    int n = 8;
    float input[] = {3, 1, 7, 0, 4, 1, 6, 3};
    float output[8];

    printf("Input: ");
    for (int i = 0; i < n; i++) printf("%g ", input[i]);
    printf("\n");

    prefixSum(input, output, n, true);
    printf("Inclusive: ");
    for (int i = 0; i < n; i++) printf("%g ", output[i]);
    printf("\n");

    prefixSum(input, output, n, false);
    printf("Exclusive: ");
    for (int i = 0; i < n; i++) printf("%g ", output[i]);
    printf("\n");

    return 0;
}
