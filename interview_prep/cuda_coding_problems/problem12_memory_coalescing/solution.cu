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
 * BAD: Uncoalesced - adjacent threads access strided memory
 */
__global__ void uncoaclescedCopy(float* input, float* output, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx * stride];  // Strided access!
    }
}

/**
 * GOOD: Coalesced - use shared memory to reorganize
 */
__global__ void coalescedCopy(float* input, float* output, int n, int stride) {
    __shared__ float tile[BLOCK_SIZE];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int threadLane = threadIdx.x;

    // Coalesced read: consecutive threads read consecutive memory
    int readIdx = blockIdx.x * blockDim.x * stride + threadLane;
    if (readIdx < n * stride && threadLane < BLOCK_SIZE) {
        // Load BLOCK_SIZE consecutive elements
        for (int i = 0; i < stride && (readIdx + i * BLOCK_SIZE) < n * stride; i++) {
            if (idx < n) {
                tile[threadLane] = input[readIdx + i * BLOCK_SIZE];
                __syncthreads();
                // Now threads can access tile in any pattern
                if (threadLane < stride) {
                    output[idx] = tile[threadLane];
                }
                __syncthreads();
            }
        }
    }

    // Alternative: Simple coalesced approach
    // Each thread processes multiple elements
    if (idx < n) {
        output[idx] = input[idx * stride];
    }
}

/**
 * BEST: Coalesced with shared memory tile
 */
__global__ void coalescedCopyOptimized(float* input, float* output,
                                        int n, int stride) {
    __shared__ float tile[BLOCK_SIZE];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // Coalesced load into shared memory
        int load_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (load_idx < n) {
            tile[threadIdx.x] = input[load_idx];
        }
        __syncthreads();

        // Strided access to shared memory (fast)
        if (idx < n && (threadIdx.x * stride) < blockDim.x) {
            output[idx] = tile[threadIdx.x * stride];
        }
    }
}

void testCoalescing() {
    int n = 1024;
    int stride = 4;
    float *h_input = new float[n * stride];
    float *h_output = new float[n];
    float *d_input, *d_output;

    for (int i = 0; i < n * stride; i++) h_input[i] = i;

    CUDA_CHECK(cudaMalloc(&d_input, n * stride * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, n * stride * sizeof(float),
                          cudaMemcpyHostToDevice));

    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Compare kernels
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Uncoalesced
    cudaEventRecord(start);
    uncoalescedCopy<<<numBlocks, BLOCK_SIZE>>>(d_input, d_output, n, stride);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_uncoalesced;
    cudaEventElapsedTime(&ms_uncoalesced, start, stop);

    // Coalesced
    cudaEventRecord(start);
    coalescedCopyOptimized<<<numBlocks, BLOCK_SIZE>>>(d_input, d_output, n, stride);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_coalesced;
    cudaEventElapsedTime(&ms_coalesced, start, stop);

    printf("Uncoalesced: %.3f ms\n", ms_uncoalesced);
    printf("Coalesced: %.3f ms\n", ms_coalesced);
    printf("Speedup: %.2fx\n", ms_uncoalesced / ms_coalesced);

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    delete[] h_input;
    delete[] h_output;
}

int main() {
    testCoalescing();
    return 0;
}
