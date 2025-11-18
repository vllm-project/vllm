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
 * Naive histogram using global atomic operations
 * Simple but slow due to memory contention
 */
__global__ void histogramNaive(int* input, int* histogram, int n, int numBins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride) {
        int bin = input[i];
        if (bin >= 0 && bin < numBins) {
            atomicAdd(&histogram[bin], 1);
        }
    }
}

/**
 * Optimized histogram using shared memory privatization
 * Each block maintains private histogram in shared memory
 * Much faster due to reduced global memory contention
 */
__global__ void histogramPrivatized(int* input, int* histogram, int n, int numBins) {
    extern __shared__ int localHist[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Initialize local histogram
    for (int i = tid; i < numBins; i += blockDim.x) {
        localHist[i] = 0;
    }
    __syncthreads();

    // Accumulate to local histogram
    for (int i = idx; i < n; i += stride) {
        int bin = input[i];
        if (bin >= 0 && bin < numBins) {
            atomicAdd(&localHist[bin], 1);
        }
    }
    __syncthreads();

    // Merge local histograms to global
    for (int i = tid; i < numBins; i += blockDim.x) {
        if (localHist[i] > 0) {
            atomicAdd(&histogram[i], localHist[i]);
        }
    }
}

/**
 * Host function
 */
void computeHistogram(int* h_input, int* h_histogram, int n, int numBins) {
    int *d_input, *d_histogram;

    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_histogram, numBins * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_histogram, 0, numBins * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice));

    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    numBlocks = min(numBlocks, 512);

    // Use privatized version if bins fit in shared memory
    if (numBins * sizeof(int) <= 48 * 1024) {
        histogramPrivatized<<<numBlocks, BLOCK_SIZE, numBins * sizeof(int)>>>(
            d_input, d_histogram, n, numBins);
    } else {
        histogramNaive<<<numBlocks, BLOCK_SIZE>>>(d_input, d_histogram, n, numBins);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_histogram, d_histogram, numBins * sizeof(int),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_histogram));
}

int main() {
    int input[] = {0,1,2,1,0,3,2,1,0,2};
    int n = 10;
    int numBins = 4;
    int histogram[4] = {0};

    computeHistogram(input, histogram, n, numBins);

    printf("Histogram: ");
    for (int i = 0; i < numBins; i++) {
        printf("%d ", histogram[i]);
    }
    printf("\n");

    return 0;
}
