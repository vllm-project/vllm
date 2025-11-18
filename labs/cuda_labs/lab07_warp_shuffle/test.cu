/**
 * Lab 07: Warp Shuffle - Test Suite
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define CUDA_CHECK(call) do { cudaError_t err = call; if (err != cudaSuccess) { \
    fprintf(stderr, "Error: %s\n", cudaGetErrorString(err)); exit(1); } } while(0)

#define WARP_SIZE 32
#define BLOCK_SIZE 256

__device__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void reduceShfl(const float *input, float *output, int n) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float val = (i < n) ? input[i] : 0.0f;
    val = warpReduceSum(val);

    __shared__ float warp_sums[BLOCK_SIZE / WARP_SIZE];
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    if (lane_id == 0) warp_sums[warp_id] = val;
    __syncthreads();

    if (tid < BLOCK_SIZE / WARP_SIZE) {
        val = warp_sums[tid];
    } else {
        val = 0.0f;
    }

    if (warp_id == 0) {
        val = warpReduceSum(val);
        if (lane_id == 0) output[blockIdx.x] = val;
    }
}

int main() {
    printf("Warp Shuffle Test Suite\n");

    int sizes[] = {1024, 65536, 1<<20};
    int passed = 0;

    for (int t = 0; t < 3; t++) {
        int n = sizes[t];
        float *h_in = (float*)malloc(n * sizeof(float));
        for (int i = 0; i < n; i++) h_in[i] = 1.0f;

        float *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, 1024 * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice));

        int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        reduceShfl<<<grid, BLOCK_SIZE>>>(d_in, d_out, n);

        float *h_out = (float*)malloc(grid * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_out, d_out, grid * sizeof(float), cudaMemcpyDeviceToHost));

        float sum = 0.0f;
        for (int i = 0; i < grid; i++) sum += h_out[i];

        bool ok = fabs(sum - n) < 1.0f;
        printf("Size %d: %s (%.0f vs %d)\n", n, ok ? "PASS" : "FAIL", sum, n);
        if (ok) passed++;

        free(h_in); free(h_out);
        CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_out));
    }

    printf("\nResults: %d/3 tests passed\n", passed);
    return passed == 3 ? 0 : 1;
}
