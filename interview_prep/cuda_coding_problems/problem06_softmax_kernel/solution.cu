#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define CUDA_CHECK(call) \
    do { cudaError_t err = call; \
         if (err != cudaSuccess) { \
             fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
             exit(1); \
         } \
    } while(0)

__device__ float warpReduceMax(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * Numerically stable softmax kernel
 * Each block processes one row (for small cols)
 * Three passes: max, exp+sum, normalize
 */
__global__ void softmaxKernel(float* input, float* output, int rows, int cols) {
    __shared__ float shared_max[BLOCK_SIZE / WARP_SIZE];
    __shared__ float shared_sum[BLOCK_SIZE / WARP_SIZE];

    int row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;

    // Pass 1: Find max value in row
    float max_val = -INFINITY;
    for (int i = tid; i < cols; i += blockDim.x) {
        max_val = fmaxf(max_val, input[row * cols + i]);
    }
    max_val = warpReduceMax(max_val);

    if (lane == 0) shared_max[wid] = max_val;
    __syncthreads();

    // Reduce across warps
    if (tid < BLOCK_SIZE / WARP_SIZE) {
        max_val = shared_max[tid];
    } else {
        max_val = -INFINITY;
    }
    if (wid == 0) {
        max_val = warpReduceMax(max_val);
        if (lane == 0) shared_max[0] = max_val;
    }
    __syncthreads();
    max_val = shared_max[0];

    // Pass 2: Compute exp(x - max) and sum
    float sum_exp = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float val = expf(input[row * cols + i] - max_val);
        output[row * cols + i] = val;  // Store temporarily
        sum_exp += val;
    }
    sum_exp = warpReduceSum(sum_exp);

    if (lane == 0) shared_sum[wid] = sum_exp;
    __syncthreads();

    if (tid < BLOCK_SIZE / WARP_SIZE) {
        sum_exp = shared_sum[tid];
    } else {
        sum_exp = 0.0f;
    }
    if (wid == 0) {
        sum_exp = warpReduceSum(sum_exp);
        if (lane == 0) shared_sum[0] = sum_exp;
    }
    __syncthreads();
    sum_exp = shared_sum[0];

    // Pass 3: Normalize
    for (int i = tid; i < cols; i += blockDim.x) {
        output[row * cols + i] /= sum_exp;
    }
}

void softmax(float* h_input, float* h_output, int rows, int cols) {
    float *d_input, *d_output;
    size_t size = rows * cols * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    dim3 grid(rows);
    dim3 block(min(BLOCK_SIZE, (cols + 31) / 32 * 32));

    softmaxKernel<<<grid, block>>>(d_input, d_output, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

int main() {
    int rows = 2, cols = 3;
    float input[] = {1.0f, 2.0f, 3.0f,
                     1.0f, 1.0f, 1.0f};
    float output[6];

    softmax(input, output, rows, cols);

    printf("Softmax output:\n");
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            printf("%.4f ", output[i * cols + j]);
            sum += output[i * cols + j];
        }
        printf(" (sum=%.4f)\n", sum);
    }

    return 0;
}
