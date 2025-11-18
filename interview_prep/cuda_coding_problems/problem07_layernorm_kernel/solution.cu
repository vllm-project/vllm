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

__device__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * Layer Normalization kernel using Welford's online algorithm
 * Computes mean and variance in a single pass for numerical stability
 */
__global__ void layerNormKernel(float* input, float* output, float* gamma,
                                float* beta, int rows, int cols, float eps) {
    __shared__ float shared_mean[BLOCK_SIZE / WARP_SIZE];
    __shared__ float shared_var[BLOCK_SIZE / WARP_SIZE];

    int row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;

    // Pass 1: Compute mean using Welford's algorithm
    float mean = 0.0f;
    float M2 = 0.0f;
    int count = 0;

    for (int i = tid; i < cols; i += blockDim.x) {
        float val = input[row * cols + i];
        count++;
        float delta = val - mean;
        mean += delta / count;
        float delta2 = val - mean;
        M2 += delta * delta2;
    }

    // Reduce sum across thread block
    float sum_val = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        sum_val += input[row * cols + i];
    }
    sum_val = warpReduceSum(sum_val);

    if (lane == 0) shared_mean[wid] = sum_val;
    __syncthreads();

    if (tid < BLOCK_SIZE / WARP_SIZE) {
        sum_val = shared_mean[tid];
    } else {
        sum_val = 0.0f;
    }
    if (wid == 0) {
        sum_val = warpReduceSum(sum_val);
        if (lane == 0) shared_mean[0] = sum_val / cols;
    }
    __syncthreads();
    mean = shared_mean[0];

    // Pass 2: Compute variance
    float var_sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float diff = input[row * cols + i] - mean;
        var_sum += diff * diff;
    }
    var_sum = warpReduceSum(var_sum);

    if (lane == 0) shared_var[wid] = var_sum;
    __syncthreads();

    if (tid < BLOCK_SIZE / WARP_SIZE) {
        var_sum = shared_var[tid];
    } else {
        var_sum = 0.0f;
    }
    if (wid == 0) {
        var_sum = warpReduceSum(var_sum);
        if (lane == 0) shared_var[0] = var_sum / cols;
    }
    __syncthreads();
    float variance = shared_var[0];

    // Pass 3: Normalize and apply affine transformation
    float inv_std = rsqrtf(variance + eps);
    for (int i = tid; i < cols; i += blockDim.x) {
        float normalized = (input[row * cols + i] - mean) * inv_std;
        output[row * cols + i] = gamma[i] * normalized + beta[i];
    }
}

void layerNorm(float* h_input, float* h_output, float* h_gamma, float* h_beta,
               int rows, int cols, float eps) {
    float *d_input, *d_output, *d_gamma, *d_beta;
    size_t size = rows * cols * sizeof(float);
    size_t param_size = cols * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));
    CUDA_CHECK(cudaMalloc(&d_gamma, param_size));
    CUDA_CHECK(cudaMalloc(&d_beta, param_size));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma, param_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, h_beta, param_size, cudaMemcpyHostToDevice));

    dim3 grid(rows);
    dim3 block(BLOCK_SIZE);

    layerNormKernel<<<grid, block>>>(d_input, d_output, d_gamma, d_beta, rows, cols, eps);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_beta));
}

int main() {
    int rows = 1, cols = 3;
    float input[] = {1.0f, 2.0f, 3.0f};
    float gamma[] = {1.0f, 1.0f, 1.0f};
    float beta[] = {0.0f, 0.0f, 0.0f};
    float output[3];

    layerNorm(input, output, gamma, beta, rows, cols, 1e-5f);

    printf("LayerNorm output: ");
    for (int i = 0; i < cols; i++) {
        printf("%.4f ", output[i]);
    }
    printf("\n");

    return 0;
}
