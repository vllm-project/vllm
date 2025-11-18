#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_CHECK(call) \
    do { cudaError_t err = call; \
         if (err != cudaSuccess) { \
             fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
             exit(1); \
         } \
    } while(0)

/**
 * Fused element-wise kernel
 * D[i] = alpha * A[i] + beta * B[i] * C[i] + gamma
 *
 * Grid-stride loop handles arbitrary array sizes efficiently
 */
__global__ void fusedElementWise(float* A, float* B, float* C, float* D,
                                 float alpha, float beta, float gamma, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Grid-stride loop
    for (int i = idx; i < n; i += stride) {
        D[i] = alpha * A[i] + beta * B[i] * C[i] + gamma;
    }
}

/**
 * Vectorized version using float4 for better memory bandwidth
 * Requires n % 4 == 0
 */
__global__ void fusedElementWiseVectorized(float* A, float* B, float* C, float* D,
                                           float alpha, float beta, float gamma, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    float4* A4 = reinterpret_cast<float4*>(A);
    float4* B4 = reinterpret_cast<float4*>(B);
    float4* C4 = reinterpret_cast<float4*>(C);
    float4* D4 = reinterpret_cast<float4*>(D);

    int n4 = n / 4;

    for (int i = idx; i < n4; i += stride) {
        float4 a = A4[i];
        float4 b = B4[i];
        float4 c = C4[i];
        float4 d;

        d.x = alpha * a.x + beta * b.x * c.x + gamma;
        d.y = alpha * a.y + beta * b.y * c.y + gamma;
        d.z = alpha * a.z + beta * b.z * c.z + gamma;
        d.w = alpha * a.w + beta * b.w * c.w + gamma;

        D4[i] = d;
    }

    // Handle remainder
    int remainder_start = n4 * 4;
    for (int i = remainder_start + threadIdx.x; i < n; i += blockDim.x) {
        if (i < n) {
            D[i] = alpha * A[i] + beta * B[i] * C[i] + gamma;
        }
    }
}

void runFusedElementWise(float* h_A, float* h_B, float* h_C, float* h_D,
                         float alpha, float beta, float gamma, int n) {
    float *d_A, *d_B, *d_C, *d_D;
    size_t size = n * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));
    CUDA_CHECK(cudaMalloc(&d_D, size));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice));

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    numBlocks = min(numBlocks, 2048);

    fusedElementWise<<<numBlocks, blockSize>>>(d_A, d_B, d_C, d_D,
                                               alpha, beta, gamma, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_D, d_D, size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_D));
}

int main() {
    int n = 1000000;
    float *h_A = new float[n];
    float *h_B = new float[n];
    float *h_C = new float[n];
    float *h_D = new float[n];

    for (int i = 0; i < n; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
        h_C[i] = 1.0f;
    }

    runFusedElementWise(h_A, h_B, h_C, h_D, 2.0f, 3.0f, 1.0f, n);

    printf("D[0] = %f (expected 9.0)\n", h_D[0]);
    printf("D[100] = %f (expected 9.0)\n", h_D[100]);

    delete[] h_A; delete[] h_B; delete[] h_C; delete[] h_D;
    return 0;
}
