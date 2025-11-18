#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_SIZE 16
#define CUDA_CHECK(call) \
    do { cudaError_t err = call; \
         if (err != cudaSuccess) { \
             fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
             exit(1); \
         } \
    } while(0)

/**
 * Quantize FP32 to INT8
 */
__global__ void quantizeKernel(float* input, int8_t* output, float scale,
                               int8_t zero_point, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx] / scale;
        val = roundf(val) + zero_point;
        val = fmaxf(-128.0f, fminf(127.0f, val));  // Clamp to INT8 range
        output[idx] = (int8_t)val;
    }
}

/**
 * INT8 GEMM with dequantization
 * C = dequantize(A @ B)
 */
__global__ void quantizedGEMM(int8_t* A, int8_t* B, float* C,
                               float scale_A, float scale_B,
                               int M, int N, int K) {
    __shared__ int8_t A_tile[TILE_SIZE][TILE_SIZE];
    __shared__ int8_t B_tile[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    int32_t sum = 0;  // INT32 accumulator to avoid overflow

    // Tiled matrix multiplication in INT8
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles
        int a_col = t * TILE_SIZE + threadIdx.x;
        if (row < M && a_col < K) {
            A_tile[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            A_tile[threadIdx.y][threadIdx.x] = 0;
        }

        int b_row = t * TILE_SIZE + threadIdx.y;
        if (b_row < K && col < N) {
            B_tile[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            B_tile[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        // Compute partial product (INT8 × INT8 = INT32)
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += (int32_t)A_tile[threadIdx.y][k] *
                   (int32_t)B_tile[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Dequantize: convert INT32 to FP32 and apply scales
    if (row < M && col < N) {
        C[row * N + col] = (float)sum * scale_A * scale_B;
    }
}

/**
 * DP4A version (4 INT8 dot products at once) - requires SM_61+
 */
#if __CUDA_ARCH__ >= 610
__global__ void quantizedGEMM_DP4A(int8_t* A, int8_t* B, float* C,
                                   float scale_A, float scale_B,
                                   int M, int N, int K) {
    // Using __dp4a for 4x throughput
    // __dp4a(a, b, c) = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3] + c

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        int32_t sum = 0;

        // Process 4 elements at a time
        for (int k = 0; k < K; k += 4) {
            if (k + 3 < K) {
                int32_t a_vec = *reinterpret_cast<int32_t*>(&A[row * K + k]);
                int32_t b_vec = *reinterpret_cast<int32_t*>(&B[k * N + col]);
                sum = __dp4a(a_vec, b_vec, sum);
            } else {
                // Handle remainder
                for (int i = k; i < K; i++) {
                    sum += (int32_t)A[row * K + i] * (int32_t)B[i * N + col];
                }
            }
        }

        C[row * N + col] = (float)sum * scale_A * scale_B;
    }
}
#endif

void quantizedMatMul(float* h_A, float* h_B, float* h_C,
                     int M, int N, int K,
                     float scale_A, float scale_B) {
    int8_t *d_A_int8, *d_B_int8;
    float *d_A_fp32, *d_B_fp32, *d_C;

    // Allocate memory
    CUDA_CHECK(cudaMalloc(&d_A_fp32, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B_fp32, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_A_int8, M * K * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_B_int8, K * N * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

    // Copy input
    CUDA_CHECK(cudaMemcpy(d_A_fp32, h_A, M * K * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_fp32, h_B, K * N * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Quantize
    int blockSize = 256;
    quantizeKernel<<<(M*K + blockSize - 1)/blockSize, blockSize>>>(
        d_A_fp32, d_A_int8, scale_A, 0, M * K);
    quantizeKernel<<<(K*N + blockSize - 1)/blockSize, blockSize>>>(
        d_B_fp32, d_B_int8, scale_B, 0, K * N);

    // INT8 GEMM
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE);

    quantizedGEMM<<<grid, block>>>(d_A_int8, d_B_int8, d_C,
                                   scale_A, scale_B, M, N, K);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result
    CUDA_CHECK(cudaMemcpy(h_C, d_C, M * N * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CHECK(cudaFree(d_A_fp32));
    CUDA_CHECK(cudaFree(d_B_fp32));
    CUDA_CHECK(cudaFree(d_A_int8));
    CUDA_CHECK(cudaFree(d_B_int8));
    CUDA_CHECK(cudaFree(d_C));
}

int main() {
    int M = 64, N = 64, K = 64;
    float *A = new float[M * K];
    float *B = new float[K * N];
    float *C = new float[M * N];

    for (int i = 0; i < M * K; i++) A[i] = 0.1f;
    for (int i = 0; i < K * N; i++) B[i] = 0.1f;

    quantizedMatMul(A, B, C, M, N, K, 0.01f, 0.01f);

    printf("INT8 GEMM result C[0] = %.6f\n", C[0]);

    delete[] A; delete[] B; delete[] C;
    return 0;
}
