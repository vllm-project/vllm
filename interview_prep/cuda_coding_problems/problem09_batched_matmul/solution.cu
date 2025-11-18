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
 * Batched matrix multiplication: C[b] = A[b] @ B[b]
 * Each batch processes independently
 */
__global__ void batchedMatMulKernel(float* A, float* B, float* C,
                                    int batch_size, int M, int K, int N) {
    __shared__ float A_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float B_tile[TILE_SIZE][TILE_SIZE];

    int batch = blockIdx.z;
    if (batch >= batch_size) return;

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Offset for this batch
    int A_offset = batch * M * K;
    int B_offset = batch * K * N;
    int C_offset = batch * M * N;

    // Tile across K dimension
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load A tile
        int a_col = t * TILE_SIZE + threadIdx.x;
        if (row < M && a_col < K) {
            A_tile[threadIdx.y][threadIdx.x] = A[A_offset + row * K + a_col];
        } else {
            A_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load B tile
        int b_row = t * TILE_SIZE + threadIdx.y;
        if (b_row < K && col < N) {
            B_tile[threadIdx.y][threadIdx.x] = B[B_offset + b_row * N + col];
        } else {
            B_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial result
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += A_tile[threadIdx.y][k] * B_tile[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        C[C_offset + row * N + col] = sum;
    }
}

void batchedMatMul(float* h_A, float* h_B, float* h_C,
                   int batch_size, int M, int K, int N) {
    float *d_A, *d_B, *d_C;

    size_t A_size = batch_size * M * K * sizeof(float);
    size_t B_size = batch_size * K * N * sizeof(float);
    size_t C_size = batch_size * M * N * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_A, A_size));
    CUDA_CHECK(cudaMalloc(&d_B, B_size));
    CUDA_CHECK(cudaMalloc(&d_C, C_size));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, A_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, B_size, cudaMemcpyHostToDevice));

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE,
              batch_size);

    batchedMatMulKernel<<<grid, block>>>(d_A, d_B, d_C, batch_size, M, K, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C, d_C, C_size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

int main() {
    int batch = 2, M = 2, K = 2, N = 2;
    float A[] = {1,2,3,4, 5,6,7,8};
    float B[] = {1,0,0,1, 1,1,1,1};
    float C[8];

    batchedMatMul(A, B, C, batch, M, K, N);

    printf("Batched MatMul Results:\n");
    for (int b = 0; b < batch; b++) {
        printf("Batch %d:\n", b);
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                printf("%.0f ", C[b*M*N + i*N + j]);
            }
            printf("\n");
        }
    }

    return 0;
}
