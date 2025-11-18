#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define TILE_SIZE 32
#define CUDA_CHECK(call) \
    do { cudaError_t err = call; \
         if (err != cudaSuccess) { \
             fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
             exit(1); \
         } \
    } while(0)

/**
 * Tiled matrix multiplication for Q * K^T with scaling
 * Q: (seq_len_q × d_k)
 * K: (seq_len_k × d_k)
 * Output: (seq_len_q × seq_len_k)
 */
__global__ void attentionScoresKernel(float* Q, float* K, float* scores,
                                      int seq_len_q, int seq_len_k, int d_k) {
    __shared__ float Q_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float K_tile[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    float scale = rsqrtf((float)d_k);

    // Tile across d_k dimension
    for (int t = 0; t < (d_k + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load Q tile
        int q_col = t * TILE_SIZE + threadIdx.x;
        if (row < seq_len_q && q_col < d_k) {
            Q_tile[threadIdx.y][threadIdx.x] = Q[row * d_k + q_col];
        } else {
            Q_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load K tile (note: we want K^T, so swap indices)
        int k_row = col;  // K row corresponds to output column
        int k_col = t * TILE_SIZE + threadIdx.y;
        if (k_row < seq_len_k && k_col < d_k) {
            K_tile[threadIdx.y][threadIdx.x] = K[k_row * d_k + k_col];
        } else {
            K_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += Q_tile[threadIdx.y][k] * K_tile[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write result with scaling
    if (row < seq_len_q && col < seq_len_k) {
        scores[row * seq_len_k + col] = sum * scale;
    }
}

void computeAttentionScores(float* h_Q, float* h_K, float* h_scores,
                            int seq_len_q, int seq_len_k, int d_k) {
    float *d_Q, *d_K, *d_scores;

    CUDA_CHECK(cudaMalloc(&d_Q, seq_len_q * d_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, seq_len_k * d_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scores, seq_len_q * seq_len_k * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, seq_len_q * d_k * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, seq_len_k * d_k * sizeof(float),
                          cudaMemcpyHostToDevice));

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((seq_len_k + TILE_SIZE - 1) / TILE_SIZE,
              (seq_len_q + TILE_SIZE - 1) / TILE_SIZE);

    attentionScoresKernel<<<grid, block>>>(d_Q, d_K, d_scores,
                                           seq_len_q, seq_len_k, d_k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_scores, d_scores,
                          seq_len_q * seq_len_k * sizeof(float),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_scores));
}

int main() {
    int seq_len_q = 3, seq_len_k = 2, d_k = 2;
    float Q[] = {1,0, 0,1, 1,1};
    float K[] = {1,0, 0,1};
    float scores[6];

    computeAttentionScores(Q, K, scores, seq_len_q, seq_len_k, d_k);

    printf("Attention Scores (%dx%d):\n", seq_len_q, seq_len_k);
    for (int i = 0; i < seq_len_q; i++) {
        for (int j = 0; j < seq_len_k; j++) {
            printf("%.4f ", scores[i * seq_len_k + j]);
        }
        printf("\n");
    }

    return 0;
}
