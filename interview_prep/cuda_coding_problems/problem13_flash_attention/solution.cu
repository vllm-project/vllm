#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define BLOCK_SIZE 32
#define CUDA_CHECK(call) \
    do { cudaError_t err = call; \
         if (err != cudaSuccess) { \
             fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
             exit(1); \
         } \
    } while(0)

/**
 * Simplified Flash Attention kernel
 * Computes attention in blocks using online softmax
 *
 * Key idea:
 * - Process K,V in chunks
 * - Maintain running max, sum, and output
 * - Update incrementally (online softmax)
 */
__global__ void flashAttentionKernel(float* Q, float* K, float* V, float* O,
                                     int seq_len, int d, int block_size) {
    extern __shared__ float shared_mem[];

    // Shared memory layout
    float* Q_tile = shared_mem;  // block_size × d
    float* K_tile = Q_tile + block_size * d;  // block_size × d
    float* V_tile = K_tile + block_size * d;  // block_size × d
    float* scores = V_tile + block_size * d;  // block_size × block_size

    int q_start = blockIdx.x * block_size;
    int tid = threadIdx.x;

    // Each block processes block_size queries
    if (q_start + tid >= seq_len) return;

    // Initialize output and normalization factors
    float output[64];  // Assuming d <= 64
    float row_max = -INFINITY;
    float row_sum = 0.0f;

    for (int i = 0; i < d; i++) {
        output[i] = 0.0f;
    }

    // Load Q tile
    for (int i = 0; i < d; i++) {
        if (q_start + tid < seq_len) {
            Q_tile[tid * d + i] = Q[(q_start + tid) * d + i];
        }
    }
    __syncthreads();

    // Process K, V in blocks (online softmax)
    for (int k_start = 0; k_start < seq_len; k_start += block_size) {
        // Load K, V tiles
        for (int i = tid; i < block_size * d; i += blockDim.x) {
            int local_idx = i / d;
            int dim_idx = i % d;
            int global_idx = k_start + local_idx;

            if (global_idx < seq_len) {
                K_tile[i] = K[global_idx * d + dim_idx];
                V_tile[i] = V[global_idx * d + dim_idx];
            } else {
                K_tile[i] = 0.0f;
                V_tile[i] = 0.0f;
            }
        }
        __syncthreads();

        // Compute attention scores for this block
        if (tid < block_size && q_start + tid < seq_len) {
            for (int j = 0; j < block_size; j++) {
                if (k_start + j < seq_len) {
                    float score = 0.0f;
                    for (int i = 0; i < d; i++) {
                        score += Q_tile[tid * d + i] * K_tile[j * d + i];
                    }
                    score /= sqrtf((float)d);
                    scores[tid * block_size + j] = score;

                    // Online softmax: update max
                    float old_max = row_max;
                    row_max = fmaxf(row_max, score);

                    // Rescale previous output and sum
                    if (old_max != -INFINITY) {
                        float scale = expf(old_max - row_max);
                        row_sum *= scale;
                        for (int i = 0; i < d; i++) {
                            output[i] *= scale;
                        }
                    }

                    // Add contribution from this element
                    float exp_score = expf(score - row_max);
                    row_sum += exp_score;

                    for (int i = 0; i < d; i++) {
                        output[i] += exp_score * V_tile[j * d + i];
                    }
                }
            }
        }
        __syncthreads();
    }

    // Final normalization
    if (tid < block_size && q_start + tid < seq_len) {
        for (int i = 0; i < d; i++) {
            O[(q_start + tid) * d + i] = output[i] / row_sum;
        }
    }
}

void flashAttention(float* h_Q, float* h_K, float* h_V, float* h_O,
                   int seq_len, int d) {
    float *d_Q, *d_K, *d_V, *d_O;
    size_t size = seq_len * d * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_Q, size));
    CUDA_CHECK(cudaMalloc(&d_K, size));
    CUDA_CHECK(cudaMalloc(&d_V, size));
    CUDA_CHECK(cudaMalloc(&d_O, size));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, size, cudaMemcpyHostToDevice));

    int block_size = min(BLOCK_SIZE, seq_len);
    int numBlocks = (seq_len + block_size - 1) / block_size;

    size_t shared_mem = (3 * block_size * d + block_size * block_size) * sizeof(float);

    flashAttentionKernel<<<numBlocks, block_size, shared_mem>>>(
        d_Q, d_K, d_V, d_O, seq_len, d, block_size);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_O, d_O, size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));
}

int main() {
    int seq_len = 64;
    int d = 8;

    float *Q = new float[seq_len * d];
    float *K = new float[seq_len * d];
    float *V = new float[seq_len * d];
    float *O = new float[seq_len * d];

    // Initialize with simple values
    for (int i = 0; i < seq_len * d; i++) {
        Q[i] = 0.1f;
        K[i] = 0.1f;
        V[i] = 1.0f;
    }

    flashAttention(Q, K, V, O, seq_len, d);

    printf("Flash Attention output[0]: %.4f\n", O[0]);

    delete[] Q; delete[] K; delete[] V; delete[] O;
    return 0;
}
