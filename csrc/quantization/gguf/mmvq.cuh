// copied and adapted from https://github.com/ggerganov/llama.cpp/blob/b2899/ggml-cuda/mmvq.cu
template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda>
static __global__ void mul_mat_vec_q(const void * __restrict__ vx, const void * __restrict__ vy, half * __restrict__ dst, const int ncols, const int nrows) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;

// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = threadIdx.x / (qi/vdr); i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs  = vdr * (threadIdx.x % (qi/vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_q_cuda(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE/2; mask > 0; mask >>= 1) {
        tmp += VLLM_SHFL_XOR_SYNC(tmp, mask);
    }

    if (threadIdx.x == 0) {
        dst[row] = __float2half(tmp);
    }
}

static void mul_mat_vec_q4_0_q8_1_cuda(const void * vx, const void * vy, half * dst, const int ncols, const int nrows, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, 1, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows);
}

static void mul_mat_vec_q4_1_q8_1_cuda(const void * vx, const void * vy, half * dst, const int ncols, const int nrows, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, 1, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<QK4_0, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows);
}

static void mul_mat_vec_q5_0_q8_1_cuda(const void * vx, const void * vy, half * dst, const int ncols, const int nrows, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, 1, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows);
}

static void mul_mat_vec_q5_1_q8_1_cuda(const void * vx, const void * vy, half * dst, const int ncols, const int nrows, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, 1, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows);
}

static void mul_mat_vec_q8_0_q8_1_cuda(const void * vx, const void * vy, half * dst, const int ncols, const int nrows, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, 1, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows);
}

static void mul_mat_vec_q2_K_q8_1_cuda(const void * vx, const void * vy, half * dst, const int ncols, const int nrows, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, 1, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows);
}

static void mul_mat_vec_q3_K_q8_1_cuda(const void * vx, const void * vy, half * dst, const int ncols, const int nrows, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, 1, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows);
}

static void mul_mat_vec_q4_K_q8_1_cuda(const void * vx, const void * vy, half * dst, const int ncols, const int nrows, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, 1, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows);
}

static void mul_mat_vec_q5_K_q8_1_cuda(const void * vx, const void * vy, half * dst, const int ncols, const int nrows, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, 1, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows);
}

static void mul_mat_vec_q6_K_q8_1_cuda(const void * vx, const void * vy, half * dst, const int ncols, const int nrows, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, 1, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows);
}

static void mul_mat_vec_iq2_xxs_q8_1_cuda(const void * vx, const void * vy, half * dst, const int ncols, const int nrows, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, 1, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<QK_K, QI2_XXS, block_iq2_xxs, 1, vec_dot_iq2_xxs_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows);
}

static void mul_mat_vec_iq2_xs_q8_1_cuda(const void * vx, const void * vy, half * dst, const int ncols, const int nrows, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, 1, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<QK_K, QI2_XS, block_iq2_xs, 1, vec_dot_iq2_xs_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows);
}

static void mul_mat_vec_iq2_s_q8_1_cuda(const void * vx, const void * vy, half * dst, const int ncols, const int nrows, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, 1, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<QK_K, QI2_S, block_iq2_s, 1, vec_dot_iq2_s_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows);
}

static void mul_mat_vec_iq3_xxs_q8_1_cuda(const void * vx, const void * vy, half * dst, const int ncols, const int nrows, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, 1, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<QK_K, QI3_XXS, block_iq3_xxs, 1, vec_dot_iq3_xxs_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows);
}

static void mul_mat_vec_iq1_s_q8_1_cuda(const void * vx, const void * vy, half * dst, const int ncols, const int nrows, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, 1, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<QK_K, QI1_S, block_iq1_s, 1, vec_dot_iq1_s_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows);
}

static void mul_mat_vec_iq1_m_q8_1_cuda(const void * vx, const void * vy, half * dst, const int ncols, const int nrows, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, 1, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<QK_K, QI1_M, block_iq1_m, 1, vec_dot_iq1_m_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows);
}

static void mul_mat_vec_iq4_nl_q8_1_cuda(const void * vx, const void * vy, half * dst, const int ncols, const int nrows, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, 1, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<QK4_NL, QI4_NL, block_iq4_nl, VDR_Q4_0_Q8_1_MMVQ, vec_dot_iq4_nl_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows);
}

static void mul_mat_vec_iq4_xs_q8_1_cuda(const void * vx, const void * vy, half * dst, const int ncols, const int nrows, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, 1, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<QK_K, QI4_XS, block_iq4_xs, 1, vec_dot_iq4_xs_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows);
}

static void mul_mat_vec_iq3_s_q8_1_cuda(const void * vx, const void * vy, half * dst, const int ncols, const int nrows, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, 1, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<QK_K, QI3_XS, block_iq3_s, 1, vec_dot_iq3_s_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows);
}
