#include <cstdint>

template <int qk, int qr, int qi, bool need_sum, typename block_q_t, int mmq_x,
          int mmq_y, int nwarps, allocate_tiles_cuda_t allocate_tiles,
          load_tiles_cuda_t load_tiles, int vdr,
          vec_dot_q_mul_mat_cuda_t vec_dot>
static __device__ __forceinline__ void moe_q(
    const void* __restrict__ vx, const void* __restrict__ vy,
    half* __restrict__ dst, const int* __restrict__ sorted_token_ids,
    const int* __restrict__ expert_ids, const int exp_stride, const int ncols_x,
    const int nrows_x, const int ncols_y, const int nrows_y,
    const int nrows_dst, const int top_k) {
  const int blocks_per_row_x = ncols_x / qk;
  const int blocks_per_col_y = nrows_y / QK8_1;
  const int blocks_per_warp = WARP_SIZE_GGUF / qi;

  const int ncols_dst = ncols_y * top_k;

  const int row_dst_0 = blockIdx.x * mmq_y;
  const int& row_x_0 = row_dst_0;

  const int col_dst_0 = blockIdx.y * mmq_x;

  int token_ids[mmq_x / nwarps];
  int token_offs[mmq_x / nwarps];
  for (int i = 0; i < mmq_x; i += nwarps) {
    token_offs[i / nwarps] = sorted_token_ids[col_dst_0 + threadIdx.y + i];
    // printf("thread %d/%d, %d tok id %d, tok offset %d \n",
    //         threadIdx.x, threadIdx.y, i / nwarps,
    //        token_ids[i / nwarps], token_offs[i / nwarps]);
  }

  for (int i = 0; i < mmq_x; i += nwarps * QI8_1) {
    const int ids =
        (i + threadIdx.y * QI8_1 + threadIdx.x / (WARP_SIZE_GGUF / QI8_1)) %
        mmq_x;
    token_ids[i / (nwarps * QI8_1)] = token_offs[ids] / top_k;
  }
  const int exp_idx = expert_ids[blockIdx.y];
  if (exp_idx > 255 || exp_idx < 0) return;

  // if ((blockIdx.y == 119 || blockIdx.y == 119)&& blockIdx.x == 9 &&
  // threadIdx.x == 3 && threadIdx.y == 3)
  //     if (threadIdx.x == 0 && threadIdx.y == 0)
  //     {
  //     printf(
  //         "running kernel for expert id %d, row_dst %d \
// blocks per warp %d \n",
  //         exp_idx, row_dst_0, blocks_per_warp);
  //     for (int i = 0; i < mmq_x; i += nwarps) {
  //       printf("block %d, %d/%d, %d tok id %d, tok offset %d
  //       \n",blockIdx.y,threadIdx.x, threadIdx.y, i / nwarps,
  //              token_ids[i / nwarps], token_offs[i / nwarps]);
  //     }
  //   }

  const block_q_t* x = (const block_q_t*)((char*)vx + exp_idx * exp_stride);
  const block_q8_1* y = (const block_q8_1*)(vy);

  int* tile_x_ql = nullptr;
  half2* tile_x_dm = nullptr;
  int* tile_x_qh = nullptr;
  int* tile_x_sc = nullptr;

  allocate_tiles(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc);

  __shared__ int tile_y_qs[mmq_x * WARP_SIZE_GGUF];
  __shared__ half2 tile_y_ds[mmq_x * WARP_SIZE_GGUF / QI8_1];

  float sum[mmq_y / WARP_SIZE_GGUF][mmq_x / nwarps] = {{0.0f}};

  for (int ib0 = 0; ib0 < blocks_per_row_x; ib0 += blocks_per_warp) {
    // if ((blockIdx.y == 0 || blockIdx.y == 1)&& blockIdx.x == 0 &&
    // (threadIdx.x == 0 || threadIdx.x == 0))
    // {
    //   printf("stepping tile");
    // }
    load_tiles(x + row_x_0 * blocks_per_row_x + ib0, tile_x_ql, tile_x_dm,
               tile_x_qh, tile_x_sc, threadIdx.y, nrows_x - row_x_0 - 1,
               threadIdx.x, blocks_per_row_x);

#pragma unroll
    for (int ir = 0; ir < qr; ++ir) {
      const int kqs = ir * WARP_SIZE_GGUF + threadIdx.x;
      const int kbxd = kqs / QI8_1;

#pragma unroll
      for (int i = 0; i < mmq_x; i += nwarps) {
        // const int col_y_eff = min(col_y_0 + threadIdx.y + i, ncols_y-1); //
        // to prevent out-of-bounds memory accesses
        const int col_y_eff =
            min(token_ids[i / nwarps],
                ncols_y - 1);  // to prevent out-of-bounds memory accesses
        const block_q8_1* by0 =
            &y[col_y_eff * blocks_per_col_y + ib0 * (qk / QK8_1) + kbxd];
        const int index_y =
            (threadIdx.y + i) * WARP_SIZE_GGUF + kqs % WARP_SIZE_GGUF;
        tile_y_qs[index_y] =
            get_int_from_int8_aligned(by0->qs, threadIdx.x % QI8_1);
        // if (blockIdx.x == 0 && blockIdx.y == 0 && ib0 == 0 && ir == 0 && i ==
        // 0)
        // {
        //     printf("thread %d/%d loading data from col eff %d, idx %d, val
        //     %d\n",
        //             threadIdx.x, threadIdx.y, col_y_eff, index_y,
        //             tile_y_qs[index_y]);
        // }
      }

#pragma unroll
      for (int ids0 = 0; ids0 < mmq_x; ids0 += nwarps * QI8_1) {
        const int ids = (ids0 + threadIdx.y * QI8_1 +
                         threadIdx.x / (WARP_SIZE_GGUF / QI8_1)) %
                        mmq_x;
        const int kby = threadIdx.x % (WARP_SIZE_GGUF / QI8_1);
        const int col_y_eff = min(token_ids[ids], ncols_y - 1);
        // if ((blockIdx.y == 0 || blockIdx.y == 1)&& blockIdx.x == 0 &&
        // (threadIdx.x == 0 || threadIdx.x == 0)) if ((blockIdx.y == 0 ||
        // blockIdx.y == 0)&& blockIdx.x == 0)
        // {
        //     printf("block %d thread %d/%d, ids %d loading next scale %d, %d,
        //     %d, %d, %d, to %d/%d\n",
        //             blockIdx.y,threadIdx.x, threadIdx.y, ids, col_y_eff, kby,
        //             ib0, ir, col_y_eff * blocks_per_col_y + ib0 * (qk /
        //             QK8_1) + ir * (WARP_SIZE_GGUF / QI8_1) + kby, ids *
        //             (WARP_SIZE_GGUF / QI8_1) + kby,
        //             mmq_x*WARP_SIZE_GGUF/QI8_1
        //             );
        // }
        // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 &&
        // threadIdx.y == 0)
        // {
        //     printf("loading scale from col eff %d\n", col_y_eff);
        // }
        // const int col_y_eff = min(col_y_0 + threadIdx.y + i, ncols_y-1); //
        // to prevent out-of-bounds memory accesses const int col_y_eff =
        // min(token_ids[i/nwarps], ncols_y-1); // to prevent out-of-bounds
        // memory accesses

        // if the sum is not needed it's faster to transform the scale to f32
        // ahead of time
        const half2* dsi_src =
            &y[col_y_eff * blocks_per_col_y + ib0 * (qk / QK8_1) +
               ir * (WARP_SIZE_GGUF / QI8_1) + kby]
                 .ds;
        half2* dsi_dst =
            &tile_y_ds[threadIdx.y * (WARP_SIZE_GGUF / QI8_1) + kby];
        // half2* dsi_dst = &tile_y_ds[threadIdx.y];
        if (need_sum) {
          *dsi_dst = *dsi_src;
        } else {
          float* dfi_dst = (float*)dsi_dst;
          *dfi_dst = __low2float(*dsi_src);
        }
      }

      __syncthreads();

      // #pragma unroll // unrolling this loop causes too much register pressure
      for (int k = ir * WARP_SIZE_GGUF / qr; k < (ir + 1) * WARP_SIZE_GGUF / qr;
           k += vdr) {
#pragma unroll
        for (int j = 0; j < mmq_x; j += nwarps) {
#pragma unroll
          for (int i = 0; i < mmq_y; i += WARP_SIZE_GGUF) {
            sum[i / WARP_SIZE_GGUF][j / nwarps] +=
                vec_dot(tile_x_ql, tile_x_dm, tile_x_qh, tile_x_sc, tile_y_qs,
                        tile_y_ds, threadIdx.x + i, threadIdx.y + j, k);
            // if ((blockIdx.y == 0 || blockIdx.y == 1)&& blockIdx.x == 0 &&
            // (threadIdx.x == 0 || threadIdx.x == 0))
            // {
            //     printf("block %d thread %d/%d doing next sum %f \n",
            //     blockIdx.y,threadIdx.x, threadIdx.y, sum[i /
            //     WARP_SIZE_GGUF][j / nwarps]);
            // }
          }
        }
      }
      __syncthreads();
    }
  }

#pragma unroll
  for (int j = 0; j < mmq_x; j += nwarps) {
    // const int col_dst = col_dst_0 + j + threadIdx.y;
    const int col_dst = token_offs[j / nwarps];
    if (col_dst >= ncols_dst) {
      return;
    }

#pragma unroll
    for (int i = 0; i < mmq_y; i += WARP_SIZE_GGUF) {
      const int row_dst = row_dst_0 + threadIdx.x + i;
      if (row_dst >= nrows_dst) {
        continue;
      }
      // if (blockIdx.x == 0 && blockIdx.y == 0) {
      //   printf("thread %d/%d saving %f tocol %d, row %d\n", threadIdx.x,
      //   threadIdx.y,
      //          sum[i / WARP_SIZE_GGUF][j / nwarps], col_dst, row_dst);
      // }
      dst[col_dst * nrows_dst + row_dst] =
          __float2half(sum[i / WARP_SIZE_GGUF][j / nwarps]);
    }
  }
}

#if defined(USE_ROCM)
  #define MMQ_X_Q2_K 64
  #define MMQ_Y_Q2_K 128
  #define NWARPS_Q2_K 8
#else
  #define MMQ_X_Q2_K 4
  #define MMQ_Y_Q2_K 32
  #define NWARPS_Q2_K 4
#endif

template <bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q2_K, 2)
#endif
    moe_q2_K(const void* __restrict__ vx, const void* __restrict__ vy,
             half* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int exp_stride, const int ncols_x,
             const int nrows_x, const int ncols_y, const int nrows_y,
             const int nrows_dst, const int top_k) {
  const int mmq_x = MMQ_X_Q2_K;
  const int mmq_y = MMQ_Y_Q2_K;
  const int nwarps = NWARPS_Q2_K;

  moe_q<QK_K, QR2_K, QI2_K, false, block_q2_K, mmq_x, mmq_y, nwarps,
        allocate_tiles_q2_K<mmq_y>, load_tiles_q2_K<mmq_y, nwarps, need_check>,
        VDR_Q2_K_Q8_1_MMQ, vec_dot_q2_K_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, exp_stride, ncols_x, nrows_x,
      ncols_y, nrows_y, nrows_dst, top_k);
}

static void ggml_moe_q2_K_q8_1_cuda(const void* inp, const void* w, half* dst,
                                    const int* sorted_token_ids,
                                    const int* expert_ids, const int exp_stride,
                                    const int ncols_x, const int nrows_x,
                                    const int ncols_y, const int nrows_y,
                                    const int nrows_dst, const int top_k,
                                    const int tokens_post_padded,
                                    cudaStream_t stream) {
  const int mmq_x = MMQ_X_Q2_K;
  const int mmq_y = MMQ_Y_Q2_K;
  const int nwarps = NWARPS_Q2_K;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  // const int block_num_y = (ncols_y * top_k + mmq_x - 1) / mmq_x;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);
  // printf("running block size %d %d, grid size %d %d \n", WARP_SIZE_GGUF,
  // nwarps, block_num_x, block_num_y);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_q2_K<need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, exp_stride, ncols_x, nrows_x,
        ncols_y, nrows_y, nrows_dst, top_k);
  } else {
    constexpr bool need_check = true;
    moe_q2_K<need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, exp_stride, ncols_x, nrows_x,
        ncols_y, nrows_y, nrows_dst, top_k);
  }
}

#if defined(USE_ROCM)
  #define MMQ_X_Q3_K 64
  #define MMQ_Y_Q3_K 128
  #define NWARPS_Q3_K 8
#else
  #define MMQ_X_Q3_K 4
  #define MMQ_Y_Q3_K 32
  #define NWARPS_Q3_K 4
#endif

template <bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q3_K, 2)
#endif
    moe_q3_K(const void* __restrict__ vx, const void* __restrict__ vy,
             half* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int exp_stride, const int ncols_x,
             const int nrows_x, const int ncols_y, const int nrows_y,
             const int nrows_dst, const int top_k) {

  const int mmq_x = MMQ_X_Q3_K;
  const int mmq_y = MMQ_Y_Q3_K;
  const int nwarps = NWARPS_Q3_K;

  moe_q<QK_K, QR3_K, QI3_K, false, block_q3_K, mmq_x, mmq_y, nwarps,
        allocate_tiles_q3_K<mmq_y>, load_tiles_q3_K<mmq_y, nwarps, need_check>,
        VDR_Q3_K_Q8_1_MMQ, vec_dot_q3_K_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, exp_stride, ncols_x, nrows_x,
      ncols_y, nrows_y, nrows_dst, top_k);
}

static void ggml_moe_q3_K_q8_1_cuda(const void* inp, const void* w, half* dst,
                                    const int* sorted_token_ids,
                                    const int* expert_ids, const int exp_stride,
                                    const int ncols_x, const int nrows_x,
                                    const int ncols_y, const int nrows_y,
                                    const int nrows_dst, const int top_k,
                                    const int tokens_post_padded,
                                    cudaStream_t stream) {
  const int mmq_x = MMQ_X_Q3_K;
  const int mmq_y = MMQ_Y_Q3_K;
  const int nwarps = NWARPS_Q3_K;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  // const int block_num_y = (ncols_y * top_k + mmq_x - 1) / mmq_x;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);
  // printf("running block size %d %d, grid size %d %d \n", WARP_SIZE_GGUF,
  // nwarps, block_num_x, block_num_y);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_q3_K<need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, exp_stride, ncols_x, nrows_x,
        ncols_y, nrows_y, nrows_dst, top_k);
  } else {
    constexpr bool need_check = true;
    moe_q3_K<need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, exp_stride, ncols_x, nrows_x,
        ncols_y, nrows_y, nrows_dst, top_k);
  }
}
