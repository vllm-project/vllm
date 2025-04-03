// copied from https://github.com/ggerganov/llama.cpp/blob/b2899/ggml-cuda/mmq.cu
template <typename scalar_t, int qk, int qr, int qi, bool need_sum, typename block_q_t, int mmq_x, int mmq_y, int nwarps,
              allocate_tiles_cuda_t allocate_tiles, load_tiles_cuda_t load_tiles, int vdr, vec_dot_q_mul_mat_cuda_t vec_dot>
static __device__ __forceinline__ void mul_mat_q(
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    const int blocks_per_row_x = ncols_x / qk;
    const int blocks_per_col_y = nrows_y / QK8_1;
    const int blocks_per_warp = WARP_SIZE_GGUF / qi;

    const int & ncols_dst = ncols_y;

    const auto row_dst_0 = blockIdx.x*mmq_y;
    const int & row_x_0 = row_dst_0;

    const auto col_dst_0 = blockIdx.y*mmq_x;
    const int & col_y_0 = col_dst_0;

    int   * tile_x_ql = nullptr;
    half2 * tile_x_dm = nullptr;
    int   * tile_x_qh = nullptr;
    int   * tile_x_sc = nullptr;

    allocate_tiles(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc);

    __shared__ int    tile_y_qs[mmq_x * WARP_SIZE_GGUF];
    __shared__ half2  tile_y_ds[mmq_x * WARP_SIZE_GGUF/QI8_1];

    float sum[mmq_y/WARP_SIZE_GGUF][mmq_x/nwarps] = {{0.0f}};

    for (int ib0 = 0; ib0 < blocks_per_row_x; ib0 += blocks_per_warp) {

        load_tiles(x + row_x_0*blocks_per_row_x + ib0, tile_x_ql, tile_x_dm, tile_x_qh, tile_x_sc,
                   threadIdx.y, nrows_x-row_x_0-1, threadIdx.x, blocks_per_row_x);

#pragma unroll
        for (int ir = 0; ir < qr && ib0 + ir * blocks_per_warp/qr < blocks_per_row_x; ++ir) {
            const auto kqs = ir*WARP_SIZE_GGUF + threadIdx.x;
            const int kbxd = kqs / QI8_1;

#pragma unroll
            for (int i = 0; i < mmq_x; i += nwarps) {
                const int col_y_eff = min(col_y_0 + threadIdx.y + i, ncols_y-1); // to prevent out-of-bounds memory accesses
                const block_q8_1 * by0 = &y[col_y_eff*blocks_per_col_y + ib0 * (qk/QK8_1) + kbxd];
                const int index_y = (threadIdx.y + i) * WARP_SIZE_GGUF + kqs % WARP_SIZE_GGUF;
                tile_y_qs[index_y] = get_int_from_int8_aligned(by0->qs, threadIdx.x % QI8_1);
            }

#pragma unroll
            for (int ids0 = 0; ids0 < mmq_x; ids0 += nwarps * QI8_1) {
                const int ids = (ids0 + threadIdx.y * QI8_1 + threadIdx.x / (WARP_SIZE_GGUF/QI8_1)) % mmq_x;
                const auto kby = threadIdx.x % (WARP_SIZE_GGUF/QI8_1);
                const int col_y_eff = min(col_y_0 + ids, ncols_y-1);

                // if the sum is not needed it's faster to transform the scale to f32 ahead of time
                const half2 * dsi_src = &y[col_y_eff*blocks_per_col_y + ib0 * (qk/QK8_1) + ir*(WARP_SIZE_GGUF/QI8_1) + kby].ds;
                half2       * dsi_dst = &tile_y_ds[ids * (WARP_SIZE_GGUF/QI8_1) + kby];
                if (need_sum) {
                    *dsi_dst = *dsi_src;
                } else {
                    float * dfi_dst = (float *) dsi_dst;
                    *dfi_dst = __low2float(*dsi_src);
                }
            }

            __syncthreads();

// #pragma unroll // unrolling this loop causes too much register pressure
            for (int k = ir*WARP_SIZE_GGUF/qr; k < (ir+1)*WARP_SIZE_GGUF/qr; k += vdr) {
#pragma unroll
                for (int j = 0; j < mmq_x; j += nwarps) {
#pragma unroll
                    for (int i = 0; i < mmq_y; i += WARP_SIZE_GGUF) {
                        sum[i/WARP_SIZE_GGUF][j/nwarps] += vec_dot(
                            tile_x_ql, tile_x_dm, tile_x_qh, tile_x_sc, tile_y_qs, tile_y_ds,
                            threadIdx.x + i, threadIdx.y + j, k);
                    }
                }
            }
            __syncthreads();
        }
    }

#pragma unroll
    for (int j = 0; j < mmq_x; j += nwarps) {
        const auto col_dst = col_dst_0 + j + threadIdx.y;
        if (col_dst >= ncols_dst) {
            return;
        }

#pragma unroll
        for (int i = 0; i < mmq_y; i += WARP_SIZE_GGUF) {
            const auto row_dst = row_dst_0 + threadIdx.x + i;
            if (row_dst >= nrows_dst) {
                continue;
            }
            dst[col_dst*nrows_dst + row_dst] = sum[i/WARP_SIZE_GGUF][j/nwarps];
        }
    }
}

#if defined(USE_ROCM)
#define  MMQ_X_Q4_0  64
#define  MMQ_Y_Q4_0  128
#define NWARPS_Q4_0  8
#else
#define  MMQ_X_Q4_0 4
#define  MMQ_Y_Q4_0 32
#define NWARPS_Q4_0 4
#endif

template<typename scalar_t, bool need_check> static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF*NWARPS_Q4_0, 2)
#endif
mul_mat_q4_0(
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q4_0;
    const int mmq_y  =  MMQ_Y_Q4_0;
    const int nwarps = NWARPS_Q4_0;

    mul_mat_q<scalar_t, QK4_0, QR4_0, QI4_0, true, block_q4_0, mmq_x, mmq_y, nwarps, allocate_tiles_q4_0<mmq_y>,
        load_tiles_q4_0<mmq_y, nwarps, need_check>, VDR_Q4_0_Q8_1_MMQ, vec_dot_q4_0_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

template<typename scalar_t>
static void ggml_mul_mat_q4_0_q8_1_cuda(
    const void * vx, const void * vy, scalar_t * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    int mmq_x  =  MMQ_X_Q4_0;
    int mmq_y  =  MMQ_Y_Q4_0;
    int nwarps = NWARPS_Q4_0;

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q4_0<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q4_0<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

#if defined(USE_ROCM)
#define  MMQ_X_Q4_1 64
#define  MMQ_Y_Q4_1 128
#define NWARPS_Q4_1 8
#else
#define  MMQ_X_Q4_1 4
#define  MMQ_Y_Q4_1 32
#define NWARPS_Q4_1 4
#endif

template<typename scalar_t, bool need_check> static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF*NWARPS_Q4_1, 2)
#endif
mul_mat_q4_1(
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q4_1;
    const int mmq_y  =  MMQ_Y_Q4_1;
    const int nwarps = NWARPS_Q4_1;

    mul_mat_q<scalar_t, QK4_1, QR4_1, QI4_1, true, block_q4_1, mmq_x, mmq_y, nwarps, allocate_tiles_q4_1<mmq_y>,
        load_tiles_q4_1<mmq_y, nwarps, need_check>, VDR_Q4_1_Q8_1_MMQ, vec_dot_q4_1_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

template<typename scalar_t>
static void ggml_mul_mat_q4_1_q8_1_cuda(
    const void * vx, const void * vy, scalar_t * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    int mmq_x  =  MMQ_X_Q4_1;
    int mmq_y  =  MMQ_Y_Q4_1;
    int nwarps = NWARPS_Q4_1;

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q4_1<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q4_1<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

#if defined(USE_ROCM)
#define  MMQ_X_Q5_0 64
#define  MMQ_Y_Q5_0 128
#define NWARPS_Q5_0 8
#else
#define  MMQ_X_Q5_0 4
#define  MMQ_Y_Q5_0 32
#define NWARPS_Q5_0 4
#endif

template<typename scalar_t, bool need_check> static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF*NWARPS_Q5_0, 2)
#endif
mul_mat_q5_0(
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q5_0;
    const int mmq_y  =  MMQ_Y_Q5_0;
    const int nwarps = NWARPS_Q5_0;

    mul_mat_q<scalar_t, QK5_0, QR5_0, QI5_0, false, block_q5_0, mmq_x, mmq_y, nwarps, allocate_tiles_q5_0<mmq_y>,
        load_tiles_q5_0<mmq_y, nwarps, need_check>, VDR_Q5_0_Q8_1_MMQ, vec_dot_q5_0_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

template<typename scalar_t>
static void ggml_mul_mat_q5_0_q8_1_cuda(
    const void * vx, const void * vy, scalar_t * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    const int mmq_x  =  MMQ_X_Q5_0;
    const int mmq_y  =  MMQ_Y_Q5_0;
    const int nwarps = NWARPS_Q5_0;

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q5_0<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q5_0<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

#if defined(USE_ROCM)
#define  MMQ_X_Q5_1 64
#define  MMQ_Y_Q5_1 128
#define NWARPS_Q5_1 8
#else
#define  MMQ_X_Q5_1 4
#define  MMQ_Y_Q5_1 32
#define NWARPS_Q5_1 4
#endif

template<typename scalar_t, bool need_check> static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF*NWARPS_Q5_1, 2)
#endif
mul_mat_q5_1(
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q5_1;
    const int mmq_y  =  MMQ_Y_Q5_1;
    const int nwarps = NWARPS_Q5_1;

    mul_mat_q<scalar_t, QK5_1, QR5_1, QI5_1, true, block_q5_1, mmq_x, mmq_y, nwarps, allocate_tiles_q5_1<mmq_y>,
        load_tiles_q5_1<mmq_y, nwarps, need_check>, VDR_Q5_1_Q8_1_MMQ, vec_dot_q5_1_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

template<typename scalar_t>
static void ggml_mul_mat_q5_1_q8_1_cuda(
    const void * vx, const void * vy, scalar_t * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {
    const int mmq_x  =  MMQ_X_Q5_1;
    const int mmq_y  =  MMQ_Y_Q5_1;
    const int nwarps = NWARPS_Q5_1;

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q5_1<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q5_1<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

#if defined(USE_ROCM)
#define  MMQ_X_Q8_0 64
#define  MMQ_Y_Q8_0 128
#define NWARPS_Q8_0 8
#else
#define  MMQ_X_Q8_0 4
#define  MMQ_Y_Q8_0 32
#define NWARPS_Q8_0 4
#endif

template<typename scalar_t, bool need_check> static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF*NWARPS_Q8_0, 2)
#endif
mul_mat_q8_0(
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q8_0;
    const int mmq_y  =  MMQ_Y_Q8_0;
    const int nwarps = NWARPS_Q8_0;

    mul_mat_q<scalar_t, QK8_0, QR8_0, QI8_0, false, block_q8_0, mmq_x, mmq_y, nwarps, allocate_tiles_q8_0<mmq_y>,
        load_tiles_q8_0<mmq_y, nwarps, need_check>, VDR_Q8_0_Q8_1_MMQ, vec_dot_q8_0_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

template<typename scalar_t>
static void ggml_mul_mat_q8_0_q8_1_cuda(
    const void * vx, const void * vy, scalar_t * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {
    const int mmq_x  =  MMQ_X_Q8_0;
    const int mmq_y  =  MMQ_Y_Q8_0;
    const int nwarps = NWARPS_Q8_0;

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q8_0<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q8_0<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

#if defined(USE_ROCM)
#define  MMQ_X_Q2_K 64
#define  MMQ_Y_Q2_K 128
#define NWARPS_Q2_K 8
#else
#define  MMQ_X_Q2_K 4
#define  MMQ_Y_Q2_K 32
#define NWARPS_Q2_K 4
#endif

template<typename scalar_t, bool need_check> static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF*NWARPS_Q2_K, 2)
#endif
mul_mat_q2_K(
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q2_K;
    const int mmq_y  =  MMQ_Y_Q2_K;
    const int nwarps = NWARPS_Q2_K;

    mul_mat_q<scalar_t, QK_K, QR2_K, QI2_K, false, block_q2_K, mmq_x, mmq_y, nwarps, allocate_tiles_q2_K<mmq_y>,
        load_tiles_q2_K<mmq_y, nwarps, need_check>, VDR_Q2_K_Q8_1_MMQ, vec_dot_q2_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

template<typename scalar_t>
static void ggml_mul_mat_q2_K_q8_1_cuda(
    const void * vx, const void * vy, scalar_t * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {
    const int mmq_x  =  MMQ_X_Q2_K;
    const int mmq_y  =  MMQ_Y_Q2_K;
    const int nwarps = NWARPS_Q2_K;

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q2_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q2_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

#if defined(USE_ROCM)
#define  MMQ_X_Q3_K 64
#define  MMQ_Y_Q3_K 128
#define NWARPS_Q3_K 8
#else
#define  MMQ_X_Q3_K 4
#define  MMQ_Y_Q3_K 32
#define NWARPS_Q3_K 4
#endif

template<typename scalar_t, bool need_check> static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF*NWARPS_Q3_K, 2)
#endif
mul_mat_q3_K(
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {

    const int mmq_x  =  MMQ_X_Q3_K;
    const int mmq_y  =  MMQ_Y_Q3_K;
    const int nwarps = NWARPS_Q3_K;

    mul_mat_q<scalar_t, QK_K, QR3_K, QI3_K, false, block_q3_K, mmq_x, mmq_y, nwarps, allocate_tiles_q3_K<mmq_y>,
        load_tiles_q3_K<mmq_y, nwarps, need_check>, VDR_Q3_K_Q8_1_MMQ, vec_dot_q3_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

template<typename scalar_t>
static void ggml_mul_mat_q3_K_q8_1_cuda(
    const void * vx, const void * vy, scalar_t * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    const int mmq_x  =  MMQ_X_Q3_K;
    const int mmq_y  =  MMQ_Y_Q3_K;
    const int nwarps = NWARPS_Q3_K;

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q3_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q3_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

#if defined(USE_ROCM)
#define  MMQ_X_Q4_K 64
#define  MMQ_Y_Q4_K 128
#define NWARPS_Q4_K 8
#else
#define  MMQ_X_Q4_K 4
#define  MMQ_Y_Q4_K 32
#define NWARPS_Q4_K 4
#endif

template<typename scalar_t, bool need_check> static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF*NWARPS_Q4_K, 2)
#endif
mul_mat_q4_K(
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q4_K;
    const int mmq_y  =  MMQ_Y_Q4_K;
    const int nwarps = NWARPS_Q4_K;

    mul_mat_q<scalar_t, QK_K, QR4_K, QI4_K, true, block_q4_K, mmq_x, mmq_y, nwarps, allocate_tiles_q4_K<mmq_y>,
        load_tiles_q4_K<mmq_y, nwarps, need_check>, VDR_Q4_K_Q8_1_MMQ, vec_dot_q4_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

template<typename scalar_t>
static void ggml_mul_mat_q4_K_q8_1_cuda(
    const void * vx, const void * vy, scalar_t * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {
    const int mmq_x  =  MMQ_X_Q4_K;
    const int mmq_y  =  MMQ_Y_Q4_K;
    const int nwarps = NWARPS_Q4_K;

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q4_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q4_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

#if defined(USE_ROCM)
#define  MMQ_X_Q5_K 64
#define  MMQ_Y_Q5_K 128
#define NWARPS_Q5_K 8
#else
#define  MMQ_X_Q5_K 4
#define  MMQ_Y_Q5_K 32
#define NWARPS_Q5_K 4
#endif

template<typename scalar_t, bool need_check> static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF*NWARPS_Q5_K, 2)
#endif
mul_mat_q5_K(
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q5_K;
    const int mmq_y  =  MMQ_Y_Q5_K;
    const int nwarps = NWARPS_Q5_K;

    mul_mat_q<scalar_t, QK_K, QR5_K, QI5_K, true, block_q5_K, mmq_x, mmq_y, nwarps, allocate_tiles_q5_K<mmq_y>,
        load_tiles_q5_K<mmq_y, nwarps, need_check>, VDR_Q5_K_Q8_1_MMQ, vec_dot_q5_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

template<typename scalar_t>
static void ggml_mul_mat_q5_K_q8_1_cuda(
    const void * vx, const void * vy, scalar_t * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    const int mmq_x  =  MMQ_X_Q5_K;
    const int mmq_y  =  MMQ_Y_Q5_K;
    const int nwarps = NWARPS_Q5_K;

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q5_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q5_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

#if defined(USE_ROCM)
#define  MMQ_X_Q6_K 64
#define  MMQ_Y_Q6_K 128
#define NWARPS_Q6_K 8
#else
#define  MMQ_X_Q6_K 4
#define  MMQ_Y_Q6_K 32
#define NWARPS_Q6_K 4
#endif

template<typename scalar_t, bool need_check> static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF*NWARPS_Q6_K, 2)
#endif
mul_mat_q6_K(
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q6_K;
    const int mmq_y  =  MMQ_Y_Q6_K;
    const int nwarps = NWARPS_Q6_K;

    mul_mat_q<scalar_t, QK_K, QR6_K, QI6_K, false, block_q6_K, mmq_x, mmq_y, nwarps, allocate_tiles_q6_K<mmq_y>,
        load_tiles_q6_K<mmq_y, nwarps, need_check>, VDR_Q6_K_Q8_1_MMQ, vec_dot_q6_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

template<typename scalar_t>
static void ggml_mul_mat_q6_K_q8_1_cuda(
    const void * vx, const void * vy, scalar_t * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {
    const int mmq_x  =  MMQ_X_Q6_K;
    const int mmq_y  =  MMQ_Y_Q6_K;
    const int nwarps = NWARPS_Q6_K;

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q6_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q6_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}
