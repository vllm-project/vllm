// SPDX-License-Identifier: Apache-2.0
// TurboQuant warp-per-Q-head decode kernel.
//
// Two variants:
//   1. tq_decode_wph_kernel — original, one token at a time
//   2. tq_decode_wph_smem_multi_kernel — shared-memory cooperative,
//      TOKENS_PER_ITER=16 tokens loaded to smem per iteration (1.3-1.9x faster)
//
// Supports both 4-bit uniform quantized values and FP8 E4M3 values (hybrid mode).
// When value_fp8=1, values are loaded as FP8 bytes with no scale/zero.
//
// Compile-time defines TQ_HEAD_DIM and TQ_KV_GROUP_SIZE control which
// template instantiations are emitted. When set, only the exact combo is
// compiled, keeping binary size minimal for CUDA graph capture.

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>

#define FULL_MASK 0xFFFFFFFF

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(FULL_MASK, val, offset);
    return val;
}

// FP8 E4M3 to float conversion via bit manipulation.
// Handles normal values and zero. Subnormals treated as zero (acceptable for inference).
__device__ __forceinline__ float fp8e4m3_to_float(uint8_t x) {
    uint32_t sign = ((uint32_t)(x & 0x80)) << 24;  // bit 7 → bit 31
    uint32_t exp = (x >> 3) & 0xF;
    uint32_t mant = x & 0x7;
    if (exp == 0) return __uint_as_float(sign);  // ±0 or subnormal→0
    // FP8 bias=7, FP32 bias=127 → exp_fp32 = exp + 120
    uint32_t bits = sign | ((exp + 120u) << 23) | (mant << 20);
    return __uint_as_float(bits);
}

// ---------------------------------------------------------------------------
// Original WPH kernel (no shared memory)
// ---------------------------------------------------------------------------
template <int D, int KV_GROUP_SIZE>
__global__ void __launch_bounds__(32 * KV_GROUP_SIZE)
tq_decode_wph_kernel(
    const float* __restrict__ q_rot,
    const float* __restrict__ q_proj,
    const uint8_t* __restrict__ kv_cache,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seq_lens,
    const float* __restrict__ centroids,
    float* __restrict__ mid_o,
    int stride_qb, int stride_qh,
    int64_t stride_cache_block, int stride_cache_pos, int stride_cache_head,
    int stride_bt_b,
    int stride_mid_b, int stride_mid_h, int stride_mid_s,
    int num_kv_splits, int block_size,
    int mse_bytes, int qjl_bytes, int kps, int val_data_bytes,
    int value_fp8,
    float correction, float attn_scale)
{
    constexpr int DIMS_PER_THREAD = D / 32;

    const int bid = blockIdx.x;
    const int kv_hid = blockIdx.y;
    const int sid = blockIdx.z;

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;

    const int q_hid = kv_hid * KV_GROUP_SIZE + warp_id;

    const float c0 = centroids[0];
    const float c1 = centroids[1];
    const float c2 = centroids[2];
    const float c3 = centroids[3];

    const int seq_len = seq_lens[bid];
    const int split_len = (seq_len + num_kv_splits - 1) / num_kv_splits;
    const int split_start = split_len * sid;
    const int split_end = min(split_start + split_len, seq_len);

    if (split_start >= split_end) return;

    const int q_base = bid * stride_qb + q_hid * stride_qh;
    float q_r[DIMS_PER_THREAD];
    float q_p[DIMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < DIMS_PER_THREAD; i++) {
        const int d = lane_id * DIMS_PER_THREAD + i;
        q_r[i] = q_rot[q_base + d];
        q_p[i] = q_proj[q_base + d];
    }

    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    float acc[DIMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < DIMS_PER_THREAD; i++) acc[i] = 0.0f;

    const int bt_base = bid * stride_bt_b;
    const int norm_off = mse_bytes + qjl_bytes;
    const int vparam_off = kps + val_data_bytes;

    for (int pos = split_start; pos < split_end; pos++) {
        const int page_idx = pos / block_size;
        const int page_off = pos % block_size;
        const int block_num = block_table[bt_base + page_idx];

        const uint8_t* slot = kv_cache
            + (int64_t)block_num * stride_cache_block
            + page_off * stride_cache_pos
            + kv_hid * stride_cache_head;

        float partial_t1 = 0.0f;
        float partial_t2 = 0.0f;

        #pragma unroll
        for (int i = 0; i < DIMS_PER_THREAD; i++) {
            const int d = lane_id * DIMS_PER_THREAD + i;

            const uint8_t mse_b = slot[d >> 2];
            const int mse_idx = (mse_b >> ((d & 3) << 1)) & 0x3;
            float c_val = (mse_idx == 0) ? c0 : (mse_idx == 1) ? c1 : (mse_idx == 2) ? c2 : c3;
            partial_t1 += q_r[i] * c_val;

            const uint8_t qjl_b = slot[mse_bytes + (d >> 3)];
            const float sign = ((qjl_b >> (d & 7)) & 1) ? 1.0f : -1.0f;
            partial_t2 += q_p[i] * sign;
        }

        float sum_t1 = warp_reduce_sum(partial_t1);
        float sum_t2 = warp_reduce_sum(partial_t2);

        float score;
        if (lane_id == 0) {
            uint16_t n_u16 = (uint16_t)slot[norm_off]
                           | ((uint16_t)slot[norm_off + 1] << 8);
            float vec_norm = __half2float(*reinterpret_cast<const __half*>(&n_u16));

            uint16_t g_u16 = (uint16_t)slot[norm_off + 2]
                           | ((uint16_t)slot[norm_off + 3] << 8);
            float res_norm = __half2float(*reinterpret_cast<const __half*>(&g_u16));

            score = vec_norm * (sum_t1 + correction * res_norm * sum_t2) * attn_scale;
        }
        score = __shfl_sync(FULL_MASK, score, 0);

        const float m_new = fmaxf(score, m_prev);
        const float alpha = __expf(m_prev - m_new);
        const float p = __expf(score - m_new);

        if (value_fp8) {
            // FP8 E4M3 values: 1 byte per dim, no scale/zero
            #pragma unroll
            for (int i = 0; i < DIMS_PER_THREAD; i++) {
                const int d = lane_id * DIMS_PER_THREAD + i;
                const float val = fp8e4m3_to_float(slot[kps + d]);
                acc[i] = acc[i] * alpha + p * val;
            }
        } else {
            // 4-bit uniform quantized values with scale+zero
            float v_scale, v_zero;
            if (lane_id == 0) {
                uint16_t sc_u16 = (uint16_t)slot[vparam_off]
                                | ((uint16_t)slot[vparam_off + 1] << 8);
                v_scale = __half2float(*reinterpret_cast<const __half*>(&sc_u16));

                uint16_t zr_u16 = (uint16_t)slot[vparam_off + 2]
                                | ((uint16_t)slot[vparam_off + 3] << 8);
                v_zero = __half2float(*reinterpret_cast<const __half*>(&zr_u16));
            }
            v_scale = __shfl_sync(FULL_MASK, v_scale, 0);
            v_zero = __shfl_sync(FULL_MASK, v_zero, 0);

            #pragma unroll
            for (int i = 0; i < DIMS_PER_THREAD; i++) {
                const int d = lane_id * DIMS_PER_THREAD + i;
                const uint8_t val_b = slot[kps + (d >> 1)];
                const float v_idx = (float)((val_b >> ((d & 1) << 2)) & 0xF);
                const float val = v_idx * v_scale + v_zero;
                acc[i] = acc[i] * alpha + p * val;
            }
        }

        l_prev = l_prev * alpha + p;
        m_prev = m_new;
    }

    const int out_base = bid * stride_mid_b + q_hid * stride_mid_h + sid * stride_mid_s;
    #pragma unroll
    for (int i = 0; i < DIMS_PER_THREAD; i++) {
        const int d = lane_id * DIMS_PER_THREAD + i;
        mid_o[out_base + d] = acc[i] / l_prev;
    }
    if (lane_id == 0) {
        mid_o[out_base + D] = m_prev + logf(l_prev);
    }
}

// ---------------------------------------------------------------------------
// Shared-memory cooperative multi-token WPH kernel (OPT 5b)
// With vectorized uint32 smem copy and FP8 value support.
// ---------------------------------------------------------------------------
template <int D, int KV_GROUP_SIZE, int TOKENS_PER_ITER>
__global__ void __launch_bounds__(32 * KV_GROUP_SIZE)
tq_decode_wph_smem_multi_kernel(
    const float* __restrict__ q_rot,
    const float* __restrict__ q_proj,
    const uint8_t* __restrict__ kv_cache,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seq_lens,
    const float* __restrict__ centroids,
    float* __restrict__ mid_o,
    int stride_qb, int stride_qh,
    int64_t stride_cache_block, int stride_cache_pos, int stride_cache_head,
    int stride_bt_b,
    int stride_mid_b, int stride_mid_h, int stride_mid_s,
    int num_kv_splits, int block_size,
    int mse_bytes, int qjl_bytes, int kps, int val_data_bytes,
    int slot_bytes,
    int value_fp8,
    float correction, float attn_scale)
{
    constexpr int DIMS_PER_THREAD = D / 32;

    const int bid = blockIdx.x;
    const int kv_hid = blockIdx.y;
    const int sid = blockIdx.z;

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int tid = threadIdx.x;
    const int num_threads = 32 * KV_GROUP_SIZE;

    const int q_hid = kv_hid * KV_GROUP_SIZE + warp_id;

    const float c[4] = {centroids[0], centroids[1], centroids[2], centroids[3]};

    const int seq_len = seq_lens[bid];
    const int split_len = (seq_len + num_kv_splits - 1) / num_kv_splits;
    const int split_start = split_len * sid;
    const int split_end = min(split_start + split_len, seq_len);

    if (split_start >= split_end) return;

    const int q_base = bid * stride_qb + q_hid * stride_qh;
    float q_r[DIMS_PER_THREAD];
    float q_p[DIMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < DIMS_PER_THREAD; i++) {
        const int d = lane_id * DIMS_PER_THREAD + i;
        q_r[i] = q_rot[q_base + d];
        q_p[i] = q_proj[q_base + d];
    }

    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    float acc[DIMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < DIMS_PER_THREAD; i++) acc[i] = 0.0f;

    const int bt_base = bid * stride_bt_b;
    const int norm_off = mse_bytes + qjl_bytes;
    const int vparam_off = kps + val_data_bytes;

    // Dynamic shared memory for TOKENS_PER_ITER tokens
    extern __shared__ uint8_t slot_smem[];

    // Precompute uint32 word count for vectorized copy
    const int slot_words = (slot_bytes + 3) / 4;

    for (int pos_base = split_start; pos_base < split_end; pos_base += TOKENS_PER_ITER) {
        const int tokens_this_iter = min(TOKENS_PER_ITER, split_end - pos_base);

        // Cooperative load of multiple tokens into shared memory
        // VECTORIZED: copy as uint32 (4x fewer transactions than byte-by-byte)
        for (int t = 0; t < tokens_this_iter; t++) {
            const int pos = pos_base + t;
            const int page_idx = pos / block_size;
            const int page_off = pos % block_size;
            const int block_num = block_table[bt_base + page_idx];

            const uint8_t* slot_global = kv_cache
                + (int64_t)block_num * stride_cache_block
                + page_off * stride_cache_pos
                + kv_hid * stride_cache_head;

            // Vectorized uint32 copy — slot_global is aligned (padded_slot is power-of-2)
            const uint32_t* src32 = reinterpret_cast<const uint32_t*>(slot_global);
            uint32_t* dst32 = reinterpret_cast<uint32_t*>(slot_smem + t * slot_bytes);
            for (int i = tid; i < slot_words; i += num_threads) {
                dst32[i] = src32[i];
            }
        }
        __syncthreads();

        // Process all loaded tokens — vectorized: pre-load MSE/QJL/value
        // bytes into registers to reduce repeated smem byte-level reads.
        for (int t = 0; t < tokens_this_iter; t++) {
            const uint8_t* slot = slot_smem + t * slot_bytes;

            // Pre-load MSE bytes (2 bytes for DIMS_PER_THREAD=8 dims at 2 bits/dim)
            const int mse_base = lane_id * (DIMS_PER_THREAD / 4);
            const uint8_t mse_b0 = slot[mse_base];
            const uint8_t mse_b1 = slot[mse_base + 1];

            // Pre-load QJL byte (1 byte for 8 dims at 1 bit/dim)
            const uint8_t qjl_b = slot[mse_bytes + lane_id];

            float partial_t1 = 0.0f;
            float partial_t2 = 0.0f;

            #pragma unroll
            for (int i = 0; i < DIMS_PER_THREAD; i++) {
                const int d = lane_id * DIMS_PER_THREAD + i;
                const uint8_t mse_byte = (i < 4) ? mse_b0 : mse_b1;
                const int mse_idx = (mse_byte >> ((d & 3) << 1)) & 0x3;
                partial_t1 += q_r[i] * c[mse_idx];

                const float sign = ((qjl_b >> (d & 7)) & 1) ? 1.0f : -1.0f;
                partial_t2 += q_p[i] * sign;
            }

            float sum_t1 = warp_reduce_sum(partial_t1);
            float sum_t2 = warp_reduce_sum(partial_t2);

            // All threads read norms from smem (broadcast read, no shuffle needed)
            uint16_t n_u16 = (uint16_t)slot[norm_off]
                           | ((uint16_t)slot[norm_off + 1] << 8);
            float vec_norm = __half2float(*reinterpret_cast<const __half*>(&n_u16));
            uint16_t g_u16 = (uint16_t)slot[norm_off + 2]
                           | ((uint16_t)slot[norm_off + 3] << 8);
            float res_norm = __half2float(*reinterpret_cast<const __half*>(&g_u16));

            float score;
            if (lane_id == 0) {
                score = vec_norm * (sum_t1 + correction * res_norm * sum_t2) * attn_scale;
            }
            score = __shfl_sync(FULL_MASK, score, 0);

            const float m_new = fmaxf(score, m_prev);
            const float alpha = __expf(m_prev - m_new);
            const float p = __expf(score - m_new);

            if (value_fp8) {
                // FP8 E4M3 values: 1 byte per dim, no scale/zero
                // Pre-load value bytes (DIMS_PER_THREAD bytes for FP8)
                uint8_t fp8_bytes[DIMS_PER_THREAD];
                const int val_base = kps + lane_id * DIMS_PER_THREAD;
                #pragma unroll
                for (int vi = 0; vi < DIMS_PER_THREAD; vi++) {
                    fp8_bytes[vi] = slot[val_base + vi];
                }
                #pragma unroll
                for (int i = 0; i < DIMS_PER_THREAD; i++) {
                    const float val = fp8e4m3_to_float(fp8_bytes[i]);
                    acc[i] = acc[i] * alpha + p * val;
                }
            } else {
                // 4-bit uniform quantized values with scale+zero
                // All threads read v_scale/v_zero from smem (no shuffle needed)
                uint16_t sc_u16 = (uint16_t)slot[vparam_off]
                                | ((uint16_t)slot[vparam_off + 1] << 8);
                float v_scale = __half2float(*reinterpret_cast<const __half*>(&sc_u16));
                uint16_t zr_u16 = (uint16_t)slot[vparam_off + 2]
                                | ((uint16_t)slot[vparam_off + 3] << 8);
                float v_zero = __half2float(*reinterpret_cast<const __half*>(&zr_u16));

                // Pre-load value bytes (4 bytes for 8 dims at 4 bits/dim)
                const int val_byte_base = kps + lane_id * (DIMS_PER_THREAD / 2);
                uint8_t val_bytes[DIMS_PER_THREAD / 2];
                #pragma unroll
                for (int vi = 0; vi < DIMS_PER_THREAD / 2; vi++) {
                    val_bytes[vi] = slot[val_byte_base + vi];
                }

                #pragma unroll
                for (int i = 0; i < DIMS_PER_THREAD; i++) {
                    const float v_idx = (float)((val_bytes[i >> 1] >> ((i & 1) << 2)) & 0xF);
                    const float val = v_idx * v_scale + v_zero;
                    acc[i] = acc[i] * alpha + p * val;
                }
            }

            l_prev = l_prev * alpha + p;
            m_prev = m_new;
        }

        __syncthreads();
    }

    const int out_base = bid * stride_mid_b + q_hid * stride_mid_h + sid * stride_mid_s;
    #pragma unroll
    for (int i = 0; i < DIMS_PER_THREAD; i++) {
        const int d = lane_id * DIMS_PER_THREAD + i;
        mid_o[out_base + d] = acc[i] / l_prev;
    }
    if (lane_id == 0) {
        mid_o[out_base + D] = m_prev + logf(l_prev);
    }
}

// ---------------------------------------------------------------------------
// Launch wrappers — use compile-time defines when available to minimize
// template instantiations and binary size.
// ---------------------------------------------------------------------------

// When TQ_HEAD_DIM and TQ_KV_GROUP_SIZE are defined, only instantiate that
// specific combo (2 kernels total: original + smem). Otherwise fall back to
// the full switch for generality.

void tq_decode_wph_launch(
    torch::Tensor q_rot, torch::Tensor q_proj,
    torch::Tensor kv_cache, torch::Tensor block_table,
    torch::Tensor seq_lens, torch::Tensor centroids,
    torch::Tensor mid_o,
    int64_t num_kv_splits, int64_t head_dim,
    int64_t num_kv_heads, int64_t kv_group_size,
    int64_t block_size,
    int64_t mse_bytes, int64_t qjl_bytes,
    int64_t kps, int64_t val_data_bytes,
    int64_t value_fp8,
    double correction, double attn_scale)
{
    const int B = q_rot.size(0);
    const int Hk = num_kv_heads;

    dim3 grid(B, Hk, num_kv_splits);
    dim3 block(32 * kv_group_size);

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    #define LAUNCH_WPH(D_VAL, GS_VAL) \
        tq_decode_wph_kernel<D_VAL, GS_VAL><<<grid, block, 0, stream>>>( \
            q_rot.data_ptr<float>(), q_proj.data_ptr<float>(), \
            kv_cache.data_ptr<uint8_t>(), block_table.data_ptr<int32_t>(), \
            seq_lens.data_ptr<int32_t>(), centroids.data_ptr<float>(), \
            mid_o.data_ptr<float>(), \
            (int)q_rot.stride(0), (int)q_rot.stride(1), \
            (int64_t)kv_cache.stride(0), (int)kv_cache.stride(1), (int)kv_cache.stride(2), \
            (int)block_table.stride(0), \
            (int)mid_o.stride(0), (int)mid_o.stride(1), (int)mid_o.stride(2), \
            (int)num_kv_splits, (int)block_size, \
            (int)mse_bytes, (int)qjl_bytes, (int)kps, (int)val_data_bytes, \
            (int)value_fp8, \
            (float)correction, (float)attn_scale)

#if defined(TQ_HEAD_DIM) && defined(TQ_KV_GROUP_SIZE)
    // Minimal instantiation: only the configured combo
    TORCH_CHECK(head_dim == TQ_HEAD_DIM && kv_group_size == TQ_KV_GROUP_SIZE,
        "Compiled for D=", TQ_HEAD_DIM, " GS=", TQ_KV_GROUP_SIZE,
        " but got D=", head_dim, " GS=", kv_group_size);
    LAUNCH_WPH(TQ_HEAD_DIM, TQ_KV_GROUP_SIZE);
#else
    // Full switch (fallback)
    if (head_dim == 256) {
        switch (kv_group_size) {
            case 1:  LAUNCH_WPH(256, 1); break;
            case 2:  LAUNCH_WPH(256, 2); break;
            case 4:  LAUNCH_WPH(256, 4); break;
            case 8:  LAUNCH_WPH(256, 8); break;
            case 16: LAUNCH_WPH(256, 16); break;
            default: TORCH_CHECK(false, "Unsupported kv_group_size=", kv_group_size);
        }
    } else if (head_dim == 128) {
        switch (kv_group_size) {
            case 1:  LAUNCH_WPH(128, 1); break;
            case 2:  LAUNCH_WPH(128, 2); break;
            case 4:  LAUNCH_WPH(128, 4); break;
            case 8:  LAUNCH_WPH(128, 8); break;
            case 16: LAUNCH_WPH(128, 16); break;
            default: TORCH_CHECK(false, "Unsupported kv_group_size=", kv_group_size);
        }
    } else {
        TORCH_CHECK(false, "Unsupported head_dim=", head_dim);
    }
#endif
    #undef LAUNCH_WPH
}

void tq_decode_wph_smem_launch(
    torch::Tensor q_rot, torch::Tensor q_proj,
    torch::Tensor kv_cache, torch::Tensor block_table,
    torch::Tensor seq_lens, torch::Tensor centroids,
    torch::Tensor mid_o,
    int64_t num_kv_splits, int64_t head_dim,
    int64_t num_kv_heads, int64_t kv_group_size,
    int64_t block_size,
    int64_t mse_bytes, int64_t qjl_bytes,
    int64_t kps, int64_t val_data_bytes,
    int64_t slot_bytes,
    int64_t value_fp8,
    double correction, double attn_scale)
{
    const int B = q_rot.size(0);
    const int Hk = num_kv_heads;
    constexpr int TOKENS_PER_ITER = 16;

    dim3 grid(B, Hk, num_kv_splits);
    dim3 block(32 * kv_group_size);

    // Dynamic shared memory: TOKENS_PER_ITER * slot_bytes
    const int smem_bytes = TOKENS_PER_ITER * (int)slot_bytes;

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    #define LAUNCH_SMEM(D_VAL, GS_VAL) \
        tq_decode_wph_smem_multi_kernel<D_VAL, GS_VAL, TOKENS_PER_ITER><<<grid, block, smem_bytes, stream>>>( \
            q_rot.data_ptr<float>(), q_proj.data_ptr<float>(), \
            kv_cache.data_ptr<uint8_t>(), block_table.data_ptr<int32_t>(), \
            seq_lens.data_ptr<int32_t>(), centroids.data_ptr<float>(), \
            mid_o.data_ptr<float>(), \
            (int)q_rot.stride(0), (int)q_rot.stride(1), \
            (int64_t)kv_cache.stride(0), (int)kv_cache.stride(1), (int)kv_cache.stride(2), \
            (int)block_table.stride(0), \
            (int)mid_o.stride(0), (int)mid_o.stride(1), (int)mid_o.stride(2), \
            (int)num_kv_splits, (int)block_size, \
            (int)mse_bytes, (int)qjl_bytes, (int)kps, (int)val_data_bytes, \
            (int)slot_bytes, \
            (int)value_fp8, \
            (float)correction, (float)attn_scale)

#if defined(TQ_HEAD_DIM) && defined(TQ_KV_GROUP_SIZE)
    TORCH_CHECK(head_dim == TQ_HEAD_DIM && kv_group_size == TQ_KV_GROUP_SIZE,
        "Compiled for D=", TQ_HEAD_DIM, " GS=", TQ_KV_GROUP_SIZE,
        " but got D=", head_dim, " GS=", kv_group_size);
    LAUNCH_SMEM(TQ_HEAD_DIM, TQ_KV_GROUP_SIZE);
#else
    if (head_dim == 256) {
        switch (kv_group_size) {
            case 1:  LAUNCH_SMEM(256, 1); break;
            case 2:  LAUNCH_SMEM(256, 2); break;
            case 4:  LAUNCH_SMEM(256, 4); break;
            case 8:  LAUNCH_SMEM(256, 8); break;
            case 16: LAUNCH_SMEM(256, 16); break;
            default: TORCH_CHECK(false, "Unsupported kv_group_size=", kv_group_size);
        }
    } else if (head_dim == 128) {
        switch (kv_group_size) {
            case 1:  LAUNCH_SMEM(128, 1); break;
            case 2:  LAUNCH_SMEM(128, 2); break;
            case 4:  LAUNCH_SMEM(128, 4); break;
            case 8:  LAUNCH_SMEM(128, 8); break;
            case 16: LAUNCH_SMEM(128, 16); break;
            default: TORCH_CHECK(false, "Unsupported kv_group_size=", kv_group_size);
        }
    } else {
        TORCH_CHECK(false, "Unsupported head_dim=", head_dim);
    }
#endif
    #undef LAUNCH_SMEM
}

// ---------------------------------------------------------------------------
// Full dequant kernel: KV cache → fp16 K, V buffers (for 5a graph-compat)
// Grid: (alloc_seq_len, B * Hk), Block: 32 threads
// For pos >= seq_lens[bid], writes K=0, V=0.
// ---------------------------------------------------------------------------
template <int D>
__global__ void __launch_bounds__(32)
tq_full_dequant_kv_kernel(
    const uint8_t* __restrict__ kv_cache,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seq_lens,
    const float* __restrict__ centroids,
    __half* __restrict__ k_out,
    __half* __restrict__ v_out,
    int stride_ko_b, int stride_ko_h, int stride_ko_s,
    int stride_vo_b, int stride_vo_h, int stride_vo_s,
    int64_t stride_cache_block, int stride_cache_pos, int stride_cache_head,
    int stride_bt_b,
    int num_kv_heads, int block_size,
    int mse_bytes, int qjl_bytes, int kps, int val_data_bytes,
    float correction)
{
    constexpr int DIMS_PER_THREAD = D / 32;
    const int pos = blockIdx.x;
    const int bh = blockIdx.y;
    const int bid = bh / num_kv_heads;
    const int hid = bh % num_kv_heads;
    const int tid = threadIdx.x;

    const int seq_len = seq_lens[bid];
    const int ko_base = bid * stride_ko_b + hid * stride_ko_h + pos * stride_ko_s;
    const int vo_base = bid * stride_vo_b + hid * stride_vo_h + pos * stride_vo_s;

    if (pos >= seq_len) {
        #pragma unroll
        for (int i = 0; i < DIMS_PER_THREAD; i++) {
            const int d = tid * DIMS_PER_THREAD + i;
            k_out[ko_base + d] = __float2half(0.0f);
            v_out[vo_base + d] = __float2half(0.0f);
        }
        return;
    }

    const int page_idx = pos / block_size;
    const int page_off = pos % block_size;
    const int block_num = block_table[bid * stride_bt_b + page_idx];
    const uint8_t* slot = kv_cache
        + (int64_t)block_num * stride_cache_block
        + page_off * stride_cache_pos
        + hid * stride_cache_head;

    const float c0 = centroids[0], c1 = centroids[1];
    const float c2 = centroids[2], c3 = centroids[3];

    const int norm_off = mse_bytes + qjl_bytes;
    uint16_t n_u16 = (uint16_t)slot[norm_off] | ((uint16_t)slot[norm_off+1] << 8);
    float vec_norm = __half2float(*reinterpret_cast<const __half*>(&n_u16));
    uint16_t g_u16 = (uint16_t)slot[norm_off+2] | ((uint16_t)slot[norm_off+3] << 8);
    float res_norm = __half2float(*reinterpret_cast<const __half*>(&g_u16));

    const int vparam_off = kps + val_data_bytes;
    uint16_t sc_u16 = (uint16_t)slot[vparam_off] | ((uint16_t)slot[vparam_off+1] << 8);
    float v_scale = __half2float(*reinterpret_cast<const __half*>(&sc_u16));
    uint16_t zr_u16 = (uint16_t)slot[vparam_off+2] | ((uint16_t)slot[vparam_off+3] << 8);
    float v_zero = __half2float(*reinterpret_cast<const __half*>(&zr_u16));

    #pragma unroll
    for (int i = 0; i < DIMS_PER_THREAD; i++) {
        const int d = tid * DIMS_PER_THREAD + i;
        const uint8_t mse_b = slot[d >> 2];
        const int mse_idx = (mse_b >> ((d & 3) << 1)) & 0x3;
        float k_mse = (mse_idx == 0) ? c0 : (mse_idx == 1) ? c1 : (mse_idx == 2) ? c2 : c3;
        const uint8_t qjl_b = slot[mse_bytes + (d >> 3)];
        const float sign = ((qjl_b >> (d & 7)) & 1) ? 1.0f : -1.0f;
        k_out[ko_base + d] = __float2half(vec_norm * (k_mse + correction * res_norm * sign));

        const uint8_t val_b = slot[kps + (d >> 1)];
        const float v_idx = (float)((val_b >> ((d & 1) << 2)) & 0xF);
        v_out[vo_base + d] = __float2half(v_idx * v_scale + v_zero);
    }
}

// ---------------------------------------------------------------------------
// Masked softmax kernel
// Grid: (num_rows), Block: 256 threads
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256)
tq_masked_softmax_kernel(
    float* __restrict__ scores,
    const int32_t* __restrict__ seq_lens,
    int S, int num_q_heads)
{
    const int row = blockIdx.x;
    const int bid = row / num_q_heads;
    const int seq_len = seq_lens[bid];
    const int tid = threadIdx.x;
    float* row_ptr = scores + (int64_t)row * S;

    // Pass 1: find max with masking
    float local_max = -INFINITY;
    for (int s = tid; s < S; s += 256) {
        float val;
        if (s >= seq_len) { row_ptr[s] = -INFINITY; val = -INFINITY; }
        else { val = row_ptr[s]; }
        local_max = fmaxf(local_max, val);
    }
    __shared__ float smem[256];
    smem[tid] = local_max;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
        __syncthreads();
    }
    float row_max = smem[0];

    // Pass 2: exp and sum
    float local_sum = 0.0f;
    for (int s = tid; s < S; s += 256) {
        float val = row_ptr[s];
        float e = (val > -INFINITY) ? __expf(val - row_max) : 0.0f;
        row_ptr[s] = e;
        local_sum += e;
    }
    smem[tid] = local_sum;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    float inv_sum = (smem[0] > 0.0f) ? 1.0f / smem[0] : 0.0f;

    // Pass 3: normalize
    for (int s = tid; s < S; s += 256) { row_ptr[s] *= inv_sum; }
}

// ---------------------------------------------------------------------------
// Dequant + softmax launch wrappers
// ---------------------------------------------------------------------------
void tq_full_dequant_kv_launch(
    torch::Tensor kv_cache, torch::Tensor block_table,
    torch::Tensor seq_lens, torch::Tensor centroids,
    torch::Tensor k_out, torch::Tensor v_out,
    int64_t alloc_seq_len, int64_t head_dim,
    int64_t num_kv_heads, int64_t block_size,
    int64_t mse_bytes, int64_t qjl_bytes,
    int64_t kps, int64_t val_data_bytes,
    double correction)
{
    const int B = seq_lens.size(0);
    const int Hk = num_kv_heads;
    dim3 grid(alloc_seq_len, B * Hk);
    dim3 block(32);
    auto stream = at::cuda::getCurrentCUDAStream().stream();

#if defined(TQ_HEAD_DIM)
    TORCH_CHECK(head_dim == TQ_HEAD_DIM, "Compiled for D=", TQ_HEAD_DIM,
                " but got D=", head_dim);
    tq_full_dequant_kv_kernel<TQ_HEAD_DIM><<<grid, block, 0, stream>>>(
#else
    // Fallback: instantiate both
    auto launch = [&](auto fn) {
        fn<<<grid, block, 0, stream>>>(
#endif
            kv_cache.data_ptr<uint8_t>(), block_table.data_ptr<int32_t>(),
            seq_lens.data_ptr<int32_t>(), centroids.data_ptr<float>(),
            reinterpret_cast<__half*>(k_out.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(v_out.data_ptr<at::Half>()),
            (int)k_out.stride(0), (int)k_out.stride(1), (int)k_out.stride(2),
            (int)v_out.stride(0), (int)v_out.stride(1), (int)v_out.stride(2),
            (int64_t)kv_cache.stride(0), (int)kv_cache.stride(1), (int)kv_cache.stride(2),
            (int)block_table.stride(0),
            (int)Hk, (int)block_size,
            (int)mse_bytes, (int)qjl_bytes, (int)kps, (int)val_data_bytes,
            (float)correction);
#if defined(TQ_HEAD_DIM)
    // single instantiation done above
#else
    };
    if (head_dim == 256) { launch(tq_full_dequant_kv_kernel<256>); }
    else if (head_dim == 128) { launch(tq_full_dequant_kv_kernel<128>); }
    else { TORCH_CHECK(false, "Unsupported head_dim=", head_dim); }
#endif
}

void tq_masked_softmax_launch(
    torch::Tensor scores,
    torch::Tensor seq_lens,
    int64_t alloc_seq_len,
    int64_t num_q_heads)
{
    dim3 grid(scores.size(0));
    dim3 block(256);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    tq_masked_softmax_kernel<<<grid, block, 0, stream>>>(
        scores.data_ptr<float>(),
        seq_lens.data_ptr<int32_t>(),
        (int)alloc_seq_len, (int)num_q_heads);
}
