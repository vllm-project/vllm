// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
/**
 * HIP Indexer Compressor Kernel - DeepSeek-V4 head_dim=128 (FP8 + MXFP4)
 *
 * The indexer's compressor is CSA geometry (ratio=4, overlap, K_POOL=8) at
 * head_dim=128. The compress -> softmax -> weighted-sum -> RMSNorm -> RoPE
 * front-end is shared; only the quant tail differs. Two compile-time
 * specializations (template<QFMT>), selected at launch by use_fp4_cache, mirror
 * vLLM's two Triton kernels (_fused_kv_compress_norm_rope_insert_indexer_attn /
 * ..._indexer_mxfp4_attn). Both are the byte-exact targets.
 *
 *   QFMT_FP8  : whole 128-dim post-RoPE row -> FP8 + single ue8m0 scale (fp32).
 *               token_stride=128, scale_dim=4 (1 fp32). native v_cvt_pk_fp8_f32.
 *   QFMT_MXFP4: even/odd pairs -> E2M1 nibbles, per-32 ue8m0 block scale.
 *               token_stride=64, scale_dim=4 (4 ue8m0 bytes). native
 *               v_cvt_scalef32_pk_fp4_f32.
 *
 * Mapping = M1 (CSA port): wave-per-token, 64 lanes * 2 dims = 128. RMSNorm /
 * FP8 absmax are wave-local warp reductions; MXFP4 block absmax is a 16-lane
 * subgroup reduction. No LDS, no __syncthreads. Plan-free: launched per-token,
 * non-boundary waves return. (P3: M1 is at max occupancy / latency-bound;
 * thread-per-token is infeasible (per-dim softmax state). See PLAN_INDEXER.md.)
 *
 * Built into the _rocm_C extension for all VLLM_GPU_ARCHES. The native FP8 /
 * FP4 cvt builtins exist only on gfx950 (CDNA4), so the kernel BODY is compiled
 * only for the gfx950 device pass (and the host pass for symbol parity); on
 * other device passes it is an empty stub. The host launcher always compiles,
 * and the Python dispatcher only routes to this op on gfx950.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <cmath>
#include <cstdint>

#include "dsv4_compress_common.cuh"  // DSV4_GFX950, dsv4::warp_reduce_sum

// Constants for DeepSeek-V4 indexer config (ratio=4, overlap, head_dim=128)
[[maybe_unused]] constexpr int K_POOL = 8;
[[maybe_unused]] constexpr int HEAD_SIZE = 128;
[[maybe_unused]] constexpr int ROPE_HEAD_DIM = 64;
[[maybe_unused]] constexpr int NOPE_HEAD_DIM = HEAD_SIZE - ROPE_HEAD_DIM;  // 64
[[maybe_unused]] constexpr int STATE_WIDTH = 256;   // overlap: coff*head_dim = 2*128

[[maybe_unused]] constexpr int BLOCK_SIZE = 256;  // 4 waves
[[maybe_unused]] constexpr int WARP_SIZE = 64;    // AMD wavefront
[[maybe_unused]] constexpr int DIMS_PER_LANE = HEAD_SIZE / WARP_SIZE;  // 2

[[maybe_unused]] constexpr float FP8_MAX = 448.0f;
[[maybe_unused]] constexpr float INV_FP8_MAX = 1.0f / FP8_MAX;
[[maybe_unused]] constexpr float FP4_MAX = 6.0f;          // E2M1 max
[[maybe_unused]] constexpr float INV_FP4_MAX = 1.0f / FP4_MAX;

// Quant-format specializations.
[[maybe_unused]] constexpr int QFMT_FP8 = 0;
[[maybe_unused]] constexpr int QFMT_MXFP4 = 1;

// ============================================================================
// Helpers
// ============================================================================
// warp_reduce_sum comes from dsv4_compress_common.cuh (dsv4::warp_reduce_sum).
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor(val, offset));
    }
    return val;
}

// Max over a 16-lane subgroup (one MXFP4 block = 16 pairs = 16 consecutive lanes).
__device__ __forceinline__ float subgroup16_reduce_max(float val) {
    #pragma unroll
    for (int offset = 8; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor(val, offset));
    }
    return val;
}

// ============================================================================
// Kernel: fully fused, plan-free, wave-per-token (M1), templated quant tail
// ============================================================================
template <int QFMT, typename WeightT>
__global__ void indexer_compress_kernel(
    int num_tokens,
    const __hip_bfloat16* __restrict__ state_cache,
    int64_t state_stride0,
    int64_t state_stride1,
    const float* __restrict__ ape,          // RAW [4, 2*HEAD_SIZE=256] fp32
    int64_t ape_stride,                      // = ape.stride(0) (e.g. 256)
    const int32_t* __restrict__ token_to_req_indices,
    const int64_t* __restrict__ positions,
    const int64_t* __restrict__ slot_mapping,
    const int32_t* __restrict__ block_table,
    int64_t block_table_stride,
    int32_t block_size,
    const WeightT* __restrict__ rms_norm_weight,
    float rms_norm_eps,
    const float* __restrict__ cos_sin_cache,
    int64_t cos_sin_stride,
    uint8_t* __restrict__ kv_cache,
    const int64_t* __restrict__ kv_slot_mapping,
    int32_t kv_cache_block_size,
    int64_t kv_block_stride,
    int32_t scale_dim
) {
#if defined(DSV4_COMPILE_GFX950_BODY)
    // Per-token byte stride: FP8 = all 128 bytes; MXFP4 = 64 packed nibbles.
    constexpr int TOKEN_STRIDE = (QFMT == QFMT_MXFP4) ? HEAD_SIZE / 2 : HEAD_SIZE;

    int wave_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int token_idx = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + wave_id;
    if (token_idx >= num_tokens) return;

    // On-device boundary derivation (ratio=4): (position+1) % 4 == 0.
    int64_t kv_slot_idx = kv_slot_mapping[token_idx];
    int64_t position = positions[token_idx];
    if (((position + 1) & 3) != 0 || kv_slot_idx < 0) return;  // not a boundary
    int64_t slot_id = slot_mapping[token_idx];
    if (slot_id < 0) return;

    int32_t req_idx = token_to_req_indices[token_idx];
    int64_t start_pos = position - (K_POOL - 1);
    int dim_base = lane_id * DIMS_PER_LANE;  // 0,2,...,126; pair index == lane_id

    // ── Phase 1: resolve K_POOL row base pointers (overlap second half @ k>=4) ─
    const __hip_bfloat16* row_ptrs[K_POOL];
    bool valid_k[K_POOL];
    #pragma unroll
    for (int k = 0; k < K_POOL; k++) {
        int64_t pos_k = start_pos + k;
        if (pos_k < 0) {
            valid_k[k] = false;
            row_ptrs[k] = nullptr;
            continue;
        }
        valid_k[k] = true;
        int64_t blk_idx = pos_k / block_size;
        int64_t blk_off = pos_k % block_size;
        int32_t blk_num = block_table[req_idx * block_table_stride + blk_idx];
        int head_offset = (k >= 4) ? HEAD_SIZE : 0;
        row_ptrs[k] = state_cache + blk_num * state_stride0
                    + blk_off * state_stride1 + head_offset;
    }

    // ── Online softmax (per-dim over K_POOL); APE from raw [4,256] ────────────
    float m[DIMS_PER_LANE], l[DIMS_PER_LANE], acc[DIMS_PER_LANE];
    #pragma unroll
    for (int d = 0; d < DIMS_PER_LANE; d++) {
        m[d] = -INFINITY; l[d] = 0.0f; acc[d] = 0.0f;
    }
    #pragma unroll
    for (int k = 0; k < K_POOL; k++) {
        if (!valid_k[k]) continue;  // exp(-inf) contributes nothing
        union { float v; __hip_bfloat16 h[2]; } us, uk;
        us.v = *reinterpret_cast<const float*>(row_ptrs[k] + STATE_WIDTH + dim_base);
        uk.v = *reinterpret_cast<const float*>(row_ptrs[k] + dim_base);
        const float* ape_ptr = ape + (int64_t)(k & 3) * ape_stride
                             + ((k >= 4) ? HEAD_SIZE : 0) + dim_base;
        #pragma unroll
        for (int d = 0; d < DIMS_PER_LANE; d++) {
            float s = __bfloat162float(us.h[d]) + ape_ptr[d];
            float kvv = __bfloat162float(uk.h[d]);
            float new_m = fmaxf(m[d], s);
            float corr = __expf(m[d] - new_m);   // 0 on first valid k (m=-inf)
            float p = __expf(s - new_m);
            l[d] = l[d] * corr + p;
            acc[d] = acc[d] * corr + p * kvv;
            m[d] = new_m;
        }
    }
    float comp[DIMS_PER_LANE];
    #pragma unroll
    for (int d = 0; d < DIMS_PER_LANE; d++) comp[d] = acc[d] / l[d];

    // ── RMSNorm: full-wave warp_reduce_sum over all 128 dims ──────────────────
    float sq = 0.0f;
    #pragma unroll
    for (int d = 0; d < DIMS_PER_LANE; d++) sq += comp[d] * comp[d];
    float variance = dsv4::warp_reduce_sum(sq) / HEAD_SIZE;
    float rrms = rsqrtf(variance + rms_norm_eps);

    float normed[DIMS_PER_LANE];
    #pragma unroll
    for (int d = 0; d < DIMS_PER_LANE; d++) {
        float w = dsv4::load_rms_norm_weight(rms_norm_weight, dim_base + d);
        normed[d] = comp[d] * rrms * w;
    }

    // ── GPT-J RoPE on last ROPE_HEAD_DIM dims; result[0]=new_even,1=new_odd ───
    float result[DIMS_PER_LANE];
    if (dim_base < NOPE_HEAD_DIM) {
        #pragma unroll
        for (int d = 0; d < DIMS_PER_LANE; d++) result[d] = normed[d];
    } else {
        int rope_local = dim_base - NOPE_HEAD_DIM;   // 0,2,...,62
        int cs_idx = rope_local / 2;
        int64_t comp_pos = position & (~3LL);         // (position//4)*4
        const float* cs = cos_sin_cache + comp_pos * cos_sin_stride;
        float cos_v = cs[cs_idx];
        float sin_v = cs[ROPE_HEAD_DIM / 2 + cs_idx];
        float e = normed[0], o = normed[1];           // one pair per lane
        result[0] = e * cos_v - o * sin_v;            // new_even
        result[1] = o * cos_v + e * sin_v;            // new_odd
    }
    // bf16 roundtrip the whole row for parity with the reference.
    float q[DIMS_PER_LANE];
    #pragma unroll
    for (int d = 0; d < DIMS_PER_LANE; d++) {
        q[d] = __bfloat162float(__float2bfloat16(result[d]));
    }

    // ── Output addressing ─────────────────────────────────────────────────────
    int64_t kv_blk_idx = kv_slot_idx / kv_cache_block_size;
    int64_t kv_pos = kv_slot_idx % kv_cache_block_size;
    uint8_t* cache_block = kv_cache + kv_blk_idx * kv_block_stride;
    uint8_t* val_ptr = cache_block + kv_pos * TOKEN_STRIDE;
    uint8_t* scale_ptr = cache_block + kv_cache_block_size * TOKEN_STRIDE
                       + kv_pos * scale_dim;

    if constexpr (QFMT == QFMT_FP8) {
        // Single-block FP8 ue8m0: absmax over the WHOLE 128-dim row.
        float lmax = 0.0f;
        #pragma unroll
        for (int d = 0; d < DIMS_PER_LANE; d++) lmax = fmaxf(lmax, fabsf(q[d]));
        float blk_max = fmaxf(warp_reduce_max(lmax), 1e-4f);
        float exponent = ceilf(log2f(blk_max * INV_FP8_MAX));
        float inv_scale = exp2f(-exponent);

        // Native E4M3 pack: 2 f32 -> 2 fp8 (low 16 bits), one uint16 store.
        // dim_base = lane*2 -> val_ptr+dim_base is 2-byte aligned & contiguous.
        int w = 0;
        w = __builtin_amdgcn_cvt_pk_fp8_f32(q[0] * inv_scale, q[1] * inv_scale, w, false);
        *reinterpret_cast<uint16_t*>(val_ptr + dim_base) = (uint16_t)(w & 0xFFFF);

        // Single fp32 scale per token (value = 2^exponent), one writer.
        if (lane_id == 0) {
            *reinterpret_cast<float*>(scale_ptr) = exp2f(exponent);
        }
    } else {  // QFMT_MXFP4
        // Per-32-element block ue8m0; pair index == lane, block = lane / 16.
        // amax over the block (max of |new_even|,|new_odd| across 16 lanes).
        float lmax = fmaxf(fabsf(q[0]), fabsf(q[1]));
        float amax = fmaxf(subgroup16_reduce_max(lmax), FP4_MAX * exp2f(-126.0f));
        float log2r = ceilf(log2f(amax * INV_FP4_MAX));
        log2r = fminf(fmaxf(log2r, -127.0f), 127.0f);
        float inv_scale = exp2f(-log2r);

        // Native E2M1 pack: (even,odd) -> 2 nibbles (1 byte). Pre-scale by
        // inv_scale and pass scale=1.0 to mirror Triton's
        // _fp32x2_to_fp4x2(even*inv_scale, odd*inv_scale).
        uint32_t packed = 0;
        packed = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(
            packed, q[0] * inv_scale, q[1] * inv_scale, 1.0f, 0);
        val_ptr[lane_id] = (uint8_t)(packed & 0xFF);

        // ue8m0 block scale byte (one writer per 16-lane block).
        if ((lane_id & 15) == 0) {
            scale_ptr[lane_id >> 4] = (uint8_t)((int)(log2r + 127.0f));
        }
    }
#else
    dsv4::ignore_unused(num_tokens, state_cache, state_stride0, state_stride1,
                        ape, ape_stride, token_to_req_indices, positions,
                        slot_mapping, block_table, block_table_stride,
                        block_size, rms_norm_weight, rms_norm_eps,
                        cos_sin_cache, cos_sin_stride, kv_cache,
                        kv_slot_mapping, kv_cache_block_size, kv_block_stride,
                        scale_dim);
#endif  // DSV4_COMPILE_GFX950_BODY
}

// ============================================================================
// Launcher
// ============================================================================
extern "C" void launch_indexer_compress(
    int num_tokens,
    int use_fp4_cache,
    const void* state_cache_ptr, int64_t state_stride0, int64_t state_stride1,
    const float* ape_raw, int64_t ape_stride,
    const int32_t* token_to_req_indices,
    const int64_t* positions,
    const int64_t* slot_mapping,
    const int32_t* block_table, int64_t block_table_stride, int32_t block_size,
    const void* rms_norm_weight, bool rms_norm_weight_is_bf16,
    float rms_norm_eps,
    const float* cos_sin_cache, int64_t cos_sin_stride,
    uint8_t* kv_cache, const int64_t* kv_slot_mapping,
    int32_t kv_cache_block_size, int64_t kv_block_stride, int32_t scale_dim,
    void* stream_ptr
) {
    if (num_tokens == 0) return;

    const __hip_bfloat16* state_cache =
        reinterpret_cast<const __hip_bfloat16*>(state_cache_ptr);
    hipStream_t stream = reinterpret_cast<hipStream_t>(stream_ptr);

    constexpr int WAVES_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;  // 4
    int num_blocks = (num_tokens + WAVES_PER_BLOCK - 1) / WAVES_PER_BLOCK;
    dim3 grid(num_blocks);
    dim3 block(BLOCK_SIZE);

    #define IDX_FWD_ARGS \
        num_tokens, state_cache, state_stride0, state_stride1, \
        ape_raw, ape_stride, token_to_req_indices, positions, slot_mapping, \
        block_table, block_table_stride, block_size, \
        rms_norm_weight_t, rms_norm_eps, cos_sin_cache, cos_sin_stride, \
        kv_cache, kv_slot_mapping, kv_cache_block_size, kv_block_stride, scale_dim

    // Parens around the templated kernel name: hipLaunchKernelGGL macro would
    // otherwise split the template args on the comma.
    if (rms_norm_weight_is_bf16) {
        const __hip_bfloat16* rms_norm_weight_t =
            reinterpret_cast<const __hip_bfloat16*>(rms_norm_weight);
        if (use_fp4_cache) {
            hipLaunchKernelGGL((indexer_compress_kernel<QFMT_MXFP4, __hip_bfloat16>),
                               grid, block, 0, stream, IDX_FWD_ARGS);
        } else {
            hipLaunchKernelGGL((indexer_compress_kernel<QFMT_FP8, __hip_bfloat16>),
                               grid, block, 0, stream, IDX_FWD_ARGS);
        }
    } else {
        const float* rms_norm_weight_t =
            reinterpret_cast<const float*>(rms_norm_weight);
        if (use_fp4_cache) {
            hipLaunchKernelGGL((indexer_compress_kernel<QFMT_MXFP4, float>),
                               grid, block, 0, stream, IDX_FWD_ARGS);
        } else {
            hipLaunchKernelGGL((indexer_compress_kernel<QFMT_FP8, float>),
                               grid, block, 0, stream, IDX_FWD_ARGS);
        }
    }
    #undef IDX_FWD_ARGS
}
