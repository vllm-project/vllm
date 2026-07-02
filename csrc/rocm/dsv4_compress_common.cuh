// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
/**
 * Shared device code for the DeepSeek-V4 fused compressors (gfx950 / CDNA4).
 *
 * The CSA (ratio=4) and HCA (ratio=128) compressors share the entire output
 * half — RMSNorm → GPT-J RoPE → native FP8 E4M3 + UE8M0 scale → packed store —
 * over a head_dim=512 row (8 dims/lane). That writer lives here as
 * ``dsv4::write_output_512`` (the only difference between the two is the RoPE
 * compressed-position mask, passed as ``ratio``). ``dsv4::warp_reduce_sum`` is
 * shared by all three compressors (CSA / HCA / indexer).
 *
 * The native FP8 cvt builtin is CDNA4-only, so the writer body compiles only for
 * the gfx950 device pass (and the host pass for symbol parity); on other device
 * passes it is an empty stub. See dsv4_*_compress.cu for the per-shape kernels.
 */
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <cmath>
#include <cstdint>

// Defined only when the device pass targets gfx950 (CDNA4). Mirrors the
// __HIP__RDNA3__ pattern in q_gemm_rdna3.cu.
#if defined(__HIPCC__) && defined(__gfx950__)
  #define DSV4_GFX950
#endif

#if defined(DSV4_GFX950) || !defined(__HIP_DEVICE_COMPILE__)
  #define DSV4_COMPILE_GFX950_BODY 1
#endif

namespace dsv4 {

template <typename... Args>
__host__ __device__ __forceinline__ void ignore_unused(const Args&...) {}

constexpr int WARP_SIZE = 64;     // AMD wavefront
constexpr int HEAD_SIZE = 512;    // CSA / HCA head_dim
constexpr int ROPE_HEAD_DIM = 64;
constexpr int NOPE_HEAD_DIM = HEAD_SIZE - ROPE_HEAD_DIM;       // 448
constexpr int QUANT_BLOCK = 64;
constexpr int N_NOPE_BLOCKS = NOPE_HEAD_DIM / QUANT_BLOCK;     // 7
constexpr int TOKEN_STRIDE = NOPE_HEAD_DIM + ROPE_HEAD_DIM * 2;  // 576
constexpr int DIMS_PER_LANE = HEAD_SIZE / WARP_SIZE;          // 8
constexpr float FP8_MAX = 448.0f;
constexpr float INV_FP8_MAX = 1.0f / FP8_MAX;

// Warp-level sum reduction (64-lane wavefront).
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

__device__ __forceinline__ float load_rms_norm_weight(
    const float* __restrict__ rms_norm_weight,
    int idx
) {
    return rms_norm_weight[idx];
}

__device__ __forceinline__ float load_rms_norm_weight(
    const __hip_bfloat16* __restrict__ rms_norm_weight,
    int idx
) {
    return __bfloat162float(rms_norm_weight[idx]);
}

// head_dim=512 output half, shared by CSA and HCA. Must be called by all 64
// lanes of a single wave; comp[] holds that wave's 8 dims/lane (512 total).
// RMSNorm (warp_reduce) + GPT-J RoPE + native FP8 E4M3 + UE8M0 scale + paged
// scatter. ``ratio`` selects the RoPE compressed-position mask (4 for CSA, 128
// for HCA).
template <typename WeightT>
__device__ __forceinline__ void write_output_512(
    float comp[DIMS_PER_LANE], int lane_id, int dim_base,
    int64_t position, int64_t kv_slot_idx, int ratio,
    const WeightT* __restrict__ rms_norm_weight,
    float rms_norm_eps,
    const float* __restrict__ cos_sin_cache, int64_t cos_sin_stride,
    uint8_t* __restrict__ kv_cache, int32_t kv_cache_block_size,
    int64_t kv_block_stride, int32_t scale_dim
) {
#if defined(DSV4_COMPILE_GFX950_BODY)
    // RMSNorm: wave-local single warp_reduce_sum (no barrier, no LDS).
    float sq = 0.0f;
    #pragma unroll
    for (int d = 0; d < DIMS_PER_LANE; d++) sq += comp[d] * comp[d];
    float variance = warp_reduce_sum(sq) / HEAD_SIZE;
    float rrms = rsqrtf(variance + rms_norm_eps);

    float normed[DIMS_PER_LANE];
    #pragma unroll
    for (int d = 0; d < DIMS_PER_LANE; d++) {
        float w = load_rms_norm_weight(rms_norm_weight, dim_base + d);
        normed[d] = comp[d] * rrms * w;
    }

    int64_t kv_blk_idx = kv_slot_idx / kv_cache_block_size;
    int64_t kv_pos = kv_slot_idx % kv_cache_block_size;
    uint8_t* cache_block = kv_cache + kv_blk_idx * kv_block_stride;
    uint8_t* fp8_ptr = cache_block + kv_pos * TOKEN_STRIDE;
    uint8_t* scale_ptr = cache_block + kv_cache_block_size * TOKEN_STRIDE
                       + kv_pos * scale_dim;

    // BF16 roundtrip for parity with the fused Triton path.
    float quant[DIMS_PER_LANE];
    #pragma unroll
    for (int d = 0; d < DIMS_PER_LANE; d++) {
        quant[d] = __bfloat162float(__float2bfloat16(normed[d]));
    }

    // NOPE: each lane owns 8 dims = 1/8 of a 64-dim quant block. absmax over the
    // aligned 8-lane group (xor 4,2,1), native E4M3 pack (2 f32 -> 2 fp8/call),
    // one dwordx2 store of the 8 contiguous NOPE bytes; one UE8M0 byte per block.
    if (dim_base < NOPE_HEAD_DIM) {
        float local_max = 0.0f;
        #pragma unroll
        for (int d = 0; d < DIMS_PER_LANE; d++) {
            local_max = fmaxf(local_max, fabsf(quant[d]));
        }
        float rmax = local_max;
        rmax = fmaxf(rmax, __shfl_xor(rmax, 4));
        rmax = fmaxf(rmax, __shfl_xor(rmax, 2));
        rmax = fmaxf(rmax, __shfl_xor(rmax, 1));
        float blk_max = fmaxf(rmax, 1e-4f);

        float raw_scale = blk_max * INV_FP8_MAX;
        float exponent = ceilf(log2f(raw_scale));
        float inv_scale = exp2f(-exponent);
        int enc_scale = (int)(exponent + 127.0f);
        enc_scale = max(0, min(255, enc_scale));

        uint32_t words[2];
        #pragma unroll
        for (int p = 0; p < 2; p++) {
            int b = p * 4;
            int w = 0;
            w = __builtin_amdgcn_cvt_pk_fp8_f32(quant[b]   * inv_scale,
                                                quant[b+1] * inv_scale, w, false);
            w = __builtin_amdgcn_cvt_pk_fp8_f32(quant[b+2] * inv_scale,
                                                quant[b+3] * inv_scale, w, true);
            words[p] = (uint32_t)w;
        }
        *reinterpret_cast<uint64_t*>(fp8_ptr + dim_base) =
            (uint64_t)words[0] | ((uint64_t)words[1] << 32);

        int qb = dim_base / QUANT_BLOCK;
        if ((lane_id % 8) == 0 && qb < N_NOPE_BLOCKS) {
            scale_ptr[qb] = (uint8_t)enc_scale;
        }
    }

    if (lane_id == 0) {
        scale_ptr[N_NOPE_BLOCKS] = 0;  // padding scale
    }

    // ROPE: GPT-J interleaved rotation + bf16 store for dims [NOPE, HEAD_SIZE).
    if (dim_base >= NOPE_HEAD_DIM) {
        __hip_bfloat16* bf16_out =
            reinterpret_cast<__hip_bfloat16*>(fp8_ptr + NOPE_HEAD_DIM);
        int64_t comp_pos = position & (~(int64_t)(ratio - 1));  // (pos/ratio)*ratio
        const float* cs = cos_sin_cache + comp_pos * cos_sin_stride;
        #pragma unroll
        for (int d = 0; d < DIMS_PER_LANE; d += 2) {
            int rope_local = (dim_base - NOPE_HEAD_DIM) + d;
            int cs_idx = rope_local / 2;
            float cos_v = cs[cs_idx];
            float sin_v = cs[ROPE_HEAD_DIM / 2 + cs_idx];
            float e = normed[d], o = normed[d + 1];
            bf16_out[rope_local]     = __float2bfloat16(e * cos_v - o * sin_v);
            bf16_out[rope_local + 1] = __float2bfloat16(o * cos_v + e * sin_v);
        }
    }
#else
    ignore_unused(comp, lane_id, dim_base, position, kv_slot_idx, ratio,
                  rms_norm_weight, rms_norm_eps, cos_sin_cache, cos_sin_stride,
                  kv_cache, kv_cache_block_size, kv_block_stride, scale_dim);
#endif  // DSV4_COMPILE_GFX950_BODY
}

}  // namespace dsv4
