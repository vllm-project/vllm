// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
/**
 * HIP CSA Compressor Kernel - DeepSeek-V4 Ratio=4 Specialized BF16 Fast Path
 *
 * Fully fused, plan-free kernel for gfx950 (CDNA4 / MI350):
 *   - wave-per-boundary mapping (each wave owns one token, 8 dims/lane);
 *     RMSNorm variance is a single warp_reduce_sum -> NO __syncthreads, NO LDS.
 *   - single fused load pass + online softmax (kv and score read together).
 *     (A two-pass max-then-exp form halves the transcendentals but adds a
 *     second score load pass; benchmarked -14% at 32k prefill because this
 *     kernel is load/bandwidth-bound at scale, not exp-bound. Kept single-pass.)
 *   - on-device boundary derivation: launched per-TOKEN, non-boundary waves
 *     return immediately, so there is NO host plan / no positions[].item() syncs.
 *   - raw APE [4, 2*HEAD_SIZE] read directly (no host-side expansion).
 * 128-bit vectorized loads; native v_cvt_pk_fp8_f32 E4M3 quant; 79 VGPR /
 * 0 spill -> 6 waves/SIMD.
 *
 * Built into the _rocm_C extension for all VLLM_GPU_ARCHES. The native FP8 cvt
 * builtin exists only on gfx950 (CDNA4), so the kernel BODY is compiled only for
 * the gfx950 device pass (and the host pass for symbol parity); on other device
 * passes it is an empty stub. The host launcher always compiles, and the Python
 * dispatcher only routes to this op on gfx950.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <cmath>
#include <cstdint>

#include "dsv4_compress_common.cuh"
using namespace dsv4;  // HEAD_SIZE, WARP_SIZE, DIMS_PER_LANE, warp_reduce_sum, write_output_512, ...

// CSA-specific config (ratio=4, overlap). Shared 512-dim constants + the output
// writer live in dsv4_compress_common.cuh.
[[maybe_unused]] constexpr int K_POOL = 8;
[[maybe_unused]] constexpr int STATE_WIDTH = 1024;  // overlap mode: 2 * HEAD_SIZE
[[maybe_unused]] constexpr int BLOCK_SIZE = 256;    // 4 waves

// (An XCD-locality block remap was tried here to capture the K_POOL overlap
// reuse -- adjacent boundaries share 4 of 8 rows. It only lifted 32k L2 hit
// 30%->32% and REGRESSED device time -8%: the default sequential block order is
// already HBM-streaming friendly, and intra-XCD concurrency evicts the shared
// rows faster than the reuse distance, so static swizzle loses streaming
// locality without capturing reuse. Reverted. The overlap reuse needs explicit
// per-wave load-union, not scheduling alone.)

// ============================================================================
// Kernel: fully fused, plan-free, wave-per-boundary
// ============================================================================
// block = 256 threads = 4 waves; each wave maps to ONE token and owns 8
// contiguous hidden dims (64 lanes * 8 = 512). The boundary is derived
// on-device; non-boundary waves return immediately. grid = ceil(num_tokens/4).
template <typename WeightT>
__global__ void csa_compress_kernel(
    int num_tokens,
    const __hip_bfloat16* __restrict__ state_cache,
    int64_t state_stride0,
    int64_t state_stride1,
    const float* __restrict__ ape,          // RAW [4, 2*HEAD_SIZE] fp32
    int64_t ape_stride,                      // = ape.stride(0) (e.g. 1024)
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
    int wave_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int token_idx = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + wave_id;
    if (token_idx >= num_tokens) return;

    // On-device boundary derivation (replaces the host plan).
    int64_t kv_slot_idx = kv_slot_mapping[token_idx];
    int64_t position = positions[token_idx];
    if (((position + 1) & 3) != 0 || kv_slot_idx < 0) return;  // not a boundary
    int64_t slot_id = slot_mapping[token_idx];
    if (slot_id < 0) return;

    int32_t req_idx = token_to_req_indices[token_idx];
    int64_t start_pos = position - (K_POOL - 1);
    int dim_base = lane_id * DIMS_PER_LANE;  // 0,8,...,504

    // Phase 1: resolve row base pointers (wave-uniform -> SGPR; gather batches).
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

    // Online softmax: single fused pass loading score AND kv per k. APE is read
    // straight from raw [4, 1024]: expanded[k] = ape[k&3, (k>=4?HEAD:0) + d].
    float m[DIMS_PER_LANE], l[DIMS_PER_LANE], acc[DIMS_PER_LANE];
    #pragma unroll
    for (int d = 0; d < DIMS_PER_LANE; d++) {
        m[d] = -INFINITY; l[d] = 0.0f; acc[d] = 0.0f;
    }
    #pragma unroll
    for (int k = 0; k < K_POOL; k++) {
        if (!valid_k[k]) continue;  // exp(-inf) contributes nothing
        union { float4 v; __hip_bfloat16 h[8]; } us, uk;
        us.v = *reinterpret_cast<const float4*>(row_ptrs[k] + STATE_WIDTH + dim_base);
        uk.v = *reinterpret_cast<const float4*>(row_ptrs[k] + dim_base);
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

    // Shared head_dim=512 output half (RMSNorm + GPT-J RoPE + FP8 + store).
    // ratio=4 selects the CSA RoPE compressed-position mask.
    write_output_512(comp, lane_id, dim_base, position, kv_slot_idx, /*ratio=*/4,
                     rms_norm_weight, rms_norm_eps,
                     cos_sin_cache, cos_sin_stride, kv_cache, kv_cache_block_size,
                     kv_block_stride, scale_dim);
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

extern "C" void launch_csa_compress(
    int num_tokens,
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

    const __hip_bfloat16* state_cache = reinterpret_cast<const __hip_bfloat16*>(state_cache_ptr);
    hipStream_t stream = reinterpret_cast<hipStream_t>(stream_ptr);

    constexpr int WAVES_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;  // 4
    int num_blocks = (num_tokens + WAVES_PER_BLOCK - 1) / WAVES_PER_BLOCK;
    dim3 grid(num_blocks);
    dim3 block(BLOCK_SIZE);

    #define CSA_FWD_ARGS \
        num_tokens, state_cache, state_stride0, state_stride1, \
        ape_raw, ape_stride, token_to_req_indices, positions, slot_mapping, \
        block_table, block_table_stride, block_size, rms_norm_weight_t, \
        rms_norm_eps, cos_sin_cache, cos_sin_stride, kv_cache, kv_slot_mapping, \
        kv_cache_block_size, kv_block_stride, scale_dim

    if (rms_norm_weight_is_bf16) {
        const __hip_bfloat16* rms_norm_weight_t =
            reinterpret_cast<const __hip_bfloat16*>(rms_norm_weight);
        hipLaunchKernelGGL((csa_compress_kernel<__hip_bfloat16>),
                           grid, block, 0, stream, CSA_FWD_ARGS);
    } else {
        const float* rms_norm_weight_t =
            reinterpret_cast<const float*>(rms_norm_weight);
        hipLaunchKernelGGL((csa_compress_kernel<float>),
                           grid, block, 0, stream, CSA_FWD_ARGS);
    }
    #undef CSA_FWD_ARGS
}
