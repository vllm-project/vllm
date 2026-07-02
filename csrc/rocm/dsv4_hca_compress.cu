// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
/**
 * HIP HCA Compressor Kernel - DeepSeek-V4 Ratio=128 Main Path (BF16 state)
 *
 * HCA differs from CSA (ratio=4) in the front half only:
 *   - ratio = 128, K_POOL = 128 (NO overlap, coff=1) -> each boundary
 *     softmax-weight-sums 128 state rows (vs 8 for CSA).
 *   - state_width = HEAD_SIZE = 512 (single region; no k>=4 head offset).
 *   - boundary: (position + 1) % 128 == 0; window = [position-127, position]
 *     (always >= 0, so no padding mask is needed).
 *   - APE is raw [128, HEAD_SIZE] fp32; expanded[k] = ape[k, d].
 *   - state-cache page size (block_size) is 8 (vs 4 for CSA).
 * The output half (RMSNorm warp_reduce, native v_cvt_pk_fp8_f32 E4M3 + UE8M0
 * scale + GPT-J bf16 RoPE, packed dwordx2 store) is identical to CSA and is
 * factored into hca_write_output() (called by one full wave).
 *
 * Two kernels:
 *   v0  hca_compress_kernel        - wave-per-boundary, K=128 serial.
 *       Good when boundaries are dense enough to fill the machine (huge N);
 *       4 tokens/block, no LDS.
 *   v1  hca_compress_ksplit_kernel<NW>  - block-per-boundary, K split across
 *       NW waves + LDS cross-wave online-softmax reduce. Lifts occupancy when
 *       boundaries are sparse (ratio=128 -> always sparse at small/mid N).
 * launch_hca_compress() auto-dispatches NW by estimated boundary count.
 *
 * Built into the _rocm_C extension for all VLLM_GPU_ARCHES. The native FP8 cvt
 * builtin (in hca_write_output) exists only on gfx950 (CDNA4), so that helper's
 * BODY is compiled only for the gfx950 device pass (and the host pass for symbol
 * parity); on other device passes it is an empty stub. Everything else (the
 * kernels, build_plan, launchers) always compiles, and the Python dispatcher
 * only routes to this op on gfx950.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <cmath>
#include <cstdint>

#include "dsv4_compress_common.cuh"
using namespace dsv4;  // HEAD_SIZE, WARP_SIZE, DIMS_PER_LANE, warp_reduce_sum, write_output_512, ...

// HCA-specific config (ratio=128, no overlap). Shared 512-dim constants + the
// output writer live in dsv4_compress_common.cuh.
[[maybe_unused]] constexpr int RATIO = 128;
[[maybe_unused]] constexpr int K_POOL = RATIO;          // no overlap -> K_POOL == ratio
[[maybe_unused]] constexpr int STATE_WIDTH = HEAD_SIZE; // no overlap: kv | score each HEAD_SIZE
[[maybe_unused]] constexpr int V0_BLOCK_SIZE = 256;     // v0: 4 waves (4 tokens/block)

// Per-(k) online-softmax update of one lane's 8 dims, reading row k of the
// state cache + APE. Used by both kernels.
__device__ __forceinline__ void hca_softmax_step(
    int k, int64_t start_pos, int32_t req_idx,
    const __hip_bfloat16* __restrict__ state_cache,
    int64_t state_stride0, int64_t state_stride1,
    const int32_t* __restrict__ block_table, int64_t block_table_stride,
    int32_t block_size, const float* __restrict__ ape, int64_t ape_stride,
    int dim_base, float m[DIMS_PER_LANE], float l[DIMS_PER_LANE],
    float acc[DIMS_PER_LANE]
) {
#if defined(DSV4_COMPILE_GFX950_BODY)
    int64_t pos_k = start_pos + k;
    if (pos_k < 0) return;  // dead for HCA (boundary position >= 127)
    int64_t blk_idx = pos_k / block_size;
    int64_t blk_off = pos_k % block_size;
    int32_t blk_num = block_table[req_idx * block_table_stride + blk_idx];
    const __hip_bfloat16* row = state_cache + blk_num * state_stride0
                              + blk_off * state_stride1;
    union { float4 v; __hip_bfloat16 h[8]; } us, uk;
    us.v = *reinterpret_cast<const float4*>(row + STATE_WIDTH + dim_base);
    uk.v = *reinterpret_cast<const float4*>(row + dim_base);
    const float* ape_ptr = ape + (int64_t)k * ape_stride + dim_base;
    #pragma unroll
    for (int d = 0; d < DIMS_PER_LANE; d++) {
        float s = __bfloat162float(us.h[d]) + ape_ptr[d];
        float kvv = __bfloat162float(uk.h[d]);
        float new_m = fmaxf(m[d], s);
        float corr = __expf(m[d] - new_m);
        float p = __expf(s - new_m);
        l[d] = l[d] * corr + p;
        acc[d] = acc[d] * corr + p * kvv;
        m[d] = new_m;
    }
#else
    dsv4::ignore_unused(k, start_pos, req_idx, state_cache, state_stride0,
                        state_stride1, block_table, block_table_stride,
                        block_size, ape, ape_stride, dim_base, m, l, acc);
#endif  // DSV4_COMPILE_GFX950_BODY
}

// ============================================================================
// v0: wave-per-boundary, K=128 serial. grid = ceil(num_tokens / 4).
// ============================================================================
template <typename WeightT>
__global__ void hca_compress_kernel(
    int num_tokens,
    const __hip_bfloat16* __restrict__ state_cache,
    int64_t state_stride0, int64_t state_stride1,
    const float* __restrict__ ape, int64_t ape_stride,
    const int32_t* __restrict__ token_to_req_indices,
    const int64_t* __restrict__ positions,
    const int64_t* __restrict__ slot_mapping,
    const int32_t* __restrict__ block_table, int64_t block_table_stride,
    int32_t block_size,
    const WeightT* __restrict__ rms_norm_weight,
    float rms_norm_eps,
    const float* __restrict__ cos_sin_cache, int64_t cos_sin_stride,
    uint8_t* __restrict__ kv_cache, const int64_t* __restrict__ kv_slot_mapping,
    int32_t kv_cache_block_size, int64_t kv_block_stride, int32_t scale_dim
) {
#if defined(DSV4_COMPILE_GFX950_BODY)
    int wave_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int token_idx = blockIdx.x * (V0_BLOCK_SIZE / WARP_SIZE) + wave_id;
    if (token_idx >= num_tokens) return;

    int64_t kv_slot_idx = kv_slot_mapping[token_idx];
    int64_t position = positions[token_idx];
    if (((position + 1) & (RATIO - 1)) != 0 || kv_slot_idx < 0) return;
    if (slot_mapping[token_idx] < 0) return;

    int32_t req_idx = token_to_req_indices[token_idx];
    int64_t start_pos = position - (K_POOL - 1);
    int dim_base = lane_id * DIMS_PER_LANE;

    float m[DIMS_PER_LANE], l[DIMS_PER_LANE], acc[DIMS_PER_LANE];
    #pragma unroll
    for (int d = 0; d < DIMS_PER_LANE; d++) { m[d] = -INFINITY; l[d] = 0.0f; acc[d] = 0.0f; }
    for (int k = 0; k < K_POOL; k++) {
        hca_softmax_step(k, start_pos, req_idx, state_cache, state_stride0,
                         state_stride1, block_table, block_table_stride,
                         block_size, ape, ape_stride, dim_base, m, l, acc);
    }
    float comp[DIMS_PER_LANE];
    #pragma unroll
    for (int d = 0; d < DIMS_PER_LANE; d++) comp[d] = acc[d] / l[d];

    write_output_512(comp, lane_id, dim_base, position, kv_slot_idx, RATIO,
                     rms_norm_weight, rms_norm_eps,
                     cos_sin_cache, cos_sin_stride, kv_cache,
                     kv_cache_block_size, kv_block_stride, scale_dim);
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

// K-split body: NW waves split the 128 rows, LDS cross-wave online-softmax
// reduce, wave 0 writes output. Called uniformly by all block threads for a
// resolved boundary (token_idx/position/kv_slot_idx already known valid).
//
// (A dim-tiled merge to halve the LDS — NW*256*3 over 2 passes instead of
// NW*512*3 — was tried to lift occupancy from the 48KB-LDS-locked 2 waves/SIMD
// toward 4. It REGRESSED 5-37%: VGPR rose 118->130 (comp[] lives across passes)
// capping occupancy at 3, and the extra barriers serialized the merge and hurt
// large-N throughput. The 2-waves/SIMD occupancy was not the real limiter —
// large N is bandwidth-bound (~45% peak), small N is boundary-count-bound.
// Reverted; single-pass merge below.)
template <int NW, typename WeightT>
__device__ __forceinline__ void hca_ksplit_body(
    int token_idx, int64_t position, int64_t kv_slot_idx,
    const __hip_bfloat16* __restrict__ state_cache,
    int64_t state_stride0, int64_t state_stride1,
    const float* __restrict__ ape, int64_t ape_stride,
    const int32_t* __restrict__ token_to_req_indices,
    const int32_t* __restrict__ block_table, int64_t block_table_stride,
    int32_t block_size,
    const WeightT* __restrict__ rms_norm_weight,
    float rms_norm_eps,
    const float* __restrict__ cos_sin_cache, int64_t cos_sin_stride,
    uint8_t* __restrict__ kv_cache,
    int32_t kv_cache_block_size, int64_t kv_block_stride, int32_t scale_dim
) {
#if defined(DSV4_COMPILE_GFX950_BODY)
    constexpr int K_PER_WAVE = K_POOL / NW;
    int wid = threadIdx.x / WARP_SIZE;     // 0..NW-1
    int lane_id = threadIdx.x % WARP_SIZE; // 0..63
    int dim_base = lane_id * DIMS_PER_LANE;
    int32_t req_idx = token_to_req_indices[token_idx];
    int64_t start_pos = position - (K_POOL - 1);

    float m[DIMS_PER_LANE], l[DIMS_PER_LANE], acc[DIMS_PER_LANE];
    #pragma unroll
    for (int d = 0; d < DIMS_PER_LANE; d++) { m[d] = -INFINITY; l[d] = 0.0f; acc[d] = 0.0f; }
    int k0 = wid * K_PER_WAVE;
    #pragma unroll
    for (int kk = 0; kk < K_PER_WAVE; kk++) {
        hca_softmax_step(k0 + kk, start_pos, req_idx, state_cache, state_stride0,
                         state_stride1, block_table, block_table_stride,
                         block_size, ape, ape_stride, dim_base, m, l, acc);
    }

    __shared__ float sm[NW * HEAD_SIZE];
    __shared__ float sl[NW * HEAD_SIZE];
    __shared__ float sa[NW * HEAD_SIZE];
    #pragma unroll
    for (int d = 0; d < DIMS_PER_LANE; d++) {
        int idx = wid * HEAD_SIZE + dim_base + d;
        sm[idx] = m[d]; sl[idx] = l[d]; sa[idx] = acc[d];
    }
    __syncthreads();
    if (wid != 0) return;  // only wave 0 reduces + writes output

    float comp[DIMS_PER_LANE];
    #pragma unroll
    for (int d = 0; d < DIMS_PER_LANE; d++) {
        int dd = dim_base + d;
        float mg = -INFINITY;
        #pragma unroll
        for (int w = 0; w < NW; w++) mg = fmaxf(mg, sm[w * HEAD_SIZE + dd]);
        float ls = 0.0f, as = 0.0f;
        #pragma unroll
        for (int w = 0; w < NW; w++) {
            float sc = __expf(sm[w * HEAD_SIZE + dd] - mg);
            ls += sl[w * HEAD_SIZE + dd] * sc;
            as += sa[w * HEAD_SIZE + dd] * sc;
        }
        comp[d] = as / ls;
    }
    write_output_512(comp, lane_id, dim_base, position, kv_slot_idx, RATIO,
                     rms_norm_weight, rms_norm_eps,
                     cos_sin_cache, cos_sin_stride, kv_cache,
                     kv_cache_block_size, kv_block_stride, scale_dim);
#else
    dsv4::ignore_unused(token_idx, position, kv_slot_idx, state_cache,
                        state_stride0, state_stride1, ape, ape_stride,
                        token_to_req_indices, block_table, block_table_stride,
                        block_size, rms_norm_weight, rms_norm_eps,
                        cos_sin_cache, cos_sin_stride, kv_cache,
                        kv_cache_block_size, kv_block_stride, scale_dim);
#endif  // DSV4_COMPILE_GFX950_BODY
}

// ----------------------------------------------------------------------------
// v1: plan-free. grid = num_tokens; block-per-token; non-boundary blocks bail.
// ----------------------------------------------------------------------------
template <int NW, typename WeightT>
__global__ void __launch_bounds__(WARP_SIZE * NW)
hca_compress_ksplit_kernel(
    int num_tokens,
    const __hip_bfloat16* __restrict__ state_cache,
    int64_t state_stride0, int64_t state_stride1,
    const float* __restrict__ ape, int64_t ape_stride,
    const int32_t* __restrict__ token_to_req_indices,
    const int64_t* __restrict__ positions,
    const int64_t* __restrict__ slot_mapping,
    const int32_t* __restrict__ block_table, int64_t block_table_stride,
    int32_t block_size,
    const WeightT* __restrict__ rms_norm_weight,
    float rms_norm_eps,
    const float* __restrict__ cos_sin_cache, int64_t cos_sin_stride,
    uint8_t* __restrict__ kv_cache, const int64_t* __restrict__ kv_slot_mapping,
    int32_t kv_cache_block_size, int64_t kv_block_stride, int32_t scale_dim
) {
#if defined(DSV4_COMPILE_GFX950_BODY)
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;
    int64_t kv_slot_idx = kv_slot_mapping[token_idx];
    int64_t position = positions[token_idx];
    if (((position + 1) & (RATIO - 1)) != 0 || kv_slot_idx < 0) return;
    if (slot_mapping[token_idx] < 0) return;
    hca_ksplit_body<NW, WeightT>(token_idx, position, kv_slot_idx, state_cache,
        state_stride0, state_stride1, ape, ape_stride, token_to_req_indices,
        block_table, block_table_stride, block_size, rms_norm_weight,
        rms_norm_eps, cos_sin_cache, cos_sin_stride, kv_cache, kv_cache_block_size,
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
// v2: compact plan. A prep kernel atomically compacts boundary token indices
// into plan[]; the compress kernel launches grid = plan_capacity (host upper
// bound) and bails on sentinel rows -> no num_tokens-sized wasted launches.
// ============================================================================
extern "C" __global__ void hca_build_plan_kernel(
    int num_tokens,
    const int64_t* __restrict__ positions,
    const int64_t* __restrict__ slot_mapping,
    const int64_t* __restrict__ kv_slot_mapping,
    int32_t* __restrict__ plan,        // [plan_capacity] init to -1
    int32_t* __restrict__ counter      // [1] init to 0
) {
#if defined(DSV4_COMPILE_GFX950_BODY)
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_tokens) return;
    int64_t kv_slot = kv_slot_mapping[t];
    int64_t pos = positions[t];
    if (((pos + 1) & (RATIO - 1)) != 0 || kv_slot < 0) return;
    if (slot_mapping[t] < 0) return;
    int slot = atomicAdd(counter, 1);
    plan[slot] = t;  // plan_capacity is a safe upper bound -> no overflow
#else
    dsv4::ignore_unused(num_tokens, positions, slot_mapping, kv_slot_mapping,
                        plan, counter);
#endif  // DSV4_COMPILE_GFX950_BODY
}

template <int NW, typename WeightT>
__global__ void __launch_bounds__(WARP_SIZE * NW)
hca_compress_plan_ksplit_kernel(
    const int32_t* __restrict__ plan,  // [plan_capacity]; -1 = sentinel
    const __hip_bfloat16* __restrict__ state_cache,
    int64_t state_stride0, int64_t state_stride1,
    const float* __restrict__ ape, int64_t ape_stride,
    const int32_t* __restrict__ token_to_req_indices,
    const int64_t* __restrict__ positions,
    const int32_t* __restrict__ block_table, int64_t block_table_stride,
    int32_t block_size,
    const WeightT* __restrict__ rms_norm_weight,
    float rms_norm_eps,
    const float* __restrict__ cos_sin_cache, int64_t cos_sin_stride,
    uint8_t* __restrict__ kv_cache, const int64_t* __restrict__ kv_slot_mapping,
    int32_t kv_cache_block_size, int64_t kv_block_stride, int32_t scale_dim
) {
#if defined(DSV4_COMPILE_GFX950_BODY)
    int token_idx = plan[blockIdx.x];
    if (token_idx < 0) return;  // sentinel padding row
    int64_t position = positions[token_idx];
    int64_t kv_slot_idx = kv_slot_mapping[token_idx];
    hca_ksplit_body<NW, WeightT>(token_idx, position, kv_slot_idx, state_cache,
        state_stride0, state_stride1, ape, ape_stride, token_to_req_indices,
        block_table, block_table_stride, block_size, rms_norm_weight,
        rms_norm_eps, cos_sin_cache, cos_sin_stride, kv_cache,
        kv_cache_block_size, kv_block_stride, scale_dim);
#else
    dsv4::ignore_unused(plan, state_cache, state_stride0, state_stride1, ape,
                        ape_stride, token_to_req_indices, positions,
                        block_table, block_table_stride, block_size,
                        rms_norm_weight, rms_norm_eps, cos_sin_cache,
                        cos_sin_stride, kv_cache, kv_slot_mapping,
                        kv_cache_block_size, kv_block_stride, scale_dim);
#endif  // DSV4_COMPILE_GFX950_BODY
}

// ============================================================================
// Launchers
// ============================================================================

template <int NW, typename WeightT>
static void ksplit_launch(
    int num_tokens, const __hip_bfloat16* state_cache,
    int64_t state_stride0, int64_t state_stride1,
    const float* ape, int64_t ape_stride,
    const int32_t* token_to_req_indices, const int64_t* positions,
    const int64_t* slot_mapping, const int32_t* block_table,
    int64_t block_table_stride, int32_t block_size,
    const WeightT* rms_norm_weight,
    float rms_norm_eps,
    const float* cos_sin_cache, int64_t cos_sin_stride,
    uint8_t* kv_cache, const int64_t* kv_slot_mapping,
    int32_t kv_cache_block_size, int64_t kv_block_stride, int32_t scale_dim,
    hipStream_t stream
) {
    dim3 grid(num_tokens);
    dim3 block(WARP_SIZE * NW);
    hipLaunchKernelGGL(
        (hca_compress_ksplit_kernel<NW, WeightT>), grid, block, 0, stream,
        num_tokens, state_cache, state_stride0, state_stride1,
        ape, ape_stride, token_to_req_indices, positions, slot_mapping,
        block_table, block_table_stride, block_size,
        rms_norm_weight, rms_norm_eps,
        cos_sin_cache, cos_sin_stride,
        kv_cache, kv_slot_mapping, kv_cache_block_size, kv_block_stride, scale_dim);
}

template <typename WeightT>
static void v0_launch(
    int num_tokens, const __hip_bfloat16* state_cache,
    int64_t state_stride0, int64_t state_stride1,
    const float* ape, int64_t ape_stride,
    const int32_t* token_to_req_indices, const int64_t* positions,
    const int64_t* slot_mapping, const int32_t* block_table,
    int64_t block_table_stride, int32_t block_size,
    const WeightT* rms_norm_weight,
    float rms_norm_eps,
    const float* cos_sin_cache, int64_t cos_sin_stride,
    uint8_t* kv_cache, const int64_t* kv_slot_mapping,
    int32_t kv_cache_block_size, int64_t kv_block_stride, int32_t scale_dim,
    hipStream_t stream
) {
    int num_blocks = (num_tokens + (V0_BLOCK_SIZE / WARP_SIZE) - 1) / (V0_BLOCK_SIZE / WARP_SIZE);
    hipLaunchKernelGGL(
        (hca_compress_kernel<WeightT>), dim3(num_blocks), dim3(V0_BLOCK_SIZE), 0, stream,
        num_tokens, state_cache, state_stride0, state_stride1,
        ape, ape_stride, token_to_req_indices, positions, slot_mapping,
        block_table, block_table_stride, block_size,
        rms_norm_weight, rms_norm_eps,
        cos_sin_cache, cos_sin_stride,
        kv_cache, kv_slot_mapping, kv_cache_block_size, kv_block_stride, scale_dim);
}

template <int NW, typename WeightT>
static void plan_ksplit_launch(
    int plan_capacity, const int32_t* plan, const __hip_bfloat16* state_cache,
    int64_t state_stride0, int64_t state_stride1,
    const float* ape, int64_t ape_stride,
    const int32_t* token_to_req_indices, const int64_t* positions,
    const int32_t* block_table, int64_t block_table_stride, int32_t block_size,
    const WeightT* rms_norm_weight,
    float rms_norm_eps,
    const float* cos_sin_cache, int64_t cos_sin_stride,
    uint8_t* kv_cache, const int64_t* kv_slot_mapping,
    int32_t kv_cache_block_size, int64_t kv_block_stride, int32_t scale_dim,
    hipStream_t stream
) {
    hipLaunchKernelGGL(
        (hca_compress_plan_ksplit_kernel<NW, WeightT>),
        dim3(plan_capacity), dim3(WARP_SIZE * NW),
        0, stream, plan, state_cache, state_stride0, state_stride1,
        ape, ape_stride, token_to_req_indices, positions,
        block_table, block_table_stride, block_size,
        rms_norm_weight, rms_norm_eps,
        cos_sin_cache, cos_sin_stride,
        kv_cache, kv_slot_mapping, kv_cache_block_size, kv_block_stride, scale_dim);
}

#define HCA_FWD_ARGS \
    state_cache, state_stride0, state_stride1, ape_raw, ape_stride, \
    token_to_req_indices, positions, slot_mapping, block_table, \
    block_table_stride, block_size, rms_norm_weight_t, rms_norm_eps, \
    cos_sin_cache, cos_sin_stride, kv_cache, kv_slot_mapping, \
    kv_cache_block_size, kv_block_stride, scale_dim, stream

#define HCA_PLAN_ARGS \
    state_cache, state_stride0, state_stride1, ape_raw, ape_stride, \
    token_to_req_indices, positions, block_table, block_table_stride, block_size, \
    rms_norm_weight_t, rms_norm_eps, cos_sin_cache, cos_sin_stride, kv_cache, \
    kv_slot_mapping, kv_cache_block_size, kv_block_stride, scale_dim, stream

// nw_select: explicit NW override (0 = auto by estimated boundary count).
extern "C" void launch_hca_compress_nw(
    int num_tokens, int nw_select,
    const void* state_cache_ptr, int64_t state_stride0, int64_t state_stride1,
    const float* ape_raw, int64_t ape_stride,
    const int32_t* token_to_req_indices, const int64_t* positions,
    const int64_t* slot_mapping, const int32_t* block_table,
    int64_t block_table_stride, int32_t block_size,
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

    int nw = nw_select;
    if (nw == 0) {
        // Auto: estimate boundaries ~= num_tokens / RATIO. Tuned on MI350 graph
        // timing (v1 K-split). Few boundaries -> NW=8 (max blocks for occupancy);
        // more boundaries -> NW=4 (48KB LDS at NW=8 caps occupancy to 1 block/CU,
        // 24KB at NW=4 allows 2). NW=4 wins from ~128 boundaries up through the
        // tested range (256); revisit NW=2/1 with much larger N (machine saturates).
        int est = num_tokens / RATIO;
        nw = (est <= 64) ? 8 : 4;
    }
    if (rms_norm_weight_is_bf16) {
        const __hip_bfloat16* rms_norm_weight_t =
            reinterpret_cast<const __hip_bfloat16*>(rms_norm_weight);
        switch (nw) {
            case 8: ksplit_launch<8>(num_tokens, HCA_FWD_ARGS); break;
            case 4: ksplit_launch<4>(num_tokens, HCA_FWD_ARGS); break;
            case 2: ksplit_launch<2>(num_tokens, HCA_FWD_ARGS); break;
            default: v0_launch(num_tokens, HCA_FWD_ARGS); break;  // nw == 1
        }
    } else {
        const float* rms_norm_weight_t =
            reinterpret_cast<const float*>(rms_norm_weight);
        switch (nw) {
            case 8: ksplit_launch<8>(num_tokens, HCA_FWD_ARGS); break;
            case 4: ksplit_launch<4>(num_tokens, HCA_FWD_ARGS); break;
            case 2: ksplit_launch<2>(num_tokens, HCA_FWD_ARGS); break;
            default: v0_launch(num_tokens, HCA_FWD_ARGS); break;  // nw == 1
        }
    }
}

// Default entry (used by the torch binding): auto NW dispatch.
extern "C" void launch_hca_compress(
    int num_tokens,
    const void* state_cache_ptr, int64_t state_stride0, int64_t state_stride1,
    const float* ape_raw, int64_t ape_stride,
    const int32_t* token_to_req_indices, const int64_t* positions,
    const int64_t* slot_mapping, const int32_t* block_table,
    int64_t block_table_stride, int32_t block_size,
    const void* rms_norm_weight, bool rms_norm_weight_is_bf16,
    float rms_norm_eps,
    const float* cos_sin_cache, int64_t cos_sin_stride,
    uint8_t* kv_cache, const int64_t* kv_slot_mapping,
    int32_t kv_cache_block_size, int64_t kv_block_stride, int32_t scale_dim,
    void* stream_ptr
) {
    launch_hca_compress_nw(
        num_tokens, 0, state_cache_ptr, state_stride0, state_stride1,
        ape_raw, ape_stride, token_to_req_indices, positions, slot_mapping,
        block_table, block_table_stride, block_size, rms_norm_weight,
        rms_norm_weight_is_bf16, rms_norm_eps, cos_sin_cache, cos_sin_stride,
        kv_cache, kv_slot_mapping,
        kv_cache_block_size, kv_block_stride, scale_dim, stream_ptr);
}

// v2: compact-plan path. Caller pre-allocates scratch: plan[plan_capacity] i32
// and counter[1] i32 (sized plan_capacity >= num_tokens/RATIO + num_reqs).
// Two memsets + build_plan + compact compress launch (grid = plan_capacity);
// all graph-capturable. nw_select: 0 = auto.
extern "C" void launch_hca_compress_plan(
    int num_tokens, int nw_select,
    const void* state_cache_ptr, int64_t state_stride0, int64_t state_stride1,
    const float* ape_raw, int64_t ape_stride,
    const int32_t* token_to_req_indices, const int64_t* positions,
    const int64_t* slot_mapping, const int32_t* block_table,
    int64_t block_table_stride, int32_t block_size,
    const void* rms_norm_weight, bool rms_norm_weight_is_bf16,
    float rms_norm_eps,
    const float* cos_sin_cache, int64_t cos_sin_stride,
    uint8_t* kv_cache, const int64_t* kv_slot_mapping,
    int32_t kv_cache_block_size, int64_t kv_block_stride, int32_t scale_dim,
    int32_t* plan, int32_t* counter, int plan_capacity,
    void* stream_ptr
) {
    if (num_tokens == 0) return;
    const __hip_bfloat16* state_cache =
        reinterpret_cast<const __hip_bfloat16*>(state_cache_ptr);
    hipStream_t stream = reinterpret_cast<hipStream_t>(stream_ptr);

    (void)hipMemsetAsync(counter, 0, sizeof(int32_t), stream);
    (void)hipMemsetAsync(plan, 0xFF, (size_t)plan_capacity * sizeof(int32_t), stream);  // -1
    int prep_blocks = (num_tokens + 255) / 256;
    hipLaunchKernelGGL(hca_build_plan_kernel, dim3(prep_blocks), dim3(256), 0, stream,
                       num_tokens, positions, slot_mapping, kv_slot_mapping, plan, counter);

    // Auto NW (compact-plan path). Swept on MI350 graph timing: with the compact
    // grid (~num_compress blocks), NW=8 wins all the way to ~512 boundaries
    // (its 48KB LDS caps occupancy to 1 block/CU, but there are few blocks so it
    // doesn't bite, and the higher per-boundary parallelism dominates); NW=4
    // takes over at ~1024+ boundaries. (This differs from the plan-FREE v1
    // optimum, est<=64, because there NW=8 also paid num_tokens-block + LDS
    // overhead.) Crossover ~640 boundaries.
    int nw = nw_select;
    if (nw == 0) { int est = num_tokens / RATIO; nw = (est <= 640) ? 8 : 4; }
    if (rms_norm_weight_is_bf16) {
        const __hip_bfloat16* rms_norm_weight_t =
            reinterpret_cast<const __hip_bfloat16*>(rms_norm_weight);
        switch (nw) {
            case 8: plan_ksplit_launch<8>(plan_capacity, plan, HCA_PLAN_ARGS); break;
            case 4: plan_ksplit_launch<4>(plan_capacity, plan, HCA_PLAN_ARGS); break;
            case 2: plan_ksplit_launch<2>(plan_capacity, plan, HCA_PLAN_ARGS); break;
            default: plan_ksplit_launch<1>(plan_capacity, plan, HCA_PLAN_ARGS); break;
        }
    } else {
        const float* rms_norm_weight_t =
            reinterpret_cast<const float*>(rms_norm_weight);
        switch (nw) {
            case 8: plan_ksplit_launch<8>(plan_capacity, plan, HCA_PLAN_ARGS); break;
            case 4: plan_ksplit_launch<4>(plan_capacity, plan, HCA_PLAN_ARGS); break;
            case 2: plan_ksplit_launch<2>(plan_capacity, plan, HCA_PLAN_ARGS); break;
            default: plan_ksplit_launch<1>(plan_capacity, plan, HCA_PLAN_ARGS); break;
        }
    }
}
