/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 *
 * Horizontally-fused DeepseekV4-MLA kernel:
 *   - Q side:  per-head RMSNorm (no weight) + GPT-J RoPE on last ROPE_DIM
 *   - KV side: GPT-J RoPE on last ROPE_DIM + UE8M0 FP8 quant on NoPE + paged
 *              cache insert
 *
 * Structured after `applyMLARopeAndAssignQKVKernelGeneration` in
 * TensorRT-LLM's mlaKernels.cu: one kernel, one grid, with head-slot
 * dispatch choosing Q vs KV work per warp.  The per-warp RMSNorm/RoPE
 * skeleton is adapted from vllm-deepseek_v4's existing
 * `fusedQKNormRopeKernel` (csrc/fused_qknorm_rope_kernel.cu).
 *
 * Assumptions (hard-coded for DeepseekV4 attention):
 *   HEAD_DIM  = 512
 *   ROPE_DIM  = 64   (RoPE applied to dims [NOPE_DIM, HEAD_DIM))
 *   NOPE_DIM  = 448
 *   QUANT_BLOCK = 64 (UE8M0 FP8 quant block)
 *   FP8_MAX   = 448.0f
 *   is_neox=false (GPT-J interleaved pairs)
 *   cos_sin_cache layout [max_pos, rope_dim] = cos || sin (cos first, sin
 *     second along last dim; each half is rope_dim/2 = 32 values)
 *
 * Cache layout per paged-cache block (block_size tokens):
 *   [0,            bs*576):          token data, 448 fp8 + 128 bf16 each
 *   [bs*576,       bs*576 + bs*8):   UE8M0 scales, 7 real + 1 pad per token
 */

#include "torch_utils.h"

#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/csrc/stable/device.h>

#include <cmath>
#include "cuda_compat.h"
#include "dispatch_utils.h"
#include "type_convert.cuh"

#ifndef USE_ROCM
  #include <cuda_fp8.h>
#else
  #include <hip/hip_fp8.h>
#endif
#include <cuda_runtime.h>
#include <type_traits>

#ifndef FINAL_MASK
  #ifdef USE_ROCM
    #define FINAL_MASK 0xffffffffffffffffULL
  #else
    #define FINAL_MASK 0xffffffffu
  #endif
#endif

#ifdef USE_ROCM
// ROCm-compatible FP8 conversion helpers
__device__ __forceinline__ uint8_t rocm_cvt_float_to_fp8_e4m3(float val) {
  #if defined(HIP_FP8_TYPE_OCP)
  __hip_fp8_e4m3 fp8_val(val);
  #else
  __hip_fp8_e4m3_fnuz fp8_val(val);
  #endif
  return reinterpret_cast<uint8_t&>(fp8_val);
}
#endif

namespace vllm {
namespace deepseek_v4_fused_ops {

namespace {
inline int getSMVersion() {
  auto* props = get_device_prop();
  return props->major * 10 + props->minor;
}
}  // namespace

// ────────────────────────────────────────────────────────────────────────────
// Constants
// ────────────────────────────────────────────────────────────────────────────
constexpr int kHeadDim = 512;
constexpr int kRopeDim = 64;
constexpr int kNopeDim = kHeadDim - kRopeDim;  // 448
constexpr int kQuantBlock = 64;
constexpr int kNumQuantBlocks = kNopeDim / kQuantBlock;   // 7
constexpr int kScaleBytesPerToken = kNumQuantBlocks + 1;  // 8 (7 real + 1 pad)
constexpr int kTokenDataBytes = kNopeDim + kRopeDim * 2;  // 448 + 128 = 576
constexpr float kFp8Max = 448.0f;

#ifndef USE_ROCM
// When num_tokens is less than this threshold,
// run the reduced grid variant on cuda
constexpr float NUM_TOKEN_CUTOFF = 1024;
#endif

// Per-warp layout:  32 lanes × 16 elems/lane = 512 elems = HEAD_DIM.
constexpr int kNumLanes = 32;
constexpr int kElemsPerLane = kHeadDim / kNumLanes;  // 16

// Pack this lane's 16 fp32 elements into per-tensor E4M3 FP8 (one uint4 = 16
// B), scaling by `scale` (a reciprocal scale) and saturating to ±448.  Used by
// the FlashInfer full-cache path for both the Q and KV stores.
__device__ __forceinline__ uint4 packFp8E4M3x16(float const* values,
                                                float const scale) {
#ifndef USE_ROCM
  uint4 out;
  auto* out2 = reinterpret_cast<__nv_fp8x2_storage_t*>(&out);
  #pragma unroll
  for (int i = 0; i < kElemsPerLane / 2; i++) {
    float2 scaled =
        make_float2(values[2 * i] * scale, values[2 * i + 1] * scale);
    scaled.x = fminf(fmaxf(scaled.x, -kFp8Max), kFp8Max);
    scaled.y = fminf(fmaxf(scaled.y, -kFp8Max), kFp8Max);
    out2[i] = __nv_cvt_float2_to_fp8x2(scaled, __NV_SATFINITE, __NV_E4M3);
  }
  return out;
#else
  uint8_t out_bytes[kElemsPerLane];
  #pragma unroll
  for (int i = 0; i < kElemsPerLane; i++) {
    float scaled = values[i] * scale;
    scaled = fminf(fmaxf(scaled, -kFp8Max), kFp8Max);
    out_bytes[i] = rocm_cvt_float_to_fp8_e4m3(scaled);
  }
  return *reinterpret_cast<uint4 const*>(out_bytes);
#endif
}

// ────────────────────────────────────────────────────────────────────────────
// Small inline helpers
// ────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ float warp4MaxAbs(float val) {
  // Reduce absolute max across 4 consecutive lanes (lane id & 3 group).
  float peer = __shfl_xor_sync(FINAL_MASK, val, 1);
  val = fmaxf(val, peer);
  peer = __shfl_xor_sync(FINAL_MASK, val, 2);
  val = fmaxf(val, peer);
  return val;
}

template <typename T>
__device__ __forceinline__ float warpSum(float val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  }
  return val;
}

// ────────────────────────────────────────────────────────────────────────────
// Per-slot inner pipeline
// ────────────────────────────────────────────────────────────────────────────
// Shared by both kernel variants: 1 CTA per (token, head) pair vs. 1 CTA per
// token.  Templated on `kNumHeadsQPadded` so the KV-sentinel comparison and
// q_out stride fold to compile-time constants.
//
// Slot layout (per token):
//   slot < num_heads_q                          → live-Q   (RMSNorm + RoPE,
//                                                           read q_in →
//                                                           write q_out)
//   num_heads_q <= slot < kNumHeadsQPadded      → pad-Q    (zero-fill q_out;
//                                                           v0/v1 unused)
//   slot == kNumHeadsQPadded                    → KV       (RoPE + UE8M0 quant
//                                                           + paged-cache
//                                                           insert)
template <typename scalar_t_in, int kNumHeadsQPadded>
__device__ __forceinline__ void processDeepseekV4Slot(
    uint4 v0, uint4 v1, int const tokenIdx, int const slotIdx,
    int const dim_base, int const laneId, int const num_heads_q,
    float const eps, scalar_t_in* __restrict__ q_out,
    uint8_t* __restrict__ k_cache, int64_t const* __restrict__ slot_mapping,
    int64_t const* __restrict__ position_ids,
    float const* __restrict__ cos_sin_cache, int const cache_block_size,
    int const kv_block_stride) {
  using Converter = vllm::_typeConvert<scalar_t_in>;
  bool const isKV = (slotIdx == kNumHeadsQPadded);
  bool const isPadQ = !isKV && (slotIdx >= num_heads_q);

  // ── Pad-Q branch: write 32 B of zeros and exit. ─────────────────────────
  // FlashMLA reads these slots; bf16 +0.0 is bit pattern 0x0000, so a uint4
  // zero literal is correct.  Matches the live-Q branch's vectorized store.
  if (isPadQ) {
    scalar_t_in* dst =
        q_out +
        (static_cast<int64_t>(tokenIdx) * kNumHeadsQPadded + slotIdx) *
            kHeadDim +
        dim_base;
    uint4 const zero4 = {0u, 0u, 0u, 0u};
    *reinterpret_cast<uint4*>(dst) = zero4;
    *reinterpret_cast<uint4*>(dst + 8) = zero4;
    return;
  }

  // ── Decode the bf16 → 16 fp32 registers ─────────────────────────────
  float elements[kElemsPerLane];
  {
    typename Converter::packed_hip_type const* p0 =
        reinterpret_cast<typename Converter::packed_hip_type const*>(&v0);
    typename Converter::packed_hip_type const* p1 =
        reinterpret_cast<typename Converter::packed_hip_type const*>(&v1);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      float2 f2 = Converter::convert(p0[i]);
      elements[2 * i] = f2.x;
      elements[2 * i + 1] = f2.y;
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
      float2 f2 = Converter::convert(p1[i]);
      elements[8 + 2 * i] = f2.x;
      elements[8 + 2 * i + 1] = f2.y;
    }
  }

  // ── Q branch: RMSNorm (no weight) ───────────────────────────────────
  if (!isKV) {
    float sumOfSquares = 0.0f;
#pragma unroll
    for (int i = 0; i < kElemsPerLane; i++) {
      sumOfSquares += elements[i] * elements[i];
    }
    sumOfSquares = warpSum<float>(sumOfSquares);
    float const rms_rcp =
        rsqrtf(sumOfSquares / static_cast<float>(kHeadDim) + eps);
#pragma unroll
    for (int i = 0; i < kElemsPerLane; i++) {
      elements[i] = elements[i] * rms_rcp;
    }
  }

  // ── GPT-J RoPE on dims [NOPE_DIM, HEAD_DIM) ─────────────────────────────
  // All math in fp32.  cos_sin_cache is loaded as fp32 (its native storage).
  bool const is_rope_lane = dim_base >= kNopeDim;
  if (is_rope_lane) {
    int64_t const pos = position_ids[tokenIdx];
    constexpr int kHalfRope = kRopeDim / 2;
    float const* cos_ptr = cos_sin_cache + pos * kRopeDim;
    float const* sin_ptr = cos_ptr + kHalfRope;

    int const rope_local_base = dim_base - kNopeDim;
    int const half_base = rope_local_base >> 1;

    // Load phase: 4 vectorized LDGs issue back-to-back.
    float4 const c0 = *reinterpret_cast<float4 const*>(cos_ptr + half_base);
    float4 const c1 = *reinterpret_cast<float4 const*>(cos_ptr + half_base + 4);
    float4 const s0 = *reinterpret_cast<float4 const*>(sin_ptr + half_base);
    float4 const s1 = *reinterpret_cast<float4 const*>(sin_ptr + half_base + 4);
    float const cos_arr[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
    float const sin_arr[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};

#pragma unroll
    for (int p = 0; p < kElemsPerLane / 2; p++) {
      float const x_even = elements[2 * p];
      float const x_odd = elements[2 * p + 1];
      elements[2 * p] = x_even * cos_arr[p] - x_odd * sin_arr[p];
      elements[2 * p + 1] = x_even * sin_arr[p] + x_odd * cos_arr[p];
    }
  }

  // ═══════════════════════════════════════════════════════════════════
  // Q / KV branch dispatch. Restructured as if/else (no early `return`)
  // so every code path lands at the same exit point — callers own PDL
  // triggering and per-iteration buffer rotation.
  // ═══════════════════════════════════════════════════════════════════
  if (!isKV) {
    // ── Live-Q: cast back to bf16 and store into the padded q_out. ─────
    uint4 out0, out1;
    typename Converter::packed_hip_type* po0 =
        reinterpret_cast<typename Converter::packed_hip_type*>(&out0);
    typename Converter::packed_hip_type* po1 =
        reinterpret_cast<typename Converter::packed_hip_type*>(&out1);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      po0[i] =
          Converter::convert(make_float2(elements[2 * i], elements[2 * i + 1]));
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
      po1[i] = Converter::convert(
          make_float2(elements[8 + 2 * i], elements[8 + 2 * i + 1]));
    }
    scalar_t_in* dst =
        q_out +
        (static_cast<int64_t>(tokenIdx) * kNumHeadsQPadded + slotIdx) *
            kHeadDim +
        dim_base;
    *reinterpret_cast<uint4*>(dst) = out0;
    *reinterpret_cast<uint4*>(dst + 8) = out1;
  } else {
    // ── KV: FP8 quant on NoPE + bf16 store on RoPE + cache insert.
    int64_t const slot_id = slot_mapping[tokenIdx];
    if (slot_id >= 0) {
      int64_t const block_idx = slot_id / cache_block_size;
      int64_t const pos_in_block = slot_id % cache_block_size;
      uint8_t* block_base =
          k_cache + block_idx * static_cast<int64_t>(kv_block_stride);
      uint8_t* token_fp8_ptr = block_base + pos_in_block * kTokenDataBytes;
      uint8_t* token_bf16_ptr = token_fp8_ptr + kNopeDim;
      uint8_t* token_scale_ptr =
          block_base +
          static_cast<int64_t>(cache_block_size) * kTokenDataBytes +
          pos_in_block * kScaleBytesPerToken;

#pragma unroll
      for (int i = 0; i < kElemsPerLane; i++) {
        elements[i] = Converter::convert(Converter::convert(elements[i]));
      }

      float local_absmax = 0.0f;
#pragma unroll
      for (int i = 0; i < kElemsPerLane; i++) {
        local_absmax = fmaxf(local_absmax, fabsf(elements[i]));
      }
      float const absmax = fmaxf(warp4MaxAbs(local_absmax), 1e-4f);
      float const exponent = ceilf(log2f(absmax / kFp8Max));
      float const inv_scale = exp2f(-exponent);

      if (!is_rope_lane) {
        uint8_t out_bytes[kElemsPerLane];
#pragma unroll
        for (int i = 0; i < kElemsPerLane; i++) {
          float scaled = elements[i] * inv_scale;
          scaled = fminf(fmaxf(scaled, -kFp8Max), kFp8Max);
#ifndef USE_ROCM
          __nv_fp8_storage_t s =
              __nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3);
          out_bytes[i] = static_cast<uint8_t>(s);
#else
          out_bytes[i] = rocm_cvt_float_to_fp8_e4m3(scaled);
#endif
        }
        *reinterpret_cast<uint4*>(token_fp8_ptr + dim_base) =
            *reinterpret_cast<uint4 const*>(out_bytes);

        if ((laneId & 3) == 0) {
          int const q_block_idx = laneId >> 2;
          float encoded = fmaxf(fminf(exponent + 127.0f, 255.0f), 0.0f);
          token_scale_ptr[q_block_idx] = static_cast<uint8_t>(encoded);
        }
        if (laneId == 0) {
          token_scale_ptr[kNumQuantBlocks] = 0;
        }
      } else {
        uint4 out0, out1;
        typename Converter::packed_hip_type* po0 =
            reinterpret_cast<typename Converter::packed_hip_type*>(&out0);
        typename Converter::packed_hip_type* po1 =
            reinterpret_cast<typename Converter::packed_hip_type*>(&out1);
#pragma unroll
        for (int i = 0; i < 4; i++) {
          po0[i] = Converter::convert(
              make_float2(elements[2 * i], elements[2 * i + 1]));
        }
#pragma unroll
        for (int i = 0; i < 4; i++) {
          po1[i] = Converter::convert(
              make_float2(elements[8 + 2 * i], elements[8 + 2 * i + 1]));
        }
        int const rope_local_base = dim_base - kNopeDim;
        scalar_t_in* bf16_dst =
            reinterpret_cast<scalar_t_in*>(token_bf16_ptr) + rope_local_base;
        *reinterpret_cast<uint4*>(bf16_dst) = out0;
        *reinterpret_cast<uint4*>(bf16_dst + 8) = out1;
      }
    }
  }
}

// ────────────────────────────────────────────────────────────────────────────
// Kernel
// ────────────────────────────────────────────────────────────────────────────
//
// Grid: 1D, gridDim.x = ceil(num_tokens_full * (kNumHeadsQPadded + 1) /
// warps_per_block) Block: blockDim.x = 256 threads (8 warps per block) Each
// warp handles one (token, head_slot) pair.
//   slot < num_heads_q                              → live-Q branch
//                                                     (RMSNorm + RoPE,
//                                                      read q_in → write q_out)
//   num_heads_q <= slot < kNumHeadsQPadded          → pad-Q branch
//                                                     (zero-fill q_out)
//   slot == kNumHeadsQPadded                        → KV branch
//                                                     (RoPE + UE8M0 quant +
//                                                      paged-cache insert)
//
// `kNumHeadsQPadded` is a template parameter (compile-time constant) so the
// divisions in the grid math and the KV-sentinel comparison fold to fast
// constant operations.  The launch wrapper dispatches the runtime value to
// the matching instantiation.
//
// With DP padding, q/kv/position_ids can have more rows than slot_mapping.
// The live-Q and pad-Q branches cover all `num_tokens_full` rows (downstream
// attention uses them).  The KV branch only inserts the first
// `num_tokens_insert` tokens (= slot_mapping length) into the paged cache.
//
template <typename scalar_t_in, int kNumHeadsQPadded>
__global__ void fusedDeepseekV4QNormRopeKVRopeQuantInsertKernel(
    scalar_t_in const* __restrict__ q_in,      // [N, num_heads_q,      512]
    scalar_t_in* __restrict__ q_out,           // [N, kNumHeadsQPadded, 512]
    scalar_t_in const* __restrict__ kv_in,     // [N, 512] bf16
    uint8_t* __restrict__ k_cache,             // [num_blocks, block_stride]
    int64_t const* __restrict__ slot_mapping,  // [num_tokens_insert] i64
    int64_t const* __restrict__ position_ids,  // [N] i64
    float const* __restrict__ cos_sin_cache,   // [max_pos, 64] fp32
    float const eps,
    int const num_tokens_full,    // = q.size(0) = kv.size(0)
    int const num_tokens_insert,  // = slot_mapping.size(0), ≤ num_tokens_full
    int const num_heads_q,        // live Q heads (input layout)
    int const cache_block_size,   // tokens per paged-cache block
    int const kv_block_stride) {  // bytes per paged-cache block
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
  // BF16 _typeConvert specialization is unavailable on pre-Ampere.  The
  // DeepseekV4 kernel only runs with bf16 inputs in practice, so compile a
  // no-op stub for sm_70/sm_75 to keep multi-arch builds happy.
  if constexpr (std::is_same_v<scalar_t_in, c10::BFloat16>) {
    return;
  } else {
#endif
    int const warpsPerBlock = blockDim.x / 32;
    int const warpId = threadIdx.x / 32;
    int const laneId = threadIdx.x % 32;
    int const globalWarpIdx = blockIdx.x * warpsPerBlock + warpId;

    constexpr int kTotalSlotsPerToken = kNumHeadsQPadded + 1;
    int const tokenIdx = globalWarpIdx / kTotalSlotsPerToken;
    int const slotIdx = globalWarpIdx % kTotalSlotsPerToken;
    if (tokenIdx >= num_tokens_full) return;

    bool const isKV = (slotIdx == kNumHeadsQPadded);
    bool const isPadQ = !isKV && (slotIdx >= num_heads_q);
    // KV branch: skip DP-padded tokens (no slot reserved for them).
    if (isKV && tokenIdx >= num_tokens_insert) return;

    // PDL: wait for predecessor kernel (upstream q/kv producer) to signal
    // before touching any global memory.  No-op when PDL is not enabled on
    // the launch.  The CUDA runtime wrapper emits the griddepcontrol.wait
    // PTX with the required memory clobber internally.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaGridDependencySynchronize();
#endif

    // Dim range this lane owns within the 512-wide head.
    int const dim_base = laneId * kElemsPerLane;  // in [0, 512) step 16

    // Load only for live-Q and KV slots; pad-Q skips the read (q_in beyond
    // num_heads_q is out of bounds) and the helper zero-fills its output.
    uint4 v0, v1;
    if (!isPadQ) {
      scalar_t_in const* src_ptr;
      if (isKV) {
        src_ptr = kv_in + static_cast<int64_t>(tokenIdx) * kHeadDim + dim_base;
      } else {
        int64_t const q_row_offset =
            (static_cast<int64_t>(tokenIdx) * num_heads_q + slotIdx) *
                kHeadDim +
            dim_base;
        src_ptr = q_in + q_row_offset;
      }
      v0 = *reinterpret_cast<uint4 const*>(src_ptr);
      v1 = *reinterpret_cast<uint4 const*>(src_ptr + 8);
    }

    processDeepseekV4Slot<scalar_t_in, kNumHeadsQPadded>(
        v0, v1, tokenIdx, slotIdx, dim_base, laneId, num_heads_q, eps, q_out,
        k_cache, slot_mapping, position_ids, cos_sin_cache, cache_block_size,
        kv_block_stride);

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaTriggerProgrammaticLaunchCompletion();
#endif
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
  }
#endif
}

// ────────────────────────────────────────────────────────────────────────────
// Kernel
// ────────────────────────────────────────────────────────────────────────────
//
// Grid: 1D, gridDim.x = num_tokens_full
// Block: blockDim.x = 256 threads (8 warps per block) Each
// warp handles one token, iterating over each head.
// Q branch (RMSNorm + RoPE, in place) head_slot == num_heads_q
// KV branch (RoPE + UE8M0 quant + insert)
//
template <typename scalar_t_in, int kNumHeadsQPadded>
__global__ void fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelReducedGrid(
    scalar_t_in const* __restrict__ q_in, scalar_t_in* __restrict__ q_out,
    scalar_t_in const* __restrict__ kv_in, uint8_t* __restrict__ k_cache,
    int64_t const* __restrict__ slot_mapping,
    int64_t const* __restrict__ position_ids,
    float const* __restrict__ cos_sin_cache, float const eps,
    int const num_tokens_full, int const num_tokens_insert,
    int const num_heads_q, int const cache_block_size,
    int const kv_block_stride) {
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
  if constexpr (std::is_same_v<scalar_t_in, c10::BFloat16>) {
    return;
  } else {
#endif
    int const warpsPerBlock = blockDim.x / 32;
    int const warpId = threadIdx.x / 32;
    int const laneId = threadIdx.x % 32;

    int const tokenIdx = blockIdx.x;
    if (tokenIdx >= num_tokens_full) return;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaGridDependencySynchronize();
#endif

    int const dim_base = laneId * kElemsPerLane;  // in [0, 512) step 16
    // Slot enumeration: live-Q + pad-Q + (KV if this token has a slot).
    int const slot_end = (tokenIdx >= num_tokens_insert)
                             ? kNumHeadsQPadded
                             : (kNumHeadsQPadded + 1);

    auto load_slot = [&](int s, uint4& va, uint4& vb) {
      // pad-Q slots skip the load — q_in beyond num_heads_q is OOB.
      if (s >= num_heads_q && s < kNumHeadsQPadded) return;
      scalar_t_in const* src;
      if (s == kNumHeadsQPadded) {
        src = kv_in + static_cast<int64_t>(tokenIdx) * kHeadDim + dim_base;
      } else {
        src = q_in +
              (static_cast<int64_t>(tokenIdx) * num_heads_q +
               static_cast<int64_t>(s)) *
                  kHeadDim +
              dim_base;
      }
      va = *reinterpret_cast<uint4 const*>(src);
      vb = *reinterpret_cast<uint4 const*>(src + 8);
    };

    if (warpId < slot_end) {
      int curr_slot = warpId;
      uint4 v0_curr, v1_curr;
      load_slot(curr_slot, v0_curr, v1_curr);

      while (curr_slot < slot_end) {
        int const next_slot = curr_slot + warpsPerBlock;
        bool const has_next = (next_slot < slot_end);

        // Prefetch src for the next slot
        uint4 v0_next, v1_next;
        if (has_next) {
          load_slot(next_slot, v0_next, v1_next);
        }

        processDeepseekV4Slot<scalar_t_in, kNumHeadsQPadded>(
            v0_curr, v1_curr, tokenIdx, curr_slot, dim_base, laneId,
            num_heads_q, eps, q_out, k_cache, slot_mapping, position_ids,
            cos_sin_cache, cache_block_size, kv_block_stride);

        // ── Buffer rotation: hand the prefetched LDGs to the next iter.
        v0_curr = v0_next;
        v1_curr = v1_next;
        curr_slot = next_slot;
      }  // while
    }  // if (warpId < slot_end)

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaTriggerProgrammaticLaunchCompletion();
#endif
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
  }
#endif
}

// ────────────────────────────────────────────────────────────────────────────
// Launch wrapper
// ────────────────────────────────────────────────────────────────────────────
template <typename scalar_t_in, int kNumHeadsQPadded>
static void launchFusedDeepseekV4Templated(
    scalar_t_in const* q_in, scalar_t_in* q_out, scalar_t_in const* kv_in,
    uint8_t* k_cache, int64_t const* slot_mapping, int64_t const* position_ids,
    float const* cos_sin_cache, float const eps, int const num_tokens_full,
    int const num_tokens_insert, int const num_heads_q,
    int const cache_block_size, int const kv_block_stride,
    cudaStream_t stream) {
  constexpr int kBlockSize = 256;
  constexpr int kWarpsPerBlock = kBlockSize / 32;
  int64_t const total_warps =
      static_cast<int64_t>(num_tokens_full) * (kNumHeadsQPadded + 1);
  int const grid =
      static_cast<int>((total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock);

  // PDL: enable programmatic stream serialization whenever the hardware
  // supports it (SM90+).  On pre-Hopper GPUs the attribute is unavailable,
  // so leave numAttrs = 0 and launch as a regular kernel.
#ifndef USE_ROCM
  static int const sm_version = getSMVersion();
  // Host-side guard: the device kernel body is compiled as a no-op for
  // bf16 on pre-Ampere (sm_70/sm_75) because _typeConvert<BFloat16> is
  // unavailable there.  Refuse the launch loudly instead of silently
  // skipping the work.
  STD_TORCH_CHECK(
      sm_version >= 80,
      "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert requires sm_80+ "
      "(Ampere or newer); got sm_",
      sm_version);
  cudaLaunchConfig_t config;
  config.gridDim = dim3(grid);
  config.blockDim = dim3(kBlockSize);
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attrs;
  config.numAttrs = (sm_version >= 90) ? 1 : 0;

  if (num_tokens_full < NUM_TOKEN_CUTOFF) {
    cudaLaunchKernelEx(
        &config,
        fusedDeepseekV4QNormRopeKVRopeQuantInsertKernel<scalar_t_in,
                                                        kNumHeadsQPadded>,
        q_in, q_out, kv_in, k_cache, slot_mapping, position_ids, cos_sin_cache,
        eps, num_tokens_full, num_tokens_insert, num_heads_q, cache_block_size,
        kv_block_stride);
  } else {
    config.gridDim = dim3(num_tokens_full);
    cudaLaunchKernelEx(
        &config,
        fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelReducedGrid<
            scalar_t_in, kNumHeadsQPadded>,
        q_in, q_out, kv_in, k_cache, slot_mapping, position_ids, cos_sin_cache,
        eps, num_tokens_full, num_tokens_insert, num_heads_q, cache_block_size,
        kv_block_stride);
  }
#else
  // ROCm: use standard kernel launch syntax (no PDL/stream serialization)
  // clang-format off
  fusedDeepseekV4QNormRopeKVRopeQuantInsertKernel<scalar_t_in, kNumHeadsQPadded>
      <<<grid, kBlockSize, 0, stream>>>(
          q_in, q_out, kv_in, k_cache, slot_mapping, position_ids,
          cos_sin_cache, eps, num_tokens_full, num_tokens_insert, num_heads_q,
          cache_block_size, kv_block_stride);
#endif
}

// Runtime dispatch into one of the precompiled `kNumHeadsQPadded`
// instantiations.  Supported padded head counts: 8, 16, 32, 64, 128.
template <typename scalar_t_in>
void launchFusedDeepseekV4QNormRopeKVRopeQuantInsert(
    scalar_t_in const* q_in, scalar_t_in* q_out, scalar_t_in const* kv_in,
    uint8_t* k_cache, int64_t const* slot_mapping,
    int64_t const* position_ids, float const* cos_sin_cache, float const eps,
    int const num_tokens_full, int const num_tokens_insert,
    int const num_heads_q, int const num_heads_q_padded,
    int const cache_block_size, int const kv_block_stride,
    cudaStream_t stream) {
#define DISPATCH(N)                                                         \
  case N:                                                                   \
    launchFusedDeepseekV4Templated<scalar_t_in, N>(                         \
        q_in, q_out, kv_in, k_cache, slot_mapping, position_ids,            \
        cos_sin_cache, eps, num_tokens_full, num_tokens_insert, num_heads_q, \
        cache_block_size, kv_block_stride, stream);                         \
    return;

  switch (num_heads_q_padded) {
    DISPATCH(8)
    DISPATCH(16)
    DISPATCH(32)
    DISPATCH(64)
    DISPATCH(128)
    default:
      STD_TORCH_CHECK(false,
                  "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert: "
                  "unsupported num_heads_q_padded=",
                  num_heads_q_padded,
                  " (compiled instantiations: 8, 16, 32, 64, 128).");
  }
#undef DISPATCH
}

// ────────────────────────────────────────────────────────────────────────────
// FlashInfer full-cache kernel
// ────────────────────────────────────────────────────────────────────────────
//
// Sibling to the FlashMLA kernel above, used by the FlashInfer V4 sparse-MLA
// backend.  Differences from the legacy path:
//   * No Q head padding — output Q layout matches the input num_heads_q.
//   * KV is written as a *contiguous* 512-wide row per token (token-strided),
//     not the legacy UE8M0 paged layout with a separate scale tail.
//   * Q/KV are stored either as bf16 or as per-tensor E4M3 FP8 (one global
//     scale), selected by the STORE_Q_FP8 / STORE_KV_FP8 template flags.
//
// Grid: 1D, gridDim.x = ceil(num_tokens_full * (num_heads_q + 1) / warps).
// Each warp handles one (token, slot): slot < num_heads_q → Q, slot ==
// num_heads_q → KV.
template <typename scalar_t_in, bool STORE_Q_FP8, bool STORE_KV_FP8>
__global__ void fusedDeepseekV4FullCacheKernel(
    scalar_t_in* __restrict__ q_inout,          // [N, H, 512], in place (bf16)
    uint8_t* __restrict__ q_fp8_out,            // [N, H, 512] fp8, optional
    int64_t const q_fp8_stride0,                // elements (fp8 == bytes)
    int64_t const q_fp8_stride1,                // elements (fp8 == bytes)
    scalar_t_in const* __restrict__ kv_in,      // [N, 512] bf16
    uint8_t* __restrict__ k_cache,              // contiguous bf16 or fp8 cache
    int64_t const* __restrict__ slot_mapping,   // [num_tokens_insert] i64
    int64_t const* __restrict__ position_ids,   // [N] i64
    float const* __restrict__ cos_sin_cache,    // [max_pos, 64] fp32
    float const* __restrict__ fp8_scale_ptr,    // scalar, KV fp8 only
    float const* __restrict__ q_fp8_scale_inv,  // scalar, Q fp8 only
    float const eps,
    int const num_tokens_full,      // = q.size(0) = kv.size(0)
    int const num_tokens_insert,    // = slot_mapping.size(0)
    int const num_heads_q,          // H (no padding)
    int const cache_block_size,     // tokens per cache block
    int64_t const kv_block_stride,  // bytes per cache block
    int64_t const kv_token_stride) {  // bytes per cache token
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
  if constexpr (std::is_same_v<scalar_t_in, c10::BFloat16>) {
    return;
  } else {
#endif
    using Converter = vllm::_typeConvert<scalar_t_in>;
    int const warpsPerBlock = blockDim.x / 32;
    int const warpId = threadIdx.x / 32;
    int const laneId = threadIdx.x % 32;
    int const globalWarpIdx = blockIdx.x * warpsPerBlock + warpId;

    int const slotsPerToken = num_heads_q + 1;
    int const tokenIdx = globalWarpIdx / slotsPerToken;
    int const slotIdx = globalWarpIdx % slotsPerToken;
    if (tokenIdx >= num_tokens_full) return;
    bool const isKV = (slotIdx == num_heads_q);
    // KV branch: skip DP-padded tokens (no slot reserved for them).
    if (isKV && tokenIdx >= num_tokens_insert) return;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaGridDependencySynchronize();
#endif

    int const dim_base = laneId * kElemsPerLane;  // in [0, 512) step 16
    scalar_t_in const* src_ptr;
    if (isKV) {
      src_ptr = kv_in + static_cast<int64_t>(tokenIdx) * kHeadDim + dim_base;
    } else {
      src_ptr = q_inout +
                (static_cast<int64_t>(tokenIdx) * num_heads_q + slotIdx) *
                    kHeadDim +
                dim_base;
    }
    uint4 const v0 = *reinterpret_cast<uint4 const*>(src_ptr);
    uint4 const v1 = *reinterpret_cast<uint4 const*>(src_ptr + 8);

    // ── Decode bf16 → 16 fp32 registers ───────────────────────────────────
    float elements[kElemsPerLane];
    {
      auto const* p0 =
          reinterpret_cast<typename Converter::packed_hip_type const*>(&v0);
      auto const* p1 =
          reinterpret_cast<typename Converter::packed_hip_type const*>(&v1);
#pragma unroll
      for (int i = 0; i < 4; i++) {
        float2 f2 = Converter::convert(p0[i]);
        elements[2 * i] = f2.x;
        elements[2 * i + 1] = f2.y;
      }
#pragma unroll
      for (int i = 0; i < 4; i++) {
        float2 f2 = Converter::convert(p1[i]);
        elements[8 + 2 * i] = f2.x;
        elements[8 + 2 * i + 1] = f2.y;
      }
    }

    // ── Q branch: RMSNorm (no weight) ─────────────────────────────────────
    if (!isKV) {
      float sumOfSquares = 0.0f;
#pragma unroll
      for (int i = 0; i < kElemsPerLane; i++) {
        sumOfSquares += elements[i] * elements[i];
      }
      sumOfSquares = warpSum<float>(sumOfSquares);
      float const rms_rcp =
          rsqrtf(sumOfSquares / static_cast<float>(kHeadDim) + eps);
#pragma unroll
      for (int i = 0; i < kElemsPerLane; i++) {
        elements[i] = elements[i] * rms_rcp;
      }
    }

    // ── GPT-J RoPE on dims [NOPE_DIM, HEAD_DIM) ───────────────────────────
    bool const is_rope_lane = dim_base >= kNopeDim;
    if (is_rope_lane) {
      int64_t const pos = position_ids[tokenIdx];
      constexpr int kHalfRope = kRopeDim / 2;
      float const* cos_ptr = cos_sin_cache + pos * kRopeDim;
      float const* sin_ptr = cos_ptr + kHalfRope;
      int const rope_local_base = dim_base - kNopeDim;
      int const half_base = rope_local_base >> 1;
      float4 const c0 = *reinterpret_cast<float4 const*>(cos_ptr + half_base);
      float4 const c1 = *reinterpret_cast<float4 const*>(cos_ptr + half_base + 4);
      float4 const s0 = *reinterpret_cast<float4 const*>(sin_ptr + half_base);
      float4 const s1 = *reinterpret_cast<float4 const*>(sin_ptr + half_base + 4);
      float const cos_arr[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
      float const sin_arr[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
#pragma unroll
      for (int p = 0; p < kElemsPerLane / 2; p++) {
        float const x_even = elements[2 * p];
        float const x_odd = elements[2 * p + 1];
        elements[2 * p] = x_even * cos_arr[p] - x_odd * sin_arr[p];
        elements[2 * p + 1] = x_even * sin_arr[p] + x_odd * cos_arr[p];
      }
    }

    // ── Store ─────────────────────────────────────────────────────────────
    if (!isKV) {
      if constexpr (STORE_Q_FP8) {
        float const scale_inv = VLLM_LDG(q_fp8_scale_inv);
        uint4 const out = packFp8E4M3x16(elements, scale_inv);
        uint8_t* dst = q_fp8_out +
                       static_cast<int64_t>(tokenIdx) * q_fp8_stride0 +
                       static_cast<int64_t>(slotIdx) * q_fp8_stride1 + dim_base;
        *reinterpret_cast<uint4*>(dst) = out;
      } else {
        uint4 out0, out1;
        auto* po0 = reinterpret_cast<typename Converter::packed_hip_type*>(&out0);
        auto* po1 = reinterpret_cast<typename Converter::packed_hip_type*>(&out1);
#pragma unroll
        for (int i = 0; i < 4; i++) {
          po0[i] = Converter::convert(
              make_float2(elements[2 * i], elements[2 * i + 1]));
        }
#pragma unroll
        for (int i = 0; i < 4; i++) {
          po1[i] = Converter::convert(
              make_float2(elements[8 + 2 * i], elements[8 + 2 * i + 1]));
        }
        scalar_t_in* dst =
            q_inout +
            (static_cast<int64_t>(tokenIdx) * num_heads_q + slotIdx) * kHeadDim +
            dim_base;
        *reinterpret_cast<uint4*>(dst) = out0;
        *reinterpret_cast<uint4*>(dst + 8) = out1;
      }
    } else {
      int64_t const slot_id = slot_mapping[tokenIdx];
      if (slot_id >= 0) {
        int64_t const block_idx = slot_id / cache_block_size;
        int64_t const pos_in_block = slot_id % cache_block_size;
        uint8_t* cache_row =
            k_cache + block_idx * kv_block_stride + pos_in_block * kv_token_stride;
        if constexpr (STORE_KV_FP8) {
          float const inv_scale = 1.0f / VLLM_LDG(fp8_scale_ptr);
          uint4 const out = packFp8E4M3x16(elements, inv_scale);
          *reinterpret_cast<uint4*>(cache_row + dim_base) = out;
        } else {
          uint4 out0, out1;
          auto* po0 =
              reinterpret_cast<typename Converter::packed_hip_type*>(&out0);
          auto* po1 =
              reinterpret_cast<typename Converter::packed_hip_type*>(&out1);
#pragma unroll
          for (int i = 0; i < 4; i++) {
            po0[i] = Converter::convert(
                make_float2(elements[2 * i], elements[2 * i + 1]));
          }
#pragma unroll
          for (int i = 0; i < 4; i++) {
            po1[i] = Converter::convert(
                make_float2(elements[8 + 2 * i], elements[8 + 2 * i + 1]));
          }
          scalar_t_in* dst = reinterpret_cast<scalar_t_in*>(cache_row) + dim_base;
          *reinterpret_cast<uint4*>(dst) = out0;
          *reinterpret_cast<uint4*>(dst + 8) = out1;
        }
      }
    }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaTriggerProgrammaticLaunchCompletion();
#endif
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
  }
#endif
}

// Configure + launch helper shared by the bf16 and fp8 full-cache launchers.
template <typename scalar_t_in, bool STORE_Q_FP8, bool STORE_KV_FP8>
static void launchFullCacheKernel(
    scalar_t_in* q_inout, uint8_t* q_fp8_out, int64_t q_fp8_stride0,
    int64_t q_fp8_stride1, scalar_t_in const* kv_in, uint8_t* k_cache,
    int64_t const* slot_mapping, int64_t const* position_ids,
    float const* cos_sin_cache, float const* fp8_scale,
    float const* q_fp8_scale_inv, float const eps, int const num_tokens_full,
    int const num_tokens_insert, int const num_heads_q,
    int const cache_block_size, int64_t const kv_block_stride,
    int64_t const kv_token_stride, char const* op_name, cudaStream_t stream) {
  constexpr int kBlockSize = 256;
  constexpr int kWarpsPerBlock = kBlockSize / 32;
  int64_t const total_warps =
      static_cast<int64_t>(num_tokens_full) * (num_heads_q + 1);
  int const grid =
      static_cast<int>((total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock);
  auto* kernel =
      fusedDeepseekV4FullCacheKernel<scalar_t_in, STORE_Q_FP8, STORE_KV_FP8>;
#ifndef USE_ROCM
  static int const sm_version = getSMVersion();
  STD_TORCH_CHECK(sm_version >= 80, op_name,
                  " requires sm_80+ (Ampere or newer); got sm_", sm_version);
  cudaLaunchConfig_t config;
  config.gridDim = dim3(grid);
  config.blockDim = dim3(kBlockSize);
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attrs;
  config.numAttrs = (sm_version >= 90) ? 1 : 0;
  cudaLaunchKernelEx(&config, kernel, q_inout, q_fp8_out, q_fp8_stride0,
                     q_fp8_stride1, kv_in, k_cache, slot_mapping, position_ids,
                     cos_sin_cache, fp8_scale, q_fp8_scale_inv, eps,
                     num_tokens_full, num_tokens_insert, num_heads_q,
                     cache_block_size, kv_block_stride, kv_token_stride);
#else
  kernel<<<grid, kBlockSize, 0, stream>>>(
      q_inout, q_fp8_out, q_fp8_stride0, q_fp8_stride1, kv_in, k_cache,
      slot_mapping, position_ids, cos_sin_cache, fp8_scale, q_fp8_scale_inv,
      eps, num_tokens_full, num_tokens_insert, num_heads_q, cache_block_size,
      kv_block_stride, kv_token_stride);
#endif
}

}  // namespace deepseek_v4_fused_ops
}  // namespace vllm

// ────────────────────────────────────────────────────────────────────────────
// Torch op wrapper
// ────────────────────────────────────────────────────────────────────────────
torch::stable::Tensor fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
    torch::stable::Tensor const& q_in,           // [N, num_heads_q, 512] bf16
    torch::stable::Tensor const& kv,             // [N, 512] bf16 (read-only)
    torch::stable::Tensor& k_cache,              // [num_blocks, block_bytes] uint8
    torch::stable::Tensor const& slot_mapping,   // [N] int64
    torch::stable::Tensor const& position_ids,   // [N] int64
    torch::stable::Tensor const& cos_sin_cache,  // [max_pos, rope_dim] bf16
    int64_t q_head_padded,                       // padded Q head count for output
    double eps, int64_t cache_block_size) {
  STD_TORCH_CHECK(q_in.device().is_cuda() && q_in.is_contiguous(),
                  "q_in must be contiguous CUDA");
  STD_TORCH_CHECK(kv.device().is_cuda() && kv.is_contiguous(),
                  "kv must be contiguous CUDA");
  STD_TORCH_CHECK(k_cache.device().is_cuda(), "k_cache must be CUDA");
  STD_TORCH_CHECK(slot_mapping.device().is_cuda() &&
                      slot_mapping.scalar_type() ==
                          torch::headeronly::ScalarType::Long,
                  "slot_mapping must be int64 CUDA");
  STD_TORCH_CHECK(position_ids.device().is_cuda() &&
                      position_ids.scalar_type() ==
                          torch::headeronly::ScalarType::Long,
                  "position_ids must be int64 CUDA");
  STD_TORCH_CHECK(cos_sin_cache.device().is_cuda(), "cos_sin_cache must be CUDA");
  STD_TORCH_CHECK(q_in.dim() == 3 && q_in.size(2) == 512,
                  "q_in shape [N, num_heads_q, 512]");
  STD_TORCH_CHECK(kv.dim() == 2 && kv.size(1) == 512, "kv shape [N, 512]");
  STD_TORCH_CHECK(q_in.scalar_type() == kv.scalar_type(),
                  "q_in and kv dtype must match");
  STD_TORCH_CHECK(q_head_padded >= q_in.size(1),
                  "q_head_padded must be >= q_in.size(1) (num_heads_q)");
  STD_TORCH_CHECK(k_cache.scalar_type() == torch::headeronly::ScalarType::Byte,
                  "k_cache must be uint8");
  STD_TORCH_CHECK(cos_sin_cache.dim() == 2 && cos_sin_cache.size(1) == 64,
                  "cos_sin_cache shape [max_pos, 64]");
  STD_TORCH_CHECK(cos_sin_cache.scalar_type() ==
                      torch::headeronly::ScalarType::Float,
                  "cos_sin_cache must be float32");

  // With DP padding, slot_mapping can be shorter than q/kv/positions.
  // Q-norm+RoPE runs on all q.size(0) rows (downstream attention uses them);
  // KV quant+insert runs only on the first slot_mapping.size(0) rows.
  int const num_tokens_full = static_cast<int>(q_in.size(0));
  int const num_tokens_insert = static_cast<int>(slot_mapping.size(0));
  STD_TORCH_CHECK(static_cast<int>(kv.size(0)) == num_tokens_full &&
                      static_cast<int>(position_ids.size(0)) == num_tokens_full,
                  "q/kv/position_ids row counts must match");
  STD_TORCH_CHECK(num_tokens_insert <= num_tokens_full,
                  "slot_mapping must not exceed q row count");
  int const num_heads_q = static_cast<int>(q_in.size(1));
  int const num_heads_q_padded = static_cast<int>(q_head_padded);
  int const cache_block_size_i = static_cast<int>(cache_block_size);
  int const kv_block_stride = static_cast<int>(k_cache.stride(0));

  const torch::stable::accelerator::DeviceGuard device_guard(
      q_in.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(q_in.get_device_index());

  // Allocate the padded q output.  The kernel writes every element (live
  // region gets RMSNorm+RoPE; pad region gets zeros), so `empty` is safe.
  auto q_out = torch::stable::new_empty(
      q_in, {q_in.size(0), q_head_padded, q_in.size(2)}, q_in.scalar_type());

  VLLM_STABLE_DISPATCH_HALF_TYPES(
      q_in.scalar_type(), "fused_deepseek_v4_qnorm_rope_kv_insert", [&] {
        using qkv_scalar_t = scalar_t;
        vllm::deepseek_v4_fused_ops::
            launchFusedDeepseekV4QNormRopeKVRopeQuantInsert<qkv_scalar_t>(
                reinterpret_cast<qkv_scalar_t const*>(q_in.const_data_ptr()),
                reinterpret_cast<qkv_scalar_t*>(q_out.mutable_data_ptr()),
                reinterpret_cast<qkv_scalar_t const*>(kv.const_data_ptr()),
                reinterpret_cast<uint8_t*>(k_cache.mutable_data_ptr()),
                slot_mapping.const_data_ptr<int64_t>(),
                position_ids.const_data_ptr<int64_t>(),
                cos_sin_cache.const_data_ptr<float>(), static_cast<float>(eps),
                num_tokens_full, num_tokens_insert, num_heads_q,
                num_heads_q_padded, cache_block_size_i, kv_block_stride,
                stream);
      });
  return q_out;
}

// ────────────────────────────────────────────────────────────────────────────
// FlashInfer full-cache torch ops
// ────────────────────────────────────────────────────────────────────────────
void fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert(
    torch::stable::Tensor& q,                    // [N, H, 512] bf16, in place
    torch::stable::Tensor const& kv,             // [N, 512] bf16, read-only
    torch::stable::Tensor& k_cache,              // [num_blocks, bs, 512] bf16
    torch::stable::Tensor const& slot_mapping,   // [num_tokens_insert] int64
    torch::stable::Tensor const& position_ids,   // [N] int64
    torch::stable::Tensor const& cos_sin_cache,  // [max_pos, 64] float32
    double eps, int64_t cache_block_size) {
  using torch::headeronly::ScalarType;
  STD_TORCH_CHECK(q.device().is_cuda() && q.is_contiguous(),
                  "q must be contiguous CUDA");
  STD_TORCH_CHECK(kv.device().is_cuda() && kv.is_contiguous(),
                  "kv must be contiguous CUDA");
  STD_TORCH_CHECK(k_cache.device().is_cuda(), "k_cache must be CUDA");
  STD_TORCH_CHECK(slot_mapping.device().is_cuda() &&
                      slot_mapping.scalar_type() == ScalarType::Long,
                  "slot_mapping must be int64 CUDA");
  STD_TORCH_CHECK(position_ids.device().is_cuda() &&
                      position_ids.scalar_type() == ScalarType::Long,
                  "position_ids must be int64 CUDA");
  STD_TORCH_CHECK(cos_sin_cache.device().is_cuda() &&
                      cos_sin_cache.scalar_type() == ScalarType::Float &&
                      cos_sin_cache.dim() == 2 && cos_sin_cache.size(1) == 64,
                  "cos_sin_cache shape [max_pos, 64] float32");
  STD_TORCH_CHECK(q.dim() == 3 && q.size(2) == 512, "q shape [N, H, 512]");
  STD_TORCH_CHECK(kv.dim() == 2 && kv.size(1) == 512, "kv shape [N, 512]");
  STD_TORCH_CHECK(q.scalar_type() == ScalarType::BFloat16 &&
                      kv.scalar_type() == ScalarType::BFloat16,
                  "q and kv must be bfloat16");
  STD_TORCH_CHECK(k_cache.dim() == 3 && k_cache.size(1) == cache_block_size &&
                      k_cache.size(2) == 512 && k_cache.stride(2) == 1,
                  "k_cache shape [num_blocks, cache_block_size, 512] contiguous");
  STD_TORCH_CHECK(k_cache.scalar_type() == ScalarType::BFloat16,
                  "k_cache must be bfloat16");

  int const num_tokens_full = static_cast<int>(q.size(0));
  int const num_tokens_insert = static_cast<int>(slot_mapping.size(0));
  STD_TORCH_CHECK(static_cast<int>(kv.size(0)) == num_tokens_full &&
                      static_cast<int>(position_ids.size(0)) == num_tokens_full,
                  "q/kv/position_ids row counts must match");
  STD_TORCH_CHECK(num_tokens_insert <= num_tokens_full,
                  "slot_mapping must not exceed q row count");
  int const num_heads_q = static_cast<int>(q.size(1));

  const torch::stable::accelerator::DeviceGuard device_guard(
      q.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(q.get_device_index());

  // bf16 cache: 2 bytes/element -> byte strides for the uint8-addressed kernel.
  int64_t const kv_block_stride = k_cache.stride(0) * 2;
  int64_t const kv_token_stride = k_cache.stride(1) * 2;

  VLLM_STABLE_DISPATCH_HALF_TYPES(
      q.scalar_type(),
      "fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert", [&] {
        vllm::deepseek_v4_fused_ops::launchFullCacheKernel<scalar_t, false,
                                                           false>(
            reinterpret_cast<scalar_t*>(q.mutable_data_ptr()), nullptr, 0, 0,
            reinterpret_cast<scalar_t const*>(kv.const_data_ptr()),
            reinterpret_cast<uint8_t*>(k_cache.mutable_data_ptr()),
            slot_mapping.const_data_ptr<int64_t>(),
            position_ids.const_data_ptr<int64_t>(),
            cos_sin_cache.const_data_ptr<float>(), nullptr, nullptr,
            static_cast<float>(eps), num_tokens_full, num_tokens_insert,
            num_heads_q, static_cast<int>(cache_block_size), kv_block_stride,
            kv_token_stride,
            "fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert",
            stream);
      });
}

void fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert(
    torch::stable::Tensor const& q,                // [N, H, 512] bf16, read-only
    torch::stable::Tensor const& kv,               // [N, 512] bf16, read-only
    torch::stable::Tensor& q_fp8,                  // [N, H, 512] fp8 e4m3
    torch::stable::Tensor& k_cache,                // [num_blocks, bs, 512] fp8
    torch::stable::Tensor const& slot_mapping,     // [num_tokens_insert] int64
    torch::stable::Tensor const& position_ids,     // [N] int64
    torch::stable::Tensor const& cos_sin_cache,    // [max_pos, 64] float32
    torch::stable::Tensor const& fp8_scale,        // scalar float32 (KV scale)
    torch::stable::Tensor const& q_fp8_scale_inv,  // scalar float32 (1 / Q scale)
    double eps, int64_t cache_block_size) {
  using torch::headeronly::ScalarType;
  STD_TORCH_CHECK(q.device().is_cuda() && q.is_contiguous(),
                  "q must be contiguous CUDA");
  STD_TORCH_CHECK(kv.device().is_cuda() && kv.is_contiguous(),
                  "kv must be contiguous CUDA");
  STD_TORCH_CHECK(q_fp8.device().is_cuda() && q_fp8.is_contiguous() &&
                      q_fp8.scalar_type() == ScalarType::Float8_e4m3fn &&
                      q_fp8.dim() == 3 && q_fp8.size(0) == q.size(0) &&
                      q_fp8.size(1) == q.size(1) && q_fp8.size(2) == q.size(2),
                  "q_fp8 must be a contiguous float8_e4m3fn tensor matching q");
  STD_TORCH_CHECK(k_cache.device().is_cuda(), "k_cache must be CUDA");
  STD_TORCH_CHECK(slot_mapping.device().is_cuda() &&
                      slot_mapping.scalar_type() == ScalarType::Long,
                  "slot_mapping must be int64 CUDA");
  STD_TORCH_CHECK(position_ids.device().is_cuda() &&
                      position_ids.scalar_type() == ScalarType::Long,
                  "position_ids must be int64 CUDA");
  STD_TORCH_CHECK(cos_sin_cache.device().is_cuda() &&
                      cos_sin_cache.scalar_type() == ScalarType::Float &&
                      cos_sin_cache.dim() == 2 && cos_sin_cache.size(1) == 64,
                  "cos_sin_cache shape [max_pos, 64] float32");
  STD_TORCH_CHECK(fp8_scale.device().is_cuda() &&
                      fp8_scale.scalar_type() == ScalarType::Float &&
                      fp8_scale.size(0) == 1,
                  "fp8_scale must be a scalar float32 CUDA tensor");
  STD_TORCH_CHECK(q_fp8_scale_inv.device().is_cuda() &&
                      q_fp8_scale_inv.scalar_type() == ScalarType::Float &&
                      q_fp8_scale_inv.size(0) == 1,
                  "q_fp8_scale_inv must be a scalar float32 CUDA tensor");
  STD_TORCH_CHECK(q.dim() == 3 && q.size(2) == 512, "q shape [N, H, 512]");
  STD_TORCH_CHECK(kv.dim() == 2 && kv.size(1) == 512, "kv shape [N, 512]");
  STD_TORCH_CHECK(q.scalar_type() == kv.scalar_type(),
                  "q and kv dtype must match");
  STD_TORCH_CHECK(k_cache.dim() == 3 && k_cache.size(1) == cache_block_size &&
                      k_cache.size(2) == 512 && k_cache.stride(2) == 1,
                  "k_cache shape [num_blocks, cache_block_size, 512] contiguous");
  STD_TORCH_CHECK(k_cache.scalar_type() == ScalarType::Float8_e4m3fn,
                  "k_cache must be float8_e4m3fn");

  int const num_tokens_full = static_cast<int>(q.size(0));
  int const num_tokens_insert = static_cast<int>(slot_mapping.size(0));
  STD_TORCH_CHECK(static_cast<int>(kv.size(0)) == num_tokens_full &&
                      static_cast<int>(position_ids.size(0)) == num_tokens_full,
                  "q/kv/position_ids row counts must match");
  STD_TORCH_CHECK(num_tokens_insert <= num_tokens_full,
                  "slot_mapping must not exceed q row count");
  int const num_heads_q = static_cast<int>(q.size(1));

  const torch::stable::accelerator::DeviceGuard device_guard(
      q.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(q.get_device_index());

  VLLM_STABLE_DISPATCH_HALF_TYPES(
      q.scalar_type(),
      "fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert", [&] {
        vllm::deepseek_v4_fused_ops::launchFullCacheKernel<scalar_t, true,
                                                           true>(
            // q is read-only in the fp8 path (the kernel writes q_fp8); the
            // launcher signature is non-const, so cast away const on the ptr.
            reinterpret_cast<scalar_t*>(
                const_cast<void*>(q.const_data_ptr())),
            reinterpret_cast<uint8_t*>(q_fp8.mutable_data_ptr()),
            q_fp8.stride(0), q_fp8.stride(1),
            reinterpret_cast<scalar_t const*>(kv.const_data_ptr()),
            reinterpret_cast<uint8_t*>(k_cache.mutable_data_ptr()),
            slot_mapping.const_data_ptr<int64_t>(),
            position_ids.const_data_ptr<int64_t>(),
            cos_sin_cache.const_data_ptr<float>(),
            fp8_scale.const_data_ptr<float>(),
            q_fp8_scale_inv.const_data_ptr<float>(), static_cast<float>(eps),
            num_tokens_full, num_tokens_insert, num_heads_q,
            static_cast<int>(cache_block_size),
            // fp8 cache: 1 byte/element -> stride already in bytes.
            k_cache.stride(0), k_cache.stride(1),
            "fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert",
            stream);
      });
}
