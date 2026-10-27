/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 *
 * Horizontally-fused MiniMax-M3 attention pre-processing kernel.
 *
 * Replaces the per-token Python sequence in
 * ``MiniMaxM3SparseAttention.forward`` / ``MiniMaxM3Attention.forward``:
 *
 *     q  = q_norm(q);  k = k_norm(k);  q, k = rotary_emb(pos, q, k)
 *     index_q = index_q_norm(index_q);  index_k = index_k_norm(index_k)
 *     index_q, index_k = rotary_emb(pos, index_q, index_k)
 *     _insert_kv(k, v, index_k)
 *
 * All branches share head_dim=128 and the *same* partial-NeoX RoPE table
 * (``rotary_dim`` rotated, the trailing dims pass through).  The four norms
 * are Gemma-style RMSNorm (``x * rsqrt(mean(x^2)+eps) * (1 + weight)``) with
 * independent weights.
 *
 * Everything lives in a single fused ``qkv`` tensor.  The sparse layer's
 * fused projection (MinimaxM3QKVParallelLinearWithIndexer) emits, per token::
 *
 *     [ q | k | v | index_q | index_k ]   (the "5 results")
 *
 * while the dense layer emits just ``[ q | k | v ]``.  The kernel reads the
 * index branch straight out of that packed row -- no separate index tensors.
 *
 * One kernel, one grid; each warp owns one (token, head-slot) pair.  Slot
 * enumeration per token:
 *     [0, nq)                         Q  heads   -> norm(q_w)  + RoPE, write
 * qkv [nq, nq+nkv)                    K  heads   -> norm(k_w)  + RoPE, write
 * qkv
 *                                                  (+ insert into key cache)
 *     [nq+nkv, nq+2*nkv)              V  heads   -> insert into value cache
 *     IQ heads (niq)                             -> norm(iq_w) + RoPE, write iq
 *     IK       (1)                               -> norm(ik_w) + RoPE
 *                                                  (+ insert into index cache)
 *
 * The IQ/IK warps address the index_q/index_k sub-blocks *inside* qkv at the
 * fixed physical offsets (nq+2*nkv)*128 and (nq+2*nkv+niq)*128.
 *
 * Dense vs sparse row layout and index-branch processing are separate template
 * choices. Skip-index-topk reuse layers still have sparse rows and insert main
 * K/V cache entries, but compile away index_q/index_k work and index-cache
 * writes.
 *
 * Q/K and (sparse) index_q/index_k are all rewritten in place inside the fused
 * ``qkv`` tensor.  Caches (bf16) are scatter-written by slot.
 */

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <type_traits>
#include <limits>

#include "torch_utils.h"

#include "core/cuda_compat.h"
#include "type_convert.cuh"
#include "core/vec_dtypes/dtype_fp8.cuh"
#include "dispatch_utils.h"

#ifdef USE_ROCM
  #include "quantization/w8a8/fp8/amd/quant_utils.cuh"
#else
  #include "quantization/w8a8/fp8/nvidia/quant_utils.cuh"
#endif

// Direct float -> E4M3 FP8 conversion for the indexer Q / index-K outputs.
#ifndef USE_ROCM
  #include <cuda_fp8.h>
#else
  #include <hip/hip_fp8.h>
#endif

#ifndef FINAL_MASK
  #ifdef USE_ROCM
    #define FINAL_MASK 0xffffffffffffffffULL
  #else
    #define FINAL_MASK 0xffffffffu
  #endif
#endif

#ifdef USE_ROCM
// ROCm-compatible direct float -> E4M3 FP8 conversion (mirrors the DeepSeek V4
// fused kernel).
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
namespace minimax_m3_fused_ops {

namespace {
inline int getSMVersion() {
  auto* props = get_device_prop();
  return props->major * 10 + props->minor;
}
}  // namespace

// ────────────────────────────────────────────────────────────────────────────
// Constants (hard-coded for MiniMax-M3-preview).
// ────────────────────────────────────────────────────────────────────────────
constexpr int kHeadDim = 128;
constexpr int kNumLanes = 32;
constexpr int kElemsPerLane = kHeadDim / kNumLanes;  // 4

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ float warpReduceSum(float val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  }
  return val;
}

// Gemma RMSNorm over the full head (no-op when ``weight == nullptr``), rounded
// back to scalar_t like the materialized unfused norm output, followed by
// partial NeoX RoPE on the leading ``rotary_dim`` dims. Each lane owns
// ``kElemsPerLane`` contiguous dims [laneId*4, laneId*4+4).
template <typename scalar_t>
__device__ __forceinline__ void normAndRope(
    float (&elems)[kElemsPerLane], int const laneId, float const eps,
    scalar_t const* __restrict__ weight,  // [kHeadDim] or nullptr (no norm)
    bool const do_rope, int const rotary_dim,
    scalar_t const* __restrict__ cos_ptr,  // cos_sin_cache + pos*rotary_dim
    bool const apply_norm) {
  // ── Gemma RMSNorm: x * rsqrt(mean(x^2)+eps) * (1 + w) ──────────────────
  if (apply_norm) {
    float sumsq = 0.0f;
#pragma unroll
    for (int i = 0; i < kElemsPerLane; i++) sumsq += elems[i] * elems[i];
    sumsq = warpReduceSum(sumsq);
    float const rms_rcp = rsqrtf(sumsq / static_cast<float>(kHeadDim) + eps);
#pragma unroll
    for (int i = 0; i < kElemsPerLane; i++) {
      int const dim = laneId * kElemsPerLane + i;
      float const w = 1.0f + static_cast<float>(weight[dim]);
      elems[i] = elems[i] * rms_rcp * w;
    }
  }

  // ── Partial NeoX RoPE on dims [0, rotary_dim) ──────────────────────────
  // half = rotary_dim/2.  Pair (i, i+half) for i in [0, half).  Lane L owns
  // dims [4L, 4L+4); since half is a multiple of 4, a lane lies wholly in the
  // first half (own=x[i]) or second half (own=x[i+half]); its partner lives
  // ``half/4`` lanes away (XOR with that distance).
  if (do_rope) {
    int const half = rotary_dim / 2;
    int const dim0 = laneId * kElemsPerLane;
    bool const in_rope = dim0 < rotary_dim;
    int const lane_xor = half / kElemsPerLane;  // partner-lane distance

    float partner[kElemsPerLane];
#pragma unroll
    for (int i = 0; i < kElemsPerLane; i++) {
      partner[i] = __shfl_xor_sync(FINAL_MASK, elems[i], lane_xor, 32);
    }
    if (in_rope) {
      bool const first_half = dim0 < half;
      int const i_base = first_half ? dim0 : (dim0 - half);  // cos/sin index
      scalar_t const* sin_ptr = cos_ptr + half;
#pragma unroll
      for (int i = 0; i < kElemsPerLane; i++) {
        float const c = static_cast<float>(cos_ptr[i_base + i]);
        float const s = static_cast<float>(sin_ptr[i_base + i]);
        if (first_half) {
          elems[i] = elems[i] * c - partner[i] * s;
        } else {
          elems[i] = elems[i] * c + partner[i] * s;
        }
      }
    }
  }
}

// Load 4 contiguous bf16 -> 4 fp32 registers.
template <typename scalar_t>
__device__ __forceinline__ void loadElems(scalar_t const* __restrict__ src,
                                          float (&elems)[kElemsPerLane]) {
  using Converter = vllm::_typeConvert<scalar_t>;
  uint2 v = *reinterpret_cast<uint2 const*>(src);
  auto const* p =
      reinterpret_cast<typename Converter::packed_hip_type const*>(&v);
#pragma unroll
  for (int i = 0; i < kElemsPerLane / 2; i++) {
    float2 f2 = Converter::convert(p[i]);
    elems[2 * i] = f2.x;
    elems[2 * i + 1] = f2.y;
  }
}

// Store 4 fp32 registers -> 4 contiguous bf16.
template <typename scalar_t>
__device__ __forceinline__ void storeElems(
    scalar_t* __restrict__ dst, float const (&elems)[kElemsPerLane]) {
  using Converter = vllm::_typeConvert<scalar_t>;
  uint2 v;
  auto* p = reinterpret_cast<typename Converter::packed_hip_type*>(&v);
#pragma unroll
  for (int i = 0; i < kElemsPerLane / 2; i++) {
    p[i] = Converter::convert(make_float2(elems[2 * i], elems[2 * i + 1]));
  }
  *reinterpret_cast<uint2*>(dst) = v;
}

// Main K/V cache store. kAuto = unquantized (cache_t == scalar_t); fp8 cache
// dtypes use the scaled-convert path with identity scale.
template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__device__ __forceinline__ void storeCacheElems(
    cache_t* __restrict__ dst, float const (&elems)[kElemsPerLane]) {
  if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
    // kAuto means unquantized KV cache here: cache_t == scalar_t, so store the
    // model dtype directly. FP8 cache dtypes use the conversion path below.
    storeElems<scalar_t>(reinterpret_cast<scalar_t*>(dst), elems);
  } else {
#ifdef USE_ROCM
    // Match ROCm's model-dtype materialization before FP8 cache conversion.
    using Converter = vllm::_typeConvert<scalar_t>;
    using rounded_t = typename Converter::hip_type;
    rounded_t rounded[kElemsPerLane];
  #pragma unroll
    for (int i = 0; i < kElemsPerLane; i++) {
      rounded[i] = Converter::convert(elems[i]);
    }
  #pragma unroll
    for (int i = 0; i < kElemsPerLane; i++) {
      dst[i] = fp8::scaled_convert<cache_t, rounded_t, kv_dt>(rounded[i], 1.0f);
    }
#else
  #pragma unroll
    for (int i = 0; i < kElemsPerLane; i++) {
      dst[i] = fp8::scaled_convert<cache_t, float, kv_dt>(elems[i], 1.0f);
    }
#endif
  }
}

// Store 4 fp32 registers -> 4 contiguous E4M3 FP8 bytes (direct cast,
// saturating to ±448). Used for the fp8 indexer-Q / index-K outputs; no scale
// (RMSNorm outputs are O(1) and the score path only needs relative block
// ordering).
__device__ __forceinline__ void storeElemsFp8(
    uint8_t* __restrict__ dst, float const (&elems)[kElemsPerLane]) {
  constexpr float kFp8Max = 448.0f;
#ifndef USE_ROCM
  __nv_fp8x2_storage_t out2[kElemsPerLane / 2];
  #pragma unroll
  for (int i = 0; i < kElemsPerLane / 2; i++) {
    float2 vv = make_float2(elems[2 * i], elems[2 * i + 1]);
    vv.x = fminf(fmaxf(vv.x, -kFp8Max), kFp8Max);
    vv.y = fminf(fmaxf(vv.y, -kFp8Max), kFp8Max);
    out2[i] = __nv_cvt_float2_to_fp8x2(vv, __NV_SATFINITE, __NV_E4M3);
  }
  *reinterpret_cast<uint32_t*>(dst) = *reinterpret_cast<uint32_t const*>(out2);
#else
  #pragma unroll
  for (int i = 0; i < kElemsPerLane; i++) {
    float vv = fminf(fmaxf(elems[i], -kFp8Max), kFp8Max);
    dst[i] = rocm_cvt_float_to_fp8_e4m3(vv);
  }
#endif
}

// Match scaled_fp8_quant(q_out): materialize q in scalar_t before applying the
// inverse dequantization scale and converting it to E4M3.
#ifndef USE_ROCM
// storeScaledQElemsFp8 with inv_scale = 1, from the packed model-dtype values:
// E4M3 of a scalar_t equals E4M3 of its float, the max takes NaN and values
// below -448 to -448 as storeElemsFp8's fmaxf does, and the cvt saturates the
// top at 448.
template <typename scalar_t>
__device__ __forceinline__ void storeUnitScaleQElemsFp8(
    uint8_t* __restrict__ dst, float const (&elems)[kElemsPerLane]) {
  using Converter = vllm::_typeConvert<scalar_t>;
  __nv_fp8x2_storage_t out2[kElemsPerLane / 2];
  #pragma unroll
  for (int i = 0; i < kElemsPerLane / 2; i++) {
    auto const v =
        Converter::convert(make_float2(elems[2 * i], elems[2 * i + 1]));
    if constexpr (std::is_same_v<scalar_t, c10::BFloat16>) {
      out2[i] = __nv_cvt_bfloat16raw2_to_fp8x2(
          __hmax2(v, __float2bfloat162_rn(-448.0f)), __NV_SATFINITE, __NV_E4M3);
    } else {
      out2[i] = __nv_cvt_halfraw2_to_fp8x2(
          __hmax2(v, __float2half2_rn(-448.0f)), __NV_SATFINITE, __NV_E4M3);
    }
  }
  *reinterpret_cast<uint32_t*>(dst) = *reinterpret_cast<uint32_t const*>(out2);
}
#endif

template <typename scalar_t>
__device__ __forceinline__ void storeScaledQElemsFp8(
    uint8_t* __restrict__ dst, float const (&elems)[kElemsPerLane],
    float const inv_scale) {
  using Converter = vllm::_typeConvert<scalar_t>;
#ifndef USE_ROCM
  if (inv_scale == 1.0f) {
    storeUnitScaleQElemsFp8<scalar_t>(dst, elems);
    return;
  }
#endif
  float scaled[kElemsPerLane];
#pragma unroll
  for (int i = 0; i < kElemsPerLane; i++) {
    auto const rounded = Converter::convert(elems[i]);
    scaled[i] = static_cast<float>(rounded) * inv_scale;
  }
  storeElemsFp8(dst, scaled);
}

// index_q: E4M3(scalar_t(v)) with unit scale, i.e. q is rounded to the model
// dtype before the fp8 cast. On CUDA each element pair is rounded to scalar_t
// and then converted to E4M3, with one packed cvt each.
template <typename scalar_t>
__device__ __forceinline__ void storeIndexQElemsFp8(
    uint8_t* __restrict__ dst, float const (&elems)[kElemsPerLane]) {
#ifndef USE_ROCM
  using Converter = vllm::_typeConvert<scalar_t>;
  __nv_fp8x2_storage_t out2[kElemsPerLane / 2];
  #pragma unroll
  for (int i = 0; i < kElemsPerLane / 2; i++) {
    auto const v =
        Converter::convert(make_float2(elems[2 * i], elems[2 * i + 1]));
    if constexpr (std::is_same_v<scalar_t, c10::BFloat16>) {
      out2[i] = __nv_cvt_bfloat16raw2_to_fp8x2(v, __NV_SATFINITE, __NV_E4M3);
    } else {
      out2[i] = __nv_cvt_halfraw2_to_fp8x2(v, __NV_SATFINITE, __NV_E4M3);
    }
  }
  *reinterpret_cast<uint32_t*>(dst) = *reinterpret_cast<uint32_t const*>(out2);
#else
  storeScaledQElemsFp8<scalar_t>(dst, elems, 1.0f);
#endif
}

// ────────────────────────────────────────────────────────────────────────────
// NVFP4 main K/V cache store
// ────────────────────────────────────────────────────────────────────────────
// The block encoding of reshape_and_cache_nvfp4_kernel
// (nvfp4_kv_cache_kernels.cu) in the MSA readers' head-slot page layout. The
// quantizer differs from that kernel's reciprocal-multiply math: per 16-value
// block, scale = min(448, max(E4M3(max(amax, 1e-12) / 6), 1/512)) and
// code = sign | RNE-E2M1(min(|x / scale|, 6)), with zero stored as +0, all on
// x / the global scale.
// An HND page is 2*nkv slots of [bs, 72] bytes; slot 2*head + side (K = 0,
// V = 1) holds that head's packed E2M1 data [bs, 64] followed by its E4M3 block
// scales [bs, 8]. Every head is one contiguous 2*bs*72-byte run, so a page
// split across TP ranks is a contiguous byte slice per rank. K scales are
// linear, V scales use the SM100 trtllm-gen 4x4 token-quad swizzle. One
// 16-element scale group is four lanes. SM100-family builds of this file
// (CMake gives it family-specific SM10x/11x targets) encode E2M1 with
// cvt.rn.satfinite.e2m1x2; other targets use a bit-exact software encoder.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && \
    defined(__CUDA_ARCH_FAMILY_SPECIFIC__)
  #define MINIMAX_M3_NVFP4_NATIVE_CVT 1
#else
  #define MINIMAX_M3_NVFP4_NATIVE_CVT 0
#endif

#ifndef USE_ROCM
constexpr int kNvfp4DataBytes = kHeadDim / 2;            // 64
constexpr int kNvfp4ScaleBytes = kHeadDim / 16;          // 8
constexpr int kNvfp4LanesPerGroup = 16 / kElemsPerLane;  // 4

// Round-to-nearest-even, saturating E2M1 code of ``x`` (matches
// cvt.rn.satfinite.e2m1x2.f32). Below 1 the grid is the 0.5-step subnormals;
// from 1 up it is fp32 rounded to one mantissa bit, whose (exponent, mantissa)
// bits are the code offset by 252 once the exponent is rebiased.
__device__ __forceinline__ uint32_t e2m1Code(float x) {
  float const ax = fabsf(x);
  uint32_t mag;
  if (ax < 1.0f) {
    mag = static_cast<uint32_t>(__float2int_rn(2.0f * ax));
  } else {
    uint32_t const bits = __float_as_uint(ax);
    uint32_t const rounded = bits + 0x1FFFFFu + ((bits >> 22) & 1u);
    mag = min((rounded >> 22) - 252u, 7u);
  }
  return mag | ((__float_as_uint(x) >> 28) & 0x8u);
}

__device__ __forceinline__ float rcpApproxFtz(float a) {
  float b;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(b) : "f"(a));
  return b;
}

// a / b rounded to nearest even from r = RN(1 / b): Markstein's correction of
// a * r, which is __fdiv_rn's fast path without its range-check branch. Exact
// unless a * r overflows or the residual a - q * b underflows (a within a few
// binades of the subnormals).
__device__ __forceinline__ float divRnByRcp(float a, float b, float r) {
  float const q = __fmul_rn(a, r);
  return __fmaf_rn(__fmaf_rn(-q, b, a), r, q);
}

// Block scale E4M3(max(amax, 1e-12) / 6) clamped to [1/512, 448], with
// sf_scale applied to it as to the values, and its reciprocal RN(1 / sf_value).
__device__ __forceinline__ void nvfp4GroupScale(float sf_scale, float group_max,
                                                uint8_t& sf_byte,
                                                float& sf_value,
                                                float& sf_rcp) {
  constexpr float kRcp6 = 0x1.555556p-3f;  // RN(1 / 6)
  float const amax = fmaxf(group_max, 1e-12f);
  // amax / 6 (RNE). amax * kRcp6 bounds it from above, and fminf picks it for
  // amax = inf, where the correction is NaN.
  float const amax_6 =
      fminf(divRnByRcp(amax, 6.0f, kRcp6), __fmul_rn(amax, kRcp6));
  // Flooring before the rounding gives the 1/512 minimum; satfinite caps it
  // at 448.
  __nv_fp8_e4m3 const sf(fmaxf(sf_scale * amax_6, 0x1p-9f));
  sf_byte = sf.__x;
  sf_value = float(sf);
  // rcp.approx is correctly rounded on every positive E4M3 value.
  sf_rcp = rcpApproxFtz(sf_value);
}

// nvfp4GroupScale for sf_scale = 1 and a bf16/fp16 group_max. Uncorrected,
// amax * RN(1 / 6) still rounds to the E4M3 byte of amax / 6 (checked for every
// such amax: amax / 6 is too far from an E4M3 rounding boundary to cross it,
// and exact ties round back), and the 1e-12 floor lies below the 1/512 one.
__device__ __forceinline__ void nvfp4UnitGroupScale(float group_max,
                                                    uint8_t& sf_byte,
                                                    float& sf_rcp) {
  constexpr float kRcp6 = 0x1.555556p-3f;  // RN(1 / 6)
  __nv_fp8_e4m3 const sf(fmaxf(__fmul_rn(group_max, kRcp6), 0x1p-9f));
  sf_byte = sf.__x;
  sf_rcp = rcpApproxFtz(float(sf));
}

// (x * sf_scale) / sf_value with the E2M1 code of the IEEE quotient. The
// quotient itself is the IEEE one except below 1e-35 (code 0 either way) and
// for magnitudes over 4096, which are clamped (NaN kept) so that the division
// stays finite: 4096 / 448 > 5, so their code stays the saturated 6.
__device__ __forceinline__ float nvfp4Quotient(float x, float sf_scale,
                                               float sf_value, float sf_rcp) {
  constexpr float kMaxMag = 4096.0f;
  float n = __fmul_rn(x, sf_scale);
  n = fabsf(n) > kMaxMag ? copysignf(kMaxMag, n) : n;
  return divRnByRcp(n, sf_value, sf_rcp);
}

// Zero is stored as +0: clear the sign of every nibble whose magnitude is 0.
// Adding 7 to a 3-bit magnitude carries into the sign bit exactly when it is
// nonzero.
__device__ __forceinline__ uint16_t clearNegZeroNibbles(uint16_t packed) {
  return packed & (((packed & 0x7777u) + 0x7777u) | 0x7777u);
}

// The E2M1 codes (low nibble first) of four values x, already rounded to the
// model dtype, of a 16-value group with absolute maximum group_max, and the
// group's E4M3 scale in ``sf_byte``, where ``scale`` is the dequantization
// (global) scale.
template <typename scalar_t>
__device__ __forceinline__ uint16_t
nvfp4EncodeElems(float const (&x)[kElemsPerLane], float const group_max,
                 float const scale, uint8_t& sf_byte) {
  float sf_rcp;
  float q[kElemsPerLane];  // (x / scale) / sf_value, E2M1-encoded below
  if (scale == 1.0f) {
    static_assert(sizeof(scalar_t) == 2, "the unit-scale path needs bf16/fp16");
    nvfp4UnitGroupScale(group_max, sf_byte, sf_rcp);
    // x / sf_value truncated from x * RN(1 / sf_value). RN(1 / s) >= 1 / s for
    // every E4M3 s, so a bf16/fp16 x on an E2M1 midpoint lands exactly on it
    // (and the cvt rounds it to even), while any other bf16/fp16 x is too far
    // from a midpoint to reach one. No clamp either: a finite product never
    // truncates to inf, and inf saturates in the cvt.
  #if MINIMAX_M3_NVFP4_NATIVE_CVT
    float2 const r2 = make_float2(sf_rcp, sf_rcp);
    #pragma unroll
    for (int i = 0; i < kElemsPerLane; i += 2) {
      float2 const p = __fmul2_rz(make_float2(x[i], x[i + 1]), r2);
      q[i] = p.x;
      q[i + 1] = p.y;
    }
  #else
    #pragma unroll
    for (int i = 0; i < kElemsPerLane; i++) q[i] = __fmul_rz(x[i], sf_rcp);
  #endif
  } else {
    float const sf_scale = __frcp_rn(scale);  // exactly 1.0f / scale
    float sf_value;
    nvfp4GroupScale(sf_scale, group_max, sf_byte, sf_value, sf_rcp);
    // A correctly rounded division, not a reciprocal multiply, so that exact
    // E2M1 midpoints round to even.
  #if MINIMAX_M3_NVFP4_NATIVE_CVT
    // nvfp4Quotient on element pairs, with SM100's packed FP32 instructions.
    float2 const s2 = make_float2(sf_scale, sf_scale);
    float2 const neg_v2 = make_float2(-sf_value, -sf_value);
    float2 const r2 = make_float2(sf_rcp, sf_rcp);
    #pragma unroll
    for (int i = 0; i < kElemsPerLane; i += 2) {
      float2 n = __fmul2_rn(make_float2(x[i], x[i + 1]), s2);
      // |n| <= 4096 as in nvfp4Quotient, NaN kept.
      asm("max.NaN.f32 %0, %0, 0fC5800000;\n\t"
          "min.NaN.f32 %0, %0, 0f45800000;\n\t"
          "max.NaN.f32 %1, %1, 0fC5800000;\n\t"
          "min.NaN.f32 %1, %1, 0f45800000;"
          : "+f"(n.x), "+f"(n.y));
      float2 const q0 = __fmul2_rn(n, r2);
      float2 const p = __ffma2_rn(__ffma2_rn(q0, neg_v2, n), r2, q0);
      q[i] = p.x;
      q[i + 1] = p.y;
    }
  #else
    #pragma unroll
    for (int i = 0; i < kElemsPerLane; i++) {
      q[i] = nvfp4Quotient(x[i], sf_scale, sf_value, sf_rcp);
    }
  #endif
  }
  uint16_t packed;
  #if MINIMAX_M3_NVFP4_NATIVE_CVT
  // Low nibble = even element; cvt packs its first source in the high nibble.
  asm("{\n"
      "  .reg .b8 lo, hi;\n"
      "  cvt.rn.satfinite.e2m1x2.f32 lo, %2, %1;\n"
      "  cvt.rn.satfinite.e2m1x2.f32 hi, %4, %3;\n"
      "  mov.b16 %0, {lo, hi};\n"
      "}"
      : "=h"(packed)
      : "f"(q[0]), "f"(q[1]), "f"(q[2]), "f"(q[3]));
  #else
  uint32_t codes = 0;
    #pragma unroll
  for (int i = 0; i < kElemsPerLane; i++) codes |= e2m1Code(q[i]) << (4 * i);
  packed = static_cast<uint16_t>(codes);
  #endif
  return clearNegZeroNibbles(packed);
}

// Offset, within a K (kv 0) or V (kv 1) head's scale block, of the E4M3 scale
// of a token's 16-value group: linear for K, the SM100 trtllm-gen 4x4
// token-quad swizzle for V.
__device__ __forceinline__ int nvfp4ScaleOffset(int const kv, int const token,
                                                int const group) {
  if (kv == 0) return token * kNvfp4ScaleBytes + group;
  constexpr int kGroupsPerQuad = kNvfp4ScaleBytes / 4;
  int const swizzled_t = (token / 4) * 4 + group / kGroupsPerQuad;
  int const swizzled_s = (group % kGroupsPerQuad) * 4 + token % 4;
  return swizzled_t * kNvfp4ScaleBytes + swizzled_s;
}

// The E2M1 codes of this lane's four values of a K or V head row, rounded to
// the model dtype, and in ``sf_byte`` its 16-value group's E4M3 scale, where
// ``scale`` is the dequantization (global) scale. Every lane of the warp must
// call this (the scale-group reductions shuffle).
template <typename scalar_t>
__device__ __forceinline__ uint16_t nvfp4QuantizeElems(
    float const (&elems)[kElemsPerLane], float const scale, uint8_t& sf_byte) {
  // Quantize the model-dtype value that the unfused path reads back from qkv.
  using Converter = vllm::_typeConvert<scalar_t>;
  float x[kElemsPerLane];
  float group_max = 0.0f;
  #pragma unroll
  for (int i = 0; i < kElemsPerLane; i++) {
    x[i] = static_cast<float>(Converter::convert(elems[i]));
    group_max = fmaxf(group_max, fabsf(x[i]));
  }
  group_max = fmaxf(group_max, __shfl_xor_sync(FINAL_MASK, group_max, 1, 32));
  group_max = fmaxf(group_max, __shfl_xor_sync(FINAL_MASK, group_max, 2, 32));

  return nvfp4EncodeElems<scalar_t>(x, group_max, scale, sf_byte);
}

// Store the codes and scale from nvfp4QuantizeElems of lane laneId's four
// values of a token's K (kv 0) or V (kv 1) head row.
__device__ __forceinline__ void nvfp4StoreElems(
    uint8_t* __restrict__ page, int const kv, int const head, int const token,
    int const block_size, int const laneId, uint16_t const packed,
    uint8_t const sf_byte) {
  uint8_t* const slot = page + static_cast<int64_t>(2 * head + kv) *
                                   block_size *
                                   (kNvfp4DataBytes + kNvfp4ScaleBytes);
  uint8_t* const data =
      slot + token * kNvfp4DataBytes + laneId * (kElemsPerLane / 2);
  *reinterpret_cast<uint16_t*>(data) = packed;

  if (laneId % kNvfp4LanesPerGroup == 0) {
    slot[block_size * kNvfp4DataBytes +
         nvfp4ScaleOffset(kv, token, laneId / kNvfp4LanesPerGroup)] = sf_byte;
  }
}

// nvfp4QuantizeElems for one lane holding a whole 16-value group x (already
// rounded to the model dtype): the group max is order-independent, so this is
// the same codes and scale without the shuffles. Low nibble first.
template <typename scalar_t>
__device__ __forceinline__ uint2 nvfp4QuantizeGroup(float const (&x)[16],
                                                    float const scale,
                                                    uint8_t& sf_byte) {
  float group_max = 0.0f;
  #pragma unroll
  for (int i = 0; i < 16; i++) group_max = fmaxf(group_max, fabsf(x[i]));
  uint32_t codes[4];
  #pragma unroll
  for (int k = 0; k < 4; k++) {
    float const xk[kElemsPerLane] = {x[4 * k], x[4 * k + 1], x[4 * k + 2],
                                     x[4 * k + 3]};
    codes[k] = nvfp4EncodeElems<scalar_t>(xk, group_max, scale, sf_byte);
  }
  return make_uint2(codes[0] | codes[1] << 16, codes[2] | codes[3] << 16);
}

// Store group ``group`` (its 8 code bytes and its scale) of a token's K
// (kv 0) or V (kv 1) head row into its NVFP4 page.
__device__ __forceinline__ void nvfp4StoreGroup(
    uint8_t* __restrict__ page, int const kv, int const head, int const token,
    int const block_size, int const group, uint2 const codes,
    uint8_t const sf_byte) {
  uint8_t* const slot = page + static_cast<int64_t>(2 * head + kv) *
                                   block_size *
                                   (kNvfp4DataBytes + kNvfp4ScaleBytes);
  *reinterpret_cast<uint2*>(slot + token * kNvfp4DataBytes + 8 * group) = codes;
  slot[block_size * kNvfp4DataBytes + nvfp4ScaleOffset(kv, token, group)] =
      sf_byte;
}

// Quantize one K or V head row of one token into its NVFP4 page, where
// ``scale`` is the dequantization (global) scale. Every lane of the warp must
// call this (the scale-group reductions shuffle).
template <typename scalar_t>
__device__ __forceinline__ void storeNvfp4CacheElems(
    uint8_t* __restrict__ page, int const kv, int const head, int const token,
    int const block_size, int const laneId, float const (&elems)[kElemsPerLane],
    float const scale) {
  uint8_t sf_byte;
  uint16_t const packed = nvfp4QuantizeElems<scalar_t>(elems, scale, sf_byte);
  nvfp4StoreElems(page, kv, head, token, block_size, laneId, packed, sf_byte);
}
#endif  // USE_ROCM

// ────────────────────────────────────────────────────────────────────────────
// Kernel
// ────────────────────────────────────────────────────────────────────────────
// Grid: 1D, ceil(num_tokens * slots_per_token / warps_per_block).
// Each warp = one (token, slot).
//
// `kHasIndex`, `kProcessIndex`, and `kInsertKV` are compile-time template
// bools, so branch decisions that distinguish the dense layer from the sparse
// layer (index slots, KV/index inserts, V slots) fold away per instantiation.
// Slots per token:
//     Q : nq                            (always — norm+RoPE)
//     K : nkv                           (always — norm+RoPE; +K-cache insert)
//     V : nkv  only if kInsertKV        (V-cache insert)
//     IQ: niq  only if kProcessIndex    (norm+RoPE)
//     IK: 1    only if kProcessIndex    (norm+RoPE; +index-cache insert)
// cache_t/kv_dt: main attention KV-cache dtype (auto/fp8). out_idx_t/kFp8Idx:
// indexer index-K cache + index-Q output dtype (scalar_t or e4m3 byte).
// kNvfp4Idx (implies kFp8Idx): the index-K cache is instead packed NVFP4 pages
// [nb, bs, 72] (bs*64 B E2M1, then bs*8 B linear E4M3 scales, no global scale),
// the layout of the MSA Q8KV4 indexer; index-Q stays e4m3.
// kHasIndex means the qkv row is laid out as sparse [q|k|v|index_q|index_k].
// kProcessIndex controls whether this launch actually norms/ropes the index
// branch and writes index_q/index_k outputs. Skip-index-topk reuse layers keep
// kHasIndex=true but set kProcessIndex=false.
template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt,
          bool kNvfp4, typename out_idx_t, bool kHasIndex, bool kInsertKV,
          bool kProcessIndex, bool kFp8Idx, bool kNvfp4Idx>
__global__ void fusedMiniMaxM3QNormRopeKVInsertKernel(
    scalar_t* __restrict__ qkv,  // [N, qkv_row] in/out (packs index if sparse)
    scalar_t* __restrict__ q_out,         // [N, nq*128] contiguous, or nullptr
    uint8_t* __restrict__ q_fp8_out,      // [N, nq*128] E4M3, or nullptr
    out_idx_t* __restrict__ index_q_out,  // [N, niq*128]; scalar_t or e4m3 byte
    scalar_t const* __restrict__ q_norm_w,
    scalar_t const* __restrict__ k_norm_w,
    scalar_t const* __restrict__ iq_norm_w,
    scalar_t const* __restrict__ ik_norm_w,
    scalar_t const* __restrict__ cos_sin_cache,  // [max_pos, rotary_dim]
    int64_t const* __restrict__ positions,       // [N] i64
    int64_t const* __restrict__ slot_mapping,    // main K/V slots or nullptr
    int64_t const* __restrict__ index_slot_mapping,  // index K slots/nullptr
    cache_t* __restrict__ kv_cache,       // [nb,nkv,bs,2*128] or nullptr
    out_idx_t* __restrict__ index_cache,  // [nb*bs, 128]; scalar_t or e4m3 byte
    float const eps, float const q_fp8_inv_scale, int const rotary_dim,
    int const num_tokens, int const nq, int const nkv, int const niq,
    int const block_size,
    // kv_cache strides (in elements) for logical shape [nb, nkv, bs, 2*128].
    // The content (last) dim is always innermost-contiguous (stride 1), so the
    // NHD/HND layout choice is captured by the head/token strides.
    int64_t const kv_s_block, int64_t const kv_s_head, int64_t const kv_s_token,
    int64_t const kv_s_dim,
    // NVFP4 index cache only: page stride in bytes.
    int64_t const idx_s_block,
    // NVFP4 main cache only: per-layer K/V dequantization (global) scales.
    float const* __restrict__ kv_k_scale,
    float const* __restrict__ kv_v_scale) {
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
  // _typeConvert<BFloat16> is unavailable on pre-Ampere; the M3 kernel only
  // runs with bf16/fp16 inputs in practice.  Discard the bf16 body there.
  if constexpr (std::is_same_v<scalar_t, c10::BFloat16>) {
    return;
  } else {
#endif
    int const warpsPerBlock = blockDim.x / 32;
    int const laneId = threadIdx.x % 32;
    int const globalWarpIdx = blockIdx.x * warpsPerBlock + (threadIdx.x / 32);

    static_assert(!kProcessIndex || kHasIndex,
                  "index processing requires sparse row layout");
    static_assert(!kNvfp4Idx || (kFp8Idx && kInsertKV),
                  "NVFP4 index cache requires fp8 index-Q and cache insert");

    // Slot layout (compile-time gated: V slots only when inserting, index
    // slots only on the sparse layer).
    int const v_slots = kInsertKV ? nkv : 0;
    int const idx_slots = kProcessIndex ? niq + 1 : 0;
    int const slots_per_token = nq + nkv + v_slots + idx_slots;

    // Unsigned: both are non-negative, and the signed division's sign handling
    // would cost every warp.
    unsigned const warp_u = globalWarpIdx, slots_u = slots_per_token;
    int const tokenIdx = warp_u / slots_u;
    int const slot = warp_u % slots_u;
    if (tokenIdx >= num_tokens) return;

    // Slot boundaries.
    int const k_begin = nq;
    int const v_begin = nq + nkv;             // valid only when kInsertKV
    int const iq_begin = nq + nkv + v_slots;  // index block start
    int const ik_slot = iq_begin + niq;       // valid only when kProcessIndex

    bool const isQ = slot < k_begin;
    bool const isK = slot >= k_begin && slot < v_begin;
    bool isV = false;
    if constexpr (kInsertKV) isV = slot >= v_begin && slot < v_begin + nkv;
    bool isIQ = false, isIK = false;
    if constexpr (kProcessIndex) {
      isIQ = slot >= iq_begin && slot < ik_slot;
      isIK = slot == ik_slot;
    }

    int const dim_base = laneId * kElemsPerLane;
    // Physical row width of qkv: the dense layer packs [q|k|v]; the sparse
    // layer additionally packs [index_q (niq heads) | index_k (1 head)].
    int const qkv_row = (nq + 2 * nkv + (kHasIndex ? (niq + 1) : 0)) * kHeadDim;

    // ── Resolve source pointer + per-branch parameters. ────────────────────
    scalar_t* row_ptr = nullptr;       // in-place output location
    scalar_t const* norm_w = nullptr;  // nullptr -> skip norm (V)
    bool do_rope = true;
    int head = 0;  // kv head index for inserts

    if (isQ) {
      row_ptr =
          qkv + static_cast<int64_t>(tokenIdx) * qkv_row + slot * kHeadDim;
      norm_w = q_norm_w;
    } else if (isK) {
      head = slot - k_begin;
      row_ptr =
          qkv + static_cast<int64_t>(tokenIdx) * qkv_row + slot * kHeadDim;
      norm_w = k_norm_w;
    } else if (isV) {
      // qkv V section starts at slot index (nq + nkv): slot * kHeadDim is the
      // correct in-tensor offset.
      head = slot - v_begin;
      row_ptr =
          qkv + static_cast<int64_t>(tokenIdx) * qkv_row + slot * kHeadDim;
      norm_w = nullptr;  // V: no norm, no rope
      do_rope = false;
    } else if (isIQ) {
      // index_q sub-block lives at physical offset (nq+2*nkv)*128 in qkv.
      int const ih = slot - iq_begin;
      row_ptr = qkv + static_cast<int64_t>(tokenIdx) * qkv_row +
                (nq + 2 * nkv + ih) * kHeadDim;
      norm_w = iq_norm_w;
    } else if (isIK) {
      // Single shared index key at (nq+2*nkv+niq)*128.
      row_ptr = qkv + static_cast<int64_t>(tokenIdx) * qkv_row +
                (nq + 2 * nkv + niq) * kHeadDim;
      norm_w = ik_norm_w;
    } else {
      return;
    }

    // Store destination.  Q and index_q are gathered into dedicated contiguous
    // output buffers (when provided) so the downstream SM100 sparse kernel's
    // flat TMA descriptor can address them as [tokens*heads, head_dim]; this
    // folds the de-interleaving into the store the kernel already does, instead
    // of a separate q.contiguous() copy.  Everything else stays in place.
    // Given q_fp8_out but no q_out, q is emitted only in fp8 (null store_ptr).
    scalar_t* store_ptr = row_ptr;
    if (isQ && q_out != nullptr) {
      store_ptr = q_out + static_cast<int64_t>(tokenIdx) * nq * kHeadDim +
                  slot * kHeadDim;
    } else if (isQ && q_fp8_out != nullptr) {
      store_ptr = nullptr;
    } else if (isIQ && index_q_out != nullptr) {
      // bf16 index_q_out: gather here. fp8: written by the explicit fp8 store.
      if constexpr (!kFp8Idx) {
        store_ptr = index_q_out +
                    static_cast<int64_t>(tokenIdx) * niq * kHeadDim +
                    (slot - iq_begin) * kHeadDim;
      }
    }

    // NVFP4: the per-layer global scale is not produced by the predecessor,
    // so load it before the PDL wait.
    float nvfp4_scale = 0.0f;
    if constexpr (kNvfp4) {
      if (isK || isV) nvfp4_scale = isK ? *kv_k_scale : *kv_v_scale;
    }

    // PDL: wait for the predecessor kernel (the qkv-projection GEMM that
    // produces ``qkv``) to finish before touching any global memory.  No-op
    // when PDL is not enabled on the launch.  The CUDA runtime wrapper emits
    // the griddepcontrol.wait PTX with the required memory clobber internally.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaGridDependencySynchronize();
#endif

    // ── Load -> norm+rope (fp32) -> store back in place. ───────────────────
    float elems[kElemsPerLane];
    loadElems<scalar_t>(row_ptr + dim_base, elems);

    if (!isV) {
      int64_t const pos = positions[tokenIdx];
      scalar_t const* cos_ptr = cos_sin_cache + pos * rotary_dim;
      normAndRope<scalar_t>(elems, laneId, eps, norm_w, do_rope, rotary_dim,
                            cos_ptr, /*apply_norm=*/norm_w != nullptr);
      if constexpr (kFp8Idx) {
        // index_q is E4M3(scalar_t(q)) bytes; Q/K (and in-place index_k) stay
        // scalar_t.
        if (isIQ && index_q_out != nullptr) {
          storeIndexQElemsFp8<scalar_t>(
              index_q_out + static_cast<int64_t>(tokenIdx) * niq * kHeadDim +
                  (slot - iq_begin) * kHeadDim + dim_base,
              elems);
        } else if (store_ptr != nullptr) {
          storeElems<scalar_t>(store_ptr + dim_base, elems);
        }
      } else if (store_ptr != nullptr) {
        storeElems<scalar_t>(store_ptr + dim_base, elems);
      }
      if (isQ && q_fp8_out != nullptr) {
        storeScaledQElemsFp8<scalar_t>(
            q_fp8_out + static_cast<int64_t>(tokenIdx) * nq * kHeadDim +
                slot * kHeadDim + dim_base,
            elems, q_fp8_inv_scale);
      }
    }

    // ── Cache inserts. ─────────────────────────────────────────────────────
    if constexpr (kInsertKV) {
      // Guard (not early-return) so every thread reaches the PDL trigger below.
      int64_t sm = -1;
      if (isK || isV) {
        sm = slot_mapping[tokenIdx];
      } else if constexpr (kProcessIndex) {
        if (isIK) sm = index_slot_mapping[tokenIdx];
      }
      if (sm >= 0) {  // skip padded / unscheduled tokens
        if (isIK) {
          if constexpr (kNvfp4Idx) {
#ifndef USE_ROCM
            // A single-head K side: data [bs, 64] then linear scales [bs, 8].
            storeNvfp4CacheElems<scalar_t>(
                reinterpret_cast<uint8_t*>(index_cache) +
                    (sm / block_size) * idx_s_block,
                /*kv=*/0, /*head=*/0, static_cast<int>(sm % block_size),
                block_size, laneId, elems, /*scale=*/1.0f);
#endif
          } else if constexpr (kFp8Idx) {
            storeElemsFp8(index_cache + sm * kHeadDim + dim_base, elems);
          } else {
            storeElems<scalar_t>(index_cache + sm * kHeadDim + dim_base, elems);
          }
        } else if constexpr (kNvfp4) {
#ifndef USE_ROCM
          // kv_cache is [num_blocks, 2*nkv, block_size, 72] bytes (HND), slot
          // 2*head + side; kv_s_block is its page stride.
          storeNvfp4CacheElems<scalar_t>(
              reinterpret_cast<uint8_t*>(kv_cache) +
                  (sm / block_size) * kv_s_block,
              isK ? 0 : 1, head, static_cast<int>(sm % block_size), block_size,
              laneId, elems, nvfp4_scale);
#endif
        } else if (isK || isV) {
          // kv_cache logical shape [num_blocks, nkv, block_size, 2*head_dim].
          // Paging is logical (block = sm/block_size, token = sm%block_size);
          // the physical NHD/HND layout is honoured via the passed strides.
          int64_t const b = sm / block_size;
          int64_t const t = sm % block_size;
          int const kv = isK ? 0 : 1;
          int64_t const off = b * kv_s_block + head * kv_s_head +
                              t * kv_s_token +
                              (kv * kHeadDim + dim_base) * kv_s_dim;
          if constexpr (!kHasIndex) {
            // Dense layers stand in for reshape_and_cache_flash, which
            // quantizes the k already materialized in the model dtype.
            using Converter = vllm::_typeConvert<scalar_t>;
#pragma unroll
            for (int i = 0; i < kElemsPerLane; i++) {
              elems[i] = Converter::convert(Converter::convert(elems[i]));
            }
          }
          storeCacheElems<scalar_t, cache_t, kv_dt>(kv_cache + off, elems);
        }
      }
    }

    // PDL: signal that this kernel is done so a dependent successor may launch
    // early.  No-op when PDL is not enabled on the launch.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaTriggerProgrammaticLaunchCompletion();
#endif
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
  }
#endif
}

#ifndef USE_ROCM
// ────────────────────────────────────────────────────────────────────────────
// Persistent row-group kernel
// ────────────────────────────────────────────────────────────────────────────
// Writes the same bytes as the kernel above, for every instantiation of it,
// when rotary_dim == 64. A token's work is a group of warps with fixed roles:
// the q heads kQHeads (4 or 8) per warp, then K, V and index_q four heads per
// warp, then index_k. A branch of fewer than four heads, and index_k, takes
// one warp per head in the layout of the kernel above (lane l holds dims
// 4l .. 4l + 3); the others run four-row steps, where lane (g, v) holds dims
// 32c + 4v .. 32c + 4v + 3 (c = 0..3) of row g, the dims lane 8c + v holds
// above, so the stores call the same helpers with that lane id. A CTA runs
// some or all warps of the warp groups of tokens x, x + gridDim.x, ..., and
// each warp copies the qkv rows and cos/sin row of its next tokens into a
// kStages-deep shared-memory ring with cp.async.
namespace rg {

constexpr int kRows = 4;
constexpr int kChunks = 4;
constexpr int kRotaryDim = 64;
constexpr int kStages = 3;  // ring depth: two tokens in flight per warp
// A ring row: 256 B padded to 320 B, so that rows 1 apart start 16 banks apart
// and a 4-row step's 8-byte loads are conflict-free.
constexpr int kRowPitch = kHeadDim * 2 + 64;
constexpr int kCosSinBytes = kRotaryDim * 2;
enum Role : int { kQ = 0, kK, kV, kIQ, kIK };

// SM100's packed FP32 instructions where they exist; the same RN results.
__device__ __forceinline__ float2 mul2(float2 const a, float2 const b) {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  return __fmul2_rn(a, b);
  #else
  return make_float2(__fmul_rn(a.x, b.x), __fmul_rn(a.y, b.y));
  #endif
}

__device__ __forceinline__ float2 fma2(float2 const a, float2 const b,
                                       float2 const c) {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  return __ffma2_rn(a, b, c);
  #else
  return make_float2(__fmaf_rn(a.x, b.x, c.x), __fmaf_rn(a.y, b.y, c.y));
  #endif
}

__device__ __forceinline__ float2 add2(float2 const a, float2 const b) {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  return __fadd2_rn(a, b);
  #else
  return make_float2(__fadd_rn(a.x, b.x), __fadd_rn(a.y, b.y));
  #endif
}

// A head row as one lane holds it, paired for packed FP32 math:
// x[k][i] = element i of chunks 2k and 2k + 1.
using Row = float2[2][4];

template <typename scalar_t>
__device__ __forceinline__ void loadRow(uint8_t const* row, int const v,
                                        Row& x) {
  float e[kChunks][kElemsPerLane];
  #pragma unroll
  for (int c = 0; c < kChunks; c++) {
    loadElems<scalar_t>(reinterpret_cast<scalar_t const*>(row + 64 * c + 8 * v),
                        e[c]);
  }
  #pragma unroll
  for (int k = 0; k < 2; k++) {
  #pragma unroll
    for (int i = 0; i < kElemsPerLane; i++) {
      x[k][i] = make_float2(e[2 * k][i], e[2 * k + 1][i]);
    }
  }
}

__device__ __forceinline__ void chunkElems(Row const& x, int const c,
                                           float (&e)[kElemsPerLane]) {
  #pragma unroll
  for (int i = 0; i < kElemsPerLane; i++) {
    e[i] = c % 2 ? x[c / 2][i].y : x[c / 2][i].x;
  }
}

// normAndRope (Gemma RMSNorm, then NeoX RoPE on dims [0, 64)) with its
// rounding as compiled: an FFMA sum of squares per lane of the one-warp-per-row
// layout (one chunk here), the xor-16/8/4/2/1 tree (16 pairs chunks c and
// c ^ 2, 8 pairs c and c ^ 1), FFMA(sum, 1/128, eps), rsqrtf, x * rms * w, and
// x * cos -/+ RN(partner * sin) as one FFMA.
__device__ __forceinline__ void normRope(Row& x, Row const& w1, float const eps,
                                         float const (&cos4)[4],
                                         float const (&sin4)[4]) {
  float2 sq[2];
  #pragma unroll
  for (int k = 0; k < 2; k++) {
    sq[k] = mul2(x[k][0], x[k][0]);  // = FFMA(x, x, 0)
  #pragma unroll
    for (int i = 1; i < 4; i++) sq[k] = fma2(x[k][i], x[k][i], sq[k]);
  }
  float2 const xor16 = add2(sq[0], sq[1]);
  float sum = __fadd_rn(xor16.x, xor16.y);
  sum = __fadd_rn(sum, __shfl_xor_sync(FINAL_MASK, sum, 4, 32));
  sum = __fadd_rn(sum, __shfl_xor_sync(FINAL_MASK, sum, 2, 32));
  sum = __fadd_rn(sum, __shfl_xor_sync(FINAL_MASK, sum, 1, 32));
  float const rms_rcp =
      rsqrtf(__fmaf_rn(sum, 1.0f / static_cast<float>(kHeadDim), eps));
  float2 const r2 = make_float2(rms_rcp, rms_rcp);
  #pragma unroll
  for (int k = 0; k < 2; k++) {
  #pragma unroll
    for (int i = 0; i < 4; i++) x[k][i] = mul2(mul2(x[k][i], r2), w1[k][i]);
  }
  // x[0][i] = (x1, x2): dims 4v + i and 32 + 4v + i.
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    float2 const x_sin = mul2(x[0][i], make_float2(sin4[i], sin4[i]));
    x[0][i] = make_float2(__fmaf_rn(x[0][i].x, cos4[i], -x_sin.y),
                          __fmaf_rn(x[0][i].y, cos4[i], x_sin.x));
  }
}

__device__ __forceinline__ void cpAsync16(uint32_t const dst, void const* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst),
               "l"(src)
               : "memory");
}

__device__ __forceinline__ void cpAsyncCommit() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int kPending>
__device__ __forceinline__ void cpAsyncWait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(kPending) : "memory");
}

// Warp roles of a token's warp group: the q heads kQHeads at a time, then K,
// V and index_q four heads at a time, then index_k.
// A branch of fewer than kRows heads (and index_k) takes one warp per head in
// the layout of the kernel above instead of a part-empty four-row step.
struct Roles {
  int nq, nkv, niq;
  bool kv_one, iq_one;  // one warp per head
  int q_warps, kv_warps, v_warps, iq_warps, ik_warps;
  int warps;  // one token's warp group

  __host__ __device__ Roles(int const q_heads, int const nq_, int const nkv_,
                            int const niq_, bool const insert_kv,
                            bool const process_index)
      : nq(nq_), nkv(nkv_), niq(niq_) {
    kv_one = nkv < kRows;
    iq_one = niq < kRows;
    q_warps = (nq + q_heads - 1) / q_heads;
    kv_warps = kv_one ? nkv : (nkv + kRows - 1) / kRows;
    v_warps = insert_kv ? kv_warps : 0;
    iq_warps = process_index ? (iq_one ? niq : (niq + kRows - 1) / kRows) : 0;
    ik_warps = process_index ? 1 : 0;
    warps = q_warps + kv_warps + v_warps + iq_warps + ik_warps;
  }

  // Shared-memory offset of warp w's ring: the q warps' rings, then the
  // others'.
  __host__ __device__ int ringOffset(int const w, int const q_ring,
                                     int const side_ring) const {
    return min(w, q_warps) * q_ring + max(w - q_warps, 0) * side_ring;
  }
};

// CTA size limit, for 80 registers per thread.
constexpr int kMaxWarps = 24;

template <int kQHeads>
constexpr int stageBytes() {
  return kQHeads * kRowPitch + kCosSinBytes;
}

// What a warp writes besides the caches, fixed per launch so that the stores
// of the other modes are not compiled into the loop: q (kQFp8Unit: only
// q_fp8_out, unit scale; kQFp8: only q_fp8_out; kQAny: decided per store),
// index_q (kIQOut: index_q_out; kIQInPlace: back into qkv; kNone), and the
// in-place k / index_k rewrite (kInPlace or kNone).
enum Out : int {
  kNone = 0,
  kInPlace,
  kQFp8Unit,
  kQFp8,
  kQAny,
  kIQOut,
  kIQInPlace
};

// One warp's loop over its CTA's tokens: rows [head0, head0 + nrows) of one
// branch, kSteps four-row steps per token.
template <int kRole, int kSteps, int kOut, typename scalar_t, typename cache_t,
          Fp8KVCacheDataType kv_dt, bool kNvfp4, typename out_idx_t,
          bool kHasIndex, bool kInsertKV, bool kFp8Idx, bool kNvfp4Idx,
          bool kOneRow = false>
__device__ __forceinline__ void runWarp(
    int const head0, int const nrows, int const qkv_head0, uint8_t* const ring,
    int const stage_bytes, scalar_t* __restrict__ qkv,
    scalar_t* __restrict__ q_out, uint8_t* __restrict__ q_fp8_out,
    out_idx_t* __restrict__ index_q_out, scalar_t const* __restrict__ norm_w,
    scalar_t const* __restrict__ cos_sin_cache,
    int64_t const* __restrict__ positions,
    int64_t const* __restrict__ slot_mapping, cache_t* __restrict__ kv_cache,
    out_idx_t* __restrict__ index_cache, float const eps,
    float const q_fp8_inv_scale, int const num_tokens, int const nq,
    int const niq, int const qkv_row, int const block_size,
    int64_t const kv_s_block, int64_t const kv_s_head, int64_t const kv_s_token,
    int64_t const kv_s_dim, int64_t const idx_s_block, float const kv_scale,
    bool const write_qkv) {
  constexpr bool kNorm = kRole != kV;
  constexpr bool kSlots =
      kInsertKV && (kRole == kK || kRole == kV || kRole == kIK);
  // kOneRow: a single row in the layout of the kernel above (lane l holds
  // dims 4l .. 4l + 3), for branches of one head.
  constexpr int kStepRows = kOneRow ? 1 : kRows;
  constexpr int kLaneChunks = kOneRow ? 1 : kChunks;
  int const lane = threadIdx.x % 32;
  int const g = kOneRow ? 0 : lane / 8;
  int const v = kOneRow ? lane : lane % 8;
  constexpr int kCopies = (kSteps * kStepRows * 16 + 31) / 32;
  uint32_t const ring_smem =
      static_cast<uint32_t>(__cvta_generic_to_shared(ring));

  // Not produced by the predecessor: loaded before the PDL wait, used after.
  uint2 w_raw[kChunks];
  if constexpr (kNorm && !kOneRow) {
  #pragma unroll
    for (int c = 0; c < kChunks; c++) {
      w_raw[c] = __ldg(reinterpret_cast<uint2 const*>(norm_w + 32 * c + 4 * v));
    }
  }

  // PDL: qkv comes from the predecessor (the qkv GEMM).
  #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaGridDependencySynchronize();
  #endif

  Row w1;
  if constexpr (kNorm && !kOneRow) {
    float e[kChunks][kElemsPerLane];
  #pragma unroll
    for (int c = 0; c < kChunks; c++) {
      loadElems<scalar_t>(reinterpret_cast<scalar_t const*>(&w_raw[c]), e[c]);
    }
  #pragma unroll
    for (int k = 0; k < 2; k++) {
  #pragma unroll
      for (int i = 0; i < kElemsPerLane; i++) {
        w1[k][i] = make_float2(1.0f + e[2 * k][i], 1.0f + e[2 * k + 1][i]);
      }
    }
  }

  int const stride = gridDim.x;
  int const t0 = blockIdx.x;
  int const n = (num_tokens - t0 + stride - 1) / stride;
  // Lane j holds the position (slot) of the token of iteration base + j.
  auto fetch = [&](int64_t const* src, int const base, int64_t const fill) {
    int64_t const t = t0 + static_cast<int64_t>(base + lane) * stride;
    return t < num_tokens ? static_cast<int64_t>(__ldg(
                                reinterpret_cast<long long const*>(src) + t))
                          : fill;
  };
  int pos_base = 0, slot_base = 0;
  int64_t pos_l = 0, slot_l = -1;
  if constexpr (kNorm) pos_l = fetch(positions, 0, 0);
  if constexpr (kSlots) slot_l = fetch(slot_mapping, 0, -1);

  scalar_t const* const src0 =
      qkv + static_cast<int64_t>(t0) * qkv_row + qkv_head0 * kHeadDim;
  int64_t const src_step = static_cast<int64_t>(stride) * qkv_row;
  auto issue_rows = [&](int const it) {
    if (it >= n) return;
    scalar_t const* const src = src0 + it * src_step;
    uint32_t const stage = ring_smem + (it % kStages) * stage_bytes;
  #pragma unroll
    for (int k = 0; k < kCopies; k++) {
      int const r = lane / 16 + 2 * k;
      if (r < nrows && (!kOneRow || lane < 16)) {
        cpAsync16(stage + r * kRowPitch + lane % 16 * 16,
                  src + r * kHeadDim + lane % 16 * 8);
      }
    }
  };
  // The cos/sin row of iteration it, then the commit that closes its group.
  auto issue_cos_sin = [&](int const it) {
    if constexpr (kNorm) {
      if (it < n) {
        if (it - pos_base == 32) {
          pos_base = it;
          pos_l = fetch(positions, pos_base, 0);
        }
        int64_t const pos = __shfl_sync(
            FINAL_MASK, static_cast<long long>(pos_l), it - pos_base, 32);
        if (lane < kCosSinBytes / 16) {
          cpAsync16(ring_smem + (it % kStages) * stage_bytes +
                        kSteps * kStepRows * kRowPitch + lane * 16,
                    cos_sin_cache + pos * kRotaryDim + lane * 8);
        }
      }
    }
    cpAsyncCommit();
  };

  // The prologue's rows do not wait for the positions.
  #pragma unroll
  for (int it = 0; it < kStages - 1; it++) issue_rows(it);
  #pragma unroll
  for (int it = 0; it < kStages - 1; it++) issue_cos_sin(it);

  // This lane's dims 4v .. 4v + 3 (+ 32c) of row g of the first step, at the
  // first token; each iteration advances them by stride tokens. Dims
  // 32c + 4v of a row are lane 8c + v's dims in the kernel above.
  scalar_t* in_place = qkv + static_cast<int64_t>(t0) * qkv_row +
                       (qkv_head0 + g) * kHeadDim + 4 * v;
  int const head_rows = kRole == kQ ? nq : kRole == kIQ ? niq : 0;
  int64_t const out_lane =
      (static_cast<int64_t>(t0) * head_rows + head0 + g) * kHeadDim + 4 * v;
  int64_t const out_step = static_cast<int64_t>(stride) * head_rows * kHeadDim;
  int64_t out = out_lane;

  for (int it = 0; it < n; it++) {
    issue_rows(it + kStages - 1);
    issue_cos_sin(it + kStages - 1);
    cpAsyncWait<kStages - 1>();
    __syncwarp();
    uint8_t const* const stage = ring + (it % kStages) * stage_bytes;
    float cos4[4], sin4[4];
    if constexpr (kNorm) {
      scalar_t const* const cs = reinterpret_cast<scalar_t const*>(
          stage + kSteps * kStepRows * kRowPitch);
      if constexpr (!kOneRow) {
        loadElems<scalar_t>(cs + 4 * v, cos4);
        loadElems<scalar_t>(cs + kRotaryDim / 2 + 4 * v, sin4);
      }
    }
    // Slot of the token: block and offset with one 32-bit division (the host
    // checks that every slot fits).
    int sm = -1;
    uint32_t blk = 0, tok = 0;
    if constexpr (kSlots) {
      if (it - slot_base == 32) {
        slot_base = it;
        slot_l = fetch(slot_mapping, slot_base, -1);
      }
      sm = static_cast<int>(__shfl_sync(
          FINAL_MASK, static_cast<long long>(slot_l), it - slot_base, 32));
      blk = static_cast<uint32_t>(sm) / static_cast<uint32_t>(block_size);
      tok = static_cast<uint32_t>(sm) - blk * block_size;
    }

  #pragma unroll
    for (int step = 0; step < kSteps; step++) {
      int const r = step * kStepRows + g;  // row of the warp
      int const head = head0 + r;
      bool const row_ok = r < nrows;
      int const so = step * kStepRows * kHeadDim;  // step offset, elements
      float e[kChunks][kElemsPerLane];
      if constexpr (kOneRow) {
        loadElems<scalar_t>(
            reinterpret_cast<scalar_t const*>(stage + r * kRowPitch) + 4 * v,
            e[0]);
        if constexpr (kNorm) {
          normAndRope<scalar_t>(
              e[0], lane, eps, norm_w, /*do_rope=*/true, kRotaryDim,
              reinterpret_cast<scalar_t const*>(stage + kSteps * kRowPitch),
              /*apply_norm=*/true);
        }
      } else {
        Row x;
        loadRow<scalar_t>(stage + r * kRowPitch, v, x);
        if constexpr (kNorm) normRope(x, w1, eps, cos4, sin4);
  #pragma unroll
        for (int c = 0; c < kChunks; c++) chunkElems(x, c, e[c]);
      }
      if constexpr (kRole == kQ) {
        if (row_ok) {
  #pragma unroll
          for (int c = 0; c < kLaneChunks; c++) {
            if constexpr (kOut == kQFp8Unit) {
              storeUnitScaleQElemsFp8<scalar_t>(q_fp8_out + out + so + 32 * c,
                                                e[c]);
            } else if constexpr (kOut == kQFp8) {
              storeScaledQElemsFp8<scalar_t>(q_fp8_out + out + so + 32 * c,
                                             e[c], q_fp8_inv_scale);
            } else {
              if (q_out != nullptr) {
                storeElems<scalar_t>(q_out + out + so + 32 * c, e[c]);
              } else if (q_fp8_out == nullptr && write_qkv) {
                storeElems<scalar_t>(in_place + so + 32 * c, e[c]);
              }
              if (q_fp8_out != nullptr) {
                storeScaledQElemsFp8<scalar_t>(q_fp8_out + out + so + 32 * c,
                                               e[c], q_fp8_inv_scale);
              }
            }
          }
        }
      } else if constexpr (kRole == kIQ) {
        if (row_ok) {
  #pragma unroll
          for (int c = 0; c < kLaneChunks; c++) {
            if constexpr (kOut == kIQOut) {
              if constexpr (kFp8Idx) {
                storeIndexQElemsFp8<scalar_t>(
                    reinterpret_cast<uint8_t*>(index_q_out + out + so + 32 * c),
                    e[c]);
              } else {
                storeElems<scalar_t>(index_q_out + out + so + 32 * c, e[c]);
              }
            } else if constexpr (kOut == kIQInPlace) {
              storeElems<scalar_t>(in_place + so + 32 * c, e[c]);
            }
          }
        }
      } else if constexpr (kRole == kK || kRole == kIK) {
        if constexpr (kOut == kInPlace) {
          if (row_ok) {
  #pragma unroll
            for (int c = 0; c < kLaneChunks; c++) {
              storeElems<scalar_t>(in_place + so + 32 * c, e[c]);
            }
          }
        }
      }

      if constexpr (kInsertKV && kRole == kIK) {
        if constexpr (kNvfp4Idx) {
          uint8_t* const page =
              reinterpret_cast<uint8_t*>(index_cache) + blk * idx_s_block;
  #pragma unroll
          for (int c = 0; c < kLaneChunks; c++) {
            uint8_t sf;
            uint16_t const packed =
                nvfp4QuantizeElems<scalar_t>(e[c], 1.0f, sf);
            if (sm >= 0 && row_ok) {
              nvfp4StoreElems(page, /*kv=*/0, /*head=*/0, tok, block_size,
                              kOneRow ? lane : 8 * c + v, packed, sf);
            }
          }
        } else if (sm >= 0 && row_ok) {
          int64_t const off = static_cast<int64_t>(sm) * kHeadDim + 4 * v;
  #pragma unroll
          for (int c = 0; c < kLaneChunks; c++) {
            if constexpr (kFp8Idx) {
              storeElemsFp8(
                  reinterpret_cast<uint8_t*>(index_cache) + off + 32 * c, e[c]);
            } else {
              storeElems<scalar_t>(index_cache + off + 32 * c, e[c]);
            }
          }
        }
      } else if constexpr (kInsertKV && (kRole == kK || kRole == kV)) {
        constexpr int kv = kRole == kK ? 0 : 1;
        if constexpr (kNvfp4 && !kOneRow) {
          // One 16-value scale group per lane, regrouped through the ring
          // row: lane (g, v) takes dims 16v .. 16v + 15 of row g.
          uint8_t* const row =
              ring + (it % kStages) * stage_bytes + r * kRowPitch;
          if constexpr (kRole == kK) {
            // The k in the model dtype, as the helper would quantize it (V
            // rows already hold their model-dtype values).
  #pragma unroll
            for (int c = 0; c < kChunks; c++) {
              storeElems<scalar_t>(
                  reinterpret_cast<scalar_t*>(row + 64 * c + 8 * v), e[c]);
            }
          }
          __syncwarp();
          float x[16];
  #pragma unroll
          for (int k = 0; k < 4; k++) {
            float q[kElemsPerLane];
            loadElems<scalar_t>(
                reinterpret_cast<scalar_t const*>(row + 32 * v + 8 * k), q);
  #pragma unroll
            for (int i = 0; i < kElemsPerLane; i++) x[4 * k + i] = q[i];
          }
          uint8_t sf;
          uint2 const codes = nvfp4QuantizeGroup<scalar_t>(x, kv_scale, sf);
          if (sm >= 0 && row_ok) {
            nvfp4StoreGroup(
                reinterpret_cast<uint8_t*>(kv_cache) + blk * kv_s_block, kv,
                head, tok, block_size, v, codes, sf);
          }
        } else if constexpr (kNvfp4) {
          uint8_t* const page =
              reinterpret_cast<uint8_t*>(kv_cache) + blk * kv_s_block;
  #pragma unroll
          for (int c = 0; c < kLaneChunks; c++) {
            uint8_t sf;
            uint16_t const packed =
                nvfp4QuantizeElems<scalar_t>(e[c], kv_scale, sf);
            if (sm >= 0 && row_ok) {
              nvfp4StoreElems(page, kv, head, tok, block_size,
                              kOneRow ? lane : 8 * c + v, packed, sf);
            }
          }
        } else if (sm >= 0 && row_ok) {
          cache_t* const dst = kv_cache + blk * kv_s_block + head * kv_s_head +
                               tok * kv_s_token +
                               (kv * kHeadDim + 4 * v) * kv_s_dim;
  #pragma unroll
          for (int c = 0; c < kLaneChunks; c++) {
            if constexpr (!kHasIndex) {
              using Converter = vllm::_typeConvert<scalar_t>;
  #pragma unroll
              for (int i = 0; i < kElemsPerLane; i++) {
                e[c][i] = Converter::convert(Converter::convert(e[c][i]));
              }
            }
            storeCacheElems<scalar_t, cache_t, kv_dt>(dst + 32 * c * kv_s_dim,
                                                      e[c]);
          }
        }
      }
    }
    in_place += src_step;
    out += out_step;
    __syncwarp();
  }
}

template <int kQHeads, typename scalar_t, typename cache_t,
          Fp8KVCacheDataType kv_dt, bool kNvfp4, typename out_idx_t,
          bool kHasIndex, bool kInsertKV, bool kProcessIndex, bool kFp8Idx,
          bool kNvfp4Idx>
__global__ void __launch_bounds__(kMaxWarps * 32) fusedMiniMaxM3RowGroupKernel(
    scalar_t* __restrict__ qkv, scalar_t* __restrict__ q_out,
    uint8_t* __restrict__ q_fp8_out, out_idx_t* __restrict__ index_q_out,
    scalar_t const* __restrict__ q_norm_w,
    scalar_t const* __restrict__ k_norm_w,
    scalar_t const* __restrict__ iq_norm_w,
    scalar_t const* __restrict__ ik_norm_w,
    scalar_t const* __restrict__ cos_sin_cache,
    int64_t const* __restrict__ positions,
    int64_t const* __restrict__ slot_mapping,
    int64_t const* __restrict__ index_slot_mapping,
    cache_t* __restrict__ kv_cache, out_idx_t* __restrict__ index_cache,
    float const eps, float const q_fp8_inv_scale, int const rotary_dim,
    int const num_tokens, int const nq, int const nkv, int const niq,
    int const block_size, int64_t const kv_s_block, int64_t const kv_s_head,
    int64_t const kv_s_token, int64_t const kv_s_dim, int64_t const idx_s_block,
    float const* __restrict__ kv_k_scale, float const* __restrict__ kv_v_scale,
    bool const write_qkv) {
  #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  extern __shared__ __align__(16) uint8_t smem[];
  Roles const roles(kQHeads, nq, nkv, niq, kInsertKV, kProcessIndex);
  int const qkv_row = (nq + 2 * nkv + (kHasIndex ? niq + 1 : 0)) * kHeadDim;
  // CTA (x, y) runs warps [y * cta_warps, (y + 1) * cta_warps) of the warp
  // groups of tokens x, x + gridDim.x, ...
  int const cta_warps = blockDim.x / 32;
  int w = blockIdx.y * cta_warps + threadIdx.x / 32;
  if (w >= roles.warps) return;
  constexpr int kQStage = stageBytes<kQHeads>();
  constexpr int kSideStage = stageBytes<kRows>();
  int const q_ring = kStages * kQStage;
  int const side_ring = kStages * kSideStage;
  uint8_t* const ring_w =
      smem + roles.ringOffset(w, q_ring, side_ring) -
      roles.ringOffset(blockIdx.y * cta_warps, q_ring, side_ring);
    #define RG_ARGS(NORM_W, SLOTS, CACHE_SCALE)                               \
      qkv, q_out, q_fp8_out, index_q_out, NORM_W, cos_sin_cache, positions,   \
          SLOTS, kv_cache, index_cache, eps, q_fp8_inv_scale, num_tokens, nq, \
          niq, qkv_row, block_size, kv_s_block, kv_s_head, kv_s_token,        \
          kv_s_dim, idx_s_block, CACHE_SCALE, write_qkv
    #define RG_RUN(ROLE, STEPS, OUT)                                         \
      runWarp<ROLE, STEPS, OUT, scalar_t, cache_t, kv_dt, kNvfp4, out_idx_t, \
              kHasIndex, kInsertKV, kFp8Idx, kNvfp4Idx>
    #define RG_RUN1(ROLE, OUT)                                           \
      runWarp<ROLE, 1, OUT, scalar_t, cache_t, kv_dt, kNvfp4, out_idx_t, \
              kHasIndex, kInsertKV, kFp8Idx, kNvfp4Idx, true>
  if (w < roles.q_warps) {
    int const h0 = w * kQHeads;
    constexpr int kSteps = kQHeads / kRows;
    uint8_t* const ring = ring_w;
    int const rows = min(kQHeads, nq - h0);
    if (q_out == nullptr && q_fp8_out != nullptr) {
      if (q_fp8_inv_scale == 1.0f) {
        RG_RUN(kQ, kSteps, kQFp8Unit)
        (h0, rows, h0, ring, kQStage, RG_ARGS(q_norm_w, nullptr, 1.0f));
      } else {
        RG_RUN(kQ, kSteps, kQFp8)
        (h0, rows, h0, ring, kQStage, RG_ARGS(q_norm_w, nullptr, 1.0f));
      }
    } else {
      RG_RUN(kQ, kSteps, kQAny)
      (h0, rows, h0, ring, kQStage, RG_ARGS(q_norm_w, nullptr, 1.0f));
    }
  } else {
    uint8_t* const ring = ring_w;
    w -= roles.q_warps;
    if (w < roles.kv_warps) {
      int const h0 = roles.kv_one ? w : w * kRows;
      float const scale = kNvfp4 ? *kv_k_scale : 1.0f;
      int const rows = roles.kv_one ? 1 : min(kRows, nkv - h0);
      if (rows == 1) {
        if (write_qkv) {
          RG_RUN1(kK, kInPlace)
          (h0, 1, nq + h0, ring, kSideStage,
           RG_ARGS(k_norm_w, slot_mapping, scale));
        } else {
          RG_RUN1(kK, kNone)
          (h0, 1, nq + h0, ring, kSideStage,
           RG_ARGS(k_norm_w, slot_mapping, scale));
        }
      } else if (write_qkv) {
        RG_RUN(kK, 1, kInPlace)
        (h0, rows, nq + h0, ring, kSideStage,
         RG_ARGS(k_norm_w, slot_mapping, scale));
      } else {
        RG_RUN(kK, 1, kNone)
        (h0, rows, nq + h0, ring, kSideStage,
         RG_ARGS(k_norm_w, slot_mapping, scale));
      }
    } else if ((w -= roles.kv_warps) < roles.v_warps) {
      if constexpr (kInsertKV) {
        int const h0 = roles.kv_one ? w : w * kRows;
        int const rows = roles.kv_one ? 1 : min(kRows, nkv - h0);
        float const scale = kNvfp4 ? *kv_v_scale : 1.0f;
        if (rows == 1) {
          RG_RUN1(kV, kNone)
          (h0, 1, nq + nkv + h0, ring, kSideStage,
           RG_ARGS(nullptr, slot_mapping, scale));
        } else {
          RG_RUN(kV, 1, kNone)
          (h0, rows, nq + nkv + h0, ring, kSideStage,
           RG_ARGS(nullptr, slot_mapping, scale));
        }
      }
    } else if ((w -= roles.v_warps) < roles.iq_warps) {
      if constexpr (kProcessIndex) {
        int const h0 = roles.iq_one ? w : w * kRows;
        int const rows = roles.iq_one ? 1 : min(kRows, niq - h0);
        int const qh0 = nq + 2 * nkv + h0;
        if (rows == 1) {
          if (index_q_out != nullptr) {
            RG_RUN1(kIQ, kIQOut)
            (h0, 1, qh0, ring, kSideStage, RG_ARGS(iq_norm_w, nullptr, 1.0f));
          } else if (write_qkv) {
            RG_RUN1(kIQ, kIQInPlace)
            (h0, 1, qh0, ring, kSideStage, RG_ARGS(iq_norm_w, nullptr, 1.0f));
          } else {
            RG_RUN1(kIQ, kNone)
            (h0, 1, qh0, ring, kSideStage, RG_ARGS(iq_norm_w, nullptr, 1.0f));
          }
        } else if (index_q_out != nullptr) {
          RG_RUN(kIQ, 1, kIQOut)
          (h0, rows, qh0, ring, kSideStage, RG_ARGS(iq_norm_w, nullptr, 1.0f));
        } else if (write_qkv) {
          RG_RUN(kIQ, 1, kIQInPlace)
          (h0, rows, qh0, ring, kSideStage, RG_ARGS(iq_norm_w, nullptr, 1.0f));
        } else {
          RG_RUN(kIQ, 1, kNone)
          (h0, rows, qh0, ring, kSideStage, RG_ARGS(iq_norm_w, nullptr, 1.0f));
        }
      }
    } else if constexpr (kProcessIndex) {
      if (write_qkv) {
        RG_RUN1(kIK, kInPlace)
        (0, 1, nq + 2 * nkv + niq, ring, kSideStage,
         RG_ARGS(ik_norm_w, index_slot_mapping, 1.0f));
      } else {
        RG_RUN1(kIK, kNone)
        (0, 1, nq + 2 * nkv + niq, ring, kSideStage,
         RG_ARGS(ik_norm_w, index_slot_mapping, 1.0f));
      }
    }
  }
    #undef RG_RUN
    #undef RG_RUN1
    #undef RG_ARGS
    #if __CUDA_ARCH__ >= 900
  cudaTriggerProgrammaticLaunchCompletion();
    #endif
  #endif
}

// Shared memory of the largest CTA of cta_warps warps.
template <int kQHeads>
__host__ inline int smemBytes(Roles const& r, int const cta_warps) {
  int const q_ring = kStages * stageBytes<kQHeads>();
  int const side_ring = kStages * stageBytes<kRows>();
  int bytes = 0;
  for (int w0 = 0; w0 < r.warps; w0 += cta_warps) {
    int const w1 = std::min(w0 + cta_warps, r.warps);
    bytes = std::max(bytes, r.ringOffset(w1, q_ring, side_ring) -
                                r.ringOffset(w0, q_ring, side_ring));
  }
  return bytes;
}

// Launch with the grid this instantiation's occupancy allows: the CTAs walk
// the tokens with stride gridDim.x.
template <int kQHeads, typename scalar_t, typename cache_t,
          Fp8KVCacheDataType kv_dt, bool kNvfp4, typename out_idx_t,
          bool kHasIndex, bool kInsertKV, bool kProcessIndex, bool kFp8Idx,
          bool kNvfp4Idx, typename... Args>
void launch(cudaLaunchConfig_t config, int const num_tokens, Args... args) {
  auto* const kernel =
      fusedMiniMaxM3RowGroupKernel<kQHeads, scalar_t, cache_t, kv_dt, kNvfp4,
                                   out_idx_t, kHasIndex, kInsertKV,
                                   kProcessIndex, kFp8Idx, kNvfp4Idx>;
  // Per CTA shape (warps, shared memory), computed once: the resident CTAs
  // per SM.
  static bool const opted_in =
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           get_device_prop()->sharedMemPerBlockOptin) ==
      cudaSuccess;
  static int smem_of[kMaxWarps + 1] = {};
  static int resident_of[kMaxWarps + 1] = {};
  int const warps = config.blockDim.x / 32;
  int const smem = static_cast<int>(config.dynamicSmemBytes);
  if (smem_of[warps] != smem) {
    (void)opted_in;
    int n = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kernel, warps * 32, smem);
    resident_of[warps] = n > 0 ? n : 1;
    smem_of[warps] = smem;
  }
  int const ctas_per_token = config.gridDim.y;
  config.gridDim.x = std::max(
      1, std::min(num_tokens, get_device_prop()->multiProcessorCount *
                                  resident_of[warps] / ctas_per_token));
  cudaLaunchKernelEx(&config, kernel, args...);
}

}  // namespace rg
#endif  // USE_ROCM

#ifndef USE_ROCM
// q heads per q warp: four (one step, shorter warps) for a launch of at most
// 128 q rows, or while a token has at most 32 q heads and there are few q rows
// per SM; eight otherwise (fewer, longer warps keep more CTAs resident).
inline int rgQHeads(int const num_tokens, int const nq) {
  static int const num_sms = get_device_prop()->multiProcessorCount;
  int64_t const q_rows = static_cast<int64_t>(num_tokens) * nq;
  return q_rows <= 128 || (nq <= 32 && q_rows < 64 * num_sms) ? 4 : 8;
}
// Warps per CTA: a token's whole warp group once there are enough tokens to
// give every SM CTAs; before that, eight, which spreads a token over SMs.
inline int rgCtaWarps(int const num_tokens) {
  static int const num_sms = get_device_prop()->multiProcessorCount;
  return 2 * num_tokens >= 3 * num_sms ? rg::kMaxWarps : 8;
}
#endif

// ────────────────────────────────────────────────────────────────────────────
// Launch wrapper
// ────────────────────────────────────────────────────────────────────────────
template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt,
          bool kNvfp4 = false>
void launchFusedMiniMaxM3(
    scalar_t* qkv, scalar_t* q_out, uint8_t* q_fp8_out, void* index_q_out,
    scalar_t const* q_norm_w, scalar_t const* k_norm_w,
    scalar_t const* iq_norm_w, scalar_t const* ik_norm_w,
    scalar_t const* cos_sin_cache, int64_t const* positions,
    int64_t const* slot_mapping, int64_t const* index_slot_mapping,
    cache_t* kv_cache, void* index_cache, float const eps,
    float const q_fp8_inv_scale, int const rotary_dim, int const num_tokens,
    int const nq, int const nkv, int const niq, int const block_size,
    int64_t const kv_s_block, int64_t const kv_s_head, int64_t const kv_s_token,
    int64_t const kv_s_dim, int64_t const idx_s_block, bool const has_index,
    bool const insert_kv, bool const process_index, bool const fp8_idx,
    bool const nvfp4_idx, float const* kv_k_scale, float const* kv_v_scale,
    bool const write_qkv, bool const slots_fit_int32, cudaStream_t stream) {
  // Index outputs are scalar_t (bf16) or e4m3 bytes (uint8_t); reinterpret the
  // void* pointers per instantiation in the LAUNCH macro.
  // Slot count must match the kernel's compile-time gating.
  int const v_slots = insert_kv ? nkv : 0;
  int const idx_slots = process_index ? niq + 1 : 0;
  int const slots_per_token = nq + nkv + v_slots + idx_slots;

  constexpr int kBlockSize = 256;
  constexpr int kWarpsPerBlock = kBlockSize / 32;
  int64_t const total_warps =
      static_cast<int64_t>(num_tokens) * slots_per_token;
  int const grid =
      static_cast<int>((total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock);
  if (grid == 0) return;

#ifndef USE_ROCM
  // PDL: enable programmatic stream serialization whenever the hardware
  // supports it (SM90+).  On pre-Hopper GPUs the attribute is unavailable, so
  // leave numAttrs = 0 and launch as a regular kernel via cudaLaunchKernelEx.
  static int const sm_version = getSMVersion();
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

  // The row-group kernel: same bytes, for rotary_dim == 64 on SM80+ (cp.async).
  int const q_heads = rgQHeads(num_tokens, nq);
  rg::Roles const roles(q_heads, nq, nkv, niq, insert_kv, process_index);
  auto const rg_smem = [&](int const warps) {
    return q_heads == 8 ? rg::smemBytes<8>(roles, warps)
                        : rg::smemBytes<4>(roles, warps);
  };
  int cta_warps =
      std::min({roles.warps, rgCtaWarps(num_tokens), rg::kMaxWarps});
  // Eight warps take at most 63 KB; a whole group can exceed the 99 KB per
  // block of SM 8.6 / 8.9 / 12.x.
  static int const max_block_smem = get_device_prop()->sharedMemPerBlockOptin;
  if (rg_smem(cta_warps) > max_block_smem) cta_warps = std::min(roles.warps, 8);
  auto const aligned = [](void const* p, uintptr_t const n) {
    return reinterpret_cast<uintptr_t>(p) % n == 0;
  };
  bool const row_groups =
      sm_version >= 80 && rotary_dim == rg::kRotaryDim && slots_fit_int32 &&
      aligned(qkv, 16) && aligned(cos_sin_cache, 16) && aligned(q_norm_w, 8) &&
      aligned(k_norm_w, 8) && aligned(iq_norm_w, 8) && aligned(ik_norm_w, 8);
  cudaLaunchConfig_t rg_config = config;
  rg_config.blockDim = dim3(cta_warps * 32);
  rg_config.gridDim = dim3(1, (roles.warps + cta_warps - 1) / cta_warps);
  rg_config.dynamicSmemBytes = rg_smem(cta_warps);

  #define LAUNCH_ARGS(OUT_T)                                                \
    qkv, q_out, q_fp8_out, reinterpret_cast<OUT_T*>(index_q_out), q_norm_w, \
        k_norm_w, iq_norm_w, ik_norm_w, cos_sin_cache, positions,           \
        slot_mapping, index_slot_mapping, kv_cache,                         \
        reinterpret_cast<OUT_T*>(index_cache), eps, q_fp8_inv_scale,        \
        rotary_dim, num_tokens, nq, nkv, niq, block_size, kv_s_block,       \
        kv_s_head, kv_s_token, kv_s_dim, idx_s_block, kv_k_scale, kv_v_scale
  #define LAUNCH_RG(QH, HAS_INDEX, INSERT, PROCESS_INDEX, FP8, NVFP4_IDX,      \
                    OUT_T)                                                     \
    rg::launch<QH, scalar_t, cache_t, kv_dt, kNvfp4, OUT_T, HAS_INDEX, INSERT, \
               PROCESS_INDEX, FP8, NVFP4_IDX>(rg_config, num_tokens,           \
                                              LAUNCH_ARGS(OUT_T), write_qkv)
  #define LAUNCH(HAS_INDEX, INSERT, PROCESS_INDEX, FP8, NVFP4_IDX, OUT_T)      \
    do {                                                                       \
      if (!row_groups) {                                                       \
        cudaLaunchKernelEx(                                                    \
            &config,                                                           \
            fusedMiniMaxM3QNormRopeKVInsertKernel<                             \
                scalar_t, cache_t, kv_dt, kNvfp4, OUT_T, HAS_INDEX, INSERT,    \
                PROCESS_INDEX, FP8, NVFP4_IDX>,                                \
            LAUNCH_ARGS(OUT_T));                                               \
      } else if (q_heads == 8) {                                               \
        LAUNCH_RG(8, HAS_INDEX, INSERT, PROCESS_INDEX, FP8, NVFP4_IDX, OUT_T); \
      } else {                                                                 \
        LAUNCH_RG(4, HAS_INDEX, INSERT, PROCESS_INDEX, FP8, NVFP4_IDX, OUT_T); \
      }                                                                        \
    } while (0)
#else
  // ROCm: standard kernel launch syntax (no PDL/stream serialization).
  // clang-format off
  #define LAUNCH(HAS_INDEX, INSERT, PROCESS_INDEX, FP8, NVFP4_IDX, OUT_T)   \
    fusedMiniMaxM3QNormRopeKVInsertKernel<                                  \
        scalar_t, cache_t, kv_dt, kNvfp4, OUT_T, HAS_INDEX, INSERT,         \
        PROCESS_INDEX, FP8, NVFP4_IDX><<<grid, kBlockSize, 0, stream>>>(    \
        qkv, q_out, q_fp8_out, reinterpret_cast<OUT_T*>(index_q_out),       \
        q_norm_w, k_norm_w, iq_norm_w, ik_norm_w, cos_sin_cache, positions, \
        slot_mapping, index_slot_mapping, kv_cache,                         \
        reinterpret_cast<OUT_T*>(index_cache), eps, q_fp8_inv_scale,        \
        rotary_dim, num_tokens, nq, nkv, niq, block_size, kv_s_block,       \
        kv_s_head, kv_s_token, kv_s_dim, idx_s_block, kv_k_scale,           \
        kv_v_scale)
  // clang-format on
#endif

  if constexpr (kNvfp4) {
    // NVFP4 main cache: sparse serving only (the host checks insert_kv).
    if (!process_index) {
      LAUNCH(true, true, false, false, false, scalar_t);
    } else if (nvfp4_idx) {
      LAUNCH(true, true, true, true, true, uint8_t);
    } else if (fp8_idx) {
      LAUNCH(true, true, true, true, false, uint8_t);
    } else {
      LAUNCH(true, true, true, false, false, scalar_t);
    }
  } else if (has_index) {
    if (!process_index) {
      if (insert_kv) {
        LAUNCH(true, true, false, false, false, scalar_t);
      } else {
        LAUNCH(true, false, false, false, false, scalar_t);
      }
    } else if (insert_kv) {
      if (nvfp4_idx) {
        LAUNCH(true, true, true, true, true,
               uint8_t);  // sparse serving, NVFP4 index-K, fp8 index_q
      } else if (fp8_idx) {
        LAUNCH(true, true, true, true, false,
               uint8_t);  // sparse serving, fp8 index outputs
      } else {
        LAUNCH(true, true, true, false, false,
               scalar_t);  // sparse serving, bf16
      }
    } else {
      if (fp8_idx) {
        LAUNCH(true, false, true, true, false,
               uint8_t);  // sparse profiling, fp8 index_q
      } else {
        LAUNCH(true, false, true, false, false,
               scalar_t);  // sparse profiling, bf16
      }
    }
  } else if (insert_kv) {
    LAUNCH(false, true, false, false, false, scalar_t);  // dense, KV insert
  } else {
    LAUNCH(false, false, false, false, false, scalar_t);  // dense
  }
#undef LAUNCH
#undef LAUNCH_RG
#undef LAUNCH_ARGS
}

#ifndef USE_ROCM
// ────────────────────────────────────────────────────────────────────────────
// DSpark draft context K/V: every draft layer in one launch (bf16 in).
// ────────────────────────────────────────────────────────────────────────────
// The draft projects the target's context states to the K/V of all its layers
// with one GEMM, ``all_kv [N, L, 2, nkv, 128]``. blockIdx.z picks the layer,
// blockIdx.y the head, blockIdx.x a run of tokens. Eight lanes own one token;
// lane l holds two 8-dim chunks of K and of V, A = dims [8l, 8l + 8) and
// B = dims [64 + 8l, 64 + 8l + 8), so with rotary_dim == 64 every lane rotates
// its A chunk (partner lane l ^ 4) and no lane idles through the RoPE. K gets
// the layer's Gemma k-norm (rounded to bf16 like the unfused norm) and partial
// NeoX RoPE (fp32 math rounded once, like rotary_embedding); K and V go into
// the token's cache row like reshape_and_cache_flash (fp8: e4m3(bf16(x) *
// (1 / scale)); a unit scale converts the bf16 pairs directly).
// Layer caches are separate allocations of one layout, logical shape
// [nb, nkv, bs, 2*128] with K at [..., :128] and V at [..., 128:]; they are
// addressed as layer 0's cache plus a per-layer element offset (likewise the
// fp8 k/v scales from layer 0's k scale), so the launch is CUDA-graph safe.
constexpr int kCtxLanesPerRow = 8;
constexpr int kCtxElems = 8;  // bf16 per 16-byte chunk
constexpr int kCtxHalfVecs = kHeadDim / 2 / kCtxElems;  // 8: chunk B offset
constexpr int kCtxBlockSize = 256;
constexpr int kCtxRowsPerBlock = kCtxBlockSize / kCtxLanesPerRow;  // 32

__device__ __forceinline__ float bf16lo(uint32_t v) {
  return __uint_as_float(v << 16);
}
__device__ __forceinline__ float bf16hi(uint32_t v) {
  return __uint_as_float(v & 0xffff0000u);
}
__device__ __forceinline__ uint32_t packBf16(float lo, float hi) {
  __nv_bfloat162 const r = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t const*>(&r);
}
// Loads of what the PDL predecessor may still be writing: coherent (an .nc
// load may be hoisted above griddepcontrol.wait).
__device__ __forceinline__ uint4 ldgV4(uint4 const* p) {
  uint4 r;
  asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p));
  return r;
}
// Low 32 bits of an int64 (positions and slots fit).
__device__ __forceinline__ int ldgLo32(int64_t const* p) {
  int r;
  asm volatile("ld.global.b32 %0, [%1];" : "=r"(r) : "l"(p));
  return r;
}
// Same approximation as flashinfer's gemma_rmsnorm.
__device__ __forceinline__ float rsqrtApproxFtz(float x) {
  float r;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
// e4m3 of four floats, low byte first.
__device__ __forceinline__ uint32_t packE4m3(float a, float b, float c,
                                             float d) {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  uint32_t r;
  asm("{\n"
      "  .reg .b16 lo, hi;\n"
      "  cvt.rn.satfinite.e4m3x2.f32 lo, %2, %1;\n"
      "  cvt.rn.satfinite.e4m3x2.f32 hi, %4, %3;\n"
      "  mov.b32 %0, {lo, hi};\n"
      "}"
      : "=r"(r)
      : "f"(a), "f"(b), "f"(c), "f"(d));
  return r;
  #else
  uint32_t const lo =
      __nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
  uint32_t const hi =
      __nv_cvt_float2_to_fp8x2(make_float2(c, d), __NV_SATFINITE, __NV_E4M3);
  return lo | (hi << 16);
  #endif
}
// Eight bf16 (4 packed pairs) * inv_scale -> e4m3, low byte first.
__device__ __forceinline__ uint2 bf16x8ToE4m3(uint4 const p,
                                              float const inv_scale) {
  uint32_t const w[4] = {p.x, p.y, p.z, p.w};
  uint32_t r[2];
  #pragma unroll
  for (int i = 0; i < 2; i++) {
    r[i] = packE4m3(bf16lo(w[2 * i]) * inv_scale, bf16hi(w[2 * i]) * inv_scale,
                    bf16lo(w[2 * i + 1]) * inv_scale,
                    bf16hi(w[2 * i + 1]) * inv_scale);
  }
  return make_uint2(r[0], r[1]);
}
  #if (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && \
       __CUDA_ARCH_FAMILY_SPECIFIC__ >= 1000) || \
      (defined(__CUDA_ARCH_SPECIFIC__) && __CUDA_ARCH_SPECIFIC__ >= 1000)
    #define VLLM_CTX_KV_BF16X2_TO_E4M3X2
  #endif
// Eight bf16 -> e4m3 with a unit scale (exact); one convert per pair where
// the arch converts bf16x2 -> e4m3x2.
__device__ __forceinline__ uint2 bf16x8ToE4m3(uint4 const p) {
  #ifdef VLLM_CTX_KV_BF16X2_TO_E4M3X2
  uint32_t r[2];
  uint32_t const w[4] = {p.x, p.y, p.z, p.w};
    #pragma unroll
  for (int i = 0; i < 2; i++) {
    asm("{\n"
        "  .reg .b16 lo, hi;\n"
        "  cvt.rn.satfinite.e4m3x2.bf16x2 lo, %1;\n"
        "  cvt.rn.satfinite.e4m3x2.bf16x2 hi, %2;\n"
        "  mov.b32 %0, {lo, hi};\n"
        "}"
        : "=r"(r[i])
        : "r"(w[2 * i]), "r"(w[2 * i + 1]));
  }
  return make_uint2(r[0], r[1]);
  #else
  return bf16x8ToE4m3(p, 1.0f);
  #endif
}

template <bool kFp8>
__global__ void __launch_bounds__(kCtxBlockSize)
    fusedMiniMaxM3DSparkContextKVKernel(
        uint4 const* __restrict__ all_kv,           // bf16 [N, L, 2, nkv, 128]
        uint4 const* __restrict__ k_norm_w,         // bf16 [L, 128]
        uint4 const* __restrict__ cos_sin_cache,    // bf16 [max_pos, rot]
        int64_t const* __restrict__ positions,      // [N]
        int64_t const* __restrict__ slot_mapping,   // [N], < 0: no write
        uint8_t* __restrict__ kv_cache,             // layer 0's cache, bytes
        int64_t const* __restrict__ cache_offsets,  // [L] elements from it
        float const* __restrict__ scales,           // layer 0's k scale
        int64_t const* __restrict__ scale_offsets,  // [2, L] k, v; floats
        float const eps, int const rotary_dim, int const num_tokens,
        int const num_layers, int const block_shift, int const s_token,
        int const s_layer, int const s_side, int const s_head,
        int const kv_s_block, int const kv_s_head, int const kv_s_token) {
  // Strides are in 16-byte (8 x bf16) units for all_kv / cos_sin, elements
  // for the cache.
  constexpr int kElemBytes = kFp8 ? 1 : 2;
  int const row0 = blockIdx.x * kCtxRowsPerBlock;
  // Warps with no token exit; the others run their tail rows too (the
  // shuffles need every lane) and skip the store.
  if (row0 + threadIdx.x / 32 * (32 / kCtxLanesPerRow) >= num_tokens) return;
  int const head = blockIdx.y;
  int const layer = blockIdx.z;
  int const lane = threadIdx.x % kCtxLanesPerRow;
  int const dim0 = lane * kCtxElems;  // chunk A; chunk B is 64 dims later
  int const t = row0 + threadIdx.x / kCtxLanesPerRow;

  int const half = rotary_dim / 2;
  bool const in_rope = dim0 < rotary_dim;
  bool const first_half = dim0 < half;
  int const cs_vec =
      in_rope ? (first_half ? dim0 : dim0 - half) / kCtxElems : 0;
  int const lane_xor = half / kCtxElems;
  int const rot_vecs = rotary_dim / kCtxElems;
  // x * cos - partner * sin in the first half, + in the second.
  uint32_t const sin_sign = first_half ? 0x80008000u : 0u;

  uint8_t* const cache =
      kv_cache + (cache_offsets[layer] + head * kv_s_head + dim0) * kElemBytes;
  float k_scale = 1.0f, v_scale = 1.0f;
  if constexpr (kFp8) {
    int64_t const k_off = scale_offsets[layer];
    int64_t const v_off = scale_offsets[num_layers + layer];
    k_scale = scales[k_off];
    v_scale = scales[v_off];
  }
  float g[2][kCtxElems];  // 1 + w, chunks A and B
  #pragma unroll
  for (int c = 0; c < 2; c++) {
    uint4 const w =
        k_norm_w[layer * (kHeadDim / kCtxElems) + c * kCtxHalfVecs + lane];
    uint32_t const wp[4] = {w.x, w.y, w.z, w.w};
  #pragma unroll
    for (int i = 0; i < 4; i++) {
      g[c][2 * i] = 1.0f + bf16lo(wp[i]);
      g[c][2 * i + 1] = 1.0f + bf16hi(wp[i]);
    }
  }

  #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaGridDependencySynchronize();
  #endif

  bool const valid = t < num_tokens;
  int const ts = valid ? t : 0;
  int const pos = ldgLo32(positions + ts);
  int const slot = valid ? ldgLo32(slot_mapping + ts) : -1;
  uint4 const* const src =
      all_kv + ts * s_token + layer * s_layer + head * s_head + lane;
  uint4 kr[2], vr[2];
  #pragma unroll
  for (int c = 0; c < 2; c++) {
    kr[c] = ldgV4(src + c * kCtxHalfVecs);
    vr[c] = ldgV4(src + s_side + c * kCtxHalfVecs);
  }
  // Lanes past the rotary dims load (and later discard) a valid row too.
  uint4 const* const cs = cos_sin_cache + pos * rot_vecs + cs_vec;
  uint4 const cr = cs[0];
  uint4 const sr = cs[lane_xor];

  bool const store = slot >= 0;
  // Power-of-two block size: shift and mask, not division.
  uint8_t* const dst =
      cache + (static_cast<int64_t>(slot >> block_shift) * kv_s_block +
               (slot & ((1 << block_shift) - 1)) * kv_s_token) *
                  kElemBytes;
  constexpr int kChunkB = kHeadDim / 2 * kElemBytes;
  constexpr int kV = kHeadDim * kElemBytes;
  float const k_inv = __fdividef(1.0f, k_scale);
  float const v_inv = __fdividef(1.0f, v_scale);
  #ifdef VLLM_CTX_KV_BF16X2_TO_E4M3X2
  bool const unit_scale = k_scale == 1.0f && v_scale == 1.0f;
  #else
  constexpr bool unit_scale = false;
  #endif
  // Eight bf16 -> the cache row at byte offset `off`.
  auto put = [&](int const off, uint4 const x, float const inv_scale) {
    if constexpr (kFp8) {
      uint2 const y = unit_scale ? bf16x8ToE4m3(x) : bf16x8ToE4m3(x, inv_scale);
      if (store) *reinterpret_cast<uint2*>(dst + off) = y;
    } else {
      if (store) *reinterpret_cast<uint4*>(dst + off) = x;
    }
  };

  // V first, so its registers free up before the K math.
  #pragma unroll
  for (int c = 0; c < 2; c++) put(kV + c * kChunkB, vr[c], v_inv);

  // K: Gemma RMSNorm over the 8 lanes of the row.
  float x[2][kCtxElems];
  float sumsq = 0.0f;
  #pragma unroll
  for (int c = 0; c < 2; c++) {
    uint32_t const kp[4] = {kr[c].x, kr[c].y, kr[c].z, kr[c].w};
  #pragma unroll
    for (int j = 0; j < 4; j++) {
      x[c][2 * j] = bf16lo(kp[j]);
      x[c][2 * j + 1] = bf16hi(kp[j]);
      sumsq = fmaf(x[c][2 * j], x[c][2 * j], sumsq);
      sumsq = fmaf(x[c][2 * j + 1], x[c][2 * j + 1], sumsq);
    }
  }
  #pragma unroll
  for (int mask = kCtxLanesPerRow / 2; mask > 0; mask >>= 1) {
    sumsq += __shfl_xor_sync(FINAL_MASK, sumsq, mask, 32);
  }
  float const rr = rsqrtApproxFtz(sumsq * (1.0f / kHeadDim) + eps);
  // bf16-rounded norm output, packed.
  uint32_t kn[2][4];
  #pragma unroll
  for (int c = 0; c < 2; c++) {
  #pragma unroll
    for (int j = 0; j < 4; j++) {
      kn[c][j] = packBf16(x[c][2 * j] * rr * g[c][2 * j],
                          x[c][2 * j + 1] * rr * g[c][2 * j + 1]);
    }
  }
  // RoPE on chunk A; the partner chunk is half / 8 lanes away.
  uint32_t const cp[4] = {cr.x, cr.y, cr.z, cr.w};
  uint32_t const sp[4] = {sr.x ^ sin_sign, sr.y ^ sin_sign, sr.z ^ sin_sign,
                          sr.w ^ sin_sign};
  #pragma unroll
  for (int j = 0; j < 4; j++) {
    uint32_t const pn = __shfl_xor_sync(FINAL_MASK, kn[0][j], lane_xor, 32);
    uint32_t const rot =
        packBf16(__fmaf_rn(bf16lo(kn[0][j]), bf16lo(cp[j]),
                           __fmul_rn(bf16lo(pn), bf16lo(sp[j]))),
                 __fmaf_rn(bf16hi(kn[0][j]), bf16hi(cp[j]),
                           __fmul_rn(bf16hi(pn), bf16hi(sp[j]))));
    kn[0][j] = in_rope ? rot : kn[0][j];
  }
  #pragma unroll
  for (int c = 0; c < 2; c++) {
    put(c * kChunkB, make_uint4(kn[c][0], kn[c][1], kn[c][2], kn[c][3]), k_inv);
  }

  #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaTriggerProgrammaticLaunchCompletion();
  #endif
}

template <bool kFp8>
void launchFusedMiniMaxM3DSparkContextKV(
    uint4 const* all_kv, uint4 const* k_norm_w, uint4 const* cos_sin_cache,
    int64_t const* positions, int64_t const* slot_mapping, uint8_t* kv_cache,
    int64_t const* cache_offsets, float const* scales,
    int64_t const* scale_offsets, float const eps, int const rotary_dim,
    int const num_tokens, int const num_layers, int const nkv,
    int const block_shift, int const s_token, int const s_layer,
    int const s_side, int const s_head, int const kv_s_block,
    int const kv_s_head, int const kv_s_token, cudaStream_t stream) {
  if (num_tokens == 0) return;
  static int const sm_version = getSMVersion();
  cudaLaunchConfig_t config;
  config.gridDim = dim3((num_tokens + kCtxRowsPerBlock - 1) / kCtxRowsPerBlock,
                        nkv, num_layers);
  config.blockDim = dim3(kCtxBlockSize);
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attrs;
  config.numAttrs = (sm_version >= 90) ? 1 : 0;
  cudaLaunchKernelEx(&config, fusedMiniMaxM3DSparkContextKVKernel<kFp8>, all_kv,
                     k_norm_w, cos_sin_cache, positions, slot_mapping, kv_cache,
                     cache_offsets, scales, scale_offsets, eps, rotary_dim,
                     num_tokens, num_layers, block_shift, s_token, s_layer,
                     s_side, s_head, kv_s_block, kv_s_head, kv_s_token);
}
#endif  // !USE_ROCM

}  // namespace minimax_m3_fused_ops
}  // namespace vllm

// clang-format off
#define CALL_FUSED_MINIMAX_M3_IMPL(CACHE_T, KV_DTYPE, NVFP4)              \
  vllm::minimax_m3_fused_ops::launchFusedMiniMaxM3<st, CACHE_T, KV_DTYPE,      \
                                                   NVFP4>(                \
      reinterpret_cast<st*>(qkv.data_ptr()),                                   \
      q_out.has_value() ? reinterpret_cast<st*>(q_out->data_ptr()) : nullptr,  \
      q_fp8_out.has_value()                                                     \
          ? reinterpret_cast<uint8_t*>(q_fp8_out->data_ptr())                  \
          : nullptr,                                                           \
      index_q_out.has_value()                                                  \
          ? reinterpret_cast<void*>(index_q_out->data_ptr())                   \
          : nullptr,                                                           \
      reinterpret_cast<st const*>(q_norm_weight.data_ptr()),                   \
      reinterpret_cast<st const*>(k_norm_weight.data_ptr()),                   \
      process_index                                                            \
          ? reinterpret_cast<st const*>(index_q_norm_weight->data_ptr())       \
          : nullptr,                                                           \
      process_index                                                            \
          ? reinterpret_cast<st const*>(index_k_norm_weight->data_ptr())       \
          : nullptr,                                                           \
      reinterpret_cast<st const*>(cos_sin_cache.data_ptr()),                   \
      reinterpret_cast<int64_t const*>(positions.data_ptr()),                  \
      insert_kv ? reinterpret_cast<int64_t const*>(slot_mapping->data_ptr())   \
                : nullptr,                                                     \
      (insert_kv && process_index)                                             \
          ? reinterpret_cast<int64_t const*>(                                  \
                effective_index_slot_mapping->data_ptr())                      \
          : nullptr,                                                           \
      insert_kv ? reinterpret_cast<CACHE_T*>(kv_cache->data_ptr()) : nullptr,  \
      (insert_kv && process_index)                                             \
          ? reinterpret_cast<void*>(index_cache->data_ptr())                   \
          : nullptr,                                                           \
      static_cast<float>(eps), 1.0f / static_cast<float>(q_fp8_scale),          \
      static_cast<int>(rotary_dim), num_tokens, nq, nkv, niq,                  \
      static_cast<int>(block_size), kv_s_block, kv_s_head, kv_s_token,         \
      kv_s_dim, idx_s_block, has_index, insert_kv, process_index, fp8_idx,     \
      nvfp4_idx, kv_k_scale_ptr, kv_v_scale_ptr, write_qkv, slots_fit_int32,  \
      stream)
#define CALL_FUSED_MINIMAX_M3(_RAW_T, CACHE_T, KV_DTYPE)                       \
  CALL_FUSED_MINIMAX_M3_IMPL(CACHE_T, KV_DTYPE, false)
// clang-format on

// ────────────────────────────────────────────────────────────────────────────
// Torch op wrapper
// ────────────────────────────────────────────────────────────────────────────
void fused_minimax_m3_qknorm_rope_kv_insert(
    torch::stable::Tensor& qkv,  // [N, qkv_row] (packs index if sparse)
    torch::stable::Tensor const& q_norm_weight,  // [128]
    torch::stable::Tensor const& k_norm_weight,  // [128]
    torch::stable::Tensor const& cos_sin_cache,  // [max_pos, rotary_dim]
    torch::stable::Tensor const& positions,      // [N] i64
    int64_t num_heads, int64_t num_kv_heads, int64_t rotary_dim, double eps,
    std::optional<torch::stable::Tensor> index_q_norm_weight,  // [128]
    std::optional<torch::stable::Tensor> index_k_norm_weight,  // [128]
    int64_t num_index_heads,                                  // niq; 0 => dense
    std::optional<torch::stable::Tensor> slot_mapping,        // [N] i64
    std::optional<torch::stable::Tensor> index_slot_mapping,  // [N] i64
    std::optional<torch::stable::Tensor> kv_cache,     // [nb,nkv,bs,2*128]
    std::optional<torch::stable::Tensor> index_cache,  // [nb,bs,128]
    int64_t block_size,
    std::optional<torch::stable::Tensor> q_out,  // [N, nq*128] contiguous
    std::optional<torch::stable::Tensor>
        index_q_out,  // [N, niq*128] contiguous
    const std::string& kv_cache_dtype, bool skip_index_branch,
    std::optional<torch::stable::Tensor>
        q_fp8_out,  // [N, nq*128] contiguous E4M3
    double q_fp8_scale,
    std::optional<torch::stable::Tensor> kv_k_scale,  // [] f32, NVFP4 only
    std::optional<torch::stable::Tensor> kv_v_scale, bool write_qkv) {
  STD_TORCH_CHECK(qkv.is_cuda() && qkv.is_contiguous(),
                  "qkv must be contiguous CUDA");
  STD_TORCH_CHECK(
      qkv.scalar_type() == torch::headeronly::ScalarType::Half ||
          qkv.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "qkv must be float16 or bfloat16");
  STD_TORCH_CHECK(
      positions.is_cuda() &&
          positions.scalar_type() == torch::headeronly::ScalarType::Long,
      "positions must be int64 CUDA");
  STD_TORCH_CHECK(cos_sin_cache.is_cuda() && cos_sin_cache.is_contiguous(),
                  "cos_sin_cache must be contiguous CUDA");
  STD_TORCH_CHECK(cos_sin_cache.scalar_type() == qkv.scalar_type(),
                  "cos_sin_cache dtype must match qkv");
  STD_TORCH_CHECK(
      cos_sin_cache.dim() == 2 && cos_sin_cache.size(1) == rotary_dim,
      "cos_sin_cache shape [max_pos, rotary_dim]");

  STD_TORCH_CHECK(q_norm_weight.scalar_type() == qkv.scalar_type() &&
                      k_norm_weight.scalar_type() == qkv.scalar_type(),
                  "q/k norm weight dtype must match qkv");
  STD_TORCH_CHECK(
      q_norm_weight.numel() == vllm::minimax_m3_fused_ops::kHeadDim &&
          k_norm_weight.numel() == vllm::minimax_m3_fused_ops::kHeadDim,
      "q/k norm weight must have 128 elements");
  STD_TORCH_CHECK(rotary_dim > 0 && rotary_dim % 8 == 0 &&
                      rotary_dim <= vllm::minimax_m3_fused_ops::kHeadDim,
                  "rotary_dim must be a positive multiple of 8 and <= 128");

  int const num_tokens = static_cast<int>(qkv.size(0));
  int const nq = static_cast<int>(num_heads);
  int const nkv = static_cast<int>(num_kv_heads);
  int const niq = static_cast<int>(num_index_heads);

  // The sparse layer packs the index branch ([index_q (niq heads) | index_k
  // (1 head)]) right after [q|k|v] in the same row; the dense layer does not.
  bool const has_index = niq > 0;
  bool const insert_kv = kv_cache.has_value();
  bool const process_index = has_index && !skip_index_branch;
  bool const nvfp4_kv = kv_cache_dtype == "nvfp4";
  vllm::Fp8KVCacheDataType const kv_dt =
      nvfp4_kv ? vllm::Fp8KVCacheDataType::kAuto
               : vllm::get_fp8_kv_cache_data_type(kv_cache_dtype);
  int const kHeadDim = vllm::minimax_m3_fused_ops::kHeadDim;
  int const expected_row =
      (nq + 2 * nkv + (has_index ? niq + 1 : 0)) * kHeadDim;
  STD_TORCH_CHECK(qkv.size(1) == expected_row,
                  "qkv last dim must be (num_heads + 2*num_kv_heads"
                  " + num_index_heads + 1) * 128 for sparse, "
                  "(num_heads + 2*num_kv_heads) * 128 for dense");

  STD_TORCH_CHECK(!nvfp4_kv || has_index,
                  "nvfp4 kv_cache requires the index branch (sparse layer)");
  STD_TORCH_CHECK(has_index || !skip_index_branch,
                  "skip_index_branch requires sparse qkv rows");
  if (process_index) {
    STD_TORCH_CHECK(
        index_q_norm_weight.has_value() && index_k_norm_weight.has_value(),
        "index branch requires both index norm weights");
    STD_TORCH_CHECK(index_q_norm_weight->scalar_type() == qkv.scalar_type() &&
                        index_k_norm_weight->scalar_type() == qkv.scalar_type(),
                    "index norm weights dtype must match qkv");
    STD_TORCH_CHECK(index_q_norm_weight->numel() == kHeadDim &&
                        index_k_norm_weight->numel() == kHeadDim,
                    "index norm weights must have 128 elements");
  }
  // kv_cache strides (logical shape [nb, nkv, bs, 2*head_dim]). Read straight
  // off the tensor so the kernel honours whatever physical layout the attention
  // backend allocated (NHD: stride order (0,2,1,3); HND: (0,1,2,3)). No new
  // op argument is needed -- the strides ride along with the tensor itself.
  int64_t kv_s_block = 0, kv_s_head = 0, kv_s_token = 0, kv_s_dim = 0;
  torch::stable::Tensor const* effective_index_slot_mapping = nullptr;
  if (insert_kv) {
    STD_TORCH_CHECK(
        slot_mapping.has_value() && slot_mapping->is_cuda() &&
            slot_mapping->scalar_type() == torch::headeronly::ScalarType::Long,
        "insert mode requires int64 CUDA slot_mapping");
    if (process_index) {
      STD_TORCH_CHECK(
          !index_slot_mapping.has_value() ||
              (index_slot_mapping->is_cuda() &&
               index_slot_mapping->scalar_type() ==
                   torch::headeronly::ScalarType::Long &&
               index_slot_mapping->numel() == slot_mapping->numel()),
          "index_slot_mapping must be int64 CUDA with slot_mapping length");
    }
    // Main attention KV cache: auto matches qkv, fp8 and nvfp4 use uint8
    // storage.
    if (nvfp4_kv) {
      STD_TORCH_CHECK(
          kv_cache->scalar_type() == torch::headeronly::ScalarType::Byte,
          "nvfp4 kv_cache must use uint8 storage");
    } else if (kv_dt == vllm::Fp8KVCacheDataType::kAuto) {
      STD_TORCH_CHECK(kv_cache->scalar_type() == qkv.scalar_type(),
                      "auto kv_cache dtype must match qkv");
    } else {
      STD_TORCH_CHECK(
          kv_cache->scalar_type() == torch::headeronly::ScalarType::Byte,
          "fp8 kv_cache must use uint8 storage");
    }
    // Indexer index-K cache: independent dtype -- qkv dtype, fp8 e4m3, or
    // uint8 packed NVFP4 pages.
    if (process_index) {
      STD_TORCH_CHECK(
          index_cache.has_value() &&
              (index_cache->scalar_type() == qkv.scalar_type() ||
               index_cache->scalar_type() ==
                   torch::headeronly::ScalarType::Float8_e4m3fn ||
               index_cache->scalar_type() ==
                   torch::headeronly::ScalarType::Byte),
          "insert mode requires index_cache matching qkv dtype, fp8 e4m3, "
          "or uint8 (nvfp4)");
    }
    STD_TORCH_CHECK(kv_cache->dim() == 4 && kv_cache->stride(3) == 1,
                    "kv_cache must be [nb,nkv,bs,2*head_dim] with contiguous "
                    "content dim (stride(3)==1)");
    if (nvfp4_kv) {
      // Packed HND pages [nb, 2*nkv, bs, 72]: slot 2*head + side is one
      // contiguous [bs, 72] block, which the store addresses directly.
      int64_t const full_dim = kHeadDim / 2 + kHeadDim / 16;
      STD_TORCH_CHECK(
          kv_cache->size(1) == 2 * nkv && kv_cache->size(2) == block_size &&
              kv_cache->size(3) == full_dim &&
              kv_cache->stride(2) == full_dim &&
              kv_cache->stride(1) == block_size * full_dim &&
              block_size % 4 == 0,
          "nvfp4 kv_cache must be HND [nb, 2*num_kv_heads, block_size, 72] "
          "with contiguous pages and block_size % 4 == 0");
      for (auto const* scale : {&kv_k_scale, &kv_v_scale}) {
        STD_TORCH_CHECK(
            scale->has_value() && (*scale)->is_cuda() &&
                (*scale)->scalar_type() ==
                    torch::headeronly::ScalarType::Float &&
                (*scale)->numel() >= 1,
            "nvfp4 kv_cache requires float32 CUDA kv_k_scale and kv_v_scale");
      }
    }
    kv_s_block = kv_cache->stride(0);
    kv_s_head = kv_cache->stride(1);
    kv_s_token = kv_cache->stride(2);
    kv_s_dim = kv_cache->stride(3);
    if (process_index) {
      effective_index_slot_mapping = index_slot_mapping.has_value()
                                         ? &index_slot_mapping.value()
                                         : &slot_mapping.value();
    }
  }
  // Optional contiguous gather targets: when given, the normed/roped q (and
  // index_q) are written here instead of in place, so callers avoid a separate
  // .contiguous() copy.  index_q_out only makes sense on the sparse path.
  if (q_out.has_value()) {
    STD_TORCH_CHECK(
        q_out->is_cuda() && q_out->is_contiguous() &&
            q_out->scalar_type() == qkv.scalar_type(),
        "q_out must be a contiguous CUDA tensor matching qkv dtype");
    STD_TORCH_CHECK(
        q_out->numel() == static_cast<int64_t>(num_tokens) * nq * kHeadDim,
        "q_out must have num_tokens * num_heads * 128 elements");
  }
  if (q_fp8_out.has_value()) {
    STD_TORCH_CHECK(q_fp8_out->is_cuda() && q_fp8_out->is_contiguous() &&
                        q_fp8_out->scalar_type() ==
                            torch::headeronly::ScalarType::Float8_e4m3fn,
                    "q_fp8_out must be a contiguous CUDA fp8 e4m3 tensor");
    STD_TORCH_CHECK(
        q_fp8_out->numel() == static_cast<int64_t>(num_tokens) * nq * kHeadDim,
        "q_fp8_out must have num_tokens * num_heads * 128 elements");
    STD_TORCH_CHECK(std::isfinite(q_fp8_scale) && q_fp8_scale > 0.0,
                    "q_fp8_scale must be finite and positive");
  }
  STD_TORCH_CHECK(write_qkv || (!q_out.has_value() && q_fp8_out.has_value()),
                  "write_qkv=False writes q only to q_fp8_out: pass q_fp8_out "
                  "and no q_out");
  if (index_q_out.has_value()) {
    STD_TORCH_CHECK(process_index,
                    "index_q_out requires index branch processing");
    STD_TORCH_CHECK(
        index_q_out->is_cuda() && index_q_out->is_contiguous() &&
            (index_q_out->scalar_type() == qkv.scalar_type() ||
             index_q_out->scalar_type() ==
                 torch::headeronly::ScalarType::Float8_e4m3fn),
        "index_q_out must be contiguous CUDA, qkv dtype or fp8 e4m3");
    STD_TORCH_CHECK(index_q_out->numel() ==
                        static_cast<int64_t>(num_tokens) * niq * kHeadDim,
                    "index_q_out must have num_tokens * num_index_heads * 128 "
                    "elements");
  }

  // fp8 index path: the index-K cache and index-Q outputs are e4m3 bytes while
  // q/k/v + q_out stay qkv dtype. Both index outputs must agree. The NVFP4
  // index path keeps e4m3 index-Q and stores index-K as packed NVFP4 pages.
  auto const kFp8 = torch::headeronly::ScalarType::Float8_e4m3fn;
  bool const nvfp4_idx =
      process_index && insert_kv &&
      index_cache->scalar_type() == torch::headeronly::ScalarType::Byte;
  bool const fp8_idx =
      process_index &&
      (nvfp4_idx ||
       (index_cache.has_value() && index_cache->scalar_type() == kFp8) ||
       (index_q_out.has_value() && index_q_out->scalar_type() == kFp8));
  int64_t idx_s_block = 0;
  if (nvfp4_idx) {
#ifdef USE_ROCM
    STD_TORCH_CHECK(false, "nvfp4 index_cache is not supported on ROCm");
#endif
    int64_t const page_width = kHeadDim / 2 + kHeadDim / 16;
    STD_TORCH_CHECK(
        index_cache->dim() == 3 && index_cache->size(1) == block_size &&
            index_cache->size(2) == page_width && index_cache->stride(2) == 1 &&
            index_cache->stride(1) == page_width &&
            index_cache->stride(0) >= block_size * page_width,
        "nvfp4 index_cache must be [nb, block_size, 72] uint8 with "
        "contiguous pages");
    idx_s_block = index_cache->stride(0);
  }
  if (fp8_idx) {
    STD_TORCH_CHECK(!index_cache.has_value() || nvfp4_idx ||
                        index_cache->scalar_type() == kFp8,
                    "fp8 index path: index_cache must be fp8 e4m3");
    STD_TORCH_CHECK(
        !index_q_out.has_value() || index_q_out->scalar_type() == kFp8,
        "fp8 index path: index_q_out must be fp8 e4m3");
  }

  STD_TORCH_CHECK(!nvfp4_kv || insert_kv,
                  "nvfp4 kv_cache_dtype requires kv_cache (insert mode)");
  float const* kv_k_scale_ptr =
      nvfp4_kv ? kv_k_scale->const_data_ptr<float>() : nullptr;
  float const* kv_v_scale_ptr =
      nvfp4_kv ? kv_v_scale->const_data_ptr<float>() : nullptr;

  // The row-group kernel indexes cache slots in 32 bits.
  bool const slots_fit_int32 =
      !insert_kv ||
      (kv_cache->size(0) * block_size <= INT32_MAX &&
       (!process_index || index_cache->size(0) * block_size <= INT32_MAX));

  const torch::stable::accelerator::DeviceGuard device_guard(
      qkv.get_device_index());
  auto stream = get_current_cuda_stream(qkv.get_device_index());

  if (nvfp4_kv) {
#ifndef USE_ROCM
    VLLM_STABLE_DISPATCH_HALF_TYPES(
        qkv.scalar_type(), "fused_minimax_m3_qknorm_rope_kv_insert_nvfp4", [&] {
          using st = scalar_t;
          CALL_FUSED_MINIMAX_M3_IMPL(uint8_t, vllm::Fp8KVCacheDataType::kAuto,
                                     true);
        });
#else
    STD_TORCH_CHECK(false, "nvfp4 kv_cache is not supported on ROCm");
#endif
    return;
  }

  VLLM_STABLE_DISPATCH_HALF_TYPES(
      qkv.scalar_type(), "fused_minimax_m3_qknorm_rope_kv_insert", [&] {
        using st = scalar_t;
        DISPATCH_BY_KV_CACHE_DTYPE(qkv.scalar_type(), kv_cache_dtype,
                                   CALL_FUSED_MINIMAX_M3);
      });
}

#undef CALL_FUSED_MINIMAX_M3
#undef CALL_FUSED_MINIMAX_M3_IMPL

// ────────────────────────────────────────────────────────────────────────────
// DSpark context K/V torch op
// ────────────────────────────────────────────────────────────────────────────
void fused_minimax_m3_dspark_context_kv(
    torch::stable::Tensor const& all_kv,         // bf16 [N, L, 2, nkv, 128]
    torch::stable::Tensor const& k_norm_weight,  // bf16 [L, 128]
    torch::stable::Tensor const& cos_sin_cache,  // bf16 [max_pos, rotary_dim]
    torch::stable::Tensor const& positions,      // [N] i64
    torch::stable::Tensor const& slot_mapping,   // [N] i64
    torch::stable::Tensor& kv_cache,  // layer 0's [nb, nkv, bs, 2*128]
    torch::stable::Tensor const& cache_offsets,  // [L] i64, elements
    torch::stable::Tensor const& k_scale,        // layer 0's [] f32 k scale
    torch::stable::Tensor const& scale_offsets,  // [2, L] i64, floats
    int64_t rotary_dim, double eps, const std::string& kv_cache_dtype) {
#ifdef USE_ROCM
  STD_TORCH_CHECK(false, "fused_minimax_m3_dspark_context_kv is CUDA-only");
#else
  using torch::headeronly::ScalarType;
  int const kHeadDim = vllm::minimax_m3_fused_ops::kHeadDim;
  STD_TORCH_CHECK(all_kv.is_cuda() && all_kv.dim() == 5 &&
                      all_kv.size(2) == 2 && all_kv.size(4) == kHeadDim &&
                      all_kv.stride(4) == 1 &&
                      all_kv.scalar_type() == ScalarType::BFloat16,
                  "all_kv must be bf16 CUDA [N, L, 2, nkv, 128], last dim "
                  "contiguous");
  for (int d = 0; d < 4; d++) {
    STD_TORCH_CHECK(all_kv.stride(d) % 8 == 0,
                    "all_kv strides must be multiples of 8 elements");
  }
  STD_TORCH_CHECK(reinterpret_cast<uintptr_t>(all_kv.data_ptr()) % 16 == 0,
                  "all_kv must be 16-byte aligned");
  int const num_tokens = static_cast<int>(all_kv.size(0));
  int const num_layers = static_cast<int>(all_kv.size(1));
  int const nkv = static_cast<int>(all_kv.size(3));
  STD_TORCH_CHECK(k_norm_weight.is_contiguous() &&
                      k_norm_weight.scalar_type() == ScalarType::BFloat16 &&
                      k_norm_weight.numel() == num_layers * kHeadDim,
                  "k_norm_weight must be contiguous bf16 [L, 128]");
  STD_TORCH_CHECK(cos_sin_cache.is_contiguous() &&
                      cos_sin_cache.scalar_type() == ScalarType::BFloat16 &&
                      cos_sin_cache.dim() == 2 &&
                      cos_sin_cache.size(1) == rotary_dim,
                  "cos_sin_cache must be contiguous bf16 [max_pos, "
                  "rotary_dim]");
  STD_TORCH_CHECK(rotary_dim == 16 || rotary_dim == 32 || rotary_dim == 64,
                  "rotary_dim must be 16, 32 or 64");
  for (auto const* t : {&positions, &slot_mapping}) {
    STD_TORCH_CHECK(t->is_cuda() && t->is_contiguous() &&
                        t->scalar_type() == ScalarType::Long &&
                        t->numel() == num_tokens,
                    "positions and slot_mapping must be contiguous int64 [N]");
  }
  STD_TORCH_CHECK(cache_offsets.is_cuda() && cache_offsets.is_contiguous() &&
                      cache_offsets.scalar_type() == ScalarType::Long &&
                      cache_offsets.numel() == num_layers,
                  "cache_offsets must be int64 [L]");
  STD_TORCH_CHECK(scale_offsets.is_cuda() && scale_offsets.is_contiguous() &&
                      scale_offsets.scalar_type() == ScalarType::Long &&
                      scale_offsets.numel() == 2 * num_layers,
                  "scale_offsets must be int64 [2, L]");
  STD_TORCH_CHECK(
      k_scale.is_cuda() && k_scale.scalar_type() == ScalarType::Float,
      "k_scale must be a float32 CUDA tensor");
  bool const fp8 = kv_cache_dtype == "fp8" || kv_cache_dtype == "fp8_e4m3";
  STD_TORCH_CHECK(fp8 || kv_cache_dtype == "auto",
                  "kv_cache_dtype must be auto, fp8 or fp8_e4m3");
  STD_TORCH_CHECK(fp8 ? kv_cache.scalar_type() == ScalarType::Byte
                      : kv_cache.scalar_type() == ScalarType::BFloat16,
                  "kv_cache must be uint8 (fp8) or bf16 (auto)");
  STD_TORCH_CHECK(
      kv_cache.dim() == 4 && kv_cache.size(1) == nkv &&
          kv_cache.size(3) == 2 * kHeadDim && kv_cache.stride(3) == 1 &&
          kv_cache.stride(0) % 16 == 0 && kv_cache.stride(1) % 16 == 0 &&
          kv_cache.stride(2) % 16 == 0,
      "kv_cache must be [nb, nkv, bs, 256] with a contiguous last dim");
  int64_t const block_size = kv_cache.size(2);
  STD_TORCH_CHECK(block_size > 0 && (block_size & (block_size - 1)) == 0,
                  "kv_cache block size must be a power of two");
  int block_shift = 0;
  while ((int64_t{1} << block_shift) < block_size) block_shift++;
  int64_t constexpr kIntMax = std::numeric_limits<int>::max();
  STD_TORCH_CHECK(all_kv.numel() / 8 <= kIntMax &&
                      kv_cache.stride(0) <= kIntMax &&
                      kv_cache.size(0) * block_size <= kIntMax,
                  "all_kv / kv_cache too large for 32-bit indexing");

  const torch::stable::accelerator::DeviceGuard device_guard(
      all_kv.get_device_index());
  auto stream = get_current_cuda_stream(all_kv.get_device_index());
  // all_kv strides in 16-byte vectors.
  auto launch = [&](auto fp8_tag) {
    constexpr bool kFp8 = decltype(fp8_tag)::value;
    vllm::minimax_m3_fused_ops::launchFusedMiniMaxM3DSparkContextKV<kFp8>(
        reinterpret_cast<uint4 const*>(all_kv.data_ptr()),
        reinterpret_cast<uint4 const*>(k_norm_weight.data_ptr()),
        reinterpret_cast<uint4 const*>(cos_sin_cache.data_ptr()),
        reinterpret_cast<int64_t const*>(positions.data_ptr()),
        reinterpret_cast<int64_t const*>(slot_mapping.data_ptr()),
        reinterpret_cast<uint8_t*>(kv_cache.data_ptr()),
        reinterpret_cast<int64_t const*>(cache_offsets.data_ptr()),
        reinterpret_cast<float const*>(k_scale.data_ptr()),
        reinterpret_cast<int64_t const*>(scale_offsets.data_ptr()),
        static_cast<float>(eps), static_cast<int>(rotary_dim), num_tokens,
        num_layers, nkv, block_shift, static_cast<int>(all_kv.stride(0) / 8),
        static_cast<int>(all_kv.stride(1) / 8),
        static_cast<int>(all_kv.stride(2) / 8),
        static_cast<int>(all_kv.stride(3) / 8),
        static_cast<int>(kv_cache.stride(0)),
        static_cast<int>(kv_cache.stride(1)),
        static_cast<int>(kv_cache.stride(2)), stream);
  };
  if (fp8) {
    launch(std::true_type{});
  } else {
    launch(std::false_type{});
  }
#endif
}
