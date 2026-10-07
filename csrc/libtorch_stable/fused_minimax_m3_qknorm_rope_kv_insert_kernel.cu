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

#include <cmath>
#include <cuda_runtime.h>
#include <type_traits>

#include "torch_utils.h"

#include "../cuda_compat.h"
#include "type_convert.cuh"
#include "../attention/dtype_fp8.cuh"
#include "dispatch_utils.h"

#ifdef USE_ROCM
  #include "../quantization/w8a8/fp8/amd/quant_utils.cuh"
#else
  #include "../quantization/w8a8/fp8/nvidia/quant_utils.cuh"
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
template <typename scalar_t>
__device__ __forceinline__ void storeScaledQElemsFp8(
    uint8_t* __restrict__ dst, float const (&elems)[kElemsPerLane],
    float const inv_scale) {
  using Converter = vllm::_typeConvert<scalar_t>;
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

// Quantize one K or V head row of one token into its NVFP4 page, where
// ``scale`` is the dequantization (global) scale. Every lane of the warp must
// call this (the scale-group reductions shuffle).
template <typename scalar_t>
__device__ __forceinline__ void storeNvfp4CacheElems(
    uint8_t* __restrict__ page, int const kv, int const head, int const token,
    int const block_size, int const laneId, float const (&elems)[kElemsPerLane],
    float const scale) {
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

  uint8_t sf_byte;
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
  packed = clearNegZeroNibbles(packed);

  uint8_t* const slot = page + static_cast<int64_t>(2 * head + kv) *
                                   block_size *
                                   (kNvfp4DataBytes + kNvfp4ScaleBytes);
  uint8_t* const data =
      slot + token * kNvfp4DataBytes + laneId * (kElemsPerLane / 2);
  *reinterpret_cast<uint16_t*>(data) = packed;

  if (laneId % kNvfp4LanesPerGroup == 0) {
    int const group = laneId / kNvfp4LanesPerGroup;
    int offset;
    if (kv == 0) {
      offset = token * kNvfp4ScaleBytes + group;
    } else {
      constexpr int kGroupsPerQuad = kNvfp4ScaleBytes / 4;
      int const swizzled_t = (token / 4) * 4 + group / kGroupsPerQuad;
      int const swizzled_s = (group % kGroupsPerQuad) * 4 + token % 4;
      offset = swizzled_t * kNvfp4ScaleBytes + swizzled_s;
    }
    slot[block_size * kNvfp4DataBytes + offset] = sf_byte;
  }
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
//     V : nkv  only if kInsertKV        (V-cache insert; no warps in dense)
//     IQ: niq  only if kProcessIndex    (norm+RoPE)
//     IK: 1    only if kProcessIndex    (norm+RoPE; +index-cache insert)
// cache_t/kv_dt: main attention KV-cache dtype (auto/fp8). out_idx_t/kFp8Idx:
// indexer index-K cache + index-Q output dtype (scalar_t or e4m3 byte).
// kHasIndex means the qkv row is laid out as sparse [q|k|v|index_q|index_k].
// kProcessIndex controls whether this launch actually norms/ropes the index
// branch and writes index_q/index_k outputs. Skip-index-topk reuse layers keep
// kHasIndex=true but set kProcessIndex=false.
template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt,
          bool kNvfp4, typename out_idx_t, bool kHasIndex, bool kInsertKV,
          bool kProcessIndex, bool kFp8Idx>
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

    // Slot layout (compile-time gated: dense has neither V nor index slots).
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

    // ── Cache inserts (sparse serving only). ───────────────────────────────
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
          if constexpr (kFp8Idx) {
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
    int64_t const kv_s_dim, bool const has_index, bool const insert_kv,
    bool const process_index, bool const fp8_idx, float const* kv_k_scale,
    float const* kv_v_scale, cudaStream_t stream) {
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

  #define LAUNCH(HAS_INDEX, INSERT, PROCESS_INDEX, FP8, OUT_T)              \
    cudaLaunchKernelEx(                                                     \
        &config,                                                            \
        fusedMiniMaxM3QNormRopeKVInsertKernel<scalar_t, cache_t, kv_dt,     \
                                              kNvfp4, OUT_T, HAS_INDEX,     \
                                              INSERT, PROCESS_INDEX, FP8>,  \
        qkv, q_out, q_fp8_out, reinterpret_cast<OUT_T*>(index_q_out),       \
        q_norm_w, k_norm_w, iq_norm_w, ik_norm_w, cos_sin_cache, positions, \
        slot_mapping, index_slot_mapping, kv_cache,                         \
        reinterpret_cast<OUT_T*>(index_cache), eps, q_fp8_inv_scale,        \
        rotary_dim, num_tokens, nq, nkv, niq, block_size, kv_s_block,       \
        kv_s_head, kv_s_token, kv_s_dim, kv_k_scale, kv_v_scale)
#else
  // ROCm: standard kernel launch syntax (no PDL/stream serialization).
  // clang-format off
  #define LAUNCH(HAS_INDEX, INSERT, PROCESS_INDEX, FP8, OUT_T)              \
    fusedMiniMaxM3QNormRopeKVInsertKernel<                                  \
        scalar_t, cache_t, kv_dt, kNvfp4, OUT_T, HAS_INDEX, INSERT,         \
        PROCESS_INDEX, FP8><<<grid, kBlockSize, 0, stream>>>(               \
        qkv, q_out, q_fp8_out, reinterpret_cast<OUT_T*>(index_q_out),       \
        q_norm_w, k_norm_w, iq_norm_w, ik_norm_w, cos_sin_cache, positions, \
        slot_mapping, index_slot_mapping, kv_cache,                         \
        reinterpret_cast<OUT_T*>(index_cache), eps, q_fp8_inv_scale,        \
        rotary_dim, num_tokens, nq, nkv, niq, block_size, kv_s_block,       \
        kv_s_head, kv_s_token, kv_s_dim, kv_k_scale, kv_v_scale)
  // clang-format on
#endif

  if constexpr (kNvfp4) {
    // NVFP4 main cache: sparse serving only (the host checks insert_kv).
    if (!process_index) {
      LAUNCH(true, true, false, false, scalar_t);
    } else if (fp8_idx) {
      LAUNCH(true, true, true, true, uint8_t);
    } else {
      LAUNCH(true, true, true, false, scalar_t);
    }
  } else if (has_index) {
    if (!process_index) {
      if (insert_kv) {
        LAUNCH(true, true, false, false, scalar_t);
      } else {
        LAUNCH(true, false, false, false, scalar_t);
      }
    } else if (insert_kv) {
      if (fp8_idx) {
        LAUNCH(true, true, true, true,
               uint8_t);  // sparse serving, fp8 index outputs
      } else {
        LAUNCH(true, true, true, false, scalar_t);  // sparse serving, bf16
      }
    } else {
      if (fp8_idx) {
        LAUNCH(true, false, true, true,
               uint8_t);  // sparse profiling, fp8 index_q
      } else {
        LAUNCH(true, false, true, false, scalar_t);  // sparse profiling, bf16
      }
    }
  } else {
    // Dense layer: never has an index branch and never inserts here (the
    // generic Attention layer owns the KV insert).
    LAUNCH(false, false, false, false, scalar_t);
  }
#undef LAUNCH
}

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
      kv_s_dim, has_index, insert_kv, process_index, fp8_idx, kv_k_scale_ptr,  \
      kv_v_scale_ptr, stream)
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
    std::optional<torch::stable::Tensor> kv_v_scale) {
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

  // Only the sparse layer inserts here (dense lets the generic Attention layer
  // own the KV write); there is no dense+insert kernel instantiation.
  STD_TORCH_CHECK(
      !insert_kv || has_index,
      "insert mode (kv_cache) requires the index branch (sparse layer)");
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
    // Indexer index-K cache: independent dtype -- qkv dtype or fp8 e4m3.
    if (process_index) {
      STD_TORCH_CHECK(
          index_cache.has_value() &&
              (index_cache->scalar_type() == qkv.scalar_type() ||
               index_cache->scalar_type() ==
                   torch::headeronly::ScalarType::Float8_e4m3fn),
          "insert mode requires index_cache matching qkv dtype or fp8 e4m3");
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
  // q/k/v + q_out stay qkv dtype. Both index outputs must agree.
  auto const kFp8 = torch::headeronly::ScalarType::Float8_e4m3fn;
  bool const fp8_idx =
      process_index &&
      ((index_cache.has_value() && index_cache->scalar_type() == kFp8) ||
       (index_q_out.has_value() && index_q_out->scalar_type() == kFp8));
  if (fp8_idx) {
    STD_TORCH_CHECK(
        !index_cache.has_value() || index_cache->scalar_type() == kFp8,
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
