/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 *
 * Fused Muse Spark attention pre-processing: weightless QK RMSNorm, GPT-J
 * interleaved RoPE (skipped on NoPE layers, ``cos_sin_cache == None``) and
 * the K/V insert of ``reshape_and_cache_flash``, over ``[q | k | v | o_gate]``
 * rows.
 *
 * A group of ``head_dim * sizeof(scalar_t) / 16`` lanes owns one head (8 lanes
 * for bf16 head_dim=64), so each lane moves one 16-byte vector, the rotated
 * pair (2i, 2i+1) is lane-local, and the norm reduction is 3 shuffles.
 *
 * Slot enumeration per token:
 *     [0, nq)                 Q heads  -> norm (+RoPE), in place or to
 *                                         ``q_fp8_out``
 *     [nq, nq+nkv)            K heads  -> norm (+RoPE), in place and into
 *                                         the K cache
 *     [nq+nkv, nq+2*nkv)      V heads  -> copied into the V cache (no norm)
 * V slots exist only when a KV cache is bound.
 *
 * Norm and rotation run in fp32 and round once at the store. Both fp8 paths
 * quantize after rounding to the model dtype, the same boundary
 * ``reshape_and_cache_flash`` and ``scaled_fp8_quant`` use.
 */

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <type_traits>

#include "torch_utils.h"

#include "core/cuda_compat.h"
#include "type_convert.cuh"
#include "core/vec_dtypes/dtype_fp8.cuh"
#include "dispatch_utils.h"

#ifndef USE_ROCM
  #include <cuda_fp8.h>
  #include "quantization/w8a8/fp8/nvidia/quant_utils.cuh"
#else
  #include <hip/hip_fp8.h>
  #include "quantization/w8a8/fp8/amd/quant_utils.cuh"
#endif

#ifndef FINAL_MASK
  #ifdef USE_ROCM
    #define FINAL_MASK 0xffffffffffffffffULL
  #else
    #define FINAL_MASK 0xffffffffu
  #endif
#endif

namespace vllm {
namespace muse_spark_fused_ops {

namespace {
inline int getSMVersion() {
  auto* props = get_device_prop();
  return props->major * 10 + props->minor;
}
inline int getSMCount() {
  auto* props = get_device_prop();
  return props->multiProcessorCount;
}
}  // namespace

// 16 bytes per lane: the widest vector access, and the reason a head is owned
// by a group of lanes rather than a whole warp.
constexpr int kBytesPerLane = 16;

#ifdef USE_ROCM
using WarpMask = uint64_t;
constexpr unsigned kWarpSize = 64;
#else
using WarpMask = unsigned;
constexpr unsigned kWarpSize = 32;
#endif

// Slot -> (token, head) needs a division by a value only known at launch. The
// kernel is issue-bound, and a 32-bit hardware division is ~20 instructions of
// it, so the reciprocal is computed once on the host instead.
struct FastDiv {
  uint32_t mul;
  uint32_t shr;
  int div;
};

inline FastDiv makeFastDiv(int denom) {
  // The reciprocal form needs denom >= 2; denom == 1 is carried as a flag the
  // device side selects on. Dividends stay below 2^31, which covers every slot
  // index this kernel can form.
  if (denom <= 1) return FastDiv{0u, 0u, 1};
  int lg = 31 - __builtin_clz(denom);
  if ((1 << lg) < denom) lg++;
  uint32_t const p = 31 + static_cast<uint32_t>(lg);
  uint64_t const m = ((1ull << p) + static_cast<uint64_t>(denom) - 1) /
                     static_cast<uint64_t>(denom);
  return FastDiv{static_cast<uint32_t>(m), p - 32, denom};
}

__device__ __forceinline__ void fastDivmod(int& quo, int& rem, int src,
                                           FastDiv const fd) {
  int const q =
      static_cast<int>(__umulhi(static_cast<uint32_t>(src), fd.mul) >> fd.shr);
  // A select, not a branch: fd.div is uniform across the launch.
  quo = (fd.div == 1) ? src : q;
  rem = src - quo * fd.div;
}

// Sum across the LANES contiguous lanes that share one head. The group is
// LANES-aligned inside the warp, so an XOR butterfly over masks < LANES never
// leaves it. `group_mask` names exactly the group: whole groups are uniformly
// active or inactive, so a warp whose other groups bailed out still syncs.
template <int LANES>
__device__ __forceinline__ float groupReduceSum(float val,
                                                WarpMask const group_mask) {
#pragma unroll
  for (int mask = LANES >> 1; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(group_mask, val, mask, 32);
  }
  return val;
}

// 16 contiguous bytes -> kElems fp32 registers, keeping the raw vector so a
// pass-through slot (V) can be re-stored without a conversion round trip.
template <typename scalar_t, int kElems>
__device__ __forceinline__ uint4 loadElems(scalar_t const* __restrict__ src,
                                           float (&elems)[kElems]) {
  using Converter = vllm::_typeConvert<scalar_t>;
  using T2 = typename Converter::packed_hip_type;
  uint4 raw = *reinterpret_cast<uint4 const*>(src);
  auto const* p = reinterpret_cast<T2 const*>(&raw);
#pragma unroll
  for (int i = 0; i < kElems / 2; i++) {
    float2 f2 = Converter::convert(p[i]);
    elems[2 * i] = f2.x;
    elems[2 * i + 1] = f2.y;
  }
  return raw;
}

// The lane's slice of the rope table: kPairs cos and kPairs sin values, each
// run contiguous, so each is one vector load (16 bytes for the fp32 table, 8
// for a model-dtype one) instead of kPairs scalar ones. rotary_dim being a
// multiple of 16 is what keeps both runs naturally aligned.
template <typename rope_t, int kPairs>
__device__ __forceinline__ void loadRopeVec(rope_t const* __restrict__ cos_ptr,
                                            rope_t const* __restrict__ sin_ptr,
                                            float (&c)[kPairs],
                                            float (&s)[kPairs]) {
  static_assert(kPairs == 4, "rope vector load assumes 4 pairs per lane");
  if constexpr (std::is_same_v<rope_t, float>) {
    float4 const cv = VLLM_LDG(reinterpret_cast<float4 const*>(cos_ptr));
    float4 const sv = VLLM_LDG(reinterpret_cast<float4 const*>(sin_ptr));
    c[0] = cv.x, c[1] = cv.y, c[2] = cv.z, c[3] = cv.w;
    s[0] = sv.x, s[1] = sv.y, s[2] = sv.z, s[3] = sv.w;
  } else {
    using Converter = vllm::_typeConvert<rope_t>;
    using T2 = typename Converter::packed_hip_type;
    uint2 const cv = VLLM_LDG(reinterpret_cast<uint2 const*>(cos_ptr));
    uint2 const sv = VLLM_LDG(reinterpret_cast<uint2 const*>(sin_ptr));
    auto const* cp = reinterpret_cast<T2 const*>(&cv);
    auto const* sp = reinterpret_cast<T2 const*>(&sv);
#pragma unroll
    for (int i = 0; i < kPairs / 2; i++) {
      float2 const cf = Converter::convert(cp[i]);
      float2 const sf = Converter::convert(sp[i]);
      c[2 * i] = cf.x, c[2 * i + 1] = cf.y;
      s[2 * i] = sf.x, s[2 * i + 1] = sf.y;
    }
  }
}

// kElems fp32 registers -> the 16-byte scalar_t vector they round to. Returned
// rather than stored so the fp8 cache path can reuse the *rounded* values,
// which is what reshape_and_cache_flash quantizes from.
template <typename scalar_t, int kElems>
__device__ __forceinline__ uint4 packElems(float const (&elems)[kElems]) {
  using Converter = vllm::_typeConvert<scalar_t>;
  using T2 = typename Converter::packed_hip_type;
  uint4 raw;
  auto* p = reinterpret_cast<T2*>(&raw);
#pragma unroll
  for (int i = 0; i < kElems / 2; i++) {
    p[i] = Converter::convert(make_float2(elems[2 * i], elems[2 * i + 1]));
  }
  return raw;
}

// Scatter one lane's 16 bytes into the KV cache. kAuto stores the model dtype
// verbatim; the fp8 dtypes quantize the value *after* it has been rounded to
// scalar_t, which is the boundary reshape_and_cache_flash uses.
//
// `inv_scale` is taken once per thread: per-element fp32 divisions would cost
// more than the rest of the kernel. The extra rounding is 2^-24 relative.
template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt,
          int kElems>
__device__ __forceinline__ void storeCache(cache_t* __restrict__ dst,
                                           uint4 const& packed,
                                           float const inv_scale) {
  if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
    *reinterpret_cast<uint4*>(dst) = packed;
  } else {
    using Converter = vllm::_typeConvert<scalar_t>;
    using HipT = typename Converter::hip_type;
    auto const* rounded = reinterpret_cast<HipT const*>(&packed);
#ifndef USE_ROCM
    constexpr __nv_fp8_interpretation_t kInterp =
        (kv_dt == Fp8KVCacheDataType::kFp8E4M3) ? __NV_E4M3 : __NV_E5M2;
    __nv_fp8x2_storage_t out[kElems / 2];
  #pragma unroll
    for (int i = 0; i < kElems / 2; i++) {
      float2 v =
          make_float2(Converter::convert(rounded[2 * i]) * inv_scale,
                      Converter::convert(rounded[2 * i + 1]) * inv_scale);
      out[i] = __nv_cvt_float2_to_fp8x2(v, __NV_SATFINITE, kInterp);
    }
    // kElems fp8 bytes: 8 for the 2-byte dtypes this kernel serves.
    static_assert(kElems == 8, "fp8 store assumes 8 elements per lane");
    *reinterpret_cast<uint2*>(dst) = *reinterpret_cast<uint2 const*>(out);
#else
  #pragma unroll
    for (int i = 0; i < kElems; i++) {
      float const v = Converter::convert(rounded[i]) * inv_scale;
      dst[i] = fp8::scaled_convert<cache_t, float, kv_dt>(v, 1.0f);
    }
#endif
  }
}

// Grid is 1D over (token, head-chunk) pairs. A chunk is HEADS_PER_GROUP
// consecutive head slots of one token, served by kLanes contiguous lanes; the
// whole chunk shares the token's position and its slice of the rope table, so
// the slot division, the position load and the cos/sin load are paid once per
// chunk. `slots_per_token % HEADS_PER_GROUP == 0` keeps a chunk inside one
// token. `rope_t` is the cos/sin cache dtype, independent of the qkv dtype.
template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt,
          typename rope_t, int HEAD_DIM, int HEADS_PER_GROUP, bool kDoRope,
          bool kInsertKV>
__global__
__launch_bounds__(256, 6) void fusedMuseSparkQKNormRopeKVInsertKernel(
    scalar_t* __restrict__ qkv,                // [N, qkv_stride] in/out
    rope_t const* __restrict__ cos_sin_cache,  // [max_pos, rotary_dim]
    int64_t const* __restrict__ positions,     // [N] i64
    int64_t const* __restrict__ slot_mapping,  // [N] i64 or nullptr
    cache_t* __restrict__ kv_cache,            // [nb, nkv, bs, 2*HEAD_DIM]
    float const* __restrict__ k_scale,         // [1] or nullptr
    float const* __restrict__ v_scale,         // [1] or nullptr
    uint8_t* __restrict__ q_fp8_out,           // [N, nq * HEAD_DIM] or nullptr
    float const* __restrict__ q_scale,         // [1] or nullptr
    float const eps, int const rotary_dim, int const num_tokens, int const nq,
    int const nkv, FastDiv const chunks_div, int64_t const qkv_stride,
    FastDiv const block_div, int64_t const kv_s_block, int64_t const kv_s_head,
    int64_t const kv_s_token) {
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
  // _typeConvert<BFloat16> is unavailable on pre-Ampere; this kernel only ever
  // runs with bf16/fp16 inputs. Discard the bf16 body there.
  if constexpr (std::is_same_v<scalar_t, c10::BFloat16>) {
    return;
  } else {
#endif
    constexpr int kElems = kBytesPerLane / sizeof(scalar_t);
    constexpr int kLanes = HEAD_DIM / kElems;
    constexpr int kPairs = kElems / 2;
    constexpr int kH = HEADS_PER_GROUP;
    static_assert(kLanes > 0 && kLanes <= 32 && (32 % kLanes) == 0,
                  "head_dim must give a power-of-two lane group <= 32");

    int const tid = blockIdx.x * blockDim.x + threadIdx.x;
    int const groupIdx = tid / kLanes;
    int const dimBase = (tid % kLanes) * kElems;

    int tokenIdx, chunk;
    fastDivmod(tokenIdx, chunk, groupIdx, chunks_div);
    int const firstSlot = chunk * kH;

    // Out-of-range groups must not exit before the reductions below: the XOR
    // butterfly needs every lane the mask names, and the mask covers the whole
    // group. They ride along and skip every memory access instead.
    bool const active = tokenIdx < num_tokens;

    constexpr WarpMask kGroupBits =
        (kLanes == 32) ? static_cast<WarpMask>(0xffffffffu)
                       : ((WarpMask{1} << kLanes) - WarpMask{1});
    unsigned const group_shift =
        (threadIdx.x & (kWarpSize - 1)) & ~static_cast<unsigned>(kLanes - 1);
    WarpMask const group_mask = kGroupBits << group_shift;

    // PDL: wait for the qkv-projection GEMM that produces `qkv`. No-op when
    // PDL is not enabled on the launch.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaGridDependencySynchronize();
#endif

    // Q, K and V are contiguous head slots at the head of the row, so the slot
    // index doubles as the head offset. The trailing o-gate is skipped by the
    // row stride alone.
    scalar_t* const base = qkv + static_cast<int64_t>(tokenIdx) * qkv_stride +
                           static_cast<int64_t>(firstSlot) * HEAD_DIM + dimBase;

    // Issue every load in the chunk before touching any of it, so the group has
    // kH loads in flight instead of one. An inactive group leaves these unset;
    // nothing downstream reads them, since every store is gated on `active`.
    uint4 raw[kH];
    if (active) {
#pragma unroll
      for (int h = 0; h < kH; h++) {
        raw[h] = *reinterpret_cast<uint4 const*>(base + h * HEAD_DIM);
      }
    }

    float cosv[kPairs], sinv[kPairs];
    bool const laneRotates = kDoRope && dimBase < rotary_dim;
    if constexpr (kDoRope) {
      if (active && laneRotates) {
        int64_t const pos = positions[tokenIdx];
        rope_t const* cache_ptr = cos_sin_cache + pos * rotary_dim;
        loadRopeVec<rope_t, kPairs>(cache_ptr + dimBase / 2,
                                    cache_ptr + rotary_dim / 2 + dimBase / 2,
                                    cosv, sinv);
      }
    }

    int64_t sm = -1;
    if constexpr (kInsertKV) {
      if (active) sm = slot_mapping[tokenIdx];
    }
    int cache_block = 0, cache_tok = 0;
    float k_inv = 1.0f, v_inv = 1.0f;
    float const q_inv = q_fp8_out ? 1.0f / *q_scale : 1.0f;
    if constexpr (kInsertKV) {
      if (sm >= 0) {
        fastDivmod(cache_block, cache_tok, static_cast<int>(sm), block_div);
      }
      if constexpr (kv_dt != Fp8KVCacheDataType::kAuto) {
        k_inv = 1.0f / *k_scale;
        v_inv = 1.0f / *v_scale;
      }
    }

#pragma unroll
    for (int h = 0; h < kH; h++) {
      int const slot = firstSlot + h;
      bool const isV = kInsertKV && slot >= nq + nkv;
      bool const isK = slot >= nq && slot < nq + nkv;
      bool const isQK = active && !isV;

      float elems[kElems];
      uint4 packed = raw[h];
      if (isQK) {
        using Converter = vllm::_typeConvert<scalar_t>;
        using T2 = typename Converter::packed_hip_type;
        auto const* p = reinterpret_cast<T2 const*>(&raw[h]);
#pragma unroll
        for (int i = 0; i < kElems / 2; i++) {
          float2 const f2 = Converter::convert(p[i]);
          elems[2 * i] = f2.x;
          elems[2 * i + 1] = f2.y;
        }

        float sumsq = 0.0f;
#pragma unroll
        for (int i = 0; i < kElems; i++) sumsq += elems[i] * elems[i];
        sumsq = groupReduceSum<kLanes>(sumsq, group_mask);
        float const rms_rcp =
            rsqrtf(sumsq / static_cast<float>(HEAD_DIM) + eps);
#pragma unroll
        for (int i = 0; i < kElems; i++) elems[i] *= rms_rcp;

        if constexpr (kDoRope) {
          if (laneRotates) {
#pragma unroll
            for (int i = 0; i < kPairs; i++) {
              float const v0 = elems[2 * i];
              float const v1 = elems[2 * i + 1];
              elems[2 * i] = v0 * cosv[i] - v1 * sinv[i];
              elems[2 * i + 1] = v0 * sinv[i] + v1 * cosv[i];
            }
          }
        }
        // Rounding to scalar_t happens exactly once, here, and both the
        // Q output and the fp8 cache quantization consume the same
        // rounded value. V passes through with the vector it was loaded with.
        packed = packElems<scalar_t, kElems>(elems);
        if (q_fp8_out && slot < nq) {
          int64_t const q_off =
              (static_cast<int64_t>(tokenIdx) * nq + slot) * HEAD_DIM + dimBase;
          storeCache<scalar_t, uint8_t, Fp8KVCacheDataType::kFp8E4M3, kElems>(
              q_fp8_out + q_off, packed, q_inv);
        } else {
          *reinterpret_cast<uint4*>(base + h * HEAD_DIM) = packed;
        }
      }

      if constexpr (kInsertKV) {
        if (sm >= 0 && (isK || isV)) {  // sm < 0 marks a padded token
          int const head = isK ? (slot - nq) : (slot - nq - nkv);
          int64_t const off = static_cast<int64_t>(cache_block) * kv_s_block +
                              head * kv_s_head +
                              static_cast<int64_t>(cache_tok) * kv_s_token +
                              (isK ? 0 : HEAD_DIM) + dimBase;
          storeCache<scalar_t, cache_t, kv_dt, kElems>(kv_cache + off, packed,
                                                       isK ? k_inv : v_inv);
        }
      }
    }

    // PDL: let a dependent successor launch early.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaTriggerProgrammaticLaunchCompletion();
#endif
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
  }
#endif
}

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt,
          typename rope_t, int HEAD_DIM, int HEADS_PER_GROUP>
void launchFusedMuseSparkTyped(
    scalar_t* qkv, rope_t const* cos_sin_cache, int64_t const* positions,
    int64_t const* slot_mapping, cache_t* kv_cache, float const* k_scale,
    float const* v_scale, uint8_t* q_fp8_out, float const* q_scale,
    float const eps, int const rotary_dim, int const num_tokens, int const nq,
    int const nkv, int64_t const qkv_stride, int const block_size,
    int64_t const kv_s_block, int64_t const kv_s_head, int64_t const kv_s_token,
    bool const do_rope, bool const insert_kv, cudaStream_t stream) {
  FastDiv const block_div = makeFastDiv(block_size);
  constexpr int kElems = kBytesPerLane / sizeof(scalar_t);
  constexpr int kLanes = HEAD_DIM / kElems;

  int const slots_per_token = nq + nkv + (insert_kv ? nkv : 0);
  int const chunks_per_token = slots_per_token / HEADS_PER_GROUP;
  FastDiv const chunks_div = makeFastDiv(chunks_per_token);
  int64_t const total_lanes =
      static_cast<int64_t>(num_tokens) * chunks_per_token * kLanes;
  if (total_lanes == 0) return;

  constexpr int kBlockSize = 256;
  int const grid =
      static_cast<int>((total_lanes + kBlockSize - 1) / kBlockSize);

#ifndef USE_ROCM
  // PDL wherever the hardware supports it (SM90+); a plain launch elsewhere.
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

  #define LAUNCH_MUSE(DO_ROPE, INSERT)                                         \
    cudaLaunchKernelEx(&config,                                                \
                       fusedMuseSparkQKNormRopeKVInsertKernel<                 \
                           scalar_t, cache_t, kv_dt, rope_t, HEAD_DIM,         \
                           HEADS_PER_GROUP, DO_ROPE, INSERT>,                  \
                       qkv, cos_sin_cache, positions, slot_mapping, kv_cache,  \
                       k_scale, v_scale, q_fp8_out, q_scale, eps, rotary_dim,  \
                       num_tokens, nq, nkv, chunks_div, qkv_stride, block_div, \
                       kv_s_block, kv_s_head, kv_s_token)
#else
  // clang-format off
  #define LAUNCH_MUSE(DO_ROPE, INSERT)                                        \
    fusedMuseSparkQKNormRopeKVInsertKernel<                                   \
        scalar_t, cache_t, kv_dt, rope_t, HEAD_DIM, HEADS_PER_GROUP, DO_ROPE, \
        INSERT><<<grid, kBlockSize, 0, stream>>>(                             \
        qkv, cos_sin_cache, positions, slot_mapping, kv_cache, k_scale,       \
        v_scale, q_fp8_out, q_scale, eps, rotary_dim, num_tokens, nq, nkv,    \
        chunks_div, qkv_stride, block_div, kv_s_block, kv_s_head, kv_s_token)
  // clang-format on
#endif

  if (do_rope) {
    if (insert_kv) {
      LAUNCH_MUSE(true, true);
    } else {
      LAUNCH_MUSE(true, false);
    }
  } else {
    if (insert_kv) {
      LAUNCH_MUSE(false, true);
    } else {
      LAUNCH_MUSE(false, false);
    }
  }
#undef LAUNCH_MUSE
}

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
void launchFusedMuseSpark(
    scalar_t* qkv, void const* cos_sin_cache, bool const rope_is_fp32,
    int64_t const* positions, int64_t const* slot_mapping, cache_t* kv_cache,
    float const* k_scale, float const* v_scale, uint8_t* q_fp8_out,
    float const* q_scale, float const eps, int const rotary_dim,
    int const num_tokens, int const nq, int const nkv, int const head_dim,
    int64_t const qkv_stride, int const block_size, int64_t const kv_s_block,
    int64_t const kv_s_head, int64_t const kv_s_token, bool const do_rope,
    bool const insert_kv, cudaStream_t stream) {
  // The group width must divide the slot count. Widen only while the grid
  // still covers a full wave; decode-sized batches keep the narrow schedule.
  int const slots_per_token = nq + nkv + (insert_kv ? nkv : 0);
  constexpr int kElemsSel = kBytesPerLane / sizeof(scalar_t);
  constexpr int kBlockSel = 256;
  int const lanes_per_head = head_dim / kElemsSel;
  int64_t const full_wave =
      static_cast<int64_t>(getSMCount()) * (2048 / kBlockSel);
  int heads_per_group = 1;
  for (int h : {4, 2}) {
    if (slots_per_token % h != 0) continue;
    int64_t const blocks = (static_cast<int64_t>(num_tokens) *
                                (slots_per_token / h) * lanes_per_head +
                            kBlockSel - 1) /
                           kBlockSel;
    if (blocks >= full_wave) {
      heads_per_group = h;
      break;
    }
  }

  // TP4/TP8 amortize head addressing before filling a full CTA wave. RoPE
  // benefits earlier; FP8 conversion keeps wider groups expensive for longer.
  if (insert_kv && head_dim == 64 &&
      ((nq == 16 && nkv == 2) || (nq == 32 && nkv == 4))) {
    int64_t const query_heads = static_cast<int64_t>(num_tokens) * nq;
    if (do_rope && query_heads >= 24576) {
      heads_per_group = std::max(heads_per_group, 2);
      if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
        if (query_heads >= 49152) heads_per_group = 4;
      }
    } else if (!do_rope && query_heads >= 32768) {
      if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
        heads_per_group = std::max(heads_per_group, 2);
      }
    }
  }

#define DISPATCH_ARGS(RT, HD, HPG)                                         \
  launchFusedMuseSparkTyped<scalar_t, cache_t, kv_dt, RT, HD, HPG>(        \
      qkv, reinterpret_cast<RT const*>(cos_sin_cache), positions,          \
      slot_mapping, kv_cache, k_scale, v_scale, q_fp8_out, q_scale, eps,   \
      rotary_dim, num_tokens, nq, nkv, qkv_stride, block_size, kv_s_block, \
      kv_s_head, kv_s_token, do_rope, insert_kv, stream)

#define DISPATCH_HPG(RT, HD)    \
  switch (heads_per_group) {    \
    case 4:                     \
      DISPATCH_ARGS(RT, HD, 4); \
      break;                    \
    case 2:                     \
      DISPATCH_ARGS(RT, HD, 2); \
      break;                    \
    default:                    \
      DISPATCH_ARGS(RT, HD, 1); \
      break;                    \
  }

  // A NoPE launch never reads the table, so it can share the fp32 branch.
#define DISPATCH_ROPE_T(HD)       \
  if (!do_rope || rope_is_fp32) { \
    DISPATCH_HPG(float, HD);      \
  } else {                        \
    DISPATCH_HPG(scalar_t, HD);   \
  }

  // Which head_dim arm this translation unit instantiates. 0 (the default)
  // keeps all of them; the build compiles this file once per lane instead, each
  // with FUSED_MUSE_SPARK_LANE = 64/128/256, so the device code of one head_dim
  // is generated per unit instead of all three landing on one core.
#ifndef FUSED_MUSE_SPARK_LANE
  #define FUSED_MUSE_SPARK_LANE 0
#endif

  switch (head_dim) {
    case 64:
      if constexpr (FUSED_MUSE_SPARK_LANE == 0 || FUSED_MUSE_SPARK_LANE == 64) {
        DISPATCH_ROPE_T(64);
      } else {
        STD_TORCH_CHECK(false, "head_dim 64 not compiled");
      }
      break;
    case 128:
      if constexpr (FUSED_MUSE_SPARK_LANE == 0 ||
                    FUSED_MUSE_SPARK_LANE == 128) {
        DISPATCH_ROPE_T(128);
      } else {
        STD_TORCH_CHECK(false, "head_dim 128 not compiled");
      }
      break;
    case 256:
      if constexpr (FUSED_MUSE_SPARK_LANE == 0 ||
                    FUSED_MUSE_SPARK_LANE == 256) {
        DISPATCH_ROPE_T(256);
      } else {
        STD_TORCH_CHECK(false, "head_dim 256 not compiled");
      }
      break;
    default:
      STD_TORCH_CHECK(false, "unsupported head_dim: ", head_dim);
  }
#undef DISPATCH_ROPE_T
#undef DISPATCH_HPG
#undef DISPATCH_ARGS
}

}  // namespace muse_spark_fused_ops
}  // namespace vllm

// clang-format off
#define CALL_FUSED_MUSE_SPARK(_RAW_T, CACHE_T, KV_DTYPE)                       \
  vllm::muse_spark_fused_ops::launchFusedMuseSpark<st, CACHE_T, KV_DTYPE>(     \
      reinterpret_cast<st*>(qkv.data_ptr()),                                   \
      do_rope ? cos_sin_cache->data_ptr() : nullptr, rope_is_fp32,             \
      reinterpret_cast<int64_t const*>(positions.data_ptr()),                  \
      insert_kv ? reinterpret_cast<int64_t const*>(slot_mapping->data_ptr())   \
                : nullptr,                                                     \
      insert_kv ? reinterpret_cast<CACHE_T*>(kv_cache->data_ptr()) : nullptr,  \
      needs_scales ? reinterpret_cast<float const*>(k_scale->data_ptr())       \
                   : nullptr,                                                  \
      needs_scales ? reinterpret_cast<float const*>(v_scale->data_ptr())       \
                   : nullptr,                                                  \
      q_fp8_out ? reinterpret_cast<uint8_t*>(q_fp8_out->data_ptr()) : nullptr, \
      q_scale ? reinterpret_cast<float const*>(q_scale->data_ptr()) : nullptr, \
      static_cast<float>(eps), static_cast<int>(rotary_dim), num_tokens, nq,   \
      nkv, hd, qkv_stride, static_cast<int>(block_size), kv_s_block,           \
      kv_s_head, kv_s_token, do_rope, insert_kv, stream)
// clang-format on

void fused_muse_spark_qknorm_rope_kv_insert(
    torch::stable::Tensor& qkv,              // [N, (nq + 2*nkv + gate)*hd]
    torch::stable::Tensor const& positions,  // [N] i64
    int64_t num_heads, int64_t num_kv_heads, int64_t head_dim, double eps,
    std::optional<torch::stable::Tensor> cos_sin_cache,  // None => NoPE layer
    int64_t rotary_dim,
    std::optional<torch::stable::Tensor> slot_mapping,  // [N] i64
    std::optional<torch::stable::Tensor> kv_cache,      // [nb, nkv, bs, 2*hd]
    int64_t block_size, const std::string& kv_cache_dtype,
    std::optional<torch::stable::Tensor> k_scale,  // [1] fp32, fp8 cache only
    std::optional<torch::stable::Tensor> v_scale,
    std::optional<torch::stable::Tensor> q_fp8_out,
    std::optional<torch::stable::Tensor> q_scale) {
  STD_TORCH_CHECK(qkv.is_cuda() && qkv.dim() == 2 && qkv.stride(1) == 1,
                  "qkv must be a 2-D CUDA tensor with a contiguous row");
  STD_TORCH_CHECK(
      qkv.scalar_type() == torch::headeronly::ScalarType::Half ||
          qkv.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "qkv must be float16 or bfloat16");
  STD_TORCH_CHECK(
      positions.is_cuda() &&
          positions.scalar_type() == torch::headeronly::ScalarType::Long,
      "positions must be int64 CUDA");

  int const num_tokens = static_cast<int>(qkv.size(0));
  int const nq = static_cast<int>(num_heads);
  int const nkv = static_cast<int>(num_kv_heads);
  int const hd = static_cast<int>(head_dim);
  int64_t const qkv_stride = qkv.stride(0);

  STD_TORCH_CHECK(q_fp8_out.has_value() == q_scale.has_value(),
                  "q_fp8_out and q_scale must be provided together");
  if (q_fp8_out) {
    STD_TORCH_CHECK(
        q_fp8_out->is_cuda() &&
            q_fp8_out->get_device_index() == qkv.get_device_index() &&
            q_fp8_out->scalar_type() ==
                torch::headeronly::ScalarType::Float8_e4m3fn &&
            q_fp8_out->dim() == 2 && q_fp8_out->is_contiguous() &&
            q_fp8_out->size(0) == num_tokens &&
            q_fp8_out->size(1) == static_cast<int64_t>(nq) * hd,
        "q_fp8_out must be contiguous E4M3 [num_tokens, num_heads * head_dim] "
        "on the qkv device");
    STD_TORCH_CHECK(
        q_scale->is_cuda() &&
            q_scale->get_device_index() == qkv.get_device_index() &&
            q_scale->scalar_type() == torch::headeronly::ScalarType::Float &&
            q_scale->numel() == 1,
        "q_scale must be a single fp32 value on the qkv device");
  }

  STD_TORCH_CHECK(hd == 64 || hd == 128 || hd == 256,
                  "head_dim must be 64, 128 or 256");
  STD_TORCH_CHECK(qkv.size(1) >= static_cast<int64_t>(nq + 2 * nkv) * hd,
                  "qkv row must hold at least [q|k|v]");
  // A lane moves 16 bytes, so every row start and every head start must land
  // on a 16-byte boundary. Every supported head_dim is a multiple of 8 at 2
  // bytes, so heads are fine; the row stride is the one that can break it.
  STD_TORCH_CHECK(qkv_stride % 8 == 0,
                  "qkv row stride must be a multiple of 8 elements, got ",
                  qkv_stride);

  bool const do_rope = cos_sin_cache.has_value();
  bool rope_is_fp32 = true;
  if (do_rope) {
    STD_TORCH_CHECK(cos_sin_cache->is_cuda() && cos_sin_cache->dim() == 2 &&
                        cos_sin_cache->is_contiguous(),
                    "cos_sin_cache must be a contiguous 2-D CUDA tensor");
    STD_TORCH_CHECK(cos_sin_cache->size(1) == rotary_dim,
                    "cos_sin_cache shape must be [max_pos, rotary_dim]");
    STD_TORCH_CHECK(
        cos_sin_cache->scalar_type() == torch::headeronly::ScalarType::Float ||
            cos_sin_cache->scalar_type() == qkv.scalar_type(),
        "cos_sin_cache must be float32 or match the qkv dtype");
    rope_is_fp32 =
        cos_sin_cache->scalar_type() == torch::headeronly::ScalarType::Float;
    // A lane owns 16 bytes = 8 elements of the head and must be wholly inside
    // or wholly outside the rotated prefix.
    STD_TORCH_CHECK(rotary_dim > 0 && rotary_dim <= hd && rotary_dim % 16 == 0,
                    "rotary_dim must be a positive multiple of 16 and <= "
                    "head_dim");
  }

  vllm::Fp8KVCacheDataType const kv_dt =
      vllm::get_fp8_kv_cache_data_type(kv_cache_dtype);
  bool const insert_kv = kv_cache.has_value() && kv_cache->numel() > 0;
  bool const needs_scales =
      insert_kv && kv_dt != vllm::Fp8KVCacheDataType::kAuto;

  int64_t kv_s_block = 0, kv_s_head = 0, kv_s_token = 0;
  if (insert_kv) {
    STD_TORCH_CHECK(
        slot_mapping.has_value() && slot_mapping->is_cuda() &&
            slot_mapping->scalar_type() == torch::headeronly::ScalarType::Long,
        "insert mode requires an int64 CUDA slot_mapping");
    STD_TORCH_CHECK(slot_mapping->numel() >= num_tokens,
                    "slot_mapping must cover every token");
    STD_TORCH_CHECK(kv_cache->dim() == 4 && kv_cache->stride(3) == 1,
                    "kv_cache must be [nb, nkv, bs, 2*head_dim] with a "
                    "contiguous content dim");
    STD_TORCH_CHECK(kv_cache->size(3) == 2 * hd,
                    "kv_cache content dim must be 2*head_dim");
    STD_TORCH_CHECK(kv_cache->size(1) == nkv,
                    "kv_cache head dim must equal num_kv_heads");
    STD_TORCH_CHECK(kv_cache->size(2) == block_size,
                    "kv_cache token dim must equal block_size");
    if (kv_dt == vllm::Fp8KVCacheDataType::kAuto) {
      STD_TORCH_CHECK(kv_cache->scalar_type() == qkv.scalar_type(),
                      "auto kv_cache dtype must match qkv");
    } else {
      STD_TORCH_CHECK(
          kv_cache->scalar_type() == torch::headeronly::ScalarType::Byte ||
              kv_cache->scalar_type() ==
                  torch::headeronly::ScalarType::Float8_e4m3fn,
          "fp8 kv_cache must use uint8 or float8_e4m3fn storage");
      STD_TORCH_CHECK(k_scale.has_value() && v_scale.has_value(),
                      "fp8 kv cache requires k_scale and v_scale");
      STD_TORCH_CHECK(
          k_scale->numel() == 1 && v_scale->numel() == 1 &&
              k_scale->scalar_type() == torch::headeronly::ScalarType::Float &&
              v_scale->scalar_type() == torch::headeronly::ScalarType::Float,
          "k_scale and v_scale must be single fp32 values");
    }
    kv_s_block = kv_cache->stride(0);
    kv_s_head = kv_cache->stride(1);
    kv_s_token = kv_cache->stride(2);
  }

  const torch::stable::accelerator::DeviceGuard device_guard(
      qkv.get_device_index());
  auto stream = get_current_cuda_stream(qkv.get_device_index());

  VLLM_STABLE_DISPATCH_HALF_TYPES(
      qkv.scalar_type(), "fused_muse_spark_qknorm_rope_kv_insert", [&] {
        using st = scalar_t;
        DISPATCH_BY_KV_CACHE_DTYPE(qkv.scalar_type(), kv_cache_dtype,
                                   CALL_FUSED_MUSE_SPARK);
      });
}

#undef CALL_FUSED_MUSE_SPARK
