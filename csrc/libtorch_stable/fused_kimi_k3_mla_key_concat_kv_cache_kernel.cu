/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 *
 * Fused Kimi-K3 MLA prefill + decode epilogues with optional RoPE.
 *
 * Prefill: runs after the q_b_proj / kv_b_proj GEMMs, one launch per token
 * slice. Decode: runs after BMM1 (q_nope x W_UK) right before forward_mqa,
 * concatenating mqa_q = [ql_nope | q_pe] and inserting the latent cache
 * (fused_kimi_k3_mla_decode_q_concat_kv_cache_{,fp8_,ds_mla_}insert).
 *
 * Prefill variants:
 *
 *   bf16 (fused_kimi_k3_mla_key_concat_kv_cache_insert):
 *     - optional in-place q RoPE: rotate q[t, h, 128:192]
 *     - full key concat:    k_out[t, h] = [k_nope[t, h] | k_pe[t]]  (per head)
 *     - latent cache insert: cache[slot(t)] = [kv_c_normed[t] | k_pe[t]]
 *     (v is used as-is in bf16, so it is not touched here.)
 *
 *   fp8 (fused_kimi_k3_mla_qkv_quant_kv_cache_fp8_insert):
 *     - q_fp8[t, h]  = quant(q[t, h],                q_scale)
 *     - k_fp8[t, h]  = quant([k_nope[t, h] | k_pe[t]], k_scale)
 *     - v_fp8[t, h]  = quant(v[t, h],                v_scale)
 *     - cache[slot(t)] = quant([kv_c_normed[t] | k_pe[t]], k_scale)
 *     matching MLA's _q_scale / _k_scale / _v_scale (the cache latent uses a
 *     separate cache scale). Per-tensor E4M3.
 *
 *   fp8_ds_mla (fused_kimi_k3_mla_key_concat_ds_mla_insert):
 *     - full key concat (bf16), and cache insert in DeepSeek's 656-byte
 *       block-scaled layout (NoPE fp8 in 4 tiles of 128 with per-tile dynamic
 *       scales, RoPE bf16), bit-compatible with concat_and_cache_ds_mla_kernel.
 *
 * Both use Programmatic Dependent Launch (PDL) to overlap the tail of the
 * producing GEMMs on sm_90+, and are structured after
 * `fusedDeepseekV4FullCacheKernel`: one grid, one warp per (token, slot) with
 * `slotsPerToken = num_heads + 1`. Slots [0, H) do the per-head work; the extra
 * slot H does the per-token cache insert. All dims are multiples of 8, so bf16
 * copies move one uint4 (8 elems) per step and fp8 stores pack 8 elems into a
 * uint2.
 */

#include "torch_utils.h"

#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/csrc/stable/device.h>

#include "cuda_compat.h"
#include "dispatch_utils.h"
#include "type_convert.cuh"

#ifndef USE_ROCM
  #include <cuda_fp8.h>
  #include "../quantization/w8a8/fp8/nvidia/quant_utils.cuh"
#else
  #include <hip/hip_fp8.h>
  #include "../quantization/w8a8/fp8/amd/quant_utils.cuh"
#endif
#include <cuda_runtime.h>
#include <cfloat>
#include <type_traits>

#ifdef USE_ROCM
__device__ __forceinline__ uint8_t rocm_cvt_float_to_fp8_e4m3(float val) {
  #if defined(__gfx942__)
  __hip_fp8_e4m3_fnuz fp8_val(val);
  #else
  __hip_fp8_e4m3 fp8_val(val);
  #endif
  return reinterpret_cast<uint8_t&>(fp8_val);
}
#endif

namespace vllm {
namespace kimi_k3_fused_ops {

namespace {
inline int getSMVersion() {
  auto* props = get_device_prop();
  return props->major * 10 + props->minor;
}
}  // namespace

// ────────────────────────────────────────────────────────────────────────────
// Constants (Kimi-K3 MLA)
// ────────────────────────────────────────────────────────────────────────────
constexpr int kKvLoraRank = 512;                             // L
constexpr int kQkNopeHeadDim = 128;                          // P
constexpr int kQkRopeHeadDim = 64;                           // R
constexpr int kQkHeadDim = kQkNopeHeadDim + kQkRopeHeadDim;  // 192
constexpr int kVHeadDim = 128;                               // V
constexpr int kCacheEntry = kKvLoraRank + kQkRopeHeadDim;    // 576
constexpr int kVecElems = 8;  // 8 bf16 == one uint4 load / one uint2 fp8 store

#if defined(USE_ROCM) && defined(__gfx942__)
constexpr float kFp8Max = 224.0f;
#else
constexpr float kFp8Max = 448.0f;
#endif
// Divisor for fp8_ds_mla per-tile dynamic scales (matches cache_kernels.cu).
// fp8_ds_mla 656B entry: [0,512) NoPE fp8 (4 tiles of 128), [512,528) 4 fp32
// tile scales, [528,656) RoPE 64 bf16.
constexpr float kFp8ScaleDivisor = kFp8Max;

// Copy 8 source elements (one uint4 of bf16/fp16) to `dst`. FP8=false stores a
// uint4 (bf16); FP8=true decodes to fp32, scales by `scale_inv`, saturates to
// ±kFp8Max and packs into a uint2 of E4M3.
template <typename scalar_t, bool FP8, bool APPLY_ROPE = false>
__device__ __forceinline__ void copyChunk8(void* dst, const scalar_t* src,
                                           float scale_inv,
                                           const float* cos_sin = nullptr,
                                           int rope_elem_base = 0) {
  uint4 const v = *reinterpret_cast<const uint4*>(src);
  if constexpr (FP8 || APPLY_ROPE) {
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
    // _typeConvert<BFloat16> is unavailable on pre-Ampere. Kimi K3 uses
    // bf16 inputs, so discard unsupported conversion paths in those builds.
    if constexpr (std::is_same_v<scalar_t, c10::BFloat16>) {
      return;
    } else {
#endif
      using Converter = vllm::_typeConvert<scalar_t>;
      auto const* p =
          reinterpret_cast<typename Converter::packed_hip_type const*>(&v);
      float f[kVecElems];
#pragma unroll
      for (int i = 0; i < 4; i++) {
        float2 x = Converter::convert(p[i]);
        f[2 * i] = x.x;
        f[2 * i + 1] = x.y;
      }
      if constexpr (APPLY_ROPE) {
#pragma unroll
        for (int i = 0; i < kVecElems / 2; i++) {
          int const pair_idx = rope_elem_base / 2 + i;
          float const cos = static_cast<float>(cos_sin[pair_idx]);
          float const sin =
              static_cast<float>(cos_sin[pair_idx + kQkRopeHeadDim / 2]);
          float const x = f[2 * i];
          float const y = f[2 * i + 1];
          f[2 * i] = x * cos - y * sin;
          f[2 * i + 1] = x * sin + y * cos;
        }
      }
      if constexpr (!FP8) {
        uint4 out;
        auto* o = reinterpret_cast<typename Converter::packed_hip_type*>(&out);
#pragma unroll
        for (int i = 0; i < kVecElems / 2; i++) {
          o[i] = Converter::convert(make_float2(f[2 * i], f[2 * i + 1]));
        }
        *reinterpret_cast<uint4*>(dst) = out;
        return;
      }
#ifndef USE_ROCM
      uint2 out;
      auto* o2 = reinterpret_cast<__nv_fp8x2_storage_t*>(&out);
  #pragma unroll
      for (int i = 0; i < 4; i++) {
        float2 s = make_float2(f[2 * i] * scale_inv, f[2 * i + 1] * scale_inv);
        s.x = fminf(fmaxf(s.x, -kFp8Max), kFp8Max);
        s.y = fminf(fmaxf(s.y, -kFp8Max), kFp8Max);
        o2[i] = __nv_cvt_float2_to_fp8x2(s, __NV_SATFINITE, __NV_E4M3);
      }
      *reinterpret_cast<uint2*>(dst) = out;
#else
    uint8_t out[kVecElems];
  #pragma unroll
    for (int i = 0; i < kVecElems; i++) {
      float s = fminf(fmaxf(f[i] * scale_inv, -kFp8Max), kFp8Max);
      out[i] = rocm_cvt_float_to_fp8_e4m3(s);
    }
    *reinterpret_cast<uint2*>(dst) = *reinterpret_cast<uint2 const*>(out);
#endif
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
    }
#endif
  } else {
    *reinterpret_cast<uint4*>(dst) = v;
  }
}

// Concat + store one head's full key: dst[e] = [k_nope | k_pe], e in [0, 192).
// FP8 dst is byte-addressed; bf16 dst is scalar_t-addressed (dst_elem_size).
template <typename scalar_t, bool FP8, bool APPLY_ROPE = false>
__device__ __forceinline__ void writeFullKey(void* dst, const scalar_t* k_nope,
                                             const scalar_t* k_pe, int laneId,
                                             int dst_elem_size, float scale_inv,
                                             const float* cos_sin = nullptr) {
  auto* d = reinterpret_cast<uint8_t*>(dst);
  for (int e = laneId * kVecElems; e < kQkHeadDim; e += 32 * kVecElems) {
    if (e < kQkNopeHeadDim) {
      copyChunk8<scalar_t, FP8>(d + e * dst_elem_size, k_nope + e, scale_inv);
    } else {
      int const rope_e = e - kQkNopeHeadDim;
      copyChunk8<scalar_t, FP8, APPLY_ROPE>(
          d + e * dst_elem_size, k_pe + rope_e, scale_inv, cos_sin, rope_e);
    }
  }
}

// Store a prefill query, rotating only q[..., 128:192]. For bf16 dst may alias
// q (in-place); fp8 writes the quantized query directly to its output.
template <typename scalar_t, bool FP8, bool APPLY_ROPE = false>
__device__ __forceinline__ void writePrefillQuery(
    void* dst, const scalar_t* q, int laneId, int dst_elem_size,
    float scale_inv, const float* cos_sin = nullptr) {
  auto* d = reinterpret_cast<uint8_t*>(dst);
  if constexpr (FP8) {
    for (int e = laneId * kVecElems; e < kQkHeadDim; e += 32 * kVecElems) {
      if (e < kQkNopeHeadDim) {
        copyChunk8<scalar_t, true>(d + e * dst_elem_size, q + e, scale_inv);
      } else {
        int const rope_e = e - kQkNopeHeadDim;
        copyChunk8<scalar_t, true, APPLY_ROPE>(d + e * dst_elem_size, q + e,
                                               scale_inv, cos_sin, rope_e);
      }
    }
  } else if constexpr (APPLY_ROPE) {
    for (int e = laneId * kVecElems; e < kQkRopeHeadDim; e += 32 * kVecElems) {
      copyChunk8<scalar_t, false, true>(
          d + (kQkNopeHeadDim + e) * dst_elem_size, q + kQkNopeHeadDim + e,
          scale_inv, cos_sin, e);
    }
  }
}

// Concat + store a 576-wide latent: dst[e] = [a512 | b64], e in [0, 576). Used
// for the decode query mqa_q = [ql_nope | q_pe] and the plain latent cache
// entry [kv_c | k_pe]. FP8 packs to E4M3 (dst_elem_size 1); bf16 stores uint4.
template <typename scalar_t, bool FP8, bool APPLY_ROPE = false>
__device__ __forceinline__ void writeLatent576(void* dst, const scalar_t* a512,
                                               const scalar_t* b64, int laneId,
                                               int dst_elem_size,
                                               float scale_inv,
                                               const float* cos_sin = nullptr) {
  auto* d = reinterpret_cast<uint8_t*>(dst);
  for (int e = laneId * kVecElems; e < kCacheEntry; e += 32 * kVecElems) {
    if (e < kKvLoraRank) {
      copyChunk8<scalar_t, FP8>(d + e * dst_elem_size, a512 + e, scale_inv);
    } else {
      int const rope_e = e - kKvLoraRank;
      copyChunk8<scalar_t, FP8, APPLY_ROPE>(d + e * dst_elem_size, b64 + rope_e,
                                            scale_inv, cos_sin, rope_e);
    }
  }
}

// Write [kv_c | k_pe] into the fp8_ds_mla 656B entry using one warp: NoPE 512
// as fp8 in 4 tiles of 128 (per-tile dynamic absmax scale, 4 fp32 scales at
// [512,528)), RoPE 64 as bf16 at [528,656). Bit-compatible with
// concat_and_cache_ds_mla_kernel.
template <typename scalar_t, bool APPLY_ROPE = false>
__device__ __forceinline__ void writeDsMlaCache(
    uint8_t* row, const scalar_t* kvc, const scalar_t* pe, int laneId,
    const float* cos_sin = nullptr) {
  constexpr int kElemsPerLane = kKvLoraRank / 32;  // 16
  int const tile = laneId >> 3;                    // 8 lanes per tile
  scalar_t vals[kElemsPerLane];
  *reinterpret_cast<uint4*>(vals) =
      *reinterpret_cast<const uint4*>(kvc + laneId * kElemsPerLane);
  *reinterpret_cast<uint4*>(vals + 8) =
      *reinterpret_cast<const uint4*>(kvc + laneId * kElemsPerLane + 8);

  float max_abs = 0.0f;
#pragma unroll
  for (int i = 0; i < kElemsPerLane; i++) {
    max_abs = fmaxf(max_abs, fabsf(static_cast<float>(vals[i])));
  }
#pragma unroll
  for (int offset = 4; offset > 0; offset /= 2) {
    max_abs = fmaxf(max_abs, VLLM_SHFL_XOR_SYNC_WIDTH(max_abs, offset, 8));
  }
  float const tile_scale = fmaxf(max_abs / kFp8ScaleDivisor, FLT_MIN);
  if ((laneId & 7) == 0) {
    reinterpret_cast<float*>(row)[kKvLoraRank / 4 + tile] = tile_scale;
  }
  uint8_t res[kElemsPerLane];
#pragma unroll
  for (int i = 0; i < kElemsPerLane; i++) {
    res[i] =
        fp8::scaled_convert<uint8_t, scalar_t, Fp8KVCacheDataType::kFp8E4M3>(
            vals[i], tile_scale);
  }
  *reinterpret_cast<uint4*>(row + laneId * kElemsPerLane) =
      *reinterpret_cast<const uint4*>(res);
  scalar_t* row16 = reinterpret_cast<scalar_t*>(row);
  scalar_t* rope_dst = row16 + kKvLoraRank / 2 + 8 + laneId * 2;
  if constexpr (APPLY_ROPE) {
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
    if constexpr (std::is_same_v<scalar_t, c10::BFloat16>) {
      return;
    } else {
#endif
      using Converter = vllm::_typeConvert<scalar_t>;
      using packed_t = typename Converter::packed_hip_type;
      packed_t const src = *reinterpret_cast<packed_t const*>(pe + laneId * 2);
      float2 const xy = Converter::convert(src);
      float const cos = static_cast<float>(cos_sin[laneId]);
      float const sin =
          static_cast<float>(cos_sin[laneId + kQkRopeHeadDim / 2]);
      *reinterpret_cast<packed_t*>(rope_dst) = Converter::convert(
          make_float2(xy.x * cos - xy.y * sin, xy.x * sin + xy.y * cos));
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
    }
#endif
  } else {
    *reinterpret_cast<int32_t*>(rope_dst) =
        *reinterpret_cast<const int32_t*>(pe + laneId * 2);
  }
}

// ────────────────────────────────────────────────────────────────────────────
// bf16 variant: optional q RoPE + full-key concat + latent cache insert
// ────────────────────────────────────────────────────────────────────────────
template <typename scalar_t, bool APPLY_ROPE>
__global__ void fusedKimiK3MLAKeyConcatKVCacheInsertKernel(
    scalar_t* __restrict__ q, int64_t const q_tok_stride,
    int64_t const q_head_stride, const scalar_t* __restrict__ k_nope,
    int64_t const kn_tok_stride, int64_t const kn_head_stride,
    const scalar_t* __restrict__ k_pe, int64_t const k_pe_tok_stride,
    const scalar_t* __restrict__ kv_c, int64_t const kv_c_tok_stride,
    scalar_t* __restrict__ k_out, int64_t const ko_tok_stride,
    int64_t const ko_head_stride, scalar_t* __restrict__ k_cache,
    int64_t const cache_block_stride, int64_t const cache_token_stride,
    const int64_t* __restrict__ slot_mapping,
    const int64_t* __restrict__ position_ids,
    const float* __restrict__ cos_sin_cache, int const num_tokens,
    int const num_heads, int const cache_block_size) {
  int const warpsPerBlock = blockDim.x / 32;
  int const laneId = threadIdx.x % 32;
  int const globalWarpIdx = blockIdx.x * warpsPerBlock + threadIdx.x / 32;
  int const slotsPerToken = num_heads + 1;
  int const tokenIdx = globalWarpIdx / slotsPerToken;
  int const slotIdx = globalWarpIdx % slotsPerToken;
  if (tokenIdx >= num_tokens) return;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaGridDependencySynchronize();
#endif

  const float* rope_cache = nullptr;
  if constexpr (APPLY_ROPE) {
    rope_cache = cos_sin_cache + position_ids[tokenIdx] * kQkRopeHeadDim;
  }

  if (slotIdx < num_heads) {
    scalar_t* qh = q + tokenIdx * q_tok_stride + slotIdx * q_head_stride;
    writePrefillQuery<scalar_t, false, APPLY_ROPE>(
        qh, qh, laneId, sizeof(scalar_t), 1.0f, rope_cache);
    writeFullKey<scalar_t, false, APPLY_ROPE>(
        k_out + tokenIdx * ko_tok_stride + slotIdx * ko_head_stride,
        k_nope + tokenIdx * kn_tok_stride + slotIdx * kn_head_stride,
        k_pe + tokenIdx * k_pe_tok_stride, laneId, sizeof(scalar_t), 1.0f,
        rope_cache);
  } else {
    int64_t const slot_id = slot_mapping[tokenIdx];
    if (slot_id >= 0) {
      scalar_t* row = k_cache +
                      (slot_id / cache_block_size) * cache_block_stride +
                      (slot_id % cache_block_size) * cache_token_stride;
      writeLatent576<scalar_t, false, APPLY_ROPE>(
          row, kv_c + tokenIdx * kv_c_tok_stride,
          k_pe + tokenIdx * k_pe_tok_stride, laneId, sizeof(scalar_t), 1.0f,
          rope_cache);
    }
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// ────────────────────────────────────────────────────────────────────────────
// fp8 variant: quant q / k / v + latent cache insert
// ────────────────────────────────────────────────────────────────────────────
template <typename scalar_t, bool APPLY_ROPE>
__global__ void fusedKimiK3MLAQKVQuantKVCacheFp8Kernel(
    const scalar_t* __restrict__ q, int64_t const q_tok_stride,
    int64_t const q_head_stride, const scalar_t* __restrict__ k_nope,
    int64_t const kn_tok_stride, int64_t const kn_head_stride,
    const scalar_t* __restrict__ k_pe, int64_t const k_pe_tok_stride,
    const scalar_t* __restrict__ kv_c, int64_t const kv_c_tok_stride,
    const scalar_t* __restrict__ v, int64_t const v_tok_stride,
    int64_t const v_head_stride, uint8_t* __restrict__ q_fp8,
    int64_t const qo_tok_stride, int64_t const qo_head_stride,
    uint8_t* __restrict__ k_fp8, int64_t const ko_tok_stride,
    int64_t const ko_head_stride, uint8_t* __restrict__ v_fp8,
    int64_t const vo_tok_stride, int64_t const vo_head_stride,
    uint8_t* __restrict__ k_cache, int64_t const cache_block_stride,
    int64_t const cache_token_stride, const int64_t* __restrict__ slot_mapping,
    const float* __restrict__ q_scale_inv,
    const float* __restrict__ k_scale_inv,
    const float* __restrict__ v_scale_inv,
    const float* __restrict__ cache_scale_inv, int const num_tokens,
    int const num_heads, int const cache_block_size,
    const int64_t* __restrict__ position_ids,
    const float* __restrict__ cos_sin_cache) {
  int const warpsPerBlock = blockDim.x / 32;
  int const laneId = threadIdx.x % 32;
  int const globalWarpIdx = blockIdx.x * warpsPerBlock + threadIdx.x / 32;
  int const slotsPerToken = num_heads + 1;
  int const tokenIdx = globalWarpIdx / slotsPerToken;
  int const slotIdx = globalWarpIdx % slotsPerToken;
  if (tokenIdx >= num_tokens) return;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaGridDependencySynchronize();
#endif

  const float* rope_cache = nullptr;
  if constexpr (APPLY_ROPE) {
    rope_cache = cos_sin_cache + position_ids[tokenIdx] * kQkRopeHeadDim;
  }

  if (slotIdx < num_heads) {
    int const h = slotIdx;
    // q_fp8[t, h] = quant(q[t, h], q_scale)
    float const qsi = __ldg(q_scale_inv);
    const scalar_t* qh = q + tokenIdx * q_tok_stride + h * q_head_stride;
    uint8_t* qo = q_fp8 + tokenIdx * qo_tok_stride + h * qo_head_stride;
    writePrefillQuery<scalar_t, true, APPLY_ROPE>(qo, qh, laneId, 1, qsi,
                                                  rope_cache);
    // k_fp8[t, h] = quant([k_nope | k_pe], k_scale)
    writeFullKey<scalar_t, true, APPLY_ROPE>(
        k_fp8 + tokenIdx * ko_tok_stride + h * ko_head_stride,
        k_nope + tokenIdx * kn_tok_stride + h * kn_head_stride,
        k_pe + tokenIdx * k_pe_tok_stride, laneId, 1, __ldg(k_scale_inv),
        rope_cache);
    // v_fp8[t, h] = quant(v[t, h], v_scale)
    float const vsi = __ldg(v_scale_inv);
    const scalar_t* vh = v + tokenIdx * v_tok_stride + h * v_head_stride;
    uint8_t* vo = v_fp8 + tokenIdx * vo_tok_stride + h * vo_head_stride;
    for (int e = laneId * kVecElems; e < kVHeadDim; e += 32 * kVecElems) {
      copyChunk8<scalar_t, true>(vo + e, vh + e, vsi);
    }
  } else {
    int64_t const slot_id = slot_mapping[tokenIdx];
    if (slot_id >= 0) {
      // The cache latent uses _k_scale (read back by decode / context); the
      // attention key (k_fp8 above) uses its own k_scale.
      float const ksi = __ldg(cache_scale_inv);
      uint8_t* row = k_cache +
                     (slot_id / cache_block_size) * cache_block_stride +
                     (slot_id % cache_block_size) * cache_token_stride;
      writeLatent576<scalar_t, true, APPLY_ROPE>(
          row, kv_c + tokenIdx * kv_c_tok_stride,
          k_pe + tokenIdx * k_pe_tok_stride, laneId, 1, ksi, rope_cache);
    }
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// ────────────────────────────────────────────────────────────────────────────
// ds_mla variant: concat full key (bf16) + fp8_ds_mla latent cache insert
//
// Cache entry (656 bytes), matching concat_and_cache_ds_mla_kernel:
//   [0, 512)   NoPE 512 vals as fp8, 4 tiles of 128, each dynamically scaled
//   [512, 528) 4 fp32 per-tile scales
//   [528, 656) RoPE 64 vals as bf16 (unquantized)
// The cache slot uses one warp: lane L quantizes NoPE elems [L*16, L*16+16)
// (tile = L>>3, absmax-reduced within its 8-lane tile group), then all 32 lanes
// write 2 RoPE bf16 each.
// ────────────────────────────────────────────────────────────────────────────
template <typename scalar_t, bool APPLY_ROPE>
__global__ void fusedKimiK3MLAKeyConcatDsMlaInsertKernel(
    scalar_t* __restrict__ q, int64_t const q_tok_stride,
    int64_t const q_head_stride, const scalar_t* __restrict__ k_nope,
    int64_t const kn_tok_stride, int64_t const kn_head_stride,
    const scalar_t* __restrict__ k_pe, int64_t const k_pe_tok_stride,
    const scalar_t* __restrict__ kv_c, int64_t const kv_c_tok_stride,
    scalar_t* __restrict__ k_out, int64_t const ko_tok_stride,
    int64_t const ko_head_stride, uint8_t* __restrict__ k_cache,
    int64_t const cache_block_stride, int64_t const cache_token_stride,
    const int64_t* __restrict__ slot_mapping, int const num_tokens,
    int const num_heads, int const cache_block_size,
    const int64_t* __restrict__ position_ids,
    const float* __restrict__ cos_sin_cache) {
  int const warpsPerBlock = blockDim.x / 32;
  int const laneId = threadIdx.x % 32;
  int const globalWarpIdx = blockIdx.x * warpsPerBlock + threadIdx.x / 32;
  int const slotsPerToken = num_heads + 1;
  int const tokenIdx = globalWarpIdx / slotsPerToken;
  int const slotIdx = globalWarpIdx % slotsPerToken;
  if (tokenIdx >= num_tokens) return;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaGridDependencySynchronize();
#endif

  const float* rope_cache = nullptr;
  if constexpr (APPLY_ROPE) {
    rope_cache = cos_sin_cache + position_ids[tokenIdx] * kQkRopeHeadDim;
  }

  if (slotIdx < num_heads) {
    scalar_t* qh = q + tokenIdx * q_tok_stride + slotIdx * q_head_stride;
    writePrefillQuery<scalar_t, false, APPLY_ROPE>(
        qh, qh, laneId, sizeof(scalar_t), 1.0f, rope_cache);
    // Full key (bf16): k_out[t, h] = [k_nope[t, h] | k_pe[t]].
    writeFullKey<scalar_t, false, APPLY_ROPE>(
        k_out + tokenIdx * ko_tok_stride + slotIdx * ko_head_stride,
        k_nope + tokenIdx * kn_tok_stride + slotIdx * kn_head_stride,
        k_pe + tokenIdx * k_pe_tok_stride, laneId, sizeof(scalar_t), 1.0f,
        rope_cache);
  } else {
    int64_t const slot_id = slot_mapping[tokenIdx];
    if (slot_id >= 0) {
      uint8_t* row = k_cache +
                     (slot_id / cache_block_size) * cache_block_stride +
                     (slot_id % cache_block_size) * cache_token_stride;
      writeDsMlaCache<scalar_t, APPLY_ROPE>(
          row, kv_c + tokenIdx * kv_c_tok_stride,
          k_pe + tokenIdx * k_pe_tok_stride, laneId, rope_cache);
    }
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// ────────────────────────────────────────────────────────────────────────────
// Decode epilogue: concat mqa_q = [ql_nope | q_pe] (576) + latent cache insert,
// run right before forward_mqa. Q_FP8 quantizes mqa_q; KV_FP8 quantizes the
// plain per-tensor cache. (ds_mla cache uses the separate kernel below.)
// ────────────────────────────────────────────────────────────────────────────
template <typename scalar_t, bool Q_FP8, bool KV_FP8, bool APPLY_ROPE>
__global__ void fusedKimiK3MLADecodeQConcatKVCacheKernel(
    const scalar_t* __restrict__ ql_nope, int64_t const qn_tok_stride,
    int64_t const qn_head_stride, const scalar_t* __restrict__ q_pe,
    int64_t const qpe_tok_stride, int64_t const qpe_head_stride,
    const scalar_t* __restrict__ kv_c, int64_t const kv_c_tok_stride,
    const scalar_t* __restrict__ k_pe, int64_t const k_pe_tok_stride,
    void* __restrict__ mqa_q, int64_t const mq_tok_stride,
    int64_t const mq_head_stride, void* __restrict__ k_cache,
    int64_t const cache_block_stride, int64_t const cache_token_stride,
    const int64_t* __restrict__ slot_mapping,
    const float* __restrict__ q_scale_inv,
    const float* __restrict__ cache_scale_inv, int const num_tokens,
    int const num_heads, int const cache_block_size,
    const int64_t* __restrict__ position_ids,
    const float* __restrict__ cos_sin_cache) {
  constexpr int kMqElem = Q_FP8 ? 1 : sizeof(scalar_t);
  constexpr int kCacheElem = KV_FP8 ? 1 : sizeof(scalar_t);
  int const warpsPerBlock = blockDim.x / 32;
  int const laneId = threadIdx.x % 32;
  int const globalWarpIdx = blockIdx.x * warpsPerBlock + threadIdx.x / 32;
  int const slotsPerToken = num_heads + 1;
  int const tokenIdx = globalWarpIdx / slotsPerToken;
  int const slotIdx = globalWarpIdx % slotsPerToken;
  if (tokenIdx >= num_tokens) return;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaGridDependencySynchronize();
#endif

  const float* rope_cache = nullptr;
  if constexpr (APPLY_ROPE) {
    rope_cache = cos_sin_cache + position_ids[tokenIdx] * kQkRopeHeadDim;
  }

  if (slotIdx < num_heads) {
    float const qsi = Q_FP8 ? __ldg(q_scale_inv) : 1.0f;
    writeLatent576<scalar_t, Q_FP8, APPLY_ROPE>(
        reinterpret_cast<uint8_t*>(mqa_q) +
            (tokenIdx * mq_tok_stride + slotIdx * mq_head_stride) * kMqElem,
        ql_nope + tokenIdx * qn_tok_stride + slotIdx * qn_head_stride,
        q_pe + tokenIdx * qpe_tok_stride + slotIdx * qpe_head_stride, laneId,
        kMqElem, qsi, rope_cache);
  } else {
    int64_t const slot_id = slot_mapping[tokenIdx];
    if (slot_id >= 0) {
      float const ksi = KV_FP8 ? __ldg(cache_scale_inv) : 1.0f;
      writeLatent576<scalar_t, KV_FP8, APPLY_ROPE>(
          reinterpret_cast<uint8_t*>(k_cache) +
              (slot_id / cache_block_size * cache_block_stride +
               slot_id % cache_block_size * cache_token_stride) *
                  kCacheElem,
          kv_c + tokenIdx * kv_c_tok_stride, k_pe + tokenIdx * k_pe_tok_stride,
          laneId, kCacheElem, ksi, rope_cache);
    }
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// Decode epilogue for fp8_ds_mla: concat mqa_q (bf16) + ds_mla cache insert.
template <typename scalar_t, bool APPLY_ROPE>
__global__ void fusedKimiK3MLADecodeQConcatDsMlaKernel(
    const scalar_t* __restrict__ ql_nope, int64_t const qn_tok_stride,
    int64_t const qn_head_stride, const scalar_t* __restrict__ q_pe,
    int64_t const qpe_tok_stride, int64_t const qpe_head_stride,
    const scalar_t* __restrict__ kv_c, int64_t const kv_c_tok_stride,
    const scalar_t* __restrict__ k_pe, int64_t const k_pe_tok_stride,
    scalar_t* __restrict__ mqa_q, int64_t const mq_tok_stride,
    int64_t const mq_head_stride, uint8_t* __restrict__ k_cache,
    int64_t const cache_block_stride, int64_t const cache_token_stride,
    const int64_t* __restrict__ slot_mapping, int const num_tokens,
    int const num_heads, int const cache_block_size,
    const int64_t* __restrict__ position_ids,
    const float* __restrict__ cos_sin_cache) {
  int const warpsPerBlock = blockDim.x / 32;
  int const laneId = threadIdx.x % 32;
  int const globalWarpIdx = blockIdx.x * warpsPerBlock + threadIdx.x / 32;
  int const slotsPerToken = num_heads + 1;
  int const tokenIdx = globalWarpIdx / slotsPerToken;
  int const slotIdx = globalWarpIdx % slotsPerToken;
  if (tokenIdx >= num_tokens) return;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaGridDependencySynchronize();
#endif

  const float* rope_cache = nullptr;
  if constexpr (APPLY_ROPE) {
    rope_cache = cos_sin_cache + position_ids[tokenIdx] * kQkRopeHeadDim;
  }

  if (slotIdx < num_heads) {
    writeLatent576<scalar_t, false, APPLY_ROPE>(
        mqa_q + tokenIdx * mq_tok_stride + slotIdx * mq_head_stride,
        ql_nope + tokenIdx * qn_tok_stride + slotIdx * qn_head_stride,
        q_pe + tokenIdx * qpe_tok_stride + slotIdx * qpe_head_stride, laneId,
        sizeof(scalar_t), 1.0f, rope_cache);
  } else {
    int64_t const slot_id = slot_mapping[tokenIdx];
    if (slot_id >= 0) {
      uint8_t* row = k_cache +
                     (slot_id / cache_block_size) * cache_block_stride +
                     (slot_id % cache_block_size) * cache_token_stride;
      writeDsMlaCache<scalar_t, APPLY_ROPE>(
          row, kv_c + tokenIdx * kv_c_tok_stride,
          k_pe + tokenIdx * k_pe_tok_stride, laneId, rope_cache);
    }
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// PDL-aware launch of a (token, num_heads + 1)-warp grid.
template <typename KernelT, typename... Args>
static void launchPdl(KernelT kernel, int num_tokens, int num_heads,
                      cudaStream_t stream, Args... args) {
  constexpr int kBlockSize = 256;
  constexpr int kWarpsPerBlock = kBlockSize / 32;
  int64_t const total_warps =
      static_cast<int64_t>(num_tokens) * (num_heads + 1);
  int const grid =
      static_cast<int>((total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock);
#ifndef USE_ROCM
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
  cudaLaunchKernelEx(&config, kernel, args...);
#else
  // clang-format off
  // hipify's CUDA->HIP regex catastrophically backtracks on "> > >"; keep the
  // launch closer as ">>>". clang-format would otherwise re-split it (it does
  // not parse the CUDA launch syntax in this libtorch_stable file).
  kernel<<<grid, kBlockSize, 0, stream>>>(args...);
  // clang-format on
#endif
}

void checkBfloat16Support(torch::headeronly::ScalarType dtype) {
#ifndef USE_ROCM
  if (dtype == torch::headeronly::ScalarType::BFloat16) {
    static int const sm_version = getSMVersion();
    STD_TORCH_CHECK(
        sm_version >= 80,
        "Kimi K3 fused MLA operations require sm_80+ (Ampere or newer); got "
        "sm_",
        sm_version);
  }
#else
  (void)dtype;
#endif
}

}  // namespace kimi_k3_fused_ops
}  // namespace vllm

// ────────────────────────────────────────────────────────────────────────────
// Torch op wrappers
// ────────────────────────────────────────────────────────────────────────────
namespace {
bool check_rope_inputs(
    std::optional<torch::stable::Tensor> const& position_ids,
    std::optional<torch::stable::Tensor> const& cos_sin_cache,
    torch::stable::Tensor const& /*input*/, int64_t num_tokens) {
  using torch::headeronly::ScalarType;
  STD_TORCH_CHECK(position_ids.has_value() == cos_sin_cache.has_value(),
                  "position_ids and cos_sin_cache must be provided together");
  if (!position_ids.has_value()) return false;

  auto const& positions = position_ids.value();
  auto const& rope_cache = cos_sin_cache.value();
  STD_TORCH_CHECK(positions.device().is_cuda() && positions.dim() == 1 &&
                      positions.scalar_type() == ScalarType::Long &&
                      positions.size(0) == num_tokens,
                  "position_ids must be int64 CUDA with shape [num_tokens]");
  STD_TORCH_CHECK(rope_cache.device().is_cuda() && rope_cache.dim() == 2 &&
                      rope_cache.size(1) == 64 && rope_cache.stride(1) == 1 &&
                      rope_cache.scalar_type() == ScalarType::Float,
                  "cos_sin_cache must have shape [max_position, 64], unit "
                  "last-dim stride, and be fp32 (RoPE math runs in fp32)");
  return true;
}
}  // namespace

void fused_kimi_k3_mla_key_concat_kv_cache_insert(
    torch::stable::Tensor& q,                   // [Tp, H, 192]
    torch::stable::Tensor const& k_nope,        // [Tp, H, 128]
    torch::stable::Tensor const& k_pe,          // [Tp, 64]
    torch::stable::Tensor const& kv_c_normed,   // [Tp, 512]
    torch::stable::Tensor& k_out,               // [Tp, H, 192] bf16, written
    torch::stable::Tensor& k_cache,             // [nblk, bs, 576] bf16, written
    torch::stable::Tensor const& slot_mapping,  // [Tp] int64
    int64_t cache_block_size, std::optional<torch::stable::Tensor> position_ids,
    std::optional<torch::stable::Tensor> cos_sin_cache) {
  using torch::headeronly::ScalarType;
  namespace kk3 = vllm::kimi_k3_fused_ops;
  STD_TORCH_CHECK(
      k_nope.device().is_cuda() && k_nope.dim() == 3 && k_nope.size(2) == 128,
      "k_nope shape [Tp, H, 128] CUDA");
  STD_TORCH_CHECK(q.device().is_cuda() && q.dim() == 3 && q.size(2) == 192,
                  "q shape [Tp, H, 192] CUDA");
  // k_pe is a strided view of the fused QKV-LoRA GEMM output; the kernel takes
  // its row stride and reads the 64 cols contiguously, so unit last-dim stride
  // (not full contiguity) is all that is required.
  STD_TORCH_CHECK(k_pe.device().is_cuda() && k_pe.dim() == 2 &&
                      k_pe.stride(1) == 1 && k_pe.size(1) == 64,
                  "k_pe shape [Tp, 64], unit last-dim stride, CUDA");
  STD_TORCH_CHECK(kv_c_normed.device().is_cuda() &&
                      kv_c_normed.is_contiguous() && kv_c_normed.dim() == 2 &&
                      kv_c_normed.size(1) == 512,
                  "kv_c_normed shape [Tp, 512] contiguous CUDA");
  STD_TORCH_CHECK(k_out.device().is_cuda() && k_out.is_contiguous() &&
                      k_out.dim() == 3 && k_out.size(2) == 192,
                  "k_out shape [Tp, H, 192] contiguous CUDA");
  STD_TORCH_CHECK(k_cache.device().is_cuda() && k_cache.dim() == 3 &&
                      k_cache.size(1) == cache_block_size &&
                      k_cache.size(2) == 576 && k_cache.stride(2) == 1,
                  "k_cache shape [nblk, block_size, 576] contiguous CUDA");
  STD_TORCH_CHECK(slot_mapping.device().is_cuda() &&
                      slot_mapping.scalar_type() == ScalarType::Long,
                  "slot_mapping must be int64 CUDA");
  ScalarType const dt = k_nope.scalar_type();
  STD_TORCH_CHECK(q.scalar_type() == dt && k_pe.scalar_type() == dt &&
                      kv_c_normed.scalar_type() == dt &&
                      k_out.scalar_type() == dt && k_cache.scalar_type() == dt,
                  "all tensors must share k_nope's (bf16/fp16) dtype");
  kk3::checkBfloat16Support(dt);

  int const num_tokens = static_cast<int>(k_nope.size(0));
  int const num_heads = static_cast<int>(k_nope.size(1));
  STD_TORCH_CHECK(static_cast<int>(k_out.size(1)) == num_heads,
                  "k_out head count must match k_nope");
  STD_TORCH_CHECK(q.size(0) == num_tokens && q.size(1) == num_heads,
                  "q token/head dimensions must match k_nope");
  bool const apply_rope =
      check_rope_inputs(position_ids, cos_sin_cache, q, num_tokens);
  if (num_tokens == 0) return;

  const torch::stable::accelerator::DeviceGuard device_guard(
      k_nope.get_device_index());
  const cudaStream_t stream =
      get_current_cuda_stream(k_nope.get_device_index());

  VLLM_STABLE_DISPATCH_HALF_TYPES(
      dt, "fused_kimi_k3_mla_key_concat_kv_cache_insert", [&] {
        auto launch = [&](auto kernel) {
          kk3::launchPdl(
              kernel, num_tokens, num_heads, stream,
              reinterpret_cast<scalar_t*>(q.mutable_data_ptr()), q.stride(0),
              q.stride(1),
              reinterpret_cast<scalar_t const*>(k_nope.const_data_ptr()),
              k_nope.stride(0), k_nope.stride(1),
              reinterpret_cast<scalar_t const*>(k_pe.const_data_ptr()),
              k_pe.stride(0),
              reinterpret_cast<scalar_t const*>(kv_c_normed.const_data_ptr()),
              kv_c_normed.stride(0),
              reinterpret_cast<scalar_t*>(k_out.mutable_data_ptr()),
              k_out.stride(0), k_out.stride(1),
              reinterpret_cast<scalar_t*>(k_cache.mutable_data_ptr()),
              k_cache.stride(0), k_cache.stride(1),
              slot_mapping.const_data_ptr<int64_t>(),
              apply_rope ? position_ids.value().const_data_ptr<int64_t>()
                         : nullptr,
              apply_rope ? reinterpret_cast<float const*>(
                               cos_sin_cache.value().const_data_ptr())
                         : nullptr,
              num_tokens, num_heads, static_cast<int>(cache_block_size));
        };
        if (apply_rope) {
          launch(
              kk3::fusedKimiK3MLAKeyConcatKVCacheInsertKernel<scalar_t, true>);
        } else {
          launch(
              kk3::fusedKimiK3MLAKeyConcatKVCacheInsertKernel<scalar_t, false>);
        }
      });
}

void fused_kimi_k3_mla_key_concat_ds_mla_insert(
    torch::stable::Tensor& q,                  // [Tp, H, 192]
    torch::stable::Tensor const& k_nope,       // [Tp, H, 128] bf16
    torch::stable::Tensor const& k_pe,         // [Tp, 64] bf16
    torch::stable::Tensor const& kv_c_normed,  // [Tp, 512] bf16
    torch::stable::Tensor& k_out,              // [Tp, H, 192] bf16, written
    torch::stable::Tensor& k_cache,            // [nblk, bs, 656] uint8, written
    torch::stable::Tensor const& slot_mapping,  // [Tp] int64
    int64_t cache_block_size, std::optional<torch::stable::Tensor> position_ids,
    std::optional<torch::stable::Tensor> cos_sin_cache) {
  using torch::headeronly::ScalarType;
  namespace kk3 = vllm::kimi_k3_fused_ops;
  ScalarType const dt = k_nope.scalar_type();
  STD_TORCH_CHECK(
      k_nope.device().is_cuda() && k_nope.dim() == 3 && k_nope.size(2) == 128,
      "k_nope shape [Tp, H, 128] CUDA");
  STD_TORCH_CHECK(q.device().is_cuda() && q.scalar_type() == dt &&
                      q.dim() == 3 && q.size(2) == 192,
                  "q shape [Tp, H, 192]");
  STD_TORCH_CHECK(k_pe.device().is_cuda() && k_pe.dim() == 2 &&
                      k_pe.stride(1) == 1 && k_pe.scalar_type() == dt &&
                      k_pe.size(1) == 64,
                  "k_pe shape [Tp, 64], unit last-dim stride");
  STD_TORCH_CHECK(kv_c_normed.device().is_cuda() &&
                      kv_c_normed.is_contiguous() &&
                      kv_c_normed.scalar_type() == dt &&
                      kv_c_normed.dim() == 2 && kv_c_normed.size(1) == 512,
                  "kv_c_normed shape [Tp, 512] contiguous");
  STD_TORCH_CHECK(k_out.device().is_cuda() && k_out.is_contiguous() &&
                      k_out.scalar_type() == dt && k_out.dim() == 3 &&
                      k_out.size(2) == 192,
                  "k_out shape [Tp, H, 192] contiguous");
  // fp8_ds_mla entry is 656 bytes stored as uint8.
  STD_TORCH_CHECK(
      k_cache.device().is_cuda() && k_cache.scalar_type() == ScalarType::Byte &&
          k_cache.dim() == 3 && k_cache.size(1) == cache_block_size &&
          k_cache.size(2) == 656 && k_cache.stride(2) == 1,
      "k_cache shape [nblk, block_size, 656] uint8 contiguous");
  STD_TORCH_CHECK(slot_mapping.device().is_cuda() &&
                      slot_mapping.scalar_type() == ScalarType::Long,
                  "slot_mapping must be int64 CUDA");
  kk3::checkBfloat16Support(dt);

  int const num_tokens = static_cast<int>(k_nope.size(0));
  int const num_heads = static_cast<int>(k_nope.size(1));
  STD_TORCH_CHECK(static_cast<int>(k_out.size(1)) == num_heads,
                  "k_out head count must match k_nope");
  STD_TORCH_CHECK(q.size(0) == num_tokens && q.size(1) == num_heads,
                  "q token/head dimensions must match k_nope");
  bool const apply_rope =
      check_rope_inputs(position_ids, cos_sin_cache, q, num_tokens);
  if (num_tokens == 0) return;

  const torch::stable::accelerator::DeviceGuard device_guard(
      k_nope.get_device_index());
  const cudaStream_t stream =
      get_current_cuda_stream(k_nope.get_device_index());

  VLLM_STABLE_DISPATCH_HALF_TYPES(
      dt, "fused_kimi_k3_mla_key_concat_ds_mla_insert", [&] {
        auto launch = [&](auto kernel) {
          kk3::launchPdl(
              kernel, num_tokens, num_heads, stream,
              reinterpret_cast<scalar_t*>(q.mutable_data_ptr()), q.stride(0),
              q.stride(1),
              reinterpret_cast<scalar_t const*>(k_nope.const_data_ptr()),
              k_nope.stride(0), k_nope.stride(1),
              reinterpret_cast<scalar_t const*>(k_pe.const_data_ptr()),
              k_pe.stride(0),
              reinterpret_cast<scalar_t const*>(kv_c_normed.const_data_ptr()),
              kv_c_normed.stride(0),
              reinterpret_cast<scalar_t*>(k_out.mutable_data_ptr()),
              k_out.stride(0), k_out.stride(1),
              reinterpret_cast<uint8_t*>(k_cache.mutable_data_ptr()),
              k_cache.stride(0), k_cache.stride(1),
              slot_mapping.const_data_ptr<int64_t>(), num_tokens, num_heads,
              static_cast<int>(cache_block_size),
              apply_rope ? position_ids.value().const_data_ptr<int64_t>()
                         : nullptr,
              apply_rope ? reinterpret_cast<float const*>(
                               cos_sin_cache.value().const_data_ptr())
                         : nullptr);
        };
        if (apply_rope) {
          launch(kk3::fusedKimiK3MLAKeyConcatDsMlaInsertKernel<scalar_t, true>);
        } else {
          launch(
              kk3::fusedKimiK3MLAKeyConcatDsMlaInsertKernel<scalar_t, false>);
        }
      });
}

void fused_kimi_k3_mla_qkv_quant_kv_cache_fp8_insert(
    torch::stable::Tensor const& q,             // [Tp, H, 192] bf16
    torch::stable::Tensor const& k_nope,        // [Tp, H, 128] bf16
    torch::stable::Tensor const& k_pe,          // [Tp, 64] bf16
    torch::stable::Tensor const& kv_c_normed,   // [Tp, 512] bf16
    torch::stable::Tensor const& v,             // [Tp, H, 128] bf16
    torch::stable::Tensor& q_fp8,               // [Tp, H, 192] fp8, written
    torch::stable::Tensor& k_fp8,               // [Tp, H, 192] fp8, written
    torch::stable::Tensor& v_fp8,               // [Tp, H, 128] fp8, written
    torch::stable::Tensor& k_cache,             // [nblk, bs, 576] fp8, written
    torch::stable::Tensor const& slot_mapping,  // [Tp] int64
    torch::stable::Tensor const& q_scale_inv,   // scalar fp32 (1 / q scale)
    torch::stable::Tensor const& k_scale_inv,   // scalar fp32 (1 / k scale)
    torch::stable::Tensor const& v_scale_inv,   // scalar fp32 (1 / v scale)
    torch::stable::Tensor const& cache_scale_inv,  // scalar fp32 (1 / kv scale)
    int64_t cache_block_size, std::optional<torch::stable::Tensor> position_ids,
    std::optional<torch::stable::Tensor> cos_sin_cache) {
  using torch::headeronly::ScalarType;
  namespace kk3 = vllm::kimi_k3_fused_ops;
  ScalarType const dt = k_nope.scalar_type();
  auto check_in = [&](torch::stable::Tensor const& t, int d2, char const* n) {
    STD_TORCH_CHECK(t.device().is_cuda() && t.scalar_type() == dt &&
                        t.dim() == 3 && t.size(2) == d2,
                    n);
  };
  check_in(q, 192, "q shape [Tp, H, 192]");
  check_in(k_nope, 128, "k_nope shape [Tp, H, 128]");
  check_in(v, 128, "v shape [Tp, H, 128]");
  STD_TORCH_CHECK(k_pe.device().is_cuda() && k_pe.dim() == 2 &&
                      k_pe.stride(1) == 1 && k_pe.scalar_type() == dt &&
                      k_pe.size(1) == 64,
                  "k_pe shape [Tp, 64], unit last-dim stride");
  STD_TORCH_CHECK(kv_c_normed.device().is_cuda() &&
                      kv_c_normed.is_contiguous() &&
                      kv_c_normed.scalar_type() == dt &&
                      kv_c_normed.dim() == 2 && kv_c_normed.size(1) == 512,
                  "kv_c_normed shape [Tp, 512] contiguous");
  auto check_out = [&](torch::stable::Tensor const& t, int d2, char const* n) {
    STD_TORCH_CHECK(t.device().is_cuda() && t.is_contiguous() &&
                        t.scalar_type() == ScalarType::Float8_e4m3fn &&
                        t.dim() == 3 && t.size(2) == d2,
                    n);
  };
  check_out(q_fp8, 192, "q_fp8 shape [Tp, H, 192] fp8 contiguous");
  check_out(k_fp8, 192, "k_fp8 shape [Tp, H, 192] fp8 contiguous");
  check_out(v_fp8, 128, "v_fp8 shape [Tp, H, 128] fp8 contiguous");
  STD_TORCH_CHECK(k_cache.device().is_cuda() && k_cache.dim() == 3 &&
                      k_cache.size(1) == cache_block_size &&
                      k_cache.size(2) == 576 && k_cache.stride(2) == 1 &&
                      k_cache.scalar_type() == ScalarType::Float8_e4m3fn,
                  "k_cache shape [nblk, block_size, 576] fp8 contiguous");
  STD_TORCH_CHECK(slot_mapping.device().is_cuda() &&
                      slot_mapping.scalar_type() == ScalarType::Long,
                  "slot_mapping must be int64 CUDA");
  auto check_scale = [&](torch::stable::Tensor const& s, char const* n) {
    STD_TORCH_CHECK(s.device().is_cuda() &&
                        s.scalar_type() == ScalarType::Float && s.size(0) == 1,
                    n);
  };
  check_scale(q_scale_inv, "q_scale_inv must be scalar float32 CUDA");
  check_scale(k_scale_inv, "k_scale_inv must be scalar float32 CUDA");
  check_scale(v_scale_inv, "v_scale_inv must be scalar float32 CUDA");
  check_scale(cache_scale_inv, "cache_scale_inv must be scalar float32 CUDA");
  kk3::checkBfloat16Support(dt);

  int const num_tokens = static_cast<int>(k_nope.size(0));
  int const num_heads = static_cast<int>(k_nope.size(1));
  bool const apply_rope =
      check_rope_inputs(position_ids, cos_sin_cache, q, num_tokens);
  if (num_tokens == 0) return;

  const torch::stable::accelerator::DeviceGuard device_guard(
      k_nope.get_device_index());
  const cudaStream_t stream =
      get_current_cuda_stream(k_nope.get_device_index());

  VLLM_STABLE_DISPATCH_HALF_TYPES(
      dt, "fused_kimi_k3_mla_qkv_quant_kv_cache_fp8_insert", [&] {
        auto launch = [&](auto kernel) {
          kk3::launchPdl(
              kernel, num_tokens, num_heads, stream,
              reinterpret_cast<scalar_t const*>(q.const_data_ptr()),
              q.stride(0), q.stride(1),
              reinterpret_cast<scalar_t const*>(k_nope.const_data_ptr()),
              k_nope.stride(0), k_nope.stride(1),
              reinterpret_cast<scalar_t const*>(k_pe.const_data_ptr()),
              k_pe.stride(0),
              reinterpret_cast<scalar_t const*>(kv_c_normed.const_data_ptr()),
              kv_c_normed.stride(0),
              reinterpret_cast<scalar_t const*>(v.const_data_ptr()),
              v.stride(0), v.stride(1),
              reinterpret_cast<uint8_t*>(q_fp8.mutable_data_ptr()),
              q_fp8.stride(0), q_fp8.stride(1),
              reinterpret_cast<uint8_t*>(k_fp8.mutable_data_ptr()),
              k_fp8.stride(0), k_fp8.stride(1),
              reinterpret_cast<uint8_t*>(v_fp8.mutable_data_ptr()),
              v_fp8.stride(0), v_fp8.stride(1),
              reinterpret_cast<uint8_t*>(k_cache.mutable_data_ptr()),
              k_cache.stride(0), k_cache.stride(1),
              slot_mapping.const_data_ptr<int64_t>(),
              q_scale_inv.const_data_ptr<float>(),
              k_scale_inv.const_data_ptr<float>(),
              v_scale_inv.const_data_ptr<float>(),
              cache_scale_inv.const_data_ptr<float>(), num_tokens, num_heads,
              static_cast<int>(cache_block_size),
              apply_rope ? position_ids.value().const_data_ptr<int64_t>()
                         : nullptr,
              apply_rope ? reinterpret_cast<float const*>(
                               cos_sin_cache.value().const_data_ptr())
                         : nullptr);
        };
        if (apply_rope) {
          launch(kk3::fusedKimiK3MLAQKVQuantKVCacheFp8Kernel<scalar_t, true>);
        } else {
          launch(kk3::fusedKimiK3MLAQKVQuantKVCacheFp8Kernel<scalar_t, false>);
        }
      });
}

// ────────────────────────────────────────────────────────────────────────────
// Decode epilogue torch ops
// ────────────────────────────────────────────────────────────────────────────
namespace {
// Shared shape checks for the decode ops. Verifies the query/latent inputs and
// slot_mapping; caller checks mqa_q / k_cache dtypes for its variant.
void check_decode_inputs(torch::stable::Tensor const& ql_nope,
                         torch::stable::Tensor const& q_pe,
                         torch::stable::Tensor const& kv_c_normed,
                         torch::stable::Tensor const& k_pe,
                         torch::stable::Tensor const& mqa_q,
                         torch::stable::Tensor const& slot_mapping) {
  using torch::headeronly::ScalarType;
  auto const dt = ql_nope.scalar_type();
  STD_TORCH_CHECK(ql_nope.device().is_cuda() && ql_nope.dim() == 3 &&
                      ql_nope.size(2) == 512,
                  "ql_nope shape [B, H, 512] CUDA");
  STD_TORCH_CHECK(q_pe.device().is_cuda() && q_pe.scalar_type() == dt &&
                      q_pe.dim() == 3 && q_pe.size(2) == 64,
                  "q_pe shape [B, H, 64]");
  STD_TORCH_CHECK(kv_c_normed.device().is_cuda() &&
                      kv_c_normed.is_contiguous() &&
                      kv_c_normed.scalar_type() == dt &&
                      kv_c_normed.dim() == 2 && kv_c_normed.size(1) == 512,
                  "kv_c_normed shape [B, 512] contiguous");
  STD_TORCH_CHECK(k_pe.device().is_cuda() && k_pe.dim() == 2 &&
                      k_pe.stride(1) == 1 && k_pe.scalar_type() == dt &&
                      k_pe.size(1) == 64,
                  "k_pe shape [B, 64], unit last-dim stride");
  STD_TORCH_CHECK(mqa_q.device().is_cuda() && mqa_q.is_contiguous() &&
                      mqa_q.dim() == 3 && mqa_q.size(2) == 576,
                  "mqa_q shape [B, H, 576] contiguous");
  STD_TORCH_CHECK(slot_mapping.device().is_cuda() &&
                      slot_mapping.scalar_type() == ScalarType::Long,
                  "slot_mapping must be int64 CUDA");
}
}  // namespace

void fused_kimi_k3_mla_decode_q_concat_kv_cache_insert(
    torch::stable::Tensor const& ql_nope,       // [B, H, 512] bf16
    torch::stable::Tensor const& q_pe,          // [B, H, 64] bf16
    torch::stable::Tensor const& kv_c_normed,   // [B, 512] bf16
    torch::stable::Tensor const& k_pe,          // [B, 64] bf16
    torch::stable::Tensor& mqa_q,               // [B, H, 576] bf16, written
    torch::stable::Tensor& k_cache,             // [nblk, bs, 576] bf16, written
    torch::stable::Tensor const& slot_mapping,  // [B] int64
    int64_t cache_block_size, std::optional<torch::stable::Tensor> position_ids,
    std::optional<torch::stable::Tensor> cos_sin_cache) {
  using torch::headeronly::ScalarType;
  namespace kk3 = vllm::kimi_k3_fused_ops;
  ScalarType const dt = ql_nope.scalar_type();
  check_decode_inputs(ql_nope, q_pe, kv_c_normed, k_pe, mqa_q, slot_mapping);
  STD_TORCH_CHECK(mqa_q.scalar_type() == dt && k_cache.scalar_type() == dt,
                  "mqa_q / k_cache must match ql_nope dtype (bf16)");
  STD_TORCH_CHECK(k_cache.device().is_cuda() && k_cache.dim() == 3 &&
                      k_cache.size(1) == cache_block_size &&
                      k_cache.size(2) == 576 && k_cache.stride(2) == 1,
                  "k_cache shape [nblk, block_size, 576] contiguous");
  kk3::checkBfloat16Support(dt);

  int const num_tokens = static_cast<int>(ql_nope.size(0));
  int const num_heads = static_cast<int>(ql_nope.size(1));
  bool const apply_rope =
      check_rope_inputs(position_ids, cos_sin_cache, q_pe, num_tokens);
  if (num_tokens == 0) return;
  const torch::stable::accelerator::DeviceGuard device_guard(
      ql_nope.get_device_index());
  const cudaStream_t stream =
      get_current_cuda_stream(ql_nope.get_device_index());

  VLLM_STABLE_DISPATCH_HALF_TYPES(
      dt, "fused_kimi_k3_mla_decode_q_concat_kv_cache_insert", [&] {
        auto launch = [&](auto kernel) {
          kk3::launchPdl(
              kernel, num_tokens, num_heads, stream,
              reinterpret_cast<scalar_t const*>(ql_nope.const_data_ptr()),
              ql_nope.stride(0), ql_nope.stride(1),
              reinterpret_cast<scalar_t const*>(q_pe.const_data_ptr()),
              q_pe.stride(0), q_pe.stride(1),
              reinterpret_cast<scalar_t const*>(kv_c_normed.const_data_ptr()),
              kv_c_normed.stride(0),
              reinterpret_cast<scalar_t const*>(k_pe.const_data_ptr()),
              k_pe.stride(0), mqa_q.mutable_data_ptr(), mqa_q.stride(0),
              mqa_q.stride(1), k_cache.mutable_data_ptr(), k_cache.stride(0),
              k_cache.stride(1), slot_mapping.const_data_ptr<int64_t>(),
              nullptr, nullptr, num_tokens, num_heads,
              static_cast<int>(cache_block_size),
              apply_rope ? position_ids.value().const_data_ptr<int64_t>()
                         : nullptr,
              apply_rope ? reinterpret_cast<float const*>(
                               cos_sin_cache.value().const_data_ptr())
                         : nullptr);
        };
        if (apply_rope) {
          launch(kk3::fusedKimiK3MLADecodeQConcatKVCacheKernel<scalar_t, false,
                                                               false, true>);
        } else {
          launch(kk3::fusedKimiK3MLADecodeQConcatKVCacheKernel<scalar_t, false,
                                                               false, false>);
        }
      });
}

void fused_kimi_k3_mla_decode_q_concat_kv_cache_fp8_insert(
    torch::stable::Tensor const& ql_nope,       // [B, H, 512] bf16
    torch::stable::Tensor const& q_pe,          // [B, H, 64] bf16
    torch::stable::Tensor const& kv_c_normed,   // [B, 512] bf16
    torch::stable::Tensor const& k_pe,          // [B, 64] bf16
    torch::stable::Tensor& mqa_q,               // [B, H, 576] fp8, written
    torch::stable::Tensor& k_cache,             // [nblk, bs, 576] fp8, written
    torch::stable::Tensor const& slot_mapping,  // [B] int64
    torch::stable::Tensor const& q_scale_inv,   // scalar fp32 (1 / q scale)
    torch::stable::Tensor const& cache_scale_inv,  // scalar fp32 (1 / kv scale)
    int64_t cache_block_size, std::optional<torch::stable::Tensor> position_ids,
    std::optional<torch::stable::Tensor> cos_sin_cache) {
  using torch::headeronly::ScalarType;
  namespace kk3 = vllm::kimi_k3_fused_ops;
  ScalarType const dt = ql_nope.scalar_type();
  check_decode_inputs(ql_nope, q_pe, kv_c_normed, k_pe, mqa_q, slot_mapping);
  STD_TORCH_CHECK(mqa_q.scalar_type() == ScalarType::Float8_e4m3fn,
                  "mqa_q must be float8_e4m3fn");
  STD_TORCH_CHECK(k_cache.device().is_cuda() && k_cache.dim() == 3 &&
                      k_cache.size(1) == cache_block_size &&
                      k_cache.size(2) == 576 && k_cache.stride(2) == 1 &&
                      k_cache.scalar_type() == ScalarType::Float8_e4m3fn,
                  "k_cache shape [nblk, block_size, 576] fp8 contiguous");
  auto check_scale = [&](torch::stable::Tensor const& s, char const* n) {
    STD_TORCH_CHECK(s.device().is_cuda() &&
                        s.scalar_type() == ScalarType::Float && s.size(0) == 1,
                    n);
  };
  check_scale(q_scale_inv, "q_scale_inv must be scalar float32 CUDA");
  check_scale(cache_scale_inv, "cache_scale_inv must be scalar float32 CUDA");
  kk3::checkBfloat16Support(dt);

  int const num_tokens = static_cast<int>(ql_nope.size(0));
  int const num_heads = static_cast<int>(ql_nope.size(1));
  bool const apply_rope =
      check_rope_inputs(position_ids, cos_sin_cache, q_pe, num_tokens);
  if (num_tokens == 0) return;
  const torch::stable::accelerator::DeviceGuard device_guard(
      ql_nope.get_device_index());
  const cudaStream_t stream =
      get_current_cuda_stream(ql_nope.get_device_index());

  VLLM_STABLE_DISPATCH_HALF_TYPES(
      dt, "fused_kimi_k3_mla_decode_q_concat_kv_cache_fp8_insert", [&] {
        auto launch = [&](auto kernel) {
          kk3::launchPdl(
              kernel, num_tokens, num_heads, stream,
              reinterpret_cast<scalar_t const*>(ql_nope.const_data_ptr()),
              ql_nope.stride(0), ql_nope.stride(1),
              reinterpret_cast<scalar_t const*>(q_pe.const_data_ptr()),
              q_pe.stride(0), q_pe.stride(1),
              reinterpret_cast<scalar_t const*>(kv_c_normed.const_data_ptr()),
              kv_c_normed.stride(0),
              reinterpret_cast<scalar_t const*>(k_pe.const_data_ptr()),
              k_pe.stride(0), mqa_q.mutable_data_ptr(), mqa_q.stride(0),
              mqa_q.stride(1), k_cache.mutable_data_ptr(), k_cache.stride(0),
              k_cache.stride(1), slot_mapping.const_data_ptr<int64_t>(),
              q_scale_inv.const_data_ptr<float>(),
              cache_scale_inv.const_data_ptr<float>(), num_tokens, num_heads,
              static_cast<int>(cache_block_size),
              apply_rope ? position_ids.value().const_data_ptr<int64_t>()
                         : nullptr,
              apply_rope ? reinterpret_cast<float const*>(
                               cos_sin_cache.value().const_data_ptr())
                         : nullptr);
        };
        if (apply_rope) {
          launch(kk3::fusedKimiK3MLADecodeQConcatKVCacheKernel<scalar_t, true,
                                                               true, true>);
        } else {
          launch(kk3::fusedKimiK3MLADecodeQConcatKVCacheKernel<scalar_t, true,
                                                               true, false>);
        }
      });
}

void fused_kimi_k3_mla_decode_q_concat_ds_mla_insert(
    torch::stable::Tensor const& ql_nope,      // [B, H, 512] bf16
    torch::stable::Tensor const& q_pe,         // [B, H, 64] bf16
    torch::stable::Tensor const& kv_c_normed,  // [B, 512] bf16
    torch::stable::Tensor const& k_pe,         // [B, 64] bf16
    torch::stable::Tensor& mqa_q,              // [B, H, 576] bf16, written
    torch::stable::Tensor& k_cache,            // [nblk, bs, 656] uint8, written
    torch::stable::Tensor const& slot_mapping,  // [B] int64
    int64_t cache_block_size, std::optional<torch::stable::Tensor> position_ids,
    std::optional<torch::stable::Tensor> cos_sin_cache) {
  using torch::headeronly::ScalarType;
  namespace kk3 = vllm::kimi_k3_fused_ops;
  ScalarType const dt = ql_nope.scalar_type();
  check_decode_inputs(ql_nope, q_pe, kv_c_normed, k_pe, mqa_q, slot_mapping);
  STD_TORCH_CHECK(mqa_q.scalar_type() == dt, "mqa_q must be bf16 for ds_mla");
  STD_TORCH_CHECK(
      k_cache.device().is_cuda() && k_cache.scalar_type() == ScalarType::Byte &&
          k_cache.dim() == 3 && k_cache.size(1) == cache_block_size &&
          k_cache.size(2) == 656 && k_cache.stride(2) == 1,
      "k_cache shape [nblk, block_size, 656] uint8 contiguous");
  kk3::checkBfloat16Support(dt);

  int const num_tokens = static_cast<int>(ql_nope.size(0));
  int const num_heads = static_cast<int>(ql_nope.size(1));
  bool const apply_rope =
      check_rope_inputs(position_ids, cos_sin_cache, q_pe, num_tokens);
  if (num_tokens == 0) return;
  const torch::stable::accelerator::DeviceGuard device_guard(
      ql_nope.get_device_index());
  const cudaStream_t stream =
      get_current_cuda_stream(ql_nope.get_device_index());

  VLLM_STABLE_DISPATCH_HALF_TYPES(
      dt, "fused_kimi_k3_mla_decode_q_concat_ds_mla_insert", [&] {
        auto launch = [&](auto kernel) {
          kk3::launchPdl(
              kernel, num_tokens, num_heads, stream,
              reinterpret_cast<scalar_t const*>(ql_nope.const_data_ptr()),
              ql_nope.stride(0), ql_nope.stride(1),
              reinterpret_cast<scalar_t const*>(q_pe.const_data_ptr()),
              q_pe.stride(0), q_pe.stride(1),
              reinterpret_cast<scalar_t const*>(kv_c_normed.const_data_ptr()),
              kv_c_normed.stride(0),
              reinterpret_cast<scalar_t const*>(k_pe.const_data_ptr()),
              k_pe.stride(0),
              reinterpret_cast<scalar_t*>(mqa_q.mutable_data_ptr()),
              mqa_q.stride(0), mqa_q.stride(1),
              reinterpret_cast<uint8_t*>(k_cache.mutable_data_ptr()),
              k_cache.stride(0), k_cache.stride(1),
              slot_mapping.const_data_ptr<int64_t>(), num_tokens, num_heads,
              static_cast<int>(cache_block_size),
              apply_rope ? position_ids.value().const_data_ptr<int64_t>()
                         : nullptr,
              apply_rope ? reinterpret_cast<float const*>(
                               cos_sin_cache.value().const_data_ptr())
                         : nullptr);
        };
        if (apply_rope) {
          launch(kk3::fusedKimiK3MLADecodeQConcatDsMlaKernel<scalar_t, true>);
        } else {
          launch(kk3::fusedKimiK3MLADecodeQConcatDsMlaKernel<scalar_t, false>);
        }
      });
}
