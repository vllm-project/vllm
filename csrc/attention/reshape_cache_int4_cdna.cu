// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Per-token-head INT4 reshape_and_cache for CDNA.
//
// Inputs:
//   key, value      : fp16/bf16 [num_tokens, num_kv_heads, head_size]
//   key_cache       : uint8     [num_blocks, block_size, num_kv_heads, head_size/2]
//   value_cache     : same layout
//   k_scale_cache   : fp32      [num_blocks, block_size, num_kv_heads]
//   v_scale_cache   : same layout
//   slot_mapping    : int64     [num_tokens]  — flat slot index per token
//
// One CTA per token. WAVE threads compute per-(token, head) min/max, then
// derive (scale, zp). The same wave packs nibble pairs and writes them
// out, and packs the (scale, zp) fp32 with the steganography convention
// (low 4 bits = zp).

#include <cstdint>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include "paged_prefill_attn_cdna.cuh"

namespace vllm {
namespace reshape_cache_int4_cdna {

#if defined(__HIPCC__) && \
    (defined(__gfx90a__) || defined(__gfx942__) || defined(__gfx950__))

using vllm::prefill_attn_cdna::bf16_t;
using vllm::prefill_attn_cdna::to_float;

constexpr int LANES = 64;

__device__ __forceinline__ float wave64_max(float v) {
  v = fmaxf(v, __shfl_xor(v, 1));
  v = fmaxf(v, __shfl_xor(v, 2));
  v = fmaxf(v, __shfl_xor(v, 4));
  v = fmaxf(v, __shfl_xor(v, 8));
  v = fmaxf(v, __shfl_xor(v, 16));
  v = fmaxf(v, __shfl_xor(v, 32));
  return v;
}
__device__ __forceinline__ float wave64_min(float v) {
  v = fminf(v, __shfl_xor(v, 1));
  v = fminf(v, __shfl_xor(v, 2));
  v = fminf(v, __shfl_xor(v, 4));
  v = fminf(v, __shfl_xor(v, 8));
  v = fminf(v, __shfl_xor(v, 16));
  v = fminf(v, __shfl_xor(v, 32));
  return v;
}

template <typename T, int HEAD_SIZE>
__global__ __launch_bounds__(LANES, 8)
void reshape_and_cache_int4_kernel(
    const T* __restrict__ key, const T* __restrict__ value,
    uint8_t* __restrict__ key_cache, uint8_t* __restrict__ value_cache,
    float* __restrict__ k_scale_cache, float* __restrict__ v_scale_cache,
    const int64_t* __restrict__ slot_mapping,
    int num_kv_heads, int block_size,
    int64_t stride_k_token, int64_t stride_k_head,
    int64_t stride_v_token, int64_t stride_v_head,
    int64_t stride_kc_block, int64_t stride_kc_slot, int64_t stride_kc_head,
    int64_t stride_vc_block, int64_t stride_vc_slot, int64_t stride_vc_head,
    int64_t stride_ks_blk, int64_t stride_ks_slot, int64_t stride_ks_head,
    int64_t stride_vs_blk, int64_t stride_vs_slot, int64_t stride_vs_head) {
  int token_idx = blockIdx.x;
  int head_idx = blockIdx.y;
  int tid = threadIdx.x;

  int64_t slot = slot_mapping[token_idx];
  if (slot < 0) return;
  int blk = slot / block_size;
  int slot_in_blk = (int)(slot % block_size);

  // ----- Quantize K and write K cache  ---------------------------------
  // Each thread holds HEAD_SIZE/64 elements and computes its private
  // min/max, then we wave-reduce to a per-(token, head) min/max.
  constexpr int OWN = HEAD_SIZE / LANES;
  static_assert(HEAD_SIZE % LANES == 0,
                "HEAD_SIZE must be a multiple of 64");

  auto quant_and_pack = [&](const T* src_row, uint8_t* dst_packed,
                             float* dst_scale) {
    float vals[OWN];
    float lo = INFINITY, hi = -INFINITY;
    #pragma unroll
    for (int i = 0; i < OWN; ++i) {
      int d = tid * OWN + i;
      vals[i] = to_float<T>(src_row[d]);
      lo = fminf(lo, vals[i]);
      hi = fmaxf(hi, vals[i]);
    }
    lo = wave64_min(lo);
    hi = wave64_max(hi);
    float range = fmaxf(hi - lo, 1e-8f);
    float scale = range / 15.0f;
    int zp = (int)__float2int_rn(-lo / scale);
    if (zp < 0) zp = 0;
    if (zp > 15) zp = 15;

    // Quantize this thread's slice and pack into nibbles. Two consecutive
    // dim entries (2i, 2i+1) pack into one byte at byte_idx = d/2.
    // Each thread owns OWN dim entries starting at tid*OWN. For a pair
    // (2i, 2i+1) to be owned by the same thread, OWN must be even — true
    // when HEAD_SIZE >= 128 (OWN=2 for 128, OWN=1 for 64).
    //
    // HEAD_SIZE=64 path: OWN=1, so each thread owns one nibble. We need a
    // wave shuffle to bring neighbour nibbles together into a byte.
    if constexpr (OWN >= 2) {
      #pragma unroll
      for (int i = 0; i < OWN; i += 2) {
        int v0 = (int)__float2int_rn(vals[i]     / scale + (float)zp);
        int v1 = (int)__float2int_rn(vals[i + 1] / scale + (float)zp);
        v0 = max(0, min(15, v0));
        v1 = max(0, min(15, v1));
        int d = tid * OWN + i;
        dst_packed[d / 2] = (uint8_t)(v0 | (v1 << 4));
      }
    } else {
      // HEAD_SIZE=64 path. Each thread owns one nibble for one even-or-odd
      // dim slot. Use shfl to pair adjacent threads.
      int v0 = (int)__float2int_rn(vals[0] / scale + (float)zp);
      v0 = max(0, min(15, v0));
      // Pair thread tid (owning d=tid) with thread tid^1 (owning d=tid^1).
      int paired = __shfl_xor(v0, 1);
      bool is_even = (tid % 2) == 0;
      int lo_nib = is_even ? v0 : paired;
      int hi_nib = is_even ? paired : v0;
      if (is_even) {
        dst_packed[tid / 2] = (uint8_t)(lo_nib | (hi_nib << 4));
      }
    }

    // Lane 0 writes the steganographed scale-zp word.
    if (tid == 0) {
      uint32_t bits;
      __builtin_memcpy(&bits, &scale, 4);
      bits = (bits & ~0xFu) | ((uint32_t)zp & 0xFu);
      float out;
      __builtin_memcpy(&out, &bits, 4);
      *dst_scale = out;
    }
  };

  const T* key_row = key + (int64_t)token_idx * stride_k_token +
                     head_idx * stride_k_head;
  uint8_t* key_dst = key_cache + (int64_t)blk * stride_kc_block +
                     (int64_t)slot_in_blk * stride_kc_slot +
                     (int64_t)head_idx * stride_kc_head;
  float* key_scale_dst = k_scale_cache + (int64_t)blk * stride_ks_blk +
                         (int64_t)slot_in_blk * stride_ks_slot +
                         (int64_t)head_idx * stride_ks_head;
  quant_and_pack(key_row, key_dst, key_scale_dst);

  const T* val_row = value + (int64_t)token_idx * stride_v_token +
                     head_idx * stride_v_head;
  uint8_t* val_dst = value_cache + (int64_t)blk * stride_vc_block +
                     (int64_t)slot_in_blk * stride_vc_slot +
                     (int64_t)head_idx * stride_vc_head;
  float* val_scale_dst = v_scale_cache + (int64_t)blk * stride_vs_blk +
                         (int64_t)slot_in_blk * stride_vs_slot +
                         (int64_t)head_idx * stride_vs_head;
  quant_and_pack(val_row, val_dst, val_scale_dst);
}

template <typename T, int HEAD_SIZE>
void launch(const T* key, const T* value, uint8_t* key_cache,
            uint8_t* value_cache, float* k_scale_cache, float* v_scale_cache,
            const int64_t* slot_mapping, int num_tokens, int num_kv_heads,
            int block_size,
            int64_t stride_k_token, int64_t stride_k_head,
            int64_t stride_v_token, int64_t stride_v_head,
            int64_t stride_kc_block, int64_t stride_kc_slot,
            int64_t stride_kc_head,
            int64_t stride_vc_block, int64_t stride_vc_slot,
            int64_t stride_vc_head,
            int64_t stride_ks_blk, int64_t stride_ks_slot,
            int64_t stride_ks_head,
            int64_t stride_vs_blk, int64_t stride_vs_slot,
            int64_t stride_vs_head,
            cudaStream_t stream) {
  dim3 block(LANES);
  dim3 grid(num_tokens, num_kv_heads);
  reshape_and_cache_int4_kernel<T, HEAD_SIZE><<<grid, block, 0, stream>>>(
      key, value, key_cache, value_cache, k_scale_cache, v_scale_cache,
      slot_mapping, num_kv_heads, block_size,
      stride_k_token, stride_k_head, stride_v_token, stride_v_head,
      stride_kc_block, stride_kc_slot, stride_kc_head,
      stride_vc_block, stride_vc_slot, stride_vc_head,
      stride_ks_blk, stride_ks_slot, stride_ks_head,
      stride_vs_blk, stride_vs_slot, stride_vs_head);
}

#endif  // gfx90a/942/950

}  // namespace reshape_cache_int4_cdna
}  // namespace vllm

void reshape_and_cache_int4_cdna(
    torch::Tensor key, torch::Tensor value, torch::Tensor key_cache,
    torch::Tensor value_cache, torch::Tensor k_scale_cache,
    torch::Tensor v_scale_cache, torch::Tensor slot_mapping) {
#if defined(__HIPCC__) && \
    (defined(__gfx90a__) || defined(__gfx942__) || defined(__gfx950__))
  using namespace vllm::reshape_cache_int4_cdna;
  using vllm::prefill_attn_cdna::bf16_t;

  TORCH_CHECK(key.dtype() == at::kHalf || key.dtype() == at::kBFloat16);
  TORCH_CHECK(key_cache.dtype() == at::kByte);
  TORCH_CHECK(slot_mapping.dtype() == at::kLong);

  int num_tokens = key.size(0);
  int num_kv_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(1);
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  #define LAUNCH(T, HS)                                                        \
    launch<T, HS>((const T*)key.data_ptr(), (const T*)value.data_ptr(),        \
                  (uint8_t*)key_cache.data_ptr(),                              \
                  (uint8_t*)value_cache.data_ptr(),                            \
                  (float*)k_scale_cache.data_ptr(),                            \
                  (float*)v_scale_cache.data_ptr(),                            \
                  (const int64_t*)slot_mapping.data_ptr(),                     \
                  num_tokens, num_kv_heads, block_size,                        \
                  key.stride(0), key.stride(1),                                \
                  value.stride(0), value.stride(1),                            \
                  key_cache.stride(0), key_cache.stride(1),                    \
                  key_cache.stride(2),                                         \
                  value_cache.stride(0), value_cache.stride(1),                \
                  value_cache.stride(2),                                       \
                  k_scale_cache.stride(0), k_scale_cache.stride(1),            \
                  k_scale_cache.stride(2),                                     \
                  v_scale_cache.stride(0), v_scale_cache.stride(1),            \
                  v_scale_cache.stride(2),                                     \
                  stream)

  if (key.dtype() == at::kHalf) {
    using T = _Float16;
    switch (head_size) {
      case 64:  LAUNCH(T, 64);  break;
      case 128: LAUNCH(T, 128); break;
      default: TORCH_CHECK(false, "unsupported head_size=", head_size);
    }
  } else {
    using T = vllm::prefill_attn_cdna::bf16_t;
    switch (head_size) {
      case 64:  LAUNCH(T, 64);  break;
      case 128: LAUNCH(T, 128); break;
      default: TORCH_CHECK(false, "unsupported head_size=", head_size);
    }
  }
  #undef LAUNCH
#else
  TORCH_CHECK(false, "reshape_and_cache_int4_cdna requires gfx942/950/90a");
#endif
}
