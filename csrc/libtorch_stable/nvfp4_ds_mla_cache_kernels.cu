// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// nvfp4_ds_mla KV cache writer and gather/upconvert reader.
//
// Separate TU because its e2m1 conversions are only accepted by ptxas for a
// suffixed Blackwell target (sm_100a / sm_100f); only this file is compiled
// with the suffixed arch list.

#include "nvfp4_ds_mla_cache.h"

#include "quantization/fp4/nvfp4_utils.cuh"  // vllm::fp32_vec16_to_e2m1()

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12080
  #include <cuda_fp4.h>
  #define VLLM_HAS_CUDA_FP4 1
#else
  #define VLLM_HAS_CUDA_FP4 0
#endif

// __CUDA_ARCH_FAMILY_SPECIFIC__ is defined only for the suffixed targets. The
// non-native paths trap rather than emulate; the host entry points reject this
// format off SM100, so they are unreachable.
#if VLLM_HAS_CUDA_FP4 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && \
    defined(__CUDA_ARCH_FAMILY_SPECIFIC__)
  #define VLLM_NVFP4_NATIVE_CVT 1
#else
  #define VLLM_NVFP4_NATIVE_CVT 0
#endif

namespace vllm {

// nvfp4_ds_mla cache format, 352 B/token:
//   [0,   256)  512 x e2m1 NoPE, packed 2/byte (low nibble = even element)
//   [256, 320)  64  x e4m3 RoPE (unscaled: e4m3's 4 exponent bits span the
//               RoPE range unaided)
//   [320, 352)  32  x e4m3 NoPE scale factors (one per 16 elements), permuted
// Quantization:   sf = e4m3(max(amax_16 / 6, 2^-9)), q = round(x / float(sf))
// Dequantization: x = float(q) * float(sf)
//
// The scale-factor region is an 8x4 -> 4x8 transpose: element block s -- NoPE
// dims [16s, 16s + 16) -- has its scale at byte kSfNopeOff + nvfp4_sf_byte(s),
// and byte p holds the scale of element block 4 * (p & 7) + (p >> 3). Keep
// nvfp4_sf_byte() in lockstep with its counterpart in FlashMLA
// (csrc/sm100/decode/head64/config.h and tests/quant.py).
constexpr int kSfNopeOff = 256 + 64;

__host__ __device__ constexpr int nvfp4_sf_byte(int s) {
  return 8 * (s & 3) + (s >> 2);
}

// Thread q's scale groups {4c + q} must land on the contiguous bytes 8q + c.
// This also makes it a permutation of [0, 32), since both sides cover that
// range exactly once.
constexpr bool nvfp4_sf_perm_ok() {
  for (int q = 0; q < 4; ++q) {
    for (int c = 0; c < 8; ++c) {
      if (nvfp4_sf_byte(4 * c + q) != 8 * q + c) {
        return false;
      }
    }
  }
  return true;
}
static_assert(nvfp4_sf_perm_ok(),
              "NVFP4 SF byte order must match FlashMLA's nvfp4_sf_byte()");

// `cvt.rn.satfinite.e2m1x2.f32 d, a, b` puts a in d's HIGH nibble and b in its
// LOW nibble, so the odd element of a pair is the first source.
__device__ __forceinline__ uint64_t
nvfp4_pack16_e2m1_rn(const float (&vals)[16], float inv_sf) {
#if VLLM_NVFP4_NATIVE_CVT
  float2 pairs[8];
  #pragma unroll
  for (int i = 0; i < 8; i++) {
    pairs[i] = make_float2(vals[2 * i] * inv_sf, vals[2 * i + 1] * inv_sf);
  }
  const u32x2 packed = fp32_vec16_to_e2m1(pairs);
  return (static_cast<uint64_t>(packed.hi) << 32) | packed.lo;
#else
  __trap();
  return 0;
#endif
}

// The e2m1 data and its e4m3 scale both convert to f16 and multiply there; the
// product then goes f16 -> f32 -> bf16, as no f16 -> bf16 instruction exists.
__device__ __forceinline__ void nvfp4_unpack16_e2m1(uint64_t raw,
                                                    uint8_t sf_byte,
                                                    __nv_bfloat16 (&out)[16]) {
#if VLLM_NVFP4_NATIVE_CVT
  const __half2_raw sf2_raw = __nv_cvt_fp8x2_to_halfraw2(
      static_cast<__nv_fp8x2_storage_t>(sf_byte * 0x0101u), __NV_E4M3);
  const __half2 sf2 = *reinterpret_cast<const __half2*>(&sf2_raw);
  __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out);
  #pragma unroll
  for (int i = 0; i < 8; i++) {
    // low nibble -> .x, so .x = element 2i, matching the writer.
    const __half2_raw v_raw = __nv_cvt_fp4x2_to_halfraw2(
        static_cast<__nv_fp4x2_storage_t>(raw >> (8 * i)), __NV_E2M1);
    const __half2 v = __hmul2(*reinterpret_cast<const __half2*>(&v_raw), sf2);
    out2[i] = __float22bfloat162_rn(__half22float2(v));
  }
#else
  __trap();
#endif
}

// Rounded UP to the next e4m3 value so amax / sf never exceeds the payload
// range: round-to-nearest can land up to 33% low near e4m3's 2^-9 floor, which
// would saturate the tile's largest values.
__device__ __forceinline__ float nvfp4_tile_scale(const float* vals,
                                                  float max_val,
                                                  uint8_t* sf_out) {
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; i++) {
    amax = fmaxf(amax, fabsf(vals[i]));
  }
  const float sf_f = fmaxf(amax / max_val, 0.001953125f);  // 2^-9
  __nv_fp8_e4m3 sf8(sf_f);
  uint8_t sf_bits = *reinterpret_cast<const uint8_t*>(&sf8);
  if (float(sf8) < sf_f && sf_bits < 0x7E) {
    // Bump to the next e4m3 value (bit patterns of positive e4m3 values are
    // monotonic; 0x7E = 448 is the max finite value).
    sf_bits += 1;
    sf8 = *reinterpret_cast<const __nv_fp8_e4m3*>(&sf_bits);
  }
  *sf_out = sf_bits;
  return float(sf8);
}

// One CUDA block (64 threads) per token.
// Warp 0: 32 threads, one 16-element NoPE tile each.
// Warp 1: 32 threads, two RoPE elements each.
__global__ void concat_and_cache_nvfp4_ds_mla_kernel(
    const __nv_bfloat16* __restrict__ kv_c,    // [num_tokens, 512]
    const __nv_bfloat16* __restrict__ k_pe,    // [num_tokens, 64]
    uint8_t* __restrict__ kv_cache,            // [num_blocks, block_size, 352]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int block_stride,                    //
    const int entry_stride,                    //
    const int kv_c_stride,                     //
    const int k_pe_stride,                     //
    const int block_size) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  // NOTE: slot_idx can be -1 if the token is padded
  if (slot_idx < 0) {
    return;
  }
  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  uint8_t* token_ptr =
      kv_cache + block_idx * block_stride + block_offset * entry_stride;

  if (threadIdx.x < 32) {
    const int tile = threadIdx.x;
    const __nv_bfloat16* src = kv_c + token_idx * kv_c_stride + tile * 16;
    float vals[16];
#pragma unroll
    for (int i = 0; i < 16; i++) {
      vals[i] = __bfloat162float(src[i]);
    }
    // `tile` is the element-block index; its scale byte is permuted.
    const float sf = nvfp4_tile_scale(
        vals, 6.0f, token_ptr + kSfNopeOff + nvfp4_sf_byte(tile));
    const float inv_sf = 1.0f / sf;
    *reinterpret_cast<uint64_t*>(token_ptr + tile * 8) =
        nvfp4_pack16_e2m1_rn(vals, inv_sf);
  } else {
    // Plain e4m3, unscaled. All 32 lanes take 2 elements each, matching the
    // reader and concat_and_cache_ds_mla_kernel.
    const int lane = threadIdx.x - 32;
    const __nv_bfloat16* src = k_pe + token_idx * k_pe_stride + lane * 2;
    uint8_t q[2];
#pragma unroll
    for (int i = 0; i < 2; i++) {
      const __nv_fp8_e4m3 v8(__bfloat162float(src[i]));
      q[i] = *reinterpret_cast<const uint8_t*>(&v8);
    }
    *reinterpret_cast<uint16_t*>(token_ptr + 256 + lane * 2) =
        *reinterpret_cast<const uint16_t*>(q);
  }
}

// One warp per token (mirrors cp_gather_and_upconvert_fp8_kv_cache).
__global__ void cp_gather_and_upconvert_nvfp4_kv_cache_kernel(
    const uint8_t* __restrict__ src_cache,    // [NUM_BLOCKS, BLOCK_SIZE, 352]
    __nv_bfloat16* __restrict__ dst,          // [total_tokens, 576]
    const int32_t* __restrict__ block_table,  // [num_reqs, BLOCK_INDICES]
    const int32_t* __restrict__ workspace_starts,  // [num_reqs]
    const int32_t num_reqs, const int32_t block_size,
    const int32_t total_tokens, const int64_t block_table_stride,
    const int64_t cache_block_stride, const int64_t cache_entry_stride,
    const int64_t dst_entry_stride) {
  const int flat_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (flat_warp_id >= total_tokens) return;
  const int lane_id = threadIdx.x & 31;

  // Binary search to find which request owns this output token
  int lo = 0, hi = num_reqs - 1;
  while (lo < hi) {
    int mid = (lo + hi + 1) >> 1;
    if (workspace_starts[mid] <= flat_warp_id)
      lo = mid;
    else
      hi = mid - 1;
  }
  const int req_id = lo;

  const int out_token_id = flat_warp_id;
  const int token_offset = out_token_id - workspace_starts[req_id];
  const int cache_block_idx = token_offset / block_size;
  const int offset_in_block = token_offset % block_size;
  const int physical_block =
      block_table[req_id * block_table_stride + cache_block_idx];

  const uint8_t* token_ptr = src_cache + physical_block * cache_block_stride +
                             offset_in_block * cache_entry_stride;
  __nv_bfloat16* dst_ptr = dst + out_token_id * dst_entry_stride;

  // `lane_id` is the element-block index, so its scale byte is permuted.
  {
    const uint64_t raw =
        *reinterpret_cast<const uint64_t*>(token_ptr + lane_id * 8);
    const uint8_t sf_byte = token_ptr[kSfNopeOff + nvfp4_sf_byte(lane_id)];
    __nv_bfloat16 out[16];
    nvfp4_unpack16_e2m1(raw, sf_byte, out);
    int4* nope_dst = reinterpret_cast<int4*>(dst_ptr) + lane_id * 2;
    nope_dst[0] = *reinterpret_cast<const int4*>(&out[0]);
    nope_dst[1] = *reinterpret_cast<const int4*>(&out[8]);
  }

  // RoPE is unscaled e4m3.
  {
    const __nv_fp8_e4m3* raw =
        reinterpret_cast<const __nv_fp8_e4m3*>(token_ptr + 256 + lane_id * 2);
    const __nv_bfloat16 out[2] = {__float2bfloat16_rn(float(raw[0])),
                                  __float2bfloat16_rn(float(raw[1]))};
    *reinterpret_cast<uint32_t*>(dst_ptr + 512 + lane_id * 2) =
        *reinterpret_cast<const uint32_t*>(out);
  }
}

void launch_concat_and_cache_nvfp4_ds_mla(const void* kv_c, const void* k_pe,
                                          void* kv_cache,
                                          const int64_t* slot_mapping,
                                          int block_stride, int entry_stride,
                                          int kv_c_stride, int k_pe_stride,
                                          int block_size, int num_tokens,
                                          cudaStream_t stream) {
  dim3 grid(num_tokens);
  dim3 block(64);  // warp 0: 32 NoPE tiles; warp 1: 4 RoPE tiles
  concat_and_cache_nvfp4_ds_mla_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(kv_c),
      reinterpret_cast<const __nv_bfloat16*>(k_pe),
      reinterpret_cast<uint8_t*>(kv_cache), slot_mapping, block_stride,
      entry_stride, kv_c_stride, k_pe_stride, block_size);
}

void launch_cp_gather_and_upconvert_nvfp4_kv_cache(
    const uint8_t* src_cache, void* dst, const int32_t* block_table,
    const int32_t* workspace_starts, int32_t num_reqs, int32_t block_size,
    int32_t total_tokens, int64_t block_table_stride,
    int64_t cache_block_stride, int64_t cache_entry_stride,
    int64_t dst_entry_stride, cudaStream_t stream) {
  constexpr int warps_per_block = 8;
  const int grid_size = (total_tokens + warps_per_block - 1) / warps_per_block;
  const int block_size_threads = warps_per_block * 32;  // 256 threads
  cp_gather_and_upconvert_nvfp4_kv_cache_kernel<<<grid_size, block_size_threads,
                                                  0, stream>>>(
      src_cache, reinterpret_cast<__nv_bfloat16*>(dst), block_table,
      workspace_starts, num_reqs, block_size, total_tokens, block_table_stride,
      cache_block_stride, cache_entry_stride, dst_entry_stride);
}

}  // namespace vllm
