/*
 * Copyright (c) 2024, The vLLM team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_bf16.h>
#include "cuda_compat.h"

#include <algorithm>
#include "../attention/dtype_fp8.cuh"
#include "../quantization/fp8/amd/quant_utils.cuh"

#if defined(__HIPCC__) && (defined(__gfx90a__) || defined(__gfx940__) || \
                           defined(__gfx941__) || defined(__gfx942__))
  #define __HIP__MI300_MI250__
#endif

#if defined(NDEBUG)
  #undef NDEBUG
  #include <assert.h>
  #define UNREACHABLE_CODE assert(false);
  #define NDEBUG
#else
  #define UNREACHABLE_CODE assert(false);
#endif

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

#if defined(__HIP__MI300_MI250__)  // TODO: Add NAVI support

  #define GCN_MFMA_INSTR1 __builtin_amdgcn_mfma_f32_16x16x4f32
  #define GCN_MFMA_INSTR __builtin_amdgcn_mfma_f32_4x4x4f16

using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using float16x4 =
    __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
typedef float16x4 _Half4;
typedef struct _Half8 {
  _Half4 xy[2];
} _Half8;

using bit16_t = uint16_t;
using bit16x4 = __attribute__((__vector_size__(4 * sizeof(uint16_t)))) uint16_t;
typedef bit16x4 _B16x4;
typedef struct _B16x8 {
  _B16x4 xy[2];
} _B16x8;

using _B8x8 = uint2;

////// Non temporal load stores ///////

template <typename T>
__device__ __forceinline__ T load(T* addr) {
  return addr[0];
}

template <typename T>
__device__ __forceinline__ void store(T value, T* addr) {
  addr[0] = value;
}

template <typename T, int absz, int cbid, int blgp>
__device__ __forceinline__ floatx4 gcn_mfma_instr(const _B16x4& inpA,
                                                  const _B16x4& inpB,
                                                  const floatx4& inpC) {
  if constexpr (std::is_same<T, _Float16>::value) {
    return __builtin_amdgcn_mfma_f32_4x4x4f16(inpA, inpB, inpC, absz, cbid,
                                              blgp);
  } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
    return __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(inpA, inpB, inpC, absz, cbid,
                                                  blgp);
  } else {
    static_assert(false, "unsupported 16b dtype");
  }
}

template <typename T>
__device__ __forceinline__ float to_float(const T& inp) {
  if constexpr (std::is_same<T, _Float16>::value) {
    return (float)inp;
  } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
    return __bfloat162float(inp);
  } else {
    static_assert(false, "unsupported 16b dtype");
  }
}

template <typename T>
__device__ __forceinline__ T from_float(const float& inp) {
  if constexpr (std::is_same<T, _Float16>::value) {
    return (_Float16)inp;
  } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
    return __float2bfloat16(inp);
  } else {
    static_assert(false, "unsupported 16b dtype");
  }
}

template <typename T>
__device__ __forceinline__ _B16x4 from_floatx4(const floatx4& inp) {
  union tmpcvt {
    uint16_t u;
    _Float16 f;
    __hip_bfloat16 b;
  } t16;
  _B16x4 ret;
  if constexpr (std::is_same<T, _Float16>::value) {
  #pragma unroll
    for (int i = 0; i < 4; i++) {
      t16.f = (_Float16)inp[i];
      ret[i] = t16.u;
    }
    return ret;
  } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
  #pragma unroll
    for (int i = 0; i < 4; i++) {
      t16.b = __float2bfloat16(inp[i]);
      ret[i] = t16.u;
    }
    return ret;
  } else {
    static_assert(false, "unsupported 16b dtype");
  }
}

template <typename T>
__device__ __forceinline__ _B16x4 addx4(const _B16x4& inp1,
                                        const _B16x4& inp2) {
  union tmpcvt {
    uint16_t u;
    _Float16 f;
    __hip_bfloat16 b;
  } t1, t2, res;
  _B16x4 ret;
  if constexpr (std::is_same<T, _Float16>::value) {
  #pragma unroll
    for (int i = 0; i < 4; i++) {
      t1.u = inp1[i];
      t2.u = inp2[i];
      res.f = t1.f + t2.f;
      ret[i] = res.u;
    }
    return ret;
  } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
  #pragma unroll
    for (int i = 0; i < 4; i++) {
      t1.u = inp1[i];
      t2.u = inp2[i];
      res.b = t1.b + t2.b;
      ret[i] = res.u;
    }
    return ret;
  } else {
    static_assert(false, "unsupported 16b dtype");
  }
}

template <typename T, vllm::Fp8KVCacheDataType KV_DTYPE>
__device__ __forceinline__ _B16x8 scaled_convert_b8x8(const _B8x8 input,
                                                      const float scale) {
  union alignas(16) {
    uint4 u4;
    _B16x8 u16x8;
    vllm::bf16_8_t b16x8;
  } tmp;
  if constexpr (std::is_same<T, _Float16>::value) {
    tmp.u4 = vllm::fp8::scaled_convert<uint4, _B8x8, KV_DTYPE>(input, scale);
    return tmp.u16x8;
  } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
    tmp.b16x8 = vllm::fp8::scaled_convert<vllm::bf16_8_t, _B8x8, KV_DTYPE>(
        input, scale);
    return tmp.u16x8;
  } else {
    static_assert(false, "unsupported 16b dtype");
  }
}

///////////////////////////////////////

// grid (num_seqs, num_partitions,num_heads/gqa_ratio)
// block (partition size)
template <typename scalar_t, typename cache_t,
          vllm::Fp8KVCacheDataType KV_DTYPE, int BLOCK_SIZE, int HEAD_SIZE,
          int NUM_THREADS,
          int GQA_RATIO>
__global__ __launch_bounds__(NUM_THREADS) void paged_attention_ll4mi_QKV_kernel(
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
    const int num_kv_heads, const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ context_lens,  // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    float* __restrict__ exp_sums,  // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,  // [num_seqs, num_heads,
                                     // max_num_partitions]
    scalar_t* __restrict__ out,  // [num_seqs, num_heads, max_num_partitions,
                                 // head_size]
    scalar_t* __restrict__ final_out,  // [num_seqs, num_heads, head_size]
    int max_ctx_blocks, float k_scale, float v_scale) {
  constexpr int NWARPS = NUM_THREADS / WARP_SIZE;
  const int warpid = threadIdx.x / WARP_SIZE;
  const int laneid = threadIdx.x % WARP_SIZE;
  const int lane4id = laneid % 4;

  const int seq_idx = blockIdx.x;
  const int partition_idx = blockIdx.y;
  const int partition_size = blockDim.x;
  const int max_num_partitions = gridDim.y;

  const int context_len = context_lens[seq_idx];
  const int partition_start_token_idx = partition_idx * partition_size;
  // exit if partition is out of context for seq
  if (partition_start_token_idx >= context_len) {
    return;
  }
  constexpr int QHLOOP =
      DIVIDE_ROUND_UP(GQA_RATIO, 4);  // each 4 lanes fetch 4 different qheads,
                                      // total qheads =8, so qhloop is 2
  constexpr int GQA_RATIO4 = 4 * QHLOOP;
  __shared__ float shared_qk_max[NWARPS][GQA_RATIO4 + 1];
  __shared__ float shared_exp_sum[NWARPS][GQA_RATIO4 + 1];
  _B16x8 Qlocal[QHLOOP];
  constexpr int x = 16 / sizeof(scalar_t);
  constexpr int KHELOOP = HEAD_SIZE / x;
  _B16x8 Klocal[KHELOOP];
  _B8x8 Klocalb8[KHELOOP];
  constexpr int VHELOOP =
      HEAD_SIZE /
      WARP_SIZE;  // v head_size dimension is distributed across lanes
  constexpr int VTLOOP = 8;  // 16 separate 4xtokens across warp -> 16/2
                             // 8xtokens
  _B16x8 Vlocal[VHELOOP][VTLOOP];
  _B8x8 Vlocalb8[VHELOOP][VTLOOP];
  floatx4 dout[QHLOOP];
  float qk_max[QHLOOP];
  #pragma unroll
  for (int h = 0; h < QHLOOP; h++) {
    dout[h] = {0};
    qk_max[h] = -FLT_MAX;
  }

  const int wg_start_head_idx = blockIdx.z * GQA_RATIO;
  const int wg_start_kv_head_idx = blockIdx.z;

  const int warp_start_token_idx =
      partition_start_token_idx + warpid * WARP_SIZE;

  if (warp_start_token_idx >= context_len) {  // warp out of context
  #pragma unroll
    for (int h = 0; h < GQA_RATIO4; h++) {
      shared_qk_max[warpid][h] = -FLT_MAX;
      shared_exp_sum[warpid][h] = 0.0f;
    }
  } else {  // warp within context

    const int num_context_blocks = DIVIDE_ROUND_UP(context_len, BLOCK_SIZE);
    const int last_ctx_block = num_context_blocks - 1;

    const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;

    const int local_token_idx = threadIdx.x;
    const int global_token_idx = partition_start_token_idx + local_token_idx;

    const int block_idx = (global_token_idx < context_len)
                              ? global_token_idx / BLOCK_SIZE
                              : last_ctx_block;
    // fetch block number for q and k
    // int32 physical_block_number leads to overflow when multiplied with
    // kv_block_stride
    const int64_t physical_block_number =
        static_cast<int64_t>(block_table[block_idx]);

    // fetch vphysical block numbers up front
    constexpr int VBLOCKS = 8 * VTLOOP / BLOCK_SIZE;
    int vphysical_blocks[VBLOCKS];

    const int warp_start_block_idx = warp_start_token_idx / BLOCK_SIZE;
  #pragma unroll
    for (int b = 0; b < VBLOCKS; b++) {
      const int vblock_idx = warp_start_block_idx + b;
      const int vblock_idx_ctx =
          (vblock_idx <= last_ctx_block) ? vblock_idx : last_ctx_block;
      vphysical_blocks[b] = block_table[vblock_idx_ctx];
    }

    // each 4 lanes fetch 8 helems, so warp fetches 8*16 = 128 helems
    const scalar_t* q_ptr =
        q + seq_idx * q_stride + wg_start_head_idx * HEAD_SIZE;
    const _B16x8* q_ptrh8 = reinterpret_cast<const _B16x8*>(q_ptr);
    const int qhead_elemh8 = laneid / 4;
  #pragma unroll
    for (int h = 0; h < QHLOOP - 1; h++) {
      const int qhead_idx = h * 4 + lane4id;
      Qlocal[h] = q_ptrh8[qhead_idx * HEAD_SIZE / 8 + qhead_elemh8];
    }
    const int final_qhead_idx = 4 * (QHLOOP - 1) + lane4id;
    if (final_qhead_idx < GQA_RATIO) {
      Qlocal[QHLOOP - 1] =
          q_ptrh8[final_qhead_idx * HEAD_SIZE / 8 + qhead_elemh8];
    } else {
      Qlocal[QHLOOP - 1].xy[0] = {0};
      Qlocal[QHLOOP - 1].xy[1] = {0};
    }

    const cache_t* k_ptr = k_cache + physical_block_number * kv_block_stride +
                           wg_start_kv_head_idx * kv_head_stride;

    const int physical_block_offset =
        local_token_idx % BLOCK_SIZE;  // since x=half8, physical_block_offset
                                       // is already cast as _H8
    if constexpr (KV_DTYPE == vllm::Fp8KVCacheDataType::kAuto) {
      const _B16x8* k_ptrh8 = reinterpret_cast<const _B16x8*>(k_ptr);
  #pragma unroll
      for (int d = 0; d < KHELOOP; d++) {
        Klocal[d] = k_ptrh8[d * BLOCK_SIZE + physical_block_offset];
      }
    } else {
      constexpr int X = 16 / sizeof(cache_t);
      const cache_t* k_ptr2 = k_ptr + physical_block_offset * X;
  #pragma unroll
      for (int d = 0; d < KHELOOP; d++) {
        const int head_elem = d * 8;
        const int offset1 = head_elem / X;
        const int offset2 = head_elem % X;
        const cache_t* k_ptr3 = k_ptr2 + offset1 * BLOCK_SIZE * X + offset2;
        Klocalb8[d] = *reinterpret_cast<const _B8x8*>(k_ptr3);
      }
    }

    float alibi_slope[QHLOOP];
    if (alibi_slopes != nullptr) {
  #pragma unroll
      for (int h = 0; h < QHLOOP; h++) {
        const int qhead_idx = h * 4 + lane4id;
        alibi_slope[h] = (qhead_idx < GQA_RATIO)
                             ? alibi_slopes[wg_start_head_idx + qhead_idx]
                             : 0.f;
      }
    }

    const cache_t* v_ptr = v_cache + wg_start_kv_head_idx * kv_head_stride;
    if constexpr (KV_DTYPE == vllm::Fp8KVCacheDataType::kAuto) {
      const _B16x8* v_ptrh8 = reinterpret_cast<const _B16x8*>(v_ptr);
      // iterate over each v block
  #pragma unroll
      for (int b = 0; b < VBLOCKS; b++) {
        // int32 physical_block_number leads to overflow when multiplied with
        // kv_block_stride
        const int64_t vphysical_block_number =
            static_cast<int64_t>(vphysical_blocks[b]);
        const _B16x8* v_ptrh8b =
            v_ptrh8 + (vphysical_block_number * kv_block_stride) / 8;
        // iterate over each head elem (within head_size)
  #pragma unroll
        for (int h = 0; h < VHELOOP; h++) {
          const int head_size_elem = h * WARP_SIZE + laneid;
          const _B16x8* v_ptrh8be = v_ptrh8b + head_size_elem * BLOCK_SIZE / 8;
          // iterate over all velems within block
  #pragma unroll
          for (int d = 0; d < BLOCK_SIZE / 8; d++) {
            Vlocal[h][b * BLOCK_SIZE / 8 + d] = v_ptrh8be[d];
          }
        }
      }
    } else {
      const _B8x8* v_ptrh8 = reinterpret_cast<const _B8x8*>(v_ptr);
      // iterate over each v block
  #pragma unroll
      for (int b = 0; b < VBLOCKS; b++) {
        // int32 physical_block_number leads to overflow when multiplied with
        // kv_block_stride
        const int64_t vphysical_block_number =
            static_cast<int64_t>(vphysical_blocks[b]);
        const _B8x8* v_ptrh8b =
            v_ptrh8 + (vphysical_block_number * kv_block_stride) / 8;
        // iterate over each head elem (within head_size)
  #pragma unroll
        for (int h = 0; h < VHELOOP; h++) {
          const int head_size_elem = h * WARP_SIZE + laneid;
          const _B8x8* v_ptrh8be = v_ptrh8b + head_size_elem * BLOCK_SIZE / 8;
          // iterate over all velems within block
  #pragma unroll
          for (int d = 0; d < BLOCK_SIZE / 8; d++) {
            // Vlocalb8[h][b * BLOCK_SIZE / 8 + d] = v_ptrh8be[d];
            const _B8x8 Vlocalb8 = v_ptrh8be[d];
            Vlocal[h][b * BLOCK_SIZE / 8 + d] =
                scaled_convert_b8x8<scalar_t, KV_DTYPE>(Vlocalb8, v_scale);
          }
        }
      }
    }

    if constexpr (KV_DTYPE != vllm::Fp8KVCacheDataType::kAuto) {
  #pragma unroll
      for (int d = 0; d < KHELOOP; d++) {
        Klocal[d] =
            scaled_convert_b8x8<scalar_t, KV_DTYPE>(Klocalb8[d], k_scale);
      }
    }

  #pragma unroll
    for (int h = 0; h < QHLOOP; h++) {
      dout[h] = gcn_mfma_instr<scalar_t, 4, 0, 0>(Qlocal[h].xy[0],
                                                  Klocal[0].xy[0], dout[h]);
      dout[h] = gcn_mfma_instr<scalar_t, 4, 0, 0>(Qlocal[h].xy[1],
                                                  Klocal[0].xy[1], dout[h]);
      dout[h] = gcn_mfma_instr<scalar_t, 4, 1, 0>(Qlocal[h].xy[0],
                                                  Klocal[1].xy[0], dout[h]);
      dout[h] = gcn_mfma_instr<scalar_t, 4, 1, 0>(Qlocal[h].xy[1],
                                                  Klocal[1].xy[1], dout[h]);
      dout[h] = gcn_mfma_instr<scalar_t, 4, 2, 0>(Qlocal[h].xy[0],
                                                  Klocal[2].xy[0], dout[h]);
      dout[h] = gcn_mfma_instr<scalar_t, 4, 2, 0>(Qlocal[h].xy[1],
                                                  Klocal[2].xy[1], dout[h]);
      dout[h] = gcn_mfma_instr<scalar_t, 4, 3, 0>(Qlocal[h].xy[0],
                                                  Klocal[3].xy[0], dout[h]);
      dout[h] = gcn_mfma_instr<scalar_t, 4, 3, 0>(Qlocal[h].xy[1],
                                                  Klocal[3].xy[1], dout[h]);
      dout[h] = gcn_mfma_instr<scalar_t, 4, 4, 0>(Qlocal[h].xy[0],
                                                  Klocal[4].xy[0], dout[h]);
      dout[h] = gcn_mfma_instr<scalar_t, 4, 4, 0>(Qlocal[h].xy[1],
                                                  Klocal[4].xy[1], dout[h]);
      dout[h] = gcn_mfma_instr<scalar_t, 4, 5, 0>(Qlocal[h].xy[0],
                                                  Klocal[5].xy[0], dout[h]);
      dout[h] = gcn_mfma_instr<scalar_t, 4, 5, 0>(Qlocal[h].xy[1],
                                                  Klocal[5].xy[1], dout[h]);
      dout[h] = gcn_mfma_instr<scalar_t, 4, 6, 0>(Qlocal[h].xy[0],
                                                  Klocal[6].xy[0], dout[h]);
      dout[h] = gcn_mfma_instr<scalar_t, 4, 6, 0>(Qlocal[h].xy[1],
                                                  Klocal[6].xy[1], dout[h]);
      dout[h] = gcn_mfma_instr<scalar_t, 4, 7, 0>(Qlocal[h].xy[0],
                                                  Klocal[7].xy[0], dout[h]);
      dout[h] = gcn_mfma_instr<scalar_t, 4, 7, 0>(Qlocal[h].xy[1],
                                                  Klocal[7].xy[1], dout[h]);
      if constexpr (KHELOOP > 8) {
        dout[h] = gcn_mfma_instr<scalar_t, 4, 8, 0>(Qlocal[h].xy[0],
                                                    Klocal[8].xy[0], dout[h]);
        dout[h] = gcn_mfma_instr<scalar_t, 4, 8, 0>(Qlocal[h].xy[1],
                                                    Klocal[8].xy[1], dout[h]);
        dout[h] = gcn_mfma_instr<scalar_t, 4, 9, 0>(Qlocal[h].xy[0],
                                                    Klocal[9].xy[0], dout[h]);
        dout[h] = gcn_mfma_instr<scalar_t, 4, 9, 0>(Qlocal[h].xy[1],
                                                    Klocal[9].xy[1], dout[h]);
        dout[h] = gcn_mfma_instr<scalar_t, 4, 10, 0>(Qlocal[h].xy[0],
                                                     Klocal[10].xy[0], dout[h]);
        dout[h] = gcn_mfma_instr<scalar_t, 4, 10, 0>(Qlocal[h].xy[1],
                                                     Klocal[10].xy[1], dout[h]);
        dout[h] = gcn_mfma_instr<scalar_t, 4, 11, 0>(Qlocal[h].xy[0],
                                                     Klocal[11].xy[0], dout[h]);
        dout[h] = gcn_mfma_instr<scalar_t, 4, 11, 0>(Qlocal[h].xy[1],
                                                     Klocal[11].xy[1], dout[h]);
        dout[h] = gcn_mfma_instr<scalar_t, 4, 12, 0>(Qlocal[h].xy[0],
                                                     Klocal[12].xy[0], dout[h]);
        dout[h] = gcn_mfma_instr<scalar_t, 4, 12, 0>(Qlocal[h].xy[1],
                                                     Klocal[12].xy[1], dout[h]);
        dout[h] = gcn_mfma_instr<scalar_t, 4, 13, 0>(Qlocal[h].xy[0],
                                                     Klocal[13].xy[0], dout[h]);
        dout[h] = gcn_mfma_instr<scalar_t, 4, 13, 0>(Qlocal[h].xy[1],
                                                     Klocal[13].xy[1], dout[h]);
        dout[h] = gcn_mfma_instr<scalar_t, 4, 14, 0>(Qlocal[h].xy[0],
                                                     Klocal[14].xy[0], dout[h]);
        dout[h] = gcn_mfma_instr<scalar_t, 4, 14, 0>(Qlocal[h].xy[1],
                                                     Klocal[14].xy[1], dout[h]);
        dout[h] = gcn_mfma_instr<scalar_t, 4, 15, 0>(Qlocal[h].xy[0],
                                                     Klocal[15].xy[0], dout[h]);
        dout[h] = gcn_mfma_instr<scalar_t, 4, 15, 0>(Qlocal[h].xy[1],
                                                     Klocal[15].xy[1], dout[h]);
      }  // KHELOOP>8
      dout[h] *= scale;
    }
  // transpose dout so that 4 token ids are in each lane, and 4 heads are across
  // 4 lanes
  #pragma unroll
    for (int h = 0; h < QHLOOP; h++) {
      floatx4 tmp = {0};
  #pragma unroll
      for (int i = 0; i < 4; i++) {
        const float B = (lane4id == i) ? 1.0f : 0.0f;
        // const float A = (global_token_idx < context_len) ? dout[h][i] : 0.0f;
        tmp = __builtin_amdgcn_mfma_f32_4x4x1f32(dout[h][i], B, tmp, 0, 0, 0);
        // tmp = __builtin_amdgcn_mfma_f32_4x4x1f32(A, B, tmp, 0, 0, 0);
      }
      dout[h] = tmp;
    }

    const int lane4_token_idx = 4 * (global_token_idx >> 2);
    const int alibi_offset = lane4_token_idx - context_len + 1;
    if (alibi_slopes != nullptr) {
  #pragma unroll
      for (int h = 0; h < QHLOOP; h++) {
  #pragma unroll
        for (int i = 0; i < 4; i++) {
          dout[h][i] += alibi_slope[h] * (alibi_offset + i);
        }
      }
    }

  #pragma unroll
    for (int h = 0; h < QHLOOP; h++) {
      qk_max[h] = -FLT_MAX;
  #pragma unroll
      for (int i = 0; i < 4; i++) {
        qk_max[h] = (lane4_token_idx + i < context_len)
                        ? fmaxf(qk_max[h], dout[h][i])
                        : qk_max[h];
      }
  #pragma unroll
      for (int mask = WARP_SIZE / 2; mask >= 4; mask /= 2) {
        qk_max[h] = fmaxf(qk_max[h], __shfl_xor(qk_max[h], mask));
      }
    }

    float exp_sum[QHLOOP];
  #pragma unroll
    for (int h = 0; h < QHLOOP; h++) {
      exp_sum[h] = 0.0f;
  #pragma unroll
      for (int i = 0; i < 4; i++) {
        dout[h][i] = (lane4_token_idx + i < context_len)
                         ? __expf(dout[h][i] - qk_max[h])
                         : 0.0f;
        exp_sum[h] += dout[h][i];
      }
  #pragma unroll
      for (int mask = WARP_SIZE / 2; mask >= 4; mask /= 2) {
        exp_sum[h] += __shfl_xor(exp_sum[h], mask);
      }
    }

  #pragma unroll
    for (int h = 0; h < QHLOOP; h++) {
      const int head_idx = 4 * h + lane4id;
      shared_qk_max[warpid][head_idx] = qk_max[h];
      shared_exp_sum[warpid][head_idx] = exp_sum[h];
    }
  }  // warp within context

  __syncthreads();

  const int num_heads = gridDim.z * GQA_RATIO;
  float* max_logits_ptr =
      max_logits + seq_idx * num_heads * max_num_partitions + partition_idx;
  float* exp_sums_ptr =
      exp_sums + seq_idx * num_heads * max_num_partitions + partition_idx;
  #pragma unroll
  for (int h = 0; h < QHLOOP; h++) {
    float global_qk_max = -FLT_MAX;
    float warp_qk_max[NWARPS];
    const int head_idx = 4 * h + lane4id;
  #pragma unroll
    for (int w = 0; w < NWARPS; w++) {
      warp_qk_max[w] = shared_qk_max[w][head_idx];
      global_qk_max = fmaxf(global_qk_max, warp_qk_max[w]);
    }
    float global_exp_sum = 0.0f;
  #pragma unroll
    for (int w = 0; w < NWARPS; w++) {
      global_exp_sum +=
          shared_exp_sum[w][head_idx] * __expf(warp_qk_max[w] - global_qk_max);
    }
    if (head_idx < GQA_RATIO) {
      max_logits_ptr[(wg_start_head_idx + head_idx) * max_num_partitions] =
          global_qk_max;
      exp_sums_ptr[(wg_start_head_idx + head_idx) * max_num_partitions] =
          global_exp_sum;
    }
    const float global_inv_sum_scale = __fdividef(1.f, global_exp_sum + 1e-6f) *
                                       __expf(qk_max[h] - global_qk_max);
    dout[h] *= global_inv_sum_scale;
  }
  // logits[h] -> every 4 lanes hold 4 heads, each lane holds 4 tokens, there
  // are 4x16 tokens across warp
  _B16x4 logits[QHLOOP];
  #pragma unroll
  for (int h = 0; h < QHLOOP; h++) {
    logits[h] = from_floatx4<scalar_t>(dout[h]);
  }

  __shared__ _B16x4 vout_shared[QHLOOP][VHELOOP][WARP_SIZE][NWARPS + 1];

  if (warp_start_token_idx >= context_len) {  // warp out of context
  #pragma unroll
    for (int qh = 0; qh < QHLOOP; qh++) {
  #pragma unroll
      for (int vh = 0; vh < VHELOOP; vh++) {
        vout_shared[qh][vh][laneid][warpid] = {0};
      }
    }
  } else {  // warp in context
  // iterate across heads
  #pragma unroll
    for (int qh = 0; qh < QHLOOP; qh++) {
  // iterate over each v head elem (within head_size)
  #pragma unroll
      for (int vh = 0; vh < VHELOOP; vh++) {
        floatx4 acc = {0};
        // iterate over tokens
        acc = gcn_mfma_instr<scalar_t, 4, 0, 0>(logits[qh], Vlocal[vh][0].xy[0],
                                                acc);
        acc = gcn_mfma_instr<scalar_t, 4, 1, 0>(logits[qh], Vlocal[vh][0].xy[1],
                                                acc);
        acc = gcn_mfma_instr<scalar_t, 4, 2, 0>(logits[qh], Vlocal[vh][1].xy[0],
                                                acc);
        acc = gcn_mfma_instr<scalar_t, 4, 3, 0>(logits[qh], Vlocal[vh][1].xy[1],
                                                acc);
        acc = gcn_mfma_instr<scalar_t, 4, 4, 0>(logits[qh], Vlocal[vh][2].xy[0],
                                                acc);
        acc = gcn_mfma_instr<scalar_t, 4, 5, 0>(logits[qh], Vlocal[vh][2].xy[1],
                                                acc);
        acc = gcn_mfma_instr<scalar_t, 4, 6, 0>(logits[qh], Vlocal[vh][3].xy[0],
                                                acc);
        acc = gcn_mfma_instr<scalar_t, 4, 7, 0>(logits[qh], Vlocal[vh][3].xy[1],
                                                acc);
        acc = gcn_mfma_instr<scalar_t, 4, 8, 0>(logits[qh], Vlocal[vh][4].xy[0],
                                                acc);
        acc = gcn_mfma_instr<scalar_t, 4, 9, 0>(logits[qh], Vlocal[vh][4].xy[1],
                                                acc);
        acc = gcn_mfma_instr<scalar_t, 4, 10, 0>(logits[qh],
                                                 Vlocal[vh][5].xy[0], acc);
        acc = gcn_mfma_instr<scalar_t, 4, 11, 0>(logits[qh],
                                                 Vlocal[vh][5].xy[1], acc);
        acc = gcn_mfma_instr<scalar_t, 4, 12, 0>(logits[qh],
                                                 Vlocal[vh][6].xy[0], acc);
        acc = gcn_mfma_instr<scalar_t, 4, 13, 0>(logits[qh],
                                                 Vlocal[vh][6].xy[1], acc);
        acc = gcn_mfma_instr<scalar_t, 4, 14, 0>(logits[qh],
                                                 Vlocal[vh][7].xy[0], acc);
        acc = gcn_mfma_instr<scalar_t, 4, 15, 0>(logits[qh],
                                                 Vlocal[vh][7].xy[1], acc);
        vout_shared[qh][vh][laneid][warpid] = from_floatx4<scalar_t>(acc);
      }
    }
  }  // warp in context

  __syncthreads();

  if (warpid == 0) {
    _B16x4 vout[QHLOOP][VHELOOP];
    // iterate across heads
    scalar_t* out_ptr;
    int out_num_partitions;
    if (context_len > partition_size) {
      out_num_partitions = max_num_partitions;
      out_ptr = out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
                partition_idx * HEAD_SIZE;
    } else {
      out_num_partitions = 1;
      out_ptr = final_out + seq_idx * num_heads * HEAD_SIZE;
    }
  #pragma unroll
    for (int qh = 0; qh < QHLOOP; qh++) {
  // iterate over each v head elem (within head_size)
  #pragma unroll
      for (int vh = 0; vh < VHELOOP; vh++) {
        vout[qh][vh] = {0};
  #pragma unroll
        for (int w = 0; w < NWARPS; w++) {
          vout[qh][vh] =
              addx4<scalar_t>(vout[qh][vh], vout_shared[qh][vh][laneid][w]);
        }
        const int head_size_elem = vh * WARP_SIZE + laneid;
        bit16_t* out_ptr_b16 = reinterpret_cast<bit16_t*>(out_ptr);
  #pragma unroll
        for (int i = 0; i < 4; i++) {
          const int head_idx = 4 * qh + i;
          if (head_idx < GQA_RATIO) {
            out_ptr_b16[(wg_start_head_idx + head_idx) * out_num_partitions *
                            HEAD_SIZE +
                        head_size_elem] = vout[qh][vh][i];
          }
        }
      }
    }
  }
}

// Grid: (num_heads, num_seqs).
template <typename scalar_t, int HEAD_SIZE, int NUM_THREADS,
          int PARTITION_SIZE>
__global__
__launch_bounds__(NUM_THREADS) void paged_attention_ll4mi_reduce_kernel(
    scalar_t* __restrict__ out,            // [num_seqs, num_heads, head_size]
    const float* __restrict__ exp_sums,    // [num_seqs, num_heads,
                                           // max_num_partitions]
    const float* __restrict__ max_logits,  // [num_seqs, num_heads,
                                           // max_num_partitions]
    const scalar_t* __restrict__ tmp_out,  // [num_seqs, num_heads,
                                           // max_num_partitions, head_size]
    const int* __restrict__ context_lens,  // [num_seqs]
    const int max_num_partitions) {
  const int num_heads = gridDim.x;
  const int head_idx = blockIdx.x;
  const int seq_idx = blockIdx.y;
  const int context_len = context_lens[seq_idx];
  const int num_partitions = DIVIDE_ROUND_UP(context_len, PARTITION_SIZE);
  if (num_partitions == 1) {
    // if num_partitions==1, main kernel will write to out directly, no work in
    // reduction kernel
    return;
  }

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int warpid = threadIdx.x / WARP_SIZE;
  const int laneid = threadIdx.x % WARP_SIZE;

  __shared__ float shared_global_exp_sum;
  __shared__ float shared_exp_sums[2 * WARP_SIZE];

  if (warpid == 0) {
    const float* max_logits_ptr = max_logits +
                                  seq_idx * num_heads * max_num_partitions +
                                  head_idx * max_num_partitions;

    // valid partition is the last valid partition in case threadid > num
    // partitions
    const int valid_partition =
        (threadIdx.x < num_partitions) ? threadIdx.x : num_partitions - 1;
    const int valid_partition2 = (WARP_SIZE + threadIdx.x < num_partitions)
                                     ? WARP_SIZE + threadIdx.x
                                     : num_partitions - 1;
    float reg_max_logit = max_logits_ptr[valid_partition];
    float reg_max_logit2 = max_logits_ptr[valid_partition2];
    float max_logit = fmaxf(reg_max_logit, reg_max_logit2);

  #pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
      max_logit = fmaxf(max_logit, __shfl_xor(max_logit, mask));
    }

    const float* exp_sums_ptr = exp_sums +
                                seq_idx * num_heads * max_num_partitions +
                                head_idx * max_num_partitions;

    float global_exp_sum = 0.0f;
    float rescaled_exp_sum = exp_sums_ptr[valid_partition];
    float rescaled_exp_sum2 = exp_sums_ptr[valid_partition2];
    rescaled_exp_sum *=
        (threadIdx.x < num_partitions) ? expf(reg_max_logit - max_logit) : 0.0f;
    rescaled_exp_sum2 *= (threadIdx.x + WARP_SIZE < num_partitions)
                             ? expf(reg_max_logit2 - max_logit)
                             : 0.0f;
    global_exp_sum += rescaled_exp_sum + rescaled_exp_sum2;
    shared_exp_sums[threadIdx.x] = rescaled_exp_sum;
    shared_exp_sums[threadIdx.x + WARP_SIZE] = rescaled_exp_sum2;

  #pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
      global_exp_sum += __shfl_xor(global_exp_sum, mask);
    }
    if (threadIdx.x == 0) {
      shared_global_exp_sum = global_exp_sum;
    }
  }  // warpid == 0
  const scalar_t* tmp_out_ptr =
      tmp_out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
      head_idx * max_num_partitions * HEAD_SIZE + threadIdx.x;
  constexpr int MAX_NPAR = 64;
  scalar_t tmps[MAX_NPAR];
  const float dzero = 0.0f;
  #pragma unroll
  for (int j = 0; j < MAX_NPAR; j++) {
    tmps[j] = from_float<scalar_t>(dzero);
  }
  const int last_partition_offset = (num_partitions - 1) * HEAD_SIZE;
  const int num_partition_offset = (num_partitions)*HEAD_SIZE;
  int idx = 0;

  constexpr int JCHUNK = 16;

  #pragma unroll
  for (int j = 0; j < JCHUNK * HEAD_SIZE; j += HEAD_SIZE) {
    // lastj is last valid partition
    const int lastj_offset =
        (j < num_partition_offset) ? j : last_partition_offset;
    tmps[idx] = tmp_out_ptr[lastj_offset];
    idx++;
  }
  __syncthreads();

  if (num_partitions > JCHUNK) {
  #pragma unroll
    for (int j = JCHUNK * HEAD_SIZE; j < 2 * JCHUNK * HEAD_SIZE;
         j += HEAD_SIZE) {
      const int lastj_offset =
          (j < num_partition_offset) ? j : last_partition_offset;
      tmps[idx] = tmp_out_ptr[lastj_offset];
      idx++;
    }

    if (num_partitions > 2 * JCHUNK) {
  #pragma unroll
      for (int j = 2 * JCHUNK * HEAD_SIZE; j < MAX_NPAR * HEAD_SIZE;
           j += HEAD_SIZE) {
        const int lastj_offset =
            (j < num_partition_offset) ? j : last_partition_offset;
        tmps[idx] = tmp_out_ptr[lastj_offset];
        idx++;
      }
    }
  }  // num_partitions > JCHUNK

  // Aggregate tmp_out to out.
  float acc = 0.0f;
  #pragma unroll
  for (int j = 0; j < JCHUNK; j++) {
    acc += to_float<scalar_t>(tmps[j]) * shared_exp_sums[j];
  }
  if (num_partitions > JCHUNK) {
  #pragma unroll
    for (int j = JCHUNK; j < 2 * JCHUNK; j++) {
      acc += to_float<scalar_t>(tmps[j]) * shared_exp_sums[j];
    }
    if (num_partitions > 2 * JCHUNK) {
  #pragma unroll
      for (int j = 2 * JCHUNK; j < MAX_NPAR; j++) {
        acc += to_float<scalar_t>(tmps[j]) * shared_exp_sums[j];
      }
    }
  }

  if (num_partitions > MAX_NPAR) {
    idx = 0;
  #pragma unroll
    for (int j = MAX_NPAR * HEAD_SIZE; j < 2 * MAX_NPAR * HEAD_SIZE;
         j += HEAD_SIZE) {
      // lastj is last valid partition
      const int lastj_offset =
          (j < num_partition_offset) ? j : last_partition_offset;
      tmps[idx] = tmp_out_ptr[lastj_offset];
      idx++;
    }

  #pragma unroll
    for (int j = 0; j < MAX_NPAR; j++) {
      acc += to_float<scalar_t>(tmps[j]) * shared_exp_sums[j + MAX_NPAR];
    }
  }

  const float inv_global_exp_sum =
      __fdividef(1.0f, shared_global_exp_sum + 1e-6f);
  acc *= inv_global_exp_sum;
  scalar_t* out_ptr =
      out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
  out_ptr[threadIdx.x] = from_float<scalar_t>(acc);
}

#else  // !defined(__HIP__MI300_MI250__) TODO: Add NAVI support

template <typename scalar_t, typename cache_t,
          vllm::Fp8KVCacheDataType KV_DTYPE, int BLOCK_SIZE, int HEAD_SIZE,
          int NUM_THREADS,
          int GQA_RATIO>
__global__ __launch_bounds__(NUM_THREADS) void paged_attention_ll4mi_QKV_kernel(
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
    const int num_kv_heads, const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ context_lens,  // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    float* __restrict__ exp_sums,  // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,  // [num_seqs, num_heads,
                                     // max_num_partitions]
    scalar_t* __restrict__ out,  // [num_seqs, num_heads, max_num_partitions,
                                 // head_size]
    scalar_t* __restrict__ final_out,  // [num_seqs, num_heads, head_size]
    int max_ctx_blocks, float k_scale, float v_scale) {
  UNREACHABLE_CODE
}

// Grid: (num_heads, num_seqs).
template <typename scalar_t, int HEAD_SIZE, int NUM_THREADS,
          int PARTITION_SIZE>
__global__
__launch_bounds__(NUM_THREADS) void paged_attention_ll4mi_reduce_kernel(
    scalar_t* __restrict__ out,            // [num_seqs, num_heads, head_size]
    const float* __restrict__ exp_sums,    // [num_seqs, num_heads,
                                           // max_num_partitions]
    const float* __restrict__ max_logits,  // [num_seqs, num_heads,
                                           // max_num_partitions]
    const scalar_t* __restrict__ tmp_out,  // [num_seqs, num_heads,
                                           // max_num_partitions, head_size]
    const int* __restrict__ context_lens,  // [num_seqs]
    const int max_num_partitions){UNREACHABLE_CODE}

#endif  // defined(__HIP__MI300_MI250__) TODO: Add NAVI support

#define LAUNCH_CUSTOM_ATTENTION(GQA_RATIO)                                    \
  paged_attention_ll4mi_QKV_kernel<T, KVT, KV_DTYPE, BLOCK_SIZE, HEAD_SIZE,   \
                                   NTHR, GQA_RATIO>                           \
      <<<grid, block, 0, stream>>>(                                           \
          query_ptr, key_cache_ptr, value_cache_ptr, num_kv_heads, scale,     \
          block_tables_ptr, context_lens_ptr, max_num_blocks_per_seq,         \
          alibi_slopes_ptr, q_stride, kv_block_stride, kv_head_stride,        \
          exp_sums_ptr, max_logits_ptr, tmp_out_ptr, out_ptr, max_ctx_blocks, \
          k_scale, v_scale);

template <typename T, typename KVT, vllm::Fp8KVCacheDataType KV_DTYPE,
          int BLOCK_SIZE, int HEAD_SIZE, int PARTITION_SIZE = 512>
void paged_attention_custom_launcher(
    torch::Tensor& out, torch::Tensor& exp_sums, torch::Tensor& max_logits,
    torch::Tensor& tmp_out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, const int num_kv_heads, float scale,
    torch::Tensor& block_tables, torch::Tensor& context_lens,
    int max_context_len, const std::optional<torch::Tensor>& alibi_slopes,
    float k_scale, float v_scale) {
  int num_seqs = query.size(0);
  int num_heads = query.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);
  int q_stride = query.stride(0);
  int kv_block_stride = key_cache.stride(0);
  int kv_head_stride = key_cache.stride(1);

  // NOTE: alibi_slopes is optional.
  const float* alibi_slopes_ptr =
      alibi_slopes
          ? reinterpret_cast<const float*>(alibi_slopes.value().data_ptr())
          : nullptr;

  T* out_ptr = reinterpret_cast<T*>(out.data_ptr());
  float* exp_sums_ptr = reinterpret_cast<float*>(exp_sums.data_ptr());
  float* max_logits_ptr = reinterpret_cast<float*>(max_logits.data_ptr());
  T* tmp_out_ptr = reinterpret_cast<T*>(tmp_out.data_ptr());
  T* query_ptr = reinterpret_cast<T*>(query.data_ptr());
  KVT* key_cache_ptr = reinterpret_cast<KVT*>(key_cache.data_ptr());
  KVT* value_cache_ptr = reinterpret_cast<KVT*>(value_cache.data_ptr());
  int* block_tables_ptr = block_tables.data_ptr<int>();
  int* context_lens_ptr = context_lens.data_ptr<int>();

  const int max_ctx_blocks = DIVIDE_ROUND_UP(max_context_len, BLOCK_SIZE);
  const int max_num_partitions =
      DIVIDE_ROUND_UP(max_context_len, PARTITION_SIZE);
  const int gqa_ratio = num_heads / num_kv_heads;
  assert(num_heads % num_kv_heads == 0);
  assert(head_size == HEAD_SIZE);
  assert(max_num_partitions <= 128);

  constexpr int NTHR = PARTITION_SIZE;
  dim3 grid(num_seqs, max_num_partitions, num_kv_heads);
  dim3 block(NTHR);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  switch (gqa_ratio) {
    case 1:
      LAUNCH_CUSTOM_ATTENTION(1);
      break;
    case 2:
      LAUNCH_CUSTOM_ATTENTION(2);
      break;
    case 3:
      LAUNCH_CUSTOM_ATTENTION(3);
      break;
    case 4:
      LAUNCH_CUSTOM_ATTENTION(4);
      break;
    case 5:
      LAUNCH_CUSTOM_ATTENTION(5);
      break;
    case 6:
      LAUNCH_CUSTOM_ATTENTION(6);
      break;
    case 7:
      LAUNCH_CUSTOM_ATTENTION(7);
      break;
    case 8:
      LAUNCH_CUSTOM_ATTENTION(8);
      break;
    case 9:
      LAUNCH_CUSTOM_ATTENTION(9);
      break;
    case 10:
      LAUNCH_CUSTOM_ATTENTION(10);
      break;
    case 11:
      LAUNCH_CUSTOM_ATTENTION(11);
      break;
    case 12:
      LAUNCH_CUSTOM_ATTENTION(12);
      break;
    case 13:
      LAUNCH_CUSTOM_ATTENTION(13);
      break;
    case 14:
      LAUNCH_CUSTOM_ATTENTION(14);
      break;
    case 15:
      LAUNCH_CUSTOM_ATTENTION(15);
      break;
    case 16:
      LAUNCH_CUSTOM_ATTENTION(16);
      break;
    default:
      TORCH_CHECK(false, "Unsupported gqa ratio: ", gqa_ratio);
      break;
  }
  // dim3 grid2(num_heads,num_seqs,head_size/HEAD_ELEMS_PER_WG);
  // dim3 block2(1024);
  //  LAUNCH_CUSTOM_ATTENTION2;

  // reduction kernel is only required if max_context_len > partition size,
  // otherwise main kernel writes directly to final output
  //  note there are cases with graphing where max_context_len is the max
  //  supported by graphing, not the actual max among all the sequences: in that
  //  case reduction kernel will still run but return immediately
  if (max_context_len > PARTITION_SIZE) {
    dim3 reduce_grid(num_heads, num_seqs);
    dim3 reduce_block(head_size);
    paged_attention_ll4mi_reduce_kernel<T, HEAD_SIZE, HEAD_SIZE, PARTITION_SIZE>
        <<<reduce_grid, reduce_block, 0, stream>>>(
            out_ptr, exp_sums_ptr, max_logits_ptr, tmp_out_ptr,
            context_lens_ptr, max_num_partitions);
  }
}

#define CALL_CUSTOM_LAUNCHER(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE)       \
  paged_attention_custom_launcher<T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE>( \
      out, exp_sums, max_logits, tmp_out, query, key_cache, value_cache,  \
      num_kv_heads, scale, block_tables, context_lens, max_context_len,   \
      alibi_slopes, k_scale, v_scale);

#define CALL_CUSTOM_LAUNCHER_BLK(T, KVT, KV_DTYPE, HEAD_SIZE)     \
  switch (block_size) {                                           \
    case 16:                                                      \
      CALL_CUSTOM_LAUNCHER(T, KVT, KV_DTYPE, 16, HEAD_SIZE);      \
      break;                                                      \
    case 32:                                                      \
      CALL_CUSTOM_LAUNCHER(T, KVT, KV_DTYPE, 32, HEAD_SIZE);      \
      break;                                                      \
    default:                                                      \
      TORCH_CHECK(false, "Unsupported block size: ", block_size); \
      break;                                                      \
  }

#define CALL_CUSTOM_LAUNCHER_BLK_HEAD(T, KVT, KV_DTYPE)         \
  switch (head_size) {                                          \
    case 64:                                                    \
      CALL_CUSTOM_LAUNCHER_BLK(T, KVT, KV_DTYPE, 64);           \
      break;                                                    \
    case 128:                                                   \
      CALL_CUSTOM_LAUNCHER_BLK(T, KVT, KV_DTYPE, 128);          \
      break;                                                    \
    default:                                                    \
      TORCH_CHECK(false, "Unsupported head size: ", head_size); \
      break;                                                    \
  }

void paged_attention(
    torch::Tensor& out,         // [num_seqs, num_heads, head_size]
    torch::Tensor& exp_sums,    // [num_seqs, num_heads, max_num_partitions]
    torch::Tensor& max_logits,  // [num_seqs, num_heads, max_num_partitions]
    torch::Tensor&
        tmp_out,  // [num_seqs, num_heads, max_num_partitions, head_size]
    torch::Tensor& query,  // [num_seqs, num_heads, head_size]
    torch::Tensor&
        key_cache,  // [num_blocks, num_heads, head_size/x, block_size, x]
    torch::Tensor&
        value_cache,  // [num_blocks, num_heads, head_size, block_size]
    int64_t num_kv_heads, double scale,
    torch::Tensor& block_tables,  // [num_seqs, max_num_blocks_per_seq]
    torch::Tensor& context_lens,  // [num_seqs]
    int64_t block_size, int64_t max_context_len,
    const std::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, double k_scale, double v_scale) {
  const int head_size = query.size(2);
  if (kv_cache_dtype == "auto") {
    if (query.dtype() == at::ScalarType::Half) {
      CALL_CUSTOM_LAUNCHER_BLK_HEAD(_Float16, _Float16,
                                    vllm::Fp8KVCacheDataType::kAuto);
    } else if (query.dtype() == at::ScalarType::BFloat16) {
      CALL_CUSTOM_LAUNCHER_BLK_HEAD(__hip_bfloat16, __hip_bfloat16,
                                    vllm::Fp8KVCacheDataType::kAuto);
    } else {
      TORCH_CHECK(false, "Unsupported data type: ", query.dtype());
    }
  } else if (kv_cache_dtype == "fp8" || kv_cache_dtype == "fp8_e4m3") {
    if (query.dtype() == at::ScalarType::Half) {
      CALL_CUSTOM_LAUNCHER_BLK_HEAD(_Float16, uint8_t,
                                    vllm::Fp8KVCacheDataType::kFp8E4M3);
    } else if (query.dtype() == at::ScalarType::BFloat16) {
      CALL_CUSTOM_LAUNCHER_BLK_HEAD(__hip_bfloat16, uint8_t,
                                    vllm::Fp8KVCacheDataType::kFp8E4M3);
    } else {
      TORCH_CHECK(false, "Unsupported data type: ", query.dtype());
    }
  } else {
    TORCH_CHECK(false, "Unsupported KV cache dtype: ", kv_cache_dtype);
  }
}

#undef WARP_SIZE
#undef MAX
#undef MIN
#undef DIVIDE_ROUND_UP