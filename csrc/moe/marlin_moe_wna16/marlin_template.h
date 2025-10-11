/*
 * Modified by Neural Magic
 * Copyright (C) Marlin.2024 Elias Frantar
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Adapted from https://github.com/IST-DASLab/marlin
 */

#ifndef MARLIN_NAMESPACE_NAME
  #define MARLIN_NAMESPACE_NAME marlin_moe_wna16
#endif

#include "quantization/gptq_marlin/marlin.cuh"
#include "quantization/gptq_marlin/marlin_dtypes.cuh"
#include "quantization/gptq_marlin/dequant.h"
#include "core/scalar_type.hpp"

#define STATIC_ASSERT_SCALAR_TYPE_VALID(scalar_t)               \
  static_assert(std::is_same<scalar_t, half>::value ||          \
                    std::is_same<scalar_t, nv_bfloat16>::value, \
                "only float16 and bfloat16 is supported");

namespace MARLIN_NAMESPACE_NAME {

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800

template <typename scalar_t,  // compute dtype, half or nv_float16
          const vllm::ScalarTypeId w_type_id,  // weight ScalarType id
          const int threads,          // number of threads in a threadblock
          const int thread_m_blocks,  // number of 16x16 blocks in the m
                                      // dimension (batchsize) of the
                                      // threadblock
          const int thread_n_blocks,  // same for n dimension (output)
          const int thread_k_blocks,  // same for k dimension (reduction)
          const bool m_block_size_8,  // whether m_block_size == 8
                                      // only works when thread_m_blocks == 1
          const int stages,  // number of stages for the async global->shared
                             // fetch pipeline
          const int group_blocks,  // number of consecutive 16x16 blocks
                                   // with a separate quantization scale
          const bool is_zp_float   // is zero point of float16 type?
          >
__global__ void Marlin(
    const int4* __restrict__ A,  // fp16 input matrix of shape mxk
    const int4* __restrict__ B,  // 4bit quantized weight matrix of shape kxn
    int4* __restrict__ C,        // fp16 output buffer of shape mxn
    int4* __restrict__ C_tmp,    // fp32 tmp output buffer (for reduce)
    const int4* __restrict__ scales_ptr,  // fp16 quantization scales of shape
                                          // (k/groupsize)xn
    const int4* __restrict__ zp_ptr,      // 4bit packed zero-points of shape
                                          // (k/groupsize)x(n/pack_factor)
    const int* __restrict__ g_idx,        // int32 group indices of shape k
    const int32_t* __restrict__ sorted_token_ids_ptr,        // moe sorted_ids
    const int32_t* __restrict__ expert_ids_ptr,              // moe expert ids
    const int32_t* __restrict__ num_tokens_past_padded_ptr,  // moe num tokens
    const float* __restrict__ topk_weights_ptr,              // moe top weights
    int top_k,              // num of experts per token
    bool mul_topk_weights,  // mul topk weights or not
    bool is_ep,             // expert parallelism
    int num_groups,         // number of scale groups per output channel
    int prob_m,             // batch dimension m
    int prob_n,             // output dimension n
    int prob_k,             // reduction dimension k
    int* locks,             // extra global storage for barrier synchronization
    bool use_atomic_add,    // whether to use atomic add to reduce
    bool use_fp32_reduce,   // whether to use fp32 global reduce
    int max_shared_mem) {}

}  // namespace MARLIN_NAMESPACE_NAME

#else

// m16n8k16 tensor core mma instruction with fp16 inputs and fp32
// output/accumulation.
template <typename scalar_t>
__device__ inline void mma(const typename ScalarType<scalar_t>::FragA& a_frag,
                           const typename ScalarType<scalar_t>::FragB& frag_b,
                           typename ScalarType<scalar_t>::FragC& frag_c) {
  const uint32_t* a = reinterpret_cast<const uint32_t*>(&a_frag);
  const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
  float* c = reinterpret_cast<float*>(&frag_c);
  if constexpr (std::is_same<scalar_t, half>::value) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
  } else if constexpr (std::is_same<scalar_t, nv_bfloat16>::value) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
  } else {
    STATIC_ASSERT_SCALAR_TYPE_VALID(scalar_t);
  }
}

template <typename scalar_t>
__device__ inline void mma_trans(
    const typename ScalarType<scalar_t>::FragA& a_frag,
    const typename ScalarType<scalar_t>::FragB& frag_b,
    const typename ScalarType<scalar_t>::FragB& frag_b2,
    typename ScalarType<scalar_t>::FragC& frag_c) {
  const uint32_t* a = reinterpret_cast<const uint32_t*>(&a_frag);
  const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
  const uint32_t* b2 = reinterpret_cast<const uint32_t*>(&frag_b2);
  float* c = reinterpret_cast<float*>(&frag_c);
  if constexpr (std::is_same<scalar_t, half>::value) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(b[0]), "r"(b2[0]), "r"(b[1]), "r"(b2[1]), "r"(a[0]), "r"(a[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
  } else if constexpr (std::is_same<scalar_t, nv_bfloat16>::value) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(b[0]), "r"(b2[0]), "r"(b[1]), "r"(b2[1]), "r"(a[0]), "r"(a[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
  } else {
    STATIC_ASSERT_SCALAR_TYPE_VALID(scalar_t);
  }
}

// Instruction for loading a full 16x16 matrix fragment of operand A from shared
// memory, directly in tensor core layout.
template <int count, typename scalar_t>
__device__ inline void ldsm(typename ScalarType<scalar_t>::FragA& frag_a,
                            const void* smem_ptr) {
  uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  if constexpr (count == 4) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
        : "r"(smem));
  } else if constexpr (count == 2) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(a[0]), "=r"(a[1])
                 : "r"(smem));
  } else if constexpr (count == 1) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                 : "=r"(a[0])
                 : "r"(smem));
  } else {
    static_assert(count == 1 || count == 2 || count == 4, "invalid count");
  }
}

// Multiply dequantized values by the corresponding quantization scale; used
// only for grouped quantization.
template <typename scalar_t>
__device__ inline void scale(typename ScalarType<scalar_t>::FragB& frag_b,
                             typename ScalarType<scalar_t>::FragS& frag_s,
                             int i) {
  using scalar_t2 = typename ScalarType<scalar_t>::scalar_t2;
  scalar_t2 s =
      ScalarType<scalar_t>::num2num2(reinterpret_cast<scalar_t*>(&frag_s)[i]);
  frag_b[0] = __hmul2(frag_b[0], s);
  frag_b[1] = __hmul2(frag_b[1], s);
}

template <typename scalar_t>
__device__ inline void scale_and_sub(
    typename ScalarType<scalar_t>::FragB& frag_b, scalar_t s, scalar_t zp) {
  using scalar_t2 = typename ScalarType<scalar_t>::scalar_t2;
  scalar_t2 s2 = ScalarType<scalar_t>::num2num2(s);
  scalar_t2 zp2 = ScalarType<scalar_t>::num2num2(zp);
  frag_b[0] = __hfma2(frag_b[0], s2, __hneg2(zp2));
  frag_b[1] = __hfma2(frag_b[1], s2, __hneg2(zp2));
}

template <typename scalar_t>
__device__ inline void sub_zp(typename ScalarType<scalar_t>::FragB& frag_b,
                              typename ScalarType<scalar_t>::scalar_t2& frag_zp,
                              int i) {
  using scalar_t2 = typename ScalarType<scalar_t>::scalar_t2;
  scalar_t2 zp =
      ScalarType<scalar_t>::num2num2(reinterpret_cast<scalar_t*>(&frag_zp)[i]);
  frag_b[0] = __hsub2(frag_b[0], zp);
  frag_b[1] = __hsub2(frag_b[1], zp);
}

// Same as above, but for act_order (each K is multiplied individually)
template <typename scalar_t>
__device__ inline void scale4(typename ScalarType<scalar_t>::FragB& frag_b,
                              typename ScalarType<scalar_t>::FragS& frag_s_1,
                              typename ScalarType<scalar_t>::FragS& frag_s_2,
                              typename ScalarType<scalar_t>::FragS& frag_s_3,
                              typename ScalarType<scalar_t>::FragS& frag_s_4,
                              int i) {
  using scalar_t2 = typename ScalarType<scalar_t>::scalar_t2;
  scalar_t2 s_val_1_2;
  s_val_1_2.x = reinterpret_cast<scalar_t*>(&frag_s_1)[i];
  s_val_1_2.y = reinterpret_cast<scalar_t*>(&frag_s_2)[i];

  scalar_t2 s_val_3_4;
  s_val_3_4.x = reinterpret_cast<scalar_t*>(&frag_s_3)[i];
  s_val_3_4.y = reinterpret_cast<scalar_t*>(&frag_s_4)[i];

  frag_b[0] = __hmul2(frag_b[0], s_val_1_2);
  frag_b[1] = __hmul2(frag_b[1], s_val_3_4);
}

// Given 2 floats multiply by 2 scales (halves)
template <typename scalar_t>
__device__ inline void scale_float(float* c,
                                   typename ScalarType<scalar_t>::FragS& s) {
  scalar_t* s_ptr = reinterpret_cast<scalar_t*>(&s);
  c[0] = __fmul_rn(c[0], ScalarType<scalar_t>::num2float(s_ptr[0]));
  c[1] = __fmul_rn(c[1], ScalarType<scalar_t>::num2float(s_ptr[1]));
}

// Wait until barrier reaches `count`, then lock for current threadblock.
__device__ inline void barrier_acquire(int* lock, int count) {
  if (threadIdx.x == 0) {
    int state = -1;
    do
      // Guarantee that subsequent writes by this threadblock will be visible
      // globally.
      asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
                   : "=r"(state)
                   : "l"(lock));
    while (state != count);
  }
  __syncthreads();
}

// Release barrier and increment visitation count.
__device__ inline void barrier_release(int* lock, bool reset = false) {
  __syncthreads();
  if (threadIdx.x == 0) {
    if (reset) {
      lock[0] = 0;
      return;
    }
    int val = 1;
    // Make sure that all writes since acquiring this barrier are visible
    // globally, while releasing the barrier.
    asm volatile("fence.acq_rel.gpu;\n");
    asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n"
                 :
                 : "l"(lock), "r"(val));
  }
}

// Wait until value of lock to be negative, and then add 1
__device__ inline void wait_negative_and_add(int* lock) {
  if (threadIdx.x == 0) {
    int state = 0;
    do
      // Guarantee that subsequent writes by this threadblock will be visible
      // globally.
      asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
                   : "=r"(state)
                   : "l"(lock));
    while (state >= 0);
    atomicAdd(lock, 1);
  }
  __syncthreads();
}

template <typename scalar_t,  // compute dtype, half or nv_float16
          const vllm::ScalarTypeId w_type_id,  // weight ScalarType id
          const vllm::ScalarTypeId s_type_id,  // weight scale ScalarType id
          const int threads,          // number of threads in a threadblock
          const int thread_m_blocks,  // number of 16x16 blocks in the m
                                      // dimension (batchsize) of the
                                      // threadblock
          const int thread_n_blocks,  // same for n dimension (output)
          const int thread_k_blocks,  // same for k dimension (reduction)
          const bool m_block_size_8,  // whether m_block_size == 8
                                      // only works when thread_m_blocks == 1
          const int stages,  // number of stages for the async global->shared
                             // fetch pipeline
          const int group_blocks,  // number of consecutive 16x16 blocks
                                   // with a separate quantization scale
          const bool is_zp_float   // is zero point of float16 type?
          >
__global__ void Marlin(
    const int4* __restrict__ A,  // fp16 input matrix of shape mxk
    const int4* __restrict__ B,  // 4bit quantized weight matrix of shape kxn
    int4* __restrict__ C,        // fp16 output buffer of shape mxn
    int4* __restrict__ C_tmp,    // fp32 tmp output buffer (for reduce)
    const int4* __restrict__ b_bias_ptr,
    const int4* __restrict__ scales_ptr,  // fp16 quantization scales of shape
                                          // (k/groupsize)xn
    const uint16_t* __restrict__ scale2_ptr,  // fp16 global scale (for nvfp4
                                              // only)
    const int4* __restrict__ zp_ptr,  // 4bit packed zero-points of shape
                                      // (k/groupsize)x(n/pack_factor)
    const int* __restrict__ g_idx,    // int32 group indices of shape k
    const int32_t* __restrict__ sorted_token_ids_ptr,        // moe sorted_ids
    const int32_t* __restrict__ expert_ids_ptr,              // moe expert ids
    const int32_t* __restrict__ num_tokens_past_padded_ptr,  // moe num tokens
    const float* __restrict__ topk_weights_ptr,              // moe top weights
    int top_k,              // num of experts per token
    bool mul_topk_weights,  // mul topk weights or not
    bool is_ep,             // expert parallelism
    int num_groups,         // number of scale groups per output channel
    int prob_m,             // batch dimension m
    int prob_n,             // output dimension n
    int prob_k,             // reduction dimension k
    int* locks,             // extra global storage for barrier synchronization
    bool has_bias,
    bool use_atomic_add,   // whether to use atomic add to reduce
    bool use_fp32_reduce,  // whether to use fp32 global reduce
    int max_shared_mem) {
  // Each threadblock processes one "stripe" of the B matrix with (roughly) the
  // same size, which might involve multiple column "slices" (of width 16 *
  // `thread_n_blocks`). Stripes are defined as shown in the 3x3 matrix 5 SM
  // example:
  //   0 1 3
  //   0 2 3
  //   1 2 4
  // While this kind of partitioning makes things somewhat more complicated, it
  // ensures good utilization of all SMs for many kinds of shape and GPU
  // configurations, while requiring as few slow global cross-threadblock
  // reductions as possible.
  using Dtype = ScalarType<scalar_t>;
  using scalar_t2 = typename ScalarType<scalar_t>::scalar_t2;
  using FragA = typename ScalarType<scalar_t>::FragA;
  using FragB = typename ScalarType<scalar_t>::FragB;
  using FragC = typename ScalarType<scalar_t>::FragC;
  using FragS = typename ScalarType<scalar_t>::FragS;
  using FragZP = typename ScalarType<scalar_t>::FragZP;

  extern __shared__ int4 sh[];
  static constexpr auto w_type = vllm::ScalarType::from_id(w_type_id);
  static constexpr auto s_type = vllm::ScalarType::from_id(s_type_id);
  if constexpr (w_type == vllm::kFE2M1f) {
    static_assert(s_type == vllm::kFE4M3fn && group_blocks == 1 ||
                  s_type == vllm::kFE8M0fnu && group_blocks == 2);
  } else if constexpr (std::is_same<scalar_t, nv_bfloat16>::value) {
    static_assert(s_type == vllm::kBFloat16);
  } else if constexpr (std::is_same<scalar_t, half>::value) {
    static_assert(s_type == vllm::kFloat16);
  }

  constexpr bool has_zp = w_type == vllm::kU4 || w_type == vllm::kU8;
  constexpr bool is_int_type = w_type == vllm::kU4 || w_type == vllm::kU8 ||
                               w_type == vllm::kU4B8 || w_type == vllm::kU8B128;
  // see comments of dequant.h for more details
  constexpr bool dequant_skip_flop =
      w_type == vllm::kFE4M3fn ||
      w_type == vllm::kFE2M1f && s_type == vllm::kFE4M3fn ||
      has_zp && !is_zp_float && !std::is_same<scalar_t, nv_bfloat16>::value ||
      has_zp && !is_zp_float && !(w_type == vllm::kU8);

  scalar_t2 global_scale;

  constexpr bool has_act_order = group_blocks == 0;

  constexpr int pack_factor = 32 / w_type.size_bits();
  static_assert(thread_m_blocks == 1 || !m_block_size_8);
  constexpr int moe_block_size = m_block_size_8 ? 8 : (16 * thread_m_blocks);
  const int group_size =
      (!has_act_order && group_blocks == -1) ? prob_k : prob_k / num_groups;
  const int scales_expert_stride =
      prob_n * prob_k / group_size / (w_type == vllm::kFE2M1f ? 16 : 8);
  const int zp_expert_stride =
      is_zp_float ? prob_n * prob_k / group_size / 8
                  : prob_n * prob_k / group_size / (pack_factor * 4);
  const int b_bias_expert_stride = prob_n / 8;

  // parallel: num valid moe blocks
  int num_tokens_past_padded = num_tokens_past_padded_ptr[0];
  int parallel = num_tokens_past_padded / moe_block_size;
  int num_valid_blocks = parallel;
  if (is_ep) {
    for (int i = 0; i < parallel; i++) {
      if (expert_ids_ptr[i] == -1) num_valid_blocks--;
    }
  }
  int num_invalid_blocks = parallel - num_valid_blocks;
  parallel = num_valid_blocks;

  int k_tiles = prob_k / 16 / thread_k_blocks;
  int n_tiles = prob_n / 16 / thread_n_blocks;
  int iters = div_ceil(k_tiles * n_tiles * parallel, gridDim.x);

  if constexpr (!has_act_order && group_blocks != -1) {
    if (group_blocks >= thread_k_blocks) {
      // Ensure that the number of tiles in each stripe is a multiple of the
      // groupsize; this avoids an annoying special case where a stripe starts
      // in the middle of group.
      iters = (group_blocks / thread_k_blocks) *
              div_ceil(iters, (group_blocks / thread_k_blocks));
    }
  }

  int slice_row = (iters * blockIdx.x) % k_tiles;
  int slice_col_par = (iters * blockIdx.x) / k_tiles;
  int slice_col = slice_col_par;
  int slice_iters;  // number of threadblock tiles in the current slice
  int slice_count =
      0;          // total number of active threadblocks in the current slice
  int slice_idx;  // index of threadblock in current slice; numbered bottom to
                  // top

  int par_id = 0;
  int block_id = -1;
  int64_t expert_id = 0;  // use int64 to avoid computation result overflow
  int old_expert_id = 0;
  int64_t B_expert_off = 0;

  int4* sh_block_sorted_ids_int4 = sh;
  int4* sh_rd_block_sorted_ids_int4 =
      sh_block_sorted_ids_int4 + moe_block_size / 4;
  int4* sh_block_topk_weights_int4 =
      sh_rd_block_sorted_ids_int4 + moe_block_size / 4;
  // sh_block_topk_weights_int4 only need (moe_block_size / 4);
  // but we pad to align to 256 bytes
  int4* sh_new =
      sh_block_topk_weights_int4 + moe_block_size / 2 + moe_block_size;
  int32_t* sh_block_sorted_ids =
      reinterpret_cast<int*>(sh_block_sorted_ids_int4);
  int32_t* sh_rd_block_sorted_ids =
      reinterpret_cast<int*>(sh_rd_block_sorted_ids_int4);
  scalar_t2* sh_block_topk_weights =
      reinterpret_cast<scalar_t2*>(sh_block_topk_weights_int4);

  int32_t block_num_valid_tokens = 0;
  int32_t locks_off = 0;

  // We can easily implement parallel problem execution by just remapping
  // indices and advancing global pointers
  if (slice_col_par >= n_tiles) {
    slice_col = slice_col_par % n_tiles;
    par_id = slice_col_par / n_tiles;
  }
  if (parallel * n_tiles >= gridDim.x) {
    // when parallel * n_tiles >= sms
    // then there are at most $sms$ conflict tile blocks
    locks_off = blockIdx.x;
  } else {
    locks_off = (iters * blockIdx.x) / k_tiles - 1;
  }

  // read moe block data given block_id
  // block_sorted_ids / block_num_valid_tokens / block_topk_weights
  auto read_moe_block_data = [&](int block_id) {
    block_num_valid_tokens = moe_block_size;
  #pragma unroll
    for (int i = 0; i < moe_block_size / 4; i++) {
      int4 sorted_token_ids_int4 = reinterpret_cast<const int4*>(
          sorted_token_ids_ptr)[block_id * moe_block_size / 4 + i];
      int* sorted_token_ids = reinterpret_cast<int*>(&sorted_token_ids_int4);
  #pragma unroll
      for (int j = 0; j < 4; j++) {
        if (sorted_token_ids[j] >= prob_m * top_k) {
          block_num_valid_tokens = i * 4 + j;
          break;
        }
      }
      if (block_num_valid_tokens != moe_block_size) break;
    }

    __syncthreads();
    int tid4 = threadIdx.x / 4;
    if (threadIdx.x % 4 == 0 && threadIdx.x < block_num_valid_tokens) {
      sh_block_sorted_ids_int4[tid4] = reinterpret_cast<const int4*>(
          sorted_token_ids_ptr)[block_id * moe_block_size / 4 + tid4];

  #pragma unroll
      for (int i = 0; i < 4; i++)
        sh_rd_block_sorted_ids[tid4 * 4 + i] =
            sh_block_sorted_ids[tid4 * 4 + i] / top_k;

      if (mul_topk_weights) {
  #pragma unroll
        for (int i = 0; i < 4; i++) {
          int idx = tid4 * 4 + i;
          idx = idx < block_num_valid_tokens ? idx : 0;
          if constexpr (w_type == vllm::kFE2M1f && s_type == vllm::kFE4M3fn) {
            sh_block_topk_weights[idx] = __hmul2(
                global_scale, Dtype::num2num2(Dtype::float2num(
                                  topk_weights_ptr[sh_block_sorted_ids[idx]])));
          } else {
            sh_block_topk_weights[idx] = Dtype::num2num2(
                Dtype::float2num(topk_weights_ptr[sh_block_sorted_ids[idx]]));
          }
        }
      }
    }
    __syncthreads();
  };

  // when move to next moe block, find the next block_id and expert_id
  // and then read moe block data
  auto update_next_moe_block_data = [&]() {
    if (par_id >= parallel) return;

    old_expert_id = expert_id;
    if (num_invalid_blocks > 0) {
      int skip_count = block_id == -1 ? par_id : 0;
      block_id++;
      for (int i = block_id; i < num_tokens_past_padded / moe_block_size; i++) {
        expert_id = expert_ids_ptr[i];
        if (expert_id != -1) {
          if (skip_count == 0) {
            block_id = i;
            break;
          };
          skip_count--;
        };
      }
    } else {
      block_id = par_id;
      expert_id = expert_ids_ptr[block_id];
    }

    if constexpr (w_type == vllm::kFE2M1f && s_type == vllm::kFE4M3fn) {
      uint16_t val = scale2_ptr[expert_id];
      global_scale = Dtype::num2num2(*reinterpret_cast<scalar_t*>(&val));
    }

    B_expert_off = expert_id * prob_n * prob_k / (pack_factor * 4);
    scales_ptr += (expert_id - old_expert_id) * scales_expert_stride;
    if constexpr (has_zp) {
      zp_ptr += (expert_id - old_expert_id) * zp_expert_stride;
    }
    if constexpr (has_act_order) {
      g_idx += (expert_id - old_expert_id) * prob_k;
    }
    if (has_bias) {
      b_bias_ptr += (expert_id - old_expert_id) * b_bias_expert_stride;
    }

    read_moe_block_data(block_id);
  };

  // Compute all information about the current slice which is required for
  // synchronization.
  auto init_slice = [&](bool first_init = false) {
    slice_iters =
        iters * (blockIdx.x + 1) - (k_tiles * slice_col_par + slice_row);
    if (slice_iters < 0 || slice_col_par >= n_tiles * parallel) slice_iters = 0;
    if (slice_iters == 0) return;
    if (slice_row + slice_iters > k_tiles) slice_iters = k_tiles - slice_row;
    slice_count = 1;
    slice_idx = 0;
    int col_first = iters * div_ceil(k_tiles * slice_col_par, iters);
    if (col_first <= k_tiles * (slice_col_par + 1)) {
      int col_off = col_first - k_tiles * slice_col_par;
      slice_count = div_ceil(k_tiles - col_off, iters);
      if (col_off > 0) slice_count++;
      int delta_first = iters * blockIdx.x - col_first;
      if (delta_first < 0 || (col_off == 0 && delta_first == 0))
        slice_idx = slice_count - 1;
      else {
        slice_idx = slice_count - 1 - delta_first / iters;
        if (col_off > 0) slice_idx--;
      }
    }
    if (parallel * n_tiles >= gridDim.x) {
      if (slice_count > 1 && slice_idx == slice_count - 1) {
        locks_off++;
      }
    } else {
      locks_off++;
    }

    if (first_init && use_atomic_add && slice_count > 1 && slice_idx == 0) {
      constexpr int threads_per_m = 16 * thread_n_blocks / 8;
      int m_per_thread =
          div_ceil(block_num_valid_tokens, threads / threads_per_m);
      for (int i = 0; i < m_per_thread; i++) {
        int row = threads / threads_per_m * i + threadIdx.x / threads_per_m;
        if (row < block_num_valid_tokens) {
          int64_t sorted_row = sh_block_sorted_ids[row];
          int col = slice_col * 16 * thread_n_blocks / 8 +
                    threadIdx.x % threads_per_m;
          C[sorted_row * prob_n / 8 + col] = {0, 0, 0, 0};
        }
      }
      // After write zero to output, write a negative value to lock.
      // Every SM that processes the same slice would wait for
      // the negative value, and then atomicAdd 1 to it.
      // After all SMs are processed, the lock value would back to 0 again.
      __syncthreads();
      if (threadIdx.x == 0) locks[locks_off] = 1 - slice_count;
    }

    if (slice_col == n_tiles) {
      slice_col = 0;
      par_id++;
      update_next_moe_block_data();
    }
  };

  update_next_moe_block_data();
  init_slice(true);

  // A sizes/strides

  // stride of the A matrix in global memory
  int a_gl_stride = prob_k / 8;
  // stride of an A matrix tile in shared memory
  constexpr int a_sh_stride = 16 * thread_k_blocks / 8;
  // delta between subsequent A tiles in global memory
  constexpr int a_gl_rd_delta_o = 16 * thread_k_blocks / 8;
  // between subsequent accesses within a tile
  int a_gl_rd_delta_i = a_gl_stride * (threads / a_gl_rd_delta_o);
  // between shared memory writes
  constexpr int a_sh_wr_delta = a_sh_stride * (threads / a_gl_rd_delta_o);
  // between shared memory tile reads
  constexpr int a_sh_rd_delta_o = 2 * ((threads / 32) / (thread_n_blocks / 4));
  // within a shared memory tile
  constexpr int a_sh_rd_delta_i = a_sh_stride * 16;
  // overall size of a tile
  constexpr int a_sh_stage = a_sh_stride * (16 * thread_m_blocks);
  // number of shared write iterations for a tile
  constexpr int a_sh_wr_iters = div_ceil(a_sh_stage, a_sh_wr_delta);

  // B sizes/strides
  int b_gl_stride = 16 * prob_n / (pack_factor * 4);
  constexpr int b_sh_stride = ((thread_n_blocks * 16) * 16 / pack_factor) / 4;
  constexpr int b_thread_vecs = w_type.size_bits() == 4 ? 1 : 2;
  constexpr int b_sh_stride_threads = b_sh_stride / b_thread_vecs;

  int b_gl_rd_delta_o = b_gl_stride * thread_k_blocks;
  int b_gl_rd_delta_i = b_gl_stride * (threads / b_sh_stride_threads);
  constexpr int b_sh_wr_delta = threads * b_thread_vecs;
  constexpr int b_sh_rd_delta = threads * b_thread_vecs;
  constexpr int b_sh_stage = b_sh_stride * thread_k_blocks;
  constexpr int b_sh_wr_iters = b_sh_stage / b_sh_wr_delta;

  // Scale sizes/strides without act_order
  int s_gl_stride = prob_n / 8;
  constexpr int s_sh_stride = 16 * thread_n_blocks / 8;
  constexpr int s_tb_groups =
      !has_act_order && group_blocks != -1 && group_blocks < thread_k_blocks
          ? thread_k_blocks / group_blocks / (w_type == vllm::kFE2M1f ? 2 : 1)
          : 1;
  constexpr int s_sh_stage = s_tb_groups * s_sh_stride;
  int s_gl_rd_delta = s_gl_stride;

  // Scale size/strides with act_order
  constexpr int tb_k = 16 * thread_k_blocks;
  constexpr int g_idx_stage = has_act_order ? (tb_k * sizeof(int)) / 16 : 0;
  // constexpr int act_s_row_stride      = 1;
  // int           act_s_col_stride      = act_s_row_stride * num_groups;
  constexpr int act_s_max_num_groups = 32;
  int act_s_col_stride = 1;
  int act_s_col_warp_stride = act_s_col_stride * 8;
  int tb_n_warps = thread_n_blocks / 4;
  int act_s_col_tb_stride = act_s_col_warp_stride * tb_n_warps;

  // Zero-points sizes/strides
  int zp_gl_stride = is_zp_float ? prob_n / 8 : (prob_n / pack_factor) / 4;
  constexpr int zp_sh_stride = is_zp_float
                                   ? 16 * thread_n_blocks / 8
                                   : ((16 * thread_n_blocks) / pack_factor) / 4;
  constexpr int zp_tb_groups = s_tb_groups;
  constexpr int zp_sh_stage = has_zp ? zp_tb_groups * zp_sh_stride : 0;
  int zp_gl_rd_delta = zp_gl_stride;

  // Global A read index of current thread.
  int a_gl_rd_row = threadIdx.x / a_gl_rd_delta_o;
  int a_gl_rd_col = a_gl_rd_delta_o * slice_row + threadIdx.x % a_gl_rd_delta_o;

  // Shared write index of current thread.
  int a_sh_wr = a_sh_stride * (threadIdx.x / a_gl_rd_delta_o) +
                (threadIdx.x % a_gl_rd_delta_o);
  // Shared read index.
  int a_sh_rd =
      a_sh_stride * ((threadIdx.x % 32) % (16 / (m_block_size_8 ? 2 : 1))) +
      (threadIdx.x % 32) / (16 / (m_block_size_8 ? 2 : 1));
  a_sh_rd += 2 * ((threadIdx.x / 32) / (thread_n_blocks / 4));

  int b_gl_rd = b_gl_stride * (threadIdx.x / b_sh_stride_threads) +
                (threadIdx.x % b_sh_stride_threads) * b_thread_vecs;
  b_gl_rd += b_sh_stride * slice_col;
  b_gl_rd += b_gl_rd_delta_o * slice_row;
  auto b_sh_wr = threadIdx.x * b_thread_vecs;
  auto b_sh_rd = threadIdx.x * b_thread_vecs;

  // For act_order
  constexpr int k_iter_size = tb_k / b_sh_wr_iters;
  int slice_k_start = tb_k * slice_row;
  int slice_k_finish = slice_k_start + tb_k * slice_iters;
  int slice_k_start_shared_fetch = slice_k_start;
  int slice_n_offset = act_s_col_tb_stride * slice_col;

  // No act_order
  int s_gl_rd;
  if constexpr (!has_act_order) {
    if constexpr (group_blocks == -1) {
      s_gl_rd = s_sh_stride * slice_col + threadIdx.x;
    } else {
      s_gl_rd = s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) /
                    (w_type == vllm::kFE2M1f ? 2 : 1) +
                s_sh_stride * slice_col + threadIdx.x;
    }
  }
  auto s_sh_wr = threadIdx.x;
  bool s_sh_wr_pred = threadIdx.x < s_sh_stride;

  // Zero-points
  int zp_gl_rd;
  if constexpr (has_zp) {
    if constexpr (group_blocks == -1) {
      zp_gl_rd = zp_sh_stride * slice_col + threadIdx.x;
    } else {
      zp_gl_rd = zp_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) +
                 zp_sh_stride * slice_col + threadIdx.x;
    }
  }
  auto zp_sh_wr = threadIdx.x;
  bool zp_sh_wr_pred = threadIdx.x < zp_sh_stride;

  // We use a different scale layout for grouped and column-wise quantization as
  // we scale a `half2` tile in column-major layout in the former and in
  // row-major in the latter case.
  int s_sh_rd;
  if constexpr (group_blocks != -1 && w_type == vllm::kFE2M1f) {
    auto warp_id = threadIdx.x / 32;
    int n_warps = thread_n_blocks / 4;
    int warp_row = warp_id / n_warps;

    s_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) +
              (threadIdx.x % 32) / 4;
    s_sh_rd = s_sh_rd * 2 + (warp_row / group_blocks) % 2;

  } else if constexpr (group_blocks != -1)
    s_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) +
              (threadIdx.x % 32) / 4;
  else if constexpr (group_blocks == -1 &&
                     (m_block_size_8 || (has_zp && !dequant_skip_flop)))
    s_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) +
              (threadIdx.x % 32) / 8;
  else
    s_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) +
              (threadIdx.x % 32) % 4;

  int bias_sh_rd;
  if constexpr (m_block_size_8) {
    bias_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) +
                 (threadIdx.x % 32) / 8;
  } else {
    bias_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) +
                 (threadIdx.x % 32) % 4;
  }

  int bias_sh_wr = threadIdx.x;
  int bias_gl_rd = (thread_n_blocks * 16 / 8) * slice_col + threadIdx.x;

  // Zero-points have the same read layout as the scales
  // (without column-wise case)
  constexpr int num_col_threads = 8;
  constexpr int num_row_threads = 4;
  constexpr int num_ints_per_thread = 8 / pack_factor;
  int zp_sh_rd;
  if constexpr (has_zp) {
    if constexpr (is_zp_float) {
      if constexpr (group_blocks != -1) {
        zp_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) +
                   (threadIdx.x % 32) / 4;
      }
    } else {
      zp_sh_rd = num_ints_per_thread * num_col_threads *
                     ((threadIdx.x / 32) % (thread_n_blocks / 4)) +
                 num_ints_per_thread * ((threadIdx.x % 32) / num_row_threads);
    }
  }

  // To ensure that writing and reading A tiles to/from shared memory, the
  // latter in fragment format, is fully bank conflict free, we need to use a
  // rather fancy XOR-based layout. The key here is that neither reads nor
  // writes of the 16-byte `int4` blocks of 8 consecutive threads involve the
  // same shared memory banks. Further, it seems (based on NSight-Compute) that
  // each warp must also write a consecutive memory segment?
  auto transform_a = [&](int i) {
    int row = i / a_gl_rd_delta_o;
    return a_gl_rd_delta_o * row + (i % a_gl_rd_delta_o) ^ (row % 8);
  };
  // Since the computation of this remapping is non-trivial and, due to our main
  // loop unrolls, all shared memory accesses are static, we simply precompute
  // both transformed reads and writes.
  int a_sh_wr_trans[a_sh_wr_iters];
  #pragma unroll
  for (int i = 0; i < a_sh_wr_iters; i++)
    a_sh_wr_trans[i] = transform_a(a_sh_wr_delta * i + a_sh_wr);
  int a_sh_rd_trans[b_sh_wr_iters][thread_m_blocks];
  #pragma unroll
  for (int i = 0; i < b_sh_wr_iters; i++) {
  #pragma unroll
    for (int j = 0; j < thread_m_blocks; j++)
      a_sh_rd_trans[i][j] =
          transform_a(a_sh_rd_delta_o * i + a_sh_rd_delta_i * j + a_sh_rd);
  }

  // Since B-accesses have non-constant stride they have to be computed at
  // runtime; we break dependencies between subsequent accesses with a tile by
  // maintining multiple pointers (we have enough registers), a tiny
  // optimization.
  const int4* B_ptr[b_sh_wr_iters];
  #pragma unroll
  for (int i = 0; i < b_sh_wr_iters; i++)
    B_ptr[i] = B + b_gl_rd_delta_i * i + b_gl_rd;

  // Shared memory storage for global fetch pipelines.
  constexpr int sh_red_size = (2 * thread_n_blocks + 1) * 16 * thread_m_blocks;
  constexpr int sh_b_size = stages * b_sh_stage;
  int4* sh_b = sh_new;
  int4* sh_red = sh_new;

  constexpr int sh_size_b_red_min =
      (sh_red_size < sh_b_size ? sh_red_size : sh_b_size);
  constexpr int sh_size_b_red_max =
      (sh_red_size > sh_b_size ? sh_red_size : sh_b_size);
  constexpr int sh_bias_size = (thread_n_blocks * 16 / 8);
  constexpr int sh_b_red_bias_size =
      sh_size_b_red_max > (sh_size_b_red_min + sh_bias_size)
          ? sh_size_b_red_max
          : (sh_size_b_red_min + sh_bias_size);

  int4* sh_bias = sh_new + sh_size_b_red_min;
  int4* sh_g_idx = sh_new + sh_b_red_bias_size;
  int4* sh_zp = sh_g_idx + (stages * g_idx_stage);
  constexpr int sh_s_size = has_act_order ? (act_s_max_num_groups * s_sh_stride)
                                          : (stages * s_sh_stage);
  int4* sh_s = sh_zp + (stages * zp_sh_stage);
  // shared memory reused by reduction should be smaller than
  // shared memory used by weight.
  static_assert(thread_m_blocks * 16 * thread_n_blocks * 16 / 8 <=
                stages * b_sh_stage);
  int4* sh_a = sh_s + sh_s_size;
  constexpr int shm_size_used = moe_block_size +
                                stages * (g_idx_stage + zp_sh_stage) +
                                sh_s_size + sh_b_red_bias_size;

  // all remaining shared memory is used to cache A (input)
  // sh_a_max_row is at least ` stages * 16 * thread_m_blocks `
  int sh_a_max_row =
      ((max_shared_mem - 1024) / 16 - shm_size_used) / (thread_k_blocks * 2);

  // Register storage for double buffer of shared memory reads.
  FragA frag_a[2][thread_m_blocks];
  I4 frag_b_quant[2][b_thread_vecs];
  FragC frag_c[thread_m_blocks][4][2];
  FragS frag_s[2][4];  // No act-order
  FragS frag_bias[2][4];
  FragS act_frag_s[2][4][4];             // For act-order
  int frag_qzp[2][num_ints_per_thread];  // Zero-points
  FragZP frag_zp;                        // Zero-points in fp16
  FragZP frag_zpf[2];                    // Zero-points in fp16 in HQQ

  // Zero accumulators.
  auto zero_accums = [&]() {
  #pragma unroll
    for (int i = 0; i < thread_m_blocks * 4 * 2 * 4; i++)
      reinterpret_cast<float*>(frag_c)[i] = 0;
  };

  int sh_first_group_id = -1;
  int sh_num_groups = -1;

  auto fetch_act_order_scales_to_shared = [&](bool is_async, int first_group_id,
                                              int last_group_id) {
    sh_first_group_id = first_group_id;
    sh_num_groups = last_group_id - first_group_id + 1;

    if (sh_num_groups > act_s_max_num_groups) {
      sh_num_groups = act_s_max_num_groups;
    }

    if (sh_first_group_id + sh_num_groups > num_groups) {
      sh_num_groups = num_groups - sh_first_group_id;
    }

    int row_offset = first_group_id * s_gl_stride;

    if (is_async) {
      for (int i = 0; i < sh_num_groups; i++) {
        if (threadIdx.x < s_sh_stride) {
          cp_async4_pred(&sh_s[(i * s_sh_stride) + threadIdx.x],
                         &scales_ptr[row_offset + (i * s_gl_stride) +
                                     slice_n_offset + threadIdx.x]);
        }
      }
    } else {
      for (int i = 0; i < sh_num_groups; i++) {
        if (threadIdx.x < s_sh_stride) {
          sh_s[(i * s_sh_stride) + threadIdx.x] =
              scales_ptr[row_offset + (i * s_gl_stride) + slice_n_offset +
                         threadIdx.x];
        }
      }
    }
  };

  // Asynchronously fetch the next A, B and s tile from global to the next
  // shared memory pipeline location.
  bool should_load_a = true;
  int max_num_stage_groups =
      ((sh_a_max_row - moe_block_size) / moe_block_size + 1) / stages;
  max_num_stage_groups = max(max_num_stage_groups, 1);
  auto fetch_to_shared = [&](int pipe, int a_off, bool pred = true,
                             int pipe_a = 0) {
    if (pred) {
      if (should_load_a) {
        int4* sh_a_stage = sh_a + moe_block_size * a_sh_stride * pipe_a;
  #pragma unroll
        for (int i = 0; i < a_sh_wr_iters; i++) {
          int row = a_gl_rd_delta_i / a_gl_stride * i + a_gl_rd_row;
          int64_t sorted_row = 0;
          if (!m_block_size_8 || row < 8)
            sorted_row = sh_rd_block_sorted_ids[row];
          int64_t true_idx =
              sorted_row * a_gl_stride + a_gl_rd_col + a_gl_rd_delta_o * a_off;
          cp_async4_pred(&sh_a_stage[a_sh_wr_trans[i]], &A[true_idx],
                         row < block_num_valid_tokens);
        }
      }

      int4* sh_b_stage = sh_b + b_sh_stage * pipe;
  #pragma unroll
      for (int i = 0; i < b_sh_wr_iters; i++) {
  #pragma unroll
        for (int j = 0; j < b_thread_vecs; j++) {
          cp_async4(&sh_b_stage[b_sh_wr_delta * i + b_sh_wr + j],
                    B_ptr[i] + j + B_expert_off);
        }

        B_ptr[i] += b_gl_rd_delta_o;
      }

      if constexpr (has_act_order) {
        // Fetch g_idx thread-block portion
        int full_pipe = a_off;
        int cur_k = slice_k_start_shared_fetch + tb_k * full_pipe;
        if (cur_k < prob_k && cur_k < slice_k_finish) {
          int4* sh_g_idx_stage = sh_g_idx + g_idx_stage * pipe;

          int4 const* cur_g_idx_stage_ptr =
              reinterpret_cast<int4 const*>(&g_idx[cur_k]);

          if (threadIdx.x < g_idx_stage) {
            cp_async4_pred(&sh_g_idx_stage[threadIdx.x],
                           &cur_g_idx_stage_ptr[threadIdx.x]);
          }
        }
      } else {
        if constexpr (group_blocks != -1) {
          int4* sh_s_stage = sh_s + s_sh_stage * pipe;

          if constexpr (group_blocks >= thread_k_blocks) {
            // Only fetch scales if this tile starts a new group
            if (pipe % (group_blocks / thread_k_blocks) == 0) {
              if (s_sh_wr_pred) {
                cp_async4(&sh_s_stage[s_sh_wr], &scales_ptr[s_gl_rd]);
              }
              s_gl_rd += s_gl_rd_delta;
            }
          } else {
            for (int i = 0; i < s_tb_groups; i++) {
              if (s_sh_wr_pred) {
                cp_async4(&sh_s_stage[i * s_sh_stride + s_sh_wr],
                          &scales_ptr[s_gl_rd]);
              }
              s_gl_rd += s_gl_rd_delta;
            }
          }
        }

        if constexpr (has_zp && group_blocks != -1) {
          int4* sh_zp_stage = sh_zp + zp_sh_stage * pipe;

          if constexpr (group_blocks >= thread_k_blocks) {
            // Only fetch zero-points if this tile starts a new group
            if (pipe % (group_blocks / thread_k_blocks) == 0) {
              if (zp_sh_wr_pred) {
                cp_async4(&sh_zp_stage[zp_sh_wr], &zp_ptr[zp_gl_rd]);
              }
              zp_gl_rd += zp_gl_rd_delta;
            }
          } else {
            for (int i = 0; i < zp_tb_groups; i++) {
              if (zp_sh_wr_pred) {
                cp_async4(&sh_zp_stage[i * zp_sh_stride + zp_sh_wr],
                          &zp_ptr[zp_gl_rd]);
              }
              zp_gl_rd += zp_gl_rd_delta;
            }
          }
        }
      }
    }
    // Insert a fence even when we are winding down the pipeline to ensure that
    // waiting is also correct at this point.
    cp_async_fence();
  };

  auto fetch_col_zp_to_shared = [&]() {
    if (zp_sh_wr_pred) {
      cp_async4(&sh_zp[zp_sh_wr], &zp_ptr[zp_gl_rd]);
    }
  };

  auto fetch_col_scale_to_shared = [&]() {
    if (s_sh_wr_pred) {
      cp_async4(&sh_s[s_sh_wr], &scales_ptr[s_gl_rd]);
    }
  };

  // Wait until the next thread tile has been loaded to shared memory.
  auto wait_for_stage = [&]() {
    // We only have `stages - 2` active fetches since we are double buffering
    // and can only issue the next fetch when it is guaranteed that the previous
    // shared memory load is fully complete (as it may otherwise be
    // overwritten).
    cp_async_wait<stages - 2>();
    __syncthreads();
  };

  // Load the next sub-tile from the current location in the shared memory pipe
  // into the current register buffer.
  auto fetch_to_registers = [&](int k, int pipe, int pipe_a = 0) {
    int4* sh_a_stage = sh_a + moe_block_size * a_sh_stride * pipe_a;
  #pragma unroll
    for (int i = 0; i < thread_m_blocks; i++)
      ldsm<m_block_size_8 ? 2 : 4, scalar_t>(
          frag_a[k % 2][i], &sh_a_stage[a_sh_rd_trans[k % b_sh_wr_iters][i]]);
    int4* sh_b_stage = sh_b + b_sh_stage * pipe;

  #pragma unroll
    for (int i = 0; i < b_thread_vecs; i++) {
      frag_b_quant[k % 2][i] = *reinterpret_cast<I4*>(
          &sh_b_stage[b_sh_rd_delta * (k % b_sh_wr_iters) + b_sh_rd + i]);
    }
  };

  bool is_same_group[stages];
  int same_group_id[stages];

  auto init_same_group = [&](int pipe) {
    if constexpr (!has_act_order) {
      return;
    }

    int4* sh_g_idx_stage = sh_g_idx + g_idx_stage * pipe;
    int* sh_g_idx_int_ptr = reinterpret_cast<int*>(sh_g_idx_stage);

    int group_id_1 = sh_g_idx_int_ptr[0];
    int group_id_2 = sh_g_idx_int_ptr[tb_k - 1];

    is_same_group[pipe] = group_id_1 == group_id_2;
    same_group_id[pipe] = group_id_1;
  };

  auto fetch_scales_to_registers = [&](int k, int full_pipe) {
    int pipe = full_pipe % stages;

    if constexpr (!has_act_order) {
      // No act-order case
      if constexpr (group_blocks == -1) {
        // load only when starting a new slice
        if (k == 0 && full_pipe == 0) {
          reinterpret_cast<int4*>(&frag_s)[0] = sh_s[s_sh_rd];
          reinterpret_cast<int4*>(&frag_s)[1] = sh_s[s_sh_rd + 4];
        }
      } else if constexpr (group_blocks != -1) {
        if constexpr (group_blocks >= thread_k_blocks) {
          if (k % b_sh_wr_iters == 0) {
            int4* sh_s_stage =
                sh_s + s_sh_stage * ((group_blocks / thread_k_blocks) *
                                     (pipe / (group_blocks / thread_k_blocks)));
            reinterpret_cast<int4*>(&frag_s[k % 2])[0] = sh_s_stage[s_sh_rd];
          } else {
            reinterpret_cast<int4*>(&frag_s[1])[0] =
                reinterpret_cast<int4*>(&frag_s[0])[0];
          }
        } else {
          auto warp_id = threadIdx.x / 32;
          int n_warps = thread_n_blocks / 4;

          int warp_row = warp_id / n_warps;

          int cur_k = warp_row * 16;
          cur_k += k_iter_size * (k % b_sh_wr_iters);

          int k_blocks = cur_k / 16;
          int cur_group_id =
              k_blocks / (group_blocks * (w_type == vllm::kFE2M1f ? 2 : 1));

          int4* sh_s_stage = sh_s + s_sh_stage * pipe;

          if constexpr (w_type_id != vllm::kFE2M1f.id()) {
            reinterpret_cast<int4*>(&frag_s[k % 2])[0] =
                sh_s_stage[s_sh_rd + cur_group_id * s_sh_stride];
          } else if constexpr (group_blocks == 1 || thread_k_blocks > 4) {
            reinterpret_cast<int2*>(&frag_s[k % 2])[0] =
                reinterpret_cast<int2*>(
                    sh_s_stage)[s_sh_rd + cur_group_id * (2 * s_sh_stride)];
          } else {
            reinterpret_cast<int2*>(&frag_s[k % 2])[0] =
                reinterpret_cast<int2*>(
                    sh_s_stage)[s_sh_rd + cur_group_id * (2 * s_sh_stride) +
                                k % 2];
          }
        }
      }

      return;
    }

    // Act-order case

    // Determine K of the "current" thread-block
    int cur_k = slice_k_start + tb_k * full_pipe;
    if (cur_k >= prob_k || cur_k >= slice_k_finish) {
      return;
    }

    // Reset (to current thread-block) since we read g_idx portion from the
    // shared memory
    cur_k = 0;

    // Progress to current iteration
    cur_k += k_iter_size * (k % b_sh_wr_iters);

    // Determine "position" inside the thread-block (based on warp and
    // thread-id)
    auto warp_id = threadIdx.x / 32;
    int n_warps =
        thread_n_blocks / 4;  // Each warp processes 4 16-size tiles over N

    int warp_row = warp_id / n_warps;
    int warp_col = warp_id % n_warps;

    cur_k += warp_row * 16;

    auto th_id = threadIdx.x % 32;
    cur_k += (th_id % 4) * 2;  // Due to tensor-core layout for fp16 B matrix

    int s_col_shift =
        /*slice_n_offset +*/ (act_s_col_warp_stride * warp_col) +
        (th_id / 4) * act_s_col_stride;

    if (is_same_group[pipe]) {
      if (k % 2 == 0) {
        *(reinterpret_cast<int4*>(&(act_frag_s[k % 2][0][0]))) =
            sh_s[(same_group_id[pipe] - sh_first_group_id) * s_sh_stride +
                 s_col_shift];
      } else {
        *(reinterpret_cast<int4*>(&(act_frag_s[k % 2][0][0]))) =
            *(reinterpret_cast<int4*>(&(act_frag_s[(k - 1) % 2][0][0])));
      }

      for (int i = 1; i < 4; i++) {
        *(reinterpret_cast<int4*>(&(act_frag_s[k % 2][i][0]))) =
            *(reinterpret_cast<int4*>(&(act_frag_s[k % 2][0][0])));
      }
      return;
    }

    int4* sh_g_idx_stage = sh_g_idx + g_idx_stage * pipe;
    int* sh_g_idx_int_ptr = reinterpret_cast<int*>(sh_g_idx_stage);

    constexpr int k_frag_offsets[4] = {0, 1, 8,
                                       9};  // Tensor core offsets per thread

  #pragma unroll
    for (int i = 0; i < 4; i++) {
      int actual_k = cur_k + k_frag_offsets[i];

      int group_id = sh_g_idx_int_ptr[actual_k];
      int rel_group_id = group_id - sh_first_group_id;

      *(reinterpret_cast<int4*>(&(act_frag_s[k % 2][i][0]))) =
          sh_s[rel_group_id * s_sh_stride + s_col_shift];
    }
  };

  auto fetch_zp_to_registers = [&](int k, int full_pipe) {
    // This code does not handle group_blocks == 0,
    // which signifies act_order.
    // has_zp implies AWQ, which doesn't have act_order,
    static_assert(!has_zp || group_blocks != 0);

    if constexpr (has_zp && !is_zp_float) {
      int pipe = full_pipe % stages;

      if constexpr (group_blocks == -1) {
        // load only when starting a new slice
        if (k == 0 && full_pipe == 0) {
  #pragma unroll
          for (int i = 0; i < num_ints_per_thread; i++) {
            frag_qzp[k % 2][i] = (reinterpret_cast<int*>(sh_zp))[zp_sh_rd + i];
          }
        }

      } else if constexpr (group_blocks >= thread_k_blocks) {
        if (k % b_sh_wr_iters == 0) {
          int4* sh_zp_stage =
              sh_zp + zp_sh_stage * ((group_blocks / thread_k_blocks) *
                                     (pipe / (group_blocks / thread_k_blocks)));
  #pragma unroll
          for (int i = 0; i < num_ints_per_thread; i++) {
            frag_qzp[k % 2][i] =
                (reinterpret_cast<int*>(sh_zp_stage))[zp_sh_rd + i];
          }
        }
      } else {
        auto warp_id = threadIdx.x / 32;
        int n_warps = thread_n_blocks / 4;

        int warp_row = warp_id / n_warps;

        int cur_k = warp_row * 16;
        cur_k += k_iter_size * (k % b_sh_wr_iters);

        int k_blocks = cur_k / 16;
        int cur_group_id = 0;

        // Suppress bogus and persistent divide-by-zero warning
  #pragma nv_diagnostic push
  #pragma nv_diag_suppress divide_by_zero
        cur_group_id = k_blocks / group_blocks;
  #pragma nv_diagnostic pop

        int4* sh_zp_stage = sh_zp + zp_sh_stage * pipe;

        sh_zp_stage += cur_group_id * zp_sh_stride;

  #pragma unroll
        for (int i = 0; i < num_ints_per_thread; i++) {
          frag_qzp[k % 2][i] =
              (reinterpret_cast<int*>(sh_zp_stage))[zp_sh_rd + i];
        }
      }
    }

    else if constexpr (has_zp && is_zp_float) {
      int pipe = full_pipe % stages;

      if constexpr (group_blocks != -1) {
        if constexpr (group_blocks >= thread_k_blocks) {
          if (k % b_sh_wr_iters == 0) {
            int4* sh_zp_stage =
                sh_zp +
                zp_sh_stage * ((group_blocks / thread_k_blocks) *
                               (pipe / (group_blocks / thread_k_blocks)));
            reinterpret_cast<int4*>(&frag_zpf[k % 2])[0] =
                sh_zp_stage[zp_sh_rd];
          }
        } else {
          auto warp_id = threadIdx.x / 32;
          int n_warps = thread_n_blocks / 4;

          int warp_row = warp_id / n_warps;

          int cur_k = warp_row * 16;
          cur_k += k_iter_size * (k % b_sh_wr_iters);

          int k_blocks = cur_k / 16;
          // Suppress bogus and persistent divide-by-zero warning
  #pragma nv_diagnostic push
  #pragma nv_diag_suppress divide_by_zero
          int cur_group_id = k_blocks / group_blocks;
  #pragma nv_diagnostic pop

          int4* sh_zp_stage = sh_zp + zp_sh_stage * pipe;

          reinterpret_cast<int4*>(&frag_zpf[k % 2])[0] =
              sh_zp_stage[zp_sh_rd + cur_group_id * zp_sh_stride];
        }
      }
    }
  };

  auto dequant_data = [&](int q, scalar_t2* frag_b_ptr) {
    dequant<scalar_t2, w_type_id, dequant_skip_flop>(q, frag_b_ptr);
  };

  // Execute the actual tensor core matmul of a sub-tile.
  bool is_first_matmul_in_slice = true;
  auto matmul = [&](int k) {
    int k2 = k % 2;
    const bool is_new_zp =
        ((group_blocks != -1) && (group_blocks < thread_k_blocks || k == 0)) ||
        (group_blocks == -1 && is_first_matmul_in_slice);
    if constexpr (has_zp && !is_zp_float) {
      if (is_new_zp) {
        if constexpr (group_blocks == -1) is_first_matmul_in_slice = false;
        int zp_quant_0, zp_quant_1;

        if constexpr (w_type.size_bits() == 4) {
          zp_quant_0 = frag_qzp[k2][0];
          zp_quant_1 = zp_quant_0 >> 8;
        } else {
          static_assert(w_type.size_bits() == 8);
          zp_quant_0 = frag_qzp[k2][0];
          zp_quant_1 = frag_qzp[k2][1];
        }

        dequant_data(zp_quant_0, reinterpret_cast<scalar_t2*>(&frag_zp));
        dequant_data(zp_quant_1, reinterpret_cast<scalar_t2*>(&frag_zp) + 2);
      }
    }
    if constexpr (!dequant_skip_flop && has_zp && is_zp_float) {
      if (is_new_zp) {
        reinterpret_cast<int4*>(&frag_zp)[0] =
            reinterpret_cast<int4*>(&frag_zpf[k2])[0];
      }
    }

    if constexpr (w_type == vllm::kFE2M1f) {
      int s_quant_0 = reinterpret_cast<int*>(frag_s[k2])[0];
      int s_quant_1 = reinterpret_cast<int*>(frag_s[k2])[1];

      dequant_fp8_scales<scalar_t2, s_type_id>(
          s_quant_0, reinterpret_cast<scalar_t2*>(&frag_s[k2]));
      dequant_fp8_scales<scalar_t2, s_type_id>(
          s_quant_1, reinterpret_cast<scalar_t2*>(&frag_s[k2]) + 2);
    }

  // We have the m dimension as the inner loop in order to encourage overlapping
  // dequantization and matmul operations.
  #pragma unroll
    for (int j = 0; j < 4; j++) {
      FragB frag_b0;
      FragB frag_b1;
      int b_quant_0, b_quant_1;

      if constexpr (w_type_id == vllm::kFE2M1f.id()) {
        b_quant_1 = frag_b_quant[k2][0][j];
        b_quant_0 = b_quant_1 << 8;
      } else if constexpr (w_type.size_bits() == 4) {
        b_quant_0 = frag_b_quant[k2][0][j];
        b_quant_1 = b_quant_0 >> 8;
      } else {
        static_assert(w_type.size_bits() == 8);
        int* frag_b_quant_ptr = reinterpret_cast<int*>(frag_b_quant[k2]);
        b_quant_0 = frag_b_quant_ptr[j * 2 + 0];
        b_quant_1 = frag_b_quant_ptr[j * 2 + 1];
      }

      dequant_data(b_quant_0, reinterpret_cast<scalar_t2*>(&frag_b0));
      dequant_data(b_quant_1, reinterpret_cast<scalar_t2*>(&frag_b1));

      if constexpr (dequant_skip_flop && has_zp && !is_zp_float) {
        sub_zp<scalar_t>(frag_b0, frag_zp[j], 0);
        sub_zp<scalar_t>(frag_b1, frag_zp[j], 1);
      }

      // Apply scale to frag_b0
      if constexpr (has_act_order) {
        static_assert(group_blocks != -1);
        scale4<scalar_t>(frag_b0, act_frag_s[k2][0][j], act_frag_s[k2][1][j],
                         act_frag_s[k2][2][j], act_frag_s[k2][3][j], 0);
        scale4<scalar_t>(frag_b1, act_frag_s[k2][0][j], act_frag_s[k2][1][j],
                         act_frag_s[k2][2][j], act_frag_s[k2][3][j], 1);
      } else if constexpr (!dequant_skip_flop && has_zp && !is_zp_float &&
                           group_blocks == -1) {
        int idx = (threadIdx.x / 4) % 2;
        scalar_t2 s2 = Dtype::nums2num2(
            reinterpret_cast<scalar_t*>(&frag_s[j / 2][j % 2 * 2 + 0])[idx],
            reinterpret_cast<scalar_t*>(&frag_s[j / 2][j % 2 * 2 + 1])[idx]);
        if (is_new_zp) frag_zp[j] = __hmul2(frag_zp[j], s2);
        scale_and_sub<scalar_t>(frag_b0, s2.x, frag_zp[j].x);
        scale_and_sub<scalar_t>(frag_b1, s2.y, frag_zp[j].y);
      } else if constexpr (!dequant_skip_flop && has_zp && group_blocks != -1) {
        if (is_new_zp)
          frag_zp[j] = __hmul2(frag_zp[j],
                               *reinterpret_cast<scalar_t2*>(&frag_s[k2][j]));
        scale_and_sub<scalar_t>(frag_b0, frag_s[k2][j][0].x, frag_zp[j].x);
        scale_and_sub<scalar_t>(frag_b1, frag_s[k2][j][0].y, frag_zp[j].y);
      } else if constexpr (group_blocks != -1) {
        scale<scalar_t>(frag_b0, frag_s[k2][j], 0);
        scale<scalar_t>(frag_b1, frag_s[k2][j], 1);
      }

  #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
        if constexpr (m_block_size_8) {
          mma_trans<scalar_t>(frag_a[k2][i], frag_b0, frag_b1, frag_c[i][j][0]);
        } else {
          mma<scalar_t>(frag_a[k2][i], frag_b0, frag_c[i][j][0]);
          mma<scalar_t>(frag_a[k2][i], frag_b1, frag_c[i][j][1]);
        }
      }
    }
  };

  // Since we slice across the k dimension of a tile in order to increase the
  // number of warps while keeping the n dimension of a tile reasonable, we have
  // multiple warps that accumulate their partial sums of the same output
  // location; which we have to reduce over in the end. We do in shared memory.
  auto thread_block_reduce = [&]() {
    constexpr int red_off = threads / b_sh_stride_threads / 2;
    if (red_off >= 1) {
      auto red_idx = threadIdx.x / b_sh_stride_threads;
      constexpr int red_sh_stride = b_sh_stride_threads * 4 * 2;
      constexpr int red_sh_delta = b_sh_stride_threads;
      int red_sh_rd = red_sh_stride * (threadIdx.x / b_sh_stride_threads) +
                      (threadIdx.x % b_sh_stride_threads);

      // Parallel logarithmic shared memory reduction. We make sure to avoid any
      // unnecessary read or write iterations, e.g., for two warps we write only
      // once by warp 1 and read only once by warp 0.

  #pragma unroll
      for (int m_block = 0; m_block < thread_m_blocks; m_block++) {
  #pragma unroll
        for (int i = red_off; i > 0; i /= 2) {
          if (i <= red_idx && red_idx < 2 * i) {
  #pragma unroll
            for (int j = 0; j < 4 * 2; j += (m_block_size_8 ? 2 : 1)) {
              int red_sh_wr =
                  red_sh_delta * j + (red_sh_rd - red_sh_stride * i);
              if (i < red_off) {
                float* c_rd = reinterpret_cast<float*>(
                    &sh_red[red_sh_delta * j + red_sh_rd]);
                float* c_wr = reinterpret_cast<float*>(&sh_red[red_sh_wr]);
  #pragma unroll
                for (int k = 0; k < 4; k++)
                  reinterpret_cast<FragC*>(frag_c)[4 * 2 * m_block + j][k] +=
                      c_rd[k] + c_wr[k];
              }
              sh_red[red_sh_wr] =
                  reinterpret_cast<int4*>(&frag_c)[4 * 2 * m_block + j];
            }
          }
          __syncthreads();
        }
        if (red_idx == 0) {
  #pragma unroll
          for (int i = 0; i < 4 * 2; i += (m_block_size_8 ? 2 : 1)) {
            float* c_rd =
                reinterpret_cast<float*>(&sh_red[red_sh_delta * i + red_sh_rd]);
  #pragma unroll
            for (int j = 0; j < 4; j++)
              reinterpret_cast<FragC*>(frag_c)[4 * 2 * m_block + i][j] +=
                  c_rd[j];
          }
        }
        __syncthreads();
      }
    }
  };

  // Since multiple threadblocks may process parts of the same column slice, we
  // finally have to globally reduce over the results. As the striped
  // partitioning minimizes the number of such reductions and our outputs are
  // usually rather small, we perform this reduction serially in L2 cache.
  auto global_reduce_fp16 = [&](bool first = false, bool last = false) {
    // We are very careful here to reduce directly in the output buffer to
    // maximize L2 cache utilization in this step. To do this, we write out
    // results in FP16 (but still reduce with FP32 compute).
    constexpr int active_threads = 32 * thread_n_blocks / 4;
    bool is_th_active = threadIdx.x < active_threads;
    if (!is_th_active) {
      return;
    }

    int c_gl_stride = prob_n / 8;
    int c_gl_wr_delta_o = 8 * c_gl_stride;
    int c_gl_wr_delta_i = 4 * (active_threads / 32);
    int c_gl_wr;
    if constexpr (m_block_size_8) {
      c_gl_wr = c_gl_stride * ((threadIdx.x % 4) * 2) + 4 * (threadIdx.x / 32) +
                (threadIdx.x % 32) / 8;
      c_gl_wr += (2 * thread_n_blocks) * slice_col;
    } else {
      c_gl_wr = c_gl_stride * ((threadIdx.x % 32) / 4) +
                4 * (threadIdx.x / 32) + threadIdx.x % 4;
      c_gl_wr += (2 * thread_n_blocks) * slice_col;
    }
    constexpr int c_sh_wr_delta = active_threads;
    int c_sh_wr = threadIdx.x;

    if (!first) {

  #pragma unroll
      for (int i = 0; i < (m_block_size_8 ? 2 : thread_m_blocks * 4); i++) {
        int c_idx;
        if constexpr (m_block_size_8)
          c_idx = c_gl_wr + i * c_gl_stride +
                  (threadIdx.x % 8) / 4 * c_gl_wr_delta_i;
        else
          c_idx =
              c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2);
        if (c_idx / c_gl_stride < block_num_valid_tokens) {
          int64_t sorted_row = sh_block_sorted_ids[c_idx / c_gl_stride];
          int64_t true_idx = sorted_row * c_gl_stride + c_idx % c_gl_stride;
          sh_red[c_sh_wr + c_sh_wr_delta * i] = C[true_idx];
        }
      }
    }

  #pragma unroll
    for (int i = 0; i < (m_block_size_8 ? 2 : thread_m_blocks * 4); i++) {
      if (!first) {
        int4 c_red = sh_red[c_sh_wr + i * c_sh_wr_delta];
  #pragma unroll
        for (int j = 0; j < 2 * 4; j++) {
          int delta = 0;
          if constexpr (m_block_size_8) {
            delta = j % 2 == 1 ? -2 : 0;
          }
          reinterpret_cast<float*>(
              &frag_c)[4 * 2 * 4 * (i / 4) + 4 * j + (i % 4) + delta] +=
              Dtype::num2float(reinterpret_cast<scalar_t*>(&c_red)[j]);
        }
      }
      if (!last) {
        int4 c;
  #pragma unroll
        for (int j = 0; j < 2 * 4; j++) {
          int delta = 0;
          if constexpr (m_block_size_8) {
            delta = j % 2 == 1 ? -2 : 0;
          }
          reinterpret_cast<scalar_t*>(&c)[j] =
              Dtype::float2num(reinterpret_cast<float*>(
                  &frag_c)[4 * 2 * 4 * (i / 4) + 4 * j + (i % 4) + delta]);
        }

        int c_idx;
        if constexpr (m_block_size_8)
          c_idx = c_gl_wr + i * c_gl_stride +
                  (threadIdx.x % 8) / 4 * c_gl_wr_delta_i;
        else
          c_idx =
              c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2);
        if (c_idx / c_gl_stride < block_num_valid_tokens) {
          int64_t sorted_row = sh_block_sorted_ids[c_idx / c_gl_stride];
          int64_t true_idx = sorted_row * c_gl_stride + c_idx % c_gl_stride;
          C[true_idx] = c;
        }
      }
    }
  };

  // Globally reduce over threadblocks that compute the same column block.
  // We use a tmp C buffer to reduce in full fp32 precision.
  auto global_reduce_fp32 = [&](bool first = false, bool last = false) {
    constexpr int tb_m = thread_m_blocks * 16;
    constexpr int tb_n = thread_n_blocks * 16;

    constexpr int c_size = tb_m * tb_n * sizeof(float) / 16;

    constexpr int active_threads = 32 * thread_n_blocks / 4;
    bool is_th_active = threadIdx.x < active_threads;

    constexpr int num_floats = thread_m_blocks * 4 * 2 * 4;
    constexpr int th_size = num_floats * sizeof(float) / 16;

    int c_cur_offset = locks_off * c_size;

    if (!is_th_active) {
      return;
    }

    if (!first) {
      float* frag_c_ptr = reinterpret_cast<float*>(&frag_c);
  #pragma unroll
      for (int k = 0; k < th_size; k++) {
        if constexpr (m_block_size_8) {
          if (k % 2) continue;
        } else {
          if (k / 8 * 16 + (threadIdx.x % 32) / 4 >= block_num_valid_tokens)
            continue;
        }

        sh_red[threadIdx.x] =
            C_tmp[c_cur_offset + active_threads * k + threadIdx.x];

        float* sh_c_ptr = reinterpret_cast<float*>(&sh_red[threadIdx.x]);
  #pragma unroll
        for (int f = 0; f < 4; f++) {
          frag_c_ptr[k * 4 + f] += sh_c_ptr[f];
        }
      }
    }

    if (!last) {
      int4* frag_c_ptr = reinterpret_cast<int4*>(&frag_c);
  #pragma unroll
      for (int k = 0; k < th_size; k++) {
        if constexpr (m_block_size_8) {
          if (k % 2) continue;
        } else {
          if (k / 8 * 16 + (threadIdx.x % 32) / 4 >= block_num_valid_tokens)
            continue;
        }

        C_tmp[c_cur_offset + active_threads * k + threadIdx.x] = frag_c_ptr[k];
      }
    }
  };

  // Write out the reduce final result in the correct layout. We only actually
  // reshuffle matrix fragments in this step, the reduction above is performed
  // in fragment layout.
  auto write_result = [&](bool last) {
    int c_gl_stride = prob_n / 8;
    constexpr int c_sh_stride = 2 * thread_n_blocks + 1;
    int c_gl_wr_delta = c_gl_stride * (threads / (2 * thread_n_blocks));
    constexpr int c_sh_rd_delta =
        c_sh_stride * (threads / (2 * thread_n_blocks));

    int c_gl_wr = c_gl_stride * (threadIdx.x / (2 * thread_n_blocks)) +
                  (threadIdx.x % (2 * thread_n_blocks));
    c_gl_wr += (2 * thread_n_blocks) * slice_col;
    int c_sh_wr;
    if constexpr (m_block_size_8) {
      c_sh_wr = (8 * c_sh_stride) * ((threadIdx.x % 32) % 4 * 2) +
                (threadIdx.x % 32) / 4;
      c_sh_wr += 64 * (threadIdx.x / 32);
    } else {
      c_sh_wr =
          (4 * c_sh_stride) * ((threadIdx.x % 32) / 4) + (threadIdx.x % 32) % 4;
      c_sh_wr += 32 * (threadIdx.x / 32);
    }

    int c_sh_rd = c_sh_stride * (threadIdx.x / (2 * thread_n_blocks)) +
                  (threadIdx.x % (2 * thread_n_blocks));

    // We first reorder in shared memory to guarantee the most efficient final
    // global write patterns
    auto write = [&](int idx, float c0, float c1, FragS& s, FragS& b_bias) {
      scalar_t2 res =
          Dtype::nums2num2(Dtype::float2num(c0), Dtype::float2num(c1));

      // For per-column quantization we finally apply the scale here (only for
      // 4-bit)
      if constexpr (!has_act_order && group_blocks == -1 &&
                    w_type.size_bits() == 4 &&
                    (has_zp && dequant_skip_flop || !has_zp)) {
        scalar_t2 tmp_scale = s[0];
        if constexpr (m_block_size_8) {
          tmp_scale = Dtype::num2num2(
              reinterpret_cast<scalar_t*>(&s[0])[(threadIdx.x % 8) / 4]);
        }
        res = __hmul2(res, tmp_scale);
      }

      if constexpr (w_type == vllm::kFE2M1f && s_type == vllm::kFE4M3fn) {
        if (!mul_topk_weights) {
          res = __hmul2(res, global_scale);
        }
      }
      if (has_bias && last) {
        scalar_t2 tmp_bias = b_bias[0];
        if constexpr (m_block_size_8) {
          tmp_bias = Dtype::num2num2(
              reinterpret_cast<scalar_t*>(&b_bias[0])[(threadIdx.x % 8) / 4]);
        }
        res = __hadd2(res, tmp_bias);
      }

      if constexpr (m_block_size_8) {
        ((scalar_t*)sh_red)[idx] = res.x;
        ((scalar_t*)sh_red)[idx + 8 * c_sh_stride] = res.y;
      } else {
        ((scalar_t2*)sh_red)[idx] = res;
      }
    };

    if (threadIdx.x / 32 < thread_n_blocks / 4) {
  #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
  #pragma unroll
        for (int j = 0; j < 4; j++) {
          if constexpr (m_block_size_8) {
            int wr = c_sh_wr + 16 * j;
            write(wr, frag_c[i][j][0][0], frag_c[i][j][0][1],
                  frag_s[j / 2][2 * (j % 2) + 0],
                  frag_bias[j / 2][2 * (j % 2) + 0]);
            write(wr + 8, frag_c[i][j][0][2], frag_c[i][j][0][3],
                  frag_s[j / 2][2 * (j % 2) + 1],
                  frag_bias[j / 2][2 * (j % 2) + 1]);
          } else {
            int wr = c_sh_wr + 8 * j;
            write(wr + (4 * c_sh_stride) * 0 + 0, frag_c[i][j][0][0],
                  frag_c[i][j][0][1], frag_s[j / 2][2 * (j % 2) + 0],
                  frag_bias[j / 2][2 * (j % 2) + 0]);
            write(wr + (4 * c_sh_stride) * 8 + 0, frag_c[i][j][0][2],
                  frag_c[i][j][0][3], frag_s[j / 2][2 * (j % 2) + 0],
                  frag_bias[j / 2][2 * (j % 2) + 0]);
            write(wr + (4 * c_sh_stride) * 0 + 4, frag_c[i][j][1][0],
                  frag_c[i][j][1][1], frag_s[j / 2][2 * (j % 2) + 1],
                  frag_bias[j / 2][2 * (j % 2) + 1]);
            write(wr + (4 * c_sh_stride) * 8 + 4, frag_c[i][j][1][2],
                  frag_c[i][j][1][3], frag_s[j / 2][2 * (j % 2) + 1],
                  frag_bias[j / 2][2 * (j % 2) + 1]);
          }
        }
        c_sh_wr += 16 * (4 * c_sh_stride);
      }
    }
    __syncthreads();

  #pragma unroll
    for (int i = 0;
         i < div_ceil(16 * thread_m_blocks, threads / (2 * thread_n_blocks));
         i++) {
      int row = c_gl_wr / c_gl_stride;
      if (row < block_num_valid_tokens) {
        int64_t sorted_row = sh_block_sorted_ids[row];
        int64_t true_idx = sorted_row * c_gl_stride + c_gl_wr % c_gl_stride;
        scalar_t2 topk_weight_score;
        if (mul_topk_weights) topk_weight_score = sh_block_topk_weights[row];
        if (use_atomic_add && slice_count > 1 || mul_topk_weights) {
          scalar_t2* C_half2 = reinterpret_cast<scalar_t2*>(&C[true_idx]);
          scalar_t2* sh_red_half2 =
              reinterpret_cast<scalar_t2*>(&sh_red[c_sh_rd]);
  #pragma unroll
          for (int a = 0; a < 4; a++) {
            scalar_t2 res = sh_red_half2[a];
            if (mul_topk_weights) {
              res = __hmul2(res, topk_weight_score);
            }

            if (use_atomic_add && slice_count > 1) {
              atomicAdd(&C_half2[a], res);
            } else {
              C_half2[a] = res;
            };
          }
        } else {
          C[true_idx] = sh_red[c_sh_rd];
        }
        c_gl_wr += c_gl_wr_delta;
        c_sh_rd += c_sh_rd_delta;
      }
    }
    __syncthreads();
  };

  // Start global fetch and register load pipelines.
  auto start_pipes = [&]() {

  #pragma unroll
    for (int i = 0; i < stages - 1; i++) {
      if (has_act_order && i == 0) {
        int last_g_idx = slice_k_start + stages * tb_k * 2;
        if (last_g_idx >= prob_k) {
          last_g_idx = prob_k - 1;
        }
        fetch_act_order_scales_to_shared(true, g_idx[slice_k_start],
                                         g_idx[last_g_idx]);
      }

      if constexpr (has_zp && !is_zp_float && group_blocks == -1) {
        if (i == 0) {
          fetch_col_zp_to_shared();
          if constexpr (!dequant_skip_flop) {
            fetch_col_scale_to_shared();
          }
        }
      }
      fetch_to_shared(i, i, i < slice_iters, i);
    }

    zero_accums();
    wait_for_stage();
    init_same_group(0);
    fetch_to_registers(0, 0);
    fetch_scales_to_registers(0, 0);
    fetch_zp_to_registers(0, 0);
    a_gl_rd_col += a_gl_rd_delta_o * (stages - 1);
    if constexpr (has_act_order) {
      slice_k_start_shared_fetch += tb_k * (stages - 1);
    }
  };
  if (slice_iters) {
    start_pipes();
  }

  // Main loop.
  while (slice_iters) {
    // We unroll over both the global fetch and the register load pipeline to
    // ensure all shared memory accesses are static. Note that both pipelines
    // have even length meaning that the next iteration will always start at
    // index 0.

    for (int stage_group_id = 0; stage_group_id < max_num_stage_groups;
         stage_group_id++) {
  #pragma unroll
      for (int pipe = 0; pipe < stages;) {
  #pragma unroll
        for (int k = 0; k < b_sh_wr_iters; k++) {
          int idx =
              (pipe >= stages && stage_group_id == max_num_stage_groups - 1)
                  ? (pipe - stages)
                  : (pipe + stage_group_id * stages);
          fetch_to_registers(k + 1, pipe % stages, idx);
          fetch_scales_to_registers(k + 1, pipe);
          fetch_zp_to_registers(k + 1, pipe);
          if (k == b_sh_wr_iters - 2) {
            int idx = (pipe >= 1 && stage_group_id == max_num_stage_groups - 1)
                          ? (pipe - 1)
                          : (pipe + (stage_group_id + 1) * stages - 1);
            fetch_to_shared((pipe + stages - 1) % stages, pipe,
                            slice_iters >= stages, idx);
            pipe++;
            wait_for_stage();
            init_same_group(pipe % stages);
          }
          matmul(k);
        }
        slice_iters--;
        if (slice_iters == 0) {
          break;
        }
      }

      a_gl_rd_col += a_gl_rd_delta_o * stages;

      if constexpr (has_act_order) {
        slice_k_start += tb_k * stages;

        if (slice_k_start < prob_k) {
          slice_k_start_shared_fetch += tb_k * stages;
          int first_group_id = g_idx[slice_k_start];
          int last_g_idx = slice_k_start + stages * tb_k * 2;
          if (last_g_idx >= prob_k) {
            last_g_idx = prob_k - 1;
          }
          int last_group_id = g_idx[last_g_idx];
          if (last_group_id >= sh_first_group_id + sh_num_groups) {
            fetch_act_order_scales_to_shared(false, first_group_id,
                                             last_group_id);
            __syncthreads();
          }
        }
      }
      if (slice_iters == 0) {
        break;
      }
    }

    // Process results and, if necessary, proceed to the next column slice.
    // While this pattern may not be the most readable, other ways of writing
    // the loop seemed to noticeably worse performance after compilation.
    if (slice_iters == 0) {
      cp_async_wait<0>();
      bool last = slice_idx == slice_count - 1;
      // For per-column scales, we only fetch them here in the final step before
      // write-out
      if constexpr (!has_act_order && group_blocks == -1 &&
                    (has_zp && dequant_skip_flop || !has_zp)) {
        if (w_type.size_bits() == 8 || (last || use_atomic_add)) {
          if (s_sh_wr_pred) {
            cp_async4(&sh_s[s_sh_wr], &scales_ptr[s_gl_rd]);
          }
          cp_async_fence();
        }
      }

      thread_block_reduce();

      if (has_bias && last) {
        __syncthreads();
        cp_async4_pred(&sh_bias[bias_sh_wr], &b_bias_ptr[bias_gl_rd],
                       threadIdx.x < 16 * thread_n_blocks / 8);
        cp_async_fence();
      }

      if constexpr (!has_act_order && group_blocks == -1 &&
                    (has_zp && dequant_skip_flop || !has_zp)) {
        if (w_type.size_bits() == 8 || (last || use_atomic_add)) {
          cp_async_wait<0>();
          __syncthreads();
          if (threadIdx.x / 32 < thread_n_blocks / 4) {
            reinterpret_cast<int4*>(&frag_s)[0] = sh_s[s_sh_rd + 0];
            reinterpret_cast<int4*>(&frag_s)[1] = sh_s[s_sh_rd + 4];
            if constexpr (m_block_size_8) {
              int idx = (threadIdx.x / 4) % 2;
              scalar_t2* frag_s_half2 = reinterpret_cast<scalar_t2*>(frag_s);
  #pragma unroll
              for (int i = 0; i < 8; i++) {
                frag_s_half2[i] = Dtype::num2num2(
                    reinterpret_cast<scalar_t*>(&frag_s_half2[i])[idx]);
              }
            }
          }
        }
      }

      // For 8-bit channelwise, we apply the scale before the global reduction
      // that converts the fp32 results to fp16 (so that we avoid possible
      // overflow in fp16)
      if constexpr (!has_act_order && group_blocks == -1 &&
                    w_type.size_bits() == 8 &&
                    (has_zp && dequant_skip_flop || !has_zp)) {
        if (threadIdx.x / 32 < thread_n_blocks / 4) {
  #pragma unroll
          for (int i = 0; i < thread_m_blocks; i++) {
  #pragma unroll
            for (int j = 0; j < 4; j++) {
              scale_float<scalar_t>(
                  reinterpret_cast<float*>(&frag_c[i][j][0][0]),
                  frag_s[j / 2][2 * (j % 2) + 0]);
              scale_float<scalar_t>(
                  reinterpret_cast<float*>(&frag_c[i][j][0][2]),
                  frag_s[j / 2][2 * (j % 2) + (m_block_size_8 ? 1 : 0)]);

              if constexpr (!m_block_size_8) {
                scale_float<scalar_t>(
                    reinterpret_cast<float*>(&frag_c[i][j][1][0]),
                    frag_s[j / 2][2 * (j % 2) + 1]);
                scale_float<scalar_t>(
                    reinterpret_cast<float*>(&frag_c[i][j][1][2]),
                    frag_s[j / 2][2 * (j % 2) + 1]);
              }
            }
          }
        }
      }

      if (slice_count > 1 && !use_atomic_add) {
        // only globally reduce if there is more than one block in a slice
        barrier_acquire(&locks[locks_off], slice_idx);
        if (use_fp32_reduce) {
          global_reduce_fp32(slice_idx == 0, last);
        } else {
          global_reduce_fp16(slice_idx == 0, last);
        }
        barrier_release(&locks[locks_off], last);
      }

      if (has_bias && last) {
        cp_async_wait<0>();
        __syncthreads();
        reinterpret_cast<int4*>(&frag_bias)[0] = sh_bias[bias_sh_rd];
        reinterpret_cast<int4*>(&frag_bias)[1] = sh_bias[bias_sh_rd + 4];
        __syncthreads();
      }

      if (use_atomic_add && slice_count > 1 && slice_idx != 0)
        wait_negative_and_add(&locks[locks_off]);
      if (last || use_atomic_add)
        // only the last block in a slice actually writes the result
        write_result(last);
      int old_slice_row = slice_row;
      slice_row = 0;
      slice_col_par++;
      slice_col++;
      is_first_matmul_in_slice = true;
      init_slice();

      // Should we load A matrix in next slice?
      // `slice_col == 0`: when move to a new moe block
      // `old_slice_row > 0`:
      //    when the last slice is not starting from k_index == 0
      //    (only happen when it is the first slice of a threadblock)
      // `prob_k > thread_k_blocks * 16 * stages * max_num_stage_groups`:
      //    when the required shared memory size is larger than
      //    the remaining shared memory
      if (slice_col == 0 || old_slice_row ||
          prob_k > thread_k_blocks * 16 * stages * max_num_stage_groups) {
        should_load_a = true;
      } else {
        should_load_a = false;
      }

      if (slice_iters) {
        a_gl_rd_col = (threadIdx.x % a_gl_rd_delta_o);
  #pragma unroll
        for (int i = 0; i < b_sh_wr_iters; i++)
          B_ptr[i] += b_sh_stride - b_gl_rd_delta_o * k_tiles;
        if (slice_col == 0) {
  #pragma unroll
          for (int i = 0; i < b_sh_wr_iters; i++) B_ptr[i] -= b_gl_stride;
        }

        bias_gl_rd = (thread_n_blocks * 16 / 8) * slice_col + threadIdx.x;
        // Update slice k/n for scales loading
        if constexpr (has_act_order) {
          slice_k_start = tb_k * slice_row;
          slice_k_finish = slice_k_start + tb_k * slice_iters;
          slice_k_start_shared_fetch = slice_k_start;
          slice_n_offset = act_s_col_tb_stride * slice_col;
        } else {
          s_gl_rd = s_sh_stride * slice_col + threadIdx.x;
          zp_gl_rd = zp_sh_stride * slice_col + threadIdx.x;
        }
        start_pipes();
      }
    }
  }
}

}  // namespace MARLIN_NAMESPACE_NAME

#endif
