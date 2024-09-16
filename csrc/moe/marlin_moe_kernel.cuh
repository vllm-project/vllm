#pragma once

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <iostream>

#include "core/scalar_type.hpp"

namespace marlin_moe {

constexpr int ceildiv(int a, int b) { return (a + b - 1) / b; }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

// Instances of `Vec` are used to organize groups of >>registers<<, as needed
// for instance as inputs to tensor core operations. Consequently, all
// corresponding index accesses must be compile-time constants, which is why we
// extensively use `#pragma unroll` throughout the kernel code to guarantee
// this.
template <typename T, int n>
struct Vec {
  T elems[n];
  __device__ T& operator[](int i) { return elems[i]; }
};

using I4 = Vec<int, 4>;

// Matrix fragments for tensor core instructions; their precise layout is
// documented here:
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-floating-point-type
using FragA = Vec<half2, 4>;
using FragB = Vec<half2, 2>;
using FragC = Vec<float, 4>;
using FragS = Vec<half2, 1>;  // quantization scales
using FragZP = Vec<half2, 1>;

// Predicated asynchronous global->shared copy; used for inputs A where we apply
// predication to handle batchsizes that are not multiples of 16.
__device__ inline void cp_async4_pred(void* smem_ptr, const void* glob_ptr,
                                      bool pred = true) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "{\n"
      "   .reg .pred p;\n"
      "   setp.ne.b32 p, %0, 0;\n"
      "   @p cp.async.cg.shared.global [%1], [%2], %3;\n"
      "}\n" ::"r"((int)pred),
      "r"(smem), "l"(glob_ptr), "n"(BYTES));
}

// Asynchronous global->shared copy
__device__ inline void cp_async4(void* smem_ptr, const void* glob_ptr) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "{\n"
      "   cp.async.cg.shared.global [%0], [%1], %2;\n"
      "}\n" ::"r"(smem),
      "l"(glob_ptr), "n"(BYTES));
}

// Async copy fence.
__device__ inline void cp_async_fence() {
  asm volatile("cp.async.commit_group;\n" ::);
}

// Wait until at most `n` async copy stages are still pending.
template <int n>
__device__ inline void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
}

// m16n8k16 tensor core mma instruction with fp16 inputs and fp32
// output/accumulation.
__device__ inline void mma(const FragA& a_frag, const FragB& frag_b,
                           FragC& frag_c) {
  const uint32_t* a = reinterpret_cast<const uint32_t*>(&a_frag);
  const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
  float* c = reinterpret_cast<float*>(&frag_c);
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

// Instruction for loading a full 16x16 matrix fragment of operand A from shared
// memory, directly in tensor core layout.
__device__ inline void ldsm4(FragA& frag_a, const void* smem_ptr) {
  uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
               : "r"(smem));
}

// Lookup-table based 3-input logical operation; explicitly used for
// dequantization as the compiler does not seem to automatically recognize it in
// all cases.
template <int lut>
__device__ inline int lop3(int a, int b, int c) {
  int res;
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(res)
               : "r"(a), "r"(b), "r"(c), "n"(lut));
  return res;
}

// Constructs destination register by taking bytes from 2 sources (based on
// mask)
template <int start_byte, int mask>
__device__ inline uint32_t prmt(uint32_t a) {
  uint32_t res;
  asm volatile("prmt.b32 %0, %1, %2, %3;\n"
               : "=r"(res)
               : "r"(a), "n"(start_byte), "n"(mask));
  return res;
}

template <vllm::ScalarTypeId w_type_id>
__device__ inline FragB dequant(int q);

// Efficiently dequantize 4bit values packed in an int32 value into a full
// B-fragment of 4 fp16 values. We mostly follow the strategy in the link below,
// with some small changes:
// https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h#L215-L287
template <>
__device__ inline FragB dequant<vllm::kU4B8.id()>(int q) {
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400;
  // Guarantee that the `(a & b) | c` operations are LOP3s.
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
  // We want signed int4 outputs, hence we fuse the `-8` symmetric zero point
  // directly into `SUB` and `ADD`.
  const int SUB = 0x64086408;
  const int MUL = 0x2c002c00;
  const int ADD = 0xd480d480;
  FragB frag_b;
  frag_b[0] = __hsub2(*reinterpret_cast<half2*>(&lo),
                      *reinterpret_cast<const half2*>(&SUB));
  frag_b[1] = __hfma2(*reinterpret_cast<half2*>(&hi),
                      *reinterpret_cast<const half2*>(&MUL),
                      *reinterpret_cast<const half2*>(&ADD));
  return frag_b;
}

// Fast Int8ToFp16: Efficiently dequantize 8bit int values to fp16
// Reference:
// https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h#L53-L85
template <>
__device__ inline FragB dequant<vllm::kU8B128.id()>(int q) {
  static constexpr uint32_t mask_for_elt_01 = 0x5250;
  static constexpr uint32_t mask_for_elt_23 = 0x5351;
  static constexpr uint32_t start_byte_for_fp16 = 0x64646464;

  uint32_t lo = prmt<start_byte_for_fp16, mask_for_elt_01>(q);
  uint32_t hi = prmt<start_byte_for_fp16, mask_for_elt_23>(q);

  static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64806480;

  FragB frag_b;
  frag_b[0] = __hsub2(*reinterpret_cast<half2*>(&lo),
                      *reinterpret_cast<const half2*>(&I8s_TO_F16s_MAGIC_NUM));
  frag_b[1] = __hsub2(*reinterpret_cast<half2*>(&hi),
                      *reinterpret_cast<const half2*>(&I8s_TO_F16s_MAGIC_NUM));
  return frag_b;
}

template <>
__device__ inline FragB dequant<vllm::kU4.id()>(int q) {
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400;
  // Guarantee that the `(a & b) | c` operations are LOP3s.
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);

  const int SUB = 0x64006400;
  const int MUL = 0x2c002c00;
  const int ADD = 0xd400d400;
  FragB frag_b;
  frag_b[0] = __hsub2(*reinterpret_cast<half2*>(&lo),
                      *reinterpret_cast<const half2*>(&SUB));
  frag_b[1] = __hfma2(*reinterpret_cast<half2*>(&hi),
                      *reinterpret_cast<const half2*>(&MUL),
                      *reinterpret_cast<const half2*>(&ADD));
  return frag_b;
}

template <>
__device__ inline FragB dequant<vllm::kU8.id()>(int q) {
  static constexpr uint32_t mask_for_elt_01 = 0x5250;
  static constexpr uint32_t mask_for_elt_23 = 0x5351;
  static constexpr uint32_t start_byte_for_fp16 = 0x64646464;

  uint32_t lo = prmt<start_byte_for_fp16, mask_for_elt_01>(q);
  uint32_t hi = prmt<start_byte_for_fp16, mask_for_elt_23>(q);

  static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64006400;

  FragB frag_b;
  frag_b[0] = __hsub2(*reinterpret_cast<half2*>(&lo),
                      *reinterpret_cast<const half2*>(&I8s_TO_F16s_MAGIC_NUM));
  frag_b[1] = __hsub2(*reinterpret_cast<half2*>(&hi),
                      *reinterpret_cast<const half2*>(&I8s_TO_F16s_MAGIC_NUM));
  return frag_b;
}

// Multiply dequantized values by the corresponding quantization scale; used
// only for grouped quantization.
__device__ inline void scale(FragB& frag_b, FragS& frag_s, int i) {
  half2 s = __half2half2(reinterpret_cast<__half*>(&frag_s)[i]);
  frag_b[0] = __hmul2(frag_b[0], s);
  frag_b[1] = __hmul2(frag_b[1], s);
}

__device__ inline void sub_zp(FragB& frag_b, half2& frag_zp, int i) {
  half2 zp = __half2half2(reinterpret_cast<__half*>(&frag_zp)[i]);
  frag_b[0] = __hsub2(frag_b[0], zp);
  frag_b[1] = __hsub2(frag_b[1], zp);
}

// Given 2 floats multiply by 2 scales (halves)
__device__ inline void scale_float(float* c, FragS& s) {
  __half* s_ptr = reinterpret_cast<__half*>(&s);
  c[0] = __fmul_rn(c[0], __half2float(s_ptr[0]));
  c[1] = __fmul_rn(c[1], __half2float(s_ptr[1]));
}

// Same as above, but for act_order (each K is multiplied individually)
__device__ inline void scale4(FragB& frag_b, FragS& frag_s_1, FragS& frag_s_2,
                              FragS& frag_s_3, FragS& frag_s_4, int i) {
  __half2 s_val_1_2;
  s_val_1_2.x = reinterpret_cast<__half*>(&frag_s_1)[i];
  s_val_1_2.y = reinterpret_cast<__half*>(&frag_s_2)[i];

  __half2 s_val_3_4;
  s_val_3_4.x = reinterpret_cast<__half*>(&frag_s_3)[i];
  s_val_3_4.y = reinterpret_cast<__half*>(&frag_s_4)[i];

  frag_b[0] = __hmul2(frag_b[0], s_val_1_2);
  frag_b[1] = __hmul2(frag_b[1], s_val_3_4);
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

template <const vllm::ScalarTypeId w_type_id,  // weight ScalarType id
          const int threads,          // number of threads in a threadblock
          const int thread_m_blocks,  // number of 16x16 blocks in the m
                                      // dimension (batchsize) of the
                                      // threadblock
          const int thread_n_blocks,  // same for n dimension (output)
          const int thread_k_blocks,  // same for k dimension (reduction)
          const int stages,  // number of stages for the async global->shared
                             // fetch pipeline
          const bool has_act_order,    // whether act_order is enabled
          const bool has_zp,           // whether zero-points are enabled
          const int group_blocks = -1  // number of consecutive 16x16 blocks
                                       // with a separate quantization scale
          >
__device__ inline void MarlinMoESingle(
    const int4* __restrict__ A,  // fp16 input matrix of shape mxk
    const int4* __restrict__ B,  // 4bit quantized weight matrix of shape kxn
    int4* __restrict__ C,        // fp16 output buffer of shape mxn
    const int* __restrict__ sorted_ids,      // int32 sorted ids of experts
    const float* __restrict__ topk_weights,  // float topk weights
    const int4* __restrict__ scales_ptr,  // fp16 quantization scales of shape
                                          // (k/groupsize)xn
    const int4* __restrict__ zp_ptr,      // 4bit packed zero-points of shape
                                          // (k/groupsize)x(n/pack_factor)
    const int* __restrict__ g_idx,        // int32 group indices of shape k
    const int* __restrict__ expert_offsets,
    int num_groups,        // number of scale groups per output channel
    int expert_idx,        // idx of current expert
    int num_experts,       // number of experts
    int topk,              // topk parameter of moe
    int prob_m,            // batch dimension m
    int prob_n,            // output dimension n
    int prob_k,            // reduction dimension k
    int tot_m,             // total number of rows in A and C
    int* locks,            // extra global storage for barrier synchronization
    bool replicate_input,  // do we use the same input for each expert?
    bool apply_weights,    // apply weights to output
    int current_m_block    // current m block to start kernel computation from
);

template <const vllm::ScalarTypeId w_type_id,  // weight ScalarType id
          const int threads,          // number of threads in a threadblock
          const int thread_m_blocks,  // number of 16x16 blocks in the m
                                      // dimension (batchsize) of the
                                      // threadblock
          const int thread_n_blocks,  // same for n dimension (output)
          const int thread_k_blocks,  // same for k dimension (reduction)
          const int stages,  // number of stages for the async global->shared
                             // fetch pipeline
          const bool has_act_order,    // whether act_order is enabled
          const bool has_zp,           // whether zero-points are enabled
          const int group_blocks = -1  // number of consecutive 16x16 blocks
                                       // with a separate quantization scale
          >
__global__ void MarlinMoE(
    const int4* __restrict__ A,  // fp16 input matrix of shape mxk
    const int4* __restrict__ B,  // 4bit quantized weight matrix of shape kxn
    int4* __restrict__ C,        // fp16 output buffer of shape mxn
    const int* __restrict__ sorted_ids_base,  // int32 sorted ids of experts
    const float* __restrict__ topk_weights,   // float topk weights
    const int4* __restrict__ scales_ptr,  // fp16 quantization scales of shape
                                          // (k/groupsize)xn
    const int4* __restrict__ zp_ptr,      // 4bit packed zero-points of shape
                                          // (k/groupsize)x(n/pack_factor)
    const int* __restrict__ g_idx,        // int32 group indices of shape k
    const int* __restrict__ expert_offsets,
    int num_groups,        // number of scale groups per output channel
    int expert_idx,        // idx of current expert
    int num_experts,       // number of experts
    int topk,              // topk parameter of moe
    int prob_m,            // batch dimension m
    int prob_n,            // output dimension n
    int prob_k,            // reduction dimension k
    int tot_m,             // total number of rows in A and C
    int* locks,            // extra global storage for barrier synchronization
    bool replicate_input,  // do we use the same input for each expert?
    bool apply_weights,    // apply weights to output
    int current_m_block,   // current m block to start kernel computation from
    int max_par,           // maximum parallelism
    int cfg_max_m_blocks   // upper bound on m blocks
);

#else

template <const vllm::ScalarTypeId w_type_id,  // weight ScalarType id
          const int threads,          // number of threads in a threadblock
          const int thread_m_blocks,  // number of 16x16 blocks in the m
                                      // dimension (batchsize) of the
                                      // threadblock
          const int thread_n_blocks,  // same for n dimension (output)
          const int thread_k_blocks,  // same for k dimension (reduction)
          const int stages,  // number of stages for the async global->shared
                             // fetch pipeline
          const bool has_act_order,    // whether act_order is enabled
          const bool has_zp,           // whether zero-points are enabled
          const int group_blocks = -1  // number of consecutive 16x16 blocks
                                       // with a separate quantization scale
          >
__global__ void MarlinMoE(
    const int4* __restrict__ A,  // fp16 input matrix of shape mxk
    const int4* __restrict__ B,  // 4bit quantized weight matrix of shape kxn
    int4* __restrict__ C,        // fp16 output buffer of shape mxn
    const int* __restrict__ sorted_ids,      // int32 sorted ids of experts
    const float* __restrict__ topk_weights,  // float topk weights
    const int4* __restrict__ scales_ptr,  // fp16 quantization scales of shape
                                          // (k/groupsize)xn
    const int4* __restrict__ zp_ptr,      // 4bit packed zero-points of shape
                                          // (k/groupsize)x(n/pack_factor)
    const int* __restrict__ g_idx,        // int32 group indices of shape k
    const int* __restrict__ expert_offsets,
    int num_groups,        // number of scale groups per output channel
    int expert_idx,        // idx of current expert
    int num_experts,       // number of experts
    int topk,              // topk parameter of moe
    int prob_m,            // batch dimension m
    int prob_n,            // output dimension n
    int prob_k,            // reduction dimension k
    int tot_m,             // total number of rows in A and C
    int* locks,            // extra global storage for barrier synchronization
    bool replicate_input,  // do we use the same input for each expert?
    bool apply_weights,    // apply weights to output
    int current_m_block,   // current m block to start kernel computation from
    int max_par,           // maximum parallelism
    int cfg_max_m_blocks   // upper bound on m blocks

) {
  // Marlin is not implemented yet for SM < 8.0
  assert(false);
  return;
}

#endif

// 8 warps are a good choice since every SM has 4 schedulers and having more
// than 1 warp per schedule allows some more latency hiding. At the same time,
// we want relatively few warps to have many registers per warp and small tiles.
const int USER_THREADS =
    256;               // Note: This is only used with user-provided thread_k/n
const int STAGES = 4;  // 4 pipeline stages fit into shared memory
// const int SHARED_MEM =
//     96 * 1024; // max shared memory on compute capability 8.6 (< 8.0)

static constexpr int min_thread_n = 64;
static constexpr int min_thread_k = 64;

// #define __CALL_IF_MOE(W_TYPE, THREAD_M_BLOCKS, THREAD_N_BLOCKS,               \
//                       THREAD_K_BLOCKS, HAS_ACT_ORDER, HAS_ZP, GROUP_BLOCKS,   \
//                       NUM_THREADS)                                            \
//   else if (q_type == W_TYPE && thread_m_blocks == THREAD_M_BLOCKS &&          \
//            thread_n_blocks == THREAD_N_BLOCKS &&                              \
//            thread_k_blocks == THREAD_K_BLOCKS &&                              \
//            has_act_order == HAS_ACT_ORDER && has_zp == HAS_ZP &&              \
//            group_blocks == GROUP_BLOCKS && num_threads == NUM_THREADS) {      \
//     cudaFuncSetAttribute(MarlinMoE<W_TYPE.id(), NUM_THREADS, THREAD_M_BLOCKS, \
//                                    THREAD_N_BLOCKS, THREAD_K_BLOCKS, STAGES,  \
//                                    HAS_ACT_ORDER, HAS_ZP, GROUP_BLOCKS>,      \
//                          cudaFuncAttributeMaxDynamicSharedMemorySize,         \
//                          max_shared_mem);                                     \
//     MarlinMoE<W_TYPE.id(), NUM_THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS,     \
//               THREAD_K_BLOCKS, STAGES, HAS_ACT_ORDER, HAS_ZP, GROUP_BLOCKS>   \
//         <<<blocks, NUM_THREADS, max_shared_mem, stream>>>(                    \
//             A_ptr, B_ptr, C_ptr, sorted_ids_ptr, topk_weights_ptr, s_ptr,     \
//             zp_ptr, g_idx_ptr, expert_offsets_ptr, num_groups, expert_idx,    \
//             num_experts, topk, prob_m, prob_n, prob_k, tot_m, locks,          \
//             replicate_input, apply_weights, m_block, max_par,                 \
//             cfg_max_m_blocks);                                                \
//   }

// #define GPTQ_CALL_IF_MOE(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)             \
//   __CALL_IF_MOE(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, false, 0, NUM_THREADS)   \
//   __CALL_IF_MOE(W_TYPE, 2, N_BLOCKS, K_BLOCKS, true, false, 0, NUM_THREADS)   \
//   __CALL_IF_MOE(W_TYPE, 3, N_BLOCKS, K_BLOCKS, true, false, 0, NUM_THREADS)   \
//   __CALL_IF_MOE(W_TYPE, 4, N_BLOCKS, K_BLOCKS, true, false, 0, NUM_THREADS)   \
//                                                                               \
//   __CALL_IF_MOE(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, false, -1, NUM_THREADS) \
//   __CALL_IF_MOE(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, false, 2, NUM_THREADS)  \
//   __CALL_IF_MOE(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, false, 4, NUM_THREADS)  \
//   __CALL_IF_MOE(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, false, 8, NUM_THREADS)  \
//                                                                               \
//   __CALL_IF_MOE(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, false, -1, NUM_THREADS) \
//   __CALL_IF_MOE(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, false, 2, NUM_THREADS)  \
//   __CALL_IF_MOE(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, false, 4, NUM_THREADS)  \
//   __CALL_IF_MOE(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, false, 8, NUM_THREADS)  \
//                                                                               \
//   __CALL_IF_MOE(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, false, -1, NUM_THREADS) \
//   __CALL_IF_MOE(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, false, 2, NUM_THREADS)  \
//   __CALL_IF_MOE(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, false, 4, NUM_THREADS)  \
//   __CALL_IF_MOE(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, false, 8, NUM_THREADS)  \
//                                                                               \
//   __CALL_IF_MOE(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, false, -1, NUM_THREADS) \
//   __CALL_IF_MOE(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, false, 2, NUM_THREADS)  \
//   __CALL_IF_MOE(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, false, 4, NUM_THREADS)  \
//   __CALL_IF_MOE(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, false, 8, NUM_THREADS)

// #define AWQ_CALL_IF_MOE(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)             \
//   __CALL_IF_MOE(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, true, -1, NUM_THREADS) \
//   __CALL_IF_MOE(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, true, 2, NUM_THREADS)  \
//   __CALL_IF_MOE(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, true, 4, NUM_THREADS)  \
//   __CALL_IF_MOE(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, true, 8, NUM_THREADS)  \
//                                                                              \
//   __CALL_IF_MOE(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, true, -1, NUM_THREADS) \
//   __CALL_IF_MOE(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, true, 2, NUM_THREADS)  \
//   __CALL_IF_MOE(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, true, 4, NUM_THREADS)  \
//   __CALL_IF_MOE(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, true, 8, NUM_THREADS)  \
//                                                                              \
//   __CALL_IF_MOE(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, true, -1, NUM_THREADS) \
//   __CALL_IF_MOE(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, true, 2, NUM_THREADS)  \
//   __CALL_IF_MOE(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, true, 4, NUM_THREADS)  \
//   __CALL_IF_MOE(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, true, 8, NUM_THREADS)  \
//                                                                              \
//   __CALL_IF_MOE(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, true, -1, NUM_THREADS) \
//   __CALL_IF_MOE(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, true, 2, NUM_THREADS)  \
//   __CALL_IF_MOE(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, true, 4, NUM_THREADS)  \
//   __CALL_IF_MOE(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, true, 8, NUM_THREADS)


}  // namespace marlin_moe
