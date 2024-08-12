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

#include "marlin.cuh"
#include "marlin_dtypes.cuh"
#include "core/scalar_type.hpp"

#define STATIC_ASSERT_SCALAR_TYPE_VALID(scalar_t)               \
  static_assert(std::is_same<scalar_t, half>::value ||          \
                    std::is_same<scalar_t, nv_bfloat16>::value, \
                "only float16 and bfloat16 is supported");

template <typename T>
inline std::string str(T x) {
  return std::to_string(x);
}

namespace marlin {

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800

__global__ void permute_cols_kernel(int4 const* __restrict__ a_int4_ptr,
                                    int const* __restrict__ perm_int_ptr,
                                    int4* __restrict__ out_int4_ptr, int size_m,
                                    int size_k, int block_rows) {}

template <typename scalar_t,          // compute dtype, half or nv_float16
          const int num_bits,         // number of bits used for weights
          const int threads,          // number of threads in a threadblock
          const int thread_m_blocks,  // number of 16x16 blocks in the m
                                      // dimension (batchsize) of the
                                      // threadblock
          const int thread_n_blocks,  // same for n dimension (output)
          const int thread_k_blocks,  // same for k dimension (reduction)
          const int stages,  // number of stages for the async global->shared
                             // fetch pipeline
          const bool has_act_order,    // whether act_order is enabled
          const int group_blocks = -1  // number of consecutive 16x16 blocks
                                       // with a separate quantization scale
          >
__global__ void Marlin(
    const int4* __restrict__ A,  // fp16 input matrix of shape mxk
    const int4* __restrict__ B,  // 4bit quantized weight matrix of shape kxn
    int4* __restrict__ C,        // fp16 output buffer of shape mxn
    int4* __restrict__ C_tmp,    // fp32 tmp output buffer (for reduce)
    const int4* __restrict__ scales_ptr,  // fp16 quantization scales of shape
                                          // (k/groupsize)xn
    const int* __restrict__ g_idx,        // int32 group indices of shape k
    int num_groups,       // number of scale groups per output channel
    int prob_m,           // batch dimension m
    int prob_n,           // output dimension n
    int prob_k,           // reduction dimension k
    int* locks,           // extra global storage for barrier synchronization
    bool use_fp32_reduce  // whether to use fp32 global reduce
) {}

}  // namespace marlin

torch::Tensor gptq_marlin_gemm(torch::Tensor& a, torch::Tensor& b_q_weight,
                               torch::Tensor& b_scales, torch::Tensor& b_zeros,
                               torch::Tensor& g_idx, torch::Tensor& perm,
                               torch::Tensor& workspace,
                               vllm::ScalarTypeTorchPtr const& b_q_type,
                               int64_t size_m, int64_t size_n, int64_t size_k,
                               bool is_k_full, bool has_zp) {
  TORCH_CHECK_NOT_IMPLEMENTED(false,
                              "marlin_gemm(..) requires CUDA_ARCH >= 8.0");
  return torch::empty({1, 1});
}

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

// Instruction for loading a full 16x16 matrix fragment of operand A from shared
// memory, directly in tensor core layout.
template <typename scalar_t>
__device__ inline void ldsm4(typename ScalarType<scalar_t>::FragA& frag_a,
                             const void* smem_ptr) {
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

// Efficiently dequantize an int32 value into a full B-fragment of 4 fp16
// values. We mostly follow the strategy in the link below, with some small
// changes:
// - FP16:
// https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h#L215-L287
// - BF16:
// https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h#L327-L385
template <typename scalar_t>
__device__ inline typename ScalarType<scalar_t>::FragB dequant_4bit(int q) {
  STATIC_ASSERT_SCALAR_TYPE_VALID(scalar_t);
}

template <>
__device__ inline typename ScalarType<half>::FragB dequant_4bit<half>(int q) {
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
  typename ScalarType<half>::FragB frag_b;
  frag_b[0] = __hsub2(*reinterpret_cast<half2*>(&lo),
                      *reinterpret_cast<const half2*>(&SUB));
  frag_b[1] = __hfma2(*reinterpret_cast<half2*>(&hi),
                      *reinterpret_cast<const half2*>(&MUL),
                      *reinterpret_cast<const half2*>(&ADD));
  return frag_b;
}

template <>
__device__ inline typename ScalarType<nv_bfloat16>::FragB
dequant_4bit<nv_bfloat16>(int q) {
  static constexpr uint32_t MASK = 0x000f000f;
  static constexpr uint32_t EX = 0x43004300;

  // Guarantee that the `(a & b) | c` operations are LOP3s.

  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
  q >>= 4;
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);

  typename ScalarType<nv_bfloat16>::FragB frag_b;
  static constexpr uint32_t MUL = 0x3F803F80;
  static constexpr uint32_t ADD = 0xC308C308;

  frag_b[0] = __hfma2(*reinterpret_cast<nv_bfloat162*>(&lo),
                      *reinterpret_cast<const nv_bfloat162*>(&MUL),
                      *reinterpret_cast<const nv_bfloat162*>(&ADD));
  frag_b[1] = __hfma2(*reinterpret_cast<nv_bfloat162*>(&hi),
                      *reinterpret_cast<const nv_bfloat162*>(&MUL),
                      *reinterpret_cast<const nv_bfloat162*>(&ADD));
  return frag_b;
}

// Fast Int8ToFp16/Int8ToBf16: Efficiently dequantize 8bit int values to fp16 or
// bf16 Reference:
// - FP16:
// https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h#L53-L85
// - BF16:
// https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h#L125-L175
template <typename scalar_t>
__device__ inline typename ScalarType<scalar_t>::FragB dequant_8bit(int q) {
  STATIC_ASSERT_SCALAR_TYPE_VALID(scalar_t);
}

template <>
__device__ inline typename ScalarType<half>::FragB dequant_8bit<half>(int q) {
  static constexpr uint32_t mask_for_elt_01 = 0x5250;
  static constexpr uint32_t mask_for_elt_23 = 0x5351;
  static constexpr uint32_t start_byte_for_fp16 = 0x64646464;

  uint32_t lo = prmt<start_byte_for_fp16, mask_for_elt_01>(q);
  uint32_t hi = prmt<start_byte_for_fp16, mask_for_elt_23>(q);

  static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64806480;

  typename ScalarType<half>::FragB frag_b;
  frag_b[0] = __hsub2(*reinterpret_cast<half2*>(&lo),
                      *reinterpret_cast<const half2*>(&I8s_TO_F16s_MAGIC_NUM));
  frag_b[1] = __hsub2(*reinterpret_cast<half2*>(&hi),
                      *reinterpret_cast<const half2*>(&I8s_TO_F16s_MAGIC_NUM));
  return frag_b;
}

template <>
__device__ inline typename ScalarType<nv_bfloat16>::FragB
dequant_8bit<nv_bfloat16>(int q) {
  typename ScalarType<nv_bfloat16>::FragB frag_b;

  float fp32_intermediates[4];
  uint32_t* fp32_intermediates_casted =
      reinterpret_cast<uint32_t*>(fp32_intermediates);

  static constexpr uint32_t fp32_base = 0x4B000000;
  fp32_intermediates_casted[0] = __byte_perm(q, fp32_base, 0x7650);
  fp32_intermediates_casted[1] = __byte_perm(q, fp32_base, 0x7652);
  fp32_intermediates_casted[2] = __byte_perm(q, fp32_base, 0x7651);
  fp32_intermediates_casted[3] = __byte_perm(q, fp32_base, 0x7653);

  fp32_intermediates[0] -= 8388736.f;
  fp32_intermediates[1] -= 8388736.f;
  fp32_intermediates[2] -= 8388736.f;
  fp32_intermediates[3] -= 8388736.f;

  uint32_t* bf16_result_ptr = reinterpret_cast<uint32_t*>(&frag_b);
  bf16_result_ptr[0] = __byte_perm(fp32_intermediates_casted[0],
                                   fp32_intermediates_casted[1], 0x7632);
  bf16_result_ptr[1] = __byte_perm(fp32_intermediates_casted[2],
                                   fp32_intermediates_casted[3], 0x7632);

  return frag_b;
}

// Zero-point dequantizers

template <typename scalar_t>
__device__ inline typename ScalarType<scalar_t>::FragB dequant_4bit_zp(int q) {
  STATIC_ASSERT_SCALAR_TYPE_VALID(scalar_t);
}

template <>
__device__ inline typename ScalarType<half>::FragB dequant_4bit_zp<half>(
    int q) {
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400;
  // Guarantee that the `(a & b) | c` operations are LOP3s.
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);

  const int SUB = 0x64006400;
  const int MUL = 0x2c002c00;
  const int ADD = 0xd400d400;
  typename ScalarType<half>::FragB frag_b;
  frag_b[0] = __hsub2(*reinterpret_cast<half2*>(&lo),
                      *reinterpret_cast<const half2*>(&SUB));
  frag_b[1] = __hfma2(*reinterpret_cast<half2*>(&hi),
                      *reinterpret_cast<const half2*>(&MUL),
                      *reinterpret_cast<const half2*>(&ADD));
  return frag_b;
}

template <>
__device__ inline typename ScalarType<nv_bfloat16>::FragB
dequant_4bit_zp<nv_bfloat16>(int q) {
  static constexpr uint32_t MASK = 0x000f000f;
  static constexpr uint32_t EX = 0x43004300;

  // Guarantee that the `(a & b) | c` operations are LOP3s.

  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
  q >>= 4;
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);

  typename ScalarType<nv_bfloat16>::FragB frag_b;
  static constexpr uint32_t MUL = 0x3F803F80;
  static constexpr uint32_t ADD = 0xC300C300;

  frag_b[0] = __hfma2(*reinterpret_cast<nv_bfloat162*>(&lo),
                      *reinterpret_cast<const nv_bfloat162*>(&MUL),
                      *reinterpret_cast<const nv_bfloat162*>(&ADD));
  frag_b[1] = __hfma2(*reinterpret_cast<nv_bfloat162*>(&hi),
                      *reinterpret_cast<const nv_bfloat162*>(&MUL),
                      *reinterpret_cast<const nv_bfloat162*>(&ADD));
  return frag_b;
}

template <typename scalar_t>
__device__ inline typename ScalarType<scalar_t>::FragB dequant_8bit_zp(int q) {
  STATIC_ASSERT_SCALAR_TYPE_VALID(scalar_t);
}

template <>
__device__ inline typename ScalarType<half>::FragB dequant_8bit_zp<half>(
    int q) {
  static constexpr uint32_t mask_for_elt_01 = 0x5250;
  static constexpr uint32_t mask_for_elt_23 = 0x5351;
  static constexpr uint32_t start_byte_for_fp16 = 0x64646464;

  uint32_t lo = prmt<start_byte_for_fp16, mask_for_elt_01>(q);
  uint32_t hi = prmt<start_byte_for_fp16, mask_for_elt_23>(q);

  static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64006400;

  typename ScalarType<half>::FragB frag_b;
  frag_b[0] = __hsub2(*reinterpret_cast<half2*>(&lo),
                      *reinterpret_cast<const half2*>(&I8s_TO_F16s_MAGIC_NUM));
  frag_b[1] = __hsub2(*reinterpret_cast<half2*>(&hi),
                      *reinterpret_cast<const half2*>(&I8s_TO_F16s_MAGIC_NUM));
  return frag_b;
}

template <>
__device__ inline typename ScalarType<nv_bfloat16>::FragB
dequant_8bit_zp<nv_bfloat16>(int q) {
  typename ScalarType<nv_bfloat16>::FragB frag_b;

  float fp32_intermediates[4];
  uint32_t* fp32_intermediates_casted =
      reinterpret_cast<uint32_t*>(fp32_intermediates);

  static constexpr uint32_t fp32_base = 0x4B000000;
  fp32_intermediates_casted[0] = __byte_perm(q, fp32_base, 0x7650);
  fp32_intermediates_casted[1] = __byte_perm(q, fp32_base, 0x7652);
  fp32_intermediates_casted[2] = __byte_perm(q, fp32_base, 0x7651);
  fp32_intermediates_casted[3] = __byte_perm(q, fp32_base, 0x7653);

  fp32_intermediates[0] -= 8388608.f;
  fp32_intermediates[1] -= 8388608.f;
  fp32_intermediates[2] -= 8388608.f;
  fp32_intermediates[3] -= 8388608.f;

  uint32_t* bf16_result_ptr = reinterpret_cast<uint32_t*>(&frag_b);
  bf16_result_ptr[0] = __byte_perm(fp32_intermediates_casted[0],
                                   fp32_intermediates_casted[1], 0x7632);
  bf16_result_ptr[1] = __byte_perm(fp32_intermediates_casted[2],
                                   fp32_intermediates_casted[3], 0x7632);

  return frag_b;
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

// For a given "a" of size [M,K] performs a permutation of the K columns based
// on the given "perm" indices.
__global__ void permute_cols_kernel(int4 const* __restrict__ a_int4_ptr,
                                    int const* __restrict__ perm_int_ptr,
                                    int4* __restrict__ out_int4_ptr, int size_m,
                                    int size_k, int block_rows) {
  int start_row = block_rows * blockIdx.x;
  int finish_row = start_row + block_rows;
  if (finish_row > size_m) {
    finish_row = size_m;
  }
  int cur_block_rows = finish_row - start_row;

  int row_stride = size_k * sizeof(half) / 16;

  auto permute_row = [&](int row) {
    int iters = size_k / default_threads;
    int rest = size_k % default_threads;

    int offset = row * row_stride;

    half const* a_row_half = reinterpret_cast<half const*>(a_int4_ptr + offset);
    half* out_half = reinterpret_cast<half*>(out_int4_ptr + offset);

    int base_k = 0;

    for (int i = 0; i < iters; i++) {
      int cur_k = base_k + threadIdx.x;
      int src_pos = perm_int_ptr[cur_k];

      out_half[cur_k] = a_row_half[src_pos];

      base_k += default_threads;
    }

    if (rest) {
      if (threadIdx.x < rest) {
        int cur_k = base_k + threadIdx.x;
        int src_pos = perm_int_ptr[cur_k];

        out_half[cur_k] = a_row_half[src_pos];
      }
    }
  };

  for (int i = 0; i < cur_block_rows; i++) {
    int cur_row = start_row + i;
    if (cur_row < size_m) {
      permute_row(cur_row);
    }
  }
}

template <typename scalar_t,          // compute dtype, half or nv_float16
          const int num_bits,         // number of bits used for weights
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
    int num_groups,       // number of scale groups per output channel
    int prob_m,           // batch dimension m
    int prob_n,           // output dimension n
    int prob_k,           // reduction dimension k
    int* locks,           // extra global storage for barrier synchronization
    bool use_fp32_reduce  // whether to use fp32 global reduce
) {
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

  constexpr int pack_factor = 32 / num_bits;

  // For larger GEMMs we run multiple batchsize 64 versions in parallel for a
  // better partitioning with less reductions
  int parallel = 1;
  if (prob_m > 16 * thread_m_blocks) {
    parallel = prob_m / (16 * thread_m_blocks);
    prob_m = 16 * thread_m_blocks;
  }

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

  // We can easily implement parallel problem execution by just remapping
  // indices and advancing global pointers
  if (slice_col_par >= n_tiles) {
    A += (slice_col_par / n_tiles) * 16 * thread_m_blocks * prob_k / 8;
    C += (slice_col_par / n_tiles) * 16 * thread_m_blocks * prob_n / 8;
    locks += (slice_col_par / n_tiles) * n_tiles;
    slice_col = slice_col_par % n_tiles;
    par_id = slice_col_par / n_tiles;
  }

  // Compute all information about the current slice which is required for
  // synchronization.
  auto init_slice = [&]() {
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
    if (slice_col == n_tiles) {
      A += 16 * thread_m_blocks * prob_k / 8;
      C += 16 * thread_m_blocks * prob_n / 8;
      locks += n_tiles;
      slice_col = 0;
      par_id++;
    }
  };
  init_slice();

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
  constexpr int b_thread_vecs = num_bits == 4 ? 1 : 2;
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
          ? thread_k_blocks / group_blocks
          : 1;
  constexpr int s_sh_stage = s_tb_groups * s_sh_stride;
  int s_gl_rd_delta = s_gl_stride;

  // Scale size/strides with act_order
  constexpr int tb_k = 16 * thread_k_blocks;
  constexpr int g_idx_stage = has_act_order ? (tb_k * sizeof(int)) / 16 : 0;
  // constexpr int act_s_row_stride      = 1;
  // int           act_s_col_stride      = act_s_row_stride * num_groups;
  int act_s_col_stride = 1;
  int act_s_col_warp_stride = act_s_col_stride * 8;
  int tb_n_warps = thread_n_blocks / 4;
  int act_s_col_tb_stride = act_s_col_warp_stride * tb_n_warps;

  // Zero-points sizes/strides
  int zp_gl_stride = (prob_n / pack_factor) / 4;
  constexpr int zp_sh_stride = ((16 * thread_n_blocks) / pack_factor) / 4;
  constexpr int zp_tb_groups = s_tb_groups;
  constexpr int zp_sh_stage = has_zp ? zp_tb_groups * zp_sh_stride : 0;
  int zp_gl_rd_delta = zp_gl_stride;

  // Global A read index of current thread.
  int a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) +
                (threadIdx.x % a_gl_rd_delta_o);
  a_gl_rd += a_gl_rd_delta_o * slice_row;
  // Shared write index of current thread.
  int a_sh_wr = a_sh_stride * (threadIdx.x / a_gl_rd_delta_o) +
                (threadIdx.x % a_gl_rd_delta_o);
  // Shared read index.
  int a_sh_rd =
      a_sh_stride * ((threadIdx.x % 32) % 16) + (threadIdx.x % 32) / 16;
  a_sh_rd += 2 * ((threadIdx.x / 32) / (thread_n_blocks / 4));

  int b_gl_rd = b_gl_stride * (threadIdx.x / b_sh_stride_threads) +
                (threadIdx.x % b_sh_stride_threads) * b_thread_vecs;
  b_gl_rd += b_sh_stride * slice_col;
  b_gl_rd += b_gl_rd_delta_o * slice_row;
  int b_sh_wr = threadIdx.x * b_thread_vecs;
  int b_sh_rd = threadIdx.x * b_thread_vecs;

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
      s_gl_rd = s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) +
                s_sh_stride * slice_col + threadIdx.x;
    }
  }
  int s_sh_wr = threadIdx.x;
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
  int zp_sh_wr = threadIdx.x;
  bool zp_sh_wr_pred = threadIdx.x < zp_sh_stride;

  // We use a different scale layout for grouped and column-wise quantization as
  // we scale a `half2` tile in column-major layout in the former and in
  // row-major in the latter case.
  int s_sh_rd;
  if constexpr (group_blocks != -1)
    s_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) +
              (threadIdx.x % 32) / 4;
  else
    s_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) +
              (threadIdx.x % 32) % 4;

  // Zero-points have the same read layout as the scales
  // (without column-wise case)
  constexpr int num_col_threads = 8;
  constexpr int num_row_threads = 4;
  constexpr int num_ints_per_thread = 8 / pack_factor;
  int zp_sh_rd;
  if constexpr (has_zp) {
    zp_sh_rd = num_ints_per_thread * num_col_threads *
                   ((threadIdx.x / 32) % (thread_n_blocks / 4)) +
               num_ints_per_thread * ((threadIdx.x % 32) / num_row_threads);
  }

  // Precompute which thread should not read memory in which iterations; this is
  // needed if there are more threads than required for a certain tilesize or
  // when the batchsize is not a multiple of 16.
  bool a_sh_wr_pred[a_sh_wr_iters];
  #pragma unroll
  for (int i = 0; i < a_sh_wr_iters; i++)
    a_sh_wr_pred[i] = a_sh_wr_delta * i + a_sh_wr < a_sh_stride * prob_m;

  // To ensure that writing and reading A tiles to/from shared memory, the
  // latter in fragment format, is fully bank conflict free, we need to use a
  // rather fancy XOR-based layout. The key here is that neither reads nor
  // writes of the 16-byte `int4` blocks of 8 consecutive threads involve the
  // same shared memory banks. Further, it seems (based on NSight-Compute) that
  // each warp must also write a consecutive memory segment?
  auto transform_a = [&](int i) {
    int row = i / a_gl_rd_delta_o;
    return a_gl_rd_delta_o * row + (i % a_gl_rd_delta_o) ^ row;
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

  extern __shared__ int4 sh[];
  // Shared memory storage for global fetch pipelines.
  int4* sh_a = sh;
  int4* sh_b = sh_a + (stages * a_sh_stage);
  int4* sh_g_idx = sh_b + (stages * b_sh_stage);
  int4* sh_zp = sh_g_idx + (stages * g_idx_stage);
  int4* sh_s = sh_zp + (stages * zp_sh_stage);

  // Register storage for double buffer of shared memory reads.
  FragA frag_a[2][thread_m_blocks];
  I4 frag_b_quant[2][b_thread_vecs];
  FragC frag_c[thread_m_blocks][4][2];
  FragS frag_s[2][4];                    // No act-order
  FragS act_frag_s[2][4][4];             // For act-order
  int frag_qzp[2][num_ints_per_thread];  // Zero-points
  FragZP frag_zp;                        // Zero-points in fp16

  // Zero accumulators.
  auto zero_accums = [&]() {
  #pragma unroll
    for (int i = 0; i < thread_m_blocks * 4 * 2 * 4; i++)
      reinterpret_cast<float*>(frag_c)[i] = 0;
  };

  int sh_first_group_id = -1;
  int sh_num_groups = -1;
  constexpr int sh_max_num_groups = 32;

  auto fetch_scales_to_shared = [&](bool is_async, int first_group_id,
                                    int last_group_id) {
    sh_first_group_id = first_group_id;
    sh_num_groups = last_group_id - first_group_id + 1;

    if (sh_num_groups < sh_max_num_groups) {
      sh_num_groups = sh_max_num_groups;
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
  auto fetch_to_shared = [&](int pipe, int a_off, bool pred = true) {
    if (pred) {
      int4* sh_a_stage = sh_a + a_sh_stage * pipe;
  #pragma unroll
      for (int i = 0; i < a_sh_wr_iters; i++) {
        cp_async4_pred(
            &sh_a_stage[a_sh_wr_trans[i]],
            &A[a_gl_rd_delta_i * i + a_gl_rd + a_gl_rd_delta_o * a_off],
            a_sh_wr_pred[i]);
      }
      int4* sh_b_stage = sh_b + b_sh_stage * pipe;
  #pragma unroll
      for (int i = 0; i < b_sh_wr_iters; i++) {
  #pragma unroll
        for (int j = 0; j < b_thread_vecs; j++) {
          cp_async4(&sh_b_stage[b_sh_wr_delta * i + b_sh_wr + j], B_ptr[i] + j);
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

  auto fetch_zp_to_shared = [&]() {
    if (zp_sh_wr_pred) {
      cp_async4(&sh_zp[zp_sh_wr], &zp_ptr[zp_gl_rd]);
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
  auto fetch_to_registers = [&](int k, int pipe) {
    int4* sh_a_stage = sh_a + a_sh_stage * pipe;
  #pragma unroll
    for (int i = 0; i < thread_m_blocks; i++)
      ldsm4<scalar_t>(frag_a[k % 2][i],
                      &sh_a_stage[a_sh_rd_trans[k % b_sh_wr_iters][i]]);
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
      is_same_group[pipe] = false;
      same_group_id[pipe] = 0;
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
      if constexpr (group_blocks != -1) {
        if constexpr (group_blocks >= thread_k_blocks) {
          int4* sh_s_stage =
              sh_s + s_sh_stage * ((group_blocks / thread_k_blocks) *
                                   (pipe / (group_blocks / thread_k_blocks)));
          reinterpret_cast<int4*>(&frag_s[k % 2])[0] = sh_s_stage[s_sh_rd];
        } else {
          int warp_id = threadIdx.x / 32;
          int n_warps = thread_n_blocks / 4;

          int warp_row = warp_id / n_warps;

          int cur_k = warp_row * 16;
          cur_k += k_iter_size * (k % b_sh_wr_iters);

          int k_blocks = cur_k / 16;
          int cur_group_id = k_blocks / group_blocks;

          int4* sh_s_stage = sh_s + s_sh_stage * pipe;

          reinterpret_cast<int4*>(&frag_s[k % 2])[0] =
              sh_s_stage[s_sh_rd + cur_group_id * s_sh_stride];
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
    int warp_id = threadIdx.x / 32;
    int n_warps =
        thread_n_blocks / 4;  // Each warp processes 4 16-size tiles over N

    int warp_row = warp_id / n_warps;
    int warp_col = warp_id % n_warps;

    cur_k += warp_row * 16;

    int th_id = threadIdx.x % 32;
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

    if constexpr (has_zp) {
      int pipe = full_pipe % stages;

      if constexpr (group_blocks == -1) {
        for (int i = 0; i < num_ints_per_thread; i++) {
          frag_qzp[k % 2][i] = (reinterpret_cast<int*>(sh_zp))[zp_sh_rd + i];
        }

      } else if constexpr (group_blocks >= thread_k_blocks) {
        int4* sh_zp_stage =
            sh_zp + zp_sh_stage * ((group_blocks / thread_k_blocks) *
                                   (pipe / (group_blocks / thread_k_blocks)));
        for (int i = 0; i < num_ints_per_thread; i++) {
          frag_qzp[k % 2][i] =
              (reinterpret_cast<int*>(sh_zp_stage))[zp_sh_rd + i];
        }
      } else {
        int warp_id = threadIdx.x / 32;
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

        for (int i = 0; i < num_ints_per_thread; i++) {
          frag_qzp[k % 2][i] =
              (reinterpret_cast<int*>(sh_zp_stage))[zp_sh_rd + i];
        }
      }
    }
  };

  // Execute the actual tensor core matmul of a sub-tile.
  auto matmul = [&](int k) {
    if constexpr (has_zp) {
      FragB frag_zp_0;
      FragB frag_zp_1;
      if constexpr (num_bits == 4) {
        int zp_quant = frag_qzp[k % 2][0];
        int zp_quant_shift = zp_quant >> 8;
        frag_zp_0 = dequant_4bit_zp<scalar_t>(zp_quant);
        frag_zp_1 = dequant_4bit_zp<scalar_t>(zp_quant_shift);

      } else {
        int zp_quant_0 = frag_qzp[k % 2][0];
        int zp_quant_1 = frag_qzp[k % 2][1];
        frag_zp_0 = dequant_8bit_zp<scalar_t>(zp_quant_0);
        frag_zp_1 = dequant_8bit_zp<scalar_t>(zp_quant_1);
      }

      frag_zp[0] = frag_zp_0[0];
      frag_zp[1] = frag_zp_0[1];
      frag_zp[2] = frag_zp_1[0];
      frag_zp[3] = frag_zp_1[1];
    }

  // We have the m dimension as the inner loop in order to encourage overlapping
  // dequantization and matmul operations.
  #pragma unroll
    for (int j = 0; j < 4; j++) {
      FragB frag_b0;
      FragB frag_b1;
      if constexpr (num_bits == 4) {
        int b_quant = frag_b_quant[k % 2][0][j];
        int b_quant_shift = b_quant >> 8;

        if constexpr (has_zp) {
          frag_b0 = dequant_4bit_zp<scalar_t>(b_quant);
          frag_b1 = dequant_4bit_zp<scalar_t>(b_quant_shift);

        } else {
          frag_b0 = dequant_4bit<scalar_t>(b_quant);
          frag_b1 = dequant_4bit<scalar_t>(b_quant_shift);
        }

      } else {
        int* frag_b_quant_ptr = reinterpret_cast<int*>(frag_b_quant[k % 2]);
        int b_quant_0 = frag_b_quant_ptr[j * 2 + 0];
        int b_quant_1 = frag_b_quant_ptr[j * 2 + 1];

        if constexpr (has_zp) {
          frag_b0 = dequant_8bit_zp<scalar_t>(b_quant_0);
          frag_b1 = dequant_8bit_zp<scalar_t>(b_quant_1);
        } else {
          frag_b0 = dequant_8bit<scalar_t>(b_quant_0);
          frag_b1 = dequant_8bit<scalar_t>(b_quant_1);
        }
      }

      // Apply zero-point to frag_b0
      if constexpr (has_zp) {
        sub_zp<scalar_t>(frag_b0, frag_zp[j], 0);
      }

      // Apply scale to frag_b0
      if constexpr (has_act_order) {
        scale4<scalar_t>(frag_b0, act_frag_s[k % 2][0][j],
                         act_frag_s[k % 2][1][j], act_frag_s[k % 2][2][j],
                         act_frag_s[k % 2][3][j], 0);
      } else {
        if constexpr (group_blocks != -1) {
          scale<scalar_t>(frag_b0, frag_s[k % 2][j], 0);
        }
      }

      // Apply zero-point to frag_b1
      if constexpr (has_zp) {
        sub_zp<scalar_t>(frag_b1, frag_zp[j], 1);
      }

      // Apply scale to frag_b1
      if constexpr (has_act_order) {
        scale4<scalar_t>(frag_b1, act_frag_s[k % 2][0][j],
                         act_frag_s[k % 2][1][j], act_frag_s[k % 2][2][j],
                         act_frag_s[k % 2][3][j], 1);

      } else {
        if constexpr (group_blocks != -1) {
          scale<scalar_t>(frag_b1, frag_s[k % 2][j], 1);
        }
      }

  #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
        mma<scalar_t>(frag_a[k % 2][i], frag_b0, frag_c[i][j][0]);
        mma<scalar_t>(frag_a[k % 2][i], frag_b1, frag_c[i][j][1]);
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
      int red_idx = threadIdx.x / b_sh_stride_threads;
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
            for (int j = 0; j < 4 * 2; j++) {
              int red_sh_wr =
                  red_sh_delta * j + (red_sh_rd - red_sh_stride * i);
              if (i < red_off) {
                float* c_rd =
                    reinterpret_cast<float*>(&sh[red_sh_delta * j + red_sh_rd]);
                float* c_wr = reinterpret_cast<float*>(&sh[red_sh_wr]);
  #pragma unroll
                for (int k = 0; k < 4; k++)
                  reinterpret_cast<FragC*>(frag_c)[4 * 2 * m_block + j][k] +=
                      c_rd[k] + c_wr[k];
              }
              sh[red_sh_wr] =
                  reinterpret_cast<int4*>(&frag_c)[4 * 2 * m_block + j];
            }
          }
          __syncthreads();
        }
        if (red_idx == 0) {
  #pragma unroll
          for (int i = 0; i < 4 * 2; i++) {
            float* c_rd =
                reinterpret_cast<float*>(&sh[red_sh_delta * i + red_sh_rd]);
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
    if (threadIdx.x < active_threads) {
      int c_gl_stride = prob_n / 8;
      int c_gl_wr_delta_o = 8 * c_gl_stride;
      int c_gl_wr_delta_i = 4 * (active_threads / 32);
      int c_gl_wr = c_gl_stride * ((threadIdx.x % 32) / 4) +
                    4 * (threadIdx.x / 32) + threadIdx.x % 4;
      c_gl_wr += (2 * thread_n_blocks) * slice_col;
      constexpr int c_sh_wr_delta = active_threads;
      int c_sh_wr = threadIdx.x;

      int row = (threadIdx.x % 32) / 4;

      if (!first) {
  // Interestingly, doing direct global accesses here really seems to mess up
  // the compiler and lead to slowdowns, hence we also use async-copies even
  // though these fetches are not actually asynchronous.
  #pragma unroll
        for (int i = 0; i < thread_m_blocks * 4; i++) {
          cp_async4_pred(
              &sh[c_sh_wr + c_sh_wr_delta * i],
              &C[c_gl_wr + c_gl_wr_delta_o * (i / 2) +
                 c_gl_wr_delta_i * (i % 2)],
              i < (thread_m_blocks - 1) * 4 || 8 * (i / 2) + row < prob_m);
        }
        cp_async_fence();
        cp_async_wait<0>();
      }

  #pragma unroll
      for (int i = 0; i < thread_m_blocks * 4; i++) {
        if (i < (thread_m_blocks - 1) * 4 || 8 * (i / 2) + row < prob_m) {
          if (!first) {
            int4 c_red = sh[c_sh_wr + i * c_sh_wr_delta];
  #pragma unroll
            for (int j = 0; j < 2 * 4; j++) {
              reinterpret_cast<float*>(
                  &frag_c)[4 * 2 * 4 * (i / 4) + 4 * j + (i % 4)] +=
                  Dtype::num2float(reinterpret_cast<scalar_t*>(&c_red)[j]);
            }
          }
          if (!last) {
            int4 c;
  #pragma unroll
            for (int j = 0; j < 2 * 4; j++) {
              reinterpret_cast<scalar_t*>(&c)[j] =
                  Dtype::float2num(reinterpret_cast<float*>(
                      &frag_c)[4 * 2 * 4 * (i / 4) + 4 * j + (i % 4)]);
            }
            C[c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2)] =
                c;
          }
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

    int par_offset = c_size * n_tiles * par_id;
    int slice_offset = c_size * slice_col;

    constexpr int num_floats = thread_m_blocks * 4 * 2 * 4;
    constexpr int th_size = num_floats * sizeof(float) / 16;

    int c_cur_offset = par_offset + slice_offset;

    if (!is_th_active) {
      return;
    }

    if (!first) {
      float* frag_c_ptr = reinterpret_cast<float*>(&frag_c);
  #pragma unroll
      for (int k = 0; k < th_size; k++) {
        sh[threadIdx.x] =
            C_tmp[c_cur_offset + active_threads * k + threadIdx.x];

        float* sh_c_ptr = reinterpret_cast<float*>(&sh[threadIdx.x]);
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
        C_tmp[c_cur_offset + active_threads * k + threadIdx.x] = frag_c_ptr[k];
      }
    }
  };

  // Write out the reduce final result in the correct layout. We only actually
  // reshuffle matrix fragments in this step, the reduction above is performed
  // in fragment layout.
  auto write_result = [&]() {
    int c_gl_stride = prob_n / 8;
    constexpr int c_sh_stride = 2 * thread_n_blocks + 1;
    int c_gl_wr_delta = c_gl_stride * (threads / (2 * thread_n_blocks));
    constexpr int c_sh_rd_delta =
        c_sh_stride * (threads / (2 * thread_n_blocks));

    int c_gl_wr = c_gl_stride * (threadIdx.x / (2 * thread_n_blocks)) +
                  (threadIdx.x % (2 * thread_n_blocks));
    c_gl_wr += (2 * thread_n_blocks) * slice_col;
    int c_sh_wr =
        (4 * c_sh_stride) * ((threadIdx.x % 32) / 4) + (threadIdx.x % 32) % 4;
    c_sh_wr += 32 * (threadIdx.x / 32);
    int c_sh_rd = c_sh_stride * (threadIdx.x / (2 * thread_n_blocks)) +
                  (threadIdx.x % (2 * thread_n_blocks));

    int c_gl_wr_end = c_gl_stride * prob_m;

    // We first reorder in shared memory to guarantee the most efficient final
    // global write patterns
    auto write = [&](int idx, float c0, float c1, FragS& s) {
      scalar_t2 res =
          Dtype::nums2num2(Dtype::float2num(c0), Dtype::float2num(c1));

      // For per-column quantization we finally apply the scale here (only for
      // 4-bit)
      if constexpr (!has_act_order && group_blocks == -1 && num_bits == 4) {
        res = __hmul2(res, s[0]);
      }

      ((scalar_t2*)sh)[idx] = res;
    };

    if (threadIdx.x / 32 < thread_n_blocks / 4) {
  #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
  #pragma unroll
        for (int j = 0; j < 4; j++) {
          int wr = c_sh_wr + 8 * j;
          write(wr + (4 * c_sh_stride) * 0 + 0, frag_c[i][j][0][0],
                frag_c[i][j][0][1], frag_s[j / 2][2 * (j % 2) + 0]);
          write(wr + (4 * c_sh_stride) * 8 + 0, frag_c[i][j][0][2],
                frag_c[i][j][0][3], frag_s[j / 2][2 * (j % 2) + 0]);
          write(wr + (4 * c_sh_stride) * 0 + 4, frag_c[i][j][1][0],
                frag_c[i][j][1][1], frag_s[j / 2][2 * (j % 2) + 1]);
          write(wr + (4 * c_sh_stride) * 8 + 4, frag_c[i][j][1][2],
                frag_c[i][j][1][3], frag_s[j / 2][2 * (j % 2) + 1]);
        }
        c_sh_wr += 16 * (4 * c_sh_stride);
      }
    }
    __syncthreads();

  #pragma unroll
    for (int i = 0;
         i < div_ceil(16 * thread_m_blocks, threads / (2 * thread_n_blocks));
         i++) {
      if (c_gl_wr < c_gl_wr_end) {
        C[c_gl_wr] = sh[c_sh_rd];
        c_gl_wr += c_gl_wr_delta;
        c_sh_rd += c_sh_rd_delta;
      }
    }
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
        fetch_scales_to_shared(true, g_idx[slice_k_start], g_idx[last_g_idx]);
      }

      if constexpr (has_zp && group_blocks == -1) {
        if (i == 0) {
          fetch_zp_to_shared();
        }
      }
      fetch_to_shared(i, i, i < slice_iters);
    }

    zero_accums();
    wait_for_stage();
    init_same_group(0);
    fetch_to_registers(0, 0);
    fetch_scales_to_registers(0, 0);
    fetch_zp_to_registers(0, 0);
    a_gl_rd += a_gl_rd_delta_o * (stages - 1);
    slice_k_start_shared_fetch += tb_k * (stages - 1);
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

  #pragma unroll
    for (int pipe = 0; pipe < stages;) {
  #pragma unroll
      for (int k = 0; k < b_sh_wr_iters; k++) {
        fetch_to_registers(k + 1, pipe % stages);
        fetch_scales_to_registers(k + 1, pipe);
        fetch_zp_to_registers(k + 1, pipe);
        if (k == b_sh_wr_iters - 2) {
          fetch_to_shared((pipe + stages - 1) % stages, pipe,
                          slice_iters >= stages);
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

    a_gl_rd += a_gl_rd_delta_o * stages;
    slice_k_start += tb_k * stages;
    slice_k_start_shared_fetch += tb_k * stages;

    if constexpr (has_act_order) {
      int first_group_id = g_idx[slice_k_start];
      int last_g_idx = slice_k_start + stages * tb_k * 2;
      if (last_g_idx >= prob_k) {
        last_g_idx = prob_k - 1;
      }
      int last_group_id = g_idx[last_g_idx];
      if (last_group_id >= sh_first_group_id + sh_num_groups) {
        fetch_scales_to_shared(false, first_group_id, last_group_id);
        __syncthreads();
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
      if constexpr (!has_act_order && group_blocks == -1) {
        if constexpr (num_bits == 8) {
          if (s_sh_wr_pred) {
            cp_async4(&sh_s[s_sh_wr], &scales_ptr[s_gl_rd]);
          }
          cp_async_fence();
        } else {
          if (last) {
            if (s_sh_wr_pred) {
              cp_async4(&sh_s[s_sh_wr], &scales_ptr[s_gl_rd]);
            }
            cp_async_fence();
          }
        }
      }

      thread_block_reduce();
      if constexpr (!has_act_order && group_blocks == -1) {
        if constexpr (num_bits == 8) {
          cp_async_wait<0>();
          __syncthreads();
          if (threadIdx.x / 32 < thread_n_blocks / 4) {
            reinterpret_cast<int4*>(&frag_s)[0] = sh_s[s_sh_rd + 0];
            reinterpret_cast<int4*>(&frag_s)[1] = sh_s[s_sh_rd + 4];
          }

        } else {
          if (last) {
            cp_async_wait<0>();
            __syncthreads();
            if (threadIdx.x / 32 < thread_n_blocks / 4) {
              reinterpret_cast<int4*>(&frag_s)[0] = sh_s[s_sh_rd + 0];
              reinterpret_cast<int4*>(&frag_s)[1] = sh_s[s_sh_rd + 4];
            }
          }
        }
      }

      // For 8-bit channelwise, we apply the scale before the global reduction
      // that converts the fp32 results to fp16 (so that we avoid possible
      // overflow in fp16)
      if constexpr (!has_act_order && group_blocks == -1 && num_bits == 8) {
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
                  frag_s[j / 2][2 * (j % 2) + 0]);

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

      if (slice_count > 1) {  // only globally reduce if there is more than one
                              // block in a slice
        barrier_acquire(&locks[slice_col], slice_idx);
        if (use_fp32_reduce) {
          global_reduce_fp32(slice_idx == 0, last);
        } else {
          global_reduce_fp16(slice_idx == 0, last);
        }
        barrier_release(&locks[slice_col], last);
      }
      if (last)  // only the last block in a slice actually writes the result
        write_result();
      slice_row = 0;
      slice_col_par++;
      slice_col++;
      init_slice();
      if (slice_iters) {
        a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) +
                  (threadIdx.x % a_gl_rd_delta_o);
  #pragma unroll
        for (int i = 0; i < b_sh_wr_iters; i++)
          B_ptr[i] += b_sh_stride - b_gl_rd_delta_o * k_tiles;
        if (slice_col == 0) {
  #pragma unroll
          for (int i = 0; i < b_sh_wr_iters; i++) B_ptr[i] -= b_gl_stride;
        }

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

  #define __CALL_IF(NUM_BITS, THREAD_M_BLOCKS, THREAD_N_BLOCKS,                \
                    THREAD_K_BLOCKS, HAS_ACT_ORDER, HAS_ZP, GROUP_BLOCKS,      \
                    NUM_THREADS)                                               \
    else if (num_bits == NUM_BITS && thread_m_blocks == THREAD_M_BLOCKS &&     \
             thread_n_blocks == THREAD_N_BLOCKS &&                             \
             thread_k_blocks == THREAD_K_BLOCKS &&                             \
             has_act_order == HAS_ACT_ORDER && has_zp == HAS_ZP &&             \
             group_blocks == GROUP_BLOCKS && num_threads == NUM_THREADS) {     \
      cudaFuncSetAttribute(                                                    \
          Marlin<scalar_t, NUM_BITS, NUM_THREADS, THREAD_M_BLOCKS,             \
                 THREAD_N_BLOCKS, THREAD_K_BLOCKS, pipe_stages, HAS_ACT_ORDER, \
                 HAS_ZP, GROUP_BLOCKS>,                                        \
          cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);        \
      Marlin<scalar_t, NUM_BITS, NUM_THREADS, THREAD_M_BLOCKS,                 \
             THREAD_N_BLOCKS, THREAD_K_BLOCKS, pipe_stages, HAS_ACT_ORDER,     \
             HAS_ZP, GROUP_BLOCKS>                                             \
          <<<blocks, NUM_THREADS, max_shared_mem, stream>>>(                   \
              A_ptr, B_ptr, C_ptr, C_tmp_ptr, s_ptr, zp_ptr, g_idx_ptr,        \
              num_groups, prob_m, prob_n, prob_k, locks, use_fp32_reduce);     \
    }

typedef struct {
  int thread_k;
  int thread_n;
  int num_threads;
} thread_config_t;

typedef struct {
  int max_m_blocks;
  thread_config_t tb_cfg;
} exec_config_t;

thread_config_t small_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {128, 128, 256},
    {64, 128, 128},
    {128, 64, 128},
};

thread_config_t large_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {64, 256, 256},
    {64, 128, 128},
    {128, 64, 128},

};

int get_scales_cache_size(thread_config_t const& th_config, int prob_m,
                          int prob_n, int prob_k, int num_bits, int group_size,
                          bool has_act_order, bool is_k_full) {
  bool cache_scales_chunk = has_act_order && !is_k_full;

  int tb_n = th_config.thread_n;
  int tb_k = th_config.thread_k;

  // Get max scale groups per thread-block
  int tb_groups;
  if (group_size == -1) {
    tb_groups = 1;
  } else if (group_size == 0) {
    tb_groups = div_ceil(tb_k, 32);  // Worst case is 32 group size
  } else {
    tb_groups = div_ceil(tb_k, group_size);
  }

  if (cache_scales_chunk) {
    int load_groups =
        tb_groups * pipe_stages * 2;     // Chunk size is 2x pipeline over dim K
    load_groups = max(load_groups, 32);  // We load at least 32 scale groups
    return load_groups * tb_n * 2;

  } else {
    int tb_scales = tb_groups * tb_n * 2;

    return tb_scales * pipe_stages;
  }
}

bool is_valid_cache_size(thread_config_t const& th_config, int max_m_blocks,
                         int prob_m, int prob_n, int prob_k, int num_bits,
                         int scales_cache_size, int max_shared_mem) {
  int pack_factor = 32 / num_bits;

  // Get B size
  int tb_k = th_config.thread_k;
  int tb_n = th_config.thread_n;

  int b_size = (tb_k * tb_n / pack_factor) * 4;

  // Get A size
  int m_blocks = div_ceil(prob_m, 16);
  int tb_max_m = 16;

  while (true) {
    if (m_blocks >= max_m_blocks) {
      tb_max_m *= max_m_blocks;
      break;
    }

    max_m_blocks--;
    if (max_m_blocks == 0) {
      TORCH_CHECK(false, "Unexpected m_blocks = ", m_blocks);
    }
  }

  int a_size = (tb_max_m * tb_k) * 2;

  float pipe_size = (a_size + b_size) * pipe_stages;

  TORCH_CHECK(max_shared_mem / 2 > scales_cache_size);  // Sanity

  return pipe_size < 0.95f * (max_shared_mem - scales_cache_size);
}

bool is_valid_config(thread_config_t const& th_config, int max_m_blocks,
                     int prob_m, int prob_n, int prob_k, int num_bits,
                     int group_size, bool has_act_order, bool is_k_full,
                     int max_shared_mem) {
  // Sanity
  if (th_config.thread_k == -1 || th_config.thread_n == -1 ||
      th_config.num_threads == -1) {
    return false;
  }

  // Verify K/N are divisible by thread K/N
  if (prob_k % th_config.thread_k != 0 || prob_n % th_config.thread_n != 0) {
    return false;
  }

  // Verify min for thread K/N
  if (th_config.thread_n < min_thread_n || th_config.thread_k < min_thread_k) {
    return false;
  }

  // num_threads must be at least 128 (= 4 warps)
  if (th_config.num_threads < 128) {
    return false;
  }

  //  Determine cache for scales
  int scales_cache_size =
      get_scales_cache_size(th_config, prob_m, prob_n, prob_k, num_bits,
                            group_size, has_act_order, is_k_full);

  // Check that pipeline fits into cache
  if (!is_valid_cache_size(th_config, max_m_blocks, prob_m, prob_n, prob_k,
                           num_bits, scales_cache_size, max_shared_mem)) {
    return false;
  }

  return true;
}

int determine_reduce_max_m(int prob_m, int max_par) {
  constexpr int tile_m_size = 16;

  if (prob_m <= tile_m_size) {
    return tile_m_size;

  } else if (prob_m <= tile_m_size * 2) {
    return tile_m_size * 2;

  } else if (prob_m <= tile_m_size * 3) {
    return tile_m_size * 3;

  } else if (prob_m <= tile_m_size * 4) {
    return tile_m_size * 4;

  } else {
    int cur_par = min(div_ceil(prob_m, tile_m_size * 4), max_par);
    return tile_m_size * 4 * cur_par;
  }
}

exec_config_t determine_thread_config(int prob_m, int prob_n, int prob_k,
                                      int num_bits, int group_size,
                                      bool has_act_order, bool is_k_full,
                                      int max_shared_mem) {
  int max_m_blocks = 4;
  while (max_m_blocks > 0) {
    if (prob_m <= 16) {
      for (auto th_config : small_batch_thread_configs) {
        if (is_valid_config(th_config, max_m_blocks, prob_m, prob_n, prob_k,
                            num_bits, group_size, has_act_order, is_k_full,
                            max_shared_mem)) {
          return exec_config_t{max_m_blocks, th_config};
        }
      }
    } else {
      for (auto th_config : large_batch_thread_configs) {
        if (is_valid_config(th_config, max_m_blocks, prob_m, prob_n, prob_k,
                            num_bits, group_size, has_act_order, is_k_full,
                            max_shared_mem)) {
          return exec_config_t{max_m_blocks, th_config};
        }
      }
    }

    max_m_blocks--;  // Process less M blocks per invocation to reduce cache
                     // usage
  }

  return exec_config_t{0, {-1, -1, -1}};
}

  #define GPTQ_CALL_IF(NUM_BITS, N_BLOCKS, K_BLOCKS, NUM_THREADS)             \
    __CALL_IF(NUM_BITS, 1, N_BLOCKS, K_BLOCKS, true, false, 0, NUM_THREADS)   \
    __CALL_IF(NUM_BITS, 2, N_BLOCKS, K_BLOCKS, true, false, 0, NUM_THREADS)   \
    __CALL_IF(NUM_BITS, 3, N_BLOCKS, K_BLOCKS, true, false, 0, NUM_THREADS)   \
    __CALL_IF(NUM_BITS, 4, N_BLOCKS, K_BLOCKS, true, false, 0, NUM_THREADS)   \
                                                                              \
    __CALL_IF(NUM_BITS, 1, N_BLOCKS, K_BLOCKS, false, false, -1, NUM_THREADS) \
    __CALL_IF(NUM_BITS, 1, N_BLOCKS, K_BLOCKS, false, false, 2, NUM_THREADS)  \
    __CALL_IF(NUM_BITS, 1, N_BLOCKS, K_BLOCKS, false, false, 4, NUM_THREADS)  \
    __CALL_IF(NUM_BITS, 1, N_BLOCKS, K_BLOCKS, false, false, 8, NUM_THREADS)  \
                                                                              \
    __CALL_IF(NUM_BITS, 2, N_BLOCKS, K_BLOCKS, false, false, -1, NUM_THREADS) \
    __CALL_IF(NUM_BITS, 2, N_BLOCKS, K_BLOCKS, false, false, 2, NUM_THREADS)  \
    __CALL_IF(NUM_BITS, 2, N_BLOCKS, K_BLOCKS, false, false, 4, NUM_THREADS)  \
    __CALL_IF(NUM_BITS, 2, N_BLOCKS, K_BLOCKS, false, false, 8, NUM_THREADS)  \
                                                                              \
    __CALL_IF(NUM_BITS, 3, N_BLOCKS, K_BLOCKS, false, false, -1, NUM_THREADS) \
    __CALL_IF(NUM_BITS, 3, N_BLOCKS, K_BLOCKS, false, false, 2, NUM_THREADS)  \
    __CALL_IF(NUM_BITS, 3, N_BLOCKS, K_BLOCKS, false, false, 4, NUM_THREADS)  \
    __CALL_IF(NUM_BITS, 3, N_BLOCKS, K_BLOCKS, false, false, 8, NUM_THREADS)  \
                                                                              \
    __CALL_IF(NUM_BITS, 4, N_BLOCKS, K_BLOCKS, false, false, -1, NUM_THREADS) \
    __CALL_IF(NUM_BITS, 4, N_BLOCKS, K_BLOCKS, false, false, 2, NUM_THREADS)  \
    __CALL_IF(NUM_BITS, 4, N_BLOCKS, K_BLOCKS, false, false, 4, NUM_THREADS)  \
    __CALL_IF(NUM_BITS, 4, N_BLOCKS, K_BLOCKS, false, false, 8, NUM_THREADS)

  #define AWQ_CALL_IF(NUM_BITS, N_BLOCKS, K_BLOCKS, NUM_THREADS)             \
    __CALL_IF(NUM_BITS, 1, N_BLOCKS, K_BLOCKS, false, true, -1, NUM_THREADS) \
    __CALL_IF(NUM_BITS, 1, N_BLOCKS, K_BLOCKS, false, true, 2, NUM_THREADS)  \
    __CALL_IF(NUM_BITS, 1, N_BLOCKS, K_BLOCKS, false, true, 4, NUM_THREADS)  \
    __CALL_IF(NUM_BITS, 1, N_BLOCKS, K_BLOCKS, false, true, 8, NUM_THREADS)  \
                                                                             \
    __CALL_IF(NUM_BITS, 2, N_BLOCKS, K_BLOCKS, false, true, -1, NUM_THREADS) \
    __CALL_IF(NUM_BITS, 2, N_BLOCKS, K_BLOCKS, false, true, 2, NUM_THREADS)  \
    __CALL_IF(NUM_BITS, 2, N_BLOCKS, K_BLOCKS, false, true, 4, NUM_THREADS)  \
    __CALL_IF(NUM_BITS, 2, N_BLOCKS, K_BLOCKS, false, true, 8, NUM_THREADS)  \
                                                                             \
    __CALL_IF(NUM_BITS, 3, N_BLOCKS, K_BLOCKS, false, true, -1, NUM_THREADS) \
    __CALL_IF(NUM_BITS, 3, N_BLOCKS, K_BLOCKS, false, true, 2, NUM_THREADS)  \
    __CALL_IF(NUM_BITS, 3, N_BLOCKS, K_BLOCKS, false, true, 4, NUM_THREADS)  \
    __CALL_IF(NUM_BITS, 3, N_BLOCKS, K_BLOCKS, false, true, 8, NUM_THREADS)  \
                                                                             \
    __CALL_IF(NUM_BITS, 4, N_BLOCKS, K_BLOCKS, false, true, -1, NUM_THREADS) \
    __CALL_IF(NUM_BITS, 4, N_BLOCKS, K_BLOCKS, false, true, 2, NUM_THREADS)  \
    __CALL_IF(NUM_BITS, 4, N_BLOCKS, K_BLOCKS, false, true, 4, NUM_THREADS)  \
    __CALL_IF(NUM_BITS, 4, N_BLOCKS, K_BLOCKS, false, true, 8, NUM_THREADS)

template <typename scalar_t>
void marlin_mm(const void* A, const void* B, void* C, void* C_tmp, void* s,
               void* zp, void* g_idx, void* perm, void* a_tmp, int prob_m,
               int prob_n, int prob_k, void* workspace,
               vllm::ScalarType const& q_type, bool has_act_order,
               bool is_k_full, bool has_zp, int num_groups, int group_size,
               int dev, cudaStream_t stream, int thread_k, int thread_n,
               int sms, int max_par, bool use_fp32_reduce) {
  if (has_zp) {
    TORCH_CHECK(
        q_type == vllm::kU4 || q_type == vllm::kU8,
        "q_type must be u4 or u8 when has_zp = True. Got = ", q_type.str());
  } else {
    TORCH_CHECK(
        q_type == vllm::kU4B8 || q_type == vllm::kU8B128,
        "q_type must be uint4b8 or uint8b128 when has_zp = False. Got = ",
        q_type.str());
  }

  TORCH_CHECK(prob_m > 0 && prob_n > 0 && prob_k > 0, "Invalid MNK = [", prob_m,
              ", ", prob_n, ", ", prob_k, "]");

  // TODO: remove alias when we start supporting other 8bit types
  int num_bits = q_type.size_bits();
  int tot_m = prob_m;
  int tot_m_blocks = div_ceil(tot_m, 16);
  int pad = 16 * tot_m_blocks - tot_m;

  if (sms == -1) {
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  }

  int max_shared_mem = 0;
  cudaDeviceGetAttribute(&max_shared_mem,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  TORCH_CHECK(max_shared_mem > 0);

  // Set thread config
  exec_config_t exec_cfg;
  if (thread_k != -1 && thread_n != -1) {
    // User-defined config
    exec_cfg =
        exec_config_t{4, thread_config_t{thread_k, thread_n, default_threads}};
  } else {
    // Auto config
    exec_cfg =
        determine_thread_config(prob_m, prob_n, prob_k, num_bits, group_size,
                                has_act_order, is_k_full, max_shared_mem);
  }

  TORCH_CHECK(exec_cfg.max_m_blocks > 0 &&
                  is_valid_config(exec_cfg.tb_cfg, exec_cfg.max_m_blocks,
                                  prob_m, prob_n, prob_k, num_bits, group_size,
                                  has_act_order, is_k_full, max_shared_mem),
              "Invalid thread config: max_m_blocks = ", exec_cfg.max_m_blocks,
              ", thread_k = ", exec_cfg.tb_cfg.thread_k,
              ", thread_n = ", exec_cfg.tb_cfg.thread_n,
              ", num_threads = ", exec_cfg.tb_cfg.num_threads, " for MKN = [",
              prob_m, ", ", prob_k, ", ", prob_n, "] and num_bits = ", num_bits,
              ", group_size = ", group_size,
              ", has_act_order = ", has_act_order, ", is_k_full = ", is_k_full,
              ", max_shared_mem = ", max_shared_mem);

  int num_threads = exec_cfg.tb_cfg.num_threads;
  thread_k = exec_cfg.tb_cfg.thread_k;
  thread_n = exec_cfg.tb_cfg.thread_n;

  int thread_k_blocks = thread_k / 16;
  int thread_n_blocks = thread_n / 16;

  int blocks = sms;

  TORCH_CHECK(prob_n % thread_n == 0, "prob_n = ", prob_n,
              " is not divisible by thread_n = ", thread_n);
  TORCH_CHECK(prob_k % thread_k == 0, "prob_k = ", prob_k,
              " is not divisible by thread_k = ", thread_k);

  int group_blocks = 0;
  if (has_act_order) {
    if (is_k_full) {
      TORCH_CHECK(group_size != -1);
      group_blocks = group_size / 16;
      TORCH_CHECK(prob_k % group_blocks == 0, "prob_k = ", prob_k,
                  " is not divisible by group_blocks = ", group_blocks);
    } else {
      TORCH_CHECK(group_size == 0);
      group_blocks = 0;
    }

  } else {
    if (group_size == -1) {
      group_blocks = -1;
    } else {
      group_blocks = group_size / 16;
      TORCH_CHECK(prob_k % group_blocks == 0, "prob_k = ", prob_k,
                  " is not divisible by group_blocks = ", group_blocks);
    }
  }

  const int4* A_ptr = (const int4*)A;
  const int4* B_ptr = (const int4*)B;
  int4* C_ptr = (int4*)C;
  int4* C_tmp_ptr = (int4*)C_tmp;
  const int4* s_ptr = (const int4*)s;
  const int4* zp_ptr = (const int4*)zp;
  const int* g_idx_ptr = (const int*)g_idx;
  const int* perm_ptr = (const int*)perm;
  int4* a_tmp_ptr = (int4*)a_tmp;

  int* locks = (int*)workspace;

  if (has_act_order) {
    // Permute A columns
    int block_rows = div_ceil(prob_m, blocks);
    permute_cols_kernel<<<blocks, default_threads, 0, stream>>>(
        A_ptr, perm_ptr, a_tmp_ptr, prob_m, prob_k, block_rows);
    A_ptr = a_tmp_ptr;
  }

  // If we have a full K, then we can run the non-act-order version of Marlin
  // (since the weight rows are reordered by increasing group ids, and by having
  // a full K, we have full original groups)
  if (is_k_full) {
    has_act_order = false;
  }

  // Main loop
  for (int i = 0; i < tot_m_blocks; i += exec_cfg.max_m_blocks) {
    int thread_m_blocks = tot_m_blocks - i;
    prob_m = tot_m - 16 * i;
    int par = 1;
    if (thread_m_blocks > exec_cfg.max_m_blocks) {
      // Note that parallel > 1 currently only works for inputs without any
      // padding
      par = (16 * thread_m_blocks - pad) / (16 * exec_cfg.max_m_blocks);
      if (par > max_par) par = max_par;
      prob_m = (16 * exec_cfg.max_m_blocks) * par;
      i += exec_cfg.max_m_blocks * (par - 1);
      thread_m_blocks = exec_cfg.max_m_blocks;
    }

    if (false) {
    }
    GPTQ_CALL_IF(4, 16, 4, 256)
    GPTQ_CALL_IF(4, 8, 8, 256)
    GPTQ_CALL_IF(4, 8, 4, 128)
    GPTQ_CALL_IF(4, 4, 8, 128)
    GPTQ_CALL_IF(8, 16, 4, 256)
    GPTQ_CALL_IF(8, 8, 8, 256)
    GPTQ_CALL_IF(8, 8, 4, 128)
    GPTQ_CALL_IF(8, 4, 8, 128)

    AWQ_CALL_IF(4, 16, 4, 256)
    AWQ_CALL_IF(4, 8, 8, 256)
    AWQ_CALL_IF(4, 8, 4, 128)
    AWQ_CALL_IF(4, 4, 8, 128)
    AWQ_CALL_IF(8, 16, 4, 256)
    AWQ_CALL_IF(8, 8, 8, 256)
    AWQ_CALL_IF(8, 8, 4, 128)
    AWQ_CALL_IF(8, 4, 8, 128)
    else {
      TORCH_CHECK(false, "Unsupported shapes: MNK = [", prob_m, ", ", prob_n,
                  ", ", prob_k, "]", ", has_act_order = ", has_act_order,
                  ", num_groups = ", num_groups, ", group_size = ", group_size,
                  ", thread_m_blocks = ", thread_m_blocks,
                  ", thread_n_blocks = ", thread_n_blocks,
                  ", thread_k_blocks = ", thread_k_blocks,
                  ", num_bits = ", num_bits);
    }

    A_ptr += 16 * thread_m_blocks * (prob_k / 8) * par;
    C_ptr += 16 * thread_m_blocks * (prob_n / 8) * par;
  }
}

}  // namespace marlin

torch::Tensor gptq_marlin_gemm(torch::Tensor& a, torch::Tensor& b_q_weight,
                               torch::Tensor& b_scales, torch::Tensor& b_zeros,
                               torch::Tensor& g_idx, torch::Tensor& perm,
                               torch::Tensor& workspace,
                               vllm::ScalarTypeTorchPtr const& b_q_type,
                               int64_t size_m, int64_t size_n, int64_t size_k,
                               bool is_k_full, bool has_zp,
                               bool use_fp32_reduce) {
  if (has_zp) {
    TORCH_CHECK(*b_q_type == vllm::kU4 || *b_q_type == vllm::kU8,
                "b_q_type must be u4 or u8 when has_zp = True. Got = ",
                b_q_type->str());
  } else {
    TORCH_CHECK(
        *b_q_type == vllm::kU4B8 || *b_q_type == vllm::kU8B128,
        "b_q_type must be uint4b8 or uint8b128 when has_zp = False. Got = ",
        b_q_type->str());
  }

  int pack_factor = 32 / b_q_type->size_bits();

  // Verify A
  TORCH_CHECK(a.size(0) == size_m, "Shape mismatch: a.size(0) = ", a.size(0),
              ", size_m = ", size_m);
  TORCH_CHECK(a.size(1) == size_k, "Shape mismatch: a.size(1) = ", a.size(1),
              ", size_k = ", size_k);

  // Verify B
  TORCH_CHECK(size_k % marlin::tile_size == 0, "size_k = ", size_k,
              " is not divisible by tile_size = ", marlin::tile_size);
  TORCH_CHECK((size_k / marlin::tile_size) == b_q_weight.size(0),
              "Shape mismatch: b_q_weight.size(0) = ", b_q_weight.size(0),
              ", size_k = ", size_k, ", tile_size = ", marlin::tile_size);
  TORCH_CHECK(b_q_weight.size(1) % marlin::tile_size == 0,
              "b_q_weight.size(1) = ", b_q_weight.size(1),
              " is not divisible by tile_size = ", marlin::tile_size);
  int actual_size_n = (b_q_weight.size(1) / marlin::tile_size) * pack_factor;
  TORCH_CHECK(size_n == actual_size_n, "size_n = ", size_n,
              ", actual_size_n = ", actual_size_n);

  // Verify device and strides
  TORCH_CHECK(a.device().is_cuda(), "A is not on GPU");
  TORCH_CHECK(a.is_contiguous(), "A is not contiguous");

  TORCH_CHECK(b_q_weight.device().is_cuda(), "b_q_weight is not on GPU");
  TORCH_CHECK(b_q_weight.is_contiguous(), "b_q_weight is not contiguous");

  TORCH_CHECK(b_scales.device().is_cuda(), "b_scales is not on GPU");
  TORCH_CHECK(b_scales.is_contiguous(), "b_scales is not contiguous");

  TORCH_CHECK(b_zeros.device().is_cuda(), "b_zeros is not on GPU");
  TORCH_CHECK(b_zeros.is_contiguous(), "b_zeros is not contiguous");

  TORCH_CHECK(g_idx.device().is_cuda(), "g_idx is not on GPU");
  TORCH_CHECK(g_idx.is_contiguous(), "g_idx is not contiguous");

  TORCH_CHECK(perm.device().is_cuda(), "perm is not on GPU");
  TORCH_CHECK(perm.is_contiguous(), "perm is not contiguous");

  // Alloc buffers
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto options = torch::TensorOptions().dtype(a.dtype()).device(a.device());
  torch::Tensor c = torch::empty({size_m, size_n}, options);
  torch::Tensor a_tmp = torch::empty({size_m, size_k}, options);

  // Alloc C tmp buffer that is going to be used for the global reduce
  int reduce_max_m = marlin::determine_reduce_max_m(size_m, marlin::max_par);
  int reduce_n = size_n;
  auto options_fp32 =
      torch::TensorOptions().dtype(at::kFloat).device(a.device());
  if (!use_fp32_reduce) {
    reduce_max_m = 0;
    reduce_n = 0;
  }
  torch::Tensor c_tmp = torch::empty({reduce_max_m, reduce_n}, options_fp32);

  // thread_k: `k` size of a thread_tile in `weights` (can usually be left as
  // auto -1)
  int thread_k = -1;
  // thread_n: `n` size of a thread_tile in `weights` (can usually be left as
  // auto -1)
  int thread_n = -1;
  // sms: number of SMs to use for the kernel (can usually be left as auto -1)
  int sms = -1;

  // Verify g_idx and perm
  TORCH_CHECK((g_idx.size(0) == 0 && perm.size(0) == 0) ||
                  (g_idx.size(0) == size_k && perm.size(0) == size_k),
              "Unexpected g_idx.size(0) = ", g_idx.size(0),
              " and perm.size(0) = ", perm.size(0),
              ", where size_k = ", size_k);

  // Detect groupsize and act_order
  int num_groups = -1;
  int group_size = -1;
  bool has_act_order = g_idx.size(0) != 0;

  int rank = b_scales.sizes().size();
  TORCH_CHECK(rank == 2, "b_scales rank = ", rank, " is not 2");
  TORCH_CHECK(b_scales.size(1) == size_n, "b_scales dim 1 = ", b_scales.size(1),
              " is not size_n = ", size_n);
  num_groups = b_scales.size(0);

  if (has_act_order) {
    if (is_k_full) {
      TORCH_CHECK(num_groups > 1, "For act_order, num_groups must be > 1");
      TORCH_CHECK(size_k % num_groups == 0, "size_k = ", size_k,
                  ", is not divisible by num_groups = ", num_groups);
      group_size = size_k / num_groups;
    } else {
      group_size = 0;
    }

  } else {
    if (num_groups > 1) {
      TORCH_CHECK(
          size_k % num_groups == 0, "size_k = ", size_k,
          ", is not divisible by b_scales.size(0) = ", b_scales.size(0));
      group_size = size_k / num_groups;
    } else {
      group_size = -1;
    }
  }

  // Verify b_zeros
  if (has_zp) {
    int rank = b_zeros.sizes().size();
    TORCH_CHECK(rank == 2, "b_zeros rank = ", rank, " is not 2");
    TORCH_CHECK(b_zeros.size(0) == num_groups,
                "b_zeros dim 0 = ", b_zeros.size(0),
                " is not num_groups = ", num_groups);
    TORCH_CHECK(b_zeros.size(1) == size_n / pack_factor,
                "b_zeros dim 1 = ", b_scales.size(1),
                " is not size_n / pack_factor = ", size_n / pack_factor);
  }

  // Verify workspace size
  TORCH_CHECK(size_n % marlin::min_thread_n == 0, "size_n = ", size_n,
              ", is not divisible by min_thread_n = ", marlin::min_thread_n);
  int min_workspace_size = (size_n / marlin::min_thread_n) * marlin::max_par;
  TORCH_CHECK(workspace.numel() >= min_workspace_size,
              "workspace.numel = ", workspace.numel(),
              " is below min_workspace_size = ", min_workspace_size);

  int dev = a.get_device();
  if (a.scalar_type() == at::ScalarType::Half) {
    marlin::marlin_mm<half>(
        a.data_ptr<at::Half>(), b_q_weight.data_ptr(), c.data_ptr<at::Half>(),
        c_tmp.data_ptr<float>(), b_scales.data_ptr<at::Half>(),
        b_zeros.data_ptr(), g_idx.data_ptr(), perm.data_ptr(),
        a_tmp.data_ptr<at::Half>(), size_m, size_n, size_k,
        workspace.data_ptr(), *b_q_type, has_act_order, is_k_full, has_zp,
        num_groups, group_size, dev, at::cuda::getCurrentCUDAStream(dev),
        thread_k, thread_n, sms, marlin::max_par, use_fp32_reduce);
  } else if (a.scalar_type() == at::ScalarType::BFloat16) {
    marlin::marlin_mm<nv_bfloat16>(
        a.data_ptr<at::BFloat16>(), b_q_weight.data_ptr(),
        c.data_ptr<at::BFloat16>(), c_tmp.data_ptr<float>(),
        b_scales.data_ptr<at::BFloat16>(), b_zeros.data_ptr(), g_idx.data_ptr(),
        perm.data_ptr(), a_tmp.data_ptr<at::BFloat16>(), size_m, size_n, size_k,
        workspace.data_ptr(), *b_q_type, has_act_order, is_k_full, has_zp,
        num_groups, group_size, dev, at::cuda::getCurrentCUDAStream(dev),
        thread_k, thread_n, sms, marlin::max_par, use_fp32_reduce);
  } else {
    TORCH_CHECK(false, "gpt_marlin_gemm only supports bfloat16 and float16");
  }

  return c;
}

#endif
