/*
Adapted from https://github.com/mit-han-lab/llm-awq
Modified from NVIDIA FasterTransformer:
https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h
@article{lin2023awq,
  title={AWQ: Activation-aware Weight Quantization for LLM Compression and
Acceleration}, author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang,
Shang and Dang, Xingyu and Han, Song}, journal={arXiv}, year={2023}
}
*/

#pragma once

#include <cuda_bf16.h>

namespace vllm {
namespace awq {

__device__ uint4 dequantize_s4_to_fp16x2(uint32_t const& source) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
  assert(false);
#else
  uint4 result;

  uint32_t* h = reinterpret_cast<uint32_t*>(&result);
  uint32_t const i4s = reinterpret_cast<uint32_t const&>(source);

  // First, we extract the i4s and construct an intermediate fp16 number.
  static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
  static constexpr uint32_t BOTTOM_MASK = 0x000f000f;
  static constexpr uint32_t TOP_MASK = 0x00f000f0;
  static constexpr uint32_t I4s_TO_F16s_MAGIC_NUM = 0x64006400;

  // Note that the entire sequence only requires 1 shift instruction. This is
  // thanks to the register packing format and the fact that we force our
  // integers to be unsigned, and account for this in the fp16 subtractions. In
  // addition, I exploit the fact that sub and fma have the same throughput in
  // order to convert elt_23 and elt_67 to fp16 without having to shift them to
  // the bottom bits before hand.

  // Shift right by 8 to now consider elt_45 and elt_67. Issue first to hide RAW
  // dependency if we issue immediately before required.
  const uint32_t top_i4s = i4s >> 8;
  // Extract elt_01 - (i4s & 0x000f000f) | 0x64006400
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[0])
               : "r"(i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM),
                 "n"(immLut));
  // Extract elt_23 (i4s & 0x00f000f0) | 0x64006400
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[1])
               : "r"(i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM),
                 "n"(immLut));
  // Extract elt_45 (top_i4s & 0x000f000f) | 0x64006400
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[2])
               : "r"(top_i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM),
                 "n"(immLut));
  // Extract elt_67 (top_i4s & 0x00f000f0) | 0x64006400
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[3])
               : "r"(top_i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM),
                 "n"(immLut));

  // I use inline PTX below because I am not sure if the compiler will emit
  // float2half instructions if I use the half2 ctor. In this case, I chose
  // performance reliability over code readability.

  // This is the half2 {1032, 1032} represented as an integer.
  // static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64086408;
  // Haotian: subtract {1024, 1024} instead, we do not need to map to [-8, 7]
  static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64006400;
  // This is the half2 {1 / 16, 1 / 16} represented as an integer.
  static constexpr uint32_t ONE_SIXTEENTH = 0x2c002c00;
  // This is the half2 {-72, -72} represented as an integer.
  // static constexpr uint32_t NEG_72 = 0xd480d480;
  // Haotian: Let's use {-64, -64}.
  static constexpr uint32_t NEG_64 = 0xd400d400;

  // Finally, we construct the output numbers.
  // Convert elt_01
  asm volatile("sub.f16x2 %0, %1, %2;\n"
               : "=r"(h[0])
               : "r"(h[0]), "r"(FP16_TOP_MAGIC_NUM));
  // Convert elt_23
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
               : "=r"(h[1])
               : "r"(h[1]), "r"(ONE_SIXTEENTH), "r"(NEG_64));
  // Convert elt_45
  asm volatile("sub.f16x2 %0, %1, %2;\n"
               : "=r"(h[2])
               : "r"(h[2]), "r"(FP16_TOP_MAGIC_NUM));
  // Convert elt_67
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
               : "=r"(h[3])
               : "r"(h[3]), "r"(ONE_SIXTEENTH), "r"(NEG_64));

  return result;
#endif
  __builtin_unreachable();  // Suppress missing return statement warning
}

__device__ uint4 dequantize_s4_to_bf16x2(uint32_t const& source) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  static_assert(false, "dequantize_s4_to_bf16x2 requires CUDA ARCH >= 800 (Ampere)");
#else
  uint4 result;

  uint32_t* h = reinterpret_cast<uint32_t*>(&result);
  uint32_t const i4s = reinterpret_cast<uint32_t const&>(source);

  // First, we extract the i4s and construct an intermediate bf16 number.
  static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
  static constexpr uint32_t BOTTOM_MASK = 0x000f000f;
  static constexpr uint32_t TOP_MASK = 0x00f000f0;
  
  // Magic number for the int->float conversion trick using addition.
  // The value is 128.0f in bfloat16, which is 0x4300.
  static constexpr uint32_t I4s_TO_BF16s_MAGIC_NUM = 0x43004300;

  // Shift right by 8 to now consider elt_45 and elt_67
  const uint32_t top_i4s = i4s >> 8;
  
  // Extract elt_01
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[0])
               : "r"(i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_BF16s_MAGIC_NUM), "n"(immLut));
  // Extract elt_23
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[1])
               : "r"(i4s), "n"(TOP_MASK), "n"(I4s_TO_BF16s_MAGIC_NUM), "n"(immLut));
  // Extract elt_45
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[2])
               : "r"(top_i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_BF16s_MAGIC_NUM), "n"(immLut));
  // Extract elt_67
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[3])
               : "r"(top_i4s), "n"(TOP_MASK), "n"(I4s_TO_BF16s_MAGIC_NUM), "n"(immLut));

  // Constants for bfloat16 conversion
  // This is {128.0, 128.0} as bfloat16x2. We subtract this to finish the conversion.
  static constexpr uint32_t BF16_SUB_MAGIC_NUM = 0x43004300;
  // This is {1/16, 1/16} as bfloat16x2. Value is 0x3D80.
  static constexpr uint32_t ONE_SIXTEENTH_BF16 = 0x3D803D80;
  // This is {-8.0, -8.0} as bfloat16x2. Value is 0xC100.
  // It's used to cancel out the (128.0f * 1/16) term in the FMA operation.
  static constexpr uint32_t NEG_8_BF16 = 0xC100C100;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  // SM90+ (Hopper) implementation using PTX assembly
  // Convert elt_01, elt_45: sub( (i4 + 128.0), 128.0 ) -> i4
  asm volatile("sub.bf16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(BF16_SUB_MAGIC_NUM));
  asm volatile("sub.bf16x2 %0, %1, %2;\n" : "=r"(h[2]) : "r"(h[2]), "r"(BF16_SUB_MAGIC_NUM));
          
  // Convert elt_23, elt_67: fma( (i4*16 + 128.0), 1/16, -8.0 ) -> i4
  asm volatile("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(h[1]) : "r"(h[1]), "r"(ONE_SIXTEENTH_BF16), "r"(NEG_8_BF16));
  asm volatile("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(h[3]) : "r"(h[3]), "r"(ONE_SIXTEENTH_BF16), "r"(NEG_8_BF16));
#else
  // SM80-SM89 (Ampere) implementation using CUDA intrinsics
  __nv_bfloat162 h0 = *reinterpret_cast<__nv_bfloat162*>(&h[0]);
  __nv_bfloat162 h1 = *reinterpret_cast<__nv_bfloat162*>(&h[1]);
  __nv_bfloat162 h2 = *reinterpret_cast<__nv_bfloat162*>(&h[2]);
  __nv_bfloat162 h3 = *reinterpret_cast<__nv_bfloat162*>(&h[3]);

  const __nv_bfloat162 sub_magic = *reinterpret_cast<const __nv_bfloat162*>(&BF16_SUB_MAGIC_NUM);
  const __nv_bfloat162 scale_1_16 = *reinterpret_cast<const __nv_bfloat162*>(&ONE_SIXTEENTH_BF16);
  const __nv_bfloat162 offset_neg_8 = *reinterpret_cast<const __nv_bfloat162*>(&NEG_8_BF16);

  // Convert elt_01, elt_45
  h0 = __hsub2(h0, sub_magic);
  h2 = __hsub2(h2, sub_magic);

  // Convert elt_23, elt_67
  h1 = __hfma2(h1, scale_1_16, offset_neg_8);
  h3 = __hfma2(h3, scale_1_16, offset_neg_8);

  // Store results back
  h[0] = *reinterpret_cast<uint32_t*>(&h0);
  h[1] = *reinterpret_cast<uint32_t*>(&h1);
  h[2] = *reinterpret_cast<uint32_t*>(&h2);
  h[3] = *reinterpret_cast<uint32_t*>(&h3);
#endif

  return result;
#endif
  __builtin_unreachable();
}
}  // namespace awq
}  // namespace vllm