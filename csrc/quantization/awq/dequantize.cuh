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
  assert(false);
#else
  uint4 result;

  uint32_t* h = reinterpret_cast<uint32_t*>(&result);
  uint32_t const i4s = reinterpret_cast<uint32_t const&>(source);

  // First, we extract the i4s and construct an intermediate bf16 number.
  static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
  static constexpr uint32_t BOTTOM_MASK = 0x000f000f;
  static constexpr uint32_t TOP_MASK = 0x00f000f0;
  
  // For BF16, we'll use a different magic number that accounts for the different exponent bias
  static constexpr uint32_t I4s_TO_BF16s_MAGIC_NUM = 0x5F005F00; // {0x5F00, 0x5F00} in bfloat16

  // Shift right by 8 to now consider elt_45 and elt_67
  const uint32_t top_i4s = i4s >> 8;
  
  // Extract elt_01 - (i4s & 0x000f000f) | 0x5F005F00
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[0])
               : "r"(i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_BF16s_MAGIC_NUM),
                 "n"(immLut));
  // Extract elt_23 (i4s & 0x00f000f0) | 0x5F005F00
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[1])
               : "r"(i4s), "n"(TOP_MASK), "n"(I4s_TO_BF16s_MAGIC_NUM),
                 "n"(immLut));
  // Extract elt_45 (top_i4s & 0x000f000f) | 0x5F005F00
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[2])
               : "r"(top_i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_BF16s_MAGIC_NUM),
                 "n"(immLut));
  // Extract elt_67 (top_i4s & 0x00f000f0) | 0x5F005F00
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[3])
               : "r"(top_i4s), "n"(TOP_MASK), "n"(I4s_TO_BF16s_MAGIC_NUM),
                 "n"(immLut));

  // Constants for bfloat16 conversion
  // For INT4 to BF16, we need to:
  // 1. Subtract the bias (which is different from FP16)
  // 2. Handle the sign bit properly (INT4 is signed)
  
  // This is the bfloat16 {0x5F00, 0x5F00} which represents {1.0, 1.0} * 2^0
  static constexpr uint32_t BF16_BIAS = 0x5F005F00;
  // This is the bfloat16 {0x3C00, 0x3C00} which represents {1/16, 1/16}
  static constexpr uint32_t ONE_SIXTEENTH_BF16 = 0x3C003C00;
  // This is the bfloat16 {-64, -64} in bfloat16 format
  static constexpr uint32_t NEG_64_BF16 = 0xC200C200;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  // SM90+ implementation using assembly instructions
  // Similar to FP16 implementation, we use different operations for even/odd indices
  // For h[0] and h[2], we use subtraction
  asm volatile("sub.bf16x2 %0, %1, %2;\n"
               : "=r"(h[0])
               : "r"(h[0]), "r"(BF16_BIAS));
  asm volatile("sub.bf16x2 %0, %1, %2;\n"
               : "=r"(h[2])
               : "r"(h[2]), "r"(BF16_BIAS));
               
  // For h[1] and h[3], we use fused multiply-add like in the FP16 version
  asm volatile("fma.rn.bf16x2 %0, %1, %2, %3;\n"
               : "=r"(h[1])
               : "r"(h[1]), "r"(ONE_SIXTEENTH_BF16), "r"(NEG_64_BF16));
  asm volatile("fma.rn.bf16x2 %0, %1, %2, %3;\n"
               : "=r"(h[3])
               : "r"(h[3]), "r"(ONE_SIXTEENTH_BF16), "r"(NEG_64_BF16));
#else
  // SM80-SM90 implementation using __nv_bfloat162 operations
  // Convert from BF16_BIAS representation to actual values with scaling
  
  // Process each pair (two bfloat16 values packed in a uint32_t)
  // Following the same pattern as in the FP16 version
  
  // For h[0] and h[2], we use subtraction (like sub.bf16x2)
  // This processes elements 0,1,4,5 (direct conversion by subtracting bias)
  __nv_bfloat162 bf16_pair_0 = *reinterpret_cast<__nv_bfloat162*>(&h[0]);
  __nv_bfloat162 bf16_pair_2 = *reinterpret_cast<__nv_bfloat162*>(&h[2]);
  __nv_bfloat162 bias_pair = *reinterpret_cast<const __nv_bfloat162*>(&BF16_BIAS);
  
  // Subtract the bias to get the actual INT4 values in bf16 format
  bf16_pair_0 = __hsub2(bf16_pair_0, bias_pair);
  bf16_pair_2 = __hsub2(bf16_pair_2, bias_pair);
  
  // Store results back
  h[0] = *reinterpret_cast<uint32_t*>(&bf16_pair_0);
  h[2] = *reinterpret_cast<uint32_t*>(&bf16_pair_2);
  
  // For h[1] and h[3], we use fused multiply-add (like fma.rn.bf16x2)
  // This processes elements 2,3,6,7 (conversion via multiply-add operation)
  __nv_bfloat162 bf16_pair_1 = *reinterpret_cast<__nv_bfloat162*>(&h[1]);
  __nv_bfloat162 bf16_pair_3 = *reinterpret_cast<__nv_bfloat162*>(&h[3]);
  __nv_bfloat162 scale_pair = *reinterpret_cast<const __nv_bfloat162*>(&ONE_SIXTEENTH_BF16);
  __nv_bfloat162 offset_pair = *reinterpret_cast<const __nv_bfloat162*>(&NEG_64_BF16);
  
  // Apply scale and offset: (x + bias) * (1/16) + (-64)
  // This effectively converts and scales the INT4 values appropriately
  bf16_pair_1 = __hfma2(bf16_pair_1, scale_pair, offset_pair);
  bf16_pair_3 = __hfma2(bf16_pair_3, scale_pair, offset_pair);
  
  // Store results back
  h[1] = *reinterpret_cast<uint32_t*>(&bf16_pair_1);
  h[3] = *reinterpret_cast<uint32_t*>(&bf16_pair_3);
#endif

  return result;
#endif
  __builtin_unreachable();
}
}  // namespace awq
}  // namespace vllm
