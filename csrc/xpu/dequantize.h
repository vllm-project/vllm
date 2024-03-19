#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>
#include "utils.h"
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

namespace vllm {
namespace awq {

sycl::uint4 dequantize_s4_to_fp16x2(uint32_t const& source) {
  sycl::uint4 result;

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
  h[0] = (i4s & BOTTOM_MASK) | I4s_TO_F16s_MAGIC_NUM;
  h[1] = (i4s & TOP_MASK) | I4s_TO_F16s_MAGIC_NUM;
  h[2] = (top_i4s & BOTTOM_MASK) | I4s_TO_F16s_MAGIC_NUM;
  h[3] = (top_i4s & TOP_MASK) | I4s_TO_F16s_MAGIC_NUM;

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
  *(sycl::half2*)(&h[0]) = sycl_half_sub2(
      *(sycl::half2*)(&h[0]), *(sycl::half2*)(&FP16_TOP_MAGIC_NUM));
  *(sycl::half2*)(&h[1]) = sycl_half_fma2(
      *(sycl::half2*)(&h[1]),
      *(sycl::half2*)(&ONE_SIXTEENTH),
      *(sycl::half2*)(&NEG_64));
  *(sycl::half2*)(&h[2]) = sycl_half_sub2(
      *(sycl::half2*)(&h[2]), *(sycl::half2*)(&FP16_TOP_MAGIC_NUM));
  *(sycl::half2*)(&h[3]) = sycl_half_fma2(
      *(sycl::half2*)(&h[3]),
      *(sycl::half2*)(&ONE_SIXTEENTH),
      *(sycl::half2*)(&NEG_64));

  return result;
}

} // namespace awq
} // namespace vllm