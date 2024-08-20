#pragma once

#include "cutlass/numeric_conversion.h"
#include "cutlass_extensions/vllm_custom_types.cuh"
#include "cutlass_extensions/cute_utils.cuh"

// this file extends:
//   https://github.com/NVIDIA/cutlass/blob/cutlass-3.5.0/include/cutlass/numeric_conversion.h
// with vllm specific type conversions, namely: vllm_uint4b8_t, vllm_uint8b128_t
// as well as adds interleaved numeric array converters for specific types.
// (interleaved numeric array converters can be more efficient for subbyte
// types)

namespace cutlass {

// InterleavedNumericArrayConverter is like NumericArrayConverter but also
// deinterleaves converted elements based on IlvBlkLayout, interleaving can
// make subbyte converts more efficient by allowing for efficient extraction
// of subbyte elements from a 32bit register.
template <typename IlvBlkLayout, typename T, typename S, int N,
          FloatRoundStyle Round = FloatRoundStyle::round_to_nearest,
          class Enable = void>
struct InterleavedNumericArrayConverter {
  using Converter = NumericArrayConverter<T, S, N, Round>;

  using result_type = typename Converter::result_type;
  using source_type = typename Converter::source_type;

  CUTLASS_DEVICE
  static result_type convert(source_type const& source) {
    CUTE_INVALID_CONTROL_PATH(
        "InterleavedNumericArrayConverter not implemented\n");
    return {};
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const& s) const { return convert(s); }
};

template <typename IlvBlkLayout, typename T, typename S, int N,
          FloatRoundStyle Round>
struct InterleavedNumericArrayConverter<
    IlvBlkLayout, T, S, N, Round,
    std::enable_if_t<is_identity_layout<IlvBlkLayout>()>> {
  using Converter = NumericArrayConverter<T, S, N, Round>;

  using result_type = typename Converter::result_type;
  using source_type = typename Converter::source_type;

  CUTLASS_DEVICE
  static result_type convert(source_type const& source) {
    return Converter::convert(source);
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const& s) const { return convert(s); }
};

// TODO (LucasWilkinson): Implement
// for Array<cutlass::float8_e4m3fn, N> <= Array<vllm_uint4b8_t, N>

// ....

template <typename RegConvert32bit, typename T, typename S, int N>
struct ArrayConverterPacked32Bit {
  using result_type = Array<T, N>;
  using source_type = Array<S, N>;

  using result_packed_8_t = Array<T, 8>;
  using result_packed_4_t = Array<T, 4>;
  using result_packed_2_t = Array<T, 2>;
  using src_packed_8_t = Array<S, 8>;
  using src_packed_4_t = Array<S, 4>;
  using src_packed_2_t = Array<S, 2>;

  static_assert(N % 2 == 0, "N must be a multiple of 2");
  static_assert(cutlass::sizeof_bits_v<S> >= 4);  // TODO: add 16 packed sources
  static_assert(32 % cutlass::sizeof_bits_v<S> == 0);
  static constexpr auto src_elems_per_32bit_reg =
      32 / cutlass::sizeof_bits_v<S>;

  // Maybe not Valid. ScalarConverter will not actually work unless
  // NumericConverter<T, S, Round> is implemented. However it won't be used
  // anyways since we assert N % 2 == 0, just here for compliance with
  // VectorizedConverter.
  using ScalarConverter = NumericConverter<T, S>;

  template <typename PackedSrc>
  CUTLASS_DEVICE static uint32_t to_reg(PackedSrc const& source) {
    if constexpr (sizeof(PackedSrc) == 1) {
      return static_cast<uint32_t>(reinterpret_cast<const uint8_t&>(source));
    } else if constexpr (sizeof(PackedSrc) == 2) {
      return static_cast<uint32_t>(reinterpret_cast<const uint16_t&>(source));
    } else {
      static_assert(sizeof(PackedSrc) == 4);
      return reinterpret_cast<const uint32_t&>(source);
    }
  }

  // The core converter uses bit tricks to construct a known FP16 number, then
  // does a subtraction in FP16 for the final result.
  template <typename PackedResultType, typename PackedSrcType>
  CUTLASS_DEVICE static PackedResultType packed_convert(
      PackedSrcType const& source) {
    static_assert(PackedSrcType::kElements == PackedResultType::kElements);
    static_assert(PackedResultType::kElements == 2 ||
                      PackedResultType::kElements == 4 ||
                      PackedResultType::kElements == 8,
                  "Invalid PackedResultType must be 2, 4 or 8.");
    static_assert(std::is_same_v<typename PackedSrcType::Element, S>);
    static_assert(std::is_same_v<typename PackedResultType::Element, T>);

    return RegConvert32bit::template convert<PackedResultType>(to_reg(source));
  }

  friend class detail::VectorizedConverter;

 public:
  CUTLASS_DEVICE static result_type convert(source_type const& source) {
    result_type result;
    using ConverterType =
        ArrayConverterPacked32Bit<RegConvert32bit,
                                  typename result_type::Element,
                                  typename source_type::Element, N>;

    if constexpr (src_elems_per_32bit_reg >= 8) {
      detail::VectorizedConverter::convert<
          ConverterType, result_packed_8_t, src_packed_8_t, result_packed_4_t,
          src_packed_4_t, result_packed_2_t, src_packed_2_t>(result, source);
    } else if constexpr (src_elems_per_32bit_reg >= 4) {
      detail::VectorizedConverter::convert<ConverterType, result_packed_4_t,
                                           src_packed_4_t, result_packed_2_t,
                                           src_packed_2_t>(result, source);
    } else {
      detail::VectorizedConverter::convert<ConverterType, result_packed_2_t,
                                           src_packed_2_t>(result, source);
    }

    return result;
  }
};

// for Array<cutlass::half_t, N> <= Array<vllm_uint4b8_t, N>
template <FloatRoundStyle Round, int N>
struct NumericArrayConverter<cutlass::half_t, vllm_uint4b8_t, N, Round> {
  using result_type = Array<cutlass::half_t, N>;
  using source_type = Array<vllm_uint4b8_t, N>;

  struct RegConvert {
    template <typename PackedResultType>
    CUTLASS_DEVICE static PackedResultType convert(uint32_t src) {
      using RegArray =
          cutlass::AlignedArray<uint32_t, PackedResultType::kElements / 2,
                                sizeof(PackedResultType)>;
      RegArray r;

      // Below constructs the following temporary:
      // fp16s_01 = {0x00, i4_01, 0x00, i4_01}
      // fp16s_23 = {0x00, i4_23, 0x00, i4_23}
      // fp16s_45 = {0x00, i4_45, 0x00, i4_45}
      // fp16s_67 = {0x00, i4_67, 0x00, i4_67}
      // We use inline asm instead of __byte_perm intrinsic since we don't want
      // the documented (& 0x7) on the index. NVCC might be able to optimize it
      // out since the index is a constexpr, but we choose to be safe about it
      // here.
      uint32_t prmt_indices[4] = {0x4040, 0x4141, 0x4242, 0x4343};
      static_assert(RegArray::kElements <= 4,
                    "Too many inputs for F16 -> I4 vector converter");
      CUTLASS_PRAGMA_UNROLL
      for (int ii = 0; ii < RegArray::kElements; ++ii) {
        asm volatile(
            "{\n"
            "  prmt.b32 %0, %1, %2, %3;\n"
            "}\n"
            : "=r"(r[ii])
            : "r"(src), "n"(0), "r"(prmt_indices[ii]));
      }

      // Since the stored 4bit values are biased by 8 we get stored_val = (x+8)
      //  we are trying to construct x and a fp16 value
      // The below XOR does the following:
      //  1) Sets the exponent bits of the FP16 to the correct value for the
      //  FP16 magic_num. We will be constructing {1024+16*(x1+8), 1024+(x0+8)},
      //  where x1 in the high nibble and x0 is the low nibble then using hfma
      //  to subtract 1032 from that
      // The AND does the following:
      //  1) Clear the set bits for the int4 we will ignore.
      // We use lop3 so that we can use 1 instruction for AND and XOR.
      static constexpr uint32_t xor_mask = 0x64006400;
      static constexpr uint32_t and_mask = 0xFFF0FF0F;
      static constexpr uint32_t immLut = (0xf0 & 0xcc) ^ 0xaa;

      // For each operand, computes:
      // r[i] = (r[i] & and_mask) ^ xor_mask
      CUTLASS_PRAGMA_UNROLL
      for (int ii = 0; ii < RegArray::kElements; ++ii) {
        asm volatile(
            "{\n"
            "  lop3.b32 %0, %0, %1, %2, %3;\n"
            "}\n"
            : "+r"(r[ii])
            : "n"(and_mask), "n"(xor_mask), "n"(immLut));
      }

      // We will issue 2 hfmas that do the following:
      // {x1, x0} = {1024+16*(x1+8), 1024+(x0+8)} * {1/16, 1} - {72, 1032}
      //          = {x1 + 1152, x0 + 1032} * {1/16, 1} - {72, 1032}
      static constexpr uint32_t hfma_bias_rep = 0xD480E408;   // {72, 1032}
      static constexpr uint32_t hfma_scale_rep = 0x2C003C00;  // {1 / 16, 1}

      const half2& hfma_bias = reinterpret_cast<const half2&>(hfma_bias_rep);
      const half2& hfma_scale = reinterpret_cast<const half2&>(hfma_scale_rep);
      CUTLASS_PRAGMA_UNROLL
      for (int ii = 0; ii < RegArray::kElements; ++ii) {
        half2& fp16x2_val = reinterpret_cast<__half2&>(r[ii]);
        fp16x2_val = __hfma2(hfma_scale, fp16x2_val, hfma_bias);
      }

      return reinterpret_cast<PackedResultType&>(r);
    };
  };

 public:
  CUTLASS_DEVICE
  static result_type convert(source_type const& source) {
    return ArrayConverterPacked32Bit<RegConvert, typename result_type::Element,
                                     typename source_type::Element,
                                     N>::convert(source);
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const& s) const { return convert(s); }
};

// for Array<cutlass::half_t, N> <= Array<vllm_uint4b8_t, N>
//   for IlvdLayout: (2, 4):(4, 1)
template <FloatRoundStyle Round, int N>
struct InterleavedNumericArrayConverter<Layout<Shape<_2, _4>, Stride<_4, _1>>,
                                        cutlass::half_t, vllm_uint4b8_t, N,
                                        Round, void> {
  using IlvdLayout = Layout<Shape<_2, _4>, Stride<_4, _1>>;
  static_assert(N % size(IlvdLayout{}) == 0);

  using result_type = Array<cutlass::half_t, N>;
  using source_type = Array<vllm_uint4b8_t, N>;

  static FloatRoundStyle const round_style = Round;

 private:
  struct RegConvert {
    template <typename PackedResultType>
    CUTLASS_DEVICE static PackedResultType convert(uint32_t src) {
      using RegArray =
          cutlass::AlignedArray<uint32_t, PackedResultType::kElements / 2,
                                sizeof(PackedResultType)>;
      RegArray r;

      static_assert(PackedResultType::kElements <= size(IlvdLayout{}));
      static constexpr uint32_t xor_mask = 0x64006400;

      for (int ii = 0; ii < RegArray::kElements; ii += 2) {
        auto src_ = src >> (4 * (ii));
        r[ii + 0] = src_;
        r[ii + 1] = src_;

        static constexpr uint32_t and_xor_imm_lut = (0xf0 & 0xcc) ^ 0xaa;

        static constexpr uint32_t low_nib_mask = 0x000F000F;
        static constexpr uint32_t high_nib_mask = 0x00F000F0;

        asm volatile(
            "{\n"
            "  lop3.b32 %0, %0, %1, %2, %3;\n"
            "}\n"
            : "+r"(r[ii + 0])
            : "n"(low_nib_mask), "n"(xor_mask), "n"(and_xor_imm_lut));

        asm volatile(
            "{\n"
            "  lop3.b32 %0, %0, %1, %2, %3;\n"
            "}\n"
            : "+r"(r[ii + 1])
            : "n"(high_nib_mask), "n"(xor_mask), "n"(and_xor_imm_lut));

        // For low nibble:
        //  {x1, x0} = {1024+(x1+8), 1024+(x0+8)} * {1, 1} - {1032, 1032}
        // For high nibble:
        //  {x1, x0} = {1024+16*(x1+8), 1024+16*(x0+8)} * {1/16, 1/16}
        //             - {72, 72}
        static constexpr uint32_t low_nib_bias = 0x64086408;    // {1032, 1032}
        static constexpr uint32_t high_nib_scale = 0x2C002C00;  // {1/16, 1/16}
        static constexpr uint32_t high_nib_bias = 0xD480D480;   // {-72, -72}

        {
          half2& fp16x2_val = reinterpret_cast<__half2&>(r[ii + 0]);
          fp16x2_val =
              __hsub2(fp16x2_val, reinterpret_cast<const half2&>(low_nib_bias));
        }

        {
          half2& fp16x2_val = reinterpret_cast<__half2&>(r[ii + 1]);
          fp16x2_val = __hfma2(fp16x2_val,
                               reinterpret_cast<const half2&>(high_nib_scale),
                               reinterpret_cast<const half2&>(high_nib_bias));
        }
      }

      return reinterpret_cast<PackedResultType&>(r);
    };
  };

 public:
  CUTLASS_DEVICE
  static result_type convert(source_type const& source) {
    return ArrayConverterPacked32Bit<RegConvert, typename result_type::Element,
                                     typename source_type::Element,
                                     N>::convert(source);
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const& s) const { return convert(s); }
};

// for Array<cutlass::half_t, N> <= Array<uint4_t, N>
//   for IlvdLayout: (2, 4):(4, 1)
template <FloatRoundStyle Round, int N>
struct InterleavedNumericArrayConverter<Layout<Shape<_2, _4>, Stride<_4, _1>>,
                                        cutlass::half_t, uint4_t, N, Round,
                                        void> {
  using IlvdLayout = Layout<Shape<_2, _4>, Stride<_4, _1>>;
  static_assert(N % size(IlvdLayout{}) == 0);

  using result_type = Array<cutlass::half_t, N>;
  using source_type = Array<uint4_t, N>;

  static FloatRoundStyle const round_style = Round;

 private:
  struct RegConvert {
    template <typename PackedResultType>
    CUTLASS_DEVICE static PackedResultType convert(uint32_t src) {
      using RegArray =
          cutlass::AlignedArray<uint32_t, PackedResultType::kElements / 2,
                                sizeof(PackedResultType)>;
      RegArray r;

      static_assert(PackedResultType::kElements <= size(IlvdLayout{}));
      static constexpr uint32_t xor_mask = 0x64006400;

      for (int ii = 0; ii < RegArray::kElements; ii += 2) {
        auto src_ = src >> (4 * (ii));
        r[ii + 0] = src_;
        r[ii + 1] = src_;

        static constexpr uint32_t and_xor_imm_lut = (0xf0 & 0xcc) ^ 0xaa;

        static constexpr uint32_t low_nib_mask = 0x000F000F;
        static constexpr uint32_t high_nib_mask = 0x00F000F0;

        asm volatile(
            "{\n"
            "  lop3.b32 %0, %0, %1, %2, %3;\n"
            "}\n"
            : "+r"(r[ii + 0])
            : "n"(low_nib_mask), "n"(xor_mask), "n"(and_xor_imm_lut));

        asm volatile(
            "{\n"
            "  lop3.b32 %0, %0, %1, %2, %3;\n"
            "}\n"
            : "+r"(r[ii + 1])
            : "n"(high_nib_mask), "n"(xor_mask), "n"(and_xor_imm_lut));

        // For low nibble:
        //  {x1, x0} = {1024+x1, 1024+x0} - {1024, 1024}
        // For high nibble:
        //  {x1, x0} = {1024+16*x1, 1024+16*x0} * {1/16, 1/16} - {64, 64}
        static constexpr uint32_t low_nib_bias = 0x64006400;    // {1024, 1024}
        static constexpr uint32_t high_nib_scale = 0x2C002C00;  // {1/16, 1/16}
        static constexpr uint32_t high_nib_bias = 0xD400D400;   // {-64, -64}

        {
          half2& fp16x2_val = reinterpret_cast<__half2&>(r[ii + 0]);
          fp16x2_val =
              __hsub2(fp16x2_val, reinterpret_cast<const half2&>(low_nib_bias));
        }

        {
          half2& fp16x2_val = reinterpret_cast<__half2&>(r[ii + 1]);
          fp16x2_val = __hfma2(fp16x2_val,
                               reinterpret_cast<const half2&>(high_nib_scale),
                               reinterpret_cast<const half2&>(high_nib_bias));
        }
      }

      return reinterpret_cast<PackedResultType&>(r);
    };
  };

 public:
  CUTLASS_DEVICE
  static result_type convert(source_type const& source) {
    return ArrayConverterPacked32Bit<RegConvert, typename result_type::Element,
                                     typename source_type::Element,
                                     N>::convert(source);
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const& s) const { return convert(s); }
};

// for Array<cutlass::half_t, N> <= Array<vllm_uint8b128_t, N>
template <FloatRoundStyle Round, int N>
struct NumericArrayConverter<cutlass::half_t, vllm_uint8b128_t, N, Round> {
  using result_type = Array<cutlass::half_t, N>;
  using source_type = Array<vllm_uint8b128_t, N>;

  struct RegConvert {
    template <typename PackedResultType>
    CUTLASS_DEVICE static PackedResultType convert(uint32_t src) {
      // Hold output FP16s in reg. We need 1 reg for every 2 elements
      using RegArray =
          cutlass::AlignedArray<uint32_t, PackedResultType::kElements / 2,
                                sizeof(PackedResultType)>;
      RegArray r;

      uint32_t const prmt_indices[2] = {0x5150, 0x5352};
      static constexpr uint32_t start_byte_for_fp16 = 0x64646464;

      for (int ii = 0; ii < RegArray::kElements; ++ii) {
        asm volatile("prmt.b32 %0,%1,%2,%3;\n"
                     : "=r"(r[ii])
                     : "r"(src), "n"(start_byte_for_fp16),
                       "r"(prmt_indices[ii]));
      }

      // -128 is folded into bias subtraction, i.e. the 0x80 in the low bytes
      static constexpr uint32_t bias_rep = 0x64806480;
      const half2& bias = reinterpret_cast<const half2&>(bias_rep);
      CUTLASS_PRAGMA_UNROLL
      for (int ii = 0; ii < RegArray::kElements; ++ii) {
        half2& fp16x2_val = reinterpret_cast<__half2&>(r[ii]);
        fp16x2_val = __hsub2(fp16x2_val, bias);
      }

      return reinterpret_cast<PackedResultType&>(r);
    };
  };

 public:
  CUTLASS_DEVICE
  static result_type convert(source_type const& source) {
    return ArrayConverterPacked32Bit<RegConvert, typename result_type::Element,
                                     typename source_type::Element,
                                     N>::convert(source);
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const& s) const { return convert(s); }
};

// for Array<cutlass::float, N> <= Array<vllm_uint8b128_t, N>
template <FloatRoundStyle Round, int N>
struct NumericArrayConverter<float, vllm_uint8b128_t, N, Round> {
  using result_type = Array<float, N>;
  using source_type = Array<vllm_uint8b128_t, N>;
  static FloatRoundStyle const round_style = Round;

 private:
  struct RegConvert {
    template <typename PackedResultType>
    CUTLASS_DEVICE static PackedResultType convert(uint32_t src) {
      PackedResultType r;

      // __byte_perm simulates the add.u32 0x4B000000 to every u8 element of
      // u8x4 source and stores the result in r (without introducing extra
      // cvt.u32.u8 instruction)
      uint32_t const prmt_indices[4] = {0x7650, 0x7651, 0x7652, 0x7653};
      uint32_t* result_as_int = reinterpret_cast<uint32_t*>(&r);
      for (int ii = 0; ii < PackedResultType::kElements; ++ii) {
        result_as_int[ii] = __byte_perm(src, 0x4B000000, prmt_indices[ii]);
        // Subtract the magic number 0x4B000000 from tmp in floating-point
        // arithmetic to obtain final result
        r[ii] -= (8388608.f + 128.f);  // fold in -128 bias
      }

      return r;
    };
  };

 public:
  CUTLASS_DEVICE
  static result_type convert(source_type const& source) {
    return ArrayConverterPacked32Bit<RegConvert, typename result_type::Element,
                                     typename source_type::Element,
                                     N>::convert(source);
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const& s) const { return convert(s); }
};

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

// for Array<cutlass::bfloat16_t, N> <= Array<vllm_uint4b8_t, N>
template <FloatRoundStyle Round, int N>
struct NumericArrayConverter<cutlass::bfloat16_t, vllm_uint4b8_t, N, Round> {
  using result_type = Array<cutlass::bfloat16_t, N>;
  using source_type = Array<vllm_uint4b8_t, N>;

  static FloatRoundStyle const round_style = Round;

 private:
  struct RegConvert {
    template <typename PackedResultType>
    CUTLASS_DEVICE static PackedResultType convert(uint32_t src_reg) {
      // Hold output BF16s in reg. We need 1 reg for every 2 elements
      using RegArray =
          cutlass::AlignedArray<uint32_t, PackedResultType::kElements / 2,
                                sizeof(PackedResultType)>;
      RegArray r;
      uint32_t src_reg_shifted = src_reg >> 4;

      // Below constructs the following temporary:
      uint32_t const prmt_indices[4] = {0xF4F0, 0xF5F1, 0xF6F2, 0xF7F3};
      static_assert(RegArray::kElements <= 4,
                    "Too many inputs for uint4b8_t -> BF16 vector converter");
      CUTLASS_PRAGMA_UNROLL
      for (int ii = 0; ii < RegArray::kElements; ++ii) {
        asm volatile(
            "{\n"
            "  prmt.b32 %0, %1, %2, %3;\n"
            "}\n"
            : "=r"(r[ii])
            : "r"(src_reg), "r"(src_reg_shifted), "r"(prmt_indices[ii]));
      }

      // Since the stored 4bit values are biased by 8 we get stored_val = (x+8)
      //  we are trying to construct x and a BF16 value
      // The below XOR does the following:
      //  1) Sets the exponent bits of the BF16 to the correct value for the
      //  BF16 magic_num. We will be constructing {128 + (x1+8), 128 + (x0+8)}
      //  and subtracting 136 to get {x1, x0}
      static constexpr uint32_t xor_mask = 0x43004300;
      static constexpr uint32_t and_mask = 0x000F000F;
      static constexpr uint32_t immLut = (0xf0 & 0xcc) ^ 0xaa;

      // For each operand, computes:
      // r[i] = (r[i] & and_mask) ^ xor_mask
      CUTLASS_PRAGMA_UNROLL
      for (int ii = 0; ii < RegArray::kElements; ++ii) {
        asm volatile(
            "{\n"
            "  lop3.b32 %0, %0, %1, %2, %3;\n"
            "}\n"
            : "+r"(r[ii])
            : "n"(and_mask), "n"(xor_mask), "n"(immLut));
      }

      // We will issue 2 bfmas that do the following:
      // high BF16:
      // hi_bf16 - 136, lo_bf16 - 136

      // This is the BF16 {136, 136} represented as an integer.
      static constexpr uint32_t bias_rep = 0x43084308;
      const __nv_bfloat162& bias =
          reinterpret_cast<const __nv_bfloat162&>(bias_rep);

      CUTLASS_PRAGMA_UNROLL
      for (int ii = 0; ii < RegArray::kElements; ++ii) {
        __nv_bfloat162& bf16x2_val = reinterpret_cast<__nv_bfloat162&>(r[ii]);
        bf16x2_val = __hsub2(bf16x2_val, bias);
      }

      return reinterpret_cast<PackedResultType&>(r);
    }
  };

 public:
  CUTLASS_DEVICE
  static result_type convert(source_type const& source) {
    return ArrayConverterPacked32Bit<RegConvert, typename result_type::Element,
                                     typename source_type::Element,
                                     N>::convert(source);
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const& s) const { return convert(s); }
};

// for Array<cutlass::bfloat16_t, N> <= Array<vllm_uint4b8_t, N>
//   for IlvdLayout: (2, 4):(4, 1)
template <FloatRoundStyle Round, int N>
struct InterleavedNumericArrayConverter<Layout<Shape<_2, _4>, Stride<_4, _1>>,
                                        cutlass::bfloat16_t, vllm_uint4b8_t, N,
                                        Round, void> {
  using IlvdLayout = Layout<Shape<_2, _4>, Stride<_4, _1>>;
  static_assert(N % size(IlvdLayout{}) == 0);

  using result_type = Array<cutlass::bfloat16_t, N>;
  using source_type = Array<vllm_uint4b8_t, N>;

 private:
  struct RegConvert {
    template <typename PackedResultType>
    CUTLASS_DEVICE static PackedResultType convert(uint32_t src) {
      using RegArray =
          cutlass::AlignedArray<uint32_t, PackedResultType::kElements / 2,
                                sizeof(PackedResultType)>;
      RegArray r;

      static_assert(PackedResultType::kElements <= size(IlvdLayout{}));
      static constexpr uint32_t or_mask = 0x43004300;

      // Unlike float16 where the mantissa is large enough to contain 2
      // nibbles, bfloat16 can only fit one, so we can only convert one
      // nibble at a time
      for (int ii = 0; ii < RegArray::kElements; ++ii) {
        r[ii] = src >> (4 * ii);

        static constexpr uint32_t and_or_imm_lut = (0xf0 & 0xcc) | 0xaa;
        static constexpr uint32_t low_nib_mask = 0x000F000F;

        asm volatile(
            "{\n"
            "  lop3.b32 %0, %0, %1, %2, %3;\n"
            "}\n"
            : "+r"(r[ii + 0])
            : "n"(low_nib_mask), "n"(or_mask), "n"(and_or_imm_lut));

        // For low nibble:
        //  {x1, x0} = {128+(x1+8), 128+(x0+8)} * {1, 1} - {136, 136}
        static constexpr uint32_t low_nib_bias = 0x43084308;  // {136, 136}

        {
          __nv_bfloat162& fp16x2_val = reinterpret_cast<__nv_bfloat162&>(r[ii]);
          fp16x2_val =
              __hsub2(fp16x2_val,
                      reinterpret_cast<const __nv_bfloat162&>(low_nib_bias));
        }
      }

      return reinterpret_cast<PackedResultType&>(r);
    };
  };

 public:
  CUTLASS_DEVICE
  static result_type convert(source_type const& source) {
    return ArrayConverterPacked32Bit<RegConvert, typename result_type::Element,
                                     typename source_type::Element,
                                     N>::convert(source);
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const& s) const { return convert(s); }
};

// for Array<cutlass::bfloat16_t, N> <= Array<uint4_t, N>
//   for IlvdLayout: (2, 4):(4, 1)
template <FloatRoundStyle Round, int N>
struct InterleavedNumericArrayConverter<Layout<Shape<_2, _4>, Stride<_4, _1>>,
                                        cutlass::bfloat16_t, uint4_t, N, Round,
                                        void> {
  using IlvdLayout = Layout<Shape<_2, _4>, Stride<_4, _1>>;
  static_assert(N % size(IlvdLayout{}) == 0);

  using result_type = Array<cutlass::bfloat16_t, N>;
  using source_type = Array<uint4_t, N>;

 private:
  struct RegConvert {
    template <typename PackedResultType>
    CUTLASS_DEVICE static PackedResultType convert(uint32_t src) {
      using RegArray =
          cutlass::AlignedArray<uint32_t, PackedResultType::kElements / 2,
                                sizeof(PackedResultType)>;
      RegArray r;

      static_assert(PackedResultType::kElements <= size(IlvdLayout{}));
      static constexpr uint32_t or_mask = 0x43004300;

      // Unlike float16 where the mantissa is large enough to contain 2
      // nibbles, bfloat16 can only fit one, so we can only convert one
      // nibble at a time
      for (int ii = 0; ii < RegArray::kElements; ++ii) {
        r[ii] = src >> (4 * ii);

        static constexpr uint32_t and_or_imm_lut = (0xf0 & 0xcc) | 0xaa;
        static constexpr uint32_t low_nib_mask = 0x000F000F;

        asm volatile(
            "{\n"
            "  lop3.b32 %0, %0, %1, %2, %3;\n"
            "}\n"
            : "+r"(r[ii])
            : "n"(low_nib_mask), "n"(or_mask), "n"(and_or_imm_lut));

        // For low nibble:
        //  {x1, x0} = {128 + x1, 128 + x0} * {1, 1} - {128, 128}
        static constexpr uint32_t low_nib_bias = 0x43004300;  // {128, 128}

        {
          __nv_bfloat162& fp16x2_val = reinterpret_cast<__nv_bfloat162&>(r[ii]);
          fp16x2_val =
              __hsub2(fp16x2_val,
                      reinterpret_cast<const __nv_bfloat162&>(low_nib_bias));
        }
      }

      return reinterpret_cast<PackedResultType&>(r);
    };
  };

 public:
  CUTLASS_DEVICE
  static result_type convert(source_type const& source) {
    return ArrayConverterPacked32Bit<RegConvert, typename result_type::Element,
                                     typename source_type::Element,
                                     N>::convert(source);
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const& s) const { return convert(s); }
};

// for Array<cutlass::bfloat16_t, N> <= Array<vllm_uint8b128_t, N>
template <FloatRoundStyle Round, int N>
struct NumericArrayConverter<cutlass::bfloat16_t, vllm_uint8b128_t, N, Round> {
  using result_type = Array<cutlass::bfloat16_t, N>;
  using source_type = Array<vllm_uint8b128_t, N>;
  static FloatRoundStyle const round_style = Round;

 private:
  using result_packed_4_t = Array<cutlass::bfloat16_t, 4>;
  using result_packed_2_t = Array<cutlass::bfloat16_t, 2>;
  using src_packed_4_t = Array<vllm_uint8b128_t, 4>;
  using src_packed_2_t = Array<vllm_uint8b128_t, 2>;

  // Not Valid, not supported, only here to satisfy the interface and to avoid
  //  a compile error. ScalarConverter will not actually work until
  //  NumericConverter<cutlass::bfloat16_t, vllm_uint8b128_t, Round> is
  //  implemented
  using ScalarConverter =
      NumericConverter<cutlass::bfloat16_t, vllm_uint8b128_t, Round>;

  template <typename PackedResultType, typename PackedSrcType>
  CUTLASS_DEVICE static PackedResultType packed_convert(
      PackedSrcType const& source) {
    static_assert(
        (platform::is_same<PackedSrcType, src_packed_2_t>::value &&
         platform::is_same<PackedResultType, result_packed_2_t>::value) ||
            (platform::is_same<PackedSrcType, src_packed_4_t>::value &&
             platform::is_same<PackedResultType, result_packed_4_t>::value),
        "Invalid PackedSrcType/PackedResultType must be 2 or 4 to use private "
        "convert dispatch.");

    NumericArrayConverter<float, vllm_uint8b128_t, PackedResultType::kElements,
                          Round>
        convert_uint8_to_f32;
    Array<float, PackedResultType::kElements> tmp =
        convert_uint8_to_f32(source);
    NumericArrayConverter<cutlass::bfloat16_t, float,
                          PackedResultType::kElements, Round>
        convert_f32_to_bf16_;
    return convert_f32_to_bf16_(tmp);
  }

  friend class detail::VectorizedConverter;

 public:
  CUTLASS_DEVICE
  static result_type convert(source_type const& source) {
    result_type result;
    using ConverterType =
        NumericArrayConverter<typename result_type::Element,
                              typename source_type::Element, N, Round>;
    detail::VectorizedConverter::convert<ConverterType, result_packed_4_t,
                                         src_packed_4_t, result_packed_2_t,
                                         src_packed_2_t>(result, source);

    return result;
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const& s) const { return convert(s); }
};

#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
