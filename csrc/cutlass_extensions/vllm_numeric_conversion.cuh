#pragma once

#include "cutlass/numeric_conversion.h"
#include "cutlass_extensions/vllm_custom_types.cuh"
#include "stable/cutlass_extensions/cute_utils.cuh"
#include "cutlass_extensions/vllm_type_utils.cuh"

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
    if (cute::elect_one_sync()) {
      if constexpr (std::is_same_v<IlvBlkLayout, void>) {
        printf(
            "Convert %s <= %s (N = %d, IlvBlkLayout = void), not implemented\n",
            nameof_v<T>, nameof_v<S>, N);
      } else {
        printf(
            "Convert %s <= %s (N = %d, size(IlvBlkLayout{}) = %d), not "
            "implemented\n",
            nameof_v<T>, nameof_v<S>, N, size(IlvBlkLayout{}));
      }
      __brkpt();
    }
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
  CUTLASS_DEVICE static auto to_regs(PackedSrc const& src) {
    if constexpr (sizeof(PackedSrc) == 1) {
      return Array<uint32_t, 1>{reinterpret_cast<uint8_t const&>(src)};
    } else if constexpr (sizeof(PackedSrc) == 2) {
      return Array<uint32_t, 1>{reinterpret_cast<uint16_t const&>(src)};
    } else if constexpr (sizeof(PackedSrc) == 4) {
      return Array<uint32_t, 1>{reinterpret_cast<uint32_t const&>(src)};
    } else {
      static_assert(sizeof(PackedSrc) == 8);
      return reinterpret_cast<Array<uint32_t, 2> const&>(src);
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

    return RegConvert32bit::template convert<PackedResultType>(to_regs(source));
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

// Convert 8 4bit values packed into a 32bit register to 8 8bit values packed
// into 2 32bit register.
template <uint8_t LUT0, uint8_t LUT1, uint8_t LUT2, uint8_t LUT3,    //
          uint8_t LUT4, uint8_t LUT5, uint8_t LUT6, uint8_t LUT7,    //
          uint8_t LUT8, uint8_t LUT9, uint8_t LUT10, uint8_t LUT11,  //
          uint8_t LUT12, uint8_t LUT13, uint8_t LUT14, uint8_t LUT15>
CUTLASS_DEVICE cutlass::AlignedArray<uint32_t, 2> lut_4bit_to_8bit_convert(
    uint32_t src) {
  cutlass::AlignedArray<uint32_t, 2> r;
  // Determines if the value is in the top half of the LUT if set or
  //  (i.e. LUT[8:15]) in the bottom half (i.e. LUT[0:7]) if not set. Then move
  //  into bit position 0x4 of each nibble so when or'd with final_prmt_base it
  //  selects the correct candidate. When elements in final_prmt_base
  //  are >= 0x4, the high candidate is selected (i.e. LUT[8:15]), when elements
  //  are  < 0x4, the low candidate is selected (i.e. LUT[0:7])
  uint32_t high_bit = (src & 0x88888888) >> 1;

  // `high_bit` is OR'd with 0x31203120 to find the correct value in the LUT
  // (selects correct high or low candidate)
  const uint32_t final_prmt_base = 0x32103210;

  // Ignore the high bit when indexing into LUT, for each 4bit value
  //  we index into both the high and low candidates then use
  //  high_bit | final_prmt_base to select the correct candidate
  uint32_t lut_idx = (src & 0x77777777);

  auto pack = [](uint8_t a, uint8_t b, uint8_t c, uint8_t d) {
    return uint32_t(a) | (uint32_t(b) << 8) | (uint32_t(c) << 16) |
           (uint32_t(d) << 24);
  };

  static constexpr uint32_t LOW_0 = pack(LUT0, LUT1, LUT2, LUT3);
  static constexpr uint32_t LOW_1 = pack(LUT4, LUT5, LUT6, LUT7);
  static constexpr uint32_t HIGH_0 = pack(LUT8, LUT9, LUT10, LUT11);
  static constexpr uint32_t HIGH_1 = pack(LUT12, LUT13, LUT14, LUT15);

  CUTLASS_PRAGMA_UNROLL
  for (int ii = 0; ii < 2; ++ii, lut_idx >>= 16, high_bit >>= 16) {
    uint32_t final_prmt_idx = final_prmt_base | high_bit;

    // This uses a look up table to convert packed int4s to packed int8s,
    // using the int4 value as the index to prmt. It first select both the
    // high and low candidates, then uses the high bit (i.e. `high_bit`) to
    // select the correct candidate.
    asm volatile(
        "{\n"
        "  .reg .b32 low, high;\n"
        "  prmt.b32 low, %1, %2, %5;\n"
        "  prmt.b32 high, %3, %4, %5;\n"
        "  prmt.b32 %0, low, high, %6;\n"
        "}\n"
        : "=r"(r[ii])
        : "n"(LOW_0), "n"(LOW_1), "n"(HIGH_0), "n"(HIGH_1), "r"(lut_idx),
          "r"(final_prmt_idx));
  }

  return r;
};

// for Array<int8_t, N> <= Array<vllm_uint4b8_t, N>
template <FloatRoundStyle Round, int N>
struct NumericArrayConverter<int8_t, vllm_uint4b8_t, N, Round> {
  using result_type = Array<int8_t, N>;
  using source_type = Array<vllm_uint4b8_t, N>;

  static FloatRoundStyle const round_style = Round;

 private:
  struct RegConvert {
    template <typename PackedResultType>
    CUTLASS_DEVICE static PackedResultType convert(Array<uint32_t, 1> src_) {
      // [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7] as int8s
      auto r = lut_4bit_to_8bit_convert<0xF8, 0xF9, 0xFA, 0xFB,  //
                                        0xFC, 0xFD, 0xFE, 0xFF,  //
                                        0x00, 0x01, 0x02, 0x03,  //
                                        0x04, 0x05, 0x06, 0x07>(src_[0]);
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

// for Array<cutlass::float_e4m3_t, N> <= Array<vllm_uint4b8_t, N>
template <FloatRoundStyle Round, int N>
struct NumericArrayConverter<cutlass::float_e4m3_t, vllm_uint4b8_t, N, Round> {
  using result_type = Array<cutlass::float_e4m3_t, N>;
  using source_type = Array<vllm_uint4b8_t, N>;

  static FloatRoundStyle const round_style = Round;

 private:
  struct RegConvert {
    template <typename PackedResultType>
    CUTLASS_DEVICE static PackedResultType convert(Array<uint32_t, 1> src_) {
      // [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7] as fp8s
      auto r = lut_4bit_to_8bit_convert<0xD0, 0xCE, 0xCC, 0xCA,  //
                                        0xC8, 0xC4, 0xC0, 0xB8,  //
                                        0x00, 0x38, 0x40, 0x44,  //
                                        0x48, 0x4A, 0x4C, 0x4E>(src_[0]);
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
template <FloatRoundStyle Round, int N>
struct NumericArrayConverter<cutlass::half_t, vllm_uint4b8_t, N, Round> {
  using result_type = Array<cutlass::half_t, N>;
  using source_type = Array<vllm_uint4b8_t, N>;

  struct RegConvert {
    template <typename PackedResultType>
    CUTLASS_DEVICE static PackedResultType convert(Array<uint32_t, 1> src_) {
      uint32_t src = src_[0];
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
    CUTLASS_DEVICE static PackedResultType convert(Array<uint32_t, 1> src_) {
      uint32_t src = src_[0];
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
    CUTLASS_DEVICE static PackedResultType convert(Array<uint32_t, 1> src_) {
      uint32_t src = src_[0];
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
    CUTLASS_DEVICE static PackedResultType convert(Array<uint32_t, 1> src_) {
      uint32_t src = src_[0];
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
    CUTLASS_DEVICE static PackedResultType convert(Array<uint32_t, 1> src_) {
      uint32_t src = src_[0];
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
    CUTLASS_DEVICE static PackedResultType convert(Array<uint32_t, 1> src_) {
      uint32_t src_reg = src_[0];
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
    CUTLASS_DEVICE static PackedResultType convert(Array<uint32_t, 1> src_) {
      uint32_t src = src_[0];
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
    CUTLASS_DEVICE static PackedResultType convert(Array<uint32_t, 1> src_) {
      uint32_t src = src_[0];
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

// for Array<int8_t, N> <= Array<cutlass::half_t, N>
//   FastFP16toINT8 from https://arxiv.org/pdf/2406.09904
template <FloatRoundStyle Round, int N>
struct NumericArrayConverter<int8_t, cutlass::half_t, N, Round> {
  using result_type = Array<int8_t, N>;
  using source_type = Array<cutlass::half_t, N>;

  struct RegConvert {
    // FastFP16toINT8 from https://arxiv.org/pdf/2406.09904
    template <typename PackedResultType, int src_regs>
    CUTLASS_DEVICE static PackedResultType convert(
        Array<uint32_t, src_regs> src) {
      // Hold output int8s in reg. We need 1 reg for every 4 elements
      using RegArray = cutlass::AlignedArray<
          uint32_t, std::max(PackedResultType::kElements / 4, size_t(1))>;
      RegArray r;

      static constexpr uint32_t MAGIC_BIAS_ = 0x64806480;
      auto MAGIC_BIAS = *reinterpret_cast<const half2*>(&MAGIC_BIAS_);

      *reinterpret_cast<half2*>(&src[0]) =
          __hadd2(*reinterpret_cast<half2*>(&src[0]), MAGIC_BIAS);

      if constexpr (src_regs > 1) {
        *reinterpret_cast<half2*>(&src[1]) =
            __hadd2(*reinterpret_cast<half2*>(&src[1]), MAGIC_BIAS);
      }

      static_assert(PackedResultType::kElements <= 4);
      uint32_t uint8s;
      static constexpr uint32_t MASK_0246 = 0x6420;
      static constexpr uint32_t UINT8s_TO_INT8s_MASK = 0x80808080;
      asm volatile("prmt.b32 %0,%1,%2,%3;\n"
                   : "=r"(uint8s)
                   : "r"(src[0]), "r"((src_regs > 1) ? src[1] : src[0]),
                     "n"(MASK_0246));

      uint32_t int8s = (uint8s ^ UINT8s_TO_INT8s_MASK);

      return reinterpret_cast<PackedResultType&>(int8s);
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

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
