
#include "marlin_dtypes.cuh"

namespace MARLIN_NAMESPACE_NAME {

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
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

template <typename scalar_t2, vllm::ScalarTypeId w_type_id>
__device__ inline void dequant(int q, scalar_t2* frag_b);

//
// Efficiently dequantize 4bit values packed in an int32 value into a full
// B-fragment of 4 fp16 values. We mostly follow the strategy in the link below,
// with some small changes:
// - FP16:
// https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h#L215-L287
// - BF16:
// https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h#L327-L385
//
template <>
__device__ inline void dequant<half2, vllm::kU4B8.id()>(int q, half2* frag_b) {
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400;
  // Guarantee that the `(a & b) | c` operations are LOP3s.
  // clang-format off
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
  // clang-format on
  // We want signed int4 outputs, hence we fuse the `-8` symmetric zero point
  // directly into `SUB` and `ADD`.
  const int SUB = 0x64086408;
  const int MUL = 0x2c002c00;
  const int ADD = 0xd480d480;
  frag_b[0] = __hsub2(*reinterpret_cast<half2*>(&lo),
                      *reinterpret_cast<const half2*>(&SUB));
  frag_b[1] = __hfma2(*reinterpret_cast<half2*>(&hi),
                      *reinterpret_cast<const half2*>(&MUL),
                      *reinterpret_cast<const half2*>(&ADD));
}

template <>
__device__ inline void dequant<half2, vllm::kU4.id()>(int q, half2* frag_b) {
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400;
  // Guarantee that the `(a & b) | c` operations are LOP3s.
  // clang-format off
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
  // clang-format on
  // We want signed int4 outputs, hence we fuse the `-8` symmetric zero point
  // directly into `SUB` and `ADD`.
  const int SUB = 0x64006400;
  const int MUL = 0x2c002c00;
  const int ADD = 0xd400d400;
  frag_b[0] = __hsub2(*reinterpret_cast<half2*>(&lo),
                      *reinterpret_cast<const half2*>(&SUB));
  frag_b[1] = __hfma2(*reinterpret_cast<half2*>(&hi),
                      *reinterpret_cast<const half2*>(&MUL),
                      *reinterpret_cast<const half2*>(&ADD));
}

template <>
__device__ inline void dequant<nv_bfloat162, vllm::kU4B8.id()>(
    int q, nv_bfloat162* frag_b) {
  static constexpr uint32_t MASK = 0x000f000f;
  static constexpr uint32_t EX = 0x43004300;

  // Guarantee that the `(a & b) | c` operations are LOP3s.
  // clang-format off
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
  q >>= 4;
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
  // clang-format on

  static constexpr uint32_t MUL = 0x3F803F80;
  static constexpr uint32_t ADD = 0xC308C308;

  frag_b[0] = __hfma2(*reinterpret_cast<nv_bfloat162*>(&lo),
                      *reinterpret_cast<const nv_bfloat162*>(&MUL),
                      *reinterpret_cast<const nv_bfloat162*>(&ADD));
  frag_b[1] = __hfma2(*reinterpret_cast<nv_bfloat162*>(&hi),
                      *reinterpret_cast<const nv_bfloat162*>(&MUL),
                      *reinterpret_cast<const nv_bfloat162*>(&ADD));
}

template <>
__device__ inline void dequant<nv_bfloat162, vllm::kU4.id()>(
    int q, nv_bfloat162* frag_b) {
  static constexpr uint32_t MASK = 0x000f000f;
  static constexpr uint32_t EX = 0x43004300;

  // Guarantee that the `(a & b) | c` operations are LOP3s.
  // clang-format off
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
  q >>= 4;
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
  // clang-format on

  static constexpr uint32_t MUL = 0x3F803F80;
  static constexpr uint32_t ADD = 0xC300C300;

  frag_b[0] = __hfma2(*reinterpret_cast<nv_bfloat162*>(&lo),
                      *reinterpret_cast<const nv_bfloat162*>(&MUL),
                      *reinterpret_cast<const nv_bfloat162*>(&ADD));
  frag_b[1] = __hfma2(*reinterpret_cast<nv_bfloat162*>(&hi),
                      *reinterpret_cast<const nv_bfloat162*>(&MUL),
                      *reinterpret_cast<const nv_bfloat162*>(&ADD));
}

//
// Fast Int8ToFp16/Int8ToBf16: Efficiently dequantize 8bit int values to fp16 or
// bf16 Reference:
// - FP16:
// https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h#L53-L85
// - BF16:
// https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h#L125-L175
//
template <>
__device__ inline void dequant<half2, vllm::kU8B128.id()>(int q,
                                                          half2* frag_b) {
  static constexpr uint32_t mask_for_elt_01 = 0x5250;
  static constexpr uint32_t mask_for_elt_23 = 0x5351;
  static constexpr uint32_t start_byte_for_fp16 = 0x64646464;

  uint32_t lo = prmt<start_byte_for_fp16, mask_for_elt_01>(q);
  uint32_t hi = prmt<start_byte_for_fp16, mask_for_elt_23>(q);

  static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64806480;

  frag_b[0] = __hsub2(*reinterpret_cast<half2*>(&lo),
                      *reinterpret_cast<const half2*>(&I8s_TO_F16s_MAGIC_NUM));
  frag_b[1] = __hsub2(*reinterpret_cast<half2*>(&hi),
                      *reinterpret_cast<const half2*>(&I8s_TO_F16s_MAGIC_NUM));
}

template <>
__device__ inline void dequant<half2, vllm::kU8.id()>(int q, half2* frag_b) {
  static constexpr uint32_t mask_for_elt_01 = 0x5250;
  static constexpr uint32_t mask_for_elt_23 = 0x5351;
  static constexpr uint32_t start_byte_for_fp16 = 0x64646464;

  uint32_t lo = prmt<start_byte_for_fp16, mask_for_elt_01>(q);
  uint32_t hi = prmt<start_byte_for_fp16, mask_for_elt_23>(q);

  static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64006400;

  frag_b[0] = __hsub2(*reinterpret_cast<half2*>(&lo),
                      *reinterpret_cast<const half2*>(&I8s_TO_F16s_MAGIC_NUM));
  frag_b[1] = __hsub2(*reinterpret_cast<half2*>(&hi),
                      *reinterpret_cast<const half2*>(&I8s_TO_F16s_MAGIC_NUM));
}

template <>
__device__ inline void dequant<nv_bfloat162, vllm::kU8B128.id()>(
    int q, nv_bfloat162* frag_b) {
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

  uint32_t* bf16_result_ptr = reinterpret_cast<uint32_t*>(frag_b);
  bf16_result_ptr[0] = __byte_perm(fp32_intermediates_casted[0],
                                   fp32_intermediates_casted[1], 0x7632);
  bf16_result_ptr[1] = __byte_perm(fp32_intermediates_casted[2],
                                   fp32_intermediates_casted[3], 0x7632);
}

template <>
__device__ inline void dequant<nv_bfloat162, vllm::kU8.id()>(
    int q, nv_bfloat162* frag_b) {
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

  uint32_t* bf16_result_ptr = reinterpret_cast<uint32_t*>(frag_b);
  bf16_result_ptr[0] = __byte_perm(fp32_intermediates_casted[0],
                                   fp32_intermediates_casted[1], 0x7632);
  bf16_result_ptr[1] = __byte_perm(fp32_intermediates_casted[2],
                                   fp32_intermediates_casted[3], 0x7632);
}

template <>
__device__ inline void dequant<half2, vllm::kFE4M3fn.id()>(int q,
                                                           half2* frag_b) {
  // Constants for FP8 (E4M3) and FP16 formats
  constexpr int FP8_EXPONENT = 4, FP8_MANTISSA = 3, FP16_EXPONENT = 5;
  constexpr int RIGHT_SHIFT = FP16_EXPONENT - FP8_EXPONENT;

  // Calculate MASK for extracting mantissa and exponent
  constexpr int MASK1 = 0x80000000;
  constexpr int MASK2 = MASK1 >> (FP8_EXPONENT + FP8_MANTISSA);
  constexpr int MASK3 = MASK2 & 0x7fffffff;
  constexpr int MASK = MASK3 | (MASK3 >> 16);
  // Final MASK value: 0x7F007F00

  // Extract and shift FP8 values to FP16 format
  int Out1 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);
  int Out2 = ((q << 8) & 0x80008000) | (((q << 8) & MASK) >> RIGHT_SHIFT);

  // Construct and apply exponent bias
  constexpr int BIAS_OFFSET =
      (1 << (FP16_EXPONENT - 1)) - (1 << (FP8_EXPONENT - 1));
  const half2 bias_reg = __float2half2_rn(float(1 << BIAS_OFFSET));

  // Convert to half2 and apply bias
  // Note: reverse indexing is intentional because weights are permuted
  frag_b[1] = __hmul2(*reinterpret_cast<const half2*>(&Out1), bias_reg);
  frag_b[0] = __hmul2(*reinterpret_cast<const half2*>(&Out2), bias_reg);
}

template <>
__device__ inline void dequant<nv_bfloat162, vllm::kFE4M3fn.id()>(
    int q, nv_bfloat162* frag_b) {
  // Constants for FP8 (E4M3) and BF16 formats
  constexpr int FP8_EXPONENT = 4, FP8_MANTISSA = 3, BF16_EXPONENT = 8;
  constexpr int RIGHT_SHIFT = BF16_EXPONENT - FP8_EXPONENT;

  // Calculate MASK for extracting mantissa and exponent
  constexpr int MASK1 = 0x80000000;
  constexpr int MASK2 = MASK1 >> (FP8_EXPONENT + FP8_MANTISSA);
  constexpr int MASK3 = MASK2 & 0x7fffffff;
  constexpr int MASK = MASK3 | (MASK3 >> 16);
  // Final MASK value: 0x7F007F00

  // Extract and shift FP8 values to BF16 format
  int Out1 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);
  int Out2 = ((q << 8) & 0x80008000) | (((q << 8) & MASK) >> RIGHT_SHIFT);

  // Construct and apply exponent bias
  constexpr int BIAS_OFFSET =
      (1 << (BF16_EXPONENT - 1)) - (1 << (FP8_EXPONENT - 1));
  // Add 127 (float exponent bias) to BIAS_OFFSET and shift to float exponent
  // position
  constexpr uint32_t BIAS = (BIAS_OFFSET + 127) << 23;
  const nv_bfloat162 bias_reg =
      __float2bfloat162_rn(*reinterpret_cast<const float*>(&BIAS));

  // Convert to bfloat162 and apply bias
  // Note: reverse indexing is intentional because weights are permuted
  frag_b[1] = __hmul2(*reinterpret_cast<const nv_bfloat162*>(&Out1), bias_reg);
  frag_b[0] = __hmul2(*reinterpret_cast<const nv_bfloat162*>(&Out2), bias_reg);
}

#endif

}  // namespace MARLIN_NAMESPACE_NAME
