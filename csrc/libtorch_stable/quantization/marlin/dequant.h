/*
Fast Dequantization (Converting INT4/INT8/FP4/FP8 to FP16/BF16)

The process of fast dequantization can be summarized as a combination
of bitwise operations and floating-point computations:

weight =>(bit_op / bitwise operations)=>
f16_value =>(flop / floating-point computation)=>
dequantized_weight

Since the dequantized weights typically require subtracting the zero point and
applying a scale factor, the floating-point computation step can be fused with
the zero-point subtraction and scaling operations.

The following are the parts that need to be modified for the fused operation
of zero-point subtraction and scaling.

## INT4 => FP16/BF16 or INT8 => FP16

The floating-point computation is `__hsub2`

If has zero points:

    flop(bit_op(weight)) - flop(bit_op(zp))
  = sub(bit_op(weight), bias) - sub(bit_op(zp), bias)
  = bit_op(weight) - bit_op(zp)

so we don't need additional modification.

If has float zero points:

    flop(bit_op(weight)) - fzp
  = sub(bit_op(weight), bias) - fzp
  = bit_op(weight) - (fzp + bias)

where the `fzp + bias` can be computed at weight loading. But this
may have accuracy issue, so we should not use this in most cases.

If has not zero points:

    scale(flop(bit_op(weight)))
  = scale(sub(bit_op(weight), bias))
  = scale(bit_op(weight)) - scale(bias)
  = fma(bit_op(weight), scale_factor, scale(bias))

where the `scale(bias)` can be cached. But this may have accuracy issue,
so we should not use this in most cases.


## INT8 => BF16

INT8 => BF16 is a special case, it use byte_perm instead of flop.
We cannot fused byte_perm with scaling.


## FP4/FP8 => FP16/BF16

    scale(flop(bit_op(weight)))
  = scale(mul(bit_op(weight), multiplier))
  = mul(bit_op(weight), scale_factor * multiplier)

where `scale_factor * multiplier` can be computed at weight loading.

*/

#include "marlin_dtypes.cuh"

namespace MARLIN_NAMESPACE_NAME {

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 750
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

template <typename scalar_t2, vllm::ScalarTypeId w_type_id,
          bool skip_flop = false>
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
__device__ inline void dequant<half2, vllm::kU4B8.id(), true>(int q,
                                                              half2* frag_b) {
  const int MASK = 0x000f000f;
  const int EX = 0x64006400;
  // Guarantee that the `(a & b) | c` operations are LOP3s.
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
  q >>= 4;
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);

  frag_b[0] = *reinterpret_cast<half2*>(&lo);
  frag_b[1] = *reinterpret_cast<half2*>(&hi);
}

template <>
__device__ inline void dequant<half2, vllm::kU4B8.id(), false>(int q,
                                                               half2* frag_b) {
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
__device__ inline void dequant<half2, vllm::kU4.id(), true>(int q,
                                                            half2* frag_b) {
  dequant<half2, vllm::kU4B8.id(), true>(q, frag_b);
}

template <>
__device__ inline void dequant<half2, vllm::kU4.id(), false>(int q,
                                                             half2* frag_b) {
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
__device__ inline void dequant<nv_bfloat162, vllm::kU4B8.id(), true>(
    int q, nv_bfloat162* frag_b) {
  static constexpr uint32_t MASK = 0x000f000f;
  static constexpr uint32_t EX = 0x43004300;

  // Guarantee that the `(a & b) | c` operations are LOP3s.
  // clang-format off
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
  q >>= 4;
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
  // clang-format on

  frag_b[0] = *reinterpret_cast<nv_bfloat162*>(&lo);
  frag_b[1] = *reinterpret_cast<nv_bfloat162*>(&hi);
}

template <>
__device__ inline void dequant<nv_bfloat162, vllm::kU4B8.id(), false>(
    int q, nv_bfloat162* frag_b) {
  dequant<nv_bfloat162, vllm::kU4B8.id(), true>(q, frag_b);

  static constexpr uint32_t SUB = 0x43084308;

  frag_b[0] = __hsub2(frag_b[0], *reinterpret_cast<const nv_bfloat162*>(&SUB));
  frag_b[1] = __hsub2(frag_b[1], *reinterpret_cast<const nv_bfloat162*>(&SUB));
}

template <>
__device__ inline void dequant<nv_bfloat162, vllm::kU4.id(), true>(
    int q, nv_bfloat162* frag_b) {
  dequant<nv_bfloat162, vllm::kU4B8.id(), true>(q, frag_b);
}

template <>
__device__ inline void dequant<nv_bfloat162, vllm::kU4.id(), false>(
    int q, nv_bfloat162* frag_b) {
  dequant<nv_bfloat162, vllm::kU4.id(), true>(q, frag_b);

  static constexpr uint32_t SUB = 0x43004300;

  frag_b[0] = __hsub2(frag_b[0], *reinterpret_cast<const nv_bfloat162*>(&SUB));
  frag_b[1] = __hsub2(frag_b[1], *reinterpret_cast<const nv_bfloat162*>(&SUB));
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
__device__ inline void dequant<half2, vllm::kU8B128.id(), true>(int q,
                                                                half2* frag_b) {
  static constexpr uint32_t mask_for_elt_01 = 0x5250;
  static constexpr uint32_t mask_for_elt_23 = 0x5351;
  static constexpr uint32_t start_byte_for_fp16 = 0x64646464;

  uint32_t lo = prmt<start_byte_for_fp16, mask_for_elt_01>(q);
  uint32_t hi = prmt<start_byte_for_fp16, mask_for_elt_23>(q);

  frag_b[0] = *reinterpret_cast<half2*>(&lo);
  frag_b[1] = *reinterpret_cast<half2*>(&hi);
}

template <>
__device__ inline void dequant<half2, vllm::kU8B128.id(), false>(
    int q, half2* frag_b) {
  dequant<half2, vllm::kU8B128.id(), true>(q, frag_b);

  static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64806480;
  frag_b[0] = __hsub2(frag_b[0],
                      *reinterpret_cast<const half2*>(&I8s_TO_F16s_MAGIC_NUM));
  frag_b[1] = __hsub2(frag_b[1],
                      *reinterpret_cast<const half2*>(&I8s_TO_F16s_MAGIC_NUM));
}

template <>
__device__ inline void dequant<half2, vllm::kU8.id(), true>(int q,
                                                            half2* frag_b) {
  dequant<half2, vllm::kU8B128.id(), true>(q, frag_b);
}

template <>
__device__ inline void dequant<half2, vllm::kU8.id(), false>(int q,
                                                             half2* frag_b) {
  dequant<half2, vllm::kU8.id(), true>(q, frag_b);

  static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64006400;
  frag_b[0] = __hsub2(frag_b[0],
                      *reinterpret_cast<const half2*>(&I8s_TO_F16s_MAGIC_NUM));
  frag_b[1] = __hsub2(frag_b[1],
                      *reinterpret_cast<const half2*>(&I8s_TO_F16s_MAGIC_NUM));
}

template <>
__device__ inline void dequant<nv_bfloat162, vllm::kU8B128.id(), false>(
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
__device__ inline void dequant<nv_bfloat162, vllm::kU8.id(), false>(
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
__device__ inline void dequant<half2, vllm::kFE4M3fn.id(), true>(
    int q, half2* frag_b) {
  // Constants for FP8 (E4M3) and FP16 formats
  constexpr int FP8_EXPONENT = 4, FP16_EXPONENT = 5;
  constexpr int RIGHT_SHIFT = FP16_EXPONENT - FP8_EXPONENT;
  constexpr int MASK = 0x7F007F00;

  // Extract and shift FP8 values to FP16 format
  int Out1 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);
  q <<= 8;
  int Out2 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);

  // Note: reverse indexing is intentional because weights are permuted
  frag_b[1] = *reinterpret_cast<const half2*>(&Out1);
  frag_b[0] = *reinterpret_cast<const half2*>(&Out2);
}

template <>
__device__ inline void dequant<half2, vllm::kFE4M3fn.id(), false>(
    int q, half2* frag_b) {
  dequant<half2, vllm::kFE4M3fn.id(), true>(q, frag_b);

  // Constants for FP8 (E4M3) and FP16 formats
  constexpr int FP8_EXPONENT = 4, FP16_EXPONENT = 5;

  // Construct and apply exponent bias
  constexpr int BIAS_OFFSET =
      (1 << (FP16_EXPONENT - 1)) - (1 << (FP8_EXPONENT - 1));
  const half2 bias_reg = __float2half2_rn(float(1 << BIAS_OFFSET));

  // Convert to half2 and apply bias
  frag_b[1] = __hmul2(frag_b[1], bias_reg);
  frag_b[0] = __hmul2(frag_b[0], bias_reg);
}

template <>
__device__ inline void dequant<nv_bfloat162, vllm::kFE4M3fn.id(), true>(
    int q, nv_bfloat162* frag_b) {
  // Constants for FP8 (E4M3) and BF16 formats
  constexpr int FP8_EXPONENT = 4, BF16_EXPONENT = 8;
  constexpr int RIGHT_SHIFT = BF16_EXPONENT - FP8_EXPONENT;

  constexpr int MASK = 0x7F007F00;

  // Extract and shift FP8 values to BF16 format
  int Out1 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);
  q <<= 8;
  int Out2 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);

  // Note: reverse indexing is intentional because weights are permuted
  frag_b[1] = *reinterpret_cast<const nv_bfloat162*>(&Out1);
  frag_b[0] = *reinterpret_cast<const nv_bfloat162*>(&Out2);
}

template <>
__device__ inline void dequant<nv_bfloat162, vllm::kFE4M3fn.id(), false>(
    int q, nv_bfloat162* frag_b) {
  dequant<nv_bfloat162, vllm::kFE4M3fn.id(), true>(q, frag_b);

  // Constants for FP8 (E4M3) and BF16 formats
  constexpr int FP8_EXPONENT = 4, BF16_EXPONENT = 8;

  // Construct and apply exponent bias
  constexpr int BIAS_OFFSET =
      (1 << (BF16_EXPONENT - 1)) - (1 << (FP8_EXPONENT - 1));
  // Add 127 (float exponent bias) to BIAS_OFFSET and shift to float exponent
  // position
  constexpr uint32_t BIAS = (BIAS_OFFSET + 127) << 23;
  const nv_bfloat162 bias_reg =
      __float2bfloat162_rn(*reinterpret_cast<const float*>(&BIAS));

  // Convert to bfloat162 and apply bias
  frag_b[1] = __hmul2(frag_b[1], bias_reg);
  frag_b[0] = __hmul2(frag_b[0], bias_reg);
}

template <>
__device__ inline void dequant<half2, vllm::kFE2M1f.id(), true>(int q,
                                                                half2* frag_b) {
  // Constants for FP4 (E2M1) and FP16 formats
  constexpr int FP4_EXPONENT = 2, FP16_EXPONENT = 5;
  constexpr int RIGHT_SHIFT = FP16_EXPONENT - FP4_EXPONENT;
  constexpr int MASK = 0x70007000;

  // Extract and shift FP4 values to FP16 format
  int Out1 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);
  q <<= 4;
  int Out2 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);

  // Note: reverse indexing is intentional because weights are permuted
  frag_b[1] = *reinterpret_cast<const half2*>(&Out1);
  frag_b[0] = *reinterpret_cast<const half2*>(&Out2);
}

template <>
__device__ inline void dequant<half2, vllm::kFE2M1f.id(), false>(
    int q, half2* frag_b) {
  dequant<half2, vllm::kFE2M1f.id(), true>(q, frag_b);

  // Constants for FP4 (E2M1) and FP16 formats
  constexpr int FP4_EXPONENT = 2, FP16_EXPONENT = 5;

  // Construct and apply exponent bias
  constexpr int BIAS_OFFSET =
      (1 << (FP16_EXPONENT - 1)) - (1 << (FP4_EXPONENT - 1));
  const half2 bias_reg = __float2half2_rn(float(1 << BIAS_OFFSET));

  // Convert to half2 and apply bias
  frag_b[1] = __hmul2(frag_b[1], bias_reg);
  frag_b[0] = __hmul2(frag_b[0], bias_reg);
}

template <>
__device__ inline void dequant<nv_bfloat162, vllm::kFE2M1f.id(), true>(
    int q, nv_bfloat162* frag_b) {
  // Constants for FP4 (E2M1) and FP16 formats
  constexpr int FP4_EXPONENT = 2, BF16_EXPONENT = 8;
  constexpr int RIGHT_SHIFT = BF16_EXPONENT - FP4_EXPONENT;
  constexpr int MASK = 0x70007000;

  // Extract and shift FP4 values to FP16 format
  int Out1 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);
  q <<= 4;
  int Out2 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);

  // Note: reverse indexing is intentional because weights are permuted
  frag_b[1] = *reinterpret_cast<const nv_bfloat162*>(&Out1);
  frag_b[0] = *reinterpret_cast<const nv_bfloat162*>(&Out2);
}

template <>
__device__ inline void dequant<nv_bfloat162, vllm::kFE2M1f.id(), false>(
    int q, nv_bfloat162* frag_b) {
  dequant<nv_bfloat162, vllm::kFE2M1f.id(), true>(q, frag_b);

  // Constants for FP4 (E2M1) and BF16 formats
  constexpr int FP4_EXPONENT = 2, BF16_EXPONENT = 8;

  // Construct and apply exponent bias
  constexpr int BIAS_OFFSET =
      (1 << (BF16_EXPONENT - 1)) - (1 << (FP4_EXPONENT - 1));
  // Add 127 (float exponent bias) to BIAS_OFFSET and shift to float exponent
  // position
  constexpr uint32_t BIAS = (BIAS_OFFSET + 127) << 23;
  const nv_bfloat162 bias_reg =
      __float2bfloat162_rn(*reinterpret_cast<const float*>(&BIAS));

  // Convert to half2 and apply bias
  frag_b[1] = __hmul2(frag_b[1], bias_reg);
  frag_b[0] = __hmul2(frag_b[0], bias_reg);
}

template <>
__device__ inline void dequant<__nv_fp8x4_e4m3, vllm::kFE2M1f.id(), true>(
    int q, __nv_fp8x4_e4m3* frag_b) {
  // Constants for FP4 (E2M1) and FP16 formats
  constexpr int FP4_EXPONENT = 2, FP8_EXPONENT = 4;
  constexpr int RIGHT_SHIFT = FP8_EXPONENT - FP4_EXPONENT;
  constexpr int MASK = 0x70707070;

  // Extract and shift FP4 values to FP16 format
  int Out1 = (q & 0x80808080) | ((q & MASK) >> RIGHT_SHIFT);
  q <<= 4;
  int Out2 = (q & 0x80808080) | ((q & MASK) >> RIGHT_SHIFT);

  // Note1: reverse indexing is intentional because weights are permuted
  // Note2: when dequant to 8bit type, we write to `frag_b[2]` instead of
  //        `frag_b[1]` to fit the layout of tensorcore
  frag_b[1] = *reinterpret_cast<const __nv_fp8x4_e4m3*>(&Out1);
  frag_b[0] = *reinterpret_cast<const __nv_fp8x4_e4m3*>(&Out2);
}

template <>
__device__ inline void dequant<int32_t, vllm::kU4B8.id(), true>(
    int q, int32_t* frag_b) {
  constexpr int repeated_zp = 0x08080808;
  constexpr int MASK = 0x80808080;

  frag_b[0] = ((q & 0x0F0F0F0F | MASK) - repeated_zp) ^ MASK;
  q >>= 4;
  frag_b[1] = ((q & 0x0F0F0F0F | MASK) - repeated_zp) ^ MASK;
}

template <>
__device__ inline void dequant<__nv_fp8x4_e4m3, vllm::kU4B8.id(), true>(
    int q, __nv_fp8x4_e4m3* frag_b) {
  int s = q & 0x08080808;
  int Out1 = ((q & 0x07070707) | (s << 4)) + (s >> 3);
  q >>= 4;
  s = q & 0x08080808;
  int Out2 = ((q & 0x07070707) | (s << 4)) + (s >> 3);

  frag_b[0] = *reinterpret_cast<const __nv_fp8x4_e4m3*>(&Out1);
  frag_b[1] = *reinterpret_cast<const __nv_fp8x4_e4m3*>(&Out2);
}

template <typename scalar_t2, vllm::ScalarTypeId s_type_id>
__device__ inline void dequant_fp8_scales(int q, scalar_t2* frag_b);

template <>
__device__ inline void dequant_fp8_scales<half2, vllm::kFE4M3fn.id()>(
    int q, half2* frag_b) {
  int Out1 = (q & 0xFF00FF00) >> 1;
  ;
  q <<= 8;
  int Out2 = (q & 0xFF00FF00) >> 1;

  // Note: reverse indexing is intentional because weights are permuted
  frag_b[1] = *reinterpret_cast<const half2*>(&Out1);
  frag_b[0] = *reinterpret_cast<const half2*>(&Out2);
};

template <>
__device__ inline void dequant_fp8_scales<nv_bfloat162, vllm::kFE4M3fn.id()>(
    int q, nv_bfloat162* frag_b) {
  constexpr int FP8_EXPONENT = 4, BF16_EXPONENT = 8;
  constexpr int RIGHT_SHIFT = BF16_EXPONENT - FP8_EXPONENT;
  constexpr int MASK = 0x7F007F00;

  // Extract and shift FP8 values to BF16 format
  int Out1 = ((q & 0x80008000) >> 1) | ((q & MASK) >> RIGHT_SHIFT);
  q <<= 8;
  int Out2 = ((q & 0x80008000) >> 1) | ((q & MASK) >> RIGHT_SHIFT);

  // Note: reverse indexing is intentional because weights are permuted
  frag_b[1] = *reinterpret_cast<const nv_bfloat162*>(&Out1);
  frag_b[0] = *reinterpret_cast<const nv_bfloat162*>(&Out2);
}

template <>
__device__ inline void dequant_fp8_scales<nv_bfloat162, vllm::kFE8M0fnu.id()>(
    int q, nv_bfloat162* frag_b) {
  // In this conversion, 2 ** -127 in FP8E8M0 would become 0 in BF16,
  // but we assume that such a extreme value would not occur in real models.
  int Out1 = (q & 0xFF00FF00) >> 1;
  q <<= 7;
  int Out2 = q & 0x7F807F80;

  // Note: reverse indexing is intentional because weights are permuted
  frag_b[1] = *reinterpret_cast<const nv_bfloat162*>(&Out1);
  frag_b[0] = *reinterpret_cast<const nv_bfloat162*>(&Out2);
};

// subtract zero point in quanted format and then dequant
template <typename scalar_t2, vllm::ScalarTypeId w_type_id,
          bool skip_flop = false>
__device__ inline void sub_zp_and_dequant(int q, scalar_t2* frag_b, int zp);

template <>
__device__ inline void sub_zp_and_dequant<int32_t, vllm::kU4.id(), true>(
    int q, int32_t* frag_b, int zp) {
  // INT4 with zp -> INT8
  // see https://github.com/vllm-project/vllm/pull/24722
  int repeated_zp = 0x01010101 * zp;
  int MASK = 0x80808080;

  frag_b[0] = ((q & 0x0F0F0F0F | MASK) - repeated_zp) ^ MASK;
  q >>= 4;
  frag_b[1] = ((q & 0x0F0F0F0F | MASK) - repeated_zp) ^ MASK;
}

template <>
__device__ inline void sub_zp_and_dequant<__nv_fp8x4_e4m3, vllm::kU4.id(),
                                          true>(int q, __nv_fp8x4_e4m3* frag_b,
                                                int zp) {
  // INT4 with zp -> FP8
  // see https://github.com/vllm-project/vllm/pull/24722
  uint32_t u_q = *reinterpret_cast<uint32_t*>(&q);
  uint32_t u_zp = *reinterpret_cast<uint32_t*>(&zp);
  uint32_t u_zp1 = u_zp + 1;
  uint32_t repeated_zp = 0x01010101 * u_zp;

  uint32_t q0, s;
  q0 = (u_q & 0x0F0F0F0F) | 0x70707070;
  s = (q0 + repeated_zp) & 0x80808080;
  uint32_t Out1 = (q0 + (s >> 7) * u_zp1) & 0x0F0F0F0F | s;

  u_q >>= 4;
  q0 = (u_q & 0x0F0F0F0F) | 0x70707070;
  s = (q0 + repeated_zp) & 0x80808080;
  uint32_t Out2 = (q0 + (s >> 7) * u_zp1) & 0x0F0F0F0F | s;

  frag_b[0] = *reinterpret_cast<const __nv_fp8x4_e4m3*>(&Out1);
  frag_b[1] = *reinterpret_cast<const __nv_fp8x4_e4m3*>(&Out2);
}

#endif

}  // namespace MARLIN_NAMESPACE_NAME
