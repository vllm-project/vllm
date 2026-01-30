
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace vllm {
namespace moe {

template <typename scalar_t>
class ScalarType {};

template <>
class ScalarType<half> {
 public:
  using scalar_t = half;
  using scalar_t2 = half2;

  static __device__ float inline num2float(const half x) {
    return __half2float(x);
  }

  static __device__ half2 inline num2num2(const half x) {
    return __half2half2(x);
  }

  static __device__ half2 inline nums2num2(const half x1, const half x2) {
    return __halves2half2(x1, x2);
  }

  static __host__ __device__ half inline float2num(const float x) {
    return __float2half(x);
  }

  static __host__ __device__ half inline int2num(const float x) {
    return __int2half_rn(x);
  }

  static __host__ __device__ float2 inline num22float2(const half2 x) {
    return __half22float2(x);
  }

  static __host__ __device__ half2 inline float22num2(const float2 x) {
    return __float22half2_rn(x);
  }
};

template <>
class ScalarType<nv_bfloat16> {
 public:
  using scalar_t = nv_bfloat16;
  using scalar_t2 = nv_bfloat162;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  static __device__ float inline num2float(const nv_bfloat16 x) {
    return __bfloat162float(x);
  }

  static __device__ nv_bfloat162 inline num2num2(const nv_bfloat16 x) {
    return __bfloat162bfloat162(x);
  }

  static __device__ nv_bfloat162 inline nums2num2(const nv_bfloat16 x1,
                                                  const nv_bfloat16 x2) {
    return __halves2bfloat162(x1, x2);
  }

  static __host__ __device__ nv_bfloat16 inline float2num(const float x) {
    return __float2bfloat16(x);
  }

  static __host__ __device__ nv_bfloat16 inline int2num(const float x) {
    return __int2bfloat16_rn(x);
  }

  static __host__ __device__ float2 inline num22float2(const nv_bfloat162 x) {
    return __bfloat1622float2(x);
  }

  static __host__ __device__ nv_bfloat162 inline float22num2(const float2 x) {
    return __float22bfloat162_rn(x);
  }
#endif
};

}  // namespace moe
}  // namespace vllm

template <int lut>
__device__ inline int lop3(int a, int b, int c) {
  int res;
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(res)
               : "r"(a), "r"(b), "r"(c), "n"(lut));
  return res;
}

template <int start_byte, int mask>
__device__ inline uint32_t prmt(uint32_t a) {
  uint32_t res;
  asm volatile("prmt.b32 %0, %1, %2, %3;\n"
               : "=r"(res)
               : "r"(a), "n"(start_byte), "n"(mask));
  return res;
}

template <typename scalar_t2, int bit>
__device__ inline void dequant(int q, scalar_t2* res) {}

template <>
__device__ inline void dequant<half2, 4>(int q, half2* res) {
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400;
  const int SUB = 0x64006400;
  const int MUL = 0x2c002c00;
  const int ADD = 0xd400d400;

  int lo0 = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi0 = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
  q >>= 8;
  int lo1 = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi1 = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);

  res[0] = __hsub2(*reinterpret_cast<half2*>(&lo0),
                   *reinterpret_cast<const half2*>(&SUB));
  res[1] = __hfma2(*reinterpret_cast<half2*>(&hi0),
                   *reinterpret_cast<const half2*>(&MUL),
                   *reinterpret_cast<const half2*>(&ADD));
  res[2] = __hsub2(*reinterpret_cast<half2*>(&lo1),
                   *reinterpret_cast<const half2*>(&SUB));
  res[3] = __hfma2(*reinterpret_cast<half2*>(&hi1),
                   *reinterpret_cast<const half2*>(&MUL),
                   *reinterpret_cast<const half2*>(&ADD));
}

template <>
__device__ inline void dequant<half2, 8>(int q, half2* res) {
  static constexpr uint32_t mask_for_elt_01 = 0x5250;
  static constexpr uint32_t mask_for_elt_23 = 0x5351;
  static constexpr uint32_t start_byte_for_fp16 = 0x64646464;

  uint32_t lo = prmt<start_byte_for_fp16, mask_for_elt_01>(q);
  uint32_t hi = prmt<start_byte_for_fp16, mask_for_elt_23>(q);

  static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64006400;

  res[0] = __hsub2(*reinterpret_cast<half2*>(&lo),
                   *reinterpret_cast<const half2*>(&I8s_TO_F16s_MAGIC_NUM));
  res[1] = __hsub2(*reinterpret_cast<half2*>(&hi),
                   *reinterpret_cast<const half2*>(&I8s_TO_F16s_MAGIC_NUM));
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
template <>
__device__ inline void dequant<nv_bfloat162, 4>(int q, nv_bfloat162* res) {
  static constexpr uint32_t MASK = 0x000f000f;
  static constexpr uint32_t EX = 0x43004300;

  int lo0 = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
  q >>= 4;
  int hi0 = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
  q >>= 4;
  int lo1 = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
  q >>= 4;
  int hi1 = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);

  static constexpr uint32_t MUL = 0x3F803F80;
  static constexpr uint32_t ADD = 0xC300C300;

  res[0] = __hfma2(*reinterpret_cast<nv_bfloat162*>(&lo0),
                   *reinterpret_cast<const nv_bfloat162*>(&MUL),
                   *reinterpret_cast<const nv_bfloat162*>(&ADD));
  res[1] = __hfma2(*reinterpret_cast<nv_bfloat162*>(&hi0),
                   *reinterpret_cast<const nv_bfloat162*>(&MUL),
                   *reinterpret_cast<const nv_bfloat162*>(&ADD));
  res[2] = __hfma2(*reinterpret_cast<nv_bfloat162*>(&lo1),
                   *reinterpret_cast<const nv_bfloat162*>(&MUL),
                   *reinterpret_cast<const nv_bfloat162*>(&ADD));
  res[3] = __hfma2(*reinterpret_cast<nv_bfloat162*>(&hi1),
                   *reinterpret_cast<const nv_bfloat162*>(&MUL),
                   *reinterpret_cast<const nv_bfloat162*>(&ADD));
}

template <>
__device__ inline void dequant<nv_bfloat162, 8>(int q, nv_bfloat162* res) {
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

  uint32_t* bf16_result_ptr = reinterpret_cast<uint32_t*>(res);
  bf16_result_ptr[0] = __byte_perm(fp32_intermediates_casted[0],
                                   fp32_intermediates_casted[1], 0x7632);
  bf16_result_ptr[1] = __byte_perm(fp32_intermediates_casted[2],
                                   fp32_intermediates_casted[3], 0x7632);
}
#endif
