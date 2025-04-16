#pragma once

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <iostream>
#include "../gptq_marlin/marlin_dtypes.cuh"
using marlin::ScalarType;

namespace allspark {

#define CHECK_CUDA(cmd)                                             \
  do {                                                              \
    cudaError_t cuda_status = cmd;                                  \
    if (cuda_status != cudaSuccess) {                               \
      std::string err_str = cudaGetErrorString(cuda_status);        \
      std::cerr << "Failed: " << __FILE__ << ":" << __LINE__ << " " \
                << err_str;                                         \
      exit(-1);                                                     \
    }                                                               \
  } while (0)

#define CHECK_CUBLAS(cmd)                                            \
  do {                                                               \
    cublasStatus_t cublas_status = cmd;                              \
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {                    \
      std::cerr << "Failed:  " << __FILE__ << ":" << __LINE__ << " " \
                << cublas_status << std::endl;                       \
      exit(-1);                                                      \
    }                                                                \
  } while (0)

template <typename FType, typename QType>
struct SM8x_GEMM_W8A16_Splitk_Params {
  const FType* A_ptr;
  const QType* B_ptr;
  const FType* B_scale_ptr;
  const FType* B_zero_ptr;
  FType* C_ptr;
  int M;
  int N;
  int K;
  int SplitK;
  int GroupCnt;
  int GroupSize;
  FType* C_split_ptr;       // for non-fused splitk reduce
  float* C_tmp_ptr;         // for fused splitk reduce
  uint32_t* red_count_ptr;  // for fused splitk reduce
};

struct alignas(16) BlockTileSplitkParams {
  int Mtile;
  int Ntile;
  int SplitK;
  bool EnableFuse;
};

template <typename FType, int BLOCK, int N_MATRIX>
__global__ void f16_gemm_splitk_reduce_kernel(const FType* C_split, FType* C,
                                              uint32_t n, uint32_t n_matrix,
                                              uint32_t matrix_size) {
  auto idx = blockIdx.x * BLOCK + threadIdx.x;

  if (idx >= matrix_size) {
    return;
  }

  float sum = 0.f;

  int n_mat = N_MATRIX > 0 ? N_MATRIX : (int)n_matrix;
  for (int i = 0; i < n_mat; ++i) {
    sum += ScalarType<FType>::num2float(C_split[idx + i * matrix_size]);
  }

  C[idx] = ScalarType<FType>::float2num(sum);
}

template <typename FType>
void f16_gemm_splitk_reduce(const FType* C_split, FType* C, const uint32_t m,
                            const uint32_t n, const uint32_t n_matrix,
                            cudaStream_t stream) {
  const int BLOCK = 128;
  uint32_t matrix_size = m * n;
  int grid = (matrix_size + BLOCK - 1) / BLOCK;

  void (*kernel)(const FType*, FType*, uint32_t, uint32_t, uint32_t) = nullptr;

  switch (n_matrix) {
    case 4:
      kernel = f16_gemm_splitk_reduce_kernel<FType, BLOCK, 4>;
      break;
    case 5:
      kernel = f16_gemm_splitk_reduce_kernel<FType, BLOCK, 5>;
      break;
    case 6:
      kernel = f16_gemm_splitk_reduce_kernel<FType, BLOCK, 6>;
      break;
    case 7:
      kernel = f16_gemm_splitk_reduce_kernel<FType, BLOCK, 7>;
      break;
    case 8:
      kernel = f16_gemm_splitk_reduce_kernel<FType, BLOCK, 8>;
      break;
    case 9:
      kernel = f16_gemm_splitk_reduce_kernel<FType, BLOCK, 9>;
      break;
    case 10:
      kernel = f16_gemm_splitk_reduce_kernel<FType, BLOCK, 10>;
      break;
    case 11:
      kernel = f16_gemm_splitk_reduce_kernel<FType, BLOCK, 11>;
      break;
    case 12:
      kernel = f16_gemm_splitk_reduce_kernel<FType, BLOCK, 12>;
      break;
    default:
      kernel = f16_gemm_splitk_reduce_kernel<FType, BLOCK, -1>;
      break;
  }

  kernel<<<grid, BLOCK, 0, stream>>>(C_split, C, n, n_matrix, matrix_size);
}

template <typename T>
struct HalfType;
template <>
struct HalfType<half> {
  using T1 = __half;
  using T2 = __half2;
};
template <>
struct HalfType<__nv_bfloat16> {
  using T1 = __nv_bfloat16;
  using T2 = __nv_bfloat162;
};

// convert 64-bit pointer to 32-bit smem addr
__device__ __forceinline__ uint32_t smem_u32addr(const void* smem_ptr) {
  uint32_t addr;
  asm("{.reg .u64 u64addr;\n"
      " cvta.to.shared.u64 u64addr, %1;\n"
      " cvt.u32.u64 %0, u64addr;}\n"
      : "=r"(addr)
      : "l"(smem_ptr));

  return addr;
}

template <typename T>
__device__ __forceinline__ void ldg16_cg_0(T& r0, const void* ptr, bool guard) {
  static_assert(sizeof(T) == 2, "ldg16_cg_0: invalid T");

  asm volatile(
      "{.reg .pred p;\n"
      " setp.ne.b32 p, %2, 0;\n"
      " @!p mov.b16 %0, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750
      " @p ld.global.cg.L2::128B.b16 {%0}, [%1];}\n"
#else
      " @p ld.global.ca.b16 {%0}, [%1];}\n"
#endif
      : "=h"(reinterpret_cast<uint16_t&>(r0))
      : "l"(ptr), "r"((int)guard));
}

template <typename T>
__device__ __forceinline__ void ldg64_ca(T& r0, T& r1, const void* ptr,
                                         bool guard) {
  static_assert(sizeof(T) == 4, "ldg64_ca: invalid T");

  asm volatile(
      "{.reg .pred p;\n"
      " setp.ne.b32 p, %3, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750
      " @p ld.global.ca.L2::128B.v2.b32 {%0, %1}, [%2];}\n"
#else
      " @p ld.global.ca.v2.b32 {%0, %1}, [%2];}\n"
#endif
      : "=r"(reinterpret_cast<uint32_t&>(r0)),
        "=r"(reinterpret_cast<uint32_t&>(r1))
      : "l"(ptr), "r"((int)guard));
}

template <typename T>
__device__ __forceinline__ void ldg128_cg_0(T& r0, T& r1, T& r2, T& r3,
                                            const void* ptr, bool guard) {
  static_assert(sizeof(T) == 4, "ldg128_cg_0: invalid T");

  asm volatile(
      "{.reg .pred p;\n"
      " setp.ne.b32 p, %5, 0;\n"
      " @!p mov.b32 %0, 0;\n"
      " @!p mov.b32 %1, 0;\n"
      " @!p mov.b32 %2, 0;\n"
      " @!p mov.b32 %3, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750
      " @p ld.global.cg.L2::128B.v4.b32 {%0, %1, %2, %3}, [%4];}\n"
#else
      " @p ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];}\n"
#endif
      : "=r"(reinterpret_cast<uint32_t&>(r0)),
        "=r"(reinterpret_cast<uint32_t&>(r1)),
        "=r"(reinterpret_cast<uint32_t&>(r2)),
        "=r"(reinterpret_cast<uint32_t&>(r3))
      : "l"(ptr), "r"((int)guard));
}

template <typename T>
__device__ __forceinline__ void lds128(T& reg0, T& reg1, T& reg2, T& reg3,
                                       const uint32_t addr) {
  static_assert(sizeof(T) == 4, "lds128: invalid T");

  asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(reinterpret_cast<uint32_t&>(reg0)),
                 "=r"(reinterpret_cast<uint32_t&>(reg1)),
                 "=r"(reinterpret_cast<uint32_t&>(reg2)),
                 "=r"(reinterpret_cast<uint32_t&>(reg3))
               : "r"(addr));
}

template <typename T>
__device__ __forceinline__ void stg128(const T& r0, const T& r1, const T& r2,
                                       const T& r3, const void* ptr,
                                       bool guard) {
  static_assert(sizeof(T) == 4, "stg128: invalid T");

  asm volatile(
      "{.reg .pred p;\n"
      " setp.ne.b32 p, %1, 0;\n"
      " @p st.global.v4.b32 [%0], {%2, %3, %4, %5};}\n"
      :
      : "l"(ptr), "r"((int)guard), "r"(reinterpret_cast<const uint32_t&>(r0)),
        "r"(reinterpret_cast<const uint32_t&>(r1)),
        "r"(reinterpret_cast<const uint32_t&>(r2)),
        "r"(reinterpret_cast<const uint32_t&>(r3)));
}

template <typename T>
__device__ __forceinline__ void ldsm_4(T& r0, T& r1, T& r2, T& r3,
                                       const uint32_t& addr) {
  static_assert(sizeof(T) == 4, "ldsm_4: invalid T");
#if (__CUDA_ARCH__ >= 750) && (__CUDACC_VER_MAJOR__ >= 11)
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(reinterpret_cast<uint32_t&>(r0)),
        "=r"(reinterpret_cast<uint32_t&>(r1)),
        "=r"(reinterpret_cast<uint32_t&>(r2)),
        "=r"(reinterpret_cast<uint32_t&>(r3))
      : "r"(addr));
#endif
}

template <typename FType>
__device__ __forceinline__ void hmma16816_f32(float (&d)[4],
                                              const uint32_t (&a)[4],
                                              const uint32_t (&b)[2]);

template <>
__device__ __forceinline__ void hmma16816_f32<__half>(float (&d)[4],
                                                      const uint32_t (&a)[4],
                                                      const uint32_t (&b)[2]) {
#if (__CUDA_ARCH__ >= 800) && (__CUDACC_VER_MAJOR__ >= 11)
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

template <>
__device__ __forceinline__ void hmma16816_f32<__nv_bfloat16>(
    float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
#if (__CUDA_ARCH__ >= 800) && (__CUDACC_VER_MAJOR__ >= 11)
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

template <int SIZE_IN_BYTES>
__device__ __forceinline__ void cp_async(const uint32_t smem_addr,
                                         const void* gmem_ptr,
                                         const int src_in_bytes, bool guard) {
  static_assert(
      (SIZE_IN_BYTES == 4 || SIZE_IN_BYTES == 8 || SIZE_IN_BYTES == 16),
      "Size is not supported");
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDA_ARCH__ >= 800
  asm volatile(
      "{.reg.pred p;\n"
      " setp.ne.b32 p, %4, 0;\n"
  #if __CUDACC_VER_MINOR__ >= 4
      " @p cp.async.cg.shared.global.L2::256B [%0], [%1], %2, %3;}\n"
  #else
      " @p cp.async.cg.shared.global [%0], [%1], %2, %3;}\n"
  #endif
      ::"r"(smem_addr),
      "l"(gmem_ptr), "n"(SIZE_IN_BYTES), "r"(src_in_bytes), "r"((int)guard));
#endif
}

template <int SIZE_IN_BYTES>
__device__ __forceinline__ void cp_async_ca(const uint32_t smem_addr,
                                            const void* gmem_ptr,
                                            const int src_in_bytes,
                                            bool guard) {
  static_assert(
      (SIZE_IN_BYTES == 4 || SIZE_IN_BYTES == 8 || SIZE_IN_BYTES == 16),
      "Size is not supported");
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDA_ARCH__ >= 800
  asm volatile(
      "{.reg.pred p;\n"
      " setp.ne.b32 p, %4, 0;\n"
  #if __CUDACC_VER_MINOR__ >= 4
      " @p cp.async.ca.shared.global.L2::256B [%0], [%1], %2, %3;}\n"
  #else
      " @p cp.async.ca.shared.global [%0], [%1], %2, %3;}\n"
  #endif
      ::"r"(smem_addr),
      "l"(gmem_ptr), "n"(SIZE_IN_BYTES), "r"(src_in_bytes), "r"((int)guard));
#endif
}

__device__ __forceinline__ void cp_async_commit_group() {
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n");
#endif
}

template <int N>
__device__ __forceinline__ void cp_asyc_wait_group() {
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;\n" : : "n"(N));
#endif
}

template <typename T>
__device__ __forceinline__ void cvt_8bx4_to_16bx4_bias128(const uint32_t& idata,
                                                          T* fdata);

template <>
// fast conversion: 4xuint8 to 4xhalf, subtracting bias = 128
__device__ __forceinline__ void cvt_8bx4_to_16bx4_bias128<__half2>(
    const uint32_t& idata, __half2* fdata) {
  uint32_t i10, i32;
  asm volatile(
      "prmt.b32 %0, %2, 0x64, 0x4140;"
      "prmt.b32 %1, %2, 0x64, 0x4342;"
      : "=r"(i10), "=r"(i32)
      : "r"(idata));

  static constexpr uint32_t MAGIC_NUM = 0x64806480;
  fdata[0] = __hsub2(reinterpret_cast<const __half2&>(i10),
                     reinterpret_cast<const __half2&>(MAGIC_NUM));
  fdata[1] = __hsub2(reinterpret_cast<const __half2&>(i32),
                     reinterpret_cast<const __half2&>(MAGIC_NUM));
}

template <>
// fast conversion: 4xuint8 to 4xbfloat16, subtracting bias = 128
// reference from marlin fast implementation
__device__ __forceinline__ void cvt_8bx4_to_16bx4_bias128<__nv_bfloat162>(
    const uint32_t& idata, __nv_bfloat162* fdata) {
  float fp32_imd[4];
  uint32_t* fp32_imd_casted = reinterpret_cast<uint32_t*>(fp32_imd);
  asm volatile(
      "prmt.b32 %0, %4, 0x4B000000, 0x7650;"
      "prmt.b32 %1, %4, 0x4B000000, 0x7651;"
      "prmt.b32 %2, %4, 0x4B000000, 0x7652;"
      "prmt.b32 %3, %4, 0x4B000000, 0x7653;"
      : "=r"(fp32_imd_casted[0]), "=r"(fp32_imd_casted[1]),
        "=r"(fp32_imd_casted[2]), "=r"(fp32_imd_casted[3])
      : "r"(idata));

  fp32_imd[0] -= 8388736.f;
  fp32_imd[1] -= 8388736.f;
  fp32_imd[2] -= 8388736.f;
  fp32_imd[3] -= 8388736.f;

  uint32_t* bf16_res = reinterpret_cast<uint32_t*>(fdata);
  asm volatile(
      "prmt.b32 %0, %2, %3, 0x7632;"
      "prmt.b32 %1, %4, %5, 0x7632;"
      : "=r"(bf16_res[0]), "=r"(bf16_res[1])
      : "r"(fp32_imd_casted[0]), "r"(fp32_imd_casted[1]),
        "r"(fp32_imd_casted[2]), "r"(fp32_imd_casted[3]));
}

static __device__ nv_bfloat162 inline num2num2(const nv_bfloat16 x) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  assert(false);
#else
  return __bfloat162bfloat162(x);
#endif
  __builtin_unreachable();  // Suppress missing return statement warning
}

static __device__ half2 inline num2num2(const half x) {
  return __half2half2(x);
}

}  // namespace allspark
