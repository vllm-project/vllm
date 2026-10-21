// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Kernels and shared helpers of the "skinny" (wave-split-K) GEMMs. The host
// launchers live in one translation unit per kernel family, see
// skinny_gemms_{llmm1,wvsplitk,wvsplitkrc,wvsplitkq}.cu.

#pragma once

// <torch/all.h> is not needed here: it also pulls in autograd, nn and jit,
// which is several seconds of frontend for every skinny GEMM translation unit.
#include <torch/types.h>            // torch::Tensor, torch::empty, AT_DISPATCH
#include <ATen/cuda/CUDAContext.h>  // at::cuda::getCurrentCUDAStream
#include <c10/cuda/CUDAGuard.h>     // at::cuda::OptionalCUDAGuard

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>
#include <algorithm>
#include <map>
#include <mutex>
#include <utility>

#include "core/cuda_compat.h"
#include "core/utils.hpp"

// TODO(rasmith): The kernels in this file are susceptible to integer overflow
// issues, do not take strides, and are unable to handle PyTorch tensors that
// return is_contiguous() as False (the tensors may actually be contiguous
// in memory).
//
// However, it may be possible to fix these kernels to handle both issues.

// Arch macros. Guarded with #ifndef because this is now a header: a second copy
// of any of them (e.g. from another ROCm kernel header) must not warn or
// redefine.
#if !defined(__HIP__GFX9__) && defined(__HIPCC__) && \
    (defined(__gfx90a__) || defined(__gfx942__) || defined(__gfx950__))
  #define __HIP__GFX9__
#endif

// Combined RDNA macro (gfx11 + gfx12) - both use 32-wide wavefronts
#if !defined(__HIP__GFX1X__) && (defined(__GFX11__) || defined(__GFX12__))
  #define __HIP__GFX1X__
#endif

#if !defined(__HIP__MI3XX__) && defined(__HIPCC__) && \
    (defined(__gfx942__) || defined(__gfx950__))
  #define __HIP__MI3XX__
#endif

#ifndef LDS_SIZE
  #if defined(__gfx950__)
    #define LDS_SIZE 160 * 1024
  #else
    #define LDS_SIZE 64 * 1024
  #endif
#endif

// Traps unconditionally, so it does not depend on NDEBUG or on <assert.h> being
// included with a particular state (this used to #undef/#define NDEBUG around
// the include, which is order-dependent now that the file is a header). Note
// this trades the device-assert message (file/line) for a bare trap: these
// stubs are never launched, they only exist so the arch-specific #if in the
// launchers has something to call.
#define UNREACHABLE_CODE __builtin_trap()

// The helpers below keep function-local statics (device properties); because
// they are inline, the C++ rules make that one object for the whole program
// instead of one per translation unit, which is what the callers want.
inline int get_lds_size() {
  static const int result = [] {
    const auto* dprops = at::cuda::getCurrentDeviceProperties();
    const std::string device_arch = dprops->gcnArchName;
    return device_arch.find("gfx95") == std::string::npos ? 64 * 1024
                                                          : 160 * 1024;
  }();
  return result;
}

inline bool on_gfx1x() {
  static const bool result = [] {
    const auto* dprops = at::cuda::getCurrentDeviceProperties();
    const std::string device_arch = dprops->gcnArchName;
    return device_arch.find("gfx11") != std::string::npos ||
           device_arch.find("gfx12") != std::string::npos;
  }();
  return result;
}

inline bool on_gfx12() {
  static const bool result = [] {
    const auto* dprops = at::cuda::getCurrentDeviceProperties();
    const std::string device_arch = dprops->gcnArchName;
    return device_arch.find("gfx12") != std::string::npos;
  }();
  return result;
}

inline bool on_gfx1151() {
  static const bool result = [] {
    const auto* dprops = at::cuda::getCurrentDeviceProperties();
    const std::string device_arch = dprops->gcnArchName;
    return device_arch.find("gfx1151") != std::string::npos;
  }();
  return result;
}

template <typename T>
struct scalar {};

template <typename T>
struct scalar2 {};

template <typename T>
__device__ __forceinline__ float2 __s22float2(T v);

template <typename T>
__device__ __forceinline__ T __float2s(float v);

template <typename T>
__device__ __forceinline__ T __float22s2_rn(float2 v);

// Definitions and cvt functions for fp16
template <>
struct scalar<c10::Half> {
  using type = half;
};

template <>
struct scalar2<c10::Half> {
  using type = __half2;
};

template <>
__device__ __forceinline__ half __float2s(float v) {
  return __float2half(v);
}

template <>
__device__ __forceinline__ float2 __s22float2(__half2 v) {
  return __half22float2(v);
}

template <>
__device__ __forceinline__ __half2 __float22s2_rn(float2 v) {
  return __float22half2_rn(v);
}

// Definitions and cvt functions for bf16
template <>
struct scalar<c10::BFloat16> {
  using type = __hip_bfloat16;
};

template <>
struct scalar2<c10::BFloat16> {
  using type = __hip_bfloat162;
};

template <>
__device__ __forceinline__ __hip_bfloat16 __float2s(float v) {
  return __float2bfloat16(v);
}

template <>
__device__ __forceinline__ float2 __s22float2(__hip_bfloat162 v) {
  return __bfloat1622float2(v);
}

template <>
__device__ __forceinline__ __hip_bfloat162 __float22s2_rn(float2 v) {
  return __float22bfloat162_rn(v);
}

template <typename T>
__device__ __forceinline__ T loadnt(T* addr) {
  return __builtin_nontemporal_load(addr);
}

__device__ __forceinline__ float4 load_ntmprl(const float4* addr) {
  auto addr_alias = reinterpret_cast<const float*>(addr);
  auto dat0 = loadnt(addr_alias);
  auto dat1 = loadnt(addr_alias + 1);
  auto dat2 = loadnt(addr_alias + 2);
  auto dat3 = loadnt(addr_alias + 3);
  return make_float4(dat0, dat1, dat2, dat3);
}

// TBlock fetches entire rows of A, and entire col of B (K dimension); assume
// N=1 for time being grid is M/A_NUM_ROWS blocks
template <typename scalar_t, int NUM_A_ROWS_PER_BLOCK>
__global__ void LLGemm1_kernel(const scalar_t* in_a, const scalar_t* in_b,
                               scalar_t* out_c, const int K) {
  using scalar2_t = typename scalar2<scalar_t>::type;
  auto af4 = reinterpret_cast<const float4*>(in_a);
  auto bf4 = reinterpret_cast<const scalar2_t*>(in_b);
  auto c = reinterpret_cast<scalar2_t*>(out_c);
  __shared__ float red_smem[NUM_A_ROWS_PER_BLOCK][WARP_SIZE];
  const int row_addr = blockIdx.x * NUM_A_ROWS_PER_BLOCK * K / 8;
  const int threadid = threadIdx.x;
  const int warp = threadIdx.x / WARP_SIZE;
  const int lane = threadIdx.x % WARP_SIZE;
  const int num_warps = blockDim.x / WARP_SIZE;
  const int qwarpid = threadid / 16;
  const int qthreadid = threadid % 16;
  float4 rowA_elem4[NUM_A_ROWS_PER_BLOCK];
  scalar2_t colB_elem4x, colB_elem4y, colB_elem4z, colB_elem4w;
  float acc[NUM_A_ROWS_PER_BLOCK];
  scalar2_t acch2;
  scalar2_t oval;

  // As we later use warp shuffle operations, we may have more threads in the
  // block than the actual available data, hence the if guard here.
  if (threadid * 8 < K) {
#pragma unroll
    for (int i = 0; i < NUM_A_ROWS_PER_BLOCK; i++) {
      // rowA_elem4[i] holds 8 * half numbers seen as a single float4.
      rowA_elem4[i] = load_ntmprl(&af4[row_addr + threadid + K / 8 * i]);
    }
    colB_elem4x = bf4[threadid * 4 + 0];
    colB_elem4y = bf4[threadid * 4 + 1];
    colB_elem4z = bf4[threadid * 4 + 2];
    colB_elem4w = bf4[threadid * 4 + 3];
  }

  scalar2_t Af2;
  float2 S;

  auto Ah2ptr = reinterpret_cast<scalar2_t*>(&rowA_elem4);
  scalar2_t* ah2lptr;

#pragma unroll
  for (int i = 0; i < NUM_A_ROWS_PER_BLOCK; i++) {
    // Multiply-add on 8 scalar_t.
    ah2lptr = Ah2ptr + i * 4;
    Af2 = *(ah2lptr);
    acch2 = __hmul2(Af2, colB_elem4x);
    Af2 = *(ah2lptr + 1);
    acch2 = __hfma2(Af2, colB_elem4y, acch2);
    Af2 = *(ah2lptr + 2);
    acch2 = __hfma2(Af2, colB_elem4z, acch2);
    Af2 = *(ah2lptr + 3);
    acch2 = __hfma2(Af2, colB_elem4w, acch2);
    S = __s22float2(acch2);

    // See comment above concerning the if guard.
    acc[i] = (threadid * 8 < K ? S.x + S.y : 0.f);
  }

// all reduce across warp.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
#pragma unroll
    for (int i = 0; i < NUM_A_ROWS_PER_BLOCK; i++) {
      acc[i] += __shfl_xor(acc[i], mask);
    }
  }

  // Warp leaders store the data to shared memory.
  if (lane < NUM_A_ROWS_PER_BLOCK) {
    red_smem[lane][warp] = acc[lane];
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  if (qwarpid < NUM_A_ROWS_PER_BLOCK) {
    acc[qwarpid] = qthreadid < num_warps ? red_smem[qwarpid][qthreadid] : 0.f;
#pragma unroll
    for (int mask = 16 / 2; mask >= 1; mask /= 2) {
      acc[qwarpid] += __shfl_xor(acc[qwarpid], mask);
    }
    float oval2 = __shfl_xor(acc[qwarpid], 16);

    if (lane % 32 == 0) {
      oval = __float22s2_rn<scalar2_t>(make_float2(acc[qwarpid], oval2));
      c[blockIdx.x * NUM_A_ROWS_PER_BLOCK / 2 + qwarpid / 2] = oval;
    }
  }
}

#if defined(__HIP__GFX9__) && !defined(__HIP__GFX1X__)
  #define DOT2C(V0, V2, V3)                                          \
    if constexpr (std::is_same_v<scalar_t, half>) {                  \
      asm("v_dot2c_f32_f16 %0, %2, %3"                               \
          : "=v"(V0)                                                 \
          : "0"(V0), "v"(V2), "v"(V3));                              \
    } else if constexpr (std::is_same_v<scalar_t, __hip_bfloat16>) { \
      float2 s = __bfloat1622float2(*((__hip_bfloat162*)(&(V2)))) *  \
                 __bfloat1622float2(*((__hip_bfloat162*)(&(V3))));   \
      V0 += (s.x + s.y);                                             \
    }
#elif defined(__HIP__GFX1X__)
  // gfx1x: v_dot2_f32_f16 (VOP3-P, dot10-insts, available on gfx11+gfx12)
  #define DOT2C(V0, V2, V3)                                               \
    if constexpr (std::is_same_v<scalar_t, half>) {                       \
      asm("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(V0) : "v"(V2), "v"(V3)); \
    } else if constexpr (std::is_same_v<scalar_t, __hip_bfloat16>) {      \
      float2 s = __bfloat1622float2(*((__hip_bfloat162*)(&(V2)))) *       \
                 __bfloat1622float2(*((__hip_bfloat162*)(&(V3))));        \
      V0 += (s.x + s.y);                                                  \
    }
#endif

// To avoid LLVM silently upcasting to double
__device__ inline unsigned int min__(uint32_t a, uint32_t b) {
  return min(a, b);
}

#if defined(__HIP__GFX9__) || defined(__HIP__GFX1X__)
// This version targets cases where A[] fits LDS capacity
template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK,
          int UNRL, int N>
__global__ void __launch_bounds__(WvPrGrp* THRDS)
    wvSplitK_hf_sml_(const int K, const int Kbp, const int Kap, const int M,
                     const int Bx, const int By, const scalar_t* B,
                     const scalar_t* __restrict__ A,
                     const scalar_t* __restrict__ BIAS, scalar_t* C,
                     const int _WvPrGrp, const int CuCount) {
  constexpr int max_lds_len = LDS_SIZE / 2;
  #if defined(__HIP__MI3XX__)
  constexpr bool use_mfma = (std::is_same_v<scalar_t, __hip_bfloat16>);
  #else
  constexpr bool use_mfma = false;
  #endif
  using scalar8 =
      __attribute__((__vector_size__((A_CHUNK / 2) * sizeof(float)))) float;
  using half4 =
      __attribute__((__vector_size__((A_CHUNK / 2) * sizeof(__bf16)))) __bf16;
  union bigType {
    scalar_t h[A_CHUNK];
    float f[A_CHUNK / 2];
    float2 f2[A_CHUNK / 4];
    double d[A_CHUNK / 4];
    half4 h4[A_CHUNK / 4];
    scalar8 h8;
  };

  //----------------------------------------------------
  // Reserving 64/160 KB of LDS to have 1 WG / CU
  // Goal is to bring the activation matrix A to the LDS
  // and use it across the lifetime of the work group
  // TODO: When activation matrix is larger than 64 KB
  //	     then this is not going to work!
  //----------------------------------------------------
  __shared__ scalar_t s[max_lds_len];

  //----------------------------------------------------
  // Fetch the activation matrix to LDS
  // Loop iteration:
  // - Each thread (lane) is fetching 8 elements (A_Chunk)
  // - Each wave will fetch 64*8=> 512 elements
  // - Each WG will fetch 512 * 16 => 8K elements
  // - Then the WG will move to another 8 K elements
  // TODO: Logic below will only work when K is multiple of 8
  //----------------------------------------------------
  for (uint32_t k = (threadIdx.y * THRDS + threadIdx.x) * A_CHUNK;
       k < min__(Kap * N, max_lds_len); k += THRDS * WvPrGrp * A_CHUNK) {
  #if defined(__gfx950__)
    __builtin_amdgcn_global_load_lds((int*)(&A[k]), (int*)(&s[k]), 16, 0, 0);
  #else
    *((bigType*)(&s[k])) = *((bigType*)(&A[k]));
  #endif
  }
  __syncthreads();

  if (threadIdx.y >= _WvPrGrp) return;

  uint32_t m = (blockIdx.x * _WvPrGrp + (threadIdx.y % _WvPrGrp)) * YTILE;

  //----------------------------------------------------
  // Each wave works on a single column of weight matrix.
  // There are 16 waves per WG, and hence, each WG is
  // working on 16 columns of weight matrix. Moreover,
  // we tile in column direction by YTILE, so when YTILE=1
  // the above math is right, however, when YTILE=2 then
  // each wave  will be working on 2 columns and WG will
  // be working on 32 columns.
  //
  // Top level loop that makes WGs persistent!
  // - WGs iterates across columns of weight matrix
  // - Each wave within WG works on a given column(s)
  // - After completing first set of columns, WGs start
  //   working on the next set of available columns
  //----------------------------------------------------
  while (m < M) {
    //----------------------------------------------------
    // 'sum' accumulates the matrix A x B computation
    // split across 64 lanes.
    //
    // YTILE represents how many column of weight matrix
    // are being worked on by each wave.
    //----------------------------------------------------
    float sum[N][YTILE] = {};
    scalar8 sum4[N][YTILE] = {};

    for (uint32_t k1 = 0; k1 < K; k1 += THRDS * A_CHUNK * UNRL) {
      bigType bigA[N][UNRL] = {};
      bigType bigB[YTILE][UNRL];
      // Fetch the weight matrix from memory!
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        const scalar_t* B_ = &B[min__(k_, K - A_CHUNK)];
        for (int y = 0; y < YTILE; y++)
          bigB[y][k2].h8 = (loadnt((scalar8*)(&B_[min__(y + m, M - 1) * Kbp])));
      }

      // Fetch activation matrix from either just LDS or from both LDS / memory
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;
        for (int n = 0; n < N; n++) {
          bigA[n][k2] = *((const bigType*)(&(s[k_ + Kap * n])));
        }
      }

      // Do the matrix multiplication in interleaved manner
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        for (uint32_t n = 0; n < N; n++) {
          for (int y = 0; y < YTILE; y++) {
            if constexpr (!use_mfma)
              for (uint32_t b = 0; b < A_CHUNK / 2; b++) {
                DOT2C(sum[n][y], bigA[n][k2].f[b], bigB[y][k2].f[b])
              }
            else
              for (uint32_t b = 0; b < A_CHUNK / 4; b++)
                sum4[n][y] = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(
                    bigA[n][k2].h4[b], bigB[y][k2].h4[b], sum4[n][y], 0, 0, 0);
          }
        }
      }
    }
    __builtin_amdgcn_sched_barrier(0);
    //----------------------------------------------------
    // Final reduction step using shuffle
    //----------------------------------------------------
    if constexpr (!use_mfma) {
      for (int n = 0; n < N; n++) {
        for (int y = 0; y < YTILE; y++) {
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x118, 0xf, 0xf,
                                                1);  // row_shr8
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x114, 0xf, 0xf,
                                                1);  // row_shr4
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x112, 0xf, 0xf,
                                                1);  // row_shr2
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x111, 0xf, 0xf,
                                                1);  // row_shr1
  #if defined(__HIP__GFX9__)
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x142, 0xf, 0xf,
                                                1);  // ROW_BCAST15
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x143, 0xf, 0xf,
                                                1);  // ROW_BCAST31
  #else
          sum[n][y] += __shfl_xor(sum[n][y], 16);
  #endif
        }
      }

      if (threadIdx.x == (THRDS - 1)) {
        scalar_t biases[N][YTILE] = {};
        if (BIAS)
          for (int n = 0; n < N; n++) {
            for (int y = 0; y < YTILE; y++) {
              biases[n][y] = BIAS[(m + y) % Bx + (n % By) * Bx];
            }
          }
        for (int n = 0; n < N; n++) {
          for (int y = 0; y < YTILE; y++) {
            if constexpr (std::is_same_v<scalar_t, half>) {
              sum[n][y] += __half2float(biases[n][y]);
            } else if constexpr (std::is_same_v<scalar_t, __hip_bfloat16>) {
              sum[n][y] += __bfloat162float(biases[n][y]);
            }
            C[m + y + n * M] = __float2s<scalar_t>(sum[n][y]);
          }
        }
      }
    } else {
  #ifdef __HIP__GFX9__
    #pragma unroll
      for (int n = 0; n < N; n++) {
    #pragma unroll
        for (int y = 0; y < YTILE; y++) {
          /*float accm1 = 0;
           for (int i=0; i<64; i++)
              accm1 += __shfl(sum4[n][y][i%4], i);
          sum4[n][y][0] = accm1;*/
          float accm = sum4[n][y][0];
          accm += __builtin_amdgcn_mov_dpp(sum4[n][y][1], 0x101, 0xf, 0xf,
                                           1);  // row_shl1
          accm += __builtin_amdgcn_mov_dpp(sum4[n][y][2], 0x102, 0xf, 0xf,
                                           1);  // row_shl2
          accm += __builtin_amdgcn_mov_dpp(sum4[n][y][3], 0x103, 0xf, 0xf,
                                           1);  // row_shl3
          accm += __builtin_amdgcn_mov_dpp(accm, 0x104, 0xf, 0xf,
                                           1);  // row_shl4
          accm += __builtin_amdgcn_mov_dpp(accm, 0x108, 0xf, 0xf,
                                           1);  // row_shl8
          accm = __builtin_amdgcn_mov_dpp(accm, 0x11f, 0xf, 0xf,
                                          1);  // row_shr15
          accm += __builtin_amdgcn_mov_dpp(accm, 0x142, 0xf, 0xf,
                                           1);  // ROW_BCAST15
          accm += __builtin_amdgcn_mov_dpp(accm, 0x143, 0xf, 0xf,
                                           1);  // ROW_BCAST31

          sum4[n][y][0] = accm;
        }
      }
      if (threadIdx.x == (THRDS - 1)) {
        scalar_t biases[N][YTILE] = {};
        if (BIAS)
          for (int n = 0; n < N; n++) {
            for (int y = 0; y < YTILE; y++) {
              biases[n][y] = BIAS[(m + y) % Bx + (n % By) * Bx];
            }
          }
        for (int n = 0; n < N; n++) {
          for (int y = 0; y < YTILE; y++) {
            sum4[n][y][0] += __bfloat162float(biases[n][y]);
            C[m + y + n * M] = __float2bfloat16(sum4[n][y][0]);
          }
        }
      }
  #endif  // __HIP__GFX9__ (MFMA path)
    }
    m += CuCount * _WvPrGrp * YTILE;
  }
}
#else
template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK,
          int UNRL, int N>
__global__ void wvSplitK_hf_sml_(const int K, const int Kbp, const int Kap,
                                 const int M, const int Bx, const int By,
                                 const scalar_t* B,
                                 const scalar_t* __restrict__ A,
                                 const scalar_t* __restrict__ BIAS, scalar_t* C,
                                 const int _WvPrGrp, const int CuCount) {
  UNREACHABLE_CODE;
}
#endif

#if defined(__HIP__GFX9__) || defined(__HIP__GFX1X__)
// This version targets cases where A[] marginally exceeds LDS capacity
template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK,
          int UNRL, int N>
__global__ void __launch_bounds__(WvPrGrp* THRDS)
    wvSplitK_hf_(const int K, const int Kbp, const int Kap, const int M,
                 const int Bx, const int By, const scalar_t* B,
                 const scalar_t* __restrict__ A,
                 const scalar_t* __restrict__ BIAS, scalar_t* C,
                 const int _WvPrGrp, const int CuCount) {
  constexpr int max_lds_len = LDS_SIZE / 2;
  #if defined(__HIP__MI3XX__)
  constexpr bool use_mfma = (std::is_same_v<scalar_t, __hip_bfloat16>);
  #else
  constexpr bool use_mfma = false;
  #endif

  using scalar8 =
      __attribute__((__vector_size__((A_CHUNK / 2) * sizeof(float)))) float;
  using half4 =
      __attribute__((__vector_size__((A_CHUNK / 2) * sizeof(__bf16)))) __bf16;
  union bigType {
    scalar_t h[A_CHUNK];
    float f[A_CHUNK / 2];
    float2 f2[A_CHUNK / 4];
    double d[A_CHUNK / 4];
    half4 h4[A_CHUNK / 4];
    scalar8 h8;
  };

  __shared__ scalar_t s[max_lds_len];

  //----------------------------------------------------
  // Computation of columns that need to be committed to memory!
  //----------------------------------------------------
  uint32_t commitColumn[YTILE];
  for (uint32_t i = 0; i < YTILE; i++) {
    commitColumn[i] = 1;
  }

  uint32_t m = (blockIdx.x * _WvPrGrp + threadIdx.y) * YTILE;

  // Check whether there will be fragmentation!
  // This will happen only for the last wave!
  if (m < M && (m + YTILE) >= M) {
    uint32_t startColumn = M - YTILE;
    for (uint32_t i = 0; i < (m - startColumn); i++) {
      commitColumn[i] = 0;
    }
    m = startColumn;
  }

  for (uint32_t k = (threadIdx.y * THRDS + threadIdx.x) * A_CHUNK;
       k < min__(Kap * N, max_lds_len); k += THRDS * WvPrGrp * A_CHUNK) {
  #if defined(__gfx950__)
    __builtin_amdgcn_global_load_lds((int*)(&A[k]), (int*)(&s[k]), 16, 0, 0);
  #else
    *((bigType*)(&s[k])) = *((bigType*)(&A[k]));
  #endif
  }

  __syncthreads();

  if (threadIdx.y >= _WvPrGrp) return;

  while (m < M) {
    float sum[N][YTILE] = {};
    scalar8 sum4[N][YTILE] = {};

    for (uint32_t k1 = 0; k1 < K; k1 += THRDS * A_CHUNK * UNRL) {
      bigType bigA[N][UNRL] = {};
      bigType bigB[YTILE][UNRL];
      // Fetch the weight matrix from memory!
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        const scalar_t* B_ = &B[min__(k_, K - A_CHUNK)];
        for (int y = 0; y < YTILE; y++)
          bigB[y][k2].h8 = (loadnt((scalar8*)(&B_[min__(y + m, M - 1) * Kbp])));
      }

      // Fetch activation matrix from either just LDS or from both LDS / memory
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;
        for (int n = 0; n < N; n++) {
          if (k_ + Kap * n < max_lds_len)
            bigA[n][k2] = *((const bigType*)(&(s[k_ + Kap * n])));
          else
            bigA[n][k2] = *((const bigType*)(&(A[k_ + Kap * n])));
        }
      }

      // Do the matrix multiplication in interleaved manner
      for (uint32_t n = 0; n < N; n++) {
        for (uint32_t k2 = 0; k2 < UNRL; k2++) {
          for (int y = 0; y < YTILE; y++) {
            if constexpr (!use_mfma)
              for (uint32_t b = 0; b < A_CHUNK / 2; b++) {
                DOT2C(sum[n][y], bigA[n][k2].f[b], bigB[y][k2].f[b])
              }
            else
              for (uint32_t b = 0; b < A_CHUNK / 4; b++)
                sum4[n][y] = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(
                    bigA[n][k2].h4[b], bigB[y][k2].h4[b], sum4[n][y], 0, 0, 0);
          }
        }
      }
    }

    //----------------------------------------------------
    // Final reduction step using shuffle
    //----------------------------------------------------
    if constexpr (!use_mfma) {
      for (int n = 0; n < N; n++) {
        for (int y = 0; y < YTILE; y++) {
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x118, 0xf, 0xf,
                                                1);  // row_shr8
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x114, 0xf, 0xf,
                                                1);  // row_shr4
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x112, 0xf, 0xf,
                                                1);  // row_shr2
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x111, 0xf, 0xf,
                                                1);  // row_shr1
  #if defined(__HIP__GFX9__)
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x142, 0xf, 0xf,
                                                1);  // ROW_BCAST15
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x143, 0xf, 0xf,
                                                1);  // ROW_BCAST31
  #else
          sum[n][y] += __shfl_xor(sum[n][y], 16);
  #endif
        }
      }

      if (threadIdx.x == (THRDS - 1)) {
        scalar_t biases[N][YTILE] = {};
        if (BIAS)
          for (int n = 0; n < N; n++) {
            for (int y = 0; y < YTILE; y++) {
              biases[n][y] = BIAS[(m + y) % Bx + (n % By) * Bx];
            }
          }
        for (int n = 0; n < N; n++) {
          for (int y = 0; y < YTILE; y++) {
            if (commitColumn[y]) {
              if constexpr (std::is_same_v<scalar_t, half>) {
                sum[n][y] += __half2float(biases[n][y]);
              } else if constexpr (std::is_same_v<scalar_t, __hip_bfloat16>) {
                sum[n][y] += __bfloat162float(biases[n][y]);
              }
              C[m + y + n * M] = __float2s<scalar_t>(sum[n][y]);
            }
          }
        }
      }
    } else {
  #ifdef __HIP__GFX9__
    #pragma unroll
      for (int n = 0; n < N; n++) {
    #pragma unroll
        for (int y = 0; y < YTILE; y++) {
          // float accm1 = 0;
          // for (int i=0; i<64; i++)
          //    accm1 += __shfl(sum4[n][y][i%4], i);
          float accm = sum4[n][y][0];
          accm += __builtin_amdgcn_mov_dpp(sum4[n][y][1], 0x101, 0xf, 0xf,
                                           1);  // row_shl1
          accm += __builtin_amdgcn_mov_dpp(sum4[n][y][2], 0x102, 0xf, 0xf,
                                           1);  // row_shl2
          accm += __builtin_amdgcn_mov_dpp(sum4[n][y][3], 0x103, 0xf, 0xf,
                                           1);  // row_shl3
          accm += __builtin_amdgcn_mov_dpp(accm, 0x104, 0xf, 0xf,
                                           1);  // row_shl4
          accm += __builtin_amdgcn_mov_dpp(accm, 0x108, 0xf, 0xf,
                                           1);  // row_shl8
          accm = __builtin_amdgcn_mov_dpp(accm, 0x11f, 0xf, 0xf,
                                          1);  // row_shr15
          accm += __builtin_amdgcn_mov_dpp(accm, 0x142, 0xf, 0xf,
                                           1);  // ROW_BCAST15
          accm += __builtin_amdgcn_mov_dpp(accm, 0x143, 0xf, 0xf,
                                           1);  // ROW_BCAST31
          sum4[n][y][0] = accm;
        }
      }
      if (threadIdx.x == (THRDS - 1)) {
        scalar_t biases[N][YTILE] = {};
        if (BIAS)
          for (int n = 0; n < N; n++) {
            for (int y = 0; y < YTILE; y++) {
              biases[n][y] = BIAS[(m + y) % Bx + (n % By) * Bx];
            }
          }
        for (int n = 0; n < N; n++) {
          for (int y = 0; y < YTILE; y++) {
            if (commitColumn[y]) {
              sum4[n][y][0] += __bfloat162float(biases[n][y]);
              C[m + y + n * M] = __float2bfloat16(sum4[n][y][0]);
            }
          }
        }
      }
  #endif  // __HIP__GFX9__ (MFMA path)
    }

    m += CuCount * _WvPrGrp * YTILE;

    // Check whether there will be fragmentation!
    // This will happen only for the last wave!
    if (m < M && (m + YTILE) >= M) {
      uint32_t startColumn = M - YTILE;
      for (uint32_t i = 0; i < (m - startColumn); i++) {
        commitColumn[i] = 0;
      }
      m = startColumn;
    }
  }
}

#else
template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK,
          int UNRL, int N>
__global__ void wvSplitK_hf_(const int K, const int Kbp, const int Kap,
                             const int M, const int Bx, const int By,
                             const scalar_t* B, const scalar_t* __restrict__ A,
                             const scalar_t* __restrict__ BIAS, scalar_t* C,
                             const int _WvPrGrp, const int CuCount) {
  UNREACHABLE_CODE;
}
#endif

#if defined(__HIP__GFX9__) || defined(__HIP__GFX1X__)
// This version targets big A[] cases, where it is much larger than LDS capacity
template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK,
          int UNRL, int N>
__global__ void __launch_bounds__(WvPrGrp* THRDS)
    wvSplitK_hf_big_(const int K, const int Kbp, const int Kap, const int M,
                     const int Bx, const int By, const scalar_t* B,
                     const scalar_t* __restrict__ A,
                     const scalar_t* __restrict__ BIAS, scalar_t* C,
                     const int _WvPrGrp, const int CuCount) {
  constexpr int max_lds_len = LDS_SIZE / 2;
  #if defined(__HIP__MI3XX__)
  constexpr bool use_mfma = (std::is_same_v<scalar_t, __hip_bfloat16>);
  #else
  constexpr bool use_mfma = false;
  #endif

  using scalar8 =
      __attribute__((__vector_size__((A_CHUNK / 2) * sizeof(float)))) float;
  using half4 =
      __attribute__((__vector_size__((A_CHUNK / 2) * sizeof(__bf16)))) __bf16;
  union bigType {
    scalar_t h[A_CHUNK];
    float f[A_CHUNK / 2];
    float2 f2[A_CHUNK / 4];
    double d[A_CHUNK / 4];
    half4 h4[A_CHUNK / 4];
    scalar8 h8;
  };

  //----------------------------------------------------
  // Reserving 64/160 KB of LDS to have 1 WG / CU
  // Goal is to bring the activation matrix A to the LDS
  // and use it across the lifetime of the work group
  // TODO: When activation matrix is larger than 64 KB
  //	     then this is not going to work!
  //----------------------------------------------------
  __shared__ scalar_t s[max_lds_len];

  //----------------------------------------------------
  // Computation of columns that need to be committed to memory!
  //----------------------------------------------------
  uint32_t commitColumn[YTILE];
  for (uint32_t i = 0; i < YTILE; i++) {
    commitColumn[i] = 1;
  }

  // int _WvPrGrp = mindiv(N, CuCount * YTILE, WvPrGrp);
  if (threadIdx.y >= _WvPrGrp) return;

  //----------------------------------------------------
  // Indexing function into the column of weight matrix B
  // Algorithm does 64 lane k-splitting / wave and uses
  // WG ID and Thread ID to find the index.
  //----------------------------------------------------
  uint32_t m = (blockIdx.x * _WvPrGrp + threadIdx.y) * YTILE;

  // Check whether there will be fragmentation!
  // This will happen only for the last wave!
  if (m < M && (m + YTILE) >= M) {
    uint32_t startColumn = M - YTILE;
    for (uint32_t i = 0; i < (m - startColumn); i++) {
      commitColumn[i] = 0;
    }
    m = startColumn;
  }

  //----------------------------------------------------
  // Fetch the activation matrix to LDS
  // Loop iteration:
  // - Each thread (lane) is fetching 8 elements (A_Chunk)
  // - Each wave will fetch 64*8=> 512 elements
  // - Each WG will fetch 512 * 16 => 8K elements
  // - Then the WG will move to another 8 K elements
  // TODO: Logic below will only work when K is multiple of 8
  //----------------------------------------------------
  #define PCML
  #ifndef PCML
  for (uint32_t k = (threadIdx.y * THRDS + threadIdx.x) * A_CHUNK;
       k < min__(Kap * N, max_lds_len); k += THRDS * WvPrGrp * A_CHUNK) {
    #if defined(__gfx950__)
    __builtin_amdgcn_global_load_lds((int*)(&A[k]), (int*)(&s[k]), 16, 0, 0);
    #else
    *((bigType*)(&s[k])) = *((bigType*)(&A[k]));
    #endif
  }
  __syncthreads();
  #endif

  #define TUC (THRDS * UNRL * A_CHUNK)
  uint32_t kBase = 0;
  // find biggest k size that fits in LDS
  uint32_t kFit = (max_lds_len) / N;
  // kFit = (kFit%TWC==0) ? kFit : (kFit-kFit%TWC+TWC); //round up to multiple
  // of TUC
  kFit = (kFit % TUC == 0)
             ? kFit
             : (kFit - kFit % TUC);  // round up to multiple of TUC
  // if (kFit == 0) kFit = TUC;
  kFit = min__(kFit, Kap);

  //----------------------------------------------------
  // Each wave works on a single column of weight matrix.
  // There are 16 waves per WG, and hence, each WG is
  // working on 16 columns of weight matrix. Moreover,
  // we tile in column direction by YTILE, so when YTILE=1
  // the above math is right, however, when YTILE=2 then
  // each wave  will be working on 2 columns and WG will
  // be working on 32 columns.
  //
  // Top level loop that makes WGs persistent!
  // - WGs iterates across columns of weight matrix
  // - Each wave within WG works on a given column(s)
  // - After completing first set of columns, WGs start
  //   working on the next set of available columns
  //----------------------------------------------------
  #ifdef PCML
  int YW = (YTILE * _WvPrGrp);
  uint32_t Mrndp = (M % YW == 0) ? M : (M - M % YW + YW);
  while (m < Mrndp) {
  #else
  while (m < M) {
  #endif
    //----------------------------------------------------
    // 'sum' accumulates the matrix A x B computation
    // split across 64 lanes.
    //
    // YTILE represents how many column of weight matrix
    // are being worked on by each wave.
    //----------------------------------------------------
    float sum[N][YTILE] = {};
    scalar8 sum4[N][YTILE] = {};

    //----------------------------------------------------
    // Fetch weight matrix B in interleaved K-split!
    // - Each thread (lane) is fetching 8 elements (A_Chunk)
    // - Each wave will fetch 64*8=> 512 elements (1024B)
    // - YTILE represents the number of column being serviced
    //   by wave
    // - Loop for fetching weight matrix (B) are unrolled
    //
    // Fetch activation matrix A from LDS
    // - Loop for fetching activation matrix (A) are unrolled
    //
    // Finally, do the matrix multiplication in an unrolled
    // fashion. This provides lot of food for compiler
    // scheduling.
    //
    // TODO: Logic below will only work when K is multiple of 8
    //----------------------------------------------------
    for (uint32_t k1 = 0; k1 < K; k1 += THRDS * A_CHUNK * UNRL) {
      bigType bigA[N][UNRL] = {};
      bigType bigB[YTILE][UNRL];

  #ifdef PCML
      if ((k1 == 0) || (k1 == kBase + kFit)) {  // load next chunk of A[] to LDS
        if (k1 != 0) kBase += kFit;
        __syncthreads();
        for (uint32_t k = 0; k < kFit; k += THRDS * _WvPrGrp * A_CHUNK) {
          uint32_t kOff = k + ((threadIdx.y * THRDS + threadIdx.x) * A_CHUNK);
          if (kBase + kOff >= Kap) break;
          if (kOff >= kFit) break;
          for (uint32_t n = 0; n < N; n++) {
            uint32_t k_in = kBase + n * Kap + kOff;
            uint32_t k_ot = n * kFit + kOff;
    #if defined(__gfx950__)
            __builtin_amdgcn_global_load_lds((int*)(&A[k_in]), (int*)(&s[k_ot]),
                                             16, 0, 0);
    #else
            *((bigType*)(&s[k_ot])) = *((bigType*)(&A[k_in]));
    #endif
          }
        }
        __syncthreads();
      }
      if (m >= M) continue;
  #endif

      // Fetch the weight matrix from memory!
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        const scalar_t* B_ = &B[min__(k_, K - A_CHUNK)];
        for (int y = 0; y < YTILE; y++)
          bigB[y][k2].h8 = (loadnt((scalar8*)(&B_[min__(y + m, M - 1) * Kbp])));
      }

      // Fetch activation matrix from either just LDS or from both LDS / memory
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;
        for (int n = 0; n < N; n++) {
  #ifdef PCML
          bigA[n][k2] = *((const bigType*)(&(s[k_ - kBase + kFit * n])));
  #else
          if (k_ + Kap * n < max_lds_len)
            bigA[n][k2] = *((const bigType*)(&(s[k_ + Kap * n])));
          else
            bigA[n][k2] = *((const bigType*)(&(A[k_ + Kap * n])));
  #endif
        }
      }

      // Do the matrix multiplication in interleaved manner
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        for (uint32_t n = 0; n < N; n++) {
          for (int y = 0; y < YTILE; y++) {
            if constexpr (!use_mfma)
              for (uint32_t b = 0; b < A_CHUNK / 2; b++) {
                DOT2C(sum[n][y], bigA[n][k2].f[b], bigB[y][k2].f[b])
              }
            else
              for (uint32_t b = 0; b < A_CHUNK / 4; b++)
                sum4[n][y] = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(
                    bigA[n][k2].h4[b], bigB[y][k2].h4[b], sum4[n][y], 0, 0, 0);
          }
        }
      }
    }

  #ifdef PCML
    if (m >= M) {
      m += CuCount * _WvPrGrp * YTILE;
      kBase = 0;
      continue;
    }
  #endif

    //----------------------------------------------------
    // Final reduction step using shuffle
    //----------------------------------------------------
    if constexpr (!use_mfma) {
      for (int n = 0; n < N; n++) {
        for (int y = 0; y < YTILE; y++) {
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x118, 0xf, 0xf,
                                                1);  // row_shr8
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x114, 0xf, 0xf,
                                                1);  // row_shr4
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x112, 0xf, 0xf,
                                                1);  // row_shr2
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x111, 0xf, 0xf,
                                                1);  // row_shr1
  #if defined(__HIP__GFX9__)
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x142, 0xf, 0xf,
                                                1);  // ROW_BCAST15
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x143, 0xf, 0xf,
                                                1);  // ROW_BCAST31
  #else
          sum[n][y] += __shfl_xor(sum[n][y], 16);
  #endif
        }
      }

      if (threadIdx.x == (THRDS - 1)) {
        scalar_t biases[N][YTILE] = {};
        if (BIAS)
          for (int n = 0; n < N; n++) {
            for (int y = 0; y < YTILE; y++) {
              biases[n][y] = BIAS[(m + y) % Bx + (n % By) * Bx];
            }
          }
        for (int n = 0; n < N; n++) {
          for (int y = 0; y < YTILE; y++) {
            if (commitColumn[y]) {
              if constexpr (std::is_same_v<scalar_t, half>) {
                sum[n][y] += __half2float(biases[n][y]);
              } else if constexpr (std::is_same_v<scalar_t, __hip_bfloat16>) {
                sum[n][y] += __bfloat162float(biases[n][y]);
              }
              C[m + y + n * M] = __float2s<scalar_t>(sum[n][y]);
            }
          }
        }
      }
    } else {
  #ifdef __HIP__GFX9__
    #pragma unroll
      for (int n = 0; n < N; n++) {
    #pragma unroll
        for (int y = 0; y < YTILE; y++) {
          float accm = sum4[n][y][0];
          accm += __builtin_amdgcn_mov_dpp(sum4[n][y][1], 0x101, 0xf, 0xf,
                                           1);  // row_shl1
          accm += __builtin_amdgcn_mov_dpp(sum4[n][y][2], 0x102, 0xf, 0xf,
                                           1);  // row_shl2
          accm += __builtin_amdgcn_mov_dpp(sum4[n][y][3], 0x103, 0xf, 0xf,
                                           1);  // row_shl3
          accm += __builtin_amdgcn_mov_dpp(accm, 0x104, 0xf, 0xf,
                                           1);  // row_shl4
          accm += __builtin_amdgcn_mov_dpp(accm, 0x108, 0xf, 0xf,
                                           1);  // row_shl8
          accm = __builtin_amdgcn_mov_dpp(accm, 0x11f, 0xf, 0xf,
                                          1);  // row_shr15
          accm += __builtin_amdgcn_mov_dpp(accm, 0x142, 0xf, 0xf,
                                           1);  // ROW_BCAST15
          accm += __builtin_amdgcn_mov_dpp(accm, 0x143, 0xf, 0xf,
                                           1);  // ROW_BCAST31
          sum4[n][y][0] = accm;
        }
      }
      if (threadIdx.x == (THRDS - 1)) {
        scalar_t biases[N][YTILE] = {};
        if (BIAS)
          for (int n = 0; n < N; n++) {
            for (int y = 0; y < YTILE; y++) {
              biases[n][y] = BIAS[(m + y) % Bx + (n % By) * Bx];
            }
          }
        for (int n = 0; n < N; n++) {
          for (int y = 0; y < YTILE; y++) {
            if (commitColumn[y]) {
              sum4[n][y][0] += __bfloat162float(biases[n][y]);
              C[m + y + n * M] = __float2bfloat16(sum4[n][y][0]);
            }
          }
        }
      }
  #endif  // __HIP__GFX9__ (MFMA path)
    }

    m += CuCount * _WvPrGrp * YTILE;
    kBase = 0;

    // Check whether there will be fragmentation!
    // This will happen only for the last wave!
    if (m < M && (m + YTILE) >= M) {
      uint32_t startColumn = M - YTILE;
      for (uint32_t i = 0; i < (m - startColumn); i++) {
        commitColumn[i] = 0;
      }
      m = startColumn;
    }
  }
}
#else
template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK,
          int UNRL, int N>
__global__ void wvSplitK_hf_big_(const int K, const int Kbp, const int Kap,
                                 const int M, const int Bx, const int By,
                                 const scalar_t* B,
                                 const scalar_t* __restrict__ A,
                                 const scalar_t* __restrict__ BIAS, scalar_t* C,
                                 const int _WvPrGrp, const int CuCount) {
  UNREACHABLE_CODE;
}
#endif

// Find the min val of div2 that doesn't increase N/(div1*div2)
inline int mindiv(int N, int div1, int div2) {
  int nPrRnd = div1 * div2;
  int rnds[13];
  for (int i = 0; i < 13; i++) {
    rnds[i] = (N + nPrRnd - 1) / nPrRnd;
    nPrRnd -= div1;
  }
  for (int i = 12; i >= 0; i--)
    if (rnds[0] == rnds[i]) return (div2 - i);
  return 0;
}

// This version targets cases skinny where CUs are not filled
// Wave-SplitK is used with reduction done via atomics.
#if defined(__gfx950__)
  #define WVSPLITKRC_1KPASS
template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK,
          int UNRL, int N, int GrpsShrB, int CHUNKK, int DTRMNSTC>
__global__ void __launch_bounds__(WvPrGrp* THRDS)
    __attribute__((amdgpu_waves_per_eu(1, 1)))
    wvSplitKrc_(const int actlN, const int K, const int Kap, const int M,
                const int Bx, const int By, const scalar_t* __restrict__ A,
                const scalar_t* __restrict__ B,
                const scalar_t* __restrict__ BIAS, float* glbl, int* cntr,
                scalar_t* C, const int CuCount) {
  constexpr int NTILE = 16;
  constexpr int APAD = 1;
  constexpr int ASTRD = 64;
  constexpr int BPAD = 1;
  constexpr int WVLDS_ = THRDS * A_CHUNK / CHUNKK;
  constexpr int WVLDS = ((WVLDS_ + A_CHUNK * BPAD)) * YTILE;

  constexpr int max_lds_len = LDS_SIZE / 2;

  using scalar16 =
      __attribute__((__vector_size__((A_CHUNK * 2) * sizeof(float)))) float;
  using scalar8 =
      __attribute__((__vector_size__((A_CHUNK / 2) * sizeof(float)))) float;
  using half4 =
      __attribute__((__vector_size__((A_CHUNK / 2) * sizeof(__bf16)))) __bf16;
  union bigType {
    scalar_t h[A_CHUNK];
    float f[A_CHUNK / 2];
    unsigned int i[A_CHUNK / 2];
    float2 f2[A_CHUNK / 4];
    unsigned long l[A_CHUNK / 4];
    double d[A_CHUNK / 4];
    half4 h4[A_CHUNK / 4];
    scalar8 h8;
  };
  using big4 = __attribute__((__vector_size__(4 * sizeof(bigType)))) __bf16;

  // The main-loop staging areas are separated from the split-K readback by a
  // __syncthreads(), so the readback may reuse all of LDS.
  __shared__ union {
    struct {
      scalar_t stg[WvPrGrp * WVLDS / GrpsShrB];
      scalar_t s[max_lds_len - WvPrGrp * WVLDS / GrpsShrB];
    } mainloop;
    scalar_t all[max_lds_len];
  } lds;
  auto& stg = lds.mainloop.stg;
  auto& s = lds.mainloop.s;
  unsigned int* myStg = (unsigned int*)(&stg[WVLDS * (threadIdx.y / GrpsShrB)]);

  #ifndef WVSPLITKRC_1KPASS
  constexpr int TUC_ = (THRDS * UNRL * A_CHUNK);
  // find biggest k size that fits padded into LDS
  constexpr uint32_t kFit__ = (max_lds_len - WvPrGrp * WVLDS / GrpsShrB) / N;
  constexpr uint32_t kFit_ = (kFit__ * ASTRD) / (APAD + ASTRD);
  uint32_t kFit = kFit_ - (kFit_ % TUC_);
  uint32_t kfitsPerRdc = (K + kFit - 1) / kFit;

  // find best k split to fill the CUs
  if (((K + kfitsPerRdc * kFit - 1) / (kfitsPerRdc * kFit)) * numCuWithFullK <=
      CuCount)
    while (true) {
      while (kFit > TUC_) {
        uint32_t kFit_ = kFit - TUC_;
        if (((K + (kfitsPerRdc * kFit_ - 1)) / (kfitsPerRdc * kFit_)) *
                numCuWithFullK >
            CuCount)
          break;
        kFit = kFit_;
      }
      if (((K + ((kfitsPerRdc - 1) * kFit - 1)) / ((kfitsPerRdc - 1) * kFit)) *
              numCuWithFullK <=
          CuCount)
        kfitsPerRdc--;
      else
        break;
    }
  #else
  int constexpr kFit = 512 / CHUNKK;
  int constexpr kfitsPerRdc = 1;
  #endif

  bool doRdc = true;  // Assuming (kfitsPerRdc * kFit < K) is always true
  uint32_t numCuWithFullK =
      ((M + (WvPrGrp * YTILE / GrpsShrB) - 1) / (WvPrGrp * YTILE / GrpsShrB));
  uint32_t Mmod = numCuWithFullK * (WvPrGrp * YTILE / GrpsShrB);

  // given above k-split, find this wave's position
  uint32_t kFitPdd = kFit * CHUNKK + ((kFit * CHUNKK) / ASTRD) * APAD;
  uint32_t m0 = (blockIdx.x * WvPrGrp / GrpsShrB) * YTILE;
  uint32_t m1 = ((threadIdx.y % WvPrGrp) / GrpsShrB) * YTILE;
  uint32_t m = (m0 + m1) % Mmod;
  const uint32_t k_str = (m0 / Mmod) * kFit * kfitsPerRdc;
  uint32_t k_end = (m0 / Mmod + 1) * kFit * kfitsPerRdc;
  const uint32_t k_rnd = (K + kFit * kfitsPerRdc - 1) / (kFit * kfitsPerRdc);

  scalar8 sum4[N / NTILE / GrpsShrB][1] = {0};
  bigType bigB_[YTILE / GrpsShrB / CHUNKK][UNRL];
  const uint32_t bLoader = (threadIdx.y % GrpsShrB);
  uint32_t kBase = 0;
  if (k_str >= K) return;
  if (m >= Mmod) return;

  bool noreloada = false;
  constexpr bool FAST_UNSAFE_RDC_INIT = false;

  #ifdef WVSPLITKRC_1KPASS
  // Early glbl init, B[] loading, if 1KPASS
  if constexpr (FAST_UNSAFE_RDC_INIT) {
    if (m + (threadIdx.x % 16) < M)
      if (doRdc)
        if (k_str == 0) {
          int mindx = m + (threadIdx.x % 16);
          int nindx_ = (0 + (threadIdx.x / 16) * 4) + 0 * NTILE +
                       (N / GrpsShrB) * (threadIdx.y % GrpsShrB);
          int adr_ = mindx + M * nindx_ / 4;
          __hip_atomic_store(&cntr[adr_], 0, __ATOMIC_RELAXED,
                             __HIP_MEMORY_SCOPE_AGENT);
          for (uint32_t nt = 0; nt < N / NTILE / GrpsShrB; nt++) {
            for (uint32_t j = 0; j < 4; j++) {
              int nindx = (j + (threadIdx.x / 16) * 4) + nt * NTILE +
                          (N / GrpsShrB) * (threadIdx.y % GrpsShrB);
              int adr = mindx + M * nindx;
              __hip_atomic_store(&glbl[adr], 0, __ATOMIC_RELAXED,
                                 __HIP_MEMORY_SCOPE_AGENT);
            }
          }
        }
  }

    // Load first B[] chunk
    #pragma unroll
  for (uint32_t k2 = 0; k2 < UNRL; k2++) {
    uint32_t k = k_str + k2 * THRDS * A_CHUNK;
    uint32_t k_ = k + (threadIdx.x % (THRDS / CHUNKK)) * A_CHUNK;
    const scalar_t* B_ = &B[min__(k_, K - A_CHUNK)];
    #pragma unroll
    for (uint32_t y = 0; y < YTILE / GrpsShrB; y += CHUNKK)
      bigB_[y / CHUNKK][k2].h8 = (loadnt(
          (scalar8*)(&B_[min__((y + threadIdx.x / (THRDS / CHUNKK)) * GrpsShrB +
                                   bLoader + m,
                               M - 1) *
                         K])));
  }
  {
  #else
  while (m < Mmod) {
  #endif

  #ifndef WVSPLITKRC_1KPASS
    if constexpr (FAST_UNSAFE_RDC_INIT) {
      if (m + (threadIdx.x % 16) < M)
        if (doRdc)
          if (k_str == 0) {
            int mindx = m + (threadIdx.x % 16);
            int nindx_ = (0 + (threadIdx.x / 16) * 4) + 0 * NTILE +
                         (N / GrpsShrB) * (threadIdx.y % GrpsShrB);
            int adr_ = mindx + M * nindx_ / 4;
            __hip_atomic_store(&cntr[adr_], 0, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT);
            for (uint32_t nt = 0; nt < N / NTILE / GrpsShrB; nt++) {
              for (uint32_t j = 0; j < 4; j++) {
                int nindx = (j + (threadIdx.x / 16) * 4) + nt * NTILE +
                            (N / GrpsShrB) * (threadIdx.y % GrpsShrB);
                int adr = mindx + M * nindx;
                __hip_atomic_store(&glbl[adr], 0, __ATOMIC_RELAXED,
                                   __HIP_MEMORY_SCOPE_AGENT);
              }
            }
          }
    }

  #endif

  #ifndef WVSPLITKRC_1KPASS
    for (uint32_t k1 = k_str; k1 < k_end; k1 += THRDS * A_CHUNK * UNRL) {
  #else
    const uint32_t k1 = k_str;
    {
  #endif
  #ifndef WVSPLITKRC_1KPASS
      const bool reloada = (!noreloada) &&
                           ((k1 == k_str) || (k1 == k_str + kBase + kFit)) &&
                           (k1 < k_end);
      // load next chunk of A[] to LDS
      if (reloada) {
        if (k1 != k_str) kBase += kFit;
        __syncthreads();
  #else
      const bool reloada = (!noreloada) &&
                           ((k1 == k_str) || (k1 == k_str + kBase + kFit)) &&
                           (k1 < k_end);
      if (reloada) {
  #endif
        constexpr int sprdN = 4;
        const uint32_t thrd = threadIdx.x % (THRDS / CHUNKK);

  #ifndef WVSPLITKRC_1KPASS
    #pragma unroll
        for (int k = 0; k < kFit;
             k += (THRDS * (WvPrGrp / sprdN) * A_CHUNK) / CHUNKK) {
  #else
        const unsigned int k = 0;
        {
  #endif
          unsigned int kOff = k + (thrd * A_CHUNK);
          unsigned int kOffcp = min__(K - A_CHUNK, k_str + kOff);
          for (unsigned int n = 0; n < N; n += CHUNKK * sprdN) {
            __builtin_amdgcn_global_load_lds(
                (int*)(&A[min__(Kap * actlN - A_CHUNK,
                                kOffcp + Kap * (n / CHUNKK +
                                                (N / CHUNKK) * (threadIdx.x /
                                                                (64 / CHUNKK)) +
                                                (threadIdx.y % sprdN)))]),
                (int*)(&s[(k +
                           kFitPdd * ((n / CHUNKK) + (threadIdx.y % sprdN)))]),
                16, 0, 0);
          }

          // Stage loaded B[] to LDS for MFMA swizzling...
          for (uint32_t k2 = 0; k2 < UNRL; k2++) {
            uint32_t k = k1 + k2 * THRDS * A_CHUNK;
            uint32_t k_ = k + (threadIdx.x % (THRDS / CHUNKK)) * A_CHUNK;
            const bool oob_k = (k_ >= K);
            for (uint32_t y = 0; y < YTILE / GrpsShrB; y += CHUNKK) {
              uint32_t idx =
                  (threadIdx.x % (THRDS / CHUNKK)) * 4 +
                  ((y + threadIdx.x / (THRDS / CHUNKK)) * GrpsShrB + bLoader) *
                      ((THRDS / CHUNKK + BPAD) * 4);
              // zero out if oob
              *((scalar8*)&myStg[idx]) =
                  (oob_k)  // TODO: ever necessary (y*GrpsShrB+bLoader+m>=M) ?
                      ? 0
                      : bigB_[y / CHUNKK][k2].h8;
            }
          }
        }
      }
    }
  #ifndef WVSPLITKRC_1KPASS
    // Fire load of next B[] chunk...
    if ((k1 + THRDS * A_CHUNK * UNRL < k_end) &&
        (k1 + THRDS * A_CHUNK * UNRL < K))
    #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + THRDS * A_CHUNK * UNRL + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        const scalar_t* B_ = &B[min__(k_, K - A_CHUNK)];
    #pragma unroll
        for (uint32_t y = 0; y < YTILE / GrpsShrB; y += CHUNKK)
          bigB_[y / CHUNKK][k2].h8 = (loadnt(
              (scalar8*)(&B_[min__((y + threadIdx.x / (THRDS / CHUNKK)) *
                                           GrpsShrB +
                                       bLoader + m,
                                   M - 1) *
                             K])));
      }
  #endif

    // B[] staging is cooperative across GrpsShrB, so sync here before reading
    // back. This wait is currently inserted by compiler, but not guaranteed.
    asm volatile("s_waitcnt 0");
    __syncthreads();

    // read back B[] swizzled for MFMA...
    bigType bigB[YTILE / CHUNKK][UNRL];
    for (uint32_t k2 = 0; k2 < UNRL; k2++) {
      for (uint32_t y = 0; y < YTILE / CHUNKK; y++) {
        unsigned int idx =
            (threadIdx.x % YTILE) * ((THRDS / CHUNKK + BPAD) * 4) +
            (threadIdx.x / YTILE) * 4 + y * 16;
        bigB[y][k2].h8 = *((scalar8*)&myStg[idx]);
      }
    }

    // rReadback A[] swizzled for MFMA...
    bigType bigA[N / GrpsShrB / CHUNKK][UNRL];
  #pragma unroll
    for (uint32_t k2 = 0; k2 < UNRL; k2++) {
      uint32_t k = k1 + k2 * THRDS * A_CHUNK - kBase - k_str;
  #pragma unroll
      for (uint32_t nt = 0; nt < N / GrpsShrB; nt += NTILE)
  #pragma unroll
        for (uint32_t n = 0; n < NTILE / CHUNKK; n++) {
          uint32_t idxa =
              ((nt + (N / GrpsShrB) * (threadIdx.y % GrpsShrB)) % (N / CHUNKK) +
               (threadIdx.x % NTILE)) *
                  kFitPdd +
              ((nt + (N / GrpsShrB) * (threadIdx.y % GrpsShrB)) /
               (N / CHUNKK)) *
                  A_CHUNK * (64 / CHUNKK) +
              A_CHUNK * ((threadIdx.x / NTILE) + n * 4) + k;
          bigA[nt / CHUNKK + n][k2] = *((const bigType*)(&(s[idxa])));
        }
    }

    // Do the MFMAs
  #pragma unroll
    for (uint32_t k2 = 0; k2 < UNRL; k2++) {
  #pragma unroll
      for (uint32_t nt = 0; nt < N / NTILE / GrpsShrB; nt++) {
  #pragma unroll
        for (uint32_t j = 0; j < YTILE / CHUNKK; j++) {
          if constexpr (std::is_same_v<scalar_t, half>) {
            sum4[nt][0] = __builtin_amdgcn_mfma_f32_16x16x32_f16(
                bigA[nt * (YTILE / CHUNKK) + j][k2].h8, bigB[j][k2].h8,
                sum4[nt][0], 0, 0, 0);
          } else {  // bf16
            sum4[nt][0] = __builtin_amdgcn_mfma_f32_16x16x32_bf16(
                bigA[nt * (YTILE / CHUNKK) + j][k2].h8, bigB[j][k2].h8,
                sum4[nt][0], 0, 0, 0);
          }
        }
      }
    }
  }

  union flt4 {
    scalar8 s8;
    float2 f2[2];
    float4 f4;
  };
  if (m + (threadIdx.x % 16) < M) {
    int my_cntr;
    int mindx = m + (threadIdx.x % 16);
    int g_mindx = m * 4 + (threadIdx.x % 64);  // coalesced atomic reduction
    scalar_t biases[N / NTILE / GrpsShrB][4] = {};
    // Atomic add the output, read biases
    for (uint32_t nt = 0; nt < N / NTILE / GrpsShrB; nt++) {
      int g_nindx =
          (nt * NTILE + (N / GrpsShrB) * (threadIdx.y % GrpsShrB)) / 4;
      int g_adr = g_mindx * 4 + 0 + M * g_nindx * 4;
      if (DTRMNSTC) {
        flt4 flt4_ = {.s8 = sum4[nt][0]};
        __hip_atomic_store((float2*)&glbl[g_adr + M * N * (m0 / Mmod)],
                           flt4_.f2[0], __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_AGENT);
        __hip_atomic_store((float2*)&glbl[g_adr + 2 + M * N * (m0 / Mmod)],
                           flt4_.f2[1], __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_AGENT);
      } else {
        for (uint32_t j = 0; j < 4; j++)
          atomicAdd((&glbl[g_adr + j]), sum4[nt][0][j]);
      }
    }

    __atomic_signal_fence(__ATOMIC_SEQ_CST);
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    __atomic_signal_fence(__ATOMIC_SEQ_CST);

    int nindx_ = (0 + (threadIdx.x / 16) * 4) + 0 * NTILE +
                 (N / GrpsShrB) * (threadIdx.y % GrpsShrB);
    int adr_ = mindx + M * nindx_ / 4;
    my_cntr = atomicAdd(&cntr[adr_], 1);

    // make sure LDS is free for write out staging
    if (DTRMNSTC) __syncthreads();

    // Update the complete counter
    flt4 vals[N / NTILE / GrpsShrB] = {};
    // If we're the last k-shard, read back the value and convert...
    if (my_cntr + 1 == k_rnd) {
      cntr[adr_] = 0;  // clear for next round
      if constexpr (DTRMNSTC) {
        constexpr uint32_t NT = N / NTILE / GrpsShrB;
        // The readback stages k-shard partials through LDS, which bounds how
        // many shards may be in flight at once.
        constexpr uint32_t LDS_F4 =
            max_lds_len * sizeof(scalar_t) / sizeof(float4);
        constexpr uint32_t KSBMAX = LDS_F4 / (THRDS * 4 * NT);
        static_assert(KSBMAX >= 1, "LDS cannot stage even one k-shard");
        const uint32_t KSB = (k_rnd < KSBMAX) ? k_rnd : KSBMAX;

        // First batch is peeled; it is the only batch for most shapes.
  #pragma unroll
        for (uint32_t kb = 0; kb < KSB; kb++) {
          for (uint32_t nt = 0; nt < NT; nt++) {
            int g_nindx =
                (nt * NTILE + (N / GrpsShrB) * (threadIdx.y % GrpsShrB)) / 4;
            int g_adr = g_mindx * 4 + 0 + M * g_nindx * 4;
            __builtin_amdgcn_global_load_lds(
                (float4*)(&glbl[g_adr + M * N * kb]),
                &(((float4*)lds.all)[(threadIdx.y * THRDS) + kb * THRDS * 4 +
                                     nt * THRDS * 4 * KSB]),
                16, 0, 0);
          }
        }
        if (BIAS)  // overlaps the DMAs above
          for (uint32_t nt = 0; nt < NT; nt++) {
            for (uint32_t j = 0; j < 4; j++) {
              int nindx = (j + (threadIdx.x / 16) * 4) + nt * NTILE +
                          (N / GrpsShrB) * (threadIdx.y % GrpsShrB);
              biases[nt][j] = BIAS[(mindx % Bx) + (nindx % By) * Bx];
            }
          }
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        for (uint32_t kb = 0; kb < KSB; kb++) {
          for (uint32_t nt = 0; nt < NT; nt++) {
            vals[nt].f4 +=
                ((float4*)lds.all)[(threadIdx.x + threadIdx.y * THRDS) +
                                   kb * THRDS * 4 + nt * THRDS * 4 * KSB];
          }
        }

        for (uint32_t ks0 = KSB; ks0 < k_rnd; ks0 += KSB) {
          const uint32_t ksn = (k_rnd - ks0 < KSB) ? (k_rnd - ks0) : KSB;
          // prior reads must retire before the staging is overwritten
          asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
          for (uint32_t kb = 0; kb < ksn; kb++) {
            for (uint32_t nt = 0; nt < NT; nt++) {
              int g_nindx =
                  (nt * NTILE + (N / GrpsShrB) * (threadIdx.y % GrpsShrB)) / 4;
              int g_adr = g_mindx * 4 + 0 + M * g_nindx * 4;
              __builtin_amdgcn_global_load_lds(
                  (float4*)(&glbl[g_adr + M * N * (ks0 + kb)]),
                  &(((float4*)lds.all)[(threadIdx.y * THRDS) + kb * THRDS * 4 +
                                       nt * THRDS * 4 * KSB]),
                  16, 0, 0);
            }
          }
          asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
          for (uint32_t kb = 0; kb < ksn; kb++) {
            for (uint32_t nt = 0; nt < NT; nt++) {
              vals[nt].f4 +=
                  ((float4*)lds.all)[(threadIdx.x + threadIdx.y * THRDS) +
                                     kb * THRDS * 4 + nt * THRDS * 4 * KSB];
            }
          }
        }
      } else {
        for (uint32_t nt = 0; nt < N / NTILE / GrpsShrB; nt++) {
          int g_nindx =
              (nt * NTILE + (N / GrpsShrB) * (threadIdx.y % GrpsShrB)) / 4;
          int g_adr = g_mindx * 4 + 0 + M * g_nindx * 4;
          vals[nt].f4 = *(float4*)(&glbl[g_adr]);
          *(float4*)(&glbl[g_adr]) = {};  // clear out for next round
        }
        if (BIAS)
          for (uint32_t nt = 0; nt < N / NTILE / GrpsShrB; nt++) {
            for (uint32_t j = 0; j < 4; j++) {
              int nindx = (j + (threadIdx.x / 16) * 4) + nt * NTILE +
                          (N / GrpsShrB) * (threadIdx.y % GrpsShrB);
              biases[nt][j] = BIAS[(mindx % Bx) + (nindx % By) * Bx];
            }
          }
      }
      __builtin_amdgcn_sched_barrier(0);
      for (uint32_t nt = 0; nt < N / NTILE / GrpsShrB; nt++) {
        for (uint32_t j = 0; j < 4; j++) {
          int nindx = (j + (threadIdx.x / 16) * 4) + nt * NTILE +
                      (N / GrpsShrB) * (threadIdx.y % GrpsShrB);
          if (nindx < actlN) {
            int adr = mindx + M * nindx;
            if constexpr (std::is_same_v<scalar_t, __hip_bfloat16>) {
              vals[nt].s8[j] += __bfloat162float(biases[nt][j]);
              C[adr] = __float2bfloat16(vals[nt].s8[j]);
            } else {
              vals[nt].s8[j] += __half2float(biases[nt][j]);
              C[adr] = __float2half(vals[nt].s8[j]);
            }
          }
        }
      }
    }

  #ifndef WVSPLITKRC_1KPASS
    m0 += CuCount * WvPrGrp * YTILE / GrpsShrB;
    m = (m0 + m1) % Mmod;
    k_str = (m0 / Mmod) * kFit * kfitsPerRdc;
    k_end = (m0 / Mmod + 1) * kFit * kfitsPerRdc;
    if (k_str >= K) break;
    kBase = 0;
  #endif
  }
}
#else
template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK,
          int UNRL, int N, int GrpsShrB, int CHUNKK, int DTRMNSTC>
__global__ void wvSplitKrc_(const int actlN, const int K, const int Kap,
                            const int M, const int Bx, const int By,
                            const scalar_t* B, const scalar_t* __restrict__ A,
                            const scalar_t* __restrict__ BIAS, float* glbl,
                            int* cntr, scalar_t* C, const int CuCount) {
  UNREACHABLE_CODE;
}
#endif  // defined(__HIP__GFX9__) TODO: Add NAVI support

#if defined(__HIP__MI3XX__) || defined(__GFX12__)
template <typename scalar_t, typename fp8_t, int THRDS, int YTILE, int WvPrGrp,
          int A_CHUNK, int UNRL, int N>
__global__ void __launch_bounds__(WvPrGrp* THRDS)
    wvSplitKQ_hf_sml_(const int K, const int Kap, const int Kbp, const int M,
                      const int Bx, const int By, const fp8_t* B,
                      const fp8_t* __restrict__ A,
                      const scalar_t* __restrict__ BIAS, scalar_t* C,
                      const float* __restrict__ s_A,
                      const float* __restrict__ s_B, const int _WvPrGrp,
                      const int CuCount) {
  constexpr int max_lds_len = LDS_SIZE;
  using scalar8 =
      __attribute__((__vector_size__((A_CHUNK / 4) * sizeof(float)))) float;
  using intx2 = __attribute__((__vector_size__(2 * sizeof(int)))) int;
  using intx4 = __attribute__((__vector_size__(4 * sizeof(int)))) int;
  union bigType {
    char f8[A_CHUNK];
    char2 c2[A_CHUNK / 2];
    scalar_t h[A_CHUNK / 2];
    float f[A_CHUNK / 4];
    int i[A_CHUNK / 4];
    long l[A_CHUNK / 8];
    intx4 l2[A_CHUNK / 16];
    scalar8 h8;
  };

  __shared__ fp8_t s[max_lds_len];

  for (uint32_t k = (threadIdx.y * THRDS + threadIdx.x) * A_CHUNK;
       k < min__(Kap * N, max_lds_len); k += THRDS * WvPrGrp * A_CHUNK) {
  #if defined(__gfx950__)
    __builtin_amdgcn_global_load_lds((int*)(&A[k]), (int*)(&s[k]), 16, 0, 0);
  #else
    *((bigType*)(&s[k])) = *((bigType*)(&A[k]));
  #endif
  }
  asm volatile("s_waitcnt vmcnt(0)");
  __syncthreads();

  if (threadIdx.y >= _WvPrGrp) return;

  uint32_t m = (blockIdx.x * _WvPrGrp + (threadIdx.y % _WvPrGrp)) * YTILE;

  float sA = *s_A;
  float sB = *s_B;

  while (m < M) {
  #ifdef __GFX12__
    // gfx12: per-lane scalar accumulation via v_dot4_f32_fp8_fp8
    float sum[N][YTILE] = {};
  #else
    // gfx9: MFMA accumulation
    scalar8 sum[N][YTILE] = {};
  #endif
    for (uint32_t k1 = 0; k1 < K; k1 += THRDS * A_CHUNK * UNRL) {
      bigType bigA[N][UNRL] = {};
      bigType bigB[YTILE][UNRL];

      // Fetch the weight matrix from memory!
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        const fp8_t* B_ = &B[min__(k_, K - A_CHUNK)];
  #pragma unroll
        for (uint32_t y = 0; y < YTILE; ++y) {
          bigB[y][k2].h8 = (loadnt((scalar8*)(&B_[min__(y + m, M - 1) * Kbp])));
        }
      }

  // Fetch activation matrix from either just LDS or from both LDS / memory
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;
        for (int n = 0; n < N; n++) {
          bigA[n][k2] = *((const bigType*)(&(s[k_ + Kap * n])));
        }
      }

  // Do the matrix multiplication in interleaved manner
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        for (uint32_t n = 0; n < N; n++) {
  #ifdef __GFX12__
          // gfx12: 4 x dot4 per A_CHUNK=16 bytes (4 FP8 per dot4)
          for (int y = 0; y < YTILE; ++y) {
    #pragma unroll
            for (int i = 0; i < A_CHUNK / 4; i++) {
              sum[n][y] = __builtin_amdgcn_dot4_f32_fp8_fp8(
                  bigA[n][k2].i[i], bigB[y][k2].i[i], sum[n][y]);
            }
          }
  #else
          // gfx9: MFMA path
          for (int i = 0; i < A_CHUNK; i += 8) {
            for (int y = 0; y < YTILE; ++y) {
              sum[n][y] = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(
                  bigA[n][k2].l[i / 8], bigB[y][k2].l[i / 8], sum[n][y], 0, 0,
                  0);
            }
          }
  #endif
        }
      }
    }

    // Final reduction
  #ifdef __GFX12__
    // gfx12 wave32: DPP row_shr within 16-lane rows + cross-row shuffle
    for (int n = 0; n < N; n++) {
      for (int y = 0; y < YTILE; y++) {
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:8 bound_ctrl:0 "
            : "=v"(sum[n][y])
            : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:4 bound_ctrl:0 "
            : "=v"(sum[n][y])
            : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:2 bound_ctrl:0 "
            : "=v"(sum[n][y])
            : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:1 bound_ctrl:0 "
            : "=v"(sum[n][y])
            : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
        sum[n][y] += __shfl_xor(sum[n][y], 16);
      }
    }
  #else
    // gfx9 MFMA reduction
    for (int n = 0; n < N; n++) {
      for (int y = 0; y < YTILE; y++) {
        float accm0 = sum[n][y][0];
        accm0 += __builtin_amdgcn_mov_dpp(sum[n][y][1], 0x101, 0xf, 0xf,
                                          1);  // row_shl1
        accm0 += __builtin_amdgcn_mov_dpp(sum[n][y][2], 0x102, 0xf, 0xf,
                                          1);  // row_shl2
        accm0 += __builtin_amdgcn_mov_dpp(sum[n][y][3], 0x103, 0xf, 0xf,
                                          1);  // row_shl3
        accm0 += __shfl_down(accm0, 20);
        accm0 += __shfl_down(accm0, 40);
        sum[n][y][0] = accm0;
      }
    }
  #endif

    const bool writeback_lane =
  #ifdef __GFX12__
        threadIdx.x == (THRDS - 1);
  #else
        threadIdx.x == 0;
  #endif
    if (writeback_lane) {
      scalar_t biases[N][YTILE] = {};
      if (BIAS)
        for (int n = 0; n < N; n++) {
          for (int y = 0; y < YTILE; y++) {
            biases[n][y] = BIAS[(m + y) % Bx + (n % By) * Bx];
          }
        }
      for (int n = 0; n < N; n++) {
        for (int y = 0; y < YTILE; y++) {
          if (y + m >= M) break;  // To avoid mem access fault.
  #ifdef __GFX12__
          float result = sum[n][y] * sA * sB;
  #else
          float result = sum[n][y][0] * sA * sB;
  #endif
          if constexpr (std::is_same_v<scalar_t, half>) {
            result += __half2float(biases[n][y]);
          } else if constexpr (std::is_same_v<scalar_t, __hip_bfloat16>) {
            result += __bfloat162float(biases[n][y]);
          }
          C[m + y + n * M] = __float2s<scalar_t>(result);
        }
      }
    }

    m += CuCount * _WvPrGrp * YTILE;
  }
}
#else   // !defined(__HIP__MI3XX__) && !defined(__GFX12__)
template <typename scalar_t, typename fp8_t, int THRDS, int YTILE, int WvPrGrp,
          int A_CHUNK, int UNRL, int N>
__global__ void wvSplitKQ_hf_sml_(const int K, const int Kap, const int Kbp,
                                  const int M, const int Bx, const int By,
                                  const fp8_t* B, const fp8_t* __restrict__ A,
                                  const scalar_t* __restrict__ BIAS,
                                  scalar_t* C, const float* __restrict__ s_A,
                                  const float* __restrict__ s_B,
                                  const int _WvPrGrp, const int CuCount) {
  UNREACHABLE_CODE;
}
#endif  // defined(__HIP__MI3XX__) || defined(__GFX12__)

#if defined(__HIP__MI3XX__) || defined(__GFX12__)
template <typename scalar_t, typename fp8_t, int THRDS, int YTILE, int WvPrGrp,
          int A_CHUNK, int UNRL, int N>
__global__ void __launch_bounds__(WvPrGrp* THRDS)
    wvSplitKQ_hf_(const int K, const int Kap, const int Kbp, const int M,
                  const int Bx, const int By, const fp8_t* B,
                  const fp8_t* __restrict__ A,
                  const scalar_t* __restrict__ BIAS, scalar_t* C,
                  const float* __restrict__ s_A, const float* __restrict__ s_B,
                  const int _WvPrGrp, const int CuCount) {
  constexpr int max_lds_len = LDS_SIZE;
  using scalar8 =
      __attribute__((__vector_size__((A_CHUNK / 4) * sizeof(float)))) float;
  using intx2 = __attribute__((__vector_size__(2 * sizeof(int)))) int;
  using intx4 = __attribute__((__vector_size__(4 * sizeof(int)))) int;
  union bigType {
    char f8[A_CHUNK];
    char2 c2[A_CHUNK / 2];
    scalar_t h[A_CHUNK / 2];
    float f[A_CHUNK / 4];
    int i[A_CHUNK / 4];
    long l[A_CHUNK / 8];
    intx4 l2[A_CHUNK / 16];
    scalar8 h8;
  };

  __shared__ fp8_t s[max_lds_len];

  for (uint32_t k = (threadIdx.y * THRDS + threadIdx.x) * A_CHUNK;
       k < min__(Kap * N, max_lds_len); k += THRDS * WvPrGrp * A_CHUNK) {
  #if defined(__gfx950__)
    __builtin_amdgcn_global_load_lds((int*)(&A[k]), (int*)(&s[k]), 16, 0, 0);
  #else
    *((bigType*)(&s[k])) = *((bigType*)(&A[k]));
  #endif
  }
  asm volatile("s_waitcnt vmcnt(0)");
  __syncthreads();

  if (threadIdx.y >= _WvPrGrp) return;

  uint32_t m = (blockIdx.x * _WvPrGrp + (threadIdx.y % _WvPrGrp)) * YTILE;

  float sA = *s_A;
  float sB = *s_B;

  while (m < M) {
  #ifdef __GFX12__
    // gfx12: per-lane scalar accumulation via v_dot4_f32_fp8_fp8
    float sum[N][YTILE] = {};
  #else
    // gfx9: MFMA accumulation
    scalar8 sum[N][YTILE] = {};
  #endif
    for (uint32_t k1 = 0; k1 < K; k1 += THRDS * A_CHUNK * UNRL) {
      bigType bigA[N][UNRL] = {};
      bigType bigB[YTILE][UNRL];

      // Fetch the weight matrix from memory!
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        const fp8_t* B_ = &B[min__(k_, K - A_CHUNK)];
        for (int y = 0; y < YTILE; ++y) {
          bigB[y][k2].h8 = (loadnt((scalar8*)(&B_[min__(y + m, M - 1) * Kbp])));
        }
      }

  // Fetch activation matrix from either just LDS or from both LDS / memory
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;
        for (int n = 0; n < N; n++) {
          if (k_ + Kap * n < max_lds_len)
            bigA[n][k2] = *((const bigType*)(&(s[k_ + Kap * n])));
          else
            bigA[n][k2] = *((const bigType*)(&(A[k_ + Kap * n])));
        }
      }

  // Do the matrix multiplication in interleaved manner
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        for (uint32_t n = 0; n < N; n++) {
  #ifdef __GFX12__
          // gfx12: 4 x dot4 per A_CHUNK=16 bytes (4 FP8 per dot4)
          for (int y = 0; y < YTILE; ++y) {
    #pragma unroll
            for (int i = 0; i < A_CHUNK / 4; i++) {
              sum[n][y] = __builtin_amdgcn_dot4_f32_fp8_fp8(
                  bigA[n][k2].i[i], bigB[y][k2].i[i], sum[n][y]);
            }
          }
  #else
          // gfx9: MFMA path
          for (int i = 0; i < A_CHUNK; i += 8) {
            for (int y = 0; y < YTILE; ++y) {
              sum[n][y] = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(
                  bigA[n][k2].l[i / 8], bigB[y][k2].l[i / 8], sum[n][y], 0, 0,
                  0);
            }
          }
  #endif
        }
      }
    }

    // Final reduction
  #ifdef __GFX12__
    // gfx12 wave32: DPP row_shr within 16-lane rows + cross-row shuffle
    for (int n = 0; n < N; n++) {
      for (int y = 0; y < YTILE; y++) {
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:8 bound_ctrl:0 "
            : "=v"(sum[n][y])
            : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:4 bound_ctrl:0 "
            : "=v"(sum[n][y])
            : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:2 bound_ctrl:0 "
            : "=v"(sum[n][y])
            : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:1 bound_ctrl:0 "
            : "=v"(sum[n][y])
            : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
        sum[n][y] += __shfl_xor(sum[n][y], 16);
      }
    }
  #else
    // gfx9 MFMA reduction
    for (int n = 0; n < N; n++) {
      for (int y = 0; y < YTILE; y++) {
        float accm0 = sum[n][y][0];
        accm0 += __builtin_amdgcn_mov_dpp(sum[n][y][1], 0x101, 0xf, 0xf,
                                          1);  // row_shl1
        accm0 += __builtin_amdgcn_mov_dpp(sum[n][y][2], 0x102, 0xf, 0xf,
                                          1);  // row_shl2
        accm0 += __builtin_amdgcn_mov_dpp(sum[n][y][3], 0x103, 0xf, 0xf,
                                          1);  // row_shl3
        accm0 += __shfl_down(accm0, 20);
        accm0 += __shfl_down(accm0, 40);
        sum[n][y][0] = accm0;
      }
    }
  #endif

    const bool writeback_lane =
  #ifdef __GFX12__
        threadIdx.x == (THRDS - 1);
  #else
        threadIdx.x == 0;
  #endif
    if (writeback_lane) {
      scalar_t biases[N][YTILE] = {};
      if (BIAS)
        for (int n = 0; n < N; n++) {
          for (int y = 0; y < YTILE; y++) {
            biases[n][y] = BIAS[(m + y) % Bx + (n % By) * Bx];
          }
        }
      for (int n = 0; n < N; n++) {
        for (int y = 0; y < YTILE; y++) {
          if (y + m >= M) break;  // To avoid mem access fault.
  #ifdef __GFX12__
          float result = sum[n][y] * sA * sB;
  #else
          float result = sum[n][y][0] * sA * sB;
  #endif
          if constexpr (std::is_same_v<scalar_t, half>) {
            result += __half2float(biases[n][y]);
          } else if constexpr (std::is_same_v<scalar_t, __hip_bfloat16>) {
            result += __bfloat162float(biases[n][y]);
          }
          C[m + y + n * M] = __float2s<scalar_t>(result);
        }
      }
    }

    m += CuCount * _WvPrGrp * YTILE;
  }
}
#else   // !defined(__HIP__MI3XX__) && !defined(__GFX12__)
template <typename scalar_t, typename fp8_t, int THRDS, int YTILE, int WvPrGrp,
          int A_CHUNK, int UNRL, int N>
__global__ void wvSplitKQ_hf_(const int K, const int Kap, const int Kbp,
                              const int M, const int Bx, const int By,
                              const fp8_t* B, const fp8_t* __restrict__ A,
                              const scalar_t* __restrict__ BIAS, scalar_t* C,
                              const float* __restrict__ s_A,
                              const float* __restrict__ s_B, const int _WvPrGrp,
                              const int CuCount) {
  UNREACHABLE_CODE;
}
#endif  // defined(__HIP__MI3XX__) || defined(__GFX12__)

// Do not leak this into the including translation units: skinny_gemms_int4.cu
// defines its own LDS_SIZE (64 KiB, no gfx950 case), so a shared TU must
// never pick up one definition from the other by accident.
#undef LDS_SIZE
