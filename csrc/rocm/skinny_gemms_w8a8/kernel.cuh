#pragma once

#include <ATen/cuda/CUDAContext.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>

#include "../../cuda_compat.h"

#if defined(__HIPCC__) && \
    (defined(__gfx90a__) || defined(__gfx942__) || defined(__gfx950__))
  #define __HIP__GFX9__
#endif

// Total LDS budget per workgroup. We reserve LDS_AUX_BYTES at the end for
// auxiliary shared-memory arrays (a_scales_dyn, warp_max) used by the dynamic
// quantization path. The activation buffer s[] gets the remainder.
#define LDS_TOTAL (64 * 1024)
#define LDS_AUX_BYTES 128
#define LDS_SIZE (LDS_TOTAL - LDS_AUX_BYTES)

inline int get_lds_size_w8a8() {
  static bool is_cached = false;
  static int result;
  if (is_cached == false) {
    auto dprops = at::cuda::getCurrentDeviceProperties();
    std::string device_arch = dprops->gcnArchName;
    size_t substring = device_arch.find("gfx95");
    // Subtract LDS_AUX_BYTES to leave room for auxiliary shared-memory arrays
    // (a_scales_dyn, warp_max) used by the dynamic quantization path.
    result = (substring == std::string::npos ? 64 * 1024 - LDS_AUX_BYTES
                                             : 160 * 1024 - LDS_AUX_BYTES);
    is_cached = true;
  }
  return result;
}

inline bool is_gfx11_w8a8() {
  static const bool result = [] {
    auto dprops = at::cuda::getCurrentDeviceProperties();
    std::string device_arch = dprops->gcnArchName;
    return device_arch.find("gfx11") != std::string::npos;
  }();
  return result;
}

// Check if this is a low-bandwidth gfx11 variant (gfx1150, gfx1152, gfx1153)
// — RDNA 3.5 mobile parts. gfx1151 (Strix Halo) has higher bandwidth and is
// handled like gfx9. gfx1103 (RDNA 3 mobile, Radeon 760M) has different
// instruction-level preferences and gets its own branch via
// is_gfx1103_w8a8().
inline bool is_low_bandwidth_gfx11_w8a8() {
  static const bool result = [] {
    auto dprops = at::cuda::getCurrentDeviceProperties();
    std::string device_arch = dprops->gcnArchName;
    return device_arch.find("gfx1150") != std::string::npos ||
           device_arch.find("gfx1152") != std::string::npos ||
           device_arch.find("gfx1153") != std::string::npos;
  }();
  return result;
}

// gfx1103 (Radeon 760M, RDNA 3 mobile) — strongly prefers ur=4 across most
// shapes, unlike its RDNA 3.5 cousins (gfx1150/1152/1153) which prefer ur=2.
inline bool is_gfx1103_w8a8() {
  static const bool result = [] {
    auto dprops = at::cuda::getCurrentDeviceProperties();
    std::string device_arch = dprops->gcnArchName;
    return device_arch.find("gfx1103") != std::string::npos;
  }();
  return result;
}

#if defined(NDEBUG)
  #undef NDEBUG
  #include <assert.h>
  #define UNREACHABLE_CODE assert(false);
  #define NDEBUG
#else
  #define UNREACHABLE_CODE assert(false);
#endif

template <typename T>
__device__ __forceinline__ float __s2float(T v);

template <>
__device__ __forceinline__ float __s2float(half v) {
  return __half2float(v);
}

template <>
__device__ __forceinline__ float __s2float(__hip_bfloat16 v) {
  return __bfloat162float(v);
}

template <typename T>
__device__ __forceinline__ T __float2s(float v);

template <>
__device__ __forceinline__ half __float2s(float v) {
  return __float2half(v);
}

template <>
__device__ __forceinline__ __hip_bfloat16 __float2s(float v) {
  return __float2bfloat16(v);
}

template <typename T>
__device__ __forceinline__ T loadnt(T* addr) {
  return __builtin_nontemporal_load(addr);
}

template <typename T>
struct scalar {};

template <>
struct scalar<c10::Half> {
  using type = half;
};

template <>
struct scalar<c10::BFloat16> {
  using type = __hip_bfloat16;
};

#define DOT2C(V0, V2, V3)                                                   \
  if constexpr (std::is_same_v<scalar_t, half>) {                           \
    V0 = __builtin_amdgcn_fdot2(*((half2*)(&(V2))), *((half2*)(&(V3))), V0, \
                                false);                                     \
  } else if constexpr (std::is_same_v<scalar_t, __hip_bfloat16>) {          \
    float2 s = __bfloat1622float2(*((__hip_bfloat162*)(&(V2)))) *           \
               __bfloat1622float2(*((__hip_bfloat162*)(&(V3))));            \
    V0 += (s.x + s.y);                                                      \
  }

#if defined(__GFX11__)
  // Int8x4 dot product for RDNA3/4: v_dot4_i32_iu8
  // 4 signed int8 multiplies + int32 accumulate in one instruction.
  #define DOT4_I8(V0, V2, V3)                                  \
    V0 = __builtin_amdgcn_sudot4(true, *((int*)(&(V2))), true, \
                                 *((int*)(&(V3))), V0, false);
#endif

__device__ inline unsigned int min_w8a8(uint32_t a, uint32_t b) {
  return min(a, b);
}

// Round-to-nearest-even float→int8 with saturation.
// Matches the ROCm implementation in scaled_quant.cu (uses nearbyint + clamp).
static __device__ __forceinline__ int8_t float_to_int8_rn(float x) {
  static constexpr auto i8_min =
      static_cast<float>(std::numeric_limits<int8_t>::min());
  static constexpr auto i8_max =
      static_cast<float>(std::numeric_limits<int8_t>::max());
  float dst = std::nearbyint(x);
  dst = dst < i8_min ? i8_min : (dst > i8_max ? i8_max : dst);
  return static_cast<int8_t>(dst);
}

inline int mindiv_w8a8(int N, int div1, int div2) {
  int nPrRnd = div1 * div2;
  int limit = div2 < 13 ? div2 : 13;
  int rnds[16];
  for (int i = 0; i < limit; i++) {
    rnds[i] = (N + nPrRnd - 1) / nPrRnd;
    nPrRnd -= div1;
  }
  for (int i = limit - 1; i >= 0; i--)
    if (rnds[0] == rnds[i]) return (div2 - i);
  return 0;
}

// W8A8 skinny GEMM kernel: int8 weights, int8 activations
// Both operands are int8. Activations stored in LDS as int8 (1 byte each),
// giving 2x the LDS capacity compared to the W8A16 variant.
// Epilogue: result = sum * w_scale[m] * a_scale (per-channel weight, per-tensor
// activation)
//
// Quantization modes for activations (quant_mode):
//   1 = fused static quant (bf16 in, known a_scale)
//   2 = fused dynamic quant (bf16 in, compute a_scale per row)
#if defined(__HIP__GFX9__) || defined(__GFX11__)
template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK,
          int UNRL, int N>
__global__ void __launch_bounds__(WvPrGrp* THRDS)
    wvSplitK_w8a8_hf_sml_(const int K, const int M, const int Bx, const int By,
                          const int8_t* B, const void* __restrict__ A_raw,
                          const scalar_t* w_scale,
                          const float* __restrict__ a_scale,
                          const scalar_t* __restrict__ BIAS, scalar_t* C,
                          const int _WvPrGrp, const int CuCount,
                          const int quant_mode) {
  // LDS stores int8 activations: 1 byte each (vs 2 bytes for fp16 in W8A16)
  constexpr int max_lds_ints = LDS_SIZE;

  // Activation load union: A_CHUNK int8 values = A_CHUNK bytes
  union bigTypeA {
    int8_t b[A_CHUNK];
    float f[A_CHUNK / 4];
  };

  // Converted activation values: A_CHUNK fp16/bf16 values = 2*A_CHUNK bytes
  union bigTypeAcvt {
    scalar_t h[A_CHUNK];
    float f[A_CHUNK / 2];
  };

  // Weight union: A_CHUNK int8 values = A_CHUNK bytes
  union bigTypeW {
    int8_t b[A_CHUNK];
    float f[A_CHUNK / 4];
  };

  __shared__ int8_t s[max_lds_ints];
  // Per-row activation scales computed by dynamic quantization (mode 2).
  // Max N=5 rows. Only written/read when quant_mode == 2.
  __shared__ float a_scales_dyn[5];
  // Scratch space for cross-warp max reduction (one float per warp).
  __shared__ float warp_max[WvPrGrp];

  // Fetch activation matrix to LDS as int8.
  if (quant_mode == 2) {
    // --- Mode 2: fused dynamic quantization ---
    // Activations arrive in bf16/fp16. For each row, find absmax,
    // compute scale, then quantize to int8.
    const scalar_t* A_fp = (const scalar_t*)A_raw;
    const uint32_t tid = threadIdx.y * THRDS + threadIdx.x;
    const uint32_t stride = THRDS * WvPrGrp;

    for (int n = 0; n < N; n++) {
      const scalar_t* A_row = &A_fp[n * K];

      // Phase 1: find per-thread absmax for this row
      float tmax = 0.0f;
      for (uint32_t k = tid; k < (uint32_t)K; k += stride) {
        tmax = fmaxf(tmax, fabsf(__s2float(A_row[k])));
      }

      // Phase 2: warp-level max reduction
  #if defined(__GFX11__)
      // GFX11 wave32: row_shr DPP cascade for absmax (lanes 0-15),
      // then __shfl_xor(16) to combine the two halves.
      asm("s_nop 0\n\tv_max_f32 %0, %2, %3 row_shr:8 bound_ctrl:0 "
          : "=v"(tmax)
          : "0"(tmax), "v"(tmax), "v"(tmax));
      asm("s_nop 0\n\tv_max_f32 %0, %2, %3 row_shr:4 bound_ctrl:0 "
          : "=v"(tmax)
          : "0"(tmax), "v"(tmax), "v"(tmax));
      asm("s_nop 0\n\tv_max_f32 %0, %2, %3 row_shr:2 bound_ctrl:0 "
          : "=v"(tmax)
          : "0"(tmax), "v"(tmax), "v"(tmax));
      asm("s_nop 0\n\tv_max_f32 %0, %2, %3 row_shr:1 bound_ctrl:0 "
          : "=v"(tmax)
          : "0"(tmax), "v"(tmax), "v"(tmax));
      tmax = fmaxf(tmax, __shfl_xor(tmax, 16));
      // Broadcast from lane (THRDS-1) so the LDS write at lane 0 below
      // (shared with the GFX9 path) sees the reduced value.
      tmax = __shfl(tmax, THRDS - 1);
  #else
      // GFX9 wave64: DPP reduction for max (same shift pattern as sum
      // reduction)
      asm("s_nop 0\n\tv_max_f32 %0, %2, %3 row_shr:8 bound_ctrl:0 "
          : "=v"(tmax)
          : "0"(tmax), "v"(tmax), "v"(tmax));
      asm("s_nop 0\n\tv_max_f32 %0, %2, %3 row_shr:4 bound_ctrl:0 "
          : "=v"(tmax)
          : "0"(tmax), "v"(tmax), "v"(tmax));
      asm("s_nop 0\n\tv_max_f32 %0, %2, %3 row_shr:2 bound_ctrl:0 "
          : "=v"(tmax)
          : "0"(tmax), "v"(tmax), "v"(tmax));
      asm("s_nop 0\n\tv_max_f32 %0, %2, %3 wave_shr:1 bound_ctrl:0"
          : "=v"(tmax)
          : "0"(tmax), "v"(tmax), "v"(tmax));
      asm("s_nop 0\n\tv_max_f32 %0, %2, %3 row_bcast:15 bound_ctrl:0"
          : "=v"(tmax)
          : "0"(tmax), "v"(tmax), "v"(tmax));
      asm("s_nop 0\n\tv_max_f32 %0, %2, %3 row_bcast:31 bound_ctrl:0"
          : "=v"(tmax)
          : "0"(tmax), "v"(tmax), "v"(tmax));
  #endif

      // Phase 3: cross-warp max reduction via LDS
      if (threadIdx.x == 0) warp_max[threadIdx.y] = tmax;
      __syncthreads();

      float absmax = 0.0f;
      for (int w = 0; w < WvPrGrp; w++) absmax = fmaxf(absmax, warp_max[w]);
      float inv_s = (absmax > 0.0f) ? 127.0f / absmax : 0.0f;

      // Store scale for epilogue
      if (threadIdx.x == 0 && threadIdx.y == 0)
        a_scales_dyn[n] = (absmax > 0.0f) ? absmax / 127.0f : 0.0f;
      __syncthreads();

      // Phase 4: quantize and store to LDS
      int8_t* s_row = &s[n * K];
      for (uint32_t k = tid; k < (uint32_t)K; k += stride) {
        s_row[k] = float_to_int8_rn(__s2float(A_row[k]) * inv_s);
      }
      __syncthreads();
    }
  } else {
    // --- Mode 1: fused static quantization ---
    // Activations arrive in bf16/fp16 with known per-tensor a_scale.
    const scalar_t* A_fp = (const scalar_t*)A_raw;
    const float inv_scale = 1.0f / (*a_scale);
    for (uint32_t k = 0; k < min_w8a8(K * N, max_lds_ints);
         k += THRDS * WvPrGrp * A_CHUNK) {
      uint32_t k_in = k + ((threadIdx.y * THRDS + threadIdx.x) * A_CHUNK);
      if (k_in >= min_w8a8(K * N, max_lds_ints)) break;
  #pragma unroll
      for (int i = 0; i < A_CHUNK; i++) {
        float val = __s2float(A_fp[k_in + i]);
        s[k_in + i] = float_to_int8_rn(val * inv_scale);
      }
    }
    __syncthreads();
  }

  if (threadIdx.y >= _WvPrGrp) return;

  uint32_t m = (blockIdx.x * _WvPrGrp + (threadIdx.y % _WvPrGrp)) * YTILE;

  // Load per-tensor activation scale once (mode 1 only; mode 2 uses
  // a_scales_dyn)
  const float a_scale_val = (quant_mode != 2) ? *a_scale : 0.0f;

  #if defined(__GFX11__)
  int32_t sum[N][YTILE];
  #else
  float sum[N][YTILE];
  #endif

  while (m < M) {
    for (int i = 0; i < YTILE; i++)
      for (int n = 0; n < N; n++) sum[n][i] = 0;

  #if defined(__GFX11__)
    bigTypeA bigA[N][UNRL];
  #else
    bigTypeAcvt bigA[N][UNRL];
  #endif
    bigTypeW bigB[YTILE][UNRL];

    for (uint32_t k1 = 0; k1 < K; k1 += THRDS * A_CHUNK * UNRL) {
      // Fetch int8 weights from global memory
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;

        const int8_t* B_ = &B[(m + 0) * K + k_];
        for (int y = 0; y < YTILE; y++) {
          // Cast to float* for 4-byte non-temporal loads (not arithmetic).
          const float* src = (const float*)(&B_[y * K]);
  #pragma unroll
          for (int i = 0; i < A_CHUNK / 4; i++)
            bigB[y][k2].f[i] = loadnt((float*)&src[i]);
        }
      }

      // Fetch int8 activations from LDS
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;

        for (int n = 0; n < N; n++) {
  #if defined(__GFX11__)
          // Direct int8 load (no conversion needed for int8 dot product)
          bigA[n][k2] = *((const bigTypeA*)(&(s[k_ + K * n])));
  #else
          bigTypeA rawA = *((const bigTypeA*)(&(s[k_ + K * n])));
            // Convert int8 activations to fp16/bf16
    #pragma unroll
          for (uint32_t b = 0; b < A_CHUNK; b++) {
            bigA[n][k2].h[b] = rawA.b[b];
          }
  #endif
        }
      }

      // Matrix multiply
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;

  #pragma unroll
        for (uint32_t n = 0; n < N; n++) {
  #pragma unroll
          for (int y = 0; y < YTILE; y++) {
  #if defined(__GFX11__)
              // Direct int8x int8 -> int32 dot product (4 elements per
              // instruction)
    #pragma unroll
            for (uint32_t b = 0; b < A_CHUNK / 4; b++) {
              DOT4_I8(sum[n][y], bigA[n][k2].f[b], bigB[y][k2].f[b])
            }
  #else
            // Convert int8 weights to fp16/bf16, then DOT2C
            bigTypeAcvt cvtB;
    #pragma unroll
            for (uint32_t b = 0; b < A_CHUNK; b++) {
              cvtB.h[b] = bigB[y][k2].b[b];
            }
    #pragma unroll
            for (uint32_t b = 0; b < A_CHUNK / 2; b++) {
              DOT2C(sum[n][y], bigA[n][k2].f[b], cvtB.f[b])
            }
  #endif
          }
        }
      }
    }

    // Reduction
  #if defined(__GFX11__)
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
        sum[n][y] += __shfl_xor(sum[n][y], 16);
      }
    }

    if (threadIdx.x == (THRDS - 1)) {
      for (int n = 0; n < N; n++) {
        float a_s = (quant_mode == 2) ? a_scales_dyn[n] : a_scale_val;
        for (int i = 0; i < YTILE; i++) {
          // Convert int32 accumulator to float for scaling
          float sum_f = static_cast<float>(sum[n][i]);
          sum_f *= __s2float(w_scale[m + i]) * a_s;
          if (BIAS) sum_f += __s2float(BIAS[(m + i) % Bx + (n % By) * M]);
          C[m + i + n * M] = __float2s<scalar_t>(sum_f);
        }
      }
    }
  #else   // GFX9 wave64 path
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
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 wave_shr:1 bound_ctrl:0"
            : "=v"(sum[n][y])
            : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_bcast:15 bound_ctrl:0"
            : "=v"(sum[n][y])
            : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_bcast:31 bound_ctrl:0"
            : "=v"(sum[n][y])
            : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
      }
    }

    if (threadIdx.x == 63) {
      for (int n = 0; n < N; n++) {
        float a_s = (quant_mode == 2) ? a_scales_dyn[n] : a_scale_val;
        for (int i = 0; i < YTILE; i++) {
          sum[n][i] *= __s2float(w_scale[m + i]) * a_s;
          if (BIAS) sum[n][i] += __s2float(BIAS[(m + i) % Bx + (n % By) * M]);
          C[m + i + n * M] = __float2s<scalar_t>(sum[n][i]);
        }
      }
    }
  #endif  // defined(__GFX11__)
    m += CuCount * _WvPrGrp * YTILE;
  }
}
#else   // !defined(__HIP__GFX9__) && !defined(__GFX11__)
template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK,
          int UNRL, int N>
__global__ void wvSplitK_w8a8_hf_sml_(
    const int K, const int M, const int Bx, const int By, const int8_t* B,
    const void* __restrict__ A_raw, const scalar_t* w_scale,
    const float* __restrict__ a_scale, const scalar_t* __restrict__ BIAS,
    scalar_t* C, const int _WvPrGrp, const int CuCount, const int quant_mode) {
  UNREACHABLE_CODE
}
#endif  // defined(__HIP__GFX9__) || defined(__GFX11__)
