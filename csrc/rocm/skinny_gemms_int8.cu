#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>
#include <algorithm>

#include "../cuda_compat.h"
#include "dispatch_utils.h"

#if defined(__HIPCC__) && \
    (defined(__gfx90a__) || defined(__gfx942__) || defined(__gfx950__))
  #define __HIP__GFX9__
#endif

#define LDS_SIZE 64 * 1024

int get_lds_size_int8() {
  static bool is_cached = false;
  static int result;
  if (is_cached == false) {
    auto dprops = at::cuda::getCurrentDeviceProperties();
    std::string device_arch = dprops->gcnArchName;
    size_t substring = device_arch.find("gfx95");
    result = (substring == std::string::npos ? 64 * 1024 : 160 * 1024);
    is_cached = true;
  }
  return result;
}

bool is_gfx11_int8() {
  static const bool result = [] {
    auto dprops = at::cuda::getCurrentDeviceProperties();
    std::string device_arch = dprops->gcnArchName;
    return device_arch.find("gfx11") != std::string::npos;
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

__device__ inline unsigned int min__(uint32_t a, uint32_t b) {
  return min(a, b);
}

// W8A16 skinny GEMM kernel: int8 weights, fp16/bf16 activations
// Targets the "sml" case where activations fit in LDS.
// A_CHUNK=16: each thread processes 16 int8 weight elements per step.
#if defined(__HIP__GFX9__) || defined(__GFX11__)
template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK,
          int UNRL, int N>
__global__ void __launch_bounds__(WvPrGrp* THRDS)
    wvSplitK_int8_hf_sml_(const int K, const int M, const int Bx, const int By,
                          const int8_t* B, const scalar_t* __restrict__ A,
                          const scalar_t* scale,
                          const scalar_t* __restrict__ BIAS, scalar_t* C,
                          const int _WvPrGrp, const int CuCount) {
  constexpr int max_lds_len = LDS_SIZE / 2;

  // Activation union: 16 fp16/bf16 values = 32 bytes
  union bigTypeA {
    scalar_t h[A_CHUNK];
    float f[A_CHUNK / 2];
  };

  // Weight union: 16 int8 values = 16 bytes
  union bigTypeW {
    int8_t b[A_CHUNK];
    float f[A_CHUNK / 4];
  };

  __shared__ scalar_t s[max_lds_len];

  // Fetch activation matrix to LDS
  // Each thread fetches A_CHUNK fp16 elements = 32 bytes
  for (uint32_t k = 0; k < min__(K * N, max_lds_len);
       k += THRDS * WvPrGrp * A_CHUNK) {
    uint32_t k_in = k + ((threadIdx.y * THRDS + threadIdx.x) * A_CHUNK);

    if (k_in >= min__(K * N, max_lds_len)) break;

    *((bigTypeA*)(&s[k_in])) = *((bigTypeA*)(&A[k_in]));
  }
  __syncthreads();

  if (threadIdx.y >= _WvPrGrp) return;

  uint32_t m = (blockIdx.x * _WvPrGrp + (threadIdx.y % _WvPrGrp)) * YTILE;

  float sum[N][YTILE];

  while (m < M) {
    for (int i = 0; i < YTILE; i++)
      for (int n = 0; n < N; n++) sum[n][i] = 0;

    bigTypeA bigA[N][UNRL];
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
          // 16 bytes = 4 floats worth of int8 data
          const float* src = (const float*)(&B_[y * K]);
  #pragma unroll
          for (int i = 0; i < A_CHUNK / 4; i++)
            bigB[y][k2].f[i] = loadnt((float*)&src[i]);
        }
      }

      // Fetch fp16/bf16 activations from LDS
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;

        for (int n = 0; n < N; n++) {
          bigA[n][k2] = *((const bigTypeA*)(&(s[k_ + K * n])));
        }
      }

      // Matrix multiply: convert int8 weight pairs to fp16, then DOT2C
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;

  #pragma unroll
        for (uint32_t n = 0; n < N; n++) {
  #pragma unroll
          for (int y = 0; y < YTILE; y++) {
            // Convert 16 int8 weights to 8 fp16 pairs stored in a bigTypeA
            // union
            bigTypeA cvtB;
  #pragma unroll
            for (uint32_t b = 0; b < A_CHUNK; b++) {
              cvtB.h[b] = bigB[y][k2].b[b];
            }
  #pragma unroll
            for (uint32_t b = 0; b < A_CHUNK / 2; b++) {
              DOT2C(sum[n][y], bigA[n][k2].f[b], cvtB.f[b])
            }
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
        for (int i = 0; i < YTILE; i++) {
          sum[n][i] *= __s2float(scale[m + i]);
          if (BIAS) sum[n][i] += __s2float(BIAS[(m + i) % Bx + (n % By) * M]);
          C[m + i + n * M] = __float2s<scalar_t>(sum[n][i]);
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
        for (int i = 0; i < YTILE; i++) {
          sum[n][i] *= __s2float(scale[m + i]);
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
__global__ void wvSplitK_int8_hf_sml_(const int K, const int M, const int Bx,
                                      const int By, const int8_t* B,
                                      const scalar_t* __restrict__ A,
                                      const scalar_t* scale,
                                      const scalar_t* __restrict__ BIAS,
                                      scalar_t* C, const int _WvPrGrp,
                                      const int CuCount) {
  UNREACHABLE_CODE
}
#endif  // defined(__HIP__GFX9__) || defined(__GFX11__)

int mindiv_int8(int N, int div1, int div2) {
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

torch::Tensor wvSplitK_int8(const at::Tensor& in_a, const at::Tensor& in_b,
                            const at::Tensor& in_scale,
                            const std::optional<at::Tensor>& in_bias,
                            const int64_t CuCount) {
  auto M_in = in_a.size(0);
  auto K_in = in_a.size(1);
  auto N_in = in_b.size(0);
  auto Bx_in =
      (in_bias.has_value() && in_bias->numel() > 0)
          ? (in_bias->sizes().size() == 2) ? in_bias->size(1) : in_bias->size(0)
          : 1;
  auto By_in = (in_bias.has_value() && in_bias->numel() > 0 &&
                in_bias->sizes().size() == 2)
                   ? in_bias->size(0)
                   : 1;

  TORCH_CHECK(in_a.dtype() == torch::kInt8, "Weight must be int8");
  TORCH_CHECK(
      in_b.dtype() == torch::kFloat16 || in_b.dtype() == torch::kBFloat16,
      "Activation must be float16 or bfloat16");
  TORCH_CHECK(in_scale.dtype() == in_b.dtype(),
              "Scale dtype must match activation dtype");
  TORCH_CHECK(in_scale.size(0) == M_in, "Scale size must match M");
  TORCH_CHECK(K_in % 16 == 0, "K must be divisible by 16 for int8 kernel");

  const int max_lds_len = get_lds_size_int8() / 2;
  TORCH_CHECK(K_in * N_in <= max_lds_len,
              "K*N exceeds LDS capacity; only sml variant is supported. "
              "K=",
              K_in, " N=", N_in, " K*N=", K_in * N_in, " max=", max_lds_len);

  auto out_c = torch::empty(
      {N_in, M_in},
      torch::TensorOptions().dtype(in_b.dtype()).device(in_b.device()));

  dim3 grid(CuCount);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_a));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

#define WVSPLITK_INT8_LAUNCH(_THRDS, _YTILE, _UNRL, _N)                        \
  {                                                                            \
    dim3 block(_THRDS, 16);                                                    \
    int __wvPrGrp = mindiv_int8(M_in, CuCount * _YTILE, 16);                   \
    TORCH_CHECK(M_in % _YTILE == 0, "M must be divisible by YTILE=", _YTILE);  \
    wvSplitK_int8_hf_sml_<fptype, _THRDS, _YTILE, 16, 16, _UNRL, _N>           \
        <<<grid, block, 0, stream>>>(K_in, M_in, Bx_in, By_in, wptr, aptr,     \
                                     sptr, biasptr, cptr, __wvPrGrp, CuCount); \
  }

#define WVSPLITK_INT8(_YTILE, _UNRL, _N)        \
  if (is_gfx11_int8())                          \
    WVSPLITK_INT8_LAUNCH(32, _YTILE, _UNRL, _N) \
  else                                          \
    WVSPLITK_INT8_LAUNCH(64, _YTILE, _UNRL, _N)

#define WVSPLIT_INT8_TILE(_sYT, __N) \
  {                                  \
    if (__N >= 4 && _sYT >= 480)     \
      WVSPLITK_INT8(4, 1, __N)       \
    else                             \
      WVSPLITK_INT8(1, 4, __N)       \
  }

  AT_DISPATCH_REDUCED_FLOATING_TYPES(in_b.scalar_type(), "wvSplitK_int8", [&] {
    using fptype = typename scalar<scalar_t>::type;
    const int8_t* wptr = in_a.data_ptr<int8_t>();
    const fptype* aptr = reinterpret_cast<const fptype*>(in_b.data_ptr());
    const fptype* sptr = reinterpret_cast<const fptype*>(in_scale.data_ptr());
    const fptype* biasptr =
        (in_bias.has_value() && in_bias->numel() > 0)
            ? reinterpret_cast<const fptype*>(in_bias->data_ptr())
            : nullptr;
    fptype* cptr = reinterpret_cast<fptype*>(out_c.data_ptr());

    int sYT = (M_in + CuCount * 4 - 1) / (CuCount * 4);

    switch (N_in) {
      case 1:
        WVSPLIT_INT8_TILE(sYT, 1)
        break;
      case 2:
        WVSPLIT_INT8_TILE(sYT, 2)
        break;
      case 3:
        WVSPLIT_INT8_TILE(sYT, 3)
        break;
      case 4:
        WVSPLIT_INT8_TILE(sYT, 4)
        break;
      case 5:
        WVSPLIT_INT8_TILE(sYT, 5)
        break;
      default:
        throw std::runtime_error(
            "Unsupported N value: " + std::to_string(M_in) + "," +
            std::to_string(K_in) + "," + std::to_string(N_in));
    }
  });

#undef WVSPLITK_INT8_LAUNCH
#undef WVSPLITK_INT8
#undef WVSPLIT_INT8_TILE

  return out_c;
}

// Sweep function disabled by default to reduce compile time.
// Build with -DVLLM_SKINNY_GEMM_SWEEP to enable.
#ifdef VLLM_SKINNY_GEMM_SWEEP
torch::Tensor wvSplitK_int8_sweep(const at::Tensor& in_a,
                                  const at::Tensor& in_b,
                                  const at::Tensor& in_scale,
                                  const std::optional<at::Tensor>& in_bias,
                                  const int64_t CuCount, const int64_t ytile,
                                  const int64_t unrl, const int64_t achunk,
                                  const int64_t wvprgrp) {
  auto M_in = in_a.size(0);
  auto K_in = in_a.size(1);
  auto N_in = in_b.size(0);

  TORCH_CHECK(in_a.dtype() == torch::kInt8, "Weight must be int8");
  TORCH_CHECK(in_b.dtype() == torch::kFloat16,
              "Sweep only supports float16 activations");
  TORCH_CHECK(in_scale.dtype() == torch::kFloat16,
              "Sweep only supports float16 scale");
  TORCH_CHECK(in_scale.size(0) == M_in, "Scale size must match M");
  TORCH_CHECK(K_in % achunk == 0, "K must be divisible by achunk=", achunk);
  TORCH_CHECK(M_in % ytile == 0, "M must be divisible by ytile=", ytile);

  const int max_lds_len = get_lds_size_int8() / 2;
  TORCH_CHECK(K_in * N_in <= max_lds_len, "K*N exceeds LDS capacity. K=", K_in,
              " N=", N_in);

  auto out_c = torch::empty(
      {N_in, M_in},
      torch::TensorOptions().dtype(in_b.dtype()).device(in_b.device()));

  dim3 grid(CuCount);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_a));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  using fptype = half;
  const int8_t* wptr = in_a.data_ptr<int8_t>();
  const fptype* aptr = reinterpret_cast<const fptype*>(in_b.data_ptr());
  const fptype* sptr = reinterpret_cast<const fptype*>(in_scale.data_ptr());
  const fptype* biasptr = nullptr;
  fptype* cptr = reinterpret_cast<fptype*>(out_c.data_ptr());

  const int THRDS = is_gfx11_int8() ? 32 : 64;

  #define SWEEP_LAUNCH(_THRDS, _YTILE, _WVPRGRP, _ACHUNK, _UNRL, _N)          \
    {                                                                         \
      dim3 block(_THRDS, _WVPRGRP);                                           \
      int __wvPrGrp = mindiv_int8(M_in, CuCount * _YTILE, _WVPRGRP);          \
      wvSplitK_int8_hf_sml_<fptype, _THRDS, _YTILE, _WVPRGRP, _ACHUNK, _UNRL, \
                            _N>                                               \
          <<<grid, block, 0, stream>>>(K_in, M_in, 1, 1, wptr, aptr, sptr,    \
                                       biasptr, cptr, __wvPrGrp, CuCount);    \
    }

  #define SWEEP_N(_THRDS, _YTILE, _WVPRGRP, _ACHUNK, _UNRL)              \
    switch (N_in) {                                                      \
      case 1:                                                            \
        SWEEP_LAUNCH(_THRDS, _YTILE, _WVPRGRP, _ACHUNK, _UNRL, 1) break; \
      case 2:                                                            \
        SWEEP_LAUNCH(_THRDS, _YTILE, _WVPRGRP, _ACHUNK, _UNRL, 2) break; \
      case 3:                                                            \
        SWEEP_LAUNCH(_THRDS, _YTILE, _WVPRGRP, _ACHUNK, _UNRL, 3) break; \
      case 4:                                                            \
        SWEEP_LAUNCH(_THRDS, _YTILE, _WVPRGRP, _ACHUNK, _UNRL, 4) break; \
      default:                                                           \
        TORCH_CHECK(false, "Unsupported N=", N_in);                      \
    }

  #define SWEEP_UNRL(_THRDS, _YTILE, _WVPRGRP, _ACHUNK) \
    if (unrl == 1) {                                    \
      SWEEP_N(_THRDS, _YTILE, _WVPRGRP, _ACHUNK, 1)     \
    } else if (unrl == 2) {                             \
      SWEEP_N(_THRDS, _YTILE, _WVPRGRP, _ACHUNK, 2)     \
    } else if (unrl == 4) {                             \
      SWEEP_N(_THRDS, _YTILE, _WVPRGRP, _ACHUNK, 4)     \
    } else {                                            \
      TORCH_CHECK(false, "Unsupported unrl=", unrl);    \
    }

  #define SWEEP_YTILE(_THRDS, _WVPRGRP, _ACHUNK)       \
    if (ytile == 1) {                                  \
      SWEEP_UNRL(_THRDS, 1, _WVPRGRP, _ACHUNK)         \
    } else if (ytile == 2) {                           \
      SWEEP_UNRL(_THRDS, 2, _WVPRGRP, _ACHUNK)         \
    } else if (ytile == 4) {                           \
      SWEEP_UNRL(_THRDS, 4, _WVPRGRP, _ACHUNK)         \
    } else {                                           \
      TORCH_CHECK(false, "Unsupported ytile=", ytile); \
    }

  #define SWEEP_WVPRGRP(_THRDS, _ACHUNK)                   \
    if (wvprgrp == 8) {                                    \
      SWEEP_YTILE(_THRDS, 8, _ACHUNK)                      \
    } else if (wvprgrp == 12) {                            \
      SWEEP_YTILE(_THRDS, 12, _ACHUNK)                     \
    } else if (wvprgrp == 16) {                            \
      SWEEP_YTILE(_THRDS, 16, _ACHUNK)                     \
    } else {                                               \
      TORCH_CHECK(false, "Unsupported wvprgrp=", wvprgrp); \
    }

  if (THRDS == 32) {
    if (achunk == 8) {
      SWEEP_WVPRGRP(32, 8)
    } else if (achunk == 16) {
      SWEEP_WVPRGRP(32, 16)
    } else if (achunk == 32) {
      SWEEP_WVPRGRP(32, 32)
    } else {
      TORCH_CHECK(false, "Unsupported achunk=", achunk);
    }
  } else {
    if (achunk == 8) {
      SWEEP_WVPRGRP(64, 8)
    } else if (achunk == 16) {
      SWEEP_WVPRGRP(64, 16)
    } else if (achunk == 32) {
      SWEEP_WVPRGRP(64, 32)
    } else {
      TORCH_CHECK(false, "Unsupported achunk=", achunk);
    }
  }

  #undef SWEEP_LAUNCH
  #undef SWEEP_N
  #undef SWEEP_UNRL
  #undef SWEEP_YTILE
  #undef SWEEP_WVPRGRP

  return out_c;
}
#endif  // VLLM_SKINNY_GEMM_SWEEP
