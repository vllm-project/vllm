#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <string>
#include <type_traits>
#include <utility>  // std::in_range

#include "core/cuda_compat.h"

// Combined RDNA macro (gfx11 + gfx12) - both use 32-wide wavefronts
#if defined(__HIPCC__) && (defined(__GFX11__) || defined(__GFX12__))
  #define __HIP__GFX1X__
#endif

static constexpr int LDS_SIZE = 64 * 1024;

static bool on_gfx1x_int8() {
  static const bool result = [] {
    const std::string arch =
        at::cuda::getCurrentDeviceProperties()->gcnArchName;
    return arch.rfind("gfx11", 0) == 0 || arch.rfind("gfx12", 0) == 0;
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
    typedef short __attribute__((ext_vector_type(2))) bf16x2_t;             \
    V0 = __builtin_amdgcn_fdot2_f32_bf16(*((bf16x2_t*)(&(V2))),             \
                                         *((bf16x2_t*)(&(V3))), V0, false); \
  }

__device__ inline unsigned int min__(uint32_t a, uint32_t b) {
  return min(a, b);
}

// DPP-based wave32 reduction for GFX1X (matches skinny_gemms.cu).
// row_shr within 16-lane rows, then cross-row __shfl_xor.
// Result accumulates in lane (THRDS - 1).
#if defined(__HIP__GFX1X__)
  #define REDUCE_SUM_DPP_WAVE32(val)                                        \
    do {                                                                    \
      val += __builtin_amdgcn_mov_dpp(val, 0x118, 0xf, 0xf, 1); /* shr8  */ \
      val += __builtin_amdgcn_mov_dpp(val, 0x114, 0xf, 0xf, 1); /* shr4  */ \
      val += __builtin_amdgcn_mov_dpp(val, 0x112, 0xf, 0xf, 1); /* shr2  */ \
      val += __builtin_amdgcn_mov_dpp(val, 0x111, 0xf, 0xf, 1); /* shr1  */ \
      val += __shfl_xor(val, 16);                                           \
    } while (0)
#endif

// W8A16 skinny GEMM kernel: int8 weights, fp16/bf16 activations
// Targets the "sml" case where activations fit in LDS.
// A_CHUNK=16: each thread processes 16 int8 weight elements per step.
// Per-channel scale [M], applied once per output row after the K reduction
#if defined(__HIP__GFX1X__)
template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK,
          int UNRL, int N>
__global__ void __launch_bounds__(WvPrGrp* THRDS)
    wvSplitK_int8_hf_sml_(const int K, const int M, const int8_t* B,
                          const scalar_t* __restrict__ A, const scalar_t* scale,
                          const scalar_t* __restrict__ BIAS, scalar_t* C,
                          const int _WvPrGrp, const int CuCount,
                          const int B_row_stride) {
  constexpr int max_lds_len = LDS_SIZE / 2;

  // Activation union: 16 fp16/bf16 values = 32 bytes
  union bigTypeA {
    scalar_t h[A_CHUNK];
    float f[A_CHUNK / 2];
    uint32_t u[A_CHUNK / 2];
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

        const int8_t* B_ = B + (size_t)m * B_row_stride + k_;
        for (int y = 0; y < YTILE; y++) {
          // 16 bytes = 4 floats worth of int8 data
          const float* src = (const float*)(B_ + (size_t)y * B_row_stride);
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
            if constexpr (std::is_same_v<scalar_t, __hip_bfloat16>) {
              // gfx11 has no fp32->bf16 convert, so the default
              // round-to-nearest is emulated (~4 extra VALU per weight);
              // truncating is exact for every int8 value.
  #pragma unroll
              for (uint32_t b = 0; b < A_CHUNK / 2; b++) {
                const uint32_t lo =
                    __float_as_uint(static_cast<float>(bigB[y][k2].b[2 * b]));
                const uint32_t hi = __float_as_uint(
                    static_cast<float>(bigB[y][k2].b[2 * b + 1]));
                cvtB.u[b] = (hi & 0xFFFF0000u) | (lo >> 16);
              }
            } else {
  #pragma unroll
              for (uint32_t b = 0; b < A_CHUNK; b++) {
                cvtB.h[b] = bigB[y][k2].b[b];
              }
            }

  #pragma unroll
            for (uint32_t b = 0; b < A_CHUNK / 2; b++) {
              DOT2C(sum[n][y], bigA[n][k2].f[b], cvtB.f[b])
            }
          }
        }
      }
    }

    for (int n = 0; n < N; n++)
      for (int y = 0; y < YTILE; y++) REDUCE_SUM_DPP_WAVE32(sum[n][y]);

    if (threadIdx.x == (THRDS - 1)) {
      for (int n = 0; n < N; n++) {
        for (int i = 0; i < YTILE; i++) {
          sum[n][i] *= __s2float(scale[m + i]);
          if (BIAS) sum[n][i] += __s2float(BIAS[m + i]);
          C[m + i + n * M] = __float2s<scalar_t>(sum[n][i]);
        }
      }
    }
    m += CuCount * _WvPrGrp * YTILE;
  }
}
#else   // !defined(__HIP__GFX1X__)
template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK,
          int UNRL, int N>
__global__ void wvSplitK_int8_hf_sml_(
    const int K, const int M, const int8_t* B, const scalar_t* __restrict__ A,
    const scalar_t* scale, const scalar_t* __restrict__ BIAS, scalar_t* C,
    const int _WvPrGrp, const int CuCount, const int B_row_stride) {
  UNREACHABLE_CODE
}
#endif  // defined(__HIP__GFX1X__)

static int mindiv_int8(int N, int div1, int div2) {
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
  TORCH_CHECK(on_gfx1x_int8(), "wvSplitK_int8 supports gfx11/gfx12 only");
  TORCH_CHECK(in_a.dim() == 2 && in_b.dim() == 2,
              "Weight and activation must be 2-D, got ", in_a.sizes(), " and ",
              in_b.sizes());
  TORCH_CHECK(
      in_a.device() == in_b.device() && in_scale.device() == in_b.device(),
      "Weight, activation and scale must be on the same device");

  const auto M_in = in_a.size(0);
  const auto K_in = in_b.size(1);
  const auto N_in = in_b.size(0);

  TORCH_CHECK(in_a.dtype() == torch::kInt8, "Weight must be int8, got ",
              in_a.dtype());
  TORCH_CHECK(
      in_b.dtype() == torch::kFloat16 || in_b.dtype() == torch::kBFloat16,
      "Activation must be float16 or bfloat16, got ", in_b.dtype());
  TORCH_CHECK(in_a.size(1) == K_in, "Weight K (", in_a.size(1),
              ") must match activation K (", K_in, ")");
  TORCH_CHECK(K_in % 16 == 0, "K must be divisible by 16, got ", K_in);
  // The kernel stages activations into LDS as one dense [N, K] block.
  TORCH_CHECK(in_b.is_contiguous(), "Activation must be contiguous");

  TORCH_CHECK(in_a.stride(1) == 1, "Weight rows must be contiguous");
  const int64_t b_row_stride = in_a.stride(0);  // int8: elements == bytes
  TORCH_CHECK(b_row_stride >= K_in, "Weight row stride (", b_row_stride,
              ") must be at least K=", K_in);
  TORCH_CHECK(b_row_stride % 16 == 0,
              "Weight row stride must be a multiple of 16, got ", b_row_stride);
  TORCH_CHECK(std::in_range<int>(b_row_stride), "Weight row stride (",
              b_row_stride, ") exceeds int range");

  TORCH_CHECK(in_scale.dtype() == in_b.dtype(),
              "Scale dtype must match activation dtype");
  TORCH_CHECK(in_scale.dim() == 1 && in_scale.size(0) == M_in,
              "Per-channel scale must be [M] = [", M_in, "], got ",
              in_scale.sizes());
  TORCH_CHECK(in_scale.is_contiguous(), "Scale must be contiguous");

  const bool has_bias = in_bias.has_value() && in_bias->numel() > 0;
  if (has_bias) {
    TORCH_CHECK(in_bias->dtype() == in_b.dtype(),
                "Bias dtype must match activation dtype");
    TORCH_CHECK(in_bias->device() == in_b.device(),
                "Bias must be on the activation's device");
    TORCH_CHECK(in_bias->dim() == 1 && in_bias->size(0) == M_in,
                "Bias must be [M] = [", M_in, "], got ", in_bias->sizes());
    TORCH_CHECK(in_bias->is_contiguous(), "Bias must be contiguous");
  }

  const int max_lds_len = LDS_SIZE / 2;
  TORCH_CHECK(K_in * N_in <= max_lds_len, "K*N exceeds LDS capacity: K=", K_in,
              " N=", N_in, " K*N=", K_in * N_in, " max=", max_lds_len);

  auto out_c = torch::empty(
      {N_in, M_in},
      torch::TensorOptions().dtype(in_b.dtype()).device(in_b.device()));

  dim3 grid(CuCount);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_a));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

#define WVSPLITK_INT8_LAUNCH(_THRDS, _YTILE, _UNRL, _N)                       \
  {                                                                           \
    dim3 block(_THRDS, 16);                                                   \
    int __wvPrGrp = mindiv_int8(M_in, CuCount * _YTILE, 16);                  \
    TORCH_CHECK(M_in % _YTILE == 0, "M must be divisible by YTILE=", _YTILE); \
    wvSplitK_int8_hf_sml_<fptype, _THRDS, _YTILE, 16, 16, _UNRL, _N>          \
        <<<grid, block, 0, stream>>>(K_in, M_in, wptr, aptr, sptr, biasptr,   \
                                     cptr, __wvPrGrp, CuCount,                \
                                     static_cast<int>(b_row_stride));         \
  }

#define WVSPLITK_INT8(_YTILE, _UNRL, _N) \
  WVSPLITK_INT8_LAUNCH(32, _YTILE, _UNRL, _N)

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
        has_bias ? reinterpret_cast<const fptype*>(in_bias->data_ptr())
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
        TORCH_CHECK(false, "wvSplitK_int8 supports N <= 5, got N=", N_in);
    }
  });

#undef WVSPLITK_INT8_LAUNCH
#undef WVSPLITK_INT8
#undef WVSPLIT_INT8_TILE

  return out_c;
}
