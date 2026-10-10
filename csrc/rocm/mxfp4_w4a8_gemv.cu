// MXFP4 x int8 (W4A8) GEMV for RDNA3/RDNA3.5 (gfx11) small-batch decode.
//
// Activations are quantized to int8 per 32-element group (symmetric, one fp32
// scale per group), E2M1 weight codes map exactly to int8 (2 * e2m1), and each
// 32-element group is reduced with eight v_dot4_i32_iu8 instructions. One fp32
// multiply per group applies the E8M0 weight scale and the activation scale.
// The scheme follows ggml's MMVQ / q8_1 path (llama.cpp, MIT License).
//
// Layouts (Quark OCP MX checkpoints, no repack):
//   a        [M, K]     fp16 / bf16, 1 <= M <= 8
//   b_q      [N, K/2]   uint8, two E2M1 codes per byte, low nibble = even k
//   b_scale  [N, K/32]  uint8 E8M0
//   out      [M, N]     same dtype as a

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <string>

#include "../cuda_compat.h"

#if defined(__GFX11__)
  #define __HIP__MXFP4_W4A8_GFX11__
#endif

namespace {

constexpr int kMaxM = 8;
constexpr int kGroup = 32;
constexpr int kWarpsPerBlock = 8;

template <typename T>
__device__ __forceinline__ float to_float(T v);
template <>
__device__ __forceinline__ float to_float(half v) {
  return __half2float(v);
}
template <>
__device__ __forceinline__ float to_float(__hip_bfloat16 v) {
  return __bfloat162float(v);
}

template <typename T>
__device__ __forceinline__ T from_float(float v);
template <>
__device__ __forceinline__ half from_float(float v) {
  return __float2half(v);
}
template <>
__device__ __forceinline__ __hip_bfloat16 from_float(float v) {
  return __float2bfloat16(v);
}

#if defined(__HIP__MXFP4_W4A8_GFX11__) || !defined(__HIP_DEVICE_COMPILE__)

// 2 * E2M1 value for each 4-bit code (sign in bit 3). Exact in int8.
__device__ __constant__ int8_t kE2M1x2[16] = {0, 1,  2,  3,  4,  6,  8,  12,
                                              0, -1, -2, -3, -4, -6, -8, -12};

template <typename T>
__global__ void mxfp4_w4a8_quant_act_kernel(const T* __restrict__ a,
                                            int8_t* __restrict__ a_q,
                                            float* __restrict__ a_s, int M,
                                            int K) {
  const int groups = K / kGroup;
  const int m = blockIdx.x;
  const int g = blockIdx.y * blockDim.x + threadIdx.x;
  if (m >= M || g >= groups) return;

  const T* row = a + (int64_t)m * K + g * kGroup;
  float v[kGroup];
  float amax = 0.f;
  #pragma unroll
  for (int i = 0; i < kGroup; ++i) {
    v[i] = to_float(row[i]);
    amax = fmaxf(amax, fabsf(v[i]));
  }
  const float scale = fmaxf(amax / 127.0f, 1e-12f);
  a_s[(int64_t)m * groups + g] = scale;

  int8_t* out = a_q + (int64_t)m * K + g * kGroup;
  #pragma unroll
  for (int i = 0; i < kGroup; ++i) {
    out[i] = (int8_t)fmaxf(-127.f, fminf(127.f, rintf(v[i] / scale)));
  }
}

// One wave (32 lanes) per output column n. Lanes stride over the K groups;
// each lane decodes its weight group once and reuses it for all M rows.
template <int M, typename T>
__global__ void mxfp4_w4a8_gemv_kernel(const int* __restrict__ a_q,
                                       const float* __restrict__ a_s,
                                       const uint8_t* __restrict__ b_q,
                                       const uint8_t* __restrict__ b_scale,
                                       T* __restrict__ out, int N, int K) {
  const int groups = K / kGroup;
  const int lane = threadIdx.x;
  const int n = blockIdx.x * blockDim.y + threadIdx.y;
  if (n >= N) return;

  const uint8_t* w_row = b_q + (int64_t)n * (K / 2);
  const uint8_t* s_row = b_scale + (int64_t)n * groups;

  float acc[M];
  #pragma unroll
  for (int m = 0; m < M; ++m) acc[m] = 0.f;

  for (int g = lane; g < groups; g += 32) {
    const uint8_t* gp = w_row + g * (kGroup / 2);
    int w4[8];
  #pragma unroll
    for (int j = 0; j < 8; ++j) {
      const uint32_t two = gp[2 * j] | (gp[2 * j + 1] << 8);
      const int w0 = kE2M1x2[two & 0xF];
      const int w1 = kE2M1x2[(two >> 4) & 0xF];
      const int w2 = kE2M1x2[(two >> 8) & 0xF];
      const int w3 = kE2M1x2[(two >> 12) & 0xF];
      w4[j] = (w0 & 0xFF) | ((w1 & 0xFF) << 8) | ((w2 & 0xFF) << 16) |
              ((w3 & 0xFF) << 24);
    }
    // 2^(e - 127) / 2: undoes the factor 2 folded into kE2M1x2.
    const float w_scale = __int_as_float((int)s_row[g] << 23) * 0.5f;
  #pragma unroll
    for (int m = 0; m < M; ++m) {
      const int* ap = a_q + (int64_t)m * (K / 4) + g * (kGroup / 4);
      int isum = 0;
  #pragma unroll
      for (int j = 0; j < 8; ++j) {
        isum = __builtin_amdgcn_sudot4(true, w4[j], true, ap[j], isum, false);
      }
      acc[m] += (float)isum * (w_scale * a_s[(int64_t)m * groups + g]);
    }
  }

  #pragma unroll
  for (int m = 0; m < M; ++m) {
    float v = acc[m];
  #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += VLLM_SHFL_XOR_SYNC_WIDTH(v, o, 32);
    if (lane == 0) out[(int64_t)m * N + n] = from_float<T>(v);
  }
}

#else  // non-gfx11 device pass: empty kernels for symbol parity.

template <typename T>
__global__ void mxfp4_w4a8_quant_act_kernel(const T*, int8_t*, float*, int,
                                            int) {}

template <int M, typename T>
__global__ void mxfp4_w4a8_gemv_kernel(const int*, const float*, const uint8_t*,
                                       const uint8_t*, T*, int, int) {}

#endif  // __HIP__MXFP4_W4A8_GFX11__ || !__HIP_DEVICE_COMPILE__

bool mxfp4_w4a8_on_gfx11() {
  static const bool result = [] {
    const auto* dprops = at::cuda::getCurrentDeviceProperties();
    const std::string arch = dprops->gcnArchName;
    return arch.rfind("gfx11", 0) == 0;
  }();
  return result;
}

template <typename T>
void launch(const at::Tensor& a, const at::Tensor& b_q,
            const at::Tensor& b_scale, at::Tensor& out, int M, int N, int K) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int groups = K / kGroup;

  auto a_q = torch::empty({M, K}, a.options().dtype(torch::kInt8));
  auto a_s = torch::empty({M, groups}, a.options().dtype(torch::kFloat32));

  const dim3 q_block(256);
  const dim3 q_grid(M, (groups + 255) / 256);
  mxfp4_w4a8_quant_act_kernel<T><<<q_grid, q_block, 0, stream>>>(
      reinterpret_cast<const T*>(a.data_ptr()), a_q.data_ptr<int8_t>(),
      a_s.data_ptr<float>(), M, K);

  const dim3 block(32, kWarpsPerBlock);
  const dim3 grid((N + kWarpsPerBlock - 1) / kWarpsPerBlock);
  const int* a_q32 = reinterpret_cast<const int*>(a_q.data_ptr<int8_t>());
  const float* a_s_ptr = a_s.data_ptr<float>();
  const uint8_t* b_q_ptr = b_q.data_ptr<uint8_t>();
  const uint8_t* b_s_ptr = b_scale.data_ptr<uint8_t>();
  T* out_ptr = reinterpret_cast<T*>(out.data_ptr());

#define MXFP4_W4A8_LAUNCH(MM)                                  \
  case MM:                                                     \
    mxfp4_w4a8_gemv_kernel<MM, T><<<grid, block, 0, stream>>>( \
        a_q32, a_s_ptr, b_q_ptr, b_s_ptr, out_ptr, N, K);      \
    break;

  switch (M) {
    MXFP4_W4A8_LAUNCH(1)
    MXFP4_W4A8_LAUNCH(2)
    MXFP4_W4A8_LAUNCH(3)
    MXFP4_W4A8_LAUNCH(4)
    MXFP4_W4A8_LAUNCH(5)
    MXFP4_W4A8_LAUNCH(6)
    MXFP4_W4A8_LAUNCH(7)
    MXFP4_W4A8_LAUNCH(8)
  }
#undef MXFP4_W4A8_LAUNCH
}

}  // namespace

torch::Tensor mxfp4_w4a8_gemv(const at::Tensor& a, const at::Tensor& b_q,
                              const at::Tensor& b_scale) {
  TORCH_CHECK(a.is_cuda() && b_q.is_cuda() && b_scale.is_cuda(),
              "mxfp4_w4a8_gemv: all inputs must be on the GPU");
  TORCH_CHECK(mxfp4_w4a8_on_gfx11(),
              "mxfp4_w4a8_gemv requires an RDNA3/RDNA3.5 (gfx11) GPU");
  TORCH_CHECK(a.scalar_type() == at::kHalf || a.scalar_type() == at::kBFloat16,
              "mxfp4_w4a8_gemv: a must be float16 or bfloat16");
  TORCH_CHECK(
      b_q.scalar_type() == at::kByte && b_scale.scalar_type() == at::kByte,
      "mxfp4_w4a8_gemv: b_q and b_scale must be uint8");
  TORCH_CHECK(a.dim() == 2 && b_q.dim() == 2 && b_scale.dim() == 2,
              "mxfp4_w4a8_gemv: inputs must be 2D");
  TORCH_CHECK(
      a.is_contiguous() && b_q.is_contiguous() && b_scale.is_contiguous(),
      "mxfp4_w4a8_gemv: inputs must be contiguous");

  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b_q.size(0);
  TORCH_CHECK(M >= 1 && M <= kMaxM,
              "mxfp4_w4a8_gemv supports 1 <= M <= ", kMaxM, ", got M=", M);
  TORCH_CHECK(K % kGroup == 0, "mxfp4_w4a8_gemv: K must be a multiple of 32");
  TORCH_CHECK(b_q.size(1) * 2 == K, "mxfp4_w4a8_gemv: b_q must be [N, K/2]");
  TORCH_CHECK(b_scale.size(0) == N && b_scale.size(1) == K / kGroup,
              "mxfp4_w4a8_gemv: b_scale must be [N, K/32]");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto out = torch::empty({M, N}, a.options());

  if (a.scalar_type() == at::kHalf) {
    launch<half>(a, b_q, b_scale, out, M, N, K);
  } else {
    launch<__hip_bfloat16>(a, b_q, b_scale, out, M, N, K);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
