#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "../cuda_compat.h"
#include "dispatch_utils.h"
#include "quantization/w8a8/fp8/common.cuh"

namespace {

constexpr int kWaveSize = 32;
constexpr int kWavesPerWG = 16;
[[maybe_unused]] constexpr int kThreadsPerWG = kWaveSize * kWavesPerWG;
[[maybe_unused]] constexpr int kAChunk = 16;
[[maybe_unused]] constexpr int kYTile = 2;
constexpr int kFixedK = 4096;
constexpr int kFixedKGroups = kFixedK / 128;
[[maybe_unused]] constexpr int kMaxTokens = 8;
constexpr std::uintptr_t kVecLoadBytes = 16;

using __nv_bfloat16 = __hip_bfloat16;
using __nv_bfloat162 = __hip_bfloat162;
typedef short __attribute__((ext_vector_type(2))) bf16x2_t;

struct bf16_16_t {
  __nv_bfloat162 p0;
  __nv_bfloat162 p1;
  __nv_bfloat162 p2;
  __nv_bfloat162 p3;
  __nv_bfloat162 p4;
  __nv_bfloat162 p5;
  __nv_bfloat162 p6;
  __nv_bfloat162 p7;
};

bool on_gfx1151() {
  static const bool result = [] {
    const auto* dprops = at::cuda::getCurrentDeviceProperties();
    const std::string device_arch = dprops->gcnArchName;
    return device_arch.find("gfx1151") != std::string::npos;
  }();
  return result;
}

struct LogicalScaleStrides {
  int64_t outer;
  int64_t k;
};

LogicalScaleStrides resolve_activation_scale_strides(const at::Tensor& scale,
                                                     int64_t tokens,
                                                     int64_t k_groups) {
  TORCH_CHECK(scale.dim() == 2, "activation_scale must be rank-2");
  if (scale.size(0) == tokens && scale.size(1) == k_groups) {
    return {scale.stride(0), scale.stride(1)};
  }
  if (scale.size(0) == k_groups && scale.size(1) == tokens) {
    return {scale.stride(1), scale.stride(0)};
  }
  TORCH_CHECK(false,
              "activation_scale must be [tokens, K/128] or [K/128, tokens], "
              "got ",
              scale.sizes());
}

LogicalScaleStrides resolve_weight_scale_strides(const at::Tensor& scale,
                                                 int64_t n_groups,
                                                 int64_t k_groups) {
  TORCH_CHECK(scale.dim() == 2, "weight_scale must be rank-2");
  const int64_t s0 = scale.stride(0);
  const int64_t s1 = scale.stride(1);
  const bool shape_canonical =
      scale.size(0) == n_groups && scale.size(1) == k_groups;
  const bool shape_transposed =
      scale.size(0) == k_groups && scale.size(1) == n_groups;
  TORCH_CHECK(shape_canonical || shape_transposed,
              "weight_scale must be [N/128, K/128] or [K/128, N/128], got ",
              scale.sizes());

  // When n_groups == k_groups (e.g. the wo_b case with N == K == 4096) the two
  // legal shapes are identical, so shape alone cannot tell a canonical
  // [N/128, K/128] tensor from a transposed [K/128, N/128] view. Disambiguate
  // from the contiguous (unit-stride) dimension instead of silently trusting
  // the shape-order match, which would otherwise read the transposed element
  // scale[k_group, n_group] in place of scale[n_group, k_group].
  if (n_groups == k_groups) {
    const bool k_contiguous = (s1 == 1);
    const bool n_contiguous = (s0 == 1);
    TORCH_CHECK(k_contiguous != n_contiguous,
                "square weight_scale layout is ambiguous; expected exactly one "
                "contiguous dimension, got strides ",
                scale.strides());
    // k_contiguous  -> row-major [n_groups, k_groups]
    // n_contiguous  -> transposed view of [n_groups, k_groups]
    return k_contiguous ? LogicalScaleStrides{s0, s1}
                        : LogicalScaleStrides{s1, s0};
  }

  return shape_canonical ? LogicalScaleStrides{s0, s1}
                         : LogicalScaleStrides{s1, s0};
}

__device__ __forceinline__ const uint8_t* weight_ptr(const uint8_t* base,
                                                     int logical_n,
                                                     int logical_k,
                                                     int64_t stride0,
                                                     int64_t stride1) {
  // Winner-only skinny path over the original logical [N, K] weight layout.
  // The kernel consumes the exact tensor/strides presented by PyTorch and does
  // not assume an in-place bpreshuffle transform.
  return base + logical_n * stride0 + logical_k * stride1;
}

#if defined(__GFX11__) || defined(__GFX12__)
__device__ __forceinline__ float dot2_bf16(__nv_bfloat162 lhs,
                                           __nv_bfloat162 rhs, float acc) {
  return __builtin_amdgcn_fdot2_f32_bf16(
      *reinterpret_cast<const bf16x2_t*>(&lhs),
      *reinterpret_cast<const bf16x2_t*>(&rhs), acc, /*clamp=*/false);
}

__device__ __forceinline__ float dot16_bf16(const bf16_16_t& lhs,
                                            const bf16_16_t& rhs) {
  float acc = 0.0f;
  acc = dot2_bf16(lhs.p0, rhs.p0, acc);
  acc = dot2_bf16(lhs.p1, rhs.p1, acc);
  acc = dot2_bf16(lhs.p2, rhs.p2, acc);
  acc = dot2_bf16(lhs.p3, rhs.p3, acc);
  acc = dot2_bf16(lhs.p4, rhs.p4, acc);
  acc = dot2_bf16(lhs.p5, rhs.p5, acc);
  acc = dot2_bf16(lhs.p6, rhs.p6, acc);
  acc = dot2_bf16(lhs.p7, rhs.p7, acc);
  return acc;
}

__device__ __forceinline__ float wave32_reduce_sum(float x) {
  x += __builtin_amdgcn_mov_dpp(x, 0x118, 0xf, 0xf, 1);
  x += __builtin_amdgcn_mov_dpp(x, 0x114, 0xf, 0xf, 1);
  x += __builtin_amdgcn_mov_dpp(x, 0x112, 0xf, 0xf, 1);
  x += __builtin_amdgcn_mov_dpp(x, 0x111, 0xf, 0xf, 1);
  x += __shfl_xor(x, 16);
  return x;
}

__device__ __forceinline__ float subgroup_broadcast8(float x, int lane) {
  const int leader = lane & ~7;
  return __shfl(x, leader, kWaveSize);
}

__device__ __forceinline__ bf16_16_t fp8x16_to_bf16(uint4 packed) {
  union packed16_t {
    uint4 u4;
    uint16_t u16[8];
  };
  packed16_t tmp;
  tmp.u4 = packed;
  bf16_16_t out;
  out.p0 = vllm::fp8::vec_conversion<__nv_bfloat162, uint16_t>(tmp.u16[0]);
  out.p1 = vllm::fp8::vec_conversion<__nv_bfloat162, uint16_t>(tmp.u16[1]);
  out.p2 = vllm::fp8::vec_conversion<__nv_bfloat162, uint16_t>(tmp.u16[2]);
  out.p3 = vllm::fp8::vec_conversion<__nv_bfloat162, uint16_t>(tmp.u16[3]);
  out.p4 = vllm::fp8::vec_conversion<__nv_bfloat162, uint16_t>(tmp.u16[4]);
  out.p5 = vllm::fp8::vec_conversion<__nv_bfloat162, uint16_t>(tmp.u16[5]);
  out.p6 = vllm::fp8::vec_conversion<__nv_bfloat162, uint16_t>(tmp.u16[6]);
  out.p7 = vllm::fp8::vec_conversion<__nv_bfloat162, uint16_t>(tmp.u16[7]);
  return out;
}

template <int TOKENS>
__global__
__launch_bounds__(kThreadsPerWG) void wvSplitKQBlockScaleBpreshuffleGfx1151Kernel(
    const uint8_t* __restrict__ weight, const uint8_t* __restrict__ activation,
    const float* __restrict__ activation_scale,
    const float* __restrict__ weight_scale, __hip_bfloat16* __restrict__ out,
    int N, int64_t weight_stride0, int64_t weight_stride1,
    int64_t activation_stride0, int64_t as_token_stride, int64_t as_k_stride,
    int64_t ws_n_stride, int64_t ws_k_stride, int64_t out_stride0,
    int64_t out_stride1, int cu_count) {
  static_assert(TOKENS >= 1 && TOKENS <= kMaxTokens);
  __shared__ uint4 sA[(TOKENS * kFixedK) / kAChunk];

  const int lane = threadIdx.x;
  const int wave = threadIdx.y;
  const int linear_tid = wave * kWaveSize + lane;

  for (int idx = linear_tid * kAChunk; idx < TOKENS * kFixedK;
       idx += kThreadsPerWG * kAChunk) {
    const int token = idx / kFixedK;
    const int k = idx % kFixedK;
    const uint8_t* src = activation + token * activation_stride0 + k;
    sA[idx / kAChunk] = *reinterpret_cast<const uint4*>(src);
  }

  __syncthreads();

  int out_n = (blockIdx.x * kWavesPerWG + wave) * kYTile;
  while (out_n < N) {
    float sum[TOKENS][kYTile] = {};

    for (int k_base = 0; k_base < kFixedK; k_base += kWaveSize * kAChunk) {
      const int logical_k = k_base + lane * kAChunk;
      const int k_group = logical_k >> 7;

      bf16_16_t weight_frag[kYTile];
      bool active_y[kYTile];
  #pragma unroll
      for (int y = 0; y < kYTile; ++y) {
        active_y[y] = (out_n + y) < N;
        if (active_y[y]) {
          const uint8_t* w_ptr = weight_ptr(
              weight, out_n + y, logical_k, weight_stride0, weight_stride1);
          weight_frag[y] =
              fp8x16_to_bf16(*reinterpret_cast<const uint4*>(w_ptr));
        } else {
          weight_frag[y] = {};
        }
      }

      float ws[kYTile] = {};
      if ((lane & 7) == 0) {
  #pragma unroll
        for (int y = 0; y < kYTile; ++y) {
          if (active_y[y]) {
            ws[y] = weight_scale[((out_n + y) >> 7) * ws_n_stride +
                                 k_group * ws_k_stride];
          }
        }
      }
  #pragma unroll
      for (int y = 0; y < kYTile; ++y) {
        ws[y] = subgroup_broadcast8(ws[y], lane);
      }

  #pragma unroll
      for (int token = 0; token < TOKENS; ++token) {
        const uint4 packed_a = sA[(token * kFixedK + logical_k) / kAChunk];
        const bf16_16_t act_frag = fp8x16_to_bf16(packed_a);

        float as = 0.0f;
        if ((lane & 7) == 0) {
          as =
              activation_scale[token * as_token_stride + k_group * as_k_stride];
        }
        as = subgroup_broadcast8(as, lane);

  #pragma unroll
        for (int y = 0; y < kYTile; ++y) {
          if (active_y[y]) {
            const float partial = dot16_bf16(act_frag, weight_frag[y]);
            sum[token][y] += partial * as * ws[y];
          }
        }
      }
    }

  #pragma unroll
    for (int token = 0; token < TOKENS; ++token) {
  #pragma unroll
      for (int y = 0; y < kYTile; ++y) {
        sum[token][y] = wave32_reduce_sum(sum[token][y]);
      }
    }

    if (lane == (kWaveSize - 1)) {
  #pragma unroll
      for (int token = 0; token < TOKENS; ++token) {
  #pragma unroll
        for (int y = 0; y < kYTile; ++y) {
          if (out_n + y < N) {
            out[token * out_stride0 + (out_n + y) * out_stride1] =
                __float2bfloat16(sum[token][y]);
          }
        }
      }
    }

    out_n += cu_count * kWavesPerWG * kYTile;
  }
}
#else
template <int TOKENS>
__global__ void wvSplitKQBlockScaleBpreshuffleGfx1151Kernel(
    const uint8_t* __restrict__, const uint8_t* __restrict__,
    const float* __restrict__, const float* __restrict__,
    __hip_bfloat16* __restrict__, int, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, int) {}
#endif

template <int TOKENS, typename fp8_t>
void launch_wvsplitk_blockscale_bpreshuffle_gfx1151(
    const at::Tensor& weight, const at::Tensor& activation,
    const at::Tensor& activation_scale, const at::Tensor& weight_scale,
    at::Tensor& out, int64_t cu_count) {
  const auto as =
      resolve_activation_scale_strides(activation_scale, TOKENS, kFixedKGroups);
  const auto ws = resolve_weight_scale_strides(weight_scale, out.size(1) / 128,
                                               kFixedKGroups);

  dim3 grid(cu_count);
  dim3 block(kWaveSize, kWavesPerWG);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(activation));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto* weight_ptr = reinterpret_cast<const uint8_t*>(weight.data_ptr<fp8_t>());
  auto* act_ptr =
      reinterpret_cast<const uint8_t*>(activation.data_ptr<fp8_t>());
  auto* out_ptr =
      reinterpret_cast<__hip_bfloat16*>(out.data_ptr<c10::BFloat16>());

  wvSplitKQBlockScaleBpreshuffleGfx1151Kernel<TOKENS>
      <<<grid, block, 0, stream>>>(
          weight_ptr, act_ptr, activation_scale.data_ptr<float>(),
          weight_scale.data_ptr<float>(), out_ptr,
          static_cast<int>(out.size(1)), weight.stride(0), weight.stride(1),
          activation.stride(0), as.outer, as.k, ws.outer, ws.k, out.stride(0),
          out.stride(1), static_cast<int>(cu_count));
}

}  // namespace

void wvSplitKQBlockScale(const at::Tensor& weight, const at::Tensor& activation,
                         const at::Tensor& activation_scale,
                         const at::Tensor& weight_scale, at::Tensor& out,
                         const int64_t CuCount, const bool bpreshuffle) {
  static c10::ScalarType kFp8Type = is_fp8_ocp()
                                        ? c10::ScalarType::Float8_e4m3fn
                                        : c10::ScalarType::Float8_e4m3fnuz;

  TORCH_CHECK(on_gfx1151(),
              "wvSplitKQBlockScale is currently supported on gfx1151 only");
  TORCH_CHECK(!bpreshuffle,
              "wvSplitKQBlockScale non-preshuffled path expects original "
              "weight layout");
  TORCH_CHECK(weight.is_cuda() && activation.is_cuda() &&
                  activation_scale.is_cuda() && weight_scale.is_cuda() &&
                  out.is_cuda(),
              "wvSplitKQBlockScale expects CUDA/HIP tensors");
  TORCH_CHECK(weight.device() == activation.device() &&
                  weight.device() == activation_scale.device() &&
                  weight.device() == weight_scale.device() &&
                  weight.device() == out.device(),
              "wvSplitKQBlockScale tensors must share device");
  TORCH_CHECK(activation.scalar_type() == kFp8Type,
              "activation must use ROCm native fp8 type");
  TORCH_CHECK(weight.scalar_type() == kFp8Type,
              "weight must use ROCm native fp8 type");
  TORCH_CHECK(activation_scale.scalar_type() == at::kFloat,
              "activation_scale must be float32");
  TORCH_CHECK(weight_scale.scalar_type() == at::kFloat,
              "weight_scale must be float32");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16, "out must be bfloat16");

  TORCH_CHECK(weight.dim() == 2, "weight must be rank-2");
  TORCH_CHECK(activation.dim() == 2, "activation must be rank-2");
  TORCH_CHECK(out.dim() == 2, "out must be rank-2");
  TORCH_CHECK(
      reinterpret_cast<std::uintptr_t>(weight.data_ptr()) % kVecLoadBytes == 0,
      "weight data pointer must be 16-byte aligned");
  TORCH_CHECK(weight.stride(0) % kAChunk == 0,
              "weight row stride must be a multiple of 16 elements");
  TORCH_CHECK(weight.stride(1) == 1, "weight must have contiguous K dimension");
  TORCH_CHECK(
      reinterpret_cast<std::uintptr_t>(activation.data_ptr()) % kVecLoadBytes ==
          0,
      "activation data pointer must be 16-byte aligned");
  TORCH_CHECK(activation.stride(0) % kAChunk == 0,
              "activation row stride must be a multiple of 16 elements");
  TORCH_CHECK(activation.stride(1) == 1,
              "activation must have contiguous K dimension");
  TORCH_CHECK(out.stride(1) == 1, "out must have contiguous N dimension");

  const int64_t tokens = activation.size(0);
  const int64_t K = activation.size(1);
  TORCH_CHECK(tokens >= 1 && tokens <= 8,
              "activation token count must be in [1, 8], got ", tokens);
  TORCH_CHECK(K == kFixedK, "logical K must be 4096, got ", K);

  const int64_t N = weight.size(0);
  const int64_t logical_K = weight.size(1);
  TORCH_CHECK(logical_K == K,
              "weight shape does not match activation K");
  TORCH_CHECK(
      N == 4096 || N == 1536,
      "first version supports logical N in {4096, 1536}, got ", N);
  TORCH_CHECK(out.size(0) == tokens && out.size(1) == N,
              "out must be shaped [tokens, N], got ", out.sizes(),
              " for logical N=", N, " tokens=", tokens);
  TORCH_CHECK(CuCount > 0, "CuCount must be positive");

  VLLM_DISPATCH_FP8_TYPES(activation.scalar_type(), "wvSplitKQBlockScale", [&] {
    switch (tokens) {
      case 1:
        launch_wvsplitk_blockscale_bpreshuffle_gfx1151<1, fp8_t>(
            weight, activation, activation_scale, weight_scale, out, CuCount);
        break;
      case 2:
        launch_wvsplitk_blockscale_bpreshuffle_gfx1151<2, fp8_t>(
            weight, activation, activation_scale, weight_scale, out, CuCount);
        break;
      case 3:
        launch_wvsplitk_blockscale_bpreshuffle_gfx1151<3, fp8_t>(
            weight, activation, activation_scale, weight_scale, out, CuCount);
        break;
      case 4:
        launch_wvsplitk_blockscale_bpreshuffle_gfx1151<4, fp8_t>(
            weight, activation, activation_scale, weight_scale, out, CuCount);
        break;
      case 5:
        launch_wvsplitk_blockscale_bpreshuffle_gfx1151<5, fp8_t>(
            weight, activation, activation_scale, weight_scale, out, CuCount);
        break;
      case 6:
        launch_wvsplitk_blockscale_bpreshuffle_gfx1151<6, fp8_t>(
            weight, activation, activation_scale, weight_scale, out, CuCount);
        break;
      case 7:
        launch_wvsplitk_blockscale_bpreshuffle_gfx1151<7, fp8_t>(
            weight, activation, activation_scale, weight_scale, out, CuCount);
        break;
      case 8:
        launch_wvsplitk_blockscale_bpreshuffle_gfx1151<8, fp8_t>(
            weight, activation, activation_scale, weight_scale, out, CuCount);
        break;
      default:
        TORCH_CHECK(false, "unsupported token count ", tokens);
    }
  });
}
