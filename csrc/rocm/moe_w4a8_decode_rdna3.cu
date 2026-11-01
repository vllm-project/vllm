// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Fused MoE decode for 4-bit group-quantized experts with int8 activations
// (W4A8) on RDNA3/RDNA3.5 (gfx11), for small token counts.
//
// Three launches, no expert sort, no padding, no atomics (deterministic):
//   1. quant_act: x -> int8 per 32 values, fp32 scale and int32 sum per group.
//   2. gate_up:   one block per (token, slot, 32 intermediate columns). Gate
//                 and up dot products, h = silu(gate) * up, and h is
//                 re-quantized to int8 per 32 values.
//   3. down:      one block per (token, 32 output columns). Loops over the
//                 token's top_k slots, applies the routing weights and writes
//                 the output once.
// Each 32-value group is reduced with eight v_dot4_i32_iu8 instructions and
// one fp32 multiply-add applies both scales and the weight zero point:
//   acc += a_scale * w_scale * (dot(a, w) - w_zero * sum(a)).
// The scheme follows ggml's MMVQ MoE path (llama.cpp, MIT License).
//
// Layouts (the TritonWNA16Experts tensors, no repack):
//   x         [M, K]          fp16 / bf16
//   w13       [E, 2N, K/2]    uint8, low nibble = even k;
//                             rows [0, N) gate, rows [N, 2N) up
//   w13_scale [E, 2N, K/G1]   same dtype as x
//   w13_zp    [E, N, K/G1]    uint8, two zero points per byte along rows,
//                             low nibble = even row; absent = symmetric (8)
//   w2        [E, K, N/2], w2_scale [E, K, N/G2], w2_zp [E, K/2, N/G2]
//   topk_ids  [M, top_k] int32, entries outside [0, E) are skipped
//   topk_w    [M, top_k] fp32
//   out       [M, K]          same dtype as x
//
// Quantized activations store each group of 8 values in the order
// (0, 2, 4, 6, 1, 3, 5, 7), so the low and high nibbles of four weight bytes
// line up with two consecutive int32 words of activations.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <optional>
#include <string>
#include <type_traits>

#include "core/cuda_compat.h"

#if defined(__GFX11__)
  #define __HIP__MOE_W4A8_GFX11__
#endif

namespace {

constexpr int kGroup = 32;
constexpr int kWave = 32;
constexpr int kWavesPerBlock = 8;
constexpr int kColsPerWave = 4;
constexpr int kTileCols = kWavesPerBlock * kColsPerWave;

static_assert(kTileCols == kGroup,
              "gate_up re-quantizes one 32-column tile as one group");
static_assert(kTileCols == kWave,
              "gate_up epilogue maps one lane to one column");

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

#if defined(__HIP__MOE_W4A8_GFX11__) || !defined(__HIP_DEVICE_COMPILE__)

// Storage position of value i inside its 32-value group.
__device__ __forceinline__ int permuted_pos(int i) {
  const int r = i & 7;
  return (i & ~7) | ((r & 1) ? 4 + (r >> 1) : (r >> 1));
}

__device__ __forceinline__ float wave_sum(float v) {
  #pragma unroll
  for (int o = kWave / 2; o > 0; o >>= 1)
    v += VLLM_SHFL_XOR_SYNC_WIDTH(v, o, kWave);
  return v;
}

__device__ __forceinline__ int wave_sum(int v) {
  #pragma unroll
  for (int o = kWave / 2; o > 0; o >>= 1)
    v += VLLM_SHFL_XOR_SYNC_WIDTH(v, o, kWave);
  return v;
}

__device__ __forceinline__ float wave_max(float v) {
  #pragma unroll
  for (int o = kWave / 2; o > 0; o >>= 1)
    v = fmaxf(v, VLLM_SHFL_XOR_SYNC_WIDTH(v, o, kWave));
  return v;
}

// Integer dot product of 32 unsigned 4-bit weights (16 bytes, low nibble =
// even k) with 32 permuted signed int8 activations.
__device__ __forceinline__ int dot_group(const int4& w, const int4& a0,
                                         const int4& a1) {
  constexpr int kLo = 0x0F0F0F0F;
  int acc = 0;
  acc = __builtin_amdgcn_sudot4(true, a0.x, false, w.x & kLo, acc, false);
  acc =
      __builtin_amdgcn_sudot4(true, a0.y, false, (w.x >> 4) & kLo, acc, false);
  acc = __builtin_amdgcn_sudot4(true, a0.z, false, w.y & kLo, acc, false);
  acc =
      __builtin_amdgcn_sudot4(true, a0.w, false, (w.y >> 4) & kLo, acc, false);
  acc = __builtin_amdgcn_sudot4(true, a1.x, false, w.z & kLo, acc, false);
  acc =
      __builtin_amdgcn_sudot4(true, a1.y, false, (w.z >> 4) & kLo, acc, false);
  acc = __builtin_amdgcn_sudot4(true, a1.z, false, w.w & kLo, acc, false);
  acc =
      __builtin_amdgcn_sudot4(true, a1.w, false, (w.w >> 4) & kLo, acc, false);
  return acc;
}

template <bool kHasZp>
__device__ __forceinline__ int zero_point(const uint8_t* __restrict__ zp,
                                          int row, int g, int groups) {
  if constexpr (kHasZp) {
    const int byte = zp[(int64_t)(row >> 1) * groups + g];
    return (row & 1) ? (byte >> 4) : (byte & 0xF);
  } else {
    return 8;
  }
}

// One thread per 32-value group of one token.
template <typename T>
__global__ void moe_w4a8_quant_act_kernel(const T* __restrict__ x,
                                          int8_t* __restrict__ xq,
                                          float* __restrict__ xs,
                                          int* __restrict__ xsum, int M,
                                          int K) {
  const int groups = K / kGroup;
  const int m = blockIdx.x;
  const int g = blockIdx.y * blockDim.x + threadIdx.x;
  if (m >= M || g >= groups) return;

  const T* src = x + (int64_t)m * K + g * kGroup;
  float v[kGroup];
  float amax = 0.f;
  #pragma unroll
  for (int i = 0; i < kGroup; ++i) {
    v[i] = to_float(src[i]);
    amax = fmaxf(amax, fabsf(v[i]));
  }
  const float scale = fmaxf(amax / 127.0f, 1e-12f);

  int8_t* dst = xq + (int64_t)m * K + g * kGroup;
  int sum = 0;
  #pragma unroll
  for (int i = 0; i < kGroup; ++i) {
    const int q = (int)fmaxf(-127.f, fminf(127.f, rintf(v[i] / scale)));
    dst[permuted_pos(i)] = (int8_t)q;
    sum += q;
  }
  xs[(int64_t)m * groups + g] = scale;
  xsum[(int64_t)m * groups + g] = sum;
}

// grid (M * top_k, N / 32), block (32, 8). Each wave computes the gate and
// up rows of 4 intermediate columns; lanes stride over the K groups.
template <typename T, bool kHasZp>
__global__ void __launch_bounds__(kWave* kWavesPerBlock)
    moe_w4a8_gate_up_kernel(
        const int8_t* __restrict__ xq, const float* __restrict__ xs,
        const int* __restrict__ xsum, const int* __restrict__ topk_ids,
        const uint8_t* __restrict__ w13, const T* __restrict__ w13_scale,
        const uint8_t* __restrict__ w13_zp, int8_t* __restrict__ hq,
        float* __restrict__ hs, int* __restrict__ hsum, int top_k,
        int num_experts, int K, int N, int group_size) {
  const int ms = blockIdx.x;
  const int e = topk_ids[ms];
  if (e < 0 || e >= num_experts) return;

  const int m = ms / top_k;
  const int lane = threadIdx.x;
  const int wave = threadIdx.y;
  const int col0 = blockIdx.y * kTileCols + wave * kColsPerWave;
  const int k_groups = K / kGroup;
  const int s_groups = K / group_size;

  const int8_t* xq_m = xq + (int64_t)m * K;
  const float* xs_m = xs + (int64_t)m * k_groups;
  const int* xsum_m = xsum + (int64_t)m * k_groups;
  const uint8_t* w_e = w13 + (int64_t)e * (2 * N) * (K / 2);
  const T* s_e = w13_scale + (int64_t)e * (2 * N) * s_groups;
  const uint8_t* z_e = kHasZp ? w13_zp + (int64_t)e * N * s_groups : nullptr;

  // acc[0, 4) gate rows, acc[4, 8) up rows.
  float acc[2 * kColsPerWave];
  #pragma unroll
  for (int r = 0; r < 2 * kColsPerWave; ++r) acc[r] = 0.f;

  for (int b = lane; b < k_groups; b += kWave) {
    const int4 a0 = *reinterpret_cast<const int4*>(xq_m + b * kGroup);
    const int4 a1 = *reinterpret_cast<const int4*>(xq_m + b * kGroup + 16);
    const float a_s = xs_m[b];
    const int a_sum = xsum_m[b];
    const int g = b * kGroup / group_size;
  #pragma unroll
    for (int r = 0; r < 2 * kColsPerWave; ++r) {
      const int row =
          r < kColsPerWave ? col0 + r : N + col0 + (r - kColsPerWave);
      const int4 w = *reinterpret_cast<const int4*>(
          w_e + (int64_t)row * (K / 2) + b * (kGroup / 2));
      const int dot = dot_group(w, a0, a1);
      const int z = zero_point<kHasZp>(z_e, row, g, s_groups);
      const float w_s = to_float(s_e[(int64_t)row * s_groups + g]);
      acc[r] += a_s * w_s * (float)(dot - z * a_sum);
    }
  }

  __shared__ float s_gate[kTileCols];
  __shared__ float s_up[kTileCols];
  #pragma unroll
  for (int r = 0; r < 2 * kColsPerWave; ++r) {
    const float v = wave_sum(acc[r]);
    if (lane == 0) {
      if (r < kColsPerWave)
        s_gate[wave * kColsPerWave + r] = v;
      else
        s_up[wave * kColsPerWave + r - kColsPerWave] = v;
    }
  }
  __syncthreads();
  if (wave != 0) return;

  const float gate = s_gate[lane];
  const float h = gate / (1.f + __expf(-gate)) * s_up[lane];
  const float amax = wave_max(fabsf(h));
  const float scale = fmaxf(amax / 127.0f, 1e-12f);
  const int q = (int)fmaxf(-127.f, fminf(127.f, rintf(h / scale)));

  const int n_groups = N / kGroup;
  hq[(int64_t)ms * N + blockIdx.y * kGroup + permuted_pos(lane)] = (int8_t)q;
  const int q_sum = wave_sum(q);
  if (lane == 0) {
    hs[(int64_t)ms * n_groups + blockIdx.y] = scale;
    hsum[(int64_t)ms * n_groups + blockIdx.y] = q_sum;
  }
}

// grid (M, K / 32), block (32, 8). Each wave computes 4 output columns;
// lanes stride over (slot, N group) pairs of the token.
template <typename T, bool kHasZp>
__global__ void __launch_bounds__(kWave* kWavesPerBlock) moe_w4a8_down_kernel(
    const int8_t* __restrict__ hq, const float* __restrict__ hs,
    const int* __restrict__ hsum, const int* __restrict__ topk_ids,
    const float* __restrict__ topk_w, const uint8_t* __restrict__ w2,
    const T* __restrict__ w2_scale, const uint8_t* __restrict__ w2_zp,
    T* __restrict__ out, int top_k, int num_experts, int K, int N,
    int group_size) {
  const int m = blockIdx.x;
  const int lane = threadIdx.x;
  const int wave = threadIdx.y;
  const int col0 = blockIdx.y * kTileCols + wave * kColsPerWave;
  const int n_groups = N / kGroup;
  const int s_groups = N / group_size;

  float acc[kColsPerWave];
  #pragma unroll
  for (int c = 0; c < kColsPerWave; ++c) acc[c] = 0.f;

  for (int t = lane; t < top_k * n_groups; t += kWave) {
    const int slot = t / n_groups;
    const int b = t - slot * n_groups;
    const int ms = m * top_k + slot;
    const int e = topk_ids[ms];
    if (e < 0 || e >= num_experts) continue;

    const int8_t* hp = hq + (int64_t)ms * N + b * kGroup;
    const int4 a0 = *reinterpret_cast<const int4*>(hp);
    const int4 a1 = *reinterpret_cast<const int4*>(hp + 16);
    const float a_s = hs[(int64_t)ms * n_groups + b] * topk_w[ms];
    const int a_sum = hsum[(int64_t)ms * n_groups + b];
    const int g = b * kGroup / group_size;

    const uint8_t* w_e = w2 + (int64_t)e * K * (N / 2);
    const T* s_e = w2_scale + (int64_t)e * K * s_groups;
    const uint8_t* z_e =
        kHasZp ? w2_zp + (int64_t)e * (K / 2) * s_groups : nullptr;
  #pragma unroll
    for (int c = 0; c < kColsPerWave; ++c) {
      const int row = col0 + c;
      const int4 w = *reinterpret_cast<const int4*>(
          w_e + (int64_t)row * (N / 2) + b * (kGroup / 2));
      const int dot = dot_group(w, a0, a1);
      const int z = zero_point<kHasZp>(z_e, row, g, s_groups);
      const float w_s = to_float(s_e[(int64_t)row * s_groups + g]);
      acc[c] += a_s * w_s * (float)(dot - z * a_sum);
    }
  }

  #pragma unroll
  for (int c = 0; c < kColsPerWave; ++c) {
    const float v = wave_sum(acc[c]);
    if (lane == c) out[(int64_t)m * K + col0 + c] = from_float<T>(v);
  }
}

#else  // non-gfx11 device pass: empty kernels for symbol parity.

template <typename T>
__global__ void moe_w4a8_quant_act_kernel(const T*, int8_t*, float*, int*, int,
                                          int) {}

template <typename T, bool kHasZp>
__global__ void moe_w4a8_gate_up_kernel(const int8_t*, const float*, const int*,
                                        const int*, const uint8_t*, const T*,
                                        const uint8_t*, int8_t*, float*, int*,
                                        int, int, int, int, int) {}

template <typename T, bool kHasZp>
__global__ void moe_w4a8_down_kernel(const int8_t*, const float*, const int*,
                                     const int*, const float*, const uint8_t*,
                                     const T*, const uint8_t*, T*, int, int,
                                     int, int, int) {}

#endif  // __HIP__MOE_W4A8_GFX11__ || !__HIP_DEVICE_COMPILE__

bool moe_w4a8_on_gfx11() {
  static const bool result = [] {
    const auto* dprops = at::cuda::getCurrentDeviceProperties();
    const std::string arch = dprops->gcnArchName;
    return arch.rfind("gfx11", 0) == 0;
  }();
  return result;
}

bool aligned16(const at::Tensor& t) {
  return reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0;
}

template <typename T>
void launch(at::Tensor& out, const at::Tensor& x, const at::Tensor& topk_ids,
            const at::Tensor& topk_weights, const at::Tensor& w13,
            const at::Tensor& w13_scale,
            const std::optional<at::Tensor>& w13_zp, const at::Tensor& w2,
            const at::Tensor& w2_scale, const std::optional<at::Tensor>& w2_zp,
            int M, int K, int N, int E, int top_k, int group1, int group2) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int k_groups = K / kGroup;
  const int n_groups = N / kGroup;
  const int MS = M * top_k;

  auto i8 = x.options().dtype(torch::kInt8);
  auto f32 = x.options().dtype(torch::kFloat32);
  auto i32 = x.options().dtype(torch::kInt32);
  auto xq = torch::empty({M, K}, i8);
  auto xs = torch::empty({M, k_groups}, f32);
  auto xsum = torch::empty({M, k_groups}, i32);
  auto hq = torch::empty({MS, N}, i8);
  auto hs = torch::empty({MS, n_groups}, f32);
  auto hsum = torch::empty({MS, n_groups}, i32);

  const dim3 q_block(64);
  const dim3 q_grid(M, (k_groups + 63) / 64);
  moe_w4a8_quant_act_kernel<T><<<q_grid, q_block, 0, stream>>>(
      reinterpret_cast<const T*>(x.data_ptr()), xq.data_ptr<int8_t>(),
      xs.data_ptr<float>(), xsum.data_ptr<int>(), M, K);

  const dim3 block(kWave, kWavesPerBlock);
  const int* ids = topk_ids.data_ptr<int>();

  const dim3 gu_grid(MS, N / kTileCols);
  auto gate_up = [&](auto has_zp) {
    constexpr bool kHasZp = decltype(has_zp)::value;
    moe_w4a8_gate_up_kernel<T, kHasZp><<<gu_grid, block, 0, stream>>>(
        xq.data_ptr<int8_t>(), xs.data_ptr<float>(), xsum.data_ptr<int>(), ids,
        w13.data_ptr<uint8_t>(),
        reinterpret_cast<const T*>(w13_scale.data_ptr()),
        kHasZp ? w13_zp->data_ptr<uint8_t>() : nullptr, hq.data_ptr<int8_t>(),
        hs.data_ptr<float>(), hsum.data_ptr<int>(), top_k, E, K, N, group1);
  };
  if (w13_zp.has_value())
    gate_up(std::true_type{});
  else
    gate_up(std::false_type{});

  const dim3 down_grid(M, K / kTileCols);
  auto down = [&](auto has_zp) {
    constexpr bool kHasZp = decltype(has_zp)::value;
    moe_w4a8_down_kernel<T, kHasZp><<<down_grid, block, 0, stream>>>(
        hq.data_ptr<int8_t>(), hs.data_ptr<float>(), hsum.data_ptr<int>(), ids,
        topk_weights.data_ptr<float>(), w2.data_ptr<uint8_t>(),
        reinterpret_cast<const T*>(w2_scale.data_ptr()),
        kHasZp ? w2_zp->data_ptr<uint8_t>() : nullptr,
        reinterpret_cast<T*>(out.data_ptr()), top_k, E, K, N, group2);
  };
  if (w2_zp.has_value())
    down(std::true_type{});
  else
    down(std::false_type{});
}

void check_zp(const std::optional<at::Tensor>& zp, int64_t E, int64_t rows,
              int64_t groups, const char* name) {
  if (!zp.has_value()) return;
  TORCH_CHECK(
      zp->is_cuda() && zp->scalar_type() == at::kByte && zp->is_contiguous(),
      "moe_w4a8_decode_rdna3: ", name,
      " must be a contiguous uint8 GPU tensor");
  TORCH_CHECK(zp->dim() == 3 && zp->size(0) == E && zp->size(1) == rows / 2 &&
                  zp->size(2) == groups,
              "moe_w4a8_decode_rdna3: ", name, " must be [E, rows/2, groups]");
}

}  // namespace

void moe_w4a8_decode_rdna3(at::Tensor& out, const at::Tensor& x,
                           const at::Tensor& topk_ids,
                           const at::Tensor& topk_weights,
                           const at::Tensor& w13, const at::Tensor& w13_scale,
                           const std::optional<at::Tensor>& w13_zp,
                           const at::Tensor& w2, const at::Tensor& w2_scale,
                           const std::optional<at::Tensor>& w2_zp) {
  TORCH_CHECK(moe_w4a8_on_gfx11(),
              "moe_w4a8_decode_rdna3 requires an RDNA3/RDNA3.5 (gfx11) GPU");
  for (const at::Tensor* t : std::initializer_list<const at::Tensor*>{
           &out, &x, &topk_ids, &topk_weights, &w13, &w13_scale, &w2,
           &w2_scale}) {
    TORCH_CHECK(t->is_cuda() && t->is_contiguous(),
                "moe_w4a8_decode_rdna3: all tensors must be contiguous and on "
                "the GPU");
  }
  TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
              "moe_w4a8_decode_rdna3: x must be float16 or bfloat16");
  TORCH_CHECK(out.scalar_type() == x.scalar_type() &&
                  w13_scale.scalar_type() == x.scalar_type() &&
                  w2_scale.scalar_type() == x.scalar_type(),
              "moe_w4a8_decode_rdna3: out and scales must have the dtype of x");
  TORCH_CHECK(
      w13.scalar_type() == at::kByte && w2.scalar_type() == at::kByte,
      "moe_w4a8_decode_rdna3: w13 and w2 must be uint8 (two codes per byte)");
  TORCH_CHECK(topk_ids.scalar_type() == at::kInt &&
                  topk_weights.scalar_type() == at::kFloat,
              "moe_w4a8_decode_rdna3: topk_ids must be int32 and topk_weights "
              "float32");
  TORCH_CHECK(x.dim() == 2 && out.dim() == 2 && topk_ids.dim() == 2 &&
                  w13.dim() == 3 && w13_scale.dim() == 3 && w2.dim() == 3 &&
                  w2_scale.dim() == 3,
              "moe_w4a8_decode_rdna3: unexpected tensor rank");

  const int64_t M = x.size(0);
  const int64_t K = x.size(1);
  const int64_t E = w13.size(0);
  const int64_t N = w13.size(1) / 2;
  const int64_t top_k = topk_ids.size(1);
  TORCH_CHECK(M >= 1, "moe_w4a8_decode_rdna3: empty input");
  TORCH_CHECK(K % kTileCols == 0 && N % kTileCols == 0,
              "moe_w4a8_decode_rdna3: hidden and intermediate sizes must be "
              "multiples of 32, got K=",
              K, ", N=", N);
  TORCH_CHECK(out.size(0) == M && out.size(1) == K,
              "moe_w4a8_decode_rdna3: out must be [M, K]");
  TORCH_CHECK(topk_ids.size(0) == M && topk_weights.sizes() == topk_ids.sizes(),
              "moe_w4a8_decode_rdna3: topk_ids and topk_weights must be "
              "[M, top_k]");
  TORCH_CHECK(w13.size(1) == 2 * N && w13.size(2) * 2 == K,
              "moe_w4a8_decode_rdna3: w13 must be [E, 2N, K/2]");
  TORCH_CHECK(w2.size(0) == E && w2.size(1) == K && w2.size(2) * 2 == N,
              "moe_w4a8_decode_rdna3: w2 must be [E, K, N/2]");
  TORCH_CHECK(w13_scale.size(0) == E && w13_scale.size(1) == 2 * N &&
                  K % w13_scale.size(2) == 0,
              "moe_w4a8_decode_rdna3: w13_scale must be [E, 2N, K/group]");
  TORCH_CHECK(w2_scale.size(0) == E && w2_scale.size(1) == K &&
                  N % w2_scale.size(2) == 0,
              "moe_w4a8_decode_rdna3: w2_scale must be [E, K, N/group]");
  const int64_t group1 = K / w13_scale.size(2);
  const int64_t group2 = N / w2_scale.size(2);
  TORCH_CHECK(group1 % kGroup == 0 && group2 % kGroup == 0,
              "moe_w4a8_decode_rdna3: group sizes must be multiples of 32, "
              "got ",
              group1, " and ", group2);
  check_zp(w13_zp, E, 2 * N, w13_scale.size(2), "w13_zp");
  check_zp(w2_zp, E, K, w2_scale.size(2), "w2_zp");
  TORCH_CHECK(aligned16(w13) && aligned16(w2),
              "moe_w4a8_decode_rdna3: w13 and w2 must be 16-byte aligned");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  if (x.scalar_type() == at::kHalf) {
    launch<half>(out, x, topk_ids, topk_weights, w13, w13_scale, w13_zp, w2,
                 w2_scale, w2_zp, M, K, N, E, top_k, group1, group2);
  } else {
    launch<__hip_bfloat16>(out, x, topk_ids, topk_weights, w13, w13_scale,
                           w13_zp, w2, w2_scale, w2_zp, M, K, N, E, top_k,
                           group1, group2);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
