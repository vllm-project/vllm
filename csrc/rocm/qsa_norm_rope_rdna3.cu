// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Qwen4Exp QSA indexer: Gemma RMSNorm + NeoX (M)RoPE of the index heads in
// one launch (gfx11). Unfused, GemmaRMSNorm.forward_native plus the rotary
// glue are ~30 small launches per indexer call, ~1.1 ms per decode step.
//
//   n = fp16(x * rsqrt(mean(x^2) + eps) * (1 + w))      (fp32 math)
//   lo half: n1 * cos - n2 * sin,  hi half: n2 * cos + n1 * sin
//
// The rotation runs as separately rounded fp16 operations, like the eager
// ApplyRotaryEmb. With MRoPE positions ([3, T]) each frequency takes the
// T/H/W position of its section (interleaved or contiguous sections).
//
// One wave per (token, head); each lane owns 4 consecutive dims of the
// 128-dim head, so the NeoX partner of a rotated dim sits rd/8 lanes away.

#include <cstdint>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

namespace vllm {
namespace qsa_norm_rope_rdna3 {

constexpr int kHeadDim = 128;
constexpr int kDimsPerLane = 4;

struct Rope {
  int rd;
  int half;
  int st, sh, sw;
  bool mrope, interleaved;
};

__device__ __forceinline__ int mrope_axis(int f, const Rope& r) {
  if (r.interleaved) {
    if ((f % 3) == 1 && f <= 3 * r.sh) return 1;
    if ((f % 3) == 2 && f <= 3 * r.sw) return 2;
    return 0;
  }
  return f < r.st ? 0 : (f < r.st + r.sh ? 1 : 2);
}

// Explicit fp16 instructions: the build's fast-math flags would otherwise
// fold the intermediate roundings away.
__device__ __forceinline__ half rot(half a, half b, half c, half s, bool lo) {
  _Float16 ac, bs, o;
  const _Float16 a16 = a, b16 = b, c16 = c, s16 = s;
  asm volatile("v_mul_f16 %0, %1, %2" : "=v"(ac) : "v"(a16), "v"(c16));
  asm volatile("v_mul_f16 %0, %1, %2" : "=v"(bs) : "v"(b16), "v"(s16));
  if (lo)
    asm volatile("v_sub_f16 %0, %1, %2" : "=v"(o) : "v"(ac), "v"(bs));
  else
    asm volatile("v_add_f16 %0, %1, %2" : "=v"(o) : "v"(ac), "v"(bs));
  return o;
}

__global__ void __launch_bounds__(256)
    norm_rope_kernel(const half* __restrict__ x, half* __restrict__ out,
                     const half* __restrict__ w,
                     const int64_t* __restrict__ pos,
                     const half* __restrict__ cache, float eps, int n_heads,
                     int64_t sxt, int64_t sxh, int64_t spa, int64_t spt,
                     int64_t scache, Rope r) {
  const int t = blockIdx.x;
  const int h = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  if (h >= n_heads) return;
  const int d0 = lane * kDimsPerLane;

  const uint2 raw = *reinterpret_cast<const uint2*>(x + (int64_t)t * sxt +
                                                    (int64_t)h * sxh + d0);
  const half* xv = reinterpret_cast<const half*>(&raw);
  float xf[kDimsPerLane];
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < kDimsPerLane; ++i) {
    xf[i] = __half2float(xv[i]);
    ss += xf[i] * xf[i];
  }
#pragma unroll
  for (int m = 16; m >= 1; m >>= 1) ss += __shfl_xor(ss, m, 32);
  const float inv = rsqrtf(ss * (1.0f / kHeadDim) + eps);

  half n[kDimsPerLane];
#pragma unroll
  for (int i = 0; i < kDimsPerLane; ++i) {
    const float wf = __half2float(w[d0 + i]) + 1.0f;
    n[i] = __float2half_rn((xf[i] * inv) * wf);
  }

  // Partner halves of the rotated dims (lane +- rd/8).
  uint2 mine;
  half* mh = reinterpret_cast<half*>(&mine);
#pragma unroll
  for (int i = 0; i < kDimsPerLane; ++i) mh[i] = n[i];
  const int pl = r.half / kDimsPerLane;
  uint2 other;
  other.x = __shfl_xor(mine.x, pl, 32);
  other.y = __shfl_xor(mine.y, pl, 32);
  const half* oh = reinterpret_cast<const half*>(&other);

  uint2 res;
  half* rh = reinterpret_cast<half*>(&res);
  if (d0 < r.rd) {
    const bool lo = d0 < r.half;
    const int f0 = lo ? d0 : d0 - r.half;
#pragma unroll
    for (int i = 0; i < kDimsPerLane; ++i) {
      const int f = f0 + i;
      const int axis = r.mrope ? mrope_axis(f, r) : 0;
      const int64_t p = pos[(int64_t)axis * spa + (int64_t)t * spt];
      const half* row = cache + p * scache;
      rh[i] = rot(n[i], oh[i], row[f], row[r.half + f], lo);
    }
  } else {
#pragma unroll
    for (int i = 0; i < kDimsPerLane; ++i) rh[i] = n[i];
  }
  *reinterpret_cast<uint2*>(out + ((int64_t)t * n_heads + h) * kHeadDim + d0) =
      res;
}

}  // namespace qsa_norm_rope_rdna3
}  // namespace vllm

void qsa_norm_rope_rdna3(torch::Tensor x, torch::Tensor out,
                         torch::Tensor weight, torch::Tensor positions,
                         torch::Tensor cos_sin_cache, double eps,
                         int64_t rotary_dim, int64_t section_t,
                         int64_t section_h, int64_t section_w,
                         bool interleaved) {
  using namespace vllm::qsa_norm_rope_rdna3;
  TORCH_CHECK(x.dim() == 3 && x.size(2) == kHeadDim && x.stride(2) == 1 &&
                  x.scalar_type() == torch::kHalf,
              "qsa_norm_rope_rdna3: x must be fp16 [T, H, 128], unit stride");
  TORCH_CHECK(x.size(1) >= 1 && x.size(1) <= 8, "at most 8 heads");
  TORCH_CHECK(x.stride(0) % 4 == 0 && x.stride(1) % 4 == 0 &&
                  reinterpret_cast<uintptr_t>(x.data_ptr()) % 8 == 0,
              "x rows must be 8-byte aligned");
  TORCH_CHECK(out.is_contiguous() && out.sizes() == x.sizes() &&
              out.scalar_type() == torch::kHalf);
  TORCH_CHECK(weight.is_contiguous() && weight.numel() == kHeadDim &&
              weight.scalar_type() == torch::kHalf);
  TORCH_CHECK(cos_sin_cache.dim() == 2 && cos_sin_cache.stride(1) == 1 &&
              cos_sin_cache.size(1) == rotary_dim &&
              cos_sin_cache.scalar_type() == torch::kHalf);
  const int half_rd = (int)rotary_dim / 2;
  const int pl = half_rd / kDimsPerLane;
  TORCH_CHECK(rotary_dim > 0 && rotary_dim <= kHeadDim &&
                  half_rd % kDimsPerLane == 0 && (pl & (pl - 1)) == 0,
              "rotary_dim must be 8 * 2^k <= 128");
  TORCH_CHECK(positions.scalar_type() == torch::kInt64 &&
              (positions.dim() == 1 || positions.dim() == 2));
  const bool mrope = positions.dim() == 2;
  TORCH_CHECK(!mrope || positions.size(0) == 3);
  TORCH_CHECK(positions.size(-1) == x.size(0));
  const int tokens = x.size(0);
  if (tokens == 0) return;
  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream();
  Rope r{(int)rotary_dim, half_rd, (int)section_t, (int)section_h,
         (int)section_w,  mrope,   interleaved};
  const int heads = x.size(1);
  using vllm::qsa_norm_rope_rdna3::norm_rope_kernel;
  norm_rope_kernel<<<tokens, 32 * heads, 0, stream>>>(
      (const half*)x.data_ptr(), (half*)out.data_ptr(),
      (const half*)weight.data_ptr(), positions.data_ptr<int64_t>(),
      (const half*)cos_sin_cache.data_ptr(), (float)eps, heads, x.stride(0),
      x.stride(1), mrope ? positions.stride(0) : 0, positions.stride(-1),
      cos_sin_cache.stride(0), r);
}
