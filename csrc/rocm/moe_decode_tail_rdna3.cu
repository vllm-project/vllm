// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Decode tail of a Qwen3-Next style MoE block on RDNA3 (gfx1100), one launch:
//   - softmax top-k routing of the router logits (same arithmetic and tie
//     order as topkGatingSoftmax for 384 experts on wave32),
//   - shared expert: SiLU-and-mul, down projection and sigmoid gate, written
//     to the fp32 `out`, which the routed experts then accumulate into,
//   - zeroes the routed gate_up workspace that the expert kernels accumulate
//     into with atomics.
// Replaces topk_softmax, silu_and_mul, the down GEMM, the gate kernel, the
// zero fill and the shared + routed add.

#include <cstdint>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

namespace vllm {
namespace moe_decode_tail_rdna3 {

constexpr int kThreads = 256;
constexpr int kWaves = kThreads / 32;
constexpr int kRowsPerBlock = 64;
constexpr int kMaxI = 256;
constexpr int kExperts = 384;
constexpr int kMaxTopK = 16;

struct alignas(16) half8 {
  half v[8];
};

// One wave per token row: the topkGatingSoftmax schedule for 384 experts,
// fp16 logits, 4-byte loads, 32 threads per row.
__device__ __forceinline__ void topk_row(const half* __restrict__ logits, float* topk_w,
                         int32_t* topk_ids, int k, bool renormalize, int lane) {
  constexpr int kLdg = 6, kVpt = 12, kColsPerLdg = 64;
  float v[kVpt];
#pragma unroll
  for (int ii = 0; ii < kLdg; ++ii) {
    const float2 f = __half22float2(
        *reinterpret_cast<const __half2*>(logits + lane * 2 + ii * kColsPerLdg));
    v[2 * ii] = f.x;
    v[2 * ii + 1] = f.y;
  }
  float mx = v[0];
#pragma unroll
  for (int i = 1; i < kVpt; ++i) mx = max(mx, v[i]);
#pragma unroll
  for (int s = 16; s > 0; s >>= 1) mx = max(mx, __shfl_xor(mx, s, 32));
  float sum = 0.f;
#pragma unroll
  for (int i = 0; i < kVpt; ++i) {
    v[i] = expf(v[i] - mx);
    sum += v[i];
  }
#pragma unroll
  for (int s = 16; s > 0; s >>= 1) sum += __shfl_xor(sum, s, 32);
  const float rs = 1.f / sum;
#pragma unroll
  for (int i = 0; i < kVpt; ++i) {
    v[i] *= rs;
    if (isnan(v[i]) || isinf(v[i])) v[i] = 0.f;
  }
  float choice[kVpt];
#pragma unroll
  for (int i = 0; i < kVpt; ++i) choice[i] = v[i];

  __shared__ float sel[kMaxTopK];
  float selected_sum = 0.f;
#pragma unroll 1
  for (int kk = 0; kk < k; ++kk) {
    float best_c = choice[0], best = v[0];
    int expert = lane * 2;
#pragma unroll
    for (int ii = 0; ii < kLdg; ++ii)
#pragma unroll
      for (int j = 0; j < 2; ++j)
        if (choice[2 * ii + j] > best_c) {
          best_c = choice[2 * ii + j];
          best = v[2 * ii + j];
          expert = lane * 2 + ii * kColsPerLdg + j;
        }
#pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
      const float oc = __shfl_xor(best_c, s, 32);
      const float ov = __shfl_xor(best, s, 32);
      const int oe = __shfl_xor(expert, s, 32);
      if (oc > best_c || (oc == best_c && oe < expert)) {
        best_c = oc;
        best = ov;
        expert = oe;
      }
    }
    if (lane == 0) {
      sel[kk] = best;
      topk_ids[kk] = expert;
    }
    if (renormalize) selected_sum += best;
    if (kk + 1 < k && lane == (expert / 2) % 32) {
      const int slot = (expert / kColsPerLdg) * 2 + expert % 2;
#pragma unroll
      for (int i = 0; i < kVpt; ++i)
        if (i == slot) choice[i] = -10000.f;
    }
  }
  if (lane == 0) {
    float scale = 1.f;
    if (renormalize) scale /= selected_sum > 0.f ? selected_sum : 1.f;
    for (int kk = 0; kk < k; ++kk) topk_w[kk] = sel[kk] * scale;
  }
}

template <int MT>
__global__ void __launch_bounds__(kThreads)
    tail_kernel(const half* __restrict__ logits, int64_t s_logits,
                const half* __restrict__ gu, int64_t s_gu,
                const half* __restrict__ x, int64_t s_x,
                const half* __restrict__ w_gate,
                const half* __restrict__ w_down, float* __restrict__ out,
                int64_t s_out, float* __restrict__ topk_w,
                int32_t* __restrict__ topk_ids, uint4* __restrict__ zero_buf,
                int64_t zero_n16, int M, int H, int I, int k, bool renormalize,
                int down_blocks) {
  const int t = threadIdx.x;
  const int lane = t & 31;
  const int wave = t >> 5;

  for (int64_t i = (int64_t)blockIdx.x * kThreads + t; i < zero_n16;
       i += (int64_t)gridDim.x * kThreads)
    zero_buf[i] = make_uint4(0, 0, 0, 0);

  if (blockIdx.x >= down_blocks) {
    const int m = blockIdx.x - down_blocks;
    if (wave == 0)
      topk_row(logits + m * s_logits, topk_w + m * k, topk_ids + m * k, k,
               renormalize, lane);
    return;
  }

  __shared__ __align__(16) half h[MT][kMaxI];
  __shared__ float part[MT][kWaves];
  __shared__ float gate[MT];

  // SiLU-and-mul of the shared gate_up rows, rounded to fp16 like the
  // activation tensor it replaces.
  for (int e = t; e < MT * I; e += kThreads) {
    const int m = e / I, i = e - m * I;
    float r = 0.f;
    if (m < M) {
      const float g = __half2float(gu[m * s_gu + i]);
      const float u = __half2float(gu[m * s_gu + I + i]);
      r = g / (1.f + __expf(-g)) * u;
    }
    h[m][i] = __float2half_rn(r);
  }

  // sigmoid(x . w_gate): the shared_expert_gate_rdna3 reduction order.
  float acc[MT];
#pragma unroll
  for (int m = 0; m < MT; ++m) acc[m] = 0.f;
  for (int c = t; c < H / 8; c += kThreads) {
    const half8 b = *reinterpret_cast<const half8*>(w_gate + c * 8);
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      if (m < M) {
        const half8 a = *reinterpret_cast<const half8*>(x + m * s_x + c * 8);
#pragma unroll
        for (int i = 0; i < 8; ++i)
          acc[m] += __half2float(a.v[i]) * __half2float(b.v[i]);
      }
    }
  }
#pragma unroll
  for (int m = 0; m < MT; ++m) {
#pragma unroll
    for (int s = 16; s >= 1; s >>= 1) acc[m] += __shfl_xor(acc[m], s, 32);
    if (lane == 0) part[m][wave] = acc[m];
  }
  __syncthreads();
  if (t < MT) {
    float g = 0.f;
#pragma unroll
    for (int i = 0; i < kWaves; ++i) g += part[t][i];
    const float gl = __half2float(__float2half_rn(g));
    gate[t] = __half2float(__float2half_rn(1.f / (1.f + expf(-gl))));
  }
  __syncthreads();

  // Down projection: four lanes per output row split I in 16-byte chunks.
  const int q = lane & 3;
  const int n = blockIdx.x * kRowsPerBlock + wave * 8 + (lane >> 2);
  float d[MT];
#pragma unroll
  for (int m = 0; m < MT; ++m) d[m] = 0.f;
  if (n < H) {
    const half* wr = w_down + (int64_t)n * I;
    for (int c = q; c < I / 8; c += 4) {
      const uint4 wv = *reinterpret_cast<const uint4*>(wr + c * 8);
      const half2* w2 = reinterpret_cast<const half2*>(&wv);
#pragma unroll
      for (int m = 0; m < MT; ++m) {
        if (m < M) {
          const uint4 hv = *reinterpret_cast<const uint4*>(&h[m][c * 8]);
          const half2* h2 = reinterpret_cast<const half2*>(&hv);
#pragma unroll
          for (int j = 0; j < 4; ++j)
            d[m] = __builtin_amdgcn_fdot2(w2[j], h2[j], d[m], false);
        }
      }
    }
  }
#pragma unroll
  for (int m = 0; m < MT; ++m) {
    d[m] += __shfl_xor(d[m], 1, 32);
    d[m] += __shfl_xor(d[m], 2, 32);
  }
  if (n < H) {
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      if (m < M && (m & 3) == q) {
        const float o = __half2float(__float2half_rn(d[m]));
        out[m * s_out + n] = __half2float(__float2half_rn(o * gate[m]));
      }
    }
  }
}

}  // namespace moe_decode_tail_rdna3
}  // namespace vllm

// logits    [M, >=384] fp16, row stride s_logits (unit column stride)
// gate_up   [M, >=2I] fp16: shared expert [gate | up]
// x         [M, H] fp16: MoE input, for the shared expert gate
// w_gate    [1, H] fp16; w_down [H, I] fp16
// out       [M, H] fp32 <- sigmoid(x . w_gate) * down(silu(gate) * up),
//           rounded to fp16 values
// topk_w    [M, k] fp32, topk_ids [M, k] int32
// zero_buf  fp16 tensor zeroed in full (numel % 8 == 0)
void moe_decode_tail_rdna3(torch::Tensor logits, torch::Tensor gate_up,
                           torch::Tensor x, torch::Tensor w_gate,
                           torch::Tensor w_down, torch::Tensor out,
                           torch::Tensor topk_w, torch::Tensor topk_ids,
                           torch::Tensor zero_buf, bool renormalize) {
  using namespace vllm::moe_decode_tail_rdna3;
  const int M = x.size(0), H = x.size(1), I = w_down.size(1);
  const int k = topk_w.size(1);
  for (const auto* p : {&logits, &gate_up, &x, &w_gate, &w_down, &zero_buf})
    TORCH_CHECK(p->scalar_type() == torch::kHalf && p->stride(-1) == 1 &&
                    reinterpret_cast<uintptr_t>(p->data_ptr()) % 16 == 0,
                "moe_decode_tail_rdna3: fp16, unit column stride, aligned");
  TORCH_CHECK(logits.size(0) == M && logits.size(1) >= kExperts,
              "moe_decode_tail_rdna3: logits [M, 384]");
  TORCH_CHECK(gate_up.size(0) == M && gate_up.size(1) >= 2 * I,
              "moe_decode_tail_rdna3: gate_up [M, 2I]");
  TORCH_CHECK(w_down.size(0) == H && w_gate.numel() == H && out.size(0) == M &&
                  out.size(1) == H,
              "moe_decode_tail_rdna3: shape mismatch");
  TORCH_CHECK(out.scalar_type() == torch::kFloat && out.stride(1) == 1,
              "moe_decode_tail_rdna3: out fp32 with unit column stride");
  TORCH_CHECK(w_down.is_contiguous() && w_gate.is_contiguous(),
              "moe_decode_tail_rdna3: contiguous weights");
  TORCH_CHECK(I % 8 == 0 && I <= kMaxI && H % 8 == 0 &&
                  x.stride(0) % 8 == 0 && logits.stride(0) % 2 == 0,
              "moe_decode_tail_rdna3: unsupported sizes");
  TORCH_CHECK(topk_w.scalar_type() == torch::kFloat &&
                  topk_ids.scalar_type() == torch::kInt &&
                  topk_w.is_contiguous() && topk_ids.is_contiguous() &&
                  topk_w.size(0) == M && topk_ids.size(0) == M &&
                  topk_ids.size(1) == k && k <= kMaxTopK,
              "moe_decode_tail_rdna3: topk outputs [M, k] fp32 / int32");
  TORCH_CHECK(zero_buf.is_contiguous() && zero_buf.numel() % 8 == 0,
              "moe_decode_tail_rdna3: zero_buf");
  if (M == 0) return;
  TORCH_CHECK(M <= 32, "moe_decode_tail_rdna3: at most 32 rows");

  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int down_blocks = (H + kRowsPerBlock - 1) / kRowsPerBlock;
  const dim3 grid(down_blocks + M);

#define TAIL_LAUNCH(MT)                                                       \
  tail_kernel<MT><<<grid, kThreads, 0, stream>>>(                             \
      (const half*)logits.data_ptr(), logits.stride(0),                       \
      (const half*)gate_up.data_ptr(), gate_up.stride(0),                     \
      (const half*)x.data_ptr(), x.stride(0),                                 \
      (const half*)w_gate.data_ptr(), (const half*)w_down.data_ptr(),         \
      out.data_ptr<float>(), out.stride(0), topk_w.data_ptr<float>(),         \
      topk_ids.data_ptr<int32_t>(), (uint4*)zero_buf.data_ptr(),              \
      zero_buf.numel() / 8, M, H, I, k, renormalize, down_blocks)

  if (M <= 1)
    TAIL_LAUNCH(1);
  else if (M <= 2)
    TAIL_LAUNCH(2);
  else if (M <= 4)
    TAIL_LAUNCH(4);
  else if (M <= 8)
    TAIL_LAUNCH(8);
  else if (M <= 16)
    TAIL_LAUNCH(16);
  else
    TAIL_LAUNCH(32);
#undef TAIL_LAUNCH
}
