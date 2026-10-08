// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// fp16 GEMM for a few tokens (5..32) on gfx1100: out[N, M] = x[N, K] @ w[M,
// K]^T.
//
// Decode with several sequences and speculative tokens hands the dense layers
// 8..32 rows. wvSplitK stops at 16 and re-reads the weight per chunk above it,
// hipBLASLt picks slow tiles for these shapes; both read the weight more than
// once or at a fraction of the bandwidth. Here every weight byte is read once:
// one v_wmma_f32_16x16x16_f16 multiplies 16 output features by 16 tokens over
// 16 K, and a second one reuses the same weight fragment for tokens 16..31.
//
// Block: 8 waves = RT row tiles (16 features each) x WK waves splitting K.
// Grid: (row tiles / RT, split). With split > 1 every block writes fp32
// partials that skinny_wmma_reduce sums; otherwise it writes the output.

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

namespace vllm {
namespace skinny_wmma {

typedef _Float16 h16v __attribute__((ext_vector_type(16)));
typedef _Float16 h8v __attribute__((ext_vector_type(8)));
typedef float f8v __attribute__((ext_vector_type(8)));

constexpr int kWaves = 8;
constexpr int kChunk = 16;                // K per WMMA
constexpr int kGroup = 4;                 // chunks per load group (64 K)
constexpr int kGroupK = kChunk * kGroup;  // 64

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
    defined(__gfx1103__) || !defined(__HIP_DEVICE_COMPILE__)
  #define SKINNY_WMMA_BODY 1
#endif

// 16 consecutive halves of a row, or zeros when the row or K is out of range
// (K is a multiple of 16, so a chunk is either fully in or fully out).
__device__ __forceinline__ h16v load16(const _Float16* p, bool ok) {
  if (!ok) return (h16v){};
  const h8v lo = *reinterpret_cast<const h8v*>(p);
  const h8v hi = *reinterpret_cast<const h8v*>(p + 8);
  return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                 12, 13, 14, 15);
}

// Two consecutive 16-K chunks of a row for a WMMA operand, where lanes l and
// l + 16 must hold the same chunk: lanes 0-15 load the first and lanes 16-31
// the second, then v_permlanex16 hands each half the other's.
__device__ __forceinline__ void load16x2(const _Float16* p, bool ok, int lane,
                                         h16v& c0, h16v& c1) {
  const bool hi = lane >= 16;
  const h16v mine = load16(p + (hi ? kChunk : 0), ok);
  typedef int i8v __attribute__((ext_vector_type(8)));
  const i8v m = __builtin_bit_cast(i8v, mine);
  i8v o;
#pragma unroll
  for (int i = 0; i < 8; ++i)
    o[i] = __builtin_amdgcn_permlanex16(m[i], m[i], 0x76543210, 0xfedcba98,
                                        false, false);
  const h16v other = __builtin_bit_cast(h16v, o);
  c0 = hi ? other : mine;
  c1 = hi ? mine : other;
}

template <int NT, int RT>
__global__ void __launch_bounds__(kWaves * 32)
    skinny_wmma_kernel(const _Float16* __restrict__ x,
                       const _Float16* __restrict__ w,
                       _Float16* __restrict__ out, float* __restrict__ part,
                       const int N, const int M, const int K,
                       const int k_per_block) {
#ifdef SKINNY_WMMA_BODY
  constexpr int WK = kWaves / RT;
  const int lane = threadIdx.x & 31;
  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  const int rt = wave % RT, kw = wave / RT;
  const int f0 = (blockIdx.x * RT + rt) * 16;
  const int r = lane & 15;
  const int k_begin = blockIdx.y * k_per_block;
  const int k_end = min(K, k_begin + k_per_block);

  const bool row_ok = f0 + r < M;
  const _Float16* wrow = w + (int64_t)(row_ok ? f0 + r : 0) * K;
  const _Float16* xrow[NT];
  bool tok_ok[NT];
  #pragma unroll
  for (int t = 0; t < NT; ++t) {
    const int tok = t * 16 + r;
    tok_ok[t] = tok < N;
    xrow[t] = x + (int64_t)(tok_ok[t] ? tok : 0) * K;
  }

  f8v acc[NT];
  #pragma unroll
  for (int t = 0; t < NT; ++t) acc[t] = (f8v){};

  // Waves of a row tile take groups of 64 K in turn: the 4 waves read 4
  // consecutive 128-byte stretches of each of the 16 rows. The next group of
  // weights is requested before this one's x, so it is in flight while the
  // WMMAs run.
  const int step = WK * kGroupK;
  int k = k_begin + kw * kGroupK;
  h16v a[kGroup];
  #pragma unroll
  for (int c = 0; c < kGroup; c += 2) {
    const int kc = k + c * kChunk;
    load16x2(wrow + kc, row_ok && kc < k_end, lane, a[c], a[c + 1]);
  }
  for (; k < k_end; k += step) {
    h16v b[kGroup][NT];
  #pragma unroll
    for (int c = 0; c < kGroup; ++c) {
      const int kc = k + c * kChunk;
  #pragma unroll
      for (int t = 0; t < NT; ++t)
        b[c][t] = load16(xrow[t] + kc, tok_ok[t] && kc < k_end);
    }
    h16v an[kGroup];
  #pragma unroll
    for (int c = 0; c < kGroup; c += 2) {
      const int kc = k + step + c * kChunk;
      load16x2(wrow + kc, row_ok && kc < k_end, lane, an[c], an[c + 1]);
    }
  #pragma unroll
    for (int c = 0; c < kGroup; ++c)
  #pragma unroll
      for (int t = 0; t < NT; ++t)
        acc[t] =
            __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[c], b[c][t], acc[t]);
  #pragma unroll
    for (int c = 0; c < kGroup; ++c) a[c] = an[c];
  }

  // acc[t][i] of lane l is feature f0 + 2i + (l >> 4), token 16t + (l & 15).
  __shared__ float red[kWaves][NT][8][32];
  #pragma unroll
  for (int t = 0; t < NT; ++t)
  #pragma unroll
    for (int i = 0; i < 8; ++i) red[wave][t][i][lane] = acc[t][i];
  __syncthreads();
  if (kw != 0) return;
  #pragma unroll
  for (int t = 0; t < NT; ++t) {
  #pragma unroll
    for (int i = 0; i < 8; ++i) {
      float v = 0.f;
  #pragma unroll
      for (int j = 0; j < WK; ++j) v += red[j * RT + rt][t][i][lane];
      const int f = f0 + 2 * i + (lane >> 4);
      const int tok = t * 16 + r;
      if (f >= M || tok >= N) continue;
      if (part != nullptr)
        part[((int64_t)blockIdx.y * N + tok) * M + f] = v;
      else
        out[(int64_t)tok * M + f] = (_Float16)v;
    }
  }
#endif
}

__global__ void skinny_wmma_reduce(const float* __restrict__ part,
                                   _Float16* __restrict__ out, const int NM,
                                   const int split) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= NM) return;
  float v = 0.f;
  for (int s = 0; s < split; ++s) v += part[(int64_t)s * NM + i];
  out[i] = (_Float16)v;
}

template <int NT, int RT>
void launch(const _Float16* x, const _Float16* w, _Float16* out, float* part,
            int N, int M, int K, int split, int k_per_block,
            hipStream_t stream) {
  const dim3 grid((M + 16 * RT - 1) / (16 * RT), split);
  skinny_wmma_kernel<NT, RT>
      <<<grid, kWaves * 32, 0, stream>>>(x, w, out, part, N, M, K, k_per_block);
}

}  // namespace skinny_wmma
}  // namespace vllm

// x [N, K] and w [M, K] fp16 contiguous, 1 <= N <= 32, K % 16 == 0.
// Returns [N, M]. split > 1 splits K over that many blocks per row tile.
torch::Tensor skinny_wmma_f16(const torch::Tensor& x, const torch::Tensor& w,
                              int64_t split) {
  using namespace vllm::skinny_wmma;
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && x.size(1) == w.size(1),
              "skinny_wmma_f16: x [N, K], w [M, K]");
  TORCH_CHECK(x.scalar_type() == at::kHalf && w.scalar_type() == at::kHalf,
              "skinny_wmma_f16: fp16 only");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous(),
              "skinny_wmma_f16: contiguous operands");
  const int N = x.size(0), K = x.size(1), M = w.size(0);
  TORCH_CHECK(N >= 1 && N <= 32, "skinny_wmma_f16: 1 <= N <= 32");
  TORCH_CHECK(K % 16 == 0, "skinny_wmma_f16: K % 16 == 0");
  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto out = torch::empty({N, M}, x.options());
  // K per block, a multiple of the 64-K load group; split <= 0 aims at ~2
  // blocks per CU with at least one group per wave.
  // Split only when the row tiles alone leave CUs idle: every split costs
  // a reduce launch.
  const bool tall = M >= 16 * 2 * 384;
  const int row_blocks = (M + 16 * (tall ? 2 : 1) - 1) / (16 * (tall ? 2 : 1));
  if (split <= 0)
    split = row_blocks >= 96
                ? 1
                : std::max<int64_t>(
                      1, std::min<int64_t>((192 + row_blocks - 1) / row_blocks,
                                           K / (kWaves * kGroupK)));
  int kpb = (K + split - 1) / split;
  kpb = (kpb + kGroupK - 1) / kGroupK * kGroupK;
  const int s = (K + kpb - 1) / kpb;
  torch::Tensor part;
  if (s > 1) part = torch::empty({s, N, M}, x.options().dtype(at::kFloat));
  const auto* xp = reinterpret_cast<const _Float16*>(x.data_ptr());
  const auto* wp = reinterpret_cast<const _Float16*>(w.data_ptr());
  auto* op = reinterpret_cast<_Float16*>(out.data_ptr());
  float* pp = s > 1 ? part.data_ptr<float>() : nullptr;
  // One row tile per block, 8 waves along K; two for very tall weights.
  if (N <= 16) {
    if (tall)
      launch<1, 2>(xp, wp, op, pp, N, M, K, s, kpb, stream);
    else
      launch<1, 1>(xp, wp, op, pp, N, M, K, s, kpb, stream);
  } else {
    if (tall)
      launch<2, 2>(xp, wp, op, pp, N, M, K, s, kpb, stream);
    else
      launch<2, 1>(xp, wp, op, pp, N, M, K, s, kpb, stream);
  }
  if (s > 1) {
    const int NM = N * M;
    skinny_wmma_reduce<<<(NM + 255) / 256, 256, 0, stream>>>(pp, op, NM, s);
  }
  return out;
}
