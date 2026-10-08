// Fused HyperConnection combine+mix for qwen4_exp decode on gfx1100 (HC=4,
// H=2560, lora R=320).
//
// Reference (vllm/models/qwen4_exp/amd/hyperconnection.py, combine_and_mix with
// use_combine):
//   h'  = fp16(h + block_out * 2*sigmoid(inj/HC))           per stream
//   xn  = fp16(rms(h') per stream * (1 + w))                  Gemma RMSNorm
//   dl  = fp16(xn @ Wd^T)                Wd: [R + HC (+pad), HC*H]
//   s   = fp16(silu(dl[:R] / HC)),  inj' = dl[R:R+HC]
//   g   = fp16(s @ Wu^T)                 Wu: [HC*H, R]
//   out = fp16(sum_b sigmoid(g_b) * xn_b / HC)
//
// The unfused path is five kernels per call and ~30 us at M=1 on a 7900 XTX
// (13 MB of weights, replicated on every TP rank, 96 calls per decode step).
// Here:
//   down: block i owns Wd rows [4i, 4i+4) over the whole K. Wave w covers half
//         a stream, so the stream RMS is two waves exchanging sums through
//         LDS, and the cross-wave sum of the dot products is a fixed-order LDS
//         reduction (deterministic, no split-K partials).
//   up:   each wave produces DPW outputs d from the HC rows b*H+d of Wu (8
//         lanes per row) and applies the gate mix in registers.
// Valid for 1 <= M <= 8 tokens; prefill keeps the unfused path.
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

namespace {

constexpr int HC = 4;
constexpr int H = 2560;
constexpr int D = HC * H;  // 10240
constexpr int R = 320;
constexpr int RO =
    R + HC;  // rows of Wd actually used (the rest is alignment padding)
constexpr int RPB = 4;                    // Wd rows per block
constexpr int NB = (RO + RPB - 1) / RPB;  // 81 blocks
constexpr int T1 = 256;  // 8 waves: wave w covers K [w*1280, (w+1)*1280)
constexpr int KW = D / (T1 / 32);   // 1280 = half a stream
constexpr int CPL = KW / (32 * 8);  // 5 chunks of 8 halves per lane
constexpr int MMAX = 8;

using half8 = __attribute__((__vector_size__(8 * sizeof(_Float16)))) _Float16;
using half2v = __attribute__((__vector_size__(2 * sizeof(_Float16)))) _Float16;

__device__ __forceinline__ float sigm(float x) {
  return 1.f / (1.f + __expf(-x));
}

// Lane xor within a row of 16 via DPP row_xmask (gfx10+), and across the two
// rows via v_permlanex16: ~1-2 cycles each instead of an LDS round trip for
// ds_bpermute.
template <int X>
__device__ __forceinline__ float xor_lane(float v) {
  if constexpr (X == 16) {
    const int i = __float_as_int(v);
    return __int_as_float(__builtin_amdgcn_permlanex16(
        i, i, 0x76543210, 0xfedcba98, false, false));
  } else {
    return __int_as_float(__builtin_amdgcn_update_dpp(
        0, __float_as_int(v), 0x160 + X, 0xf, 0xf, false));
  }
}

__device__ __forceinline__ float wave_sum(float v) {
  v += xor_lane<1>(v);
  v += xor_lane<2>(v);
  v += xor_lane<4>(v);
  v += xor_lane<8>(v);
  v += xor_lane<16>(v);
  return v;
}

__device__ __forceinline__ float dot8(const half8& a, const half8& b,
                                      float acc) {
#pragma unroll
  for (int e = 0; e < 8; e += 2) {
    half2v x = {a[e], a[e + 1]};
    half2v y = {b[e], b[e + 1]};
    acc = __builtin_amdgcn_fdot2(x, y, acc, false);
  }
  return acc;
}

// Down half: Wd rows [RPB*i, RPB*i + RPB) of block i over the whole K. The
// weights are requested first; the combine + RMSNorm overlaps with their
// arrival. Without xn_in, pass 1 only sums squares and pass 2 rebuilds h' token
// by token, which keeps register use flat in M. Each block writes h' and xn for
// the 8-column chunks g with g % NB == i.
template <int M>
__global__ void __launch_bounds__(T1)
    hc_down_kernel(const _Float16* __restrict__ h,
                   const _Float16* __restrict__ bo,
                   const _Float16* __restrict__ inj,
                   const _Float16* __restrict__ nw, const int nw_shared,
                   const _Float16* __restrict__ wd,
                   _Float16* __restrict__ out_h, _Float16* __restrict__ out_xn,
                   _Float16* __restrict__ s_out, _Float16* __restrict__ inj_out,
                   const float eps, const int Mr,
                   const _Float16* __restrict__ xn_in) {
  __shared__ float lds[(T1 / 32) * M * (1 + RPB)];
  float (*sq)[M] = reinterpret_cast<float (*)[M]>(lds);
  float (*red)[RPB][M] =
      reinterpret_cast<float (*)[RPB][M]>(lds + (T1 / 32) * M);
  const int t = threadIdx.x, lane = t & 31, wv = t >> 5;
  const int b = wv >> 1;  // stream of this wave
  const int r0 = blockIdx.x * RPB;

  half8 wr[RPB][CPL];
#pragma unroll
  for (int p = 0; p < RPB; ++p) {
    const int r = min(r0 + p, RO - 1);
#pragma unroll
    for (int u = 0; u < CPL; ++u)
      wr[p][u] = *reinterpret_cast<const half8*>(
          &wd[(size_t)r * D + wv * KW + (lane + 32 * u) * 8]);
  }

  // Pass 1 (no xn_in): sum of squares of h' per token, nothing kept in
  // registers.
  float acc[RPB][M];
#pragma unroll
  for (int p = 0; p < RPB; ++p)
#pragma unroll
    for (int m = 0; m < M; ++m) acc[p][m] = 0.f;
  float cc[M];
  if (!xn_in) {
#pragma unroll
    for (int m = 0; m < M; ++m) {
      float ss = 0.f;
      cc[m] = 0.f;
      if (m < Mr) {
        cc[m] = 2.f * sigm((float)inj[m * HC + b] / HC);
#pragma unroll
        for (int u = 0; u < CPL; ++u) {
          const int k = wv * KW + (lane + 32 * u) * 8;
          const half8 hh = *reinterpret_cast<const half8*>(&h[m * D + k]);
          const half8 bb =
              *reinterpret_cast<const half8*>(&bo[m * H + (k - b * H)]);
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const float v =
                (float)(_Float16)((float)hh[e] + (float)bb[e] * cc[m]);
            ss += v * v;
          }
        }
      }
      ss = wave_sum(ss);
      if (lane == 0) sq[wv][m] = ss;
    }
    __syncthreads();
  }
  // Pass 2, one token at a time: rebuild h' (L2-hot), normalize, write the
  // owned chunks and accumulate the dot products straight away.
  half8 ww[CPL];
  if (!xn_in) {
#pragma unroll
    for (int u = 0; u < CPL; ++u) {
      const int k = wv * KW + (lane + 32 * u) * 8;
      ww[u] = *reinterpret_cast<const half8*>(&nw[nw_shared ? (k - b * H) : k]);
    }
  }
#pragma unroll
  for (int m = 0; m < M; ++m) {
    if (m >= Mr) break;
    const float rr =
        xn_in ? 0.f : rsqrtf((sq[2 * b][m] + sq[2 * b + 1][m]) / H + eps);
#pragma unroll
    for (int u = 0; u < CPL; ++u) {
      const int k = wv * KW + (lane + 32 * u) * 8;
      half8 xo;
      if (xn_in) {
        xo = *reinterpret_cast<const half8*>(&xn_in[m * D + k]);
      } else {
        const half8 hh = *reinterpret_cast<const half8*>(&h[m * D + k]);
        const half8 bb =
            *reinterpret_cast<const half8*>(&bo[m * H + (k - b * H)]);
        half8 hv;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          hv[e] = (_Float16)((float)hh[e] + (float)bb[e] * cc[m]);
          float y = (float)hv[e] * rr;
          y += y * (float)ww[u][e];
          xo[e] = (_Float16)y;
        }
        if ((k >> 3) % NB ==
            (int)blockIdx.x) {  // each 8-column chunk has one owner
          *reinterpret_cast<half8*>(&out_h[m * D + k]) = hv;
          *reinterpret_cast<half8*>(&out_xn[m * D + k]) = xo;
        }
      }
#pragma unroll
      for (int p = 0; p < RPB; ++p) acc[p][m] = dot8(wr[p][u], xo, acc[p][m]);
    }
  }
#pragma unroll
  for (int p = 0; p < RPB; ++p) {
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const float a = wave_sum(acc[p][m]);
      if (lane == 0) red[wv][p][m] = a;
    }
  }
  __syncthreads();
  if (t < RPB * M) {
    const int p = t / M, m = t % M, r = r0 + p;
    if (m < Mr && r < RO) {
      float v = 0.f;
#pragma unroll
      for (int w = 0; w < T1 / 32; ++w) v += red[w][p][m];
      const _Float16 dl = (_Float16)v;
      if (r < R) {
        const float x = (float)dl / HC;
        s_out[m * R + r] = (_Float16)(x * sigm(x));
      } else {
        inj_out[m * HC + (r - R)] = dl;
      }
    }
  }
}

constexpr int T2 = 256;               // threads per block, kernel 2 (8 waves)
constexpr int DPW = 4;                // outputs d per wave
constexpr int DPB = DPW * (T2 / 32);  // 32 outputs per block -> 80 blocks
static_assert(H % DPB == 0);

// Up half: gate = s @ Wu^T and the gate mix, DPB outputs per block.
template <int M>
__global__ void __launch_bounds__(T2)
    hc_up_kernel(const _Float16* __restrict__ s,
                 const _Float16* __restrict__ wu,
                 const _Float16* __restrict__ xn, _Float16* __restrict__ out,
                 const int Mr) {
  __shared__ __attribute__((aligned(16))) _Float16 ss[M][R];
  const int t = threadIdx.x, lane = t & 31, wv = t >> 5;
  const int bstream = lane >> 3,
            q = lane & 7;  // 8 lanes per Wu row, one row per stream
  const int d0 = blockIdx.x * DPB + wv * DPW;
  // Weights first (40 chunks of 8 halves per row: lane q takes chunks q, q+8,
  // ..., q+32), then the xn values of the gate mix, then s into LDS: the three
  // latencies overlap.
  half8 wr[DPW][5];
#pragma unroll
  for (int p = 0; p < DPW; ++p) {
    const _Float16* row = wu + (size_t)(bstream * H + d0 + p) * R;
#pragma unroll
    for (int u = 0; u < 5; ++u)
      wr[p][u] = *reinterpret_cast<const half8*>(&row[(q + 8 * u) * 8]);
  }
  float xnv[DPW][M];
#pragma unroll
  for (int p = 0; p < DPW; ++p)
#pragma unroll
    for (int m = 0; m < M; ++m)
      xnv[p][m] =
          (q == 0 && m < Mr) ? (float)xn[m * D + bstream * H + d0 + p] : 0.f;
  for (int i = t; i < M * R; i += T2)
    ss[i / R][i % R] = i < Mr * R ? s[i] : (_Float16)0.f;
  __syncthreads();
#pragma unroll
  for (int m = 0; m < M; ++m) {
    if (m >= Mr) break;
    half8 sv[5];
#pragma unroll
    for (int u = 0; u < 5; ++u)
      sv[u] = *reinterpret_cast<const half8*>(&ss[m][(q + 8 * u) * 8]);
#pragma unroll
    for (int p = 0; p < DPW; ++p) {
      float a = 0.f;
#pragma unroll
      for (int u = 0; u < 5; ++u) a = dot8(wr[p][u], sv[u], a);
      a += xor_lane<4>(a);
      a += xor_lane<2>(a);
      a += xor_lane<1>(a);
      const float g = (float)(_Float16)a;
      float mix = (q == 0) ? sigm(g) * xnv[p][m] : 0.f;
      mix += xor_lane<8>(mix);
      mix += xor_lane<16>(mix);
      if (lane == 0) out[m * H + d0 + p] = (_Float16)(mix / HC);
    }
  }
}

template <int M>
void launch(const at::Tensor& h, const at::Tensor& bo, const at::Tensor& inj,
            const at::Tensor& nw, const at::Tensor& wd, const at::Tensor& wu,
            at::Tensor& out_h, at::Tensor& xn, at::Tensor& s,
            at::Tensor& inj_out, at::Tensor& out, float eps, hipStream_t st,
            const _Float16* xn_in = nullptr) {
  auto P = [](const at::Tensor& x) {
    return reinterpret_cast<const _Float16*>(x.data_ptr());
  };
  auto W = [](at::Tensor& x) {
    return reinterpret_cast<_Float16*>(x.data_ptr());
  };
  hc_down_kernel<M><<<NB, T1, 0, st>>>(
      P(h), P(bo), P(inj), P(nw), nw.numel() == H, P(wd), W(out_h), W(xn), W(s),
      W(inj_out), eps, (int)h.size(0), xn_in);
  hc_up_kernel<M><<<H / DPB, T2, 0, st>>>(P(s), P(wu), xn_in ? xn_in : P(xn),
                                          W(out), (int)h.size(0));
}

template <typename... Args>
void dispatch(int M, Args&&... args) {
  if (M == 1)
    launch<1>(args...);
  else if (M == 2)
    launch<2>(args...);
  else if (M <= 4)
    launch<4>(args...);
  else
    launch<8>(args...);
}

void check_weights(const torch::Tensor& wd, const torch::Tensor& wu) {
  TORCH_CHECK(wd.dim() == 2 && wd.size(1) == D && wd.size(0) >= RO,
              "qwen4_hc: bad down weight");
  TORCH_CHECK(wu.dim() == 2 && wu.size(0) == D && wu.size(1) == R,
              "qwen4_hc: bad up weight");
  for (auto* x : {&wd, &wu})
    TORCH_CHECK(x->is_contiguous() && x->scalar_type() == at::kHalf,
                "qwen4_hc: fp16 contiguous");
}

}  // namespace

// HyperConnection combine_and_mix (use_combine) for 1 <= M <= 8 decode tokens.
// Returns (residual h', block input, injection logits).
std::vector<torch::Tensor> qwen4_hc_combine_mix(
    const torch::Tensor& h, const torch::Tensor& block_out,
    const torch::Tensor& inj, const torch::Tensor& norm_w,
    const torch::Tensor& wd, const torch::Tensor& wu, double eps) {
  const int M = h.size(0);
  TORCH_CHECK(M >= 1 && M <= MMAX, "qwen4_hc_combine_mix: 1 <= M <= 8");
  TORCH_CHECK(h.size(1) == D && block_out.size(1) == H && inj.size(1) == HC);
  TORCH_CHECK(norm_w.numel() == H || norm_w.numel() == D);
  for (auto* x : {&h, &block_out, &inj, &norm_w})
    TORCH_CHECK(x->is_contiguous() && x->scalar_type() == at::kHalf,
                "qwen4_hc: fp16 contiguous");
  check_weights(wd, wu);
  const at::cuda::OptionalCUDAGuard guard(device_of(h));
  auto out_h = torch::empty_like(h);
  auto xn = torch::empty_like(h);
  auto s = torch::empty({M, R}, h.options());
  auto inj_out = torch::empty({M, HC}, h.options());
  auto out = torch::empty({M, H}, h.options());
  dispatch(M, h, block_out, inj, norm_w, wd, wu, out_h, xn, s, inj_out, out,
           (float)eps, at::cuda::getCurrentCUDAStream(), nullptr);
  return {out_h, out, inj_out};
}

// The same mix from a precomputed normalized input xn (the combine + RMSNorm
// already ran). Returns (block input, injection logits).
std::vector<torch::Tensor> qwen4_hc_mix_xn(const torch::Tensor& xn,
                                           const torch::Tensor& wd,
                                           const torch::Tensor& wu) {
  const int M = xn.size(0);
  TORCH_CHECK(M >= 1 && M <= MMAX, "qwen4_hc_mix_xn: 1 <= M <= 8");
  TORCH_CHECK(xn.size(1) == D && xn.is_contiguous() &&
              xn.scalar_type() == at::kHalf);
  check_weights(wd, wu);
  const at::cuda::OptionalCUDAGuard guard(device_of(xn));
  auto s = torch::empty({M, R}, xn.options());
  auto inj_out = torch::empty({M, HC}, xn.options());
  auto out = torch::empty({M, H}, xn.options());
  const auto* x = reinterpret_cast<const _Float16*>(xn.data_ptr());
  // h/block_out/inj/norm_w/out_h/xn are not touched on this path.
  torch::Tensor unused = xn;
  dispatch(M, xn, xn, xn, xn, wd, wu, unused, unused, s, inj_out, out, 0.f,
           at::cuda::getCurrentCUDAStream(), x);
  return {out, inj_out};
}
