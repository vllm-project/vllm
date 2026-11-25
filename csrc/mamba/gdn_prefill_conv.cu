// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// GDN prefill: causal conv1d + post-conv prep in one kernel, replacing
// causal_conv1d_fn + fused_post_conv_prep on pure-prefill batches. One warp per
// (sequence chunk, head), 4 channels per lane; H and HV are runtime values.
// Numerics: conv state bit-exact; q/k/v within ~2 bf16 ulp (fp32 conv,
// tanh.approx SiLU); g/beta within 1e-4 rel (__expf/__logf).
#include <cuda_bf16.h>

#include "../torch_utils.h"
#include <torch/csrc/stable/macros.h>

#define DEVINL __device__ __forceinline__

DEVINL float4 bf16x4_to_f32(uint2 v) {
  __nv_bfloat162 lo = *reinterpret_cast<__nv_bfloat162*>(&v.x);
  __nv_bfloat162 hi = *reinterpret_cast<__nv_bfloat162*>(&v.y);
  return {__bfloat162float(lo.x), __bfloat162float(lo.y),
          __bfloat162float(hi.x), __bfloat162float(hi.y)};
}

DEVINL uint2 f32_to_bf16x4(float4 f) {
  __nv_bfloat162 lo = __floats2bfloat162_rn(f.x, f.y);
  __nv_bfloat162 hi = __floats2bfloat162_rn(f.z, f.w);
  return {*reinterpret_cast<unsigned*>(&lo), *reinterpret_cast<unsigned*>(&hi)};
}

DEVINL float tanh_approx(float x) {
  float y;
  asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// bf16x2 pair -> fp32 pair without CVT: low element shifts up, high is masked.
DEVINL float2 bf16x2_to_f32x2(unsigned v) {
  return {__uint_as_float(v << 16), __uint_as_float(v & 0xFFFF0000u)};
}

// Packed FFMA2/FMUL2 for the conv taps and SiLU, sm_103a only: sm_100a/100f
// cubins spill in the hot loop (+20%, nvcc 13.1 and 13.4, cause unknown), and
// sm_110/120 lack the instructions.
#define GDN_HAS_F32X2 \
  (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1030) && (__CUDA_ARCH__ < 1100))

DEVINL float2 ffma2(float2 a, float2 b, float2 c) {
#if GDN_HAS_F32X2
  return __ffma2_rn(a, b, c);
#else
  return {fmaf(a.x, b.x, c.x), fmaf(a.y, b.y, c.y)};
#endif
}

DEVINL float2 fmul2(float2 a, float2 b) {
#if GDN_HAS_F32X2
  return __fmul2_rn(a, b);
#else
  return {a.x * b.x, a.y * b.y};
#endif
}

namespace gdn_conv {

struct Ptrs {
  const __nv_bfloat16* x;  // (tokens, dim), row stride arbitrary
  const __nv_bfloat16* w;  // (dim, 4) row-major
  __nv_bfloat16* state;    // (lines, dim, >=3), arbitrary strides
  const int* cache_indices;
  // the layer may hand a (seqs, S) view: read column 0 via the row stride
  long stride_cache_indices;
  const bool* has_init;
  const int* qsl;
  const __nv_bfloat16* a;  // (tokens, HV), row stride arbitrary
  const __nv_bfloat16* b;
  const float* A_log;
  const void* dt_bias;  // fp32 or bf16 (dt_bias follows the model dtype)
  bool dt_bias_bf16;
  __nv_bfloat16* q;
  __nv_bfloat16* k;
  __nv_bfloat16* v;
  float* g;
  float* beta;
  int num_seqs;
  int search_steps;
  int search_stride;
  long stride_x_token;
  long stride_state_seq;
  long stride_state_dim;
  long stride_state_token;
  long stride_a_token;
  long stride_b_token;
  int null_block_id;
  int H, HV;
  bool output_g_exp;
  float beta_scale;
};

// noinline: inlining the epilogues slows the main loop ~13% at 16K tokens.
__device__ __noinline__ void gbeta_epilogue(
    const __nv_bfloat16* a, const __nv_bfloat16* b, const float* A_log_p,
    const void* dt_bias_p, bool dt_bias_bf16, float* g, float* beta,
    long stride_a, long stride_b, int HV, bool output_g_exp, float beta_scale,
    int hv, int row0, int end, int seq_start, int lane) {
  const float A_log = A_log_p[hv];
  const float dt_bias =
      dt_bias_bf16 ? __bfloat162float(
                         reinterpret_cast<const __nv_bfloat16*>(dt_bias_p)[hv])
                   : static_cast<const float*>(dt_bias_p)[hv];
  for (int r = row0 + lane; r < end; r += 32) {
    long tok = seq_start + r;
    float av = __bfloat162float(a[tok * stride_a + hv]);
    float bv = __bfloat162float(b[tok * stride_b + hv]);
    float xg = av + dt_bias;
    float sp =
        xg > 0.f ? xg + __logf(1.f + __expf(-xg)) : __logf(1.f + __expf(xg));
    sp = xg <= 20.f ? sp : xg;
    float gv = -__expf(A_log) * sp;
    if (output_g_exp) gv = __expf(gv);
    g[tok * HV + hv] = gv;
    beta[tok * HV + hv] = beta_scale / (1.f + __expf(-bv));
  }
}

__device__ __noinline__ void state_epilogue(long stride_state_dim,
                                            long stride_state_token,
                                            __nv_bfloat16* state_head, int c0,
                                            int seqlen, bool load_init,
                                            const char* x_head, long xstep) {
  // Last 3 inputs, with old state shifted in for short sequences. All reads
  // precede the stores.
  auto hist_row = [&](int pos) -> float4 {
    if (pos >= 0)
      return bf16x4_to_f32(
          *reinterpret_cast<const uint2*>(x_head + (long)pos * xstep));
    float4 r = {0.f, 0.f, 0.f, 0.f};
    if (load_init) {
      float* e = &r.x;
#pragma unroll
      for (int i = 0; i < 4; i++)
        e[i] = __bfloat162float(state_head[(long)(c0 + i) * stride_state_dim +
                                           (3 + pos) * stride_state_token]);
    }
    return r;
  };
  float4 n0 = hist_row(seqlen - 3);
  float4 n1 = hist_row(seqlen - 2);
  float4 n2 = hist_row(seqlen - 1);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    __nv_bfloat16* base = state_head + (long)(c0 + i) * stride_state_dim;
    base[0] = __float2bfloat16_rn((&n0.x)[i]);
    base[stride_state_token] = __float2bfloat16_rn((&n1.x)[i]);
    base[2 * stride_state_token] = __float2bfloat16_rn((&n2.x)[i]);
  }
}

template <int CHUNK>
// Caps ptxas at 80 regs (uses 72), which measured best.
__global__ void __launch_bounds__(32, 25) gdn_prefill_conv_kernel(Ptrs p) {
  constexpr int D = 128;
  constexpr int PF = 8;  // rows prefetched ahead in the register queue
  const int lane = threadIdx.x & 31;
  // 1D grid, head fastest: adjacent CTAs share x rows; no grid.y 65535 cap.
  const int head = blockIdx.x % (2 * p.H + p.HV);
  const int pid = blockIdx.x / (2 * p.H + p.HV);

  // Radix-32 search: sequence s owns chunks [g(s), g(s+1)) with
  // g(s) = (qsl[s] - qsl[0]) / CHUNK + s; find last s with g(s) <= pid.
  const int qsl0 = p.qsl[0];
  int lo = 0;
  int stride = p.search_stride;
  for (int s = 0; s < p.search_steps; s++) {
    int idx = lo + lane * stride;
    bool in = idx < p.num_seqs;
    int g = in ? (p.qsl[idx] - qsl0) / CHUNK + idx : 0;
    unsigned ball = __ballot_sync(0xffffffffu, in && (g <= pid));
    lo += (__popc(ball) - 1) * stride;
    stride /= 32;
  }
  const int seq = lo;
  const int seq_start = p.qsl[seq];
  const int chunk = pid - ((seq_start - qsl0) / CHUNK + seq);
  const int seqlen = p.qsl[seq + 1] - seq_start;
  const int row0 = chunk * CHUNK;
  if (row0 >= seqlen) return;  // spare program (also: empty sequences)
  const int slot = p.cache_indices[(long)seq * p.stride_cache_indices];
  if (p.null_block_id >= 0 && slot == p.null_block_id) return;
  const bool load_init = p.has_init[seq];
  const int end = min(seqlen, row0 + CHUNK);

  const int c0 = head * D + lane * 4;  // first of this thread's 4 channels
  const __nv_bfloat16* x_head = p.x + (long)seq_start * p.stride_x_token + c0;
  __nv_bfloat16* out_head;
  int out_stride;
  if (head < p.H) {
    out_head = p.q + (long)seq_start * (p.H * D) + c0;
    out_stride = p.H * D;
  } else if (head < 2 * p.H) {
    out_head = p.k + (long)seq_start * (p.H * D) + c0 - p.H * D;
    out_stride = p.H * D;
  } else {
    out_head = p.v + (long)seq_start * (p.HV * D) + c0 - 2 * p.H * D;
    out_stride = p.HV * D;
  }
  const long xstep = p.stride_x_token * 2;
  const long ostep = (long)out_stride * 2;
  const char* x_base = reinterpret_cast<const char*>(x_head);
  const char* x_pf = x_base + (long)(row0 + PF) * xstep;
  const char* x_pf_end = x_base + (long)end * xstep;
  char* o_cur = reinterpret_cast<char*>(out_head) + (long)row0 * ostep;

  // 4 channels x 4 taps, as fp32 pairs.
  float2 wt[4][2];  // [tap][channel pair]
#pragma unroll
  for (int pr = 0; pr < 2; pr++) {
    const __nv_bfloat16* wrow = p.w + (long)(c0 + 2 * pr) * 4;
#pragma unroll
    for (int j = 0; j < 4; j++)
      wt[j][pr] = {__bfloat162float(wrow[j]), __bfloat162float(wrow[4 + j])};
  }

  // Input row `pos` of the sequence; before it, the conv state (or zero).
  __nv_bfloat16* state_head = p.state + (long)slot * p.stride_state_seq;
  auto hist_row = [&](int pos) -> float4 {
    if (pos >= 0)
      return bf16x4_to_f32(
          *reinterpret_cast<const uint2*>(x_base + (long)pos * xstep));
    float4 r = {0.f, 0.f, 0.f, 0.f};
    if (load_init) {
      float* e = &r.x;
#pragma unroll
      for (int i = 0; i < 4; i++)
        e[i] = __bfloat162float(state_head[(long)(c0 + i) * p.stride_state_dim +
                                           (3 + pos) * p.stride_state_token]);
    }
    return r;
  };

  // Channel pairs stay float2 throughout: no pairing moves in the loop.
  float2 h0[2], h1[2], h2[2];  // [channel pair]
  {
    float4 t0 = hist_row(row0 - 3);
    float4 t1 = hist_row(row0 - 2);
    float4 t2 = hist_row(row0 - 1);
    h0[0] = {t0.x, t0.y};
    h0[1] = {t0.z, t0.w};
    h1[0] = {t1.x, t1.y};
    h1[1] = {t1.z, t1.w};
    h2[0] = {t2.x, t2.y};
    h2[1] = {t2.z, t2.w};
  }

  uint2 pf[PF];
#pragma unroll
  for (int j = 0; j < PF; j++) {
    const char* addr = x_base + (long)(row0 + j) * xstep;
    pf[j] = (addr < x_pf_end) ? *reinterpret_cast<const uint2*>(addr)
                              : uint2{0u, 0u};
  }

#pragma unroll 8
  for (int r = row0; r < end; r++) {
    uint2 raw = pf[0];
#pragma unroll
    for (int j = 0; j < PF - 1; j++) pf[j] = pf[j + 1];
    pf[PF - 1] = (x_pf < x_pf_end) ? *reinterpret_cast<const uint2*>(x_pf)
                                   : uint2{0u, 0u};
    x_pf += xstep;
    float2 cur0 = bf16x2_to_f32x2(raw.x);
    float2 cur1 = bf16x2_to_f32x2(raw.y);
    float2 a0 = fmul2(cur0, wt[3][0]);
    float2 a1 = fmul2(cur1, wt[3][1]);
    a0 = ffma2(h2[0], wt[2][0], a0);
    a1 = ffma2(h2[1], wt[2][1], a1);
    a0 = ffma2(h1[0], wt[1][0], a0);
    a1 = ffma2(h1[1], wt[1][1], a1);
    a0 = ffma2(h0[0], wt[0][0], a0);
    a1 = ffma2(h0[1], wt[0][1], a1);
    // SiLU(x) = h + h * tanh(h), h = x / 2.
    float2 hf0 = fmul2(a0, {0.5f, 0.5f});
    float2 hf1 = fmul2(a1, {0.5f, 0.5f});
    float2 t0 = {tanh_approx(hf0.x), tanh_approx(hf0.y)};
    float2 t1 = {tanh_approx(hf1.x), tanh_approx(hf1.y)};
    a0 = ffma2(hf0, t0, hf0);
    a1 = ffma2(hf1, t1, hf1);
    float4 y32 = {a0.x, a0.y, a1.x, a1.y};
    if (head < 2 * p.H) {
      // L2-normalize q/k over the head.
      float s = y32.x * y32.x + y32.y * y32.y + y32.z * y32.z + y32.w * y32.w;
#pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        s += __shfl_xor_sync(0xffffffffu, s, off);
      float inv = rsqrtf(s + 1e-6f);
      y32.x *= inv;
      y32.y *= inv;
      y32.z *= inv;
      y32.w *= inv;
    }
    *reinterpret_cast<uint2*>(o_cur) = f32_to_bf16x4(y32);
    o_cur += ostep;
    h0[0] = h1[0];
    h0[1] = h1[1];
    h1[0] = h2[0];
    h1[1] = h2[1];
    h2[0] = cur0;
    h2[1] = cur1;
  }

  if (head >= 2 * p.H)
    gbeta_epilogue(p.a, p.b, p.A_log, p.dt_bias, p.dt_bias_bf16, p.g, p.beta,
                   p.stride_a_token, p.stride_b_token, p.HV, p.output_g_exp,
                   p.beta_scale, head - 2 * p.H, row0, end, seq_start, lane);
  if (chunk == 0)
    state_epilogue(p.stride_state_dim, p.stride_state_token, state_head, c0,
                   seqlen, load_init, x_base, xstep);
}

template <int CHUNK>
void launch(const Ptrs& p, int num_tokens, cudaStream_t stream) {
  int num_heads = 2 * p.H + p.HV;
  int chunks = (num_tokens + CHUNK - 1) / CHUNK + p.num_seqs;
  dim3 grid(num_heads * chunks);  // one warp per (chunk, head), head fastest
  gdn_prefill_conv_kernel<CHUNK><<<grid, 32, 0, stream>>>(p);
}

void launch_auto(const Ptrs& p, int num_tokens, cudaStream_t stream) {
  // Chunk sizes from a GB300 sweep.
  if (num_tokens <= 512)
    launch<16>(p, num_tokens, stream);
  else if (num_tokens <= 2048)
    launch<32>(p, num_tokens, stream);
  else
    launch<64>(p, num_tokens, stream);
}

}  // namespace gdn_conv

// Outputs are allocated by the Python wrapper.
void gdn_prefill_conv(
    const torch::stable::Tensor& x, const torch::stable::Tensor& weight,
    const torch::stable::Tensor& conv_states,
    const torch::stable::Tensor& query_start_loc,
    const torch::stable::Tensor& cache_indices,
    const torch::stable::Tensor& has_initial_state,
    const torch::stable::Tensor& a, const torch::stable::Tensor& b,
    const torch::stable::Tensor& A_log, const torch::stable::Tensor& dt_bias,
    const torch::stable::Tensor& q, const torch::stable::Tensor& k,
    const torch::stable::Tensor& v, const torch::stable::Tensor& g,
    const torch::stable::Tensor& beta, int64_t num_k_heads, int64_t head_dim,
    bool output_g_exp, double beta_scale, int64_t null_block_id) {
  STD_TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.stride(1) == 1);
  STD_TORCH_CHECK(x.scalar_type() == torch::headeronly::ScalarType::BFloat16);
  STD_TORCH_CHECK(head_dim == 128 && x.size(1) % 128 == 0);
  STD_TORCH_CHECK(weight.stride(1) == 1 && weight.stride(0) == 4 &&
                  weight.size(1) == 4 && weight.size(0) == x.size(1));
  STD_TORCH_CHECK(conv_states.dim() == 3 && conv_states.size(1) == x.size(1) &&
                  conv_states.size(2) >= 3);
  STD_TORCH_CHECK(A_log.scalar_type() == torch::headeronly::ScalarType::Float);
  STD_TORCH_CHECK(
      dt_bias.scalar_type() == torch::headeronly::ScalarType::BFloat16 ||
      dt_bias.scalar_type() == torch::headeronly::ScalarType::Float);
  STD_TORCH_CHECK((reinterpret_cast<uintptr_t>(x.data_ptr()) & 7) == 0);
  STD_TORCH_CHECK((x.stride(0) * 2) % 8 == 0);

  const int num_tokens = x.size(0);
  const int H = num_k_heads;
  const int HV = A_log.size(0);
  const int num_seqs = query_start_loc.size(0) - 1;

  gdn_conv::Ptrs p{};
  p.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  p.w = reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr());
  p.state = reinterpret_cast<__nv_bfloat16*>(conv_states.data_ptr());
  p.cache_indices = static_cast<const int*>(cache_indices.data_ptr());
  p.stride_cache_indices = cache_indices.stride(0);
  p.has_init = static_cast<const bool*>(has_initial_state.data_ptr());
  p.qsl = static_cast<const int*>(query_start_loc.data_ptr());
  p.a = reinterpret_cast<const __nv_bfloat16*>(a.data_ptr());
  p.b = reinterpret_cast<const __nv_bfloat16*>(b.data_ptr());
  p.A_log = static_cast<const float*>(A_log.data_ptr());
  p.dt_bias = dt_bias.data_ptr();
  p.dt_bias_bf16 =
      dt_bias.scalar_type() == torch::headeronly::ScalarType::BFloat16;
  p.q = reinterpret_cast<__nv_bfloat16*>(q.data_ptr());
  p.k = reinterpret_cast<__nv_bfloat16*>(k.data_ptr());
  p.v = reinterpret_cast<__nv_bfloat16*>(v.data_ptr());
  p.g = static_cast<float*>(g.data_ptr());
  p.beta = static_cast<float*>(beta.data_ptr());
  p.num_seqs = num_seqs;
  int search_steps = 0;
  int64_t pow32 = 1;
  while (pow32 < num_seqs) {
    pow32 *= 32;
    search_steps++;
  }
  p.search_steps = search_steps;
  p.search_stride = (int)(search_steps > 0 ? pow32 / 32 : 1);
  p.stride_x_token = x.stride(0);
  p.stride_state_seq = conv_states.stride(0);
  p.stride_state_dim = conv_states.stride(1);
  p.stride_state_token = conv_states.stride(2);
  p.stride_a_token = a.stride(0);
  p.stride_b_token = b.stride(0);
  p.null_block_id = (int)null_block_id;
  p.H = H;
  p.HV = HV;
  p.output_g_exp = output_g_exp;
  p.beta_scale = (float)beta_scale;

  gdn_conv::launch_auto(p, num_tokens, get_current_cuda_stream());
}
