// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// SM90 (sm_90a) KDA chunked prefill for GLM-5.3-Flash style KDA layers
// (head_dim 128, bounded gate), a drop-in for FlashKDA's fwd: the q|k|v
// sections of the packed conv output, raw gate and beta logits, fp32 state.
//
// Two kernels, chunks of 32 tokens (or of 64 when K2 holds at most a quarter
// of the SMs).
// K1 (prep): one CTA per (chunk, head): norms, gates, decayed q/k, kk/qk,
//   (I+L)^-1 -> an operand image in the workspace, laid out exactly as K2's
//   shared-memory stage.
// K2 (recurrence): one CTA per (sequence, head, half of the value rows); one
//   warpgroup keeps its 64 state rows as fp32 wgmma accumulators, warp 0 also
//   streams the images (one bulk copy per chunk) and V.
//
// G_i: log decay since the chunk start (inclusive), per key column;
//   E_i = exp(G_i - G_15) keeps |log| <= 80.
//   kk = tril_-1(<k_i E_i, k_j / E_j>), qk = scale tril(<q_i E_i, k_j / E_j>),
//   AI = (I + diag(b) kk)^-1 diag(b)
//   Kp = k exp(G), Qp = scale q exp(G), K_end = k exp(G_31 - G),
//   Gamma_C = exp(G_31)
// Recurrence per chunk (S0 = state at chunk start), Y = V - Kp S0^T:
//   U = AI Y,  O = Qp S0^T + qk U,  S = S0 o Gamma_C + U^T K_end
// 64-token chunks: K1 also forms W = AI^T K_end and (qk AI)^T, so that K2 needs
// no U:
//   O = Qp S0^T + (qk AI) Y,  S = S0 o Gamma_C + Y^T W
#include <cuda_bf16.h>

#include <algorithm>
#include <initializer_list>
#include <optional>
#include <type_traits>

#include "../torch_utils.h"
#include "kda_prefill_sm90_utils.cuh"

namespace vllm {
namespace kda_sm90 {
namespace {

constexpr int C = 32;  // tokens per chunk
constexpr float L2E = 1.4426950408889634f;

// ---- operand image (K1 output, K2 stage): A1 = [Kp; Qp] 64x128 K-major (2
// sub-tiles of 8 KB), KE = K_end 32x128 (2 sub-tiles of 4 KB, read MN-major),
// QA = 32 rows x 64 cols: cols 0..31 qk^T (third MN-major N-block after KE),
// cols 32..63 AI (K-major), GC = Gamma_C fp32 [128]
constexpr int ST_A = 0, ST_KE = 16384, ST_QA = 24576, ST_GC = 28672,
              IMG_BYTES = 29184;
// ---- K1 shared memory: input q, k, g as [32 x 64] SW128 sub-tiles + beta box
// [32][8]; image; scratch sub-tile s (columns 64 s .. 64 s + 63): k at 8192 s,
// q at 8192 s + 4096, g at 16384 + 4096 s, so that the mid-frame [Kp; Qp]
// written over k, q is one 64-row K-major operand per sub-tile and Km written
// over g is a 32-row one
constexpr int IN_K = 0, IN_Q = 4096, IN_G = 16384, IN_KQS = 8192, IN_GS = 4096;
// fp32 blocks of L and of its inverse reuse the input tiles once the
// elementwise pass has read them
constexpr int PS_LD = 0, PS_L10 = 2592, PS_EI = 3872, PS_TM = 6464;
constexpr int PS_RQ = 0, PS_RK = 128, PS_BS = 256, PS_T0 = 384, PS_T1 = 896,
              PS_BYTES = 1408;
// only the image's QA and GC parts live in shared memory (right after the
// inputs); Kp, Qp, K_end go to global directly
constexpr int K1_IN = 0, K1_IMG = 24576 - ST_QA, K1_PS = K1_IMG + IMG_BYTES,
              K1_BAR = K1_PS + PS_BYTES;
constexpr int K1_TM = (K1_BAR + 64 + 127) / 128 * 128,
              K1_SMEM = K1_TM + 384 + 1024;
// ---- K2 shared memory: stages (image + this CTA's V columns 32x64 SW128), out
// tiles [2] 32x64 SW128
constexpr int ST_V = 29696, ST_STRIDE = 33792;
constexpr int OFF_ST = 0;
constexpr int k2_smem(int nst, int ob) {
  return nst * ST_STRIDE + 8192 * ob + 128 + 1024;
}  // ob = 0: direct output stores

struct Params {
  // tensor maps of this call: [0] qkv as [T][3H][128] bf16, box {64, 1, 32},
  // 128B swizzle; [1], [2] out as [T][H][128] bf16, box {16, 1, 32} and {16, 1,
  // 128}, 32B swizzle; [3] g as [T][H][128], box {64, 1, 32}, 128B swizzle
  CUtensorMap tmpl[4];
  const __nv_bfloat16* qkv;
  const __nv_bfloat16* g;
  const __nv_bfloat16* beta;
  const float* A_log;
  const float* dt_bias;
  const float* initial_state;
  const int32_t* cu_seqlens;
  const int32_t* ckpt;
  __nv_bfloat16* out;
  float* final_state;
  float* ckpt_state;
  uint8_t* ws;  // operand images [chunk][H]
  int* meta;    // [N] first global chunk of each sequence, written by K1
  int* flags;   // overlapped mode (zeroed): [1 + slot * H + h] image ready (2
                // readers)
  int64_t qs, gs, bs, os;  // token strides in elements of qkv, g, beta, out
  int T, N, H;
  int lat_stride;  // overlapped 64-token mode: workspace slots per sequence
  int kord;        // K1 task order (see kda_prep)
  int early;  // K1 triggers K2 at its start (K1 has fewer tasks than there are
              // SMs)
  int sms;
  int c0;  // overlapped 64-token mode: first chunk of this prep grid (grid z +
           // c0)
  int trig;  // overlapped 64-token mode: this prep grid lets the recurrence
             // kernel launch at its start
  float scale, lb;
};

// byte offset of (row r, col c) in a [rows x 64] bf16 tile with 128B swizzle
__device__ __forceinline__ uint32_t sw(int r, int c) {
  return r * 128 + ((((c >> 3) ^ r) & 7) << 4) + ((c & 7) << 1);
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ float2 unpack_bf16(uint32_t u) {
  __nv_bfloat162 v = *reinterpret_cast<__nv_bfloat162*>(&u);
  return __bfloat1622float2(v);
}
template <typename T>
__device__ __forceinline__ T& at(uint8_t* base, uint32_t off) {
  return *reinterpret_cast<T*>(base + off);
}

__device__ __forceinline__ uint64_t kdesc(uint32_t addr) {
  return ptx::wgmma_desc_sw128(addr, 16, 1024);
}

// Chunks of a sequence: 32 tokens, except that a checkpoint offset at 16 mod 32
// ends a 16-token chunk.
struct Chunking {
  int len, shift, m;
  __device__ void init(int len_, int off) {
    len = len_;
    shift = (off > 0 && (off & 31) == 16) ? 16 : 0;
    m = off >> 5;
  }
  __device__ int count() const { return (len + shift + C - 1) / C; }
  __device__ int start(int c) const {
    return C * c - ((shift && c > m) ? 16 : 0);
  }
  __device__ int end(int c) const { return min(start(c + 1), len); }
};
template <class CH>
__device__ __forceinline__ int seq_chunks_t(const Params& p, int n) {
  CH ch;
  ch.init(p.cu_seqlens[n + 1] - p.cu_seqlens[n], p.ckpt[n]);
  return ch.count();
}
__device__ __forceinline__ int seq_chunks(const Params& p, int n) {
  return seq_chunks_t<Chunking>(p, n);
}
// Block-wide (NT threads) exclusive prefix of the per-sequence chunk counts.
// by_chunk: find the sequence n holding global chunk key; else key is n.
// Returns (n, first global chunk of n); n = N when key is past the last chunk.
template <int NT, class CH>
__device__ __forceinline__ int2 locate_t(const Params& p, int key,
                                         bool by_chunk, int* sh) {
  constexpr int NW = NT / 32;
  const int t = threadIdx.x, lane = t & 31, wp = t >> 5;
  if (t == 0) sh[NW] = p.N, sh[NW + 1] = 0;
  int carry = 0;
  for (int i0 = 0; i0 < p.N; i0 += NT) {
    const int i = i0 + t;
    int b0 = 0, b1 = 0, ck = 0, cnt = 0;
    if (i < p.N) {
      b0 = p.cu_seqlens[i], b1 = p.cu_seqlens[i + 1], ck = p.ckpt[i];
      CH chk;
      chk.init(b1 - b0, ck);
      cnt = chk.count();
    }
    int v = cnt;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const int u = __shfl_up_sync(0xffffffffu, v, d);
      if (lane >= d) v += u;
    }
    if (lane == 31) sh[wp] = v;
    __syncthreads();
    int ws = 0, all = 0;
    for (int k = 0; k < NW; ++k) {
      ws += k < wp ? sh[k] : 0;
      all += sh[k];
    }
    const int incl = carry + ws + v, excl = incl - cnt;
    if (i < p.N && (by_chunk ? (excl <= key && key < incl) : i == key))
      sh[NW] = i, sh[NW + 1] = excl, sh[NW + 2] = b0, sh[NW + 3] = b1,
      sh[NW + 4] = ck;  // bounds of the found sequence
    carry += all;
    __syncthreads();
  }
  return make_int2(sh[NW], sh[NW + 1]);
}
__device__ __forceinline__ int2 locate(const Params& p, int key, bool by_chunk,
                                       int* sh) {
  return locate_t<128, Chunking>(p, key, by_chunk, sh);
}

// Sequences of 1..TINY tokens take no image: K2's two CTAs run the recurrence
// token by token and K1's CTA exits. Warp w holds keys 4 lane .. 4 lane + 3 of
// state rows 64 half + 4 r + w, one row per instruction.
constexpr int TINY = 4;

__device__ __forceinline__ bool is_tiny(int len) {
  return len >= 1 && len <= TINY;
}
__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) x += __shfl_xor_sync(0xffffffffu, x, d);
  return x;
}
// x[r] per lane, r < 16 -> the warp's sum of x[r] for r = lane >> 1 (lanes 2 r
// and 2 r + 1 both hold it): each butterfly step sends half of the remaining
// values, 16 shuffles for the 16 sums
__device__ __forceinline__ float warp_sum16(const float (&x)[16]) {
  const int l = threadIdx.x & 31;
  float y[8], z[4], u[2];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const bool hi = l & 16;
    y[j] = (hi ? x[j + 8] : x[j]) +
           __shfl_xor_sync(0xffffffffu, hi ? x[j] : x[j + 8], 16);
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const bool hi = l & 8;
    z[j] = (hi ? y[j + 4] : y[j]) +
           __shfl_xor_sync(0xffffffffu, hi ? y[j] : y[j + 4], 8);
  }
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const bool hi = l & 4;
    u[j] = (hi ? z[j + 2] : z[j]) +
           __shfl_xor_sync(0xffffffffu, hi ? z[j] : z[j + 2], 4);
  }
  const bool hi = l & 2;
  float v =
      (hi ? u[1] : u[0]) + __shfl_xor_sync(0xffffffffu, hi ? u[0] : u[1], 2);
  return v + __shfl_xor_sync(0xffffffffu, v, 1);
}
__device__ __forceinline__ void tiny_direct(const Params& p, int n, int h,
                                            int half, int bos, int len) {
  const int t = threadIdx.x, w = t >> 5, l = t & 31, H = p.H;
  const float ea = expf(p.A_log[h]);
  const float4 db =
      *reinterpret_cast<const float4*>(p.dt_bias + h * 128 + 4 * l);
  const size_t sbase =
      ((size_t)n * H + h) * 16384 + (64 * half + w) * 128 + 4 * l;
  float4 S[16];  // all loads issued before the first use
#pragma unroll
  for (int r = 0; r < 16; ++r)
    S[r] = *reinterpret_cast<const float4*>(p.initial_state + sbase + r * 512);
  for (int i = 0; i < len; ++i) {
    const size_t tok = bos + i;
    const __nv_bfloat16* x = p.qkv + tok * p.qs + h * 128;
    const uint2 qw = *reinterpret_cast<const uint2*>(x + 4 * l),
                kw = *reinterpret_cast<const uint2*>(x + (size_t)H * 128 +
                                                     4 * l);
    const uint2 gw =
        *reinterpret_cast<const uint2*>(p.g + tok * p.gs + h * 128 + 4 * l);
    const float vmine =
        __bfloat162float(x[(size_t)2 * H * 128 + 64 * half + 4 * (l >> 1) +
                           w]);  // v of row r = lane >> 1
    const float bet =
        1.f / (1.f + __expf(-__bfloat162float(p.beta[tok * p.bs + h])));
    const float2 q0 = unpack_bf16(qw.x), q1 = unpack_bf16(qw.y),
                 k0 = unpack_bf16(kw.x), k1 = unpack_bf16(kw.y);
    const float2 g0 = unpack_bf16(gw.x), g1 = unpack_bf16(gw.y);
    const float qa[4] = {q0.x, q0.y, q1.x, q1.y},
                ka[4] = {k0.x, k0.y, k1.x, k1.y},
                ga[4] = {g0.x, g0.y, g1.x, g1.y};
    const float da[4] = {db.x, db.y, db.z, db.w};
    const float rq = rsqrtf(warp_sum(qa[0] * qa[0] + qa[1] * qa[1] +
                                     qa[2] * qa[2] + qa[3] * qa[3]) +
                            1e-6f) *
                     p.scale;
    const float rk = rsqrtf(warp_sum(ka[0] * ka[0] + ka[1] * ka[1] +
                                     ka[2] * ka[2] + ka[3] * ka[3]) +
                            1e-6f);
    float gmul[4], kh[4], qh[4], gk[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      gmul[e] = __expf(p.lb / (1.f + __expf(-ea * (ga[e] + da[e]))));
      kh[e] = ka[e] * rk;
      qh[e] = qa[e] * rq;
      gk[e] = gmul[e] * kh[e];
    }
    float xr[16];
#pragma unroll
    for (int r = 0; r < 16; ++r)
      xr[r] = fmaf(S[r].x, gk[0],
                   fmaf(S[r].y, gk[1], fmaf(S[r].z, gk[2], S[r].w * gk[3])));
    const float um = bet * (vmine - warp_sum16(xr));  // u of row lane >> 1
#pragma unroll
    for (int r = 0; r < 16; ++r) {
      const float4 sv = S[r];
      const float u = __shfl_sync(0xffffffffu, um, 2 * r);
      S[r] = make_float4(
          fmaf(sv.x, gmul[0], u * kh[0]), fmaf(sv.y, gmul[1], u * kh[1]),
          fmaf(sv.z, gmul[2], u * kh[2]), fmaf(sv.w, gmul[3], u * kh[3]));
      xr[r] = fmaf(S[r].x, qh[0],
                   fmaf(S[r].y, qh[1], fmaf(S[r].z, qh[2], S[r].w * qh[3])));
    }
    const float o = warp_sum16(xr);
    if (!(l & 1))
      p.out[tok * p.os + h * 128 + 64 * half + 4 * (l >> 1) + w] =
          __float2bfloat16_rn(o);
  }
#pragma unroll
  for (int r = 0; r < 16; ++r)
    *reinterpret_cast<float4*>(p.final_state + sbase + r * 512) = S[r];
}

// K1 CTA (0, 0): longest chunk count and number of one- to four-token sequences
// into meta[N], meta[N + 1], for K2's pacing of the latter (out of line: it
// would raise K1's register pressure)
__device__ __noinline__ void tiny_stats(const Params& p, int* red) {
  const int pt = threadIdx.x, pw = pt >> 5, pl = pt & 31;
  int mx = 0, nt = 0;
  for (int i = pt; i < p.N; i += 128) {
    const int len = p.cu_seqlens[i + 1] - p.cu_seqlens[i];
    Chunking q;
    q.init(len, p.ckpt[i]);
    mx = max(mx, q.count());
    nt += is_tiny(len) ? 1 : 0;
  }
#pragma unroll
  for (int d = 16; d > 0; d >>= 1)
    mx = max(mx, __shfl_xor_sync(0xffffffffu, mx, d)),
    nt += __shfl_xor_sync(0xffffffffu, nt, d);
  if (pl == 0) red[pw] = mx, red[4 + pw] = nt;
  __syncthreads();
  if (pt == 0) {
    p.meta[p.N] = max(max(red[0], red[1]), max(red[2], red[3]));
    p.meta[p.N + 1] = red[4] + red[5] + red[6] + red[7];
  }
  __syncthreads();
}

// ============================== K1: prep ==============================
// LAT (overlapped with K2, launched after it as a programmatic dependent): grid
// (H, N, chunks bound), so chunks complete in order across heads and sequences;
// each image is flagged ready for K2.
template <bool LAT>
__global__ void __launch_bounds__(128, LAT ? 6 : 5)
    kda_prep(const __grid_constant__ Params p) {
  if (LAT ? p.trig : p.early)
    ptx::griddep_launch_dependents();  // few tasks / early grid: K2's CTAs may
                                       // start
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* sm = smem_raw + ((1024 - (ptx::smem_u32(smem_raw) & 1023)) & 1023);
  const uint32_t sb = ptx::smem_u32(sm);
  const int pt = threadIdx.x, pw = pt >> 5, pl = pt & 31, H = p.H,
            h = blockIdx.x;
  int n, c, slot;
  Chunking ch;
  if (!LAT && blockIdx.x == 0 && blockIdx.y == 0)
    tiny_stats(p, reinterpret_cast<int*>(sm + K1_IN));
  if (LAT) {
    n = blockIdx.y;
    c = blockIdx.z + p.c0;
    if (c >= seq_chunks(p, n)) return;
    slot = locate(p, n, false, reinterpret_cast<int*>(sm + K1_BAR)).y + c;
  } else {
    // K2 reads the images written last from L2: kord 1 (K2 in one wave or
    // ranked): stripes of the sequence-major slots, each in descending order,
    // so that the first chunks of every sequence come last; kord 2 (many
    // sequences, K2 in sequence order N-1 first): sequence-major, chunks
    // descending within each sequence
    int key = blockIdx.y;
    if (p.kord == 1) {
      const int q = gridDim.y / p.N;
      key = (blockIdx.y % p.N) * q + (q - 1 - (int)blockIdx.y / p.N);
    }
    const int2 nb = locate(p, key, true, reinterpret_cast<int*>(sm + K1_BAR));
    n = nb.x;
    if (n >= p.N) return;
    c = key - nb.y;
    if (p.kord == 2) c = seq_chunks(p, n) - 1 - c;
    slot = nb.y + c;
  }
  const int* shb = reinterpret_cast<const int*>(sm + K1_BAR) +
                   6;  // bounds of n from locate: bos, eos, ckpt
  const int bos = shb[0];
  ch.init(shb[1] - bos, shb[2]);
  if (is_tiny(ch.len)) return;  // K2 runs this sequence without an image
  const int L = ch.end(c) - ch.start(c);
  if (!LAT && c == 0 && h == 0 && pt == 0) p.meta[n] = slot;
  uint8_t* inp = sm + K1_IN;
  uint8_t* stp = sm + K1_IMG;
  uint8_t* wsi = p.ws + ((size_t)slot * H + h) *
                            IMG_BYTES;  // this task's image in the workspace
  const uint32_t st = sb + K1_IMG;
  uint8_t* ps = sm + K1_PS;
  float* rqS = reinterpret_cast<float*>(ps + PS_RQ);
  float* rkS = reinterpret_cast<float*>(ps + PS_RK);
  float* bS = reinterpret_cast<float*>(ps + PS_BS);
  float* t0S = reinterpret_cast<float*>(ps + PS_T0);
  float* t1S = reinterpret_cast<float*>(ps + PS_T1);
  float* Ld = reinterpret_cast<float*>(
      sm + K1_IN + PS_LD);  // diagonal 16x16 blocks of L, [2][16][20] (+4 pad)
  float* L10 = reinterpret_cast<float*>(
      sm + K1_IN + PS_L10);  // L rows 16..31, cols 0..15, [16][20]
  float* Ei = reinterpret_cast<float*>(
      sm + K1_IN + PS_EI);  // inverses of the diagonal blocks, [2][16][20]
  float* Tm =
      reinterpret_cast<float*>(sm + K1_IN + PS_TM);  // L10 E0^-1, [16][20]
  const uint32_t ibar = sb + K1_BAR;
  if (pt == 0) {
    // q, k, g rows of this chunk into [32 x 64] 128B-swizzled sub-tiles, one
    // TMA box each; rows past T are zero
    ptx::mbarrier_init(ibar, 1);
    ptx::fence_mbarrier_init();
    ptx::mbarrier_arrive_expect_tx(ibar, 6 * 4096);
    const int t0 = bos + ch.start(c);
#pragma unroll
    for (int sub = 0; sub < 2; ++sub) {
      ptx::tma_load_3d(sb + K1_IN + IN_Q + sub * IN_KQS, &p.tmpl[0], 64 * sub,
                       h, t0, ibar);
      ptx::tma_load_3d(sb + K1_IN + IN_K + sub * IN_KQS, &p.tmpl[0], 64 * sub,
                       H + h, t0, ibar);
      ptx::tma_load_3d(sb + K1_IN + IN_G + sub * IN_GS, &p.tmpl[3], 64 * sub, h,
                       t0, ibar);
    }
  }
  // the upper-right 16x16 block of AI is zero
  {
    const int r = pt >> 3, j = 16 + 2 * (pt & 7);
    at<uint32_t>(stp, ST_QA + sw(r, 32 + j)) = 0u;
  }
  // beta of row pt >> 2 (used by lanes with pt & 3 == 1), loaded before the TMA
  // wait
  const float braw =
      ((pt & 3) == 1 && (pt >> 2) < L)
          ? __bfloat162float(
                p.beta[(size_t)(bos + ch.start(c) + (pt >> 2)) * p.bs + h])
          : 0.f;
  const float eA2 = 0.5f * expf(p.A_log[h]);
  const float lb2 =
      0.5f * p.lb * L2E;  // gate in log2 units: lb2 * (1 + tanh(z))
  // P2 mapping: 4 rows x 8 columns per thread
  const int hf = pw >> 1, csub = pw & 1, rqi = pl >> 3, coi = pl & 7;
  const int r0 = 16 * hf + 4 * rqi, col0 = 64 * csub + 8 * coi;
  float dA[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) dA[e] = eA2 * p.dt_bias[h * 128 + col0 + e];
  uint32_t offu[4];  // byte offset of (row r0 + u, chunk coi) in a
                     // 128B-swizzled sub-tile
#pragma unroll
  for (int u = 0; u < 4; ++u)
    offu[u] = (r0 + u) * 128 + (((coi ^ (r0 + u)) & 7) << 4);
  __syncthreads();
  ptx::mbarrier_wait_parity(ibar, 0);

  // ---- P1: row norms of q, k; beta ----
  {
    const int r = pt >> 2, qq = pt & 3;
    float sq = 0.f, sk = 0.f;
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int cc = (qq & 1) * 4 + u;
      const uint32_t o = (qq >> 1) * IN_KQS + r * 128 + (((cc ^ r) & 7) << 4);
      const uint4 qv = at<const uint4>(inp, IN_Q + o);
      const uint4 kv = at<const uint4>(inp, IN_K + o);
      const uint32_t qa[4] = {qv.x, qv.y, qv.z, qv.w},
                     ka[4] = {kv.x, kv.y, kv.z, kv.w};
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 a = unpack_bf16(qa[e]), b = unpack_bf16(ka[e]);
        sq = fmaf(a.x, a.x, fmaf(a.y, a.y, sq));
        sk = fmaf(b.x, b.x, fmaf(b.y, b.y, sk));
      }
    }
    sq += __shfl_xor_sync(0xffffffffu, sq, 1);
    sq += __shfl_xor_sync(0xffffffffu, sq, 2);
    sk += __shfl_xor_sync(0xffffffffu, sk, 1);
    sk += __shfl_xor_sync(0xffffffffu, sk, 2);
    if (qq == 0) {
      rqS[r] = rsqrtf(sq + 1e-6f) * p.scale;
      rkS[r] = r < L ? rsqrtf(sk + 1e-6f) : 0.f;
    } else if (qq == 1) {
      bS[r] = r < L ? 1.f / (1.f + __expf(-braw)) : 0.f;
    }
  }
  // ---- P2: gates, log2 decay relative to token 15, decayed q/k ----
  uint32_t e15p[4],
      e1p[4];  // bf16x2 exp(G_15), exp(G_31 - G_15) of this thread's columns
  {
    float xg[4][8];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const float lbv = r0 + u < L ? lb2 : 0.f;
      const uint4 gw = at<const uint4>(inp, IN_G + csub * IN_GS + offu[u]);
      const uint32_t w[4] = {gw.x, gw.y, gw.z, gw.w};
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const float glo = __uint_as_float(w[q] << 16),
                    ghi = __uint_as_float(w[q] & 0xffff0000u);
        xg[u][2 * q] =
            fmaf(lbv, ptx::tanh_approx(fmaf(eA2, glo, dA[2 * q])), lbv);
        xg[u][2 * q + 1] =
            fmaf(lbv, ptx::tanh_approx(fmaf(eA2, ghi, dA[2 * q + 1])), lbv);
      }
    }
    // inclusive prefix over the half's 16 rows: 4 rows per thread, 4 threads
    // (rqi) per column
    float tot[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      xg[1][e] += xg[0][e];
      xg[2][e] += xg[1][e];
      xg[3][e] += xg[2][e];
      const float t = xg[3][e];
      float v = t;
      const float s1 = __shfl_up_sync(0xffffffffu, v, 8);
      if (rqi >= 1) v += s1;
      const float s2 = __shfl_up_sync(0xffffffffu, v, 16);
      if (rqi >= 2) v += s2;
      const float ex = v - t;
      tot[e] = __shfl_sync(0xffffffffu, v, coi + 24);
      // x = G_r - G_15: half 0 subtracts its total, half 1 is already relative
      // to token 15
      const float sh = hf ? ex : ex - tot[e];
#pragma unroll
      for (int u = 0; u < 4; ++u) xg[u][e] += sh;
    }
    if (rqi == 0) {
      float* dst = (hf ? t1S : t0S) + col0;
      *reinterpret_cast<float4*>(dst) =
          make_float4(tot[0], tot[1], tot[2], tot[3]);
      *reinterpret_cast<float4*>(dst + 4) =
          make_float4(tot[4], tot[5], tot[6], tot[7]);
    }
    __syncthreads();
    float e15[8], e1[8];
    {
      const float4 a0 = *reinterpret_cast<const float4*>(t0S + col0),
                   a1 = *reinterpret_cast<const float4*>(t0S + col0 + 4);
      const float4 b0 = *reinterpret_cast<const float4*>(t1S + col0),
                   b1 = *reinterpret_cast<const float4*>(t1S + col0 + 4);
      const float g15[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
      const float d1[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        e15[e] = ptx::ex2_approx(g15[e]);
        e1[e] = ptx::ex2_approx(d1[e]);
      }
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        e15p[q] = pack_bf16(e15[2 * q], e15[2 * q + 1]);
        e1p[q] = pack_bf16(e1[2 * q], e1[2 * q + 1]);
      }
      if (hf == 0 && rqi == 0) {
        float gc[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) gc[e] = ptx::ex2_approx(g15[e] + d1[e]);
        at<float4>(stp, ST_GC + col0 * 4) =
            make_float4(gc[0], gc[1], gc[2], gc[3]);
        at<float4>(stp, ST_GC + col0 * 4 + 16) =
            make_float4(gc[4], gc[5], gc[6], gc[7]);
      }
    }
    const float4 rkv = *reinterpret_cast<const float4*>(rkS + r0),
                 rqv = *reinterpret_cast<const float4*>(rqS + r0);
    const float rka[4] = {rkv.x, rkv.y, rkv.z, rkv.w},
                rqa[4] = {rqv.x, rqv.y, rqv.z, rqv.w};
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const uint32_t okq = csub * IN_KQS + offu[u],
                     og = IN_G + csub * IN_GS + offu[u];
      const uint4 kw4 = at<const uint4>(inp, IN_K + okq);
      const uint4 qw4 = at<const uint4>(inp, IN_Q + okq);
      const uint32_t kw[4] = {kw4.x, kw4.y, kw4.z, kw4.w},
                     qw[4] = {qw4.x, qw4.y, qw4.z, qw4.w};
      uint32_t kpm[4], qpm[4], kmm[4], kps[4], qps[4], kes[4];
      const float rk = rka[u], rq = rqa[u];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int e = 2 * q;
        const float E0 = ptx::ex2_approx(xg[u][e]),
                    E1 = ptx::ex2_approx(xg[u][e + 1]);
        const float R0 = ptx::ex2_approx(-xg[u][e]),
                    R1 = ptx::ex2_approx(-xg[u][e + 1]);
        // mid frame for kk / qk and chunk-start frame for the image, both in
        // fp32 with one bf16 rounding
        const float2 kf = unpack_bf16(kw[q]), qf = unpack_bf16(qw[q]);
        const float k0 = kf.x * rk, k1 = kf.y * rk, q0 = qf.x * rq,
                    q1 = qf.y * rq;
        const float a0 = k0 * E0, a1 = k1 * E1, b0 = k0 * R0, b1 = k1 * R1,
                    c0 = q0 * E0, c1 = q1 * E1;
        kpm[q] = pack_bf16(a0, a1);
        kmm[q] = pack_bf16(b0, b1);
        qpm[q] = pack_bf16(c0, c1);
        kps[q] = pack_bf16(a0 * e15[e], a1 * e15[e + 1]);
        qps[q] = pack_bf16(c0 * e15[e], c1 * e15[e + 1]);
        kes[q] = pack_bf16(b0 * e1[e], b1 * e1[e + 1]);
      }
      const uint32_t o32 = csub * 4096 + offu[u], o64 = csub * 8192 + offu[u];
      at<uint4>(inp, IN_K + okq) = make_uint4(kpm[0], kpm[1], kpm[2], kpm[3]);
      at<uint4>(inp, IN_Q + okq) = make_uint4(qpm[0], qpm[1], qpm[2], qpm[3]);
      at<uint4>(inp, og) = make_uint4(kmm[0], kmm[1], kmm[2], kmm[3]);
      at<uint4>(wsi, ST_A + o64) = make_uint4(kps[0], kps[1], kps[2], kps[3]);
      at<uint4>(wsi, ST_A + o64 + 4096) =
          make_uint4(qps[0], qps[1], qps[2], qps[3]);
      at<uint4>(wsi, ST_KE + o32) = make_uint4(kes[0], kes[1], kes[2], kes[3]);
    }
  }
  ptx::fence_proxy_async_smem();
  __syncthreads();

  // ---- P3: [kk; qk] = [Kp; Qp] Km^T (mid frame, 64 x 32) ----
  {
    float acc[16];
    ptx::wgmma_fence();
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      const uint64_t ad =
          kdesc(sb + K1_IN + (ks >> 2) * IN_KQS + (ks & 3) * 32);
      const uint64_t bd =
          kdesc(sb + K1_IN + IN_G + (ks >> 2) * IN_GS + (ks & 3) * 32);
      ptx::wgmma_m64n32k16_bf16_ss(acc, ad, bd, ks > 0);
    }
    ptx::wgmma_commit();
    ptx::wgmma_wait<0>();
    __syncthreads();  // the fp32 blocks below overwrite the mid-frame operands
    const int g = pl >> 2, cq = pl & 3;
#pragma unroll
    for (int hr = 0; hr < 2; ++hr) {
      const int il = g + 8 * hr;  // row within the warp's 16
      if (pw < 2) {
        const float bi = bS[16 * pw + il];
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
          const int jl = 8 * (jj & 1) + 2 * cq;  // column within a 16-block
          const float v0 = bi * acc[4 * jj + 2 * hr],
                      v1 = bi * acc[4 * jj + 2 * hr + 1];
          if (jj < 2 && pw == 1) {
            *reinterpret_cast<float2*>(L10 + il * 20 + jl) =
                make_float2(v0, v1);
          } else if ((jj < 2) == (pw == 0)) {
            *reinterpret_cast<float2*>(Ld + pw * 324 + il * 20 + jl) =
                make_float2(jl < il ? v0 : 0.f, jl + 1 < il ? v1 : 0.f);
          }
        }
      }
    }
    if (pw >= 2) {
      // qk^T: element (row j, col i) = qk[i][j] (j <= i), read MN-major beside
      // K_end; transposed by stmatrix
      uint32_t op[8];
#pragma unroll
      for (int jj = 0; jj < 4; ++jj)
#pragma unroll
        for (int hr = 0; hr < 2; ++hr) {
          const int i = 16 * (pw - 2) + g + 8 * hr, j = 8 * jj + 2 * cq;
          op[2 * jj + hr] =
              pack_bf16(j <= i ? acc[4 * jj + 2 * hr] : 0.f,
                        j + 1 <= i ? acc[4 * jj + 2 * hr + 1] : 0.f);
        }
      const int mi = pl >> 3;
#pragma unroll
      for (int hf2 = 0; hf2 < 2; ++hf2) {
        const int trow = 8 * (2 * hf2 + (mi >> 1)) + (pl & 7),
                  i0 = 16 * (pw - 2) + 8 * (mi & 1);
        ptx::stmatrix_x4_trans(
            st + ST_QA + trow * 128 + ((((i0 >> 3) ^ trow) & 7) << 4),
            op[4 * hf2], op[4 * hf2 + 1], op[4 * hf2 + 2], op[4 * hf2 + 3]);
      }
    }
  }
  __syncthreads();

  // ---- P4: AI = (I + L)^-1 diag(b): 16x16 diagonal blocks by substitution,
  // then the off-diagonal block ----
  if (pw == 0) {
    const int b = pl >> 4, cc = pl & 15;
    const float* Lb = Ld + b * 324;
    float xv[16];
#pragma unroll
    for (int r = 0; r < 16; ++r) xv[r] = 0.f;
#pragma unroll
    for (int r = 0; r < 16; ++r) {
      float acc = 0.f;
#pragma unroll
      for (int m4 = 0; m4 < r; m4 += 4) {
        const float4 lv = *reinterpret_cast<const float4*>(Lb + r * 20 + m4);
        acc = fmaf(lv.x, xv[m4], acc);
        if (m4 + 1 < r) acc = fmaf(lv.y, xv[m4 + 1], acc);
        if (m4 + 2 < r) acc = fmaf(lv.z, xv[m4 + 2], acc);
        if (m4 + 3 < r) acc = fmaf(lv.w, xv[m4 + 3], acc);
      }
      xv[r] = r == cc ? 1.f : (r < cc ? 0.f : -acc);
    }
    float* E = Ei + b * 324;
    const float bj = bS[16 * b + cc];
#pragma unroll
    for (int r = 0; r < 16; ++r) {
      E[r * 20 + cc] = xv[r];
      at<__nv_bfloat16>(stp, ST_QA + sw(16 * b + r, 32 + 16 * b + cc)) =
          __float2bfloat16_rn(xv[r] * bj);
    }
  }
  __syncthreads();
  {
    const int r = pt >> 3, c0 = (pt & 7) * 2;
    float s0 = 0.f, s1 = 0.f;
#pragma unroll
    for (int m4 = 0; m4 < 16; m4 += 4) {
      const float4 lv = *reinterpret_cast<const float4*>(L10 + r * 20 + m4);
      const float lw[4] = {lv.x, lv.y, lv.z, lv.w};
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const float2 ev =
            *reinterpret_cast<const float2*>(Ei + (m4 + q) * 20 + c0);
        s0 = fmaf(lw[q], ev.x, s0);
        s1 = fmaf(lw[q], ev.y, s1);
      }
    }
    *reinterpret_cast<float2*>(Tm + r * 20 + c0) = make_float2(s0, s1);
    __syncthreads();
    float u0 = 0.f, u1 = 0.f;
#pragma unroll
    for (int m4 = 0; m4 < 16; m4 += 4) {
      const float4 ev =
          *reinterpret_cast<const float4*>(Ei + 324 + r * 20 + m4);
      const float ew[4] = {ev.x, ev.y, ev.z, ev.w};
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const float2 tv =
            *reinterpret_cast<const float2*>(Tm + (m4 + q) * 20 + c0);
        u0 = fmaf(ew[q], tv.x, u0);
        u1 = fmaf(ew[q], tv.y, u1);
      }
    }
    at<uint32_t>(stp, ST_QA + sw(16 + r, 32 + c0)) =
        pack_bf16(-u0 * bS[c0], -u1 * bS[c0 + 1]);
  }
  ptx::fence_proxy_async_smem();
  __syncthreads();
  if (pt == 0) {
    ptx::bulk_copy_s2g(wsi + ST_QA, st + ST_QA, IMG_BYTES - ST_QA);
    ptx::bulk_commit();
    if (LAT) {
      ptx::bulk_wait<0>();
      ptx::fence_proxy_async_global();
      ptx::fence_acq_rel_gpu();
      ptx::st_release_gpu(p.flags + 1 + slot * H + h, 2);
    } else {
      ptx::bulk_wait_read<0>();
    }
  }
  if (!LAT && !p.early)
    ptx::griddep_launch_dependents();  // K2's CTAs may start their prologue;
                                       // they wait for K1's completion
}

// ============================== K2: recurrence ==============================
// One CTA per (sequence, head, half of the value rows): warpgroup 0 keeps state
// rows [64 half, 64 half + 64) as fp32 wgmma accumulators. Warp 0 also loads,
// per chunk, the operand image (one bulk copy) and this half's V columns. Per
// chunk, on the critical path: G1 = Kp S0^T -> Y = V - G1 -> U = AI Y -> S +=
// U^T K_end. The output GEMMs O = Qp S0^T + qk U of chunk c are issued after
// its state update and drained in chunk c+1. OB: chunks per output store (one
// TMA store per warp for OB full chunks); DS: output stored straight from
// registers
template <int NST, int OB, bool LAT, bool DS>
__global__ void __launch_bounds__(128, 2)
    kda_recur(const __grid_constant__ Params p) {
  if (LAT)
    ptx::griddep_launch_dependents();  // K1 may start on the SMs this grid
                                       // leaves free
  constexpr int OFF_OUT = OFF_ST + NST * ST_STRIDE,
                OFF_BAR = OFF_OUT + (DS ? 0 : 8192 * OB);
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* sm = smem_raw + ((1024 - (ptx::smem_u32(smem_raw) & 1023)) & 1023);
  const uint32_t sb = ptx::smem_u32(sm);
  const int tid = threadIdx.x, H = p.H;
  const int h = blockIdx.x >> 1, half = blockIdx.x & 1;
  int n = p.N - 1 - blockIdx.y;
  if (!LAT && p.N <= 32) {
    // a few sequences: CTA row y takes the sequence of rank y by chunk count,
    // longest first (ties: higher index first), so that the longest chains
    // start in the first wave; stage 0 is scratch before the first load
    int* sh = reinterpret_cast<int*>(sm + OFF_ST);
    const int t = threadIdx.x;
    if (t < p.N) sh[t] = seq_chunks(p, t);
    __syncthreads();
    if (t < p.N) {
      const int mine = sh[t];
      int r = 0;
      for (int m = 0; m < p.N; ++m)
        r += (sh[m] > mine || (sh[m] == mine && m > t)) ? 1 : 0;
      if (r == (int)blockIdx.y) sh[32] = t;
    }
    __syncthreads();
    n = sh[32];
  }
  const int bos = p.cu_seqlens[n];
  const int off = p.ckpt[n];
  Chunking ch;
  ch.init(p.cu_seqlens[n + 1] - bos, off);
  if (is_tiny(ch.len)) {
    const bool pace =
        !LAT && !p.early &&
        4 * H <=
            p.sms;  // a long sequence's CTAs may leave half of the SMs free
    if (pace) ptx::griddep_wait();  // K1's counts in meta
    if (pace && 4 * H * (p.N - p.meta[p.N + 1]) <= p.sms) {
      // at most M such CTAs at a time, so that their state traffic is spread
      // over the longest chain (about 0.6 us per chunk) instead of slowing its
      // first chunks (about 4 us per CTA)
      const int mx = max(p.meta[p.N], 1), nt = p.meta[p.N + 1];
      const int M = max(16, (int)((14LL * H * nt + mx - 1) / mx));
      if (threadIdx.x == 0)
        for (;;) {
          if (ptx::ld_relaxed_gpu(p.flags) < M) {
            if (atomicAdd(p.flags, 1) < M) break;
            atomicSub(p.flags, 1);
          }
          __nanosleep(2000);
        }
      __syncthreads();
      tiny_direct(p, n, h, half, bos, ch.len);
      __syncthreads();
      if (threadIdx.x == 0) atomicSub(p.flags, 1);
    } else {
      tiny_direct(p, n, h, half, bos, ch.len);
    }
    return;
  }
  const int nch = ch.count();
  const int ck_chunk = off > 0 ? (ch.shift ? ch.m : (off >> 5) - 1) : -1;
  const uint32_t bar_full = sb + OFF_BAR;
  if (tid == 0) {
    for (int i = 0; i < NST; ++i) ptx::mbarrier_init(bar_full + 8 * i, 1);
    ptx::fence_mbarrier_init();
  }
  __syncthreads();

  if (!LAT)
    ptx::griddep_wait();  // launched as a programmatic dependent of K1: its
                          // images and meta from here on
  const int base =
      LAT ? locate(p, n, false, reinterpret_cast<int*>(sm + OFF_BAR + 64)).y
          : p.meta[n];  // first chunk of n
  // warp 0: loads of chunk c into stage c % NST
  int fpre = 0;  // overlapped mode: ready flag of the next chunk to load, read
                 // one load ahead
  auto load = [&](int c) {
    const int s = c % NST, l = tid & 31;
    const uint32_t st = sb + OFF_ST + s * ST_STRIDE, full = bar_full + 8 * s;
    if (LAT && l == 0) {
      int* f = p.flags + 1 + (base + c) * H + h;
      while (fpre == 0) fpre = ptx::ld_relaxed_gpu(f);
      ptx::fence_acquire_gpu();
      ptx::fence_proxy_async_global();
      atomicSub(f, 1);
      fpre = c + 1 < nch ? ptx::ld_relaxed_gpu(f + H) : 1;
    }
    if (l == 0) {
      ptx::mbarrier_arrive_expect_tx(full, IMG_BYTES + 4096);
      ptx::bulk_copy_g2s(st, p.ws + ((size_t)(base + c) * H + h) * IMG_BYTES,
                         IMG_BYTES, full);
      ptx::tma_load_3d(st + ST_V, &p.tmpl[0], 64 * half, 2 * H + h,
                       bos + ch.start(c), full);
    }
  };
  if (tid < 32)
    for (int c = 0; c < NST - 1 && c < nch; ++c) load(c);

  // ---- consumer warpgroup ----
  const int wt = tid, w = wt >> 5, l = wt & 31, g = l >> 2, cq = l & 3;
  const int vrow = 64 * half + 16 * w + g;  // rows vrow, vrow + 8
  float S[64];
  const size_t sbase = ((size_t)n * H + h) * 128 * 128;
  {
    const float* src = p.initial_state + sbase;
#pragma unroll
    for (int j = 0; j < 16; ++j)
#pragma unroll
      for (int hr = 0; hr < 2; ++hr) {
        const float2 v = *reinterpret_cast<const float2*>(
            src + (vrow + 8 * hr) * 128 + 8 * j + 2 * cq);
        S[4 * j + 2 * hr] = v.x;
        S[4 * j + 2 * hr + 1] = v.y;
      }
  }
  auto store_state = [&](float* dst) {
#pragma unroll
    for (int j = 0; j < 16; ++j)
#pragma unroll
      for (int hr = 0; hr < 2; ++hr)
        *reinterpret_cast<float2*>(dst + (vrow + 8 * hr) * 128 + 8 * j +
                                   2 * cq) =
            make_float2(S[4 * j + 2 * hr], S[4 * j + 2 * hr + 1]);
  };
  // output of chunk c from bf16 pairs in accumulator layout: movmatrix
  // transposes each 8x8 (value rows x tokens) block so that a lane holds two
  // adjacent value rows of one token, stored straight to global memory (no
  // shared memory)
  auto store_direct = [&](const uint32_t (&op)[8], int c) {
    const int t0 = ch.start(c), L = ch.end(c) - t0;
    __nv_bfloat16* dst = p.out + (size_t)(bos + t0 + (l >> 2)) * p.os +
                         h * 128 + 64 * half + 16 * w + 2 * (l & 3);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const uint32_t v = ptx::movmatrix_trans(op[e]);
      const int tt = 8 * (e >> 1) + (l >> 2);
      if (tt < L)
        *reinterpret_cast<uint32_t*>(dst + (size_t)8 * (e >> 1) * p.os +
                                     8 * (e & 1)) = v;
    }
  };

  // output of chunk c from bf16 pairs in accumulator layout: each warp
  // transposes its 16 value columns into its own [32 OB tokens][32 B] tile (32B
  // swizzle, two buffers); OB full chunks go out with one TMA store per warp,
  // other chunks copy their valid rows. One bulk group per OB chunks.
  auto store_tma = [&](const uint32_t (&op)[8], int c) {
    const int t0 = ch.start(c), L = ch.end(c) - t0, b = c / OB,
              slot = c - b * OB;
    const bool full =
        OB > 1 ? (b + 1) * OB <= nch &&
                     ch.end(b * OB + OB - 1) - ch.start(b * OB) == 32 * OB
               : L == 32;
    const uint32_t ot = sb + OFF_OUT + ((b & 1) * 4 + w) * (OB * 1024);
    if (slot == 0) {
      if (l == 0)
        ptx::bulk_wait_read<1>();  // the store of batch b - 2 from this buffer
                                   // has read it
      __syncwarp();
    }
    const int mi = l >> 3;
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      const int trow = 8 * (2 * hf + (mi >> 1)) + (l & 7);
      ptx::stmatrix_x4_trans(
          ot + (32 * slot + trow) * 32 + ((((mi & 1) ^ (trow >> 2)) & 1) << 4),
          op[4 * hf], op[4 * hf + 1], op[4 * hf + 2], op[4 * hf + 3]);
    }
    if (full) {
      if (slot == OB - 1) {
        ptx::fence_proxy_async_smem();
        __syncwarp();
        if (l == 0) {
          ptx::tma_store_3d(&p.tmpl[OB > 1 ? 2 : 1], 64 * half + 16 * w, h,
                            bos + ch.start(b * OB), ot);
          ptx::bulk_commit();
        }
      }
    } else {
      __syncwarp();
      const uint8_t* otp = sm + (ot - sb) + 1024 * slot;
#pragma unroll
      for (int pass = 0; pass < 2; ++pass) {
        const int t = (l >> 1) + 16 * pass, j = l & 1;
        if (t < L) {
          const uint4 v = *reinterpret_cast<const uint4*>(
              otp + t * 32 + (((j ^ (t >> 2)) & 1) << 4));
          *reinterpret_cast<uint4*>(p.out + (size_t)(bos + t0 + t) * p.os +
                                    h * 128 + 64 * half + 16 * w + 8 * j) = v;
        }
      }
      if ((slot == OB - 1 || c == nch - 1) && l == 0) ptx::bulk_commit();
    }
  };

  auto store_out = [&](const uint32_t (&op)[8], int c) {
    if (DS)
      store_direct(op, c);
    else
      store_tma(op, c);
  };

  uint32_t opk[8];  // previous chunk's output, bf16 pairs in accumulator layout
#pragma unroll 1
  for (int c = 0; c < nch; ++c) {
    const int s = c % NST;
    const uint32_t st = sb + OFF_ST + s * ST_STRIDE;
    uint8_t* stp = sm + OFF_ST + s * ST_STRIDE;
    //@PB(8)
    ptx::mbarrier_wait_parity(bar_full + 8 * s, (c / NST) & 1);
    //@PE(8)
    // B fragments of AI (QA columns 32..63, rows = output tokens) for the
    // mma.sync U = AI Y: [n-tile][k 0..31 by 8]
    //@PB(9)
    uint32_t aS[8][4];
#pragma unroll
    for (int ks = 0; ks < 8; ++ks)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        aS[ks][e] = pack_bf16(S[8 * ks + 2 * e], S[8 * ks + 2 * e + 1]);
        ptx::reg_fence_u32(aS[ks][e]);
      }
    // acc1 = [Kp S0^T | Qp S0^T]^T
    float acc1[32];
    ptx::wgmma_fence();
    ptx::wgmma_m64n64k16_bf16_rs_z(acc1, aS[0], kdesc(st + ST_A));
#pragma unroll
    for (int ks = 1; ks < 8; ++ks)
      ptx::wgmma_m64n64k16_bf16_rs(
          acc1, aS[ks], kdesc(st + ST_A + (ks >> 2) * 8192 + (ks & 3) * 32), 1);
    ptx::wgmma_commit();
    if (w == 0 && c + NST - 1 < nch)
      load(c + NST - 1);  // into the stage of chunk c-1, no longer read
    uint32_t bAI[4][4];
#pragma unroll
    for (int t = 0; t < 4; ++t)
      ptx::ldmatrix_x4(bAI[t],
                       st + ST_QA + sw(8 * t + (l & 7), 32 + 8 * (l >> 3)));
    uint32_t vt[4][2];
    {
      const int mi = l >> 3, rr = l & 7;
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        const int t = 8 * (2 * hf + (mi >> 1)) + rr, v = 16 * w + 8 * (mi & 1);
        uint32_t r[4];
        ptx::ldmatrix_x4_trans(r, st + ST_V + sw(t, v));
        vt[2 * hf][0] = r[0];
        vt[2 * hf][1] = r[1];
        vt[2 * hf + 1][0] = r[2];
        vt[2 * hf + 1][1] = r[3];
      }
    }
    float2 gcf[16];
#pragma unroll
    for (int j = 0; j < 16; ++j)
      gcf[j] = at<const float2>(stp, ST_GC + (8 * j + 2 * cq) * 4);
    ptx::wgmma_wait<0>();
#pragma unroll
    for (int e = 0; e < 32; ++e) ptx::reg_fence(acc1[e]);
    // Y^T = V^T - (Kp S0^T)^T, then U^T = Y^T AI^T
    uint32_t aY[2][4];
#pragma unroll
    for (int jj = 0; jj < 4; ++jj)
#pragma unroll
      for (int hr = 0; hr < 2; ++hr) {
        const float2 vv = unpack_bf16(vt[jj][hr]);
        aY[jj >> 1][2 * (jj & 1) + hr] = pack_bf16(
            vv.x - acc1[4 * jj + 2 * hr], vv.y - acc1[4 * jj + 2 * hr + 1]);
        ptx::reg_fence_u32(aY[jj >> 1][2 * (jj & 1) + hr]);
      }
    // U^T = Y^T AI^T by mma.sync per warp (no wgmma in flight, so S0 o Gamma_C
    // runs beside it); S += U^T K_end; O2 = (qk U)^T, qk^T read MN-major
    float acc2[16], accU[16];
#pragma unroll
    for (int e = 0; e < 16; ++e) accU[e] = 0.f;
#pragma unroll
    for (int t = 0; t < 4; ++t)
#pragma unroll
      for (int ks = 0; ks < 2; ++ks)
        ptx::mma_m16n8k16_bf16(*reinterpret_cast<float (*)[4]>(accU + 4 * t),
                               aY[ks], bAI[t][2 * ks], bAI[t][2 * ks + 1]);
#pragma unroll
    for (int j = 0; j < 16; ++j) {  // S0 o Gamma_C
      S[4 * j] *= gcf[j].x;
      S[4 * j + 1] *= gcf[j].y;
      S[4 * j + 2] *= gcf[j].x;
      S[4 * j + 3] *= gcf[j].y;
    }
    uint32_t aU[2][4];
#pragma unroll
    for (int ks = 0; ks < 2; ++ks)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        aU[ks][e] = pack_bf16(accU[8 * ks + 2 * e], accU[8 * ks + 2 * e + 1]);
        ptx::reg_fence_u32(aU[ks][e]);
      }
    ptx::wgmma_fence();
    ptx::wgmma_m64n128k16_bf16_rs<1>(
        S, aU[0], ptx::wgmma_desc_sw128(st + ST_KE, 4096, 1024), 1);
    ptx::wgmma_m64n128k16_bf16_rs<1>(
        S, aU[1], ptx::wgmma_desc_sw128(st + ST_KE + 2048, 4096, 1024), 1);
    ptx::wgmma_m64n32k16_bf16_rs_z<1>(
        acc2, aU[0], ptx::wgmma_desc_sw128(st + ST_QA, 4096, 1024));
    ptx::wgmma_m64n32k16_bf16_rs<1>(
        acc2, aU[1], ptx::wgmma_desc_sw128(st + ST_QA + 2048, 4096, 1024), 1);
    ptx::wgmma_commit();
    if (c > 0) store_out(opk, c - 1);
    ptx::wgmma_wait<0>();
#pragma unroll
    for (int e = 0; e < 8; ++e)
      opk[e] = pack_bf16(acc1[16 + 2 * e] + acc2[2 * e],
                         acc1[16 + 2 * e + 1] + acc2[2 * e + 1]);
    //@PE(9)
    if (c == ck_chunk) store_state(p.ckpt_state + sbase);
  }
  if (nch > 0) store_out(opk, nch - 1);

  store_state(p.final_state + sbase);
  if (l == 0) ptx::bulk_wait_read<0>();
}

// ============================== K2 with 64-token chunks
// ============================== image: A = [Kp; Qp] 128 rows x 128 cols
// K-major (2 sub-tiles of 16 KB), W = AI^T K_end 64 tokens x 128 keys read
// MN-major (2 N-blocks of 8 KB), QA = (qk AI)^T 64 x 64 read MN-major (8 KB),
// GC = Gamma_C fp32 [128]
constexpr int C64 = 64;
constexpr int S64_A = 0, S64_W = 32768, S64_QA = 49152, S64_GC = 57344,
              IMG64 = 57856;
constexpr int S64_V = 58368, S64_STRIDE = 66560;
constexpr int k64_smem(int nst) {
  return nst * S64_STRIDE + 16384 + 128 + 1024;
}

struct Chunking64 {
  int len, shift, m;
  __device__ void init(int len_, int off) {
    len = len_;
    shift = (off > 0 && (off & 63)) ? (off & 63) : 0;
    m = off >> 6;
  }
  __device__ int count() const {
    return (len + (shift ? C64 - shift : 0) + C64 - 1) / C64;
  }
  __device__ int start(int c) const {
    return C64 * c - ((shift && c > m) ? C64 - shift : 0);
  }
  __device__ int end(int c) const { return min(start(c + 1), len); }
};

template <class P>
__device__ __forceinline__ int2 locate64(const P& p, int key, bool by_chunk,
                                         int* sh) {
  return locate_t<256, Chunking64>(p, key, by_chunk, sh);
}

// ============================== K1 with 64-token chunks
// ============================== One CTA of 256 threads per (chunk of 64
// tokens, head); warpgroup b runs the 32-token sub-chunk b as K1 does (mid
// frame relative to token 32 b + 15), then the two halves are joined:
//   kk10 = Kp1' k_end0^T, qk10 = Qp1' k_end0^T with Kp1' = k exp(G_i - G_31) (i
//   in sub 1), k_end0 = k exp(G_31 - G_j) AI = [[AI0, 0], [-AI1 kk10 AI0,
//   AI1]]; W^T = K_end^T AI (K_end from the mid-frame Km and per-key factors)
//   (qk AI)^T = AI^T qk^T
// smem: per sub b at 24576 b: k, q, g as in K1 (mid frames written over them);
// KX: Kp1'/Qp1' (16 KB), k_end0 (8 KB), later the inverse scratch and then the
// staged image tail W | (qk AI)^T, followed by GC
constexpr int P64_KX = 49152, P64_PS = P64_KX + 24576;
constexpr int P64_RQ = 0, P64_RK = 256, P64_BS = 512, P64_TS = 768,
              P64_BAR = 768 + 2048;
constexpr int P64_SMEM = P64_PS + P64_BAR + 128;
// dead k/q tiles after the GEMMs: bf16 qk [i][m] and AI [i][j] 64x64 tiles,
// fp32 AI0, AI1, kk10 [32][36]
constexpr int P64_QK = 0, P64_AI = 8192, P64_AIF = 24576,
              P64_KKF = 24576 + 2 * 4608;
constexpr int F36 = 36;

template <bool LAT>
__global__ void __launch_bounds__(256, 2)
    kda_prep64(const __grid_constant__ Params p) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* sm = smem_raw + ((1024 - (ptx::smem_u32(smem_raw) & 1023)) & 1023);
  const uint32_t sb = ptx::smem_u32(sm);
  const int tid = threadIdx.x, b = tid >> 7, pt = tid & 127, pw = pt >> 5,
            pl = pt & 31, H = p.H, h = blockIdx.x;
  int n, c, slot;
  Chunking64 ch;
  int* shl = reinterpret_cast<int*>(sm + P64_PS + P64_BAR + 16);
  if (LAT) {
    if (p.trig) ptx::griddep_launch_dependents();
    n = blockIdx.y;
    c = blockIdx.z + p.c0;
    ch.init(p.cu_seqlens[n + 1] - p.cu_seqlens[n], p.ckpt[n]);
    if (c >= ch.count()) return;
    slot = n * p.lat_stride + c;
  } else {
    const int2 nb = locate64(p, blockIdx.y, true, shl);
    n = nb.x;
    if (n >= p.N) return;
    c = blockIdx.y - nb.y;
    slot = blockIdx.y;
  }
  const int bos = p.cu_seqlens[n];
  ch.init(p.cu_seqlens[n + 1] - bos, p.ckpt[n]);
  const int L = ch.end(c) - ch.start(c);       // valid tokens of the chunk
  const int Lb = min(max(L - 32 * b, 0), 32);  // of this sub-chunk
  if (!LAT && c == 0 && h == 0 && tid == 0) p.meta[n] = slot;
  uint8_t* inp = sm + 24576 * b;  // this sub-chunk's input tiles
  const uint32_t ib = sb + 24576 * b;
  uint8_t* wsi = p.ws + ((size_t)slot * H + h) * IMG64;
  uint8_t* ps = sm + P64_PS;
  float* rqS = reinterpret_cast<float*>(ps + P64_RQ) + 32 * b;
  float* rkS = reinterpret_cast<float*>(ps + P64_RK) + 32 * b;
  float* bS = reinterpret_cast<float*>(ps + P64_BS) + 32 * b;
  float* tS = reinterpret_cast<float*>(
      ps + P64_TS);  // [sub][half][128] totals of the gate over 16 rows (log2)
  const uint32_t ibar = sb + P64_PS + P64_BAR;
  if (tid == 0) {
    ptx::mbarrier_init(ibar, 1);
    ptx::fence_mbarrier_init();
    ptx::mbarrier_arrive_expect_tx(ibar, 12 * 4096);
    const int t0 = bos + ch.start(c);
#pragma unroll
    for (int bb = 0; bb < 2; ++bb)
#pragma unroll
      for (int sub = 0; sub < 2; ++sub) {
        const uint32_t d = sb + 24576 * bb;
        ptx::tma_load_3d(d + IN_Q + sub * IN_KQS, &p.tmpl[0], 64 * sub, h,
                         t0 + 32 * bb, ibar);
        ptx::tma_load_3d(d + IN_K + sub * IN_KQS, &p.tmpl[0], 64 * sub, H + h,
                         t0 + 32 * bb, ibar);
        ptx::tma_load_3d(d + IN_G + sub * IN_GS, &p.tmpl[3], 64 * sub, h,
                         t0 + 32 * bb, ibar);
      }
  }
  const unsigned short braw =
      ((pt & 3) == 1 && (pt >> 2) < Lb)
          ? __ldg(reinterpret_cast<const unsigned short*>(p.beta) +
                  (size_t)(bos + ch.start(c) + 32 * b + (pt >> 2)) * p.bs + h)
          : 0;
  const float lb2 = 0.5f * p.lb * L2E;
  const int hf = pw >> 1, csub = pw & 1, rqi = pl >> 3, coi = pl & 7;
  const int r0 = 16 * hf + 4 * rqi, col0 = 64 * csub + 8 * coi;
  // A_log and dt_bias are loaded here and used only after the input wait, so
  // that no thread waits for them first
  const float a_log = __ldg(p.A_log + h);
  const float4 db0 =
      __ldg(reinterpret_cast<const float4*>(p.dt_bias + h * 128 + col0));
  const float4 db1 =
      __ldg(reinterpret_cast<const float4*>(p.dt_bias + h * 128 + col0 + 4));
  uint32_t offu[4];
#pragma unroll
  for (int u = 0; u < 4; ++u)
    offu[u] = (r0 + u) * 128 + (((coi ^ (r0 + u)) & 7) << 4);
  __syncthreads();
  ptx::mbarrier_wait_parity(ibar, 0);
  const float eA2 = 0.5f * expf(a_log);
  float dA[8] = {eA2 * db0.x, eA2 * db0.y, eA2 * db0.z, eA2 * db0.w,
                 eA2 * db1.x, eA2 * db1.y, eA2 * db1.z, eA2 * db1.w};

  // ---- P1: row norms of q, k; beta ----
  {
    const int r = pt >> 2, qq = pt & 3;
    float sq = 0.f, sk = 0.f;
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int cc = (qq & 1) * 4 + u;
      const uint32_t o = (qq >> 1) * IN_KQS + r * 128 + (((cc ^ r) & 7) << 4);
      const uint4 qv = at<const uint4>(inp, IN_Q + o);
      const uint4 kv = at<const uint4>(inp, IN_K + o);
      const uint32_t qa[4] = {qv.x, qv.y, qv.z, qv.w},
                     ka[4] = {kv.x, kv.y, kv.z, kv.w};
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 a = unpack_bf16(qa[e]), bb = unpack_bf16(ka[e]);
        sq = fmaf(a.x, a.x, fmaf(a.y, a.y, sq));
        sk = fmaf(bb.x, bb.x, fmaf(bb.y, bb.y, sk));
      }
    }
    sq += __shfl_xor_sync(0xffffffffu, sq, 1);
    sq += __shfl_xor_sync(0xffffffffu, sq, 2);
    sk += __shfl_xor_sync(0xffffffffu, sk, 1);
    sk += __shfl_xor_sync(0xffffffffu, sk, 2);
    if (qq == 0) {
      rqS[r] = rsqrtf(sq + 1e-6f) * p.scale;
      rkS[r] = r < Lb ? rsqrtf(sk + 1e-6f) : 0.f;
    } else if (qq == 1) {
      bS[r] = r < Lb
                  ? 1.f / (1.f + __expf(-__uint_as_float((uint32_t)braw << 16)))
                  : 0.f;
    }
  }
  // ---- P2: gates, log2 decay relative to token 32 b + 15, decayed q/k ----
  {
    float xg[4][8];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const float lbv = r0 + u < Lb ? lb2 : 0.f;
      const uint4 gw = at<const uint4>(inp, IN_G + csub * IN_GS + offu[u]);
      const uint32_t wv[4] = {gw.x, gw.y, gw.z, gw.w};
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const float glo = __uint_as_float(wv[q] << 16),
                    ghi = __uint_as_float(wv[q] & 0xffff0000u);
        xg[u][2 * q] =
            fmaf(lbv, ptx::tanh_approx(fmaf(eA2, glo, dA[2 * q])), lbv);
        xg[u][2 * q + 1] =
            fmaf(lbv, ptx::tanh_approx(fmaf(eA2, ghi, dA[2 * q + 1])), lbv);
      }
    }
    float tot[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      xg[1][e] += xg[0][e];
      xg[2][e] += xg[1][e];
      xg[3][e] += xg[2][e];
      const float t = xg[3][e];
      float v = t;
      const float s1 = __shfl_up_sync(0xffffffffu, v, 8);
      if (rqi >= 1) v += s1;
      const float s2 = __shfl_up_sync(0xffffffffu, v, 16);
      if (rqi >= 2) v += s2;
      const float ex = v - t;
      tot[e] = __shfl_sync(0xffffffffu, v, coi + 24);
      const float sh = hf ? ex : ex - tot[e];
#pragma unroll
      for (int u = 0; u < 4; ++u) xg[u][e] += sh;
    }
    if (rqi == 0) {
      float* dst = tS + (2 * b + hf) * 128 + col0;
      *reinterpret_cast<float4*>(dst) =
          make_float4(tot[0], tot[1], tot[2], tot[3]);
      *reinterpret_cast<float4*>(dst + 4) =
          make_float4(tot[4], tot[5], tot[6], tot[7]);
    }
    __syncthreads();
    // e15: exp(G_{32b+15} - G_{32b-1}); e1: exp(G_{32b+31} - G_{32b+15}); sub 1
    // also scales its image rows by exp(G_31)
    float e15[8], e1[8], eG[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const float a0 = tS[(2 * b) * 128 + col0 + e],
                  a1 = tS[(2 * b + 1) * 128 + col0 + e];
      e15[e] = ptx::ex2_approx(a0);
      e1[e] = ptx::ex2_approx(a1);
      eG[e] = b ? ptx::ex2_approx(tS[col0 + e] + tS[128 + col0 + e]) : 1.f;
      if (b == 1 && hf == 0 && rqi == 0)
        at<float>(wsi, S64_GC + (col0 + e) * 4) =
            ptx::ex2_approx(tS[col0 + e] + tS[128 + col0 + e] + a0 + a1);
    }
    const float4 rkv = *reinterpret_cast<const float4*>(rkS + r0),
                 rqv = *reinterpret_cast<const float4*>(rqS + r0);
    const float rka[4] = {rkv.x, rkv.y, rkv.z, rkv.w},
                rqa[4] = {rqv.x, rqv.y, rqv.z, rqv.w};
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const uint32_t okq = csub * IN_KQS + offu[u],
                     og = IN_G + csub * IN_GS + offu[u];
      const uint4 kw4 = at<const uint4>(inp, IN_K + okq);
      const uint4 qw4 = at<const uint4>(inp, IN_Q + okq);
      const uint32_t kw[4] = {kw4.x, kw4.y, kw4.z, kw4.w},
                     qw[4] = {qw4.x, qw4.y, qw4.z, qw4.w};
      uint32_t kpm[4], qpm[4], kmm[4], kps[4], qps[4], kx[4], qx[4];
      const float rk = rka[u], rq = rqa[u];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int e = 2 * q;
        const float E0 = ptx::ex2_approx(xg[u][e]),
                    E1 = ptx::ex2_approx(xg[u][e + 1]);
        const float R0 = ptx::ex2_approx(-xg[u][e]),
                    R1 = ptx::ex2_approx(-xg[u][e + 1]);
        const float2 kf = unpack_bf16(kw[q]), qf = unpack_bf16(qw[q]);
        const float k0 = kf.x * rk, k1 = kf.y * rk, q0 = qf.x * rq,
                    q1 = qf.y * rq;
        const float a0 = k0 * E0, a1 = k1 * E1, b0 = k0 * R0, b1 = k1 * R1,
                    c0 = q0 * E0, c1 = q1 * E1;
        kpm[q] = pack_bf16(a0, a1);
        kmm[q] = pack_bf16(b0, b1);
        qpm[q] = pack_bf16(c0, c1);
        if (b == 0) {
          kps[q] = pack_bf16(a0 * e15[e], a1 * e15[e + 1]);
          qps[q] = pack_bf16(c0 * e15[e], c1 * e15[e + 1]);
          kx[q] = pack_bf16(b0 * e1[e],
                            b1 * e1[e + 1]);  // k_end0 = k exp(G_31 - G_j)
        } else {
          const float x0 = a0 * e15[e], x1 = a1 * e15[e + 1], y0 = c0 * e15[e],
                      y1 = c1 * e15[e + 1];
          kx[q] = pack_bf16(x0, x1);  // Kp1' = k exp(G_i - G_31)
          qx[q] = pack_bf16(y0, y1);
          kps[q] = pack_bf16(x0 * eG[e], x1 * eG[e + 1]);
          qps[q] = pack_bf16(y0 * eG[e], y1 * eG[e + 1]);
        }
      }
      at<uint4>(inp, IN_K + okq) = make_uint4(kpm[0], kpm[1], kpm[2], kpm[3]);
      at<uint4>(inp, IN_Q + okq) = make_uint4(qpm[0], qpm[1], qpm[2], qpm[3]);
      at<uint4>(inp, og) = make_uint4(kmm[0], kmm[1], kmm[2], kmm[3]);
      // image A: key sub-tile csub at 16384 csub, rows 0..63 Kp, 64..127 Qp
      const uint32_t oa = csub * 16384 + 32 * b * 128 + offu[u];
      at<uint4>(wsi, S64_A + oa) = make_uint4(kps[0], kps[1], kps[2], kps[3]);
      at<uint4>(wsi, S64_A + oa + 8192) =
          make_uint4(qps[0], qps[1], qps[2], qps[3]);
      if (b == 0) {
        at<uint4>(sm, P64_KX + 16384 + csub * 4096 + offu[u]) =
            make_uint4(kx[0], kx[1], kx[2], kx[3]);
      } else {
        at<uint4>(sm, P64_KX + csub * 8192 + offu[u]) =
            make_uint4(kx[0], kx[1], kx[2], kx[3]);
        at<uint4>(sm, P64_KX + csub * 8192 + 4096 + offu[u]) =
            make_uint4(qx[0], qx[1], qx[2], qx[3]);
      }
    }
  }
  ptx::fence_proxy_async_smem();
  __syncthreads();

  // ---- P3: diagonal blocks [kk; qk]_b = [Kp; Qp]_b Km_b^T (mid frame);
  // warpgroup 0 also the cross block ----
  const int g = pl >> 2, cq = pl & 3;
  float acc[16], accX[16];
  ptx::wgmma_fence();
#pragma unroll
  for (int ks = 0; ks < 8; ++ks)
    ptx::wgmma_m64n32k16_bf16_ss(
        acc, kdesc(ib + (ks >> 2) * IN_KQS + (ks & 3) * 32),
        kdesc(ib + IN_G + (ks >> 2) * IN_GS + (ks & 3) * 32), ks > 0);
  ptx::wgmma_commit();
  if (b == 0) {
#pragma unroll
    for (int ks = 0; ks < 8; ++ks)
      ptx::wgmma_m64n32k16_bf16_ss(
          accX, kdesc(sb + P64_KX + (ks >> 2) * 8192 + (ks & 3) * 32),
          kdesc(sb + P64_KX + 16384 + (ks >> 2) * 4096 + (ks & 3) * 32),
          ks > 0);
    ptx::wgmma_commit();
  }
  ptx::wgmma_wait<0>();
  __syncthreads();  // the blocks below overwrite the k, q tiles and the KX
                    // region
  float* sc = reinterpret_cast<float*>(sm + P64_KX +
                                       7744 * b);  // this sub's inverse scratch
  float* Ld = sc;                                  // [2][16][20] (+4 pad)
  float* L10 = sc + 648;                           // [16][20]
  float* Ei = sc + 968;                            // [2][16][20] (+4 pad)
  float* Tm = sc + 1616;                           // [16][20]
  float* AIf = reinterpret_cast<float*>(sm + P64_AIF + 4608 * b);  // [32][36]
  float* KKf = reinterpret_cast<float*>(sm + P64_KKF);             // [32][36]
  const uint32_t qkt = sb + P64_QK, ait = sb + P64_AI;
  {
    const int mk = pl >> 3;
#pragma unroll
    for (int hr = 0; hr < 2; ++hr) {
      const int il = g + 8 * hr;
      if (pw < 2) {
        const float bi = bS[16 * pw + il];
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
          const int jl = 8 * (jj & 1) + 2 * cq;
          const float v0 = bi * acc[4 * jj + 2 * hr],
                      v1 = bi * acc[4 * jj + 2 * hr + 1];
          if (jj < 2 && pw == 1) {
            *reinterpret_cast<float2*>(L10 + il * 20 + jl) =
                make_float2(v0, v1);
          } else if ((jj < 2) == (pw == 0)) {
            *reinterpret_cast<float2*>(Ld + pw * 324 + il * 20 + jl) =
                make_float2(jl < il ? v0 : 0.f, jl + 1 < il ? v1 : 0.f);
          }
          if (b == 0)  // kk10 raw
            *reinterpret_cast<float2*>(KKf + (16 * pw + il) * F36 + 8 * jj +
                                       2 * cq) =
                make_float2(accX[4 * jj + 2 * hr], accX[4 * jj + 2 * hr + 1]);
        }
      }
    }
    if (pw >= 2) {
      // qk rows 32 b + i of the [i][m] tile (diagonal block masked j <= i);
      // warpgroup 0 also qk10 (rows 32 + i)
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        uint32_t r4[4], x4[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const int hr = k & 1, jj = 2 * q + (k >> 1);
          const int i = 16 * (pw - 2) + g + 8 * hr, j = 8 * jj + 2 * cq;
          r4[k] = pack_bf16(j <= i ? acc[4 * jj + 2 * hr] : 0.f,
                            j + 1 <= i ? acc[4 * jj + 2 * hr + 1] : 0.f);
          x4[k] = pack_bf16(accX[4 * jj + 2 * hr], accX[4 * jj + 2 * hr + 1]);
        }
        const int row = 16 * (pw - 2) + 8 * (mk & 1) + (pl & 7),
                  cb = 4 * b + 2 * q + (mk >> 1);
        const int rr = 32 * b + row;
        ptx::stmatrix_x4(qkt + rr * 128 + (((cb ^ rr) & 7) << 4), r4[0], r4[1],
                         r4[2], r4[3]);
        if (b == 0) {
          const int rx = 32 + row, cx = 2 * q + (mk >> 1);
          ptx::stmatrix_x4(qkt + rx * 128 + (((cx ^ rx) & 7) << 4), x4[0],
                           x4[1], x4[2], x4[3]);
        }
      }
    }
    if (b == 0) {  // zero upper-right blocks of the qk and AI tiles: rows
                   // 0..31, cols 32..63
      const int r = pt >> 2, cc = 4 + (pt & 3);
      at<uint4>(sm, P64_QK + r * 128 + (((cc ^ r) & 7) << 4)) =
          make_uint4(0, 0, 0, 0);
      at<uint4>(sm, P64_AI + r * 128 + (((cc ^ r) & 7) << 4)) =
          make_uint4(0, 0, 0, 0);
    }
  }
  __syncthreads();

  // ---- P4: AI_b = (I + L_b)^-1 diag(b_b) per sub: 16x16 diagonal blocks by
  // substitution, then the off-diagonal ----
  auto ai_store =
      [&](int r, int cc,
          float v) {  // element (r, cc) of AI_b: bf16 tile and fp32 copy
        at<__nv_bfloat16>(sm, P64_AI + sw(32 * b + r, 32 * b + cc)) =
            __float2bfloat16_rn(v);
        AIf[r * F36 + cc] = v;
      };
  if (pw == 0) {
    const int bb = pl >> 4, cc = pl & 15;
    const float* Lb = Ld + bb * 324;
    float xv[16];
#pragma unroll
    for (int r = 0; r < 16; ++r) xv[r] = 0.f;
#pragma unroll
    for (int r = 0; r < 16; ++r) {
      float a = 0.f;
#pragma unroll
      for (int m4 = 0; m4 < r; m4 += 4) {
        const float4 lv = *reinterpret_cast<const float4*>(Lb + r * 20 + m4);
        a = fmaf(lv.x, xv[m4], a);
        if (m4 + 1 < r) a = fmaf(lv.y, xv[m4 + 1], a);
        if (m4 + 2 < r) a = fmaf(lv.z, xv[m4 + 2], a);
        if (m4 + 3 < r) a = fmaf(lv.w, xv[m4 + 3], a);
      }
      xv[r] = r == cc ? 1.f : (r < cc ? 0.f : -a);
    }
    float* E = Ei + bb * 324;
    const float bj = bS[16 * bb + cc];
#pragma unroll
    for (int r = 0; r < 16; ++r) {
      E[r * 20 + cc] = xv[r];
      ai_store(16 * bb + r, 16 * bb + cc, xv[r] * bj);
    }
  } else if (pw == 1) {
    const int r = pl >> 1,
              c0 =
                  16 + 8 * (pl & 1);  // upper-right 16x16 block of AI_b is zero
#pragma unroll
    for (int e = 0; e < 8; ++e) AIf[r * F36 + c0 + e] = 0.f;
    at<uint4>(sm, P64_AI + sw(32 * b + r, 32 * b + c0)) =
        make_uint4(0, 0, 0, 0);
  }
  __syncthreads();
  {
    const int r = pt >> 3, c0 = (pt & 7) * 2;
    float s0 = 0.f, s1 = 0.f;
#pragma unroll
    for (int m4 = 0; m4 < 16; m4 += 4) {
      const float4 lv = *reinterpret_cast<const float4*>(L10 + r * 20 + m4);
      const float lw[4] = {lv.x, lv.y, lv.z, lv.w};
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const float2 ev =
            *reinterpret_cast<const float2*>(Ei + (m4 + q) * 20 + c0);
        s0 = fmaf(lw[q], ev.x, s0);
        s1 = fmaf(lw[q], ev.y, s1);
      }
    }
    *reinterpret_cast<float2*>(Tm + r * 20 + c0) = make_float2(s0, s1);
    __syncthreads();
    float u0 = 0.f, u1 = 0.f;
#pragma unroll
    for (int m4 = 0; m4 < 16; m4 += 4) {
      const float4 ev =
          *reinterpret_cast<const float4*>(Ei + 324 + r * 20 + m4);
      const float ew[4] = {ev.x, ev.y, ev.z, ev.w};
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const float2 tv =
            *reinterpret_cast<const float2*>(Tm + (m4 + q) * 20 + c0);
        u0 = fmaf(ew[q], tv.x, u0);
        u1 = fmaf(ew[q], tv.y, u1);
      }
    }
    ai_store(16 + r, c0, -u0 * bS[c0]);
    ai_store(16 + r, c0 + 1, -u1 * bS[c0 + 1]);
  }
  __syncthreads();
  // ---- X = -AI1 (kk10 AI0): rows 32..63, cols 0..31 of AI. Sequential mode
  // and the first chunks of the overlapped mode (K2 waits for them): tf32 mma,
  // warp w: rows 16 (w >> 2), cols 8 (w & 3). Later overlapped chunks: fp32
  // SIMT (with the tf32 join there, K2 ran slower beside K1) ----
  if (!LAT || c < 4) {
    const int wp = tid >> 5, r0 = 16 * (wp >> 2), n0 = 8 * (wp & 3),
              gq = pl >> 2, cq4 = pl & 3;
    const float* A0 = reinterpret_cast<const float*>(sm + P64_AIF);
    const float* A1 = A0 + 32 * F36;
    // D = A B over k = 0..31 for this warp's 16 x 8 tile; A, B fp32 [32][36] in
    // shared memory
    auto mm = [&](const float* A, const float* B, float (&d)[4]) {
#pragma unroll
      for (int e = 0; e < 4; ++e) d[e] = 0.f;
#pragma unroll
      for (int k0 = 0; k0 < 32; k0 += 8) {
        const uint32_t a[4] = {
            __float_as_uint(A[(r0 + gq) * F36 + k0 + cq4]),
            __float_as_uint(A[(r0 + gq + 8) * F36 + k0 + cq4]),
            __float_as_uint(A[(r0 + gq) * F36 + k0 + cq4 + 4]),
            __float_as_uint(A[(r0 + gq + 8) * F36 + k0 + cq4 + 4])};
        ptx::mma_m16n8k8_tf32(
            d, a, __float_as_uint(B[(k0 + cq4) * F36 + n0 + gq]),
            __float_as_uint(B[(k0 + cq4 + 4) * F36 + n0 + gq]));
      }
    };
    float t[4];
    mm(KKf, A0, t);
    __syncthreads();
    *reinterpret_cast<float2*>(KKf + (r0 + gq) * F36 + n0 + 2 * cq4) =
        make_float2(t[0], t[1]);
    *reinterpret_cast<float2*>(KKf + (r0 + gq + 8) * F36 + n0 + 2 * cq4) =
        make_float2(t[2], t[3]);
    __syncthreads();
    mm(A1, KKf, t);
    at<uint32_t>(sm, P64_AI + sw(32 + r0 + gq, n0 + 2 * cq4)) =
        pack_bf16(-t[0], -t[1]);
    at<uint32_t>(sm, P64_AI + sw(40 + r0 + gq, n0 + 2 * cq4)) =
        pack_bf16(-t[2], -t[3]);
  } else {
    const int i = tid >> 3, j0 = (tid & 7) * 4;
    const float* A0 = reinterpret_cast<const float*>(sm + P64_AIF);
    const float* A1 = A0 + 32 * F36;
    float t[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 8
    for (int m = 0; m < 32; ++m) {
      const float kv = KKf[i * F36 + m];
      const float4 av = *reinterpret_cast<const float4*>(A0 + m * F36 + j0);
      t[0] = fmaf(kv, av.x, t[0]);
      t[1] = fmaf(kv, av.y, t[1]);
      t[2] = fmaf(kv, av.z, t[2]);
      t[3] = fmaf(kv, av.w, t[3]);
    }
    __syncthreads();
    *reinterpret_cast<float4*>(KKf + i * F36 + j0) =
        make_float4(t[0], t[1], t[2], t[3]);
    __syncthreads();
    float x[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 8
    for (int m = 0; m < 32; ++m) {
      const float av = A1[i * F36 + m];
      const float4 tv = *reinterpret_cast<const float4*>(KKf + m * F36 + j0);
      x[0] = fmaf(av, tv.x, x[0]);
      x[1] = fmaf(av, tv.y, x[1]);
      x[2] = fmaf(av, tv.z, x[2]);
      x[3] = fmaf(av, tv.w, x[3]);
    }
    const uint32_t o = sw(32 + i, j0);
    at<uint32_t>(sm, P64_AI + o) = pack_bf16(-x[0], -x[1]);
    at<uint32_t>(sm, P64_AI + o + 4) = pack_bf16(-x[2], -x[3]);
  }
  ptx::fence_proxy_async_smem();
  __syncthreads();

  // ---- P5: W^T = K_end^T AI (warpgroup b: keys 64 b .. 64 b + 63) and (qk
  // AI)^T = AI^T qk^T (warpgroup b: i in sub b)
  {
    const int mk = pl >> 3;
    float accA[16], accB[32];
    ptx::wgmma_fence();
    // A = Km^T read MN-major from the g tiles (key sub-tile b), B = AI [i][j]
    // read MN-major
#pragma unroll
    for (int ks = 0; ks < 2; ++ks)
      ptx::wgmma_m64n32k16_bf16_ss<1, 1>(
          accA,
          ptx::wgmma_desc_sw128(sb + IN_G + IN_GS * b + 2048 * ks, 4096, 1024),
          ptx::wgmma_desc_sw128(ait + 2048 * ks, 8192, 1024), ks);
    ptx::wgmma_commit();
#pragma unroll
    for (int ks = 0; ks < 2; ++ks)
      ptx::wgmma_m64n64k16_bf16_ss<1, 1>(
          accB,
          ptx::wgmma_desc_sw128(sb + 24576 + IN_G + IN_GS * b + 2048 * ks, 4096,
                                1024),
          ptx::wgmma_desc_sw128(ait + 4096 + 2048 * ks, 8192, 1024), ks);
    ptx::wgmma_commit();
    ptx::wgmma_wait<0>();
    // per key row: sub 0 rows scaled by exp(G_31 - G_15) then exp(G_63 - G_31),
    // sub 1 rows by exp(G_63 - G_47)
    float fa[2], fd[2], fb[2];
#pragma unroll
    for (int hr = 0; hr < 2; ++hr) {
      const int key = 64 * b + 16 * pw + g + 8 * hr;
      fa[hr] = ptx::ex2_approx(tS[128 + key]);
      fd[hr] = ptx::ex2_approx(tS[256 + key] + tS[384 + key]);
      fb[hr] = ptx::ex2_approx(tS[384 + key]);
    }
    // W staging (KX region): N-block b at 8192 b, rows j, cols key - 64 b
    const uint32_t wt = sb + P64_KX + 8192 * b;
#pragma unroll
    for (int q = 0; q < 4; ++q) {  // 16 j columns per stmatrix.x4.trans
      uint32_t r4[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int hr = k & 1, jj = 2 * q + (k >> 1);
        float v0 = accB[4 * jj + 2 * hr] * fb[hr],
              v1 = accB[4 * jj + 2 * hr + 1] * fb[hr];
        if (jj < 4) {
          v0 = fmaf(accA[4 * jj + 2 * hr] * fa[hr], fd[hr], v0);
          v1 = fmaf(accA[4 * jj + 2 * hr + 1] * fa[hr], fd[hr], v1);
        }
        r4[k] = pack_bf16(v0, v1);
      }
      // matrix mk: (key rows 8 (mk & 1) of the warp's 16, j columns 8 (2 q +
      // (mk >> 1))); transposed -> rows j
      const int j = 8 * (2 * q + (mk >> 1)) + (pl & 7), kb = 2 * pw + (mk & 1);
      ptx::stmatrix_x4_trans(wt + j * 128 + (((kb ^ j) & 7) << 4), r4[0], r4[1],
                             r4[2], r4[3]);
    }
  }
  {
    const int mk = pl >> 3;
    float accT[16];
    ptx::wgmma_fence();
    // M = j (A = AI [m][j] read MN-major), N = i of sub b (B = qk rows 32 b..,
    // K-major)
    ptx::wgmma_m64n32k16_bf16_ss<1, 0>(
        accT, ptx::wgmma_desc_sw128(ait, 8192, 1024), kdesc(qkt + 4096 * b), 0);
    ptx::wgmma_m64n32k16_bf16_ss<1, 0>(
        accT, ptx::wgmma_desc_sw128(ait + 2048, 8192, 1024),
        kdesc(qkt + 4096 * b + 32), 1);
    if (b) {
      ptx::wgmma_m64n32k16_bf16_ss<1, 0>(
          accT, ptx::wgmma_desc_sw128(ait + 4096, 8192, 1024),
          kdesc(qkt + 4096 * b + 64), 1);
      ptx::wgmma_m64n32k16_bf16_ss<1, 0>(
          accT, ptx::wgmma_desc_sw128(ait + 6144, 8192, 1024),
          kdesc(qkt + 4096 * b + 96), 1);
    }
    ptx::wgmma_commit();
    ptx::wgmma_wait<0>();
    const uint32_t tt = sb + P64_KX + 16384;  // (qk AI)^T staging [j][i]
#pragma unroll
    for (int q = 0; q < 2; ++q) {
      uint32_t r4[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int hr = k & 1, jj = 2 * q + (k >> 1);
        r4[k] = pack_bf16(accT[4 * jj + 2 * hr], accT[4 * jj + 2 * hr + 1]);
      }
      const int row = 16 * pw + 8 * (mk & 1) + (pl & 7),
                cb = 4 * b + 2 * q + (mk >> 1);
      ptx::stmatrix_x4(tt + row * 128 + (((cb ^ row) & 7) << 4), r4[0], r4[1],
                       r4[2], r4[3]);
    }
  }
  ptx::fence_proxy_async_smem();
  __syncthreads();
  if (tid == 0) {
    ptx::bulk_copy_s2g(wsi + S64_W, sb + P64_KX, S64_GC - S64_W);
    ptx::bulk_commit();
    if (LAT) {
      ptx::bulk_wait<0>();
      ptx::fence_proxy_async_global();
      ptx::fence_acq_rel_gpu();
      ptx::st_release_gpu(p.flags + 1 + slot * H + h, 2);
    } else {
      ptx::bulk_wait_read<0>();
    }
  }
  if (!LAT) ptx::griddep_launch_dependents();
}

template <int NST, bool LAT>
__global__ void __launch_bounds__(128, 1)
    kda_recur64(const __grid_constant__ Params p) {
  if (LAT) ptx::griddep_launch_dependents();
  constexpr int OFF_OUT = NST * S64_STRIDE, OFF_BAR = OFF_OUT + 16384;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* sm = smem_raw + ((1024 - (ptx::smem_u32(smem_raw) & 1023)) & 1023);
  const uint32_t sb = ptx::smem_u32(sm);
  const int tid = threadIdx.x, H = p.H;
  const int h = blockIdx.x >> 1, half = blockIdx.x & 1;
  const int n = p.N - 1 - blockIdx.y;
  const int bos = p.cu_seqlens[n];
  const int off = p.ckpt[n];
  Chunking64 ch;
  ch.init(p.cu_seqlens[n + 1] - bos, off);
  const int nch = ch.count();
  const int ck_chunk = off > 0 ? (ch.shift ? ch.m : (off >> 6) - 1) : -1;
  const uint32_t bar_full = sb + OFF_BAR;
  if (tid == 0) {
    for (int i = 0; i < NST; ++i) ptx::mbarrier_init(bar_full + 8 * i, 1);
    ptx::fence_mbarrier_init();
  }
  __syncthreads();
  if (!LAT) ptx::griddep_wait();
  const int base = LAT ? n * p.lat_stride : p.meta[n];
  int fpre = 0;
  auto load = [&](int c) {
    const int s = c % NST, l = tid & 31;
    const uint32_t st = sb + s * S64_STRIDE, full = bar_full + 8 * s;
    if (LAT && l == 0) {
      int* f = p.flags + 1 + (base + c) * H + h;
      while (fpre == 0) fpre = ptx::ld_relaxed_gpu(f);
      ptx::fence_acquire_gpu();
      ptx::fence_proxy_async_global();
      atomicSub(f, 1);
      fpre = c + 1 < nch ? ptx::ld_relaxed_gpu(f + H) : 1;
    }
    if (l == 0) {
      ptx::mbarrier_arrive_expect_tx(full, IMG64 + 8192);
      ptx::bulk_copy_g2s(st, p.ws + ((size_t)(base + c) * H + h) * IMG64, IMG64,
                         full);
      ptx::tma_load_3d(st + S64_V, &p.tmpl[0], 64 * half, 2 * H + h,
                       bos + ch.start(c), full);
      ptx::tma_load_3d(st + S64_V + 4096, &p.tmpl[0], 64 * half, 2 * H + h,
                       bos + ch.start(c) + 32, full);
    }
  };
  if (tid < 32)
    for (int c = 0; c < NST - 1 && c < nch; ++c) load(c);

  const int w = tid >> 5, l = tid & 31, g = l >> 2, cq = l & 3;
  const int vrow = 64 * half + 16 * w + g;
  float S[64];
  const size_t sbase = ((size_t)n * H + h) * 128 * 128;
  {
    const float* src = p.initial_state + sbase;
#pragma unroll
    for (int j = 0; j < 16; ++j)
#pragma unroll
      for (int hr = 0; hr < 2; ++hr) {
        const float2 v = *reinterpret_cast<const float2*>(
            src + (vrow + 8 * hr) * 128 + 8 * j + 2 * cq);
        S[4 * j + 2 * hr] = v.x;
        S[4 * j + 2 * hr + 1] = v.y;
      }
  }
  auto store_state = [&](float* dst) {
#pragma unroll
    for (int j = 0; j < 16; ++j)
#pragma unroll
      for (int hr = 0; hr < 2; ++hr)
        *reinterpret_cast<float2*>(dst + (vrow + 8 * hr) * 128 + 8 * j +
                                   2 * cq) =
            make_float2(S[4 * j + 2 * hr], S[4 * j + 2 * hr + 1]);
  };
  // output of chunk c: each warp transposes its 16 value columns into its own
  // [64 tokens][32 B] tile (32B swizzle, two buffers); a full chunk goes out as
  // two 32-row TMA stores, others copy their valid rows
  auto store_out = [&](const uint32_t (&op)[16], int c) {
    const int t0 = ch.start(c), L = ch.end(c) - t0;
    const uint32_t ot = sb + OFF_OUT + ((c & 1) * 4 + w) * 2048;
    if (l == 0) ptx::bulk_wait_read<1>();
    __syncwarp();
    const int mi = l >> 3;
#pragma unroll
    for (int hf = 0; hf < 4; ++hf) {
      const int trow = 8 * (2 * hf + (mi >> 1)) + (l & 7);
      ptx::stmatrix_x4_trans(
          ot + trow * 32 + ((((mi & 1) ^ (trow >> 2)) & 1) << 4), op[4 * hf],
          op[4 * hf + 1], op[4 * hf + 2], op[4 * hf + 3]);
    }
    if (L == C64) {
      ptx::fence_proxy_async_smem();
      __syncwarp();
      if (l == 0) {
        ptx::tma_store_3d(&p.tmpl[1], 64 * half + 16 * w, h, bos + t0, ot);
        ptx::tma_store_3d(&p.tmpl[1], 64 * half + 16 * w, h, bos + t0 + 32,
                          ot + 1024);
        ptx::bulk_commit();
      }
    } else {
      __syncwarp();
      const uint8_t* otp = sm + (ot - sb);
#pragma unroll
      for (int pass = 0; pass < 4; ++pass) {
        const int t = (l >> 1) + 16 * pass, j = l & 1;
        if (t < L) {
          const uint4 v = *reinterpret_cast<const uint4*>(
              otp + t * 32 + (((j ^ (t >> 2)) & 1) << 4));
          *reinterpret_cast<uint4*>(p.out + (size_t)(bos + t0 + t) * p.os +
                                    h * 128 + 64 * half + 16 * w + 8 * j) = v;
        }
      }
      if (l == 0) ptx::bulk_commit();
    }
  };

  uint32_t opk[16];
#pragma unroll 1
  for (int c = 0; c < nch; ++c) {
    const int s = c % NST;
    const uint32_t st = sb + s * S64_STRIDE;
    uint8_t* stp = sm + s * S64_STRIDE;
    ptx::mbarrier_wait_parity(bar_full + 8 * s, (c / NST) & 1);
    uint32_t aS[8][4];
#pragma unroll
    for (int ks = 0; ks < 8; ++ks)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        aS[ks][e] = pack_bf16(S[8 * ks + 2 * e], S[8 * ks + 2 * e + 1]);
        ptx::reg_fence_u32(aS[ks][e]);
      }
    // acc1 = [Kp S0^T | Qp S0^T]^T: columns 0..63 Kp tokens, 64..127 Qp tokens
    float acc1[64];
    ptx::wgmma_fence();
    ptx::wgmma_m64n128k16_bf16_rs_z(acc1, aS[0], kdesc(st + S64_A));
#pragma unroll
    for (int ks = 1; ks < 8; ++ks)
      ptx::wgmma_m64n128k16_bf16_rs(
          acc1, aS[ks], kdesc(st + S64_A + (ks >> 2) * 16384 + (ks & 3) * 32),
          1);
    ptx::wgmma_commit();
    if (w == 0 && c + NST - 1 < nch) load(c + NST - 1);
    uint32_t vt[8][2];
    {
      const int mi = l >> 3, rr = l & 7;
#pragma unroll
      for (int hf = 0; hf < 4; ++hf) {
        const int t = 8 * (2 * hf + (mi >> 1)) + rr, v = 16 * w + 8 * (mi & 1);
        uint32_t r[4];
        ptx::ldmatrix_x4_trans(r, st + S64_V + sw(t, v));
        vt[2 * hf][0] = r[0];
        vt[2 * hf][1] = r[1];
        vt[2 * hf + 1][0] = r[2];
        vt[2 * hf + 1][1] = r[3];
      }
    }
    float2 gcf[16];
#pragma unroll
    for (int j = 0; j < 16; ++j)
      gcf[j] = at<const float2>(stp, S64_GC + (8 * j + 2 * cq) * 4);
    ptx::wgmma_wait<0>();
#pragma unroll
    for (int e = 0; e < 64; ++e) ptx::reg_fence(acc1[e]);
    uint32_t aY[4][4];
#pragma unroll
    for (int jj = 0; jj < 8; ++jj)
#pragma unroll
      for (int hr = 0; hr < 2; ++hr) {
        const float2 vv = unpack_bf16(vt[jj][hr]);
        aY[jj >> 1][2 * (jj & 1) + hr] = pack_bf16(
            vv.x - acc1[4 * jj + 2 * hr], vv.y - acc1[4 * jj + 2 * hr + 1]);
        ptx::reg_fence_u32(aY[jj >> 1][2 * (jj & 1) + hr]);
      }
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      S[4 * j] *= gcf[j].x;
      S[4 * j + 1] *= gcf[j].y;
      S[4 * j + 2] *= gcf[j].x;
      S[4 * j + 3] *= gcf[j].y;
    }
    // S += Y^T W, O2 = Y^T (qk AI)^T
    float acc2[32];
    ptx::wgmma_fence();
#pragma unroll
    for (int ks = 0; ks < 4; ++ks)
      ptx::wgmma_m64n128k16_bf16_rs<1>(
          S, aY[ks], ptx::wgmma_desc_sw128(st + S64_W + 2048 * ks, 8192, 1024),
          1);
    ptx::wgmma_m64n64k16_bf16_rs_z<1>(
        acc2, aY[0], ptx::wgmma_desc_sw128(st + S64_QA, 8192, 1024));
#pragma unroll
    for (int ks = 1; ks < 4; ++ks)
      ptx::wgmma_m64n64k16_bf16_rs<1>(
          acc2, aY[ks],
          ptx::wgmma_desc_sw128(st + S64_QA + 2048 * ks, 8192, 1024), 1);
    ptx::wgmma_commit();
    if (c > 0) store_out(opk, c - 1);
    ptx::wgmma_wait<0>();
#pragma unroll
    for (int e = 0; e < 16; ++e)
      opk[e] = pack_bf16(acc1[32 + 2 * e] + acc2[2 * e],
                         acc1[32 + 2 * e + 1] + acc2[2 * e + 1]);
    if (c == ck_chunk) store_state(p.ckpt_state + sbase);
  }
  if (nch > 0) store_out(opk, nch - 1);
  store_state(p.final_state + sbase);
  if (l == 0) ptx::bulk_wait_read<0>();
}

// Workspace slots: 32-token images (one per chunk, plus per-sequence slack),
// and with H N <= 16 (the only case in which the 64-token path can run)
// 64-token images, which in the overlapped mode take T / 64 + 2 slots per
// sequence.
struct Dims {
  int64_t T, N, H;
};
size_t ws_slots(const Dims& d) {
  return (size_t)(d.T + C - 1) / C + 2 * (size_t)d.N + 1;
}
size_t lat_stride(const Dims& d) { return (size_t)d.T / C64 + 2; }
size_t ws_slots64(const Dims& d) {
  return std::max((size_t)(d.T + C64 - 1) / C64 + 2 * (size_t)d.N + 1,
                  d.H * d.N <= 16 ? d.N * lat_stride(d) : 0);
}
bool may_use_64(const Dims& d) { return d.H * d.N <= 16; }
size_t ws_bytes(const Dims& d) {
  const size_t b32 = ws_slots(d) * IMG_BYTES;
  return (may_use_64(d) ? std::max(b32, ws_slots64(d) * IMG64) : b32) *
         (size_t)d.H;
}

struct Args {
  Dims dims;
  const __nv_bfloat16* qkv;  // q of token t at qkv + t * qs; k, v follow at + H
                             // * 128, + 2 * H * 128
  const __nv_bfloat16* g;
  const __nv_bfloat16* beta;
  const float* A_log;
  const float* dt_bias;
  const float* initial_state;
  const int32_t* cu_seqlens;
  const int32_t* checkpoint_offsets;
  __nv_bfloat16* out;
  float* final_state;
  float* checkpoint_state;
  int64_t qs, gs, bs, os;  // token strides (elements)
  float scale;
  float lower_bound;
};
struct Ctx {
  cudaStream_t stream;
  int device, sm_count;
  void* scratch;     // ws_bytes + meta, contents arbitrary at entry
  void* zeroed;      // zeroed_bytes, zero at entry; the kernels leave it zero
  void* memset_ptr;  // set to zero on the stream right before the first launch
                     // (after the host-side setup)
  size_t memset_bytes;  // 0: nothing to zero
};

// scratch: operand images + 384 + meta [N + 2]; zeroed: the overlapped mode's
// ready flags and the pacing counter
void kernel_workspace(const Dims& d, size_t* scratch_bytes,
                      size_t* zeroed_bytes) {
  *scratch_bytes = ws_bytes(d) + 384 + 4 * (size_t)d.N + 8;
  *zeroed_bytes = 4 * (1 + (may_use_64(d) ? std::max(ws_slots(d), ws_slots64(d))
                                          : ws_slots(d)) *
                               (size_t)d.H);
}

// Per-device kernel handles and attributes (a CUfunction belongs to one
// context).
struct DeviceState {
  bool init = false, init64 = false;
  CUfunction f_prep = nullptr, f_prepl = nullptr, f_r54l = nullptr;
  CUfunction f_r64l = nullptr, f_p64 = nullptr, f_p64l = nullptr,
             f_r64 = nullptr;
};
constexpr int kMaxDevices = 64;

// Enqueues the kernels on ctx.stream; returns false when a launch or a setup
// call fails.
bool kernel_run(const Args& a, const Ctx& ctx) {
  const int T = (int)a.dims.T, N = (int)a.dims.N, H = (int)a.dims.H;
  bool ok = true;
  auto chk = [&](auto r) { ok = ok && static_cast<int>(r) == 0; };
  Params p;
  p.qkv = a.qkv;
  p.g = a.g;
  p.beta = a.beta;
  p.A_log = a.A_log;
  p.dt_bias = a.dt_bias;
  p.initial_state = a.initial_state;
  p.cu_seqlens = a.cu_seqlens;
  p.ckpt = a.checkpoint_offsets;
  p.out = a.out;
  p.final_state = a.final_state;
  p.ckpt_state = a.checkpoint_state;
  p.ws = static_cast<uint8_t*>(ctx.scratch);
  p.flags = static_cast<int*>(ctx.zeroed);
  p.meta = reinterpret_cast<int*>(p.ws + ws_bytes(a.dims) + 384);
  p.qs = a.qs;
  p.gs = a.gs;
  p.bs = a.bs;
  p.os = a.os;
  {
    // one set of maps per head count and token strides, encoded once; per call
    // only the address and T change
    struct Maps {
      int H;
      int64_t qs, os, gs;
      bool ok;
      cu::MapPatch m[4];
    };
    static Maps cache[8];
    static int nt = 0;
    const void* base[4] = {a.qkv, a.out, a.out, a.g};
    const int64_t tstride[4] = {a.qs, a.os, a.os, a.gs};
    auto encode = [&](int k, CUtensorMap* map, const void* b, cuuint64_t t) {
      const cuuint64_t dims[3] = {128, (cuuint64_t)(k == 0 ? 3 * H : H), t};
      const cuuint64_t strides[2] = {256, (cuuint64_t)(2 * tstride[k])};
      const cuuint32_t box[3] = {k == 0 || k == 3 ? 64u : 16u, 1,
                                 k == 2 ? 128u : 32u};
      return cu::encode_tiled(map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, b, dims,
                              strides, box,
                              k == 0 || k == 3 ? CU_TENSOR_MAP_SWIZZLE_128B
                                               : CU_TENSOR_MAP_SWIZZLE_32B);
    };
    int i = 0;
    while (i < nt && !(cache[i].H == H && cache[i].qs == a.qs &&
                       cache[i].os == a.os && cache[i].gs == a.gs))
      ++i;
    if (i == nt) {
      if (nt < 8)
        ++nt;
      else
        i = 7;
      cache[i].H = H;
      cache[i].qs = a.qs;
      cache[i].os = a.os;
      cache[i].gs = a.gs;
      cache[i].ok = true;
      for (int k = 0; k < 4; ++k)
        cache[i].ok &=
            cu::probe_patch(&cache[i].m[k], base[k], (cuuint64_t)T,
                            [&](CUtensorMap* map, const void* b, cuuint64_t t) {
                              return encode(k, map, b, t);
                            });
    }
    for (int k = 0; k < 4; ++k) {
      if (cache[i].ok)
        cu::apply_patch(cache[i].m[k], &p.tmpl[k], base[k], (cuuint64_t)T);
      else
        ok = encode(k, &p.tmpl[k], base[k], (cuuint64_t)T) && ok;
    }
  }
  p.T = T;
  p.lat_stride = (int)lat_stride(a.dims);
  p.kord = N < 2 ? 0 : (N <= 32 ? 1 : 2);
  p.early = (T / C) * H <= ctx.sm_count ? 1 : 0;
  p.sms = ctx.sm_count;
  p.c0 = 0;
  p.trig = 0;
  p.N = N;
  p.H = H;
  p.scale = a.scale;
  p.lb = a.lower_bound;
  if (ctx.device < 0 || ctx.device >= kMaxDevices) return false;
  if (N == 0) return true;
  static DeviceState states[kMaxDevices];
  DeviceState& ds = states[ctx.device];
  if (!ds.init) {
    ds.f_prep = cu::driver_function(kda_prep<false>);
    ds.f_prepl = cu::driver_function(kda_prep<true>);
    ds.f_r54l = cu::driver_function(kda_recur<5, 4, true, true>);
    chk(cu::max_dynamic_smem(kda_prep<false>, K1_SMEM));
    chk(cu::max_dynamic_smem(kda_prep<true>, K1_SMEM));
    chk(cu::max_dynamic_smem(kda_recur<3, 1, false, false>, k2_smem(3, 1)));
    chk(cu::max_dynamic_smem(kda_recur<3, 1, false, true>, k2_smem(3, 0)));
    chk(cu::max_dynamic_smem(kda_recur<5, 4, false, true>, k2_smem(5, 0)));
    chk(cu::max_dynamic_smem(kda_recur<5, 4, true, true>, k2_smem(5, 0)));
    if (!ok) return false;
    ds.init = true;
  }
  if (ctx.memset_bytes)
    chk(cudaMemsetAsync(ctx.memset_ptr, 0, ctx.memset_bytes, ctx.stream));
  // K2 on at most a quarter of the SMs: 64-token chunks (K2's chain is the
  // longer one; K1 has SMs to spare)
  if (8 * H * N <= ctx.sm_count && H * N <= 16) {
    if (!ds.init64) {
      ds.f_r64l = cu::driver_function(kda_recur64<3, true>);
      ds.f_p64 = cu::driver_function(kda_prep64<false>);
      ds.f_p64l = cu::driver_function(kda_prep64<true>);
      ds.f_r64 = cu::driver_function(kda_recur64<3, false>);
      chk(cu::max_dynamic_smem(kda_prep64<false>, P64_SMEM));
      chk(cu::max_dynamic_smem(kda_prep64<true>, P64_SMEM));
      chk(cu::max_dynamic_smem(kda_recur64<3, false>, k64_smem(3)));
      chk(cu::max_dynamic_smem(kda_recur64<3, true>, k64_smem(3)));
      if (!ok) return false;
      ds.init64 = true;
    }
    if ((T / C) * H >= 2048) {
      // the first E chunks of every sequence are prepped by a grid launched
      // before K2 on the SMs K2 leaves free, the rest by a grid launched as
      // K2's programmatic dependent
      // E = 12 at H * N = 16 (measured best at T=8192, H=16, one sequence)
      const int E = std::min(192 / (H * N), T / C64 + 1);
      Params pe = p;
      pe.c0 = 0;
      pe.trig = 1;
      if (E > 0) {
        chk(cu::launch_driver(ds.f_p64l, dim3(H, N, E), dim3(256), P64_SMEM,
                              ctx.stream, pe));
        chk(cu::launch_driver_pdl(ds.f_r64l, dim3(2 * H, N), dim3(128),
                                  k64_smem(3), ctx.stream, p));
      } else {
        chk(cu::launch_driver(ds.f_r64l, dim3(2 * H, N), dim3(128), k64_smem(3),
                              ctx.stream, p));
      }
      pe.c0 = std::max(E, 0);
      pe.trig = 0;
      chk(cu::launch_driver_pdl(ds.f_p64l, dim3(H, N, T / C64 + 2 - pe.c0),
                                dim3(256), P64_SMEM, ctx.stream, pe));
    } else {
      chk(cu::launch_driver(ds.f_p64, dim3(H, (unsigned)ws_slots64(a.dims)),
                            dim3(256), P64_SMEM, ctx.stream, p));
      chk(cu::launch_driver_pdl(ds.f_r64, dim3(2 * H, N), dim3(128),
                                k64_smem(3), ctx.stream, p));
    }
    return ok;
  }
  if (4 * H * N <= ctx.sm_count && (T / C) * H >= 2048) {
    // K2 holds at most half of the SMs and K1 takes many waves: run K2 first
    // and K1 beside it on the free SMs, chunk images flagged ready one by one
    // the first E chunks are prepped by a grid launched before K2, the rest by
    // K2's programmatic dependent
    const int E =
        std::min(std::max(512 / (H * N), 1), T / C + 1);  // 16 at H N = 32
    Params pe = p;
    pe.c0 = 0;
    pe.trig = 1;
    chk(cu::launch_driver(ds.f_prepl, dim3(H, N, E), dim3(128), K1_SMEM,
                          ctx.stream, pe));
    chk(cu::launch_driver_pdl(ds.f_r54l, dim3(2 * H, N), dim3(128),
                              k2_smem(5, 0), ctx.stream, p));
    pe.c0 = E;
    pe.trig = 0;
    chk(cu::launch_pdl(kda_prep<true>, dim3(H, N, T / C + 2 - E), dim3(128),
                       K1_SMEM, ctx.stream, pe));
  } else {
    chk(cu::launch_driver(
        ds.f_prep, dim3(H, (unsigned)(N * ((ws_slots(a.dims) + N - 1) / N))),
        dim3(128), K1_SMEM, ctx.stream, p));
    // one CTA per SM when the CTAs fit in one wave, or in a few waves of long
    // sequences (a CTA sharing its SM with another runs its chain about 1.9x
    // slower); two per SM for many short sequences
    if (2 * H * N <= ctx.sm_count ||
        (T >= 400 * N && 2 * H * N <= 3 * ctx.sm_count))
      chk(cu::launch_pdl(kda_recur<5, 4, false, true>, dim3(2 * H, N),
                         dim3(128), k2_smem(5, 0), ctx.stream, p));
    else if (H <= 32)  // two per SM: output from registers, except with many
                       // heads (bandwidth-bound, TMA stores)
      chk(cu::launch_pdl(kda_recur<3, 1, false, true>, dim3(2 * H, N),
                         dim3(128), k2_smem(3, 0), ctx.stream, p));
    else
      chk(cu::launch_pdl(kda_recur<3, 1, false, false>, dim3(2 * H, N),
                         dim3(128), k2_smem(3, 1), ctx.stream, p));
  }
  return ok;
}

// ---- workspace layout of the op: [checkpoint-offset zeros, 4 N B][zeroed
// flags][scratch at a 256 B boundary] ----
constexpr size_t align_up(size_t x, size_t a) { return (x + a - 1) / a * a; }
struct Layout {
  size_t ckpt_bytes, zeroed_bytes, scratch_off, total;
};
Layout layout(const Dims& d) {
  size_t s = 0, z = 0;
  kernel_workspace(d, &s, &z);
  Layout l;
  l.ckpt_bytes = align_up(4 * (size_t)d.N, 16);
  l.zeroed_bytes = z;
  l.scratch_off = align_up(l.ckpt_bytes + z, 256);
  l.total = l.scratch_off + s;
  return l;
}

}  // namespace

// Workspace bytes for any call with at most T_total tokens and N sequences of H
// heads.
int64_t workspace_size(int64_t T_total, int64_t H, int64_t N) {
  // the size grows with T and N, except that H N <= 16 adds the 64-token slots:
  // also take the largest such N
  size_t total = layout(Dims{T_total, N, H}).total;
  const int64_t n64 = std::min<int64_t>(N, 16 / std::max<int64_t>(H, 1));
  if (n64 >= 1) total = std::max(total, layout(Dims{T_total, n64, H}).total);
  return static_cast<int64_t>(total);
}

}  // namespace kda_sm90
}  // namespace vllm

using torch::headeronly::ScalarType;
using torch::stable::Tensor;

namespace vllm {
namespace kda_sm90 {

// zero_workspace = false skips the memset of the zeroed region; only for a
// workspace that is kept zero between calls (the kernels leave it zero) and
// never shared with other users.
void forward(Tensor const& q, Tensor const& k, Tensor const& v, Tensor const& g,
             Tensor const& beta, double scale, Tensor& out, Tensor& workspace,
             Tensor const& A_log, Tensor const& dt_bias, double lower_bound,
             std::optional<Tensor> const& initial_state,
             std::optional<Tensor> const& final_state,
             std::optional<Tensor> const& cu_seqlens,
             std::optional<Tensor> const& checkpoint_state,
             std::optional<Tensor> const& checkpoint_offsets,
             bool zero_workspace) {
  constexpr int64_t D = 128;
  STD_TORCH_CHECK(
      initial_state.has_value() && final_state.has_value() &&
          cu_seqlens.has_value(),
      "kda_prefill_sm90 needs initial_state, final_state and cu_seqlens");
  STD_TORCH_CHECK(
      checkpoint_state.has_value() == checkpoint_offsets.has_value(),
      "checkpoint_state and checkpoint_offsets must be provided together");
  for (const Tensor* t :
       std::initializer_list<const Tensor*>{&q, &k, &v, &g, &beta, &out})
    STD_TORCH_CHECK(t->is_cuda() && t->scalar_type() == ScalarType::BFloat16,
                    "q, k, v, g, beta and out must be bfloat16 CUDA tensors");
  STD_TORCH_CHECK(q.dim() == 4 && q.size(0) == 1 && q.size(3) == D,
                  "q must be [1, T, H, 128]");
  const int64_t T = q.size(1), H = q.size(2);
  for (const Tensor* t : std::initializer_list<const Tensor*>{&k, &v, &g, &out})
    STD_TORCH_CHECK(t->dim() == 4 && t->size(0) == 1 && t->size(1) == T &&
                        t->size(2) == H && t->size(3) == D,
                    "k, v, g and out must match q's shape [1, T, H, 128]");
  STD_TORCH_CHECK(beta.dim() == 3 && beta.size(0) == 1 && beta.size(1) == T &&
                      beta.size(2) == H,
                  "beta must be [1, T, H]");
  STD_TORCH_CHECK(T > 0 && T < (int64_t(1) << 31) && H > 0,
                  "kda_prefill_sm90: bad T or H");

  const Tensor& cu = *cu_seqlens;
  STD_TORCH_CHECK(cu.is_cuda() && cu.scalar_type() == ScalarType::Int &&
                      cu.dim() == 1 && cu.is_contiguous() && cu.numel() >= 2,
                  "cu_seqlens must be a contiguous int32 CUDA tensor [N + 1]");
  const int64_t N = cu.numel() - 1;
  auto check_state = [&](const Tensor& s, const char* what) {
    STD_TORCH_CHECK(
        s.is_cuda() && s.scalar_type() == ScalarType::Float &&
            s.is_contiguous() && s.dim() == 4 && s.size(0) == N &&
            s.size(1) == H && s.size(2) == D && s.size(3) == D,
        what, " must be a contiguous float32 CUDA tensor [N, H, 128, 128]");
  };
  check_state(*initial_state, "initial_state");
  check_state(*final_state, "final_state");
  if (checkpoint_state.has_value()) {
    check_state(*checkpoint_state, "checkpoint_state");
    const Tensor& co = *checkpoint_offsets;
    STD_TORCH_CHECK(
        co.is_cuda() && co.scalar_type() == ScalarType::Int &&
            co.is_contiguous() && co.dim() == 1 && co.numel() == N,
        "checkpoint_offsets must be a contiguous int32 CUDA tensor [N]");
  }
  STD_TORCH_CHECK(A_log.is_cuda() && A_log.scalar_type() == ScalarType::Float &&
                      A_log.is_contiguous() && A_log.numel() == H,
                  "A_log must be a contiguous float32 CUDA tensor [H]");
  STD_TORCH_CHECK(dt_bias.is_cuda() &&
                      dt_bias.scalar_type() == ScalarType::Float &&
                      dt_bias.is_contiguous() && dt_bias.numel() == H * D,
                  "dt_bias must be a contiguous float32 CUDA tensor [H, 128]");

  // Token strides: rows of H * 128 contiguous elements, 16-byte aligned for TMA
  // and vector accesses.
  auto token_stride = [&](const Tensor& t, int64_t dense) {
    return T > 1 ? t.stride(1) : dense;
  };
  auto rows_ok = [&](const Tensor& t, int64_t ts, int64_t min_ts) {
    return t.stride(3) == 1 && (H == 1 || t.stride(2) == D) && ts >= min_ts &&
           ts % 8 == 0 &&
           reinterpret_cast<uintptr_t>(t.const_data_ptr()) % 16 == 0;
  };
  const int64_t qs = token_stride(q, 3 * H * D);
  STD_TORCH_CHECK(
      rows_ok(q, qs, 3 * H * D) && token_stride(k, qs) == qs &&
          token_stride(v, qs) == qs && k.stride(3) == 1 && v.stride(3) == 1 &&
          (H == 1 || (k.stride(2) == D && v.stride(2) == D)) &&
          static_cast<const char*>(k.const_data_ptr()) ==
              static_cast<const char*>(q.const_data_ptr()) + 2 * H * D &&
          static_cast<const char*>(v.const_data_ptr()) ==
              static_cast<const char*>(q.const_data_ptr()) + 4 * H * D,
      "kda_prefill_sm90: q, k, v must be the q|k|v sections of one packed "
      "buffer (k = q + H * 128, "
      "v = q + 2 * H * 128 elements, one token stride, a multiple of 8 and at "
      "least 3 * H * 128, "
      "contiguous H/D rows, 16-byte aligned)");
  const int64_t gs = token_stride(g, H * D), os = token_stride(out, H * D);
  STD_TORCH_CHECK(rows_ok(g, gs, H * D),
                  "g needs contiguous H/D rows, a token stride that is a "
                  "multiple of 8 and a "
                  "16-byte aligned start");
  STD_TORCH_CHECK(
      rows_ok(out, os, H * D),
      "out needs contiguous H/D rows, a token stride that is a multiple of 8 "
      "and a 16-byte aligned start");
  const int64_t bs = token_stride(beta, H);
  STD_TORCH_CHECK((H == 1 || beta.stride(2) == 1) && bs >= H,
                  "beta needs contiguous heads per token");

  const Dims d{T, N, H};
  const Layout lay = layout(d);
  STD_TORCH_CHECK(
      workspace.is_cuda() && workspace.scalar_type() == ScalarType::Byte &&
          workspace.is_contiguous() &&
          reinterpret_cast<uintptr_t>(workspace.const_data_ptr()) % 256 == 0,
      "workspace must be a contiguous, 256-byte aligned uint8 CUDA tensor");
  STD_TORCH_CHECK(static_cast<size_t>(workspace.numel()) >= lay.total,
                  "workspace too small: ", workspace.numel(), " < ",
                  static_cast<int64_t>(lay.total),
                  " bytes (kda_prefill_sm90_workspace_size)");

  const int device = q.get_device_index();
  torch::stable::accelerator::DeviceGuard const device_guard(device);
  const cudaDeviceProp* prop = get_device_prop();
  STD_TORCH_CHECK(prop->major == 9 && prop->minor == 0,
                  "kda_prefill_sm90 runs on SM90 only");
  cudaStream_t stream = get_current_cuda_stream(device);

  uint8_t* ws = static_cast<uint8_t*>(workspace.mutable_data_ptr());
  const bool has_ckpt = checkpoint_state.has_value();
  // The kernels need zeroed flags at entry (they leave them zero, but the
  // workspace is shared with other users) and, without checkpoints, an all-zero
  // checkpoint_offsets [N]; both sit at the start of the workspace: one memset.
  const size_t zero_begin = (zero_workspace && has_ckpt) ? lay.ckpt_bytes : 0;
  const size_t zero_end = zero_workspace ? lay.ckpt_bytes + lay.zeroed_bytes
                                         : (has_ckpt ? 0 : lay.ckpt_bytes);

  Args a;
  a.dims = d;
  a.qkv = static_cast<const __nv_bfloat16*>(q.const_data_ptr());
  a.g = static_cast<const __nv_bfloat16*>(g.const_data_ptr());
  a.beta = static_cast<const __nv_bfloat16*>(beta.const_data_ptr());
  a.A_log = static_cast<const float*>(A_log.const_data_ptr());
  a.dt_bias = static_cast<const float*>(dt_bias.const_data_ptr());
  a.initial_state = static_cast<const float*>(initial_state->const_data_ptr());
  a.cu_seqlens = static_cast<const int32_t*>(cu.const_data_ptr());
  a.checkpoint_offsets =
      has_ckpt
          ? static_cast<const int32_t*>(checkpoint_offsets->const_data_ptr())
          : reinterpret_cast<int32_t*>(ws);
  a.out = static_cast<__nv_bfloat16*>(out.mutable_data_ptr());
  a.final_state = static_cast<float*>(final_state->mutable_data_ptr());
  // with all offsets zero the kernels never write the checkpoint state
  a.checkpoint_state =
      has_ckpt ? static_cast<float*>(checkpoint_state->mutable_data_ptr())
               : nullptr;
  a.qs = qs;
  a.gs = gs;
  a.bs = bs;
  a.os = os;
  a.scale = static_cast<float>(scale);
  a.lower_bound = static_cast<float>(lower_bound);
  Ctx ctx;
  ctx.stream = stream;
  ctx.device = device;
  ctx.sm_count = prop->multiProcessorCount;
  ctx.zeroed = ws + lay.ckpt_bytes;
  ctx.scratch = ws + lay.scratch_off;
  ctx.memset_ptr = ws + zero_begin;
  ctx.memset_bytes = zero_end - zero_begin;
  STD_TORCH_CHECK(kernel_run(a, ctx),
                  "kda_prefill_sm90: kernel launch failed: ",
                  cudaGetErrorString(cudaGetLastError()));
}

}  // namespace kda_sm90
}  // namespace vllm

void kda_prefill_sm90(Tensor const& q, Tensor const& k, Tensor const& v,
                      Tensor const& g, Tensor const& beta, double scale,
                      Tensor& out, Tensor& workspace, Tensor const& A_log,
                      Tensor const& dt_bias, double lower_bound,
                      std::optional<Tensor> initial_state,
                      std::optional<Tensor> final_state,
                      std::optional<Tensor> cu_seqlens,
                      std::optional<Tensor> checkpoint_state,
                      std::optional<Tensor> checkpoint_offsets) {
  vllm::kda_sm90::forward(q, k, v, g, beta, scale, out, workspace, A_log,
                          dt_bias, lower_bound, initial_state, final_state,
                          cu_seqlens, checkpoint_state, checkpoint_offsets,
                          /*zero_workspace=*/true);
}

int64_t kda_prefill_sm90_workspace_size(int64_t T_total, int64_t H, int64_t N) {
  return vllm::kda_sm90::workspace_size(T_total, H, N);
}
