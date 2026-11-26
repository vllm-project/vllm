// KDA ReplaySSM verify (T = 6, T = 9, q_n tokens per request) step for one
// B200 rank, vLLM layouts.
//
// Request n owns the packed token rows b_n .. b_n + q_n - 1 (b_n =
// query_start_loc[n], q_n = query_start_loc[n+1] - b_n) of the token args [M,
// ...] and of out [MO, H, D]. The kernels are templated on the capacity T (6,
// 9) and mask the tokens q_n .. T - 1 (zero inputs, never stored). A request
// with q_n = 0 or slot 0 (padded) touches no cache. The out rows from
// query_start_loc[N] (the tail of the batch) to MO are zeroed: CTA (h, n) takes
// those of head h at n, n + N, ... past the tail. The ring has R = 32 rows per
// slot and head, at slot ring_slot[n] (each ring's slots stride0 apart); a
// request flushes when c + q_n > L.
//
// Work unit = (head h, request n), one CTA of 256 threads per unit. Routing by
// T and the unit count H N:
// - T = 6 up to FZ_UNITS units: step_kernel<6, FZ> (one CTA per SM), the
//   gate and output-gate projections in the CTA (bf16 mma), the head's weights
//   by bulk copy into shared memory;
// - T = 6 above FZ_UNITS, every T = 9: proj_kernel, then step_kernel<T, !FZ>
//   (two CTAs per SM) launched with PDL.
// Any other T launches an invalid configuration, so the call fails.
// Per unit: the conv history and token rows, ring records and activations by
// cp.async, then the checkpoint by one 64 KB bulk copy; conv + SiLU, q / k
// norms, gates, the ring replay, the q tokens, gated RMSNorm, ring appends,
// flush.
//
// Algebra per unit (S0 the checkpoint [V, K]; records j < c with cumulative log
// decay G_j, total G_c, E = e^{G_c}; tokens t with cumulative log decay
// Gcum_t):
//   S_c = S0 E + sum_j u_j (k_j e^{G_c - G_j})^T
//       f16 mma of the records onto S0 E (fp32 accumulate)
//   Y_2t = khat_t e^{Gcum_t}, Y_2t+1 = qhat_t e^{Gcum_t};  Q = S_c Y
//       TF32 mma over the 128 state columns
//   C_im = (khat_i e^{-Gcum_i}) . Y_m
//       token-token coefficients (TF32 mma)
//   u_t = beta_t (v_t - Q_2t - sum_{i<t} u_i C_i,2t)
//   o_t = scale (Q_2t+1 + sum_{i<=t} u_i C_i,2t+1)
// The stored checkpoint on flush is S_c (dense rows: staged in the checkpoint
// buffer, one bulk store).
//
// step_kernel: warp w holds state rows 16w .. 16w+15 in registers, in a layout
// that is at once the accumulator fragment of the record product and the A
// fragment of Q (the K order permuted within 16-column groups), so the
// checkpoint is read from shared memory and written back to global memory with
// 16-byte accesses.
#include "kda_replayssm.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "kda_replayssm_utils.cuh"

namespace {

constexpr int D = 128;
constexpr int L = 16;  // ring window
constexpr int R = 32;  // ring rows per slot and head (every family)
constexpr int TMAX = 9;
static_assert(L + TMAX <= R, "ring too short for c + q appends");
constexpr int NTHR = 256;
constexpr float LOG2E = 1.4426950408889634f;
// up to this many units (one CTA per SM) the fused weights come by bulk copy
constexpr int FZ_UNITS = 148;

__device__ __forceinline__ float fexp(float x) {
  return ptx::ex2_approx(x * LOG2E);
}
__device__ __forceinline__ float bfr(float x) {
  return __bfloat162float(__float2bfloat16_rn(x));
}
// TF32 operand rounded to nearest (ties away from zero): the mma ignores the
// low 13 bits.
__device__ __forceinline__ uint32_t tf(float x) {
  return __float_as_uint(x) + 0x1000u;
}
__device__ __forceinline__ uint32_t sa(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
// Two floats as an f16x2 operand, lo in the low half (round to nearest).
__device__ __forceinline__ uint32_t h2(float lo, float hi) {
  const __half2 v = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&v);
}
__device__ __forceinline__ float sigm(float x) {
  return __fdividef(1.f, 1.f + fexp(-x));
}
__device__ __forceinline__ float4 ld4(const float* p) {
  return *reinterpret_cast<const float4*>(p);
}
__device__ __forceinline__ float f(uint32_t x) { return __uint_as_float(x); }
// Request n's token rows and the batch total from query_start_loc; q clamped to
// [0, T].
struct Toks {
  int b, q, total;
};
__device__ __forceinline__ Toks req_tokens(const int32_t* qsl, int n, int N,
                                           int T) {
  Toks t;
  t.b = static_cast<int>(ptx::ld_global_u32(qsl + n));
  const int e = static_cast<int>(ptx::ld_global_u32(qsl + n + 1));
  t.total = static_cast<int>(ptx::ld_global_u32(qsl + N));
  t.q = min(max(e - t.b, 0), T);
  return t;
}

// The out rows total + n, total + n + N, ... below MO of head h are zero (16
// threads per row).
__device__ __forceinline__ void zero_tail(__nv_bfloat16* out, int n, int N,
                                          int h, int H, int total, int64_t MO,
                                          int tid) {
  for (int64_t r = total + n + static_cast<int64_t>(N) * (tid / 16); r < MO;
       r += static_cast<int64_t>(N) * (NTHR / 16))
    *reinterpret_cast<uint4*>(out + (r * H + h) * D + 8 * (tid % 16)) =
        make_uint4(0, 0, 0, 0);
}

// One step of a transposing warp reduction on each of NB blocks of 32 values:
// lane bit O picks which half of 2 O values of a block it keeps; after the
// steps O = 16 .. 1, lane l holds the warp sum of value 32 b + l in p[32 b]
// (values past the blocks untouched).
template <int O, int N, int NB = 1>
__device__ __forceinline__ void tred(float (&p)[N], int lane) {
  static_assert(N >= 32 * (NB - 1) + 2 * O, "tred: too few values");
  const bool up = lane & O;
#pragma unroll
  for (int b = 0; b < NB; ++b)
#pragma unroll
    for (int i = 0; i < O; ++i) {
      const float send = up ? p[32 * b + i] : p[32 * b + i + O];
      const float keep = up ? p[32 * b + i + O] : p[32 * b + i];
      p[32 * b + i] = keep + __shfl_xor_sync(0xffffffffu, send, O);
    }
}

// Warp sums of P values (P a power of two, at most 32): afterwards lane l holds
// the sum of value l % P.
template <int P>
__device__ __forceinline__ float wsums(float (&p)[P], int lane) {
  if constexpr (P > 16) tred<16>(p, lane);
  if constexpr (P > 8) tred<8>(p, lane);
  if constexpr (P > 4) tred<4>(p, lane);
  if constexpr (P > 2) tred<2>(p, lane);
  if constexpr (P > 1) tred<1>(p, lane);
#pragma unroll
  for (int off = P; off < 32; off <<= 1)
    p[0] += __shfl_xor_sync(0xffffffffu, p[0], off);
  return p[0];
}
constexpr int pow2_at_least(int n) {
  return n <= 1 ? 1 : 2 * pow2_at_least((n + 1) / 2);
}
// Row of column r in the record-key table: rows r and r ^ 4 of a 16-row group
// move apart by two rows of 32 bytes, so the B-fragment reads of rows {0, 1, 4,
// 5} (one half-warp) fall on distinct banks. Not in step_kernel<6, false>:
// there the extra addressing costs 24 bytes of stack.
template <bool SW>
__device__ __forceinline__ int ks(int r) {
  return SW ? r ^ (((r >> 2) & 1) << 1) : r;
}

// ------------------------------------------------------------------------------------------------
// projections

// The low-rank projections: out[t][ch] = sum_k a[t][k] w[ch][k] with bf16
// m16n8k16 mma (fp32 accumulate), 32 channels ch0 .. ch0 + 31 per warp. K is
// permuted within each 32-column chunk so that one 16-byte load gives a thread
// the B fragments of two k-steps (logical k 2tq, 2tq + 1, 2tq + 8, 2tq + 9 of
// the first step and 16 + those of the second at physical 8tq .. 8tq + 7), and
// one 16-byte shared load per row the matching A fragments. row stride of the
// activation rows in shared memory (16-byte loads without conflicts)
constexpr int AS = D + 32;

__device__ __forceinline__ void proj_load(uint4 (&b)[4][4],
                                          const __nv_bfloat16* w, int g,
                                          int tq) {
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int kp = 0; kp < 4; ++kp)
      b[j][kp] = ptx::ld_global_v4(w + (8 * j + g) * D + 32 * kp + 8 * tq);
}

// A rows g (and g + 8 when HI) of a [rows, AS] bf16 tile in shared memory; rows
// past `rows` are zero (only g is checked).
template <bool HI>
__device__ __forceinline__ void proj_mma(float (&acc)[4][4],
                                         const uint4 (&b)[4][4],
                                         const __nv_bfloat16* a, int rows,
                                         int g, int tq) {
#pragma unroll
  for (int kp = 0; kp < 4; ++kp) {
    const uint4 lo = g < rows
                         ? *reinterpret_cast<const uint4*>(
                               a + (g < rows ? g : 0) * AS + 32 * kp + 8 * tq)
                         : make_uint4(0, 0, 0, 0);
    const uint4 hi = HI ? *reinterpret_cast<const uint4*>(a + (g + 8) * AS +
                                                          32 * kp + 8 * tq)
                        : make_uint4(0, 0, 0, 0);
    const uint32_t a0[4] = {lo.x, hi.x, lo.y, hi.y},
                   a1[4] = {lo.z, hi.z, lo.w, hi.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      ptx::mma_m16n8k16_bf16(acc[j], a0, b[j][kp].x, b[j][kp].y);
      ptx::mma_m16n8k16_bf16(acc[j], a1, b[j][kp].z, b[j][kp].w);
    }
  }
}

// The same fragments from a [rows, D] bf16 copy of the weights in shared
// memory.
__device__ __forceinline__ void proj_lds(uint4 (&b)[4][4],
                                         const __nv_bfloat16* w, int g,
                                         int tq) {
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int kp = 0; kp < 4; ++kp)
      b[j][kp] = *reinterpret_cast<const uint4*>(w + (8 * j + g) * D + 32 * kp +
                                                 8 * tq);
}

struct ProjArgs {
  const __nv_bfloat16* act[2];  // f_a, g_a: [M, D] views
  int64_t s0[2];
  const __nv_bfloat16* wt[2];  // f_b_weight, g_b_weight [H, D, D]
  // [2][M][H*D]: raw gate (bf16-rounded), output-gate logits
  float* out;
  // query_start_loc: rows at or past qsl[N] are not computed
  const int32_t* qsl;
  int M, HD, N;
  int round_z;  // round the output-gate logits to bf16
};

// Grid (H, 2, ceil(M / PR)), 4 warps: warp w computes channels 32w .. of head
// blockIdx.x of projection blockIdx.y (0: gate, 1: output gate) for up to PR
// tokens, the activation rows staged in shared memory.
constexpr int PR = 32;
__global__ void __launch_bounds__(128)
    proj_kernel(const __grid_constant__ ProjArgs p) {
  ptx::griddep_launch_dependents();
  __shared__ alignas(16) __nv_bfloat16 at[PR][AS];
  const int h = blockIdx.x, pj = blockIdx.y, rbase = blockIdx.z * PR;
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, g = lane >> 2,
            tq = lane & 3;
  // the activation rows of the buffer (rows past the batch total are computed,
  // never stored) and the weights are loaded before the total arrives
  const int nrows_q = static_cast<int>(ptx::ld_global_u32(p.qsl + p.N));
  const __nv_bfloat16* act = p.act[0];
  int64_t s0 = p.s0[0];
  if (pj) act = p.act[1], s0 = p.s0[1];
#pragma unroll
  for (int i = 0; i < PR / 8; ++i) {
    const int idx = tid + 128 * i, r = idx >> 4, ch = idx & 15, row = rbase + r;
    if (row < p.M)
      ptx::cp_async_16(sa(&at[r][8 * ch]), act + row * s0 + 8 * ch);
    else
      *reinterpret_cast<uint4*>(&at[r][8 * ch]) = make_uint4(0, 0, 0, 0);
  }
  ptx::cp_async_commit();
  uint4 b[4][4];
  proj_load(
      b, (pj ? p.wt[1] : p.wt[0]) + (static_cast<int64_t>(h) * D + 32 * w) * D,
      g, tq);
  const int nrows = min(p.M, nrows_q);  // packed rows in use
  ptx::cp_async_wait<0>();
  if (rbase >= nrows) return;
  __syncthreads();
  const bool rnd = pj == 0 || p.round_z;
  float* out = p.out + static_cast<int64_t>(pj) * p.M * p.HD + h * D + 32 * w;
#pragma unroll
  for (int mt = 0; mt < PR / 16; ++mt) {
    const int rb = rbase + 16 * mt;
    if (rb < nrows) {
      float acc[4][4] = {};
      proj_mma<true>(acc, b, &at[16 * mt][0], 16, g, tq);
      const int ra = rb + g, rc = ra + 8;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float v[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) v[e] = rnd ? bfr(acc[j][e]) : acc[j][e];
        if (ra < nrows)
          *reinterpret_cast<float2*>(out + static_cast<int64_t>(ra) * p.HD +
                                     8 * j + 2 * tq) = {v[0], v[1]};
        if (rc < nrows)
          *reinterpret_cast<float2*>(out + static_cast<int64_t>(rc) * p.HD +
                                     8 * j + 2 * tq) = {v[2], v[3]};
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------
// step

struct StepArgs {
  kda_replayssm::Args a;
  // [2][M][H*D] gate and output-gate logits from proj_kernel (FZ false)
  const float* gz;
  // flashinfer / triton: conv output and recurrent output rounded to bf16
  int rounded;
  int round_z;  // output-gate logits rounded to bf16 (all but split_gate)
};

template <int T>
// conv history, the T new conv rows and the ring records (read until the
// records pass)
struct PreBuf {
  __nv_bfloat16 hist[3][3][D];
  __nv_bfloat16 xw[T][3][D];  // rows q .. T - 1 zero
  float rec_g[L][D];
  __nv_bfloat16 rec_k[L][D];
};
template <int T>
// Y, the token keys kt and the token-token coefficients C = kt^T Y, from the
// token phase on
struct TokBuf {
  // 2T targets in n-tiles of 8: 16, 24 at T 6, 9
  static constexpr int NB = (2 * T + 7) / 8 * 8;
  float y[NB][D + 16];
  // kt_i = khat_i e^{-Gcum_i} (fp32; the m16 tile's rows past T are zero
  // registers)
  alignas(16) float kt[T][D];
  // C[i][m] = kt_i . Y_m (used: C[i][2t] for i < t, C[i][2t + 1] for i <= t)
  alignas(16) float cs[T][NB];
};
template <int T>
union PreTok {  // the inputs until the records pass, then the token keys and Y
  PreBuf<T> pre;
  TokBuf<T> tok;
};

template <int T, bool FZ>
struct Smem {
  static_assert(!FZ || T <= 8,
                "FZ: the projections take the token rows as the low half of "
                "one m16 tile");
  // activation rows held (FZ: the projections run in the step CTA)
  static constexpr int TA = FZ ? T : 1;
  static constexpr int M = 2 * T;           // targets
  static constexpr int NB = TokBuf<T>::NB;  // Y rows (n-tiles of 8)
  // rows of rec_u: the records' u, then Q transposed (NB rows)
  static constexpr int QR = NB > L ? NB : L;
  static constexpr int YS = D + 16;  // Y row stride
  alignas(16) float sck[D][D];       // checkpoint rows
  alignas(16) PreTok<T> B;

  // u of the records; after a warp's record product, Q[v][m] of its rows v at
  // [m][v]
  alignas(16) float rec_u[QR][D + 4];
  // record keys k_j e^{G_c - G_j}, per column, j at 4 ((j % 8) / 2) + j % 2 + 2
  // (j / 8)
  alignas(16) __half kpt[D][L];
  alignas(16) __nv_bfloat16 fa[TA][AS];  // f_a rows (rows q .. zero)
  alignas(16) __nv_bfloat16 ga[TA][AS];  // g_a rows (rows q .. zero)
  alignas(16) float g1s[T][D];  // raw gate bf16(f_a f_b^T) of this head
  // output-gate logits g_a g_b^T of this head (bf16-rounded except on
  // split_gate)
  alignas(16) float zs[T][D];
  alignas(16) float ef[D];
  // squared norms of q, k per token: four warp partials
  alignas(16) float nsq[2][T][4];
  alignas(16) float ssq[T][4];
  alignas(8) uint64_t bar;   // checkpoint bulk copy
  alignas(8) uint64_t wbar;  // FZ: the head's projection weights
};

template <class S>
constexpr size_t wts_offset() {
  return (sizeof(S) + 127) / 128 * 128;
}

struct Unit {
  int n, h, slot, rslot, acc, r0, c;  // rslot: the slot of the unit's rings
  // first packed token row, token count (0 for a padded request), batch total
  // query_start_loc[N]
  int b, q, total;
};

// One CTA per unit (head blockIdx.x, request blockIdx.y), T the token capacity;
// the unit's q <= T tokens.
template <int T, bool FZ>
__global__ void __launch_bounds__(NTHR, FZ ? 1 : 2)
    step_kernel(const __grid_constant__ StepArgs P) {
  using S = Smem<T, FZ>;
  constexpr int M = S::M, NB = S::NB, NT8 = NB / 8, TA = S::TA;
  constexpr bool KS = FZ;  // record-key row swizzle
  extern __shared__ __align__(16) unsigned char smem_raw[];
  S& s = *reinterpret_cast<S*>(smem_raw);
  const kda_replayssm::Args& a = P.a;
  const int H = static_cast<int>(a.dims.H), N = static_cast<int>(a.dims.N),
            HD = H * D;
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, g = lane >> 2,
            tq = lane & 3;
  // the channel / column / value row of a thread of the first half
  const int d = tid & (D - 1);
  auto scalars = [&]() {
    Unit x;
    x.h = static_cast<int>(blockIdx.x);
    x.n = static_cast<int>(blockIdx.y);
    x.slot = static_cast<int>(ptx::ld_global_u32(a.slot + x.n));
    x.rslot = static_cast<int>(ptx::ld_global_u32(a.ring_slot + x.n));
    x.acc = static_cast<int>(ptx::ld_global_u32(a.num_accepted + x.n));
    x.r0 = static_cast<int>(ptx::ld_global_u32(a.ring_start + x.n));
    x.c = static_cast<int>(ptx::ld_global_u32(a.num_committed + x.n));
    const Toks tk = req_tokens(a.query_start_loc, x.n, N, T);
    x.b = tk.b, x.q = tk.q, x.total = tk.total;
    return x;
  };
  // The small inputs of a live unit x (one cp.async group): token rows, conv
  // history, ring records, the activations (FZ); token and activation rows q ..
  // are zeroed, u rows past c are zeroed (A operand rows of the record
  // product).
  auto issue_small = [&](const Unit& x) {
    {
      const __nv_bfloat16* xs = a.x + x.h * a.x_stride[2];
#pragma unroll
      for (int i0 = 0; i0 < T * 48; i0 += NTHR) {
        const int i = i0 + tid;
        if (i0 + NTHR <= T * 48 || i < T * 48) {
          const int t = i / 48, p = (i / 16) % 3, ch = i % 16;
          if (t < x.q)
            ptx::cp_async_16(
                sa(&s.B.pre.xw[t][p][8 * ch]),
                xs + (x.b + t) * a.x_stride[0] + p * a.x_stride[1] + 8 * ch);
          else
            *reinterpret_cast<uint4*>(&s.B.pre.xw[t][p][8 * ch]) =
                make_uint4(0, 0, 0, 0);
        }
      }
      if (FZ)  // g_a rows (first 16 TA entries), f_a rows (next 16 TA)
        for (int i = tid; i < 32 * TA; i += NTHR) {
          const int pj = i / (16 * TA), t = (i / 16) % TA, ch = i & 15;
          __nv_bfloat16* dst = pj ? &s.fa[t][8 * ch] : &s.ga[t][8 * ch];
          if (t < x.q)
            ptx::cp_async_16(sa(dst),
                             (pj ? a.f_a + (x.b + t) * a.f_a_stride[0]
                                 : a.g_a + (x.b + t) * a.g_a_stride[0]) +
                                 8 * ch);
          else
            *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
        }
    }
    {
      const __nv_bfloat16* cs = a.conv_state + x.slot * a.conv_state_stride[0] +
                                x.h * a.conv_state_stride[3];
      if (tid < 9 * 16) {
        const int r = tid / 48, p = (tid / 16) % 3, ch = tid % 16;
        ptx::cp_async_16(sa(&s.B.pre.hist[r][p][8 * ch]),
                         cs + (x.acc - 1 + r) * a.conv_state_stride[1] +
                             p * a.conv_state_stride[2] + 8 * ch);
      }
      const int64_t rbase = static_cast<int64_t>(x.h) * R;
      const int e = tid & 15;
      for (int j = tid >> 4; j < x.c; j += NTHR / 16) {
        const int64_t row = (rbase + ((x.r0 + j) & (R - 1))) * D;
        ptx::cp_async_16(sa(&s.B.pre.rec_g[j][4 * e]),
                         a.g_ring + x.rslot * a.g_ring_stride0 + row + 4 * e);
        ptx::cp_async_16(
            sa(&s.B.pre.rec_g[j][64 + 4 * e]),
            a.g_ring + x.rslot * a.g_ring_stride0 + row + 64 + 4 * e);
        ptx::cp_async_16(sa(&s.B.pre.rec_k[j][8 * e]),
                         a.k_ring + x.rslot * a.k_ring_stride0 + row + 8 * e);
      }
      ptx::cp_async_commit();
      // the records' u in a second group: read only by the record product
      for (int j = tid >> 4; j < x.c; j += NTHR / 16) {
        const int64_t row = (rbase + ((x.r0 + j) & (R - 1))) * D;
        ptx::cp_async_16(sa(&s.rec_u[j][4 * e]),
                         a.u_ring + x.rslot * a.u_ring_stride0 + row + 4 * e);
        ptx::cp_async_16(
            sa(&s.rec_u[j][64 + 4 * e]),
            a.u_ring + x.rslot * a.u_ring_stride0 + row + 64 + 4 * e);
      }
      for (int i = tid; i < (L - x.c) * (D / 4); i += NTHR)
        *reinterpret_cast<float4*>(
            &s.rec_u[x.c + i / (D / 4)][4 * (i % (D / 4))]) =
            make_float4(0.f, 0.f, 0.f, 0.f);
    }
    ptx::cp_async_commit();
  };
  // The checkpoint of unit x (64 KB, contiguous) in one bulk copy.
  auto issue_ck = [&](const Unit& x) {
    if (tid == 0) {
      ptx::mbarrier_arrive_expect_tx(sa(&s.bar), D * D * 4);
      ptx::bulk_copy_g2s(sa(&s.sck[0][0]),
                         a.checkpoint + x.slot * a.checkpoint_stride[0] +
                             x.h * a.checkpoint_stride[1],
                         D * D * 4, sa(&s.bar));
    }
  };
  // Per-head inputs in registers: conv weights of channel d (q, k in the first
  // half, v in the second), dt_bias of channel d (first half), A_log.
  float wcr[2][4], dtb = 0.f, a_log = 0.f;
  auto head_regs = [&](int hh) {
#pragma unroll
    for (int p = 0; p < 2; ++p)
#pragma unroll
      for (int j = 0; j < 4; ++j)
        wcr[p][j] = __uint_as_float(ptx::ld_global_u32(
            a.conv_weight + (((tid < D ? p : 2) * 4 + j) * H + hh) * D + d));
    if (tid < D)
      dtb = __uint_as_float(ptx::ld_global_u32(a.dt_bias + hh * D + d));
    a_log = __uint_as_float(ptx::ld_global_u32(a.A_log + hh));
  };
  // Projection outputs of channels c0 .. c0 + 31 of the token rows g < T into
  // dst [T][D].
  auto store_proj = [&](float (*dst)[D], const float (&acc)[4][4], int c0,
                        bool rnd) {
#pragma unroll
    for (int j = 0; j < 4; ++j)
      if (g < T)
        *reinterpret_cast<float2*>(&dst[g < T ? g : 0][c0 + 8 * j + 2 * tq]) = {
            rnd ? bfr(acc[j][0]) : acc[j][0], rnd ? bfr(acc[j][1]) : acc[j][1]};
  };

  // Thread 0 initializes the barriers before its scalar loads (their fence
  // would wait for the loads), then issues the weight copies while the scalars
  // are in flight.
  if (tid == 0) {
    ptx::mbarrier_init(sa(&s.bar), 1);
    if (FZ) ptx::mbarrier_init(sa(&s.wbar), 1);
    ptx::fence_mbarrier_init();
  }
  const Unit cur = scalars();
  // FZ (one CTA per SM): the head's g_b and f_b weights by two bulk copies into
  // the shared memory after S
  const __nv_bfloat16* wts =
      reinterpret_cast<const __nv_bfloat16*>(smem_raw + wts_offset<S>());
  if (FZ && tid == 0) {
    const int64_t hw = static_cast<int64_t>(blockIdx.x) * D * D;
    ptx::mbarrier_arrive_expect_tx(sa(&s.wbar), 2 * D * D * 2);
    ptx::bulk_copy_g2s(sa(wts), a.g_b_weight + hw, D * D * 2, sa(&s.wbar));
    ptx::bulk_copy_g2s(sa(wts + D * D), a.f_b_weight + hw, D * D * 2,
                       sa(&s.wbar));
  }
  zero_tail(a.out, cur.n, N, cur.h, H, cur.total, a.dims.MO, tid);
  // padded request: no cache touched, no output row of its own
  if (cur.slot <= 0 || cur.q <= 0) {
    if (tid == 0) {
      // the weight copy lands before the CTA exits
      if (FZ) ptx::mbarrier_wait_parity(sa(&s.wbar), 0);
    }
    return;
  }
  // before the per-head loads: the small rows are on the critical path
  issue_small(cur);
  const float nwv =
      tid >= D ? __uint_as_float(ptx::ld_global_u32(a.norm_weight + d)) : 0.f;
  head_regs(cur.h);
  uint4 pw[4][4];
  // FZ: the gate fragments of the second half from the bulk-copied f_b
  auto weights_from_smem = [&]() {
    ptx::mbarrier_wait_parity(sa(&s.wbar), 0);
    proj_lds(pw, wts + D * D + 32 * (w & 3) * D, g, tq);
  };

  {
    const int h = cur.h, slot = cur.slot, rslot = cur.rslot, c = cur.c,
              r0 = cur.r0, b = cur.b, q = cur.q;
    // beta logits of the tokens (second half); 0 past q
    float bt[T];
    if (tid >= D)
#pragma unroll
      for (int t = 0; t < T; ++t)
        bt[t] = t < q
                    ? __bfloat162float(__ushort_as_bfloat16(ptx::ld_global_u16(
                          a.beta + (b + t) * a.beta_stride[0] + h)))
                    : 0.f;
    ptx::cp_async_wait<1>();  // all but the records' u
    __syncthreads();
    issue_ck(cur);  // the checkpoint after the small inputs have landed

    {
      // flush when the committed records and the new tokens overflow the
      // window
      const bool flush = c + q > L;
      float* ckp = a.checkpoint + slot * a.checkpoint_stride[0] +
                   h * a.checkpoint_stride[1];
      const int64_t ring_base = static_cast<int64_t>(h) * R;

      // conv + SiLU of channel d over the window of part p (q, k: first half;
      // v: second half), with its new conv rows (history rows 1, 2, then the q
      // tokens; rows q + 2 .. kept)
      auto conv_part = [&](int p, const float (&wc)[4], float (&y)[T]) {
        float hf[T + 3];
#pragma unroll
        for (int i = 0; i < 3; ++i)
          hf[i] = __bfloat162float(s.B.pre.hist[i][p][d]);
#pragma unroll
        for (int t = 0; t < T; ++t)
          hf[3 + t] = __bfloat162float(s.B.pre.xw[t][p][d]);
#pragma unroll
        for (int t = 0; t < T; ++t) {
          const float pre =
              ((wc[0] * hf[t] + wc[1] * hf[t + 1]) + wc[2] * hf[t + 2]) +
              wc[3] * hf[t + 3];
          const float v = __fdividef(pre, 1.f + fexp(-pre));
          y[t] = P.rounded ? bfr(v) : v;
        }
        __nv_bfloat16* cs = a.conv_state + slot * a.conv_state_stride[0] +
                            p * a.conv_state_stride[2] +
                            h * a.conv_state_stride[3] + d;
        cs[0] = __float2bfloat16_rn(hf[1]);
        cs[a.conv_state_stride[1]] = __float2bfloat16_rn(hf[2]);
#pragma unroll
        for (int t = 0; t < T; ++t)
          if (t < q)
            cs[(t + 2) * a.conv_state_stride[1]] =
                __float2bfloat16_rn(hf[3 + t]);
      };
      // conv outputs of channel d: q, k (first half), v (second half)
      float qr[T], kr[T], vr[T];
      if (tid < D) {  // q and k; their squared norms
        conv_part(0, wcr[0], qr);
        conv_part(1, wcr[1], kr);
        constexpr int PS = pow2_at_least(2 * T);
        // squared q (0 .. T - 1) and k (T .. 2T - 1) of the tokens
        float ss[PS] = {};
#pragma unroll
        for (int t = 0; t < T; ++t) {
          ss[t] = qr[t] * qr[t];
          ss[T + t] = kr[t] * kr[t];
        }
        const float sw = wsums(ss, lane);
        if (lane < 2 * T) s.nsq[lane / T][lane % T][w] = sw;
        // v of value row d; the raw gate g1 = bf16(f_a f_b^T) of channels 32 (w
        // - 4) ..; the records of column d
      } else {
        if (FZ) {
          float ga[4][4] = {};
          weights_from_smem();
          proj_mma<false>(ga, pw, &s.fa[0][0], TA, g, tq);
          store_proj(s.g1s, ga, 32 * (w - 4), true);
          // gate and output-gate rows of this head from proj_kernel (rows q ..
          // zero)
        } else {
          ptx::griddep_wait();
          for (int i = d; i < 2 * T * 32; i += D) {
            const int pj = i / (32 * T), t = (i / 32) % T, ch = i % 32;
            float* dst = (pj ? &s.zs[t][0] : &s.g1s[t][0]) + 4 * ch;
            if (t < q)
              ptx::cp_async_16(sa(dst), P.gz + (pj * a.dims.M + b + t) * HD +
                                            h * D + 4 * ch);
            else
              *reinterpret_cast<float4*>(dst) = make_float4(0.f, 0.f, 0.f, 0.f);
          }
          ptx::cp_async_commit();
        }
        conv_part(2, wcr[0], vr);
        float suf = 0.f;  // sum of the gates of the records after j
        float kw[L];
#pragma unroll
        for (int j = L - 1; j >= 0; --j) {
          const float gj = j < c ? s.B.pre.rec_g[j][d] : 0.f;
          const float kj = j < c ? __bfloat162float(s.B.pre.rec_k[j][d]) : 0.f;
          kw[j] = kj * fexp(suf);
          suf += gj;
        }
        // positions 4tq' .. 4tq' + 3 hold j = 2tq', 2tq' + 1, 2tq' + 8, 2tq' +
        // 9
        uint32_t kp[8];
#pragma unroll
        for (int t4 = 0; t4 < 4; ++t4) {
          kp[2 * t4] = h2(kw[2 * t4], kw[2 * t4 + 1]);
          kp[2 * t4 + 1] = h2(kw[2 * t4 + 8], kw[2 * t4 + 9]);
        }
        *reinterpret_cast<uint4*>(&s.kpt[ks<KS>(d)][0]) =
            make_uint4(kp[0], kp[1], kp[2], kp[3]);
        *reinterpret_cast<uint4*>(&s.kpt[ks<KS>(d)][8]) =
            make_uint4(kp[4], kp[5], kp[6], kp[7]);
        s.ef[d] = fexp(suf);
        if (!FZ) ptx::cp_async_wait<0>();
      }
      __syncthreads();

      // tokens, column d: gates, Y, token keys, ring appends of k and g (tokens
      // past q: Y rows zero)
      if (tid < D) {
        const float ea = expf(a_log), lb = a.lower_bound;
        float gk[T], gcum[T], khat[T], qh[T];
#pragma unroll
        for (int t = 0; t < T; ++t) {
          const float x = ea * (s.g1s[t][d] + dtb);
          gk[t] = __fdividef(lb, 1.f + fexp(-x));
          khat[t] = kr[t] * rsqrtf(s.nsq[1][t][0] + s.nsq[1][t][1] +
                                   s.nsq[1][t][2] + s.nsq[1][t][3] + 1e-6f);
          qh[t] = qr[t] * rsqrtf(s.nsq[0][t][0] + s.nsq[0][t][1] +
                                 s.nsq[0][t][2] + s.nsq[0][t][3] + 1e-6f);
        }
        gcum[0] = gk[0];
#pragma unroll
        for (int t = 1; t < T; ++t) gcum[t] = gcum[t - 1] + gk[t];
#pragma unroll
        for (int t = 0; t < T; ++t) {
          const float eg = t < q ? fexp(gcum[t]) : 0.f;
          const float yk = khat[t] * eg, yq = qh[t] * eg;
          s.B.tok.y[2 * t][d] = yk;
          s.B.tok.y[2 * t + 1][d] = yq;
          s.B.tok.kt[t][d] = khat[t] * fexp(-gcum[t]);
          if (t < q) {
            const int64_t row = (ring_base + ((r0 + c + t) & (R - 1))) * D + d;
            a.k_ring[rslot * a.k_ring_stride0 + row] =
                __float2bfloat16_rn(khat[t]);
            a.g_ring[rslot * a.g_ring_stride0 + row] = gk[t];
          }
        }
      } else {
#pragma unroll
        // in place: sigmoid(z) (FZ: z still being written) and sigmoid(beta)
        for (int t = 0; t < T; ++t) {
          if (!FZ) s.zs[t][d] = sigm(s.zs[t][d]);
          bt[t] = __fdividef(1.f, 1.f + fexp(-bt[t]));
        }
        if (NB > M)  // Y rows past the targets
          for (int i = d; i < (NB - M) * (S::YS / 4); i += D)
            *reinterpret_cast<float4*>(
                &s.B.tok.y[M + i / (S::YS / 4)][4 * (i % (S::YS / 4))]) =
                make_float4(0.f, 0.f, 0.f, 0.f);
        // output-gate logits of channels 32 (w - 4) .. from the bulk-copied g_b
        // (needed only by the norm)
        if (FZ) {
          proj_lds(pw, wts + 32 * (w - 4) * D, g, tq);
          float za[4][4] = {};
          proj_mma<false>(za, pw, &s.ga[0][0], TA, g, tq);
          store_proj(s.zs, za, 32 * (w - 4), P.round_z);
        }
      }
      ptx::cp_async_wait<0>();  // the records' u
      __syncthreads();

      // The warp's checkpoint rows into registers: rows 16w + g (+8), columns
      // 16m + 4tq .. +3. Lanes of odd rows load the two groups of a pair in the
      // other order, so that rows g and g + 1 (one quarter-warp) read different
      // banks. while the checkpoint lands, warp w: the n-tile w of C over all
      // columns (rows g, g + 8 of the kt tile; past T zero)
      if (w < NT8) {
        float cf[2][4] = {};
#pragma unroll
        for (int m = 0; m < 8; ++m) {
          const int c0 = 16 * m + 4 * tq;
          const float4 ka = g < T ? ld4(&s.B.tok.kt[g < T ? g : 0][c0])
                                  : make_float4(0.f, 0.f, 0.f, 0.f);
          const float4 kb = g + 8 < T
                                ? ld4(&s.B.tok.kt[g + 8 < T ? g + 8 : 0][c0])
                                : make_float4(0.f, 0.f, 0.f, 0.f);
          const uint32_t a0[4] = {tf(ka.x), tf(kb.x), tf(ka.y), tf(kb.y)};
          const uint32_t a1[4] = {tf(ka.z), tf(kb.z), tf(ka.w), tf(kb.w)};
          const float4 bb = ld4(&s.B.tok.y[8 * w + g][c0]);
          ptx::mma_m16n8k8_tf32(cf[m & 1], a0, tf(bb.x), tf(bb.y));
          ptx::mma_m16n8k8_tf32(cf[m & 1], a1, tf(bb.z), tf(bb.w));
        }
        if (g < T) {
          s.B.tok.cs[g < T ? g : 0][8 * w + 2 * tq] = cf[0][0] + cf[1][0];
          s.B.tok.cs[g < T ? g : 0][8 * w + 2 * tq + 1] = cf[0][1] + cf[1][1];
        }
        if (g + 8 < T) {
          s.B.tok.cs[g + 8 < T ? g + 8 : 0][8 * w + 2 * tq] =
              cf[0][2] + cf[1][2];
          s.B.tok.cs[g + 8 < T ? g + 8 : 0][8 * w + 2 * tq + 1] =
              cf[0][3] + cf[1][3];
        }
      }
      const int v0 = 16 * w + g, v1 = v0 + 8;
      uint4 st[2][8];
      ptx::mbarrier_wait_parity(sa(&s.bar), 0);
      {
        const bool odd = g & 1;
#pragma unroll
        for (int mp = 0; mp < 8; mp += 2) {
#pragma unroll
          for (int r = 0; r < 2; ++r) {
            const float* row = &s.sck[r ? v1 : v0][0];
            const uint4 x =
                *reinterpret_cast<const uint4*>(row + 16 * (mp + odd) + 4 * tq);
            const uint4 y = *reinterpret_cast<const uint4*>(
                row + 16 * (mp + 1 - odd) + 4 * tq);
            st[r][mp] = odd ? y : x;
            st[r][mp + 1] = odd ? x : y;
          }
        }
      }
      // S_c = S0 E + U^T K' in registers (state layout = accumulator fragments
      // of two n-tiles per 16-column group).
      if (c > 0) {
        // f16 m16n8k16 over the 16 records: logical j 2tq, 2tq + 1 | 2tq + 8,
        // 2tq + 9 (kpt holds them at 4tq .. 4tq + 3)
        const float (*ru)[D + 4] = s.rec_u;
        const uint32_t au[4] = {h2(ru[2 * tq][v0], ru[2 * tq + 1][v0]),
                                h2(ru[2 * tq][v1], ru[2 * tq + 1][v1]),
                                h2(ru[2 * tq + 8][v0], ru[2 * tq + 9][v0]),
                                h2(ru[2 * tq + 8][v1], ru[2 * tq + 9][v1])};
        const int col = 4 * (g >> 1) + (g & 1);
        // ks(16m + r) = 16m + ks(r)
        const int ka_row = ks<KS>(col), kb_row = ks<KS>(col + 2);
#pragma unroll
        for (int m = 0; m < 8; ++m) {
          const float4 E = ld4(&s.ef[16 * m + 4 * tq]);
          const uint2 ka =
              *reinterpret_cast<const uint2*>(&s.kpt[16 * m + ka_row][4 * tq]);
          const uint2 kb =
              *reinterpret_cast<const uint2*>(&s.kpt[16 * m + kb_row][4 * tq]);
          float ca[4] = {f(st[0][m].x) * E.x, f(st[0][m].y) * E.y,
                         f(st[1][m].x) * E.x, f(st[1][m].y) * E.y};
          float cb[4] = {f(st[0][m].z) * E.z, f(st[0][m].w) * E.w,
                         f(st[1][m].z) * E.z, f(st[1][m].w) * E.w};
          ptx::mma_m16n8k16_f16(ca, au, ka.x, ka.y);
          ptx::mma_m16n8k16_f16(cb, au, kb.x, kb.y);
          st[0][m] = make_uint4(__float_as_uint(ca[0]), __float_as_uint(ca[1]),
                                __float_as_uint(cb[0]), __float_as_uint(cb[1]));
          st[1][m] = make_uint4(__float_as_uint(ca[2]), __float_as_uint(ca[3]),
                                __float_as_uint(cb[2]), __float_as_uint(cb[3]));
        }
      }

      // the warp's reads of the records' u before Q overwrites them
      __syncwarp();
      {  // Q = S_c Y for the warp's 16 rows (two accumulator sets per n-tile)
        float qa[2][NT8][4] = {};
#pragma unroll
        // TF32 m16n8k8 (fp32 exponent range: S_c and Y can be far below f16's)
        for (int m = 0; m < 8; ++m) {
          const uint32_t a0[4] = {tf(f(st[0][m].x)), tf(f(st[1][m].x)),
                                  tf(f(st[0][m].y)), tf(f(st[1][m].y))};
          const uint32_t a1[4] = {tf(f(st[0][m].z)), tf(f(st[1][m].z)),
                                  tf(f(st[0][m].w)), tf(f(st[1][m].w))};
#pragma unroll
          for (int nt = 0; nt < NT8; ++nt) {
            const float4 bb = ld4(&s.B.tok.y[8 * nt + g][16 * m + 4 * tq]);
            ptx::mma_m16n8k8_tf32(qa[m & 1][nt], a0, tf(bb.x), tf(bb.y));
            ptx::mma_m16n8k8_tf32(qa[m & 1][nt], a1, tf(bb.z), tf(bb.w));
          }
        }
#pragma unroll
        for (int nt = 0; nt < NT8; ++nt) {
          // Q transposed, in the warp's own columns of the records' u (NB <= QR
          // rows)
          float (*qt)[D + 4] = s.rec_u;
          qt[8 * nt + 2 * tq][v0] = qa[0][nt][0] + qa[1][nt][0];
          qt[8 * nt + 2 * tq + 1][v0] = qa[0][nt][1] + qa[1][nt][1];
          qt[8 * nt + 2 * tq][v1] = qa[0][nt][2] + qa[1][nt][2];
          qt[8 * nt + 2 * tq + 1][v1] = qa[0][nt][3] + qa[1][nt][3];
        }
      }
      float* ck0 = ckp + v0 * a.checkpoint_stride[2] + 4 * tq;
      float* ck1 = ckp + v1 * a.checkpoint_stride[2] + 4 * tq;
      // The new checkpoint S_c (committed records only): with dense rows staged
      // in the checkpoint buffer (read only into registers) and stored by one
      // bulk copy, else by the threads.
      const bool ck_bulk = flush && a.checkpoint_stride[2] == D;
      if (ck_bulk) {
        const bool odd = g & 1;  // the bank order of the loads
#pragma unroll
        for (int mp = 0; mp < 8; mp += 2)
#pragma unroll
          for (int r = 0; r < 2; ++r) {
            float* row = &s.sck[r ? v1 : v0][0];
            *reinterpret_cast<uint4*>(row + 16 * (mp + odd) + 4 * tq) =
                odd ? st[r][mp + 1] : st[r][mp];
            *reinterpret_cast<uint4*>(row + 16 * (mp + 1 - odd) + 4 * tq) =
                odd ? st[r][mp] : st[r][mp + 1];
          }
        ptx::fence_proxy_async_smem();
      } else if (flush) {
#pragma unroll
        for (int m = 0; m < 8; ++m) {
          *reinterpret_cast<uint4*>(ck0 + 16 * m) = st[0][m];
          *reinterpret_cast<uint4*>(ck1 + 16 * m) = st[1][m];
        }
      }
      __syncthreads();
      if (ck_bulk && tid == 0) {
        ptx::bulk_copy_s2g(ckp, sa(&s.sck[0][0]), D * D * 4);
        ptx::bulk_commit();
      }

      float o[T];  // recurrent output of value row d (second half)
      // value row d: the delta rule over the tokens, ring append of u, the RMS
      // sums
      if (tid >= D) {
        float qv[M];
#pragma unroll
        for (int m = 0; m < M; ++m) qv[m] = s.rec_u[m][d];
        float u_[T];
#pragma unroll
        for (int t = 0; t < T; ++t) {
          float kq = qv[2 * t];
#pragma unroll
          for (int i = 0; i < t; ++i) kq += u_[i] * s.B.tok.cs[i][2 * t];
          u_[t] = bt[t] * (vr[t] - kq);
        }
#pragma unroll
        for (int t = 0; t < T; ++t) {
          float ov = qv[2 * t + 1];
#pragma unroll
          for (int i = 0; i <= t; ++i) ov += u_[i] * s.B.tok.cs[i][2 * t + 1];
          ov *= a.scale;
          o[t] = P.rounded ? bfr(ov) : ov;
        }
#pragma unroll
        for (int t = 0; t < T; ++t)
          if (t < q)
            a.u_ring[rslot * a.u_ring_stride0 +
                     (ring_base + ((r0 + c + t) & (R - 1))) * D + d] = u_[t];
        constexpr int PO = pow2_at_least(T);
        float ss[PO] = {};
#pragma unroll
        for (int t = 0; t < T; ++t) ss[t] = o[t] * o[t];
        const float sw = wsums(ss, lane);
        if (lane < T) s.ssq[lane][w - 4] = sw;
      }
      __syncthreads();

      // gated RMSNorm of value row d, the q token rows b .. b + q - 1
      if (tid >= D) {
#pragma unroll
        for (int t = 0; t < T; ++t) {
          if (t < q) {
            const float ms =
                (s.ssq[t][0] + s.ssq[t][1] + s.ssq[t][2] + s.ssq[t][3]) *
                (1.f / D);
            const float y = o[t] * rsqrtf(ms + a.norm_eps) * nwv *
                            (FZ ? sigm(s.zs[t][d]) : s.zs[t][d]);
            a.out[((static_cast<int64_t>(b) + t) * H + h) * D + d] =
                __float2bfloat16_rn(y);
          }
        }
      }
    }
  }
  ptx::cp_async_wait<0>();
  if (tid == 0) {
    // a checkpoint bulk store has read the shared buffer before the CTA exits
    ptx::bulk_wait_read<0>();
  }
}

// B200 (sm_100): 227 KB of dynamic shared memory per block
constexpr size_t SMEM_MAX = 232448;

template <int T, bool FZ>
void launch_step(const StepArgs& sa_, const kda_replayssm::Ctx& ctx) {
  using K = Smem<T, FZ>;
  constexpr size_t bytes = FZ ? wts_offset<K>() + 2 * D * D * 2 : sizeof(K);
  static_assert(bytes <= SMEM_MAX, "step_kernel shared memory");
  static bool init = false;
  if (!init) {
    cu::max_dynamic_smem(step_kernel<T, FZ>, bytes);
    // the full carveout: two CTAs per SM (the default carveout leaves room for
    // one), and proj_kernel beside them
    cu::smem_carveout(step_kernel<T, FZ>, cudaSharedmemCarveoutMaxShared);
    cu::smem_carveout(proj_kernel, cudaSharedmemCarveoutMaxShared);
    init = true;
  }
  const kda_replayssm::Dims& d = sa_.a.dims;
  const dim3 grid(d.H, d.N);
  if (FZ)
    step_kernel<T, FZ><<<grid, dim3(NTHR), bytes, ctx.stream>>>(sa_);
  else
    cu::launch_pdl(step_kernel<T, FZ>, grid, dim3(NTHR), bytes, ctx.stream,
                   sa_);
}

// Launched with 2048 threads per block, an invalid configuration: kernel_run's
// way to fail a shape it does not support (kda_replayssm_binding.cu checks
// cudaGetLastError after kernel_run).
__global__ void unsupported_kernel() {}

}  // namespace

namespace kda_replayssm {

void kernel_workspace(const Dims& d, size_t* scratch_bytes) {
  *scratch_bytes = 2 * static_cast<size_t>(d.M * d.H * d.D) * sizeof(float);
}

void kernel_run(const Args& x, const Ctx& ctx) {
  const Dims& d = x.dims;
  if ((d.T != 6 && d.T != 9) || d.D != D || d.N < 1 || d.N > 65535 || d.H < 1) {
    // fails loudly: cudaErrorInvalidConfiguration
    unsupported_kernel<<<1, 2048, 0, ctx.stream>>>();
    return;
  }
  const int M = static_cast<int>(d.M), HD = static_cast<int>(d.H * d.D);
  // vLLM's route for the shape: KdaSplit only at TP4 with T = 6, up to 16
  // tokens per graph (split) and 32-64 tokens (split_gate); FlashInfer (other
  // TP4 T = 6 rows) and Triton (TP8 T = 6, every T = 9) round to bf16.
  const bool ks_route = d.H == 16 && d.T == 6;
  const bool split = ks_route && M <= 16;
  const bool split_gate = ks_route && M >= 32 && M <= 64;
  // The projections in the step CTAs (FZ) for T = 6 up to FZ_UNITS units (the
  // head's weights by bulk copy there, one CTA per SM); otherwise in their own
  // launch (weights read once per head and PR tokens).
  const bool fz = d.N * d.H <= FZ_UNITS && d.T == 6;
  float* gz = static_cast<float*>(ctx.scratch);
  if (!fz) {
    ProjArgs p{};
    p.act[0] = x.f_a, p.act[1] = x.g_a;
    p.s0[0] = x.f_a_stride[0], p.s0[1] = x.g_a_stride[0];
    p.wt[0] = x.f_b_weight, p.wt[1] = x.g_b_weight;
    p.out = gz;
    p.qsl = x.query_start_loc;
    p.M = M, p.HD = HD, p.N = static_cast<int>(d.N);
    p.round_z = !split_gate;
    proj_kernel<<<dim3(d.H, 2, (M + PR - 1) / PR), 128, 0, ctx.stream>>>(p);
  }
  const StepArgs sa_{x, gz, !(split || split_gate), !split_gate};
  if (d.T == 6) {
    if (fz)
      launch_step<6, true>(sa_, ctx);
    else
      launch_step<6, false>(sa_, ctx);
  } else {  // T = 9
    launch_step<9, false>(sa_, ctx);
  }
}

}  // namespace kda_replayssm
