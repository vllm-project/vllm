// One persistent decode kernel for a dense GQA transformer in BF16 as vLLM's single-request decode
// step, built for one model at a time:
//   -DMK_MODEL_LLAMA31_8B (default)  Llama-3.1-8B-Instruct
//   -DMK_MODEL_QWEN3_1_7B            Qwen3-1.7B (per-head Q/K RMSNorm before RoPE)
//
// Derived from llama.cpp's src/megakernel/llama-bf16.hip (see SOURCE). What differs for vLLM:
//   - weights are vLLM's own Hugging Face tensors (qkv_proj and gate_up_proj are row slices);
//   - the K/V cache is vLLM's paged BF16 cache, a cell found through the request's block table;
//   - RoPE is NeoX style (HF head order) from vLLM's own cos/sin cache;
//   - token, position and the grid barrier generation are read on the device, so a launch is
//     identical every step and can be captured in a HIP graph;
//   - the kernel ends with the final RMSNorm, written as BF16 hidden states; vLLM's own LM head,
//     logits processor and sampler run on them.
//
// A launch decodes one token: 64 workgroups stay resident and meet at a grid barrier, five
// stages per layer:
//
//   qkv   RMSNorm of x (every block, fp32 into LDS), Q / K / V
//   attn  per (KV head, context chunk), all 4 of its query heads: RoPE, new K/V
//         into the cache, scores, softmax partials, P.V partials
//   o     merge the chunk partials, O, x += o
//   up    RMSNorm, gate and up for the same rows, ffn = silu(g) * u
//   down  down, x += down
//
// then the final norm.
//
// GEMV: BF16 weights row-major as vLLM holds them, a wave-wide 16-byte load reads 512 contiguous
// bytes of one row; activations stay fp32 in LDS. Wave w of a block takes the w-th eighth of K,
// the 8 partials are summed in LDS.
//
//   hipcc -O3 --offload-arch=gfx1201 -shared -fPIC [-DMK_MODEL_...] -o libmk_<model>_vllm_<arch>.so dense_vllm.hip

#include "common.h"

namespace mk_dense_impl {

#if defined(MK_MODEL_QWEN3_1_7B)
constexpr int kHidden = 2048;
constexpr int kFF = 6144;
constexpr int kHeads = 16;
constexpr float kEps = 1e-6f;
constexpr bool kQKNorm = true;
#else
constexpr int kHidden = 4096;
constexpr int kFF = 14336;
constexpr int kHeads = 32;
constexpr float kEps = 1e-5f;
constexpr bool kQKNorm = false;
#endif
constexpr int kKV = 8;
constexpr int kDim = 128;
constexpr int kRot = kDim / 2;
constexpr int kQPK = kHeads / kKV;  // query heads per KV head; one attention item handles all of them
constexpr int kG = kHidden / 2048;  // groups of 256 x 8 values per block for a hidden-size vector
static_assert(kHeads * kDim == kHidden, "attention output feeds O as a hidden-size vector");
static_assert(kQPK == 2 || kQPK == 4, "attention maps 2 or 4 query heads per KV head onto 256 threads");
static_assert(kHidden % 2048 == 0 && kFF % 2048 == 0, "GEMV splits K into 8 waves x multiples of 32 chunks");
constexpr int kBlock = 256;
constexpr int kWaves = kBlock / 32;
#ifndef MK_GRID
#if !defined(MK_STANDALONE) && defined(__gfx1151__)
#define MK_GRID 40  // the host pass still sees 64: mk_vllm_dense_create asks the device
#else
#define MK_GRID 64
#endif
#endif
constexpr int kGrid = MK_GRID;  // blocks (64 on gfx1201 and gfx1100, 40 on gfx1151)
constexpr int kMaxChunk = 1024;
constexpr int kMaxSplits = 64;
// Row blocks of 32 rows per stage, and how many each block takes (row block b0 + kGrid * k).
constexpr int kRbQ = kHeads * kDim / 32, kRbK = kKV * kDim / 32;
constexpr int kRbQKV = kRbQ + 2 * kRbK, kRbH = kHidden / 32, kRbF = kFF / 32;
constexpr int per_block(int n) { return (n + kGrid - 1) / kGrid; }
constexpr int kKQKV = per_block(kRbQKV), kKO = per_block(kRbH), kKGU = per_block(kRbF), kKD = per_block(kRbH);
constexpr int cmax(int a, int b) { return a > b ? a : b; }
// Row block rb of a stage with n row blocks exists (always, when kGrid divides n).
__device__ __forceinline__ constexpr bool has_rb(int rb, int n) { return n % kGrid == 0 || rb < n; }
constexpr int kProbeLayer = 5;
constexpr int kProbeBase = 192;
constexpr int kMaxSteps = 4096;

struct Layer {
  const u4v* q;
  const u4v* k;
  const u4v* v;
  const u4v* o;
  const u4v* gate;
  const u4v* up;
  const u4v* down;
  const float* attn_norm;
  const float* ffn_norm;
  const float* q_norm;  // [kDim], when kQKNorm
  const float* k_norm;
};

// Every field is fixed for the lifetime of the engine (device pointers into vLLM's persistent
// buffers), so the launch can be captured in a HIP graph and replayed each step.
struct Args {
  int n_layers;
  int max_pos;               // rows of cos_sin
  int block_shift;           // log2 of vLLM's KV block size
  const Layer* layers;
  const uint16_t* emb;       // BF16 [vocab][hidden], row-major (vLLM's embed_tokens)
  const float* out_norm;     // fp32 copy of the final RMSNorm weight
  const uint16_t* cos_sin;   // BF16 [max_pos][kDim]: cos for dims 0..63, sin for 64..127
  const int* input_ids;      // vLLM's input_ids[0]
  const long long* positions;  // vLLM's positions[0]
  const int* block_table;    // vLLM's block_table[0][:]
  uint16_t* out_hidden;      // BF16 [hidden]: the final RMSNorm of x
  float* x;
  float* q;
  float* k;
  float* v;
  float* ffn;
  float* part_acc;  // [kHeads][kMaxSplits][kDim]
  float* part_m;
  float* part_l;
  uint16_t* const* kcache;  // [layer] -> BF16 [cell][kv][dim]: vLLM's kv_cache[0]
  uint16_t* const* vcache;  // [layer] -> vLLM's kv_cache[1]
  float* trace;
  unsigned* bar_count;
  unsigned long long* stamps;
  unsigned* keepalive;  // host memory written once per layer (mk_keepalive), null to skip
  unsigned* status;     // host memory: the watchdog flag after each step (mk_status), null to skip
};

__device__ __forceinline__ uint16_t f32_to_bf16(float f) {  // round to nearest even, as torch does
  const unsigned u = __builtin_bit_cast(unsigned, f);
  return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

struct AttnSmem {
  float q[kQPK][kDim];
  float tmp[kQPK][kDim];
  float cs[kRot];
  float sn[kRot];
  float p[kQPK][kMaxChunk];
  float vred[2][kQPK][kDim];  // P.V partials of the two token halves
  int blk[kMaxChunk / 16];  // first cache cell of each 16-position pass of this chunk
};

struct Smem {
  union {
    struct {
      float x[kHidden];
      float part[kWaves * cmax(2 * kKGU, cmax(kKQKV, kKO))][32];
    } h;  // stages with hidden-size input
    struct {
      float x[kFF];
      float part[kWaves * kKD][32];
    } f;  // down
    AttnSmem at;
  } u;
  float red[32 * kWaves];
  int best_i[kWaves];
};

// ------------------------------------------------------------- block helpers
// Wave w owns groups w, w + 8, ... of 256 values; lane l holds 256 * group + 8 * l + 0..7.
template <int G>
__device__ __forceinline__ void load_groups(const float* __restrict__ src, float (&v)[G][8]) {
  const int lane = threadIdx.x & 31, wv = threadIdx.x >> 5;
#pragma unroll
  for (int g = 0; g < G; g++) {
    const float4* p = (const float4*)(src + (wv + kWaves * g) * 256 + 8 * lane);
    const float4 a = p[0], b = p[1];
    v[g][0] = a.x, v[g][1] = a.y, v[g][2] = a.z, v[g][3] = a.w;
    v[g][4] = b.x, v[g][5] = b.y, v[g][6] = b.z, v[g][7] = b.w;
  }
}

template <int G>
__device__ __forceinline__ void store_groups(float* dst, const float (&v)[G][8]) {
  const int lane = threadIdx.x & 31, wv = threadIdx.x >> 5;
#pragma unroll
  for (int g = 0; g < G; g++) {
    float4* p = (float4*)(dst + (wv + kWaves * g) * 256 + 8 * lane);
    p[0] = make_float4(v[g][0], v[g][1], v[g][2], v[g][3]);
    p[1] = make_float4(v[g][4], v[g][5], v[g][6], v[g][7]);
  }
}

template <int G>
__device__ void groups_rmsnorm(Smem& sm, float (&v)[G][8], const float (&wg)[G][8]) {
  const int lane = threadIdx.x & 31, wv = threadIdx.x >> 5;
  float ss = 0.f;
#pragma unroll
  for (int g = 0; g < G; g++)
#pragma unroll
    for (int e = 0; e < 8; e++) ss = __builtin_fmaf(v[g][e], v[g][e], ss);
  ss = wave_sum(ss);
  lds_sync();
  if (lane == 0) sm.red[wv] = ss;
  lds_sync();
  float tot = 0.f;
#pragma unroll
  for (int k = 0; k < kWaves; k++) tot += sm.red[k];
  const float inv = 1.f / sqrtf(tot / (float)(G * kBlock * 8) + kEps);
#pragma unroll
  for (int g = 0; g < G; g++)
#pragma unroll
    for (int e = 0; e < 8; e++) v[g][e] = v[g][e] * inv * wg[g][e];
}

// Weight pointers come from the layer table in memory, so the compiler cannot tell they are
// global and would emit flat loads, which also count in DScnt: every LDS wait would then wait
// for the weights in flight. Loads through these are global loads.
template <typename T>
__device__ __forceinline__ T ldg(const T* p) {
  return *(const T __attribute__((address_space(1)))*)p;
}

// ------------------------------------------------------------------- GEMV
// W is row-major [M][K] BF16 (GGUF / ggml layout, K / 8 16-byte chunks per row), x = fp32
// activations in LDS. Returns row 32 * rb + lane over chunks [c0, c1), a multiple of 32 chunks.
// Rows go 4 at a time: the wave reads 512 contiguous bytes of each (lane l: chunk c0 + l + 32 j),
// then a transpose-reduce leaves row r's sum in lanes 8 (r % 4) .. + 7 and lane l picks row l.
__device__ __forceinline__ float bf16_dot(const u4v* __restrict__ W, int rb, int KC, int c0, int c1,
                                          const float* x) {
  constexpr int RG = 4, LPR = 32 / RG;
  const int lane = threadIdx.x & 31;
  float res = 0.f;
#pragma unroll 1
  for (int g = 0; g < 32 / RG; g++) {
    const u4v* p = W + (unsigned)(rb * 32 + RG * g) * KC;
    float acc[RG];
#pragma unroll
    for (int r = 0; r < RG; r++) acc[r] = 0.f;
#pragma unroll 4
    for (int c = c0 + lane; c < c1; c += 32) {
      const float4 xa = *(const float4*)(x + 8 * c), xb = *(const float4*)(x + 8 * c + 4);
#pragma unroll
      for (int r = 0; r < RG; r++) {
        const u4v w = ldg(p + (unsigned)r * KC + c);
        float s = bf16_lo(w.x) * xa.x;
        s = __builtin_fmaf(bf16_hi(w.x), xa.y, s);
        s = __builtin_fmaf(bf16_lo(w.y), xa.z, s);
        s = __builtin_fmaf(bf16_hi(w.y), xa.w, s);
        s = __builtin_fmaf(bf16_lo(w.z), xb.x, s);
        s = __builtin_fmaf(bf16_hi(w.z), xb.y, s);
        s = __builtin_fmaf(bf16_lo(w.w), xb.z, s);
        s = __builtin_fmaf(bf16_hi(w.w), xb.w, s);
        acc[r] += s;
      }
    }
#pragma unroll
    for (int n = RG / 2, off = 16; n >= 1; n >>= 1, off >>= 1) {
      const bool hi = lane & off;
#pragma unroll
      for (int i = 0; i < n; i++) {
        const float send = hi ? acc[i] : acc[i + n];
        const float keep = hi ? acc[i + n] : acc[i];
        acc[i] = keep + __shfl_xor(send, off);
      }
    }
#pragma unroll
    for (int off = LPR / 2; off >= 1; off >>= 1) acc[0] += __shfl_xor(acc[0], off);
    const float mine = __shfl(acc[0], LPR * (lane % RG));
    if (lane / RG == g) res = mine;
  }
  return res;
}

// Row block r of a stage is blockIdx.x + kGrid * r. Partials from the 8 waves are
// summed in wave order; threads t < NR * 32 get row block t / 32, lane t % 32.
template <int NR, int NM>
__device__ __forceinline__ bool reduce_partials(float (*part)[32], const float (&p)[NR][NM], float (&tot)[NM],
                                                int& r) {
  const int t = threadIdx.x, lane = t & 31, wv = t >> 5;
  lds_sync();
#pragma unroll
  for (int i = 0; i < NR; i++)
#pragma unroll
    for (int m = 0; m < NM; m++) part[(wv * NR + i) * NM + m][lane] = p[i][m];
  lds_sync();
  if (t >= NR * 32) return false;
  r = t >> 5;
#pragma unroll
  for (int m = 0; m < NM; m++) {
    tot[m] = 0.f;
#pragma unroll
    for (int w = 0; w < kWaves; w++) tot[m] += part[(w * NR + r) * NM + m][lane];
  }
  return true;
}

// reduce_partials for partials the waves already wrote to part[(wv * NR + i) * NM + m][lane].
template <int NR, int NM>
__device__ __forceinline__ bool reduce_stored(float (*part)[32], float (&tot)[NM], int& r) {
  const int t = threadIdx.x, lane = t & 31;
  lds_sync();
  if (t >= NR * 32) return false;
  r = t >> 5;
#pragma unroll
  for (int m = 0; m < NM; m++) {
    tot[m] = 0.f;
#pragma unroll
    for (int w = 0; w < kWaves; w++) tot[m] += part[(w * NR + r) * NM + m][lane];
  }
  return true;
}

// reduce_stored for more row blocks than waves: threads take row blocks first + t / 32.
template <int NR, int NM>
__device__ __forceinline__ bool reduce_stored_from(float (*part)[32], float (&tot)[NM], int& r, int first) {
  const int t = threadIdx.x, lane = t & 31;
  if (first == 0) lds_sync();
  if (t >= (NR - first) * 32) return false;
  r = first + (t >> 5);
#pragma unroll
  for (int m = 0; m < NM; m++) {
    tot[m] = 0.f;
#pragma unroll
    for (int w = 0; w < kWaves; w++) tot[m] += part[(w * NR + r) * NM + m][lane];
  }
  return true;
}

// --------------------------------------------------------------- attention
// vLLM's ROCm paged layout (PagedAttention.split_kv_cache), per block of bs = 2^block_shift positions:
// K [kv head][dim / 8][slot][8], V [kv head][dim][slot]. A cell is block * bs + slot.
__device__ __forceinline__ unsigned k_off(const Args& a, int cell, int kv, int d) {
  const int sh = a.block_shift, slot = cell & ((1 << sh) - 1);
  return ((unsigned)(cell >> sh) << (sh + 10)) + ((unsigned)kv << (sh + 7)) + ((unsigned)(d >> 3) << (sh + 3)) +
         (slot << 3) + (d & 7);
}
__device__ __forceinline__ unsigned v_off(const Args& a, int cell, int kv, int d) {
  const int sh = a.block_shift, slot = cell & ((1 << sh) - 1);
  return ((unsigned)(cell >> sh) << (sh + 10)) + ((unsigned)kv << (sh + 7)) + ((unsigned)d << sh) + slot;
}

// The cache cell vLLM gave position p of this request: its block table maps blocks of
// 2^block_shift positions to physical blocks.
__device__ __forceinline__ int kv_cell(const Args& a, int p) {
  const int mask = (1 << a.block_shift) - 1;
  return (a.block_table[p >> a.block_shift] << a.block_shift) | (p & mask);
}


// NeoX RoPE on one element (vLLM's Llama, HF head order): elements i and i + 64 rotate by angle
// i % 64. row[] is the head this element belongs to, staged in LDS by the caller.
__device__ __forceinline__ float rope_neox(const AttnSmem& s, const float* row, float v, int i) {
  const float other = row[i ^ kRot];
  const float c = s.cs[i & (kRot - 1)], sn = s.sn[i & (kRot - 1)];
  return (i & kRot) ? __builtin_fmaf(other, sn, v * c) : __builtin_fmaf(v, c, -(other * sn));
}

// One work item = KV head g and a chunk of the context, for its 4 query heads.
// Thread t holds query elements (head t / 128, dim t % 128) and (head t / 128 + 2, same dim).
// Scores: 16 lanes x 8 dims per token, 16 tokens per pass. P.V: one dim per thread, 8 tokens
// per thread, the two halves of the block take the 16 tokens of a pass.
__device__ void attention(const Args& a, Smem& sm, int layer, int pos, int nchunk, int chunk) {
  AttnSmem& s = sm.u.at;
  const int t = threadIdx.x, lane = t & 31, wv = t >> 5;
  const int h0 = t >> 7, i = t & 127;
  const int sub = t >> 4, part = t & 15;
  constexpr int kTPH = kBlock / kQPK, kWPH = kWaves / kQPK;  // softmax: threads and waves per head
  const int hs = t / kTPH, j0 = t % kTPH;
  const int ntok = pos + 1;
  const float scale = 1.f / sqrtf((float)kDim);

  if (t < kRot) {  // vLLM's own (llama3-scaled) table, so rotations match its prefill exactly
    const uint16_t* row = a.cos_sin + (size_t)pos * kDim;
    s.cs[t] = bf16_to_f32(row[t]);
    s.sn[t] = bf16_to_f32(row[kRot + t]);
  }

  uint16_t* const kc = a.kcache[layer];
  uint16_t* const vc = a.vcache[layer];

  for (int item = blockIdx.x; item < kKV * nchunk; item += gridDim.x) {
    const int g = item / nchunk, c = item % nchunk;
    const int t0 = c * chunk, t1 = min(t0 + chunk, ntok);

    // RoPE for the 4 query heads (and the new K, V when this chunk holds pos).
    float qa = a.q[(kQPK * g + h0) * kDim + i], qb = kQPK == 4 ? a.q[(kQPK * g + h0 + 2) * kDim + i] : 0.f;
    const bool last = t1 == ntok;
    float kr = last && h0 == 0 ? a.k[g * kDim + i] : 0.f;
    if constexpr (kQKNorm) {  // RMSNorm over each head's 128 dims (4 waves) before RoPE
      const float sa = wave_sum(qa * qa), sb = wave_sum(qb * qb), sk = wave_sum(kr * kr);
      lds_sync();
      if (lane == 0) sm.red[wv] = sa, sm.red[kWaves + wv] = sb, sm.red[2 * kWaves + wv] = sk;
      lds_sync();
      float ta = 0.f, tb = 0.f, tk = 0.f;
#pragma unroll
      for (int w = 0; w < 4; w++)
        ta += sm.red[4 * h0 + w], tb += sm.red[kWaves + 4 * h0 + w], tk += sm.red[2 * kWaves + w];
      const float qn = a.layers[layer].q_norm[i], kn = a.layers[layer].k_norm[i];
      qa *= qn / sqrtf(ta / kDim + kEps);
      qb *= qn / sqrtf(tb / kDim + kEps);
      kr *= kn / sqrtf(tk / kDim + kEps);
    }
    lds_sync();
    s.tmp[h0][i] = qa;
    if constexpr (kQPK == 4) s.tmp[h0 + 2][i] = qb;
    // The loops below step 16 positions at a time from a multiple of 16, and a block holds at
    // least 16 (mk_vllm_dense_bind), so each pass stays in one block.
    if (t < (t1 - t0 + 15) / 16) s.blk[t] = kv_cell(a, t0 + 16 * t);
    lds_sync();
    s.q[h0][i] = rope_neox(s, s.tmp[h0], qa, i);
    if constexpr (kQPK == 4) s.q[h0 + 2][i] = rope_neox(s, s.tmp[h0 + 2], qb, i);
    if (last) {
      lds_sync();
      if (h0 == 0) s.tmp[0][i] = kr;
      lds_sync();
      const int cell = kv_cell(a, pos);
      if (h0 == 0) kc[k_off(a, cell, g, i)] = f32_to_bf16(rope_neox(s, s.tmp[0], kr, i));
      else vc[v_off(a, cell, g, i)] = f32_to_bf16(a.v[g * kDim + i]);
    }
    __syncthreads();

    float q[kQPK][8];
#pragma unroll
    for (int h = 0; h < kQPK; h++)
#pragma unroll
      for (int e = 0; e < 8; e++) q[h][e] = s.q[h][8 * part + e];
#pragma unroll 2
    for (int base = t0; base < t1; base += 16) {
      const int tok = base + sub;
      const int cell = s.blk[(base - t0) >> 4] + sub;
      float sc[kQPK] = {};
      if (tok < t1) {
        const u4v kk = *(const u4v*)(kc + k_off(a, cell, g, 8 * part));
        const unsigned kw[4] = {kk.x, kk.y, kk.z, kk.w};
#pragma unroll
        for (int e = 0; e < 8; e++) {
          const float kf = (e & 1) ? bf16_hi(kw[e >> 1]) : bf16_lo(kw[e >> 1]);
#pragma unroll
          for (int h = 0; h < kQPK; h++) sc[h] = __builtin_fmaf(q[h][e], kf, sc[h]);
        }
      }
#pragma unroll
      for (int off = 8; off > 0; off >>= 1)
#pragma unroll
        for (int h = 0; h < kQPK; h++) sc[h] += __shfl_xor(sc[h], off);
      if (part < kQPK && tok < t1) {
        float mine = sc[0];
#pragma unroll
        for (int h = 1; h < kQPK; h++)
          if (part == h) mine = sc[h];
        s.p[part][tok - t0] = mine * scale;
      }
    }
    lds_sync();

    const int n = t1 - t0;
    float mx = -INFINITY;
    for (int j = j0; j < n; j += kTPH) mx = fmaxf(mx, s.p[hs][j]);
    mx = wave_max(mx);
    if (lane == 0) sm.red[wv] = mx;
    lds_sync();
    {
      float m = sm.red[kWPH * hs];
#pragma unroll
      for (int w = 1; w < kWPH; w++) m = fmaxf(m, sm.red[kWPH * hs + w]);
      mx = m;
    }
    float ls = 0.f;
    for (int j = j0; j < n; j += kTPH) {
      const float e = expf(s.p[hs][j] - mx);
      s.p[hs][j] = e;
      ls += e;
    }
    ls = wave_sum(ls);
    lds_sync();
    if (lane == 0) sm.red[wv] = ls;
    lds_sync();
    {
      float l = 0.f;
#pragma unroll
      for (int w = 0; w < kWPH; w++) l += sm.red[kWPH * hs + w];
      ls = l;
    }
    if (j0 == 0) {
      a.part_m[(kQPK * g + hs) * kMaxSplits + c] = mx;
      a.part_l[(kQPK * g + hs) * kMaxSplits + c] = ls;
    }

    float acc[kQPK] = {};
    const int hd = t & 127;
#pragma unroll 2
    for (int base = t0; base < t1; base += 16) {
      const int tok0 = base + 8 * (t >> 7);
      if (tok0 < t1) {
        const int cell = s.blk[(base - t0) >> 4] + 8 * (t >> 7);
        const u4v vv = *(const u4v*)(vc + v_off(a, cell, g, hd));
        const unsigned vw[4] = {vv.x, vv.y, vv.z, vv.w};
        float pr[kQPK][8];
#pragma unroll
        for (int h = 0; h < kQPK; h++) {
          const float4 p0 = *(const float4*)&s.p[h][tok0 - t0], p1 = *(const float4*)&s.p[h][tok0 - t0 + 4];
          pr[h][0] = p0.x, pr[h][1] = p0.y, pr[h][2] = p0.z, pr[h][3] = p0.w;
          pr[h][4] = p1.x, pr[h][5] = p1.y, pr[h][6] = p1.z, pr[h][7] = p1.w;
        }
#pragma unroll
        for (int e = 0; e < 8; e++) {
          // cells past the context hold whatever the block had (maybe NaN): never multiply them
          const bool in = tok0 + e < t1;
          const float vf = in ? ((e & 1) ? bf16_hi(vw[e >> 1]) : bf16_lo(vw[e >> 1])) : 0.f;
#pragma unroll
          for (int h = 0; h < kQPK; h++) acc[h] = __builtin_fmaf(in ? pr[h][e] : 0.f, vf, acc[h]);
        }
      }
    }
#pragma unroll
    for (int h = 0; h < kQPK; h++) s.vred[t >> 7][h][hd] = acc[h];
    lds_sync();
#pragma unroll
    for (int k = 0; k < kQPK / 2; k++) {
      const int h = h0 + 2 * k;
      const float out = s.vred[0][h][i] + s.vred[1][h][i];
      a.part_acc[((unsigned)(kQPK * g + h) * kMaxSplits + c) * kDim + i] = out;
    }
  }
}

// Group w = heads 2w, 2w + 1: lane l holds head 2w + l / 16, dims 8 * (l % 16) + 0..7.
template <int G>
__device__ void attention_merge(const Args& a, float (&v)[G][8], int nchunk) {
  const int lane = threadIdx.x & 31, wv = threadIdx.x >> 5;
#pragma unroll
  for (int gi = 0; gi < G; gi++) {
    const int grp = wv + kWaves * gi;
    const int head = 2 * grp + (lane >> 4), d0 = 8 * (lane & 15);
    const float* pm = a.part_m + head * kMaxSplits;
    const float* pl = a.part_l + head * kMaxSplits;
    float M = -INFINITY;
    for (int c = 0; c < nchunk; c++) M = fmaxf(M, pm[c]);
    float l = 0.f, o[8];
#pragma unroll
    for (int e = 0; e < 8; e++) o[e] = 0.f;
    for (int c = 0; c < nchunk; c++) {
      const float sc = expf(pm[c] - M);
      l = __builtin_fmaf(pl[c], sc, l);
      const float4* p = (const float4*)(a.part_acc + (((unsigned)head * kMaxSplits + c) * kDim + d0));
      const float4 x0 = p[0], x1 = p[1];
      const float acc[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
      for (int e = 0; e < 8; e++) o[e] = __builtin_fmaf(acc[e], sc, o[e]);
    }
#pragma unroll
    for (int e = 0; e < 8; e++) v[gi][e] = o[e] / l;
  }
}

__host__ __device__ inline void attn_chunks(int ntok, int& nchunk, int& chunk) {
  attn_split(ntok, kKV, kGrid, kMaxChunk, nchunk, chunk);
}

// ------------------------------------------------------------------ kernel
__device__ __forceinline__ void decode_step(const Args& a, Smem& sm, int pos, int token, unsigned& my_gen,
                                            unsigned long long* stamps, float* trace) {
  const int t = threadIdx.x, lane = t & 31, wv = t >> 5;
  const int b0 = blockIdx.x;
  constexpr int KCH = kHidden / 8 / kWaves;  // chunks per wave for K = hidden
  constexpr int KCF = kFF / 8 / kWaves;
  int ns = 0;
  stamp(stamps, ns);

  for (int e = blockIdx.x * kBlock + t; e < kHidden; e += gridDim.x * kBlock)
    a.x[e] = bf16_to_f32(a.emb[(unsigned)token * kHidden + e]);
  grid_barrier(a.bar_count, my_gen);
  stamp(stamps, ns);

  int nchunk, chunk;
  attn_chunks(pos + 1, nchunk, chunk);

  for (int layer = 0; layer < a.n_layers; layer++) {
    mk_keepalive(a.keepalive, layer);
    const Layer L = a.layers[layer];
    auto sub = [&](int k) {
      if (stamps && layer == kProbeLayer && blockIdx.x == 0 && t == 0) stamps[kProbeBase + k] = wall_clock64();
    };
    if (trace && blockIdx.x == 0)
      for (int e = t; e < kHidden; e += kBlock) trace[(size_t)layer * kHidden + e] = a.x[e];

    {  // qkv: Q kRbQ row blocks, K and V kRbK each -> kKQKV per block
      float v[kG][8], w[kG][8];
      load_groups(a.x, v);
      load_groups(L.attn_norm, w);
      groups_rmsnorm(sm, v, w);
      store_groups(sm.u.h.x, v);
      lds_sync();
      sub(0);
      float p[kKQKV][1];
#pragma unroll
      for (int r = 0; r < kKQKV; r++) {
        const int rb = b0 + kGrid * r;
        const u4v* W = rb < kRbQ ? L.q : rb < kRbQ + kRbK ? L.k : L.v;
        const int lr = rb < kRbQ ? rb : rb < kRbQ + kRbK ? rb - kRbQ : rb - kRbQ - kRbK;
        p[r][0] = has_rb(rb, kRbQKV) ? bf16_dot(W, lr, kHidden / 8, wv * KCH, wv * KCH + KCH, sm.u.h.x) : 0.f;
      }
      float tot[1];
      int r;
      if (reduce_partials(sm.u.h.part, p, tot, r) && has_rb(b0 + kGrid * r, kRbQKV)) {
        const int rb = b0 + kGrid * r;
        if (rb < kRbQ) a.q[rb * 32 + lane] = tot[0];
        else if (rb < kRbQ + kRbK) a.k[(rb - kRbQ) * 32 + lane] = tot[0];
        else a.v[(rb - kRbQ - kRbK) * 32 + lane] = tot[0];
      }
      sub(1);
    }
    grid_barrier(a.bar_count, my_gen);
    stamp(stamps, ns);

    attention(a, sm, layer, pos, nchunk, chunk);
    grid_barrier(a.bar_count, my_gen);
    stamp(stamps, ns);

    {  // o: kRbH row blocks -> kKO per block; x += o
      float v[kG][8];
      attention_merge(a, v, nchunk);
      store_groups(sm.u.h.x, v);
      lds_sync();
      sub(2);
      float p[kKO][1];
#pragma unroll
      for (int r = 0; r < kKO; r++)
        p[r][0] = has_rb(b0 + kGrid * r, kRbH)
                      ? bf16_dot(L.o, b0 + kGrid * r, kHidden / 8, wv * KCH, wv * KCH + KCH, sm.u.h.x) : 0.f;
      float tot[1];
      int r;
      if (reduce_partials(sm.u.h.part, p, tot, r) && has_rb(b0 + kGrid * r, kRbH))
        a.x[(b0 + kGrid * r) * 32 + lane] += tot[0];
      sub(3);
    }
    grid_barrier(a.bar_count, my_gen);
    stamp(stamps, ns);

    {  // gate, up: kRbF row blocks each -> kKGU per block
      float v[kG][8], w[kG][8];
      load_groups(a.x, v);
      load_groups(L.ffn_norm, w);
      groups_rmsnorm(sm, v, w);
      store_groups(sm.u.h.x, v);
      lds_sync();
      sub(4);
      // The loop stays rolled and its partials go straight to LDS: a register array
      // indexed by the rolled loop would live in scratch.
#pragma unroll 1
      for (int r = 0; r < kKGU; r++) {
        if constexpr (kRbF % kGrid != 0)
          if (b0 + kGrid * r >= kRbF) break;
        // separate streams for gate and up stream faster than one interleaved loop
        sm.u.h.part[(wv * kKGU + r) * 2][lane] =
            bf16_dot(L.gate, b0 + kGrid * r, kHidden / 8, wv * KCH, wv * KCH + KCH, sm.u.h.x);
        sm.u.h.part[(wv * kKGU + r) * 2 + 1][lane] =
            bf16_dot(L.up, b0 + kGrid * r, kHidden / 8, wv * KCH, wv * KCH + KCH, sm.u.h.x);
      }
      if constexpr (kKGU <= kWaves) {
        float tot[2];
        int r;
        if (reduce_stored<kKGU, 2>(sm.u.h.part, tot, r) && has_rb(b0 + kGrid * r, kRbF)) {
          const float g = tot[0], u = tot[1];
          a.ffn[(b0 + kGrid * r) * 32 + lane] = (g / (1.f + expf(-g))) * u;
        }
      } else {
#pragma unroll
        for (int first = 0; first < kKGU; first += kWaves) {
          float tot[2];
          int r;
          if (reduce_stored_from<kKGU, 2>(sm.u.h.part, tot, r, first) && has_rb(b0 + kGrid * r, kRbF)) {
            const float g = tot[0], u = tot[1];
            a.ffn[(b0 + kGrid * r) * 32 + lane] = (g / (1.f + expf(-g))) * u;
          }
        }
      }
      sub(5);
    }
    grid_barrier(a.bar_count, my_gen);
    stamp(stamps, ns);

    {  // down: kRbH row blocks over K = kFF -> kKD per block; x += down
      const float4* src = (const float4*)a.ffn;
      float4* dst = (float4*)sm.u.f.x;
      for (int e = t; e < kFF / 4; e += kBlock) dst[e] = src[e];
      lds_sync();
      sub(6);
      float p[kKD][1];
#pragma unroll
      for (int r = 0; r < kKD; r++)
        p[r][0] = has_rb(b0 + kGrid * r, kRbH)
                      ? bf16_dot(L.down, b0 + kGrid * r, kFF / 8, wv * KCF, wv * KCF + KCF, sm.u.f.x) : 0.f;
      float tot[1];
      int r;
      if (reduce_partials(sm.u.f.part, p, tot, r) && has_rb(b0 + kGrid * r, kRbH))
        a.x[(b0 + kGrid * r) * 32 + lane] += tot[0];
      sub(7);
    }
    grid_barrier(a.bar_count, my_gen);
    stamp(stamps, ns);
  }

  if (trace && blockIdx.x == 0)
    for (int e = t; e < kHidden; e += kBlock) trace[(size_t)a.n_layers * kHidden + e] = a.x[e];

  {  // final RMSNorm: every block has x after the last barrier; block 0 writes it for vLLM's LM head
    float v[kG][8], w[kG][8];
    load_groups(a.x, v);
    load_groups(a.out_norm, w);
    groups_rmsnorm(sm, v, w);
    store_groups(sm.u.h.x, v);
    lds_sync();
    if (blockIdx.x == 0)
      for (int e = t; e < kHidden; e += kBlock) a.out_hidden[e] = f32_to_bf16(sm.u.h.x[e]);
  }
  stamp(stamps, ns);
}

// The barrier counter only grows, by gridDim.x per barrier, so at a launch boundary it is a whole
// number of generations. A block that reads it late sees fewer than gridDim.x extra arrivals (the
// first barrier cannot complete without it), so the division still gives this launch's base.
__device__ __forceinline__ unsigned launch_generation(unsigned* count) {
  __shared__ unsigned gen;
  if (threadIdx.x == 0) gen = load_dev_scalar(count) / gridDim.x;
  __syncthreads();
  return gen;
}

__global__ void __launch_bounds__(kBlock) decode(Args a) {
  __shared__ Smem sm;
  unsigned my_gen = launch_generation(a.bar_count);
  decode_step(a, sm, (int)a.positions[0], a.input_ids[0], my_gen, a.stamps, a.trace);
  mk_status(a.status, a.bar_count);
}

static int barriers_per_step(int n_layers) { return 5 * n_layers + 1; }
}  // namespace mk_dense_impl

// ------------------------------------------------------------------ vLLM entry
// Plain C, called from Python through ctypes with raw device pointers. create() and bind() run
// once (outside any graph capture); launch() only enqueues the kernel on the caller's stream, so
// vLLM can capture it in its batch-1 decode graph.

using namespace mk_dense_impl;

extern "C" {

struct mk_vllm_dense_layer {
  const void* q;  // rows of qkv_proj: Q [heads * 128], K [1024], V [1024], each [hidden] BF16
  const void* k;
  const void* v;
  const void* o;
  const void* gate;  // rows of gate_up_proj: gate [ffn], up [ffn]
  const void* up;
  const void* down;
  const float* attn_norm;  // fp32 copies of the BF16 norm weights
  const float* ffn_norm;
  const float* q_norm;  // [128] per-head Q/K norms (Qwen3), else null
  const float* k_norm;
};

// The model this library was built for, for the caller to check against its config.
void mk_vllm_dense_shape(int* hidden, int* ffn, int* heads, int* kv_heads, int* head_dim, int* qk_norm) {
  *hidden = kHidden, *ffn = kFF, *heads = kHeads, *kv_heads = kKV, *head_dim = kDim, *qk_norm = kQKNorm;
}

struct mk_vllm_dense {
  int grid = 0, n_layers = 0;
  Args a{};
  Layer* d_layers = nullptr;
  float *x = nullptr, *q = nullptr, *k = nullptr, *v = nullptr, *ffn = nullptr;
  float *part_acc = nullptr, *part_m = nullptr, *part_l = nullptr;
  uint16_t **kcache = nullptr, **vcache = nullptr;
  unsigned* bar_count = nullptr;
  unsigned* keepalive = nullptr;  // pinned host memory, see mk_keepalive
  unsigned* status = nullptr;     // pinned host memory, see mk_status
};

void mk_vllm_dense_free(mk_vllm_dense* mk);

__global__ void mk_vllm_layout_grid(int* out) { *out = kGrid; }

#define MK_TRY(x, what)                                              \
  do {                                                               \
    hipError_t e_ = (x);                                             \
    if (e_ != hipSuccess) {                                          \
      snprintf(err, err_len, "%s: %s", what, hipGetErrorString(e_)); \
      mk_vllm_dense_free(mk);                                        \
      return nullptr;                                                \
    }                                                                \
  } while (0)

// NULL, with the reason in err, when this GPU cannot run the kernel.
mk_vllm_dense* mk_vllm_dense_create(int n_layers, const mk_vllm_dense_layer* layers, const void* tok_embd,
                                    const float* out_norm, const void* cos_sin, int max_pos, char* err,
                                    int err_len) {
  mk_vllm_dense* mk = new mk_vllm_dense();
  int* d_grid = nullptr;
  MK_TRY(hipMalloc(&d_grid, 4), "alloc");
  hipLaunchKernelGGL(mk_vllm_layout_grid, 1, 1, 0, 0, d_grid);
  {
    hipError_t e = hipGetLastError();
    if (e == hipSuccess) e = hipMemcpy(&mk->grid, d_grid, 4, hipMemcpyDeviceToHost);
    (void)hipFree(d_grid);
    MK_TRY(e, "layout query (is the library built for this GPU?)");
  }
  // The grid barrier needs every block resident at once.
  int per_mp = 0, mps = 0;
  MK_TRY(hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_mp, decode, kBlock, 0), "occupancy");
  MK_TRY(hipDeviceGetAttribute(&mps, hipDeviceAttributeMultiprocessorCount, 0), "multiprocessor count");
  if (per_mp * mps < mk->grid) {
    snprintf(err, err_len, "%d blocks must be resident, the GPU admits %d per multiprocessor x %d", mk->grid,
             per_mp, mps);
    mk_vllm_dense_free(mk);
    return nullptr;
  }
  mk->n_layers = n_layers;
  std::vector<Layer> hl(n_layers);
  for (int l = 0; l < n_layers; l++) {
    const mk_vllm_dense_layer& s = layers[l];
    hl[l] = {(const u4v*)s.q,  (const u4v*)s.k,    (const u4v*)s.v, (const u4v*)s.o,  (const u4v*)s.gate,
             (const u4v*)s.up, (const u4v*)s.down, s.attn_norm,      s.ffn_norm, s.q_norm, s.k_norm};
    if (kQKNorm && (!s.q_norm || !s.k_norm)) {
      snprintf(err, err_len, "layer %d: this model needs the per-head Q/K norm weights", l);
      mk_vllm_dense_free(mk);
      return nullptr;
    }
  }
  MK_TRY(hipMalloc(&mk->d_layers, hl.size() * sizeof(Layer)), "alloc");
  MK_TRY(hipMemcpy(mk->d_layers, hl.data(), hl.size() * sizeof(Layer), hipMemcpyHostToDevice), "upload");
  MK_TRY(hipMalloc(&mk->x, kHidden * 4), "alloc");
  MK_TRY(hipMalloc(&mk->q, kHidden * 4), "alloc");
  MK_TRY(hipMalloc(&mk->k, kKV * kDim * 4), "alloc");
  MK_TRY(hipMalloc(&mk->v, kKV * kDim * 4), "alloc");
  MK_TRY(hipMalloc(&mk->ffn, kFF * 4), "alloc");
  MK_TRY(hipMalloc(&mk->part_acc, (size_t)kHeads * kMaxSplits * kDim * 4), "alloc");
  MK_TRY(hipMalloc(&mk->part_m, kHeads * kMaxSplits * 4), "alloc");
  MK_TRY(hipMalloc(&mk->part_l, kHeads * kMaxSplits * 4), "alloc");
  MK_TRY(hipMalloc(&mk->kcache, n_layers * sizeof(uint16_t*)), "alloc");
  MK_TRY(hipMalloc(&mk->vcache, n_layers * sizeof(uint16_t*)), "alloc");
  MK_TRY(hipMalloc(&mk->bar_count, 8), "alloc");
  MK_TRY(hipMemset(mk->bar_count, 0, 8), "memset");
  if (!getenv("RDNA_MK_NO_KEEPALIVE"))
    MK_TRY(hipHostMalloc((void**)&mk->keepalive, 64, hipHostMallocCoherent), "pinned alloc");
  MK_TRY(hipHostMalloc((void**)&mk->status, 64, hipHostMallocCoherent), "pinned alloc");
  *mk->status = 0;

  Args& a = mk->a;
  a.n_layers = n_layers;
  a.max_pos = max_pos;
  a.layers = mk->d_layers;
  a.emb = (const uint16_t*)tok_embd;
  a.out_norm = out_norm;
  a.cos_sin = (const uint16_t*)cos_sin;
  a.x = mk->x;
  a.q = mk->q;
  a.k = mk->k;
  a.v = mk->v;
  a.ffn = mk->ffn;
  a.part_acc = mk->part_acc;
  a.part_m = mk->part_m;
  a.part_l = mk->part_l;
  a.kcache = mk->kcache;
  a.vcache = mk->vcache;
  a.bar_count = mk->bar_count;
  a.keepalive = mk->keepalive;
  a.status = mk->status;
  return mk;
}

// The per-layer halves of vLLM's kv_cache tensors ([2, blocks, block_size, kv heads, dim] BF16).
// Returns 0, or nonzero if block_size is not a power of two of at least 16.
int mk_vllm_dense_bind(mk_vllm_dense* mk, void* const* k_cache, void* const* v_cache, int block_size) {
  int shift = 0;
  while ((1 << shift) < block_size) shift++;
  if ((1 << shift) != block_size || block_size < 16) return 1;
  mk->a.block_shift = shift;
  if (hipMemcpy(mk->kcache, k_cache, mk->n_layers * sizeof(void*), hipMemcpyHostToDevice) != hipSuccess) return 2;
  if (hipMemcpy(mk->vcache, v_cache, mk->n_layers * sizeof(void*), hipMemcpyHostToDevice) != hipSuccess) return 2;
  return 0;
}

// Enqueues one decode step on `stream`: reads input_ids[0], positions[0] and block_table[0][:]
// on the device and writes the final hidden state (BF16 [hidden]) to out_hidden.
int mk_vllm_dense_launch(mk_vllm_dense* mk, const int* input_ids, const long long* positions,
                         const int* block_table, void* out_hidden, hipStream_t stream) {
  Args a = mk->a;
  a.input_ids = input_ids;
  a.positions = positions;
  a.block_table = block_table;
  a.out_hidden = (uint16_t*)out_hidden;
  hipLaunchKernelGGL(decode, mk->grid, kBlock, 0, stream, a);
  return hipGetLastError() == hipSuccess ? 0 : 1;
}

// The grid barrier's watchdog flag (nonzero: a step timed out and its output is garbage).
int mk_vllm_dense_status(mk_vllm_dense* mk) { return mk->status ? (int)*mk->status : 0; }

int mk_vllm_dense_max_ctx() { return kMaxChunk * kMaxSplits; }
int mk_vllm_dense_grid(mk_vllm_dense* mk) { return mk->grid; }

void mk_vllm_dense_free(mk_vllm_dense* mk) {
  if (!mk) return;
  for (void* p : {(void*)mk->d_layers, (void*)mk->x, (void*)mk->q, (void*)mk->k, (void*)mk->v, (void*)mk->ffn,
                  (void*)mk->part_acc, (void*)mk->part_m, (void*)mk->part_l, (void*)mk->kcache, (void*)mk->vcache,
                  (void*)mk->bar_count})
    if (p) (void)hipFree(p);
  for (void* p : {(void*)mk->keepalive, (void*)mk->status})
    if (p) (void)hipHostFree(p);
  delete mk;
}

}  // extern "C"
