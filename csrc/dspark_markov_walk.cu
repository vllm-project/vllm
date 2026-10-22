// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// The DSpark Markov walk in one persistent launch, int8 W2 kept on chip
// (Hopper, sm_90a).
//
// Adapted from markov_walk_i8s.cu of Xinwei Qiang's int8 DSpark Markov walk
// kernels (https://github.com/tie-pilot-qxw/dspark-markov-walk), which carry
// this notice:
//
//   MIT License
//
//   Copyright (c) 2026 tie-pilot-qxw
//
//   Permission is hereby granted, free of charge, to any person obtaining a
//   copy of this software and associated documentation files (the
//   "Software"), to deal in the Software without restriction, including
//   without limitation the rights to use, copy, modify, merge, publish,
//   distribute, sublicense, and/or sell copies of the Software, and to permit
//   persons to whom the Software is furnished to do so, subject to the
//   following conditions:
//
//   The above copyright notice and this permission notice shall be included
//   in all copies or substantial portions of the Software.
//
//   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
//   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
//   NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
//   DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
//   OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
//   USE OR OTHER DEALINGS IN THE SOFTWARE.
//
// What it computes, for every request b of the batch:
//
//   prev = anchor[b]
//   for k in range(num_steps):
//     logit = bf16(base[b, k] + W2 @ W1[prev])          # [V]
//     tok   = argmax(logit)  or  a sample of softmax(logit / T[b])
//     prev  = tok
//
// The eager loop launches a full-vocabulary GEMV, an add and a sampler per
// step and reads all of W2 from HBM every step. Here one cooperative grid
// (one CTA per SM) runs every step:
//
// * W2 is int8 with one fp32 scale per row. Each CTA owns kTiles 64-row tiles:
//   kRes of them stay in SMEM for the whole call, the rest stream from L2
//   through a kRing-slot TMA ring every step. W1[prev] is stored as two int8
//   planes, u ~= s_hi * q_hi + s_lo * q_lo with s_lo = s_hi / 256, so both dot
//   products are exact int32 sums on int8 tensor cores (wgmma m64n64k32, u as
//   the register A operand, W2 as the SMEM B operand). Nothing is dequantized.
// * The logits are rounded to bf16 once. Argmax and sampler read those values
//   and the same bits go to the draft-logit cache, so the rejection sampler
//   sees exactly the distribution a sampled token was drawn from.
// * Each step ends with one global argmax per request: every CTA folds its
//   best (order-preserving fp32 value, inverted row) key into a {key, count}
//   pair with atomicMax and a release add, then polls the pair. Ties go to
//   the smallest row, as torch.argmax. The pairs of a call alternate between
//   two sets by call parity, so nothing needs a memset between CUDA graph
//   replays.
// * Sampling is exact up to floating-point error: inside each 8-row item the
//   row is drawn by inverse CDF, across items by Gumbel-max on the item's
//   log2 mass. The noise is Philox keyed by the request's seed and counted by
//   (item, step, position), so a seeded request replays its draws.
//
// Contract with the caller (vllm/v1/worker/gpu/spec_decode/dspark/):
// * base: bf16 [nb, kb >= num_steps, ld] contiguous, ld % 8 == 0. Rows past
//   valid_rows (the last CTA's padding) are never read or written.
// * anchor of request b: anchor_ids[b * anchor_stride].
// * idx_mapping[b * num_steps]: the request-state index of request b, < 0 for
//   a padding row. temperature / seeds are indexed by it. temperature is
//   optional: without it every request is greedy.
// * pos[b * num_steps + k]: the position that keys the draw of (b, k).
// * tokens[b * tokens_stride + k] receives the walk.
// * draft_logits (optional): bf16 [num_reqs_max, cols, V], unit stride in the
//   last dim. Row (idx, k) receives the logits of every sampling request
//   (temperature > 0). Greedy requests leave their rows untouched.
// * state: int64 [kStateU64], 128-B aligned, zeroed once and kept.

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include "core/registration.h"
#include "torch_utils.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <optional>

namespace vllm {
namespace dspark_markov_walk {

using u64 = unsigned long long;

constexpr int kMaxSteps = 16;
constexpr int kMaxReqs = 64;
constexpr int kThreads = 512;
constexpr int kWarpGroups = 4;
constexpr int kBlockReqs = 32;  // requests per warpgroup pass (wgmma M = 64)
constexpr int kTileRows = 64;   // W2 rows per tile (wgmma N)
constexpr int kTileBytes = kTileRows * 256;
constexpr int kChunkBytes = 2048;  // one k32 chunk of a tile
constexpr int kW1RowBytes = 528;   // fragment-ordered q_hi | q_lo, s_hi, s_lo
constexpr int kRes = 11;           // SMEM-resident tiles per CTA
constexpr int kRing = 2;           // TMA ring slots for the streamed tiles
constexpr int kPairU64 = 16;       // one {key, count} pair per 128-B line
constexpr int kSetU64 = kMaxSteps * kMaxReqs * kPairU64;
constexpr int kStateU64 = 16 + 2 * kSetU64;
constexpr unsigned kNoiseTag = 0x4D57A1Bu;

// ---- memory-model helpers
__device__ __forceinline__ u64 ld_relaxed(const u64* p) {
  u64 v;
  asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];"
               : "=l"(v)
               : "l"(p)
               : "memory");
  return v;
}
// {key, count} in one 16-B load
__device__ __forceinline__ ulonglong2 ld_relaxed_v2(const u64* p) {
  ulonglong2 v;
  asm volatile("ld.relaxed.gpu.global.v2.u64 {%0, %1}, [%2];"
               : "=l"(v.x), "=l"(v.y)
               : "l"(p)
               : "memory");
  return v;
}
__device__ __forceinline__ void st_relaxed(u64* p, u64 v) {
  asm volatile("st.relaxed.gpu.global.u64 [%0], %1;" ::"l"(p), "l"(v)
               : "memory");
}
__device__ __forceinline__ void red_add_release(u64* p, u64 v) {
  asm volatile("red.release.gpu.global.add.u64 [%0], %1;" ::"l"(p), "l"(v)
               : "memory");
}

// order-preserving fp32 in the high word, inverted row in the low word:
// the max key is the best value, ties go to the smallest row
__device__ __forceinline__ u64 pack_key(float v, unsigned row) {
  unsigned b = __float_as_uint(v);
  b = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
  return (static_cast<u64>(b) << 32) | static_cast<u64>(0xFFFFFFFFu - row);
}
__device__ __forceinline__ unsigned key_row(u64 key) {
  return 0xFFFFFFFFu - static_cast<unsigned>(key & 0xFFFFFFFFull);
}

// Philox4x32-10
__device__ __forceinline__ uint4 philox(uint4 c, uint2 k) {
#pragma unroll
  for (int i = 0; i < 10; ++i) {
    const unsigned hi0 = __umulhi(0xD2511F53u, c.x), lo0 = 0xD2511F53u * c.x;
    const unsigned hi1 = __umulhi(0xCD9E8D57u, c.z), lo1 = 0xCD9E8D57u * c.z;
    c = make_uint4(hi1 ^ c.y ^ k.x, lo1, hi0 ^ c.w ^ k.y, lo0);
    k.x += 0x9E3779B9u;
    k.y += 0xBB67AE85u;
  }
  return c;
}

// lg2.approx without the denormal fix-up: every argument below is normal, 0
// or inf, where the ftz variant returns the same bits
__device__ __forceinline__ float lg2_ftz(float x) {
  float y;
  asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float ex2_ftz(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
// base-2 Gumbel up to a constant: -log2(-log2(1 - v)), small v by series
__device__ __forceinline__ float gumbel2(unsigned x) {
  const float v = (__uint2float_rn(x) + 0.5f) * 2.3283064365386963e-10f;
  const float e = v < 0.0078125f
                      ? v * 1.4426950408889634f *
                            (1.f + v * (0.5f + v * (0.33333334f + v * 0.25f)))
                      : -lg2_ftz(1.f - v);
  return -lg2_ftz(e);
}

// An 8-row run of bf16 logits as 4 packed pairs (pair i = rows 2i, 2i + 1):
// its max in both halves, and the first row holding a value
__device__ __forceinline__ unsigned bf16x8_max(unsigned p0, unsigned p1,
                                               unsigned p2, unsigned p3) {
  unsigned m;
  asm("{\n\t.reg .b32 a, b;\n\t"
      "max.bf16x2 a, %1, %2;\n\tmax.bf16x2 b, %3, %4;\n\tmax.bf16x2 a, a, "
      "b;\n\t"
      "prmt.b32 b, a, a, 0x1032;\n\tmax.bf16x2 %0, a, b;\n\t}"
      : "=r"(m)
      : "r"(p0), "r"(p1), "r"(p2), "r"(p3));
  return m;
}
__device__ __forceinline__ int bf16x8_first_eq(unsigned p0, unsigned p1,
                                               unsigned p2, unsigned p3,
                                               unsigned m) {
  int qi;
  asm("{\n\t.reg .pred e0, e1, e2, e3, e4, e5, e6, e7;\n\t"
      "setp.eq.bf16x2 e0|e1, %1, %5;\n\tsetp.eq.bf16x2 e2|e3, %2, %5;\n\t"
      "setp.eq.bf16x2 e4|e5, %3, %5;\n\tsetp.eq.bf16x2 e6|e7, %4, %5;\n\t"
      "mov.b32 %0, 7;\n\t@e6 mov.b32 %0, 6;\n\t@e5 mov.b32 %0, 5;\n\t"
      "@e4 mov.b32 %0, 4;\n\t@e3 mov.b32 %0, 3;\n\t@e2 mov.b32 %0, 2;\n\t"
      "@e1 mov.b32 %0, 1;\n\t@e0 mov.b32 %0, 0;\n\t}"
      : "=r"(qi)
      : "r"(p0), "r"(p1), "r"(p2), "r"(p3), "r"(m));
  return qi;
}

// ---- loads and stores
__device__ __forceinline__ uint4 ldg_nc(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}
__device__ __forceinline__ uint4 ldg_hint(const void* p, u64 pol) {
  uint4 v;
  asm volatile("ld.global.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p), "l"(pol));
  return v;
}
__device__ __forceinline__ void stg_hint(void* p, uint4 v, u64 pol) {
  asm volatile(
      "st.global.L2::cache_hint.v4.u32 [%0], {%1, %2, %3, %4}, %5;" ::"l"(p),
      "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w), "l"(pol)
      : "memory");
}
__device__ __forceinline__ u64 evict_first_policy() {
  u64 pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;"
               : "=l"(pol));
  return pol;
}
__device__ __forceinline__ u64 evict_last_policy() {
  u64 pol;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;"
               : "=l"(pol));
  return pol;
}

// ---- mbarrier + TMA bulk copies
__device__ __forceinline__ unsigned smem_u32(const void* p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(unsigned addr, unsigned count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(addr), "r"(count)
               : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(unsigned addr, unsigned bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(addr),
      "r"(bytes)
      : "memory");
}
// CTA-scope acquire: only this CTA's TMA wrote the SMEM (a cluster-scope wait
// would also invalidate L1 on every try)
__device__ __forceinline__ void mbar_wait(unsigned addr, unsigned parity) {
  unsigned ok = 0;
  do {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, p;\n\t}"
        : "=r"(ok)
        : "r"(addr), "r"(parity)
        : "memory");
  } while (!ok);
}
__device__ __forceinline__ void bulk_g2s(unsigned dst, const void* src,
                                         unsigned bytes, unsigned mbar,
                                         u64 pol) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes."
      "L2::cache_hint [%0], [%1], %2, [%3], %4;" ::"r"(dst),
      "l"(src), "r"(bytes), "r"(mbar), "l"(pol)
      : "memory");
}
__device__ __forceinline__ int ld_volatile_shared(const int* p) {
  int v;
  asm volatile("ld.volatile.shared.s32 %0, [%1];"
               : "=r"(v)
               : "r"(smem_u32(p))
               : "memory");
  return v;
}

// ---- wgmma m64n64k32 s8 x s8 -> s32, A from registers
__device__ __forceinline__ void wg_fence() {
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void wg_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void wg_wait0() {
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
}
// K-major, no swizzle, LBO 128 B, SBO 256 B; the high word is the constant SBO
constexpr unsigned kDescHi = 256 >> 4;
__device__ __forceinline__ unsigned wg_desc_lo(unsigned addr) {
  return ((addr & 0x3FFFF) >> 4) | ((128 >> 4) << 16);
}
#define DSPARK_WG_D32                                                       \
  "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, " \
  "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, "  \
  "%30, %31}"
#define DSPARK_WG_D32_OUT(d)                                              \
  "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), \
      "+r"(d[6]), "+r"(d[7]), "+r"(d[8]), "+r"(d[9]), "+r"(d[10]),        \
      "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]),    \
      "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]),    \
      "+r"(d[21]), "+r"(d[22]), "+r"(d[23]), "+r"(d[24]), "+r"(d[25]),    \
      "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]),    \
      "+r"(d[31])
// d (+)= A B; acc == 0 overwrites
__device__ __forceinline__ void wgmma_rs(int (&d)[32], const uint4& a,
                                         unsigned blo, int acc) {
  asm volatile(
      "{\n.reg .pred p;\n.reg .b64 bd;\nsetp.ne.b32 p, %37, 0;\n"
      "mov.b64 bd, {%36, %38};\n"
      "wgmma.mma_async.sync.aligned.m64n64k32.s32.s8.s8 " DSPARK_WG_D32
      ", {%32, %33, %34, %35}, bd, p;\n}"
      : DSPARK_WG_D32_OUT(d)
      : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(blo), "r"(acc),
        "r"(kDescHi));
}

// Layout of one (32-request block, 64-row tile) product: M row 16 w + g is
// q_hi of request 8 w + g of the block, row 16 w + g + 8 its q_lo (the host
// stores W1 rows in that fragment order, so the gather is 8 x LDG.128 into the
// A registers). The host permutes every tile's rows so that accumulator column
// n = 8 i + 2 tig + e is tile row 8 tig + 2 i + e (i < 4) or 32 + 8 tig +
// 2 (i - 4) + e: a thread owns one request and two aligned 8-row runs.
//
// Ring: the streamed tiles of a call form one FIFO q = step * kStr + rank of
// the tile among the streamed ones; each warpgroup consumes its tiles in
// increasing q. The warp whose arrival completes the consumption of q refills
// the slot with q + kRing: it writes the slot's tag first, then arms the
// barrier and issues the copy, and a consumer waits for the tag before the
// parity, so a warpgroup never mistakes fill f - 2 for fill f.
template <int kTiles>
__global__ void __launch_bounds__(kThreads, 1) markov_walk_kernel(
    const uint8_t* __restrict__ w2_res, const uint8_t* __restrict__ w2_str,
    const float* __restrict__ row_scale, const uint8_t* __restrict__ w1f,
    const __nv_bfloat16* __restrict__ base,
    const int32_t* __restrict__ anchor_ids, int anchor_stride,
    const int64_t* __restrict__ idx_mapping,
    const float* __restrict__ temperature, const int64_t* __restrict__ seeds,
    const int64_t* __restrict__ pos, int64_t* __restrict__ tokens,
    int tokens_stride, __nv_bfloat16* __restrict__ draft_logits,
    int64_t dl_stride0, int64_t dl_stride1, u64* __restrict__ state, int nb,
    int num_steps, int kb, int ld, int valid_rows, unsigned stream_mask) {
  constexpr int kRowsCta = kTiles * kTileRows;
  constexpr int kStr = kTiles - kRes;
  static_assert(kRing >= 1 && kRing <= kStr, "ring depth");
  extern __shared__ __align__(128) unsigned char smem_raw[];
  unsigned char* res_s = smem_raw;                    // [kRes][16 KiB]
  unsigned char* ring_s = res_s + kRes * kTileBytes;  // [kRing][16 KiB]
  float* scale_s = reinterpret_cast<float*>(ring_s + kRing * kTileBytes);
  __shared__ __align__(8) u64 mbar_res[kRes + 1];  // + 1: the row scales
  __shared__ __align__(8) u64 mbar_full[kRing];
  __shared__ int slot_q[kRing];    // FIFO index the slot holds or awaits
  __shared__ int slot_cnt[kRing];  // monotone consumer arrivals
  __shared__ u64 wbest_s[kMaxReqs][kWarpGroups];
  __shared__ unsigned tok_s[kMaxReqs];
  // per sampling request and step: Philox key and counter words, and the
  // draft-logit row (kept here, not in registers, for the tile loop)
  __shared__ uint4 noise_s[kMaxReqs];
  __shared__ __nv_bfloat16* dl_s[kMaxReqs];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int wg = warp >> 2, wq = warp & 3, g = lane >> 2, tig = lane & 3;
  const int row_base = blockIdx.x * kRowsCta;

  const u64 round = ld_relaxed(state);
  const int set = static_cast<int>(round & 1);
  u64* pairs = state + 16 + set * kSetU64;
  auto pair = [&](int k, int b) {
    return pairs + kPairU64 * (k * kMaxReqs + b);
  };
  // clear the set the next call uses (the previous call, which used it, has
  // finished: calls are serialized on one stream)
  if (blockIdx.x == 0)
    for (int i = threadIdx.x; i < kMaxSteps * kMaxReqs * 2; i += kThreads)
      st_relaxed(
          state + 16 + (set ^ 1) * kSetU64 + (i >> 1) * kPairU64 + (i & 1), 0);

  // ---- work split: nblk 32-request blocks, wpb warpgroups per block
  const int nblk = (nb + kBlockReqs - 1) / kBlockReqs;
  const int wpb = kWarpGroups / nblk;
  const int blk = wg / wpb, phase = wg % wpb;
  const int b = blk * kBlockReqs + 8 * wq + g;  // this thread's request
  const bool live = b < nb;
  // Run 0 (tile rows 8 tig .. +8) / run 1 (32 + 8 tig .. +8) of tile t holds
  // vocabulary rows iff t < lim0 / lim1 (valid_rows % 8 == 0). Dead requests
  // get 0, which also skips their base loads.
  const int vloc = valid_rows - row_base;
  const int lim0 =
      live ? min(kTiles, max(0, vloc - 8 * tig + kTileRows - 1) / kTileRows)
           : 0;
  const int lim1 =
      live
          ? min(kTiles, max(0, vloc - 32 - 8 * tig + kTileRows - 1) / kTileRows)
          : 0;
  const int consumers = 4 * nblk;   // ring arrivals per fill (4 warps / WG)
  const int nq = kStr * num_steps;  // streamed fills per call
  auto str_idx = [&](int t) { return __popc(stream_mask & ((1u << t) - 1u)); };

  // ---- per request: sampling parameters (temperature <= 0: greedy)
  auto request_temp = [&](int i) {
    const int64_t req = idx_mapping[static_cast<int64_t>(i) * num_steps];
    return temperature != nullptr && req >= 0 ? temperature[req] : 0.f;
  };
  auto stage_step = [&](int i, int k) {  // request i's noise and row of step k
    const int64_t req = idx_mapping[static_cast<int64_t>(i) * num_steps];
    if (request_temp(i) <= 0.f) return;
    const u64 s = static_cast<u64>(seeds[req]);
    const u64 p =
        static_cast<u64>(pos[static_cast<int64_t>(i) * num_steps + k]);
    noise_s[i] = make_uint4(
        static_cast<unsigned>(s), static_cast<unsigned>(s >> 32),
        static_cast<unsigned>(p), static_cast<unsigned>(p >> 32) ^ kNoiseTag);
    dl_s[i] = draft_logits != nullptr
                  ? draft_logits + req * dl_stride0 + k * dl_stride1
                  : nullptr;
  };
  float inv_t = 0.f;
  if (live) {
    const float t = request_temp(b);
    inv_t = t > 0.f ? 1.f / t : 0.f;
  }
  const float t2 = inv_t * 1.4426950408889634f;

  // ---- once per call: row scales and resident tiles by TMA, ring prefill
  const u64 pol_last = evict_last_policy(), pol_first = evict_first_policy();
  auto res_mbar = [&](int i) { return smem_u32(&mbar_res[i]); };
  auto issue_fill = [&](int q) {
    const int slot = q % kRing;
    slot_q[slot] = q;
    const unsigned mb = smem_u32(&mbar_full[slot]);
    mbar_expect_tx(mb, kTileBytes);
    // the last step's copies are demoted for whatever runs next
    bulk_g2s(smem_u32(ring_s + slot * kTileBytes),
             w2_str + (static_cast<size_t>(blockIdx.x) * kStr + q % kStr) *
                          kTileBytes,
             kTileBytes, mb, q >= nq - kStr ? pol_first : pol_last);
  };
  if (threadIdx.x == 0) {
    for (int i = 0; i <= kRes; ++i) mbar_init(res_mbar(i), 1);
    for (int s = 0; s < kRing; ++s) {
      mbar_init(smem_u32(&mbar_full[s]), 1);
      slot_cnt[s] = 0;
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    mbar_expect_tx(res_mbar(kRes), kRowsCta * 4);
    bulk_g2s(smem_u32(scale_s), row_scale + row_base, kRowsCta * 4,
             res_mbar(kRes), pol_first);
    for (int q = 0; q < kRing && q < nq; ++q) issue_fill(q);
    for (int i = 0; i < kRes; ++i) {
      mbar_expect_tx(res_mbar(i), kTileBytes);
      bulk_g2s(
          smem_u32(res_s + i * kTileBytes),
          w2_res + (static_cast<size_t>(blockIdx.x) * kRes + i) * kTileBytes,
          kTileBytes, res_mbar(i), pol_first);
    }
  }
  for (int i = threadIdx.x; i < nb; i += kThreads) {
    const int a = anchor_ids[static_cast<int64_t>(i) * anchor_stride];
    tok_s[i] = static_cast<unsigned>(min(max(a, 0), valid_rows - 1));
    stage_step(i, 0);
  }
  __syncthreads();  // barriers initialized; the copies are still in flight
  mbar_wait(res_mbar(kRes), 0);

  const unsigned res_desc0 = wg_desc_lo(smem_u32(res_s));
  const unsigned ring_desc0 = wg_desc_lo(smem_u32(ring_s));
  // base of this thread's next tile (bf16 x 8, two runs)
  uint4 nb0 = make_uint4(0, 0, 0, 0), nb1 = make_uint4(0, 0, 0, 0);

  for (int k = 0; k < num_steps; ++k) {
    // u fragments of this thread's request (dead requests: zeros, so the
    // warpgroup's MMA stays well defined)
    uint4 a[8];
    float s_lo = 0.f;
    if (live) {
      const uint8_t* src = w1f + static_cast<size_t>(tok_s[b]) * kW1RowBytes;
#pragma unroll
      for (int j = 0; j < 8; ++j) a[j] = ldg_nc(src + j * 64 + tig * 16);
      s_lo = __ldg(reinterpret_cast<const float*>(src + 516));
    } else {
#pragma unroll
      for (int j = 0; j < 8; ++j) a[j] = make_uint4(0, 0, 0, 0);
    }
    // element offset of (b, k, run 0 of tile 0); < 2^31, checked on the host
    const int ob = (b * kb + k) * ld + row_base + 8 * tig;
    if (k == 0) {  // later steps load their first tile under the exchange
      if (phase < lim0)
        nb0 = ldg_hint(base + ob + phase * kTileRows, pol_first);
      if (phase < lim1)
        nb1 = ldg_hint(base + ob + phase * kTileRows + 32, pol_first);
    }
    float cur_v = -INFINITY;
    int cur_r = -1;

#pragma unroll 2
    for (int t = phase; t < kTiles; t += wpb) {
      const bool streamed = (stream_mask >> t) & 1u;
      unsigned bd;
      int q = 0;
      if (streamed) {
        q = k * kStr + str_idx(t);
        const int slot = q % kRing;
        while (ld_volatile_shared(&slot_q[slot]) != q) {
        }
        mbar_wait(smem_u32(&mbar_full[slot]),
                  static_cast<unsigned>((q / kRing) & 1));
        bd = ring_desc0 + slot * (kTileBytes >> 4);
      } else {
        const int ri = t - str_idx(t);
        // first use of a resident tile waits for its own copy
        if (k == 0) mbar_wait(res_mbar(ri), 0);
        bd = res_desc0 + ri * (kTileBytes >> 4);
      }
      int d[32];
      wg_fence();
#pragma unroll
      for (int j = 0; j < 8; ++j)
        wgmma_rs(d, a[j], bd + j * (kChunkBytes >> 4), j);
      wg_commit();
      // this tile's base came one tile ago; the next one's loads under the MMA
      const int o = ob + t * kTileRows;
      const uint4 bb0 = nb0, bb1 = nb1;
      if (t + wpb < lim0) nb0 = ldg_hint(base + o + wpb * kTileRows, pol_first);
      if (t + wpb < lim1)
        nb1 = ldg_hint(base + o + wpb * kTileRows + 32, pol_first);
      wg_wait0();
      // keep every read of d below the wait (d is written asynchronously)
      asm volatile("" : DSPARK_WG_D32_OUT(d)::"memory");
      if (streamed && lane == 0) {  // this warp is done with the slot
        const int slot = q % kRing;
        const int done = atomicAdd(&slot_cnt[slot], 1) + 1;
        if (done == (q / kRing + 1) * consumers && q + kRing < nq)
          issue_fill(q + kRing);
      }
      if (!live) continue;
      // bias = s_lo * (256 d_hi + d_lo) (exact int32), logit = row_scale *
      // bias + base, rounded to bf16 once
      const float4 sa =
          *reinterpret_cast<const float4*>(scale_s + t * kTileRows + 8 * tig);
      const float4 sb = *reinterpret_cast<const float4*>(
          scale_s + t * kTileRows + 8 * tig + 4);
      const float4 sc = *reinterpret_cast<const float4*>(
          scale_s + t * kTileRows + 32 + 8 * tig);
      const float4 sd = *reinterpret_cast<const float4*>(
          scale_s + t * kTileRows + 32 + 8 * tig + 4);
      const float sr[16] = {sa.x, sa.y, sa.z, sa.w, sb.x, sb.y, sb.z, sb.w,
                            sc.x, sc.y, sc.z, sc.w, sd.x, sd.y, sd.z, sd.w};
      const unsigned bw[8] = {bb0.x, bb0.y, bb0.z, bb0.w,
                              bb1.x, bb1.y, bb1.z, bb1.w};
      float lg[16];  // [run][element]: element 2 (i % 4) + e of run i / 4
#pragma unroll
      for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int idx = 8 * (i >> 2) + 2 * (i & 3) + e;
          const float bias =
              s_lo * static_cast<float>(d[4 * i + e] * 256 + d[4 * i + 2 + e]);
          const unsigned w = bw[idx >> 1];
          lg[idx] = fmaf(sr[idx], bias,
                         __uint_as_float(e ? (w & 0xFFFF0000u) : (w << 16)));
        }
      unsigned pk[8];  // the 16 bf16 logits, packed in memory order
#pragma unroll
      for (int p2 = 0; p2 < 8; ++p2) {
        const __nv_bfloat162 h =
            __floats2bfloat162_rn(lg[2 * p2], lg[2 * p2 + 1]);
        pk[p2] = *reinterpret_cast<const unsigned*>(&h);
      }
      const int r = row_base + t * kTileRows + 8 * tig;  // run 0 rows r .. +8
      if (row_base + (t + 1) * kTileRows > valid_rows) {
        // the last CTA's padding runs: bf16 -inf
#pragma unroll
        for (int run = 0; run < 2; ++run)
          if (r + 32 * run >= valid_rows) {
#pragma unroll
            for (int qq = 0; qq < 4; ++qq) pk[4 * run + qq] = 0xFF80FF80u;
          }
      }
      if (inv_t > 0.f) {
        __nv_bfloat16* dl = dl_s[b];
        if (dl != nullptr) {
          if (t < lim0)
            stg_hint(dl + r, make_uint4(pk[0], pk[1], pk[2], pk[3]), pol_first);
          if (t < lim1)
            stg_hint(dl + r + 32, make_uint4(pk[4], pk[5], pk[6], pk[7]),
                     pol_first);
        }
#pragma unroll
        for (int p2 = 0; p2 < 8; ++p2) {
          lg[2 * p2] = __uint_as_float(pk[p2] << 16);
          lg[2 * p2 + 1] = __uint_as_float(pk[p2] & 0xFFFF0000u);
        }
        // run 0 and run 1 draw from one Philox block: (x, y) and (z, w)
        const uint4 nz = noise_s[b];
        const uint4 xr =
            philox(make_uint4(static_cast<unsigned>(r >> 3),
                              static_cast<unsigned>(k), nz.z, nz.w),
                   make_uint2(nz.x, nz.y));
#pragma unroll
        for (int run = 0; run < 2; ++run) {
          const float* l8 = lg + 8 * run;
          const float m =
              __uint_as_float(bf16x8_max(pk[4 * run], pk[4 * run + 1],
                                         pk[4 * run + 2], pk[4 * run + 3])
                              << 16);
          const float mz = m * t2;
          float cs[8], c = 0.f;
#pragma unroll
          for (int qq = 0; qq < 8; ++qq) {
            c += ex2_ftz(fmaf(l8[qq], t2, -mz));
            cs[qq] = c;
          }
          const unsigned xu = run == 0 ? xr.x : xr.z;
          const unsigned xg = run == 0 ? xr.y : xr.w;
          const float u =
              (static_cast<float>(xu >> 8) + 0.5f) * 5.9604644775390625e-8f;
          const float tt = u * c;  // < c
          int qi = 7;
#pragma unroll
          for (int qq = 6; qq >= 0; --qq) qi = tt < cs[qq] ? qq : qi;
          const float key_v = mz + lg2_ftz(c) + gumbel2(xg);
          if (key_v > cur_v) {
            cur_v = key_v;
            cur_r = r + 32 * run + qi;
          }
        }
      } else {  // greedy on the packed values
#pragma unroll
        for (int run = 0; run < 2; ++run) {
          const unsigned* p8 = pk + 4 * run;
          const unsigned m2 = bf16x8_max(p8[0], p8[1], p8[2], p8[3]);
          const int qi = bf16x8_first_eq(p8[0], p8[1], p8[2], p8[3], m2);
          const float m = __uint_as_float(m2 << 16);
          if (m > cur_v) {
            cur_v = m;
            cur_r = r + 32 * run + qi;
          }
        }
      }
    }
    // the 4 tig lanes of a request -> this warpgroup's best for it
    {
      u64 key =
          cur_r >= 0 ? pack_key(cur_v, static_cast<unsigned>(cur_r)) : 0ull;
      key = max(key, __shfl_xor_sync(0xffffffffu, key, 1));
      key = max(key, __shfl_xor_sync(0xffffffffu, key, 2));
      if (tig == 0 && live) wbest_s[b][phase] = key;
    }
    // the next step's first base tile does not depend on the tokens
    if (k + 1 < num_steps) {
      const int o1 = ob + ld + phase * kTileRows;
      if (phase < lim0) nb0 = ldg_hint(base + o1, pol_first);
      if (phase < lim1) nb1 = ldg_hint(base + o1 + 32, pol_first);
    }
    __syncthreads();
    if (warp == 0) {
      // lane holds requests lane and lane + 32
#pragma unroll
      for (int qd = 0; qd < 2; ++qd) {
        const int bq = lane + 32 * qd;
        if (bq < nb) {
          u64 bst = 0ull;
          for (int p = 0; p < wpb; ++p) bst = max(bst, wbest_s[bq][p]);
          atomicMax(pair(k, bq), bst);
          red_add_release(pair(k, bq) + 1, 1);
        }
      }
      // both requests of a lane polled in the same round trip
      const u64 n = gridDim.x;
      ulonglong2 v0 = make_ulonglong2(0, lane < nb ? 0 : n);
      ulonglong2 v1 = make_ulonglong2(0, lane + 32 < nb ? 0 : n);
      do {
        if (v0.y < n) v0 = ld_relaxed_v2(pair(k, lane));
        if (v1.y < n) v1 = ld_relaxed_v2(pair(k, lane + 32));
      } while (v0.y < n || v1.y < n);
      const u64 key[2] = {v0.x, v1.x};
#pragma unroll
      for (int qd = 0; qd < 2; ++qd) {
        const int bq = lane + 32 * qd;
        if (bq < nb) {
          // key 0: no row was a candidate anywhere (every logit NaN or -inf,
          // e.g. a padded CUDA graph row); emit row 0, as torch.argmax
          const unsigned tok = key[qd] != 0ull ? key_row(key[qd]) : 0u;
          tok_s[bq] = tok;
          if (blockIdx.x == 0)
            tokens[static_cast<int64_t>(bq) * tokens_stride + k] = tok;
          if (k + 1 < num_steps) {  // the next gather's W1 row into L2
            stage_step(bq, k + 1);
            const uint8_t* src = w1f + static_cast<size_t>(tok) * kW1RowBytes;
#pragma unroll
            for (int l = 0; l < 4; ++l)
              asm volatile(
                  "prefetch.global.L2::evict_last [%0];" ::"l"(src + 128 * l));
          }
        }
      }
    }
    __syncthreads();
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) st_relaxed(state, round + 1);
}

template <int kTiles>
void launch(const torch::stable::Tensor& w2_res,
            const torch::stable::Tensor& w2_str,
            const torch::stable::Tensor& row_scale,
            const torch::stable::Tensor& w1f, const torch::stable::Tensor& base,
            const int32_t* anchor_ids, int anchor_stride,
            const int64_t* idx_mapping, const float* temperature,
            const int64_t* seeds, const int64_t* pos, int64_t* tokens,
            int tokens_stride, __nv_bfloat16* draft_logits, int64_t dl_stride0,
            int64_t dl_stride1, u64* state, int nb, int num_steps,
            int valid_rows, unsigned stream_mask, cudaStream_t stream) {
  constexpr int kRowsCta = kTiles * kTileRows;
  constexpr int kStr = kTiles - kRes;
  const int64_t rows_pad = row_scale.numel();
  STD_TORCH_CHECK(rows_pad % kRowsCta == 0,
                  "row_scale size is not a multiple of the CTA rows");
  const int grid = static_cast<int>(rows_pad / kRowsCta);
  STD_TORCH_CHECK(grid <= get_device_prop()->multiProcessorCount,
                  "dspark_markov_walk: grid of ", grid,
                  " CTAs exceeds the SM count");
  STD_TORCH_CHECK(
      w2_res.numel() == static_cast<int64_t>(grid) * kRes * kTileBytes,
      "w2_res size does not match the tiling");
  STD_TORCH_CHECK(
      w2_str.numel() == static_cast<int64_t>(grid) * kStr * kTileBytes,
      "w2_str size does not match the tiling");
  STD_TORCH_CHECK(__builtin_popcount(stream_mask) == kStr &&
                      (kTiles == 32 || (stream_mask >> kTiles) == 0),
                  "stream_mask must stream exactly tiles - ", kRes, " tiles");
  const int kb = static_cast<int>(base.size(1));
  const int ld = static_cast<int>(base.size(2));
  // 32-bit element offsets into base, including the dead lanes of a
  // partially filled 32-request block and the padding tiles
  const int64_t lanes =
      static_cast<int64_t>(kBlockReqs) * ((nb + kBlockReqs - 1) / kBlockReqs);
  STD_TORCH_CHECK(lanes * kb * ld + rows_pad < (int64_t(1) << 31),
                  "base is too large for 32-bit offsets");
  const int smem = (kRes + kRing) * kTileBytes + kRowsCta * 4;
  auto kernel = markov_walk_kernel<kTiles>;
  static bool attr_set = false;
  if (!attr_set) {
    STD_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    attr_set = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(kThreads);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeCooperative;
  attr[0].val.cooperative = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  STD_CUDA_CHECK(cudaLaunchKernelEx(
      &cfg, kernel, reinterpret_cast<const uint8_t*>(w2_res.data_ptr()),
      reinterpret_cast<const uint8_t*>(w2_str.data_ptr()),
      reinterpret_cast<const float*>(row_scale.data_ptr()),
      reinterpret_cast<const uint8_t*>(w1f.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(base.data_ptr()), anchor_ids,
      anchor_stride, idx_mapping, temperature, seeds, pos, tokens,
      tokens_stride, draft_logits, dl_stride0, dl_stride1, state, nb, num_steps,
      kb, ld, valid_rows, stream_mask));
}

}  // namespace dspark_markov_walk
}  // namespace vllm

// See the header for the layouts; the host side that builds them is
// vllm/v1/worker/gpu/spec_decode/dspark/markov_walk.py.
void dspark_markov_walk(
    torch::stable::Tensor const& w2_res, torch::stable::Tensor const& w2_str,
    torch::stable::Tensor const& row_scale, torch::stable::Tensor const& w1f,
    torch::stable::Tensor const& base, torch::stable::Tensor const& anchor_ids,
    int64_t anchor_stride, torch::stable::Tensor const& idx_mapping,
    std::optional<torch::stable::Tensor> temperature,
    torch::stable::Tensor const& seeds, torch::stable::Tensor const& pos,
    torch::stable::Tensor& tokens,
    std::optional<torch::stable::Tensor> draft_logits,
    torch::stable::Tensor& state, int64_t num_steps, int64_t tiles,
    int64_t stream_mask) {
  namespace dmw = vllm::dspark_markov_walk;
  using torch::headeronly::ScalarType;
  STD_TORCH_CHECK(base.dim() == 3 && base.is_contiguous() &&
                      base.scalar_type() == ScalarType::BFloat16,
                  "base must be contiguous bf16 [num_reqs, steps, vocab]");
  const int64_t nb = base.size(0);
  const int64_t ld = base.size(2);
  STD_TORCH_CHECK(nb >= 1 && nb <= dmw::kMaxReqs, "batch must be in [1, ",
                  dmw::kMaxReqs, "]");
  STD_TORCH_CHECK(num_steps >= 1 && num_steps <= dmw::kMaxSteps &&
                      base.size(1) >= num_steps,
                  "num_steps must be in [1, ", dmw::kMaxSteps, "]");
  STD_TORCH_CHECK(ld % 8 == 0, "vocab size must be a multiple of 8");
  const int64_t valid_rows = w1f.size(0);
  STD_TORCH_CHECK(valid_rows == ld && valid_rows <= row_scale.numel(),
                  "W1 rows, W2 rows and the logits' vocab size must match");
  STD_TORCH_CHECK(
      w1f.dim() == 2 && w1f.size(1) == dmw::kW1RowBytes && w1f.is_contiguous(),
      "w1f must be [vocab, 528] bytes");
  STD_TORCH_CHECK(w2_res.is_contiguous() && w2_str.is_contiguous() &&
                      row_scale.is_contiguous() &&
                      row_scale.scalar_type() == ScalarType::Float,
                  "bad W2 layout");
  STD_TORCH_CHECK(anchor_ids.scalar_type() == ScalarType::Int &&
                      anchor_ids.numel() >= (nb - 1) * anchor_stride + 1,
                  "anchor_ids must be int32 with an anchor per request");
  STD_TORCH_CHECK(idx_mapping.scalar_type() == ScalarType::Long &&
                      idx_mapping.is_contiguous() &&
                      idx_mapping.numel() >= nb * num_steps,
                  "idx_mapping must be int64 [num_reqs * steps]");
  STD_TORCH_CHECK(pos.scalar_type() == ScalarType::Long &&
                      pos.is_contiguous() && pos.numel() >= nb * num_steps,
                  "pos must be int64 [num_reqs * steps]");
  STD_TORCH_CHECK(seeds.scalar_type() == ScalarType::Long,
                  "seeds must be int64");
  STD_TORCH_CHECK(tokens.dim() == 2 && tokens.size(0) >= nb &&
                      tokens.size(1) >= num_steps && tokens.stride(1) == 1 &&
                      tokens.scalar_type() == ScalarType::Long,
                  "tokens must be int64 [>= num_reqs, >= steps]");
  STD_TORCH_CHECK(state.scalar_type() == ScalarType::Long &&
                      state.numel() == dmw::kStateU64 &&
                      reinterpret_cast<uintptr_t>(state.data_ptr()) % 128 == 0,
                  "state must be a 128-B aligned int64 [", dmw::kStateU64, "]");
  const float* temp_ptr = nullptr;
  if (temperature.has_value()) {
    STD_TORCH_CHECK(temperature->scalar_type() == ScalarType::Float,
                    "temperature must be fp32");
    temp_ptr = reinterpret_cast<const float*>(temperature->data_ptr());
  }
  __nv_bfloat16* dl_ptr = nullptr;
  int64_t dl_stride0 = 0, dl_stride1 = 0;
  if (draft_logits.has_value()) {
    STD_TORCH_CHECK(
        draft_logits->dim() == 3 &&
            draft_logits->scalar_type() == ScalarType::BFloat16 &&
            draft_logits->stride(2) == 1 &&
            draft_logits->size(1) >= num_steps && draft_logits->size(2) >= ld &&
            draft_logits->stride(0) % 8 == 0 &&
            draft_logits->stride(1) % 8 == 0 &&
            reinterpret_cast<uintptr_t>(draft_logits->data_ptr()) % 16 == 0,
        "draft_logits must be 16-B aligned bf16 [reqs, cols, "
        "vocab] with unit last stride");
    dl_ptr = reinterpret_cast<__nv_bfloat16*>(draft_logits->data_ptr());
    dl_stride0 = draft_logits->stride(0);
    dl_stride1 = draft_logits->stride(1);
  }
  const torch::stable::accelerator::DeviceGuard device_guard(
      base.get_device_index());
  auto stream = get_current_cuda_stream(base.get_device_index());
  const unsigned mask = static_cast<unsigned>(stream_mask);

#define DSPARK_MARKOV_WALK_LAUNCH(TILES)                                    \
  if (tiles == TILES) {                                                     \
    dmw::launch<TILES>(                                                     \
        w2_res, w2_str, row_scale, w1f, base,                               \
        reinterpret_cast<const int32_t*>(anchor_ids.data_ptr()),            \
        static_cast<int>(anchor_stride),                                    \
        reinterpret_cast<const int64_t*>(idx_mapping.data_ptr()), temp_ptr, \
        reinterpret_cast<const int64_t*>(seeds.data_ptr()),                 \
        reinterpret_cast<const int64_t*>(pos.data_ptr()),                   \
        reinterpret_cast<int64_t*>(tokens.mutable_data_ptr()),              \
        static_cast<int>(tokens.stride(0)), dl_ptr, dl_stride0, dl_stride1, \
        reinterpret_cast<dmw::u64*>(state.mutable_data_ptr()),              \
        static_cast<int>(nb), static_cast<int>(num_steps),                  \
        static_cast<int>(valid_rows), mask, stream);                        \
    return;                                                                 \
  }
  DSPARK_MARKOV_WALK_LAUNCH(16)
  DSPARK_MARKOV_WALK_LAUNCH(18)
  DSPARK_MARKOV_WALK_LAUNCH(20)
  DSPARK_MARKOV_WALK_LAUNCH(24)
  DSPARK_MARKOV_WALK_LAUNCH(28)
  DSPARK_MARKOV_WALK_LAUNCH(32)
#undef DSPARK_MARKOV_WALK_LAUNCH
  STD_TORCH_CHECK(false, "dspark_markov_walk: no build for ", tiles,
                  " tiles per CTA");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("dspark_markov_walk", TORCH_BOX(&dspark_markov_walk));
}
