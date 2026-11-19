// Inkling2's hidden-stream short conv fused with the AttnRes mix + RMSNorm of
// the next sublayer, for prefill-sized batches of an already reduced input.
// One persistent CTA per SM walks a contiguous token range. A producer warp
// streams each token's rows (stacked mix sources, conv input, residual inputs)
// into a ring of shared-memory stages with bulk async copies as soon as a
// stage is free, so every row is read from memory once. Each thread of the
// other warps owns kVecs chunks of 8 consecutive channels; the elementwise
// math runs on fp32 pairs, and the row reductions synchronize those warps
// through mbarriers, not CTA barriers.

#include <cuda_bf16.h>

#include <climits>
#include <utility>

#include "../torch_utils.h"

namespace vllm {
namespace inkling2 {
namespace {

using bf16 = __nv_bfloat16;

constexpr int kVec = 8;
constexpr int kPairs = kVec / 2;
constexpr int kTaps = 4;
// Inkling2's hidden size 4608 in chunks of 8 channels.
constexpr int kRowVecs = 576;
constexpr int kRowBytes = kRowVecs * sizeof(uint4);
constexpr int kMaxStack = 8;
constexpr int kMaxStages = 4;
constexpr int kMaxWorld = 8;
// Dynamic shared memory for the stages and the token metadata.
constexpr int kSmemBytes = 227 * 1024 - 4096;

struct ConvMixParams {
  bf16 const* x;  // reduced conv input [haloRows + T, D]
  // Or (Gather) every rank's pushed partial rows, xSources slots of
  // xSourceStride elements, summed in rank order once every rank arrived at
  // barrier flag row gatherSlot + 1.
  int xSources;
  int64_t xSourceStride;
  int numTokens;
  // The tokens are rows [tokenBase, tokenBase + T) of the batch (the
  // metadata); x starts with the conv input of the haloRows before them.
  int tokenBase;
  int haloRows;
  int dim;
  int tokensPerCta;
  int stages;
  int stageRows;
  int blockShift;
  // Short conv over the paged conv-state cache.
  bf16* cache;
  bf16 const* convWeight;  // [D, 4]
  int64_t const* positions;
  int const* seqIdx;
  int64_t const* slots;
  int const* blockTable;
  int const* queryStart;
  int64_t cacheBlockStride;
  int64_t cacheHeadStride;
  int64_t cacheSlotStride;
  int blockTableStride;
  int maxBlocks;
  int streamOffset;
  int streamWidth;
  int numHeads;
  float convScale;
  // Residual update (res + addend) + y; res may alias convOut.
  bf16 const* res;
  bf16 const* resAddend;
  bf16* convOut;
  // AttnRes mix over [*stack, (last +) convOut] and RMSNorm.
  bf16 const* stack;
  int64_t stackStride;
  bf16 const* last;
  float const* queries;    // [Q, D]
  bf16 const* normWeight;  // [Q, D]
  float eps;
  float normEps;
  // [Q, outTokens, D], at rows tokenBase + t: out, or every rank's
  // all-gather buffer (outRanks of them, at outOffset, starting with this
  // rank's). The last CTA to finish then waits until every rank's pushes
  // have landed: it bumps gatherEpoch and exchanges it through barrier flag
  // row gatherSlot at the start of the buffers (`seq_parallel.cu`), and
  // releases the other CTAs.
  bf16* out;
  char* const* outBases;
  int64_t outOffset;
  int outRanks;
  int outTokens;
  int rank;
  int gatherSlot;
  uint32_t* gatherEpoch;
  uint32_t* doneCtas;
  uint32_t* released;
  // The conv-state cache rows later steps read: each request's last
  // tailRows tokens and the last kTaps - 1 tokens before every multiple of
  // align (prefix-cache hit boundaries). Their owners push the conv inputs to
  // the other ranks' staging rows (at stagingOffset), which every rank
  // copies to its cache after the barrier.
  int64_t stagingOffset;
  int tailRows;
  int align;
  int batchTokens;
};

__device__ __forceinline__ void storeRelease(uint32_t* ptr, uint32_t v) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(ptr), "r"(v)
               : "memory");
}

__device__ __forceinline__ uint32_t loadAcquire(uint32_t const* ptr) {
  uint32_t v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
               : "=r"(v)
               : "l"(ptr)
               : "memory");
  return v;
}

__device__ __forceinline__ uint32_t loadAcquireGpu(uint32_t const* ptr) {
  uint32_t v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
               : "=r"(v)
               : "l"(ptr)
               : "memory");
  return v;
}

__device__ __forceinline__ bool readLater(ConvMixParams const& p, int t) {
  if (p.slots[t] < 0) return false;
  int const next = t + p.tailRows;
  if (next >= p.batchTokens || p.slots[next] < 0 ||
      p.queryStart[next] != p.queryStart[t])
    return true;
  return p.positions[t] % p.align >= p.align - (kTaps - 1);
}

// Run by every CTA once the grid dependency resolved; `epoch` is this call's
// gather epoch. Returns once every rank's previous kernels (the pushes of x)
// have finished.
__device__ void pushBarrier(ConvMixParams const& p, uint32_t epoch) {
  if (threadIdx.x < p.outRanks) {
    auto flags = [&](int k) {
      return reinterpret_cast<uint32_t*>(p.outBases[k]) +
             (p.gatherSlot + 1) * kMaxWorld;
    };
    if (blockIdx.x == 0) {
      __threadfence_system();
      storeRelease(flags(threadIdx.x) + p.rank, epoch);
    }
    int const peer = (p.rank + threadIdx.x) % p.outRanks;
    while (static_cast<int>(loadAcquire(flags(0) + peer) - epoch) < 0) {
    }
  }
  __syncthreads();
}

// Run by every CTA after its pushes; `epoch` is this call's gather epoch.
// Returns once every rank's pushes have landed.
__device__ void gatherBarrier(ConvMixParams const& p, uint32_t epoch) {
  __shared__ bool lastCta;
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) lastCta = atomicAdd(p.doneCtas, 1) == gridDim.x - 1;
  __syncthreads();
  if (lastCta) {
    if (threadIdx.x < p.outRanks) {
      auto flags = [&](int k) {
        return reinterpret_cast<uint32_t*>(p.outBases[k]) +
               p.gatherSlot * kMaxWorld;
      };
      int const peer = (p.rank + threadIdx.x) % p.outRanks;
      storeRelease(flags(threadIdx.x) + p.rank, epoch);
      uint32_t const* mine = flags(0) + peer;
      while (static_cast<int>(loadAcquire(mine) - epoch) < 0) {
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      *p.doneCtas = 0;
      *p.gatherEpoch = epoch;
      asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p.released),
                   "r"(epoch)
                   : "memory");
    }
  } else if (threadIdx.x == 0) {
    while (static_cast<int>(loadAcquireGpu(p.released) - epoch) < 0) {
    }
  }
  __syncthreads();
}

__device__ __forceinline__ float rsqrtApprox(float x) {
  float y;
  asm("rsqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ uint4 load16(bf16 const* ptr) {
  return *reinterpret_cast<uint4 const*>(ptr);
}

__device__ __forceinline__ float2 unpack(uint32_t w) {
  return make_float2(__uint_as_float(w << 16),
                     __uint_as_float(w & 0xffff0000u));
}

// 8 bf16 as 4 fp32 pairs.
__device__ __forceinline__ void unpack(uint4 raw, float2 (&v)[kPairs]) {
  v[0] = unpack(raw.x);
  v[1] = unpack(raw.y);
  v[2] = unpack(raw.z);
  v[3] = unpack(raw.w);
}

// a * b + c of the bf16 pairs a and b (fp32 products, unrounded).
__device__ __forceinline__ float2 fmaBf16(uint32_t a, uint32_t b, float2 c) {
  float2 d;
  asm("{\n"
      ".reg .b16 al, ah, bl, bh;\n"
      "mov.b32 {al, ah}, %2;\n"
      "mov.b32 {bl, bh}, %3;\n"
      "fma.rn.f32.bf16 %0, al, bl, %4;\n"
      "fma.rn.f32.bf16 %1, ah, bh, %5;\n"
      "}"
      : "=f"(d.x), "=f"(d.y)
      : "r"(a), "r"(b), "f"(c.x), "f"(c.y));
  return d;
}

__device__ __forceinline__ uint4 pack(float2 const (&v)[kPairs]) {
  uint32_t w[kPairs];
#pragma unroll
  for (int i = 0; i < kPairs; ++i) {
    __nv_bfloat162 h = __float22bfloat162_rn(v[i]);
    w[i] = *reinterpret_cast<uint32_t*>(&h);
  }
  return make_uint4(w[0], w[1], w[2], w[3]);
}

__device__ __forceinline__ void roundBf16(float2 (&v)[kPairs]) {
  unpack(pack(v), v);
}

__device__ __forceinline__ uint32_t smemAddr(void const* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void barrierInit(uint64_t* bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smemAddr(bar)),
               "r"(count));
}

__device__ __forceinline__ void barrierArrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smemAddr(bar))
               : "memory");
}

__device__ __forceinline__ void expectBytes(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(
                   smemAddr(bar)),
               "r"(bytes)
               : "memory");
}

__device__ __forceinline__ void copyRow(void* dst, void const* src,
                                        uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
      "[%0], [%1], %2, [%3];" ::"r"(smemAddr(dst)),
      "l"(src), "r"(kRowBytes), "r"(smemAddr(bar))
      : "memory");
}

__device__ __forceinline__ void prefetchRow(void const* src) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src),
               "r"(kRowBytes)
               : "memory");
}

__device__ __forceinline__ void barrierWait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred done;\n"
      "wait:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1;\n"
      "@!done bra wait;\n"
      "}" ::"r"(smemAddr(bar)),
      "r"(parity)
      : "memory");
}

// Halves the values in `vals[0, 2 * Width)` across lanes `Bit` apart (lanes
// with `Bit` set keep the upper half), then adds lanes Bit / 2 ... 1 apart.
template <int Width, int Bit, int K>
__device__ __forceinline__ void halve(float (&vals)[K]) {
  if constexpr (Bit >= 1) {
    if constexpr (Width >= 1) {
      bool const upper = threadIdx.x & Bit;
#pragma unroll
      for (int i = 0; i < Width; ++i) {
        float send = upper ? vals[i] : vals[i + Width];
        float keep = upper ? vals[i + Width] : vals[i];
        vals[i] = __fadd_rn(keep, __shfl_xor_sync(0xffffffff, send, Bit));
      }
    } else {
      vals[0] = __fadd_rn(vals[0], __shfl_xor_sync(0xffffffff, vals[0], Bit));
    }
    halve<Width / 2, Bit / 2>(vals);
  }
}

// Sums each of `vals` over the CTA's Warps compute warps into `vals` of every
// thread, in two steps that may run apart: blockPost posts this warp's sums to
// `scratch` and arrives on `bar` (Warps arrivals per phase), blockGather sums
// the posts of every warp. After `halve`, lane l holds the warp sums of values
// [l * K / 32, +K / 32) (K >= 32), or of value l / (32 / K) (K < 32).
template <int K>
__device__ void blockPost(float (&vals)[K], float* scratch, uint64_t* bar) {
  static_assert(K >= 2 && K <= 64 && (K & (K - 1)) == 0);
  int const lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  halve<K / 2, 16>(vals);
  constexpr int kPerLane = K >= 32 ? K / 32 : 1;
  if (K >= 32 || lane % (32 / K) == 0) {
    int const index = K >= 32 ? lane * kPerLane : lane / (32 / K);
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) scratch[warp * K + index + i] = vals[i];
  }
  __syncwarp();
  if (lane == 0) barrierArrive(bar);
}

template <int Warps, int K>
__device__ void blockGather(float (&vals)[K], float const* scratch,
                            uint64_t* bar, uint32_t parity) {
  int const lane = threadIdx.x % 32;
  constexpr int kPerLane = K >= 32 ? K / 32 : 1;
  barrierWait(bar, parity);
  float total[kPerLane];
#pragma unroll
  for (int i = 0; i < kPerLane; ++i) {
    total[i] = 0.f;
    int const k = i * 32 + lane;
    if (k < K) {
#pragma unroll
      for (int w = 0; w < Warps; ++w)
        total[i] = __fadd_rn(total[i], scratch[w * K + k]);
    }
  }
#pragma unroll
  for (int k = 0; k < K; ++k)
    vals[k] = __shfl_sync(0xffffffff, total[k / 32], k % 32);
}

constexpr int pow2Ceil(int x) {
  int p = 2;
  while (p < x) p *= 2;
  return p;
}

// Three queries hold more registers per channel; they run with three chunks
// per thread, whose register cap (192 threads) is higher. One more warp
// issues the copies.
template <int Q>
struct Config {
  static constexpr int kVecs = Q == 1 ? 2 : 3;
  static constexpr int kThreads = kRowVecs / kVecs;
  static constexpr int kWarps = kThreads / 32;
};

// Gather: push the mixes into every rank's all-gather buffer.
template <int NStack, int Q, bool Gather>
__global__ void __launch_bounds__(Config<Q>::kThreads + 32, 1)
    convMixKernel(ConvMixParams p) {
  // Unsharded batches start at token 0 without halo rows.
  int const tokenBase = Gather ? p.tokenBase : 0;
  int const haloRows = Gather ? p.haloRows : 0;
  bool const pushed = Gather && p.xSources > 1;
  // Read before this CTA counts itself done (the last CTA bumps it).
  uint32_t const epoch = Gather ? *p.gatherEpoch + 1 : 0;
  constexpr int kVecs = Config<Q>::kVecs;
  constexpr int kThreads = Config<Q>::kThreads;
  constexpr int kWarps = Config<Q>::kWarps;
  constexpr int kSources = NStack + 1;
  constexpr int kStats = pow2Ceil(kSources * (1 + Q));
  // Reductions per token: the source statistics, then each query's mix.
  constexpr int kReductions = 1 + Q;
  // The reductions' buffers and barriers alternate between tokens.
  __shared__ float scratch[2][kWarps * kStats];
  __shared__ float sumScratch[2][Q][kWarps * 2];
  __shared__ uint64_t full[kMaxStages], empty[kMaxStages],
      reduced[2][kReductions], normFull;
  // Dynamic shared memory: each token's metadata, the norm weights, then the
  // stages.
  extern __shared__ int4 smem[];
  int4* const meta = smem;
  uint4* const normRows = reinterpret_cast<uint4*>(smem + p.tokensPerCta);
  uint4* const rows = normRows + Q * kRowVecs;
  // Stage rows: the stacked sources, the conv input (unless pushed), then
  // last, res, addend (those present).
  int const numRows = p.stageRows;
  int const xRow = NStack;
  int const lastRow = xRow + !pushed;
  int const resRow = lastRow + (p.last != nullptr);
  int const addRow = resRow + (p.res != nullptr);
  int const stages = p.stages;
  int const begin = blockIdx.x * p.tokensPerCta;
  int const end = min(begin + p.tokensPerCta, p.numTokens);
  bool const producer = threadIdx.x >= kThreads;
  // The rows other than the conv input were written two or more kernels
  // earlier, so the first stages load them before the grid dependency wait.
  auto issueOld = [&](int t, int stage) {
    uint4* dst = rows + stage * numRows * kRowVecs;
    int64_t const off = int64_t(t) * p.dim;
    expectBytes(&full[stage], numRows * kRowBytes);
    for (int n = 0; n < NStack; ++n)
      copyRow(dst + n * kRowVecs, p.stack + n * p.stackStride + off,
              &full[stage]);
    if (p.last) copyRow(dst + lastRow * kRowVecs, p.last + off, &full[stage]);
    if (p.res) copyRow(dst + resRow * kRowVecs, p.res + off, &full[stage]);
    if (p.resAddend)
      copyRow(dst + addRow * kRowVecs, p.resAddend + off, &full[stage]);
  };
  auto issueX = [&](int t, int stage) {
    bf16 const* src = p.x + int64_t(t + haloRows) * p.dim;
    if (pushed) {
      for (int r = 0; r < p.xSources; ++r)
        prefetchRow(src + r * p.xSourceStride);
    } else {
      copyRow(rows + (stage * numRows + xRow) * kRowVecs, src, &full[stage]);
    }
  };
  // The producer's first thread issues the copies; the first ones start
  // before anything else.
  bool const issuer = threadIdx.x == kThreads;
  int const prologue = min(begin + stages, end);
  if (issuer) {
    for (int s = 0; s < stages; ++s) {
      barrierInit(&full[s], 1);
      barrierInit(&empty[s], kWarps);
    }
    barrierInit(&normFull, 1);
    for (int i = 0; i < 2 * kReductions; ++i)
      barrierInit(&reduced[i / kReductions][i % kReductions], kWarps);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    expectBytes(&normFull, Q * kRowBytes);
    for (int q = 0; q < Q; ++q)
      copyRow(normRows + q * kRowVecs, p.normWeight + q * p.dim, &normFull);
    for (int t = begin; t < prologue; ++t) issueOld(t, t - begin);
  }
  for (int i = threadIdx.x; i < end - begin; i += blockDim.x) {
    int const t = begin + i;
    int const b = tokenBase + t;
    // The sign bit flags the cache rows read later.
    int const later = Gather && readLater(p, b) ? INT_MIN : 0;
    meta[i] = make_int4(p.positions[b], p.slots[b], p.queryStart[b],
                        p.seqIdx[b] | later);
  }
  int cols[kVecs];
  int64_t cacheCols[kVecs];
  // Conv taps as bf16 pairs; queries in fp32.
  uint32_t weight[kVecs][kTaps][kPairs];
  float2 query[Q][kVecs][kPairs];
#pragma unroll
  for (int v = 0; v < kVecs && !producer; ++v) {
    cols[v] = (v * kThreads + threadIdx.x) * kVec;
    int const head = min(cols[v] / p.streamWidth, p.numHeads - 1);
    cacheCols[v] =
        head * p.cacheHeadStride + p.streamOffset + cols[v] % p.streamWidth;
    uint4 raw[kTaps];
#pragma unroll
    for (int i = 0; i < kTaps; ++i)
      raw[i] = load16(p.convWeight + (int64_t(cols[v]) + 2 * i) * kTaps);
    uint16_t const* w = reinterpret_cast<uint16_t const*>(raw);
#pragma unroll
    for (int k = 0; k < kTaps; ++k)
#pragma unroll
      for (int i = 0; i < kPairs; ++i)
        weight[v][k][i] =
            w[2 * i * kTaps + k] | uint32_t(w[(2 * i + 1) * kTaps + k]) << 16;
#pragma unroll
    for (int q = 0; q < Q; ++q) {
      float4 const* qp =
          reinterpret_cast<float4 const*>(p.queries + q * p.dim + cols[v]);
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        float4 const f = qp[i];
        query[q][v][2 * i] = make_float2(f.x, f.y);
        query[q][v][2 * i + 1] = make_float2(f.z, f.w);
      }
    }
  }
  int const blockShift = p.blockShift, blockMask = (1 << blockShift) - 1;
  __syncthreads();
  cudaGridDependencySynchronize();
  if (pushed) pushBarrier(p, epoch);
  if (issuer) {
    for (int t = begin; t < prologue; ++t) issueX(t, t - begin);
    // Refill each stage once every warp released its previous token.
    int stage = 0, phase = 0;
    for (int t = prologue; t < end; ++t) {
      barrierWait(&empty[stage], phase);
      issueOld(t, stage);
      issueX(t, stage);
      if (++stage == stages) stage = 0, phase ^= 1;
    }
  }
  // The conv input chunk v of x row t.
  auto loadX = [&](int t, int v) {
    bf16 const* src = p.x + int64_t(t) * p.dim + cols[v];
    if (!pushed) return load16(src);
    float2 acc[kPairs], x[kPairs];
    unpack(__ldcg(reinterpret_cast<uint4 const*>(src)), acc);
    for (int r = 1; r < p.xSources; ++r) {
      unpack(__ldcg(reinterpret_cast<uint4 const*>(src + r * p.xSourceStride)),
             x);
#pragma unroll
      for (int i = 0; i < kPairs; ++i) acc[i] = __fadd2_rn(acc[i], x[i]);
    }
    return pack(acc);
  };
  // The conv input rows of the last kTaps - 1 tokens.
  uint4 window[kTaps - 1][kVecs];
#pragma unroll
  for (int k = 0; k < kTaps - 1 && !producer; ++k) {
    int const t = begin - (kTaps - 1) + k + haloRows;
#pragma unroll
    for (int v = 0; v < kVecs; ++v)
      window[k][v] = t >= 0 ? loadX(t, v) : make_uint4(0, 0, 0, 0);
  }

  // Per token: the conv, then the source statistics (one reduction), then
  // per query the mix and its RMS (one reduction each). With one query and
  // local stores, the next token's conv and statistics run while the last
  // reduction of a token is in flight (three queries have no registers to
  // spare; delaying the peer stores of Gather is slower end to end).
  constexpr bool kOverlap = Q == 1 && !Gather;
  int stage = 0, phase = 0;
  uint4 const* row = rows;
  float2 y[kVecs][kPairs];
  auto source = [&](int n, int v, float2(&s)[kPairs]) {
    if (n < NStack) {
      unpack(row[n * kRowVecs + v * kThreads], s);
    } else {
#pragma unroll
      for (int i = 0; i < kPairs; ++i) s[i] = y[v][i];
    }
  };
  auto front = [&](int t) {
    int4 const m = meta[t - begin];
    int const pos = m.x, slot = m.y, qstart = m.z, req = m.w & INT_MAX;
    bool const later = m.w < 0;
    bool const valid = slot >= 0;
    uint4 xRaw[kVecs];
    if (pushed) {
#pragma unroll
      for (int v = 0; v < kVecs; ++v) xRaw[v] = loadX(t + haloRows, v);
    }
    // Conv taps, oldest first: earlier rows of this forward, else the
    // conv-state cache.
    uint4 tapRaw[kTaps - 1][kVecs];
#pragma unroll
    for (int k = 0; k < kTaps - 1; ++k) {
      int const lag = kTaps - 1 - k;
      bool const inForward = tokenBase + t - lag >= qstart;
#pragma unroll
      for (int v = 0; v < kVecs; ++v)
        tapRaw[k][v] = valid && pos >= lag && inForward
                           ? window[k][v]
                           : make_uint4(0, 0, 0, 0);
      if (valid && pos >= lag && !inForward) {
        int const src = pos - lag;
        int const block = p.blockTable[int64_t(req) * p.blockTableStride +
                                       min(src >> blockShift, p.maxBlocks - 1)];
        bf16 const* base = p.cache + block * p.cacheBlockStride +
                           (src & blockMask) * p.cacheSlotStride;
#pragma unroll
        for (int v = 0; v < kVecs; ++v)
          tapRaw[k][v] = load16(base + cacheCols[v]);
      }
    }
    barrierWait(&full[stage], phase);
    row = rows + stage * numRows * kRowVecs + threadIdx.x;

    // y: the conv output (rounded), with the residual update, plus `last`.
#pragma unroll
    for (int v = 0; v < kVecs; ++v) {
      uint4 const curRaw =
          pushed ? xRaw[v] : row[xRow * kRowVecs + v * kThreads];
#pragma unroll
      for (int k = 0; k < kTaps - 2; ++k) window[k][v] = window[k + 1][v];
      window[kTaps - 2][v] = curRaw;
      if (valid) {
        *reinterpret_cast<uint4*>(
            p.cache + (slot >> blockShift) * p.cacheBlockStride +
            (slot & blockMask) * p.cacheSlotStride + cacheCols[v]) = curRaw;
      }
      if constexpr (Gather) {
        if (later) {
          int64_t const offset =
              p.stagingOffset + (int64_t(tokenBase + t) * p.dim + cols[v]) * 2;
          for (int r = 1; r < p.outRanks; ++r)
            *reinterpret_cast<uint4*>(p.outBases[r] + offset) = curRaw;
        }
      }
      float2 cur[kPairs];
      unpack(curRaw, cur);
#pragma unroll
      for (int i = 0; i < kPairs; ++i) y[v][i] = make_float2(0.f, 0.f);
#pragma unroll
      for (int k = 0; k < kTaps - 1; ++k) {
        uint32_t const* tap = reinterpret_cast<uint32_t const*>(&tapRaw[k][v]);
#pragma unroll
        for (int i = 0; i < kPairs; ++i)
          y[v][i] = fmaBf16(tap[i], weight[v][k][i], y[v][i]);
      }
      if (valid && pos >= 0) {
        uint32_t const* w = reinterpret_cast<uint32_t const*>(&curRaw);
#pragma unroll
        for (int i = 0; i < kPairs; ++i)
          y[v][i] = fmaBf16(w[i], weight[v][kTaps - 1][i], y[v][i]);
      }
      float2 const scale = make_float2(p.convScale, p.convScale);
#pragma unroll
      for (int i = 0; i < kPairs; ++i)
        y[v][i] = __fmul2_rn(__fadd2_rn(y[v][i], cur[i]), scale);
      roundBf16(y[v]);
      if (p.resAddend != nullptr) {
        float2 add[kPairs];
        unpack(row[addRow * kRowVecs + v * kThreads], add);
        if (p.res != nullptr) {
          float2 res[kPairs];
          unpack(row[resRow * kRowVecs + v * kThreads], res);
#pragma unroll
          for (int i = 0; i < kPairs; ++i) add[i] = __fadd2_rn(res[i], add[i]);
        }
#pragma unroll
        for (int i = 0; i < kPairs; ++i) y[v][i] = __fadd2_rn(add[i], y[v][i]);
        roundBf16(y[v]);
      }
      *reinterpret_cast<uint4*>(p.convOut + int64_t(t) * p.dim + cols[v]) =
          pack(y[v]);
      if (p.last != nullptr) {
        float2 last[kPairs];
        unpack(row[lastRow * kRowVecs + v * kThreads], last);
#pragma unroll
        for (int i = 0; i < kPairs; ++i) y[v][i] = __fadd2_rn(last[i], y[v][i]);
      }
    }
    // Per source, the sum of squares and the Q query dots.
    float stats[kStats] = {};
#pragma unroll
    for (int n = 0; n < kSources; ++n) {
      float2 acc[1 + Q] = {};
#pragma unroll
      for (int v = 0; v < kVecs; ++v) {
        float2 s[kPairs];
        source(n, v, s);
#pragma unroll
        for (int i = 0; i < kPairs; ++i) {
          acc[0] = __ffma2_rn(s[i], s[i], acc[0]);
#pragma unroll
          for (int q = 0; q < Q; ++q)
            acc[1 + q] = __ffma2_rn(query[q][v][i], s[i], acc[1 + q]);
        }
      }
#pragma unroll
      for (int k = 0; k <= Q; ++k)
        stats[n * (1 + Q) + k] = __fadd_rn(acc[k].x, acc[k].y);
    }
    blockPost(stats, scratch[(t - begin) & 1], &reduced[(t - begin) & 1][0]);
  };
  // out = mix * rstd * norm weight of query q.
  auto store = [&](int t, int q, float2 const(&mix)[kVecs][kPairs],
                   float sumSq) {
    float const rstd =
        rsqrtApprox(__fadd_rn(__fdiv_rn(sumSq, float(p.dim)), p.normEps));
#pragma unroll
    for (int v = 0; v < kVecs; ++v) {
      float2 w[kPairs];
      unpack(normRows[q * kRowVecs + v * kThreads + threadIdx.x], w);
#pragma unroll
      for (int i = 0; i < kPairs; ++i)
        w[i] = __fmul2_rn(__fmul2_rn(mix[v][i], make_float2(rstd, rstd)), w[i]);
      int64_t const offset =
          (int64_t(q) * p.outTokens + tokenBase + t) * p.dim + cols[v];
      uint4 const packed = pack(w);
      if constexpr (Gather) {
        for (int r = 0; r < p.outRanks; ++r)
          reinterpret_cast<uint4*>(p.outBases[r] + p.outOffset)[offset / kVec] =
              packed;
      } else {
        *reinterpret_cast<uint4*>(p.out + offset) = packed;
      }
    }
  };
  if (!producer) barrierWait(&normFull, 0);
  for (int t = producer ? end : begin; t < end; ++t) {
    if (!kOverlap || t == begin) front(t);
    int const buf = (t - begin) & 1;
    uint32_t const parity = ((t - begin) >> 1) & 1;
    float stats[kStats];
    blockGather<kWarps>(stats, scratch[buf], &reduced[buf][0], parity);

    float invRms[kSources];
#pragma unroll
    for (int n = 0; n < kSources; ++n)
      invRms[n] =
          rsqrtApprox(__fmaf_rn(stats[n * (1 + Q)], 1.f / p.dim, p.eps));
    // One query at a time: the mix, its RMS, the store.
    float2 mix[kVecs][kPairs];
    float sumSq[2];
#pragma unroll
    for (int q = 0; q < Q; ++q) {
      float logit[kSources];
      float maxLogit = -INFINITY;
#pragma unroll
      for (int n = 0; n < kSources; ++n) {
        logit[n] = __fmul_rn(stats[n * (1 + Q) + 1 + q], invRms[n]);
        maxLogit = fmaxf(maxLogit, logit[n]);
      }
      float prob[kSources];
      float total = 0.f;
#pragma unroll
      for (int n = 0; n < kSources; ++n) {
        prob[n] = expf(__fsub_rn(logit[n], maxLogit));
        total = __fadd_rn(total, prob[n]);
      }
      float const invTotal = __frcp_rn(total);
#pragma unroll
      for (int v = 0; v < kVecs; ++v)
#pragma unroll
        for (int i = 0; i < kPairs; ++i) mix[v][i] = make_float2(0.f, 0.f);
#pragma unroll
      for (int n = 0; n < kSources; ++n) {
        float const w = __fmul_rn(prob[n], invRms[n]);
#pragma unroll
        for (int v = 0; v < kVecs; ++v) {
          float2 s[kPairs];
          source(n, v, s);
#pragma unroll
          for (int i = 0; i < kPairs; ++i)
            mix[v][i] = __ffma2_rn(make_float2(w, w), s[i], mix[v][i]);
        }
      }
      float2 sq = make_float2(0.f, 0.f);
#pragma unroll
      for (int v = 0; v < kVecs; ++v)
#pragma unroll
        for (int i = 0; i < kPairs; ++i) {
          mix[v][i] = __fmul2_rn(mix[v][i], make_float2(invTotal, invTotal));
          sq = __ffma2_rn(mix[v][i], mix[v][i], sq);
        }
      sumSq[0] = __fadd_rn(sq.x, sq.y);
      sumSq[1] = 0.f;
      blockPost(sumSq, sumScratch[buf][q], &reduced[buf][1 + q]);
      if (q + 1 < Q) {
        blockGather<kWarps>(sumSq, sumScratch[buf][q], &reduced[buf][1 + q],
                            parity);
        store(t, q, mix, sumSq[0]);
      }
    }
    // This warp is done with the stage.
    __syncwarp();
    if (threadIdx.x % 32 == 0) barrierArrive(&empty[stage]);
    if (++stage == stages) stage = 0, phase ^= 1;
    if (kOverlap && t + 1 < end) front(t + 1);
    blockGather<kWarps>(sumSq, sumScratch[buf][Q - 1], &reduced[buf][Q],
                        parity);
    store(t, Q - 1, mix, sumSq[0]);
  }
  if constexpr (Gather) {
    gatherBarrier(p, epoch);
    // The cache rows read later that other ranks own: CTA b takes rows
    // b, b + grid, ...; each thread checks one, then the CTA copies the
    // flagged ones.
    __shared__ int flagged[kThreads];
    __shared__ int numFlagged;
    for (int first = 0; first * gridDim.x < p.batchTokens; first += kThreads) {
      if (threadIdx.x == 0) numFlagged = 0;
      __syncthreads();
      int const t = blockIdx.x + (first + threadIdx.x) * gridDim.x;
      if (!producer && t < p.batchTokens &&
          (t < tokenBase || t >= tokenBase + p.numTokens) && readLater(p, t))
        flagged[atomicAdd(&numFlagged, 1)] = t;
      __syncthreads();
      for (int i = 0; i < numFlagged && !producer; ++i) {
        int const row = flagged[i];
        int64_t const slot = p.slots[row];
        bf16* const dst = p.cache + (slot >> blockShift) * p.cacheBlockStride +
                          (slot & blockMask) * p.cacheSlotStride;
        bf16 const* src =
            reinterpret_cast<bf16 const*>(p.outBases[0] + p.stagingOffset) +
            int64_t(row) * p.dim;
#pragma unroll
        for (int v = 0; v < kVecs; ++v)
          *reinterpret_cast<uint4*>(dst + cacheCols[v]) =
              __ldcg(reinterpret_cast<uint4 const*>(src + cols[v]));
      }
      __syncthreads();
    }
  }
  cudaTriggerProgrammaticLaunchCompletion();
}

template <int Q, bool Gather, int... N>
auto selectKernel(int numStack, std::integer_sequence<int, N...>) {
  void (*kernel)(ConvMixParams) = nullptr;
  ((numStack == N + 1 ? (kernel = &convMixKernel<N + 1, Q, Gather>, 0) : 0),
   ...);
  return kernel;
}

template <typename T>
T const* optionalPtr(std::optional<torch::stable::Tensor> const& t) {
  return t.has_value() ? static_cast<T const*>(t->data_ptr()) : nullptr;
}

ConvMixParams makeParams(
    torch::stable::Tensor const& x, torch::stable::Tensor& cache,
    torch::stable::Tensor const& conv_weight,
    torch::stable::Tensor const& positions,
    torch::stable::Tensor const& seq_idx, torch::stable::Tensor const& slots,
    torch::stable::Tensor const& block_table,
    torch::stable::Tensor const& query_start, int64_t stream_offset,
    int64_t stream_width, double conv_scale,
    std::optional<torch::stable::Tensor> const& residual,
    std::optional<torch::stable::Tensor> const& addend,
    torch::stable::Tensor& conv_out, torch::stable::Tensor const& stack,
    std::optional<torch::stable::Tensor> const& last,
    torch::stable::Tensor const& queries,
    torch::stable::Tensor const& norm_weight, double eps, double norm_eps) {
  ConvMixParams p{};
  p.x = static_cast<bf16 const*>(x.data_ptr());
  p.xSources = x.dim() == 3 ? x.size(0) : 1;
  p.xSourceStride = x.dim() == 3 ? x.stride(0) : 0;
  p.numTokens = conv_out.size(0);
  p.dim = conv_out.size(1);
  p.cache = static_cast<bf16*>(cache.data_ptr());
  p.convWeight = static_cast<bf16 const*>(conv_weight.data_ptr());
  p.positions = static_cast<int64_t const*>(positions.data_ptr());
  p.seqIdx = static_cast<int const*>(seq_idx.data_ptr());
  p.slots = static_cast<int64_t const*>(slots.data_ptr());
  p.blockTable = static_cast<int const*>(block_table.data_ptr());
  p.queryStart = static_cast<int const*>(query_start.data_ptr());
  p.cacheBlockStride = cache.stride(0);
  p.cacheHeadStride = cache.stride(1);
  p.cacheSlotStride = cache.stride(2);
  p.blockTableStride = block_table.stride(0);
  p.maxBlocks = block_table.size(1);
  p.streamOffset = stream_offset;
  p.streamWidth = stream_width;
  p.numHeads = cache.size(1);
  p.convScale = conv_scale;
  p.res = optionalPtr<bf16>(residual);
  p.resAddend = optionalPtr<bf16>(addend);
  p.convOut = static_cast<bf16*>(conv_out.data_ptr());
  p.stack = static_cast<bf16 const*>(stack.data_ptr());
  p.stackStride = stack.stride(0);
  p.last = optionalPtr<bf16>(last);
  p.queries = static_cast<float const*>(queries.data_ptr());
  p.normWeight = static_cast<bf16 const*>(norm_weight.data_ptr());
  p.eps = eps;
  p.normEps = norm_eps;
  p.batchTokens = positions.size(0);
  int const blockSize = cache.size(2);
  STD_TORCH_CHECK(p.dim == kRowVecs * kVec,
                  "Inkling2 conv + mix requires hidden size 4608");
  STD_TORCH_CHECK(stream_width % kVec == 0 && stream_offset % kVec == 0);
  STD_TORCH_CHECK((blockSize & (blockSize - 1)) == 0);
  STD_TORCH_CHECK(cache.size(0) * blockSize <= int64_t(INT32_MAX));
  p.blockShift = __builtin_ctz(blockSize);
  return p;
}

template <bool Gather>
void launch(ConvMixParams& p, int numStack, int numQueries, int device) {
  STD_TORCH_CHECK(numStack >= 1 && numStack <= kMaxStack);
  STD_TORCH_CHECK(numQueries == 1 || numQueries == 3);
  int const ctas =
      std::min(get_device_prop()->multiProcessorCount, p.numTokens);
  p.tokensPerCta = (p.numTokens + ctas - 1) / ctas;
  // As many stages as the shared memory holds.
  p.stageRows = numStack + (p.xSources == 1) + (p.last != nullptr) +
                (p.res != nullptr) + (p.resAddend != nullptr);
  int const metaBytes = p.tokensPerCta * sizeof(int4) + numQueries * kRowBytes;
  p.stages = std::min<int>(
      kMaxStages, (kSmemBytes - metaBytes) / (p.stageRows * kRowBytes));
  STD_TORCH_CHECK(p.stages >= 1);

  auto seq = std::make_integer_sequence<int, kMaxStack>{};
  void (*kernel)(ConvMixParams) = numQueries == 1
                                      ? selectKernel<1, Gather>(numStack, seq)
                                      : selectKernel<3, Gather>(numStack, seq);
  int const smem = metaBytes + p.stages * p.stageRows * kRowBytes;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       smem);
  cudaLaunchAttribute attrs[1]{};
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3((p.numTokens + p.tokensPerCta - 1) / p.tokensPerCta);
  config.blockDim =
      dim3((numQueries == 1 ? Config<1>::kThreads : Config<3>::kThreads) + 32);
  config.dynamicSmemBytes = smem;
  config.stream = get_current_cuda_stream(device);
  config.attrs = attrs;
  config.numAttrs = 1;
  cudaError_t status = cudaLaunchKernelEx(&config, kernel, p);
  STD_TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
}

}  // namespace
}  // namespace inkling2
}  // namespace vllm

void inkling2_conv_mix(
    torch::stable::Tensor const& x, torch::stable::Tensor& cache,
    torch::stable::Tensor const& conv_weight,
    torch::stable::Tensor const& positions,
    torch::stable::Tensor const& seq_idx, torch::stable::Tensor const& slots,
    torch::stable::Tensor const& block_table,
    torch::stable::Tensor const& query_start, int64_t stream_offset,
    int64_t stream_width, double conv_scale,
    std::optional<torch::stable::Tensor> residual,
    std::optional<torch::stable::Tensor> addend,
    torch::stable::Tensor& conv_out, torch::stable::Tensor const& stack,
    std::optional<torch::stable::Tensor> last,
    torch::stable::Tensor const& queries,
    torch::stable::Tensor const& norm_weight, double eps, double norm_eps,
    torch::stable::Tensor& out) {
  using namespace vllm::inkling2;
  int const device = conv_out.get_device_index();
  torch::stable::accelerator::DeviceGuard const device_guard(device);
  ConvMixParams p = makeParams(
      x, cache, conv_weight, positions, seq_idx, slots, block_table,
      query_start, stream_offset, stream_width, conv_scale, residual, addend,
      conv_out, stack, last, queries, norm_weight, eps, norm_eps);
  p.out = static_cast<bf16*>(out.data_ptr());
  p.outTokens = p.numTokens;
  STD_TORCH_CHECK(p.xSources == 1);
  if (p.numTokens == 0) return;
  launch<false>(p, stack.size(0), queries.size(0), device);
}

void inkling2_seq_parallel_conv_mix(
    torch::stable::Tensor const& x, torch::stable::Tensor& cache,
    torch::stable::Tensor const& conv_weight,
    torch::stable::Tensor const& positions,
    torch::stable::Tensor const& seq_idx, torch::stable::Tensor const& slots,
    torch::stable::Tensor const& block_table,
    torch::stable::Tensor const& query_start, int64_t stream_offset,
    int64_t stream_width, double conv_scale,
    std::optional<torch::stable::Tensor> residual,
    std::optional<torch::stable::Tensor> addend,
    torch::stable::Tensor& conv_out, torch::stable::Tensor const& stack,
    std::optional<torch::stable::Tensor> last,
    torch::stable::Tensor const& queries,
    torch::stable::Tensor const& norm_weight, double eps, double norm_eps,
    int64_t token_base, int64_t halo_rows, int64_t out_bases,
    int64_t out_offset, int64_t world, int64_t out_tokens, int64_t rank,
    int64_t gather_slot, torch::stable::Tensor& epochs, int64_t staging_offset,
    int64_t tail_rows, int64_t align) {
  using namespace vllm::inkling2;
  int const device = conv_out.get_device_index();
  torch::stable::accelerator::DeviceGuard const device_guard(device);
  ConvMixParams p = makeParams(
      x, cache, conv_weight, positions, seq_idx, slots, block_table,
      query_start, stream_offset, stream_width, conv_scale, residual, addend,
      conv_out, stack, last, queries, norm_weight, eps, norm_eps);
  p.tokenBase = token_base;
  p.haloRows = halo_rows;
  p.outBases = reinterpret_cast<char* const*>(out_bases);
  p.outOffset = out_offset;
  p.outRanks = world;
  p.outTokens = out_tokens;
  p.rank = rank;
  p.gatherSlot = gather_slot;
  STD_TORCH_CHECK(epochs.numel() == 3);
  p.gatherEpoch = static_cast<uint32_t*>(epochs.data_ptr());
  p.doneCtas = p.gatherEpoch + 1;
  p.released = p.gatherEpoch + 2;
  p.stagingOffset = staging_offset;
  p.tailRows = tail_rows;
  p.align = align;
  STD_TORCH_CHECK(world >= 2 && world <= kMaxWorld);
  STD_TORCH_CHECK(p.xSources == 1 || p.xSources == world);
  STD_TORCH_CHECK(tail_rows >= kTaps - 1 && align >= kTaps);
  STD_TORCH_CHECK(p.numTokens > 0 && token_base + p.numTokens <= out_tokens);
  STD_TORCH_CHECK(x.size(-2) == halo_rows + p.numTokens);
  launch<true>(p, stack.size(0), queries.size(0), device);
}
