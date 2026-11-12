// Inkling2's TP all-reduce of the attention / MoE partial sums fused with the
// hidden-stream short conv and the AttnRes mix + RMSNorm of the next sublayer.
// Built on the MNNVL Lamport primitives in mnnvl_lamport.cuh. Each token runs
// on a cluster of CTAs that split its channels; each thread owns 8 consecutive
// channels, so everything but the mix's row statistics (summed over the
// cluster in distributed shared memory) stays in registers.

#include "../torch_utils.h"
#include "../mnnvl_lamport.cuh"

namespace vllm {
namespace inkling2 {

namespace ar = vllm::mnnvl;
using bf16 = __nv_bfloat16;

constexpr int kVec = 8;
constexpr int kMaxSources = 9;
constexpr int kMaxQueries = 3;
constexpr int kTaps = 4;
// Threads per token: Inkling2's hidden size 4608 / 8 channels per thread.
constexpr int kTokenThreads = 576;
constexpr int kMaxWarps = kTokenThreads / 32;
constexpr int kClearCtas = 8;

// The partial sum: this rank's bf16 rows `x`, or (MoE) its bf16 expert rows
// `down` combined in fp32 over each token's `topK` slots `src2dst` and rounded
// once. One-shot pushes every row to every rank; two-shot scatters each row's
// 1/World channel shard to its owner, which reduces it and broadcasts it in
// bf16.
enum Input { kPartial, kCombine, kPartialTwoShot, kCombineTwoShot };

struct Params {
  bf16 const* x;
  bf16 const* down;
  int const* src2dst;
  int topK;
  float* mcast;
  void* const* peers;  // every rank's unicast Lamport buffer
  float* local;
  uint32_t* flags;
  int rank;
  int numTokens;
  int dim;
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
  int blockSize;
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
  int numStack;
  bf16 const* last;
  float const* queries;    // [Q, D]
  bf16 const* normWeight;  // [Q, D]
  int numQueries;
  float eps;
  float normEps;
  bf16* out;  // [Q, T, D]
};

__device__ __forceinline__ float rsqrtApprox(float x) {
  float y;
  asm("rsqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ void loadBf16(bf16 const* ptr, float (&v)[kVec]) {
  uint4 raw = *reinterpret_cast<uint4 const*>(ptr);
  bf16 const* h = reinterpret_cast<bf16 const*>(&raw);
#pragma unroll
  for (int i = 0; i < kVec; ++i) v[i] = __bfloat162float(h[i]);
}

__device__ __forceinline__ void unpackBf16(uint4 raw, float (&v)[kVec]) {
  bf16 const* h = reinterpret_cast<bf16 const*>(&raw);
#pragma unroll
  for (int i = 0; i < kVec; ++i) v[i] = __bfloat162float(h[i]);
}

// Staging slots in shared memory, one 16-byte chunk per thread each: the conv
// taps, the norm weights, the fp32 queries, then the present sources.
constexpr int kWeightSlot = 0;
constexpr int kNormSlot = kTaps;
constexpr int kMaxSlots = kNormSlot + 3 * kMaxQueries + kMaxSources - 1 + 3;

struct Slots {
  int query, stack, last, res, add, count;
  __host__ __device__ Slots(Params const& p) {
    query = kNormSlot + p.numQueries;
    stack = query + 2 * p.numQueries;
    last = stack + p.numStack;
    res = last + (p.last != nullptr);
    add = res + (p.res != nullptr);
    count = add + (p.resAddend != nullptr);
  }
};

__device__ __forceinline__ void cpAsync(uint4* dst, void const* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(
                   static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
               "l"(src));
}

__device__ __forceinline__ void storeBf16(bf16* ptr, float const (&v)[kVec]) {
  uint4 raw;
  bf16* h = reinterpret_cast<bf16*>(&raw);
#pragma unroll
  for (int i = 0; i < kVec; ++i) h[i] = __float2bfloat16_rn(v[i]);
  *reinterpret_cast<uint4*>(ptr) = raw;
}

__device__ __forceinline__ void roundBf16(float (&v)[kVec]) {
#pragma unroll
  for (int i = 0; i < kVec; ++i)
    v[i] = __bfloat162float(__float2bfloat16_rn(v[i]));
}

// Wait until every rank's bf16 slice of the rows `t - lag` (lag = kTaps - 1 -
// k for each `rows[k]`) is in this rank's Lamport buffer, then sum the ranks
// in rank order in fp32 into `sum[k]` and round once to bf16. `Group` ranks
// of every row are polled at once.
template <int World, int Group>
__device__ void reduceRows(void const* buffer, int64_t rowStride, int t,
                           int col, bool const (&rows)[kTaps],
                           float (&sum)[kTaps][kVec]) {
  bf16 const* base = static_cast<bf16 const*>(buffer);
#pragma unroll
  for (int k = 0; k < kTaps; ++k) {
    if (!rows[k]) continue;
#pragma unroll
    for (int i = 0; i < kVec; ++i) sum[k][i] = 0.f;
  }
#pragma unroll 1
  for (int r0 = 0; r0 < World; r0 += Group) {
    uint4 vals[kTaps][Group];
    bool ready;
    do {
      ready = true;
#pragma unroll
      for (int k = 0; k < kTaps; ++k) {
        if (!rows[k]) continue;
        int64_t const row = int64_t(t - (kTaps - 1 - k)) * World + r0;
#pragma unroll
        for (int r = 0; r < Group; ++r) {
          auto loaded = ar::utils::loadPackedVolatile<float4>(
              base + (row + r) * rowStride + col);
          vals[k][r] = *reinterpret_cast<uint4 const*>(&loaded.packed);
          ready &= !ar::utils::isLamportDirty(loaded);
        }
      }
    } while (!ready);
#pragma unroll
    for (int k = 0; k < kTaps; ++k) {
      if (!rows[k]) continue;
#pragma unroll
      for (int r = 0; r < Group; ++r) {
        float v[kVec];
        unpackBf16(vals[k][r], v);
#pragma unroll
        for (int i = 0; i < kVec; ++i) sum[k][i] = __fadd_rn(sum[k][i], v[i]);
      }
    }
  }
#pragma unroll
  for (int k = 0; k < kTaps; ++k) {
    if (rows[k]) roundBf16(sum[k]);
  }
}

// The CTAs past the tokens clear every stage of the previous call's Lamport
// buffer off the tokens' critical path.
__device__ void clearDirtyBuffer(Params const& p) {
  uint4 const flags = reinterpret_cast<uint4 const*>(p.flags)[0];
  float const z = __uint_as_float(ar::utils::kNEGZERO_FP32);
  uint32_t const first = (blockIdx.y - p.numTokens) * gridDim.x + blockIdx.x;
  uint32_t const stride = (gridDim.y - p.numTokens) * gridDim.x * blockDim.x;
  // No stages before the first call.
  for (uint32_t stage = 0; stage < flags.w; ++stage) {
    float4* dirty = reinterpret_cast<float4*>(
        reinterpret_cast<char*>(p.local) +
        size_t(flags.y * flags.w + stage) * (flags.z / flags.w));
    uint32_t const n =
        (p.flags[4 + stage] + sizeof(float4) - 1) / sizeof(float4);
    for (uint32_t i = first * blockDim.x + threadIdx.x; i < n; i += stride)
      dirty[i] = make_float4(z, z, z, z);
  }
}

// Halves the values in `vals[0, 2 * Width)` across lanes `bit` apart: lanes
// with `bit` set keep the upper half.
template <int Width, int Bit, int K>
__device__ __forceinline__ void halve(float (&vals)[K]) {
  if constexpr (Width >= 1) {
    bool const upper = threadIdx.x & Bit;
#pragma unroll
    for (int i = 0; i < Width; ++i) {
      float send = upper ? vals[i] : vals[i + Width];
      float keep = upper ? vals[i + Width] : vals[i];
      vals[i] = __fadd_rn(keep, __shfl_xor_sync(0xffffffff, send, Bit));
    }
    halve<Width / 2, Bit / 2>(vals);
  }
}

__device__ __forceinline__ uint32_t smemAddr(void const* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// K row statistics of a token as partial sums of its cluster's C CTAs, in
// every CTA's shared memory. The CTAs store their partials into each other's
// `parts` through distributed shared memory, completing on `bar`.
template <int C, int K>
struct RowSums {
  float parts[C][K];
  uint64_t bar;

  __device__ void init() {
    if constexpr (C > 1) {
      asm volatile(
          "mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smemAddr(&bar)));
      asm volatile(
          "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(
              smemAddr(&bar)),
          "r"(int(sizeof(parts))));
    }
  }

  // Statistic k summed over the CTAs in order.
  __device__ float operator[](int k) const {
    float total = parts[0][k];
#pragma unroll
    for (int c = 1; c < C; ++c) total = __fadd_rn(total, parts[c][k]);
    return total;
  }
};

// Sums each of `vals` over the CTA's warps into `sums`, and (C > 1) waits for
// the other CTAs' partials.
template <int C, int K>
__device__ void rowReduce(float (&vals)[K], float* scratch,
                          RowSums<C, K>& sums) {
  int const lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  int const numWarps = blockDim.x / 32;
  halve<K / 2, 16>(vals);
  // After log2(K) steps lanes l and l ^ (bits below 32 / K) hold the same
  // value's partial sums.
  float v = vals[0];
#pragma unroll
  for (int bit = 16 / K; bit >= 1; bit /= 2)
    v = __fadd_rn(v, __shfl_xor_sync(0xffffffff, v, bit));
  int const index = lane / (32 / K);
  if (lane % (32 / K) == 0) scratch[warp * K + index] = v;
  __syncthreads();
  if (threadIdx.x < K) {
    float total = 0.f;
    for (int w = 0; w < numWarps; ++w)
      total = __fadd_rn(total, scratch[w * K + threadIdx.x]);
    if constexpr (C == 1) {
      sums.parts[0][threadIdx.x] = total;
    } else {
      uint32_t const part = smemAddr(&sums.parts[blockIdx.x][threadIdx.x]);
      uint32_t const bar = smemAddr(&sums.bar);
#pragma unroll
      for (int c = 0; c < C; ++c) {
        uint32_t remotePart, remoteBar;
        asm("mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(remotePart)
            : "r"(part), "r"(c));
        asm("mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(remoteBar)
            : "r"(bar), "r"(c));
        asm volatile(
            "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], "
            "%1, [%2];" ::"r"(remotePart),
            "r"(__float_as_uint(total)), "r"(remoteBar)
            : "memory");
      }
    }
  }
  if constexpr (C == 1) {
    __syncthreads();
  } else {
    uint32_t done;
    do {
      asm volatile(
          "{\n.reg .pred p;\n"
          "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0;\n"
          "selp.u32 %0, 1, 0, p;\n}"
          : "=r"(done)
          : "r"(smemAddr(&sums.bar))
          : "memory");
    } while (!done);
  }
}

// A source's sum of squares and query dots, at `stats[at:at + 1 + Q]`.
template <int Q, int K>
__device__ __forceinline__ void sourceStats(float const (&src)[kVec],
                                            float const (&query)[Q][kVec],
                                            float (&stats)[K], int at) {
  float sq = 0.f;
#pragma unroll
  for (int i = 0; i < kVec; ++i) sq = __fmaf_rn(src[i], src[i], sq);
  stats[at] = sq;
#pragma unroll
  for (int q = 0; q < Q; ++q) {
    float dot = 0.f;
#pragma unroll
    for (int i = 0; i < kVec; ++i) dot = __fmaf_rn(query[q][i], src[i], dot);
    stats[at + 1 + q] = dot;
  }
}

// One token per cluster of C CTAs (`blockIdx.y`), each CTA a 1/C slice of
// its channels (`blockIdx.x`).
template <int World, int In, int Q, int C>
__global__ void __launch_bounds__(kTokenThreads / C, 1)
    allReduceConvMixKernel(Params p) {
  constexpr int kThreads = kTokenThreads / C;
  constexpr bool kTwoShot = In == kPartialTwoShot || In == kCombineTwoShot;
  constexpr bool kCombined = In == kCombine || In == kCombineTwoShot;
  // Two-shot rows are read back from the reduced broadcast.
  constexpr int kRowWorld = kTwoShot ? 1 : World;
  // This row and the in-batch taps are polled at once, or as many ranks of
  // them as the registers of the CTA size allow, this row first (a whole
  // token's CTA: then each tap).
  constexpr bool kAtOnce = kRowWorld == 1 || C > 2 || (C > 1 && kRowWorld <= 4);
  constexpr int kRowGroup =
      C == 1 ? (kRowWorld < 4 ? kRowWorld : 4) : kRowWorld;
  constexpr int kTapGroup = (kRowWorld < 4 ? kRowWorld : 4);
  if (blockIdx.y >= p.numTokens) {
    cudaGridDependencySynchronize();
    clearDirtyBuffer(p);
    ar::utils::LamportFlags<float4>(p.flags, kTwoShot ? 2 : 1).ctaArrive();
    return;
  }
  constexpr int kStackStats = (kMaxSources - 1) * (1 + Q);
  constexpr int kLastStats = Q == 1 ? 2 : 4;
  __shared__ RowSums<C, kStackStats> stackStats;
  __shared__ RowSums<C, kLastStats> lastStats;
  __shared__ RowSums<C, 4> sumSqs;
  // Distributed shared memory is written only once every CTA of the cluster
  // runs (the wait is in the first row reduction).
  if constexpr (C > 1) {
    if (threadIdx.x == 0) {
      stackStats.init();
      lastStats.init();
      sumSqs.init();
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
  }
  __shared__ float scratch[kMaxWarps * 32];
  int const t = blockIdx.y;
  int const col = (blockIdx.x * kThreads + threadIdx.x) * kVec;
  int64_t const rowOff = int64_t(t) * p.dim + col;

  // Everything not produced by the preceding kernel loads before the grid
  // dependency wait: row metadata, weights, cached conv taps, and the mix and
  // residual sources (written two or more kernels earlier).
  int64_t const pos = p.positions[t];
  int64_t const slot = p.slots[t];
  int const req = p.seqIdx[t];
  int const qstart = p.queryStart[t];
  bool const valid = slot >= 0;
  int const head = min(col / p.streamWidth, p.numHeads - 1);
  int64_t const cacheCol =
      head * p.cacheHeadStride + p.streamOffset + col % p.streamWidth;
  // Conv taps, oldest first: earlier rows of this forward (`inBatch`) are
  // reduced here too, earlier steps come from the conv-state cache.
  bool inBatch[kTaps];
  uint4 cachedRaw[kTaps - 1];
#pragma unroll
  for (int k = 0; k < kTaps - 1; ++k) {
    int const lag = kTaps - 1 - k;
    cachedRaw[k] = make_uint4(0, 0, 0, 0);
    inBatch[k] = valid && pos >= lag && t - lag >= qstart;
    if (valid && pos >= lag && t - lag < qstart) {
      int64_t src = pos - lag;
      int block = p.blockTable[int64_t(req) * p.blockTableStride +
                               min(int(src / p.blockSize), p.maxBlocks - 1)];
      cachedRaw[k] = *reinterpret_cast<uint4 const*>(
          p.cache + block * p.cacheBlockStride +
          (src % p.blockSize) * p.cacheSlotStride + cacheCol);
    }
  }
  inBatch[kTaps - 1] = false;
  extern __shared__ uint4 staged[];
  auto slot16 = [&](int s) -> uint4& {
    return staged[s * kThreads + threadIdx.x];
  };
  Slots const slots(p);
  for (int n = 0; n < p.numStack; ++n)
    cpAsync(&slot16(slots.stack + n), p.stack + n * p.stackStride + rowOff);
  if (p.last) cpAsync(&slot16(slots.last), p.last + rowOff);
  if (p.res) cpAsync(&slot16(slots.res), p.res + rowOff);
  if (p.resAddend) cpAsync(&slot16(slots.add), p.resAddend + rowOff);
#pragma unroll
  for (int q = 0; q < Q; ++q) {
    cpAsync(&slot16(kNormSlot + q), p.normWeight + q * p.dim + col);
    cpAsync(&slot16(slots.query + 2 * q), p.queries + q * p.dim + col);
    cpAsync(&slot16(slots.query + 2 * q + 1), p.queries + q * p.dim + col + 4);
  }
#pragma unroll
  for (int i = 0; i < kTaps; ++i)
    cpAsync(&slot16(kWeightSlot + i),
            p.convWeight + (int64_t(col) + 2 * i) * kTaps);
  asm volatile("cp.async.commit_group;" ::: "memory");

  // The mix statistics of the stacked sources (per source, the sum of squares
  // and the Q query dots) do not depend on the preceding kernel either. They
  // are computed while the pushes are in flight.
  auto loadQueries = [&](float (&query)[Q][kVec]) {
#pragma unroll
    for (int q = 0; q < Q; ++q) {
      *reinterpret_cast<uint4*>(&query[q][0]) = slot16(slots.query + 2 * q);
      *reinterpret_cast<uint4*>(&query[q][4]) = slot16(slots.query + 2 * q + 1);
    }
  };
  auto reduceStackStats = [&] {
    asm volatile("cp.async.wait_all;" ::: "memory");
    float query[Q][kVec];
    loadQueries(query);
    float stats[kStackStats] = {};
#pragma unroll
    for (int n = 0; n < kMaxSources - 1; ++n) {
      if (n >= p.numStack) break;
      float src[kVec];
      unpackBf16(slot16(slots.stack + n), src);
      sourceStats<Q>(src, query, stats, n * (1 + Q));
    }
    if constexpr (C > 1) {
      asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    }
    rowReduce(stats, scratch, stackStats);
  };
  cudaGridDependencySynchronize();

  ar::utils::LamportFlags<float4> flag(p.flags, kTwoShot ? 2 : 1);
  auto* mcast =
      static_cast<bf16*>(flag.getCurLamportBuf(p.mcast, kTwoShot ? 1 : 0));
  // This rank's buffer of the reduced rows: the one-shot buffer, or the
  // two-shot broadcast stage.
  void* local = flag.getCurLamportBuf(p.local, kTwoShot ? 1 : 0);
  // This thread's bf16 partial. The Lamport buffer's empty pattern is a -0.0
  // fp32 word: pushed -0.0 values become +0.0.
  uint4 partial;
  uint16_t* h = reinterpret_cast<uint16_t*>(&partial);
  if constexpr (kCombined) {
    float x[kVec];
#pragma unroll
    for (int i = 0; i < kVec; ++i) x[i] = 0.f;
    for (int k = 0; k < p.topK; ++k) {
      int row = p.src2dst[t * p.topK + k];
      if (row < 0) continue;
      float v[kVec];
      loadBf16(p.down + int64_t(row) * p.dim + col, v);
#pragma unroll
      for (int i = 0; i < kVec; ++i) x[i] = __fadd_rn(x[i], v[i]);
    }
#pragma unroll
    for (int i = 0; i < kVec; ++i)
      h[i] = __bfloat16_as_ushort(__float2bfloat16_rn(x[i]));
  } else {
    partial = *reinterpret_cast<uint4 const*>(p.x + rowOff);
  }
#pragma unroll
  for (int i = 0; i < kVec; ++i)
    h[i] = h[i] == ar::utils::kNEGZERO_FP16 ? 0 : h[i];
  // The reduced rows: this one, then the in-batch taps.
  float rows[kTaps][kVec];
  bool own[kTaps] = {};
  own[kTaps - 1] = true;
  if constexpr (kTwoShot) {
    // Scatter: the channels of shard `dest` go to rank `dest`.
    int const shard = p.dim / World;
    int const dest = col / shard;
    int const shardCol = col - dest * shard;
    auto* scatter = static_cast<bf16*>(flag.getCurLamportBuf(p.peers[dest], 0));
    *reinterpret_cast<uint4*>(scatter + (int64_t(t) * World + p.rank) * shard +
                              shardCol) = partial;
    reduceStackStats();
    if (dest == p.rank) {
      // Reduce this rank's shard in rank order, broadcast it in bf16.
      reduceRows<World, World>(flag.getCurLamportBuf(p.local, 0), shard, t,
                               shardCol, own, rows);
      uint4 packed;
      uint16_t* r = reinterpret_cast<uint16_t*>(&packed);
#pragma unroll
      for (int i = 0; i < kVec; ++i) {
        uint16_t b =
            __bfloat16_as_ushort(__float2bfloat16_rn(rows[kTaps - 1][i]));
        r[i] = b == ar::utils::kNEGZERO_FP16 ? 0 : b;
      }
      *reinterpret_cast<uint4*>(mcast + int64_t(t) * p.dim + col) = packed;
    }
  } else {
    *reinterpret_cast<uint4*>(mcast + (int64_t(t) * World + p.rank) * p.dim +
                              col) = partial;
    reduceStackStats();
  }
  if constexpr (kAtOnce) {
    bool needed[kTaps];
#pragma unroll
    for (int k = 0; k < kTaps; ++k) needed[k] = own[k] || inBatch[k];
    reduceRows<kRowWorld, kRowWorld>(local, p.dim, t, col, needed, rows);
  } else {
    reduceRows<kRowWorld, kRowGroup>(local, p.dim, t, col, own, rows);
    if constexpr (C == 1) {
#pragma unroll
      for (int k = 0; k < kTaps - 1; ++k) {
        bool tap[kTaps] = {};
        tap[k] = inBatch[k];
        reduceRows<kRowWorld, kTapGroup>(local, p.dim, t, col, tap, rows);
      }
    } else {
      reduceRows<kRowWorld, kTapGroup>(local, p.dim, t, col, inBatch, rows);
    }
  }
  // Arriving after the reduction keeps the release of this CTA's pushes off
  // the critical path.
  flag.ctaArrive();
  cudaTriggerProgrammaticLaunchCompletion();
  float cur[kVec];
#pragma unroll
  for (int i = 0; i < kVec; ++i) cur[i] = rows[kTaps - 1][i];
  float taps[kTaps][kVec];
#pragma unroll
  for (int k = 0; k < kTaps - 1; ++k) {
    if (inBatch[k]) {
#pragma unroll
      for (int i = 0; i < kVec; ++i) taps[k][i] = rows[k][i];
    } else {
      unpackBf16(cachedRaw[k], taps[k]);
    }
  }
  if (valid) {
    storeBf16(p.cache + (slot / p.blockSize) * p.cacheBlockStride +
                  (slot % p.blockSize) * p.cacheSlotStride + cacheCol,
              cur);
  }
#pragma unroll
  for (int i = 0; i < kVec; ++i)
    taps[kTaps - 1][i] = valid && pos >= 0 ? cur[i] : 0.f;

  // Static conv, then the residual update (res + addend) + y.
  float y[kVec];
  asm volatile("cp.async.wait_all;" ::: "memory");
  {
    uint4 weightRaw[kTaps];
#pragma unroll
    for (int i = 0; i < kTaps; ++i) weightRaw[i] = slot16(kWeightSlot + i);
    bf16 const* w = reinterpret_cast<bf16 const*>(weightRaw);
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      float acc = 0.f;
#pragma unroll
      for (int k = 0; k < kTaps; ++k)
        acc = __fmaf_rn(taps[k][i], __bfloat162float(w[i * kTaps + k]), acc);
      y[i] = __fmul_rn(__fadd_rn(acc, cur[i]), p.convScale);
    }
  }
  roundBf16(y);
  if (p.resAddend != nullptr) {
    float addend[kVec], res[kVec];
    unpackBf16(slot16(slots.add), addend);
    unpackBf16(slot16(slots.res), res);
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      float a = p.res ? __fadd_rn(res[i], addend[i]) : addend[i];
      y[i] = __fadd_rn(a, y[i]);
    }
    roundBf16(y);
  }
  storeBf16(p.convOut + rowOff, y);

  // AttnRes mix over [*stack, (last +) y] and the next sublayer's RMSNorm.
  if (p.last != nullptr) {
    float last[kVec];
    unpackBf16(slot16(slots.last), last);
#pragma unroll
    for (int i = 0; i < kVec; ++i) y[i] = __fadd_rn(last[i], y[i]);
  }
  {
    float query[Q][kVec];
    loadQueries(query);
    float stats[kLastStats] = {};
    sourceStats<Q>(y, query, stats, 0);
    rowReduce(stats, scratch, lastStats);
  }
  int const numSources = p.numStack + 1;
  auto source = [&](int n, float (&v)[kVec]) {
    if (n < p.numStack) {
      unpackBf16(slot16(slots.stack + n), v);
    } else {
#pragma unroll
      for (int i = 0; i < kVec; ++i) v[i] = y[i];
    }
  };
  // The softmax over the sources, one lane per source: lane n's weight
  // (probability / RMS) of source n per query.
  float weight[Q], invTotal[Q];
  {
    int const n = threadIdx.x % 32;
    bool const present = n < numSources;
    auto stat = [&](int i) {
      return n < p.numStack ? stackStats[n * (1 + Q) + i] : lastStats[i];
    };
    float const invRms =
        present ? rsqrtApprox(__fmaf_rn(stat(0), 1.f / p.dim, p.eps)) : 0.f;
#pragma unroll
    for (int q = 0; q < Q; ++q) {
      float const logit = present ? __fmul_rn(stat(1 + q), invRms) : -INFINITY;
      float maxLogit = logit;
#pragma unroll
      for (int bit = 16; bit >= 1; bit /= 2)
        maxLogit = fmaxf(maxLogit, __shfl_xor_sync(0xffffffff, maxLogit, bit));
      float const prob = present ? expf(__fsub_rn(logit, maxLogit)) : 0.f;
      float total = prob;
#pragma unroll
      for (int bit = 16; bit >= 1; bit /= 2)
        total = __fadd_rn(total, __shfl_xor_sync(0xffffffff, total, bit));
      invTotal[q] = __frcp_rn(total);
      weight[q] = __fmul_rn(prob, invRms);
    }
  }
  float mix[Q][kVec];
#pragma unroll
  for (int q = 0; q < Q; ++q) {
#pragma unroll
    for (int i = 0; i < kVec; ++i) mix[q][i] = 0.f;
  }
#pragma unroll
  for (int n = 0; n < kMaxSources; ++n) {
    if (n >= numSources) break;
    float src[kVec];
    source(n, src);
#pragma unroll
    for (int q = 0; q < Q; ++q) {
      float const w = __shfl_sync(0xffffffff, weight[q], n);
#pragma unroll
      for (int i = 0; i < kVec; ++i)
        mix[q][i] = __fmaf_rn(w, src[i], mix[q][i]);
    }
  }
  float sumSq[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int q = 0; q < Q; ++q) {
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      mix[q][i] = __fmul_rn(mix[q][i], invTotal[q]);
      sumSq[q] = __fmaf_rn(mix[q][i], mix[q][i], sumSq[q]);
    }
  }
  rowReduce(sumSq, scratch, sumSqs);
#pragma unroll
  for (int q = 0; q < Q; ++q) {
    float const rstd =
        rsqrtApprox(__fadd_rn(__fdiv_rn(sumSqs[q], float(p.dim)), p.normEps));
    float w[kVec];
    unpackBf16(slot16(kNormSlot + q), w);
#pragma unroll
    for (int i = 0; i < kVec; ++i)
      w[i] = __fmul_rn(__fmul_rn(mix[q][i], rstd), w[i]);
    storeBf16(p.out + (int64_t(q) * p.numTokens + t) * p.dim + col, w);
  }

  // The two-shot stages hold every row once; the one-shot buffer every rank's
  // rows.
  uint32_t const bytes = p.numTokens * p.dim * sizeof(bf16);
  flag.waitAndUpdate(kTwoShot ? uint4{bytes, bytes, 0, 0}
                              : uint4{bytes * World, 0, 0, 0});
}

using Kernel = void (*)(Params);

template <int World, int In, int Q>
Kernel clusterKernel(int cluster) {
  switch (cluster) {
    case 1:
      return &allReduceConvMixKernel<World, In, Q, 1>;
    case 2:
      return &allReduceConvMixKernel<World, In, Q, 2>;
    case 3:
      return &allReduceConvMixKernel<World, In, Q, 3>;
    default:
      return &allReduceConvMixKernel<World, In, Q, 6>;
  }
}

template <int World, int In>
Kernel queryKernel(int numQueries, int cluster) {
  return numQueries == 1 ? clusterKernel<World, In, 1>(cluster)
                         : clusterKernel<World, In, 3>(cluster);
}

template <int World>
Kernel inputKernel(int in, int numQueries, int cluster) {
  switch (in) {
    case kPartial:
      return queryKernel<World, kPartial>(numQueries, cluster);
    case kCombine:
      return queryKernel<World, kCombine>(numQueries, cluster);
    case kPartialTwoShot:
      return queryKernel<World, kPartialTwoShot>(numQueries, cluster);
    default:
      return queryKernel<World, kCombineTwoShot>(numQueries, cluster);
  }
}

template <typename T>
T const* optionalPtr(std::optional<torch::stable::Tensor> const& t) {
  return t.has_value() ? static_cast<T const*>(t->data_ptr()) : nullptr;
}

}  // namespace inkling2
}  // namespace vllm

void inkling2_allreduce_conv_mix(
    std::optional<torch::stable::Tensor> x,
    std::optional<torch::stable::Tensor> down,
    std::optional<torch::stable::Tensor> src2dst, bool two_shot,
    int64_t multicast_ptr, int64_t local_ptr, int64_t buffer_ptrs,
    torch::stable::Tensor& flags, int64_t rank, int64_t world,
    torch::stable::Tensor& cache, torch::stable::Tensor const& conv_weight,
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
  Params p{};
  bool const combine = down.has_value();
  STD_TORCH_CHECK(combine == src2dst.has_value() && combine != x.has_value(),
                  "Inkling2 all-reduce takes x or (down, src2dst)");
  int in;
  if (combine) {
    in = two_shot ? kCombineTwoShot : kCombine;
    p.down = static_cast<bf16 const*>(down->data_ptr());
    p.src2dst = static_cast<int const*>(src2dst->data_ptr());
    p.topK = src2dst->numel() / conv_out.size(0);
  } else {
    in = two_shot ? kPartialTwoShot : kPartial;
    p.x = static_cast<bf16 const*>(x->data_ptr());
  }
  p.mcast = reinterpret_cast<float*>(multicast_ptr);
  p.local = reinterpret_cast<float*>(local_ptr);
  p.peers = reinterpret_cast<void* const*>(buffer_ptrs);
  p.flags = static_cast<uint32_t*>(flags.data_ptr());
  p.rank = rank;
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
  p.blockSize = cache.size(2);
  p.streamOffset = stream_offset;
  p.streamWidth = stream_width;
  p.numHeads = cache.size(1);
  p.convScale = conv_scale;
  p.res = optionalPtr<bf16>(residual);
  p.resAddend = optionalPtr<bf16>(addend);
  p.convOut = static_cast<bf16*>(conv_out.data_ptr());
  p.stack = static_cast<bf16 const*>(stack.data_ptr());
  p.stackStride = stack.stride(0);
  p.numStack = stack.size(0);
  p.last = optionalPtr<bf16>(last);
  p.queries = static_cast<float const*>(queries.data_ptr());
  p.normWeight = static_cast<bf16 const*>(norm_weight.data_ptr());
  p.numQueries = queries.size(0);
  p.eps = eps;
  p.normEps = norm_eps;
  p.out = static_cast<bf16*>(out.data_ptr());
  STD_TORCH_CHECK(p.numStack < kMaxSources && p.numQueries <= kMaxQueries);
  STD_TORCH_CHECK(p.dim == kTokenThreads * kVec,
                  "Inkling2 all-reduce requires hidden size 4608");
  STD_TORCH_CHECK(p.numQueries == 1 || p.numQueries == 3);
  STD_TORCH_CHECK(stream_width % kVec == 0 && stream_offset % kVec == 0);
  STD_TORCH_CHECK(world == 2 || world == 4 || world == 8,
                  "Inkling2 all-reduce supports TP2/4/8");
  if (p.numTokens == 0) return;

  // Each token's channels split over a cluster of CTAs (from e2e sweeps:
  // grids on much more than 100 SMs delay the next kernel's early launch).
  int const cluster = p.numTokens <= 16   ? 6
                      : p.numTokens <= 32 ? 3
                      : p.numTokens <= 48 ? 2
                      : p.numTokens <= 64 ? 3
                                          : 1;
  int const threads = kTokenThreads / cluster;
  cudaLaunchAttribute attrs[2]{};
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  attrs[1].id = cudaLaunchAttributeClusterDimension;
  attrs[1].val.clusterDim = {static_cast<unsigned>(cluster), 1, 1};
  cudaLaunchConfig_t config{};
  // Extra CTAs clear the previous call's buffer.
  config.gridDim =
      dim3(cluster, p.numTokens + (kClearCtas + cluster - 1) / cluster);
  config.blockDim = dim3(threads);
  config.stream = get_current_cuda_stream(device);
  config.attrs = attrs;
  config.numAttrs = cluster > 1 ? 2 : 1;
  config.dynamicSmemBytes = Slots(p).count * threads * sizeof(uint4);
  Kernel kernel;
  switch (world) {
    case 2:
      kernel = inputKernel<2>(in, p.numQueries, cluster);
      break;
    case 4:
      kernel = inputKernel<4>(in, p.numQueries, cluster);
      break;
    default:
      kernel = inputKernel<8>(in, p.numQueries, cluster);
      break;
  }
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       kMaxSlots * threads * sizeof(uint4));
  cudaError_t status = cudaLaunchKernelEx(&config, kernel, p);
  STD_TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
}
