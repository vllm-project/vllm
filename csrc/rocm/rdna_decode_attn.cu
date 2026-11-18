// RDNA3.5 decode attention over the paged KV cache
// (num_pages, NUM_KV_HEADS, PAGE_SIZE, 2 * HEAD_DIM), K and V packed per row,
// as a template on one entry of rdna_decode_configs.h, or of
// rdna_decode_batch_configs.h (one sequence per grid.y).
//
// One workgroup per (kv head, row group, KV segment) handles all query rows of
// its kv head, so the KV is read once. Both products are WMMA 16x16x16:
//
//   S^T[key][row] = K[key][:] . Q[row][:]        A = K tile, B = Q^T
//   O^T[d][row]  += V[key][d] . P[row][key]      A = V^T,    B = P^T
#pragma once

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <type_traits>

#include "rdna_attn_common.cuh"
#include "rdna_attn_launch.h"
#include "rdna_decode_batch_configs.h"
#include "rdna_decode_configs.h"

namespace rdna {

// Profiling only, gives wrong results: 1 skips V loads, 2 one P@V WMMA per
// tile, 4 skips K loads, 16 one Q@K WMMA per tile, 32 returns at once, 64 only
// loads, 128 skips the score exchange.
#ifdef ABLATE
inline constexpr int kDecodeAblate = ABLATE;
#else
inline constexpr int kDecodeAblate = 0;
#endif

// Six dims (D=96): three 4-byte aligned dwords.
struct u3w {
  unsigned w[3];
  __device__ unsigned& operator[](int i) { return w[i]; }
  __device__ unsigned operator[](int i) const { return w[i]; }
};

template <DecodeConfig C, bool kBatchedBuild = false>
struct Decode {
  static constexpr bool kBatched = kBatchedBuild;
  static constexpr int kHeadDim = static_cast<int>(C.head_dim);
  static constexpr int kNumQHeads = C.num_q_heads;
  static constexpr int kNumKvHeads = C.num_kv_heads;
  static constexpr int kWindow = C.window;
  static constexpr int kMaxQueryLen = C.max_query_len;
  static constexpr bool kBf16 = C.dtype == Dtype::kBf16;
  static constexpr int kPageSize = C.page_size;
  static constexpr int kMaxSegments = C.max_segments;
  static constexpr int kRowGroups = C.row_groups;
  static constexpr int kMinSegmentBlocks = C.min_segment_blocks;
  static constexpr int kWaves = C.waves;
  static constexpr bool kPrefetch = C.prefetch;
  static constexpr bool kVInLds = C.v_in_lds;
  static constexpr bool kDotProduct = C.dot_product;
  static constexpr int kHeadDimSplit =
      C.head_dim_split ? C.head_dim_split
                       : (kHeadDim >= 256 ? kHeadDim / 128 : 1);
  static constexpr int kAblate = kDecodeAblate;

  // Polls before a shared-merge waiter leaves its slice to the last arriver.
  static constexpr int kMaxSpins = 128;
  // Workgroups per WGP a batch aims for.
  static constexpr int kBatchWorkgroupsPerWgp = 4;
  static constexpr int kWaveSize = 32;
  static constexpr int kThreads = kWaves * kWaveSize;
  static constexpr int kQHeadsPerKvHead = kNumQHeads / kNumKvHeads;
  static constexpr int kKvRowElems = 2 * kHeadDim;
  static constexpr int kPageElems = kPageSize * kNumKvHeads * kKvRowElems;
  static constexpr float kLog2e = 1.44269504088896340736f;

  static_assert(kNumQHeads % kNumKvHeads == 0, "GQA must be integral");
  static_assert(kPageSize % 16 == 0, "a 16-key tile must sit inside one page");

  using elem_t = typename Elem<kBf16>::type;
  using e16 = typename Elem<kBf16>::v16;
  using e8 = typename Elem<kBf16>::v8;
  using e2 = typename Elem<kBf16>::v2;

  static __device__ __forceinline__ elem_t to_elem(float x) {
    return Elem<kBf16>::from_float(x);
  }

  static __device__ __forceinline__ f8 wmma(e16 a, e16 b, f8 c) {
    return Elem<kBf16>::wmma(a, b, c);
  }

  // P's high part; for bf16 it truncates and the low part carries the rest.
  static __device__ __forceinline__ unsigned pack2(float a, float b) {
    if constexpr (kBf16)
      return __builtin_amdgcn_perm(__builtin_bit_cast(unsigned, b),
                                   __builtin_bit_cast(unsigned, a),
                                   0x07060302u);
    else
      return __builtin_bit_cast(unsigned, e2{(elem_t)a, (elem_t)b});
  }

  static __device__ __forceinline__ float elem_part(float p) {
    if constexpr (kBf16)
      return __builtin_bit_cast(float,
                                __builtin_bit_cast(unsigned, p) & 0xFFFF0000u);
    else
      return (float)(elem_t)p;
  }

  // B = P^T: lane l (row l & 15) holds P for keys 2e + (l >> 4) in probs[e].
  // A and B use the same k order, so no lane selects on its half.
  static __device__ __forceinline__ e16 p_frag(const float* probs) {
    u8v packed;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      packed[i] = pack2(probs[2 * i], probs[2 * i + 1]);
      packed[4 + i] = xhalf_u(packed[i]);
    }
    return __builtin_bit_cast(e16, packed);
  }

  // The last key a query token at `position` sees: its own, or the end of the
  // multimodal range [start, end] it is in. `ranges` as LaunchArgs::mm_ranges.
  static __device__ __forceinline__ int mm_last_key(const int* ranges,
                                                    int capacity, int position,
                                                    int seqLen) {
    int key = position;
    if (ranges == nullptr) return key;
    const int count = min(ranges[0], capacity);
    for (int r = 0; r < count; ++r) {
      const int start = ranges[1 + 2 * r], end = ranges[2 + 2 * r];
      if (start < end && start <= position && position <= end)
        key = max(key, end);
    }
    return min(key, seqLen - 1);
  }

  // Dot-product path (C.dot_product). Its constants and asserts are
  // evaluated for every build; the asserts only bind its own.
  // One workgroup per (q head, segment); each wave takes kTileKeys keys at a
  // time with the head dim across its lanes.
  static constexpr int kDimsPerLane =
      kHeadDim == 128 ? 8 : kHeadDim / kWaveSize;
  static constexpr int kLanesPerKey = kHeadDim / kDimsPerLane;
  static constexpr int kKeysPerStep = kWaveSize / kLanesPerKey;
  static constexpr int kKeysPerLane = std::max(4 / kKeysPerStep, 1);
  static constexpr int kTileKeys = kKeysPerLane * kKeysPerStep;
  static constexpr int kPairsPerLane = kDimsPerLane / 2;
  static constexpr int kSliceStride = kDimsPerLane + 1;
  static constexpr int kPartialStride = kLanesPerKey * kSliceStride;
  static constexpr int kPartials = kWaves * kKeysPerStep;
  static constexpr int kAccBytes =
      kMaxQueryLen * kPartials * kPartialStride * 4;
  static constexpr int kStatBytes = kMaxQueryLen * kPartials * 4;

  static_assert(!kDotProduct ||
                    (kLanesPerKey >= 1 && kLanesPerKey <= kWaveSize &&
                     kWaveSize % kLanesPerKey == 0),
                "a key must be covered by a whole divisor of the wave");
  static_assert(!kDotProduct || kDimsPerLane % 2 == 0,
                "a lane's slice is whole dot2 pairs");
  static_assert(!kDotProduct || kPageSize % kTileKeys == 0,
                "a tile must not straddle two blocks");
  static_assert(!kDotProduct || kAccBytes + 2 * kStatBytes + 16 <= 65536,
                "the partials must fit LDS");

  typedef float fvec __attribute__((ext_vector_type(kPairsPerLane)));

  static constexpr int kDotLds = kAccBytes + 2 * kStatBytes + 16;
  static constexpr int kDotGrid = kMaxSegments * kNumQHeads;

  // Q heads of a kv head are adjacent in the grid, sharing the KV in L2.
  static __device__ __forceinline__ void dot_coords(int& segment, int& qHead) {
    const int workgroup = blockIdx.x;
    qHead = workgroup % kNumQHeads;
    segment = workgroup / kNumQHeads;
  }

  // The page of the wave's first tile, read before S is known.
  static __device__ __forceinline__ int dot_first_pages(
      const int* __restrict__ blockTable, int blockTableWidth) {
    int segment, qHead;
    dot_coords(segment, qHead);
    const int globalWave = __builtin_amdgcn_readfirstlane(
        segment * kWaves + threadIdx.x / kWaveSize);
    return blockTable[min(globalWave * kKeysPerLane * kKeysPerStep / kPageSize,
                          blockTableWidth - 1)];
  }

  static __device__ __forceinline__ float dot2(float a, float b, float c) {
    if constexpr (kBf16)
      return __builtin_amdgcn_fdot2_f32_bf16(
          __builtin_bit_cast(s2v, a), __builtin_bit_cast(s2v, b), c, false);
    else
      return __builtin_amdgcn_fdot2(__builtin_bit_cast(h2, a),
                                    __builtin_bit_cast(h2, b), c, false);
  }

  static __device__ __forceinline__ int partial_row(int waveBase, int partial) {
    return waveBase * kKeysPerStep + partial;
  }

  // Lane i takes lane i ^ step's value.
  static __device__ __forceinline__ float butterfly(float value, int step) {
    const int bits = __builtin_bit_cast(int, value);
    if (step >= 16)
      return __builtin_bit_cast(
          float, __builtin_amdgcn_permlanex16(bits, bits, 0x76543210u,
                                              0xFEDCBA98u, false, false));
    const unsigned selectLow = step == 1   ? 0x67452301u
                               : step == 2 ? 0x54761032u
                               : step == 4 ? 0x32107654u
                                           : 0xFEDCBA98u;
    const unsigned selectHigh = step == 1   ? 0xEFCDAB89u
                                : step == 2 ? 0xDCFE98BAu
                                : step == 4 ? 0xBA98FEDCu
                                            : 0x76543210u;
    return __builtin_bit_cast(
        float, __builtin_amdgcn_permlane16(bits, bits, selectLow, selectHigh,
                                           false, false));
  }

  template <typename OutT>
  static __device__ __forceinline__ void dot_body(
      const elem_t* __restrict__ q, const elem_t* __restrict__ kv,
      const int* __restrict__ blockTable, float* __restrict__ partialAcc,
      float* __restrict__ partialMax, float* __restrict__ partialSum,
      int* __restrict__ counters, OutT* __restrict__ out, const int seqLen,
      int pageIndices, int blockTableWidth, float scale, int cooperative,
      char* __restrict__ lds, const int* __restrict__ mmRanges,
      int mmCapacity) {
    (void)cooperative;
    if (blockIdx.x >= kDotGrid) return;
    const int thread = threadIdx.x;
    const int wave = __builtin_amdgcn_readfirstlane(thread / kWaveSize);
    int segment, qHead;
    dot_coords(segment, qHead);
    const int lane = thread & (kWaveSize - 1);
    const int keyLane = lane % kLanesPerKey;
    const int keyInStep = lane / kLanesPerKey;
    const int laneDim = keyLane * kDimsPerLane;
    const int kvHead = qHead / kQHeadsPerKvHead;

    fvec qRegs[kMaxQueryLen];
#pragma unroll
    for (int t = 0; t < kMaxQueryLen; ++t) {
      const unsigned qOffset =
          ((unsigned)t * kNumQHeads + (unsigned)qHead) * kHeadDim +
          (unsigned)laneDim;
      qRegs[t] = *(const fvec*)(q + qOffset);
    }

    float accum[kMaxQueryLen][kDimsPerLane], runMax[kMaxQueryLen],
        runSum[kMaxQueryLen];
#pragma unroll
    for (int t = 0; t < kMaxQueryLen; ++t) {
#pragma unroll
      for (int i = 0; i < kDimsPerLane; ++i) accum[t][i] = 0.f;
      runMax[t] = -INFINITY;
      runSum[t] = 0.f;
    }

    __builtin_assume(seqLen > 0);
    const float scaleLog2 = scale * kLog2e;
    const int context = seqLen - kMaxQueryLen;
    // Each token's last key: its own, or its multimodal range's end.
    int lastKey[kMaxQueryLen];
#pragma unroll
    for (int t = 0; t < kMaxQueryLen; ++t)
      lastKey[t] = mm_last_key(mmRanges, mmCapacity, context + t, seqLen);
    const int globalWave =
        __builtin_amdgcn_readfirstlane(segment * kWaves + wave);
    // With a window, start at the tile of the first visible key.
    const int firstKey = kWindow
                             ? max(0, seqLen - kMaxQueryLen - (kWindow - 1)) /
                                   kTileKeys * kTileKeys
                             : 0;
    const int keyStart = firstKey + globalWave * kKeysPerLane * kKeysPerStep;
    const int keyStep = kMaxSegments * kWaves * kTileKeys;
    // Pages are read one tile ahead.
    int page;
    if constexpr (kWindow)
      page = __builtin_amdgcn_readfirstlane(
          blockTable[min(keyStart / kPageSize, blockTableWidth - 1)]);
    else
      page = __builtin_amdgcn_readfirstlane(pageIndices);

    for (int tileKey = keyStart; tileKey < seqLen; tileKey += keyStep) {
      fvec kRegs[kKeysPerLane], vRegs[kKeysPerLane];
      __builtin_assume(page >= 0);
      const size_t base =
          (size_t)page * kPageElems +
          ((size_t)kvHead * kPageSize + (unsigned)tileKey % kPageSize) *
              kKvRowElems +
          (size_t)keyInStep * kKvRowElems + laneDim;
#pragma unroll
      for (int c = 0; c < kKeysPerLane; ++c) {
        const size_t offset = base + (size_t)c * kKeysPerStep * kKvRowElems;
        kRegs[c] = *(const fvec*)(kv + offset);
        vRegs[c] = *(const fvec*)(kv + offset + kHeadDim);
      }
      page = __builtin_amdgcn_readfirstlane(blockTable[min(
          (tileKey + keyStep) / kPageSize, blockTableWidth - 1)]);
      // Zero V past S so no 0 * NaN reaches the accumulator.
      if (__builtin_expect(tileKey + kTileKeys > seqLen, 0)) {
#pragma unroll
        for (int c = 0; c < kKeysPerLane; ++c)
          if (tileKey + c * kKeysPerStep + keyInStep >= seqLen)
            vRegs[c] = fvec{};
      }

      float scores[kKeysPerLane][kMaxQueryLen];
#pragma unroll
      for (int c = 0; c < kKeysPerLane; ++c)
#pragma unroll
        for (int t = 0; t < kMaxQueryLen; ++t) {
          float dot = 0.f;
#pragma unroll
          for (int e = 0; e < kPairsPerLane; ++e)
            dot = dot2(qRegs[t][e], kRegs[c][e], dot);
          scores[c][t] = dot;
        }
#pragma unroll
      for (int step = 1; step < kLanesPerKey; step <<= 1) {
#pragma unroll
        for (int c = 0; c < kKeysPerLane; ++c)
#pragma unroll
          for (int t = 0; t < kMaxQueryLen; ++t)
            scores[c][t] = butterfly(scores[c][t], step) + scores[c][t];
      }
#pragma unroll
      for (int c = 0; c < kKeysPerLane; ++c) {
        const int key = tileKey + c * kKeysPerStep + keyInStep;
#pragma unroll
        for (int t = 0; t < kMaxQueryLen; ++t) {
          const bool causal = key <= lastKey[t];
          if constexpr (kWindow)
            scores[c][t] =
                (key < context + t - (kWindow - 1)) ? -INFINITY : scores[c][t];
          scores[c][t] = causal ? scores[c][t] * scaleLog2 : -INFINITY;
        }
      }
#pragma unroll
      for (int t = 0; t < kMaxQueryLen; ++t) {
        float newMax = runMax[t];
#pragma unroll
        for (int c = 0; c < kKeysPerLane; ++c)
          newMax = fmaxf(newMax, scores[c][t]);
        const float rescale = (newMax == -INFINITY)
                                  ? 0.f
                                  : __builtin_amdgcn_exp2f(runMax[t] - newMax);
        runMax[t] = newMax;
        runSum[t] *= rescale;
#pragma unroll
        for (int i = 0; i < kDimsPerLane; ++i) accum[t][i] *= rescale;
        float tileSum = 0.f;
#pragma unroll
        for (int c = 0; c < kKeysPerLane; ++c) {
          const float prob =
              (newMax == -INFINITY)
                  ? 0.f
                  : __builtin_amdgcn_exp2f(scores[c][t] - newMax);
          scores[c][t] = prob;
          tileSum += prob;
        }
        runSum[t] += tileSum;
      }
#pragma unroll
      for (int c = 0; c < kKeysPerLane; ++c) {
        const elem_t* v = (const elem_t*)&vRegs[c];
#pragma unroll
        for (int t = 0; t < kMaxQueryLen; ++t)
#pragma unroll
          for (int i = 0; i < kDimsPerLane; ++i)
            accum[t][i] += scores[c][t] * (float)v[i];
      }
    }

    // Merge the waves in LDS, then the segments in global memory.
    float* const ldsAccum = reinterpret_cast<float*>(lds);
    float* const ldsMax = reinterpret_cast<float*>(lds + kAccBytes);
    float* const ldsSum = ldsMax + kMaxQueryLen * kPartials;
    volatile int& ldsIsLast =
        *reinterpret_cast<volatile int*>(ldsSum + kMaxQueryLen * kPartials);
#pragma unroll
    for (int t = 0; t < kMaxQueryLen; ++t) {
      if (keyLane == 0) {
        ldsMax[(t * kWaves + wave) * kKeysPerStep + keyInStep] = runMax[t];
        ldsSum[(t * kWaves + wave) * kKeysPerStep + keyInStep] = runSum[t];
      }
#pragma unroll
      for (int i = 0; i < kDimsPerLane; ++i)
        ldsAccum[((t * kWaves + wave) * kKeysPerStep + keyInStep) *
                     kPartialStride +
                 keyLane * kSliceStride + i] = accum[t][i];
    }
    lds_barrier();

#pragma unroll
    for (int t = 0; t < kMaxQueryLen; ++t) {
      const int waveBase = t * kWaves;
      float maxAll = -INFINITY;
#pragma unroll
      for (int p = 0; p < kPartials; ++p)
        maxAll = fmaxf(maxAll, ldsMax[partial_row(waveBase, p)]);
      float weights[kPartials], denominator = 0.f;
#pragma unroll
      for (int p = 0; p < kPartials; ++p) {
        const int row = partial_row(waveBase, p);
        weights[p] = weight_of(ldsMax[row], maxAll);
        denominator = fmaf(weights[p], ldsSum[row], denominator);
      }
      const size_t partialRow =
          ((size_t)qHead * kMaxSegments + segment) * kMaxQueryLen + t;
      if constexpr (kMaxSegments > 1) {
        if (thread == 0) {
          partialMax[partialRow] = maxAll;
          partialSum[partialRow] = denominator;
        }
      }
      for (int d = thread; d < kHeadDim; d += kThreads) {
        float numerator = 0.f;
#pragma unroll
        for (int p = 0; p < kPartials; ++p)
          numerator = fmaf(
              weights[p],
              ldsAccum[partial_row(waveBase, p) * kPartialStride +
                       (d / kDimsPerLane) * kSliceStride + d % kDimsPerLane],
              numerator);
        if constexpr (kMaxSegments == 1)
          out[((size_t)t * kNumQHeads + qHead) * kHeadDim + d] =
              to_elem(numerator * __builtin_amdgcn_rcpf(denominator));
        else
          partialAcc[partialRow * kHeadDim + d] = numerator;
      }
    }

    if constexpr (kMaxSegments > 1) {
      // The last segment to arrive merges and resets the counter.
      __threadfence();
      if (thread == 0)
        ldsIsLast = (atomicAdd(&counters[qHead], 1) == kMaxSegments - 1);
      lds_barrier();
      if (!ldsIsLast) return;
      if (thread == 0) counters[qHead] = 0;
      __threadfence();
#pragma unroll
      for (int t = 0; t < kMaxQueryLen; ++t) {
        const size_t rowBase = (size_t)qHead * kMaxSegments * kMaxQueryLen + t;
        float maxAll = -INFINITY;
#pragma unroll
        for (int s = 0; s < kMaxSegments; ++s)
          maxAll =
              fmaxf(maxAll, partialMax[rowBase + (size_t)s * kMaxQueryLen]);
        for (int d = thread; d < kHeadDim; d += kThreads) {
          float numerator = 0.f, denominator = 0.f;
#pragma unroll
          for (int s = 0; s < kMaxSegments; ++s) {
            const size_t row = rowBase + (size_t)s * kMaxQueryLen;
            const float weight = weight_of(partialMax[row], maxAll);
            denominator = fmaf(weight, partialSum[row], denominator);
            numerator = fmaf(weight, partialAcc[row * kHeadDim + d], numerator);
          }
          out[((size_t)t * kNumQHeads + qHead) * kHeadDim + d] =
              to_elem(numerator * __builtin_amdgcn_rcpf(denominator));
        }
      }
    }
  }

  // WMMA path, likewise.
  static constexpr int kRowsPerKvHead = kQHeadsPerKvHead * kMaxQueryLen;
  // Row groups take consecutive (q head, token) rows.
  static constexpr int kGroupRows = kRowsPerKvHead / kRowGroups;
  static constexpr int kRowTiles = (kGroupRows + 15) / 16;
  static constexpr int kPaddedRows = kRowTiles * 16;
  static constexpr int kKeyTiles = kWaves / kHeadDimSplit;
  static constexpr int kBlockKeys = 16 * kKeyTiles;
  static constexpr int kPartDims = kHeadDim / kHeadDimSplit;
  static constexpr int kPartChunks = kPartDims / 16;
  // Consecutive dims of a K or V row per lane.
  static constexpr int kLaneDims = kPartDims / 16;
  // P goes into P@V as a high and a low part for precision. With at most 8 rows
  // the low part uses the padding columns of the same WMMA.
  static constexpr bool kPackLowHalf = kGroupRows <= 8;
  // LDS tile rows are padded by 16 bytes to avoid bank conflicts.
  static constexpr int kTileRowBytes = kPartDims * 2 + 16;
  static constexpr int kTilesPerWave = kVInLds ? 2 : 1;
  static constexpr int kWaveTileBytes = kTilesPerWave * 16 * kTileRowBytes;
  static constexpr int kQRowPad = 8;
  static constexpr int kQBytes = kPaddedRows * (kHeadDim + kQRowPad) * 2;
  // Merge buffer: one slot of real rows per (live tile, part).
  static constexpr int kMergeRowStride = kPartDims + 4;
  static constexpr int kMergeSlotFloats =
      std::min(kGroupRows, 16) * kMergeRowStride;
  static constexpr int kMergeBudgetBytes = 36 * 1024;
  static constexpr int kSlotBytes = kHeadDimSplit * kMergeSlotFloats * 4;
  // Tiles left after the tree merge: the most that fit the budget.
  static constexpr int kFinalTiles =
      kKeyTiles * kSlotBytes <= kMergeBudgetBytes       ? kKeyTiles
      : kKeyTiles / 2 * kSlotBytes <= kMergeBudgetBytes ? kKeyTiles / 2
      : kKeyTiles / 4 * kSlotBytes <= kMergeBudgetBytes ? kKeyTiles / 4
      : kKeyTiles / 8 * kSlotBytes <= kMergeBudgetBytes ? kKeyTiles / 8
                                                        : 1;
  static constexpr int kMergeSlots = kFinalTiles == kKeyTiles
                                         ? kKeyTiles
                                         : std::max(kFinalTiles, kKeyTiles / 2);
  static constexpr int kMergeBytes =
      kKeyTiles == 1 ? 0 : kMergeSlots * kHeadDimSplit * kMergeSlotFloats * 4;
  static constexpr int kRawBytes =
      std::max(kQBytes + kWaves * kWaveTileBytes, kMergeBytes);
  // From 64 KiB of partials, every segment merges a slice.
  static constexpr bool kSharedMerge =
      !kDotProduct && kGroupRows * kHeadDim * kMaxSegments * 4 >= 64 * 1024;
  // Per-wave max and sum, the group's, then the merge flag.
  static constexpr int kStatsOffset = (kRawBytes + 15) / 16 * 16;
  static constexpr int kWmmaLds =
      kStatsOffset + (2 * kWaves * kPaddedRows + 2 * kPaddedRows) * 4 + 16;
  // Counters per (kv head, row group): arrivals, generation, two slice masks.
  static constexpr int kWmmaGrid = kMaxSegments * kNumKvHeads * kRowGroups;

  static_assert(kDotProduct || kRowsPerKvHead % kRowGroups == 0,
                "row groups split the rows evenly");
  static_assert(kDotProduct || kWaves % kHeadDimSplit == 0,
                "whole tiles per workgroup");
  static_assert(kDotProduct || kHeadDim / 16 % kHeadDimSplit == 0,
                "whole chunks per part");
  static_assert(kDotProduct || kLaneDims == 4 || kLaneDims == 6 ||
                    kLaneDims == 8,
                "a lane's K and V slice is one b64, b96 or b128");
  static_assert(kDotProduct || kKeyTiles <= kWaveSize,
                "one lane per tile loads its page");
  static_assert(kDotProduct || kHeadDimSplit == 1 ||
                    kRowTiles * 1024 <= 16 * kTileRowBytes,
                "the score exchange must fit a wave's K tile");
  static_assert(kDotProduct || !kVInLds || !kPrefetch,
                "V_IN_LDS replaces PREFETCH's second tile");

  using vrow_t =
      std::conditional_t<kLaneDims == 8, u4v,
                         std::conditional_t<kLaneDims == 6, u3w, u2v> >;
  // A six-dim slice is only 8-byte aligned.
  static constexpr int kRowChunk = kLaneDims % 4 ? 2 : 4;
  typedef float fchunk __attribute__((ext_vector_type(kRowChunk)));

  // A wave's share of a 16-key tile: half h holds keys 2e + h, each lane
  // kLaneDims dims of them.
  struct Tile {
    vrow_t k[8];
    vrow_t v[8];
  };

  // Lane t holds the page of tile t of a block, clamped to the last page.
  static __device__ __forceinline__ int block_pages(
      const int* __restrict__ blockTable, int block, int lane, int seqLen) {
    const int key = block * kBlockKeys + (lane % kKeyTiles) * 16;
    if constexpr (kKeyTiles == 1)
      return blockTable[__builtin_amdgcn_readfirstlane(min(key, seqLen - 1)) /
                        kPageSize];
    return blockTable[(unsigned)min(key, seqLen - 1) / kPageSize];
  }

  // As block_pages, before S is known: clamped to the table width.
  static __device__ __forceinline__ int block_pages_w(
      const int* __restrict__ blockTable, int block, int lane,
      int blockTableWidth) {
    const int tableIndex =
        (block * kBlockKeys + (lane % kKeyTiles) * 16) / kPageSize;
    if constexpr (kKeyTiles == 1)
      return blockTable[__builtin_amdgcn_readfirstlane(
          min(tableIndex, blockTableWidth - 1))];
    return blockTable[min(tableIndex, blockTableWidth - 1)];
  }

  // A = V^T for element `dim` of this lane's slice, in p_frag's key order.
  static __device__ __forceinline__ e16 v_frag(const vrow_t* v, int dim) {
    const int word = dim >> 1;
    const unsigned selector = (dim & 1) ? 0x07060302u : 0x05040100u;
    u8v packed;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      packed[i] =
          __builtin_amdgcn_perm(v[2 * i + 1][word], v[2 * i][word], selector);
      packed[4 + i] = xhalf_u(packed[i]);
    }
    return __builtin_bit_cast(e16, packed);
  }

  // Store a wave's real rows of one row tile, `stride` floats apart.
  static __device__ __forceinline__ void store_rows(
      float* dst, const f8* accum, int laneRow, int laneHalf,
      int stride = kMergeRowStride) {
    float* row = dst + laneRow * stride;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      float values[kLaneDims];
#pragma unroll
      for (int j = 0; j < kLaneDims; ++j) values[j] = accum[j][e];
#pragma unroll
      for (int k = 0; k < kLaneDims; k += kRowChunk)
        *(fchunk*)(row + kLaneDims * (2 * e + laneHalf) + k) =
            *(const fchunk*)(values + k);
    }
  }

  static __device__ __forceinline__ void add_rows(f8* accum, const float* src,
                                                  int laneRow, int laneHalf) {
    const float* row = src + laneRow * kMergeRowStride;
#pragma unroll
    for (int e = 0; e < 8; ++e)
#pragma unroll
      for (int k = 0; k < kLaneDims; k += kRowChunk) {
        const fchunk values =
            *(const fchunk*)(row + kLaneDims * (2 * e + laneHalf) + k);
#pragma unroll
        for (int j = 0; j < kRowChunk; ++j) accum[k + j][e] += values[j];
      }
  }

  // The first block's pages, read before S is known.
  static __device__ __forceinline__ int wmma_first_pages(
      const int* __restrict__ blockTable, int blockTableWidth) {
    const int segment = blockIdx.x / (kRowGroups * kNumKvHeads);
    return block_pages_w(blockTable, segment, threadIdx.x & (kWaveSize - 1),
                         blockTableWidth);
  }

  template <typename OutT>
  static __device__ __forceinline__ void wmma_body(
      const elem_t* __restrict__ q, const elem_t* __restrict__ kv,
      const int* __restrict__ blockTable, float* __restrict__ partialAcc,
      float* __restrict__ partialMax, float* __restrict__ partialSum,
      int* __restrict__ counters, OutT* __restrict__ out, const int seqLen,
      int pageIndices, int blockTableWidth, float scale, int cooperative,
      char* __restrict__ lds, const int* __restrict__ mmRanges,
      int mmCapacity) {
    const int thread = threadIdx.x;
    const int lane = thread & (kWaveSize - 1);
    const int wave = __builtin_amdgcn_readfirstlane(thread / kWaveSize);
    const int laneRow = lane & 15;
    const int laneHalf = lane >> 4;
    const int keyTile = wave / kHeadDimSplit;
    const int dimPart = wave % kHeadDimSplit;

    const int workgroup = blockIdx.x;
    const int rowGroup = workgroup % kRowGroups;
    const int kvHead = (workgroup / kRowGroups) % kNumKvHeads;
    const int segment = workgroup / (kRowGroups * kNumKvHeads);

    __builtin_assume(seqLen > 0);
    const int numBlocks = (seqLen + kBlockKeys - 1) / kBlockKeys;
    // With a window, the first block with a visible key.
    const int firstBlock =
        kWindow ? max(0, seqLen - kMaxQueryLen - (kWindow - 1)) / kBlockKeys
                : 0;
    int numSegments;
    if constexpr (kBatched) {
      // The host's segment cap is in the upper bits of cooperative.
      const int segmentCap = cooperative >> 1;
      cooperative &= 1;
      numSegments = max(1, min(min(kMaxSegments, segmentCap),
                               (numBlocks - firstBlock) / kMinSegmentBlocks));
    } else {
      numSegments = max(
          1, min(kMaxSegments, (numBlocks - firstBlock) / kMinSegmentBlocks));
    }
    if (segment >= numSegments) return;
    if constexpr (kWindow)
      pageIndices = block_pages(blockTable, firstBlock + segment, lane, seqLen);
    if constexpr (kAblate & 32) {
      if (seqLen == -1) out[0] = (OutT)0;
      return;
    }

    // Load Q before the KV.
    constexpr int kQLoads =
        (kPaddedRows * kHeadDim + kThreads * 8 - 1) / (kThreads * 8);
    e8 qLoads[kQLoads];
#pragma unroll
    for (int it = 0; it < kQLoads; ++it) {
      const int elem = (it * kThreads + thread) * 8;
      const int row = min(elem / kHeadDim, kGroupRows - 1),
                dim = elem % kHeadDim;
      const int qHead = kvHead * kQHeadsPerKvHead +
                        (rowGroup * kGroupRows + row) / kMaxQueryLen;
      const int token = (rowGroup * kGroupRows + row) % kMaxQueryLen;
      qLoads[it] = *(
          const e8*)(q + ((size_t)token * kNumQHeads + qHead) * kHeadDim + dim);
      if (elem / kHeadDim >= kGroupRows) qLoads[it] = e8{};
    }
    int* const generation =
        counters + kNumKvHeads * kRowGroups + kvHead * kRowGroups + rowGroup;
    // Bit s: segment s merges its own slice. Two masks, alternating per launch.
    int* const committedMask = counters + 2 * kNumKvHeads * kRowGroups +
                               2 * (kvHead * kRowGroups + rowGroup);
    const int startGeneration =
        kSharedMerge ? __hip_atomic_load(generation, __ATOMIC_RELAXED,
                                         __HIP_MEMORY_SCOPE_AGENT)
                     : 0;

    // The merge reuses the Q and K tile LDS.
    auto qShared = reinterpret_cast<elem_t(*)[kHeadDim + kQRowPad]>(lds);
    float* mergeShared = reinterpret_cast<float*>(lds);
    char* waveTile = lds + kQBytes + wave * kTilesPerWave * 16 * kTileRowBytes;
    // Partial scores reuse the wave's K tile.
    float* scoreShared = reinterpret_cast<float*>(lds + kQBytes);
    auto waveMax = reinterpret_cast<float (*)[kPaddedRows]>(lds + kStatsOffset);
    auto waveSum = reinterpret_cast<float (*)[kPaddedRows]>(
        lds + kStatsOffset + kWaves * kPaddedRows * 4);
    float* const groupMax = reinterpret_cast<float*>(
        lds + kStatsOffset + 2 * kWaves * kPaddedRows * 4);
    float* const groupSum = groupMax + kPaddedRows;

    const elem_t* kvHeadBase = kv + (size_t)kvHead * (kPageSize * kKvRowElems);
    const int laneDim = dimPart * kPartDims + kLaneDims * laneRow;
    const int firstChunk = dimPart * kPartChunks;

    // Load this wave's tile of `block` and the pages of the next block.
    auto issue = [&](Tile& tile, int block) {
      const int page = __builtin_amdgcn_readlane(pageIndices, keyTile);
      if (block + numSegments < numBlocks)
        pageIndices =
            block_pages(blockTable, block + numSegments, lane, seqLen);
      const elem_t* src =
          kvHeadBase + (size_t)page * kPageElems +
          (size_t)((block * kBlockKeys + keyTile * 16) % kPageSize) *
              kKvRowElems +
          (size_t)laneHalf * kKvRowElems + laneDim;
#pragma unroll
      for (int e = 0; e < 8; ++e)
        tile.v[e] =
            (kAblate & 1)
                ? vrow_t{}
                : *(const vrow_t*)(src + 2 * e * kKvRowElems + kHeadDim);
#pragma unroll
      for (int e = 0; e < 8; ++e)
        tile.k[e] = (kAblate & 4) ? vrow_t{}
                                  : *(const vrow_t*)(src + 2 * e * kKvRowElems);
      // Keep the loads from sinking to their first use.
      asm volatile("" ::: "memory");
    };

    Tile tileA;
    issue(tileA, firstBlock + segment);

#pragma unroll
    for (int it = 0; it < kQLoads; ++it) {
      const int elem = (it * kThreads + thread) * 8;
      if (elem < kPaddedRows * kHeadDim)
        *(e8*)&qShared[elem / kHeadDim][elem % kHeadDim] = qLoads[it];
    }

    const float scaleLog2 = scale * kLog2e;
    const int context = seqLen - kMaxQueryLen;

    float runMax[kRowTiles], runSum[kRowTiles];
    f8 accum[kRowTiles][kLaneDims];
#pragma unroll
    for (int rt = 0; rt < kRowTiles; ++rt) {
      runMax[rt] = -INFINITY;
      runSum[rt] = 0.f;
#pragma unroll
      for (int j = 0; j < kLaneDims; ++j) accum[rt][j] = f8{};
    }
    // Token of this lane's row in each row tile.
    int rowToken[kRowTiles];
#pragma unroll
    for (int rt = 0; rt < kRowTiles; ++rt) {
      const int row = rt * 16 + laneRow;
      rowToken[rt] = (row < kGroupRows)
                         ? ((rowGroup * kGroupRows + row) % kMaxQueryLen)
                         : (kMaxQueryLen - 1);
    }

    lds_barrier();

    // Stage a tile in this wave's LDS. The asm keeps the stores ordered after
    // earlier reads of the same bytes.
    auto stageKv = [&](const Tile& tile) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int offset =
            (2 * e + laneHalf) * kTileRowBytes + kLaneDims * laneRow * 2;
        *(vrow_t*)(waveTile + offset) = tile.k[e];
        *(vrow_t*)(waveTile + 16 * kTileRowBytes + offset) = tile.v[e];
      }
      asm volatile("" ::: "memory");
    };
    auto stageK = [&](const Tile& tile) {
#pragma unroll
      for (int e = 0; e < 8; ++e)
        *(vrow_t*)(waveTile + (2 * e + laneHalf) * kTileRowBytes +
                   kLaneDims * laneRow * 2) = tile.k[e];
      asm volatile("" ::: "memory");
    };

    // Q@K, softmax update and P@V for the tile staged at kTile.
    auto process = [&](Tile& tile, vrow_t* v, const char* kTile, int block) {
      const int tileKey = block * kBlockKeys + keyTile * 16;
      if constexpr (kAblate & 64) {
        unsigned x = 0;
#pragma unroll
        for (int e = 0; e < 8; ++e)
#pragma unroll
          for (int w = 0; w < kLaneDims / 2; ++w)
            x ^= tile.v[e][w] ^ tile.k[e][w];
        accum[0][0][0] += __builtin_bit_cast(float, x & 0x3f800000u);
        return;
      }
      f8 scores[kRowTiles];
#pragma unroll
      for (int rt = 0; rt < kRowTiles; ++rt) scores[rt] = f8{};
#pragma unroll
      for (int c = 0; c < kPartChunks; ++c) {
        const e16 kOperand =
            *(const e16*)(kTile + laneRow * kTileRowBytes + c * 32);
#pragma unroll
        for (int rt = 0; rt < kRowTiles; ++rt) {
          const e16 qOperand =
              *(const e16*)&qShared[rt * 16 + laneRow][(firstChunk + c) * 16];
          if (!(kAblate & 16) || c == 0)
            scores[rt] = wmma(kOperand, qOperand, scores[rt]);
        }
      }
      if constexpr (kHeadDimSplit > 1 && !(kAblate & 128)) {
        // Sum the parts in the same order in every wave so all agree exactly.
        asm volatile("" ::: "memory");
#pragma unroll
        for (int rt = 0; rt < kRowTiles; ++rt)
          *(f8*)&scoreShared[wave * kTilesPerWave * (16 * kTileRowBytes / 4) +
                             rt * 256 + lane * 8] = scores[rt];
        lds_barrier();
#pragma unroll
        for (int rt = 0; rt < kRowTiles; ++rt) {
          f8 sum = *(const f8*)&scoreShared[(wave - dimPart) * kTilesPerWave *
                                                (16 * kTileRowBytes / 4) +
                                            rt * 256 + lane * 8];
#pragma unroll
          for (int part = 1; part < kHeadDimSplit; ++part)
            sum += *(
                const f8*)&scoreShared[(wave - dimPart + part) * kTilesPerWave *
                                           (16 * kTileRowBytes / 4) +
                                       rt * 256 + lane * 8];
          scores[rt] = sum;
        }
        lds_barrier();
      }
      vrow_t ldsV[8];
      if constexpr (kVInLds) {
#pragma unroll
        for (int e = 0; e < 8; ++e)
          ldsV[e] = *(const vrow_t*)(kTile + 16 * kTileRowBytes +
                                     (2 * e + laneHalf) * kTileRowBytes +
                                     kLaneDims * laneRow * 2);
        v = ldsV;
      }

      // Zero V past S and before the window (possibly freed pages), so no
      // 0 * NaN reaches P@V.
      if (__builtin_expect(tileKey + 16 > seqLen, 0)) {
#pragma unroll
        for (int e = 0; e < 8; ++e)
          if (tileKey + 2 * e + laneHalf >= seqLen) v[e] = vrow_t{};
      }
      if constexpr (kWindow) {
        if (__builtin_expect(tileKey < context - (kWindow - 1), 0)) {
#pragma unroll
          for (int e = 0; e < 8; ++e)
            if (tileKey + 2 * e + laneHalf < context - (kWindow - 1))
              v[e] = vrow_t{};
        }
      }

      // Lane holds row laneRow, keys tileKey + 2e + laneHalf. Mask only edge
      // tiles; the scale is applied in the exponent.
      const bool pastContext = tileKey + 16 > context + 1;
      const bool belowWindow =
          kWindow && tileKey < context + kMaxQueryLen - kWindow;
#pragma unroll
      for (int rt = 0; rt < kRowTiles; ++rt) {
        if (__builtin_expect(pastContext || belowWindow, 0)) {
          // The row's last key: its token's own, or its multimodal range's
          // end. Here, not before the loop: a register the loop would hold.
          const int lastKey =
              mm_last_key(mmRanges, mmCapacity, context + rowToken[rt], seqLen);
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const int key = tileKey + 2 * e + laneHalf;
            const bool causal = key <= lastKey;
            if constexpr (kWindow)
              if (key < context + rowToken[rt] - (kWindow - 1))
                scores[rt][e] = -INFINITY;
            if (!causal) scores[rt][e] = -INFINITY;
          }
        }
        float tileMax = scores[rt][0];
#pragma unroll
        for (int e = 1; e < 8; ++e) tileMax = fmaxf(tileMax, scores[rt][e]);
        tileMax = fmaxf(tileMax, xhalf(tileMax)) * scaleLog2;
        const float newMax = fmaxf(runMax[rt], tileMax);
        const float rescale = (newMax == -INFINITY)
                                  ? 1.f
                                  : __builtin_amdgcn_exp2f(runMax[rt] - newMax);
        runMax[rt] = newMax;
        float probs[8], probsLow[8], tileSum = 0.f;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          probs[e] = (newMax == -INFINITY)
                         ? 0.f
                         : __builtin_amdgcn_exp2f(__builtin_fmaf(
                               scores[rt][e], scaleLog2, -newMax));
          probsLow[e] = probs[e] - elem_part(probs[e]);
          tileSum += probs[e];
        }
        runSum[rt] = runSum[rt] * rescale + tileSum + xhalf(tileSum);

        e16 pOperand, pLowOperand;
        float accumScale;
        if constexpr (kPackLowHalf) {
          // Columns 0..7 carry P's high part, 8..15 its low part; lanes 8..15
          // accumulate for row laneRow - 8.
          const u8v high = __builtin_bit_cast(u8v, p_frag(probs));
          const u8v low = __builtin_bit_cast(u8v, p_frag(probsLow));
          const unsigned upperMask = 0u - (unsigned)(laneRow >> 3);
          u8v packed;
#pragma unroll
          for (int i = 0; i < 8; ++i)
            packed[i] = (high[i] & ~upperMask) | (lower8(low[i]) & upperMask);
          pOperand = __builtin_bit_cast(e16, packed);
          const unsigned rescaleBits = __builtin_bit_cast(unsigned, rescale);
          accumScale =
              __builtin_bit_cast(float, (rescaleBits & ~upperMask) |
                                            (lower8(rescaleBits) & upperMask));
        } else {
          pOperand = p_frag(probs);
          pLowOperand = p_frag(probsLow);
          accumScale = rescale;
        }
#pragma unroll
        for (int j = 0; j < kLaneDims; ++j) {
          accum[rt][j] *= accumScale;
          if ((kAblate & 2) && j) continue;
          const e16 vOperand = v_frag(v, j);
          accum[rt][j] = wmma(vOperand, pOperand, accum[rt][j]);
          if constexpr (!kPackLowHalf)
            accum[rt][j] = wmma(vOperand, pLowOperand, accum[rt][j]);
        }
      }
      asm volatile("" ::: "memory");
    };

    if constexpr (kPrefetch) {
      Tile tileB;
      for (int block = firstBlock + segment; block < numBlocks;
           block += 2 * numSegments) {
        stageK(tileA);
        if (block + numSegments < numBlocks) issue(tileB, block + numSegments);
        process(tileA, tileA.v, waveTile, block);
        if (block + numSegments >= numBlocks) break;
        stageK(tileB);
        if (block + 2 * numSegments < numBlocks)
          issue(tileA, block + 2 * numSegments);
        process(tileB, tileB.v, waveTile, block + numSegments);
      }
    } else if constexpr (kVInLds) {
      for (int block = firstBlock + segment; block < numBlocks;
           block += numSegments) {
        stageKv(tileA);
        if (block + numSegments < numBlocks) issue(tileA, block + numSegments);
        process(tileA, nullptr, waveTile, block);
      }
    } else {
      for (int block = firstBlock + segment; block < numBlocks;
           block += numSegments) {
        stageK(tileA);
        process(tileA, tileA.v, waveTile, block);
        if (block + numSegments < numBlocks) issue(tileA, block + numSegments);
      }
    }
    if constexpr (kPackLowHalf) {
      // Fold the low-part columns onto their rows.
#pragma unroll
      for (int j = 0; j < kLaneDims; ++j)
#pragma unroll
        for (int e = 0; e < 8; ++e) accum[0][j][e] += upper8f(accum[0][j][e]);
    }

// Merge the waves: rescale to the workgroup max, tree-reduce until the
// partials fit the budget, then sum.
#pragma unroll
    for (int rt = 0; rt < kRowTiles; ++rt)
      if (laneHalf == 0) {
        waveMax[wave][rt * 16 + laneRow] = runMax[rt];
        waveSum[wave][rt * 16 + laneRow] = runSum[rt];
      }
    lds_barrier();
    float groupSums[kRowTiles];
#pragma unroll
    for (int rt = 0; rt < kRowTiles; ++rt) {
      const int row = rt * 16 + laneRow;
      float maxAll = -INFINITY;
#pragma unroll
      for (int w = dimPart; w < kWaves; w += kHeadDimSplit)
        maxAll = fmaxf(maxAll, waveMax[w][row]);
      float sum = 0.f;
#pragma unroll
      for (int w = dimPart; w < kWaves; w += kHeadDimSplit)
        sum += weight_of(waveMax[w][row], maxAll) * waveSum[w][row];
      groupSums[rt] = sum;
      if (keyTile == 0 && dimPart == 0 && laneHalf == 0) {
        groupMax[row] = maxAll;
        groupSum[row] = sum;
      }
      const float weight = weight_of(runMax[rt], maxAll);
#pragma unroll
      for (int j = 0; j < kLaneDims; ++j) accum[rt][j] *= weight;
    }

    const int group = kvHead * kRowGroups + rowGroup;
    const size_t partialBase =
        ((size_t)group * kMaxSegments + segment) * kPaddedRows;
    const bool publish = numSegments > 1;

#pragma unroll
    for (int rt = 0; rt < kRowTiles; ++rt) {
      const int realRows = min(16, kGroupRows - rt * 16);
      if (rt) lds_barrier();
#pragma unroll
      for (int stride = kKeyTiles / 2; stride >= kFinalTiles; stride >>= 1) {
        if (keyTile >= stride && keyTile < 2 * stride && laneRow < realRows)
          store_rows(
              mergeShared + ((keyTile - stride) * kHeadDimSplit + dimPart) *
                                kMergeSlotFloats,
              accum[rt], laneRow, laneHalf);
        lds_barrier();
        if (keyTile < stride && laneRow < realRows)
          add_rows(accum[rt],
                   mergeShared +
                       (keyTile * kHeadDimSplit + dimPart) * kMergeSlotFloats,
                   laneRow, laneHalf);
        lds_barrier();
      }
      if constexpr (kFinalTiles == 1) {
        // One tile left: write from registers.
        if (keyTile == 0 && laneRow < realRows) {
          const int row = rt * 16 + laneRow;
          if (!publish) {
            const int qHead = kvHead * kQHeadsPerKvHead +
                              (rowGroup * kGroupRows + row) / kMaxQueryLen;
            const int token = (rowGroup * kGroupRows + row) % kMaxQueryLen;
            const float inv = __builtin_amdgcn_rcpf(groupSums[rt]);
            OutT* dst = out + ((size_t)token * kNumQHeads + qHead) * kHeadDim +
                        dimPart * kPartDims;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
              elem_t x[kLaneDims];
#pragma unroll
              for (int j = 0; j < kLaneDims; ++j)
                x[j] = to_elem(accum[rt][j][e] * inv);
              *(vrow_t*)(dst + kLaneDims * (2 * e + laneHalf)) =
                  *(const vrow_t*)x;
            }
          } else {
            store_rows(partialAcc + (partialBase + rt * 16) * kHeadDim +
                           dimPart * kPartDims,
                       accum[rt], laneRow, laneHalf, kHeadDim);
          }
        }
      } else {
        if (keyTile < kFinalTiles && laneRow < realRows)
          store_rows(mergeShared +
                         (keyTile * kHeadDimSplit + dimPart) * kMergeSlotFloats,
                     accum[rt], laneRow, laneHalf);
        lds_barrier();
        // Eight dims of one row per thread.
        const int firstRow = rt * 16;
        const int numRows = min(16, kGroupRows - firstRow);
        for (int elem = thread * 8; elem < numRows * kHeadDim;
             elem += kThreads * 8) {
          const int tileRow = elem / kHeadDim, dim = elem % kHeadDim;
          const int row = firstRow + tileRow;
          const float* src =
              mergeShared +
              ((tileRow / 16) * kHeadDimSplit + dim / kPartDims) *
                  kMergeSlotFloats +
              (tileRow % 16) * kMergeRowStride + dim % kPartDims;
          f4 low = *(const f4*)src, high = *(const f4*)(src + 4);
#pragma unroll
          for (int t = 1; t < kFinalTiles; ++t) {
            low += *(const f4*)(src + t * kHeadDimSplit * kMergeSlotFloats);
            high +=
                *(const f4*)(src + t * kHeadDimSplit * kMergeSlotFloats + 4);
          }
          if (!publish) {
            const float inv = __builtin_amdgcn_rcpf(groupSum[row]);
            e8 values;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
              values[k] = to_elem(low[k] * inv);
              values[4 + k] = to_elem(high[k] * inv);
            }
            const int qHead = kvHead * kQHeadsPerKvHead +
                              (rowGroup * kGroupRows + row) / kMaxQueryLen;
            const int token = (rowGroup * kGroupRows + row) % kMaxQueryLen;
            *(e8*)(out + ((size_t)token * kNumQHeads + qHead) * kHeadDim +
                   dim) = values;
          } else {
            float* dst = partialAcc + (partialBase + row) * kHeadDim + dim;
            *(f4*)dst = low;
            *(f4*)(dst + 4) = high;
          }
        }
      }
    }
    if (!publish) return;

    // Split KV: publish (accum, max, sum). The last segment to arrive merges
    // and resets the counter. With a shared merge (whole grid resident), the
    // others wait for the generation bump and each merges a slice.
    for (int row = thread; row < kGroupRows; row += kThreads) {
      partialMax[partialBase + row] = groupMax[row];
      partialSum[partialBase + row] = groupSum[row];
    }
    __threadfence();
    cooperative = kSharedMerge && cooperative;
    volatile int& ldsGo =
        *reinterpret_cast<volatile int*>(groupSum + kPaddedRows);
    lds_barrier();
    if (thread == 0) {
      const bool last = atomicAdd(&counters[group], 1) == numSegments - 1;
      int go = last ? 2 : 0;
      if (last) {
        if (cooperative) {
          committedMask[(startGeneration + 1) & 1] = 0;
          __hip_atomic_store(generation, startGeneration + 1, __ATOMIC_RELEASE,
                             __HIP_MEMORY_SCOPE_AGENT);
        }
        counters[group] = 0;
      } else if (cooperative) {
        // Bounded, since residency is only estimated.
        for (int spin = 0; spin < kMaxSpins; ++spin) {
          if (__hip_atomic_load(generation, __ATOMIC_RELAXED,
                                __HIP_MEMORY_SCOPE_AGENT) != startGeneration) {
            // A late commit only makes two segments write the same values.
            atomicOr(&committedMask[startGeneration & 1], (int)(1u << segment));
            go = 1;
            break;
          }
          __builtin_amdgcn_s_sleep(1);
        }
      }
      ldsGo = go;
    }
    lds_barrier();
    if (!ldsGo) return;
    __threadfence();
    const bool merger = ldsGo == 2;
    constexpr int kGroupElems = kGroupRows * kHeadDim;
    const int sliceElems =
        cooperative ? (kGroupElems / 4 + numSegments - 1) / numSegments * 4
                    : kGroupElems;
    // Own slice first; the last arriver then takes every uncommitted one.
    unsigned pendingSlices = 0;
    for (int slice = cooperative ? segment : 0;;) {
      const int sliceBegin = slice * sliceElems;
      const int sliceEnd = min(kGroupElems, sliceBegin + sliceElems);
      for (int elem = sliceBegin + thread * 4; elem < sliceEnd;
           elem += kThreads * 4) {
        const int row = elem / kHeadDim, dim = elem % kHeadDim;
        float segMax[kMaxSegments], segSum[kMaxSegments];
        f4 segAcc[kMaxSegments];
#pragma unroll
        for (int s = 0; s < kMaxSegments; ++s) {
          const size_t partialRow =
              ((size_t)group * kMaxSegments + s) * kPaddedRows + row;
          segMax[s] = s < numSegments ? partialMax[partialRow] : -INFINITY;
          segSum[s] = s < numSegments ? partialSum[partialRow] : 0.f;
          segAcc[s] =
              s < numSegments
                  ? *(const f4*)(partialAcc + partialRow * kHeadDim + dim)
                  : f4{};
        }
        float maxAll = -INFINITY;
#pragma unroll
        for (int s = 0; s < kMaxSegments; ++s)
          maxAll = fmaxf(maxAll, segMax[s]);
        f4 numerator = {};
        float denominator = 0.f;
#pragma unroll
        for (int s = 0; s < kMaxSegments; ++s) {
          const float weight = weight_of(segMax[s], maxAll);
          denominator = fmaf(weight, segSum[s], denominator);
          numerator += weight * segAcc[s];
        }
        const float inv = __builtin_amdgcn_rcpf(denominator);
        const int qHead = kvHead * kQHeadsPerKvHead +
                          (rowGroup * kGroupRows + row) / kMaxQueryLen;
        const int token = (rowGroup * kGroupRows + row) % kMaxQueryLen;
        OutT* dst = out + ((size_t)token * kNumQHeads + qHead) * kHeadDim + dim;
#pragma unroll
        for (int k = 0; k < 4; ++k) dst[k] = to_elem(numerator[k] * inv);
      }
      if (!cooperative || !merger) break;
      if (slice == segment) {
        // Any view of the mask covers every slice.
        const unsigned committed =
            __hip_atomic_load(&committedMask[startGeneration & 1],
                              __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        pendingSlices = ~committed &
                        ((numSegments < 32 ? 1u << numSegments : 0u) - 1) &
                        ~(1u << segment);
      }
      if (!pendingSlices) break;
      slice = __builtin_ctz(pendingSlices);
      pendingSlices &= pendingSlices - 1;
    }
  }

  static constexpr int kLds = kDotProduct ? kDotLds : kWmmaLds;
  static constexpr int kGrid = kDotProduct ? kDotGrid : kWmmaGrid;
};

// Per-sequence strides of a batch.
struct DecodeBatch {
  int blockTableStride, accStride, statStride, counterStride, mmStride;
};

template <DecodeConfig C, bool kBatched, typename OutT>
__global__ __launch_bounds__((Decode<C, kBatched>::kThreads)) void decode_attn(
    const typename Decode<C, kBatched>::elem_t* __restrict__ q,
    const typename Decode<C, kBatched>::elem_t* __restrict__ kv,
    const int* __restrict__ blockTable, float* __restrict__ partialAcc,
    float* __restrict__ partialMax, float* __restrict__ partialSum,
    int* __restrict__ counters, OutT* __restrict__ out,
    const int* __restrict__ seqLens, int blockTableWidth, float scale,
    int cooperative, DecodeBatch batch, const int* __restrict__ mmRanges,
    int mmCapacity) {
  using K = Decode<C, kBatched>;
  __shared__ __attribute__((aligned(16))) char lds[K::kLds];
  if constexpr (kBatched) {
    const int seq = blockIdx.y;
    constexpr int kSeqElems = K::kMaxQueryLen * K::kNumQHeads * K::kHeadDim;
    q += (size_t)seq * kSeqElems;
    out += (size_t)seq * kSeqElems;
    blockTable += (size_t)seq * batch.blockTableStride;
    partialAcc += (size_t)seq * batch.accStride;
    partialMax += (size_t)seq * batch.statStride;
    partialSum += (size_t)seq * batch.statStride;
    counters += (size_t)seq * batch.counterStride;
    seqLens += seq;
    // mmStride is 0 without ranges.
    mmRanges += (size_t)seq * batch.mmStride;
  }
  // Load S and the first pages together.
  const int seqLen = __builtin_amdgcn_readfirstlane(*seqLens);
  if constexpr (K::kDotProduct) {
    const int pageIndices = K::dot_first_pages(blockTable, blockTableWidth);
    K::template dot_body<OutT>(q, kv, blockTable, partialAcc, partialMax,
                               partialSum, counters, out, seqLen, pageIndices,
                               blockTableWidth, scale, cooperative, lds,
                               mmRanges, mmCapacity);
  } else {
    const int pageIndices = K::wmma_first_pages(blockTable, blockTableWidth);
    if constexpr (kBatched) {
      // CUDA-graph padding.
      if (seqLen <= 0) return;
    }
    K::template wmma_body<OutT>(q, kv, blockTable, partialAcc, partialMax,
                                partialSum, counters, out, seqLen, pageIndices,
                                blockTableWidth, scale, cooperative, lds,
                                mmRanges, mmCapacity);
  }
}

// Arguments are checked by rdna_registry.cu.
template <DecodeConfig C, bool kBatched = false>
void launch_decode(const LaunchArgs& a) {
  using K = Decode<C, kBatched>;
  using elem_t = typename K::elem_t;
  static_assert(!kBatched || !K::kDotProduct,
                "a batched build runs the WMMA decomposition");
  // Row group fastest, then kv head, then segment.
  dim3 grid(K::kGrid, a.nseq), threads(K::kThreads);
  static const int residentWorkgroups = [&] {
    if (!K::kSharedMerge) return 0;
    int device, numWgps, perWgp = 0;
    (void)hipGetDevice(&device);
    (void)hipDeviceGetAttribute(&numWgps, hipDeviceAttributeMultiprocessorCount,
                                device);
    (void)hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &perWgp, decode_attn<C, kBatched, elem_t>, K::kThreads, 0);
    // The runtime assumes 64 KiB of LDS per WGP; gfx1151 has 128 KiB.
    hipFuncAttributes attrs;
    hipDeviceProp_t prop;
    if (hipFuncGetAttributes(&attrs, reinterpret_cast<const void*>(
                                         decode_attn<C, kBatched, elem_t>)) ==
            hipSuccess &&
        hipGetDeviceProperties(&prop, device) == hipSuccess &&
        strstr(prop.gcnArchName, "gfx1151")) {
      const int wavesPerSimd =
          std::min(16, 1536 / ((attrs.numRegs + 23) / 24 * 24));
      const int byWaves = 2 * (2 * wavesPerSimd / K::kWaves);
      const int byLds = attrs.sharedSizeBytes
                            ? 128 * 1024 / (int)attrs.sharedSizeBytes
                            : byWaves;
      perWgp = std::max(perWgp, std::min(byWaves, byLds));
    }
    return perWgp * numWgps;
  }();
  // The shared merge needs the whole grid resident.
  int cooperative = (int)(residentWorkgroups >= (int)(grid.x * grid.y));
  DecodeBatch batch{};
  if constexpr (kBatched) {
    // Cap the segments per sequence near kBatchWorkgroupsPerWgp per WGP.
    static const int batchTarget = [] {
      int device, numWgps = 1;
      (void)hipGetDevice(&device);
      (void)hipDeviceGetAttribute(
          &numWgps, hipDeviceAttributeMultiprocessorCount, device);
      return K::kBatchWorkgroupsPerWgp * std::max(1, numWgps);
    }();
    const int groups = K::kGrid / K::kMaxSegments;
    cooperative |=
        std::max(1, (batchTarget + a.nseq * groups - 1) / (a.nseq * groups))
        << 1;
    batch = {a.bt_stride, a.acc_stride, a.ml_stride, a.cnt_stride, a.mm_stride};
  }
  hipLaunchKernelGGL((decode_attn<C, kBatched, elem_t>), grid, threads, 0,
                     a.stream, static_cast<const elem_t*>(a.q),
                     static_cast<const elem_t*>(a.kv), a.bt, a.acc, a.m, a.l,
                     a.cnt, static_cast<elem_t*>(a.out), a.seq_lens, a.bt_width,
                     a.scale, cooperative, batch, a.mm_ranges, a.mm_capacity);
}

}  // namespace rdna
