// RDNA3.5 prefill attention for one sequence of M query tokens after a cached
// prefix, causal and optionally windowed, over the paged KV cache
// (num_pages, NUM_KV_HEADS, PAGE_SIZE, 2 * HEAD_DIM), as a template on one
// entry of rdna_singleseq_prefill_configs.h.
//
// PrefixLM: a token in a multimodal range [start, end] (absolute positions,
// inclusive) also sees the range's later keys. The window still bounds its
// earlier keys, as TRITON_ATTN with mm_prefix_clamp_sliding_window.
//
// FlashAttention-2 on WMMA 16x16x16. One workgroup per (kv head, query tile,
// KV segment); row r of a kv head is token r / G, q head kvHead * G + r % G.
// Each wave owns 16 rows over part of the head dim; a step's key_tile keys are
// staged in LDS.
//
//   S^T[key][row] = K[key][:] . Q[row][:]        A = K tile, B = Q^T
//   O^T[d][row]  += V[key][d] . P[row][key]      A = V^T,    B = P^T
#pragma once

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <type_traits>

#include "rdna_singleseq_attn_common.cuh"
#include "rdna_singleseq_attn_launch.h"
#include "rdna_singleseq_prefill_configs.h"

namespace rdna_singleseq {

// The largest divisor of n up to cap.
constexpr int largest_divisor_up_to(int n, int cap) {
  int g = cap < n ? cap : n;
  while (n % g) --g;
  return g;
}

// Internal linkage, as each build is instantiated in one translation unit: the
// kernel's LDS arrays are then internal and it compiles as a non-template
// kernel would.
namespace {

template <PrefillConfig C>
struct Prefill {
  static constexpr int kHeadDim = static_cast<int>(C.head_dim);
  static constexpr int kNumQHeads = C.num_q_heads;
  static constexpr int kNumKvHeads = C.num_kv_heads;
  static constexpr int kWindow = C.window;
  static constexpr bool kBf16 = C.dtype == Dtype::kBf16;
  static constexpr int kPageSize = C.page_size;
  static constexpr int kWaves = C.waves;
  static constexpr int kKeyTile = C.key_tile;
  static constexpr int kDimParts = C.head_dim_split;
  static constexpr bool kVT = C.v_t;
  static constexpr bool kDoubleBuffer = C.double_buffer;
  static constexpr int kKeyWaves = C.key_waves;
  static constexpr int kMaxSegments = C.max_segments;

  // Fewest steps per KV segment.
  static constexpr int kMinSegmentSteps = 16;

  static constexpr int kWaveSize = 32;
  static constexpr int kThreads = kWaves * kWaveSize;
  static constexpr int kG = kNumQHeads / kNumKvHeads;
  static constexpr int kRowWaves = kWaves / (kDimParts * kKeyWaves);
  static constexpr int kBlockRows = kRowWaves * 16;
  static constexpr int kSubTiles = kKeyTile / 16;
  static constexpr int kWaveSubTiles = kSubTiles / kKeyWaves;
  static constexpr int kPartDims = kHeadDim / kDimParts;
  static constexpr int kPartChunks = kPartDims / 16;
  // Consecutive dims of a V row per lane.
  static constexpr int kLaneDims = kPartDims / 16;
  static constexpr int kLaneWords = kLaneDims / 2;
  static constexpr int kKvRowElems = 2 * kHeadDim;
  static constexpr int kPageElems = kPageSize * kNumKvHeads * kKvRowElems;
  // LDS rows are padded by 16 bytes to avoid bank conflicts.
  static constexpr int kRowPitch = kHeadDim + 8;
  // V_T row pitch, chosen to avoid bank conflicts.
  static constexpr int kVtPitch = kKeyTile == 16 ? 16 : kKeyTile + 16;
  static constexpr int kBuffers = kDoubleBuffer ? 2 : 1;
  static constexpr int kRowChunks = kHeadDim / 8;
  static constexpr int kKUnits = kKeyTile * kRowChunks;
  // With V_T a V unit is keys kk and kk + 2 of a sub-tile.
  static constexpr int kVUnits = (kVT ? kKeyTile / 2 : kKeyTile) * kRowChunks;
  static constexpr int kKLoads = (kKUnits + kThreads - 1) / kThreads;
  static constexpr int kVLoads = (kVUnits + kThreads - 1) / kThreads;
  static constexpr int kVPerUnit = kVT ? 2 : 1;
  // Pages a step spans.
  static constexpr int kStepPages =
      kKeyTile > kPageSize ? kKeyTile / kPageSize : 1;
  static constexpr float kLog2e = 1.44269504088896340736f;
  static constexpr float kMaxSlack = 8.f;
  // D=512 uses buffer loads to save the registers of 64-bit addresses.
  static constexpr bool kBufferLoads = kHeadDim >= 512;

  static constexpr int kHeadGroup =
      largest_divisor_up_to(kNumKvHeads, C.head_group);

  static_assert(kNumQHeads % kNumKvHeads == 0, "GQA must be integral");
  static_assert(kWaves % (kDimParts * kKeyWaves) == 0, "whole row tiles");
  static_assert(kHeadDim % (16 * kDimParts) == 0, "whole chunks per part");
  static_assert(kSubTiles % kKeyWaves == 0, "whole sub-tiles per key wave");
  static_assert(kLaneDims % 2 == 0, "a lane's V slice is whole dwords");
  static_assert(kKeyTile % 16 == 0 && kPageSize % 16 == 0,
                "a 16-key sub-tile sits inside one page");
  static_assert(kRowChunks % 4 == 0,
                "a wave's keys of one load sit in one sub-tile, so one page");

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

  // The other half-wave's value, through LDS when the LDS is otherwise idle.
  static __device__ __forceinline__ unsigned xhalf_p(unsigned value) {
    if constexpr (kDimParts == 1)
      return (unsigned)__builtin_amdgcn_ds_bpermute(
          (int)(((threadIdx.x & 31) ^ 16) << 2), (int)value);
    else
      return xhalf_u(value);
  }

  static __device__ __forceinline__ unsigned pack2(float a, float b) {
    return __builtin_bit_cast(unsigned, e2{to_elem(a), to_elem(b)});
  }

  // B = P^T: lane l (row l & 15) holds P for keys 2e + (l >> 4) in probs[e].
  static __device__ __forceinline__ e16 p_frag(const float* probs) {
    u8v packed;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      packed[i] = pack2(probs[2 * i], probs[2 * i + 1]);
      packed[4 + i] = xhalf_p(packed[i]);
    }
    return __builtin_bit_cast(e16, packed);
  }

  struct VRow {
    unsigned w[kLaneWords];
  };

  // A = V^T for element `dim` of this lane's slice, in p_frag's key order.
  static __device__ __forceinline__ e16 v_frag(const VRow* v, int dim) {
    const int word = dim >> 1;
    const unsigned selector = (dim & 1) ? 0x07060302u : 0x05040100u;
    u8v packed;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      packed[i] = __builtin_amdgcn_perm(v[2 * i + 1].w[word], v[2 * i].w[word],
                                        selector);
      packed[4 + i] = xhalf_u(packed[i]);
    }
    return __builtin_bit_cast(e16, packed);
  }

  // V_T layout: the row of a dim, and the slot of a key (even keys first).
  static __device__ __forceinline__ int vt_row(int dim) {
    const int part = dim / kPartDims, local = dim % kPartDims;
    return part * kPartDims + (local % kLaneDims) * 16 + local / kLaneDims;
  }

  static __device__ __forceinline__ int vt_slot(int key) {
    return (key & 1) * 8 + (key >> 1);
  }

  // LDS per workgroup, and the waves per SIMD the register budget targets.
  static constexpr int kLdsBytes =
      kBuffers * kKeyTile * kRowPitch * sizeof(elem_t) +
      kBuffers * (kVT ? kHeadDim * kVtPitch : kKeyTile * kRowPitch) *
          sizeof(elem_t) +
      (kDimParts > 1 ? kWaves * kWaveSubTiles * 256 * 4 : 0) +
      (kKeyWaves > 1 ? (2 * kKeyWaves * kBlockRows + kWaves * kWaveSize * 8) * 4
                     : 0);
  static constexpr int kSimdWaves = kWaves >= 4 ? kWaves / 4 : 1;
  static constexpr int kWavesPerSimd =
      std::min(kWaves >= 16 ? 8 : 6, 131072 / kLdsBytes * kWaves / 4) /
      kSimdWaves * kSimdWaves;

  static __global__ void __launch_bounds__(kThreads)
      __attribute__((amdgpu_waves_per_eu(kWavesPerSimd > 1 ? kWavesPerSimd
                                                           : 1)))
      prefill_attn(const elem_t* __restrict__ q, const elem_t* __restrict__ kv,
                   const int* __restrict__ blockTable, elem_t* __restrict__ out,
                   const int* __restrict__ seqLens, int numTokens, float scale,
                   float* __restrict__ partialO, float* __restrict__ partialML,
                   int* __restrict__ counters, int numSegments, int firstRound,
                   const int* __restrict__ mmRanges, int numMmRanges) {
    __shared__ __attribute__((
        aligned(16))) elem_t kTile[kBuffers][kKeyTile][kRowPitch];
    __shared__ __attribute__((aligned(16))) elem_t
        vTile[kBuffers][kVT ? kHeadDim : kKeyTile][kVT ? kVtPitch : kRowPitch];
    __shared__ __attribute__((aligned(
        16))) float scoreX[kDimParts > 1 ? kWaves * kWaveSubTiles * 256 : 1];

    const int thread = threadIdx.x;
    const int lane = thread & (kWaveSize - 1);
    const int wave = __builtin_amdgcn_readfirstlane(thread / kWaveSize);
    const int laneRow = lane & 15;
    const int laneHalf = lane >> 4;
    const int dimPart = wave % kDimParts;
    const int keyWave = wave / kDimParts % kKeyWaves;
    const int rowWave = wave / (kDimParts * kKeyWaves);
    const int firstChunk = dimPart * kPartChunks;

    // A tile's segments are adjacent; the heaviest tiles (last tokens) go
    // first, interleaving kHeadGroup kv heads.
    const int numRows = numTokens * kG;
    const int numTiles = (numRows + kBlockRows - 1) / kBlockRows;
    const int segment = (int)blockIdx.x % numSegments;
    const int block = (int)blockIdx.x / numSegments;
    const int groupBlocks = kHeadGroup * numTiles;
    const int within = block % groupBlocks;
    const int tile = numTiles - 1 - within / kHeadGroup;
    const int kvHead = block / groupBlocks * kHeadGroup + within % kHeadGroup;

    const int seqLen = __builtin_amdgcn_readfirstlane(*seqLens);
    const int context = seqLen - numTokens;
    const int tileRow = tile * kBlockRows;
    const int firstToken = tileRow / kG;
    const int lastToken = min(numTokens - 1, (tileRow + kBlockRows - 1) / kG);
    // The last key any of tokens first..last sees: the last token, or the end
    // of a multimodal range they reach into.
    auto last_key = [&](int first, int last) __attribute__((always_inline)) {
      int key = context + last;
      if (numMmRanges == 0) return key;
      for (int r = 0; r < numMmRanges; ++r) {
        const int start = mmRanges[2 * r], end = mmRanges[2 * r + 1];
        if (start < end && start <= context + last && end >= context + first)
          key = max(key, end);
      }
      return min(key, seqLen - 1);
    };
    // Keys from the first token's window to the last key.
    const int keyEnd = last_key(firstToken, lastToken) + 1;
    const int windowFirst =
        kWindow ? max(0, context + firstToken - (kWindow - 1)) : 0;
    const int tileKeyBegin = windowFirst / kKeyTile * kKeyTile;
    // This segment's steps, possibly none.
    const int numSteps = (keyEnd - tileKeyBegin + kKeyTile - 1) / kKeyTile;
    const int keyBegin =
        tileKeyBegin + segment * numSteps / numSegments * kKeyTile;
    const int stepsEnd = min(keyEnd, tileKeyBegin + (segment + 1) * numSteps /
                                                        numSegments * kKeyTile);
    const int lastPage = (seqLen - 1) / kPageSize;

    const int waveRow = tileRow + rowWave * 16;
    const int row = min(waveRow + laneRow, numRows - 1);
    const int rowToken = row / kG;
    const int rowHead = kvHead * kG + row % kG;
    const int waveFirstToken = waveRow / kG;
    const int waveLastToken = min(numTokens - 1, (waveRow + 15) / kG);
    const int waveLastKey = last_key(waveFirstToken, waveLastToken);
    const int rowLastKey = last_key(rowToken, rowToken);

    // Stagger the Q loads of the first resident workgroups so the heaviest
    // tiles start first.
    {
      constexpr int kStaggerUnits = std::max(
          1, kBlockRows * kHeadDim * (int)sizeof(elem_t) * 3 / (2 * 8192));
      const int place = __builtin_amdgcn_readfirstlane((int)blockIdx.x);
      // Only when tile lengths differ and the KV is not split.
      const bool unequal = numSegments == 1 && __builtin_amdgcn_readfirstlane(
                                                   *seqLens) <= 2 * numTokens;
      if (unequal && place < firstRound)
        for (int i = 0; i < place * kStaggerUnits; ++i)
          __builtin_amdgcn_s_sleep(1);
    }

    // Lane l holds page l % kStepPages of a step, fetched a step ahead.
    auto step_pages = [&](int keyStart) __attribute__((always_inline)) {
      return blockTable[min(keyStart / kPageSize + lane % kStepPages,
                            lastPage)];
    };
    int pagesNext = step_pages(keyBegin);

    e16 qOperand[kPartChunks];
#pragma unroll
    for (int c = 0; c < kPartChunks; ++c)
      qOperand[c] = *(
          const e16*)(q + ((size_t)rowToken * kNumQHeads + rowHead) * kHeadDim +
                      (firstChunk + c) * 16);

    const elem_t* kvHeadBase = kv + (size_t)kvHead * (kPageSize * kKvRowElems);
    e8 kStaged[kKLoads];
    e8 vStaged[kVLoads][kVPerUnit];
    auto v_unit_pair = [&](int unit) __attribute__((always_inline)) {
      if constexpr (kVT)
        return unit / (8 * kRowChunks) * 8 + unit % 8;
      else
        return unit / kRowChunks;
    };
    auto v_unit_col = [&](int unit) __attribute__((always_inline)) {
      if constexpr (kVT)
        return unit / 8 % kRowChunks * 8;
      else
        return unit % kRowChunks * 8;
    };
    auto v_unit_key = [&](int unit) __attribute__((always_inline)) {
      const int pair = v_unit_pair(unit);
      if constexpr (kVT)
        return pair / 8 * 16 + (pair % 8 >> 1) * 4 + (pair & 1);
      else
        return pair;
    };
    // Steps with keys past S or before the window (possibly freed pages) clamp
    // their loads and store zeros, so no 0 * NaN reaches P@V.
    auto edge_step = [&](int keyStart) __attribute__((always_inline)) {
      return keyStart + kKeyTile > seqLen || keyStart < windowFirst;
    };
    // A wave's keys of one load sit in one page.
    auto load_row = [&]<bool kEdge>(int pages, int keyStart, int key,
                                    int col) __attribute__((always_inline)) {
      if constexpr (!kEdge) {
        const int first = keyStart % kPageSize;
        const int slot =
            (first + __builtin_amdgcn_readfirstlane(key)) / kPageSize;
        const elem_t* base =
            kvHeadBase +
            (size_t)__builtin_amdgcn_readlane(pages, slot) * kPageElems +
            (first - slot * kPageSize) * kKvRowElems;
        return *(const e8*)(base + (unsigned)(key * kKvRowElems + col));
      }
      const int position = min(keyStart + key, seqLen - 1);
      const int slot = __builtin_amdgcn_readfirstlane(position / kPageSize -
                                                      keyStart / kPageSize);
      const elem_t* page =
          kvHeadBase +
          (size_t)__builtin_amdgcn_readlane(pages, slot) * kPageElems;
      return *(
          const e8*)(page +
                     (unsigned)((position % kPageSize) * kKvRowElems + col));
    };
    // Per-lane byte offsets of the buffer loads, kept in registers.
    int kOff[kKLoads], vOff[kVLoads][kVPerUnit];
    if constexpr (kBufferLoads) {
#pragma unroll
      for (int i = 0; i < kKLoads; ++i) {
        const int unit = i * kThreads + thread;
        const int waveKey = (i * kThreads + wave * kWaveSize) / kRowChunks;
        kOff[i] = ((unit / kRowChunks - waveKey) * kKvRowElems +
                   unit % kRowChunks * 8) *
                  (int)sizeof(elem_t);
        asm volatile("" : "+v"(kOff[i]));
      }
#pragma unroll
      for (int i = 0; i < kVLoads; ++i) {
        const int unit = i * kThreads + thread;
        const int waveKey = v_unit_key(i * kThreads + wave * kWaveSize);
#pragma unroll
        for (int x = 0; x < kVPerUnit; ++x) {
          vOff[i][x] = ((v_unit_key(unit) - waveKey) * kKvRowElems + kHeadDim +
                        v_unit_col(unit)) *
                       (int)sizeof(elem_t);
          asm volatile("" : "+v"(vOff[i][x]));
        }
      }
    }
    // A buffer load relative to the wave's first key `waveKey`.
    auto load_buffer = [&](int pages, int keyStart, int waveKey,
                           int offset) __attribute__((always_inline)) {
      const int first = keyStart % kPageSize;
      const int slot = (first + waveKey) / kPageSize;
      const elem_t* base =
          kvHeadBase +
          (size_t)__builtin_amdgcn_readlane(pages, slot) * kPageElems +
          (first - slot * kPageSize + waveKey) * kKvRowElems;
      const __amdgpu_buffer_rsrc_t rsrc = __builtin_amdgcn_make_buffer_rsrc(
          (void*)base, 0, 0x7fffffff, 0x31004000);
      return __builtin_bit_cast(
          e8, __builtin_amdgcn_raw_buffer_load_b128(rsrc, offset, 0, 0));
    };
    // Load the step at keyStart into registers.
    auto issue_step =
        [&]<bool kEdge>(int keyStart) __attribute__((always_inline)) {
          const int pages = pagesNext;
          pagesNext = step_pages(keyStart + kKeyTile);
          if constexpr (kBufferLoads && !kEdge) {
#pragma unroll
            for (int i = 0; i < kKLoads; ++i)
              if (kKUnits % kThreads == 0 || i * kThreads + thread < kKUnits)
                kStaged[i] = load_buffer(
                    pages, keyStart,
                    (i * kThreads + wave * kWaveSize) / kRowChunks, kOff[i]);
#pragma unroll
            for (int i = 0; i < kVLoads; ++i)
              if (kVUnits % kThreads == 0 || i * kThreads + thread < kVUnits) {
                const int waveKey = v_unit_key(i * kThreads + wave * kWaveSize);
#pragma unroll
                for (int x = 0; x < kVPerUnit; ++x)
                  vStaged[i][x] =
                      load_buffer(pages, keyStart, waveKey + 2 * x, vOff[i][x]);
              }
            return;
          }
#pragma unroll
          for (int i = 0; i < kKLoads; ++i) {
            const int unit = i * kThreads + thread;
            if (kKUnits % kThreads == 0 || unit < kKUnits)
              kStaged[i] = load_row.template operator()<kEdge>(
                  pages, keyStart, unit / kRowChunks, unit % kRowChunks * 8);
          }
#pragma unroll
          for (int i = 0; i < kVLoads; ++i) {
            const int unit = i * kThreads + thread;
            if (kVUnits % kThreads == 0 || unit < kVUnits) {
              const int key = v_unit_key(unit), col = v_unit_col(unit);
#pragma unroll
              for (int x = 0; x < kVPerUnit; ++x)
                vStaged[i][x] = load_row.template operator()<kEdge>(
                    pages, keyStart, key + 2 * x, kHeadDim + col);
            }
          }
        };
    auto valid = [&](int position) __attribute__((always_inline)) {
      return position < seqLen && position >= windowFirst;
    };
    // Store the loaded step to LDS buffer `buffer`.
    auto stage_step = [&]<bool kEdge, bool kDoK = true, bool kDoV = true>(
                          int buffer,
                          int keyStart) __attribute__((always_inline)) {
#pragma unroll
      for (int i = 0; i < (kDoK ? kKLoads : 0); ++i) {
        const int unit = i * kThreads + thread;
        if (kKUnits % kThreads == 0 || unit < kKUnits) {
          const int key = unit / kRowChunks, col = unit % kRowChunks * 8;
          *(e8*)&kTile[buffer][key][col] =
              !kEdge || valid(keyStart + key) ? kStaged[i] : e8{};
        }
      }
#pragma unroll
      for (int i = 0; i < (kDoV ? kVLoads : 0); ++i) {
        const int unit = i * kThreads + thread;
        if (kVUnits % kThreads == 0 || unit < kVUnits) {
          const int key = v_unit_key(unit), col = v_unit_col(unit);
          if constexpr (kVT) {
            const e8 a = !kEdge || valid(keyStart + key) ? vStaged[i][0] : e8{};
            const e8 b =
                !kEdge || valid(keyStart + key + 2) ? vStaged[i][1] : e8{};
            const u4v aw = __builtin_bit_cast(u4v, a),
                      bw = __builtin_bit_cast(u4v, b);
            const int slot = key / 16 * 16 + vt_slot(key % 16);
#pragma unroll
            for (int d = 0; d < 8; ++d)
              *(unsigned*)&vTile[buffer][vt_row(col + d)][slot] =
                  __builtin_amdgcn_perm(bw[d >> 1], aw[d >> 1],
                                        (d & 1) ? 0x07060302u : 0x05040100u);
          } else {
            *(e8*)&vTile[buffer][key][col] =
                !kEdge || valid(keyStart + key) ? vStaged[i][0] : e8{};
          }
        }
      }
    };
    auto issue = [&](int keyStart) __attribute__((always_inline)) {
      if (edge_step(keyStart))
        issue_step.template operator()<true>(keyStart);
      else
        issue_step.template operator()<false>(keyStart);
    };
    auto stage = [&](int buffer, int keyStart) __attribute__((always_inline)) {
      if (edge_step(keyStart))
        stage_step.template operator()<true>(buffer, keyStart);
      else
        stage_step.template operator()<false>(buffer, keyStart);
    };
    auto stage_k = [&](int buffer,
                       int keyStart) __attribute__((always_inline)) {
      if (edge_step(keyStart))
        stage_step.template operator()<true, true, false>(buffer, keyStart);
      else
        stage_step.template operator()<false, true, false>(buffer, keyStart);
    };
    auto stage_v = [&](int buffer,
                       int keyStart) __attribute__((always_inline)) {
      if (edge_step(keyStart))
        stage_step.template operator()<true, false, true>(buffer, keyStart);
      else
        stage_step.template operator()<false, false, true>(buffer, keyStart);
    };
    // One buffer with a head-dim split: stage the next K after the score
    // exchange and the next V after the step, saving a barrier.
    constexpr bool kSplitStage = !kDoubleBuffer && kDimParts > 1;

    const float scaleLog2 = scale * kLog2e;
    float runMax = -INFINITY, runSum = 0.f;
    f8 accum[kLaneDims];
#pragma unroll
    for (int j = 0; j < kLaneDims; ++j) accum[j] = f8{};

    // Q@K, softmax update and P@V for the step at keyStart in `buffer`. Lane
    // holds row laneRow, keys keyStart + 16 st + 2e + laneHalf.
    auto compute = [&](int buffer, int keyStart,
                       int stageNext = -1) __attribute__((always_inline)) {
      // Skip sub-tiles no row of the wave sees.
      bool live[kWaveSubTiles];
#pragma unroll
      for (int ls = 0, st = keyWave; ls < kWaveSubTiles;
           ++ls, st += kKeyWaves) {
        const int first = keyStart + st * 16;
        live[ls] = first <= waveLastKey &&
                   (!kWindow ||
                    first + 15 >= context + waveFirstToken - (kWindow - 1));
      }
      f8 scores[kWaveSubTiles];
#pragma unroll
      for (int ls = 0; ls < kWaveSubTiles; ++ls) scores[ls] = f8{};
      auto qk = [&]<bool kCheck>() __attribute__((always_inline)) {
#pragma unroll
        for (int c = 0; c < kPartChunks; ++c)
#pragma unroll
          for (int ls = 0, st = keyWave; ls < kWaveSubTiles;
               ++ls, st += kKeyWaves) {
            const e16 kOperand = *(const e16*)&kTile[buffer][st * 16 + laneRow]
                                                    [(firstChunk + c) * 16];
            if (!kCheck || live[ls])
              scores[ls] = wmma(kOperand, qOperand[c], scores[ls]);
          }
      };
      bool allLive = true;
#pragma unroll
      for (int ls = 0; ls < kWaveSubTiles; ++ls) allLive = allLive && live[ls];
      if constexpr (kWaveSubTiles == 1) {
        if (live[0]) qk.template operator()<false>();
      } else if (__builtin_amdgcn_readfirstlane((int)allLive)) {
        // All sub-tiles live; read operands at most four WMMAs ahead.
#pragma unroll
        for (int c = 0; c < kPartChunks; ++c)
#pragma unroll
          for (int ls = 0, st = keyWave; ls < kWaveSubTiles;
               ++ls, st += kKeyWaves) {
            scores[ls] = wmma(*(const e16*)&kTile[buffer][st * 16 + laneRow]
                                                 [(firstChunk + c) * 16],
                              qOperand[c], scores[ls]);
            if ((c * kWaveSubTiles + ls + 1) % 4 == 0) {
              __builtin_amdgcn_sched_group_barrier(0x0100, 8, 0);  // DS reads
              __builtin_amdgcn_sched_group_barrier(0x0008, 4, 0);  // WMMAs
            }
          }
      } else {
        qk.template operator()<true>();
      }
      if constexpr (kDimParts > 1) {
        // Sum the parts in the same order in every wave so all agree exactly.
        auto put_scores = [&](int slot, f8 x) __attribute__((always_inline)) {
          *(f4*)&scoreX[slot * 256 + lane * 4] =
              __builtin_shufflevector(x, x, 0, 1, 2, 3);
          *(f4*)&scoreX[slot * 256 + 128 + lane * 4] =
              __builtin_shufflevector(x, x, 4, 5, 6, 7);
        };
        auto get_scores = [&](int slot) __attribute__((always_inline)) {
          const f4 lo = *(const f4*)&scoreX[slot * 256 + lane * 4];
          const f4 hi = *(const f4*)&scoreX[slot * 256 + 128 + lane * 4];
          return (f8)__builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
        };
#pragma unroll
        for (int ls = 0; ls < kWaveSubTiles; ++ls)
          put_scores(wave * kWaveSubTiles + ls, scores[ls]);
        lds_barrier();
        if constexpr (kSplitStage)
          if (stageNext >= 0) stage_k(0, stageNext);
#pragma unroll
        for (int ls = 0; ls < kWaveSubTiles; ++ls) {
          if constexpr (kDimParts == 2) {
            scores[ls] += get_scores((wave ^ 1) * kWaveSubTiles + ls);
          } else {
            f8 sum = {};
#pragma unroll
            for (int part = 0; part < kDimParts; ++part)
              sum += get_scores((wave - dimPart + part) * kWaveSubTiles + ls);
            scores[ls] = sum;
          }
        }
      }

      // Mask only diagonal and window-edge steps. No row's last key is before
      // the first token's.
      const bool causalEdge = keyStart + kKeyTile - 1 > context + firstToken;
      const bool windowEdge =
          kWindow && keyStart < context + lastToken - (kWindow - 1);
      if (__builtin_expect(causalEdge || windowEdge, 0)) {
#pragma unroll
        for (int ls = 0, st = keyWave; ls < kWaveSubTiles;
             ++ls, st += kKeyWaves)
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const int key = keyStart + st * 16 + 2 * e + laneHalf;
            if (key > rowLastKey) scores[ls][e] = -INFINITY;
            if constexpr (kWindow)
              if (key < context + rowToken - (kWindow - 1))
                scores[ls][e] = -INFINITY;
          }
      }
      float tileMax = -INFINITY;
#pragma unroll
      for (int ls = 0; ls < kWaveSubTiles; ++ls)
#pragma unroll
        for (int e = 0; e < 8; ++e) tileMax = fmaxf(tileMax, scores[ls][e]);
      tileMax = fmaxf(tileMax, xhalf(tileMax)) * scaleLog2;
      // Lazy rescale: a row's max moves only when exceeded by kMaxSlack (log2).
      const float newMax = tileMax > runMax + kMaxSlack ? tileMax : runMax;
      const float safeMax = newMax == -INFINITY ? 0.f : newMax;
      const float rescale = __builtin_amdgcn_exp2f(runMax - safeMax);
      runMax = newMax;
      if (__builtin_amdgcn_ballot_w32(rescale != 1.f))
#pragma unroll
        for (int j = 0; j < kLaneDims; ++j) accum[j] *= rescale;
      e16 pOperand[kWaveSubTiles];
      float tileSum = 0.f;
#pragma unroll
      for (int ls = 0; ls < kWaveSubTiles; ++ls) {
        float probs[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          probs[e] = __builtin_amdgcn_exp2f(
              __builtin_fmaf(scores[ls][e], scaleLog2, -safeMax));
          tileSum += probs[e];
        }
        pOperand[ls] = p_frag(probs);
      }
      runSum = runSum * rescale + tileSum + xhalf(tileSum);

#pragma unroll
      for (int ls = 0, st = keyWave; ls < kWaveSubTiles;
           ++ls, st += kKeyWaves) {
        VRow v[8];
        if constexpr (!kVT) {
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const unsigned* src =
                (const unsigned*)&vTile[buffer][st * 16 + 2 * e + laneHalf]
                                       [dimPart * kPartDims +
                                        kLaneDims * laneRow];
#pragma unroll
            for (int w = 0; w < kLaneWords; ++w) v[e].w[w] = src[w];
          }
        }
#pragma unroll
        for (int j = 0; j < kLaneDims; ++j) {
          e16 vOperand;
          if constexpr (kVT) {
            const elem_t* vRow =
                &vTile[buffer][dimPart * kPartDims + j * 16 + laneRow][st * 16];
            const e8 own = *(const e8*)(vRow + laneHalf * 8);
            const e8 other = *(const e8*)(vRow + (1 - laneHalf) * 8);
            vOperand = __builtin_shufflevector(own, other, 0, 1, 2, 3, 4, 5, 6,
                                               7, 8, 9, 10, 11, 12, 13, 14, 15);
          } else {
            vOperand = v_frag(v, j);
          }
          if (live[ls]) accum[j] = wmma(vOperand, pOperand[ls], accum[j]);
        }
      }
    };

    if (keyBegin < stepsEnd) {
      if constexpr (kDoubleBuffer) {
        issue(keyBegin);
        stage(0, keyBegin);
        if (keyBegin + kKeyTile < stepsEnd) issue(keyBegin + kKeyTile);
        lds_barrier();
        int buffer = 0;
        for (int keyStart = keyBegin; keyStart < stepsEnd;
             keyStart += kKeyTile) {
          compute(buffer, keyStart);
          if (keyStart + kKeyTile < stepsEnd) {
            stage(buffer ^ 1, keyStart + kKeyTile);
            if (keyStart + 2 * kKeyTile < stepsEnd)
              issue(keyStart + 2 * kKeyTile);
          }
          lds_barrier();
          buffer ^= 1;
        }
      } else if constexpr (kSplitStage) {
        issue(keyBegin);
        stage(0, keyBegin);
        if (keyBegin + kKeyTile < stepsEnd) issue(keyBegin + kKeyTile);
        lds_barrier();
        for (int keyStart = keyBegin; keyStart < stepsEnd;
             keyStart += kKeyTile) {
          const int next =
              keyStart + kKeyTile < stepsEnd ? keyStart + kKeyTile : -1;
          compute(0, keyStart, next);
          lds_barrier();
          if (next >= 0) {
            stage_v(0, next);
            if (next + kKeyTile < stepsEnd) issue(next + kKeyTile);
          }
        }
      } else {
        issue(keyBegin);
        for (int keyStart = keyBegin; keyStart < stepsEnd;
             keyStart += kKeyTile) {
          lds_barrier();
          stage(0, keyStart);
          lds_barrier();
          if (keyStart + kKeyTile < stepsEnd) issue(keyStart + kKeyTile);
          compute(0, keyStart);
        }
      }
    }

    if constexpr (kKeyWaves > 1) {
      // Merge the key waves into key wave 0.
      __shared__ float keyMax[kKeyWaves][kBlockRows],
          keySum[kKeyWaves][kBlockRows];
      __shared__ __attribute__((aligned(16))) f8 keySlot[kWaves][kWaveSize];
      const int localRow = rowWave * 16 + laneRow;
      __syncthreads();
      if (laneHalf == 0 && dimPart == 0) {
        keyMax[keyWave][localRow] = runMax;
        keySum[keyWave][localRow] = runSum;
      }
      __syncthreads();
      float maxAll = -INFINITY;
#pragma unroll
      for (int k = 0; k < kKeyWaves; ++k)
        maxAll = fmaxf(maxAll, keyMax[k][localRow]);
      float total = 0.f;
#pragma unroll
      for (int k = 0; k < kKeyWaves; ++k)
        total += weight_of(keyMax[k][localRow], maxAll) * keySum[k][localRow];
      const float weight = weight_of(runMax, maxAll);
#pragma unroll
      for (int j = 0; j < kLaneDims; ++j) accum[j] *= weight;
      runMax = maxAll;
      runSum = total;
#pragma unroll
      for (int j = 0; j < kLaneDims; ++j) {
        if (keyWave > 0) keySlot[wave][lane] = accum[j];
        __syncthreads();
        if (keyWave == 0)
#pragma unroll
          for (int k = 1; k < kKeyWaves; ++k)
            accum[j] += keySlot[wave + k * kDimParts][lane];
        __syncthreads();
      }
    }
    const bool owner = kKeyWaves == 1 || keyWave == 0;

    if (numSegments == 1) {
      if (!owner || waveRow + laneRow >= numRows) return;
      const float inv = __builtin_amdgcn_rcpf(runSum);
      elem_t* dst = out + ((size_t)rowToken * kNumQHeads + rowHead) * kHeadDim +
                    dimPart * kPartDims;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        elem_t x[kLaneDims];
#pragma unroll
        for (int j = 0; j < kLaneDims; ++j) x[j] = to_elem(accum[j][e] * inv);
        unsigned* d = (unsigned*)(dst + kLaneDims * (2 * e + laneHalf));
#pragma unroll
        for (int w = 0; w < kLaneWords; ++w)
          d[w] = __builtin_bit_cast(unsigned, e2{x[2 * w], x[2 * w + 1]});
      }
      return;
    }

    // Split KV: publish (O, max, sum); the last segment to arrive merges and
    // resets the counter.
    const size_t partialBase =
        ((size_t)block * numSegments + segment) * kBlockRows;
    if (owner) {
      const size_t partialRow = partialBase + rowWave * 16 + laneRow;
      float* dst = partialO + partialRow * kHeadDim + dimPart * kPartDims;
#pragma unroll
      for (int e = 0; e < 8; ++e)
#pragma unroll
        for (int j = 0; j < kLaneDims; ++j)
          dst[kLaneDims * (2 * e + laneHalf) + j] = accum[j][e];
      if (laneHalf == 0 && dimPart == 0) {
        partialML[partialRow * 2] = runMax;
        partialML[partialRow * 2 + 1] = runSum;
      }
    }
    __threadfence();
    __shared__ int isLast;
    __syncthreads();
    if (thread == 0) {
      isLast = atomicAdd(&counters[block], 1) == numSegments - 1;
      if (isLast) counters[block] = 0;
    }
    __syncthreads();
    if (!isLast) return;
    __threadfence();
    // Merge kMergeGroup segments at a time.
    constexpr int kMergeGroup = kMaxSegments < 8 ? kMaxSegments : 8;
    constexpr int kMergeIters =
        (kBlockRows * kHeadDim + kThreads * 4 - 1) / (kThreads * 4);
#pragma unroll 1
    for (int it = 0; it < kMergeIters; ++it) {
      const int elem = (it * kThreads + thread) * 4;
      const int r = elem / kHeadDim, d = elem % kHeadDim;
      const int outRow = tileRow + r;
      if (elem >= kBlockRows * kHeadDim || outRow >= numRows) continue;
      const size_t rowBase = (size_t)block * numSegments * kBlockRows + r;
      float maxAll = -INFINITY, denominator = 0.f;
      f4 numerator = {};
#pragma unroll 1
      for (int first = 0; first < numSegments; first += kMergeGroup) {
        float m[kMergeGroup], l[kMergeGroup];
        f4 o[kMergeGroup];
#pragma unroll
        for (int g = 0; g < kMergeGroup; ++g) {
          const size_t pr = rowBase + (size_t)(first + g) * kBlockRows;
          const bool in = first + g < numSegments;
          m[g] = in ? partialML[pr * 2] : -INFINITY;
          l[g] = in ? partialML[pr * 2 + 1] : 0.f;
          o[g] = in ? *(const f4*)(partialO + pr * kHeadDim + d) : f4{};
        }
        float groupMax = maxAll;
#pragma unroll
        for (int g = 0; g < kMergeGroup; ++g) groupMax = fmaxf(groupMax, m[g]);
        const float carry = weight_of(maxAll, groupMax);
        denominator *= carry;
        numerator *= carry;
#pragma unroll
        for (int g = 0; g < kMergeGroup; ++g) {
          const float weight = weight_of(m[g], groupMax);
          denominator = fmaf(weight, l[g], denominator);
          numerator += weight * o[g];
        }
        maxAll = groupMax;
      }
      const float inv = __builtin_amdgcn_rcpf(denominator);
      elem_t* dst =
          out +
          ((size_t)(outRow / kG) * kNumQHeads + kvHead * kG + outRow % kG) *
              kHeadDim +
          d;
#pragma unroll
      for (int k = 0; k < 4; ++k) dst[k] = to_elem(numerator[k] * inv);
    }
  }
};

// Arguments are checked by rdna_singleseq_registry.cu. Splits the KV
// until the GPU is full, keeping at least kMinSegmentSteps steps per segment.
template <PrefillConfig C>
void launch_prefill(const PrefillArgs& a) {
  using P = Prefill<C>;
  using elem_t = typename P::elem_t;
  static const int numWgps = [] {
    int device, n = 0;
    (void)hipGetDevice(&device);
    (void)hipDeviceGetAttribute(&n, hipDeviceAttributeMultiprocessorCount,
                                device);
    return std::max(1, n);
  }();
  constexpr int kWgpWorkgroups = std::max(
      1, std::min(131072 / P::kLdsBytes, P::kWavesPerSimd * 4 / P::kWaves));
  const int targetWorkgroups = 4 * numWgps;
  const int numTiles =
      (a.num_tokens * P::kG + P::kBlockRows - 1) / P::kBlockRows;
  const int groups = numTiles * P::kNumKvHeads;
  int segments = 1;
  if (P::kMaxSegments > 1 && groups < targetWorkgroups) {
    // Keys a tile sees.
    const int keys =
        P::kWindow ? std::min(a.max_seq_len,
                              P::kWindow + P::kBlockRows / P::kG + P::kKeyTile)
                   : a.max_seq_len;
    const int steps = (keys + P::kKeyTile - 1) / P::kKeyTile;
    segments =
        std::min({(targetWorkgroups + groups - 1) / groups, P::kMaxSegments,
                  std::max(1, steps / P::kMinSegmentSteps)});
  }
  hipLaunchKernelGGL(
      P::prefill_attn, dim3(groups * segments), dim3(P::kThreads), 0, a.stream,
      static_cast<const elem_t*>(a.q), static_cast<const elem_t*>(a.kv), a.bt,
      static_cast<elem_t*>(a.out), a.seq_lens, a.num_tokens, a.scale,
      a.partial_o, a.partial_ml, a.counters, segments, kWgpWorkgroups * numWgps,
      a.mm_ranges, a.num_mm_ranges);
}

}  // namespace

}  // namespace rdna_singleseq
