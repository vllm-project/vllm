// CTA-parallel descending bitonic + absorb a tile into a top-K buffer.
// Math warps only; sync with NamedBarrier (TMA warps are not in the CTA).
//
// Bitonic: NamedBarrier only when the partner is in another warp (stride>=32).
// Intra-warp strides use __syncwarp(). That cuts CTA barriers from 36 to 6
// for a 256-length sort (the common flush size under SPLIT_KV=256).
#pragma once

#include <cstdint>

#include <cutlass/arch/barrier.h>

namespace dsa_opt {

constexpr float kNegInf = -1.0e30f;

__device__ __forceinline__ void SwapPair(
    float* scores, int32_t* indices, int a, int b) {
    const float ts = scores[a];
    const int32_t ti = indices[a];
    scores[a] = scores[b];
    indices[a] = indices[b];
    scores[b] = ts;
    indices[b] = ti;
}

template <int kNumThreads>
__device__ __forceinline__ void SelectSync(int stride, int barrier_id) {
    if (stride >= 32) {
        cutlass::arch::NamedBarrier::sync(kNumThreads, barrier_id);
    } else {
        __syncwarp();
    }
}

// In-place bitonic sort, largest score first. n must be a power of two.
template <int kNumThreads>
__device__ void BitonicSortDesc(
    float* scores, int32_t* indices, int n, int barrier_id) {
    const int tid = static_cast<int>(threadIdx.x);
    for (int size = 2; size <= n; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = tid; i < n; i += kNumThreads) {
                const int j = i ^ stride;
                if (j > i) {
                    const bool first_half = (i & size) == 0;
                    const bool out_of_order = first_half ? (scores[i] < scores[j])
                                                         : (scores[i] > scores[j]);
                    if (out_of_order) {
                        SwapPair(scores, indices, i, j);
                    }
                }
            }
            SelectSync<kNumThreads>(stride, barrier_id);
        }
    }
}

__device__ void MergeDescTwoPtr(
    const float* a_s, const int32_t* a_i, int na,
    const float* b_s, const int32_t* b_i, int nb,
    float* o_s, int32_t* o_i, int no) {
    int ia = 0, ib = 0;
    for (int k = 0; k < no; ++k) {
        const bool take_a =
            ib >= nb || (ia < na && a_s[ia] >= b_s[ib]);
        if (take_a) {
            o_s[k] = (ia < na) ? a_s[ia] : kNegInf;
            o_i[k] = (ia < na) ? a_i[ia] : -1;
            ++ia;
        } else {
            o_s[k] = b_s[ib];
            o_i[k] = b_i[ib];
            ++ib;
        }
    }
}

// Overflow path only. Common case (n + tile <= K) writes straight into sel
// from the WGMMA epilogue and never calls this.
template <int kNumThreads, int kTopK, int kTileTokens, int kSortN>
__device__ void AbsorbTileOverflow(
    float* sel_score,
    int32_t* sel_idx,
    int& heap_n,
    int& sorted_flag,
    const float* tile_score,
    const int32_t* tile_idx,
    float* merge_s,
    int32_t* merge_i,
    int barrier_id) {
    static_assert(kSortN >= kTopK + kTileTokens, "sort buffer too small");
    const int tid = static_cast<int>(threadIdx.x);
    const int n = heap_n;
    float* pack_s = sel_score + (kSortN - kTileTokens);
    int32_t* pack_i = sel_idx + (kSortN - kTileTokens);

    for (int i = tid; i < kTileTokens; i += kNumThreads) {
        pack_s[i] = tile_score[i];
        pack_i[i] = tile_idx[i];
    }
    cutlass::arch::NamedBarrier::sync(kNumThreads, barrier_id);

    if (sorted_flag) {
        BitonicSortDesc<kNumThreads>(pack_s, pack_i, kTileTokens, barrier_id);
        if (tid == 0) {
            MergeDescTwoPtr(
                sel_score, sel_idx, kTopK, pack_s, pack_i, kTileTokens,
                merge_s, merge_i, kTopK);
        }
        cutlass::arch::NamedBarrier::sync(kNumThreads, barrier_id);
        for (int i = tid; i < kTopK; i += kNumThreads) {
            sel_score[i] = merge_s[i];
            sel_idx[i] = merge_i[i];
        }
        cutlass::arch::NamedBarrier::sync(kNumThreads, barrier_id);
        return;
    }

    for (int i = tid; i < kTileTokens; i += kNumThreads) {
        sel_score[n + i] = pack_s[i];
        sel_idx[n + i] = pack_i[i];
    }
    const int total = n + kTileTokens;
    for (int i = tid; i < kSortN; i += kNumThreads) {
        if (i >= total) {
            sel_score[i] = kNegInf;
            sel_idx[i] = -1;
        }
    }
    cutlass::arch::NamedBarrier::sync(kNumThreads, barrier_id);
    BitonicSortDesc<kNumThreads>(sel_score, sel_idx, kSortN, barrier_id);
    if (tid == 0) {
        heap_n = kTopK;
        sorted_flag = 1;
    }
    cutlass::arch::NamedBarrier::sync(kNumThreads, barrier_id);
}

// Sort the first n entries descending. Pads to next pow2, never past kTopK.
template <int kNumThreads, int kTopK>
__device__ void SortTopKDesc(
    float* scores, int32_t* indices, int n, int barrier_id) {
    const int tid = static_cast<int>(threadIdx.x);
    int p = 1;
    while (p < n) {
        p <<= 1;
    }
    if (p > kTopK) {
        p = kTopK;
    }
    for (int i = tid; i < p; i += kNumThreads) {
        if (i >= n) {
            scores[i] = kNegInf;
            indices[i] = -1;
        }
    }
    cutlass::arch::NamedBarrier::sync(kNumThreads, barrier_id);
    BitonicSortDesc<kNumThreads>(scores, indices, p, barrier_id);
}

}  // namespace dsa_opt
