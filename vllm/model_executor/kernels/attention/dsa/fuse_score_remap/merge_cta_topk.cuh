// Per-row top-K over compact (score, idx) packs from the score kernel.
#pragma once

#include <cstdint>

#include <cub/block/block_radix_sort.cuh>

#include "cta_bitonic_select.cuh"

namespace dsa_opt {

constexpr int kPackWidth = 256;
constexpr int kTopkThreads = 256;
// 64k path: each CTA sorts 8192 candidates (256 threads × IPT 32), same as
// the 8k kernel. Chunks run in parallel; a second kernel merges their top-K.
constexpr int kChunkIpt = 32;
constexpr int kChunkItems = kTopkThreads * kChunkIpt;  // 8192
constexpr int kChunkParts = kChunkItems / kPackWidth;  // 32

template <int kTopK, int kIpt>
__global__ void merge_cta_topk_kernel(
    const float* __restrict__ pack_scores,
    const int32_t* __restrict__ pack_indices,
    const int32_t* __restrict__ seq_lens,
    int32_t* __restrict__ out_indices,
    int num_rows,
    int max_parts) {
    cudaGridDependencySynchronize();
    const int row = static_cast<int>(blockIdx.x);
    if (row >= num_rows) {
        return;
    }
    const int tid = static_cast<int>(threadIdx.x);
    const int nthreads = static_cast<int>(blockDim.x);
    const int seq = seq_lens[row];
    int32_t* out = out_indices + static_cast<long long>(row) * kTopK;
    const long long row_base =
        static_cast<long long>(row) * max_parts * kPackWidth;

    if (seq <= 0) {
        for (int i = tid; i < kTopK; i += nthreads) {
            out[i] = -1;
        }
        return;
    }

    if (seq <= kTopK) {
        if (tid == 0) {
            int written = 0;
            const int nparts = (seq + kPackWidth - 1) / kPackWidth;
            for (int p = 0; p < nparts && written < kTopK; ++p) {
                const long long base = row_base + static_cast<long long>(p) * kPackWidth;
                for (int t = 0; t < kPackWidth && written < kTopK; ++t) {
                    const int32_t id = pack_indices[base + t];
                    if (id >= 0) {
                        out[written++] = id;
                    }
                }
            }
            for (int i = written; i < kTopK; ++i) {
                out[i] = -1;
            }
        }
        return;
    }

    using BlockSort = cub::BlockRadixSort<float, kTopkThreads, kIpt, int32_t>;
    __shared__ typename BlockSort::TempStorage sort_tmp;

    float keys[kIpt];
    int32_t vals[kIpt];
    const int nparts = (seq + kPackWidth - 1) / kPackWidth;
    const int ncand = nparts * kPackWidth;
    #pragma unroll
    for (int i = 0; i < kIpt; ++i) {
        const int idx = tid * kIpt + i;
        if (idx < ncand) {
            keys[i] = pack_scores[row_base + idx];
            vals[i] = pack_indices[row_base + idx];
        } else {
            keys[i] = kNegInf;
            vals[i] = -1;
        }
    }
    BlockSort(sort_tmp).SortDescending(keys, vals);
    #pragma unroll
    for (int i = 0; i < kIpt; ++i) {
        const int idx = tid * kIpt + i;
        if (idx < kTopK) {
            out[idx] = vals[i];
        }
    }
}

// 64k phase 1: one CTA per (row, 8192-chunk). Writes that chunk's top-K.
template <int kTopK>
__global__ void merge_cta_topk_chunk_partial_kernel(
    const float* __restrict__ pack_scores,
    const int32_t* __restrict__ pack_indices,
    const int32_t* __restrict__ seq_lens,
    float* __restrict__ part_scores,
    int32_t* __restrict__ part_indices,
    int num_rows,
    int max_parts,
    int nchunks) {
    cudaGridDependencySynchronize();
    const int chunk = static_cast<int>(blockIdx.x);
    const int row = static_cast<int>(blockIdx.y);
    if (row >= num_rows || chunk >= nchunks) {
        return;
    }
    const int tid = static_cast<int>(threadIdx.x);
    const int nthreads = static_cast<int>(blockDim.x);
    const int seq = seq_lens[row];
    const int ncand =
        seq > 0 ? ((seq + kPackWidth - 1) / kPackWidth) * kPackWidth : 0;
    const int off = chunk * kChunkItems;
    const long long part_base =
        (static_cast<long long>(row) * nchunks + chunk) * kTopK;
    float* ps = part_scores + part_base;
    int32_t* pi = part_indices + part_base;

    if (off >= ncand) {
        for (int i = tid; i < kTopK; i += nthreads) {
            ps[i] = kNegInf;
            pi[i] = -1;
        }
        return;
    }

    using BlockSort = cub::BlockRadixSort<float, kTopkThreads, kChunkIpt, int32_t>;
    __shared__ typename BlockSort::TempStorage sort_tmp;

    float keys[kChunkIpt];
    int32_t vals[kChunkIpt];
    const long long row_base =
        static_cast<long long>(row) * max_parts * kPackWidth;
    #pragma unroll
    for (int i = 0; i < kChunkIpt; ++i) {
        const int idx = off + tid * kChunkIpt + i;
        if (idx < ncand) {
            keys[i] = pack_scores[row_base + idx];
            vals[i] = pack_indices[row_base + idx];
        } else {
            keys[i] = kNegInf;
            vals[i] = -1;
        }
    }
    BlockSort(sort_tmp).SortDescending(keys, vals);
    #pragma unroll
    for (int i = 0; i < kChunkIpt; ++i) {
        const int idx = tid * kChunkIpt + i;
        if (idx < kTopK) {
            ps[idx] = keys[i];
            pi[idx] = vals[i];
        }
    }
}

// r-th largest (0-based) of two descending sequences. All threads may call
// this; each owns a disjoint r. Used to replace thread-0 MergeDescTwoPtr
// (that path was ~2.6ms for 7 serial K=2048 merges).
__device__ __forceinline__ void PickKthDesc(
    const float* a_s, const int32_t* a_i, int na,
    const float* b_s, const int32_t* b_i, int nb,
    int r, float& os, int32_t& oi) {
    int lo = r + 1 > nb ? r + 1 - nb : 0;
    int hi = r + 1 < na ? r + 1 : na;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        const int jb = r + 1 - mid;
        const bool a_ok =
            (mid == 0) || (jb >= nb) || (a_s[mid - 1] >= b_s[jb]);
        const bool b_ok =
            (jb == 0) || (mid >= na) || (b_s[jb - 1] >= a_s[mid]);
        if (!a_ok) {
            hi = mid;
        } else if (!b_ok) {
            lo = mid + 1;
        } else {
            lo = mid;
            break;
        }
    }
    const int ia = lo;
    const int ib = r + 1 - ia;
    const bool from_a =
        ia > 0 && (ib <= 0 || a_s[ia - 1] <= (ib > 0 ? b_s[ib - 1] : kNegInf));
    if (from_a) {
        os = a_s[ia - 1];
        oi = a_i[ia - 1];
    } else if (ib > 0) {
        os = b_s[ib - 1];
        oi = b_i[ib - 1];
    } else {
        os = kNegInf;
        oi = -1;
    }
}

template <int kTopK>
__device__ void MergeDescParallel(
    const float* a_s, const int32_t* a_i,
    const float* b_s, const int32_t* b_i,
    float* o_s, int32_t* o_i) {
    const int tid = static_cast<int>(threadIdx.x);
    const int nthreads = static_cast<int>(blockDim.x);
    for (int r = tid; r < kTopK; r += nthreads) {
        PickKthDesc(a_s, a_i, kTopK, b_s, b_i, kTopK, r, o_s[r], o_i[r]);
    }
}

// 64k phase 2: merge nchunks sorted top-K lists into one top-K per row.
template <int kTopK>
__global__ void merge_cta_topk_reduce_chunks_kernel(
    const float* __restrict__ part_scores,
    const int32_t* __restrict__ part_indices,
    const int32_t* __restrict__ seq_lens,
    int32_t* __restrict__ out_indices,
    int num_rows,
    int nchunks) {
    cudaGridDependencySynchronize();
    const int row = static_cast<int>(blockIdx.x);
    if (row >= num_rows) {
        return;
    }
    const int tid = static_cast<int>(threadIdx.x);
    const int nthreads = static_cast<int>(blockDim.x);
    int32_t* out = out_indices + static_cast<long long>(row) * kTopK;
    const int seq = seq_lens[row];
    const int ncand =
        seq > 0 ? ((seq + kPackWidth - 1) / kPackWidth) * kPackWidth : 0;
    int nvalid = (ncand + kChunkItems - 1) / kChunkItems;
    if (nvalid > nchunks) {
        nvalid = nchunks;
    }
    if (nvalid <= 0) {
        for (int i = tid; i < kTopK; i += nthreads) {
            out[i] = -1;
        }
        return;
    }

    __shared__ float run_s[kTopK];
    __shared__ int32_t run_i[kTopK];
    __shared__ float mer_s[kTopK];
    __shared__ int32_t mer_i[kTopK];
    __shared__ float chk_s[kTopK];
    __shared__ int32_t chk_i[kTopK];

    const long long c0 = static_cast<long long>(row) * nchunks * kTopK;
    for (int i = tid; i < kTopK; i += nthreads) {
        run_s[i] = part_scores[c0 + i];
        run_i[i] = part_indices[c0 + i];
    }
    __syncthreads();
    for (int c = 1; c < nvalid; ++c) {
        const long long cb = (static_cast<long long>(row) * nchunks + c) * kTopK;
        for (int i = tid; i < kTopK; i += nthreads) {
            chk_s[i] = part_scores[cb + i];
            chk_i[i] = part_indices[cb + i];
        }
        __syncthreads();
        MergeDescParallel<kTopK>(run_s, run_i, chk_s, chk_i, mer_s, mer_i);
        __syncthreads();
        for (int i = tid; i < kTopK; i += nthreads) {
            run_s[i] = mer_s[i];
            run_i[i] = mer_i[i];
        }
        __syncthreads();
    }
    for (int i = tid; i < kTopK; i += nthreads) {
        out[i] = run_i[i];
    }
}

__global__ void fill_seg_offsets_kernel(
    const int32_t* __restrict__ seq_lens,
    int* __restrict__ begin,
    int* __restrict__ end,
    int num_rows,
    int stride) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= num_rows) {
        return;
    }
    begin[i] = i * stride;
    const int n = seq_lens[i];
    int ncand = n > 0 ? ((n + kPackWidth - 1) / kPackWidth) * kPackWidth : 0;
    if (ncand > stride) {
        ncand = stride;
    }
    end[i] = begin[i] + ncand;
}

__global__ void gather_physical_kernel(
    const int32_t* __restrict__ pack_indices,
    const int32_t* __restrict__ logical,
    int32_t* __restrict__ out_indices,
    int pack_cols,
    int topk) {
    const int row = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int nthreads = static_cast<int>(blockDim.x);
    for (int i = tid; i < topk; i += nthreads) {
        const int32_t log = logical[static_cast<long long>(row) * topk + i];
        out_indices[static_cast<long long>(row) * topk + i] =
            (log >= 0) ? pack_indices[static_cast<long long>(row) * pack_cols + log]
                       : -1;
    }
}

template <int kTopK>
__global__ void take_seg_topk_kernel(
    const int32_t* __restrict__ sorted_idx,
    const int* __restrict__ begin,
    const int* __restrict__ end,
    int32_t* __restrict__ out_indices,
    int num_rows) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= num_rows) {
        return;
    }
    const int tid = static_cast<int>(threadIdx.x);
    const int nthreads = static_cast<int>(blockDim.x);
    const int nseg = end[row] - begin[row];
    const int take = nseg < kTopK ? nseg : kTopK;
    int32_t* out = out_indices + static_cast<long long>(row) * kTopK;
    const int32_t* src = sorted_idx + begin[row];
    for (int i = tid; i < kTopK; i += nthreads) {
        out[i] = (i < take) ? src[i] : -1;
    }
}

}  // namespace dsa_opt
