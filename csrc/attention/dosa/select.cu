#include "common.cuh"
#include "record.cuh"

namespace DOSA_NAMESPACE {
namespace record4_integer {
constexpr int Top = 2048, Threads = 256, Bins = 4096;
constexpr unsigned Full = 0xFFFFFFFFu, IdMask = 0xFFFFFu;
struct Args {
  const int *words, *counts, *starts, *ends, *pool;
  int *error, *output, *diag, *boundary_ids, *boundary_counts;
  int rows, cap, nk, boundary_stride;
  const uint8_t* low;
};

__device__ void fail(const Args& a, int row) {
  if (threadIdx.x == 0) {
    atomicOr(a.error + row, 1);
    a.boundary_counts[row] = 0;
    for (int j = 0; j < 2; ++j) a.diag[int64_t(row) * 2 + j] = -1;
  }
  for (int j = threadIdx.x; j < Top; j += Threads)
    a.output[int64_t(row) * Top + j] = -1;
}

__global__ __launch_bounds__(Threads) void select(Args a) {
  const int row = a.rows - 1 - int(blockIdx.x), t = threadIdx.x, warp = t / 32,
            lane = t % 32;
  const int count = a.counts[row];
  __shared__ unsigned hist[Bins], warp_count[8], bad;
  __shared__ int winning_warp, group_before, pivot, certain, uncertain,
      written_b, written_e;
  if (t == 0) {
    bad = 0;
    if (count < Top || count > a.cap) bad |= 1;
    if (a.starts[row] < 0 || a.ends[row] < a.starts[row] || a.ends[row] > a.nk)
      bad |= 4;
    // Native BF16 scores are represented exactly by these ordered cells.
    if (a.error[row] != 0 || *a.pool != 0) bad |= 16;
    winning_warp = -1;
    group_before = 0;
    pivot = -1;
    certain = uncertain = 0;
    written_b = written_e = 0;
  }
  for (int bin = t; bin < Bins; bin += Threads) hist[bin] = 0;
  __syncthreads();
  if (bad) {
    fail(a, row);
    return;
  }

  const int64_t base = int64_t(row) * a.cap;
  // These disjoint input tensors are immutable for the operation. Cache the
  // row bounds once instead of reloading both for every validated record.
  const unsigned row_start = unsigned(a.starts[row]);
  const unsigned row_end = unsigned(a.ends[row]);
  {
    DOSA_RECORD5_INDEPENDENT_WORDS(
        Threads, t, a.words, base, count, word, slot,
        const unsigned cp = word >> 20, id = word & IdMask;
        if (cp < 0x8u || cp > 0xFF7u) {
          atomicOr(&bad, 2u);
          continue;
        } if (id >= unsigned(a.nk) || id < row_start || id >= row_end) {
          atomicOr(&bad, 4u);
          continue;
        } atomicAdd(hist + cp, 1u););
  }
  __syncthreads();
  if (bad) {
    fail(a, row);
    return;
  }

  // Each warp reads one contiguous512-bin group, coalesced at each step.
  unsigned local = 0;
#pragma unroll
  for (int part = 0; part < 16; ++part)
    local += hist[warp * 512 + part * 32 + lane];
  const unsigned group_sum = __reduce_add_sync(Full, local);
  if (lane == 0) warp_count[warp] = group_sum;
  __syncthreads();
  if (t == 0) {
    unsigned before = 0;
    for (int w = 0; w < 8; ++w) {
      if (before < Top && Top <= before + warp_count[w]) {
        winning_warp = w;
        group_before = int(before);
        break;
      }
      before += warp_count[w];
    }
    if (winning_warp < 0) bad |= 16;
  }
  __syncthreads();
  if (bad) {
    fail(a, row);
    return;
  }

  // Only the selected full warp searches its16 contiguous chunks. All branches
  // are warp-uniform; no CTA barrier occurs inside this selected-warp region.
  if (warp == winning_warp) {
    unsigned rank = Top - group_before, seen = 0;
    for (int part = 0; part < 16; ++part) {
      const unsigned value = hist[warp * 512 + part * 32 + lane];
      const unsigned chunk = __reduce_add_sync(Full, value);
      if (rank > chunk) {
        rank -= chunk;
        seen += chunk;
        continue;
      }
      unsigned inclusive = value;
#pragma unroll
      for (int offset = 1; offset < 32; offset <<= 1) {
        const unsigned other = __shfl_up_sync(Full, inclusive, offset);
        if (lane >= offset) inclusive += other;
      }
      const unsigned mask = __ballot_sync(Full, inclusive >= rank);
      const int winner = __ffs(mask) - 1;
      const unsigned selected_count = __shfl_sync(Full, value, winner);
      const unsigned selected_before =
          __shfl_sync(Full, inclusive - value, winner);
      if (lane == 0) {
        pivot = warp * 512 + part * 32 + winner;
        certain = group_before + int(seen + selected_before);
        uncertain = int(selected_count);
      }
      break;
    }
  }
  __syncthreads();
  {
    __shared__ unsigned fine_hist[256];
    // The coarse histogram is dead. Reuse its4096 words for the selected cell
    // when it fits; otherwise retain the original full third-pass algorithm.
    if (t == 0) {
      group_before = uncertain;
      winning_warp = uncertain <= Bins;
    }
    fine_hist[t] = 0;
    __syncthreads();
    const unsigned coarse = unsigned(pivot);
    if (winning_warp) {
      auto collect = [&](unsigned word, int slot, bool live) {
        const unsigned hi = word >> 20, id = word & IdMask;
        const bool is_b = live && hi < coarse, is_e = live && hi == coarse;
        const unsigned bm = __ballot_sync(Full, is_b),
                       em = __ballot_sync(Full, is_e);
        if (!(bm | em)) return;
        DOSA_RECORD5_RESERVE(Full, lane, bm, em, &written_b, &written_e, bbase,
                             ebase, lower);
        if (is_b) {
          const int at = bbase + __popc(bm & lower);
          if (at < Top)
            a.output[int64_t(row) * Top + at] = int(id);
          else
            atomicOr(&bad, 16u);
        }
        if (is_e) {
          const unsigned low = unsigned(a.low[base + slot]);
          const int at = ebase + __popc(em & lower);
          if (at < Bins)
            hist[at] = (low << 20) | id;
          else
            atomicOr(&bad, 16u);
          atomicAdd(fine_hist + low, 1u);
        }
      };
      DOSA_RECORD5_COLLECTIVE_WORDS(Threads, warp, lane, a.words, base, count,
                                    collect);
    } else {
      for (int slot = t; slot < count; slot += Threads) {
        const unsigned word = unsigned(a.words[base + slot]);
        if ((word >> 20) == coarse)
          atomicAdd(fine_hist + unsigned(a.low[base + slot]), 1u);
      }
    }
    __syncthreads();
    if (t == 0 && winning_warp &&
        (written_b != certain || written_e != group_before))
      bad |= 16;
    __syncthreads();
    if (bad) {
      fail(a, row);
      return;
    }
    if (warp == 0) {
      unsigned rank = Top - certain, seen = 0;
      for (int part = 0; part < 8; ++part) {
        const unsigned value = fine_hist[part * 32 + lane];
        const unsigned chunk = __reduce_add_sync(Full, value);
        if (rank > chunk) {
          rank -= chunk;
          seen += chunk;
          continue;
        }
        unsigned inclusive = value;
#pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
          const unsigned other = __shfl_up_sync(Full, inclusive, offset);
          if (lane >= offset) inclusive += other;
        }
        const unsigned mask = __ballot_sync(Full, inclusive >= rank);
        const int winner = __ffs(mask) - 1;
        const unsigned selected_count = __shfl_sync(Full, value, winner);
        const unsigned selected_before =
            __shfl_sync(Full, inclusive - value, winner);
        if (lane == 0) {
          pivot = int((coarse << 8) | unsigned(part * 32 + winner));
          certain += int(seen + selected_before);
          uncertain = int(selected_count);
        }
        break;
      }
    }
    __syncthreads();
    if (t == 0 && winning_warp) written_e = 0;
  }
  if (t == 0) {
    const bool invalid_pivot = (pivot < 0x800 || pivot > 0xFF7FF);
    if (invalid_pivot || certain < 0 || certain >= Top || uncertain <= 0 ||
        certain + uncertain < Top)
      bad |= 16;
    if (uncertain > a.boundary_stride) bad |= 32;
  }
  __syncthreads();
  if (bad) {
    fail(a, row);
    return;
  }

  if (!winning_warp) {
    for (int begin = warp * 32; begin < count; begin += Threads) {
      const int slot = begin + lane;
      const bool live = slot < count;
      const unsigned word = live ? unsigned(a.words[base + slot]) : 0u;
      const unsigned coarse = word >> 20, id = word & IdMask;
      unsigned cp = coarse;
      {
        cp = coarse << 8;
        if (live && coarse == (unsigned(pivot) >> 8))
          cp |= unsigned(a.low[base + slot]);
      }
      const bool is_b = live && cp < unsigned(pivot),
                 is_e = live && cp == unsigned(pivot);
      const unsigned bm = __ballot_sync(Full, is_b),
                     em = __ballot_sync(Full, is_e);
      int bbase = 0, ebase = 0;
      if (lane == 0) {
        if (bm) bbase = atomicAdd(&written_b, __popc(bm));
        if (em) ebase = atomicAdd(&written_e, __popc(em));
      }
      bbase = __shfl_sync(Full, bbase, 0);
      ebase = __shfl_sync(Full, ebase, 0);
      const unsigned lower = lane ? ((1u << lane) - 1u) : 0u;
      if (is_b) {
        const int at = bbase + __popc(bm & lower);
        if (at < Top)
          a.output[int64_t(row) * Top + at] = int(id);
        else
          atomicOr(&bad, 16u);
      }
      if (is_e) {
        const int at = ebase + __popc(em & lower);
        if (at < a.boundary_stride)
          a.boundary_ids[int64_t(row) * a.boundary_stride + at] = int(id);
        else
          atomicOr(&bad, 16u);
      }
    }
  }
  {
    if (winning_warp) {
      // All cached words are immutable until this loop has drained. Fine E
      // goes to the separate global boundary tensor, never into this cache.
      for (int begin = warp * 32; begin < group_before; begin += Threads) {
        const int at = begin + lane;
        const bool live = at < group_before;
        const unsigned word = live ? hist[at] : 0u;
        const unsigned code = word >> 20, id = word & IdMask;
        const bool is_b = live && code < (unsigned(pivot) & 255u);
        const bool is_e = live && code == (unsigned(pivot) & 255u);
        const unsigned bm = __ballot_sync(Full, is_b),
                       em = __ballot_sync(Full, is_e);
        int bbase = 0, ebase = 0;
        if (lane == 0) {
          if (bm) bbase = atomicAdd(&written_b, __popc(bm));
          if (em) ebase = atomicAdd(&written_e, __popc(em));
        }
        bbase = __shfl_sync(Full, bbase, 0);
        ebase = __shfl_sync(Full, ebase, 0);
        const unsigned lower = lane ? ((1u << lane) - 1u) : 0u;
        if (is_b) {
          const int dst = bbase + __popc(bm & lower);
          if (dst < Top)
            a.output[int64_t(row) * Top + dst] = int(id);
          else
            atomicOr(&bad, 16u);
        }
        if (is_e) {
          const int dst = ebase + __popc(em & lower);
          if (dst < a.boundary_stride)
            a.boundary_ids[int64_t(row) * a.boundary_stride + dst] = int(id);
          else
            atomicOr(&bad, 16u);
        }
      }
    }
  }
  __syncthreads();
  if (t == 0 && (written_b != certain || written_e != uncertain)) bad |= 16;
  __syncthreads();
  if (bad) {
    fail(a, row);
    return;
  }
  if (t == 0) {
    a.boundary_counts[row] = uncertain;
    int* diag = a.diag + int64_t(row) * 2;
    diag[0] = certain;
    diag[1] = uncertain;
  }
}
}  // namespace record4_integer

void select_dense5(Tensor low, Tensor packed, Tensor counts, Tensor starts,
                   Tensor ends, Tensor pool_status, Tensor error, Tensor output,
                   Tensor diag, Tensor boundary_ids, Tensor boundary_counts,
                   int64_t nk) {
  // Every tensor is constructor-owned: U8/I32[Q,114688] frontier planes,
  // I32[Q,2048] output, I32[Q,4096] boundary, I32[Q,2] B/E fields and
  // independent metadata. Selection reads counts
  // and error bits on the current stream; no caller tensor is borrowed here.
  const int rows = packed.size(0), cap = packed.size(1);
  record4_integer::Args args{packed.mutable_data_ptr<int>(),
                             counts.mutable_data_ptr<int>(),
                             starts.mutable_data_ptr<int>(),
                             ends.mutable_data_ptr<int>(),
                             pool_status.mutable_data_ptr<int>(),
                             error.mutable_data_ptr<int>(),
                             output.mutable_data_ptr<int>(),
                             diag.mutable_data_ptr<int>(),
                             boundary_ids.mutable_data_ptr<int>(),
                             boundary_counts.mutable_data_ptr<int>(),
                             rows,
                             cap,
                             int(nk),
                             4096,
                             low.mutable_data_ptr<uint8_t>()};
  DeviceGuard guard(packed.get_device_index());
  record4_integer::select<<<rows, 256, 0, get_current_cuda_stream()>>>(args);
  DOSA_CUDA_CHECK(cudaGetLastError());
}

}  // namespace DOSA_NAMESPACE

STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def(DOSA_OP("select_dense5") "(Tensor low, Tensor packed, Tensor counts, Tensor starts, Tensor ends, "
      "Tensor pool_status, Tensor! error, Tensor! output, Tensor! diag, Tensor! boundary_ids, "
      "Tensor! boundary_counts, int nk) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops) {
  ops.impl(DOSA_OP("select_dense5"), TORCH_BOX(&DOSA_NAMESPACE::select_dense5));
}
