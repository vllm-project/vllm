#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

namespace paged_record5 {
constexpr int PageCells = 512, SeedCells = 2048, SeedPages = 4;
constexpr int InlinePages = 4, PrefixPages = SeedPages + InlinePages;
constexpr int MaxCells = 114688, RowPages = MaxCells / PageCells;
constexpr int Empty = -1, Building = -2, Failed = -3;
struct Arena {
  uint8_t* low;  // [nonseed pages, then seed pages], 512 cells each
  int* words;    // [physical_pages, 512]
  int* table;    // [rows, 224]
  int *error, *next_page;
  int physical_pages, rows;
};
struct Cell {
  uint8_t low;
  int word;
};

static __device__ __forceinline__ int acquire(const int* slot) {
  int value;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
               : "=r"(value)
               : "l"(slot)
               : "memory");
  return value;
}
static __device__ __forceinline__ void release(int* slot, int value) {
  asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(slot), "r"(value)
               : "memory");
}

// Seed pages remain one contiguous global prefix. The inline nonseed pages
// follow it, with one four-page range per physical row.
static __host__ __device__ __forceinline__ int prefix_page(int rows, int row,
                                                           int logical) {
  if (logical < SeedPages) return row * SeedPages + logical;
  return logical < PrefixPages
             ? rows * SeedPages + row * InlinePages + logical - SeedPages
             : -1;
}

// Called ONLY by the CTA leader. No other lane allocates or polls a page.
// A winner publishes before returning, including a permanent failure marker;
// no allocation lock is held while waiting for a different owner.
static __device__ int ensure_page(Arena a, int row, int logical_page) {
  const int prefix = prefix_page(a.rows, row, logical_page);
  if (prefix >= 0) return prefix;
  int* slot = a.table + row * RowPages + logical_page;
  int page = acquire(slot);
  if (page == Empty) page = atomicCAS(slot, Empty, Building);
  if (page == Empty) {
    page = atomicAdd(a.next_page, 1);
    if (page >= a.physical_pages) page = Failed;
    release(slot, page);
  } else {
    while (page == Building) page = acquire(slot);
  }
  if (page < 0) atomicOr(a.error + row, 16);
  return page;
}

// Existing Q4/solo emitters already reserve logical slots. This helper only
// maps a nonempty <=32-cell span; it NEVER increments the row count.
struct WarpSpan {
  int logical_page, first, last;
};
static __device__ WarpSpan warp_ensure_span(Arena a, int row, int begin, int n,
                                            unsigned mask) {
  const int lane = threadIdx.x & 31, leader = __ffs(mask) - 1;
  int first = -1, last = -1;
  if (lane == leader) {
    if (a.error[row]) {
      // No new physical allocation is useful after a sticky row failure.
    } else if (begin < SeedCells || n <= 0 || n > 32 || n != __popc(mask) ||
               begin > MaxCells - n) {
      atomicOr(a.error + row, 32);
    } else {
      first = ensure_page(a, row, begin / PageCells);
      if (first >= 0)
        last = (begin / PageCells == (begin + n - 1) / PageCells)
                   ? first
                   : ensure_page(a, row, (begin + n - 1) / PageCells);
    }
  }
  first = __shfl_sync(mask, first, leader);
  last = __shfl_sync(mask, last, leader);
  return {begin / PageCells, first, last};
}
template <bool Rank12 = false>
static __device__ bool write_span(Arena a, WarpSpan span, int ordinal,
                                  Cell cell) {
  if (span.first < 0 || span.last < 0) return false;
  const int page =
      ordinal / PageCells == span.logical_page ? span.first : span.last;
  const int at = page * PageCells + ordinal % PageCells;
  if constexpr (!Rank12) a.low[at - a.rows * SeedCells] = cell.low;
  a.words[at] = cell.word;
  return true;
}
}  // namespace paged_record5
