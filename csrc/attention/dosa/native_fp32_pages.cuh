#pragma once
#include "paged_record5.cuh"

namespace DOSA_NAMESPACE::native_fp32_pages {
// Add count ownership to the production page mapper; do not change its ABI or
// reset its table. Production seed pages have implicit deterministic addresses.
struct Arena {
  paged_record5::Arena storage;
  int* counts;
  int rows;
};
// Same whole-task reservation as the qualified standalone publisher. One CTA
// leader allocates the range; ensure_page publishes every mapping before
// return.
__device__ __forceinline__ int reserve(Arena a, int row, int n) {
  if (a.storage.error[row]) return -1;
  const int begin = atomicAdd(a.counts + row, n);
  if (begin < paged_record5::SeedCells || begin > paged_record5::MaxCells ||
      n > paged_record5::MaxCells - begin) {
    atomicOr(a.storage.error + row, 32);
    return -1;
  }
  if (!n) return begin;
  for (int page = begin / paged_record5::PageCells;
       page <= (begin + n - 1) / paged_record5::PageCells; ++page)
    if (paged_record5::ensure_page(a.storage, row, page) < 0) return -1;
  return begin;
}
__device__ __forceinline__ void write_reserved(Arena a, int row, int ordinal,
                                               paged_record5::Cell cell) {
  const int logical = ordinal / paged_record5::PageCells;
  const int prefix = paged_record5::prefix_page(a.rows, row, logical);
  const int page =
      prefix >= 0
          ? prefix
          : paged_record5::acquire(a.storage.table +
                                   row * paged_record5::RowPages + logical);
  const int at =
      page * paged_record5::PageCells + ordinal % paged_record5::PageCells;
  a.storage.low[at - a.rows * paged_record5::SeedCells] = cell.low;
  a.storage.words[at] = cell.word;
}
}  // namespace DOSA_NAMESPACE::native_fp32_pages
