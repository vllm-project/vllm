#include "common.cuh"
#include "geometry.cuh"

namespace DOSA_NAMESPACE {
// ID-only page128 pool. Each successful new page owns exactly one H32 task.
namespace fused_few_live_nk {
struct PagedArgs {
  int *pool, *table, *head, *status, *error, *solo_counts, *common_counts,
      *solo_n, *common_n;
  int2 *solo_tasks, *common_tasks;
  unsigned* words;
  int word_count;
  int nq, nk, max_records, columns, pages, solo_task_cap, common_task_cap;
  int64_t table_stride;
};
__device__ __forceinline__ void page_error(const PagedArgs& a, int row,
                                           bool common, int status) {
  atomicOr(a.status, status);
  if (common) {
#pragma unroll
    for (int j = 0; j < 4; ++j) atomicOr(a.error + row * 4 + j, 1);
  } else
    atomicOr(a.error + row, 1);
}
// Only live grid(1,Q/4): each page ordinal has one CTA/thread owner.
// Existing CTA barriers order partial-page reuse; H32 starts after this kernel.
__device__ __forceinline__ void publish_owned_page(int* slot, int page) {
  asm volatile("st.global.wb.b32 [%0],%1;" ::"l"(slot), "r"(page) : "memory");
}
__device__ __forceinline__ int ensure_id_page(const PagedArgs& a, int row,
                                              int packet, bool common) {
  if (packet < 0 || packet >= a.columns) {
    page_error(a, row, common, 2);
    return -1;
  }
  int table_row = common ? a.nq + row : row;
  int* slot = a.table + int64_t(table_row) * a.table_stride + packet;
  int old;
  asm volatile("ld.global.cg.b32 %0,[%1];" : "=r"(old) : "l"(slot) : "memory");
  if (old == -1) {
    int page = atomicAdd(a.head, 1);
    if (page < 0 || page >= a.pages) {
      page_error(a, row, common, 1);
      publish_owned_page(slot, -3);
      return -1;
    }
    int task = atomicAdd(common ? a.common_n : a.solo_n, 1);
    int cap = common ? a.common_task_cap : a.solo_task_cap;
    if (task < 0 || task >= cap) {
      page_error(a, row, common, 2);
      publish_owned_page(slot, -3);
      return -1;
    }
    (common ? a.common_tasks : a.solo_tasks)[task] =
        make_int2(row, packet * 128);
    publish_owned_page(slot, page);
    return page;
  }
  while (old == -2) {
    __nanosleep(32);
    old = atomicAdd(slot, 0);
  }
  if (old < 0 || old >= a.pages) {
    page_error(a, row, common, old == -3 ? 1 : 2);
    return -1;
  }
  return old;
}
}  // namespace fused_few_live_nk

namespace fused_few_live_nk {
// One CTA amortizes each of five stream reservations over64K raw keys.
// Each thread owns eight coalesced uint4 word reads. Input classification,
// contiguous per-row reservations and shared-page arbitration are unchanged.
// Union only inside the intersection of all four original visible intervals.
__global__ void compact_words_live_nk(
    const __grid_constant__ PagedArgs capacity, const int* budget, int seed,
    const int* starts, const int* ends, const int* live_nk) {
  using Scan = cub::BlockScan<int, 256>;
  __shared__ typename Scan::TempStorage scan;
  __shared__ int aggregate, base, valid, first_page, page_count, pages[513];
  // Each warp may consume a different lane's output work. The second component
  // is the original CUB offset plus the cumulative popcount through this word.
  __shared__ uint2 output_word[8][256];
  __shared__ int thread_end[256];
  __shared__ int epoch_nk;
  int t = threadIdx.x, row = blockIdx.y;
  if (t == 0) epoch_nk = live_nk[0];
  __syncthreads();
  // Validate before subtraction or any word/table access. Caller keeps this
  // scalar immutable through the launch, and resets output counters first.
  if (epoch_nk < seed || epoch_nk > capacity.nk) {
    if (t == 0) page_error(capacity, row, true, 2);
    return;
  }
  PagedArgs a = capacity;
  a.nk = epoch_nk;
  a.max_records = epoch_nk - seed;
  a.columns = (a.max_records + 127) / 128;
  int active_words = (a.max_records + 31) / 32;
  int family = budget[row];
  bool budget_ok = dosa_geometry::valid_certificate_budget(family);
  if (!budget_ok) {
    if (t == 0) page_error(a, row, true, 2);
    return;
  }
  int lo = max(max(starts[row * 4], starts[row * 4 + 1]),
               max(starts[row * 4 + 2], starts[row * 4 + 3]));
  int hi = min(min(ends[row * 4], ends[row * 4 + 1]),
               min(ends[row * 4 + 2], ends[row * 4 + 3]));
  // Fixed Q-grid; loop bounds depend on liveNK, never capacityNK.
  for (int first_tile = 0; first_tile < active_words; first_tile += 2048) {
    int first_word = first_tile + t;
    uint4 packed[8];
    unsigned common[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      int word = first_word + j * 256;
      packed[j] = word < active_words
                      ? reinterpret_cast<const uint4*>(
                            a.words)[int64_t(row) * a.word_count + word]
                      : make_uint4(0, 0, 0, 0);
      int key = seed + word * 32;
      int remaining = a.nk - key;
      unsigned tail = remaining >= 32
                          ? 0xffffffffu
                          : (remaining <= 0 ? 0u : ((1u << remaining) - 1u));
      packed[j].x &= tail;
      packed[j].y &= tail;
      packed[j].z &= tail;
      packed[j].w &= tail;
      int left = max(lo - key, 0), right = hi - key;
      unsigned lower = left >= 32 ? 0u : (0xffffffffu << min(left, 31));
      unsigned upper = right >= 32
                           ? 0xffffffffu
                           : (right <= 0 ? 0u : ((1u << min(right, 31)) - 1u));
      unsigned a = packed[j].x, b = packed[j].y, c = packed[j].z,
               d = packed[j].w;
      common[j] = ((a & b) | (a & c) | (a & d) | (b & c) | (b & d) | (c & d)) &
                  lower & upper;
    }
#pragma unroll
    for (int stream = 0; stream < 5; ++stream) {
      bool is_common = stream == 4;
      int output_row = is_common ? row : row * 4 + stream;
      unsigned mask[8];
      int n = 0, offset, total;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        unsigned value = stream == 0   ? packed[j].x
                         : stream == 1 ? packed[j].y
                         : stream == 2 ? packed[j].z
                                       : packed[j].w;
        mask[j] = is_common ? common[j] : value & ~common[j];
        n += __popc(mask[j]);
      }
      Scan(scan).ExclusiveSum(n, offset, total);
      if (t == 0) aggregate = total;
      __syncthreads();
      if (!aggregate) continue;
      if (n) {
        int word_end = offset;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          word_end += __popc(mask[j]);
          output_word[j][t] = make_uint2(mask[j], word_end);
        }
      }
      thread_end[t] = offset + n;
      if (t == 0) {
        base = atomicAdd(
            (is_common ? a.common_counts : a.solo_counts) + output_row,
            aggregate);
        valid = 0;
        page_count = 0;
        if (base < 0 || base >= a.max_records)
          page_error(a, output_row, is_common, 2);
        else {
          valid = min(aggregate, a.max_records - base);
          if (valid < aggregate) page_error(a, output_row, is_common, 2);
          first_page = base / 128;
          page_count = (base + valid - 1) / 128 - first_page + 1;
          if (page_count > 513) {
            page_error(a, output_row, is_common, 2);
            valid = 0;
            page_count = 0;
          }
        }
      }
      __syncthreads();
      for (int page = t; page < page_count; page += 256)
        pages[page] =
            ensure_id_page(a, output_row, first_page + page, is_common);
      __syncthreads();
      // Each output rank belongs to exactly one CTA thread, independently of
      // the source owner. Only shared-memory lookups follow; no cross-warp
      // shuffle.
      for (int rank = t; rank < aggregate; rank += 256) {
        int search = rank, owner = 0;
#pragma unroll
        for (int step = 128; step; step >>= 1) {
          int boundary = thread_end[owner + step - 1];
          if (search >= boundary) owner += step;
        }
        int source_thread = owner, word = 0;
#pragma unroll
        for (int step = 4; step; step >>= 1) {
          unsigned boundary = output_word[word + step - 1][source_thread].y;
          if (search >= int(boundary)) word += step;
        }
        uint2 source = output_word[word][source_thread];
        int bit_rank = search - (int(source.y) - __popc(source.x));
        // PTX fns includes base=0 and numbers set bits from one.
        int bit = __fns(source.x, 0, bit_rank + 1);
        int64_t position = int64_t(base) + rank;
        if (rank < valid && position >= 0 && position < a.max_records) {
          int pg = pages[position / 128 - first_page];
          if (pg >= 0 && pg < a.pages)
            a.pool[int64_t(pg) * 128 + (position & 127)] =
                seed + (first_tile + source_thread + word * 256) * 32 + bit;
        }
      }
      __syncthreads();
    }
  }  // live word tiles
}
}  // namespace fused_few_live_nk

namespace test_common_live_nk_host {
// Independent opt-in ABI. nk_capacity and all tensor shapes remain fixed;
// logical liveNK is read on device each launch. No existing entry is replaced.
void compact_candidates_live_nk(std::vector<Tensor> streams, Tensor pool,
                                Tensor table, Tensor head, Tensor status,
                                Tensor error, Tensor words, Tensor budget,
                                int64_t seed, int64_t nk_capacity,
                                Tensor starts, Tensor ends, Tensor live_nk) {
  // Owned I32 stream6, page128 pool, [Q+Q/4,capacityPages] table and
  // [Q/4,capacityWords,4] words. seed12288/K<=2^20 and Q8 padding come from
  // the owner; this stream holds liveNK fixed through scan, compact and refine.
  const int nq = int(words.size(0) * 4), nw = int(words.size(1));
  fused_few_live_nk::PagedArgs pg{};
  pg.pool = pool.mutable_data_ptr<int>();
  pg.table = table.mutable_data_ptr<int>();
  pg.head = head.mutable_data_ptr<int>();
  pg.status = status.mutable_data_ptr<int>();
  pg.error = error.mutable_data_ptr<int>();
  pg.solo_counts = streams[0].mutable_data_ptr<int>();
  pg.solo_tasks = reinterpret_cast<int2*>(streams[1].mutable_data_ptr<int>());
  pg.solo_n = streams[2].mutable_data_ptr<int>();
  pg.common_counts = streams[3].mutable_data_ptr<int>();
  pg.common_tasks = reinterpret_cast<int2*>(streams[4].mutable_data_ptr<int>());
  pg.common_n = streams[5].mutable_data_ptr<int>();
  pg.words = reinterpret_cast<unsigned*>(words.mutable_data_ptr<int>());
  pg.word_count = nw;
  pg.nq = nq;
  pg.nk = nk_capacity;
  pg.max_records = nk_capacity - seed;
  pg.columns = int(table.size(1));
  pg.table_stride = table.stride(0);
  pg.pages = int(pool.size(0));
  pg.solo_task_cap = int(streams[1].size(0));
  pg.common_task_cap = int(streams[4].size(0));
  DeviceGuard guard(words.get_device_index());
  auto stream = get_current_cuda_stream();
  // Launch even for zero physical words so invalid device NK is not hidden.
  fused_few_live_nk::compact_words_live_nk<<<dim3(1, nq / 4), 256, 0, stream>>>(
      pg, budget.mutable_data_ptr<int>(), int(seed),
      starts.mutable_data_ptr<int>(), ends.mutable_data_ptr<int>(),
      live_nk.mutable_data_ptr<int>());
  DOSA_CUDA_CHECK(cudaGetLastError());
}

}  // namespace test_common_live_nk_host

}  // namespace DOSA_NAMESPACE

STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def(DOSA_OP("compact_candidates_live_nk") "(Tensor(a!)[] streams, Tensor! pool, Tensor! table, Tensor! head, Tensor! status, Tensor! error, "
      "Tensor words, Tensor budget, int seed, int nk_capacity, Tensor starts, Tensor ends, "
      "Tensor live_nk) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops) {
  ops.impl(DOSA_OP("compact_candidates_live_nk"),
           TORCH_BOX(&DOSA_NAMESPACE::test_common_live_nk_host::
                         compact_candidates_live_nk));
}
