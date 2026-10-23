#pragma once
#include "common.cuh"
#include "score.cuh"

namespace DOSA_NAMESPACE {
namespace dosa_screen {
using certicascade_streaming_exact::make_map;
using certicascade_streaming_exact::scale_row_stride;
using Barrier = cutlass::arch::ClusterTransactionBarrier;
struct PagedArgs {
  int *pool, *table, *head, *status, *error, *solo_counts, *common_counts,
      *solo_n, *common_n;
  int2 *solo_tasks, *common_tasks;
  unsigned* words;
  int word_count;
  int nq, nk, max_records, columns, pages, solo_task_cap, common_task_cap;
};
struct Param {
  float guard, lo0, lo1, raw, out, tau;
  int start, end, hint;
};
struct Args {
  const uint4 *q, *k, *fine;
  const float *ks, *w, *fine_scale, *meta, *raw, *out, *tau;
  const uint32_t* rho;
  const int *hint, *starts, *ends, *budget;
  float* values;
  int *ids, *counts, *error, *packets, *members, *debug;
  unsigned long long* waits;
  const uint8_t* keep_ref;
  const float* score_ref;
  int nq, nk, cap, seed;
};
__device__ __forceinline__ uint32_t trusted_keep_mask(float value, float tau,
                                                      uint32_t eligible,
                                                      uint32_t bit) {
  uint32_t mask;
  asm volatile(
      "{.reg .pred finite,below,active,keep; .reg .f32 magnitude; "
      "abs.f32 magnitude,%1; setp.lt.f32 finite,magnitude,0f7f800000; "
      "setp.lt.f32 below,%1,%2; and.pred keep,finite,below; not.pred "
      "keep,keep; "
      "setp.ne.u32 active,%3,0; and.pred keep,keep,active; selp.u32 "
      "%0,%4,0,keep;}"
      : "=r"(mask)
      : "f"(value), "f"(tau), "r"(eligible), "r"(bit));
  return mask;
}
template <int Count>
__device__ __forceinline__ void load_tmem(const uint32_t address, float* out) {
  using Loader = std::conditional_t<
      Count == 64, cute::SM100_TMEM_LOAD_32dp32b64x,
      std::conditional_t<
          Count == 32, cute::SM100_TMEM_LOAD_32dp32b32x,
          std::conditional_t<
              Count == 16, cute::SM100_TMEM_LOAD_32dp32b16x,
              std::conditional_t<
                  Count == 8, cute::SM100_TMEM_LOAD_32dp32b8x,
                  std::conditional_t<Count == 4,
                                     cute::SM100_TMEM_LOAD_32dp32b4x,
                                     cute::SM100_TMEM_LOAD_32dp32b2x>>>>>;
  [&]<size_t... I>(cute::index_sequence<I...>) {
    Loader::copy(address, reinterpret_cast<uint32_t*>(out)[I]...);
  }(cute::make_index_sequence<Count>{});
}
}  // namespace dosa_screen

namespace dosa_emit_host {

struct KeyMaps {
  CUtensorMap keys, scales, rho;
};

inline KeyMaps key_maps(const std::vector<Tensor>& x, int tile) {
  using certicascade_streaming_exact::make_map;
  using certicascade_streaming_exact::scale_row_stride;
  const cuuint64_t nk = x[1].size(0);
  return {make_map(x[1].data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, {128, nk},
                   {128}, {128, cuuint32_t(tile)}, CU_TENSOR_MAP_SWIZZLE_128B),
          make_map(x[2].data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, {nk, 1},
                   {scale_row_stride(nk)}, {cuuint32_t(tile), 1},
                   CU_TENSOR_MAP_SWIZZLE_NONE),
          make_map(x[3].data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT32, {nk, 1},
                   {scale_row_stride(nk)}, {cuuint32_t(tile), 1},
                   CU_TENSOR_MAP_SWIZZLE_NONE)};
}

// The owner supplies x15/stream6 views from its fixed typed workspace.
// Args and PagedArgs retain the original device layout for every family.
template <typename Args, typename PagedArgs>
std::pair<Args, PagedArgs> bind(int paid, const std::vector<Tensor>& x,
                                const std::vector<Tensor>& streams,
                                const Tensor& pool, const Tensor& table,
                                const Tensor& head, const Tensor& status,
                                const Tensor& error, int64_t seed,
                                const Tensor& words) {
  // x15: Q,K,KS,rho,W,fine,fine_scale,meta,raw,out,hint,start,end,budget4,tau.
  // The serial owner allocates every tensor on one device with unit inner
  // strides; rows are padded to8, K capacity is<=2^20, seed is12288. It retains
  // the Q4 word matrix, page128 pool and stream6 counters/tasks until
  // completion.
  const int nq = x[0].size(0), nk = x[1].size(0);
  Args a{};
  a.q = reinterpret_cast<const uint4*>(x[0].data_ptr());
  a.k = reinterpret_cast<const uint4*>(x[1].data_ptr());
  a.ks = x[2].mutable_data_ptr<float>();
  a.rho = reinterpret_cast<const uint32_t*>(x[3].mutable_data_ptr<int>());
  a.w = x[4].mutable_data_ptr<float>();
  a.fine = reinterpret_cast<const uint4*>(x[5].data_ptr());
  a.fine_scale = x[6].mutable_data_ptr<float>();
  a.meta = x[7].mutable_data_ptr<float>();
  a.raw = x[8].mutable_data_ptr<float>();
  a.out = x[9].mutable_data_ptr<float>();
  a.hint = x[10].mutable_data_ptr<int>();
  a.starts = x[11].mutable_data_ptr<int>();
  a.ends = x[12].mutable_data_ptr<int>();
  a.budget = x[13].mutable_data_ptr<int>();
  a.tau = x[14].mutable_data_ptr<float>();
  a.nq = nq;
  a.nk = nk;
  a.seed = seed;
  a.error = error.mutable_data_ptr<int>();

  PagedArgs pg{};
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
  pg.word_count = words.size(1);
  pg.nq = nq;
  pg.nk = nk;
  pg.max_records = nk - seed;
  pg.columns = table.size(1);
  pg.pages = pool.size(0);
  pg.solo_task_cap = streams[1].size(0);
  pg.common_task_cap = streams[4].size(0);
  return {a, pg};
}

}  // namespace dosa_emit_host

}  // namespace DOSA_NAMESPACE
