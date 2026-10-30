// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Top-k / top-p masking of rows whose only live entries are a few known
// candidates (sampling from gathered top-k logits):
//
//   keep the k largest candidates (ties: lowest token id), then the shortest
//   prefix of them, by rank, whose renormalized probability reaches p;
//   every other candidate becomes -inf.
//
// Entries outside the candidates already hold a fill that is below every
// candidate and has zero probability, so they are left alone. One block per
// row replaces a sweep of the whole vocabulary.

#include <climits>
#include <cstdint>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

namespace vllm {
namespace topk_topp_candidates {

constexpr int kMaxCandidates = 128;

// Number of entries of (s_val, s_id)[0, count) ranked before (v, id): larger
// value, or equal value and lower id. count is a multiple of 4, the arrays
// 16-byte aligned; branch free so the LDS loads overlap.
__device__ __forceinline__ int rank_of(const float* s_val, const int* s_id,
                                       const int count, const float v,
                                       const int id) {
  int rank = 0;
#pragma unroll 4
  for (int j = 0; j < count; j += 4) {
    const float4 u = *reinterpret_cast<const float4*>(s_val + j);
    const int4 q = *reinterpret_cast<const int4*>(s_id + j);
    rank += (u.x > v) | ((u.x == v) & (q.x < id));
    rank += (u.y > v) | ((u.y == v) & (q.y < id));
    rank += (u.z > v) | ((u.z == v) & (q.z < id));
    rank += (u.w > v) | ((u.w == v) & (q.w < id));
  }
  return rank;
}

__global__ __launch_bounds__(kMaxCandidates) void mask_kernel(
    float* __restrict__ logits, const int64_t row_stride,
    const int64_t* __restrict__ ids, const int num_candidates,
    const int* __restrict__ top_k, const float* __restrict__ top_p) {
  __shared__ __align__(16) float s_val[kMaxCandidates];
  __shared__ __align__(16) int s_id[kMaxCandidates];
  __shared__ float s_exp[kMaxCandidates];

  const int t = threadIdx.x;
  const int64_t row = blockIdx.x;
  float* const row_logits = logits + row * row_stride;
  const bool live = t < num_candidates;
  const int id = live ? static_cast<int>(ids[row * num_candidates + t]) : INT_MAX;
  const float v = live ? row_logits[id] : -INFINITY;
  s_val[t] = v;
  s_id[t] = id;
  __syncthreads();

  const int rank =
      live ? rank_of(s_val, s_id, (num_candidates + 3) & ~3, v, id) : 0;
  const int k = max(1, min(top_k[row], num_candidates));
  bool keep = live && rank < k;

  const float p = top_p != nullptr ? top_p[row] : 1.f;
  if (p < 1.f) {
    if (rank == 0 && live) s_exp[0] = v;
    __syncthreads();
    const float vmax = s_exp[0];
    __syncthreads();
    if (keep) s_exp[rank] = __expf(v - vmax);
    __syncthreads();
    if (keep) {
      float sum = 0.f;
#pragma unroll 1
      for (int j = 0; j < k; ++j) sum += s_exp[j];
      float before = 0.f;
#pragma unroll 1
      for (int j = 0; j < rank; ++j) before += s_exp[j] / sum;
      keep = before < p;
    }
  }
  if (live && !keep) row_logits[id] = -INFINITY;
}


// Vocab-parallel top-k of fp16 logits, gathered across TP ranks as packed
// int64 entries (fp16 bits << 32 | global token id):
//
//   local_topk_kernel: blocks over slices of each row of the local shard
//     select their k largest logits (ties: lowest id) with a two-digit radix
//     select on the order-preserving 16-bit key; the last block of a row
//     merges the slices, all in one launch.
//   merge_kernel: after the all-gather, keeps the k largest of the
//     ranks * k entries of each row, sorted, as ids and fp32 values.

constexpr int kLocalThreads = 256;
constexpr int kLocalWaves = kLocalThreads / 32;
constexpr int kVec = 8;
constexpr int kMaxMerge = 256;
constexpr uint32_t kWindow = 0x800;

__device__ __forceinline__ int64_t pack(uint32_t bits, int64_t id) {
  return static_cast<int64_t>((static_cast<uint64_t>(bits) << 32) |
                              static_cast<uint32_t>(id));
}

// Order-preserving 16-bit keys of both fp16 halves of a word, and back.
__device__ __forceinline__ uint32_t order_keys2(uint32_t w) {
  return w ^ (0x80008000u | (((w >> 15) & 0x00010001u) * 0x7FFFu));
}

__device__ __forceinline__ uint32_t key_bits(uint32_t key) {
  return key ^ ((key & 0x8000u) ? 0x8000u : 0xFFFFu);
}

__device__ __forceinline__ uint32_t half_bits(const uint4& q, const int j) {
  const uint32_t w = j < 2 ? q.x : j < 4 ? q.y : j < 6 ? q.z : q.w;
  return (j & 1) ? w >> 16 : w & 0xFFFFu;
}

// The histograms keep kCopies interleaved copies of each of the 256 bins,
// picked by lane, so atomics on a hot bin from one wave rarely collide.
constexpr int kCopies = 8;

__device__ __forceinline__ void zero_hist(int* hist) {
  const int4 z = {0, 0, 0, 0};
  *reinterpret_cast<int4*>(hist + threadIdx.x * kCopies) = z;
  *reinterpret_cast<int4*>(hist + threadIdx.x * kCopies + 4) = z;
}

__device__ __forceinline__ void hist_add(int* hist, const uint32_t bin) {
  atomicAdd(&hist[bin * kCopies + (threadIdx.x & (kCopies - 1))], 1);
}

// Bin b of hist with sum(hist[b:]) >= need > sum(hist[b+1:]):
// out = {b, sum(hist[b+1:]), hist[b]}. Needs 256 threads; ends synced.
__device__ __forceinline__ void find_bin(const int* hist, int* sums,
                                         const int need, int* out) {
  const int t = threadIdx.x;
  const int4 a = *reinterpret_cast<const int4*>(hist + t * kCopies);
  const int4 b = *reinterpret_cast<const int4*>(hist + t * kCopies + 4);
  sums[t] = a.x + a.y + a.z + a.w + b.x + b.y + b.z + b.w;
  __syncthreads();
  if (t < 32) {
    const int4 lo = *reinterpret_cast<const int4*>(sums + t * 8);
    const int4 hi = *reinterpret_cast<const int4*>(sums + t * 8 + 4);
    const int c[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
    const int sum = lo.x + lo.y + lo.z + lo.w + hi.x + hi.y + hi.z + hi.w;
    int suffix = sum;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      const int v = __shfl_down(suffix, off, 32);
      if (t + off < 32) suffix += v;
    }
    int above = suffix - sum;
#pragma unroll
    for (int j = 7; j >= 0; --j) {
      if (above < need && above + c[j] >= need) {
        out[0] = t * 8 + j;
        out[1] = above;
        out[2] = c[j];
      }
      above += c[j];
    }
  }
  __syncthreads();
}

// Each block selects the top k of one slice of a row, held in registers as
// order keys (kIters 16-byte vectors per thread), and the last block of the
// row to finish merges the slices' candidates.
constexpr int kIters = 4;
constexpr int kSlice = kLocalThreads * kVec * kIters;
constexpr int kMaxTies = 64;

__global__ __launch_bounds__(kLocalThreads) void local_topk_kernel(
    const uint16_t* __restrict__ logits, const int64_t row_stride,
    const int n, const int k, const int64_t id_offset,
    int64_t* __restrict__ cand, int* __restrict__ counters,
    int64_t* __restrict__ out) {
  __shared__ __align__(16) int hist[256 * kCopies];
  __shared__ __align__(16) int sums[256];
  __shared__ __align__(16) float s_val[kMaxMerge];
  __shared__ __align__(16) int s_id[kMaxMerge];
  __shared__ int sel[3];
  __shared__ int wave_max[kLocalWaves];
  __shared__ int slot;
  __shared__ int num_ties;
  __shared__ int ties[kMaxTies];
  __shared__ int is_last;

  const int t = threadIdx.x;
  const int lane = t & 31;
  const int wave = t >> 5;
  const int row_idx = blockIdx.y;
  const int start = blockIdx.x * kSlice;
  const int len = min(kSlice, n - start);
  const int kk = min(k, len);
  const uint16_t* const row = logits + row_idx * row_stride + start;
  int64_t* const c = cand + (static_cast<int64_t>(row_idx) * gridDim.x + blockIdx.x) * k;
  const int64_t base_id = id_offset + start;

  uint4 q[kIters];
#pragma unroll
  for (int it = 0; it < kIters; ++it) {
    const int i = (it * kLocalThreads + t) * kVec;
    if (i < len) {
      q[it] = *reinterpret_cast<const uint4*>(row + i);
      q[it].x = order_keys2(q[it].x);
      q[it].y = order_keys2(q[it].y);
      q[it].z = order_keys2(q[it].z);
      q[it].w = order_keys2(q[it].w);
    }
  }

  // Most of a slice lies far below its top k: histogram only the keys within
  // kWindow of the slice maximum, unless fewer than k fall there.
  uint32_t kmax = 0;
#pragma unroll
  for (int it = 0; it < kIters; ++it) {
    if ((it * kLocalThreads + t) * kVec < len) {
#pragma unroll 1
      for (int j = 0; j < kVec; ++j) kmax = max(kmax, half_bits(q[it], j));
    }
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    kmax = max(kmax, static_cast<uint32_t>(__shfl_xor(static_cast<int>(kmax), off, 32)));
  if (lane == 0) wave_max[wave] = static_cast<int>(kmax);
  zero_hist(hist);
  if (t == 0) {
    slot = 0;
    num_ties = 0;
  }
  __syncthreads();
  kmax = static_cast<uint32_t>(wave_max[lane % kLocalWaves]);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    kmax = max(kmax, static_cast<uint32_t>(__shfl_xor(static_cast<int>(kmax), off, 32)));
  uint32_t floor_key = kmax > kWindow ? kmax - kWindow : 0;
#pragma unroll 1
  for (int pass = 0; pass < 2; ++pass) {
    int inside = 0;
#pragma unroll
    for (int it = 0; it < kIters; ++it) {
      if ((it * kLocalThreads + t) * kVec < len) {
#pragma unroll 1
        for (int j = 0; j < kVec; ++j) {
          const uint32_t key = half_bits(q[it], j);
          if (key >= floor_key) {
            hist_add(hist, key >> 8);
            ++inside;
          }
        }
      }
    }
    if (inside) atomicAdd(&slot, inside);
    __syncthreads();
    if (slot >= kk || floor_key == 0) break;
    __syncthreads();
    zero_hist(hist);
    if (t == 0) slot = 0;
    floor_key = 0;
    __syncthreads();
  }
  find_bin(hist, sums, kk, sel);
  const uint32_t hi = sel[0];
  const int above_hi = sel[1];
  zero_hist(hist);
  if (t == 0) slot = 0;
  __syncthreads();
#pragma unroll
  for (int it = 0; it < kIters; ++it) {
    if ((it * kLocalThreads + t) * kVec < len) {
#pragma unroll 1
      for (int j = 0; j < kVec; ++j) {
        const uint32_t key = half_bits(q[it], j);
        if ((key >> 8) == hi) hist_add(hist, key & 0xFFu);
      }
    }
  }
  __syncthreads();
  find_bin(hist, sums, kk - above_hi, sel);
  const uint32_t thr = (hi << 8) | static_cast<uint32_t>(sel[0]);
  const int num_gt = above_hi + sel[1];
  const int num_eq = kk - num_gt;
  const bool few_ties = sel[2] <= kMaxTies;

  // Entries above the threshold go anywhere in c[0, num_gt); the ties fill
  // c[num_gt, kk) in id order; c[kk, k) gets entries that rank last.
  if (t >= kk && t < k) c[t] = pack(0xFC00u, INT_MAX);
#pragma unroll
  for (int it = 0; it < kIters; ++it) {
    const int i = (it * kLocalThreads + t) * kVec;
    if (i < len) {
#pragma unroll 1
      for (int j = 0; j < kVec; ++j) {
        const uint32_t key = half_bits(q[it], j);
        if (key > thr) c[atomicAdd(&slot, 1)] = pack(key_bits(key), base_id + i + j);
        if (few_ties && key == thr) ties[atomicAdd(&num_ties, 1)] = i + j;
      }
    }
  }
  if (few_ties) {
    __syncthreads();
    if (t < num_ties) {
      const int idx = ties[t];
      int r = 0;
#pragma unroll 1
      for (int j = 0; j < num_ties; ++j) r += ties[j] < idx;
      if (r < num_eq) c[num_gt + r] = pack(key_bits(thr), base_id + idx);
    }
  } else {
    // Many ties: take the first num_eq in id order, through a block scan per
    // vector.
    int taken = 0;
#pragma unroll
    for (int it = 0; it < kIters; ++it) {
      const int i = (it * kLocalThreads + t) * kVec;
      int eq = 0;
      if (i < len) {
#pragma unroll 1
        for (int j = 0; j < kVec; ++j) eq += half_bits(q[it], j) == thr;
      }
      if (taken < num_eq) {
        int incl = eq;
#pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
          const int v = __shfl_up(incl, off, 32);
          if (lane >= off) incl += v;
        }
        if (lane == 31) wave_max[wave] = incl;
        __syncthreads();
        int before = 0;
#pragma unroll 1
        for (int w = 0; w < wave; ++w) before += wave_max[w];
        int total = 0;
#pragma unroll 1
        for (int w = 0; w < kLocalWaves; ++w) total += wave_max[w];
        int pos = taken + before + incl - eq;
        if (eq > 0) {
#pragma unroll 1
          for (int j = 0; j < kVec; ++j) {
            if (half_bits(q[it], j) == thr) {
              if (pos < num_eq) c[num_gt + pos] = pack(key_bits(thr), base_id + i + j);
              ++pos;
            }
          }
        }
        taken += total;
        __syncthreads();
      }
    }
  }

  // The last block of the row merges the gridDim.x * k slice candidates.
  __syncthreads();
  if (t == 0) {
    __threadfence();
    is_last = atomicAdd(&counters[row_idx], 1) == static_cast<int>(gridDim.x) - 1;
  }
  __syncthreads();
  if (!is_last) return;
  __threadfence();
  const int total = gridDim.x * k;
  const int64_t* const row_cand = cand + static_cast<int64_t>(row_idx) * total;
  float v = -INFINITY;
  int id = INT_MAX;
  int64_t e = 0;
  if (t < total) {
    e = static_cast<int64_t>(__hip_atomic_load(
        reinterpret_cast<const unsigned long long*>(row_cand + t),
        __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT));
    v = __half2float(__ushort_as_half(static_cast<uint16_t>(e >> 32)));
    id = static_cast<int>(e & 0xFFFFFFFF);
  }
  s_val[t] = v;
  s_id[t] = id;
  __syncthreads();
  if (t == 0) counters[row_idx] = 0;
  if (t < total) {
    const int r = rank_of(s_val, s_id, (total + 3) & ~3, v, id);
    if (r < k) out[static_cast<int64_t>(row_idx) * k + r] = e;
  }
}

__global__ __launch_bounds__(kMaxMerge) void merge_kernel(
    const int64_t* __restrict__ gathered, const int rows, const int ranks,
    const int k_in, const int k_out, int64_t* __restrict__ ids,
    float* __restrict__ values) {
  __shared__ __align__(16) float s_val[kMaxMerge];
  __shared__ __align__(16) int s_id[kMaxMerge];
  const int t = threadIdx.x;
  const int m = blockIdx.x;
  const int c = ranks * k_in;
  const bool live = t < c;
  float v = -INFINITY;
  int id = INT_MAX;
  if (live) {
    const int r = t / k_in;
    const int64_t q =
        gathered[(static_cast<int64_t>(r) * rows + m) * k_in + (t - r * k_in)];
    v = __half2float(__ushort_as_half(static_cast<uint16_t>(q >> 32)));
    id = static_cast<int>(q & 0xFFFFFFFF);
  }
  s_val[t] = v;
  s_id[t] = id;
  __syncthreads();
  if (!live) return;
  const int rank = rank_of(s_val, s_id, (c + 3) & ~3, v, id);
  if (rank < k_out) {
    ids[static_cast<int64_t>(m) * k_out + rank] = id;
    values[static_cast<int64_t>(m) * k_out + rank] = v;
  }
}

}  // namespace topk_topp_candidates
}  // namespace vllm

void topk_topp_candidates(torch::Tensor logits, torch::Tensor ids,
                          torch::Tensor top_k,
                          std::optional<torch::Tensor> top_p) {
  TORCH_CHECK(logits.dim() == 2 && logits.scalar_type() == torch::kFloat &&
                  logits.stride(1) == 1,
              "logits must be a row-major fp32 [M, V] tensor");
  const int64_t m = logits.size(0);
  TORCH_CHECK(ids.dim() == 2 && ids.size(0) == m && ids.is_contiguous() &&
                  ids.scalar_type() == torch::kLong,
              "ids must be a contiguous int64 [M, C] tensor");
  const int64_t c = ids.size(1);
  TORCH_CHECK(c > 0 && c <= vllm::topk_topp_candidates::kMaxCandidates,
              "at most 128 candidates per row");
  TORCH_CHECK(top_k.numel() == m && top_k.is_contiguous() &&
                  top_k.scalar_type() == torch::kInt,
              "top_k must be a contiguous int32 [M] tensor");
  const float* p_ptr = nullptr;
  if (top_p.has_value()) {
    TORCH_CHECK(top_p->numel() == m && top_p->is_contiguous() &&
                    top_p->scalar_type() == torch::kFloat,
                "top_p must be a contiguous fp32 [M] tensor");
    p_ptr = top_p->data_ptr<float>();
  }
  if (m == 0) return;
  const at::cuda::OptionalCUDAGuard guard(logits.device());
  vllm::topk_topp_candidates::mask_kernel<<<
      m, vllm::topk_topp_candidates::kMaxCandidates, 0,
      at::cuda::getCurrentCUDAStream()>>>(
      logits.data_ptr<float>(), logits.stride(0), ids.data_ptr<int64_t>(),
      static_cast<int>(c), top_k.data_ptr<int>(), p_ptr);
}

void topk_local_packed(torch::Tensor logits, int64_t k, int64_t id_offset,
                       torch::Tensor cand, torch::Tensor counters,
                       torch::Tensor out) {
  namespace tc = vllm::topk_topp_candidates;
  TORCH_CHECK(logits.dim() == 2 && logits.scalar_type() == torch::kHalf &&
                  logits.stride(1) == 1,
              "logits must be a row-major fp16 [M, V] tensor");
  const int64_t m = logits.size(0);
  const int64_t n = logits.size(1);
  TORCH_CHECK(n % tc::kVec == 0 && logits.stride(0) % tc::kVec == 0 &&
                  reinterpret_cast<uintptr_t>(logits.data_ptr()) % 16 == 0,
              "rows must be 16-byte aligned multiples of 8 entries");
  const int64_t slices = (n + tc::kSlice - 1) / tc::kSlice;
  TORCH_CHECK(k > 0 && k <= n && slices * k <= tc::kMaxMerge &&
                  n < (int64_t{1} << 31),
              "need 0 < k <= V and ceil(V / 8192) * k <= 256");
  TORCH_CHECK(out.is_contiguous() && out.scalar_type() == torch::kLong &&
                  out.size(0) == m && out.size(1) == k,
              "out must be a contiguous int64 [M, k] tensor");
  TORCH_CHECK(cand.is_contiguous() && cand.scalar_type() == torch::kLong &&
                  cand.numel() >= m * slices * k,
              "cand must be a contiguous int64 tensor of ceil(V / 8192) * M * k");
  TORCH_CHECK(counters.is_contiguous() && counters.scalar_type() == torch::kInt &&
                  counters.numel() >= m,
              "counters must be a zeroed contiguous int32 tensor of M entries");
  if (m == 0) return;
  const at::cuda::OptionalCUDAGuard guard(logits.device());
  tc::local_topk_kernel<<<dim3(slices, m), tc::kLocalThreads, 0,
                          at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint16_t*>(logits.data_ptr()), logits.stride(0),
      static_cast<int>(n), static_cast<int>(k), id_offset,
      cand.data_ptr<int64_t>(), counters.data_ptr<int>(),
      out.data_ptr<int64_t>());
}

void topk_merge_packed(torch::Tensor gathered, int64_t ranks,
                       torch::Tensor ids, torch::Tensor values) {
  namespace tc = vllm::topk_topp_candidates;
  TORCH_CHECK(gathered.dim() == 2 && gathered.is_contiguous() &&
                  gathered.scalar_type() == torch::kLong &&
                  gathered.size(0) % ranks == 0,
              "gathered must be a contiguous int64 [ranks * M, k] tensor");
  const int64_t m = gathered.size(0) / ranks;
  const int64_t k_in = gathered.size(1);
  TORCH_CHECK(ranks * k_in <= tc::kMaxMerge, "at most 256 entries per row");
  TORCH_CHECK(ids.is_contiguous() && ids.scalar_type() == torch::kLong &&
                  ids.dim() == 2 && ids.size(0) == m && ids.size(1) <= k_in,
              "ids must be a contiguous int64 [M, k] tensor");
  TORCH_CHECK(values.is_contiguous() && values.scalar_type() == torch::kFloat &&
                  values.sizes() == ids.sizes(),
              "values must be a contiguous fp32 tensor shaped like ids");
  if (m == 0) return;
  const at::cuda::OptionalCUDAGuard guard(gathered.device());
  tc::merge_kernel<<<m, tc::kMaxMerge, 0, at::cuda::getCurrentCUDAStream()>>>(
      gathered.data_ptr<int64_t>(), static_cast<int>(m),
      static_cast<int>(ranks), static_cast<int>(k_in),
      static_cast<int>(ids.size(1)), ids.data_ptr<int64_t>(),
      values.data_ptr<float>());
}
