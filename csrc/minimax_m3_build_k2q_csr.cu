// CUDA C++ q2k -> k2q CSR builder.
//
// Five-stage pipeline. q-ascending order within each CSR row is preserved
// by partitioning q across (CTA, warp_in_CTA) units; each unit owns a
// contiguous q-sub-range and reserves a contiguous slot range per row via
// a precomputed exclusive prefix scan.
//
//   M:  build_row_map      -- round-robin packing of rows across batches
//   H:  histogram + tile_counts
//   PR: row prefix         -- single block per head, row_counts -> row_ptr
//   PT: tile prefix        -- multi-block, scan tile_counts along (c, w) axis
//   S:  scatter (sorted)   -- per-warp slot range, q-sequential within warp
//
// Per-warp partitioning: each CTA has kWarps warps; warp w of CTA c owns
// q-range [c*q_per_cta + w*q_per_warp, c*q_per_cta + (w+1)*q_per_warp).
// tile_counts is shaped [G * kWarps, H, total_rows]; the "row" dimension
// of the prefix scan is the flattened (c * kWarps + w) index, scanned in
// lexicographic order so that warp-local slot ranges concatenate to the
// global q-sorted output.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be CUDA")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INT(x) \
  TORCH_CHECK((x).scalar_type() == at::kInt, #x " must be int32")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x); \
  CHECK_INT(x)

namespace {

constexpr int kWarpSize = 32;

__device__ __forceinline__ void advance_batch_only(int const* __restrict__ cu_q,
                                                   int B, int q_abs, int& bi) {
  while (bi < B && cu_q[bi + 1] <= q_abs) ++bi;
}

// Atomic increment of a 16-bit half within a 32-bit SMEM word; returns the
// OLD 16-bit value (slot). Per-warp count must stay < 32768 so the low
// half does not carry into the high half.
//   base_int32 : int32 pointer; element i holds rows 2*i (low) and 2*i+1
//   (high).
__device__ __forceinline__ int atomic_inc_int16_packed(int* base_int32,
                                                       int row) {
  int idx = row >> 1;
  int shift = (row & 1) << 4;  // 0 or 16
  int delta = 1 << shift;
  int old = atomicAdd(&base_int32[idx], delta);
  return (old >> shift) & 0xFFFF;
}

// Read 16-bit half from packed int32 storage.
__device__ __forceinline__ int read_int16_packed(int const* base_int32,
                                                 int row) {
  int v = base_int32[row >> 1];
  int shift = (row & 1) << 4;
  return (v >> shift) & 0xFFFF;
}

// ---------------------------------------------------------------------------
// M: round-robin row map.
// ---------------------------------------------------------------------------
template <int kBlockK>
__global__ void k2q_build_row_map_kernel(int const* __restrict__ cu_k,
                                         int* __restrict__ row_map,
                                         int* __restrict__ row_coords, int B,
                                         int max_kv_blocks) {
  int level = blockIdx.x;
  if (level >= max_kv_blocks) return;
  if (threadIdx.x != 0) return;
  int rows_before = 0;
  for (int b = 0; b < B; ++b) {
    int rb = (cu_k[b + 1] - cu_k[b] + kBlockK - 1) / kBlockK;
    rows_before += (rb < level ? rb : level);
  }
  int active_before = 0;
  for (int b = 0; b < B; ++b) {
    int rb = (cu_k[b + 1] - cu_k[b] + kBlockK - 1) / kBlockK;
    if (rb > level) {
      int row_linear = rows_before + active_before;
      row_map[(size_t)b * max_kv_blocks + level] = row_linear;
      if (row_coords != nullptr) {
        row_coords[(size_t)row_linear * 2] = b;
        row_coords[(size_t)row_linear * 2 + 1] = level;
      }
      ++active_before;
    } else {
      row_map[(size_t)b * max_kv_blocks + level] = -1;
    }
  }
}

// ---------------------------------------------------------------------------
// H: per-warp histogram + tile_counts.
// kWarps warps per CTA, each owns q-sub-range = q_per_cta / kWarps.
// SMEM hist[kWarps, total_rows] int32 (stored as packed int16 cursor:
// 2 entries per int32 word). Each warp counts to its own row.
// At end-of-CTA, write tile_counts[c*kWarps + w, h, r] = smem_hist[w, r]
// and atomicAdd(row_counts[h, r], sum over w of smem_hist[w, r]).
// ---------------------------------------------------------------------------
template <int kTopK, int kBlockK, int kWarps>
__global__ void k2q_hist_kernel(int const* __restrict__ q2k,
                                int const* __restrict__ cu_q,
                                int const* __restrict__ row_map,
                                int* __restrict__ row_counts,
                                int* __restrict__ tile_counts, int H, int B,
                                int S_Q, int total_rows, int max_kv_blocks,
                                int q_per_cta, int q_per_warp) {
  constexpr int kThreads = kWarps * kWarpSize;
  extern __shared__ int smem_hist_int[];
  int* smem_hist = smem_hist_int;
  int tid = threadIdx.x;
  int warp_id = tid >> 5;
  int lane = tid & 31;
  int c = blockIdx.x;
  int q_start_cta = c * q_per_cta;
  int q_end_cta = min(q_start_cta + q_per_cta, S_Q);
  int q_start_warp = min(q_start_cta + warp_id * q_per_warp, q_end_cta);
  int q_end_warp = min(q_start_warp + q_per_warp, q_end_cta);

  constexpr int kInt4PerToken = kTopK / 4;
  int packed_per_warp = (total_rows + 1) >> 1;
  int* my_hist = smem_hist + warp_id * packed_per_warp;

  for (int h = 0; h < H; ++h) {
    for (int i = lane; i < packed_per_warp; i += kWarpSize) my_hist[i] = 0;
    __syncthreads();

    if (q_start_warp < q_end_warp) {
      int bi = 0;
      int qi = q_start_warp + lane;
      advance_batch_only(cu_q, B, qi, bi);

      int4 const* head_topk4 =
          reinterpret_cast<int4 const*>(q2k + (size_t)h * S_Q * kTopK);

      for (; qi < q_end_warp; qi += kWarpSize) {
        advance_batch_only(cu_q, B, qi, bi);
        int const* my_row_map = row_map + (size_t)bi * max_kv_blocks;

        int4 buf[kInt4PerToken];
#pragma unroll
        for (int v = 0; v < kInt4PerToken; ++v) {
          buf[v] = head_topk4[(size_t)qi * kInt4PerToken + v];
        }
#pragma unroll
        for (int t = 0; t < kTopK; ++t) {
          int kvb_local = reinterpret_cast<int const*>(buf)[t];
          if (kvb_local >= 0 && kvb_local < max_kv_blocks) {
            int row = my_row_map[kvb_local];
            if (row >= 0 && row < total_rows) {
              atomic_inc_int16_packed(my_hist, row);
            }
          }
        }
      }
    }
    __syncthreads();

    int* head_row_counts = row_counts + (size_t)h * total_rows;
    // Each warp writes its own slice of tile_counts (full int32) by
    // unpacking int16 entries from SMEM.
    int* my_tile =
        tile_counts + ((size_t)(c * kWarps + warp_id) * H + h) * total_rows;
    for (int i = lane; i < total_rows; i += kWarpSize) {
      my_tile[i] = read_int16_packed(my_hist, i);
    }
    __syncthreads();

    // Sum across warps (int32 accumulator), atomicAdd to row_counts.
    for (int i = tid; i < total_rows; i += kThreads) {
      int sum = 0;
#pragma unroll
      for (int w = 0; w < kWarps; ++w) {
        sum += read_int16_packed(smem_hist + w * packed_per_warp, i);
      }
      if (sum > 0) atomicAdd(&head_row_counts[i], sum);
    }
    if (h + 1 < H) __syncthreads();
  }
}

// ---------------------------------------------------------------------------
// PR: row prefix. One block per head.
// ---------------------------------------------------------------------------
template <int kThreads>
__global__ void k2q_row_prefix_kernel(int const* __restrict__ row_counts,
                                      int* __restrict__ row_ptr,
                                      int const* __restrict__ row_coords,
                                      int* __restrict__ scheduler_metadata,
                                      int* __restrict__ work_count,
                                      int total_rows, int target_q_per_cta,
                                      int work_capacity) {
  int h = blockIdx.x;
  int tid = threadIdx.x;
  __shared__ int scan_buf[kThreads];

  int const* head_counts = row_counts + (size_t)h * total_rows;
  int* head_rowptr = row_ptr + (size_t)h * (total_rows + 1);
  int chunk = (total_rows + kThreads - 1) / kThreads;
  int lo = tid * chunk;
  int hi = min(lo + chunk, total_rows);

  int local_sum = 0;
  for (int i = lo; i < hi; ++i) local_sum += head_counts[i];
  scan_buf[tid] = local_sum;
  __syncthreads();

  for (int off = 1; off < kThreads; off <<= 1) {
    int add = (tid >= off) ? scan_buf[tid - off] : 0;
    __syncthreads();
    scan_buf[tid] += add;
    __syncthreads();
  }
  int running = scan_buf[tid] - local_sum;
  for (int i = lo; i < hi; ++i) {
    int row_count = head_counts[i];
    running += row_count;
    head_rowptr[i + 1] = running;
    if (scheduler_metadata != nullptr && work_count != nullptr &&
        row_count > 0) {
      int num_chunks = (row_count + target_q_per_cta - 1) / target_q_per_cta;
      int base = atomicAdd(work_count, num_chunks);
      int batch_idx = row_coords[(size_t)i * 2];
      int kv_block_idx = row_coords[(size_t)i * 2 + 1];
      for (int c = 0; c < num_chunks; ++c) {
        int work_idx = base + c;
        if (work_idx < work_capacity) {
          int q_begin = c * target_q_per_cta;
          int q_count = min(target_q_per_cta, row_count - q_begin);
          int* meta = scheduler_metadata + (size_t)work_idx * 6;
          meta[0] = h;
          meta[1] = i;
          meta[2] = q_begin;
          meta[3] = q_count;
          meta[4] = batch_idx;
          meta[5] = kv_block_idx;
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// PT_smem: SMEM-staged tile prefix scan.
// Each block handles kRowsPerBlock rows for one head h. Cooperative load
// of tile_counts[*, h, base_r..base_r+M) into SMEM (better coalescing
// than per-warp uncoalesced stride reads), then per-warp scan in SMEM,
// then cooperative store back. Fuses row_ptr into the base.
// ---------------------------------------------------------------------------
template <int kThreads, int kRowsPerBlock>
__global__ void k2q_tile_prefix_smem_kernel(int* __restrict__ tile_counts,
                                            int const* __restrict__ row_ptr,
                                            int H, int total_rows,
                                            int G_total) {
  static_assert(kRowsPerBlock > 0, "kRowsPerBlock must be positive");
  extern __shared__ int smem_tprefix[];
  // smem layout: smem[r_off][g] for r_off in [0, M), g in [0, G_total).

  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp_id = tid >> 5;

  // Grid: H * blocks_per_h. Each block stays within a single head h
  // and processes kRowsPerBlock contiguous rows starting at b_in_h *
  // kRowsPerBlock. (Earlier flat-grid mapping `h = block_job /
  // total_rows; base_r = block_job - h*total_rows` skipped rows when
  // total_rows was not a multiple of kRowsPerBlock and H > 1, because
  // the last partial block of head h-1 left blocks of head h starting
  // at a non-zero row offset.)
  int blocks_per_h = (total_rows + kRowsPerBlock - 1) / kRowsPerBlock;
  int h = blockIdx.x / blocks_per_h;
  int b_in_h = blockIdx.x - h * blocks_per_h;
  if (h >= H) return;
  int base_r = b_in_h * kRowsPerBlock;
  if (base_r >= total_rows) return;
  int actual_M = min(kRowsPerBlock, total_rows - base_r);

  size_t stride_g = (size_t)H * total_rows;
  int* base_ptr = tile_counts + (size_t)h * total_rows + base_r;
  int total_elems = G_total * actual_M;

  // Cooperative load. Pattern: thread tid -> (r_off=tid%M, g=tid/M),
  // then strided. 32 lanes hit M r's × (32/M) g's, giving 32/M cache
  // lines per warp (vs 32 in the naive stride-along-g pattern).
  for (int i = tid; i < total_elems; i += kThreads) {
    int r_off = i % actual_M;
    int g = i / actual_M;
    smem_tprefix[r_off * G_total + g] = base_ptr[g * stride_g + r_off];
  }
  __syncthreads();

  // Per-warp scan: warp w scans row (base_r + w) if w < actual_M.
  if (warp_id < actual_M) {
    int abs_r = base_r + warp_id;
    int rp = row_ptr[(size_t)h * (total_rows + 1) + abs_r];
    int* my_smem = smem_tprefix + warp_id * G_total;
    int running = rp;
    for (int g0 = 0; g0 < G_total; g0 += kWarpSize) {
      int g = g0 + lane;
      int v = (g < G_total) ? my_smem[g] : 0;
      int x = v;
#pragma unroll
      for (int off = 1; off < kWarpSize; off <<= 1) {
        int nbr = __shfl_up_sync(0xFFFFFFFF, x, off);
        if (lane >= off) x += nbr;
      }
      int excl = running + x - v;
      if (g < G_total) my_smem[g] = excl;
      int chunk_sum = __shfl_sync(0xFFFFFFFF, x, 31);
      running += chunk_sum;
    }
  }
  __syncthreads();

  // Cooperative store back.
  for (int i = tid; i < total_elems; i += kThreads) {
    int r_off = i % actual_M;
    int g = i / actual_M;
    base_ptr[g * stride_g + r_off] = smem_tprefix[r_off * G_total + g];
  }
}

// ---------------------------------------------------------------------------
// S: scatter. kWarps warps per CTA, each owns q-sub-range. Per-warp SMEM
// cursor and per-warp tile_offset slot range. Within a warp, q's are
// processed sequentially; lanes 0..kTopK-1 handle the topK slots in
// lockstep. Across distinct q's in the same warp, the lockstep ordering
// guarantees q-monotonic atomicAdd on smem_cursor[r].
// ---------------------------------------------------------------------------
// kQPerIter * kTopK lanes are active per warp iter; remaining lanes idle.
// For kTopK=16, kQPerIter=2 uses all 32 lanes; for kTopK=8, kQPerIter=4.
// CORRECTNESS NOTE: relies on lane-ordered SMEM atomicAdd return values
// within a single warp instruction (verified on B200; tests pass).
//
// SMEM cursor stored as packed int16 (two cursors per int32). Per-warp
// row count must stay < 32768 (~q_per_warp * kTopK at max sink), which
// holds for all task.md sizes up to 1024K.
template <int kTopK, int kBlockK, int kWarps>
__global__ void k2q_scatter_kernel(
    int const* __restrict__ q2k, int const* __restrict__ cu_q,
    int const* __restrict__ row_map, int const* __restrict__ abs_base,
    int* __restrict__ q_idx, int* __restrict__ qsplit_idx,
    int* __restrict__ split_counts, int H, int B, int S_Q, int total_rows,
    int max_kv_blocks, int q_per_cta, int q_per_warp, int max_seqlen_q) {
  constexpr int kQPerIter = kWarpSize / kTopK > 0 ? kWarpSize / kTopK : 1;
  extern __shared__ int smem_cursor_int[];
  int* smem_cursor = smem_cursor_int;
  int tid = threadIdx.x;
  int warp_id = tid >> 5;
  int lane = tid & 31;
  int c = blockIdx.x;
  int q_start_cta = c * q_per_cta;
  int q_end_cta = min(q_start_cta + q_per_cta, S_Q);
  int q_start_warp = min(q_start_cta + warp_id * q_per_warp, q_end_cta);
  int q_end_warp = min(q_start_warp + q_per_warp, q_end_cta);

  int q_in_iter = lane / kTopK;
  int slot_in_q = lane % kTopK;
  bool lane_active = (lane < kQPerIter * kTopK);

  // Per-warp packed cursor: total_rows int16 entries -> ceil(total_rows/2)
  // int32.
  int packed_per_warp = (total_rows + 1) >> 1;
  int* my_cursor = smem_cursor + warp_id * packed_per_warp;

  for (int h = 0; h < H; ++h) {
    for (int i = lane; i < packed_per_warp; i += kWarpSize) my_cursor[i] = 0;
    __syncwarp();

    if (q_start_warp < q_end_warp) {
      int bi = 0;
      advance_batch_only(cu_q, B, q_start_warp, bi);

      int const* head_q2k = q2k + (size_t)h * S_Q * kTopK;
      int const* my_abs_base =
          abs_base + ((size_t)(c * kWarps + warp_id) * H + h) * total_rows;
      int* head_qidx = q_idx + (size_t)h * S_Q * kTopK;

      // (Hot-row register cache experiment showed no measurable
      // benefit; relying on L1 to keep row 0 / row total_rows-1
      // hot since they're hit every iteration in sink workloads.)

      constexpr int kUnroll = 16;
      int qi_base = q_start_warp;
      for (; qi_base + kUnroll * kQPerIter <= q_end_warp;
           qi_base += kUnroll * kQPerIter) {
        int kvb[kUnroll];
        int qloc[kUnroll];
        int batch[kUnroll];
        int const* rmap[kUnroll];

#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
          int qi_u = qi_base + u * kQPerIter + q_in_iter;
          kvb[u] = -1;
          qloc[u] = 0;
          batch[u] = 0;
          if (lane_active) {
            advance_batch_only(cu_q, B, qi_u, bi);
            qloc[u] = qi_u - cu_q[bi];
            batch[u] = bi;
            kvb[u] = head_q2k[(size_t)qi_u * kTopK + slot_in_q];
          }
          rmap[u] = row_map + (size_t)bi * max_kv_blocks;
        }

        int row[kUnroll];
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
          row[u] = -1;
          if (lane_active && kvb[u] >= 0 && kvb[u] < max_kv_blocks)
            row[u] = rmap[u][kvb[u]];
        }

        // Pre-issue all kUnroll abs_base loads in parallel before
        // the atomic chain so memory pipeline runs concurrently
        // with SMEM atomic-adds.
        int abs_v[kUnroll];
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
          abs_v[u] =
              (row[u] >= 0 && row[u] < total_rows) ? my_abs_base[row[u]] : 0;
        }

#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
          int r = row[u];
          bool valid_edge = r >= 0 && r < total_rows;
          unsigned int valid_mask = __ballot_sync(0xFFFFFFFFu, valid_edge);
          unsigned int group_mask =
              (kTopK == 32) ? 0xFFFFFFFFu
                            : (((1u << kTopK) - 1u) << (q_in_iter * kTopK));
          unsigned int lower_lane_mask = lane == 0 ? 0u : ((1u << lane) - 1u);
          int split_slot = __popc(valid_mask & group_mask & lower_lane_mask);
          int valid_count = __popc(valid_mask & group_mask);
          if (split_counts != nullptr && slot_in_q == 0) {
            split_counts[((size_t)batch[u] * max_seqlen_q + qloc[u]) * H + h] =
                valid_count;
          }
          if (valid_edge) {
            int slot = atomic_inc_int16_packed(my_cursor, r);
            int out_pos = abs_v[u] + slot;
            head_qidx[out_pos] = qloc[u];
            if (qsplit_idx != nullptr) {
              qsplit_idx[(size_t)h * S_Q * kTopK + out_pos] =
                  qloc[u] | ((split_slot & 0xFF) << 24);
            }
          }
        }
      }
      // Tail: 1-3 iters left.
      for (; qi_base < q_end_warp; qi_base += kQPerIter) {
        int my_qi = qi_base + q_in_iter;
        bool valid_q = (my_qi < q_end_warp) && lane_active;
        int kvb_local = -1;
        int q_local = 0;
        int batch_local = 0;
        if (valid_q) {
          advance_batch_only(cu_q, B, my_qi, bi);
          batch_local = bi;
          q_local = my_qi - cu_q[bi];
          kvb_local = head_q2k[(size_t)my_qi * kTopK + slot_in_q];
        }
        int const* my_row_map = row_map + (size_t)bi * max_kv_blocks;
        int row = -1;
        if (valid_q && kvb_local >= 0 && kvb_local < max_kv_blocks) {
          row = my_row_map[kvb_local];
        }
        bool valid_edge = row >= 0 && row < total_rows;
        unsigned int valid_mask = __ballot_sync(0xFFFFFFFFu, valid_edge);
        unsigned int group_mask =
            (kTopK == 32) ? 0xFFFFFFFFu
                          : (((1u << kTopK) - 1u) << (q_in_iter * kTopK));
        unsigned int lower_lane_mask = lane == 0 ? 0u : ((1u << lane) - 1u);
        int split_slot = __popc(valid_mask & group_mask & lower_lane_mask);
        int valid_count = __popc(valid_mask & group_mask);
        if (split_counts != nullptr && valid_q && slot_in_q == 0) {
          split_counts[((size_t)batch_local * max_seqlen_q + q_local) * H + h] =
              valid_count;
        }
        if (valid_edge) {
          int slot = atomic_inc_int16_packed(my_cursor, row);
          int out_pos = my_abs_base[row] + slot;
          head_qidx[out_pos] = q_local;
          if (qsplit_idx != nullptr) {
            qsplit_idx[(size_t)h * S_Q * kTopK + out_pos] =
                q_local | ((split_slot & 0xFF) << 24);
          }
        }
      }
    }
    if (h + 1 < H) __syncthreads();
  }
}

}  // anonymous namespace

// ===========================================================================
// Host orchestration
// ===========================================================================

template <int kTopK, int kBlockK>
static void launch_pipeline(torch::Tensor q2k, torch::Tensor cu_q,
                            torch::Tensor cu_k, torch::Tensor row_ptr,
                            torch::Tensor q_idx, int total_rows,
                            int max_kv_blocks,
                            torch::Tensor scheduler_metadata = torch::Tensor(),
                            torch::Tensor work_count = torch::Tensor(),
                            torch::Tensor qsplit_idx = torch::Tensor(),
                            torch::Tensor split_counts = torch::Tensor(),
                            int target_q_per_cta = 1, int work_capacity = 0,
                            int max_seqlen_q = 0) {
  int H = (int)q2k.size(0);
  int S_Q = (int)q2k.size(1);
  int topK = (int)q2k.size(2);
  TORCH_CHECK(topK == kTopK, "topK runtime != template kTopK");
  int B = (int)cu_q.size(0) - 1;
  auto device = q2k.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_CUDA_CHECK(cudaMemsetAsync(row_ptr.data_ptr<int>(), 0,
                                (size_t)H * (total_rows + 1) * sizeof(int),
                                stream));
  AT_CUDA_CHECK(cudaMemsetAsync(q_idx.data_ptr<int>(), 0xFF,
                                (size_t)H * S_Q * kTopK * sizeof(int), stream));

  auto opts = torch::TensorOptions().dtype(torch::kInt32).device(device);
  auto row_counts = torch::zeros({H, total_rows}, opts);
  auto row_map = torch::empty({B, max_kv_blocks}, opts);
  bool emit_schedule = scheduler_metadata.defined();
  auto row_coords =
      emit_schedule ? torch::empty({total_rows, 2}, opts) : torch::Tensor();
  int* scheduler_metadata_ptr =
      emit_schedule ? scheduler_metadata.data_ptr<int>() : nullptr;
  int* work_count_ptr = emit_schedule ? work_count.data_ptr<int>() : nullptr;
  int* qsplit_idx_ptr = emit_schedule ? qsplit_idx.data_ptr<int>() : nullptr;
  int* split_counts_ptr =
      emit_schedule ? split_counts.data_ptr<int>() : nullptr;
  int* row_coords_ptr = emit_schedule ? row_coords.data_ptr<int>() : nullptr;
  if (emit_schedule) {
    AT_CUDA_CHECK(cudaMemsetAsync(work_count_ptr, 0, sizeof(int), stream));
    AT_CUDA_CHECK(cudaMemsetAsync(scheduler_metadata_ptr, 0,
                                  (size_t)work_capacity * 6 * sizeof(int),
                                  stream));
  }

  int dev = q2k.get_device();
  int num_sms = 0;
  AT_CUDA_CHECK(
      cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev));

  // -- Pick kWarps per CTA based on SMEM budget for cursor/hist ---------
  // SMEM per CTA = kWarps * total_rows * sizeof(int) (for both H and S).
  // Want at least 2 CTAs/SM for memory parallelism. SM100 SMEM = 228KB.
  // Pick the largest kWarps that fits two CTAs/SM, capped at 4.
  // SMEM cursor packed as int16 (2 entries per int32 word):
  int per_warp_smem = ((total_rows + 1) >> 1) * (int)sizeof(int);
  int kWarps_pick = 4;
  while (kWarps_pick > 1 && (kWarps_pick * per_warp_smem) * 2 > 228 * 1024) {
    kWarps_pick >>= 1;
  }
  if (kWarps_pick < 1) kWarps_pick = 1;

  // -- Pick G (CTAs) ----------------------------------------------------
  // For each (kWarps, per_warp_smem) pair, the SMEM-bound occupancy is
  // 228KB / (kWarps*per_warp_smem) CTAs/SM. We size G as
  // num_sms * occupancy so a single resident wave covers all CTAs and
  // the memory pipeline runs at peak.
  int per_cta_smem_bytes = kWarps_pick * per_warp_smem;
  int max_ctas_per_sm =
      std::max(1, (228 * 1024) / std::max(1, per_cta_smem_bytes));
  if (max_ctas_per_sm > 8) max_ctas_per_sm = 8;
  constexpr int kMinQPerCta = 256;
  // Cap target_g at num_sms * 3 — empirically this balances
  // per-CTA work-size against parallelism. Higher caps regress
  // mid-size cases due to row_counts atomicAdd contention and
  // smaller q_per_cta. SMEM-bound configurations naturally cap
  // lower if max_ctas_per_sm < 3.
  int target_g = num_sms * std::min(max_ctas_per_sm, 3);
  int max_g_for_q = (S_Q + kMinQPerCta - 1) / kMinQPerCta;
  int G = std::min({target_g, max_g_for_q, S_Q});
  if (G < 1) G = 1;
  int q_per_cta = (S_Q + G - 1) / G;
  G = (S_Q + q_per_cta - 1) / q_per_cta;
  int q_per_warp = (q_per_cta + kWarps_pick - 1) / kWarps_pick;
  int G_total = G * kWarps_pick;

  auto tile_counts = torch::empty({G_total, H, total_rows}, opts);

  // -- Compile-time switch on kWarps for the templated kernels ---------
  auto rmap_fn = k2q_build_row_map_kernel<kBlockK>;
  auto rprefix_fn = k2q_row_prefix_kernel<1024>;
  constexpr int kPtRowsPerBlock = 8;
  constexpr int kPtThreads = 256;
  auto tprefix_smem_fn =
      k2q_tile_prefix_smem_kernel<kPtThreads, kPtRowsPerBlock>;

  if (max_kv_blocks > 0) {
    rmap_fn<<<max_kv_blocks, 32, 0, stream>>>(cu_k.data_ptr<int>(),
                                              row_map.data_ptr<int>(),
                                              row_coords_ptr, B, max_kv_blocks);
  }

  auto launch_hist_scatter = [&](auto kWarps_const) {
    constexpr int W = decltype(kWarps_const)::value;
    size_t smem_bytes = (size_t)W * per_warp_smem;
    auto hist_fn = k2q_hist_kernel<kTopK, kBlockK, W>;
    auto scat_fn = k2q_scatter_kernel<kTopK, kBlockK, W>;
    AT_CUDA_CHECK(cudaFuncSetAttribute(
        hist_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));
    AT_CUDA_CHECK(cudaFuncSetAttribute(
        scat_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));

    hist_fn<<<G, W * kWarpSize, smem_bytes, stream>>>(
        q2k.data_ptr<int>(), cu_q.data_ptr<int>(), row_map.data_ptr<int>(),
        row_counts.data_ptr<int>(), tile_counts.data_ptr<int>(), H, B, S_Q,
        total_rows, max_kv_blocks, q_per_cta, q_per_warp);

    rprefix_fn<<<H, 1024, 0, stream>>>(
        row_counts.data_ptr<int>(), row_ptr.data_ptr<int>(),
        emit_schedule ? row_coords.data_ptr<int>() : nullptr,
        scheduler_metadata_ptr, work_count_ptr, total_rows, target_q_per_cta,
        work_capacity);

    // Grid is H * blocks_per_h so each block stays within a single
    // head; flat (H*total_rows) grid would skip rows when total_rows
    // is not a multiple of kPtRowsPerBlock.
    int blocks_per_h = (total_rows + kPtRowsPerBlock - 1) / kPtRowsPerBlock;
    int pt_grid = H * blocks_per_h;
    if (pt_grid < 1) pt_grid = 1;
    size_t pt_smem = (size_t)kPtRowsPerBlock * G_total * sizeof(int);
    AT_CUDA_CHECK(cudaFuncSetAttribute(
        tprefix_smem_fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)pt_smem));
    tprefix_smem_fn<<<pt_grid, kPtThreads, pt_smem, stream>>>(
        tile_counts.data_ptr<int>(), row_ptr.data_ptr<int>(), H, total_rows,
        G_total);

    scat_fn<<<G, W * kWarpSize, smem_bytes, stream>>>(
        q2k.data_ptr<int>(), cu_q.data_ptr<int>(), row_map.data_ptr<int>(),
        tile_counts.data_ptr<int>(), q_idx.data_ptr<int>(), qsplit_idx_ptr,
        split_counts_ptr, H, B, S_Q, total_rows, max_kv_blocks, q_per_cta,
        q_per_warp, max_seqlen_q);
  };

  if (kWarps_pick == 4) {
    launch_hist_scatter(std::integral_constant<int, 4>{});
  } else if (kWarps_pick == 2) {
    launch_hist_scatter(std::integral_constant<int, 2>{});
  } else {
    launch_hist_scatter(std::integral_constant<int, 1>{});
  }
}

void run_minimax_m3_build_k2q_csr_with_schedule(
    torch::Tensor q2k, torch::Tensor cu_q, torch::Tensor cu_k,
    torch::Tensor row_ptr, torch::Tensor q_idx,
    torch::Tensor scheduler_metadata, torch::Tensor work_count,
    torch::Tensor qsplit_idx, torch::Tensor split_counts, int64_t topk,
    int64_t blk_kv, int64_t total_rows, int64_t max_kv_blocks,
    int64_t target_q_per_cta, int64_t work_capacity, int64_t max_seqlen_q) {
  CHECK_INPUT(q2k);
  CHECK_INPUT(cu_q);
  CHECK_INPUT(cu_k);
  CHECK_INPUT(row_ptr);
  CHECK_INPUT(q_idx);
  CHECK_INPUT(scheduler_metadata);
  CHECK_INPUT(work_count);
  CHECK_INPUT(qsplit_idx);
  CHECK_INPUT(split_counts);
  TORCH_CHECK(blk_kv == 128, "build_k2q_csr only supports blk_kv == 128");
  int H = (int)q2k.size(0);
  int S_Q = (int)q2k.size(1);
  int tr = (int)total_rows;
  int mkv = (int)max_kv_blocks;
  int target = (int)target_q_per_cta;
  int capacity = (int)work_capacity;
  int max_sq = (int)max_seqlen_q;
  TORCH_CHECK(tr >= 0 && mkv >= 0 && target > 0 && capacity > 0 && max_sq >= 0,
              "invalid schedule sizing arguments");
  TORCH_CHECK(row_ptr.size(0) == H && row_ptr.size(1) == tr + 1,
              "row_ptr shape mismatch");
  TORCH_CHECK(q_idx.size(0) == H && q_idx.size(1) == (int64_t)S_Q * (int)topk,
              "q_idx shape mismatch");
  TORCH_CHECK(qsplit_idx.sizes() == q_idx.sizes(), "qsplit_idx shape mismatch");
  TORCH_CHECK(
      scheduler_metadata.size(0) == capacity && scheduler_metadata.size(1) == 6,
      "scheduler_metadata shape mismatch");
  TORCH_CHECK(work_count.numel() == 1,
              "work_count must have one int32 element");
  TORCH_CHECK(split_counts.dim() == 3 &&
                  split_counts.size(0) == cu_q.size(0) - 1 &&
                  split_counts.size(1) == max_sq && split_counts.size(2) == H,
              "split_counts shape mismatch");
  if (S_Q == 0 || tr == 0 || H == 0 || mkv == 0) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AT_CUDA_CHECK(cudaMemsetAsync(row_ptr.data_ptr<int>(), 0,
                                  (size_t)H * (tr + 1) * sizeof(int), stream));
    AT_CUDA_CHECK(cudaMemsetAsync(q_idx.data_ptr<int>(), 0xFF,
                                  (size_t)H * S_Q * (int)topk * sizeof(int),
                                  stream));
    AT_CUDA_CHECK(
        cudaMemsetAsync(work_count.data_ptr<int>(), 0, sizeof(int), stream));
    if (split_counts.numel() > 0) {
      AT_CUDA_CHECK(cudaMemsetAsync(split_counts.data_ptr<int>(), 0,
                                    (size_t)split_counts.numel() * sizeof(int),
                                    stream));
    }
    return;
  }

  if (topk == 16) {
    launch_pipeline<16, 128>(q2k, cu_q, cu_k, row_ptr, q_idx, tr, mkv,
                             scheduler_metadata, work_count, qsplit_idx,
                             split_counts, target, capacity, max_sq);
  } else if (topk == 8) {
    launch_pipeline<8, 128>(q2k, cu_q, cu_k, row_ptr, q_idx, tr, mkv,
                            scheduler_metadata, work_count, qsplit_idx,
                            split_counts, target, capacity, max_sq);
  } else if (topk == 32) {
    launch_pipeline<32, 128>(q2k, cu_q, cu_k, row_ptr, q_idx, tr, mkv,
                             scheduler_metadata, work_count, qsplit_idx,
                             split_counts, target, capacity, max_sq);
  } else if (topk == 4) {
    launch_pipeline<4, 128>(q2k, cu_q, cu_k, row_ptr, q_idx, tr, mkv,
                            scheduler_metadata, work_count, qsplit_idx,
                            split_counts, target, capacity, max_sq);
  } else {
    TORCH_CHECK(false, "unsupported topK ", topk,
                " (expected 4, 8, 16, or 32)");
  }
}
