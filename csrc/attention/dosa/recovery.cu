#include "common.cuh"
#include "score.cuh"
#include "geometry.cuh"

namespace DOSA_NAMESPACE {
// One original H32 arithmetic/TMEM implementation for both modes.
namespace dosa_replay_layout {
__device__ __forceinline__ int logical_extent(const int* logical_nk,
                                              int storage_capacity) {
  return logical_nk ? *logical_nk : storage_capacity;
}

__device__ __forceinline__ bool valid_interval(int begin, int end,
                                               int logical_nk,
                                               int storage_capacity) {
  return logical_nk >= 0 && logical_nk <= storage_capacity && begin >= 0 &&
         begin <= end && end <= logical_nk;
}
}  // namespace dosa_replay_layout

namespace dosa_h32_replay {
using namespace certicascade_streaming_exact;

__device__ __forceinline__ int swizzle(int row, int vec) {
  return row * 8 + (vec ^ (row % 8));
}

// Both recovery policies issue the same H32 packet and consume the same TMEM
// result. The caller retains its active-row reduction and output publication.
template <class Shared, class Inputs>
__device__ __forceinline__ void load_packet(Shared& s, const Inputs& a,
                                            int64_t key, bool active, int t,
                                            int tm, int seq,
                                            float (&acc)[kHeads]) {
// No out-of-interval K/scale read, including a partial final packet.
#pragma unroll
  for (int v = 0; v < 8; ++v)
    reinterpret_cast<uint4*>(s.keys)[swizzle(t, v)] =
        active ? reinterpret_cast<const uint4*>(a.k + uint64_t(key) * 128)[v]
               : make_uint4(0, 0, 0, 0);
  s.scale[t] = active ? a.scale[key] : 0.0f;
  cutlass::arch::fence_view_async_shared();
  __syncthreads();
  if (t == 0) {
    deep_gemm::ptx::tcgen05_after_thread_sync();
    issue_h32_mma<128, kHeads>(reinterpret_cast<cutlass::float_e4m3_t*>(s.keys),
                               reinterpret_cast<cutlass::float_e4m3_t*>(s.q), 0,
                               tm, cute::cast_smem_ptr_to_uint(&s.done));
  }
  s.done.wait(seq & 1);
  deep_gemm::ptx::tcgen05_after_thread_sync();
  tmem_load_heads(tm, acc);
}
}  // namespace dosa_h32_replay

namespace partitioned_h32_replay {
using namespace certicascade_streaming_exact;
using Barrier = cutlass::arch::ClusterTransactionBarrier;
struct alignas(1024) Smem {
  alignas(1024) uint8_t keys[128 * 128], q[kHeads * 128];
  alignas(128) float weights[kHeads], scale[128];
  Barrier done;
  uint32_t tmem;
  int job;
};
struct Args {
  const uint8_t *q, *k;
  const float *scale, *weights;
  const int *starts, *ends;
  int* error;
  float* dense;
  int* counts;
  int nk;  // Physical K storage and dense row pitch.
  const int2* tasks;
  int* state;
  int* row_slot;
  float* leaf;
  int* fast_rows;
  int* fast_state;
  int* finite;
  int* leaf_lengths;
  int chunks;
  const int* logical_nk;
  uint8_t* packed_q;
  float* packed_weights;
};
using dosa_h32_replay::swizzle;

constexpr int PartitionKeys = 4096;
// One reservation per failed query; normal rows never read Q/K or emit jobs.
__global__ void prepare_jobs(Args a, int nq, int2* tasks) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= nq) return;
  a.counts[row] = 0;
  a.row_slot[row] = -1;
  if (a.error[row] == 0) return;
  if (!(a.error[row] & 16) || (a.error[row] & ~(1 | 4 | 16)))
    atomicOr(a.state + 19, 1);
  int begin = a.starts[row], end = a.ends[row];
  int logical = dosa_replay_layout::logical_extent(a.logical_nk, a.nk);
  if (!dosa_replay_layout::valid_interval(begin, end, logical, a.nk)) {
    atomicOr(a.state + 19, 1);
    atomicOr(a.error + row, 2);
    return;
  }
  if (begin == end)
    atomicOr(a.state + 19,
             1);  // Empty intervals retain Q1's original behavior.
  if (end > 0) {
    int slot = atomicAdd(a.state + 2, 1);
    a.row_slot[row] = slot;
    a.fast_rows[slot] = row;
    if (slot < 8) {
      a.state[11 + slot] = row;
      if (begin > 0 || end < 2048) atomicOr(a.state + 3 + slot, 1);
    }
  }
  int chunks = int((int64_t(end) - begin + PartitionKeys - 1) / PartitionKeys);
  chunks = max(chunks,
               1);  // An empty visible interval can still need its -Inf prefix.
  int base = atomicAdd(a.state, chunks);
  for (int j = 0; j < chunks; ++j)
    tasks[base + j] =
        make_int2(row, int(int64_t(begin) + int64_t(j) * PartitionKeys));
}
CUTLASS_GLOBAL void __launch_bounds__(128, 4) replay_worker(Args a) {
  const int t = threadIdx.x, warp = t / 32, task_count = a.state[0];
  if (a.state[2] > 8 && a.state[19] == 0) return;
  if (blockIdx.x >= task_count) return;
  extern __shared__ __align__(1024) uint8_t buf[];
  auto& s = *reinterpret_cast<Smem*>(buf);
  if (t == 0) {
    s.done.init(1);
    cutlass::arch::fence_barrier_init();
  }
  if (warp == 0) {
    cute::TMEM::Allocator1Sm().allocate(kHeads, &s.tmem);
    cute::TMEM::Allocator1Sm().release_allocation_lock();
  }
  __syncthreads();
  const int tm = deep_gemm::ptx::ld_shared(&s.tmem);
  int seq = 0;
  for (;;) {
    if (t == 0) s.job = atomicAdd(a.state + 1, 1);
    __syncthreads();
    int job = s.job;
    if (job >= task_count) break;
    int2 task = a.tasks[job];
    int row = task.x;
    int begin = a.starts[row], end = a.ends[row], segment_begin = task.y;
    int segment_end =
        int(min(int64_t(end), int64_t(segment_begin) + PartitionKeys));
    float* dst = a.dense + uint64_t(row) * a.nk;
    int fast_slot = a.row_slot[row];
    bool emit_leaf = a.state[2] <= 8 && fast_slot >= 0 && fast_slot < 8;
    // Exactly the first partition owns the original-ID -Inf prefix. Other
    // partitions write only their disjoint visible ranges; no clearing race.
    if (segment_begin == begin)
      for (int64_t key = t; key < begin; key += 128) dst[key] = -INFINITY;
    if (segment_begin >= segment_end) {
      __syncthreads();
      continue;
    }
    for (int i = t; i < kHeads * 8; i += 128)
      reinterpret_cast<uint4*>(s.q)[swizzle(i / 8, i % 8)] =
          reinterpret_cast<const uint4*>(a.q + uint64_t(row) * kHeads * 128)[i];
    if (t < kHeads) s.weights[t] = a.weights[row * kHeads + t];
    __syncthreads();
    for (int64_t first = segment_begin; first < segment_end;
         first += 128, ++seq) {
      const int64_t key = first + t;
      const bool active = key < segment_end;
      float acc[kHeads];
      dosa_h32_replay::load_packet(s, a, key, active, t, tm, seq, acc);
      if (active) {
        float score = native_bf16_reduce(acc, s.weights, s.scale[t]);
        dst[key] = score;
        if (emit_leaf) {
          a.leaf[uint64_t(fast_slot) * a.chunks * 32768 + key] = score;
          if (!isfinite(score)) atomicOr(a.state + 3 + fast_slot, 1);
        }
      }
      deep_gemm::ptx::tcgen05_before_thread_sync();
      __syncthreads();
    }
  }
  if (warp == 0) cute::TMEM::Allocator1Sm().free(tm, kHeads);
}
// Q1 is complete here. FI consumes counts after the later BQ4 score launch,
// ordered on the same stream when this wrapper publishes an eligible route.
__global__ void publish_counts(Args a, int nq) {
  int index = blockIdx.x * blockDim.x + threadIdx.x, total = a.state[2];
  if (index < nq) {
    int count = 0;
    if (a.error[index] != 0) {
      int begin = a.starts[index], end = a.ends[index];
      int logical = dosa_replay_layout::logical_extent(a.logical_nk, a.nk);
      if (dosa_replay_layout::valid_interval(begin, end, logical, a.nk))
        count = end;
    }
    a.counts[index] = count;
  }
  if (index == 0) {
    a.fast_state[0] = total;
    a.fast_state[1] = total > 8 && a.state[19] == 0;
  }
  if (index < 8) {
    a.fast_rows[index] = index < total ? a.state[11 + index] : -1;
    a.finite[index] = total <= 8 && index < total && a.state[3 + index] == 0;
  }
  if (index < 8 * a.chunks) {
    int slot = index / a.chunks, seg = index % a.chunks;
    bool valid = total <= 8 && slot < total && a.state[3 + slot] == 0;
    a.leaf_lengths[index] =
        valid ? max(0, min(32768, a.ends[a.state[11 + slot]] - seg * 32768))
              : 0;
  }
}

// State/map are owner-retained. Padded Q4 lanes are initialized here and never
// read a stale row-map slot. Packed storage is reused after the scorer in
// stream order.
__global__ void pack_recovery(Args a) {
  if (!a.fast_state[1]) return;
  int slot = blockIdx.x, total = a.fast_state[0];
  if (slot >= ((total + 3) & ~3)) return;
  int row = slot < total ? a.fast_rows[slot] : 0;
  for (int v = threadIdx.x; v < kHeads * 8; v += blockDim.x)
    reinterpret_cast<uint4*>(a.packed_q + uint64_t(slot) * kHeads * 128)[v] =
        slot < total ? reinterpret_cast<const uint4*>(a.q + uint64_t(row) *
                                                                kHeads * 128)[v]
                     : make_uint4(0, 0, 0, 0);
  if (threadIdx.x < kHeads)
    a.packed_weights[slot * kHeads + threadIdx.x] =
        slot < total ? a.weights[row * kHeads + threadIdx.x] : 0.0f;
}
}  // namespace partitioned_h32_replay
static void replay_and_leaf_impl(Tensor q, Tensor k, Tensor scale,
                                 Tensor weights, Tensor starts, Tensor ends,
                                 Tensor error_status, Tensor dense_scores,
                                 Tensor counts, Tensor leaf, Tensor fast_rows,
                                 Tensor fast_state, Tensor finite,
                                 Tensor leaf_lengths, Tensor logical_nk,
                                 Tensor packed_q, Tensor packed_weights,
                                 Tensor tasks, Tensor state, Tensor row_slot) {
  using namespace partitioned_h32_replay;
  DeviceGuard guard(q.get_device_index());
  const int64_t nq = q.size(0), nk = k.size(0);
  // Owned Q[<=512,H,128]/K[Kcap,128] FP8, F32 dense rows and scales,
  // I32 row status/intervals and the immutable I32[1] live extent. Kcap<=2^20.
  // Leaf storage is F32[8*ceil(Kcap/32768),32768] with independent I32
  // metadata. Owner map[512]/state[2] and packed Q/W[512] are shared with the
  // later scorer; state publishes {failed rows, eligible} on every invocation.
  // All buffers are retained by the owner's recovery graph on this stream.
  const int* logical = logical_nk.mutable_data_ptr<int>();
  const int chunks = int((nk + 32767) / 32768);
  const int64_t max_tasks =
      nq * std::max<int64_t>(1, (nk + PartitionKeys - 1) / PartitionKeys);
  const int blocks = int(
      std::min<int64_t>(max_tasks, 4 * get_device_prop()->multiProcessorCount));
  DOSA_CUDA_CHECK(cudaMemsetAsync(state.data_ptr(), 0, 20 * sizeof(int),
                                  get_current_cuda_stream()));
  Args a{static_cast<const uint8_t*>(q.data_ptr()),
         static_cast<const uint8_t*>(k.data_ptr()),
         scale.mutable_data_ptr<float>(),
         weights.mutable_data_ptr<float>(),
         starts.mutable_data_ptr<int>(),
         ends.mutable_data_ptr<int>(),
         error_status.mutable_data_ptr<int>(),
         dense_scores.mutable_data_ptr<float>(),
         counts.mutable_data_ptr<int>(),
         int(nk),
         reinterpret_cast<const int2*>(tasks.mutable_data_ptr<int>()),
         state.mutable_data_ptr<int>(),
         row_slot.mutable_data_ptr<int>(),
         leaf.mutable_data_ptr<float>(),
         fast_rows.mutable_data_ptr<int>(),
         fast_state.mutable_data_ptr<int>(),
         finite.mutable_data_ptr<int>(),
         leaf_lengths.mutable_data_ptr<int>(),
         chunks,
         logical,
         static_cast<uint8_t*>(packed_q.data_ptr()),
         packed_weights.mutable_data_ptr<float>()};
  auto stream = get_current_cuda_stream();
  prepare_jobs<<<(nq + 127) / 128, 128, 0, stream>>>(
      a, int(nq), reinterpret_cast<int2*>(tasks.mutable_data_ptr<int>()));
  DOSA_CUDA_CHECK(cudaGetLastError());
  DOSA_CUDA_CHECK(cudaFuncSetAttribute(
      replay_worker, cudaFuncAttributeMaxDynamicSharedMemorySize,
      sizeof(Smem)));
  replay_worker<<<blocks, 128, sizeof(Smem), stream>>>(a);
  DOSA_CUDA_CHECK(cudaGetLastError());
  publish_counts<<<(std::max<int64_t>(nq, 8 * chunks) + 127) / 128, 128, 0,
                   stream>>>(a, int(nq));
  DOSA_CUDA_CHECK(cudaGetLastError());
  pack_recovery<<<packed_q.size(0), 128, 0, stream>>>(a);
  DOSA_CUDA_CHECK(cudaGetLastError());
}

void replay_and_leaf_capacity(Tensor q, Tensor k, Tensor scale, Tensor weights,
                              Tensor starts, Tensor ends, Tensor error_status,
                              Tensor dense_scores, Tensor counts, Tensor leaf,
                              Tensor fast_rows, Tensor fast_state,
                              Tensor finite, Tensor leaf_lengths,
                              Tensor logical_nk, Tensor packed_q,
                              Tensor packed_weights, Tensor tasks, Tensor state,
                              Tensor row_slot) {
  replay_and_leaf_impl(q, k, scale, weights, starts, ends, error_status,
                       dense_scores, counts, leaf, fast_rows, fast_state,
                       finite, leaf_lengths, logical_nk, packed_q,
                       packed_weights, tasks, state, row_slot);
}

constexpr int Limit = 8, Segment = 32768, Top = 2048;
__global__ void gather_kernel(const float* leaf, const int* selected,
                              const int* lengths, int chunks, const int* rows,
                              const int* state, int* finite, float* merge,
                              int* ids) {
  if (*state > Limit) return;
  int slot = blockIdx.x / chunks, seg = blockIdx.x % chunks;
  if (rows[slot] < 0) return;
  // The finite mask is published by a prior kernel. A later selected-ID fault
  // only rejects this row; all remaining writes are bounded temporary data.
  if (!finite[slot]) return;
  int length = lengths[blockIdx.x];
  for (int j = threadIdx.x; j < Top; j += blockDim.x) {
    int i = selected[int64_t(blockIdx.x) * Top + j];
    int64_t out = int64_t(slot) * (chunks * Top) + seg * Top + j;
    if (i >= 0 && i < length) {
      merge[out] = leaf[int64_t(blockIdx.x) * Segment + i];
      ids[out] = seg * Segment + i;
    } else {
      merge[out] = -INFINITY;
      ids[out] = -1;
      if (i != -1 || length >= Top) atomicExch(finite + slot, 0);
    }
  }
}
__global__ void final_meta_kernel(const int* counts, int nq, int chunks,
                                  const int* rows, const int* state,
                                  const int* finite, int* merge_lengths,
                                  int* fallback_lengths) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r < Limit)
    merge_lengths[r] =
        (*state <= Limit && rows[r] >= 0 && finite[r]) ? chunks * Top : 0;
  if (r >= nq) return;
  bool fast = false;
  if (*state <= Limit)
    for (int i = 0; i < Limit; ++i) fast |= rows[i] == r && finite[i];
  fallback_lengths[r] = fast ? 0 : counts[r];
}
__global__ void publish_kernel(const int* ids, const int* selected, int width,
                               const int* rows, const int* state,
                               const int* finite, int* out) {
  int slot = blockIdx.x;
  if (*state > Limit || rows[slot] < 0 || !finite[slot]) return;
  for (int j = threadIdx.x; j < Top; j += blockDim.x) {
    int i = selected[slot * Top + j];
    // Original FI on >=2048 finite candidates guarantees this range and a
    // nonnegative selected ID. Retain a bounds guard before any indexed load.
    out[int64_t(rows[slot]) * Top + j] =
        (i >= 0 && i < width) ? ids[int64_t(slot) * width + i] : -1;
  }
}

void gather(Tensor leaf, Tensor selected, Tensor lengths, Tensor rows,
            Tensor state, Tensor finite, Tensor merge, Tensor ids) {
  DeviceGuard guard(leaf.get_device_index());
  // Owned F32 leaf/merge and I32[8*chunks,2048] selections/IDs on one stream.
  const int chunks = leaf.size(0) / Limit;
  gather_kernel<<<Limit * chunks, 256, 0, get_current_cuda_stream()>>>(
      leaf.mutable_data_ptr<float>(), selected.mutable_data_ptr<int>(),
      lengths.mutable_data_ptr<int>(), chunks, rows.mutable_data_ptr<int>(),
      state.mutable_data_ptr<int>(), finite.mutable_data_ptr<int>(),
      merge.mutable_data_ptr<float>(), ids.mutable_data_ptr<int>());
  DOSA_CUDA_CHECK(cudaGetLastError());
}
void final_meta(Tensor counts, Tensor rows, Tensor state, Tensor finite,
                Tensor merge_lengths, Tensor fallback_lengths, int64_t chunks) {
  DeviceGuard guard(counts.get_device_index());
  // Owned I32 counts/row routing; chunks derives from the fixed K capacity.
  final_meta_kernel<<<(std::max<int64_t>(counts.numel(), Limit) + 127) / 128,
                      128, 0, get_current_cuda_stream()>>>(
      counts.mutable_data_ptr<int>(), counts.numel(), chunks,
      rows.mutable_data_ptr<int>(), state.mutable_data_ptr<int>(),
      finite.mutable_data_ptr<int>(), merge_lengths.mutable_data_ptr<int>(),
      fallback_lengths.mutable_data_ptr<int>());
  DOSA_CUDA_CHECK(cudaGetLastError());
}
void publish(Tensor ids, Tensor selected, Tensor rows, Tensor state,
             Tensor finite, Tensor output) {
  DeviceGuard guard(ids.get_device_index());
  // Owned I32[8,chunks*2048] IDs and selected[8,2048]; output has2048 columns.
  publish_kernel<<<Limit, 256, 0, get_current_cuda_stream()>>>(
      ids.mutable_data_ptr<int>(), selected.mutable_data_ptr<int>(),
      ids.size(1), rows.mutable_data_ptr<int>(), state.mutable_data_ptr<int>(),
      finite.mutable_data_ptr<int>(), output.mutable_data_ptr<int>());
  DOSA_CUDA_CHECK(cudaGetLastError());
}

__global__ void replay_publish(const int* selected, const int64_t* order,
                               const int* error, int* out, int64_t out_stride) {
  int row = blockIdx.x;
  if (error[row] == 0) return;
  int64_t destination = order[row];
  for (int j = threadIdx.x; j < 2048; j += blockDim.x)
    out[destination * out_stride + j] = selected[int64_t(row) * 2048 + j];
}

void recovered(Tensor selected, Tensor order, Tensor error, Tensor output) {
  // Owned I32[n,2048] recovered IDs, I64[n] original-row order and I32[n]
  // status. Current-stream publication writes only failed rows in owned output.
  DeviceGuard guard(selected.get_device_index());
  replay_publish<<<selected.size(0), 256, 0, get_current_cuda_stream()>>>(
      selected.mutable_data_ptr<int>(), order.mutable_data_ptr<int64_t>(),
      error.mutable_data_ptr<int>(), output.mutable_data_ptr<int>(),
      output.stride(0));
  DOSA_CUDA_CHECK(cudaGetLastError());
}

}  // namespace DOSA_NAMESPACE

STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def(DOSA_OP("replay_and_leaf_capacity") "(Tensor q, Tensor k, Tensor scales, Tensor weights, Tensor starts, Tensor ends, Tensor! error, "
      "Tensor! scores, Tensor! counts, Tensor! leaf, Tensor! rows, Tensor! state, Tensor! finite, "
      "Tensor! lengths, Tensor logical_nk, Tensor! packed_q, Tensor! packed_weights, Tensor! tasks, "
      "Tensor! work_state, Tensor! row_slot) -> ()");
  ops.def(DOSA_OP("gather") "(Tensor leaf, Tensor selected, Tensor lengths, Tensor rows, Tensor state, Tensor! finite, "
      "Tensor! merge, Tensor! ids) -> ()");
  ops.def(DOSA_OP("final_meta") "(Tensor counts, Tensor rows, Tensor state, Tensor finite, Tensor! merge_lengths, "
      "Tensor! fallback_lengths, int chunks) -> ()");
  ops.def(DOSA_OP("publish") "(Tensor ids, Tensor selected, Tensor rows, Tensor state, Tensor finite, Tensor! output) -> ()");
  ops.def(DOSA_OP("recovered") "(Tensor selected, Tensor order, Tensor error, Tensor! output) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops) {
  ops.impl(DOSA_OP("replay_and_leaf_capacity"),
           TORCH_BOX(&DOSA_NAMESPACE::replay_and_leaf_capacity));
  ops.impl(DOSA_OP("gather"), TORCH_BOX(&DOSA_NAMESPACE::gather));
  ops.impl(DOSA_OP("final_meta"), TORCH_BOX(&DOSA_NAMESPACE::final_meta));
  ops.impl(DOSA_OP("publish"), TORCH_BOX(&DOSA_NAMESPACE::publish));
  ops.impl(DOSA_OP("recovered"), TORCH_BOX(&DOSA_NAMESPACE::recovered));
}
