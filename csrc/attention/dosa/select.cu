#include "common.cuh"
#include "../../fused_mqa_topk/common.cuh"

namespace DOSA_NAMESPACE::record_selection {
constexpr int Top = 2048, Threads = 256, Bins = 4096, IdMask = 0xFFFFF;
constexpr unsigned Full = 0xFFFFFFFFu;
constexpr int FinishThreads = 128;
template <int Items>
using Sort = cub::BlockRadixSort<int, FinishThreads, Items, int>;
struct Args {
  const uint8_t* low;
  const int64_t* order;
  const int *counts, *starts, *ends, *permutation;
  int *words, *error, *output, *partition;
  int rows, cap, nk;
  int64_t low_stride, word_stride;
};

__device__ bool failed(const Args& a, int row, unsigned bad) {
  if (!bad) return false;
  if (threadIdx.x == 0) atomicOr(a.error + row, 1);
  for (int j = threadIdx.x; j < Top; j += Threads)
    a.output[a.order[row] * Top + j] = -1;
  return true;
}

struct Boundary {
  int bin, before, count;
};

// One full warp searches consecutive 32-bin chunks for the requested rank.
template <int Chunks>
__device__ __forceinline__ Boundary find_boundary(const unsigned* hist,
                                                  unsigned rank) {
  const int lane = threadIdx.x % 32;
  unsigned seen = 0;
  for (int part = 0; part < Chunks; ++part) {
    const unsigned value = hist[part * 32 + lane];
    const unsigned chunk = __reduce_add_sync(Full, value);
    if (rank > chunk) {
      rank -= chunk;
      seen += chunk;
      continue;
    }
    const unsigned inclusive = fused_mqa_topk::warp_inclusive_scan(value, lane);
    const int winner = __ffs(__ballot_sync(Full, inclusive >= rank)) - 1;
    return {part * 32 + winner,
            int(seen + __shfl_sync(Full, inclusive - value, winner)),
            int(__shfl_sync(Full, value, winner))};
  }
  return {-1, 0, 0};
}

// Warp-compact B/E; coarse mode caches low8 and builds the fine histogram.
template <bool Coarse = false>
__device__ __forceinline__ void publish_partition(
    const Args& a, int row, unsigned payload, bool is_b, bool is_e,
    unsigned* boundary, unsigned* fine_hist, int& written_b, int& written_e,
    unsigned& bad) {
  const int lane = threadIdx.x % 32;
  const unsigned bm = __ballot_sync(Full, is_b), em = __ballot_sync(Full, is_e);
  if constexpr (Coarse)
    if (!(bm | em)) return;
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
      a.output[a.order[row] * Top + at] = int(payload);
    else
      atomicOr(&bad, 16u);
  }
  if (is_e) {
    const int at = ebase + __popc(em & lower);
    if (at < Bins)
      boundary[at] = payload;
    else
      atomicOr(&bad, 16u);
    if constexpr (Coarse) atomicAdd(fine_hist + (payload >> 20), 1u);
  }
}

template <int Items>
__device__ void finish(Args a, int row, int certain, int uncertain,
                       void* storage) {
  const int* boundary = a.words + int64_t(row) * a.word_stride;
  int keys[Items], values[Items];
#pragma unroll
  for (int i = 0; i < Items; ++i) {
    const int at = i * FinishThreads + threadIdx.x;
    const int id = at < uncertain ? boundary[at] : -1;
    keys[i] = at < uncertain ? a.permutation[id] : INT_MAX;
    values[i] = id;
  }
  Sort<Items>(*static_cast<typename Sort<Items>::TempStorage*>(storage))
      .SortBlockedToStriped(keys, values);
#pragma unroll
  for (int i = 0; i < Items; ++i) {
    const int at = i * FinishThreads + threadIdx.x;
    if (at < Top - certain)
      a.output[a.order[row] * Top + certain + at] = values[i];
  }
}

template <bool Large>
__global__ void complete(Args a) {
  const int row = blockIdx.x;
  const unsigned partition = uint32_t(a.partition[row]);
  const int certain = partition >> 13, uncertain = partition & 8191;
  if (!uncertain || (uncertain > 512) != Large) return;
  extern __shared__ __align__(16) uint8_t storage[];
  if constexpr (Large)
    finish<32>(a, row, certain, uncertain, storage);
  else if (uncertain <= 128)
    finish<1>(a, row, certain, uncertain, storage);
  else
    finish<4>(a, row, certain, uncertain, storage);
}

__global__ __launch_bounds__(Threads) void select(Args a) {
  const int row = a.rows - 1 - int(blockIdx.x), t = threadIdx.x, warp = t / 32,
            lane = t % 32;
  const int count = a.counts[row];
  extern __shared__ __align__(16) uint8_t storage[];
  unsigned* hist = reinterpret_cast<unsigned*>(storage);
  __shared__ unsigned warp_count[8], bad;
  __shared__ int winning_warp, group_before, pivot, certain, uncertain,
      written_b, written_e;
  if (t == 0) {
    bad = 0;
    if (count < Top || count > a.cap) bad |= 1;
    if (a.starts[row] < 0 || a.ends[row] < a.starts[row] || a.ends[row] > a.nk)
      bad |= 4;
    if (a.error[row] != 0) bad |= 16;
    a.partition[row] = 0;
    winning_warp = -1;
    group_before = 0;
    pivot = -1;
    certain = uncertain = 0;
    written_b = written_e = 0;
  }
  for (int bin = t; bin < Bins; bin += Threads) hist[bin] = 0;
  __syncthreads();
  if (failed(a, row, bad)) return;

  int* const words = a.words + int64_t(row) * a.word_stride;
  const uint8_t* const low = a.low + int64_t(row) * a.low_stride;
  // Row bounds stay live until reads drain, before prefix reuse.
  const unsigned row_start = unsigned(a.starts[row]);
  const unsigned row_end = unsigned(a.ends[row]);
  {
    auto count_word = [&](unsigned word) {
      const unsigned cp = word >> 20, id = word & IdMask;
      if (cp < 0x8u || cp > 0xFF7u)
        atomicOr(&bad, 2u);
      else if (id >= unsigned(a.nk) || id < row_start || id >= row_end)
        atomicOr(&bad, 4u);
      else
        atomicAdd(hist + cp, 1u);
    };
    int64_t scalar_begin = t;
    if ((reinterpret_cast<uintptr_t>(words) & 15u) == 0) {
      const int64_t vectors = count / 4;
      for (int64_t vector = t; vector < vectors; vector += Threads) {
        const uint4 quad = *reinterpret_cast<const uint4*>(words + vector * 4);
        const unsigned values[4] = {quad.x, quad.y, quad.z, quad.w};
#pragma unroll
        for (int item = 0; item < 4; ++item) count_word(values[item]);
      }
      scalar_begin = vectors * 4 + t;
    }
    for (int64_t slot = scalar_begin; slot < count; slot += Threads)
      count_word(unsigned(words[slot]));
  }
  __syncthreads();
  if (failed(a, row, bad)) return;

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
  if (failed(a, row, bad)) return;

  if (warp == winning_warp) {
    const auto found = find_boundary<16>(hist + warp * 512, Top - group_before);
    if (lane == 0) {
      pivot = warp * 512 + found.bin;
      certain = group_before + found.before;
      uncertain = found.count;
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
      auto collect = [&](unsigned word, int64_t slot, bool live) {
        const unsigned hi = word >> 20, id = word & IdMask;
        const bool is_b = live && hi < coarse, is_e = live && hi == coarse;
        // Only cached E entries carry their low score byte above the key ID.
        const unsigned payload = is_e ? id | (unsigned(low[slot]) << 20) : id;
        publish_partition<true>(a, row, payload, is_b, is_e, hist, fine_hist,
                                written_b, written_e, bad);
      };
      int64_t scalar_first = 0;
      if ((reinterpret_cast<uintptr_t>(words) & 15u) == 0) {
        scalar_first = (count / 128) * 128;
        for (int64_t begin = warp * 128; begin < scalar_first;
             begin += Threads * 4) {
          const int64_t slot = begin + lane * 4;
          const uint4 quad = *reinterpret_cast<const uint4*>(words + slot);
          const unsigned values[4] = {quad.x, quad.y, quad.z, quad.w};
#pragma unroll
          for (int item = 0; item < 4; ++item)
            collect(values[item], slot + item, true);
        }
      }
      for (int64_t begin = scalar_first + warp * 32; begin < count;
           begin += Threads) {
        const int64_t slot = begin + lane;
        const bool live = slot < count;
        collect(live ? unsigned(words[slot]) : 0u, slot, live);
      }
    } else {
      for (int64_t slot = t; slot < count; slot += Threads) {
        const unsigned word = unsigned(words[slot]);
        if ((word >> 20) == coarse)
          atomicAdd(fine_hist + unsigned(low[slot]), 1u);
      }
    }
    __syncthreads();
    if (t == 0 && winning_warp &&
        (written_b != certain || written_e != group_before))
      bad |= 16;
    __syncthreads();
    if (failed(a, row, bad)) return;
    if (warp == 0) {
      const auto found = find_boundary<8>(fine_hist, Top - certain);
      if (lane == 0) {
        pivot = int((coarse << 8) | unsigned(found.bin));
        certain += found.before;
        uncertain = found.count;
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
    if (uncertain > Bins) bad |= 32;
  }
  __syncthreads();
  if (failed(a, row, bad)) return;

  if (!winning_warp) {
    for (int64_t begin = warp * 32; begin < count; begin += Threads) {
      const int64_t slot = begin + lane;
      const bool live = slot < count;
      const unsigned word = live ? unsigned(words[slot]) : 0u;
      const unsigned coarse = word >> 20, id = word & IdMask;
      unsigned cp = coarse << 8;
      if (live && coarse == (unsigned(pivot) >> 8)) cp |= unsigned(low[slot]);
      const bool is_b = live && cp < unsigned(pivot),
                 is_e = live && cp == unsigned(pivot);
      publish_partition(a, row, id, is_b, is_e, hist, nullptr, written_b,
                        written_e, bad);
    }
  }
  if (winning_warp) {
    // Original reads drained before the cache writes the global prefix.
    for (int begin = warp * 32; begin < group_before; begin += Threads) {
      const int at = begin + lane;
      const bool live = at < group_before;
      const unsigned word = live ? hist[at] : 0u;
      const unsigned code = word >> 20, id = word & IdMask;
      const bool is_b = live && code < (unsigned(pivot) & 255u);
      const bool is_e = live && code == (unsigned(pivot) & 255u);
      publish_partition(a, row, id, is_b, is_e,
                        reinterpret_cast<unsigned*>(words), nullptr, written_b,
                        written_e, bad);
    }
  }

  __syncthreads();
  if (t == 0 && (written_b != certain || written_e != uncertain)) bad |= 16;
  __syncthreads();
  if (failed(a, row, bad)) return;
  // Both paths joined after all input reads before the prefix becomes E.
  if (!winning_warp)
    for (int at = t; at < uncertain; at += Threads) words[at] = int(hist[at]);
  if (t == 0) a.partition[row] = (certain << 13) | uncertain;
}
}  // namespace DOSA_NAMESPACE::record_selection

namespace DOSA_NAMESPACE {
void select_dense5(Tensor low, Tensor packed, Tensor counts, Tensor starts,
                   Tensor ends, Tensor error, Tensor output, Tensor order,
                   Tensor permutation, Tensor partition, int64_t nk) {
  using namespace record_selection;
  DeviceGuard guard(packed.get_device_index());
  const int cap = record5::checked_capacity(low, packed);
  STD_TORCH_CHECK(low.size(0) >= packed.size(0),
                  "Record5 low plane has insufficient query rows");
  Args args{low.const_data_ptr<uint8_t>(),
            order.const_data_ptr<int64_t>(),
            counts.const_data_ptr<int>(),
            starts.const_data_ptr<int>(),
            ends.const_data_ptr<int>(),
            permutation.const_data_ptr<int>(),
            packed.mutable_data_ptr<int>(),
            error.mutable_data_ptr<int>(),
            output.mutable_data_ptr<int>(),
            partition.mutable_data_ptr<int>(),
            int(packed.size(0)),
            cap,
            int(nk),
            low.stride(0),
            packed.stride(0)};
  launch<record_selection::select, Threads, Bins * sizeof(unsigned)>(args.rows,
                                                                     args);
  // A small boundary does not need the 32-items/thread sort's register pool.
  constexpr size_t small_bytes =
      std::max(sizeof(typename Sort<1>::TempStorage),
               sizeof(typename Sort<4>::TempStorage));
  launch<complete<false>, FinishThreads, small_bytes>(args.rows, args);
  launch<complete<true>, FinishThreads, sizeof(typename Sort<32>::TempStorage)>(
      args.rows, args);
}
namespace native_seed {
constexpr int Threads = 256, Seed = 12288, Top = 2048;
struct Output {
  uint8_t* low;
  int *words, *counts;
  float* tau;
  int *error, *status;
  int64_t low_stride, word_stride;
};

__device__ __forceinline__ uint32_t score_bits(float v) {
  return __float_as_uint(v);
}
__device__ __forceinline__ uint32_t score_bits(__nv_bfloat16 v) {
  return uint32_t(__bfloat16_as_ushort(v)) << 16;
}

// On the canonical native BF16 bits this descending-order transform is its own
// inverse.
__device__ __forceinline__ unsigned ordered16(unsigned bits) {
  return bits < 0x8000u ? bits ^ 0x7fffu : bits;
}
template <class Score>
__global__ void prepare(const Score* scores, int64_t stride, Output a) {
  const int row = blockIdx.x, t = threadIdx.x;
  const Score* input = scores + int64_t(row) * stride;
  __shared__ unsigned hist[4 * 256], pivot, before, written;
  if (t == 0) {
    a.counts[row] = a.status[row] = 0;
    a.tau[row] = -INFINITY;
    written = 0;
  }
  bool bad = false;
  for (int j = t; j < Seed; j += Threads) {
    const uint32_t bits = score_bits(input[j]);
    bad |= (bits & 0x7f800000u) == 0x7f800000u || (bits & 0xffffu) != 0;
  }
  if (__syncthreads_or(bad)) {
    if (t == 0) {
      atomicOr(a.error + row, 1);
      a.status[row] = 3;
    }
    return;
  }
  auto key = [](Score value) {
    const unsigned b = score_bits(value) >> 16;
    return ordered16((b & 0x7fffu) ? b : 0u);
  };
  for (int pass = 0; pass < 2; ++pass) {
    for (int j = t; j < 4 * 256; j += Threads) hist[j] = 0;
    __syncthreads();
    const int shift = pass == 0 ? 8 : 0;
    for (int j = t; j < Seed; j += Threads) {
      const unsigned code = key(input[j]);
      if (pass == 0 || (code >> 8) == (pivot >> 8))
        atomicAdd(hist + (t / 64) * 256 + ((code >> shift) & 255u), 1u);
    }
    __syncthreads();
    const unsigned count =
        hist[t] + hist[256 + t] + hist[512 + t] + hist[768 + t];
    __syncthreads();
    hist[t] = count;
    __syncthreads();
    if (t < 32) {
      const unsigned previous = pass == 0 ? 0u : before;
      const auto boundary =
          record_selection::find_boundary<8>(hist, Top - previous);
      if (t == 0) {
        pivot = (pass ? pivot : 0u) | (unsigned(boundary.bin) << shift);
        before = previous + unsigned(boundary.before);
      }
    }
    __syncthreads();
  }
  if (t == 0) hist[0] = 0;
  __syncthreads();
  const unsigned needed = Top - before;
  // Seed ties affect only the witness subset: every chosen score is >=tau.
  for (int j = t; j < Seed; j += Threads) {
    const unsigned code = key(input[j]);
    const bool chosen =
        code < pivot || (code == pivot && atomicAdd(hist, 1u) < needed);
    if (!chosen) continue;
    const unsigned slot = atomicAdd(&written, 1u);
    if (slot < Top) {
      const unsigned record = (code << 4) | (code < 0x8000u ? 15u : 0u);
      a.low[int64_t(row) * a.low_stride + slot] = uint8_t(record);
      a.words[int64_t(row) * a.word_stride + slot] =
          int32_t(record5::pack_word(record, unsigned(j)));
    }
  }
  __syncthreads();
  if (t == 0) {
    a.tau[row] = __uint_as_float(ordered16(pivot) << 16);
    a.counts[row] = int(written);
    if (written != Top) atomicOr(a.error + row, 1);
  }
}
}  // namespace native_seed

void prepare5(Tensor scores, Tensor low, Tensor words, Tensor counts,
              Tensor tau, Tensor error, Tensor status) {
  DeviceGuard guard(scores.get_device_index());
  STD_TORCH_CHECK(record5::checked_capacity(low, words) >= native_seed::Top &&
                      std::min(low.size(0), words.size(0)) >= scores.size(0),
                  "Record5 seed outputs must hold 2048 entries per input row");
  native_seed::Output output{low.mutable_data_ptr<uint8_t>(),
                             words.mutable_data_ptr<int>(),
                             counts.mutable_data_ptr<int>(),
                             tau.mutable_data_ptr<float>(),
                             error.mutable_data_ptr<int>(),
                             status.mutable_data_ptr<int>(),
                             low.stride(0),
                             words.stride(0)};
  if (scores.scalar_type() == torch::headeronly::ScalarType::BFloat16)
    launch<native_seed::prepare<__nv_bfloat16>, 256>(
        scores.size(0), static_cast<const __nv_bfloat16*>(scores.data_ptr()),
        scores.stride(0), output);
  else
    launch<native_seed::prepare<float>, 256>(scores.size(0),
                                             scores.const_data_ptr<float>(),
                                             scores.stride(0), output);
}
__global__ void set_if_any(cudaGraphConditionalHandle handle,
                           const int32_t* values, int count, int inactive) {
  bool active = false;
  for (int i = threadIdx.x; i < count; i += blockDim.x)
    active |= values[i] != inactive;
  const int any = __syncthreads_or(active);
  if (threadIdx.x == 0) cudaGraphSetConditional(handle, any != 0);
}

// Parent retains immutable child topology and tensors through completion.
void attach_if_any(Tensor values, int64_t inactive, int64_t body_address) {
  STD_TORCH_CHECK(body_address != 0,
                  "conditional body must name a live child graph");
  DeviceGuard guard(values.get_device_index());
  cudaStream_t stream = get_current_cuda_stream();
  cudaStreamCaptureStatus status;
  cudaGraph_t graph;
  DOSA_CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, nullptr, &graph,
                                           nullptr, nullptr, nullptr));
  STD_TORCH_CHECK(status == cudaStreamCaptureStatusActive,
                  "conditional attachment requires active stream capture");
  cudaGraphConditionalHandle handle;
  DOSA_CUDA_CHECK(cudaGraphConditionalHandleCreate(&handle, graph, 0,
                                                   cudaGraphCondAssignDefault));
  launch<set_if_any, 256>(1, handle, values.const_data_ptr<int32_t>(),
                          int(values.numel()), int(inactive));
  const cudaGraphNode_t* dependencies;
  const cudaGraphEdgeData* edges;
  size_t count;
  DOSA_CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, nullptr, &graph,
                                           &dependencies, &edges, &count));
  cudaGraphNodeParams params{};
  params.type = cudaGraphNodeTypeConditional;
  params.conditional.handle = handle;
  params.conditional.type = cudaGraphCondTypeIf;
  params.conditional.size = 1;
  cudaGraphNode_t conditional, child;
  DOSA_CUDA_CHECK(cudaGraphAddNode(&conditional, graph, dependencies, edges,
                                   count, &params));
  DOSA_CUDA_CHECK(cudaGraphAddChildGraphNode(
      &child, params.conditional.phGraph_out[0], nullptr, 0,
      reinterpret_cast<cudaGraph_t>(body_address)));
  DOSA_CUDA_CHECK(cudaStreamUpdateCaptureDependencies(
      stream, &conditional, nullptr, 1, cudaStreamSetCaptureDependencies));
}

}  // namespace DOSA_NAMESPACE

DOSA_REGISTER(
    "select_dense5",
    "(Tensor low, Tensor! packed, Tensor counts, Tensor starts, Tensor ends, "
    "Tensor! error, Tensor! output, Tensor order, Tensor permutation, Tensor! "
    "partition, int nk) -> ()",
    CUDA, select_dense5)
DOSA_REGISTER("prepare5",
              "(Tensor scores, Tensor! low, Tensor! words, Tensor! counts, "
              "Tensor! tau, Tensor! error, Tensor! status) -> ()",
              CUDA, prepare5)
DOSA_REGISTER("attach_if_any",
              "(Tensor values, int inactive, int body_address) -> ()", CUDA,
              attach_if_any)
