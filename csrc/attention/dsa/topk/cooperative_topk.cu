// Cooperative cluster TopK for DeepSeek V3 sparse attention indexer.
// See cooperative_topk.cuh for the cluster path and cooperative_grid_topk.cuh
// for the grid path (a few long FP32 rows over the whole GPU).

#include <cuda_runtime.h>

#include <atomic>
#include <optional>
#include <type_traits>

#include "torch_utils.h"

#ifndef USE_ROCM
  #include "cooperative_grid_dyn.cuh"
  #include "cooperative_grid_topk.cuh"
  #include "cooperative_topk.cuh"
namespace ct = vllm::cooperative;
namespace grid = vllm::cooperative_grid;
namespace hist4096 = vllm::topk_histogram_4096;
#endif

#ifndef USE_ROCM
template <uint32_t TopK, uint32_t CS, typename InT>
void launch_cooperative_cluster(ct::CooperativeTopKParams<TopK, InT>& params,
                                size_t smem, cudaStream_t stream) {
  auto kernel = []() {
    if constexpr (CS == 2) {
      return &ct::cooperative_topk_cs2<TopK, InT>;
    } else if constexpr (CS == 16) {
      return &ct::cooperative_topk_cs16<TopK, InT>;
    } else if constexpr (CS == 8) {
      return &ct::cooperative_topk_cs8<TopK, InT>;
    } else {
      static_assert(CS == 4, "unsupported cooperative_topk cluster size");
      return &ct::cooperative_topk_cs4<TopK, InT>;
    }
  }();
  if constexpr (CS > 8) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed,
                         1);
  }
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       smem);

  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(params.num_rows, CS);
  cfg.blockDim = dim3(hist4096::kBlockSize);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim = {1, CS, 1};
  cfg.numAttrs = 1;
  cfg.attrs = attrs;
  cudaError_t err = cudaLaunchKernelEx(&cfg, kernel, params);
  STD_TORCH_CHECK(err == cudaSuccess,
                  "cooperative_topk launch failed: ", cudaGetErrorString(err));
}

namespace {
// Co-resident 8-CTA clusters per (device, TopK, dtype), cached as count + 1
// (0 = not queried). Internal linkage keeps the cache private to this .so.
constexpr int kMaxCs8Devices = 64;
std::atomic<int> cs8_max_clusters_cache[kMaxCs8Devices][3][2];

template <uint32_t TopK>
constexpr int topk_slot() {
  return TopK == 512 ? 0 : (TopK == 1024 ? 1 : 2);
}
}  // namespace

// True if num_rows 8-CTA clusters fit on the device at once, so the CS8
// kernel runs in a single wave.
template <uint32_t TopK, typename InT>
bool cs8_clusters_fit(int64_t num_rows) {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess || device < 0 ||
      device >= kMaxCs8Devices) {
    return false;
  }
  std::atomic<int>& slot =
      cs8_max_clusters_cache[device][topk_slot<TopK>()]
                            [std::is_same_v<InT, float> ? 1 : 0];
  int cached = slot.load(std::memory_order_relaxed);
  if (cached == 0) {
    auto kernel = &ct::cooperative_topk_cs8<TopK, InT>;
    constexpr size_t smem = ct::kSmemSize8<InT>;
    int max_clusters = 0;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(1, 8);
    cfg.blockDim = dim3(hist4096::kBlockSize);
    cfg.dynamicSmemBytes = smem;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim = {1, 8, 1};
    cfg.numAttrs = 1;
    cfg.attrs = attrs;
    if (cudaFuncSetAttribute(kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem) != cudaSuccess ||
        cudaOccupancyMaxActiveClusters(&max_clusters, kernel, &cfg) !=
            cudaSuccess) {
      (void)cudaGetLastError();  // Clear the error; fall back to CS4.
      max_clusters = 0;
    }
    cached = max_clusters + 1;
    slot.store(cached, std::memory_order_relaxed);
  }
  return num_rows <= cached - 1;
}

namespace {

constexpr int kGridMaxDevices = 64;

// Co-resident clusters of `kernel` launched with `cfg` (gridDim: one
// cluster), queried once per device and cached in `cache` (0: not queried,
// -1: none fit or the query failed).
int max_active_clusters(std::atomic<int>* cache, const void* kernel,
                        const cudaLaunchConfig_t& cfg) {
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= kGridMaxDevices) {
    (void)cudaGetLastError();
    return -1;
  }
  int n = cache[dev].load(std::memory_order_relaxed);
  if (n == 0) {
    if (cudaOccupancyMaxActiveClusters(&n, kernel, &cfg) != cudaSuccess ||
        n <= 0) {
      (void)cudaGetLastError();
      n = -1;
    }
    cache[dev].store(n, std::memory_order_relaxed);
  }
  return n;
}

// ---------------------------------------------------------------------------
// Grid launch: a few FP32 rows in one cooperative launch, long rows on the
// grid path (cooperative_grid_topk.cuh), short rows on the cluster path,
// chosen on the device from the row lengths.
// ---------------------------------------------------------------------------

// Per row count, rows at least this long take the grid path and shorter ones
// the cluster path (in clusters of grid_cluster_size(rows) CTAs): the measured
// crossover on B300 (more long rows share the CTAs, so the grid path pays off
// later). kNever: the grid path never wins at that row count, so the call
// takes the cluster launch:
//  - 14-15 rows: the 8-CTA cluster launch is as fast as the grid path.
//  - 23-33 rows: the dynamic-split launch instead (below).
//  - 50-64 rows: a long row would get only 2 CTAs.
// From 34 rows the cluster launch drops to 2-CTA clusters and leaves half the
// SMs idle, so the grid launch returns, a long row getting 3-4 CTAs.
constexpr uint32_t kGridMaxRows = 64;  // threshold table size
static_assert(kGridMaxRows <= grid::kMaxRows);
constexpr uint32_t kNever = 0xFFFFFFFFu;
// clang-format off
constexpr uint32_t kGridFrom[] = {
    458752,                          // 1 row, 16-CTA clusters
    524289, 524289, 524289,          // 2-4 rows: above the cluster path's
                                     // single-pass length
    // 5-8 rows, 8-CTA clusters: above 512K, where the grid path lost 0.9% of
    // GLM-5.3 TPOT at 8 requests
    655360, 655360, 655360, 655360,
    393216, 393216, 393216,          // 9-11 rows
    524288, 524288,                  // 12-13 rows
    kNever, kNever,                  // 14-15 rows
    // 16-22 rows, 4-CTA clusters
    458752, 655360, 589824,          // 16-18 rows
    786432, 786432,                  // 19-20 rows
    786432, 786432,                  // 21-22 rows
    // 23-33 rows
    kNever, kNever, kNever, kNever, kNever, kNever, kNever, kNever, kNever,
    kNever, kNever,
    // 34-49 rows, 2-CTA clusters
    262144, 262144, 262144, 262144,                  // 34-37: 4 CTAs a row
    655360, 655360, 655360, 655360, 655360, 655360,  // 38-43: 3 CTAs
    655360, 655360, 655360, 655360, 655360, 655360,  // 44-49
    // 50-64 rows
    kNever, kNever, kNever, kNever, kNever, kNever, kNever, kNever, kNever,
    kNever, kNever, kNever, kNever, kNever, kNever};
// clang-format on
static_assert(sizeof(kGridFrom) / sizeof(kGridFrom[0]) == kGridMaxRows);
constexpr bool grid_from_valid() {
  for (const uint32_t t : kGridFrom) {
    if (t < grid::kMinRowLength) return false;
  }
  return true;
}
static_assert(grid_from_valid(), "a grid threshold is below kMinRowLength");

// Cluster size for a row count (16 needs SM100).
constexpr uint32_t grid_cluster_size(uint32_t rows) {
  return rows <= 4 ? 16 : rows <= 15 ? 8 : rows <= 33 ? 4 : 2;
}

// Position of the n-th (from 0) set bit of m; m has more than n set bits.
__device__ __forceinline__ uint32_t nth_set_bit(unsigned long long m,
                                                uint32_t n) {
  const uint32_t lo = static_cast<uint32_t>(m),
                 hi = static_cast<uint32_t>(m >> 32);
  const uint32_t c = __popc(lo);
  return n < c ? __fns(lo, 0, n + 1) : 32 + __fns(hi, 0, n - c + 1);
}

// A pool of gridDim.y CTAs (grid (1, CS * C), clusters of CS along y, every
// CTA resident; the launcher requires C >= rows). Cluster r owns row r, as in
// the cluster launch: its CTAs read the row's length themselves and, for a
// short row (< grid_from), start the cluster algorithm at once. Only the CTAs
// of a long row's cluster and the spare CTAs (clusters past the rows) read all
// the lengths (one coalesced load by warp 0 and two ballots): a long row takes
// its own cluster plus an even share of the spare CTAs, regardless of cluster
// boundaries (leftover CTAs exit). Reading every length before the short rows
// started cost them about 0.5 us per call.
template <uint32_t TopK, uint32_t CS>
__global__ void __launch_bounds__(hist4096::kBlockSize, 1)
    __cluster_dims__(1, CS, 1) cooperative_topk_fp32_grid_kernel(
        ct::CooperativeTopKParams<TopK, float> params, uint8_t* candidates,
        uint32_t* headers, uint32_t grid_from) {
  static_assert(kGridMaxRows == 64, "one warp reads two lengths per lane");
  const uint32_t rows = params.num_rows, cta = blockIdx.y, own = cta / CS;
  const auto is_short = [&](int32_t len) {
    return static_cast<uint32_t>(len > 0 ? len : 0) < grid_from;
  };
  const int32_t own_len = own < rows ? params.lengths[own] : 0;
  if (own >= rows || !is_short(own_len)) {
    __shared__ int32_t s_len[kGridMaxRows];
    __shared__ unsigned long long s_long;
    if (threadIdx.x < 32) {
      const uint32_t lane = threadIdx.x;
      const int32_t l0 = lane < rows ? params.lengths[lane] : 0;
      const int32_t l1 = lane + 32 < rows ? params.lengths[lane + 32] : 0;
      const uint32_t g0 =
          __ballot_sync(0xFFFFFFFFu, lane < rows && !is_short(l0));
      const uint32_t g1 =
          __ballot_sync(0xFFFFFFFFu, lane + 32 < rows && !is_short(l1));
      s_len[lane] = l0;
      s_len[lane + 32] = l1;
      if (lane == 0) {
        s_long = g0 | (static_cast<unsigned long long>(g1) << 32);
      }
    }
    __syncthreads();
    const unsigned long long long_mask = s_long;
    const uint32_t n_long = __popcll(long_mask);
    if (n_long == 0) return;
    const uint32_t spare = (gridDim.y - rows * CS) / n_long;
    uint32_t row, g;
    if (own < rows) {
      row = own;
      g = cta % CS;
    } else {
      const uint32_t s = cta - rows * CS, j = s / spare;
      if (j >= n_long) return;
      row = nth_set_bit(long_mask, j);
      g = CS + s % spare;
    }
    grid::row_topk<CS <= 4>(params.input, params.stride, s_len[row],
                            params.output, TopK, candidates, headers, row, g,
                            CS + spare);
    return;
  }
  // Single-pass stages off the float4 visitor here (see kSinglePassVec).
  ct::cooperative_topk_body<TopK, CS, float, false>(params, own, own_len);
}

// Dynamic shared memory of the grid kernel: the larger of the cluster path's
// (kSmemSize2 == kSmemSize4) and the grid path's.
template <uint32_t CS>
constexpr size_t kGridClusterSmem =
    CS <= 4 ? ct::kSmemSize4<float> : ct::kSmemSize8<float>;
template <uint32_t CS>
constexpr size_t kGridRowSmem =
    CS <= 4 ? grid::kDynSmemSamples : grid::kDynSmem;
template <uint32_t CS>
constexpr size_t kGridSmem =
    kGridClusterSmem<CS> > kGridRowSmem<CS> ? kGridClusterSmem<CS>
                                            : kGridRowSmem<CS>;

template <uint32_t TopK, uint32_t CS>
bool launch_grid_cs(ct::CooperativeTopKParams<TopK, float>& params,
                    uint8_t* state, cudaStream_t stream) {
  const auto kernel = &cooperative_topk_fp32_grid_kernel<TopK, CS>;
  constexpr size_t smem = kGridSmem<CS>;
  static std::atomic<int> cache[kGridMaxDevices];
  if constexpr (CS > 8) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed,
                         1);
  }
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       static_cast<int>(smem));
  // One cluster (its shape is the kernel's __cluster_dims__) for the
  // occupancy query, then the pool.
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(1, CS);
  cfg.blockDim = dim3(hist4096::kBlockSize);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  const int clusters =
      max_active_clusters(cache, reinterpret_cast<const void*>(kernel), cfg);
  // Every short row needs a cluster of its own; every CTA must be resident.
  if (clusters < static_cast<int>(params.num_rows)) return false;
  cfg.gridDim = dim3(1, CS * clusters);
  cudaLaunchAttribute attr;
  attr.id = cudaLaunchAttributeCooperative;
  attr.val.cooperative = 1;
  cfg.attrs = &attr;
  cfg.numAttrs = 1;
  const cudaError_t err = cudaLaunchKernelEx(
      &cfg, kernel, params, state + grid::kHeaderBytes,
      reinterpret_cast<uint32_t*>(state), kGridFrom[params.num_rows - 1]);
  STD_TORCH_CHECK(err == cudaSuccess, "cooperative_topk grid launch failed: ",
                  cudaGetErrorString(err));
  return true;
}

// ---------------------------------------------------------------------------
// Dynamic-split launch for 23-33 FP32 rows (cooperative_grid_dyn.cuh).
// ---------------------------------------------------------------------------

// 23-33 rows take 4-CTA clusters (all resident on B300) and leave no CTAs for
// the grid path; a few long rows set the time of the whole batch. Here every
// cluster owns its row as in the cluster launch, and rows at least kDynFrom
// long (at most kDynMaxLong of them) are split among the CTAs not busy with a
// short row: those of clusters past the rows at once, the short rows' when
// their row is done.
constexpr uint32_t kDynCS = 4, kDynRowsFrom = 23, kDynRowsTo = 33;
constexpr uint32_t kDynFrom = 524288, kDynMaxLong = 8;
static_assert(kDynFrom >= grid::kMinRowLength);

template <uint32_t TopK>
__global__ void __launch_bounds__(hist4096::kBlockSize, 1)
    __cluster_dims__(1, kDynCS, 1) cooperative_topk_fp32_dyn_kernel(
        ct::CooperativeTopKParams<TopK, float> params, uint8_t* candidates,
        uint32_t* dstate) {
  constexpr uint32_t CS = kDynCS;
  const uint32_t rows = params.num_rows, cta = blockIdx.y, row = cta / CS;
  const uint32_t lane = threadIdx.x % 32;
  const auto is_short = [](int32_t len) {
    return static_cast<uint32_t>(len > 0 ? len : 0) < kDynFrom;
  };
  // The lengths, read once by warp 0; the long-row mask by ballot.
  __shared__ int32_t s_len[kGridMaxRows];
  __shared__ unsigned long long s_long;
  if (threadIdx.x < 32) {
    const int32_t l0 = lane < rows ? params.lengths[lane] : 0;
    const int32_t l1 = lane + 32 < rows ? params.lengths[lane + 32] : 0;
    const uint32_t g0 =
        __ballot_sync(0xFFFFFFFFu, lane < rows && !is_short(l0));
    const uint32_t g1 =
        __ballot_sync(0xFFFFFFFFu, lane + 32 < rows && !is_short(l1));
    s_len[lane] = l0;
    s_len[lane + 32] = l1;
    if (lane == 0) s_long = g0 | (static_cast<unsigned long long>(g1) << 32);
  }
  __syncthreads();
  const unsigned long long long_mask = s_long;
  const uint32_t n_long = __popcll(long_mask);
  const bool split = n_long != 0 && n_long <= kDynMaxLong;
  const bool own_short = row < rows && is_short(s_len[row]);
  const bool own_long = row < rows && !is_short(s_len[row]);
  // A long row being split: its own cluster takes no part in the split, waits
  // for the row's verdict and redoes the row if the cutoff was misjudged.
  __shared__ uint32_t s_verdict;
  if (own_long && split) {
    if (threadIdx.x == 0) {
      const uint32_t* st = dstate + row * grid::kDynRowWords;
      uint32_t v;
      while ((v = grid::ld_acquire32(st + 8)) == 0) __nanosleep(64);
      s_verdict = v;
    }
    __syncthreads();
  }
  // One call site of the cluster algorithm (a second inlined copy, or one
  // before the lengths are read, made the compiler spill in it): a short row;
  // a long row without a split; a long row whose split cutoff was misjudged.
  if (own_short || (own_long && (!split || s_verdict == 2))) {
    ct::cooperative_topk_body<TopK, CS, float, false>(params, row, s_len[row]);
  }
  if (!split) return;
  if (!own_long) {
    const uint32_t start = cta % n_long;
    for (uint32_t k = 0; k < n_long; k++) {
      const uint32_t r = nth_set_bit(long_mask, (start + k) % n_long);
      grid::dyn_row_topk(params.input, params.stride, s_len[r], params.output,
                         TopK, candidates, dstate, r);
    }
  }
  // The grid's last CTA re-zeroes the long rows' state for the next launch.
  __syncthreads();
  uint32_t* exit_count = dstate + grid::kMaxRows * grid::kDynRowWords;
  __shared__ bool s_last;
  if (threadIdx.x == 0) {
    s_last = grid::add_acqrel32(exit_count, 1u) + 1 == gridDim.y;
  }
  __syncthreads();
  if (!s_last) return;
  for (uint32_t r = 0; r < rows; r++) {
    if (!((long_mask >> r) & 1ull)) continue;
    uint32_t* st = dstate + r * grid::kDynRowWords;
    for (uint32_t w = threadIdx.x; w < grid::kDynRowWords; w += blockDim.x) {
      st[w] = 0;
    }
  }
  if (threadIdx.x == 0) *exit_count = 0;
}

template <uint32_t TopK>
bool launch_dyn(ct::CooperativeTopKParams<TopK, float>& params, uint8_t* state,
                cudaStream_t stream) {
  const auto kernel = &cooperative_topk_fp32_dyn_kernel<TopK>;
  constexpr size_t smem = kGridSmem<kDynCS>;
  static std::atomic<int> cache[kGridMaxDevices];
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       static_cast<int>(smem));
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(1, kDynCS);
  cfg.blockDim = dim3(hist4096::kBlockSize);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  const int clusters =
      max_active_clusters(cache, reinterpret_cast<const void*>(kernel), cfg);
  // Every row needs a cluster of its own; every CTA must be resident.
  if (clusters < static_cast<int>(params.num_rows)) return false;
  cfg.gridDim = dim3(1, kDynCS * clusters);
  cudaLaunchAttribute attr;
  attr.id = cudaLaunchAttributeCooperative;
  attr.val.cooperative = 1;
  cfg.attrs = &attr;
  cfg.numAttrs = 1;
  const cudaError_t err = cudaLaunchKernelEx(
      &cfg, kernel, params, state + grid::kHeaderBytes,
      reinterpret_cast<uint32_t*>(state + grid::kHeaderBytes +
                                  grid::kWorkspaceBytes));
  STD_TORCH_CHECK(err == cudaSuccess, "cooperative_topk dyn launch failed: ",
                  cudaGetErrorString(err));
  return true;
}

// Launches the grid kernel if the row count and the device allow; returns
// false for the cluster launch otherwise.
template <uint32_t TopK>
bool launch_grid(ct::CooperativeTopKParams<TopK, float>& params, uint8_t* state,
                 cudaStream_t stream) {
  const uint32_t rows = params.num_rows;
  if (rows >= kDynRowsFrom && rows <= kDynRowsTo) {
    return launch_dyn<TopK>(params, state, stream);
  }
  if (rows == 0 || rows > kGridMaxRows || kGridFrom[rows - 1] == kNever) {
    return false;
  }
  switch (grid_cluster_size(rows)) {
    case 16:
      return launch_grid_cs<TopK, 16>(params, state, stream);
    case 8:
      return launch_grid_cs<TopK, 8>(params, state, stream);
    case 4:
      return launch_grid_cs<TopK, 4>(params, state, stream);
    default:
      return launch_grid_cs<TopK, 2>(params, state, stream);
  }
}

}  // namespace

template <uint32_t TopK, typename InT>
void launch_cooperative_topk_impl(
    const torch::stable::Tensor& logits, const torch::stable::Tensor& lengths,
    torch::stable::Tensor& output, torch::stable::Tensor& workspace,
    int64_t max_seq_len, std::optional<torch::stable::Tensor> const& state) {
  (void)max_seq_len;  // Kept for signature parity with persistent_topk.
  const int64_t num_rows = logits.size(0);
  const cudaStream_t stream = get_current_cuda_stream();

  const uint32_t stride = static_cast<uint32_t>(logits.stride(0));
  STD_TORCH_CHECK(
      num_rows <= 64,
      "cooperative_topk supports <=64 rows; use persistent_topk for "
      "larger batches");

  STD_TORCH_CHECK(stride % ct::kTmaAlign<InT> == 0,
                  "cooperative_topk: stride must be multiple of ",
                  ct::kTmaAlign<InT>,
                  " for TMA alignment, got stride (max_model_len)=", stride);

  STD_TORCH_CHECK(workspace.is_cuda(), "workspace must be CUDA tensor");
  STD_TORCH_CHECK(
      workspace.scalar_type() == torch::headeronly::ScalarType::Byte,
      "workspace must be uint8");

  ct::CooperativeTopKParams<TopK, InT> params;
  params.input = reinterpret_cast<const InT*>(logits.const_data_ptr());
  params.output = output.mutable_data_ptr<int32_t>();
  params.lengths = lengths.const_data_ptr<int32_t>();
  params.num_rows = static_cast<uint32_t>(num_rows);
  params.stride = stride;
  params.tie_ws =
      reinterpret_cast<hist4096::Tie*>(workspace.mutable_data_ptr<uint8_t>());

  constexpr uint32_t kTieWsPerRow = ct::kCoarseTieCapacity;
  STD_TORCH_CHECK(
      workspace.size(0) >=
          static_cast<int64_t>(num_rows * kTieWsPerRow * sizeof(hist4096::Tie)),
      "workspace too small");

  const bool supports_cluster16 = get_device_prop()->major >= 10;
  // FP32 rows with the grid path's state: one grid launch.
  if constexpr (std::is_same_v<InT, float>) {
    if (state.has_value() && supports_cluster16 && logits.stride(1) == 1 &&
        reinterpret_cast<uintptr_t>(logits.const_data_ptr()) % 16 == 0) {
      const torch::stable::Tensor& st = *state;
      STD_TORCH_CHECK(
          st.is_cuda() && st.is_contiguous() &&
              st.scalar_type() == torch::headeronly::ScalarType::Byte &&
              st.numel() >= static_cast<int64_t>(grid::kStateBytes),
          "cooperative_topk: state must be a contiguous CUDA uint8 tensor of "
          "at least ",
          grid::kStateBytes, " bytes");
      if (launch_grid<TopK>(params, st.mutable_data_ptr<uint8_t>(), stream)) {
        return;
      }
    }
  }
  if (num_rows <= 4 && supports_cluster16) {
    launch_cooperative_cluster<TopK, 16>(params, ct::kSmemSize8<InT>, stream);
  } else if (num_rows <= 8 ||
             (num_rows <= 33 && cs8_clusters_fit<TopK, InT>(num_rows))) {
    // Above 8 rows, CS8 still beats CS4 while every cluster is resident at
    // once (up to 15 rows on a 148-SM B300).
    launch_cooperative_cluster<TopK, 8>(params, ct::kSmemSize8<InT>, stream);
  } else if (num_rows <= 33) {
    launch_cooperative_cluster<TopK, 4>(params, ct::kSmemSize4<InT>, stream);
  } else {
    launch_cooperative_cluster<TopK, 2>(params, ct::kSmemSize2<InT>, stream);
  }
}

template <uint32_t TopK>
void launch_cooperative_topk_dtype(
    const torch::stable::Tensor& logits, const torch::stable::Tensor& lengths,
    torch::stable::Tensor& output, torch::stable::Tensor& workspace,
    int64_t max_seq_len, std::optional<torch::stable::Tensor> const& state) {
  if (logits.scalar_type() == torch::headeronly::ScalarType::BFloat16) {
    launch_cooperative_topk_impl<TopK, __nv_bfloat16>(
        logits, lengths, output, workspace, max_seq_len, state);
  } else {
    launch_cooperative_topk_impl<TopK, float>(logits, lengths, output,
                                              workspace, max_seq_len, state);
  }
}
#endif  // USE_ROCM

// state: optional persistent buffer (uint8, at least
// cooperative_grid::kStateBytes) for the grid launch of FP32 rows. The caller
// zeroes it once and keeps it across calls, CUDA graph replays included; calls
// that may run concurrently (other streams) need their own. Without it the
// call is the cluster launch.
void cooperative_topk(const torch::stable::Tensor& logits,
                      const torch::stable::Tensor& lengths,
                      torch::stable::Tensor& output,
                      torch::stable::Tensor& workspace, int64_t k,
                      int64_t max_seq_len,
                      std::optional<torch::stable::Tensor> const& state) {
#ifndef USE_ROCM
  STD_TORCH_CHECK(logits.is_cuda(), "logits must be CUDA tensor");
  STD_TORCH_CHECK(lengths.is_cuda(), "lengths must be CUDA tensor");
  STD_TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
  STD_TORCH_CHECK(
      logits.scalar_type() == torch::headeronly::ScalarType::Float ||
          logits.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "logits must be float32 or bfloat16");
  STD_TORCH_CHECK(lengths.scalar_type() == torch::headeronly::ScalarType::Int,
                  "lengths must be int32");
  STD_TORCH_CHECK(output.scalar_type() == torch::headeronly::ScalarType::Int,
                  "output must be int32");
  STD_TORCH_CHECK(logits.dim() == 2, "logits must be 2D");
  STD_TORCH_CHECK(lengths.dim() == 1 || lengths.dim() == 2,
                  "lengths must be 1D or 2D");
  STD_TORCH_CHECK(lengths.is_contiguous(), "lengths must be contiguous");
  STD_TORCH_CHECK(output.dim() == 2, "output must be 2D");
  const int64_t num_rows = logits.size(0);
  STD_TORCH_CHECK(lengths.numel() == num_rows, "lengths size mismatch");
  STD_TORCH_CHECK(output.size(0) == num_rows && output.size(1) == k,
                  "output size mismatch");
  STD_TORCH_CHECK(
      k == 512 || k == 1024 || k == 2048,
      "cooperative_topk supports k=512, k=1024, or k=2048, got k=", k);

  if (k == 512) {
    launch_cooperative_topk_dtype<512>(logits, lengths, output, workspace,
                                       max_seq_len, state);
  } else if (k == 1024) {
    launch_cooperative_topk_dtype<1024>(logits, lengths, output, workspace,
                                        max_seq_len, state);
  } else {
    launch_cooperative_topk_dtype<2048>(logits, lengths, output, workspace,
                                        max_seq_len, state);
  }
#else
  STD_TORCH_CHECK(false, "cooperative_topk is not supported on ROCm");
#endif
}
