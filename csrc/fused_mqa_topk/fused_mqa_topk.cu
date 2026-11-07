// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Fused FP8 MQA-logits + top-k (k = 512, 1024, 2048) prefill selection for
// 32 or 64 index heads (SM100 family), LiteTopK sample -> filter -> select
// without materializing the [rows, keys] logits:
//   1. candidate producer (fp8_indexer.cuh, persistent, one CTA per SM): per
//      block of 256 / heads rows it scores a strided key sample, takes a
//      per-row threshold from it, then scores every visible key and appends
//      the (score, key) pairs above the threshold to the row's candidate slab;
//   2. candidate selector (topk_candidates_fp32.cuh, one CTA per row): the
//      exact fp32 top-k of each slab;
//   3. recovery (fp8_recovery_kernel): rows whose slab overflowed or holds
//      fewer than min(k, visible) candidates are rescored densely into a
//      bounded scratch pool and selected exactly; other blocks exit at once.
// The kernels are adapted from SGLang PR #47469 (see fp8_indexer.cuh).
// This file is the libtorch-stable host wrapper (and holds the selector
// kernels; the producer and recovery are in fp8_indexer.cu). All scratch
// memory is passed in by the caller and nothing is read back on the host, so
// the op is CUDA-graph friendly.
//
// Acknowledgement: LiteTopK (arXiv:2607.11976) was the first to fuse
// sparse-attention indexer scoring with top-k selection without materializing
// the full logits matrix, using classic sample-based threshold filtering.

#include "../torch_utils.h"

#include <torch/headeronly/core/ScalarType.h>

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <mutex>
#include <set>
#include <utility>

#include "fp8_indexer.cuh"
#include "topk_candidates_fp32.cuh"

namespace {

using fused_mqa_topk::device::fp8_indexer::Fp8Impl;
using torch::headeronly::ScalarType;
using torch::stable::Tensor;

// Most gathered keys per call: the threshold sample strides are sized for it.
constexpr int64_t kMaxKeys = int64_t(1) << 20;

void check_cuda(cudaError_t err) {
  STD_TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
}

// cuTensorMapEncodeTiled from the driver, resolved once through the runtime
// (thread-safe static initialisation; no link-time libcuda dependency).
PFN_cuTensorMapEncodeTiled_v12000 encode_tiled() {
  static const PFN_cuTensorMapEncodeTiled_v12000 fn = [] {
    void* ptr = nullptr;
    cudaDriverEntryPointQueryResult status;
    const cudaError_t err = cudaGetDriverEntryPointByVersion(
        "cuTensorMapEncodeTiled", &ptr, 12000, cudaEnableDefault, &status);
    STD_TORCH_CHECK(err == cudaSuccess && status == cudaDriverEntryPointSuccess,
                    "failed to resolve cuTensorMapEncodeTiled");
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(ptr);
  }();
  return fn;
}

// A tiled tensor map: dims and box innermost first, byte strides of dims
// 1..rank-1.
CUtensorMap make_map(const void* base, CUtensorMapDataType dtype,
                     std::initializer_list<cuuint64_t> dims,
                     std::initializer_list<cuuint64_t> strides,
                     std::initializer_list<cuuint32_t> box,
                     CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE) {
  static constexpr cuuint32_t kUnit[5] = {1, 1, 1, 1, 1};
  CUtensorMap map{};
  const CUresult r = encode_tiled()(
      &map, dtype, static_cast<cuuint32_t>(dims.size()),
      const_cast<void*>(base), dims.begin(), strides.begin(), box.begin(),
      kUnit, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  STD_TORCH_CHECK(r == CUDA_SUCCESS,
                  "cuTensorMapEncodeTiled failed: ", static_cast<int>(r));
  return map;
}

void check_tensor(const Tensor& t, int32_t device, ScalarType dtype,
                  const char* name) {
  STD_TORCH_CHECK(t.is_cuda() && t.get_device_index() == device, name,
                  " must be on the same CUDA device as q");
  STD_TORCH_CHECK(t.scalar_type() == dtype, name, " has the wrong dtype");
  STD_TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

// TMA tensor maps and bulk copies need 16-byte aligned base addresses.
void check_aligned(const Tensor& t, const char* name) {
  STD_TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, name,
                  " must be 16-byte aligned");
}

// Opts `kernel` into `smem` bytes of dynamic shared memory on the current
// device (once per device and kernel).
void set_smem(const void* kernel, size_t smem) {
  const cudaDeviceProp* prop = get_device_prop();
  STD_TORCH_CHECK(smem <= prop->sharedMemPerBlockOptin,
                  "fused_mqa_topk_prefill needs more shared memory per block "
                  "than this GPU allows");
  int device = 0;
  check_cuda(cudaGetDevice(&device));
  static std::mutex mutex;
  static std::set<std::pair<int, const void*>> done;
  const std::lock_guard<std::mutex> lock(mutex);
  if (done.count({device, kernel}) == 0) {
    check_cuda(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem)));
    done.insert({device, kernel});
  }
}

template <typename... Params, typename... Args>
void launch(void (*kernel)(Params...), uint32_t grid, uint32_t block,
            size_t smem, cudaStream_t stream, bool pdl, Args&&... args) {
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = smem;
  config.stream = stream;
  config.attrs = pdl ? &attr : nullptr;
  config.numAttrs = pdl ? 1 : 0;
  check_cuda(cudaLaunchKernelEx(&config, kernel, std::forward<Args>(args)...));
}

// Scan pieces per block of the last, partial wave of the producer: the split
// that minimizes the modelled duration (a piece costs ~10 tile scans extra).
uint32_t tail_chunks(int64_t blocks, int64_t sms, int64_t tiles) {
  const double cost = 10.0 / double(std::max<int64_t>(tiles, 1));
  uint32_t best = 1;
  double best_time = 0.0;
  for (const uint32_t chunks : {1u, 2u, 4u, 8u, 16u, 32u}) {
    const int64_t grid = std::min<int64_t>(sms, blocks * chunks);
    const int64_t tail = blocks % grid;
    const int64_t units = (tail * chunks + grid - 1) / grid;
    const double time =
        double(blocks / grid) * (1.0 + cost) +
        (tail ? double(units) / chunks + double(units) * cost : 0.0);
    if (chunks == 1 || time < best_time) best = chunks, best_time = time;
  }
  return best;
}

// The candidate selector for top-`kTopK` over `rows` candidate slabs.
template <uint32_t kTopK>
void select_topk(const Tensor& values, const Tensor& indices,
                 const int32_t* counts, const float* thresholds,
                 const Tensor& out, cudaStream_t stream) {
  using Impl = fused_mqa_topk::TopKCandidatesFP32Impl<kTopK>;
  const int64_t rows = values.size(0), cap = values.size(1);
  const fused_mqa_topk::TopKCandidatesFP32Params p{
      .scores = static_cast<const float*>(values.data_ptr()),
      .indices = static_cast<const int32_t*>(indices.data_ptr()),
      .count = counts,
      .thr = thresholds,
      .out = static_cast<int32_t*>(out.data_ptr()),
      .cap = static_cast<uint32_t>(cap),
  };
  const auto kernel = fused_mqa_topk::topk_candidates_fp32_kernel<Impl>;
  set_smem(reinterpret_cast<const void*>(kernel), Impl::kSmemBytes);
  // PDL: the selector's prologue overlaps the producer's tail.
  launch(kernel, static_cast<uint32_t>(rows), Impl::kBlockSize,
         Impl::kSmemBytes, stream, true, p);
}

template <uint32_t kHeads>
void run(const Tensor& q, const Tensor& k, const Tensor& scales,
         const Tensor& weights, const Tensor& starts, const Tensor& ends,
         const Tensor& out, const Tensor& counter, const Tensor& codes,
         const Tensor& values, const Tensor& indices, const Tensor& row_state,
         const Tensor& recovery) {
  using Impl = Fp8Impl<kHeads, false>;
  using RecoveryImpl = Fp8Impl<kHeads, true>;
  static_assert(Impl::kSmemBytes == RecoveryImpl::kSmemBytes &&
                Impl::kBlockQ == RecoveryImpl::kBlockQ);
  constexpr uint32_t kHeadDim = Impl::kHeadDim;
  const int64_t rows = q.size(0), keys = k.size(0), topk = out.size(1);
  const int64_t blocks = (rows + Impl::kBlockQ - 1) / Impl::kBlockQ;
  STD_TORCH_CHECK(counter.numel() >= 2 * blocks,
                  "counter must hold 2 * ceil(rows / ", Impl::kBlockQ,
                  ") zeros");
  STD_TORCH_CHECK(recovery.numel() >= int64_t(Impl::kBlockQ) * keys,
                  "recovery must hold at least ", Impl::kBlockQ,
                  " * keys floats (one block of rows)");

  const int sms = get_device_prop()->multiProcessorCount;
  // One CTA per SM at most. A scan piece waits for its block's thresholds,
  // published by its own CTA or a lower one (see unit_at), so the grid does not
  // need to be co-resident.
  const uint32_t chunks = tail_chunks(blocks, sms, keys / Impl::kSplitKV);
  const auto grid =
      static_cast<uint32_t>(std::min<int64_t>(sms, blocks * chunks));
  STD_TORCH_CHECK(
      codes.numel() >= 2 * int64_t(grid) * Impl::kCodesPerCta,
      "codes must hold ", 2 * Impl::kCodesPerCta,
      " int32 per producer CTA (at most min(SMs, 32 * row blocks))");

  // Q: rows * heads rows of 128 bytes, a block's per box; KV: keys rows of 128
  // bytes, a tile's 256 per box (both 128B-swizzled); weights: a block's rows;
  // per-key scales: a tile's 256 as one row.
  fused_mqa_topk::Fp8IndexerTensorMaps tmaps{};
  tmaps.q = make_map(q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8,
                     {kHeadDim, uint64_t(rows) * kHeads}, {kHeadDim},
                     {kHeadDim, Impl::kQTmaRows}, CU_TENSOR_MAP_SWIZZLE_128B);
  tmaps.weights = make_map(weights.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                           {kHeads, uint64_t(rows)}, {kHeads * 4ull},
                           {kHeads, Impl::kBlockQ});
  tmaps.kv = make_map(k.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8,
                      {kHeadDim, uint64_t(keys)}, {kHeadDim},
                      {kHeadDim, Impl::kSplitKV}, CU_TENSOR_MAP_SWIZZLE_128B);
  // Threshold samples: KV as [keys / S, S, 128] (key = row * S + phase), one
  // map per sample stride S.
  for (uint32_t m = 0; m < 2; ++m) {
    const uint64_t S = Impl::sample_stride(m);
    tmaps.kv_strided[m] =
        make_map(k.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8,
                 {kHeadDim, S, std::max<uint64_t>(keys / S, 1)},
                 {kHeadDim, S * kHeadDim}, {kHeadDim, 1, Impl::kSplitKV},
                 CU_TENSOR_MAP_SWIZZLE_128B);
  }
  tmaps.sf_kv = make_map(
      scales.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, {uint64_t(keys), 1},
      {uint64_t((keys * 4 + 15) / 16 * 16)}, {Impl::kSplitKV, 1});

  auto* counts = static_cast<int32_t*>(row_state.data_ptr());
  auto* thresholds = reinterpret_cast<float*>(counts + rows);
  fused_mqa_topk::Fp8IndexerParams p{};
  p.ks = static_cast<const int32_t*>(starts.data_ptr());
  p.ke = static_cast<const int32_t*>(ends.data_ptr());
  p.kv_scale = static_cast<const float*>(scales.data_ptr());
  p.block_ready = static_cast<uint32_t*>(counter.data_ptr());
  p.block_done = p.block_ready + blocks;
  p.sample_codes = static_cast<uint2*>(codes.data_ptr());
  p.cand_scores = static_cast<float*>(values.data_ptr());
  p.cand_indices = static_cast<int32_t*>(indices.data_ptr());
  p.cand_count = counts;
  p.thresholds = thresholds;
  p.overflow = counts + 2 * rows;
  p.num_rows = static_cast<uint32_t>(rows);
  p.num_blocks = static_cast<uint32_t>(blocks);
  p.num_keys = static_cast<uint32_t>(keys);
  for (uint32_t m = 0; m < 2; ++m) {
    p.sample_limit[m] = static_cast<uint32_t>(keys / Impl::sample_stride(m) *
                                              Impl::sample_stride(m));
  }
  p.cap = static_cast<uint32_t>(values.size(1));
  p.topk = static_cast<uint32_t>(topk);
  // Rows that see at most top-k keys keep them all (threshold -inf).
  p.dense_bound = static_cast<uint32_t>(topk);
  p.tail_chunks = chunks;

  const cudaStream_t stream = get_current_cuda_stream(q.get_device_index());
  const auto producer = fused_mqa_topk::fp8_kernel(kHeads, false);
  set_smem(reinterpret_cast<const void*>(producer), Impl::kSmemBytes);
  launch(producer, grid, Impl::kNumThreads, Impl::kSmemBytes, stream, false, p,
         tmaps);

  switch (topk) {
    case 512:
      select_topk<512>(values, indices, counts, thresholds, out, stream);
      break;
    case 1024:
      select_topk<1024>(values, indices, counts, thresholds, out, stream);
      break;
    default:
      select_topk<2048>(values, indices, counts, thresholds, out, stream);
  }

  // The recovery: one CTA per block; blocks without a bad row exit at once.
  // A bad block borrows one of min(32, pool / (block rows * keys)) slots of the
  // pool for its dense scores.
  p.recovery_scores = static_cast<float*>(recovery.data_ptr());
  p.recovery_out = static_cast<int32_t*>(out.data_ptr());
  p.recovery_weights = static_cast<const float*>(weights.data_ptr());
  p.recovery_stride = keys;
  p.recovery_slots = static_cast<uint32_t>(std::min<int64_t>(
      32, recovery.numel() / (int64_t(Impl::kBlockQ) * keys)));
  const auto recover = fused_mqa_topk::fp8_kernel(kHeads, true);
  set_smem(reinterpret_cast<const void*>(recover), RecoveryImpl::kSmemBytes);
  launch(recover, static_cast<uint32_t>(blocks), RecoveryImpl::kNumThreads,
         RecoveryImpl::kSmemBytes, stream, false, p, tmaps);
}

}  // namespace

// Inputs: q [rows, H, 128] fp8e4m3 with H (the index heads) 32 or 64,
// k [keys, 128] fp8e4m3, scales [keys] fp32, weights [rows, H] fp32,
// starts/ends [rows] int32 (the visible key range [start, end) of each row),
// with rows >= 1, rows * H < 2^31 and 1 <= keys <= 1048576. starts/ends are
// clamped to [0, keys]; an end before its start means an empty row.
//
// Output: out [rows, K] int32 with K (the top-k) 512, 1024 or 2048: key
// indices relative to the row's start (key - start, the top_k_per_row_prefill
// convention), unordered, -1 padded when fewer than K keys are visible. Exact
// on the kernels' fp32 scores up to ties at the K-th score. The thresholds are
// fp16 distances to a sampled score of each row, so a common offset of the
// scores does not matter; rows whose top scores spread over more than 65504
// (or are packed within 2^-24) may overflow and take the recovery: exact, but
// seconds at 1M keys.
//
// All tensors are contiguous; q, k, scales, weights, out, codes, values and
// indices are 16-byte aligned (TMA / bulk copies / vector stores).
//
// Workspace (caller-owned), with B = ceil(rows / (256 / H)) row blocks and
// G <= min(SMs, 32 B) the producer's grid:
//   counter [>= 2B] int32: zero between calls; the kernels leave it zeroed, so
//     a workspace must not be shared by launches that can run concurrently
//     (e.g. on two streams),
//   codes [>= 32768 G] int32: the producer's per-CTA sample codes,
//   values [rows, C] fp32 and indices [rows, C] int32: the candidate slab, C in
//     [4096, 12288] and a multiple of 4 (a row with more candidates is
//     recovered),
//   row_state [3, rows] int32: candidate counts, thresholds (fp32 bits),
//     overflow flags,
//   recovery [>= (256 / H) * keys] fp32: the recovery's dense score pool, used
//     as min(32, numel / ((256 / H) * keys)) slots.
void fused_mqa_topk_prefill(
    torch::stable::Tensor q, torch::stable::Tensor k,
    torch::stable::Tensor scales, torch::stable::Tensor weights,
    torch::stable::Tensor starts, torch::stable::Tensor ends,
    torch::stable::Tensor out, torch::stable::Tensor counter,
    torch::stable::Tensor codes, torch::stable::Tensor values,
    torch::stable::Tensor indices, torch::stable::Tensor row_state,
    torch::stable::Tensor recovery) {
  STD_TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
  const int32_t dev = q.get_device_index();
  check_tensor(q, dev, ScalarType::Float8_e4m3fn, "q");
  check_tensor(k, dev, ScalarType::Float8_e4m3fn, "k");
  check_tensor(scales, dev, ScalarType::Float, "scales");
  check_tensor(weights, dev, ScalarType::Float, "weights");
  check_tensor(starts, dev, ScalarType::Int, "starts");
  check_tensor(ends, dev, ScalarType::Int, "ends");
  check_tensor(out, dev, ScalarType::Int, "out");
  check_tensor(counter, dev, ScalarType::Int, "counter");
  check_tensor(codes, dev, ScalarType::Int, "codes");
  check_tensor(values, dev, ScalarType::Float, "values");
  check_tensor(indices, dev, ScalarType::Int, "indices");
  check_tensor(row_state, dev, ScalarType::Int, "row_state");
  check_tensor(recovery, dev, ScalarType::Float, "recovery");

  STD_TORCH_CHECK(
      q.dim() == 3 && (q.size(1) == 32 || q.size(1) == 64) && q.size(2) == 128,
      "q must be [rows, 32 or 64, 128]");
  STD_TORCH_CHECK(k.dim() == 2 && k.size(1) == 128, "k must be [keys, 128]");
  const int64_t rows = q.size(0), heads = q.size(1), keys = k.size(0);
  // TMA row coordinates (rows * heads Q rows) are int32.
  STD_TORCH_CHECK(rows > 0 && rows * heads <= INT32_MAX,
                  "rows must be in [1, 2^31 / heads)");
  STD_TORCH_CHECK(keys > 0 && keys <= kMaxKeys, "keys must be in [1, 1048576]");
  STD_TORCH_CHECK(scales.numel() == keys, "scales must be [keys]");
  STD_TORCH_CHECK(
      weights.dim() == 2 && weights.size(0) == rows && weights.size(1) == heads,
      "weights must be [rows, heads]");
  STD_TORCH_CHECK(starts.numel() == rows && ends.numel() == rows,
                  "starts/ends must be [rows]");
  STD_TORCH_CHECK(
      out.dim() == 2 && out.size(0) == rows &&
          (out.size(1) == 512 || out.size(1) == 1024 || out.size(1) == 2048),
      "out must be [rows, K] with K in {512, 1024, 2048}");
  STD_TORCH_CHECK(values.dim() == 2 && values.size(0) == rows &&
                      indices.dim() == 2 && indices.size(0) == rows &&
                      indices.size(1) == values.size(1),
                  "values/indices must be [rows, C]");
  const int64_t cap = values.size(1);
  STD_TORCH_CHECK(cap >= 4096 && cap <= 12288 && cap % 4 == 0,
                  "values/indices must be [rows, C] with C in [4096, 12288] "
                  "and a multiple of 4");
  STD_TORCH_CHECK(row_state.numel() == 3 * rows, "row_state must be [3, rows]");
  check_aligned(q, "q");
  check_aligned(k, "k");
  check_aligned(scales, "scales");
  check_aligned(weights, "weights");
  check_aligned(out, "out");
  check_aligned(codes, "codes");
  check_aligned(values, "values");
  check_aligned(indices, "indices");

  torch::stable::accelerator::DeviceGuard guard(dev);
  if (heads == 32) {
    run<32>(q, k, scales, weights, starts, ends, out, counter, codes, values,
            indices, row_state, recovery);
  } else {
    run<64>(q, k, scales, weights, starts, ends, out, counter, codes, values,
            indices, row_state, recovery);
  }
}
