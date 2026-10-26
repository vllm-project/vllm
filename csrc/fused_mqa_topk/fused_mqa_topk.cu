// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Single-launch fused FP8 MQA-logits + top-k (k = 512, 1024, 2048) prefill
// selection (SM100 family): one persistent kernel (persistent_kernel.cuh, one
// CTA per SM; sampled gate, candidate emission and selection, exact rescan of
// out-of-range blocks) scores every visible key of a row and selects its top-k
// keys without materializing the [rows, keys] logits matrix.
// This file is the libtorch-stable host wrapper. All scratch memory is passed
// in by the caller (no allocation here), so the op is CUDA-graph friendly.

#include "../torch_utils.h"

#include <torch/headeronly/core/ScalarType.h>

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <mutex>
#include <utility>

#include "constants.h"
#include "persistent_kernel.cuh"

namespace {

using torch::headeronly::ScalarType;

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

// SM count of `device`, queried once per device.
int sm_count(int device) {
  static std::mutex mutex;
  static std::map<int, int> cache;
  const std::lock_guard<std::mutex> lock(mutex);
  auto [it, inserted] = cache.try_emplace(device, 0);
  if (inserted) {
    const cudaError_t err = cudaDeviceGetAttribute(
        &it->second, cudaDevAttrMultiProcessorCount, device);
    if (err != cudaSuccess) cache.erase(it);
    check_cuda(err);
  }
  return it->second;
}

// Adapted from DeepGEMM's make_tma_2d_desc
// (csrc/jit_kernels/impls/runtime_utils.hpp at 88965b078186)
CUtensorMap make_2d(void* ptr, CUtensorMapDataType dt, int elem_size,
                    int gmem_inner, int gmem_outer, int smem_inner,
                    int smem_outer, long gmem_outer_stride, int swizzle_mode) {
  if (swizzle_mode != 0) smem_inner = swizzle_mode / elem_size;
  CUtensorMap tm;
  const cuuint64_t gdims[2] = {(cuuint64_t)gmem_inner, (cuuint64_t)gmem_outer};
  const cuuint32_t sdims[2] = {(cuuint32_t)smem_inner, (cuuint32_t)smem_outer};
  const cuuint64_t gstrides[1] = {(cuuint64_t)(gmem_outer_stride * elem_size)};
  const cuuint32_t estrides[2] = {1, 1};
  CUtensorMapSwizzle swizzle = swizzle_mode == 128  ? CU_TENSOR_MAP_SWIZZLE_128B
                               : swizzle_mode == 64 ? CU_TENSOR_MAP_SWIZZLE_64B
                               : swizzle_mode == 32
                                   ? CU_TENSOR_MAP_SWIZZLE_32B
                                   : CU_TENSOR_MAP_SWIZZLE_NONE;
  CUresult r = encode_tiled()(&tm, dt, 2, ptr, gdims, gstrides, sdims, estrides,
                              CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
                              CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                              CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  STD_TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  return tm;
}

// KV [keys,128] bytes viewed as {128, groups, stride} with strides
// {128*stride, 128}: box {128, 256, 1} at (0, 0, o) loads keys o, o+stride, ...
// with 128B swizzle. The view spans groups * stride keys, which can exceed
// `keys` by < stride / 2; that is safe only because the kernel loads offset
// o = stride / 2, whose last key ((groups - 1) * stride + stride / 2) is
// always < keys (launch_config sizes `groups` for that).
CUtensorMap make_strided_kv(void* ptr, uint32_t groups, uint32_t stride) {
  CUtensorMap tm;
  constexpr uint64_t kRowBytes = fused_mqa_topk::kHeadDim;
  const cuuint64_t gdims[3] = {kRowBytes, groups, stride};
  const cuuint64_t gstrides[2] = {kRowBytes * stride, kRowBytes};
  const cuuint32_t box[3] = {kRowBytes, fused_mqa_topk::kTileKeys, 1};
  const cuuint32_t estrides[3] = {1, 1, 1};
  CUresult r = encode_tiled()(
      &tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, ptr, gdims, gstrides, box,
      estrides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  STD_TORCH_CHECK(r == CUDA_SUCCESS,
                  "strided KV cuTensorMapEncodeTiled failed: ", (int)r);
  return tm;
}

using fused_mqa_topk::kBlockRows;
using fused_mqa_topk::kCandidatesPerRow;
using fused_mqa_topk::kHeadDim;
using fused_mqa_topk::kMaxKeys;
using fused_mqa_topk::kNumHeads;
using fused_mqa_topk::kNumThreads;
using fused_mqa_topk::kRetainedPerRow;
using fused_mqa_topk::kSlotsPerCta;
using fused_mqa_topk::kTileKeys;
namespace persist = fused_mqa_topk::persist;

// -------- launch configuration --------
using PersistentKernel =
    decltype(&persist::fused_mqa_topk_fp8_prefill_kernel<16, 2048>);

// vLLM dispatches chunks of >= 32K keys per row here (fused_mqa_topk_min_keys);
// smaller key counts are supported but take the same <= 32K configuration.
template <uint32_t TopK>
PersistentKernel persistent_kernel(uint32_t keys) {
  return keys <= 32768    ? persist::fused_mqa_topk_fp8_prefill_kernel<16, TopK>
         : keys <= 131072 ? persist::fused_mqa_topk_fp8_prefill_kernel<32, TopK>
         : keys <= 524288
             ? persist::fused_mqa_topk_fp8_prefill_kernel<64, TopK>
             : persist::fused_mqa_topk_fp8_prefill_kernel<128, TopK>;
}

struct LaunchConfig {
  PersistentKernel kernel;
  int smem;              // dynamic shared memory bytes
  uint32_t phase_bytes;  // selection scratch bytes past the Q / weights region
  // Gate sample: sample_tiles tiles of 256 keys on a uniform grid (stride
  // sample_stride).
  uint32_t sample_target, sample_stride, sample_groups, sample_tiles;
};

// Dynamic shared memory of `kernel` on `device`: everything past its static
// shared memory up to the device's opt-in limit (232448 bytes on SM100; the
// staged selection uses it all), at least the exact rescan's layout. Sets
// the opt-in attribute; both are computed once per device and kernel.
inline cudaError_t kernel_smem(int device, PersistentKernel kernel, int& smem) {
  // The recovery rescan reuses the same dynamic shared memory with the rescan's
  // layout.
  constexpr int kFallbackSmem =
      int(sizeof(fused_mqa_topk::dense::SharedStorage) +
          sizeof(persist::RescanShared));
  constexpr int kMinSmem = int(
      sizeof(persist::KernelSmem) +
      std::max(sizeof(persist::HandoffRing), sizeof(persist::SampleShared)));
  static std::mutex mutex;
  static std::map<std::pair<int, PersistentKernel>, int> cache;
  const std::lock_guard<std::mutex> lock(mutex);
  if (const auto it = cache.find({device, kernel}); it != cache.end()) {
    smem = it->second;
    return cudaSuccess;
  }
  int max_smem = 0;
  if (const cudaError_t e = cudaDeviceGetAttribute(
          &max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
      e != cudaSuccess)
    return e;
  cudaFuncAttributes attr;
  if (const cudaError_t e =
          cudaFuncGetAttributes(&attr, reinterpret_cast<const void*>(kernel));
      e != cudaSuccess)
    return e;
  const int bytes =
      std::max(kFallbackSmem, max_smem - int(attr.sharedSizeBytes));
  if (bytes < kMinSmem) return cudaErrorInvalidConfiguration;
  if (const cudaError_t e = cudaFuncSetAttribute(
          reinterpret_cast<const void*>(kernel),
          cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
      e != cudaSuccess)
    return e;
  cache.emplace(std::make_pair(device, kernel), bytes);
  smem = bytes;
  return cudaSuccess;
}

// Picks the kernel instantiation, its shared memory and the gate sample for
// `keys` keys and top-`topk` rows on `device` (the current device).
inline cudaError_t launch_config(int device, uint32_t keys, uint32_t topk,
                                 LaunchConfig& c) {
  // The gate aims for `target` candidates per row and sizes the sample so its
  // quantile rank is `rank` (count error ~1/sqrt(rank)); a row whose count
  // falls below the top-k is rescanned exactly. Up to 128K keys the candidate
  // stream dominates, so smaller top-ks aim lower with a coarser sample. Above
  // 128K the sample is clamped to 16-32 tiles (rank ~32 at 1M keys): a sample
  // tile costs about two scan tiles, so a larger sample costs more than a
  // smaller candidate set saves. Tuned on B300 real captures.
  struct Gate {
    uint32_t target, rank;
  };
  constexpr Gate kGates[3][2] = {
      // <= 128K       > 128K
      {{1024, 32}, {1536, 64}},  // top-512
      {{1536, 48}, {2048, 48}},  // top-1024
      {{3072, 64}, {4096, 64}},  // top-2048
  };
  const bool long_ctx = keys > 131072;
  const Gate gate =
      kGates[topk == 512 ? 0 : (topk == 1024 ? 1 : 2)][long_ctx ? 1 : 0];
  c.sample_target = gate.target;
  const uint32_t rank = gate.rank;
  const uint64_t tile = kTileKeys;
  const uint32_t rank_tiles = std::max(
      1u, uint32_t((uint64_t(rank) * keys + tile * c.sample_target - 1) /
                   (tile * c.sample_target)));
  c.sample_tiles =
      long_ctx ? std::min(32u, std::max(16u, rank_tiles)) : rank_tiles;
  c.sample_stride = std::max(1u, uint32_t((keys + tile * c.sample_tiles - 1) /
                                          (tile * c.sample_tiles)));
  c.sample_groups =
      (keys - c.sample_stride / 2 + c.sample_stride - 1) / c.sample_stride;
  c.kernel = topk == 512    ? persistent_kernel<512>(keys)
             : topk == 1024 ? persistent_kernel<1024>(keys)
                            : persistent_kernel<2048>(keys);
  if (const cudaError_t e = kernel_smem(device, c.kernel, c.smem);
      e != cudaSuccess)
    return e;
  c.phase_bytes = uint32_t(c.smem) - persist::kSelBase;
  return cudaSuccess;
}

void check_tensor(const torch::stable::Tensor& t, int32_t device,
                  ScalarType dtype, const char* name) {
  STD_TORCH_CHECK(t.is_cuda() && t.get_device_index() == device, name,
                  " must be on the same CUDA device as q");
  STD_TORCH_CHECK(t.scalar_type() == dtype, name, " has the wrong dtype");
  STD_TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

// TMA tensor maps need 16-byte aligned base addresses.
void check_aligned(const torch::stable::Tensor& t, const char* name) {
  STD_TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, name,
                  " must be 16-byte aligned");
}

}  // namespace

// q [rows,32,128] fp8e4m3, k [keys,128] fp8e4m3, scales [keys] fp32,
// weights [rows,32] fp32, starts/ends [rows] int32 (visible key range
// [start,end) of each row), out [rows,K] int32 with K (the top-k) 512, 1024 or
// 2048: key indices relative to the row's start (key - start, the
// top_k_per_row_prefill convention), unordered; -1 padded when fewer than K
// keys are visible. Rows that see at most K keys get them all, in order. 0 <
// rows < 2^26, 0 < keys <= 1048576 (the last 8-row block may be partial). q, k,
// scales, weights and out must be 16-byte aligned (TMA / vector stores).
// starts/ends are expected in [0, keys] with start <= end; they are read as
// uint32_t and clamped to keys, so a negative value means `keys` and an end
// before its start means an empty row.
// The output is exact up to ties: the gated path ranks fp24 score codes (the
// low 8 mantissa bits dropped), the exact rescan ranks fp32 scores.
// Workspace (caller-owned, reused across calls), with B = ceil(rows / 8) and
// S >= 2 slots per CTA of the persistent grid (one CTA per SM):
//   counter [2] int32 (zero between calls; the kernel leaves it zeroed, so a
//     workspace must not be shared by launches that can run concurrently, e.g.
//     on two streams),
//   values [8S,16384] uint16 (low 16 bits of the fp24 candidate codes),
//   indices [8S,16384] int32 (key index in bits 0-19, the code's high 8 bits
//   in bits 20-27),
//   retained [8S,4096] fp32, retained_indices [8S,4096] int32,
//   counts [>=8B] int32, defer [>=B] int32.
void fused_mqa_topk_prefill(
    torch::stable::Tensor q, torch::stable::Tensor k,
    torch::stable::Tensor scales, torch::stable::Tensor weights,
    torch::stable::Tensor starts, torch::stable::Tensor ends,
    torch::stable::Tensor out, torch::stable::Tensor counter,
    torch::stable::Tensor values, torch::stable::Tensor indices,
    torch::stable::Tensor retained, torch::stable::Tensor retained_indices,
    torch::stable::Tensor counts, torch::stable::Tensor defer) {
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
  check_tensor(values, dev, ScalarType::UInt16, "values");
  check_tensor(indices, dev, ScalarType::Int, "indices");
  check_tensor(retained, dev, ScalarType::Float, "retained");
  check_tensor(retained_indices, dev, ScalarType::Int, "retained_indices");
  check_tensor(counts, dev, ScalarType::Int, "counts");
  check_tensor(defer, dev, ScalarType::Int, "defer");

  STD_TORCH_CHECK(
      q.dim() == 3 && q.size(1) == kNumHeads && q.size(2) == kHeadDim,
      "q must be [rows, 32, 128]");
  STD_TORCH_CHECK(k.dim() == 2 && k.size(1) == kHeadDim,
                  "k must be [keys, 128]");
  const int64_t rows = q.size(0), keys = k.size(0);
  // TMA row coordinates (rows * 32 Q rows) are int32.
  STD_TORCH_CHECK(rows > 0 && rows < (int64_t(1) << 26),
                  "rows must be in [1, 2^26)");
  STD_TORCH_CHECK(keys > 0 && keys <= kMaxKeys, "keys must be in [1, 1048576]");
  const int64_t blocks = (rows + kBlockRows - 1) / kBlockRows;
  STD_TORCH_CHECK(scales.numel() == keys, "scales must be [keys]");
  STD_TORCH_CHECK(weights.dim() == 2 && weights.size(0) == rows &&
                      weights.size(1) == kNumHeads,
                  "weights must be [rows, 32]");
  STD_TORCH_CHECK(starts.numel() == rows && ends.numel() == rows,
                  "starts/ends must be [rows]");
  STD_TORCH_CHECK(out.dim() == 2 && out.size(0) == rows &&
                      fused_mqa_topk::is_supported_topk(out.size(1)),
                  "out must be [rows, K] with K in {512, 1024, 2048}");
  check_aligned(q, "q");
  check_aligned(k, "k");
  check_aligned(scales, "scales");
  check_aligned(weights, "weights");
  check_aligned(out, "out");

  torch::stable::accelerator::DeviceGuard guard(dev);
  const cudaStream_t stream = get_current_cuda_stream(dev);
  const uint32_t urows = static_cast<uint32_t>(rows),
                 ukeys = static_cast<uint32_t>(keys),
                 ublocks = static_cast<uint32_t>(blocks),
                 topk = static_cast<uint32_t>(out.size(1));
  const auto* starts_p = static_cast<const uint32_t*>(starts.data_ptr());
  const auto* ends_p = static_cast<const uint32_t*>(ends.data_ptr());
  auto* out_p = static_cast<int*>(out.data_ptr());
  // Q: rows * 32 head rows of 128 bytes, a block's 256 per box; K: keys rows of
  // 128 bytes, a tile's 256 per box (both 128B-swizzled); per-key scales: a
  // tile's 256; weights: a block's 8 rows of 32.
  const int head_rows = int(rows) * int(kNumHeads);
  auto tq = make_2d(q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, kHeadDim,
                    head_rows, kHeadDim, kBlockRows * kNumHeads, kHeadDim, 128);
  auto tk = make_2d(k.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, kHeadDim,
                    int(keys), kHeadDim, kTileKeys, kHeadDim, 128);
  auto ts = make_2d(scales.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4,
                    int(keys), 1, kTileKeys, 1, 0, 0);
  auto tw = make_2d(weights.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4,
                    kNumHeads, int(rows), kNumHeads, kBlockRows, kNumHeads, 0);

  STD_TORCH_CHECK(counter.numel() == 2, "counter must be [2]");
  constexpr int64_t kCtaRows = kSlotsPerCta * kBlockRows;
  STD_TORCH_CHECK(values.dim() == 2 && values.size(1) == kCandidatesPerRow &&
                      values.size(0) % kCtaRows == 0 && indices.dim() == 2 &&
                      indices.size(0) == values.size(0) &&
                      indices.size(1) == kCandidatesPerRow,
                  "values/indices must be [8 * slots, 16384], slots even");
  STD_TORCH_CHECK(values.size(0) > 0, "the candidate workspace is empty");
  STD_TORCH_CHECK(retained.dim() == 2 && retained.size(1) == kRetainedPerRow &&
                      retained.size(0) == values.size(0) &&
                      retained_indices.dim() == 2 &&
                      retained_indices.size(0) == retained.size(0) &&
                      retained_indices.size(1) == kRetainedPerRow,
                  "retained/retained_indices must be [8 * slots, 4096]");
  STD_TORCH_CHECK(
      counts.numel() >= blocks * kBlockRows && defer.numel() >= blocks,
      "per-row workspace too small");
  // One persistent CTA per SM (capped by the blocks and the workspace slots);
  // CTA c owns candidate slots 2c and 2c + 1.
  const uint32_t grid = static_cast<uint32_t>(std::min<int64_t>(
      {int64_t(sm_count(dev)), blocks, values.size(0) / kCtaRows}));
  LaunchConfig c;
  const cudaError_t config_err = launch_config(dev, ukeys, topk, c);
  STD_TORCH_CHECK(config_err != cudaErrorInvalidConfiguration,
                  "fused_mqa_topk_prefill needs more shared memory per block "
                  "than this GPU allows");
  check_cuda(config_err);
  auto tks = make_strided_kv(k.data_ptr(), c.sample_groups, c.sample_stride);
  const fused_mqa_topk::rescan::RescanBuffers rescan{
      static_cast<float*>(retained.data_ptr()),
      static_cast<int*>(retained_indices.data_ptr())};
  c.kernel<<<grid, kNumThreads, c.smem, stream>>>(
      urows, ukeys, starts_p, ends_p, out_p,
      static_cast<uint16_t*>(values.data_ptr()),
      static_cast<int*>(indices.data_ptr()),
      static_cast<int*>(counts.data_ptr()), rescan, c.phase_bytes,
      c.sample_target, c.sample_stride, c.sample_groups, c.sample_tiles,
      static_cast<const float*>(scales.data_ptr()), tks, tq, ts, tk, tw,
      ublocks, static_cast<uint32_t*>(counter.data_ptr()),
      static_cast<int*>(defer.data_ptr()));
  check_cuda(cudaGetLastError());
}
