// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Single-launch fused FP8 MQA-logits + top-2048 prefill selection (SM100
// family): one kernel scores every visible key of a row and selects its
// top-2048 keys without materializing the [rows, keys] logits matrix.
//   <= 4K keys: resident_kernel.cuh (all scores of a block in shared memory).
//   >  4K keys: persistent_kernel.cuh (one CTA per SM; sampled gate, candidate
//               emission and selection; exact rescan of out-of-range blocks).
// This file is the libtorch-stable host wrapper. All scratch memory is passed
// in by the caller (no allocation here), so the op is CUDA-graph friendly.

#include "../torch_utils.h"

#include <torch/headeronly/core/ScalarType.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <algorithm>
#include <cstdint>

#include "persistent_kernel.cuh"
#include "resident_kernel.cuh"

namespace {

using torch::headeronly::ScalarType;

void check_cuda(cudaError_t err) {
  STD_TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
}

void* driver_handle() {
  static void* h = nullptr;
  if (!h) {
    h = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    STD_TORCH_CHECK(h, "failed to load libcuda.so.1");
  }
  return h;
}

CUresult enc_tiled(CUtensorMap* tm, CUtensorMapDataType dt, cuuint32_t rank,
                   void* addr, const cuuint64_t* dims,
                   const cuuint64_t* strides, const cuuint32_t* box,
                   const cuuint32_t* estrides, CUtensorMapInterleave il,
                   CUtensorMapSwizzle sw, CUtensorMapL2promotion l2,
                   CUtensorMapFloatOOBfill oob) {
  using FT =
      CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
                   const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
                   const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                   CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
  static FT f = nullptr;
  if (!f) {
    f = reinterpret_cast<FT>(dlsym(driver_handle(), "cuTensorMapEncodeTiled"));
    STD_TORCH_CHECK(f, "failed to load cuTensorMapEncodeTiled");
  }
  return f(tm, dt, rank, addr, dims, strides, box, estrides, il, sw, l2, oob);
}

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
  CUresult r = enc_tiled(&tm, dt, 2, ptr, gdims, gstrides, sdims, estrides,
                         CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
                         CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                         CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  STD_TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  return tm;
}

// KV [keys,128] bytes viewed as {128, groups, stride} with strides
// {128*stride, 128}: box {128, 256, 1} at (0, 0, o) loads keys o, o+stride, ...
// with 128B swizzle.
CUtensorMap make_strided_kv(void* ptr, unsigned groups, unsigned stride) {
  CUtensorMap tm;
  const cuuint64_t gdims[3] = {128, groups, stride};
  const cuuint64_t gstrides[2] = {128ull * stride, 128};
  const cuuint32_t box[3] = {128, 256, 1};
  const cuuint32_t estrides[3] = {1, 1, 1};
  CUresult r = enc_tiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, ptr, gdims, gstrides, box,
      estrides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  STD_TORCH_CHECK(r == CUDA_SUCCESS,
                  "strided KV cuTensorMapEncodeTiled failed: ", (int)r);
  return tm;
}

constexpr int kHeads = 32;
constexpr int kHeadDim = 128;
constexpr int kTopK = 2048;
constexpr int kCandidates = 16384;  // per-row candidate slab
constexpr int kRetained = 4096;     // per-row recovery slab

void check_tensor(const torch::stable::Tensor& t, int32_t device,
                  ScalarType dtype, const char* name) {
  STD_TORCH_CHECK(t.is_cuda() && t.get_device_index() == device, name,
                  " must be on the same CUDA device as q");
  STD_TORCH_CHECK(t.scalar_type() == dtype, name, " has the wrong dtype");
  STD_TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

}  // namespace

// q [rows,32,128] fp8e4m3, k [keys,128] fp8e4m3, scales [keys] fp32,
// weights [rows,32] fp32, starts/ends [rows] int32 (visible key range
// [start,end) of each row), out [rows,2048] int32: absolute key indices
// (columns of the logits), or with relative=true indices relative to the
// row's start (key - start, the top_k_per_row_prefill convention); -1 padded
// when fewer than 2048 keys are visible.
// 0 < rows <= 16384, 0 < keys <= 1048576 (the last 8-row block may be partial).
// Workspace (caller-owned, reused across calls), with B = ceil(rows / 8) and
// S >= 2 slots per CTA of the persistent grid (one CTA per SM):
//   counter [2] int32 (zero between calls; the kernel leaves it zeroed),
//   values [8S,16384] fp16, indices [8S,16384] int32,
//   retained [8S,4096] fp32, retained_indices [8S,4096] int32,
//   counts [>=8B] int32, defer [>=B] int32.
void fused_mqa_topk_prefill(
    torch::stable::Tensor q, torch::stable::Tensor k,
    torch::stable::Tensor scales, torch::stable::Tensor weights,
    torch::stable::Tensor starts, torch::stable::Tensor ends,
    torch::stable::Tensor out, torch::stable::Tensor counter,
    torch::stable::Tensor values, torch::stable::Tensor indices,
    torch::stable::Tensor retained, torch::stable::Tensor retained_indices,
    torch::stable::Tensor counts, torch::stable::Tensor defer, bool relative) {
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
  check_tensor(values, dev, ScalarType::Half, "values");
  check_tensor(indices, dev, ScalarType::Int, "indices");
  check_tensor(retained, dev, ScalarType::Float, "retained");
  check_tensor(retained_indices, dev, ScalarType::Int, "retained_indices");
  check_tensor(counts, dev, ScalarType::Int, "counts");
  check_tensor(defer, dev, ScalarType::Int, "defer");

  STD_TORCH_CHECK(q.dim() == 3 && q.size(1) == kHeads && q.size(2) == kHeadDim,
                  "q must be [rows, 32, 128]");
  STD_TORCH_CHECK(k.dim() == 2 && k.size(1) == kHeadDim,
                  "k must be [keys, 128]");
  const int64_t rows = q.size(0), keys = k.size(0);
  STD_TORCH_CHECK(rows > 0 && rows <= 16384, "rows must be in [1, 16384]");
  STD_TORCH_CHECK(keys > 0 && keys <= (1 << 20),
                  "keys must be in [1, 1048576]");
  const int64_t blocks = (rows + 7) / 8;
  STD_TORCH_CHECK(scales.numel() == keys, "scales must be [keys]");
  STD_TORCH_CHECK(weights.dim() == 2 && weights.size(0) == rows &&
                      weights.size(1) == kHeads,
                  "weights must be [rows, 32]");
  STD_TORCH_CHECK(starts.numel() == rows && ends.numel() == rows,
                  "starts/ends must be [rows]");
  STD_TORCH_CHECK(out.dim() == 2 && out.size(0) == rows && out.size(1) == kTopK,
                  "out must be [rows, 2048]");
  STD_TORCH_CHECK(reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0,
                  "out must be 16-byte aligned");

  torch::stable::accelerator::DeviceGuard guard(dev);
  const cudaStream_t stream = get_current_cuda_stream(dev);
  const unsigned urows = static_cast<unsigned>(rows),
                 ukeys = static_cast<unsigned>(keys),
                 ublocks = static_cast<unsigned>(blocks);
  const auto* starts_p = static_cast<const unsigned*>(starts.data_ptr());
  const auto* ends_p = static_cast<const unsigned*>(ends.data_ptr());
  auto* out_p = static_cast<int*>(out.data_ptr());
  auto tq = make_2d(q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, 128,
                    int(rows) * 32, 128, 256, 128, 128);
  auto tk = make_2d(k.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, 128,
                    int(keys), 128, 256, 128, 128);
  auto ts = make_2d(scales.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4,
                    int(keys), 1, 256, 1, 0, 0);
  auto tw = make_2d(weights.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4, 32,
                    int(rows), 32, 8, 32, 0);

  if (ukeys <= fused_mqa_resident::kResidentKeys) {
    // Every score of a block fits in shared memory: no workspace.
    constexpr int smem = int(fused_mqa_resident::kSmemBytes);
    check_cuda(cudaFuncSetAttribute(
        reinterpret_cast<const void*>(fused_mqa_topk_fp8_resident_kernel),
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    fused_mqa_topk_fp8_resident_kernel<<<ublocks, 384, smem, stream>>>(
        urows, ukeys, starts_p, ends_p, out_p, relative, tq, ts, tk, tw);
    check_cuda(cudaGetLastError());
    return;
  }

  STD_TORCH_CHECK(counter.numel() == 2, "counter must be [2]");
  STD_TORCH_CHECK(values.dim() == 2 && values.size(1) == kCandidates &&
                      values.size(0) % 16 == 0 && indices.dim() == 2 &&
                      indices.size(0) == values.size(0) &&
                      indices.size(1) == kCandidates,
                  "values/indices must be [8 * slots, 16384], slots even");
  STD_TORCH_CHECK(retained.dim() == 2 && retained.size(1) == kRetained &&
                      retained.size(0) == values.size(0) &&
                      retained_indices.dim() == 2 &&
                      retained_indices.size(0) == retained.size(0) &&
                      retained_indices.size(1) == kRetained,
                  "retained/retained_indices must be [8 * slots, 4096]");
  STD_TORCH_CHECK(counts.numel() >= blocks * 8 && defer.numel() >= blocks,
                  "per-row workspace too small");
  // One persistent CTA per SM (capped by the blocks and the workspace slots);
  // CTA c owns candidate slots 2c and 2c + 1.
  int num_sms = 0;
  check_cuda(
      cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev));
  const unsigned grid = static_cast<unsigned>(
      std::min<int64_t>({int64_t(num_sms), blocks, values.size(0) / 16}));
  fused_persist::LaunchConfig c;
  check_cuda(fused_persist::launch_config(ukeys, c));
  auto tks = make_strided_kv(k.data_ptr(), c.sample_groups, c.sample_stride);
  const FallbackBuffers fallback{static_cast<float*>(retained.data_ptr()),
                                 static_cast<int*>(retained_indices.data_ptr()),
                                 out_p};
  c.kernel<<<grid, 384, c.smem, stream>>>(
      urows, ukeys, starts_p, ends_p, static_cast<uint16_t*>(values.data_ptr()),
      static_cast<int*>(indices.data_ptr()),
      static_cast<int*>(counts.data_ptr()), fallback, c.phase_bytes,
      c.sample_target, c.sample_stride, c.sample_groups, c.sample_tiles,
      static_cast<const float*>(scales.data_ptr()), tks, tq, ts, tk, tw,
      relative, ublocks, static_cast<unsigned*>(counter.data_ptr()),
      static_cast<int*>(defer.data_ptr()));
  check_cuda(cudaGetLastError());
}
