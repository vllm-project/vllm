// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "kernel.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstddef>

extern "C" void turboquant_mma_decode_launch(
    const void* q, const void* kv_cache, const int* pt, const int* sl,
    const float* rot, const float* cen, void* out, void* part_o, void* part_ml,
    int B, int HQ, int HKV, int page_table_stride, int page_size, int head_dim,
    int splits, cudaStream_t stream);

struct LaunchPlan {
  bool use_mma;
  int splits;
  size_t workspace_bytes;
};

static int get_sm_count(int device) {
  static int sm_counts[32] = {};
  TORCH_CHECK(device >= 0 && device < 32, "invalid CUDA device index");
  if (sm_counts[device] == 0) {
    cudaDeviceGetAttribute(&sm_counts[device], cudaDevAttrMultiProcessorCount,
                           device);
  }
  return sm_counts[device];
}

static LaunchPlan make_launch_plan(int B, int HQ, int HKV,
                                   int page_table_stride, int page_size,
                                   int head_dim, int max_seq_len,
                                   int sm_count) {
  const int G = HQ / HKV;
  const int mma_npacks = (G + 15) / 16;
  const int natural = B * HKV * mma_npacks;
  const int ntiles64 = (max_seq_len + 63) / 64;
  const bool populated_wide_pack =
      G >= 4 && ntiles64 >= 1 &&
      (natural >= sm_count || (G >= 6 && 3 * natural >= sm_count) ||
       (G >= 7 && 5 * natural >= sm_count));
  const int cap = std::max(1, std::min(64, ntiles64));
  int mma_splits = 1;
  double best = 1e30;
  if (populated_wide_pack) {
    for (int tiles_per_split = 1; tiles_per_split <= ntiles64;
         ++tiles_per_split) {
      const int s = (ntiles64 + tiles_per_split - 1) / tiles_per_split;
      if (s > cap) continue;
      const double bpsm = static_cast<double>(natural) * s / sm_count;
      const double resident_waves = head_dim == 256 ? 2.0 : 3.0;
      const double tail = std::max(std::ceil(bpsm), resident_waves) / bpsm;
      const double merge =
          s > 1
              ? static_cast<double>(s) * HQ * 1050.0 /
                    (static_cast<double>(max_seq_len) * HKV * (head_dim + 6.0))
              : 0.0;
      const double cost = tail * (1.0 + merge);
      if (cost < best - 1e-12) {
        best = cost;
        mma_splits = s;
      }
      if (s == 1) break;
    }
  }

  if (populated_wide_pack) {
    const int64_t part_o_bytes = mma_splits > 1
                                     ? static_cast<int64_t>(mma_splits) * B *
                                           HQ * head_dim * sizeof(float)
                                     : 0;
    const int64_t part_ml_bytes =
        mma_splits > 1
            ? static_cast<int64_t>(mma_splits) * B * HQ * 2 * sizeof(float)
            : 0;
    return {true, mma_splits,
            static_cast<size_t>(part_o_bytes + part_ml_bytes)};
  }

  const int logical_pages = (max_seq_len + page_size - 1) / page_size;
  const DecodePlan plan =
      make_decode_plan(B, HQ, HKV, std::min(logical_pages, page_table_stride),
                       page_size, sm_count);
  const int64_t workspace_floats =
      plan.splits > 1
          ? static_cast<int64_t>(B) * HQ * plan.splits * (head_dim + 2)
          : 0;
  return {false, plan.splits,
          static_cast<size_t>(workspace_floats * sizeof(float))};
}

static int64_t workspace_size(int64_t B, int64_t HQ, int64_t HKV,
                              int64_t page_table_stride, int64_t page_size,
                              int64_t head_dim, int64_t max_seq_len,
                              int64_t device) {
  TORCH_CHECK(B > 0 && HQ > 0 && HKV > 0 && HQ % HKV == 0,
              "TurboQuant decode dimensions are invalid");
  TORCH_CHECK(page_table_stride > 0 && max_seq_len > 0 &&
                  max_seq_len <= page_table_stride * page_size,
              "TurboQuant page table dimensions are invalid");
  TORCH_CHECK(
      page_size == 16 || page_size == 32 || page_size == 64 || page_size == 128,
      "TurboQuant page size must be one of {16, 32, 64, 128}");
  TORCH_CHECK(head_dim == 64 || head_dim == 128 || head_dim == 256,
              "TurboQuant head size must be one of {64, 128, 256}");
  return static_cast<int64_t>(make_launch_plan(B, HQ, HKV, page_table_stride,
                                               page_size, head_dim, max_seq_len,
                                               get_sm_count(device))
                                  .workspace_bytes);
}

static void run_with_workspace(torch::Tensor q, torch::Tensor kv_cache,
                               torch::Tensor page_table, torch::Tensor seq_lens,
                               torch::Tensor rotation, torch::Tensor centroids,
                               torch::Tensor workspace, torch::Tensor out,
                               int64_t page_size, int64_t max_seq_len) {
  TORCH_CHECK(q.is_cuda() && q.is_contiguous() && q.dim() == 3,
              "TurboQuant query must be a contiguous CUDA tensor");
  const int head_dim = q.size(2);
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 &&
                  (head_dim == 64 || head_dim == 128 || head_dim == 256),
              "TurboQuant decode requires BF16 query with head size "
              "64, 128, or 256");
  TORCH_CHECK(kv_cache.is_cuda() && kv_cache.is_contiguous() &&
                  kv_cache.scalar_type() == at::kByte && kv_cache.dim() == 4,
              "TurboQuant cache must be a contiguous CUDA uint8 tensor");
  TORCH_CHECK(
      kv_cache.size(2) == 1 && kv_cache.size(3) == page_size * (head_dim + 6),
      "TurboQuant cache record has an incompatible size");
  TORCH_CHECK(page_table.is_cuda() && page_table.scalar_type() == at::kInt &&
                  page_table.dim() == 2 && page_table.stride(1) == 1,
              "TurboQuant page table must be a CUDA int32 row-major tensor");
  TORCH_CHECK(
      seq_lens.is_cuda() && seq_lens.is_contiguous() &&
          seq_lens.scalar_type() == at::kInt && seq_lens.dim() == 1,
      "TurboQuant sequence lengths must be a contiguous CUDA int32 tensor");
  TORCH_CHECK(rotation.is_cuda() && rotation.is_contiguous() &&
                  rotation.scalar_type() == at::kFloat && rotation.dim() == 2 &&
                  rotation.size(0) == head_dim && rotation.size(1) == head_dim,
              "TurboQuant rotation must be a contiguous CUDA float32 "
              "[head_dim, head_dim] tensor");
  TORCH_CHECK(centroids.is_cuda() && centroids.is_contiguous() &&
                  centroids.scalar_type() == at::kFloat &&
                  centroids.numel() == 16,
              "TurboQuant centroids must be a contiguous CUDA float32 tensor");
  TORCH_CHECK(workspace.is_cuda() && workspace.is_contiguous(),
              "TurboQuant workspace must be a contiguous CUDA tensor");
  TORCH_CHECK(out.is_cuda() && out.is_contiguous() &&
                  out.scalar_type() == at::kBFloat16 &&
                  out.sizes() == q.sizes(),
              "TurboQuant output must match the query");
  TORCH_CHECK(kv_cache.get_device() == q.get_device() &&
                  page_table.get_device() == q.get_device() &&
                  seq_lens.get_device() == q.get_device() &&
                  rotation.get_device() == q.get_device() &&
                  centroids.get_device() == q.get_device() &&
                  workspace.get_device() == q.get_device() &&
                  out.get_device() == q.get_device(),
              "TurboQuant tensors must be on the same CUDA device");
  TORCH_CHECK(
      page_size == 16 || page_size == 32 || page_size == 64 || page_size == 128,
      "TurboQuant page size must be one of {16, 32, 64, 128}");

  const int B = q.size(0);
  const int HQ = q.size(1);
  const int HKV = kv_cache.size(1);
  const int page_table_stride = page_table.stride(0);
  TORCH_CHECK(B > 0 && HKV > 0 && HQ % HKV == 0 && page_table.size(0) == B &&
                  seq_lens.size(0) == B,
              "TurboQuant decode shapes are incompatible");
  TORCH_CHECK(page_table.size(1) > 0 && max_seq_len > 0 &&
                  max_seq_len <= page_table.size(1) * page_size,
              "TurboQuant max sequence length exceeds the page table");

  at::cuda::CUDAGuard device_guard(q.device());
  int major = 0;
  int minor = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                         q.get_device());
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                         q.get_device());
  TORCH_CHECK(minor == 0 && (major == 8 || major == 9 || major == 10),
              "TurboQuant optimized decode requires SM80, SM90, or SM100");
  const LaunchPlan launch_plan =
      make_launch_plan(B, HQ, HKV, page_table_stride, page_size, head_dim,
                       max_seq_len, get_sm_count(q.get_device()));
  const size_t workspace_bytes = workspace.numel() * workspace.element_size();
  TORCH_CHECK(workspace_bytes >= launch_plan.workspace_bytes,
              "TurboQuant caller workspace is too small");

  char* base = static_cast<char*>(workspace.data_ptr());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  if (launch_plan.use_mma) {
    const int64_t part_o_bytes =
        launch_plan.splits > 1 ? static_cast<int64_t>(launch_plan.splits) * B *
                                     HQ * head_dim * sizeof(float)
                               : 0;
    void* part_o = launch_plan.splits > 1 ? base : nullptr;
    void* part_ml = launch_plan.splits > 1 ? base + part_o_bytes : nullptr;
    turboquant_mma_decode_launch(
        q.data_ptr(), kv_cache.data_ptr(), page_table.data_ptr<int>(),
        seq_lens.data_ptr<int>(), rotation.data_ptr<float>(),
        centroids.data_ptr<float>(), out.data_ptr(), part_o, part_ml, B, HQ,
        HKV, page_table_stride, page_size, head_dim, launch_plan.splits,
        stream);
    return;
  }

  const int logical_pages = (max_seq_len + page_size - 1) / page_size;
  const DecodePlan plan =
      make_decode_plan(B, HQ, HKV, std::min(logical_pages, page_table_stride),
                       page_size, get_sm_count(q.get_device()));
  turboquant_decode_launch(
      q.data_ptr(), kv_cache.data_ptr(), page_table.data_ptr(),
      seq_lens.data_ptr(), rotation.data_ptr(), centroids.data_ptr(),
      out.data_ptr(), reinterpret_cast<float*>(base), B, HQ, HKV,
      page_table_stride, page_size, head_dim, plan, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("workspace_size", &workspace_size);
  m.def("run_with_workspace", &run_with_workspace);
}
