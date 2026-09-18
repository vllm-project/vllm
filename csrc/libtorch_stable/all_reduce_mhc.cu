// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "torch_utils.h"
#include "custom_all_gather_reduce_scatter.cuh"
#include <cooperative_groups.h>
#include <torch/csrc/stable/library.h>

namespace {
using Tensor = torch::stable::Tensor;
using Pack = vllm::packed_t<nv_bfloat16>::P;
constexpr int kHidden = 5120;
constexpr int kRanks = 4;
constexpr int kThreads = 128;
constexpr int kCluster = 5;
constexpr int kPacks = kHidden / Pack::size;

__global__ __launch_bounds__(kThreads)
    __cluster_dims__(1, kCluster, 1) void all_reduce_mhc_kernel(
        const Pack* input, const Pack* residual, const float* post,
        const float* comb, const float* pre, const Pack* weight, Pack* output,
        Pack* normalized, Pack* local, Pack* multicast, uint32_t* epochs,
        int rank, int stage_size, float eps) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  cudaGridDependencySynchronize();
  const int token = blockIdx.x;
  const int column = blockIdx.y * kThreads + threadIdx.x;
  const int idx = token * kPacks + column;
  const int size = gridDim.x * kPacks;
  const int stage = epochs[0] % 3;
  const int dirty = vllm::mnnvl_lamport_dirty_stage(stage);
  const int dirty_size = epochs[2 + dirty];
  Pack value = input[idx];
  vllm::store_multimem_lamport_payload(
      multicast + stage * stage_size + rank * size + idx,
      vllm::sanitize_lamport_payload(value));
  // Multi-token forwards benefit from launching dependents before mHC finishes.
  if (gridDim.x > 1) cudaTriggerProgrammaticLaunchCompletion();
  vllm::lamport_cta_arrive(epochs + 1);
  for (int i = idx; i < dirty_size; i += size) {
    local[dirty * stage_size + i] = vllm::lamport_sentinel<Pack>();
  }
  Pack peers[kRanks];
  bool pending;
  do {
    vllm::LamportPack<Pack> packets[kRanks];
  #pragma unroll
    for (int peer = 0; peer < kRanks; ++peer) {
      auto ptr = local + stage * stage_size + idx + peer * size;
      asm volatile("ld.relaxed.sys.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(packets[peer].words[0]), "=r"(packets[peer].words[1]),
                     "=r"(packets[peer].words[2]), "=r"(packets[peer].words[3])
                   : "l"(ptr)
                   : "memory");
    }
    pending = false;
  #pragma unroll
    for (int peer = 0; peer < kRanks; ++peer) {
      pending |= vllm::is_lamport_dirty(packets[peer]);
      peers[peer] = packets[peer].packed;
    }
  } while (pending);
  auto sum = vllm::upcast(peers[0]);
  #pragma unroll
  for (int r = 1; r < kRanks; ++r) {
    vllm::packed_assign_add(sum, vllm::upcast(peers[r]));
  }
  // Keep the BF16 boundary between the collective and mHC post.
  auto reduced = vllm::upcast(vllm::downcast<Pack>(sum));
  decltype(sum) streams[4];
  #pragma unroll
  for (int r = 0; r < 4; ++r) {
    streams[r] = vllm::upcast(residual[(token * 4 + r) * kPacks + column]);
  }
  decltype(sum) collapsed{};
  #pragma unroll
  for (int c = 0; c < 4; ++c) {
    decltype(sum) mixed;
  #pragma unroll
    for (int e = 0; e < Pack::size; ++e) {
      // Match post*x followed by the four residual FMAs.
      float acc = __fmul_rn(post[token * 4 + c], reduced.data[e]);
  #pragma unroll
      for (int r = 0; r < 4; ++r) {
        acc = fmaf(comb[token * 16 + r * 4 + c], streams[r].data[e], acc);
      }
      mixed.data[e] = acc;
    }
    Pack rounded = vllm::downcast<Pack>(mixed);
    output[(token * 4 + c) * kPacks + column] = rounded;
    auto as_float = vllm::upcast(rounded);
  #pragma unroll
    for (int e = 0; e < Pack::size; ++e) {
      collapsed.data[e] =
          fmaf(pre[token * 4 + c], as_float.data[e], collapsed.data[e]);
    }
  }
  // RMSNorm consumes the rounded BF16 collapse in the unfused path.
  collapsed = vllm::upcast(vllm::downcast<Pack>(collapsed));
  float sqr = 0;
  #pragma unroll
  for (int e = 0; e < Pack::size; ++e) {
    sqr = fmaf(collapsed.data[e], collapsed.data[e], sqr);
  }
  #pragma unroll
  for (int delta = 16; delta; delta /= 2) {
    sqr += __shfl_xor_sync(0xffffffff, sqr, delta);
  }
  __shared__ float warp_sums[kCluster][kThreads / 32];
  auto cluster = cooperative_groups::this_cluster();
  const int lane = threadIdx.x % 32;
  if (lane < kCluster) {
    *cluster.map_shared_rank(&warp_sums[blockIdx.y][threadIdx.x / 32], lane) =
        sqr;
  }
  cluster.sync();
  float total = 0;
  #pragma unroll
  for (int c = 0; c < kCluster; ++c) {
  #pragma unroll
    for (int w = 0; w < kThreads / 32; ++w) total += warp_sums[c][w];
  }
  const float scale = rsqrtf(__fadd_rn(total / kHidden, eps));
  auto weights = vllm::upcast(weight[column]);
  #pragma unroll
  for (int e = 0; e < Pack::size; ++e) {
    collapsed.data[e] = collapsed.data[e] * scale * weights.data[e];
  }
  normalized[idx] = vllm::downcast<Pack>(collapsed);
  if (idx == 0) {
    while (*reinterpret_cast<volatile uint32_t*>(epochs + 1) <
           gridDim.x * gridDim.y) {
    }
    epochs[2 + stage] = kRanks * size;
    epochs[0] = vllm::mnnvl_lamport_next_stage(stage);
    epochs[1] = 0;
  }
  if (gridDim.x == 1) cudaTriggerProgrammaticLaunchCompletion();
#else
  asm volatile("trap;");
#endif
}

void all_reduce_mhc(Tensor input, Tensor residual, Tensor post, Tensor comb,
                    Tensor pre, Tensor weight, Tensor output, Tensor normalized,
                    int64_t local, int64_t multicast, Tensor epochs,
                    int64_t rank, int64_t stage_bytes, double eps) {
  using torch::headeronly::ScalarType;
  const torch::stable::accelerator::DeviceGuard guard(input.get_device_index());
  const int64_t n = input.size(0);
  STD_TORCH_CHECK(input.dim() == 2 && input.size(1) == kHidden && n > 0 &&
                  n <= 16);
  for (const auto& t : {input, residual, weight, output, normalized}) {
    STD_TORCH_CHECK(t.is_contiguous() &&
                    t.scalar_type() == ScalarType::BFloat16);
    STD_TORCH_CHECK(t.get_device_index() == input.get_device_index());
    STD_TORCH_CHECK(reinterpret_cast<uintptr_t>(t.const_data_ptr()) % 16 == 0);
  }
  for (const auto& t : {post, comb, pre}) {
    STD_TORCH_CHECK(t.is_contiguous() && t.scalar_type() == ScalarType::Float);
    STD_TORCH_CHECK(t.get_device_index() == input.get_device_index());
  }
  STD_TORCH_CHECK(residual.numel() == n * 4 * kHidden &&
                  output.numel() == residual.numel() &&
                  normalized.numel() == input.numel() &&
                  weight.numel() == kHidden);
  STD_TORCH_CHECK(post.numel() == n * 4 && pre.numel() == n * 4 &&
                  comb.numel() == n * 16);
  STD_TORCH_CHECK(epochs.is_contiguous() &&
                  epochs.scalar_type() == ScalarType::Int &&
                  epochs.numel() >= 5 &&
                  epochs.get_device_index() == input.get_device_index());
  STD_TORCH_CHECK(local && multicast && rank >= 0 && rank < kRanks &&
                  stage_bytes % 16 == 0 &&
                  stage_bytes >= n * kHidden * 2 * kRanks);
  STD_TORCH_CHECK(get_device_prop()->major == 10);
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t config{
      .gridDim = dim3(n, kCluster),
      .blockDim = dim3(kThreads),
      .dynamicSmemBytes = 0,
      .stream = get_current_cuda_stream(input.get_device_index()),
      .attrs = &attr,
      .numAttrs = 1};
  STD_CUDA_CHECK(cudaLaunchKernelEx(
      &config, all_reduce_mhc_kernel,
      static_cast<const Pack*>(input.const_data_ptr()),
      static_cast<const Pack*>(residual.const_data_ptr()),
      static_cast<const float*>(post.const_data_ptr()),
      static_cast<const float*>(comb.const_data_ptr()),
      static_cast<const float*>(pre.const_data_ptr()),
      static_cast<const Pack*>(weight.const_data_ptr()),
      static_cast<Pack*>(output.mutable_data_ptr()),
      static_cast<Pack*>(normalized.mutable_data_ptr()),
      reinterpret_cast<Pack*>(local), reinterpret_cast<Pack*>(multicast),
      static_cast<uint32_t*>(epochs.mutable_data_ptr()), static_cast<int>(rank),
      static_cast<int>(stage_bytes / sizeof(Pack)), static_cast<float>(eps)));
}
}  // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(_C_custom_ar, mhc) {
  mhc.def(
      "all_reduce_mhc(Tensor input, Tensor residual, Tensor post, Tensor comb, "
      "Tensor pre, Tensor weight, Tensor! output, Tensor! normalized, "
      "int local, int multicast, Tensor! epochs, int rank, int stage_bytes, "
      "float eps) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(_C_custom_ar, CUDA, mhc) {
  mhc.impl("all_reduce_mhc", TORCH_BOX(&all_reduce_mhc));
}
