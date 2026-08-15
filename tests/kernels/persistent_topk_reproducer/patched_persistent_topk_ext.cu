#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

#include "persistent_topk.cuh"

namespace P = vllm::persistent;

namespace {

inline void check_cuda(cudaError_t status, const char* operation) {
  TORCH_CHECK(status == cudaSuccess, operation, ": ",
              cudaGetErrorString(status));
}

template <int TopK>
void launch_patched(const torch::Tensor& logits, const torch::Tensor& lengths,
                    torch::Tensor& output, torch::Tensor& workspace,
                    int64_t max_seq_len) {
  const c10::cuda::CUDAGuard guard(logits.device());
  const int device_index = logits.get_device();
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, device_index),
             "cudaGetDeviceProperties failed");

  const uint32_t num_rows = static_cast<uint32_t>(logits.size(0));
  const uint32_t stride = static_cast<uint32_t>(logits.stride(0));
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_index);

  if (num_rows > 32 && properties.sharedMemPerBlockOptin >= 128 * 1024) {
    check_cuda(vllm::FilteredTopKRaggedTransform<float, int32_t, TopK>(
                   logits.data_ptr<float>(), output.data_ptr<int32_t>(),
                   lengths.data_ptr<int32_t>(), num_rows, TopK, stride, stream),
               "FilteredTopK failed");
    check_cuda(cudaGetLastError(), "patched persistent_topk launch failed");
    return;
  }

  int effective_max_smem = properties.sharedMemPerBlockOptin;
  if (num_rows <= 4) {
    effective_max_smem = std::min(
        effective_max_smem, static_cast<int>(P::kSmemMedium));
  } else if (num_rows <= 8) {
    effective_max_smem = std::min(effective_max_smem, 48 * 1024);
  }

  uint32_t vec_size = 1;
  if (stride % 4 == 0) {
    vec_size = 4;
  } else if (stride % 2 == 0) {
    vec_size = 2;
  }

  const size_t available_for_ordered =
      static_cast<size_t>(effective_max_smem) - P::kFixedSmemLarge;
  uint32_t max_chunk_elements =
      static_cast<uint32_t>(available_for_ordered / sizeof(uint32_t));
  max_chunk_elements = max_chunk_elements / vec_size * vec_size;
  const uint32_t min_chunk = vec_size * P::kThreadsPerBlock;
  max_chunk_elements = std::max(max_chunk_elements, min_chunk);

  uint32_t ctas_per_group =
      (stride + max_chunk_elements - 1) / max_chunk_elements;
  uint32_t chunk_size =
      (stride + ctas_per_group - 1) / ctas_per_group;
  chunk_size = (chunk_size + vec_size - 1) / vec_size * vec_size;
  chunk_size = std::min(chunk_size, max_chunk_elements);

  size_t smem_size =
      P::kFixedSmemLarge + static_cast<size_t>(chunk_size) * sizeof(uint32_t);
  smem_size = std::max(smem_size, P::kSmemMedium);

  int occupancy = 0;
  if (vec_size == 4) {
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &occupancy, P::persistent_topk_kernel<TopK, 4>,
                   P::kThreadsPerBlock, smem_size),
               "occupancy query failed");
  } else if (vec_size == 2) {
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &occupancy, P::persistent_topk_kernel<TopK, 2>,
                   P::kThreadsPerBlock, smem_size),
               "occupancy query failed");
  } else {
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &occupancy, P::persistent_topk_kernel<TopK, 1>,
                   P::kThreadsPerBlock, smem_size),
               "occupancy query failed");
  }
  occupancy = std::max(occupancy, 1);

  const bool needs_cooperative =
      static_cast<uint32_t>(max_seq_len) > P::RADIX_THRESHOLD;
  const uint32_t resident_capacity =
      static_cast<uint32_t>(properties.multiProcessorCount) * occupancy;
  uint32_t usable_ctas = resident_capacity;
  if (needs_cooperative) {
    const uint32_t headroom = occupancy > 1
                                  ? properties.multiProcessorCount
                                  : 1;
    if (usable_ctas >= headroom + ctas_per_group) usable_ctas -= headroom;
  }
  uint32_t num_groups =
      std::min(usable_ctas / ctas_per_group, std::max(num_rows, 1u));
  num_groups = std::max(num_groups, 1u);
  const uint32_t total_ctas = num_groups * ctas_per_group;

  if (needs_cooperative && total_ctas > resident_capacity) {
    TORCH_CHECK(properties.sharedMemPerBlockOptin >= 128 * 1024,
                "exact fallback requires at least 128 KiB shared memory");
    check_cuda(vllm::FilteredTopKRaggedTransform<float, int32_t, TopK>(
                   logits.data_ptr<float>(), output.data_ptr<int32_t>(),
                   lengths.data_ptr<int32_t>(), num_rows, TopK, stride, stream),
               "FilteredTopK fallback failed");
    return;
  }

  const size_t state_bytes = num_groups * sizeof(P::RadixRowState);
  TORCH_CHECK(workspace.numel() >= static_cast<int64_t>(state_bytes),
              "workspace too small: need ", state_bytes, " bytes");
  check_cuda(cudaMemsetAsync(workspace.data_ptr<uint8_t>(), 0, state_bytes,
                             stream),
             "workspace memset failed");

  P::PersistentTopKParams params{};
  params.input = logits.data_ptr<float>();
  params.output = output.data_ptr<int32_t>();
  params.lengths = lengths.data_ptr<int32_t>();
  params.row_states =
      reinterpret_cast<P::RadixRowState*>(workspace.data_ptr<uint8_t>());
  params.num_rows = num_rows;
  params.stride = stride;
  params.top_k = TopK;
  params.chunk_size = chunk_size;
  params.ctas_per_group = ctas_per_group;
  params.max_seq_len = static_cast<uint32_t>(max_seq_len);

#define LAUNCH_PATCHED(VS)                                                    \
  do {                                                                        \
    auto kernel = P::persistent_topk_kernel<TopK, VS>;                        \
    check_cuda(cudaFuncSetAttribute(                                           \
                   kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,       \
                   static_cast<int>(smem_size)),                              \
               "setting dynamic shared memory failed");                     \
    kernel<<<total_ctas, P::kThreadsPerBlock, smem_size, stream>>>(params);    \
  } while (0)

  if (vec_size == 4) {
    LAUNCH_PATCHED(4);
  } else if (vec_size == 2) {
    LAUNCH_PATCHED(2);
  } else {
    LAUNCH_PATCHED(1);
  }
#undef LAUNCH_PATCHED
  check_cuda(cudaGetLastError(), "patched persistent_topk launch failed");
}

void patched_persistent_topk(const torch::Tensor& logits,
                             const torch::Tensor& lengths,
                             torch::Tensor output, torch::Tensor workspace,
                             int64_t k, int64_t max_seq_len) {
  TORCH_CHECK(logits.is_cuda() && lengths.is_cuda() && output.is_cuda() &&
                  workspace.is_cuda(),
              "all tensors must be CUDA tensors");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat32,
              "logits must be float32");
  TORCH_CHECK(lengths.scalar_type() == torch::kInt32,
              "lengths must be int32");
  TORCH_CHECK(output.scalar_type() == torch::kInt32,
              "output must be int32");
  TORCH_CHECK(workspace.scalar_type() == torch::kUInt8,
              "workspace must be uint8");
  TORCH_CHECK(logits.dim() == 2 && output.dim() == 2,
              "logits and output must be 2D");
  TORCH_CHECK(logits.stride(1) == 1, "logits rows must be contiguous");
  TORCH_CHECK(lengths.is_contiguous() && workspace.is_contiguous(),
              "lengths and workspace must be contiguous");
  TORCH_CHECK(lengths.numel() == logits.size(0), "lengths size mismatch");
  TORCH_CHECK(output.size(0) == logits.size(0) && output.size(1) == k,
              "output size mismatch");
  TORCH_CHECK(max_seq_len <= logits.size(1),
              "max_seq_len exceeds row width");

  if (k == 512) {
    launch_patched<512>(logits, lengths, output, workspace, max_seq_len);
  } else if (k == 1024) {
    launch_patched<1024>(logits, lengths, output, workspace, max_seq_len);
  } else if (k == 2048) {
    launch_patched<2048>(logits, lengths, output, workspace, max_seq_len);
  } else {
    TORCH_CHECK(false, "k must be 512, 1024, or 2048");
  }
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("patched_persistent_topk", &patched_persistent_topk,
             "Overflow-safe persistent top-k test backend");
}
