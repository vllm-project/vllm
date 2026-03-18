// Persistent TopK kernel for DeepSeek V3 sparse attention indexer.
// See persistent_topk.cuh for kernel implementation.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <algorithm>
#include "persistent_topk.cuh"

void persistent_topk(const torch::Tensor& logits, const torch::Tensor& lengths,
                     torch::Tensor& output, torch::Tensor& workspace, int64_t k,
                     int64_t max_seq_len) {
  TORCH_CHECK(logits.is_cuda(), "logits must be CUDA tensor");
  TORCH_CHECK(lengths.is_cuda(), "lengths must be CUDA tensor");
  TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
  TORCH_CHECK(workspace.is_cuda(), "workspace must be CUDA tensor");
  TORCH_CHECK(logits.dtype() == torch::kFloat32, "Only float32 supported");
  TORCH_CHECK(lengths.dtype() == torch::kInt32, "lengths must be int32");
  TORCH_CHECK(output.dtype() == torch::kInt32, "output must be int32");
  TORCH_CHECK(workspace.dtype() == torch::kUInt8, "workspace must be uint8");
  TORCH_CHECK(logits.dim() == 2, "logits must be 2D");
  TORCH_CHECK(lengths.dim() == 1, "lengths must be 1D");
  TORCH_CHECK(output.dim() == 2, "output must be 2D");

  const int64_t num_rows = logits.size(0);
  const int64_t stride = logits.size(1);

  TORCH_CHECK(lengths.size(0) == num_rows, "lengths size mismatch");
  TORCH_CHECK(output.size(0) == num_rows && output.size(1) == k,
              "output size mismatch");
  namespace P = vllm::persistent;

  TORCH_CHECK(k == P::TopK, "k must be 2048");
  TORCH_CHECK(k <= stride, "k out of range");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Query device properties (cached)
  static int num_sms = 0;
  static int max_smem_per_block = 0;
  if (num_sms == 0) {
    int device;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);
    cudaDeviceGetAttribute(&max_smem_per_block,
                           cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  }

  (void)max_seq_len;  // path selection is now per-row inside the kernel

  // Compute grid configuration for worst-case (large path).
  // The kernel dynamically dispatches per-row: medium rows use only CTA 0
  // of each group, large rows use all CTAs in the group.
  // This fixed grid is CUDAGraph-safe — no host-side path branching.
  size_t available_for_ordered =
      static_cast<size_t>(max_smem_per_block) - P::kFixedSmemLarge;
  uint32_t max_chunk_elements =
      static_cast<uint32_t>(available_for_ordered / sizeof(uint32_t));

  uint32_t vec_size = 1;
  if (stride % 4 == 0)
    vec_size = 4;
  else if (stride % 2 == 0)
    vec_size = 2;

  max_chunk_elements = (max_chunk_elements / vec_size) * vec_size;
  uint32_t min_chunk = vec_size * P::kThreadsPerBlock;
  if (max_chunk_elements < min_chunk) max_chunk_elements = min_chunk;

  uint32_t ctas_per_group =
      (static_cast<uint32_t>(stride) + max_chunk_elements - 1) /
      max_chunk_elements;
  uint32_t chunk_size =
      (static_cast<uint32_t>(stride) + ctas_per_group - 1) / ctas_per_group;
  chunk_size = ((chunk_size + vec_size - 1) / vec_size) * vec_size;
  if (chunk_size > max_chunk_elements) chunk_size = max_chunk_elements;

  uint32_t num_groups =
      std::min(static_cast<uint32_t>(num_sms) / ctas_per_group,
               static_cast<uint32_t>(num_rows));
  if (num_groups == 0) num_groups = 1;
  uint32_t total_ctas = num_groups * ctas_per_group;

  // Shared memory: max of large path needs and medium path needs
  size_t smem_size = P::kFixedSmemLarge + chunk_size * sizeof(uint32_t);
  if (smem_size < P::kSmemMedium) smem_size = P::kSmemMedium;

  size_t state_bytes = num_groups * sizeof(P::RadixRowState);
  TORCH_CHECK(workspace.size(0) >= static_cast<int64_t>(state_bytes),
              "workspace too small, need ", state_bytes, " bytes");

  P::PersistentTopKParams params;
  params.input = logits.data_ptr<float>();
  params.output = output.data_ptr<int32_t>();
  params.lengths = lengths.data_ptr<int32_t>();
  params.num_rows = static_cast<uint32_t>(num_rows);
  params.stride = static_cast<uint32_t>(stride);
  params.row_states =
      reinterpret_cast<P::RadixRowState*>(workspace.data_ptr<uint8_t>());
  params.chunk_size = chunk_size;
  params.ctas_per_group = ctas_per_group;

#define LAUNCH_PERSISTENT(VS)                                                 \
  do {                                                                        \
    void (*kernel)(P::PersistentTopKParams) = &P::persistent_topk_kernel<VS>; \
    cudaError_t err = cudaFuncSetAttribute(                                   \
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);      \
    TORCH_CHECK(err == cudaSuccess,                                           \
                "Failed to set smem: ", cudaGetErrorString(err));             \
    kernel<<<total_ctas, P::kThreadsPerBlock, smem_size, stream>>>(params);   \
  } while (0)

  if (vec_size == 4) {
    LAUNCH_PERSISTENT(4);
  } else if (vec_size == 2) {
    LAUNCH_PERSISTENT(2);
  } else {
    LAUNCH_PERSISTENT(1);
  }

#undef LAUNCH_PERSISTENT

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
              "persistent_topk failed: ", cudaGetErrorString(err));
}
