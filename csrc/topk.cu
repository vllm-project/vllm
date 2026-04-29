// Persistent TopK kernel for DeepSeek V3/V4 sparse attention indexer.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <algorithm>

#ifndef USE_ROCM
  #include "persistent_topk.cuh"
#endif

namespace {

#ifndef USE_ROCM

static int g_num_sms = 0;
static int g_max_smem_per_block = 0;

void ensure_device_props() {
  if (g_num_sms == 0) {
    int device;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&g_num_sms, cudaDevAttrMultiProcessorCount, device);
    cudaDeviceGetAttribute(&g_max_smem_per_block,
                           cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  }
}

template <typename scalar_t, int TopK>
void launch_persistent_topk_impl(const scalar_t* input, int32_t* output,
                                  int32_t* lengths, uint8_t* workspace,
                                  int64_t num_rows, int64_t stride,
                                  int64_t max_seq_len, cudaStream_t stream) {
  namespace P = vllm::persistent;
  ensure_device_props();

  constexpr bool is_bf16 = std::is_same_v<scalar_t, __nv_bfloat16>;
  if (num_rows > (is_bf16 ? 64 : 32) && g_max_smem_per_block >= 128 * 1024) {
    cudaError_t status =
        vllm::FilteredTopKRaggedTransform<scalar_t, int32_t, TopK>(
            const_cast<scalar_t*>(input), output, lengths,
            static_cast<uint32_t>(num_rows), static_cast<uint32_t>(TopK),
            static_cast<uint32_t>(stride), stream);
    TORCH_CHECK(status == cudaSuccess,
                "FilteredTopK failed: ", cudaGetErrorString(status));
    return;
  }

  int effective_max_smem;
  if (num_rows <= 4) {
    effective_max_smem =
        std::min(g_max_smem_per_block, static_cast<int>(P::kSmemMedium));
  } else if (num_rows <= 8) {
    constexpr int kSmemCapMedium = 48 * 1024;
    effective_max_smem = std::min(g_max_smem_per_block, kSmemCapMedium);
  } else {
    effective_max_smem = g_max_smem_per_block;
  }

  constexpr size_t kChunkElemBytes = sizeof(uint32_t);

  size_t available_for_ordered =
      static_cast<size_t>(effective_max_smem) - P::kFixedSmemLarge;
  uint32_t max_chunk_elements =
      static_cast<uint32_t>(available_for_ordered / kChunkElemBytes);

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

  constexpr size_t kActualElemBytes =
      is_bf16 ? sizeof(uint16_t) : sizeof(uint32_t);
  size_t smem_size = P::kFixedSmemLarge + chunk_size * kActualElemBytes;
  if (smem_size < P::kSmemMedium) smem_size = P::kSmemMedium;

  int occupancy = 1;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &occupancy, P::persistent_topk_kernel<scalar_t, TopK, 4>,
      P::kThreadsPerBlock, smem_size);
  if (occupancy < 1) occupancy = 1;

  uint32_t max_resident_ctas = static_cast<uint32_t>(g_num_sms) * occupancy;
  uint32_t num_groups = std::min(max_resident_ctas / ctas_per_group,
                                 static_cast<uint32_t>(num_rows));
  if (num_groups == 0) num_groups = 1;
  uint32_t total_ctas = num_groups * ctas_per_group;

  size_t state_bytes = num_groups * sizeof(P::RadixRowState);
  TORCH_CHECK(static_cast<int64_t>(state_bytes) <= 1024 * 1024,
              "workspace too small, need ", state_bytes, " bytes");

  P::PersistentTopKParams<scalar_t> params;
  params.input = input;
  params.output = output;
  params.lengths = lengths;
  params.num_rows = static_cast<uint32_t>(num_rows);
  params.stride = static_cast<uint32_t>(stride);
  params.top_k = static_cast<uint32_t>(TopK);
  params.chunk_size = chunk_size;
  params.row_states =
      reinterpret_cast<P::RadixRowState*>(workspace);
  params.ctas_per_group = ctas_per_group;
  params.max_seq_len = static_cast<uint32_t>(max_seq_len);

#define LAUNCH_PERSISTENT(DTYPE, TOPK_VAL, VS)                                  \
  do {                                                                          \
    auto kernel = &P::persistent_topk_kernel<DTYPE, TOPK_VAL, VS>;              \
    cudaError_t err = cudaFuncSetAttribute(                                     \
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);        \
    TORCH_CHECK(err == cudaSuccess,                                             \
                "Failed to set smem: ", cudaGetErrorString(err));               \
    kernel<<<total_ctas, P::kThreadsPerBlock, smem_size, stream>>>(params);     \
  } while (0)

  if (vec_size == 4) {
    LAUNCH_PERSISTENT(scalar_t, TopK, 4);
  } else if (vec_size == 2) {
    LAUNCH_PERSISTENT(scalar_t, TopK, 2);
  } else {
    LAUNCH_PERSISTENT(scalar_t, TopK, 1);
  }
#undef LAUNCH_PERSISTENT

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
              "persistent_topk failed: ", cudaGetErrorString(err));
}

template <int TopK>
void dispatch_dtype(const torch::Tensor& logits, const torch::Tensor& lengths,
                    torch::Tensor& output, torch::Tensor& workspace,
                    int64_t max_seq_len) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto num_rows = logits.size(0);
  auto stride = logits.size(1);

  if (logits.dtype() == torch::kBFloat16) {
    launch_persistent_topk_impl<__nv_bfloat16, TopK>(
        reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
        output.data_ptr<int32_t>(), lengths.data_ptr<int32_t>(),
        workspace.data_ptr<uint8_t>(), num_rows, stride, max_seq_len, stream);
  } else {
    launch_persistent_topk_impl<float, TopK>(
        logits.data_ptr<float>(), output.data_ptr<int32_t>(),
        lengths.data_ptr<int32_t>(), workspace.data_ptr<uint8_t>(),
        num_rows, stride, max_seq_len, stream);
  }
}

#endif

}  // namespace

void persistent_topk(const torch::Tensor& logits, const torch::Tensor& lengths,
                     torch::Tensor& output, torch::Tensor& workspace, int64_t k,
                     int64_t max_seq_len) {
#ifndef USE_ROCM
  TORCH_CHECK(logits.is_cuda(), "logits must be CUDA tensor");
  TORCH_CHECK(lengths.is_cuda(), "lengths must be CUDA tensor");
  TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
  TORCH_CHECK(logits.dtype() == torch::kFloat32 ||
                  logits.dtype() == torch::kBFloat16,
              "logits must be float32 or bfloat16, got ", logits.dtype());
  TORCH_CHECK(lengths.dtype() == torch::kInt32, "lengths must be int32");
  TORCH_CHECK(output.dtype() == torch::kInt32, "output must be int32");
  TORCH_CHECK(logits.dim() == 2, "logits must be 2D");
  TORCH_CHECK(lengths.dim() == 1 || lengths.dim() == 2,
              "lengths must be 1D or 2D");
  TORCH_CHECK(lengths.is_contiguous(), "lengths must be contiguous");
  TORCH_CHECK(output.dim() == 2, "output must be 2D");

  const int64_t num_rows = logits.size(0);
  const int64_t stride = logits.size(1);

  TORCH_CHECK(lengths.numel() == num_rows, "lengths size mismatch");
  TORCH_CHECK(output.size(0) == num_rows && output.size(1) == k,
              "output size mismatch");
  TORCH_CHECK(k == 512 || k == 1024 || k == 2048,
              "persistent_topk supports k=512, k=1024, or k=2048, got k=", k);

  TORCH_CHECK(workspace.is_cuda(), "workspace must be CUDA tensor");
  TORCH_CHECK(workspace.dtype() == torch::kUInt8, "workspace must be uint8");

  if (k == 512) {
    dispatch_dtype<512>(logits, lengths, output, workspace, max_seq_len);
  } else if (k == 1024) {
    dispatch_dtype<1024>(logits, lengths, output, workspace, max_seq_len);
  } else {
    dispatch_dtype<2048>(logits, lengths, output, workspace, max_seq_len);
  }
#else
  TORCH_CHECK(false, "persistent_topk is not supported on ROCm");
#endif
}
