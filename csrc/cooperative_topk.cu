// Cooperative cluster TopK for DeepSeek V3 sparse attention indexer.
// See cooperative_topk.cuh for kernel implementation.

#include <cuda_runtime.h>

#include "torch_utils.h"

#ifndef USE_ROCM
  #include "cooperative_topk.cuh"
namespace ct = vllm::cooperative;
namespace hist4096 = vllm::topk_histogram_4096;
#endif

#ifndef USE_ROCM
  #if defined(CUDART_VERSION) && CUDART_VERSION >= 13040
constexpr bool kHasOversizedSmem = true;
  #else
constexpr bool kHasOversizedSmem = false;
  #endif

template <uint32_t TopK, uint32_t CS, uint32_t Stages = 0>
void launch_cooperative_cluster(ct::CooperativeTopKParams<TopK>& params,
                                size_t smem, cudaStream_t stream) {
  auto kernel = []() {
    if constexpr (Stages != 0) {
      return &ct::cooperative_topk_staged<TopK, CS, Stages>;
    } else if constexpr (CS == 2) {
      return &ct::cooperative_topk_cs2<TopK>;
    } else if constexpr (CS == 16) {
      return &ct::cooperative_topk_cs16<TopK>;
    } else if constexpr (CS == 8) {
      return &ct::cooperative_topk_cs8<TopK>;
    } else {
      static_assert(CS == 4, "unsupported cooperative_topk cluster size");
      return &ct::cooperative_topk_cs4<TopK>;
    }
  }();
  if constexpr (CS > 8) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed,
                         1);
  }
  const size_t optin = get_device_prop()->sharedMemPerBlockOptin;
  const bool oversized = smem > optin;
  size_t max_dynamic = smem;
  if (oversized) {
    cudaFuncAttributes func_attrs{};
    cudaFuncGetAttributes(&func_attrs, kernel);
    max_dynamic = optin - func_attrs.sharedSizeBytes;
  }
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       max_dynamic);

  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(params.num_rows, CS);
  cfg.blockDim = dim3(hist4096::kBlockSize);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attrs[2];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim = {1, CS, 1};
  cfg.numAttrs = 1;
  #if defined(CUDART_VERSION) && CUDART_VERSION >= 13040
  if (oversized) {
    attrs[1].id = cudaLaunchAttributeSharedMemoryMode;
    attrs[1].val.sharedMemoryMode =
        cudaSharedMemoryModeAllowOversizedSharedMemory;
    cfg.numAttrs = 2;
  }
  #endif
  cfg.attrs = attrs;
  cudaError_t err = cudaLaunchKernelEx(&cfg, kernel, params);
  STD_TORCH_CHECK(err == cudaSuccess,
                  "cooperative_topk launch failed: ", cudaGetErrorString(err));
}

template <uint32_t TopK>
void launch_cooperative_topk_impl(const torch::stable::Tensor& logits,
                                  const torch::stable::Tensor& lengths,
                                  torch::stable::Tensor& output,
                                  torch::stable::Tensor& workspace,
                                  int64_t max_seq_len) {
  (void)max_seq_len;  // Kept for signature parity with persistent_topk.
  const int64_t num_rows = logits.size(0);
  const cudaStream_t stream = get_current_cuda_stream();

  const uint32_t stride = static_cast<uint32_t>(logits.stride(0));
  STD_TORCH_CHECK(
      num_rows <= 64,
      "cooperative_topk supports <=64 rows; use persistent_topk for "
      "larger batches");

  STD_TORCH_CHECK(stride % 4 == 0,
                  "cooperative_topk: stride must be multiple of 4 for TMA "
                  "alignment, got stride (max_model_len)=",
                  stride);

  (void)workspace;

  ct::CooperativeTopKParams<TopK> params;
  params.input = logits.const_data_ptr<float>();
  params.output = output.mutable_data_ptr<int32_t>();
  params.lengths = lengths.const_data_ptr<int32_t>();
  params.num_rows = static_cast<uint32_t>(num_rows);
  params.stride = stride;

  const cudaDeviceProp* prop = get_device_prop();
  const bool supports_cluster16 = prop->major >= 10;
  const bool rubin_long_stride = prop->major == 10 && prop->minor == 7 &&
                                 stride > ct::kFusedMaxStrideCS16;
  if (num_rows <= 4 && rubin_long_stride) {
    launch_cooperative_cluster<TopK, 24, 3>(params, ct::kSmemSizeStaged3,
                                            stream);
  } else if (num_rows <= 4 && supports_cluster16) {
    launch_cooperative_cluster<TopK, 16>(params, ct::kSmemSize8, stream);
  } else if (num_rows <= 8 && rubin_long_stride && kHasOversizedSmem) {
    launch_cooperative_cluster<TopK, 16, 4>(params, ct::kSmemSizeStaged4,
                                            stream);
  } else if (num_rows <= 8) {
    launch_cooperative_cluster<TopK, 8>(params, ct::kSmemSize8, stream);
  } else if (num_rows <= 33) {
    launch_cooperative_cluster<TopK, 4>(params, ct::kSmemSize4, stream);
  } else {
    launch_cooperative_cluster<TopK, 2>(params, ct::kSmemSize2, stream);
  }
}
#endif  // USE_ROCM

void cooperative_topk(const torch::stable::Tensor& logits,
                      const torch::stable::Tensor& lengths,
                      torch::stable::Tensor& output,
                      torch::stable::Tensor& workspace, int64_t k,
                      int64_t max_seq_len) {
#ifndef USE_ROCM
  STD_TORCH_CHECK(logits.is_cuda(), "logits must be CUDA tensor");
  STD_TORCH_CHECK(lengths.is_cuda(), "lengths must be CUDA tensor");
  STD_TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
  STD_TORCH_CHECK(logits.scalar_type() == torch::headeronly::ScalarType::Float,
                  "Only float32 supported");
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
    launch_cooperative_topk_impl<512>(logits, lengths, output, workspace,
                                      max_seq_len);
  } else if (k == 1024) {
    launch_cooperative_topk_impl<1024>(logits, lengths, output, workspace,
                                       max_seq_len);
  } else {
    launch_cooperative_topk_impl<2048>(logits, lengths, output, workspace,
                                       max_seq_len);
  }
#else
  STD_TORCH_CHECK(false, "cooperative_topk is not supported on ROCm");
#endif
}
