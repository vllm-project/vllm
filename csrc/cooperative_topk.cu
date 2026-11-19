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
constexpr bool kCudaSupportsOversizedSmem = true;
  #else
constexpr bool kCudaSupportsOversizedSmem = false;
  #endif

template <uint32_t TopK, uint32_t CS, typename InT, bool kOversizedSmem = false>
void launch_cooperative_cluster(ct::CooperativeTopKParams<TopK, InT>& params,
                                size_t smem, cudaStream_t stream) {
  auto kernel = []() {
    if constexpr (CS == 2) {
      return &ct::cooperative_topk_cs2<TopK, InT>;
    } else if constexpr (CS == 24) {
      return &ct::cooperative_topk_cs24<TopK, InT>;
    } else if constexpr (CS == 16 && kOversizedSmem) {
      return &ct::cooperative_topk_cs16_oversized<TopK, InT>;
    } else if constexpr (CS == 16) {
      return &ct::cooperative_topk_cs16<TopK, InT>;
    } else if constexpr (CS == 8) {
      return &ct::cooperative_topk_cs8<TopK, InT>;
    } else {
      static_assert(CS == 4, "unsupported cooperative_topk cluster size");
      return &ct::cooperative_topk_cs4<TopK, InT>;
    }
  }();
  cudaError_t err = cudaSuccess;
  if constexpr (CS > 8) {
    err = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  }
  size_t max_dynamic_smem = smem;
  if constexpr (kOversizedSmem) {
    cudaFuncAttributes func_attrs{};
    if (err == cudaSuccess) {
      err = cudaFuncGetAttributes(&func_attrs, kernel);
    }
    max_dynamic_smem =
        get_device_prop()->sharedMemPerBlockOptin - func_attrs.sharedSizeBytes;
  }
  if (err == cudaSuccess) {
    err = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem);
  }
  if (err != cudaSuccess) {
    (void)cudaGetLastError();
    if constexpr (kOversizedSmem) {
      launch_cooperative_cluster<TopK, CS, InT>(params, ct::kSmemSize8<InT>,
                                                stream);
      return;
    }
  }
  STD_TORCH_CHECK(err == cudaSuccess, "cooperative_topk kernel setup failed: ",
                  cudaGetErrorString(err));

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
  if constexpr (kOversizedSmem) {
    attrs[1].id = cudaLaunchAttributeSharedMemoryMode;
    attrs[1].val.sharedMemoryMode =
        cudaSharedMemoryModeAllowOversizedSharedMemory;
    cfg.numAttrs = 2;
  }
  #endif
  cfg.attrs = attrs;
  err = cudaLaunchKernelEx(&cfg, kernel, params);
  STD_TORCH_CHECK(err == cudaSuccess,
                  "cooperative_topk launch failed: ", cudaGetErrorString(err));
}

template <uint32_t TopK, typename InT>
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

  const cudaDeviceProp* prop = get_device_prop();
  const bool supports_cluster16 = prop->major >= 10;
  const bool supports_cluster24 = prop->major == 10 && prop->minor == 7;
  const bool supports_oversized_smem =
      supports_cluster24 && kCudaSupportsOversizedSmem;
  int64_t max_rows_cs24 = 0, max_rows_cs16 = 4, max_rows_cs8 = 8,
          max_rows_cs4 = 33;
  if (supports_cluster24) {
    max_rows_cs24 = 6;
    max_rows_cs16 = 8;
    max_rows_cs8 = 22;
    max_rows_cs4 = 46;
  } else if ((prop->major == 9 || prop->major == 10) &&
             prop->multiProcessorCount >= 132) {
    max_rows_cs8 = 15;
  }
  const bool long_rows = logits.size(1) > ct::kFusedMaxLenCS16<InT>;
  if (num_rows <= max_rows_cs24 && long_rows) {
    launch_cooperative_cluster<TopK, 24, InT>(params, ct::kSmemSize24<InT>,
                                              stream);
  } else if (num_rows <= max_rows_cs16 && long_rows &&
             supports_oversized_smem) {
    launch_cooperative_cluster<TopK, 16, InT, true>(
        params, ct::kSmemSize16Oversized<InT>, stream);
  } else if (num_rows <= max_rows_cs16 && supports_cluster16) {
    launch_cooperative_cluster<TopK, 16, InT>(params, ct::kSmemSize8<InT>,
                                              stream);
  } else if (num_rows <= max_rows_cs8) {
    launch_cooperative_cluster<TopK, 8, InT>(params, ct::kSmemSize8<InT>,
                                             stream);
  } else if (num_rows <= max_rows_cs4) {
    launch_cooperative_cluster<TopK, 4, InT>(params, ct::kSmemSize4<InT>,
                                             stream);
  } else {
    launch_cooperative_cluster<TopK, 2, InT>(params, ct::kSmemSize2<InT>,
                                             stream);
  }
}

template <uint32_t TopK>
void launch_cooperative_topk_dtype(const torch::stable::Tensor& logits,
                                   const torch::stable::Tensor& lengths,
                                   torch::stable::Tensor& output,
                                   torch::stable::Tensor& workspace,
                                   int64_t max_seq_len) {
  if (logits.scalar_type() == torch::headeronly::ScalarType::BFloat16) {
    launch_cooperative_topk_impl<TopK, __nv_bfloat16>(logits, lengths, output,
                                                      workspace, max_seq_len);
  } else {
    launch_cooperative_topk_impl<TopK, float>(logits, lengths, output,
                                              workspace, max_seq_len);
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
                                       max_seq_len);
  } else if (k == 1024) {
    launch_cooperative_topk_dtype<1024>(logits, lengths, output, workspace,
                                        max_seq_len);
  } else {
    launch_cooperative_topk_dtype<2048>(logits, lengths, output, workspace,
                                        max_seq_len);
  }
#else
  STD_TORCH_CHECK(false, "cooperative_topk is not supported on ROCm");
#endif
}
