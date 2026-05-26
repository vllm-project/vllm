#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <algorithm>
#ifndef USE_ROCM
  #include "persistent_topk.cuh"
#endif
namespace ct = vllm::cooperative;

#ifndef USE_ROCM
template <uint32_t TopK, uint32_t CS>
void launch_cooperative_cluster(ct::CooperativeTopKParams<TopK>& params,
                                size_t smem, cudaStream_t stream) {
  auto kernel = (CS == 8) ? &ct::cooperative_topk_cs8<TopK>
                          : &ct::cooperative_topk_cs4<TopK>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       smem);

  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(params.num_rows, CS);
  cfg.blockDim = dim3(ct::kBlockSize);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim = {1, CS, 1};
  cfg.numAttrs = 1;
  cfg.attrs = attrs;
  cudaError_t err = cudaLaunchKernelEx(&cfg, kernel, params);
  TORCH_CHECK(err == cudaSuccess,
              "cooperative_topk launch failed: ", cudaGetErrorString(err));
}

template <uint32_t TopK>
void launch_cooperative_topk_impl(const torch::Tensor& logits,
                              const torch::Tensor& lengths,
                              torch::Tensor& output, torch::Tensor& workspace,
                              int64_t max_seq_len) {
  const int64_t num_rows = logits.size(0);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // 32 = max clusters for CS=4 (32 x 4 = 128 CTAs = 66% of SMs, leaves headroom)
  if (num_rows > 32) {
    cudaError_t status =
        vllm::FilteredTopKRaggedTransform<float, int32_t, TopK>(
            logits.data_ptr<float>(), output.data_ptr<int32_t>(),
            lengths.data_ptr<int32_t>(), static_cast<uint32_t>(num_rows), TopK,
            static_cast<uint32_t>(logits.size(1)), stream);
    TORCH_CHECK(status == cudaSuccess,
                "FilteredTopK failed: ", cudaGetErrorString(status));
    return;
  }

  TORCH_CHECK(workspace.is_cuda(), "workspace must be CUDA tensor");
  TORCH_CHECK(workspace.dtype() == torch::kUInt8, "workspace must be uint8");

  ct::CooperativeTopKParams<TopK> params;
  params.input = logits.data_ptr<float>();
  params.output = output.data_ptr<int32_t>();
  params.lengths = lengths.data_ptr<int32_t>();
  params.num_rows = static_cast<uint32_t>(num_rows);
  params.stride = static_cast<uint32_t>(logits.size(1));
  params.tie_ws =
      reinterpret_cast<ct::Tie*>(workspace.data_ptr<uint8_t>());

  // TODO (roberto): can't the workspace size be smaller now? - only used in large_topk_twopass
  TORCH_CHECK(
      workspace.size(0) >=
          static_cast<int64_t>(num_rows * ct::kMaxTies * sizeof(ct::Tie)),
      "workspace too small");

  if (num_rows <= 8) {
    launch_cooperative_cluster<TopK, 8>(params, ct::kSmemSize8, stream);
  } else {
    launch_cooperative_cluster<TopK, 4>(params, ct::kSmemSize4, stream);
  }
}
#endif  // USE_ROCM

void persistent_topk(const torch::Tensor& logits,
                         const torch::Tensor& lengths, torch::Tensor& output,
                         torch::Tensor& workspace, int64_t k,
                         int64_t max_seq_len) {
  #ifndef USE_ROCM
  TORCH_CHECK(logits.is_cuda(), "logits must be CUDA tensor");
  TORCH_CHECK(lengths.is_cuda(), "lengths must be CUDA tensor");
  TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
  TORCH_CHECK(logits.dtype() == torch::kFloat32, "Only float32 supported");
  TORCH_CHECK(lengths.dtype() == torch::kInt32, "lengths must be int32");
  TORCH_CHECK(output.dtype() == torch::kInt32, "output must be int32");
  TORCH_CHECK(logits.dim() == 2, "logits must be 2D");
  TORCH_CHECK(lengths.dim() == 1 || lengths.dim() == 2,
              "lengths must be 1D or 2D");
  TORCH_CHECK(lengths.is_contiguous(), "lengths must be contiguous");
  TORCH_CHECK(output.dim() == 2, "output must be 2D");
  const int64_t num_rows = logits.size(0);
  TORCH_CHECK(lengths.numel() == num_rows, "lengths size mismatch");
  TORCH_CHECK(output.size(0) == num_rows && output.size(1) == k,
              "output size mismatch");
  TORCH_CHECK(k == 512 || k == 1024 || k == 2048,
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
  TORCH_CHECK(false, "cooperative_topk is not supported on ROCm");
  #endif
}

