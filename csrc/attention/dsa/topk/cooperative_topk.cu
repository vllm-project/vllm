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
template <uint32_t TopK, uint32_t CS, typename InT>
void launch_cooperative_cluster(
    ct::CooperativeTopKParams<TopK, InT>& params, size_t smem,
    cudaStream_t stream,
    void (*kernel)(ct::CooperativeTopKParams<TopK, InT>) = nullptr,
    uint32_t num_clusters = 0, bool pdl = false) {
  if (kernel == nullptr) {
    if constexpr (CS == 2) {
      kernel = &ct::cooperative_topk_cs2<TopK, InT>;
    } else if constexpr (CS == 16) {
      kernel = &ct::cooperative_topk_cs16<TopK, InT>;
    } else if constexpr (CS == 8) {
      kernel = &ct::cooperative_topk_cs8<TopK, InT>;
    } else {
      static_assert(CS == 4, "unsupported cooperative_topk cluster size");
      kernel = &ct::cooperative_topk_cs4<TopK, InT>;
    }
  }
  if (num_clusters == 0) {
    num_clusters = params.num_rows;
  }
  if constexpr (CS > 8) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed,
                         1);
  }
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       smem);

  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(num_clusters, CS);
  cfg.blockDim = dim3(hist4096::kBlockSize);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attrs[2];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim = {1, CS, 1};
  // pdl: may start before the previous kernel on the stream completes (that
  // kernel releases it with griddepcontrol.launch_dependents).
  attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[1].val.programmaticStreamSerializationAllowed = 1;
  cfg.numAttrs = pdl ? 2 : 1;
  cfg.attrs = attrs;
  cudaError_t err = cudaLaunchKernelEx(&cfg, kernel, params);
  STD_TORCH_CHECK(err == cudaSuccess,
                  "cooperative_topk launch failed: ", cudaGetErrorString(err));
}

// Resident CTAs per SM and resident clusters of `kernel`, queried once.
struct MultiOccupancy {
  int ctas_per_sm = 0;
  int clusters = 0;
};

template <typename Kernel>
MultiOccupancy query_multi_occupancy(Kernel kernel, uint32_t cs, size_t smem) {
  MultiOccupancy occ;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       smem);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &occ.ctas_per_sm, kernel, hist4096::kBlockSize, smem);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(1, cs);
  cfg.blockDim = dim3(hist4096::kBlockSize);
  cfg.dynamicSmemBytes = smem;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim = {1, cs, 1};
  cfg.numAttrs = 1;
  cfg.attrs = attrs;
  cudaOccupancyMaxActiveClusters(&occ.clusters, kernel, &cfg);
  return occ;
}

// Batches above 64 rows. Three launches over the same rows; the row lengths
// (read on the device, so the launch shapes depend on the row count only)
// select each launch's work:
//  - rows of at most kHist4096MaxLen scores: one CTA each, two per SM;
//  - few long rows (1..G4-1 rows, G4 = resident 4-CTA clusters): one launch
//    wave of 4-CTA clusters, a long row per cluster;
//  - otherwise the long rows take a 2-CTA cluster each, in a launch of one
//    cluster per row (the hardware starts further clusters as running ones
//    finish, so rows of uneven cost balance).
// The tie workspace is indexed by SM, see kTieWsBySm.
template <uint32_t TopK, typename InT>
void launch_cooperative_multi(ct::CooperativeTopKParams<TopK, InT>& params,
                              int64_t workspace_bytes, cudaStream_t stream) {
  constexpr size_t smem = ct::kSmemSize4<InT>;
  static const MultiOccupancy occ4 = query_multi_occupancy(
      &ct::cooperative_topk_multi_cs4<TopK, InT>, 4, smem);
  static const MultiOccupancy occ2 = query_multi_occupancy(
      &ct::cooperative_topk_multi_cs2<TopK, InT>, 2, smem);
  STD_TORCH_CHECK(occ4.ctas_per_sm == 1 && occ2.ctas_per_sm == 1,
                  "cooperative_topk >64 rows needs exactly one resident CTA "
                  "per SM, got ",
                  occ4.ctas_per_sm, " / ", occ2.ctas_per_sm);
  STD_TORCH_CHECK(occ4.clusters >= 2 && occ2.clusters >= 1,
                  "cooperative_topk >64 rows: no resident cluster");
  const int64_t num_sms = get_device_prop()->multiProcessorCount;
  const int64_t needed = num_sms * static_cast<int64_t>(ct::kCoarseTieCapacity *
                                                         sizeof(hist4096::Tie));
  STD_TORCH_CHECK(workspace_bytes >= needed,
                  "cooperative_topk with >64 rows needs a workspace of at "
                  "least ",
                  needed, " bytes (16 KiB per SM), got ", workspace_bytes);

  const uint32_t g4 = std::min<uint32_t>(occ4.clusters, params.num_rows);
  params.few_long_max = g4 - 1;
  // Short rows: one CTA each, two CTAs per SM.
  constexpr auto short_kernel = &ct::cooperative_topk_short<TopK, InT>;
  constexpr size_t short_smem = ct::kSmemShort<TopK>;
  cudaFuncSetAttribute(short_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, short_smem);
  short_kernel<<<params.num_rows, hist4096::kBlockSize, short_smem, stream>>>(
      params);
  cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess, "cooperative_topk launch failed: ",
                  cudaGetErrorString(err));
  // Both long-row launches overlap the short rows (PDL, see
  // pdl_launch_dependents in cooperative_topk.cuh).
  launch_cooperative_cluster<TopK, 4>(params, smem, stream,
                                      &ct::cooperative_topk_multi_cs4<TopK, InT>,
                                      g4, /*pdl=*/true);
  launch_cooperative_cluster<TopK, 2>(params, smem, stream,
                                      &ct::cooperative_topk_multi_cs2<TopK, InT>,
                                      params.num_rows, /*pdl=*/true);
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
  params.tie_ws =
      reinterpret_cast<hist4096::Tie*>(workspace.mutable_data_ptr<uint8_t>());

  if (num_rows > 64) {
    launch_cooperative_multi<TopK, InT>(params, workspace.size(0), stream);
    return;
  }

  constexpr uint32_t kTieWsPerRow = ct::kCoarseTieCapacity;
  STD_TORCH_CHECK(
      workspace.size(0) >=
          static_cast<int64_t>(num_rows * kTieWsPerRow * sizeof(hist4096::Tie)),
      "workspace too small");

  const bool supports_cluster16 = get_device_prop()->major >= 10;
  if (num_rows <= 4 && supports_cluster16) {
    launch_cooperative_cluster<TopK, 16>(params, ct::kSmemSize8<InT>, stream);
  } else if (num_rows <= 8) {
    launch_cooperative_cluster<TopK, 8>(params, ct::kSmemSize8<InT>, stream);
  } else if (num_rows <= 33) {
    launch_cooperative_cluster<TopK, 4>(params, ct::kSmemSize4<InT>, stream);
  } else {
    launch_cooperative_cluster<TopK, 2>(params, ct::kSmemSize2<InT>, stream);
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
