// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Standalone TRT-LLM-gen batched expert GEMM (FlashInfer cubins): block-scaled
// NVFP4 / MXFP8 weights x expert-sorted, tile-padded activations -> bf16, with
// an optional fp32 per-row output scale. Built by FlashInfer's JIT and linked
// against FlashInfer's TRT-LLM-gen MoE module, which holds the runner.

#include <tvm/ffi/container/array.h>
#include <tvm_ffi_utils.h>

#include <map>
#include <memory>

#include "flashinfer/trtllm/batched_gemm/KernelRunner.h"

namespace inkling2 {

using tensorrt_llm::kernels::TrtllmGenBatchedGemmRunner;
using tensorrt_llm::kernels::TrtllmGenBatchedGemmRunnerOptions;
using tvm::ffi::Array;
using tvm::ffi::Optional;
namespace btg = batchedGemm::trtllm::gen;

namespace {

TrtllmGenBatchedGemmRunner& getRunner(bool nvfp4, int64_t tileN) {
  static std::map<std::pair<bool, int64_t>,
                  std::unique_ptr<TrtllmGenBatchedGemmRunner>>
      runners;
  auto& runner = runners[{nvfp4, tileN}];
  if (!runner) {
    auto const dtype = nvfp4 ? btg::Dtype::E2m1 : btg::Dtype::MxE4m3;
    TrtllmGenBatchedGemmRunnerOptions options{};
    options.dtypeA = dtype;
    options.dtypeB = dtype;
    options.dtypeC = btg::Dtype::Bfloat16;
    options.transposeMmaOutput = true;
    options.tileSize = static_cast<int32_t>(tileN);
    options.epilogueTileM = 128;
    options.useShuffledMatrix = true;
    options.usePerTokenScaling = nvfp4;
    options.perTokenSfDtype = nvfp4 ? btg::Dtype::Fp32 : btg::Dtype::Void;
    runner = std::make_unique<TrtllmGenBatchedGemmRunner>(options);
  }
  return *runner;
}

}  // namespace

// Indices of the configs of a (format, token tile) runner and their B scale
// layouts.
Array<int64_t> bmm_configs(bool nvfp4, int64_t tile_n) {
  auto& runner = getRunner(nvfp4, tile_n);
  Array<int64_t> configs;
  for (int64_t index : runner.getPassingConfigIndices()) {
    configs.push_back(index);
    configs.push_back(static_cast<int64_t>(runner.getSfLayoutB(index)));
  }
  return configs;
}

// Whether config ``index`` runs [num_experts, n, k] weights with up to
// ``max_ctas`` token tiles, and the workspace bytes it needs (-1: invalid).
int64_t bmm_workspace_size(bool nvfp4, int64_t tile_n, int64_t index, int64_t n,
                           int64_t k, int64_t num_experts, int64_t max_ctas) {
  auto& runner = getRunner(nvfp4, tile_n);
  int32_t const tokens = static_cast<int32_t>(max_ctas * tile_n);
  if (!runner.isValidConfigIndex(index, tokens, n, k, {}, tokens, num_experts,
                                 max_ctas)) {
    return -1;
  }
  return runner.getWorkspaceSizeInBytes(tokens, n, k, {}, tokens, num_experts,
                                        max_ctas, index);
}

// out[r] = bf16((act[r] @ weight[e(r)]^T) * row_scale[r]), where row r of the
// tile-padded activations belongs to expert cta_to_expert[r / tile_n].
void bmm_run(bool nvfp4, int64_t tile_n, int64_t index, TensorView weight,
             TensorView weight_scale, TensorView act, TensorView act_scale,
             Optional<TensorView> row_scale, TensorView ones, TensorView out,
             TensorView cta_to_expert, TensorView cta_row_limit,
             TensorView num_ctas, TensorView num_padded_rows,
             int64_t max_ctas) {
  int32_t const num_experts = weight.size(0);
  int32_t const n = weight.size(1);
  int32_t const k = act.size(1) * (nvfp4 ? 2 : 1);
  int32_t const tokens = static_cast<int32_t>(max_ctas * tile_n);
  TVM_FFI_ICHECK(act.size(0) >= tokens && out.size(0) >= tokens &&
                 out.size(1) == n);
  getRunner(nvfp4, tile_n)
      .run(tokens, n, k, {}, tokens, num_experts,
           static_cast<int32_t>(max_ctas), act.data_ptr(), act_scale.data_ptr(),
           weight.data_ptr(), weight_scale.data_ptr(),
           row_scale.has_value() ? row_scale.value().data_ptr() : nullptr,
           /*perTokensSfB=*/nullptr, static_cast<float const*>(ones.data_ptr()),
           /*scaleGateC=*/nullptr, /*bias=*/nullptr, /*alpha=*/nullptr,
           /*beta=*/nullptr,
           /*clampLimit=*/nullptr, out.data_ptr(), /*outSfC=*/nullptr,
           /*routeMap=*/nullptr,
           static_cast<int32_t const*>(num_padded_rows.data_ptr()),
           static_cast<int32_t const*>(cta_to_expert.data_ptr()),
           static_cast<int32_t const*>(cta_row_limit.data_ptr()),
           static_cast<int32_t const*>(num_ctas.data_ptr()),
           /*permutedIdxToBiasRowIdx=*/nullptr,
           /*workspace=*/nullptr, get_stream(act.device()),
           act.device().device_id, index, /*enable_pdl=*/true);
}

}  // namespace inkling2

TVM_FFI_DLL_EXPORT_TYPED_FUNC(bmm_configs, inkling2::bmm_configs);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(bmm_workspace_size, inkling2::bmm_workspace_size);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(bmm_run, inkling2::bmm_run);
