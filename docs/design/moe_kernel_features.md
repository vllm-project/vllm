# Fused MoE Kernel Features

The purpose of this document is to provide an overview of the various MoE kernels (both modular and non-modular) so it will be easier to select an appropriate set of kernels for any particular situation. This includes information about the all2all backends used by modular kernels.

## Fused gate routing

`MoERunner` queries `router.can_fuse_gate(hidden_states)` before applying the
gate. The router interface defaults to `False`. `FusedGateRouter` selects
the first supported entry from its ordered fused-gate backends. If every
backend declines, the runner applies the gate and uses the existing top-k path.
The gate backend is independent of the expert backend.

`create_fused_moe_router()` selects `FusedGateRouter` for bias/hash routing when
a gate projection is provided. Other bias/hash callers use `FusedTopKBiasRouter`.
`FusedGateRouter` extends that router's routed top-k computation and reuses its
fallback and shared-expert handling. Backend selection and gate metadata live
in the new router; grouped, custom, CPU and zero-expert routing keep their
existing selection priority.

The backend contract lives in
`vllm/model_executor/layers/fused_moe/router/fused_gate.py`:

- `FusedGateConfig` describes scoring, top-k, normalization, scaling, biases,
  EP rank, and whether hash routing is required.
- `FusedGateMetadata` carries per-batch image, valid-token and fixed-routing
  masks plus logical fixed expert IDs. The router builds it after selection.
- `FusedGateBackend.is_supported()` checks input geometry, device, routing
  modes, padding and capture readiness without allocating device buffers,
  launching kernels or compiling programs.
- `FusedGateBackend.prepare()` binds a call or reuses a prepared operation.
  `FusedGatePlan.forward()` returns `(weights, logical_expert_ids)`, with FP32
  weights and signed IDs. Padding rows must contain zero weights and `-1` IDs.

Backends return only the routed slots and must not apply physical expert mapping
themselves. The modular router owns shared slots, EPLB mapping, logical routing
capture and index dtype conversion. Native MegaMoE performs the corresponding
post-processing in its experts/input-staging layer.

`FusedGateRouter.set_fused_gate_backends(backends)` installs per-layer
backends in priority order before warmup and CUDA Graph capture; an empty list
disables fusion for that router. Modular routers start with an empty list and
retain their existing gate and top-k path until a backend is installed.
Library-specific switches and eligibility checks belong to the adapters.

This contract also accommodates FlashInfer's `prepare_mega_gate()` / `plan.run()`
API. An adapter exposes execution as `forward()`, reverses FlashInfer's
`(ids, weights)` output order and owns its reusable plans and workspaces.
Address-bound plan caches must distinguish
tensor addresses and layouts, routing configuration, device and stream, and
retain buffers for the lifetime of any captured graphs using them. Preparation
during capture must reuse warmed resources without device queries or JIT.
Unsupported modes must be declined before the runner skips the separate gate.
A FlashInfer adapter can be registered through the same interface.

### DeepSeek V4/V4.1 native MegaMoE

The existing SM100 `bf16_mega_gate` call in `DeepseekV4MoE`, also inherited by
V4.1, uses the same `select_fused_gate_backend()` / `prepare()` / `forward()`
contract. Its candidates come from `default_fused_gate_backends()`,
currently `DeepGemmGateBackend`, which supports SM100 only. Adding another
adapter to this list does not require a library-specific branch in the model
forward pass.

The model retains its small-batch `GateLinear` thresholds (one token for
128 experts, otherwise 16 tokens), and reuses the hash/image routing metadata
prepared once by the model. Above the threshold, unsupported configurations or
an empty candidate list fall back to the separate gate/top-k path. The adapter
forwards the original routing arguments to `bf16_mega_gate`, including image
bias and fixed hash routes. Shared experts and EPLB retain their existing
post-processing.

The DeepGEMM adapter uses the existing `bf16_mega_gate` wrapper and declines
gate-level padding masks. Native MegaMoE handles padding in its input staging;
modular routing with padding falls back to the separate gate/top-k path.

## Fused MoE Modular All2All backends

There are a number of all2all communication backends that are used to implement expert parallelism (EP) for the `MoERunner` layer. The different `FusedMoEPrepareAndFinalizeModular` subclasses provide an interface for each all2all backend.

The following table describes the relevant features of each backend, i.e. activation format, supported quantization schemes and async support.

The output activation format (standard or batched) corresponds to the output of the prepare step of the `FusedMoEPrepareAndFinalizeModular` subclass, and the finalize step requires the same format. All the backend `prepare` methods expect activations in the standard format and all the `finalize` methods return activations in standard format. More details on the formats can be found in the [Fused MoE Modular Kernel](./fused_moe_modular_kernel.md) document.

The quantization types and formats enumerate which quantization schemes are supported by each `FusedMoEPrepareAndFinalizeModular` class. The quantization can happen before or after the dispatch based on the format the all2all backend supports, e.g. deepep_high_throughput supports only block-quantized fp8 format. Any other format will result in dispatching in higher precision and quantizing afterwards. The output of the prepare step for each backend is the quantized type. The finalize step generally requires the same input type as the original activations, e.g. if the original input is bfloat16 and the quantization scheme is fp8 with per-tensor scales, `prepare` will return fp8/per-tensor scale activations and `finalize` will take bfloat16 activations. See the diagrams in [Fused MoE Modular Kernel](./fused_moe_modular_kernel.md) for more details on the types and formats of activations at each step of the MoE process. If no quantization type is specified, the kernel operates on float16 and/or bfloat16.

Async backends support the use of DBO (Dual Batch Overlap) and shared expert overlap (where shared experts are computed during the combine step).

Certain models require the topk weights to be applied to the input activations rather than the output activations when topk==1, e.g. Llama. For modular kernels, this feature is supported by the `FusedMoEPrepareAndFinalizeModular` subclass. For non-modular kernels, it is up to the experts function to deal with this flag.

Unless otherwise specified, backends are controlled via the `--all2all-backend` command-line argument (or the `all2all_backend` parameter in `ParallelConfig`). All backends except `flashinfer` only work with EP+DP or EP+TP. `Flashinfer` can work with EP or DP without EP.

<style>
td {
  padding: 0.5rem !important;
  white-space: nowrap;
}

th {
  padding: 0.5rem !important;
  min-width: 0 !important;
}
</style>

| Backend | Output act. format | Quant. types | Quant. format | Async | Apply Weight On Input | Subclass |
| ------- | ------------------ | ------------ | ------------- | ----- | --------------------- | --------- |
| naive | standard | all<sup>1</sup> | G,A,T | N | <sup>6</sup> | [`MoERunner`][vllm.model_executor.layers.fused_moe.runner.moe_runner.MoERunner] |
| deepep_high_throughput | standard | fp8 | G(128),A,T<sup>2</sup> | Y | Y | [`DeepEPHTPrepareAndFinalize`][vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ht.DeepEPHTPrepareAndFinalize] |
| deepep_low_latency | batched | fp8 | G(128),A,T<sup>3</sup> | Y | Y | [`DeepEPLLPrepareAndFinalize`][vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ll.DeepEPLLPrepareAndFinalize] |
| flashinfer_nvlink_two_sided | standard | nvfp4,fp8 | G,A,T | N | N | [`FlashInferNVLinkTwoSidedPrepareAndFinalize`][vllm.model_executor.layers.fused_moe.prepare_finalize.flashinfer_nvlink_two_sided.FlashInferNVLinkTwoSidedPrepareAndFinalize] |
| flashinfer_nvlink_one_sided | standard | nvfp4,bf16,mxfp8,fp8 | G,A,T | N | N | [`FlashInferNVLinkOneSidedPrepareAndFinalize`][vllm.model_executor.layers.fused_moe.prepare_finalize.flashinfer_nvlink_one_sided.FlashInferNVLinkOneSidedPrepareAndFinalize] |
| passthrough | standard | nvfp4,mxfp4 | G | N | Y | [`PassThroughPrepareAndFinalize`][vllm.model_executor.layers.fused_moe.prepare_finalize.passthrough.PassThroughPrepareAndFinalize] |

!!! info "Table key"
    1. All types: mxfp4, nvfp4, int4, int8, fp8
    2. A,T quantization occurs after dispatch.
    3. All quantization happens after dispatch.
    4. Controlled by `--moe-backend` (`flashinfer_cutlass` or `flashinfer_trtllm`)
    5. This is a no-op dispatcher that can be used to pair with any modular experts to produce a modular kernel that runs without dispatch or combine. These cannot be selected via environment variable. These are generally use for testing or adapting an expert subclass to the `fused_experts` API.
    6. This depends on the experts implementation.

    ---

    - G - Grouped
    - G(N) - Grouped w/block size N
    - A - Per activation token
    - T - Per tensor

Modular kernels are supported by the following `FusedMoEMethodBase` classes.

- [`ModelOptFp8MoEMethod`][vllm.model_executor.layers.quantization.modelopt.ModelOptFp8MoEMethod]
- [`Fp8MoEMethod`][vllm.model_executor.layers.quantization.fp8.Fp8MoEMethod]
- [`CompressedTensorsW4A4Nvfp4MoEMethod`][vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_nvfp4.CompressedTensorsW4A4Nvfp4MoEMethod]
- [`CompressedTensorsW8A8Fp8MoEMethod`][vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_fp8.CompressedTensorsW8A8Fp8MoEMethod]
- [`GptOssMxfp4MoEMethod`][vllm.model_executor.layers.quantization.mxfp4.GptOssMxfp4MoEMethod]
- [`UnquantizedFusedMoEMethod`][vllm.model_executor.layers.fused_moe.UnquantizedFusedMoEMethod]

## Fused Experts Kernels

There are a number of MoE experts kernel implementations for different quantization types and architectures. Most follow the general API of the base Triton [`fused_experts`][vllm.model_executor.layers.fused_moe.fused_moe.fused_experts] function. Many have modular kernel adapters, so they can be used with compatible all2all backends. This table lists each experts kernel and its particular properties.

Each kernel must be provided with one of the supported input activation formats. Some flavors of kernels support both standard and batched formats through different entry points, e.g. `TritonExperts` and `BatchedTritonExperts`. Batched format kernels are currently only needed for matching with certain all2all backends, e.g. `DeepEPLLPrepareAndFinalize`.

Similar to the backend kernels, each experts kernel only supports certain quantization formats. For non-modular experts, the activations will be in the original type and quantized internally by the kernel. Modular experts will expect the activations to already be in the quantized format. Both types of experts will yield outputs in the original activation type.

Each experts kernel supports one or more activation functions, e.g. silu or gelu, which are applied to the intermediate results.

As with the backends, some experts support applying topk weights on the input activations. The entries in the column in this table only apply to the non-modular experts.

Most experts flavors include an equivalent modular interface which will be a subclass of `FusedMoEExpertsModular`.

To be used with a particular `FusedMoEPrepareAndFinalizeModular` subclass, MoE kernels must have compatible activation formats, quantization types and quantization formats.

| Kernel | Input act. format | Quant. types | Quant. format | Activation function | Apply Weight On Input | Modular | Source |
| ------ | ----------------- | ------------ | ------------- | ------------------- | --------------------- | ------- | ------ |
| triton | standard | all<sup>1</sup> | G,A,T | silu, gelu,</br>swigluoai,</br>silu_no_mul,</br>gelu_no_mul | Y | Y | [`fused_experts`][vllm.model_executor.layers.fused_moe.fused_moe.fused_experts],</br>[`TritonExperts`][vllm.model_executor.layers.fused_moe.experts.triton_moe.TritonExperts] |
| triton (batched) | batched | all<sup>1</sup> | G,A,T | silu, gelu | <sup>6</sup> | Y | [`BatchedTritonExperts`][vllm.model_executor.layers.fused_moe.experts.fused_batched_moe.BatchedTritonExperts] |
| deep gemm | standard,</br>batched | fp8 | G(128),A,T | silu, gelu | <sup>6</sup> | Y | </br>[`DeepGemmExperts`][vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe.DeepGemmExperts],</br>[`BatchedDeepGemmExperts`][vllm.model_executor.layers.fused_moe.experts.batched_deep_gemm_moe.BatchedDeepGemmExperts] |
| cutlass_fp4 | standard,</br>batched | nvfp4 | A,T | silu | Y | Y | [`CutlassExpertsFp4`][vllm.model_executor.layers.fused_moe.experts.cutlass_moe.CutlassExpertsFp4] |
| cutlass_fp8 | standard,</br>batched | fp8 | A,T | silu, gelu | Y | Y | [`CutlassExpertsFp8`][vllm.model_executor.layers.fused_moe.experts.cutlass_moe.CutlassExpertsFp8],</br>[`CutlasBatchedExpertsFp8`][vllm.model_executor.layers.fused_moe.experts.cutlass_moe.CutlassBatchedExpertsFp8] |
| flashinfer | standard | nvfp4,</br>fp8 | T | <sup>5</sup> | N | Y | [`FlashInferExperts`][vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe.FlashInferExperts] |
| gpt oss triton | standard | N/A | N/A | <sup>5</sup> | Y | Y | [`triton_kernel_fused_experts`][vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe.triton_kernel_fused_experts],</br>[`OAITritonExperts`][vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe.OAITritonExperts] |
| marlin | standard,</br>batched | <sup>3</sup> / N/A | <sup>3</sup> / N/A | silu,</br>swigluoai | Y | Y | [`fused_marlin_moe`][vllm.model_executor.layers.fused_moe.experts.marlin_moe.fused_marlin_moe],</br>[`MarlinExperts`][vllm.model_executor.layers.fused_moe.experts.marlin_moe.MarlinExperts],</br>[`BatchedMarlinExperts`][vllm.model_executor.layers.fused_moe.experts.marlin_moe.BatchedMarlinExperts] |
| trtllm | standard | mxfp4,</br>nvfp4 | G(16),G(32) | <sup>5</sup> | N | Y | [`TrtLlmMxfp4ExpertsMonolithic`][vllm.model_executor.layers.fused_moe.experts.trtllm_mxfp4_moe.TrtLlmMxfp4ExpertsMonolithic],</br>[`TrtLlmMxfp4ExpertsModular`][vllm.model_executor.layers.fused_moe.experts.trtllm_mxfp4_moe.TrtLlmMxfp4ExpertsModular],</br>[`TrtLlmNvFp4ExpertsMonolithic`][vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe.TrtLlmNvFp4ExpertsMonolithic],</br>[`TrtLlmNvfp4ExpertsModular`][vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe.TrtLlmNvFp4ExpertsModular] |
| hpc | standard | fp8 | G(128),T | silu | Y | Y | [`HPCExperts`][vllm.model_executor.layers.fused_moe.hpc_moe.HPCExperts] |
| rocm aiter moe | standard | none,</br>mxfp4,</br>fp8 | G(32),G(128),A,T | silu, gelu,</br>swigluoai,</br>situ | Y | Y | [`rocm_aiter_fused_experts`][vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe.rocm_aiter_fused_experts],</br>[`AiterUnquantizedExperts`][vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe.AiterUnquantizedExperts],</br>[`AiterFp8BlockExperts`][vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe.AiterFp8BlockExperts],</br>[`AiterFp8TensorExperts`][vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe.AiterFp8TensorExperts],</br>[`AiterFp8ChannelExperts`][vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe.AiterFp8ChannelExperts],</br>[`AiterMxfp4Experts`][vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe.AiterMxfp4Experts] |
| cpu_moe | standard | N/A | N/A | silu, gelu,</br>gelu_tanh,</br>swigluoai | Y | N | [`X86CPUUnquantizedExperts`][vllm.model_executor.layers.fused_moe.experts.cpu_moe.X86CPUUnquantizedExperts],</br>[`ArmCPUUnquantizedExperts`][vllm.model_executor.layers.fused_moe.experts.cpu_moe.ArmCPUUnquantizedExperts],</br>[`CPUUnquantizedExperts`][vllm.model_executor.layers.fused_moe.experts.cpu_moe.CPUUnquantizedExperts] |
| naive batched<sup>4</sup> | batched | int8,</br>fp8 | G,A,T | silu, gelu | <sup>6</sup> | Y | [`NaiveBatchedExperts`][vllm.model_executor.layers.fused_moe.experts.fused_batched_moe.NaiveBatchedExperts] |

!!! info "Table key"
    1. All types: mxfp4, nvfp4, int4, int8, fp8
    2. A dispatcher wrapper around triton and deep gemm experts. Will select based on type + shape + quantization params
    3. uint4, uint8, fp8, fp4
    4. This is a naive implementation of experts that supports batched format. Mainly used for testing.
    5. The `activation` parameter is ignored and SwiGlu is used by default instead.
    6. Only handled by or supported when used with modular kernels.

## Modular Kernel "families"

The following table shows "families" of modular kernels that are intended to work together. There are some combinations which may work but have not yet been tested, e.g. flashinfer with other fp8 experts.

| backend | `FusedMoEPrepareAndFinalizeModular` subclasses | `FusedMoEExpertsModular` subclasses |
| ------- | ---------------------------------------------- | ----------------------------------- |
| deepep_high_throughput | `DeepEPHTPrepareAndFinalize` | `DeepGemmExperts`,</br>`TritonExperts`,</br>`TritonOrDeepGemmExperts`,</br>`CutlassExpertsFp8`, </br>`MarlinExperts` |
| deepep_low_latency | `DeepEPLLPrepareAndFinalize` | `BatchedDeepGemmExperts`,</br>`BatchedTritonExperts`,</br>`CutlassBatchedExpertsFp8`,</br>`BatchedMarlinExperts` |
| flashinfer | `FlashInferCutlassMoEPrepareAndFinalize` | `FlashInferExperts` |
