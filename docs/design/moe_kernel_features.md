# Fused MoE Kernel Features

The purpose of this document is to provide an overview of the various MoE kernels (both modular and non-modular) so it will be easier to select an appropriate set of kernels for any particular situation. This includes information about the all2all backends used by modular kernels.

## Fused MoE Modular All2All backends

There are a number of all2all communication backends that are used to implement expert parallelism (EP) for the `MoERunner` layer. The different `FusedMoEPrepareAndFinalizeModular` subclasses provide an interface for each all2all backend.

The following table describes the relevant features of each backend, i.e. activation format, supported quantization schemes and async support.

The output activation format (standard, padded standard, or batched) corresponds to the output of the prepare step of the `FusedMoEPrepareAndFinalizeModular` subclass, and the experts must accept that format. All the backend `prepare` methods expect activations in the standard format. `finalize` combines the experts' results into the caller-provided standard-format output buffer. More details on the formats can be found in the [Fused MoE Modular Kernel](./fused_moe_modular_kernel.md) document.

The quantization types enumerate the end-to-end MoE quantization schemes with which each `FusedMoEPrepareAndFinalizeModular` class can be paired. They do not necessarily describe the representation sent over the all2all transport. Quantization can happen before or after dispatch based on the formats the backend supports. For example, deepep_high_throughput dispatches block-quantized FP8 directly but dispatches higher-precision activations for other supported FP8 formats and quantizes them afterward. The quantization format column describes the activation formats that `prepare` can provide to the experts kernel. The finalize step generally requires the same input type as the original activations. See the diagrams in [Fused MoE Modular Kernel](./fused_moe_modular_kernel.md) for more details on the types and formats of activations at each step of the MoE process. A quantization type of `none` means the kernel operates on float16 and/or bfloat16.

Async backends support the use of DBO (Dual Batch Overlap) and shared expert overlap (where shared experts are computed during the combine step).

Certain models require the topk weights to be applied to the input activations rather than the output activations when topk==1, e.g. Llama. For modular kernels, this feature is supported by the `FusedMoEPrepareAndFinalizeModular` subclass. For non-modular kernels, it is up to the experts function to deal with this flag.

Unless otherwise specified, backends are controlled via the `--all2all-backend` command-line argument (or the `all2all_backend` parameter in `ParallelConfig`). Most distributed adapters, including FlashInfer NVLink, are gated by `FusedMoEParallelConfig.use_all2all_kernels`: EP must be enabled with DP, PCP, or sequence-parallel token distribution. DP without EP uses the naive AllGather/ReduceScatter path instead. Passthrough and the local/reference adapters have separate construction paths.

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
| deepep_high_throughput | standard | none,fp8,</br>weight-only<sup>9</sup> | N/A,G(128),A,T<sup>2</sup> | Y | Y | [`DeepEPHTPrepareAndFinalize`][vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ht.DeepEPHTPrepareAndFinalize] |
| deepep_low_latency | batched | none,fp8,nvfp4,</br>weight-only<sup>9</sup> | N/A,G(16),G(128),A,T<sup>3</sup> | Y | Y | [`DeepEPLLPrepareAndFinalize`][vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ll.DeepEPLLPrepareAndFinalize] |
| flashinfer_nvlink_two_sided | standard | nvfp4,fp8 | G,A,T | N | N | [`FlashInferNVLinkTwoSidedPrepareAndFinalize`][vllm.model_executor.layers.fused_moe.prepare_finalize.flashinfer_nvlink_two_sided.FlashInferNVLinkTwoSidedPrepareAndFinalize] |
| flashinfer_nvlink_one_sided | standard | nvfp4,bf16,mxfp8,fp8 | G,A,T | N | N | [`FlashInferNVLinkOneSidedPrepareAndFinalize`][vllm.model_executor.layers.fused_moe.prepare_finalize.flashinfer_nvlink_one_sided.FlashInferNVLinkOneSidedPrepareAndFinalize] |
| deepep_v2 | padded standard | none,nvfp4,</br>mxfp8,fp8<sup>7</sup> | N/A,G(16),G(32),</br>G(128),A,T | Y | Y | [`DeepEPV2PrepareAndFinalize`][vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_v2.DeepEPV2PrepareAndFinalize] |
| nixl_ep | batched | none,fp8<sup>7</sup> | N/A,G(128),A,T | Y | Y | [`NixlEPPrepareAndFinalize`][vllm.model_executor.layers.fused_moe.prepare_finalize.nixl_ep.NixlEPPrepareAndFinalize] |
| mori (EPv2) | standard | none,fp8 | N/A,G(128) | N | N | [`MoriEPv2PrepareAndFinalize`][vllm.model_executor.layers.fused_moe.prepare_finalize.mori_epv2.MoriEPv2PrepareAndFinalize] |
| mori (EPv1 fallback),</br>mori_high_throughput,</br>mori_low_latency | standard | none,mxfp4,fp8 | G(32),G(128),A,T | N | N | [`MoriPrepareAndFinalize`][vllm.model_executor.layers.fused_moe.prepare_finalize.mori.MoriPrepareAndFinalize] |
| moonep | standard | none | N/A | N | Y | [`MoonEPPrepareAndFinalize`][vllm.model_executor.layers.fused_moe.prepare_finalize.moonep.MoonEPPrepareAndFinalize] |
| alltoall_batched (XPU modular)<sup>8</sup> | batched | none | N/A | N | Y | [`AllToAllBatchedPrepareAndFinalize`][vllm.model_executor.layers.fused_moe.prepare_finalize.alltoall_batched.AllToAllBatchedPrepareAndFinalize] |
| allgather_reducescatter | standard | all<sup>1</sup> | G,A,T | N | <sup>6</sup> | [`MoEPrepareAndFinalizeNaiveDPEPModular`][vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep.MoEPrepareAndFinalizeNaiveDPEPModular] |
| (no dispatch)<sup>5</sup> | standard | all<sup>1</sup> | G,A,T | N | <sup>6</sup> | [`MoEPrepareAndFinalizeNoDPEPModular`][vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep.MoEPrepareAndFinalizeNoDPEPModular] |
| (batched no-op)<sup>5</sup> | batched | all<sup>1</sup> | G,A,T | N | <sup>6</sup> | [`BatchedPrepareAndFinalize`][vllm.model_executor.layers.fused_moe.prepare_finalize.batched.BatchedPrepareAndFinalize] |
| passthrough | standard | nvfp4,mxfp4 | G | N | Y | [`PassThroughPrepareAndFinalize`][vllm.model_executor.layers.fused_moe.prepare_finalize.passthrough.PassThroughPrepareAndFinalize] |

!!! info "Table key"
    1. Types include none, mxfp4, nvfp4, mxfp8, int4, int8, and fp8, subject to the paired experts and adapter's input-quantization contract.
    2. A,T quantization occurs after dispatch.
    3. Quantization can happen during or after dispatch, depending on the quantization config and DeepEP dispatch mode. Fused NVFP4 dispatch requires the optional DeepEP hybrid-EP implementation; otherwise that path dispatches BF16 for the CuteDSL experts to quantize.
    4. Controlled by `--moe-backend` (`flashinfer_cutlass` or `flashinfer_trtllm`)
    5. These adapters avoid cross-rank communication but must still match the experts' layout and input-quantization requirements. `MoEPrepareAndFinalizeNoDPEPModular` is the normal local modular adapter when no dispatch is needed. `BatchedPrepareAndFinalize` reorganizes tokens locally, including the XPU `--moe-backend batched_triton` path. Neither is selected by an `--all2all-backend` name.
    6. This depends on the experts implementation.
    7. Supported MoE quantization does not imply that the quantized representation is used for dispatch. The backend may dispatch higher-precision activations and quantize after dispatch.
    8. The production factory selects this batched adapter only for modular execution on XPU. Other platforms or monolithic execution take the naive DP/EP construction branch instead; the backend name alone does not establish a supported batched pairing.
    9. Weight-only schemes dispatch unquantized activations; examples include the INT4/INT8/FP4 schemes supported by the compatible Marlin adapter. Quantization lists summarize common pairings, not every experts configuration. DeepEP LL rejects `defer_input_quant=True`; Humming pairings therefore require a prequantized activation scheme.

    ---

    - G - Grouped
    - G(N) - Grouped w/block size N
    - A - Per activation token
    - T - Per tensor
    - Padded standard - token-major with inactive rows identified by `topk_ids == -1`

Modular kernels are supported by the following `FusedMoEMethodBase` classes (non-exhaustive — new quantization methods are added regularly):

- [`UnquantizedFusedMoEMethod`][vllm.model_executor.layers.fused_moe.UnquantizedFusedMoEMethod]
- [`Fp8MoEMethod`][vllm.model_executor.layers.quantization.fp8.Fp8MoEMethod]
- [`CompressedTensorsW8A8Fp8MoEMethod`][vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_fp8.CompressedTensorsW8A8Fp8MoEMethod]
- [`CompressedTensorsW4A4Nvfp4MoEMethod`][vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_nvfp4.CompressedTensorsW4A4Nvfp4MoEMethod]
- [`CompressedTensorsW4A4Mxfp4MoEMethod`][vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_mxfp4.CompressedTensorsW4A4Mxfp4MoEMethod]
- [`CompressedTensorsW8A8Int8MoEMethod`][vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_int8.CompressedTensorsW8A8Int8MoEMethod]
- [`CompressedTensorsW8A8Mxfp8MoEMethod`][vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_mxfp8.CompressedTensorsW8A8Mxfp8MoEMethod]
- [`CompressedTensorsWNA16MoEMethod`][vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16.CompressedTensorsWNA16MoEMethod]
- [`CompressedTensorsW4A8Fp8MoEMethod`][vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a8_fp8.CompressedTensorsW4A8Fp8MoEMethod]
- [`CompressedTensorsW4A8Int8MoEMethod`][vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a8_int8.CompressedTensorsW4A8Int8MoEMethod]
- [`HummingMoEMethod`][vllm.model_executor.layers.quantization.humming.HummingMoEMethod]
- [`MoeWNA16Method`][vllm.model_executor.layers.quantization.moe_wna16.MoeWNA16Method]
- [`AutoGPTQMoEMethod`][vllm.model_executor.layers.quantization.auto_gptq.AutoGPTQMoEMethod]
- Online quantization methods: `Fp8PerTensorOnlineMoEMethod`, `Fp8PerBlockOnlineMoEMethod`, `Mxfp4OnlineMoEMethod`, `Mxfp8OnlineMoEMethod`, `Nvfp4OnlineMoEMethod`, `Int8OnlineMoEMethod`
- Quark methods: `QuarkW8A8Fp8MoEMethod`, `QuarkW8A8Int8MoEMethod`, `QuarkW4A8Fp8MoEMethod`, `QuarkOCP_MX_MoEMethod`, `QuarkNvfp4MoEMethod`

Shared backend selection is organized under [`fused_moe/oracle/`](../../vllm/model_executor/layers/fused_moe/oracle/). Some specialized or legacy paths select and construct kernels in the quantization method or its helpers; inspect that method's call sites when extending it.

## Fused Experts Kernels

There are a number of MoE experts kernel implementations for different quantization types and architectures. Most follow the general API of the base Triton [`fused_experts`][vllm.model_executor.layers.fused_moe.fused_moe.fused_experts] function. Many have modular kernel adapters, so they can be used with compatible all2all backends. The following non-exhaustive table summarizes the principal experts kernel families and their properties.

Each kernel must be provided with one of the supported input activation formats. Some flavors of kernels support both standard and batched formats through different entry points, e.g. `TritonExperts` and `BatchedTritonExperts`. Batched format kernels are currently only needed for matching with certain all2all backends, e.g. `DeepEPLLPrepareAndFinalize`.

Similar to the backend kernels, each experts kernel only supports certain quantization formats. Quantized modular experts normally receive inputs quantized by prepare, but implementations declaring `expects_unquantized_inputs=True` perform input quantization internally. The kernel passes this requirement to prepare as `defer_input_quant=True`; the prepare/finalize implementation must support that mode. For example, `FlashInferB12xExperts` requires unquantized BF16 inputs, whereas `DeepEPLLPrepareAndFinalize` rejects deferred input quantization. Modular and non-modular experts yield outputs in the original activation type. See the [prepare/finalize contract](./fused_moe_modular_kernel.md#fusedmoeprepareandfinalizemodular) for details.

Each experts kernel supports one or more activation functions, e.g. silu or gelu, which are applied to the intermediate results.

As with the backends, some experts support applying topk weights on the input activations. The entries in the column in this table only apply to the non-modular experts.

Most experts flavors include an equivalent modular interface which will be a subclass of `FusedMoEExpertsModular`.

To be used with a particular `FusedMoEPrepareAndFinalizeModular` subclass, MoE kernels must have compatible activation formats, quantization types, quantization formats, and input-quantization ownership. Platform and parallel-configuration support must also match.

The table summarizes adapter capabilities, not every combination of its columns. Weight/activation quantization pairs, routing methods, shapes, hardware, and optional features can impose additional restrictions; the adapter's `is_supported_config()` and execution contract are authoritative. Runtime fallback wrappers require a configuration supported by both underlying experts, not the union of their capabilities.

| Kernel | Input act. format | Quant. types | Quant. format | Activation function | Apply Weight On Input | Modular | Source |
| ------ | ----------------- | ------------ | ------------- | ------------------- | --------------------- | ------- | ------ |
| triton | standard,</br>padded standard | none,int8,fp8 | G,A,T | <sup>8</sup> | Y | Y | [`fused_experts`][vllm.model_executor.layers.fused_moe.fused_moe.fused_experts],</br>[`TritonExperts`][vllm.model_executor.layers.fused_moe.experts.triton_moe.TritonExperts] |
| triton (batched) | batched | none,fp8 | G(128),A,T | silu, gelu, etc.<sup>12</sup> | <sup>6</sup> | Y | [`BatchedTritonExperts`][vllm.model_executor.layers.fused_moe.experts.fused_batched_moe.BatchedTritonExperts] |
| deep gemm | standard,</br>padded standard | fp8,</br>mxfp8 (SM100) | G(128),G(32) | silu, swiglustep,</br>swigluoai_uninterleave | <sup>6</sup> | Y | [`DeepGemmExperts`][vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe.DeepGemmExperts] |
| deep gemm (batched) | batched | fp8 | G(128) | silu | <sup>6</sup> | Y | [`BatchedDeepGemmExperts`][vllm.model_executor.layers.fused_moe.experts.batched_deep_gemm_moe.BatchedDeepGemmExperts] |
| cutlass_fp4 | standard | nvfp4 | G(16),T | <sup>8</sup> | Y | Y | [`CutlassExpertsFp4`][vllm.model_executor.layers.fused_moe.experts.cutlass_moe.CutlassExpertsFp4] |
| cutlass_fp8 | standard,</br>batched | fp8 | A,T | gated<sup>8</sup> | Y | Y | [`CutlassExpertsFp8`][vllm.model_executor.layers.fused_moe.experts.cutlass_moe.CutlassExpertsFp8],</br>[`CutlassBatchedExpertsFp8`][vllm.model_executor.layers.fused_moe.experts.cutlass_moe.CutlassBatchedExpertsFp8] |
| flashinfer cutlass | standard | none,nvfp4,mxfp4,</br>mxfp8,fp8<sup>5</sup> | N/A,G(16),G(32),</br>G(128),T | silu, gelu_tanh,</br>relu2_no_mul, swigluoai,</br>swigluoai_uninterleave | N | Y | [`FlashInferExperts`][vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe.FlashInferExperts] |
| gpt oss triton (modular) | standard | mxfp4 (weights),</br>none (activations) | G(32) weights | swigluoai | Y | Y | [`OAITritonExperts`][vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe.OAITritonExperts] |
| gpt oss triton mxfp4 | standard | mxfp4 | G(32) | swigluoai | Y | N | [`OAITritonMxfp4ExpertsMonolithic`][vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe.OAITritonMxfp4ExpertsMonolithic] |
| marlin | standard,</br>batched | <sup>3</sup> / N/A | <sup>3</sup> / N/A | silu,</br>swigluoai | Y | Y | [`fused_marlin_moe`][vllm.model_executor.layers.fused_moe.experts.marlin_moe.fused_marlin_moe],</br>[`MarlinExperts`][vllm.model_executor.layers.fused_moe.experts.marlin_moe.MarlinExperts],</br>[`BatchedMarlinExperts`][vllm.model_executor.layers.fused_moe.experts.marlin_moe.BatchedMarlinExperts] |
| trtllm | standard | mxfp4,</br>nvfp4 | G(16),G(32) | <sup>7</sup> | N | Y | [`TrtLlmMxfp4ExpertsMonolithic`][vllm.model_executor.layers.fused_moe.experts.trtllm_mxfp4_moe.TrtLlmMxfp4ExpertsMonolithic],</br>[`TrtLlmMxfp4ExpertsModular`][vllm.model_executor.layers.fused_moe.experts.trtllm_mxfp4_moe.TrtLlmMxfp4ExpertsModular],</br>[`TrtLlmNvFp4ExpertsMonolithic`][vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe.TrtLlmNvFp4ExpertsMonolithic],</br>[`TrtLlmNvfp4ExpertsModular`][vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe.TrtLlmNvFp4ExpertsModular] |
| hpc | standard | fp8 | G(128),T | silu | Y | Y | [`HPCExperts`][vllm.model_executor.layers.fused_moe.hpc_moe.HPCExperts] |
| rocm aiter moe | standard | none,</br>mxfp4,</br>fp8 | G(32),G(128),A,T | silu, gelu,</br>swigluoai,</br>situ | Y | Y | [`rocm_aiter_fused_experts`][vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe.rocm_aiter_fused_experts],</br>[`AiterUnquantizedExperts`][vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe.AiterUnquantizedExperts],</br>[`AiterFp8BlockExperts`][vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe.AiterFp8BlockExperts],</br>[`AiterFp8TensorExperts`][vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe.AiterFp8TensorExperts],</br>[`AiterFp8ChannelExperts`][vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe.AiterFp8ChannelExperts],</br>[`AiterMxfp4Experts`][vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe.AiterMxfp4Experts] |
| cpu_moe | standard | N/A | N/A | silu, gelu,</br>gelu_tanh,</br>swigluoai | Y | Y | [`X86CPUUnquantizedExperts`][vllm.model_executor.layers.fused_moe.experts.cpu_moe.X86CPUUnquantizedExperts],</br>[`ArmCPUUnquantizedExperts`][vllm.model_executor.layers.fused_moe.experts.cpu_moe.ArmCPUUnquantizedExperts],</br>[`CPUUnquantizedExperts`][vllm.model_executor.layers.fused_moe.experts.cpu_moe.CPUUnquantizedExperts] |
| naive batched<sup>4</sup> | batched | int8,</br>fp8 | G,A,T | silu, gelu | <sup>6</sup> | Y | [`NaiveBatchedExperts`][vllm.model_executor.layers.fused_moe.experts.fused_batched_moe.NaiveBatchedExperts] |
| trtllm fp8 (modular) | standard,</br>padded standard | fp8,mxfp8 | G(128),G(32) | silu,</br>swigluoai_uninterleave,</br>relu2_no_mul | <sup>6</sup> | Y | [`TrtLlmFp8ExpertsModular`][vllm.model_executor.layers.fused_moe.experts.trtllm_fp8_moe.TrtLlmFp8ExpertsModular] |
| trtllm fp8 (monolithic, block) | standard | fp8,mxfp8 | G(128),G(32) | silu,</br>swigluoai_uninterleave,</br>relu2_no_mul | N | N | [`TrtLlmFp8ExpertsMonolithic`][vllm.model_executor.layers.fused_moe.experts.trtllm_fp8_moe.TrtLlmFp8ExpertsMonolithic] |
| trtllm fp8 (monolithic, per-tensor) | standard | fp8 | T | silu, relu2_no_mul | Y<sup>10</sup> | N | [`TrtLlmFp8ExpertsMonolithic`][vllm.model_executor.layers.fused_moe.experts.trtllm_fp8_moe.TrtLlmFp8ExpertsMonolithic] |
| trtllm bf16 (modular) | standard,</br>padded standard | none | N/A | silu, relu2_no_mul,</br>swiglustep<sup>7</sup> | <sup>6</sup> | Y | [`TrtLlmBf16ExpertsModular`][vllm.model_executor.layers.fused_moe.experts.trtllm_bf16_moe.TrtLlmBf16ExpertsModular] |
| trtllm bf16 (monolithic) | standard | none | N/A | silu, relu2_no_mul,</br>swiglustep<sup>7</sup> | N | N | [`TrtLlmBf16ExpertsMonolithic`][vllm.model_executor.layers.fused_moe.experts.trtllm_bf16_moe.TrtLlmBf16ExpertsMonolithic] |
| trtllm mxint4 | standard | mxint4 (weights),</br>none (activations) | G(32) weights | silu, swigluoai (alias) | N | N | [`TrtLlmMxint4ExpertsMonolithic`][vllm.model_executor.layers.fused_moe.experts.trtllm_mxint4_moe.TrtLlmMxint4ExpertsMonolithic] |
| trtllm lora | standard | none | N/A | silu | <sup>6</sup> | Y | [`TrtLlmBf16LoRAExperts`][vllm.model_executor.layers.fused_moe.experts.trtllm_lora_moe.TrtLlmBf16LoRAExperts] |
| deep gemm fp4 | standard,</br>padded standard | mxfp4 (weights),</br>fp8 (activations) | G(32) weights,</br>G(128) activations | silu,</br>swiglustep,</br>situ | <sup>6</sup> | Y | [`DeepGemmFP4Experts`][vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe.DeepGemmFP4Experts] |
| cutlass mxfp4 | standard | mxfp4 | G(32) | <sup>8</sup> | Y | Y | [`CutlassExpertsMxfp4`][vllm.model_executor.layers.fused_moe.experts.cutlass_moe.CutlassExpertsMxfp4] |
| cutlass w4a8fp8 | standard | int4 (weights),</br>fp8 (activations) | G weights,</br>A activations | gated<sup>8</sup> | Y | Y | [`CutlassExpertsW4A8Fp8`][vllm.model_executor.layers.fused_moe.experts.cutlass_moe.CutlassExpertsW4A8Fp8] |
| flashinfer cutedsl | standard | nvfp4 | G(16),A,T | silu, swigluoai,</br>swigluoai_uninterleave,</br>relu2_no_mul, situ | N | Y | [`FlashInferCuteDSLExperts`][vllm.model_executor.layers.fused_moe.experts.flashinfer_cutedsl_moe.FlashInferCuteDSLExperts] |
| flashinfer cutedsl (batched) | batched | nvfp4 | G(16),T | silu | N | Y | [`FlashInferCuteDSLBatchedExperts`][vllm.model_executor.layers.fused_moe.experts.flashinfer_cutedsl_batched_moe.FlashInferCuteDSLBatchedExperts] |
| flashinfer b12x | standard | nvfp4<sup>9</sup> | G(16),T | silu, gelu_tanh,</br>relu2_no_mul | N | Y | [`FlashInferB12xExperts`][vllm.model_executor.layers.fused_moe.experts.flashinfer_b12x_moe.FlashInferB12xExperts] |
| humming | standard,</br>padded standard,</br>batched | mxfp4,nvfp4,</br>mxfp8,fp8,</br>int4,int8 | G,A,T | <sup>8</sup> | Y | Y | [`HummingIndexedExperts`][vllm.model_executor.layers.fused_moe.experts.fused_humming_moe.HummingIndexedExperts],</br>[`HummingGroupedExperts`][vllm.model_executor.layers.fused_moe.experts.fused_humming_moe.HummingGroupedExperts],</br>[`BatchedHummingGroupedExperts`][vllm.model_executor.layers.fused_moe.experts.fused_humming_moe.BatchedHummingGroupedExperts] |
| triton wna16 | standard,</br>padded standard | int4,int8 (weights),</br>none (activations) | G weights | <sup>8</sup> except situ | Y | Y | [`TritonWNA16Experts`][vllm.model_executor.layers.fused_moe.experts.triton_moe.TritonWNA16Experts] |
| triton-or-deepgemm | standard,</br>padded standard | fp8 | G(128) | silu, swiglustep,</br>swigluoai_uninterleave | <sup>6</sup> | Y | [`TritonOrDeepGemmExperts`][vllm.model_executor.layers.fused_moe.experts.triton_deep_gemm_moe.TritonOrDeepGemmExperts] |
| triton-or-cutlass | standard | fp8 | A,T | gated<sup>8</sup> | <sup>6</sup> | Y | [`TritonOrCutlassExperts`][vllm.model_executor.layers.fused_moe.experts.triton_cutlass_moe.TritonOrCutlassExperts] |
| aiter mxfp8 | standard,</br>padded standard | mxfp8 | G(32) | silu,</br>swigluoai_uninterleave<sup>11</sup> | <sup>6</sup> | Y | [`AiterMxfp8Experts`][vllm.model_executor.layers.fused_moe.experts.aiter_mxfp8_moe.AiterMxfp8Experts] |
| triton mxfp8 | standard,</br>padded standard | mxfp8 | G(32) | silu, gelu,</br>swigluoai | Y | Y | [`Mxfp8NativeTritonExperts`][vllm.model_executor.layers.fused_moe.experts.mxfp8_native_moe.Mxfp8NativeTritonExperts],</br>[`Mxfp8EmulationTritonExperts`][vllm.model_executor.layers.fused_moe.experts.mxfp8_emulation_moe.Mxfp8EmulationTritonExperts] |
| aiter mxfp4 w4a8 | standard | mxfp4,fp8 | G(32),T | swigluoai | Y | N | [`AiterW4A8ExpertsMonolithic`][vllm.model_executor.layers.fused_moe.experts.aiter_mxfp4_w4a8_moe.AiterW4A8ExpertsMonolithic] |
| aiter mxfp4 w4a16 | standard | mxfp4,none | G(32) | silu,swigluoai | Y | N | [`AiterW4A16ExpertsMonolithic`][vllm.model_executor.layers.fused_moe.experts.aiter_mxfp4_w4a16_moe.AiterW4A16ExpertsMonolithic] |
| xpu | standard | none,fp8,mxfp8,</br>mxfp4,int4 | G,T | silu, gelu, gelu_tanh,</br>swigluoai, relu2_no_mul | <sup>6</sup> | Y | [`XPUExperts`][vllm.model_executor.layers.fused_moe.experts.xpu_moe.XPUExperts],</br>[`XPUExpertsFp8`][vllm.model_executor.layers.fused_moe.experts.xpu_moe.XPUExpertsFp8], etc. |
| moonep | standard | none | N/A | silu | <sup>6</sup> | Y | [`MoonEPExperts`][vllm.model_executor.layers.fused_moe.experts.moonep_experts.MoonEPExperts] |

!!! info "Table key"
    1. All types: mxfp4, nvfp4, int4, int8, fp8
    2. A dispatcher wrapper around triton and deep gemm experts. Will select based on type + shape + quantization params
    3. uint4, uint8, fp8, fp4
    4. This is a naive implementation of experts that supports batched format. Mainly used for testing.
    5. FlashInfer CUTLASS's accepted weight/activation pairs depend on hardware; see `FlashInferExperts._supports_quant_scheme()`. The listed types are not freely interchangeable.
    6. Only handled by or supported when used with modular kernels.
    7. Supported activations depend on the quantization format and FlashInfer version. NVFP4 and BF16 enable `swiglustep` only when the installed FlashInfer exposes that activation.
    8. Activation functions accepted by `apply_moe_activation_supported()` in [`activation.py`](../../vllm/model_executor/layers/fused_moe/activation.py), including gated and no-mul SiLU/GELU variants, SiTU, ReLU-squared, and SwiGLU variants. Entries marked "gated" exclude the no-mul variants.
    9. B12x accepts NVFP4 weight schemes with either NVFP4 activation quantization or no checkpoint activation-quantization key (W4A16). It consumes BF16 inputs and quantizes them internally. Expert parallelism is not supported.
    10. TRTLLM per-tensor FP8 applies router weights on input for `RoutingMethodType.Llama4`; its block-scaled path rejects that mode.
    11. AITER MXFP8 requires `swiglu_alpha=1.702` and `swiglu_beta=1.0` for `swigluoai_uninterleave`.
    12. Batched Triton uses `apply_moe_activation_masked_supported()` on CUDA/ROCm. Its XPU support check excludes SiTU, `swigluoai_uninterleave`, and `swiglustep` from that set.

## Modular Kernel "families"

The following table shows "families" of modular kernels that are intended to work together. There are some combinations which may work but have not yet been tested, e.g. flashinfer with other fp8 experts.

| backend | `FusedMoEPrepareAndFinalizeModular` subclasses | `FusedMoEExpertsModular` subclasses |
| ------- | ---------------------------------------------- | ----------------------------------- |
| deepep_high_throughput | `DeepEPHTPrepareAndFinalize` | `DeepGemmExperts`,</br>`TritonExperts`,</br>`TritonOrDeepGemmExperts`,</br>`MarlinExperts`,</br>`HummingIndexedExperts`,</br>`HummingGroupedExperts` |
| deepep_low_latency | `DeepEPLLPrepareAndFinalize` | `BatchedDeepGemmExperts`,</br>`BatchedTritonExperts`,</br>`CutlassBatchedExpertsFp8`,</br>`BatchedMarlinExperts`,</br>`BatchedHummingGroupedExperts`,</br>`FlashInferCuteDSLBatchedExperts` |
| deepep_v2 | `DeepEPV2PrepareAndFinalize` | `DeepGemmExperts`,</br>`TritonExperts`,</br>`TritonOrDeepGemmExperts` |
| nixl_ep | `NixlEPPrepareAndFinalize` | `BatchedDeepGemmExperts`,</br>`BatchedTritonExperts` |
| mori (EPv2) | `MoriEPv2PrepareAndFinalize` | `AiterUnquantizedExperts`,</br>`AiterFp8BlockExperts` |
| mori (EPv1 fallback),</br>mori_high_throughput,</br>mori_low_latency | `MoriPrepareAndFinalize` | `AiterUnquantizedExperts`,</br>`AiterFp8BlockExperts`,</br>`AiterFp8TensorExperts`,</br>`AiterFp8ChannelExperts`,</br>`AiterMxfp4Experts` |
| moonep | `MoonEPPrepareAndFinalize` | `MoonEPExperts` |
| alltoall_batched (XPU modular) | `AllToAllBatchedPrepareAndFinalize` | `BatchedTritonExperts` |
| allgather_reducescatter | `MoEPrepareAndFinalizeNaiveDPEPModular` | `TritonExperts`,</br>`DeepGemmExperts`,</br>`CutlassExpertsFp8` |
| flashinfer | `FlashInferNVLinkTwoSidedPrepareAndFinalize`,</br>`FlashInferNVLinkOneSidedPrepareAndFinalize` | `FlashInferExperts`,</br>`FlashInferCuteDSLExperts` |
| (no dispatch) | `MoEPrepareAndFinalizeNoDPEPModular` | (compatible standard or padding-aware experts) |

These families remain subject to each implementation's platform, quantization, and parallel-configuration checks. `FlashInferB12xExperts` is not an EP pairing: its support check rejects `use_ep=True` and it requires all experts to be local.
