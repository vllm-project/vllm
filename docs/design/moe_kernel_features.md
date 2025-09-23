# Fused MoE Kernel features

The purpose of this document is to provide an overview of the various MoE kernels (both modular and non-modular) so it will be easier to select an appropriate set of kernels for any particular situation. This includes information about the all2all backends used by modular kernels.

## Fused MoE Modular All2All backends

There are a number of All2All communication backends that are used to implement expert parallelism (EP) for the FusedMoE layer. The different `FusedMoEPrepareAndFinalize` sub-classes provide an interface for each all2all backend.

The following table describes the relevant features of each backend, i.e. activation format, supported quantization schemes and async support.

The output activation format (standard or batched) corresponds to the output of the prepare step of the `FusedMoEPrepareAndFinalize` subclass, the finalize step requires the same format. All the backends operate on and yield activations in standard format. More details on the formats can be found in the [Fused MoE Modular Kernel](./fused_moe_modular_kernel.md) document.

The quantization types and formats enumerate what quantization schemes are supported by each `FusedMoEPrepareAndFinalize` class. The quantization can happen before or after dispatch depending on which backend is selected, e.g. deepep_high_latency performs fp8 quantization after dispatch.  The output of the prepare step for each backend is the quantized type.  The finalize step generally requires the same input type as the original activations, e.g. if the original input is bfloat16 and the quantization scheme is fp8 w/per-tensor scales, `prepare` will return fp8/per-tensor scale activations and `finalize` will take bfloat16 activations. See the diagrams in [Fused MoE Modular Kernel](./fused_moe_modular_kernel.md) for more details on the types and formats of activations at each step of the MoE process.

Async backends support the use of DBO (Dual Batch Overlap) and shared expert overlap (where shared experts are computed during the combine step).

Certain models require the topk weights to be applied to the input activations rather than the output activations when topk==1, e.g. llama. For modular kernels, this feature is supported by the `FusedMoEPrepareAndFinalize` subclass, for non-modular kernels, it is up to the experts function to deal with this flag.

unless otherwise specified, backends are controlled via `VLLM_ALL2ALL_BACKEND`.  All backends except `flashinfer` only work with EP. `Flashinfer` can work with EP or DP.

| Backend                        | Output act. format | Quant. types | Quant. format   | Async | Apply Weight On Input | Sub-class                                                                                                                                                               |
|--------------------------------|--------------------|--------------|-----------------|-------|-----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| naive                          | standard           | all <sup>1</sup>]     | G,A,T           | N     | <sup>6</sup>]                  | [layer.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/layer.py)                                                                |
| pplx                           | batched            | fp8,int8     | G,A,T           | Y     | Y                     | [`PplxPrepareAndFinalize`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/pplx_prepare_finalize.py)                                |
| deepep_high_throughput         | standard           | fp8          | G(128),A,T <sup>2</sup>] | Y     | Y                     | [`DeepEPLLPrepareAndFinalize`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/deepep_ht_prepare_finalize.py)                       |
| deepep_low_latency             | batched            | fp8          | G(128),A,T <sup>3</sup>] | Y     | Y                     | [`DeepEPHTPrepareAndFinalize`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/deepep_ll_prepare_finalize.py)                       |
| flashinfer_all2allv            | standard           | nvfp4+fp8    | G,A,T           | N     | N                     | [`FlashInferAllToAllMoEPrepareAndFinalize`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/flashinfer_cutlass_prepare_finalize.py) |
| flashinfer <sup>4</sup>]                | standard           | nvfp4+fp8    | G,A,T           | N     | N                     | [`FlashInferCutlassMoEPrepareAndFinalize`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/flashinfer_cutlass_prepare_finalize.py)  |
| flashinfer <sup>4</sup>]                | standard           | nvfp4+fp8    | G,A,T           | N     | N                     | [`FlashInferCutlassMoEPrepareAndFinalize`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/flashinfer_cutlass_prepare_finalize.py)  |
| PrepareAndFinalizeNoEP <sup>5</sup>]    | standard           | fp8,int8     | G,A,T           | N     | Y                     | [`PrepareAndFinalizeNoEP`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/prepare_finalize.py)                                     |
| BatchedPrepareAndFinalize <sup>5</sup>] | batched            | fp8,int8     | G,A,T           | N     | Y                     | [`BatchedPrepareAndFinalize`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_batched_moe.py)                                 |

1. All types: mxfp4, nvfp4, int4, int8, fp8
2. AT quantization occurs after dispatch.
3. All quantization happens after dispatch.
4. Controlled by different env vars (`VLLM_FLASHINFER_MOE_BACKEND` "throughput" or "latency")
5. This is a no-op dispatcher that can be used to pair with any modular experts to produce a modular kernel that runs w/o dispatch or combine.  These cannot be selected via environment variable.  These are generally use for testing or adapting an expert subclass to the `fused_experts` API.
6. This depends on the experts implementation.

### Quantization format key

| Quantization formats        | Symbol |
|-----------------------------|--------|
| Grouped                     | G      |
| Grouped w/block size N      | G(N)   |
| Per activation token        | A      |
| Per tensor                  | T      |

Modular kernels are supported by the following `FusedMoEMethodBase` classes.
- [`ModelOptFp8MoEMethod`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/modelopt.py)
- [`Fp8MoEMethod`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/fp8.py)
- [`CompressedTensorsW4A4MoeMethod`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py)
- [`CompressedTensorsW8A8Fp8MoEMethod`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py)
- [`Mxfp4MoEMethod`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/mxfp4.py)
- [`UnquantizedFusedMoEMethod`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/layer.py)


## Fused MoE Experts Kernels

The are a number of MoE experts kernel implementations for different quantization types and architectures. Most follow the general API of the base Triton [`fused_experts`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py) function. Many have modular kernel adatpers so they can be used with compatible all2all backends. This table lists each experts kernel and its particular properties.

Each kernel must be provided with one of the supported input activation formats.  Some flavors of kernels support both standard and batched formats through different entry points, e.g. `TritonExperts` and `BatchedTritonExperts`. Batched format kernels are currently only needed for matching with certain all2all backends, e.g. `pplx`, `DeepEPLLPrepareAndFinalize`.

Similar to the backend kernels, each experts kernel only supports certain quantization formats. For non-modular experts, the activations will be in the original type and quantized internally by the kernel. Modular experts will expect the activations to already be in the quantized format. Both types of experts will yield outputs in the original activation type.

Each experts kernel supports one or more activation functions, e.g. silu, gelu that are applied to the intermediate results.

As with the backends, some experts support applying topk weights on the input activations. The entries in the column in this table only apply to the non-modular experts.

Most experts flavors include an equivalent modular interface which will be a subclass of `FusedMoEPermuteExpertsUnpermute`.

To be used with a particular `FusedMoEPrepareAndFinalize` sub-class, MoE kernels must have compatible activation formats, quantization types and quantization formats.


| Kernel                | Input act. format | Quant. types | Quant. format | Activation function                         | Apply Weight On Input | Modular | Source                                                                                                                                                                                                                                                                                               |
|-----------------------|-------------------|--------------|---------------|---------------------------------------------|-----------------------|---------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| triton                | standard          | all <sup>1</sup>]     | G,A,T         | silu,gelu,swigluoai,silu_no_mul,gelu_no_mul | Y                     | Y       | [`fused_experts`, `TritonExperts`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py)                                                                                                                                                                 |
| triton (batched)      | batched           | all <sup>1</sup>]     | G,A,T         | silu,gelu                                   | <sup>6</sup>]                   | Y       | [`BatchedTritonExperts`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_batched_moe.py)                                                                                                                                                                   |
| deep gemm             | standard, batched | fp8          | G(128),A,T    | silu,gelu                                   | <sup>6</sup>]                  | Y       | [`deep_gemm_moe_fp8`, `DeepGemmExperts`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/deep_gemm_moe.py), [`BatchedDeepGemmExperts`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/batched_deep_gemm_moe.py)             |
| cutlass_fp4           | standard, batched | nvfp4        | A,T           | silu                                        | Y                     | Y       | [`cutlass_moe_fp4`, `CutlassExpertsFp4`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/cutlass_moe.py)                                                                                                                                                         |
| cutlass_fp8           | standard, batched | fp8          | A,T           | silu,gelu                                   | Y                     | Y       | [`cutlass_moe_fp8`, `CutlassExpertsFp8`, `CutlasBatchedExpertsFp8`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/cutlass_moe.py)                                                                                                                              |
| flashinfer            | standard          | nvfp4,fp8    | T             | <sup>5</sup>]                                        | N                     | Y       | [`flashinfer_cutlass_moe_fp4`, `FlashInferExperts`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py)                                                                                                                                   |
| gpt oss triton        | batched           | N/A          | N/A           | <sup>5</sup>]                                        | Y                     | Y       | [`triton_kernel_fused_experts`, `BatchedOAITritonExperts`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/gpt_oss_triton_kernels_moe.py)                                                                                                                        |
| deep gemm+triton <sup>2</sup>] | standard, batched | all <sup>1</sup>]     | G(128),A,T    | silu,gelu                                   | <sup>6</sup>]                  | Y       | [`TritonOrDeepGemmExperts`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/triton_deep_gemm_moe.py), [`BatchedTritonOrDeepGemmExperts`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/batched_triton_or_deep_gemm_moe.py) |
| marlin                | standard          | <sup>3</sup>]         | <sup>3</sup>]          | silu,swigluoai                              | Y                     | N       | [`fused_marlin_moe`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_marlin_moe.py)                                                                                                                                                                        |
| trtllm                | standard          | mxfp4,nvfp4  | G(16),G32)    | <sup>5</sup>]                                        | N                     | Y       | [`TrtLlmGenExperts`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/trtllm_moe.py)                                                                                                                                                                              |
| pallas                | standard          | N/A          | N/A           | silu                                        | N                     | N       | [`fused_moe`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/moe_pallas.py)                                                                                                                                                                                     |
| iterative             | standard          | N/A          | N/A           | silu                                        | N                     | N       | [`fused_moe`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/moe_torch_iterative.py)                                                                                                                                                                            |
| rocm aiter moe        | standard          | fp8          | G(128),A,T    | silu,gelu                                   | Y                     | N       | [`rocm_aiter_fused_experts`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py)                                                                                                                                                            |
| cpu_fused_moe         | standard          | N/A          | N/A           | silu                                        | N                     | N       | [`CPUFusedMOE`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/cpu_fused_moe.py)                                                                                                                                                                                |
| naive batched <sup>4</sup>]    | batched           | int8,fp8     | G,A,T         | silu,gelu                                   | <sup>6</sup>]                  | Y       | [`NaiveBatchedExperts`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_batched_moe.py)                                                                                                                                                                    |

1. All types: mxfp4, nvfp4, int4, int8, fp8
2. A dispatcher wrapper around triton and deep gemm experts.  Will select based on type + shape + quantization params
3. uint4, uint8, fp8, fp4
4. This is a naive implementation of experts that supports batched format. Mainly used for testing.
5. The `activation` parameter is ignored and SwiGlu is used by default instead.
6. Only handled by or supported when used with modular kernels.

## Modular Kernel "families"

| backend                      | `FusedMoEPrepareAndFinalize` subclasses                | `FusedMoEPermuteExpertsUnpermute` subclasses                                                                   |
|------------------------------|--------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| deepep_high_throughput, pplx | `DeepEPHTPrepareAndFinalize`, `PplxPrepareAndFinalize` | `BatchedDeepGemmExperts`, `BatchedTritonExperts`, `BatchedTritonOrDeepGemmExperts`, `CutlassBatchedExpertsFp8` |
| deepep_low_latency           | `DeepEPLLPrepareAndFinalize`                           | `DeepGemmExperts`, `TritonExperts`, `TritonOrDeepGemmExperts`, `CutlassExpertsFp8`                             |
| flashinfer                   | `FlashInferCutlassMoEPrepareAndFinalize`               | `FlashInferExperts`                                                                                            |
