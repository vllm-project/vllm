TODO: add feature support?

| Quantization formats        | Symbol |
|-----------------------------|--------|
| Grouped                     | G      |
| Grouped specific block size | G(N)   |
| Per activation token        | A      |
| Per tensor                  | T      |


| Backend                | Output act. format | Quant. types | Quant. format   | Source                                 |
|------------------------|--------------------|--------------|-----------------|----------------------------------------|
| naive                  | standard           | all [^1]     | G,A,T           | layer.py                               |
| pplx                   | batched            | fp8,int8     | G,A,T           | pplx_prepare_finalize.py               |
| deepep_high_throughput | standard           | fp8          | G(128),A,T [^2] | deepep_ht_prepare_finalize.py          |
| deepep low_latency     | batched            | fp8          | G(128),A,T [^3] | deepep_ll_prepare_finalize.py          |
| flashinfer_all2allv    | standard           | nvfp4+fp8    | G,A,T           | flashinfer_cutlass_prepare_finalize.py |
| flashinfer [^4]        | standard           | nvfp4+fp8    | G,A,T           | flashinfer_cutlass_prepare_finalize.py |

Backends are controlled via VLLM_ALL2ALL_BACKEND unless otherwise specified.

[^1] All types: mxfp4, nvfp4, int4, int8, fp8
[^2] AT quantization occurs after dispatch.
[^3] All quantization happens after dispatch.
[^4] Controlled by different env vars (VLLM_FLASHINFER_MOE_BACKEND "throughput" or "latency")

TODO: add feature support?  (expert map, chunking)

| Kernel                | Input act. format | Quant. types | Quant. format | Modular | source                                                     |
|-----------------------|-------------------|--------------|---------------|---------|------------------------------------------------------------|
| triton                | standard/batched  | all [^1]     | G,A,T         | Y       | fused_moe.py/fused_batched_moe.py                          |
| deep gemm             | standard/batched  | fp8          | G(128),A,T    | Y       | deep_gemm_moe.py/batched_deep_gemm_moe.py                  |
| cutlass_fp4           | standard/batched  | nvfp4        | A,T           | Y       | cutlass_moe.py                                             |
| cutlass_fp8           | standard/batched  | fp8          | A,T           | Y       | cutlass_moe.py                                             |
| flashinfer            | standard          | nvfp4,fp8    | T             | Y       | flashinfer_cutlass_moe.py                                  |
| gpt oss triton        | batched           | N/A          | N/A           | Y       | gpt_oss_triton_kernels_moe.py                              |
| deep gemm+triton [^2] | standard/batched  | all [^1]     | G(128),A,T    | Y       | triton_deep_gemm_moe.py/batched_triton_or_deep_gemm_moe.py |
| marlin                | standard          | [^3]         | [^3]          | N       | fused_marlin_moe.py                                        |
| trtllm                | standard          | mxfp4,nvfp4  | G(16),G32)    | Y       | trtllm_moe.py                                              |
| pallas                | standard          | N/A          | N/A           | N       | moe_pallas.py                                              |
| iterative             | standard          | N/A          | N/A           | N       | moe_torch_iterative.py                                     |
| rocm aiter moe        | standard          | fp8          | G(128),A,T    | N       | rocm_aiter_fused_moe.py                                    |
| cpu_fused_moe         | standard          | N/A          | N/A           | N       | cpu_fused_moe.py                                           |

[^1] All types: mxfp4, nvfp4, int4, int8, fp8
[^2] A dispatcher wrapper around triton and deep gemm experts.  Will select based on type + shape + quantization params
[^3] uint4, uint8, fp8, fp4

Kernels must have compatible activation formats, quantization types and quantization formats with dispatchers.

-----------------------------------------------------------------------------------------

| backend | triton   deep gemm   cutlass_fp8   cutlass_fp4   gpt oss  flashinfer
naive          (std)    all       x          x            x             ?           x         x
pplx       (batched)    >=fp4     x          x            x                         x
deepep ht      (std)   fp8+bf16   x          x            x                                   x
deepep ll  (batched)   fp8+bf16   x          x            x                         x
flashinfer     (std)   fp4+fp     x          x                          x                     x

                 formats         types
triton      batched+standard     mxfp4, int8, fp8, fp16, bf16
deep gemm   batched+standard     fp8, fp16, bf16
cutlass_fp8 batched+standard     fp8
cutlass_fp4 batched+standard     nvfp4 (mxfp4?)
flashinfer    standard           nvfp4+fp8
gpt oss       batched            fp8, fp16, bf16?
