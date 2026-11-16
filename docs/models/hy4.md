# HY4

vLLM supports `tencent/Hy4-preview` and `tencent/Hy4-preview-FP8` through
`HYV4ForCausalLM`.

## AMD ROCm graph execution

HY4 on AMD ROCm uses breakable HIP graphs by default, without `torch.compile`.
No `VLLM_USE_BREAKABLE_CUDAGRAPH` override is required. This keeps graph capture
enabled for prefill and decode while selecting kernels for the captured batch
size, including the small-batch MXFP8 linear configurations.

The AMD backbone does not support `torch.compile`. Setting
`VLLM_USE_BREAKABLE_CUDAGRAPH=0` does not restore compiler support. Use
`--enforce-eager` to disable graph execution for debugging.

See [CUDA graphs](../design/cuda_graphs.md) for graph execution modes; the same
configuration terminology applies to HIP graphs on ROCm.
