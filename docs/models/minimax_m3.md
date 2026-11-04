# MiniMax-M3

vLLM supports MiniMax-M3 (`MiniMaxAI/MiniMax-M3`, `MiniMaxAI/MiniMax-M3-MXFP8`, etc.)
through `MiniMaxM3SparseForCausalLM` and `MiniMaxM3SparseForConditionalGeneration`.

## AMD ROCm graph execution

MiniMax-M3 on AMD ROCm uses breakable HIP graphs by default, without `torch.compile`.
Decode steps replay full graphs. Steps with prefill rows, up to
`max_prefill_cudagraph_capture_tokens` tokens, replay piecewise graphs with the sparse
attention running eagerly between the pieces; without them every such step runs eagerly.
Set `VLLM_USE_BREAKABLE_CUDAGRAPH=0` to go back to decode-only graphs.

The mono decode path (`VLLM_ROCM_MINIMAX_M3_MONO=1`) serves decode steps under
decode-only graphs only: pass `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'`
or `VLLM_USE_BREAKABLE_CUDAGRAPH=0` with it.

See [CUDA graphs](../design/cuda_graphs.md) for graph execution modes; the same
configuration terminology applies to HIP graphs on ROCm.
