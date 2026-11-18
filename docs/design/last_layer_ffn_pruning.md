# Last-layer FFN row pruning (prototype)

`VLLM_ENABLE_LAST_LAYER_FFN_PRUNING=1` enables an experimental optimization for
native dense Qwen3 and Llama models, and dense Qwen2/Qwen2.5 through the Transformers
modeling backend. It is disabled by default.

During prefill, ordinary next-token generation consumes only the sampled row's
final hidden state for each request. The final attention layer
still computes all rows and writes all keys and values. Immediately afterward,
the model selects each request's last hidden-state and residual row and runs the
remaining normalization and FFN operations on these rows. No weights or attention
context are removed.

For Qwen2/Qwen2.5, only the final FFN input rows are selected, and its outputs
are scattered back to their original positions. The existing residual and
normalization operations keep their ordinary shapes. This preserves the
Transformers backend and parameter names rather than substituting a different
model implementation.

Changing GEMM row count can change floating-point rounding;
mathematical equivalence does not imply bitwise-identical BF16 output.

The prototype supports unquantized generation on NVIDIA GPUs and prefill
batches of at most 16 requests, including cached prefixes, chunked prefill,
and mixed prefill/decode batches. It also supports LoRA, tensor parallelism,
and pipeline parallelism. It supports eager execution and breakable CUDA
graphs with torch compilation disabled. Breakable graphs are keyed by both the
ordinary batch descriptor and the number of selected rows, and selection indices
use a stable device buffer across replays.

A new request-count variant is captured on first use and then replayed. Capture
uses a separate CUDA stream and executes the new graph before returning the
first request's result. Up to 16 pruned variants can be retained per token-shape
descriptor, in addition to the ordinary graph. This increases first-use latency
and graph memory; warm steady-state timings exclude those costs.

For cached prefixes and chunks, only rows scheduled in the current step are
considered. Every request retains its logits row, including decode requests in
a mixed batch; intermediate-chunk logits are discarded by the existing runner.
Attention still writes KV for all scheduled tokens, so later chunks and decode
steps retain the complete context.

LoRA uses the existing per-output adapter mapping inside the reduced FFN and
restores the per-token mapping afterward. This keeps mixed-adapter batches
aligned with the selected rows without additional metadata transfers. TP ranks
select the same rows; in PP, only the final pipeline stage prunes, while earlier
stages transmit all intermediate rows unchanged.

Prompt logprobs, pooling, multimodal inputs, speculative decoding, KV transfer,
data/context parallelism, and microbatching retain the existing path. Full CUDA graphs and torch-compiled
piecewise graphs also retain the existing path.

For native Qwen3 and Llama, selected rows are placed back at their original
indices after final normalization, and other rows are zeroed. Qwen2/Qwen2.5
restores FFN rows before residual addition and normalization. Both paths preserve
the runner's token-indexed output contract; only selected rows are consumed.

For a breakable-graph experiment, use the same configuration for both arms:

Set `VLLM_USE_BREAKABLE_CUDAGRAPH=1` before starting vLLM; setting the
compilation configuration field alone is overridden during configuration
resolution.

```python
compilation_config = {
    "mode": 0,
    "cudagraph_mode": "FULL_AND_PIECEWISE",
    "use_breakable_cudagraph": True,
}
```

Change only `VLLM_ENABLE_LAST_LAYER_FFN_PRUNING` between runs. Exclude graph
capture and warmup from timing and disable prefix caching for cold-prefill
comparisons.
