# Last-layer FFN row pruning (prototype)

`VLLM_ENABLE_LAST_LAYER_FFN_PRUNING=1` enables an experimental optimization for
native dense Qwen3 models and dense Qwen2/Qwen2.5 through the Transformers
modeling backend. It is disabled by default.

During a complete prefill, ordinary next-token generation consumes only the last
prompt token's final hidden state for each request. The final attention layer
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

The prototype supports a single NVIDIA GPU, unquantized generation, and complete
prefills of at most 16 requests. It supports eager execution and breakable CUDA
graphs with torch compilation disabled. Breakable graphs are keyed by both the
ordinary batch descriptor and the number of selected rows, and selection indices
use a stable device buffer across replays.

A new request-count variant is captured on first use and then replayed. Capture
uses a separate CUDA stream and executes the new graph before returning the
first request's result. Up to 16 pruned variants can be retained per token-shape
descriptor, in addition to the ordinary graph. This increases first-use latency
and graph memory; warm steady-state timings exclude those costs.

Cached-prefix prefills, intermediate chunks, prompt logprobs, pooling,
multimodal inputs, LoRA, speculative decoding, KV transfer, parallel execution,
and microbatching retain the existing path. Full CUDA graphs and torch-compiled
piecewise graphs also retain the existing path.

After the final normalization, selected rows are placed back at their original
indices so the model runner keeps its token-indexed output contract. Other rows
are zeroed and are not consumed on the guarded generation path.

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
