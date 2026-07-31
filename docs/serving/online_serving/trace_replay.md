# Trace Replay

Trace replay forces the engine to emit a predetermined token sequence during decoding while computing real logprobs from the model's unmodified logit distribution. The primary use case is comparing logprob distributions across different configurations:

- **Inference config diff**: compare logprobs between different quantization schemes, tensor parallelism layouts, or attention backends for the same model.
- **Train vs inference diff**: replay a training-time token sequence through the inference engine to detect logprob divergence caused by numerical differences between training and serving frameworks.

## Usage

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen3-0.6B")

# Token sequence captured from a previous run or training log
trace_tokens = [15, 284, 1026, 374]

params = SamplingParams(
    trace_decode_token_ids=trace_tokens,
    logprobs=5,
)
outputs = llm.generate(["Once upon a time"], sampling_params=params)

for token, logprob in zip(
    outputs[0].outputs[0].token_ids,
    outputs[0].outputs[0].logprobs,
):
    print(f"token={token}  logprob={logprob[token].logprob:.4f}")
```

The output tokens will always be `[15, 284, 1026, 374]`. The logprobs reflect the model's true probability for each forced token under the current inference configuration.

To compare configurations, run the same trace against two engine setups and diff the per-token logprobs.

## Behavior

When `trace_decode_token_ids` is set:

- `max_tokens` is automatically set to the trace length.
- All stop conditions (EOS, stop strings, stop token IDs) are disabled.
- Generation produces exactly the trace tokens, then stops.

## Limitations

`trace_decode_token_ids` is incompatible with the following features (raises `ValueError`):

| Feature | Reason |
| --- | --- |
| `n > 1` | Trace replay produces a single deterministic sequence |
| `prompt_logprobs` | Expanded logit layout conflicts with trace kernel indexing |
| Speculative decoding | Multi-token speculation conflicts with single-token trace stepping |
| Structured outputs | Grammar constraints conflict with forced token injection |
| `repetition_detection` | Would terminate the request on repeated patterns in the trace |
| `thinking_token_budget` | Logit masking corrupts logprobs for trace tokens |
| `bad_words` | Logit masking corrupts logprobs for trace tokens |
