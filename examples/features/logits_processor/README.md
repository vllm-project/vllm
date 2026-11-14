# Custom Logits Processors

This directory contains examples demonstrating how to use custom logits processors with vLLM's offline inference API. Logits processors allow you to modify the model's output distribution before sampling, enabling controlled generation behaviors like token masking, constrained decoding, and custom sampling strategies.

## Scripts

### `dry.py` — DRY repetition penalty

Implements the DRY (Don't Repeat Yourself) repetition penalty, ported from llama.cpp, as `DryState`, a custom logits processor. Requests turn it on through `SamplingParams.extra_args`, for example `{"dry_multiplier": 0.8}`. The example sends one prompt twice in one batch, once without DRY and once with it, and prints both outputs. `tests/v1/sample/test_dry.py` imports the processor from this file.

```bash
python examples/features/logits_processor/dry.py
```

## Key Concepts

- **Batch-level processing**: vLLM processes logits at the batch level for efficiency; see the [custom logits processors guide](../../../docs/features/custom_logitsprocs.md) for the interface.
- **`SamplingParams.extra_args`**: Use this to pass custom keyword arguments to your logits processor on a per-request basis (e.g., `target_token`).

## Further Reading

- [vLLM Sampling Parameters](https://docs.vllm.ai/en/latest/api/inference_params.html#sampling-parameters)
- [vLLM LLM API](https://docs.vllm.ai/en/latest/api/offline_inference/llm.html)
