# Custom Logits Processors

This directory contains examples demonstrating how to use custom logits processors with vLLM's offline inference API. Logits processors allow you to modify the model's output distribution before sampling, enabling controlled generation behaviors like token masking, constrained decoding, and custom sampling strategies.

## Scripts

### `custom.py` — Engine-level logits processor

Demonstrates how to instantiate vLLM with a custom logits processor class that operates at the batch level. The example uses a `DummyLogitsProcessor` that masks out all tokens except a specified `target_token` when passed via `SamplingParams.extra_args`.

```bash
python examples/offline_inference/logits_processor/custom.py
```

### `custom_req.py` — Request-level logits processor wrapper

Shows how to wrap a request-level logits processor (which operates on individual requests) to be compatible with vLLM's batch-level logits processing interface.

```bash
python examples/offline_inference/logits_processor/custom_req.py
```

### `custom_req_init.py` — Request-level processor with engine config

A special case of wrapping a request-level logits processor where the processor needs access to engine configuration or model metadata during initialization (e.g., vocabulary size, tokenizer info).

```bash
python examples/offline_inference/logits_processor/custom_req_init.py
```

## Key Concepts

- **Batch-level vs. request-level**: vLLM processes logits at the batch level for efficiency. If you have a per-request processor, you need to wrap it using the patterns shown in `custom_req.py` and `custom_req_init.py`.
- **`SamplingParams.extra_args`**: Use this to pass custom keyword arguments to your logits processor on a per-request basis (e.g., `target_token`).
- **`DummyLogitsProcessor`**: A reference implementation available in `vllm/test_utils.py` that can be used as a starting point for custom processors.

## Further Reading

- [vLLM Sampling Parameters](https://docs.vllm.ai/en/latest/api/inference_params.html#sampling-parameters)
- [vLLM LLM API](https://docs.vllm.ai/en/latest/api/offline_inference/llm.html)
