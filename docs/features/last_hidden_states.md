# Last Hidden States

A generation request can ask for the hidden state that each of its generated
tokens' logits were computed from: the tensor the model's `compute_logits`
received at that position (for most decoder models, the last layer's output
after the final norm). It comes back with the request's final output, from the
same forward pass that produced the tokens and their logprobs.

Typical use: a small head trained on top of the served model (a linear probe, a
classifier over a fixed label set, a calibration layer) that needs the model's
scores and its hidden state for the same input, without a second pass through a
pooling engine.

## Quick start

```python
from vllm import LLM, SamplingParams

llm = LLM("facebook/opt-125m", enable_return_last_hidden_states=True)
params = SamplingParams(max_tokens=4, return_last_hidden_states=True)
completion = llm.generate("The capital of France is", params)[0].outputs[0]

hidden = completion.last_hidden_states
# torch.Tensor of shape [len(completion.token_ids), hidden_size], on the CPU,
# in the model's dtype; row i is the state token i's logits came from.
```

A complete example that scores a fixed label set and applies a linear head to
the state of the same forward is in
[`examples/generate/last_hidden_states_offline.py`](../../examples/generate/last_hidden_states_offline.py).

The output is part of `CompletionOutput` in the Python API (`LLM`, `AsyncLLM`);
this feature does not add it to the OpenAI-compatible HTTP responses.

## Requirements

| Requirement | Reason |
| --- | --- |
| `--enable-return-last-hidden-states` (`enable_return_last_hidden_states=True`) | Engine-level opt-in; requests that ask on an engine without it are rejected |
| `SamplingParams.return_last_hidden_states=True` | Per-request opt-in; other requests are unchanged |
| Model Runner V2 | The rows travel with its asynchronous device-to-host output copy |

The engine rejects at startup:

- Speculative decoding
- Pipeline parallelism
- Context parallelism (decode or prefill)
- Pooling and diffusion models

## How it works

1. After sampling, the model runner gathers the rows of the hidden states that
   were passed to `compute_logits` for the requests that asked.
2. The rows are copied to the host on the output copy stream, with the sampled
   tokens.
3. The scheduler keeps one row per token the request keeps (stop tokens and
   `max_tokens` behave as for logprobs), and the output processor concatenates
   them when the request finishes.

With the engine flag off, the model runner holds no state and does no work for
this feature. Rows are not recomputed: they are the exact tensors the logits
came from, so they inherit the platform's run-to-run numerics (see
[batch invariance](batch_invariance.md) for bitwise reproducibility across
batch compositions).
