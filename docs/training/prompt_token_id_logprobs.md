# Prompt Token ID Logprobs (Teacher Scoring)

In on-policy distillation, the student records its top-K candidate tokens at
every response position, and the teacher provides the log probabilities of
exactly those candidates as the distillation target. The caller passes one
candidate list per scored row as `prompt_logprob_token_ids`; the teacher runs a
single prefill over the prompt plus response and gathers their log
probabilities on the GPU, starting from `prompt_logprob_start` (typically the
prompt length minus one, so only response positions are scored). The result is
a `[rows, K]` matrix whose row `i` scores prompt token
`prompt_logprob_start + i + 1` and whose columns follow that row's candidate
order, so it lines up position by position with the student's top-K.

## Quick start

```python
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

llm = LLM(model)
output = llm.generate(
    TokensPrompt(prompt_token_ids=prompt_ids + response_ids),
    SamplingParams(
        max_tokens=1,
        prompt_logprob_token_ids=student_topk_ids,  # [len(response_ids), K]
        prompt_logprob_start=len(prompt_ids) - 1,
    ),
)
scores = output[0].prompt_token_id_logprobs
# scores.shape == (len(response_ids), K)
# scores[i, j] = log p(student_topk_ids[i][j] | prompt_ids + response_ids[:i])
```

The same parameters are accepted by the `/inference/v1/generate` HTTP endpoint
(Python and Rust frontends), which returns the matrix as a base64-encoded
`.npy` float32 array:

```python
import base64, io
import numpy as np

scores = np.load(io.BytesIO(base64.b64decode(response["prompt_token_id_logprobs"])))
```

`prompt_logprob_token_ids` is an integer array or a list of lists with exactly
`prompt_len - 1 - prompt_logprob_start` rows; shorter rows are padded with `-1`,
and every `-1` entry scores `-inf`. Pass an `int32` NumPy array, such as the
student's top-K IDs, which is sent to the engine without conversion; nested
lists of Python ints are much slower to copy and serialize. With a logits
`--logprobs-mode`, the matrix holds logits instead of log probabilities.

## Requirements

- The V2 model runner (`VLLM_USE_V2_MODEL_RUNNER=1`).
- The longest row at most `--max-logprobs` candidates.
- No `--kv-sharing-fast-prefill`.

By default, the request skips reading the prefix cache (local and KV connector),
since cached rows have no logits to score.

## Reusing the context prefix

Set `skip_reading_prefix_cache=False` to reuse local KV before
`prompt_logprob_start`. Every requested scoring row is still computed, even if
the cache contains a previous copy of the entire response. Cache hits are rounded
down to the available block boundary. For example, a 2048-token context has
`prompt_logprob_start=2047`; with 16-token blocks, at most 2032 tokens are reused.

This path supports text token inputs, one causal full-attention KV group,
unquantized KV, `max_tokens=1`, and single-GPU or tensor-parallel
execution. Chunked prefill and prefill preemption retain the scoring boundary.
The generated token is still returned; scoring callers can discard it.

Requests that also ask for ordinary `prompt_logprobs` use full recomputation.
So do configurations with KV/offload connectors, auxiliary
output capture, hybrid or sliding-window attention, partial-block reuse,
multimodal inputs or prompt embeddings, LoRA, speculative decoding, resumable
sessions, or other parallel topologies. The existing input validation still
applies. Neither the scoring output format nor the no-read default changes.
An unsupported opt-in configuration logs a warning once before falling back.

## Limitations

- Non-streaming only; `stream=true` requests are rejected.
- Not exposed through the OpenAI-compatible endpoints.
