# Prompt Token ID Logprobs (Teacher Scoring)

In on-policy distillation, the student records its top-K candidate tokens at
every response position, and the teacher provides the log probabilities of
exactly those candidates as the distillation target. The caller passes the
candidates as `prompt_logprob_token_ids`; the teacher runs a single prefill
over the prompt plus response and gathers their log probabilities on the GPU at
every scored row, starting from `prompt_logprob_start` (typically the prompt
length minus one, so only response positions are scored). The result is a
`[rows, K]` matrix whose row `i` scores prompt token
`prompt_logprob_start + i + 1` and whose columns follow the candidate order, so
it lines up position by position with the student's top-K.

## Quick start

```python
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

llm = LLM(model)
output = llm.generate(
    TokensPrompt(prompt_token_ids=prompt_ids + response_ids),
    SamplingParams(
        max_tokens=1,
        prompt_logprob_token_ids=candidate_ids,
        prompt_logprob_start=len(prompt_ids) - 1,
    ),
)
scores = output[0].prompt_token_id_logprobs
# scores.shape == (len(response_ids), len(candidate_ids))
# scores[i, j] = log p(candidate_ids[j] | prompt_ids + response_ids[:i])
```

The same parameters are accepted by the `/inference/v1/generate` HTTP endpoint
(Python and Rust frontends), which returns the matrix as a base64-encoded
`.npy` float32 array:

```python
import base64, io
import numpy as np

scores = np.load(io.BytesIO(base64.b64decode(response["prompt_token_id_logprobs"])))
```

The matrix has `max(prompt_len - 1 - prompt_logprob_start, 0)` rows. With a
logits `--logprobs-mode`, it holds logits instead of log probabilities.

## Requirements

- The V2 model runner (`VLLM_USE_V2_MODEL_RUNNER=1`).
- `len(prompt_logprob_token_ids)` at most `--max-logprobs`.
- No `--kv-sharing-fast-prefill`.

The request skips reading the prefix cache (local and KV connector), since
cached rows have no logits to score.

## Limitations

- One candidate set per request, shared by every scored row; per-position
  candidates are planned ([#56860](https://github.com/vllm-project/vllm/issues/56860)).
- Non-streaming only; `stream=true` requests are rejected.
- Not exposed through the OpenAI-compatible endpoints.
- If a prefill starts past the first scored row (e.g. `skip_reading_prefix_cache=False`
  with a cache hit), the result is `None` rather than a partial matrix.
