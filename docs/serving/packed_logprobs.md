# Packed completion logprobs

For RL and distillation clients that consume numeric candidate scores, set
`return_top_k_logprobs: true` on `/v1/completions` with positive `logprobs` and
`return_tokens_as_token_ids: true`. Both streaming and non-streaming requests
are supported. Echo, beam search and `logprob_token_ids` are rejected.
Requests without this flag retain their existing response schema.

`choices[i].logprobs.top_k` uses the packed format proposed for
`/inference/v1/generate` in [#60915](https://github.com/vllm-project/vllm/pull/60915):
`num_positions`, `k`, `token_ids` and `logprobs`. Each base64 string contains raw
little-endian, row-major int32/float32 bytes with shape `[num_positions, k]`,
without an `.npy` header. `top_logprobs` is empty; `tokens`, `token_logprobs` and
`text_offset` still describe the sampled tokens. The completion API always
returns sampled scores in `token_logprobs`; no second flag is needed.
Sampled scores retain the completion API's lower clamp of -9999.

```python
import base64

import numpy as np
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
response = client.completions.create(
    model="facebook/opt-125m",
    prompt="Hello",
    max_tokens=16,
    logprobs=8,
    extra_body={
        "return_top_k_logprobs": True,
        "return_tokens_as_token_ids": True,
    },
)
packed = response.choices[0].logprobs.model_extra["top_k"]
ids = np.frombuffer(base64.b64decode(packed["token_ids"]), dtype="<i4")
scores = np.frombuffer(base64.b64decode(packed["logprobs"]), dtype="<f4")
ids = ids.reshape(packed["num_positions"], packed["k"])
scores = scores.reshape(packed["num_positions"], packed["k"])
```

With `stream=True`, each SSE choice carries arrays for only its new tokens.
Decode each block and concatenate once per completed choice; avoid expanding
the arrays into candidate dictionaries. Empty terminal blocks have shape
`[0, k]`. Finish reasons, usage chunks and `[DONE]` retain their usual meaning.
`stream_interval` can amortize SSE/JSON overhead across several tokens (for
example, 16). Packing does not introduce a separate buffering policy.

Candidates are the engine's top-k slots, in engine order, including ties.
Probabilities are not renormalized over k. The sampled token retains its own
score even when it is outside the head. Candidate arrays preserve negative
infinity. The server's configured logprobs mode still determines whether
scores are raw or processed; this flag does not change sampling.

This Python path uses upstream's existing list-backed `FlatLogprobs` container,
skips output-candidate detokenization, and packs its primitive columns without
reconstructing candidate `Logprob` objects. Packing allocates CPU arrays
proportional to positions × k. Prompt scores keep their existing representation
on the wire and decoding. No native extensions or GPU kernels are added.
