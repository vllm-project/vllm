# Packed completion logprobs

For RL and distillation clients that consume numeric candidate scores, set
`packed_top_logprobs: true` on `/v1/completions` with positive `logprobs` and
`return_tokens_as_token_ids: true`. Both streaming and non-streaming requests
are supported. Echo, beam search and `logprob_token_ids` are rejected.
Requests without this flag retain their existing response schema.

`choices[i].logprobs.packed_top_logprobs` contains `shape: [tokens, k]`,
`token_ids_dtype: "<i4"`, `logprobs_dtype: "<f4"`, `token_ids_b64` and
`logprobs_b64`. Each base64 string contains raw little-endian, row-major
int32/float32 bytes, without an `.npy` header. `top_logprobs` is empty;
`tokens`, `token_logprobs` and `text_offset` still describe the sampled tokens.
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
        "packed_top_logprobs": True,
        "return_tokens_as_token_ids": True,
    },
)
packed = response.choices[0].logprobs.model_extra["packed_top_logprobs"]
ids = np.frombuffer(base64.b64decode(packed["token_ids_b64"]), dtype="<i4")
scores = np.frombuffer(base64.b64decode(packed["logprobs_b64"]), dtype="<f4")
ids = ids.reshape(packed["shape"])
scores = scores.reshape(packed["shape"])
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

This Python path uses the existing `FlatLogprobs` container, skips
output-candidate detokenization, and packs directly without reconstructing
candidate `Logprob` objects. It adds no native extensions or GPU kernels.
Prompt scores keep their existing representation on the wire and decoding.

To measure CPU formatting, JSON encoding, response size and client decoding:

```bash
python benchmarks/benchmark_packed_logprobs.py --tokens 1000 --top-k 8 32 128
```

This synthetic benchmark excludes inference, engine logprob processing, HTTP,
and SSE framing; its timings are not end-to-end generation throughput.
