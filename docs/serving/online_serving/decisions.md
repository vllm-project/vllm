# Decisions API

vLLM provides a text-only implementation of OpenAI's
[`POST /v1/decisions`](https://developers.openai.com/api/docs/guides/decisions)
API. Start a supported model:

```bash
vllm serve Qwen/Qwen3-0.6B
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
decision = client.decisions.create(
    model="Qwen/Qwen3-0.6B",
    input="The package arrived with a broken screen.",
    questions=[
        {
            "type": "predicate",
            "name": "damaged",
            "instructions": "Does the customer report a damaged item?",
        },
        {
            "type": "choice",
            "name": "department",
            "instructions": "Which department should handle this complaint?",
            "choices": [{"value": "returns"}, {"value": "billing"}],
        },
        {
            "type": "score",
            "instructions": "How severe is the damage?",
            "levels": [{"label": "cosmetic"}, {"label": "broken"}],
        },
    ],
)
print(decision.answers)
```

Use OpenAI Python SDK 3.26.0 or newer for `client.decisions.create`.
`input` accepts a string or an array of user messages with string content or
`input_text` parts. Answers retain question order and echo `name`, including
`null` for unnamed questions. Choice values preserve strings and booleans.
Scores are probability-weighted averages of zero-based level indices.

The request contract follows the OpenAI OpenAPI schemas retrieved on
October 7, 2026. This MVP supports 1–64 questions, 2–26 choices, and 2–10 score
levels. Upstream permits 200 questions and 255 choices; this backend returns
a 400 when either model limit is exceeded.
The optional `safety_identifier` is caller metadata, available in request
body logging with `--enable-log-requests` and DEBUG logging; it does not
authenticate callers.

## Implementation and limits

This MVP uses the same autoregressive label-reading backend as
[structured decisions](structured_decisions.md). It supports the same Qwen
architectures and logprob modes; other models return 501. Each question reads
one token, using A–Z labels checked against the tokenizer at startup. Usage
reports the actual prompt and output tokens across these reads, including
prefix-cache counters.

Probabilities are normalized over the supplied options. `confidence` is the
winning label's probability over the full vocabulary; it is not calibrated
to OpenAI's models. Thinking is disabled for these reads.

Image inputs, other message roles, tools, streaming, and per-question refusal
scoring are outside this MVP. Unsupported request fields and input types are
rejected. The `/v1/systemone` endpoint retains its existing request format and
is available by default.
