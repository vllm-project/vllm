# Decisions API

vLLM provides an implementation of OpenAI's
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
`input_text` and `input_image` parts. Answers retain question order and echo `name`, including
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

Other message roles, tools, streaming, and per-question refusal
scoring are outside this MVP. Unsupported request fields and input types are
rejected. The `/v1/systemone` endpoint retains its existing request format and
is available by default.

## Winnow checkpoints

Winnow-trained Gemma4 checkpoints use a fixed turn serialization and candidate
labels rather than the generic chat prompt. Opt in explicitly:

```bash
vllm serve /path/to/winnow-checkpoint \
  --hf-overrides '{"decision_read_strategy":"winnow","decision_temperature":1.0}' \
  --logprobs-mode raw_logprobs --enable-prefix-caching
```

This uses the shared Decisions API and label-read backend. Each question is
scheduled independently with the same state prefix. Request-specific chat
instructions or chat-template overrides are rejected. The upstream limits of
64 questions and 26 options apply. The API's confidence and response schemas
remain unchanged; they differ from Ollaya's Jev wire responses.

`decision_temperature` divides candidate log probabilities before their
conditional softmax. Use 1 when the required scale is folded into the exported
weights. An additional validation-fitted temperature in `calibration.json` is
external: set `decision_temperature` to that value for that calibrated variant.
Temperature changes conditional answer probabilities; `confidence` retains the
winning label's original full-vocabulary probability from the engine read.
Preserve the exported
Gemma logit-softcap configuration. Loading an unrelated Gemma4 checkpoint does
not make it Winnow-trained.

The checkpoint must be compatible with the installed engine's model and
quantization loaders. This strategy does not add or change model loaders.

The Decisions API accepts text. For a checkpoint deployment that interprets
input as a JSON state, set `decision_state_format` to `json` in `--hf-overrides`
and send a JSON document as the `input` string. The strategy parses that document
before applying Winnow's canonical state serialization. This preserves object
and array states when comparing against Ollaya. The default `text` mode treats
input literally. Invalid JSON in `json` mode fails before inference.

### Image input (stacked vision change)

For a supported vision-language model, `/v1/decisions` accepts ordered user
content with `input_text` and `input_image` parts. Image URLs use the existing
vLLM multimodal renderer and its media restrictions and processor cache.

```json
{
  "model": "vision-decision-model",
  "input": [{"role": "user", "content": [
    {"type": "input_text", "text": "Inspect the image."},
    {"type": "input_image", "image_url": "https://example.org/image.png"}
  ]}],
  "questions": [{"type": "predicate", "instructions": "Is a vehicle visible?"}]
}
```

Each question retains the same ordered image/state prefix and its own question
suffix. Model-specific processors expand image placeholders before inference;
image metadata stays in the engine input. Context limits apply to the expanded
input. Text-only models reject image requests.

Winnow image input requires a native `Gemma4ForConditionalGeneration` export
with the matching vision tower and projector. A text-only
`Gemma4ForCausalLM` export cannot gain vision support by changing its architecture
name. Winnow uses its fixed trained turns and image-attachment prefix through the
Gemma multimodal renderer, rather than the generic decision chat prompt.

Implementation validation is in progress. CPU tests cover schema, ordered media
at the renderer boundary, fixed Winnow image formatting and text regressions.
These tests do not establish real image accuracy or quantized vision support;
actual model and precision validation must accompany any support claim.

### Image compatibility checks

The image API was exercised on native BF16 Winnow-derived Gemma4 and Qwen3.5
conditional-generation exports with their frozen original vision weights. Each
passed five requests covering predicate, choice and score questions, changed
image content, swapped two-image order, and a repeated image after other inputs.
Explicit CPU offload was used on a 12GB RTX4070; these are functional checks,
not resident throughput or visual-quality benchmarks. Qwen used the generic
chat read strategy, so this does not measure the trained plain decision prompt.

With a compatible vision server already running, reproduce the generated-color
checks using:

```bash
.venv/bin/python benchmarks/decision_models/check_images.py --model MODEL --output /tmp/image-check
```

The benchmark requires correct red/blue classification for its simple fixtures
and repeated probabilities within 1e-6. It does not establish broad vision
accuracy or image-specific calibration.

Compact runtime evidence is in
`benchmarks/decision_models/vision-verification.json`.
