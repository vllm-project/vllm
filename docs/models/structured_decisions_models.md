# Structured Decisions Models

Structured Decisions Models (aka JEV Models) map inputs directly to predefined
options or categories and return probabilities and confidence scores, commonly
used for classification and routing.

## Supported Models

### Text-only Models

| Architecture              | ReadStrategy        | Task             | Head              | Example HF Models       |
|---------------------------|---------------------|------------------|-------------------|-------------------------|
| `ClefFlashForJointSchema` | `ClefFlashStrategy` | `token_classify` | `JointSchemaHead` | `Cloudflare/clef-flash` |
| `Gemma4ForCausalLM`       | `NextTokenStrategy` | `generate`       | N/A               | `google/gemma-4-E2B-it` |
| `Qwen3ForCausalLM`        | `NextTokenStrategy` | `generate`       | N/A               | `Qwen/Qwen3-0.6B`, etc. |

### Multimodal Models

!!! note
    For more information about multimodal models inputs, see [this page](./supported_models.md#list-of-multimodal-language-models).

| Architecture                            | Inputs                                | ReadStrategy                   | Task             | Head              | Example HF Models                        |
|-----------------------------------------|---------------------------------------|--------------------------------|------------------|-------------------|------------------------------------------|
| `ClefOmniForJointSchema`                | T + I + V                             | `JointSchemaStrategy`          | `token_classify` | `JointSchemaHead` | `Cloudflare/clef-omni`                   |
| `D3ForSequenceClassification`           | T + I + V                             | `D3Strategy`                   | `classify`       | Linear (`score`)  | `vllm-sr/d3`, `vllm-sr/d3-mini`, etc.    |
| `Gemma4ForConditionalGeneration`        | T + I<sup>+</sup> + V + A<sup>*</sup> | `NextTokenStrategy`            | `generate`       | N/A               | `google/gemma-4-E2B-it`, etc.            |
| `Gemma4UnifiedForConditionalGeneration` | T + I<sup>+</sup> + V + A             | `NextTokenStrategy`            | `generate`       | N/A               | `google/gemma-4-12B-it`, etc.            |
| `Lfm2VlForConditionalGeneration`        | T + I<sup>+</sup>                     | `LiquidStrategy`               | `generate`       | N/A               | `LiquidAI/d1-3B`                         |
| `Qwen3_5ForConditionalGeneration`       | T + I<sup>E+</sup> + V<sup>E+</sup>   | `NextTokenStrategy`            | `generate`       | N/A               | `Qwen/Qwen3.5-0.8B`, etc.                |
| `Qwen3_5MoeForConditionalGeneration`    | T + I<sup>E+</sup> + V<sup>E+</sup>   | `NextTokenStrategy`            | `generate`       | N/A               | `Qwen/Qwen3.5-35B-A3B`, etc.             |
| `DiffusionGemmaForBlockDiffusion`       | T + I<sup>+</sup> + V<sup>+</sup>     | `DiffusionGemmaCanvasStrategy` | `generate`       | N/A               | `google/diffusiongemma-26B-A4B-it`, etc. |

## Online Serving

### TypeSafe System One Endpoint

The `/v1/systemone` endpoint answers a set of typed questions about a state and
returns a probability for every allowed answer.

The endpoint is available by default on generation models. Models without a
supported read strategy return 501. Supported architectures require
`--logprobs-mode raw_logprobs` or `processed_logprobs`, since `label_mass` needs
log probabilities.

#### How it works

1. Each question is one read. The user message is the state, followed by the
   question and its options with the type's labels, and an instruction to
   answer with one label.
2. The read is the reply's first token. Each label must be one token there,
   or the request returns 400.
3. The read returns the logprobs of that question's label tokens. Softmax over
   the labels gives the answer's probabilities.

Thinking is off unless `chat_template_kwargs` turns it on.

Probabilities can differ between identical requests. They depend on which
requests share a batch and on whether a read hits the prefix cache. Set
`VLLM_BATCH_INVARIANT=1` for reads that repeat exactly, at a cost in
throughput.

Every read of a request starts with the state, so with
`--enable-prefix-caching` the state is prefilled once and each further
question prefills only its own text, then reads one token.

#### Question types

| type | criteria | answer |
| --- | --- | --- |
| `choice` | map of option name to a description or `null` | `choice`, `probabilities` by option name, `confidence` |
| `noul` | `null`, or an object with optional `true` and `false` descriptions | `noul` (probability of yes), `probabilities`, `confidence` |
| `score` | ordered list of level names | `score` (expected 0-indexed level), `legend`, `probabilities` by level index, `confidence` |

A choice has at least one option, a score at least two levels. A question id
is a non-empty string.

#### Example

```bash
curl -s http://localhost:8000/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "state": {"ticket": "My card was charged twice for one order."},
    "questions": {
      "team": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": {
          "billing": "payments and refunds",
          "shipping": "deliveries",
          "security": "account access"
        }
      },
      "urgent": {
        "type": "noul",
        "instructions": "Does the customer need a reply within the hour?",
        "criteria": {"true": "the account is locked or money is moving"}
      },
      "tone": {
        "type": "score",
        "instructions": "How upset is the customer?",
        "criteria": ["calm", "annoyed", "furious"]
      }
    },
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

Response, with probabilities that depend on the model:

```json
{
  "id": "decision-...",
  "object": "structured_decision",
  "created": 1790000000,
  "model": "Qwen/Qwen3-0.6B",
  "answers": {
    "team": {
      "type": "choice",
      "choice": "billing",
      "probabilities": {"billing": 0.97, "shipping": 0.01, "security": 0.02},
      "confidence": 0.90
    },
    "urgent": {
      "type": "noul",
      "noul": 0.02,
      "probabilities": {"yes": 0.02, "no": 0.98},
      "confidence": 0.95
    },
    "tone": {
      "type": "score",
      "score": 0.6,
      "legend": {"0": "calm", "1": "annoyed", "2": "furious"},
      "probabilities": {"0": 0.7, "1": 0.2, "2": 0.1},
      "confidence": 0.65
    }
  },
  "usage": {"input_tokens": 142, "output_tokens": 3},
  "diagnostics": {
    "team": {"label_mass": 0.93, "argmax_is_label": true},
    "urgent": {"label_mass": 0.98, "argmax_is_label": true},
    "tone": {"label_mass": 0.91, "argmax_is_label": true}
  }
}
```

`diagnostics.label_mass` is the probability the model put on the labels across
its whole vocabulary. A low value means most of the probability went to tokens
that are not labels. `confidence` is the chosen option's probability over the
whole vocabulary: its share of the labels times `label_mass`.

#### Labels

Labels follow the type: a `choice` labels its options `A` to `Z` in the order
the request lists them, a `noul` labels its options `yes` and `no`, and a
`score` labels its levels `0` to `9`, so a level's label is its score.

#### Limits

| limit | value |
| --- | --- |
| questions per request | 64 |
| options per `choice` | 26 |
| levels per `score` | 10 |

#### Joint reads

With `"joint": true`, a diffusion model reads several questions from one canvas in
one denoising step: the prompt numbers the questions, and each answer is read at
its slot in a `number: answer` line. A one-question request is read per
question, as without `joint`.

Questions that do not fit the served canvas (`--diffusion-config
'{"canvas_length": N}'`) are split, in order, into the fewest even runs that
fit. Each run is one engine request and counts once for admission.

The response adds a `joint` object, `null` when each question was read on its own:

```json
"joint": {"canvases": [{"questions": ["urgent", "bucket", "tone"], "width": 16, "slots": [2, 6, 11]}]}
```

- Answers can differ from per-question reads, since the slots see each other
  and the question order.
- `usage.output_tokens` counts each canvas once, up to its last slot.
- Joint reads take the same limits as per-question reads: 64 questions per
  request, no thinking, and each prompt plus its canvas within `max_model_len`.
- `/v1/decisions` takes the same `joint` field as a vLLM extension.

#### Request fields

| field | meaning |
| --- | --- |
| `state` | what the questions are about: a string, or JSON that is sent as its JSON text |
| `questions` | question id to `{type, instructions, criteria}`, asked in this order |
| `instructions` | optional context placed ahead of the questions |
| `chat_template_kwargs` | passed to the chat template, for example `{"enable_thinking": false}` |
| `cache_salt` | optional [prefix-cache salt](../usage/security.md#cache-salting) applied to every question |
| `joint` | optional boolean, default `false`; see [Joint reads](#joint-reads). DiffusionGemma only, other models return 400 |
| `images`, `audio`, `videos` | optional media URLs or data URLs, for models that take them ([Clef-Flash](#clef-flash), images/video; [Clef-Omni](#clef-omni); [Decision 3.0](#decision-30), images/video; DiffusionGemma, images only, with or without `joint`; [Liquid d1](#liquid-d1), images only; multimodal `NextTokenStrategy` models such as Qwen3.5, images only) |

By default, `/v1/systemone` rejects `chat_template_kwargs.chat_template` with
HTTP 400. Start the server with `--trust-request-chat-template` to allow
request-provided chat templates. Other template kwargs, such as
`enable_thinking`, do not require this flag.

Unknown top-level fields, and question fields other than `type`,
`instructions` and `criteria`, are rejected with a 400.

When [tracing](../../examples/observability/opentelemetry/README.md) is enabled
with `--otlp-traces-endpoint`, the HTTP `traceparent` and `tracestate` headers are
forwarded to every question's generation request.

### OpenAI Decisions API

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

#### Implementation and limits

This MVP uses the same label-reading backend as
[structured decisions](structured_decisions_models.md). It supports generation
architectures with a registered read strategy, including Qwen, DiffusionGemma,
and Liquid d1; unsupported models return 501. Each question reads a model's
label tokens. Usage reports the actual prompt and output tokens across these
reads, including prefix-cache counters.

Probabilities are normalized over the supplied options. `confidence` is the
winning label's probability over the full vocabulary; it is not calibrated
to OpenAI's models. Thinking is disabled for these reads.

Image inputs, other message roles, tools, streaming, and per-question refusal
scoring are outside this MVP. Unsupported request fields and input types are
rejected. The `/v1/systemone` endpoint retains its existing request format and
is available by default.

## Examples

### Clef-Flash

[Clef-Flash](https://huggingface.co/Cloudflare/clef-flash) uses a Qwen3.5-9B
backbone and a joint schema head to score all questions in one forward pass.
vLLM detects the head and serves the checkpoint as a pooling model.

```bash
vllm serve Cloudflare/clef-flash --max-model-len 16384
```

Send requests to `/v1/systemone` using `state` and `questions`, as in the
examples above. `choice`, `noul` and `score` questions are supported.

- Text and JSON states use the checkpoint's fixed prompt. JSON is compact
  with sorted keys. `instructions` returns 400; `chat_template_kwargs`
  has no effect.
- The state is truncated to fit 16,384 tokens, or `max_model_len` if smaller.
  A schema that exceeds this bound returns 400.
- `images` and `videos` are placed before the state. Videos are sampled at
  2 fps and processed by the Qwen3.5 vision processor. Audio inputs return
  400, and video soundtracks are ignored.
- Each request performs one pooling read. `usage.input_tokens` counts the
  prompt once, `output_tokens` is 0, and `label_mass` is 1.

### Clef-Omni

[Clef-Omni](https://huggingface.co/Cloudflare/clef-omni) is a decision model:
a joint schema head on the Qwen3-Omni thinker scores every option of every
question in one forward pass. vLLM serves it as a pooling model.

```bash
vllm serve Cloudflare/clef-omni
```

- A request is one read. The prompt is the one the checkpoint's
  `joint_schema_model.py` builds: the state, then every question with its
  options. A JSON state is written as compact JSON with sorted keys.
- A state too long for the model is truncated to fit, as the reference does.
  A request whose questions alone exceed 64,000 tokens, or `max_model_len`,
  returns 400.
- `instructions` returns 400, since the prompt is fixed. `chat_template_kwargs`
  has no effect.
- There are no labels, so `label_mass` is 1 and `confidence` is the chosen
  option's probability. `usage.input_tokens` counts the prompt once, and
  `output_tokens` is 0.
- `images`, `audio` and `videos` are read as the reference does: placed
  before the state, videos sampled at 2 fps, and a video's soundtrack heard
  with it when every video has one. Audio clips cannot be sent together with
  videos that have soundtracks.

### Decision 3.0

The Decision 3.0 models are a Qwen3.5 backbone with an answer-code readout:
one forward pass per question, and the readout scores the question's options
at the last token. vLLM detects the checkpoint's `decision_config.json` and
serves it as a classification model. All six sizes take the same command:
[`vllm-sr/d3`](https://huggingface.co/vllm-sr/d3) (27B),
[`vllm-sr/d3-flash`](https://huggingface.co/vllm-sr/d3-flash) (9B),
[`vllm-sr/d3-mini`](https://huggingface.co/vllm-sr/d3-mini) (4B),
[`vllm-sr/d3-nano`](https://huggingface.co/vllm-sr/d3-nano) (2B),
[`vllm-sr/d3-lite`](https://huggingface.co/vllm-sr/d3-lite) (0.8B) and
[`vllm-sr/d3-edge`](https://huggingface.co/vllm-sr/d3-edge) (0.6B).

```bash
vllm serve vllm-sr/d3
```

- Each question is one read of the prompt the checkpoint's `d3_format.py`
  writes: the state, the question and one line per option under an answer
  code. A question's probabilities are the softmax of its options' code
  logits at the checkpoint's temperature, so picks and probabilities match
  `model.system_one()`. `confidence` and `usage` follow the definitions on
  this page instead.
- A choice can have up to 255 options. A JSON state is written as JSON text,
  and an empty state as `(empty)`.
- The full attention layers read the whole prompt in both directions, as the
  checkpoint declares, so chunked prefill and prefix caching are off.
- `instructions` returns 400, since the prompt is fixed.
  `chat_template_kwargs` has no effect.
- `images` and `videos` are placed before the text, images read at up to
  1.6 megapixels and videos at 2 fps, at most 32 frames of up to
  0.2 megapixels each, as the checkpoint's runtime does. The media are read
  once and shared by every question of the request.
- `/classify` and `/pooling` return the raw logits of all 255 codes; only the
  first codes, one per option, belong to a question.

### Liquid d1

[`LiquidAI/d1-3B`](https://huggingface.co/LiquidAI/d1-3B) is LFM2.5-VL-3B
post-trained to answer decisions as the first token of its reply. vLLM serves
it as a generative model.

```bash
vllm serve LiquidAI/d1-3B
```

- The server writes the prompt and reads the label tokens the way the
  `prompt.py` shipped with the checkpoint does, so picks and probabilities
  match `model.system_one()`. `confidence` and `usage` follow the definitions
  on this page instead.
- A JSON state is written as indented JSON. A `null` state leaves the images,
  or the question alone, as the prompt.
- `images` are placed before the state and shrunk to at most 1024 x 1024
  pixels, as the reference does.
- A `choice` whose option names are all single letters is labeled with those
  letters.
- When each request carries one question over a state that changes from
  request to request, `--disable-prefix-caching` lowers latency.

### KnowLine

The KnowLine decision models are a Qwen3.5-MoE checkpoint post-trained to
answer decisions as the first token after a literal `Answer:`. The
architecture is the stock model's, so the checkpoint opts in with a
`knowline_config` field in its model config; until the checkpoint ships
one, the same field can be passed on the command line:

```bash
vllm serve PelaAI/KnowLine-35B-A3B-Gen1.5 \
    --hf-overrides '{"knowline_config": {}}' \
    --compilation-config '{
        "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64,
            72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176,
            184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 272, 288, 304,
            320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512,
            528, 544, 560, 576, 592, 608, 624, 640, 656, 672, 688, 704, 720,
            736, 752, 768, 784, 800, 816, 832, 848, 864, 880, 896, 912, 928,
            944, 960, 976, 992, 1008, 1024, 1088, 2176, 3264, 4352, 5440,
            6528, 7616, 8704, 9792, 10880, 11968, 13056, 14144, 15232, 16384],
        "max_cudagraph_capture_size": 16384}'
```

- The server writes the prompt the checkpoint was trained on, so picks and
  probabilities match the reference server shipped with the checkpoint: the
  state as chat turns (a list of `{role, content}` messages), then one user
  turn with the instruction, the question and its lettered options (`A.`,
  `B.`, ...), the assistant opener with thinking off, and `Answer:`. A JSON
  state is flattened to indented text, not JSON. A chat template that
  rejects the conversation folds the ask into the last user turn, or
  renders the whole chat as one user message.
- Labels run `A` to `Z`, then `AA` on, each one token right after
  `Answer:`, so a choice can have up to 255 options and a score up to 255
  levels. Every label must be one token, or serving is disabled.
- This is a hybrid model (GDN linear attention). With prefix caching, vLLM
  prefills any prompt over one 1088-token mamba block as a single chunk,
  and that chunk must replay a piecewise CUDA graph: the default capture
  sizes stop at 1024 tokens, and larger chunks run the compiled pieces
  kernel by kernel, several times slower. The `--compilation-config` above
  appends the mamba block's multiples to the default capture sizes, up to
  the 16,384-token chunked-prefill budget; size the list to your own
  budget.
- Up to 128 options, the read asks for the labels' logprobs directly, so
  `diagnostics.label_mass` is the labels' share over the whole vocabulary
  and `argmax_is_label` reports the model's actual first token. Above 128
  the read masks the vocabulary to the labels, which renormalizes over
  them: `label_mass` is then near 1 and `argmax_is_label` always true, and
  the questions need `--logprobs-mode processed_logprobs` plus
  `--max-logprobs 255` (or higher).
- A request reads the state and the question a second time when the
  single-pass request fits the marker's `twice_max_tokens` bound, 3000 by
  default, as the checkpoint was trained.
- Answers keep the trained shapes: a `noul` answers the probability of true
  alone, with no probabilities or confidence; a `choice`'s and `score`'s
  `confidence` is a certainty proxy, one minus the normalized entropy of
  the labels; a `score`'s `legend` echoes the criteria as they came in.
- The model reads text: `images`, `audio` and `videos` return 400. Top-level
  `instructions` return 400, since the prompt is fixed. Other
  `chat_template_kwargs` reach the template, with thinking off.
