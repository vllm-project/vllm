# Structured reads on DiffusionGemma

A discrete diffusion model denoises a whole canvas per forward pass. If the
canvas is seeded with the answer's fixed text and only the answer slots are
left as noise, one denoise step gives a distribution over each slot. These
`extra_args` fields (`vllm_xargs` on the OpenAI server) expose that:

| field | type | meaning |
| --- | --- | --- |
| `diffusion_seed_canvas` | `list[int]`, exactly `canvas_length` ids | replaces the random initial canvas after prefill |
| `diffusion_pinned` | `list[int]` of canvas positions | held at their seed value on every denoise step, so a read past one step keeps its template |
| `diffusion_max_steps` | `int` | denoise steps before the canvas is emitted |
| `diffusion_read_only` | `bool` | emit the argmax canvas as soon as the cap is reached, end the request there, and return temperature-1 logprobs at every position |
| `diffusion_constrained` | `bool` | run the unembedding, sampler and self-conditioning over the request's `logprob_token_ids` only. Logprobs are normalized over that set. A step uses this only when every read in it has the same set |

`structured_server.py` turns a question schema into those fields, with
`diffusion_constrained` on for every read (`--no-constrained` turns it off). It serves
`/v1/chat/completions`: the system message is the schema, the user message
is the state JSON, and the reply content is one distribution per question
with a standard error over a few noise draws.

```bash
vllm serve google/diffusiongemma-26B-A4B-it \
    --diffusion-config '{"canvas_length": 64}' --max-logprobs 32 --enable-prefix-caching
python examples/features/structured_diffusion/structured_server.py \
    --upstream http://127.0.0.1:8000 --tokenizer google/diffusiongemma-26B-A4B-it --canvas 64
curl -s localhost:8011/v1/chat/completions -H 'content-type: application/json' -d '{
  "messages": [
    {"role": "system", "content": "{\"questions\": [{\"id\": \"urgent\", \"type\": \"noul\", \"instructions\": \"Does the customer need a reply within the hour?\"}]}"},
    {"role": "user", "content": "{\"ticket\": \"Everything is down and we have a demo at noon.\"}"}
  ]}'
```

The attention backend is picked as for Gemma 4: FlashAttention 4 on every
layer when available, otherwise Triton. FlashInfer cannot serve this model (a
batch mixes causal prefill with bidirectional denoising), and
`--attention-backend FLASHINFER` is rejected.

Per-request canvas widths may be smaller than the served canvas with either
synchronous or asynchronous scheduling. Omit `diffusion_canvas_length` to use
the served canvas width.

Question types: `noul` (yes/no), `choice` with `options`, `score` with
ordered `levels`. Each label must be a single token in the answer template,
which the server checks with the tokenizer when a request uses the schema.

`POST /v1/systemone` implements the Jev decision API. The body holds
`state`, `questions` (a map of id to `type`, `instructions` and `criteria`)
and `model`. Answers come back in that API's shapes: a `noul` probability,
a `choice` with `probabilities` and `confidence`, or a `score` with a
0-indexed `legend`. The schema options above go in the same body as
extensions.

A question may declare `depends_on` (read in a later stage with those
answers in its prompt), `ask_if` (asked only when a named question's answer
is among the listed ones, otherwise null) and `alone` (a read of its own).

Images attach as `multipart/form-data`, with the JSON in a part named
`request` and each image as a file part, or as an `images` array of data
URLs.

```bash
curl -s localhost:8011/v1/systemone -H 'content-type: application/json' -d '{
  "model": "jev-latest",
  "state": {"ticket": "Everything is down and we have a demo at noon."},
  "questions": {"urgent": {"type": "noul", "instructions": "Does the customer need a reply within the hour?"}}}'
```

`"think": N` in the schema lets the model write up to N tokens in its
thought channel before the read. The thought is an ordinary generation with
the chat template's thinking marker on, and the read then runs with the
thought in its prompt, so the answer slots condition on it. The noise draws
of a decision share one thought. `diagnostics.thought` returns the text, its
length in tokens, whether the model closed the channel itself and the
generation time.

## Nemotron Labs Diffusion

The same `/v1/systemone` gateway can serve Nemotron Labs Diffusion 3B/8B with
`--backend nemotron`; the [OpenAPI schema](systemone-openapi.yaml) lists the
request shapes. This backend uses mask-token answer slots instead of random
noise, one masked forward after prefill, and exact candidate logprobs. It does
not reorder options or questions. Text/JSON state and choice, Noul and Score
answers are supported; thinking, images and multiple samples are not.

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve nvidia/Nemotron-Labs-Diffusion-8B \
    --served-model-name nemotron --diffusion-config '{"canvas_length": 32}' \
    --max-logprobs 128 --enable-prefix-caching
python examples/features/structured_diffusion/structured_server.py --backend nemotron \
    --upstream http://127.0.0.1:8000 --model nemotron \
    --tokenizer nvidia/Nemotron-Labs-Diffusion-8B --canvas 32
```

A direct read sets `diffusion_read_only` and a `diffusion_seed_canvas` of
exactly the served canvas length, with the mask token (`100`) at the answer
positions, and requests the candidates with `logprob_token_ids` (up to 128).
It needs `temperature=0`, default `top_p`/`top_k` and `max_tokens` no larger
than the canvas; `diffusion_pinned`, a narrower `diffusion_canvas_length` and
`diffusion_max_steps` other than 1 are not supported. Other requests are
rejected with HTTP 400 before they reach a worker.

Verify a running gateway with:

```bash
python examples/features/structured_diffusion/nemotron_smoke.py --url http://localhost:8011
```

The smoke check validates answer shapes and finite, normalized probabilities for
all three primitives, a 100-alternative choice and a 12-question request. It is
not an accuracy benchmark.
