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

`structured_server.py` turns a question schema into those fields. It serves
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

Per-request canvas widths smaller than the served canvas require async
scheduling. The diffusion async scheduler is selected automatically; no
`--scheduler-cls` argument is needed. Synchronous execution supports full-width
canvases only.

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
