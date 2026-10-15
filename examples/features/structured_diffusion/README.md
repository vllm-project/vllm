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

`single_pass_reads` is disabled by default. On the CPU backend, enable it with
`--diffusion-config '{"canvas_length": 64, "single_pass_reads": true}'`.
For requests with `diffusion_read_only` set to `true` and `diffusion_max_steps`
set to 1, the scheduler can attach the canvas to the step that finishes the
prompt. This saves one forward pass when eligible. Chunked prompts can still
require several forward passes.

If the remaining prompt tokens and canvas do not fit the step's token budget,
or the full prompt plus canvas reaches the maximum model length, the canvas
runs in a separate step. An eligible read waits until the free KV cache holds
both. The option cannot be combined with pipeline parallelism, KV connectors,
or KV offloading.

Fusion can change emitted tokens and probabilities compared with running the
prompt and canvas separately, even with the same seed. Matching the most likely
label does not guarantee matching probabilities. Before enabling this option,
validate the outputs on your workload, including any decisions based on
probability thresholds.

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
