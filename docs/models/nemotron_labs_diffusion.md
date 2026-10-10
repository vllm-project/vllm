# Nemotron Labs Diffusion

`NemotronLabsDiffusionModel` checkpoints support text generation with masked
block diffusion. Prompt prefill and completed-block KV refresh use causal
attention; denoising uses bidirectional attention within the current block.

The same implementation supports the **3B and 8B text checkpoints** with
architecture `NemotronLabsDiffusionModel`. Layer counts, hidden dimensions,
attention heads, and RoPE settings are read from the checkpoint configuration;
no size-specific architecture override is needed. Both sizes support masked
diffusion and ordinary autoregressive inference. Linear speculation is deferred
to a separate follow-up.

```bash
vllm serve nvidia/Nemotron-Labs-Diffusion-3B \
    --attention-backend TRITON_ATTN \
    --max-num-seqs 8 \
    --diffusion-config '{"temperature": 0.0, "confidence_threshold": 0.9}'
```

The canvas length defaults to the checkpoint's `block_size`. The default
`confidence_threshold` policy reveals all masked positions whose selected-token
probability is at least 0.9, and at least the most confident position each step.
Revealed positions remain fixed. The final permitted denoising step resolves any
remaining masks before the block is committed.

`DiffusionConfig.max_denoising_steps` limits iterations **per block** and defaults
to the canvas length. `SamplingParams.max_tokens` controls the response length.
Use ordinary request temperatures: `0` is greedy, `0.7` samples at temperature
0.7, and `1` samples at temperature 1. Different temperatures can share a batch.
For example, pass `SamplingParams(temperature=0.7, top_p=0.9)` to `LLM.generate`.
Temperature scaling precedes top-k/top-p filtering and applies to returned
sampling logprobs. Confidence is measured using unscaled logits over the retained
candidates.

`DiffusionConfig.temperature`, when provided, sets the default for requests that
omit temperature; it no longer overrides explicit request temperatures.
`--override-generation-config '{"temperature": 0.7}'` sets the same standard
default and takes precedence over `DiffusionConfig.temperature`. Without either
setting, the checkpoint's ordinary generation defaults apply (temperature 1 if
unspecified). The previous `1` selector for an engine temperature is removed.
The `low_confidence` policy instead reveals a scheduled number of the most
confident positions; `leftmost` reveals that number from left to right.

The model uses vLLM's diffusion support in Model Runner V2. Triton attention is the default;
FlashAttention requires FA4. FlashInfer does not support the mixed
causal/bidirectional attention. This implementation covers
text-only masked diffusion; linear speculation and vision inputs are not included.

Masked diffusion currently requires pipeline parallel size 1.
Unsupported pipeline configurations are rejected during engine configuration.
Returned token and top-k logprobs use the distribution at the step where each
position is revealed, including when requests with different logprob settings
share a batch.

## Autoregressive inference

The same checkpoint also supports ordinary causal, token-by-token generation:

```bash
vllm serve nvidia/Nemotron-Labs-Diffusion-3B \
    --hf-overrides '{"ar_mode": true}'
```

For Python, pass `hf_overrides={"ar_mode": True}` to `LLM`. The architecture
alias `hf_overrides={"architectures": ["NemotronLabsDiffusionForCausalLM"]}`
also selects AR mode, for compatibility with existing callers.

AR mode uses the same backbone and `diffusion_head` weights, with causal
attention and vLLM's standard scheduler, KV cache, and sampler. Sampling
parameters such as temperature, top-p, and top-k are set per request. Do not
pass `diffusion_config` in AR mode; denoising policies and thresholds do not
apply. The default, without either override, remains block diffusion.

## 8B checkpoints

Point the server at the 8B checkpoint directly, using the same decoding options:

```bash
vllm serve /path/to/Nemotron-Labs-Diffusion-8B \
    --max-num-seqs 8 \
    --diffusion-config '{"canvas_length": 32}'
```

Use `--hf-overrides '{"ar_mode": true}'` instead of `--diffusion-config` for AR,
or omit both options for confidence-threshold diffusion. The 8B checkpoint's
own RoPE configuration determines its context limit; the 3B context settings
are not substituted.

The model test suite accepts either size through `NEMOTRON_DLM_MODEL_PATH`:

```bash
NEMOTRON_DLM_MODEL_PATH=/path/to/Nemotron-Labs-Diffusion-8B \
    .venv/bin/python -m pytest --confcutdir=tests/models/language/generation \
    tests/models/language/generation/test_nemotron_dllm.py -v
```

## Structured decision reads

The masked-diffusion backend also supports seeded, read-only canvases. Prompt
prefill is followed by one bidirectional masked forward, without an autoregressive
verification/commit pass. Normal generation remains unchanged.

Reuse the structured diffusion example to expose a TypeSafe-shaped
`POST /v1/systemone` API. This is an optional **separate gateway process**, not
an endpoint built into `vllm serve`:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve nvidia/Nemotron-Labs-Diffusion-8B \
  --served-model-name jev-latest --port 8000 \
  --diffusion-config '{"canvas_length":32}' --max-logprobs 128 \
  --enable-prefix-caching

API_KEY="$DECISION_API_KEY" python examples/features/structured_diffusion/structured_server.py \
  --backend nemotron --tokenizer nvidia/Nemotron-Labs-Diffusion-8B \
  --model jev-latest --canvas 32 --upstream http://localhost:8000 --port 8011
```

`jev-latest` is a local serving alias, not the TypeSafe Jev checkpoint.
The gateway supports choice, Noul and Score questions over string or JSON state:

```bash
curl http://localhost:8011/v1/systemone \
  -H "Authorization: Bearer $DECISION_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"jev-latest","state":"Customer payouts have failed for three days.",
       "questions":{"route":{"type":"choice","instructions":"Which team should handle this?",
       "criteria":{"billing":"Payments and payouts","technical":"Infrastructure outages"}},
       "urgent":{"type":"noul","instructions":"Has the problem lasted multiple days?"}}}'
```

Answers preserve the question IDs. Choice returns `choice`, `probabilities` and
`confidence`; Noul returns `noul` (the probability of yes); Score returns a
zero-indexed probability-weighted `score`, `legend`, `probabilities` and
`confidence`. These are normalized model scores, not calibrated guarantees.

This example accepts at most 64 questions and 26 alternatives per question.
It chunks answer templates that exceed the canvas and keeps input order; there
are no option- or position-ordering heuristics. Every option must resolve to one
token at the same answer position. An incompatible template fails explicitly
rather than approximating missing probabilities. Nemotron reads are text-only,
deterministic (`samples=1`), and do not support the example's thinking or multi-step
extensions. See the [OpenAPI schema](../../examples/features/structured_diffusion/systemone-openapi.yaml).

For direct OpenAI-compatible calls, pass `logprob_token_ids` (up to 128 candidates),
`return_tokens_as_token_ids=true`, `logprobs=true` and these `vllm_xargs`:

```json
{
  "diffusion_read_only": true,
  "diffusion_seed_canvas": [100, 100, 100, 100, 100, 100, 100, 100,
                            100, 100, 100, 100, 100, 100, 100, 100,
                            100, 100, 100, 100, 100, 100, 100, 100,
                            100, 100, 100, 100, 100, 100, 100, 100]
}
```

Use `temperature=0`, default `top_p=1`/`top_k=0`, and `max_tokens` no greater
than the engine canvas length. The seed must contain exactly that many valid
token IDs; `100` is the checkpoint's mask token. Non-mask positions remain fixed.
The sampled tokens only transport the per-position logprobs: decisions should
normalize the requested candidates at each mask position. Read-only scoring and
indexed logprobs use the masked-diffusion backend.
