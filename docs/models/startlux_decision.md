# StartLux decision pooling

The dense and MoE models reuse the Qwen3.5 backbone and project the last prompt
position onto the checkpoint's A–Z answer-token rows in FP32. Classification
returns raw logits; the native Decisions strategy restricts them to the current
question's legal candidates before applying its type-specific temperature.
No answer tokens are generated.

## Text, single-pass interface

Keep `decision_config.json`, the tokenizer and chat template with the checkpoint.
The strategy verifies the letter token IDs and thinking-off assistant prefix.
It supports v1 and v1.1 Instant text reads, not AutoThink generation, images,
wide-choice aggregation or an additional none-of-the-above pass.

```bash
vllm serve '<CHECKPOINT>' \
  --served-model-name startlux-decision \
  --runner pooling \
  --hf-overrides '{"architectures":["StartLuxDecisionForSequenceClassification"],"num_labels":26}' \
  --pooler-config '{"task":"classify","use_activation":false}' \
  --dtype bfloat16 --max-model-len 8192 --max-num-batched-tokens 8192 \
  --no-enable-prefix-caching --no-enable-chunked-prefill --enforce-eager
```

Use `StartLuxDecisionMoeForSequenceClassification` for a MoE checkpoint.
Pipeline parallelism must be one. For Instant, preserve the checkpoint's
`jev_prompt_attention` setting: a bidirectional prompt must be processed in
full, so chunked prefill and prefix caching must remain disabled.

```bash
curl http://localhost:8000/v1/decisions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "startlux-decision",
    "input": "The same order was charged twice.",
    "questions": [
      {"type":"choice","instructions":"Which team handles this?",
       "choices":[{"value":"billing"},{"value":"shipping"}]},
      {"type":"predicate","instructions":"Is there an immediate safety emergency?"},
      {"type":"score","instructions":"How severe is the billing issue?",
       "levels":[{"label":"low"},{"label":"high"}]}
    ]
  }'
```

The current upstream route is `/v1/decisions`, not the historical independent
`/v1/systemone` example. Choice supports 2–26 distinct single-line values;
score supports 2–10 ordered levels. A score level with a distinct description
is rendered as `label: description`. Predicate prompts use the trained yes/no
order and map the probabilities back to the public API's true probability.
Each question is a separate classify read. Input over the configured maximum
length is rejected rather than truncated.

The lower-level `/classify` endpoint still returns 26 raw logits. Its field
name `probs` does not mean those logits are normalized. Clients of that endpoint
must supply the exact prompt token IDs and implement candidate restriction and
calibration themselves.

## Validation scope

CPU unit tests cover head arithmetic, candidate bounds, typed prompt conversion,
calibration, and native handler output using an engine fixture. They do not load
trained weights or demonstrate accelerator execution. Fresh main checkpoints,
multimodal processing and real graph replay require separate runtime acceptance;
historical release results do not validate this source combination.
