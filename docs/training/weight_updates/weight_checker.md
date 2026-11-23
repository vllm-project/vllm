# Weight Checker

The Weight Checker verifies that an RL weight update landed on every rank. It
hashes each model parameter with SHA-256, so a caller can save the original
digests, reset the weights, transfer them back, and confirm they match.

To also check generated outputs before and after an update, see
[Verifying a New Model Without a Trainer](README.md#verifying-a-new-model-without-a-trainer).

Enable it with `VLLM_SERVER_DEV_MODE=1`. All operations use
`POST /weight_checker`:

| Request | Response |
| --- | --- |
| `{"action": "checksum"}` | `{"checksums": {key: sha256_hex}}` |
| `{"action": "reset"}` | `{"status": "reset"}` |
| `{"action": "compare", "baseline": {key: sha256_hex}}` | `{"match": bool, "mismatches": [key]}` |

All three actions accept `"model": "target"`, `"draft"`, or `"both"`. Omitting
`model` selects the target. Use the same selection for checksum and compare.

For a single model, keys have the form `dp{dp}:pp{pp}:pcp{pcp}:tp{tp}:{tensor_name}`.
Selecting `both` adds a `target:` or `draft:` prefix to each key. With pipeline
parallelism, draft keys come from the last stage. The caller holds the baseline.

Invalid requests return HTTP 400. A draft selection without resident draft
weight parameters returns HTTP 409. Reset also returns HTTP 409 during another
reset or an active or incomplete weight update, or if a single-model reset
would overwrite storage shared with the other model. Reset checks all
participating workers before zeroing any weights. Workers reject weight updates
while a reset holds their reservation. Wait for reset to finish before starting
a weight update. Use `both` for shared weights and restore both models before
resuming.

## RL weight-update check

```bash
URL=http://localhost:8000
MODEL=both  # Use target or draft to verify a single model.

# 1. Stop serving, then save the original digests.
curl -X POST "$URL/pause?mode=abort"
curl --fail-with-body -X POST "$URL/weight_checker" -H 'Content-Type: application/json' \
  -d "{\"action\":\"checksum\",\"model\":\"$MODEL\"}" > baseline.json

# 2. Zero the weights.
curl --fail-with-body -X POST "$URL/weight_checker" -H 'Content-Type: application/json' \
  -d "{\"action\":\"reset\",\"model\":\"$MODEL\"}"

# 3. Compare again to confirm the reset changed weights before restoring them.
jq --arg model "$MODEL" '{action:"compare", model:$model, baseline:.checksums}' \
  baseline.json | curl --fail-with-body -X POST "$URL/weight_checker" \
  -H 'Content-Type: application/json' -d @-

# 4. Transfer the original target weights with /start_weight_update,
#    /update_weights, /finish_weight_update. For the draft, use
#    /start_draft_weight_update, then the same update and finish routes.
#    For both, complete the target transfer before starting the draft transfer.

# 5. Compare against the baseline. Resume only after match is true.
jq --arg model "$MODEL" '{action:"compare", model:$model, baseline:.checksums}' \
  baseline.json | curl --fail-with-body -X POST "$URL/weight_checker" \
  -H 'Content-Type: application/json' -d @-

curl -X POST "$URL/resume"
```

## Limitations

- Stay paused from `checksum` to `compare`: the weights are invalid between
  `reset` and the transfer, and EPLB moves experts while serving.
- A request covers the engines managed by the API server it reaches. With
  several API servers (for example `--data-parallel-external-lb`), send it to
  each one.
- Buffers and LoRA adapters are excluded from checksum and reset.
- Restoring a draft requires an existing weight transfer backend and topology
  that support draft weight updates. With pipeline parallelism, `draft` and
  `both` support checksum and compare. Reset returns HTTP 409 for these
  selections because the draft weight transfer path requires PP size 1.
- CPU backends that repack linear weights hide them from the checker.
- compressed-tensors online transforms are not supported: their Hadamard
  weights are normalized once per storage, so restored copies differ.
- MoE weights padded beyond the checkpoint shape (DeepEP hidden size, MXFP4)
  keep uninitialized padding, so `compare` reports them as mismatches.
- The engine must be awake: sleep levels 1 and 2 release the weight memory.
- `--offload-backend prefetch` is not supported: parameters point at staging
  buffers, not the offloaded weights.
- Hashing copies every weight to CPU, so keep it off latency-sensitive paths.
