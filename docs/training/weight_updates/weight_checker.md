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
parallelism, draft keys come from the last stage. The endpoint keeps no state,
so the caller holds the baseline.

Invalid requests return HTTP 400, and selecting a draft without resident draft
weights returns HTTP 409.

## Shared weights and reset scope

Some speculative drafters borrow the target's embeddings, LM head, or other
parameters. Checksum and compare cover all parameters visible to the selected
model, including borrowed weights. With `both`, a shared weight appears under
both models' keys.

| Reset selection | Weights cleared | Recovery |
| --- | --- | --- |
| `target` | Target-private and target-owned shared weights | Transfer target weights. The draft's aliases see the restored values. |
| `draft` | Draft-private weights | Transfer draft weights. Omit parameters borrowed from the target. |
| `both` | Target-owned and draft-private weights, with shared weights cleared once | Complete the target transfer, then transfer the draft-private weights. |

Draft-only reset returns HTTP 409 if there are no independently resettable draft
weights. Reset also returns HTTP 409 if the worker cannot establish ownership
for overlapping tensor views or if the operation would write through an alias
to a buffer in the other model. Views with provably disjoint bytes can be reset
independently.

Private quantized copies created with `draft_lm_head_quantization` are refreshed
in place after a target reload or weight update, and after a draft weight update.
The draft transfer can omit the source LM head. Drafters with model-specific
derived state that can only be refreshed after a target update support target
reset, while `draft` and `both` resets return HTTP 409.

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
#    Omit borrowed target weights from the draft transfer.
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
- Reset validates local writes before each worker mutates its weights. A
  different worker may already have reset its weights when the server reports a
  conflict. Keep inference paused until recovery and comparison succeed.
- A request covers the engines managed by the API server it reaches. With
  several API servers (for example `--data-parallel-external-lb`), send it to
  each one.
- Buffers and LoRA adapters are excluded from checksum and reset.
- Restoring a draft requires an existing weight transfer backend and topology
  that support draft weight updates. With pipeline parallelism, target reset
  supports target-owned sharing on the last stage. `draft` and `both` support
  checksum and compare and return HTTP 409 for reset, since draft weight transfer
  requires PP size 1.
- CPU backends that repack linear weights hide them from the checker.
- compressed-tensors online transforms are not supported: their Hadamard
  weights are normalized once per storage, so restored copies differ.
- MoE weights padded beyond the checkpoint shape (DeepEP hidden size, MXFP4)
  keep uninitialized padding, so `compare` reports them as mismatches.
- The engine must be awake: sleep levels 1 and 2 release the weight memory.
- `--offload-backend prefetch` is not supported: parameters point at staging
  buffers, not the offloaded weights.
- Hashing copies every weight to CPU, so keep it off latency-sensitive paths.
