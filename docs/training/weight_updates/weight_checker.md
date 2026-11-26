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
| `{"action": "checksum", "model": "both"}` | `{"checksums": {"target:...": sha256_hex, "draft:...": sha256_hex}}` |
| `{"action": "reset"}` | `{"status": "reset"}` |
| `{"action": "compare", "baseline": {key: sha256_hex}}` | `{"match": bool, "mismatches": [key]}` |

All three actions accept `"model": "target"`, `"draft"`, or `"both"`. Omitting
`model` selects the target. Use the same selection for checksum and compare.

Target-only keys have the form `dp{dp}:pp{pp}:pcp{pcp}:tp{tp}:{tensor_name}`.
Draft keys always have a `draft:` prefix. Selecting `both` adds a `target:`
prefix to the target keys. With pipeline parallelism, the draft's `pp` rank
identifies the last target stage, where the draft resides. Save the
`checksums` mapping from the response and pass it as `baseline` when comparing.

A completed comparison returns HTTP 200 even when `match` is `false`.
`mismatches` lists keys whose digest changed and keys missing from either side.
An explicit `"model": null` or a compare request without a baseline returns
HTTP 400. Selecting `draft` or `both` requires a resident draft with at least
one nonempty weight tensor. Otherwise, the request returns HTTP 409.

## Shared weights and reset scope

Some speculative drafters borrow the target's embeddings, LM head, or other
parameters. Checksum and compare cover all parameters visible to the selected
model, including borrowed weights. With `both`, a shared weight appears under
both models' keys.

Reset is a destructive debugging operation. It clears the selected model's
weights in place, and aliases in the other model reflect the same writes.
Restore all affected weights before resuming inference. A worker with an
active weight update rejects reset with HTTP 409.

| Reset selection | Weights cleared | Recovery |
| --- | --- | --- |
| `target` | All visible target weights, including shared weights | Transfer target weights. The draft's aliases see the restored values. |
| `draft` | All visible draft weights, including shared weights | Include shared parameters in the draft transfer to verify that the draft update restores them. |
| `both` | All visible target and draft weights. A shared tensor can be cleared twice with the same result. | Complete the target transfer, then transfer the draft-private weights. |

Draft reset also clears the target's view of shared parameters. If the draft
transfer omits them, compare reports any omitted weight whose baseline differs
from zero. Restore omitted shared parameters through a target weight update
before resuming inference.

Resetting `draft` or `both` requires a weight transfer backend that supports
draft updates. A worker with an absent or unsupported backend rejects the
reset before clearing either model, and the request returns HTTP 409.

Private quantized copies created with `draft_lm_head_quantization` are refreshed
in place after a target reload or weight update, and after a draft weight update.
The draft transfer can omit the source LM head. Drafters with model-specific
derived state that can only be refreshed after a target update support target
reset, while `draft` and `both` resets return HTTP 409.

## RL weight-update check

```bash
URL=http://localhost:8000
MODEL=target  # For a resident drafter with PP=1, choose draft or both to reset it.

(
  set -eo pipefail

  # 1. Stop serving, then save the original digests.
  curl --fail-with-body -X POST "$URL/pause?mode=abort"
  curl --fail-with-body -X POST "$URL/weight_checker" -H 'Content-Type: application/json' \
    -d "{\"action\":\"checksum\",\"model\":\"$MODEL\"}" > baseline.json

  # 2. Zero the weights.
  curl --fail-with-body -X POST "$URL/weight_checker" -H 'Content-Type: application/json' \
    -d "{\"action\":\"reset\",\"model\":\"$MODEL\"}"

  # 3. Inspect which weights changed. Weights that were already zero can still match.
  jq --arg model "$MODEL" '{action:"compare", model:$model, baseline:.checksums}' \
    baseline.json | curl --fail-with-body -X POST "$URL/weight_checker" \
    -H 'Content-Type: application/json' -d @-
)
```

While inference is paused, use your trainer to restore the selected weights.
Start a target transfer with `/start_weight_update` or a draft transfer with
`/start_draft_weight_update`. Both use `/update_weights` and
`/finish_weight_update`. For `both`, finish the target transfer before starting
the draft transfer.

After recovery, run the following in the same shell to resume only on a match.

```bash
(
  set -eo pipefail
  jq --arg model "$MODEL" '{action:"compare", model:$model, baseline:.checksums}' \
    baseline.json | curl --fail-with-body -X POST "$URL/weight_checker" \
    -H 'Content-Type: application/json' -d @- | jq -e '.match == true'
  curl --fail-with-body -X POST "$URL/resume"
)
```

For a target-only update, save the target baseline while paused and before
resetting or transferring weights.

```bash
curl --fail-with-body -X POST "$URL/weight_checker" -H 'Content-Type: application/json' \
  -d '{"action":"checksum","model":"target"}' > target-baseline.json
```

After `/start_weight_update` and `/update_weights`, finish the transfer and
check the returned `match` field.

```bash
jq '{baseline: .checksums}' target-baseline.json | \
  curl --fail-with-body -X POST "$URL/finish_weight_update" \
    -H 'Content-Type: application/json' -d @- | jq -e '.match == true'
```

For a draft or both-model baseline, use the matching `model` selection in a
separate `/weight_checker` compare after the required updates have finished.

With a trainer-side weight transfer engine, `send_weights()` calls
`finish_weight_update` internally. Bind the saved target digests to the HTTP
client so a mismatch raises `RuntimeError` in the trainer. Use the trainer's
existing `init_info` and weight `source`.

```python
import json
from functools import partial
from pathlib import Path

from vllm.distributed.weight_transfer import (
    HTTPVLLMWeightSyncClient,
    WeightTransferTrainerFactory,
)

baseline = json.loads(Path("target-baseline.json").read_text())["checksums"]
client = HTTPVLLMWeightSyncClient("http://localhost:8000")
client.finish_weight_update = partial(client.finish_weight_update, baseline=baseline)
trainer = WeightTransferTrainerFactory.trainer_init(
    init_info, client=client, source=source
)
trainer.send_weights()
```

## Limitations

- Stay paused from `checksum` to `compare`: the weights are invalid between
  `reset` and the transfer, and EPLB moves experts while serving.
- Each worker validates the selected models before clearing its weights. A
  different worker may already have reset its weights when the server reports a
  conflict. Keep inference paused until recovery and comparison succeed.
- A request covers the engines managed by the API server it reaches. With
  several API servers (for example `--data-parallel-external-lb`), send it to
  each one.
- Checksum covers model parameters and can miss changes to ordinary buffers or
  LoRA adapters. Reset writes through aliases of the parameters it clears, so
  verify any affected buffer state separately.
- With pipeline parallel size greater than 1, `draft` and `both` support
  checksum and compare. Resetting either selection returns HTTP 409 because
  draft weight transfer requires PP size 1. Target reset still covers shared
  weights on the last stage.
- CPU backends that repack linear weights hide them from the checker.
- compressed-tensors online transforms are not supported: their Hadamard
  weights are normalized once per storage, so restored copies differ.
- MoE weights padded beyond the checkpoint shape (DeepEP hidden size, MXFP4)
  keep uninitialized padding, so `compare` reports them as mismatches.
- The engine must be awake: sleep levels 1 and 2 release the weight memory.
- `--offload-backend prefetch` is not supported: parameters point at staging
  buffers, not the offloaded weights.
- Hashing copies every weight to CPU, so keep it off latency-sensitive paths.
