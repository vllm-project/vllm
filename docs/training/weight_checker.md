# Weight Checker

The Weight Checker verifies that an RL weight update landed on every rank. It
hashes each model parameter with SHA-256, so a caller can save the original
digests, reset the weights, transfer them back, and confirm they match.

Enable it with `VLLM_SERVER_DEV_MODE=1`. All operations use
`POST /weight_checker`:

| Request | Response |
| --- | --- |
| `{"action": "checksum"}` | `{"checksums": {key: sha256_hex}}` |
| `{"action": "reset"}` | `{"status": "reset"}` |
| `{"action": "compare", "baseline": {key: sha256_hex}}` | `{"match": bool, "mismatches": [key]}` |

Keys have the form `dp{dp}:pp{pp}:pcp{pcp}:tp{tp}:{tensor_name}`, so each
shard is checked separately. All actions cover the target model's parameters,
which is what weight loading writes; `reset` zeroes them. The endpoint keeps no
state, so the caller holds the baseline. Invalid requests return HTTP 400.

## RL weight-update check

```bash
URL=http://localhost:8000

# 1. Stop serving, then save the original digests.
curl -X POST "$URL/pause?mode=abort"
curl -X POST $URL/weight_checker -H 'Content-Type: application/json' \
  -d '{"action":"checksum"}' > baseline.json

# 2. Zero the weights.
curl -X POST $URL/weight_checker -H 'Content-Type: application/json' \
  -d '{"action":"reset"}'

# 3. Transfer the original weights with start_weight_update and update_weights.

# 4. Finish with the baseline; expect "match": true and empty "mismatches".
curl -X POST $URL/finish_weight_update -H 'Content-Type: application/json' \
  -d "{\"baseline\":$(jq .checksums baseline.json)}"

curl -X POST $URL/resume
```

`POST /weight_checker` with `{"action": "compare", "baseline": ...}` runs the
same comparison without finishing an update.

With a trainer-side weight transfer engine, which finishes the update inside
`send_weights()`, bind the baseline to the HTTP client. `finish_weight_update`
then raises `RuntimeError` on a mismatch:

```python
from functools import partial

client = HTTPVLLMWeightSyncClient(URL)
client.finish_weight_update = partial(client.finish_weight_update, baseline=baseline)
trainer = WeightTransferTrainerFactory.trainer_init(init_info, client=client, source=source)
trainer.send_weights()
```

## Limitations

- Stay paused from `checksum` to `compare`: the weights are invalid between
  `reset` and the transfer, and EPLB moves experts while serving.
- A request covers the engines managed by the API server it reaches. With
  several API servers (for example `--data-parallel-external-lb`), send it to
  each one.
- Buffers, draft models and LoRA adapters are not checked.
- CPU backends that repack linear weights hide them from the checker.
- compressed-tensors online transforms are not supported: their Hadamard
  weights are normalized once per storage, so restored copies differ.
- MoE weights padded beyond the checkpoint shape (DeepEP hidden size, MXFP4)
  keep uninitialized padding, so `compare` reports them as mismatches.
- The engine must be awake: sleep levels 1 and 2 release the weight memory.
- `--offload-backend prefetch` is not supported: parameters point at staging
  buffers, not the offloaded weights.
- Hashing copies every weight to CPU, so keep it off latency-sensitive paths.
