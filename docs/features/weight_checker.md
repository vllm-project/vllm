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

Keys have the form `dp{dp}:pp{pp}:pcp{pcp}:tp{tp}:ep{ep}:{tensor_name}`, so each
shard is checked separately. All actions cover the target model's parameters,
which is what weight loading writes; `reset` zeroes them. The endpoint keeps no
state, so the caller holds the baseline. Invalid requests return HTTP 400.

## RL weight-update check

```bash
URL=http://localhost:8000

# 1. Save the original digests.
curl -X POST $URL/weight_checker -H 'Content-Type: application/json' \
  -d '{"action":"checksum"}' > baseline.json

# 2. Stop serving, then zero the weights.
curl -X POST "$URL/pause?mode=abort"
curl -X POST $URL/weight_checker -H 'Content-Type: application/json' \
  -d '{"action":"reset"}'

# 3. Transfer the original weights with start/update/finish_weight_update.

# 4. Compare against the baseline; expect {"match": true, "mismatches": []}.
curl -X POST $URL/weight_checker -H 'Content-Type: application/json' \
  -d "{\"action\":\"compare\",\"baseline\":$(jq .checksums baseline.json)}"

curl -X POST $URL/resume
```

## Limitations

- Between `reset` and the transfer the weights are invalid, so pause before
  `reset` and resume only after the transfer.
- A request covers the engines managed by the API server it reaches. With
  several API servers (for example `--data-parallel-external-lb`), send it to
  each one.
- Buffers, draft models and LoRA adapters are not checked.
- CPU backends that repack linear weights hide them from the checker.
- The engine must be awake: sleep level 2 discards the weight storage.
- `--offload-backend prefetch` is not supported: parameters point at staging
  buffers, not the offloaded weights.
- Hashing copies every weight to CPU, so keep it off latency-sensitive paths.
