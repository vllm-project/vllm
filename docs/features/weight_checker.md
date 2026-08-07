# Weight Checker

vLLM's Weight Checker provides development HTTP APIs for calculating SHA-256
checksums of model weights. In an RLHF weight-update workflow, it can snapshot
the current weights, reset them before transfer, and compare the transferred
weights with the pre-update snapshot. This helps verify that the original
weights were transferred correctly to every inference engine after the reset.

Key capabilities:

- **Per-tensor checksums**: Computes SHA-256 checksums for model parameters and
  weight-bearing buffers.
- **Distributed coverage**: Includes tensor-parallel shards and returns results
  from every data-parallel engine.
- **Snapshot and compare**: Detects changed, added, or missing tensors after a
  weight update.
- **Weight reset**: Overwrites weight tensors with random values before a
  weight transfer.


## Usage

### Start a server

Enable development endpoints by setting `VLLM_SERVER_DEV_MODE=1` when starting
the vLLM server.

```bash
VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-0.6B \
  --port 8000
```

All Weight Checker operations use `POST /weight_checker` with a JSON request
body containing an `action` field.

### Get current checksums

Use `checksum` to calculate the current checksums without creating or changing
a stored snapshot.

```bash
curl -X POST 'http://localhost:8000/weight_checker' \
  -H 'Content-Type: application/json' \
  -d '{"action":"checksum"}'
```

Example response:

```json
{
  "checksums": {
    "dp0:0:model.embed_tokens.weight": "0123456789abcdef..."
  },
  "engines": [
    {
      "0:model.embed_tokens.weight": "0123456789abcdef..."
    }
  ]
}
```

The `checksums` object contains the combined result. Its keys have the form
`dp{data_parallel_index}:{tensor_parallel_rank}:{tensor_name}`. The `engines`
array contains the same results grouped by data-parallel engine.

### Take a snapshot

Use `snapshot` before resetting the current weights and starting a weight
transfer:

```bash
curl -X POST 'http://localhost:8000/weight_checker' \
  -H 'Content-Type: application/json' \
  -d '{"action":"snapshot"}'
```

Example response:

```json
{
  "status": "snapshotted",
  "n_tensors": 291,
  "engines": [291]
}
```

Taking another snapshot replaces the previous snapshot.

### Reset weights before transfer

After taking a snapshot, use `reset` to overwrite the current inference weights
before starting the weight transfer:

```bash
curl -X POST 'http://localhost:8000/weight_checker' \
  -H 'Content-Type: application/json' \
  -d '{"action":"reset"}'
```

Example response:

```json
{
  "status": "reset"
}
```

### Compare weights with a snapshot

Use `compare` after the weight transfer has finished:

```bash
curl -X POST 'http://localhost:8000/weight_checker' \
  -H 'Content-Type: application/json' \
  -d '{"action":"compare"}'
```

If the weights have not changed, the response is:

```json
{
  "match": true,
  "mismatches": []
}
```

If tensors changed, were added, or are missing, `match` is `false` and
`mismatches` identifies them:

```json
{
  "match": false,
  "mismatches": [
    "dp0:0:model.layers.0.self_attn.q_proj.weight"
  ]
}
```

`compare` is a one-shot operation: it clears the stored snapshot after the
comparison. A second `compare` without a new `snapshot` returns HTTP 400.

### RLHF weight-update workflow

A typical server-side weight verification sequence is:

```bash
# 1. Calculate and return the checksums of the current inference weights.
curl -X POST 'http://localhost:8000/weight_checker' \
  -H 'Content-Type: application/json' \
  -d '{"action":"checksum"}'

# 2. Save a checksum snapshot of the current inference weights.
curl -X POST 'http://localhost:8000/weight_checker' \
  -H 'Content-Type: application/json' \
  -d '{"action":"snapshot"}'

# 3. Overwrite the current inference weights with random values.
curl -X POST 'http://localhost:8000/weight_checker' \
  -H 'Content-Type: application/json' \
  -d '{"action":"reset"}'

# 4. Start and perform the configured weight transfer.
curl -X POST 'http://localhost:8000/start_weight_update'

# ... transfer weights ...

# 5. Finish the weight transfer.
curl -X POST 'http://localhost:8000/finish_weight_update' \
  -H 'Content-Type: application/json' \
  -d '{"weight_version":"step-100"}'

# 6. Get the checksums of the transferred weights without changing the snapshot.
curl -X POST 'http://localhost:8000/weight_checker' \
  -H 'Content-Type: application/json' \
  -d '{"action":"checksum"}'

# 7. Compare the transferred weights with the pre-update snapshot.
curl -X POST 'http://localhost:8000/weight_checker' \
  -H 'Content-Type: application/json' \
  -d '{"action":"compare"}'
```

The first `checksum` action calculates and returns the hashes of the original
inference weights. The second `checksum` action returns the hashes after the
weight transfer. Neither call creates, changes, or consumes the stored
snapshot. Because this workflow transfers the original weights back after
`reset`, the expected comparison result is `match: true` with an empty
`mismatches` list.

When using data parallelism, the comparison covers every engine returned by
the frontend. A missing engine, a missing tensor, a new tensor, or a changed
checksum is reported as a mismatch.

Weight Checker verifies that the transferred weights match the pre-update
snapshot byte for byte. A `match: true` result confirms that the reset weights
were replaced with the original snapshotted weights on every covered engine.

## HTTP API summary

| Action | Description | Changes checker state | Changes weights |
| --- | --- | --- | --- |
| `checksum` | Return current per-tensor SHA-256 checksums | No | No |
| `snapshot` | Store current checksums for a later comparison | Replaces snapshot | No |
| `compare` | Compare current checksums with the snapshot | Clears snapshot | No |
| `reset` | Replace covered tensors with random values | No | Yes |

Invalid or missing actions return HTTP 400. Calling `compare` before taking a
snapshot also returns HTTP 400.

## Limitations

- Checksum calculation copies each covered tensor to CPU and hashes all of its
  bytes. This can be expensive for large models and should not be placed on a
  latency-sensitive request path.
- The snapshot is stored in the API server process and is lost when the server
  restarts.
- A snapshot can be consumed only once. Concurrent clients must coordinate
  snapshot and compare operations externally.
- Checksums describe tensor bytes, not semantic model equivalence.
- Weight Checker is currently available through development HTTP endpoints; it
  does not provide a public offline `LLM` Python API.
