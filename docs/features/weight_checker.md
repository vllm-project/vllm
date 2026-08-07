# Weight Checker

vLLM's Weight Checker provides development HTTP APIs for verifying model
weights with per-tensor SHA-256 checksums. In an RLHF update workflow, it saves
the original checksums, resets inference weights before transfer, and verifies
that the transferred weights exactly match the original checkpoint.

Key capabilities:

- **Per-tensor checksums**: Hashes parameters and persistent weight buffers.
- **Distributed coverage**: Identifies every checksum by its data-, pipeline-,
  prefill-context-, tensor-, and expert-parallel ranks.
- **Baseline and compare**: Detects changed, added, or missing tensors.
- **Weight reset**: Randomizes covered tensors before a weight transfer.

## Usage

### Start a server

Enable development endpoints by setting `VLLM_SERVER_DEV_MODE=1`:

```bash
VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-0.6B --port 8000
```

All Weight Checker operations use `POST /weight_checker` with an `action`.

### Calculate checksums and save the baseline

```bash
curl -X POST 'http://localhost:8000/weight_checker' \
  -H 'Content-Type: application/json' \
  -d '{"action":"checksum"}'
```

Example response:

```json
{
  "checksums": {
    "dp0:pp0:pcp0:tp0:ep0:model.embed_tokens.weight": "0123456789abcdef..."
  },
  "engines": [
    {
      "dp0:pp0:pcp0:tp0:ep0:model.embed_tokens.weight": "0123456789abcdef..."
    }
  ],
  "baseline_created": true
}
```

The first `checksum` in a verification cycle stores its result as the compare
baseline. Later `checksum` calls return current values without replacing that
baseline and return `"baseline_created": false`. Keys use the format
`dp{dp_rank}:pp{pp_rank}:pcp{pcp_rank}:tp{tp_rank}:ep{ep_rank}:{tensor_name}`.

### Reset weights before transfer

```bash
curl -X POST 'http://localhost:8000/weight_checker' \
  -H 'Content-Type: application/json' \
  -d '{"action":"reset"}'
```

This randomizes the covered inference tensors without changing the baseline.

### Compare weights with the baseline

```bash
curl -X POST 'http://localhost:8000/weight_checker' \
  -H 'Content-Type: application/json' \
  -d '{"action":"compare"}'
```

A successful restoration returns:

```json
{
  "match": true,
  "mismatches": []
}
```

Changed, added, or missing tensors produce `match: false`, with their fully
qualified rank and tensor names in `mismatches`.

`compare` is one-shot: it clears the baseline after comparison. Calling it
again without a new `checksum` returns HTTP 400.

### RLHF weight-update workflow

The verification sequence is `checksum -> reset -> checksum -> compare`, with
the weight transfer or reload occurring between `reset` and the second
`checksum`:

```bash
# 1. Hash the original weights and save this result as the baseline.
curl -X POST 'http://localhost:8000/weight_checker' \
  -H 'Content-Type: application/json' \
  -d '{"action":"checksum"}'

# 2. Randomize the inference weights before transfer.
curl -X POST 'http://localhost:8000/weight_checker' \
  -H 'Content-Type: application/json' \
  -d '{"action":"reset"}'

# Start, perform, and finish the configured weight transfer.
curl -X POST 'http://localhost:8000/start_weight_update'
# ... transfer the original weights ...
curl -X POST 'http://localhost:8000/finish_weight_update' \
  -H 'Content-Type: application/json' \
  -d '{"weight_version":"step-100"}'

# 3. Hash the transferred weights without replacing the original baseline.
curl -X POST 'http://localhost:8000/weight_checker' \
  -H 'Content-Type: application/json' \
  -d '{"action":"checksum"}'

# 4. Compare the transferred weights with the original baseline.
curl -X POST 'http://localhost:8000/weight_checker' \
  -H 'Content-Type: application/json' \
  -d '{"action":"compare"}'
```

Because the original weights are transferred back after `reset`, the expected
result is `match: true` with an empty `mismatches` list. Under data parallelism,
the operation covers every engine returned by the frontend.

## HTTP API summary

| Action | Description | Changes checker state | Changes weights |
| --- | --- | --- | --- |
| `checksum` | Return checksums and save the first result as the baseline | Creates baseline if absent | No |
| `reset` | Replace covered tensors with random values | No | Yes |
| `compare` | Compare current checksums with the baseline | Clears baseline | No |

Invalid or missing actions return HTTP 400. Calling `compare` before
`checksum` also returns HTTP 400.

## Limitations

- Checksum calculation copies every covered tensor to CPU and hashes all its
  bytes, so it should not be placed on a latency-sensitive request path.
- The baseline is stored in the API server process and is lost on restart.
- A baseline can be consumed only once. Concurrent clients must serialize a
  complete verification cycle.
- Checksums describe tensor bytes, not semantic model equivalence.
- Weight Checker is available only through development HTTP endpoints.
