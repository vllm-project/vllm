# Weight Checker

vLLM's Weight Checker provides development HTTP APIs for verifying model
weights with per-tensor SHA-256 checksums. In an RLHF update workflow, it saves
the original checksums, resets inference weights before transfer, and verifies
that the transferred weights exactly match the original checkpoint.

Key capabilities:

- **Per-tensor checksums**: Hashes parameters and persistent weight buffers.
- **Distributed coverage**: Identifies every checksum by its data-, pipeline-,
  prefill-context-, tensor-, and expert-parallel ranks.
- **Stateless comparison**: Detects changed, added, or missing tensors; the
  caller supplies the baseline, so the check survives load balancing across API
  server processes.
- **Weight reset**: Randomizes covered tensors before a weight transfer.

## Usage

### Start a server

Enable development endpoints by setting `VLLM_SERVER_DEV_MODE=1`:

```bash
VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-0.6B --port 8000
```

All Weight Checker operations use `POST /weight_checker` with an `action`.

### Calculate checksums

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
  }
}
```

Keys use the format
`dp{dp_rank}:pp{pp_rank}:pcp{pcp_rank}:tp{tp_rank}:ep{ep_rank}:{tensor_name}`.
The caller keeps this mapping as the comparison baseline.

### Reset weights before transfer

```bash
curl -X POST 'http://localhost:8000/weight_checker' \
  -H 'Content-Type: application/json' \
  -d '{"action":"reset"}'
```

This randomizes the covered inference tensors. The endpoint stores no baseline
state, so a reset cannot make a later `compare` use a stale baseline.

`reset` takes effect immediately, so pause the engine first: requests that run
between `reset` and the weight transfer generate from random weights, and the
prefix cache keeps those blocks. See
[Reset is not atomic](#reset-is-not-atomic).

### Compare weights with the baseline

Send the mapping saved from the `checksum` call as `baseline`:

```bash
curl -X POST 'http://localhost:8000/weight_checker' \
  -H 'Content-Type: application/json' \
  -d '{"action":"compare","baseline":{
        "dp0:pp0:pcp0:tp0:ep0:model.embed_tokens.weight":"0123456789abcdef..."
      }}'
```

A successful restoration returns:

```json
{
  "match": true,
  "mismatches": []
}
```

Changed, added, or missing tensors produce `match: false`, with their fully
qualified rank and tensor names in `mismatches`. `compare` without a `baseline`
object returns HTTP 400.

### RLHF weight-update workflow

The verification sequence is
`checksum -> pause -> reset -> transfer -> checksum -> compare -> resume`, with
the weight transfer or reload occurring between `reset` and the second
`checksum`. Keep the first checksum client-side and pass it to `compare`:

```bash
# 1. Hash the original weights and save this result as the baseline.
curl -X POST 'http://localhost:8000/weight_checker' \
  -H 'Content-Type: application/json' \
  -d '{"action":"checksum"}'

# 2. Stop serving before touching the weights.
curl -X POST 'http://localhost:8000/pause?mode=abort'

# 3. Randomize the inference weights before transfer.
curl -X POST 'http://localhost:8000/weight_checker' \
  -H 'Content-Type: application/json' \
  -d '{"action":"reset"}'

# Start, perform, and finish the configured weight transfer.
curl -X POST 'http://localhost:8000/start_weight_update'
# ... transfer the original weights ...
curl -X POST 'http://localhost:8000/finish_weight_update' \
  -H 'Content-Type: application/json' \
  -d '{"weight_version":"step-100"}'

# 4. Hash the transferred weights.
curl -X POST 'http://localhost:8000/weight_checker' \
  -H 'Content-Type: application/json' \
  -d '{"action":"checksum"}'

# 5. Compare the transferred weights with the original baseline.
curl -X POST 'http://localhost:8000/weight_checker' \
  -H 'Content-Type: application/json' \
  -d '{"action":"compare","baseline":{"...":"..."}}'

# 6. Serve again.
curl -X POST 'http://localhost:8000/resume'
```

Because the original weights are transferred back after `reset`, the expected
result is `match: true` with an empty `mismatches` list.

A single request covers every engine that the frontend it reaches manages, and
each engine covers its own workers. In a deployment with several frontends
(for example `--data-parallel-external-lb` with more than one local engine per
frontend), each frontend only reports its own engines, so verifying the whole
group requires sending the action to every API server and merging the results.
Pause and resume each frontend the same way.

### Reset is not atomic

`reset` randomizes the covered tensors in place. Until the weight transfer
restores them, the engine holds weights that do not belong to any checkpoint,
so anything served in that window is meaningless:

- Requests already running may read a mix of original and randomized tensors.
  `/pause?mode=abort` returns or drops them, and they must be discarded rather
  than compared against a pre-reset baseline.
- The prefix cache keys cached blocks by token IDs, not by weight version. A
  block computed from randomized weights stays a cache hit after the weights
  are restored, so it will be returned as if it came from the correct weights.
  `/pause` clears the prefix cache by default, but only for `mode=abort` and
  `mode=wait`; `mode=keep` freezes the cache as well as the queue. If you keep
  the cache, call `/reset_prefix_cache` after the transfer, before `/resume`.
  It answers `{"success": false}` while blocks are still held, so retry until
  it succeeds.

Pause, reset, transfer, verify, and resume must therefore stay serialized: do
not let traffic reach an engine between `reset` and `resume`. In a
multi-frontend deployment, pause and resume every frontend that can reach the
engine, since `/pause` only reaches the engines the frontend it lands on
manages.

## HTTP API summary

| Action | Description | Changes weights |
| --- | --- | --- |
| `checksum` | Return per-tensor SHA-256 digests | No |
| `reset` | Replace covered tensors with random values | Yes |
| `compare` | Diff current weights against the supplied `baseline` | No |

Invalid or missing actions return HTTP 400, and so does `compare` without a
`baseline` object. A paused engine returns HTTP 409 for every action, checked
before the per-action arguments: call `/resume` first, which is what a
weight-update cycle does anyway.

## Limitations

- The engine must be awake and unpaused. The endpoint rejects a paused engine
  with HTTP 409, but sleep state is the caller's responsibility to check, since
  `/is_paused` reports the scheduler pause only. Sleep level 2 discards the
  weight storage, so digests taken while asleep would describe freed memory and
  `reset` would write to it.
- Checksum calculation copies every covered tensor to CPU and hashes all its
  bytes, so it should not be placed on a latency-sensitive request path.
- `compare` sends the whole baseline mapping, which is large for big models.
- Checksums describe tensor bytes, not semantic model equivalence.
- Weight Checker is available only through development HTTP endpoints.
