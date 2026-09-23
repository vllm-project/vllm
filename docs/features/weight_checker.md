# Weight Checker

vLLM's Weight Checker provides development HTTP APIs for verifying model
weights with per-tensor SHA-256 checksums. In an RLHF update workflow, it saves
the original checksums, resets inference weights before transfer, and verifies
that the transferred weights exactly match the original checkpoint.

Key capabilities:

- **Per-tensor checksums**: Hashes parameters and persistent weight buffers.
- **Distributed coverage**: Identifies every checksum by its data-, pipeline-,
  prefill-context-, tensor-, and expert-parallel ranks.
- **Stateless comparison**: `compare` diffs current weights against a caller
  supplied `baseline`, and against any further reports the caller sends, so it
  covers both "did the weights land" and "do the replicas agree".
- **Coverage reporting**: Every comparison returns the rank prefixes it
  covered, so a partial deployment cannot pass unnoticed.
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

`reset` takes effect immediately, so stop serving around it: requests that run
between the `reset` and the weight transfer generate from random weights, and
the prefix cache keeps those blocks. Whether to pause is the caller's decision,
since the endpoint works either way. See
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
  "mismatches": [],
  "ranks": ["dp0:pp0:pcp0:tp0:ep0:"]
}
```

Changed, added, or missing tensors produce `match: false`, with their fully
qualified rank and tensor names in `mismatches`. `compare` without a `baseline`
object returns HTTP 400.

`ranks` lists the rank prefixes the comparison covered. It matters because the
engine only reports the ranks it manages: a `match: true` over fewer ranks than
you expected means part of the deployment was never checked.

### Check that several engines agree

Sending extra reports as `checksums` compares them against the baseline and
against each other. Every report is held to every other one:

```bash
curl -X POST 'http://localhost:8000/weight_checker' \
  -H 'Content-Type: application/json' \
  -d '{"action":"compare",
       "baseline":{"dp0:...":"abc..."},
       "checksums":[{"dp0:...":"abc..."},{"dp0:...":"abc..."}]}'
```

Only the same rank-qualified key is compared. The same tensor name on a
different rank holds a different shard of a TP or EP split, so its digest is
*expected* to differ and cross-rank comparison would report false mismatches.
Entries that not every report carries are listed in `mismatches` too, so a
rank that no report reached cannot pass as consistent.

This covers reports that share a rank space: repeated samples, the same ranks
before and after a restart, or the engines one frontend manages. It does **not**
compare two replicas that sit on different data-parallel ranks.

### Check that a replica holds the same weights

A checksum key carries the data-parallel rank, so two replicas hold the same
weights under *different* keys. Comparing them against each other therefore
reports every tensor as a mismatch, and each one only covers its own ranks.
Checking them means giving them a shared reference instead:

1. Take the baseline on one replica.
2. Call `checksum` on a second replica and confirm it reports the same rank
   prefixes. Different ranks mean the two cannot be checked against each other,
   because no key is shared. That is the case in a deployment whose frontends
   each manage a disjoint slice of the ranks, where a baseline covering all of
   them has to be collected from every frontend first.
3. Call `compare` on that second replica, passing the baseline from step 1.

Repeat step 3 for every replica. `match: true` from all of them means they all
hold the baseline's weights, so they hold the same weights as each other. What
that is worth depends on the baseline: a baseline taken from a healthy replica
or a checkpoint confirms the group agrees, while one taken after a `reset`
would confirm only that the group agrees on random weights.

### Collect a digest per weight update

Calling `checksum` after every update costs an extra full pass over the
weights. A trainer that wants a snapshot of each update can ask for it on the
finish call and get the digests back in the same response:

```bash
curl -X POST 'http://localhost:8000/finish_weight_update' \
  -H 'Content-Type: application/json' \
  -d '{"weight_version":"step-100","checksum":true}'
# {"message":"Weight update finished","checksums":{"dp0:...":"abc..."}}
```

The option is off by default: without `"checksum": true` the request and the
response are unchanged and nothing extra is hashed. The digests are taken after
the transfer engine reports the update complete, so they describe the committed
weights rather than a partially applied chunk.

The snapshot covers the ranks of whichever instance answered, and its keys
carry those ranks. Reports from two replicas therefore describe the same
weights under different keys, so comparing them means checking each one against
a shared reference rather than against each other. See
[Check that a replica holds the same weights](#check-that-a-replica-holds-the-same-weights).

### RLHF weight-update workflow

The verification sequence is
`checksum -> pause -> reset -> transfer -> compare -> resume`, with the weight
transfer or reload occurring while the engine is paused. Keep the checksum
client-side and pass it to `compare`:

```bash
# 1. Hash the original weights and save this result as the baseline.
curl -X POST 'http://localhost:8000/weight_checker' \
  -H 'Content-Type: application/json' \
  -d '{"action":"checksum"}'

# 2. Stop serving before the reset.
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

# 4. Compare the transferred weights with the original baseline.
curl -X POST 'http://localhost:8000/weight_checker' \
  -H 'Content-Type: application/json' \
  -d '{"action":"compare","baseline":{"...":"..."}}'

# 5. Serve again.
curl -X POST 'http://localhost:8000/resume'
```

The pause is the caller's choice: every action works whether or not the engine
is paused, since hashing and rewriting weights do not need the scheduler. It is
placed around the reset because the weights are random until the transfer
restores them, so any request in that window generates from random weights and
the prefix cache keeps those blocks. A pause with `mode=abort` clears that
cache, and [Reset is not atomic](#reset-is-not-atomic) covers the rest.

There is no separate `checksum` between the transfer and the `compare`:
`compare` hashes the current weights itself, so a preceding `checksum` would
only repeat that work and throw the result away. Ask for the digests separately
only if you want to read the new values rather than just confirm they match, or
pass `"checksum": true` on the finish call to get them without the extra pass.

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

The window opens at the `reset` and closes when the weights are restored.
Nothing may serve while it is open, and the caller must not treat a request that
slipped in as meaningful. In a multi-frontend deployment, pause and resume every
frontend that can reach the engine, since `/pause` only reaches the engines the
frontend it lands on manages.

## HTTP API summary

| Action | Description | Changes weights |
| --- | --- | --- |
| `checksum` | Return per-tensor SHA-256 digests | No |
| `reset` | Replace covered tensors with random values | Yes |
| `compare` | Diff current weights against `baseline`, plus any extra `checksums` | No |

Invalid or missing actions return HTTP 400, and so does `compare` without a
`baseline` object. Every action works while the engine is paused, so a caller
can bracket a weight update with `/pause` and `/resume` without ordering the
checks around them.

## Limitations

- The engine must be awake. Sleep level 2 discards the weight storage, so
  digests taken while asleep would describe freed memory and `reset` would
  write to it. The endpoint does not check this, since `/is_paused` reports the
  scheduler pause and not the sleep state.
- Checksum calculation copies every covered tensor to CPU and hashes all its
  bytes, so it should not be placed on a latency-sensitive request path.
- `compare` sends the whole baseline mapping, which is large for big models.
- Checksums describe tensor bytes, not semantic model equivalence.
- Weight Checker is available only through development HTTP endpoints.
