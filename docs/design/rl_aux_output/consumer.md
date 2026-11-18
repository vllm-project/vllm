# Reading Aux Outputs in the Trainer

This page is the contract between the engine and the trainer for session requests. How the engine names and writes objects is in the [protocol](protocol.md).

## What the rollout keeps

For every request of a session, keep from the final output:

- `prompt_token_ids` (the engine-side prompt) and the generated token ids;
- `aux_output_keys`;
- `aux_version_segments`, to send with the session's next request and to label positions with their weight version;
- the inline logprob of each sampled token, which is the behavior logprob.

With `n > 1`, every choice (`RequestOutput.outputs[i]`, `choices[i]` over HTTP) has its own `aux_output_keys` and `aux_version_segments`: read a choice's rows with its keys, and continue a branch with its segments.

A training sequence of `L` tokens needs the rows of positions `[0, L - 1)`. When one request produced the whole sequence and stopped, its keys cover them; the keys of an aborted request cover `n` positions (rule 3). A sequence that several requests produced (the turns of a multi-turn trajectory trained as one sequence, a fork, a continuation after an abort) takes each position's row as the next section says. Every request's keys stay readable until the session is released.

## Building a trajectory from several responses

A trajectory that several requests produce (turns, forks, a continuation after an abort or a compaction) takes the row of each position `p` from one response:

1. **The request that sampled the next token.** When token `p + 1` of the trajectory was sampled by a request whose inputs `[0, p]` (tokens, multimodal inputs, prompt embeddings, LoRA adapter) are the trajectory's, `p` takes that request's row. A token kept across a compaction was not sampled in this context: it counts as prompt text (rule 2). A request computes every position it samples from, `[P - 1, L - 1)` for `P` prompt tokens and `L` tokens, so its response delivers them, unless it was aborted first (rule 3) or computed them again after a preemption under another version history ([protocol §9](protocol.md#9-limits)), when it lists the new rows instead.
2. **Otherwise, the first delivery of the same prefix.** A position whose next token no request sampled (prompt text, tool output, an edited history) takes the row of the first response that delivered it for the same inputs `[0, p]`, and a later response never replaces it: a later request may have computed the position again under another version history ([protocol §9](protocol.md#9-limits)). Responses whose inputs before `p + 1` differ (other branches) do not count.
3. **Continue an aborted request only from covered tokens.** A response delivers positions `[0, min(n, L - 1))`, where `n` is the number of positions its keys cover. An aborted request can cover fewer positions than its generated tokens need: keep its prompt and its generated tokens up to token `n` (the first `max(P, min(L, n + 1))` tokens), drop the rest, send its `aux_version_segments` with the continuation and resubmit. The continuation hits the KV computed before the abort, if it is still held, and samples again from its last kept token's position.

Keys cover consecutive spans: the `i`-th full object positions `[i × S, (i + 1) × S)`, with `S` the object size (the full-attention KV block size), then the tail from the `end` of the segments; a position's offset is its distance from the start of its span. Read a training sequence with the keys and offsets these rules give. Reading the last response's keys alone is right only when every position takes its row from that response.

## Reading

```python
from vllm.distributed.aux_output_connector import object_store

store = object_store.MooncakeAuxObjectStore.create(mooncake_config_path)
rows = object_store.read_aux_rows(store, keys, "routed_experts")
```

Pass the engine's `AuxOutputConfig.mooncake_config_path`, or `None` to read `MOONCAKE_CONFIG_PATH` as an engine without it does.

`read_aux_rows(store, keys, field, timeout_s=600.0)` reads the objects of `keys` in order and concatenates one field.

Writes are asynchronous, so an object can still be in flight when the response arrives: the reader waits up to `timeout_s` in total, then raises `BlockObjectStoreError`. Keep `timeout_s` no shorter than the engine's `object_put_stall_timeout_s` (both default to 600 s), since a full store can hold writes back that long while the engine keeps running. A timeout does not by itself mean the objects are lost: behind a deep queue a healthy engine can land them later than 600 s. Check the engine first, since an engine stopped on a write stall fails its steps and pauses with the write error; if it is healthy, read again, or after its next completed pause, when every listed object is in the store. The limit is checked between reads: one `get` that blocks, for example after the master died (more than 70 s in our tests), is not interrupted, so give the reading step its own timeout. It raises the same error when an object does not hold the field, for example a field the engine did not enable.

Create the store client in a process that does not start an engine afterwards. A Mooncake client created before the engine breaks CUDA initialization in the engine's workers.

## Fields

All arrays have one row per position the keys cover, `n` rows (`L - 1` for a stopped request); row `p` belongs to position `p`, the position that produces the logits for token `p + 1`.

| Field | Shape and type | Notes |
| --- | --- | --- |
| `routed_experts` | `[n, hidden layers, experts per token]`, `uint8` up to 256 experts, else `uint16` | Logical expert ids; rows of dense layers are 0. |
| `indexer_topk` | `[n, indexer source layers, index_topk]`, `int32` | The selections exactly as captured: order, padding and invalid ids are preserved. DeepSeek V4 indices address the compressed KV cache, not token positions. See [Indexer Replay](../../training/indexer_replay.md). |
| `hidden_states` | `[n, hidden_size]`, model dtype | What the forward pass returns; for most models the input of the LM head. NumPy has no bfloat16, so a bfloat16 model's rows are `int16` bit patterns: `torch.from_numpy(rows).view(torch.bfloat16)`. |
| `topk_token_ids` | `[n, k]`, `int32` | The `k` most likely next tokens, from the model's logits before any sampling parameter. |
| `topk_logprobs` | `[n, k]`, `float32` | Their log-softmax values, in the same order. |

Logits recomputed from `hidden_states` are close to the engine's but not bitwise equal, so the behavior logprob comes from the inline sampled logprob, not from the hidden states or the top-k fields.

## Weight versions

`aux_version_segments` gives the weight version of every position below its closing entry `[end, null]`: segment `[start, v]` covers positions from `start` to the next segment's start. Positions from `end` on are in the tail object; their versions are not in the segments. For generated tokens the response's `weight_versions` spans give the version that sampled each token.

The versions are the ones the object names state. Positions the request reused carry the versions of the computation that produced their KV; positions it computed carry the version that computed them. They differ only in the documented limits ([protocol §9](protocol.md#9-limits)). A position whose row a trajectory takes from an earlier response carries that response's version for it: from its segments below their `end`; for a tail position that sampled a generated token, from that token's `weight_versions` span; the response does not state the version of its other tail positions.

## Guarantees

- Each listed object holds the rows of the computation whose KV the request used for those positions, named with the tokens and weight versions of that computation, within the limits.
- Rows are stored as the engine computed them; nothing is padded, substituted or recomputed by the reader.
- The keys of a stopped request cover `[0, num_tokens - 1)` without gaps; those of an aborted request cover `[0, n)`, where `n` can be below `num_tokens - 1` or one above it ([protocol §3](protocol.md#3-request-and-response)). An aborted request that covers no position lists none.
- After every engine that served a session has completed a pause, every object of the session's stopped, aborted and preempted requests is in the store, except the tails a prefix-cache reset drops (those of preemptions whose terminal event had not reached the workers yet) and the objects listed by the abort outputs of an immediate shutdown.

## What the trainer must do

1. **Missing objects.** If `read_aux_rows` times out, check the engine before dropping the sample: if the engine stopped on a write error, the objects it had not written are lost, and so are those missing after the abort outputs of an immediate shutdown; if it is healthy, read again, or after its next completed pause. When ranks read in a collective step, agree on the outcome across ranks before continuing.
2. **Replay.** Truncate to the training sequence's `L - 1` positions. Hook R3 replay where the model computes routing; some MoE implementations call the routing function directly and bypass a router-level hook.
3. **Behavior logprob.** Use the inline sampled logprob.
4. **Versions.** Positions of different versions are not interchangeable. Use the versions for staleness weighting or filtering.
5. **Release.** Delete a session with `aux_session_pattern` after every request of the session has been read, the training step has committed, and the engines have completed a pause since its last request stopped, for example the pause of the next weight update ([protocol §7](protocol.md#7-pause-and-release)). Never send another request with a released session name.

## Example

`wait_for_pause` returns after the engines' next completed pause and raises if the pause failed, for example because an engine stopped on a write error, or cannot come because the engine shut down. Nothing is dropped silently: a second miss after a completed pause raises. A request that lists no keys (`aux_output_keys` is `None`) has no rows, so `load` returns `None` for it; for an aborted request `load` can return fewer than `len(tokens) - 1` rows, and its generated tokens after them are dropped before it continues. Read only fields the engine enables: a missing field raises the same `BlockObjectStoreError` as a missing object, so this example waits for a pause before its second read raises. `client` in `release` is a `mooncake.store.MooncakeDistributedStore` connected to the object store ([protocol §7](protocol.md#7-pause-and-release)), not the `MooncakeAuxObjectStore` used for reading.

```python
import numpy as np
import torch
from vllm.distributed.aux_output_connector import object_store
from vllm.distributed.aux_output_connector.store import BlockObjectStoreError


def load(store, choice, tokens, wait_for_pause):
    n = len(tokens) - 1
    keys = choice.aux_output_keys
    if keys is None:
        return None
    try:
        r3 = object_store.read_aux_rows(store, keys, "routed_experts")
    except BlockObjectStoreError:
        wait_for_pause()
        r3 = object_store.read_aux_rows(store, keys, "routed_experts", timeout_s=0)
    return torch.from_numpy(r3[:n].astype(np.int64))


def release(client, object_key_prefix, session):
    client.remove_by_regex(
        object_store.aux_session_pattern(object_key_prefix, session), True
    )
```
