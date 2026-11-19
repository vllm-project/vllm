# Reading Aux Outputs in the Trainer

This page is the contract between the engine and the trainer for session requests. How the engine names and writes objects is in the [protocol](protocol.md).

## What the rollout keeps

For every request of a session, keep from the final output:

- `prompt_token_ids` (the engine-side prompt) and the generated token ids;
- `aux_output_keys`;
- `aux_version_segments`, to send with the session's next request and to label positions with their weight version;
- the inline logprob of each sampled token, which is the behavior logprob.

With `n > 1`, every choice (`RequestOutput.outputs[i]`, `choices[i]` over HTTP) has its own `aux_output_keys` and `aux_version_segments`: read a choice's rows with its keys, and continue a branch with its segments.

A training sequence of `L` tokens is read with the keys of the request whose tokens it is: they cover positions `[0, L - 1)`. For a multi-turn trajectory trained as one sequence that is the last turn. A sequence that is a shorter prefix (one turn, one segment of a compacted rollout) is read with its own request's keys. Every request's keys stay readable until the session is released.

## Reading

```python
from vllm.distributed.aux_output_connector import object_store

store = object_store.MooncakeAuxObjectStore.create(mooncake_config_path)
rows = object_store.read_aux_rows(store, keys, "routed_experts")
```

Pass the engine's `AuxOutputConfig.mooncake_config_path`, or `None` to read `MOONCAKE_CONFIG_PATH` as an engine without it does.

`read_aux_rows(store, keys, field, timeout_s=600.0)` reads the objects of `keys` in order and concatenates one field.

Writes are asynchronous, so an object can still be in flight when the response arrives: the reader waits up to `timeout_s` in total, then raises `BlockObjectStoreError`. Keep `timeout_s` no shorter than the engine's `object_put_stall_timeout_s` (both default to 600 s), since a full or unreachable store can hold writes back that long while the engine keeps running. A timeout does not by itself mean the objects are lost: behind a deep queue the engine can land them later than 600 s. Read again after the engine's next completed pause: every listed object is then in the store or lost, because the engine dropped it at the stall deadline or a store process lost it. The limit is checked between reads: one `get` that blocks, for example after the master died (more than 70 s in our tests), is not interrupted, so give the reading step its own timeout. It raises the same error when an object does not hold the field, for example a field the engine did not enable.

Create the store client in a process that does not start an engine afterwards. A Mooncake client created before the engine breaks CUDA initialization in the engine's workers.

## Fields

All arrays have one row per position; row `p` belongs to position `p`, the position that produces the logits for token `p + 1`.

| Field | Shape and type | Notes |
| --- | --- | --- |
| `routed_experts` | `[L - 1, hidden layers, experts per token]`, `uint8` up to 256 experts, else `uint16` | Logical expert ids; rows of dense layers are 0. |
| `indexer_topk` | `[L - 1, indexer source layers, index_topk]`, `int32` | The selections exactly as captured: order, padding and invalid ids are preserved. DeepSeek V4 indices address the compressed KV cache, not token positions. See [Indexer Replay](../../training/indexer_replay.md). |
| `hidden_states` | `[L - 1, hidden_size]`, model dtype | What the forward pass returns; for most models the input of the LM head. NumPy has no bfloat16, so a bfloat16 model's rows are `int16` bit patterns: `torch.from_numpy(rows).view(torch.bfloat16)`. |
| `topk_token_ids` | `[L - 1, k]`, `int32` | The `k` most likely next tokens, from the model's logits before any sampling parameter. |
| `topk_logprobs` | `[L - 1, k]`, `float32` | Their log-softmax values, in the same order. |

Logits recomputed from `hidden_states` are close to the engine's but not bitwise equal, so the behavior logprob comes from the inline sampled logprob, not from the hidden states or the top-k fields.

## Weight versions

`aux_version_segments` gives the weight version of every position below its closing entry `[end, null]`: segment `[start, v]` covers positions from `start` to the next segment's start. Positions from `end` on are in the tail object; their versions are not in the segments. For generated tokens the response's `weight_versions` spans give the version that sampled each token.

The versions are the ones the object names state. Positions the request reused carry the versions of the computation that produced their KV; positions it computed carry the version that computed them. They differ only in the documented limits ([protocol §9](protocol.md#9-limits)).

## Guarantees

- Each listed object holds the rows of the computation whose KV the request used for those positions, named with the tokens and weight versions of that computation, within the limits.
- Rows are stored as the engine computed them; nothing is padded, substituted or recomputed by the reader.
- The keys of a response that lists keys cover `[0, num_tokens - 1)` without gaps. A request that is aborted, or stops while preempted, lists none.
- After every engine that served a session has completed a pause, every object of the session's stopped requests is in the store, unless the engine dropped it at the stall deadline or a store process lost it.

## What the trainer must do

1. **Missing objects.** If `read_aux_rows` times out, read again after the engine's next completed pause; an object still missing then is lost, so drop the sample. When ranks read in a collective step, agree on the outcome across ranks before continuing.
2. **Replay.** Truncate to the training sequence's `L - 1` positions. Hook R3 replay where the model computes routing; some MoE implementations call the routing function directly and bypass a router-level hook.
3. **Behavior logprob.** Use the inline sampled logprob.
4. **Versions.** Positions of different versions are not interchangeable. Use the versions for staleness weighting or filtering.
5. **Release.** Delete a session with `aux_session_pattern` after every request of the session has been read, the training step has committed, and the engines have completed a pause since its last request stopped, for example the pause of the next weight update ([protocol §7](protocol.md#7-pause-and-release)). Never send another request with a released session name.

## Example

`wait_for_pause` returns after the engines' next completed pause and raises if the pause failed. Nothing is dropped silently: a second miss after a completed pause raises. A request that is aborted, or stops while preempted, lists no keys (`aux_output_keys` is `None`) and has no rows, so `load` returns `None` for it. Read only fields the engine enables: a missing field raises the same `BlockObjectStoreError` as a missing object, so this example waits for a pause before its second read raises. `client` in `release` is a `mooncake.store.MooncakeDistributedStore` connected to the object store ([protocol §7](protocol.md#7-pause-and-release)), not the `MooncakeAuxObjectStore` used for reading.

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
