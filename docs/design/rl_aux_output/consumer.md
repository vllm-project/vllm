# Reading Aux Outputs in the Trainer

This page is the contract between the engine and the trainer for session requests. How the engine names and writes objects is in the [protocol](protocol.md).

The `rl/aux-output-session-wise` branch (head `04495f3806`) carries the version of this page that matches its code, including how a trajectory built from several responses takes each position's row and how an aborted request is continued.

## What the rollout keeps

For every request of a session, keep from the final output:

- `prompt_token_ids` (the engine-side prompt) and the generated token ids;
- `aux_output_keys`;
- `aux_version_segments`, to send with the session's next request and to label positions with their weight version;
- the inline logprob of each sampled token, which is the behavior logprob.

A training sequence of `L` tokens is read with the keys of the request whose tokens it is: they cover positions `[0, L - 1)`. For a multi-turn trajectory trained as one sequence that is the last turn. A sequence that is a shorter prefix (one turn, one segment of a compacted rollout) is read with its own request's keys. Every request's keys stay readable until the session is released.

## Reading

```python
from vllm.distributed.aux_output_connector import object_store

store = object_store.MooncakeAuxObjectStore.create()  # uses MOONCAKE_CONFIG_PATH
rows = object_store.read_aux_rows(store, keys, "routed_experts", pipeline_parallel_size)
```

`read_aux_rows(store, keys, field, pipeline_parallel_size=1, timeout_s=60.0)` reads the objects of `keys` in order and concatenates one field. With pipeline parallelism each key names one object per stage: routed experts are concatenated over the stages in stage order, and a field that only the last stage holds (hidden states, top-k logprobs) is read from that stage. Pass the engine's pipeline-parallel size.

Writes are asynchronous, so an object can still be in flight when the response arrives: the reader waits up to `timeout_s` in total, then raises `BlockObjectStoreError`. It raises the same error when no stage of an object holds the field, for example a field the engine did not enable.

Create the store client in a process that does not start an engine afterwards. A Mooncake client created before the engine breaks CUDA initialization in the engine's workers.

## Fields

All arrays have one row per position; row `p` belongs to position `p`, the position that produces the logits for token `p + 1`.

| Field | Shape and type | Notes |
| --- | --- | --- |
| `routed_experts` | `[L - 1, MoE layers, experts per token]`, `uint8` up to 256 experts, else `uint16` | Logical expert ids. |
| `indexer_topk` | `[L - 1, indexer source layers, index_topk]`, `int32` | The selections exactly as captured: order, padding and invalid ids are preserved. DeepSeek V4 indices address the compressed KV cache, not token positions. See [Indexer Replay](../../training/indexer_replay.md). |
| `hidden_states` | `[L - 1, hidden_size]`, model dtype | What the forward pass returns; for most models the input of the LM head. NumPy has no bfloat16, so a bfloat16 model's rows are `int16` bit patterns: `torch.from_numpy(rows).view(torch.bfloat16)`. |
| `topk_token_ids` | `[L - 1, k]`, `int32` | The `k` most likely next tokens, from the model's logits before any sampling parameter. |
| `topk_logprobs` | `[L - 1, k]`, `float32` | Their log-softmax values, in the same order. |

Logits recomputed from `hidden_states` are close to the engine's but not bitwise equal, so the behavior logprob comes from the inline sampled logprob, not from the hidden states or the top-k fields.

## Weight versions

`aux_version_segments` gives the weight version of every position below its closing entry `[end, null]`: segment `[start, v]` covers positions from `start` to the next segment's start. Positions from `end` on are in the tail object; their versions are not in the segments. For generated tokens the response's `weight_versions` spans give the version that sampled each token.

The versions are the ones the object names state. Positions the request reused carry the versions of the computation that produced their KV; positions it computed carry the version that computed them. They differ only in the documented limits ([protocol §12](protocol.md#12-limits)).

## Guarantees

- Each listed object holds the rows of the computation whose KV the request used for those positions, named with the tokens and weight versions of that computation, within the limits.
- Rows are stored as the engine computed them; nothing is padded, substituted or recomputed by the reader.
- The keys of one response cover `[0, num_tokens - 1)` without gaps.
- After every engine that served a session has completed a pause, every object of the session's stopped requests is in the store.

## What the trainer must do

1. **Missing objects.** If `read_aux_rows` raises, drop the sample or retry. When ranks read in a collective step, agree on the outcome across ranks before continuing.
2. **Replay.** Truncate to the training sequence's `L - 1` positions. Hook R3 replay where the model computes routing; some MoE implementations call the routing function directly and bypass a router-level hook.
3. **Behavior logprob.** Use the inline sampled logprob.
4. **Versions.** Positions of different versions are not interchangeable. Use the versions for staleness weighting or filtering.
5. **Release.** Delete a session with `aux_session_pattern` after every request of the session has been read, the training step has committed, and the engines have completed a pause since its last request stopped, for example the pause of the next weight update ([protocol §10](protocol.md#10-pause-and-release)). Never send another request with a released session name.

## Example

```python
import numpy as np
import torch
from vllm.distributed.aux_output_connector import object_store
from vllm.distributed.aux_output_connector.store import BlockObjectStoreError


def load(store, out, tokens, pipeline_parallel_size=1):
    n = len(tokens) - 1
    try:
        r3 = object_store.read_aux_rows(
            store, out.aux_output_keys, "routed_experts", pipeline_parallel_size
        )
    except BlockObjectStoreError:
        return None  # caller agrees across ranks and skips or retries together
    return torch.from_numpy(r3[:n].astype(np.int64))


def release(client, object_key_prefix, session):
    client.remove_by_regex(
        object_store.aux_session_pattern(object_key_prefix, session), True
    )
```
