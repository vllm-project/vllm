# Reading Aux Outputs in the Trainer

This page is the contract between the engine and the trainer for requests with `aux_namespace`: what the trainer sends, keeps, splices, reads, masks and releases. How the engine names, lists and writes objects is in the [protocol](protocol.md).

## Choosing the namespace

Send every sample of a GRPO group as its own request (`n > 1` is rejected) and choose the granularity ([protocol §2](protocol.md#2-granularity-what-a-namespace-is)):

- **Group:** `aux_namespace` is the group id, for every sample, later turn, sub-agent and retry of the group. Samples sent at once can each compute the prompt, and all but one then list another sample's computation of it; to compute it once, send one sample first and the others to the same replica and rank once its prefill has finished ([protocol §2](protocol.md#samples-sent-at-once)).
- **Trajectory:** `aux_namespace` is `<group id>/<sample index>`. A fork stays in its parent's namespace, or gets one nested under it.

Use segments of `[A-Za-z0-9._-]`, never `tail`, and never reuse a namespace after its release. Routing all requests of a namespace to one engine replica is recommended, not required: it keeps the hits and, under group granularity, keeps samples on different replicas from sharing a block computed at another weight version.

## What the rollout keeps

For every request, keep:

- the prompt token ids (engine side) and the generated token ids;
- the namespace and the `aux_prompt_start` it sent, and `aux_objects`;
- the store's fault epoch when the namespace opened, before its first request, to compare before training ([Failures](#failures));
- `weight_versions`, the version that sampled each generated token (sampling provenance, see below);
- the engines that served requests of its namespace: the write barrier is a pause of each of them, since a request can list objects another engine of the namespace wrote;
- the inline logprob of each sampled token, which is the behavior logprob.

## Setting `aux_prompt_start`

A trajectory holds products for positions `[0, h)`: 0 before its first request, then whatever its listings covered. Send

```text
aux_prompt_start = min(h, longest common token prefix of the new prompt and the tokens the trajectory holds, prompt length − 1)
```

When the new prompt extends the held tokens, this is `h`. The last term lists the prompt's last position, which the request computes again and samples from: a retry or a regeneration then takes that position from its own computation, unless its prompt ends on a block boundary ([protocol §7](protocol.md#7-turns-aux_prompt_start-and-splicing)). After a retokenization, a compaction or a rewritten history, it is the first changed token. After an aborted request, `h` is the end of what that request listed; the next request computes and lists from there. A response may list fewer positions than its generated tokens need (an abort): keep the products it lists, and either continue the trajectory from them with the next request, or drop the generated tokens after `h + 1` and generate them again.

**Multimodal inputs.** Compare the media too. Block hashes cover the media items that fill placeholder tokens, so the common prefix ends at the first position where the token ids differ or a placeholder is filled by another media item (by the client's identity for the item, such as a hash of its content). In the example below, pass `prompt` and record `tokens` as pairs `(token id, media identity or None)`.

## Splicing a trajectory

A trajectory of `L` tokens needs the products of positions `[0, R)`, where `R = max(L − 1, 0)`. Keep one list of entries `(key, first, start, end)` per trajectory: `first` is the first position the object holds (the `start` it was listed with), `[start, end)` the positions the trajectory takes from it, at offset `start − first`.

1. **Cut back.** Let `e = min(aux_prompt_start, R)`, with `R` for the tokens after this request: a request aborted before it generated a token can leave fewer tokens than the positions held. Drop the entries from `e` on, and cut the last kept entry at `e`.
2. **Append, earlier copies win.** For each listed `[key, start, end]`, in order: skip it if `end ≤ e`; otherwise append `(key, start, max(start, e), end)` and set `e = end`. Entries of one listing overlap where a preempted request's tail comes before the object that computed its positions again; this step keeps the tail, the computation the request sampled from.
3. **Check before training.** Entries abut, and `e == R`. An entry that starts after `e` leaves a gap; a gap, or an end short of `R`, masks the sample. While a trajectory continues, an aborted request may leave `e < R`; the next request lists from `e`. A trajectory with `R = 0` needs no products.

Step 2 is what keeps the first stored copy: a turn's tail object keeps its positions, and the next turn's full object over the same block is used only after them ([protocol §7](protocol.md#7-turns-aux_prompt_start-and-splicing) has a worked example).

**Context parallelism.** Splice the whole trajectory first, then apply to the products array the same split of positions that is applied to the tokens; each rank reads only the objects that overlap its positions.

## Reading

```python
from vllm.distributed.aux_output_connector import object_store

store = object_store.MooncakeAuxObjectStore.create(mooncake_config_path)
rows = object_store.read_aux_rows(store, [key], "routed_experts")
```

Pass the engine's `AuxOutputConfig.mooncake_config_path`, or `None` to read `MOONCAKE_CONFIG_PATH` as an engine without it does. `read_aux_rows(store, keys, field, timeout_s=600.0)` reads the objects of `keys` in order and concatenates one field, waiting up to `timeout_s` in total for objects still being written, then raises `BlockObjectStoreError`; it raises the same error when an object does not hold the field. Read each key of a training batch once, deduplicating across samples (under group granularity the prompt objects are shared), with one deadline for the batch, and slice each entry at its offset.

- **Timeout.** Keep the batch's deadline no shorter than the engine's stall deadline (150 s by default, [protocol §10](protocol.md#10-failures)), since a full or unreachable store can hold writes back that long while the engine keeps running. Pass each read what is left of it: a full `timeout_s` per key would wait it out once per missing key.
- **Lost objects.** A timeout does not by itself mean an object is lost. Wait once per batch, not once per key: if any key is still missing, wait until every engine that served the batch's namespaces has completed a pause that started after their requests stopped, and read the missing keys again with `timeout_s=0`; an object still missing then is lost.
- **Blocking gets.** The deadline is checked between reads: one `get` that blocks, for example after the master died (more than 70 s in our tests), is not interrupted, so give the reading step its own timeout. A reader that reads with Mooncake batch calls can fail transport errors after a short bound (10 s) instead of waiting out the deadline.

Create the store client in a process that does not start an engine afterwards. A Mooncake client created before the engine breaks CUDA initialization in the engine's workers.

## Fields and versions

All arrays have one entry per position. Entry `p` of a trajectory belongs to position `p`, the position that produces the logits for token `p + 1`.

| Field | Shape and type | Notes |
| --- | --- | --- |
| `routed_experts` | `[n, hidden layers, experts per token]`, `uint8` up to 256 experts, else `uint16` | Logical expert ids; dense layers are 0. |
| `indexer_topk` | `[n, indexer source layers, index_topk]`, `int32` | Exactly as captured: order, padding and invalid ids preserved. DeepSeek V4 indices address the compressed KV cache, not token positions. |
| `hidden_states` | `[n, hidden_size]`, model dtype | What the forward pass returns; for most models the input of the LM head. bfloat16 as `int16` bit patterns: `torch.from_numpy(a).view(torch.bfloat16)`. |
| `topk_token_ids` | `[n, k]`, `int32` | The `k` most likely next tokens, before any sampling parameter. |
| `topk_logprobs` | `[n, k]`, `float32` | Their log-softmax values, in the same order. |

Logits recomputed from `hidden_states` are close to the engine's but not bitwise equal, so the behavior logprob comes from the inline sampled logprob, not from the hidden states or the top-k fields.

**Versions.** The response's `weight_versions` gives the weight version that sampled each generated token. It is sampling provenance only: it does not give the version of the products a listing names. The first stored copy of a position can come from another computation than the one that sampled the next token, at another version: a block computed again after its KV was reset or evicted, the last prompt position of a retry whose prompt ends on a block boundary, or, under group granularity, a block another sample stored ([protocol §12](protocol.md#12-limits)). One object can also hold positions computed in several steps at several versions. Positions whose next token is prompt text (the prompt, tool results) have no sampling version at all. The versions of the stored products are not delivered; a per-position version in the object header would carry them and is not implemented. Equal versions do not imply identical products without batch invariance.

## Failures

1. **A store fault drops the namespace.** Compare the store's fault epoch after reading with the one recorded when the namespace opened, before training on its samples. If it changed (a holder was lost, restarted or dropped after `client_ttl`, or the master restarted), drop every sample of the namespace not yet consumed (masked, counted as store faults), do not continue its trajectories, and release the namespace. Read misses do not detect a store fault: an object lost with a holder can be written again by another computation of the same tokens, and the read then succeeds ([protocol §10](protocol.md#10-failures)).
2. **Missing object.** Without a store fault, an object still missing after the write barrier is lost: the engine dropped it. Later requests that hit its KV list the same key without writing it; a request that computes the block again after its KV was evicted writes its own computation under the key ([protocol §12](protocol.md#12-limits)).
3. **Mask the whole sample.** A sample whose splice needs an unreadable object, or whose listings leave a position uncovered, is masked entirely, and its trajectory is not continued. Under group granularity a lost prompt object masks every sample of the group that lists it; under trajectory granularity one sample.
4. **Skip a fully masked step.** When every sample of a training step is masked or dropped, skip the update for that step and continue.
5. **Agree across ranks.** When ranks read in a collective step, agree on the masked and dropped samples before the step.
6. **Release anyway.** A masked, failed or dropped sample is settled. Release its namespace as below, lost objects included.

## Release

Remove a namespace once every sample it expects, and every sample of the namespaces nested under it, is settled (consumed with its training step committed, masked, failed, or dropped without being sent again), no request of these namespaces is running or will be sent, and every engine that served a request of these namespaces has completed a pause since their last request ended, for example the pause of the next weight update ([protocol §8](protocol.md#8-pause-and-release)). A group released while one of its samples is still to be sent would let that sample hit KV whose objects are gone. Under group granularity that is once per group, after its slowest sample settles; under trajectory granularity once per sample, or once per group for all its samples.

Release under the key prefix the namespace's listed keys begin with. The engines use a new prefix every time they start ([protocol §11](protocol.md#11-deployment)): after a restart, remove the previous prefix whole once the engines that wrote under it have stopped. `client` below is a `mooncake.store.MooncakeDistributedStore` connected to the object store, not the `MooncakeAuxObjectStore` used for reading. Force is required. On Mooncake 0.3.14 repeat the removal after the next pauses: an offload in flight can register a removed object again. Never send another request with a released namespace.

## Example

`load` reads one field for a training batch. It reads each key once, with one deadline for the batch, waits for one pause only if a key is still missing, and reads the missing keys again without waiting. `wait_for_pause` returns after the next completed pause of every engine that served the batch's namespaces, and raises if a pause failed or cannot come because an engine shut down. `store_epoch` returns the store's fault epoch; Mooncake 0.3.14 reports none, so the trainer integration supplies it ([protocol §9](protocol.md#9-the-store)). A trajectory gets `LOST` when its listings leave a position uncovered or it needs a key still missing after the pause, and `STORE_FAULT` when its namespace opened at another epoch: the caller masks both, and for `STORE_FAULT` drops the namespace's samples and releases it. Read only fields the engine enables: a missing field raises the same `BlockObjectStoreError` as a missing object.

```python
import time

import numpy as np
from vllm.distributed.aux_output_connector import object_store
from vllm.distributed.aux_output_connector.store import BlockObjectStoreError

LOST = object()
STORE_FAULT = object()


class Trajectory:
    def __init__(self, namespace: str, epoch):
        self.namespace = namespace
        self.epoch = epoch  # the store's fault epoch when the namespace opened
        self.tokens: list = []
        self.entries: list[tuple[str, int, int, int]] = []  # key, first, start, end

    @property
    def held(self) -> int:
        return self.entries[-1][3] if self.entries else 0

    def prompt_start(self, prompt: list) -> int:
        n = min(len(prompt), len(self.tokens))
        same = next((i for i in range(n) if prompt[i] != self.tokens[i]), n)
        return min(self.held, same, max(len(prompt) - 1, 0))

    def record(self, start: int, tokens: list, listing) -> None:
        start = min(start, max(len(tokens) - 1, 0))
        kept = []
        for key, first, s, e in self.entries:
            if s >= start:
                break
            kept.append((key, first, s, min(e, start)))
        end = start
        for key, first, stop in listing or ():
            if stop > end:
                kept.append((key, first, max(first, end), stop))
                end = stop
        self.entries, self.tokens = kept, tokens

    def covered(self) -> bool:
        end = 0
        for _, _, s, e in self.entries:
            if s != end:
                return False
            end = e
        return end == max(len(self.tokens) - 1, 0)


def load(store, trajectories, field, wait_for_pause, store_epoch, timeout_s=600.0):
    epoch = store_epoch()
    keys = sorted(
        {
            key
            for t in trajectories
            if t.epoch == epoch and t.covered()
            for key, _, _, _ in t.entries
        }
    )
    rows = {}

    def read(keys, deadline):
        for key in keys:
            try:
                rows[key] = object_store.read_aux_rows(
                    store, [key], field, timeout_s=max(deadline - time.monotonic(), 0)
                )
            except BlockObjectStoreError:
                pass

    read(keys, time.monotonic() + timeout_s)  # one deadline for the batch
    missing = [key for key in keys if key not in rows]
    if missing:
        wait_for_pause()  # once per batch: after it, a missing key is lost
        read(missing, time.monotonic())
    epoch = store_epoch()  # checked again before the products are used
    out = []
    for t in trajectories:
        if t.epoch != epoch:
            out.append(STORE_FAULT)
        elif not t.covered() or any(key not in rows for key, _, _, _ in t.entries):
            out.append(LOST)
        elif not t.entries:
            out.append(None)  # no positions to read
        else:
            parts = [rows[key][s - first : e - first] for key, first, s, e in t.entries]
            out.append(np.concatenate(parts))
    return out


def release(client, object_key_prefix, namespace):
    client.remove_by_regex(
        object_store.aux_namespace_pattern(object_key_prefix, namespace), True
    )
```

A namespace opens with `epoch = store_epoch()` before its first request, and each of its trajectories is created with that epoch. A request sends `{"aux_namespace": namespace, "aux_prompt_start": trajectory.prompt_start(prompt)}` in its sampling parameters; its response is recorded with `trajectory.record(start, prompt + generated, choice.aux_objects)`.
