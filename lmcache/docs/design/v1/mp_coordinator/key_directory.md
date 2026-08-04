# Key directory

Module: `lmcache/v1/mp_coordinator/key_directory.py`
Contract vocabulary: `lmcache/v1/mp_coordinator/api.py`
HTTP surface: `lmcache/v1/mp_coordinator/http_apis/directory_api.py` (`/directory/*`)

This is the first milestone (M1) of the control-plane RFC
([issue #4226](https://github.com/LMCache/LMCache/issues/4226)): a fleet-wide
directory mapping each `ObjectKey` to its known placements — which instance
holds it, on which tier (`l1`/`l2`) and backend (`dram`, `cxl`, `fs`, ...), at
what size.

## Contract

The directory is **eventually consistent, soft state, and never on the
serving hot path**:

- It is built purely from `CacheEvent` batches emitted by MP servers. It
never mutates memory and never grants access to bytes.
- Every answer is a **hint**. Consumers (P2P discovery, cache control, prefetch planning) must validate at the owning MP server before touching bytes (validate-on-use). Stale state costs a wasted probe or a missed reuse.
- All state is reconstructible from event replay plus per-instance resync; nothing is persisted at the moment.

## Event application semantics

One batch = `(instance_id, incarnation, seq, event_type, tier, backend, entries[], ts)`. `apply_batch` enforces, in order:


| mechanism           | rule                                                                                                                                                                   | why                                                                                                                                         |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Incarnation fencing | `incarnation <` current → drop batch (`STALE_INCARNATION`). `incarnation >` current → drop **all** placements the old incarnation reported, then start a fresh cursor. | An MP-server restart empties its pools; no placement may survive the pool it lived in.                                                      |
| Seq dedup           | `seq <=` last applied (same incarnation) → drop batch (`DUPLICATE`).                                                                                                   | Replays (retry, event-bus redelivery) must be idempotent.                                                                                   |
| Gap detection       | `seq >` last applied `+ 1` → set the instance's `gap_detected` flag (visible in stats), apply anyway.                                                                  | Events may be lost; the flag marks the instance's slice as needing resync. Entry application is idempotent, so applying past a gap is safe. |


Per-instance FIFO by `seq` is the **only** ordering the design needs: each
instance is the sole writer of its own facts, so there is no global order
and no cross-instance arbitration.

Entry semantics by event type:

- `STORE` — upsert the placement at identity `(instance_id, tier, backend)`; re-store replaces the size. Records the entry's
`content_hash` (when present) as the back-pointer that will keep content
index (I2) deletes O(1) once I2 lands (M2).
- `DELETE` — remove that placement identity (owners report evictions as
deletes too). Removing an absent placement/key is a no-op. A key with no
remaining placements is dropped from the directory.
- `ACCESS` — refresh the key's `last_access` recency (max of batch `ts`);
never creates records, and carries no placement identity — its
`backend` may be empty (`tier`/`backend` are ignored on apply). `ts` is
emitter wall-clock and is never compared across instances.

## Structures

```
ObjectKey → _KeyRecord {
    placements: list[Placement],   # ≤1 per (instance_id, tier, backend)
    content_hash_hex, last_access
}
instance_id → _InstanceState { incarnation, last_seq, gap_detected, keys }
```

`_InstanceState.keys` is the reverse index that makes incarnation fencing
and `drop_instance` (deregistration cleanup) proportional to the
instance's own keys instead of a full directory scan.

The Python-phase directory is keyed by `ObjectKey` directly (hashable
frozen dataclass, same as the L2 usage manager). The RFC's 16-byte
`key_hash` with interned `model_id`/`salt_id` is a memory/native-port
optimization (M6), not a semantic change.

## HTTP surface

- `POST /directory/events` — apply `CacheEventBatch` batches (list
order; per-instance emission order required). Duplicates and stale
batches are counted in the response, not errors.
- `POST /directory/lookup` — resolve keys to placements (POST because the
key list rides in the body). One result per key, request order, empty
for unknown keys.
- `POST /directory/lookup_tokens` — resolve a token sequence to keys
(prefix-exact: the fleet `TokenHasher` + per-rank fan-out, as the pin
APIs do; requires `model_name` / `world_size` / `cache_salt` since key
identity includes them) and return each key's placements.
Position-independent token matching arrives with the content index (M2).
- `GET /directory/stats` — key/placement counts plus per-instance stream
state (`incarnation`, `last_seq`, `gap_detected`) for observability and
the future resync trigger.

Type placement:

- **`api.py`** — the cache-event vocabulary (`CacheEventType`,
`CacheEventEntry`, `CacheEventBatch`): the contract between the
MP-server emitter and the directory. Plain dataclasses with intrinsic
invariants in `__post_init__` (the `ObjectKey` pattern: `seq >= 1`,
concrete tier, non-empty ids are unconstructible anywhere).
- **`key_directory.py`** — the engine plus everything the directory
itself produces (`Placement`, `ApplyResult`, stats) and its private
records.
- **`schemas.py`** — HTTP models only. 

MP-server emission of the `CacheEvent` stream (L1 + L2, `incarnation` =
server start time) is implemented — see
[cache_events.md](cache_events.md). Re-basing the legacy
`/quota/events` stream on the same emitter is still open.

## Deliberately out of scope (follow-ups)
- **Resync integration**: acting on `gap_detected` (digest/resync
backstop, `UNCONFIRMED` placement decay) — extends today's
`L2ResyncManager` pattern to L1.
- **Registry integration**: calling `drop_instance` from deregistration /
heartbeat-timeout eviction (the method exists and is tested).
- **Content index (I2)**, blend rewiring, checkpointing, and the
`DELETE_PENDING`/pin placement states used by tier-aware cache-control
directives (M2–M4 of the RFC).
- **Token store (I3)**: the opt-in `content_hash → token_ids` store for
`key → tokens` introspection, fed by `TOKENS` events and refcounted from
key records via the `content_hash` back-pointer. Nothing
correctness-bearing reads it, so it ships with its first real consumer.

