# Generic p2p plan

## Sequence Diagram

```mermaid

sequenceDiagram
    participant Cons_TM as Consumer TieringManager
    participant Cons_P2P as Consumer P2P Tier
    participant Prod_P2P as Producer P2P Tier
    participant Prod_TM as Producer TieringManager

    Cons_P2P->>Cons_P2P: Open listener thread
    Prod_P2P->>Prod_P2P: Open listener thread

    Note over Cons_TM,Prod_TM: Producer has previously cached blocks for this prompt

    Note over Cons_TM,Cons_P2P: Step N - aggregate per-block lookups

    Cons_TM->>Cons_P2P: lookup per block
    Note right of Cons_P2P: do_p2p_fetch mode<br/>Returns None and registers key

    Cons_TM->>Cons_P2P: on_schedule_end
    Cons_P2P->>Prod_P2P: 𝗖𝗧𝗥𝗟: LookupMsg kv_request_id block_hashes
    Note right of Cons_P2P: One msg per peer per request per step

    Prod_P2P->>Prod_TM: lookup(key, ctx) per block_hash
    Note right of Prod_TM: HIT / MISS resolve now,<br/>HIT_PENDING / RETRY parked for re-poll
    Prod_P2P->>Prod_TM: create_store_job(hit_keys, ctx)
    Note right of Prod_TM: Pins primary slots for hits,<br/>returns JobMetadata(job_id, keys, block_ids)

    Prod_P2P--)Cons_P2P: 𝗖𝗧𝗥𝗟: LookupRespMsg kv_request_id hit_block_hashes

    Note over Cons_TM,Cons_P2P: Step N+k - resolve

    Cons_TM->>Cons_P2P: lookup per block retry
    Note right of Cons_P2P: Hit returns True<br/>Miss returns False<br/>In-flight returns None

    Cons_TM->>Cons_P2P: submit_load hit blocks only
    Note right of Cons_P2P: CPU slots allocated for hits only

    Cons_P2P->>Prod_P2P: 𝗖𝗧𝗥𝗟: FetchMsg kv_request_id block_hashes dst_block_indexes

    Prod_P2P-)Cons_P2P: 𝗗𝗔𝗧𝗔: NIXL Transfer WRITE src_descs dst_descs
    Prod_P2P-->>Cons_P2P: 𝗖𝗧𝗥𝗟: TransferDone kv_request_id success

    Cons_TM->>Cons_P2P: get_finished
    Note right of Cons_P2P: Hits loaded into GPU as normal cache hit<br/>Misses recomputed by the engine
```

## Lookup phase

The lookup phase lets a consumer probe which of its block hashes a
producer peer currently holds, before issuing a fetch. It runs
asynchronously: per-block `lookup()` calls aggregate across a scheduler
step, one `LookupMsg` is sent per `(peer, kv_request_id)` at
`on_schedule_end()`, and the response resolves the answer for a
subsequent `lookup()` call in a later step.

### Wire protocol

- `LookupMsg(kv_request_id, block_hashes)` — consumer asks which of
  these hashes the peer holds.
- `LookupRespMsg(kv_request_id, block_hashes, hits)` — two parallel
  arrays of equal length. Each `(block_hash, hit)` pair is
  self-describing, so the producer is free to split or coalesce
  responses across multiple LookupRespMsgs for the same
  `kv_request_id`.

### Client role

State is kept per `(kv_request_id, block_hash)` in `ClientRole._lookups`:

```text
            register_lookup()         flush_pending_lookups()    LookupRespMsg
   (none) ─────────────────► PENDING ──────────────────────► IN_FLIGHT ─────────► RESOLVED(bool)
                                │            send                                        │
                                │                                       register_lookup() │ (returns
                                │                                                         │  bool, deletes)
                                ▼                                                         ▼
                           idempotent: register_lookup() while PENDING/IN_FLIGHT returns None
```

- The first `manager.lookup(key, ctx)` for a symmetric-P2P consumer
  (`p2p` sub-dict in `kv_transfer_params`) registers a PENDING entry
  and returns `LookupResult.RETRY`.
- `manager.on_schedule_end()` drives `session.flush_pending_lookups()`
  on every session. The flush groups all unsent entries by
  `kv_request_id` and emits one `LookupMsg` per group; sends are gated
  on the connection's ConnectAckMsg.
- An incoming `LookupRespMsg` walks the `(block_hash, hit)` pairs and
  sets each entry's `result`.
- A subsequent `manager.lookup()` for the same key pops the entry and
  returns `LookupResult.HIT` / `LookupResult.MISS` accordingly — HIT
  becomes a normal secondary-tier hit (the manager starts promotion);
  MISS falls back to local prefill.
- A timeout (`_LOAD_TIMEOUT_S` since flush) sets `result = False` so
  the next `register_lookup` resolves via the happy path above instead
  of looping forever.

#### Entry lifecycle

There are three ways an entry leaves `_lookups`:

- **Happy path — resolved by next `register_lookup()`.** Once the
  response has set `state.result`, the next `register_lookup()` for
  that `(kv_request_id, block_hash)` deletes the entry and returns the
  bool. State lives only as long as the answer hasn't been delivered
  to the manager.
- **Request finished mid-flight — `on_request_finished()`.** The
  manager calls `session.finish_request()`, which calls
  `ClientRole.cancel_lookups(kv_request_id)` to drop every entry for
  that request whose `result` was never picked up.
- **Session torn down — `close()`.** Clears `_lookups` along with
  `_inbound`. Unresolved entries just disappear; the manager sees no
  session for the peer on the next call and the request falls back to
  local prefill.

### Server role

The producer answers any incoming `LookupMsg` from a connected peer —
there is no per-request producer flag. The reply is computed against
the local TieringManager via the `ParentManager` handle
(`tiering/base.py`) that `TieringOffloadingManager` passes to the tier
once per scheduler step through `serve_external_requests(parent)`. The
handle is valid **only** for the duration of that call, so message
dispatch merely enqueues inbound `LookupMsg`s (`on_lookup` →
`_pending_inbound_lookups`) and all parent interaction happens inside
`serve_external_requests`:

| `ParentManager` method | Purpose |
| --- | --- |
| `on_new_request(ctx) -> RequestOffloadingContext` | Open per-request bookkeeping for the synthetic peer-driven `ctx` before its first `lookup`. |
| `lookup(key, ctx) -> LookupResult` | Hit/miss decision per hash (fans out to the other tiers, P2P excluded). |
| `create_store_job(keys, ctx) -> JobMetadata` | Pin the primary-tier slots for the HIT keys; returns parallel `keys` / `block_ids` and a fresh `job_id`. The pin survives until the engine processes the matching `JobResult`. |
| `on_request_finished(ctx) -> None` | Release per-request bookkeeping the TieringManager accumulated under the synthetic peer-driven `ctx`. |

#### Per-LookupMsg tracking

`ServerRole._inbound_lookups: dict[lookup_id, _LookupBlocks]` tracks
the state for each inbound `LookupMsg` independently. Each entry
carries its own synthetic `ReqContext`:

```text
ctx = ReqContext(req_id=f"p2p:{peer_id}:{kv_request_id}:lu{lookup_id}")
```

A fresh ctx per LookupMsg gives the TieringManager a clean,
bounded-lifetime request to attach state to — created on the first
`parent.lookup` call, closed by `parent.on_request_finished` once the
lookup's last hash has been answered on the wire.

#### `on_lookup(kv_request_id, block_hashes)` (dispatch)

Runs during `session.poll()`, where no `parent` handle is available, so
it only appends `(kv_request_id, hashes, enqueued_at)` to
`_pending_inbound_lookups`. Resolution happens in the next
`serve_external_requests`.

#### `serve_external_requests(parent)` → `_process_inbound_lookup`

Drains `_pending_inbound_lookups`. For each, build a `_LookupBlocks`
with a `deadline` of `enqueued_at + _LOOKUP_PENDING_TIMEOUT_S`, call
`parent.on_new_request(ctx)`, then for each hash call
`parent.lookup(h, ctx)` and route:

| `LookupResult` | Action |
| --- | --- |
| `HIT` | Add to the newly-HIT list; record `resolved[h] = True`. |
| `MISS` | Record `resolved[h] = False`. |
| `HIT_PENDING` / `RETRY` | Add to `lookup.pending`. |

Then:

1. If any HITs: call `parent.create_store_job(hits, ctx)` once, then feed
   the returned JobMetadata into the existing
   `ServerRole.add_stored_blocks(...)` path so the eventual FetchMsg
   matches from `_outbound[kv_request_id].available`. HITs are pinned
   **immediately** even though the wire response is deferred — waiting
   would let the block evict before the client's FetchMsg lands.
2. If `lookup.pending` is empty (everything resolved on first sight),
   emit the aggregated `LookupRespMsg` now and fire
   `parent.on_request_finished(ctx)` (see `_finalize_lookup`).
3. Otherwise stash the entry in `_inbound_lookups` — the aggregate
   response is deferred until either every pending hash settles or
   `deadline` fires.

Exactly one `LookupRespMsg` goes out per inbound `LookupMsg`, carrying
every hash in its original wire order. The single-thread guarantee
(`lookup → HIT → create_store_job` in one synchronous sequence) means
no eviction can race between HIT detection and the pin.

#### `_resolve_pending_lookups` (driven from `serve_external_requests`)

Called once per serve, after the newly-enqueued lookups are processed.
For every parked lookup:

- Re-call `parent.lookup` per still-pending hash.
    - `HIT` → move to the newly-HIT list, record `resolved[h] = True`,
      drop from `pending`.
    - `MISS` → record `resolved[h] = False`, drop from `pending`.
    - Still `HIT_PENDING` / `RETRY` → leave for next serve.
- If `now >= lookup.deadline` and `pending` is still non-empty, force
  the stragglers to MISS (record `resolved[h] = False` for each and
  clear `pending`). Guarantees the response goes out within the
  deadline even against a stuck producer.
- For the newly-HIT list: one `parent.create_store_job(...)` call per
  lookup, plumbed through `add_stored_blocks` as above.
- Any lookup whose `pending` is now empty is finalized by
  `_finalize_lookup`: emit the aggregated `LookupRespMsg` in wire
  order using `resolved`, then fire `parent.on_request_finished(ctx)`
  and drop the entry.

#### Cleanup

Closing a lookup outside a serve window cannot call `parent`, so both
paths below queue the ctx for the next `serve_external_requests` to
release via `parent.on_request_finished`:

- `finish(kv_request_id)` / `on_fetch` (via `_finish_inbound_lookups`) —
  pops every parked lookup matching `kv_request_id`, drops any
  still-unprocessed raw entry for it, and appends each popped
  `lookup.ctx` to `_finished_lookup_ctxs`. In symmetric P2P `finish`
  rarely fires: the producer has no local request lifecycle for the
  consumer's id.
- `close()` — returns `(failed_store_job_ids, orphan_ctxs)` where
  `orphan_ctxs` is every remaining parked/queued ctx, then clears the
  lookup collections. The manager stashes them in
  `_orphan_finish_ctxs` and flushes them at the top of its next
  `serve_external_requests`.

#### Pin lifetime

The pin created by `parent.create_store_job` covers the full lookup →
fetch → transfer → completion journey, which spans multiple engine steps:

```text
create_store_job      ─►  JobMetadata registered, ref_cnt += 1 on each block
add_stored_blocks     ─►  _outbound[req].available populated; _store_jobs[job_id] tracked
on_fetch              ─►  add_fetch_demand matches available, NIXL write_blocks issued
collect_results       ─►  StoreResult emitted on transfer completion
manager._finished_jobs ─► JobResult bubbled to engine's get_finished_jobs()
TieringManager        ─►  _process_finished_jobs pops _transfer_jobs[job_id] and
                          calls complete_read(keys, ctx) — ref_cnt -= 1
```

The release path is the same one PD already uses — there is no
separate `complete_store_job` callback. Failure modes (peer drops the
fetch, transfer fails, `_STORE_TIMEOUT_S` fires, session closes,
`finish` arrives) all flow through the existing terminal-finalize
code, which emits `StoreResult(success=False)` for the pinned job and
the engine still releases ref_cnt.
