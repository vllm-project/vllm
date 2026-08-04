# Cache-event emission (MP server → key directory)

Module: `lmcache/v1/mp_coordinator/cache_events.py`
Contract vocabulary: `lmcache/v1/mp_coordinator/api.py`
Consumer: `lmcache/v1/mp_coordinator/key_directory.py` (see
[key_directory.md](key_directory.md))

This is the emission half of the key directory (M1 of the control-plane
RFC, [issue #4226](https://github.com/LMCache/LMCache/issues/4226)): MP
servers turn storage-listener callbacks into `CacheEventBatch` streams
and deliver them to the coordinator's directory.

## The transport seam

Production deployments may replace direct HTTP push with Kafka or
another message queue. The design isolates that choice behind one
interface so nothing else changes:

```
storage layer ──► EventBus ──► CacheEventSubscriber ──► CacheEventSink ──► directory
 (publishes)      (drain      (event → vocabulary,        (transport)
                  thread)      seq, batching)
```

- **`CacheEventSink`** — `publish(batches)` with **at-least-once**
  delivery, preserving list order within and across calls. That is the
  entire transport contract, and it is deliberately weak: the directory
  already absorbs everything a real transport does wrong. Redelivery is
  deduplicated by the per-instance `seq` cursor, loss surfaces as a
  `seq` gap that flags the instance for resync, and restarts are fenced
  by `incarnation`. A sink never needs exactly-once or global ordering.
- **`HttpCacheEventSink`** — the first sink: one
  `POST /directory/events` per flush, batches in list order. Failures
  raise `CacheEventPublishError`; the caller decides retry vs drop
  (both are safe, see above).
- A future **Kafka sink** produces to a topic with the message key set
  to `instance_id`, so one partition carries one instance's stream —
  partition FIFO is exactly the per-instance FIFO the directory needs.
  The coordinator side gains a consumer that feeds
  `KeyDirectory.apply_batch`; the subscriber and producers are
  untouched.

## Batching and sequencing (inside the subscriber)

One `CacheEventSubscriber` per MP-server process owns the buffer, the
`seq` counter, and the sink:

- **Order-preserving batching.** The buffer is a list of *pending
  batches*: consecutive records with the same `(event_type, tier,
  backend)` identity append to the last pending batch; an identity
  change starts a new one (never merged backwards). Flushing emits one
  `CacheEventBatch` per pending batch, so the batch sequence preserves
  the total order of recorded events — a store followed by a delete of
  the same key can never be reordered into "delete, then store".
  Alternating identities therefore produce multiple pending batches of
  the same identity; that is intentional (extra batch headers, never
  reordering).
- **`seq` is consumed even when publish fails.** A failed flush drops
  the drained list (bounding memory while the coordinator is down) but
  keeps the `seq` numbers it assigned. The directory sees a gap and
  sets `gap_detected` for the instance — the honest signal that events
  were lost and the resync backstop (future work) should reconcile.
  Reusing the seqs instead would hide partial-delivery ambiguity (an
  HTTP timeout after the coordinator applied the batch).
- **`incarnation` = server start time** (`int(time.time())` at
  lifespan startup). A restarted server's first batch fences out every
  placement its previous incarnation reported, matching the fact that
  its pools restarted empty.

## Event flow (the observability bus)

The storage layer already publishes key-level events to the
observability `EventBus` (`mp_observability/event_bus.py`);
cache-event emission rides the same bus instead of adding parallel
listener plumbing or a dedicated flush task:

- **Producers.** `L1Manager` publishes `l1.write.finished`,
  `l1.write_finished_and_read_reserved`, `l1.keys.evicted` (all delete
  paths), and the new `l1.keys.accessed` (`touch_keys` — the MP request
  end's unified touch of a request's retrieved and stored keys; the
  subscriber deliberately does **not** consume `l1.read.finished`,
  which would duplicate those accesses). The placement-bearing events
  (stores and evictions) carry `meta: list[L1ObjectMeta]` — each
  object's `size_bytes` (`MemoryObj.get_size()`) and its
  `L1BackendType` medium from `L1ManagerProtocol.get_backend_type()` (the
  Device-DAX tier resolves it per object via
  `DevDaxMemoryAllocator.is_devdax_obj`, i.e. `MemoryObj.parent()`) —
  so a hybrid DRAM+DAX L1 reports exactly where each object landed,
  and deletes target the same placement identity `(instance, tier,
  backend)` their store reported. The L2 base adapter's
  listener-notify funnel publishes the new `l2.keys.stored`
  (`keys`+`sizes`+`backend`), `l2.keys.accessed`, and `l2.keys.deleted`
  events; the backend name is the registered adapter type, stamped by
  the storage manager via `set_backend_name` at build time — so
  **runtime-added adapters emit automatically**.
- **`CacheEventSubscriber`** maps those events onto the directory
  vocabulary (writes → `STORE`, evictions/deletes → `DELETE`, split per
  actual L1 medium from the event metadata; touches → `ACCESS`).
  `ACCESS` batches carry an **empty backend**: the directory only
  refreshes key-level recency on access, so there is no placement
  identity to name — the vocabulary requires a non-empty backend for
  `store`/`delete` only. The subscriber is single-threaded by design —
  everything runs on the bus's drain thread, so it needs no locking.
- **Threading.** The bus dispatches on one drain thread, which is
  exactly the per-instance FIFO the directory needs. The subscriber
  self-paces delivery: recording flushes when `flush_interval` has
  elapsed since the last flush, bounding the sink-publish rate under
  load. There is no timer of its own — the subscriber additionally
  subscribes to `l1.eviction.loop_tick` (published continuously by the
  L1 eviction loop) as a flush pump, so a burst-ending tail (e.g. L2
  store completions) is delivered within one tick of the interval
  elapsing instead of waiting for the next request. The sink posts
  synchronously with a short timeout (a slow coordinator briefly
  stalls the drain, bounded by the timeout; overflow beyond the bus's
  bounded queue is dropped and surfaces as a `seq` gap → resync).
- **Coupling.** The stream requires the bus: enabling
  `--coordinator-event-reporting` together with
  `--disable-observability` is rejected at startup. Bus-level drops
  under overload are acceptable by the same argument as transport loss —
  the directory is eventually consistent soft state.

## L1 media

L1 media are a closed set, hence the `L1BackendType` enum
(`distributed/api.py`: `DRAM`, `DEVDAX`, `GDS`); L2 backends stay
strings because adapter types are an open registry (plugins register
new type names).

## Wiring and configuration

Enabled in the MP HTTP server lifespan when a coordinator URL is set
and `--coordinator-event-reporting` (or
`LMCACHE_COORDINATOR_EVENT_REPORTING`) is on;
`--coordinator-event-flush-interval` paces the subscriber's
event-driven flushes (default 1s). The same flags also gate the
legacy quota stream (`/quota/events`), which keeps its own schema and
listener until the two streams are unified.

## Known limitations (follow-ups)

- **Bus overflow drops events silently** (bounded queue, rate-limited
  warning); the resulting `seq` gap flags the instance for resync, but
  the resync backstop itself is future work.
- **The flush pump is coupled to the eviction loop's tick** — decouple
  it (e.g. a bus-owned periodic hook) so tail freshness does not depend
  on that loop's cadence.
- The legacy quota stream (`/quota/events`) can be re-based on this
  stream: route directory-applied `l2` batches into the usage/eviction
  consumers on the coordinator, then delete the `L2EventListener`
  client and the endpoint.
