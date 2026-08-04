# Per-Request Root Span

This document describes the root span design for `MPServerTracingSubscriber`:
how a single `"request"` OTel span wraps all child operations for one request,
and how the span close is deferred correctly when GPU stores are still in flight.

## Problem

Before this change, `MPServerTracingSubscriber` emitted flat, orphaned child
spans (`mp.store`, `mp.retrieve`, `mp.lookup_prefetch`) with no parent context.
Traces in Tempo/Jaeger showed disconnected spans with no request-level view.

## Design

Each request gets one root `"request"` span that:

- Opens at `MP_REQUEST_START` — the first CPU-synchronous touch of a `request_id`
- Nests all child spans beneath it via OTel context propagation
- Closes at `MP_REQUEST_END`, deferred if async GPU stores are still in flight

### New Events

Four new `EventType` values, all CPU-synchronous:

| Event | Published from | Purpose |
|-------|---------------|---------|
| `MP_REQUEST_START` | `lookup_prefetch_start()`, top of method | Open root span at true request arrival |
| `MP_STORE_SUBMITTED` | `store()`, before `publish_on_stream(MP_STORE_START)` | Register a pending GPU store before it's enqueued |
| `MP_RETRIEVE_SUBMITTED` | `retrieve()`, before `publish_on_stream(MP_RETRIEVE_START)` | Register a pending GPU retrieve before it's enqueued |
| `MP_REQUEST_END` | `end_session()`, after `session_manager.remove()` | Signal that the session lifecycle is complete |

### Deferral Protocol

`end_session()` is CPU-synchronous; GPU store/retrieve callbacks (`MP_STORE_END`,
`MP_RETRIEVE_END`) fire later via CUDA host callbacks. Without coordination,
`MP_REQUEST_END` can arrive and close the root span before GPU work finishes —
producing orphaned child spans.

The fix: `MP_STORE_SUBMITTED` and `MP_RETRIEVE_SUBMITTED` are published *before*
the respective GPU work is enqueued, incrementing `_pending_store_count` and
`_pending_retrieve_count`. When `MP_REQUEST_END` arrives:

- If both counters are zero → close root immediately
- Otherwise → save the `REQUEST_END` timestamp; the last `MP_STORE_END` or
  `MP_RETRIEVE_END` to decrement its counter to zero (when the other counter is
  also zero) closes the root using that saved timestamp

Root end-time is always the `REQUEST_END` timestamp (the logical request end),
not the GPU callback timestamp.

**Why `MP_RETRIEVE_SUBMITTED` is needed**: vLLM's IPC completion event is
recorded on the CUDA stream between `MP_RETRIEVE_START` and `MP_RETRIEVE_END`.
When vLLM unblocks on that event, it can call `end_session()` before the GPU
callback for `MP_RETRIEVE_END` fires. EventBus queue becomes:
`→ MP_RETRIEVE_START → MP_REQUEST_END → MP_RETRIEVE_END`
Without `MP_RETRIEVE_SUBMITTED`, `_on_session_end` sees no in-flight work and
closes the root span before the retrieve child span ends.

## Root Span Attributes

In addition to `session_id`, the root `"request"` span carries three hit rate
attributes that are set when `MP_LOOKUP_PREFETCH_END` is processed:

| Attribute | OTel type | Value |
|-----------|-----------|-------|
| `hit_tokens` | `int` | tokens found in L1+L2 (numerator) |
| `requested_tokens` | `int` | chunk-aligned tokens submitted for lookup (denominator) |
| `hit_rate` | `float` | `hit_tokens / requested_tokens`; `0.0` when denominator is zero |

`hit_rate` is stored as a precomputed float because trace UIs (Tempo, Jaeger)
cannot derive it from two integer attributes at query time.

**Invariant:** these attributes are set at `MP_LOOKUP_PREFETCH_END` time, while
the root span is still open.  `LP_END` always precedes `MP_REQUEST_END` in the
event stream, so the root span is guaranteed to be live in the registry when the
attributes are written.

**Store-only requests** (no `lookup_prefetch_start()` call) never emit
`MP_LOOKUP_PREFETCH_END`, so the root span will not carry these attributes.

### CB path — `cb.request` span

The same three attributes appear on the `"cb.request"` root span and are set
when `CB_LOOKUP_END` is processed by `BlendTracingSubscriber`.

`CB_LOOKUP_END` carries `hit_tokens` and `requested_tokens` in its metadata,
computed at the emit site in `lmcache/v1/multiprocess/modules/blend.py`:

| Field | Value |
|-------|-------|
| `hit_tokens` | `storage_hits * chunk_size` |
| `requested_tokens` | `(num_tokens // chunk_size) * chunk_size` (chunk-aligned) |

All three `CB_LOOKUP_END` emit sites (no-fingerprint-match, no-GPU-context,
happy path) populate these fields, so `hit_rate` is always present on the
`cb.request` span.

A fourth attribute is also set on `"cb.request"` at `CB_LOOKUP_END` time:

| Attribute | OTel type | Value |
|-----------|-----------|-------|
| `prefix_hits` | `int` | chunks found via the prefix probe (not fingerprint matching) |

#### Prefix probe

`cb_lookup_pre_computed` has two lookup paths:

1. **Fingerprint path** — `BlendTokenRangeMatcher.match_sub_sequence` finds
   sub-sequence matches using polynomial rolling hashes.  Covers arbitrary
   (non-prefix) positions in the token sequence.
2. **Prefix probe** — a fallback that runs after the fingerprint path and fills
   in chunks at contiguous prefix positions not already covered by fingerprint
   results.  It calls `token_hasher.compute_chunk_hashes(token_ids)` to derive
   the same storage keys used by `cb_store_final` and `cb_store_pre_computed`,
   then creates `CBMatchResult(old_st==cur_st)` candidates for uncovered slots.
   These candidates flow through the same prefetch/poll/evict machinery as
   fingerprint results.

The prefix probe closes the gap between the MP and CB storage paths: chunks
written by `cb_store_final` (which only registers fingerprints when
`worker_id in [0, None]`) and chunks written via the MP `store()` path (which
uses block hashes incompatible with fingerprint matching) are both visible to
`cb_lookup_pre_computed` through the prefix probe.

#### Lazy registration

When `cb_lookup_pre_computed` returns results that came *entirely* from the
prefix probe (i.e. `fingerprint_results` is empty) and the calling worker is
rank 0 or the driver (`worker_id in [0, None]`), the found prefix chunks are
registered into `BlendTokenRangeMatcher` so that future lookups can find them
via the faster fingerprint path.  Registration is guarded by
`BlendTokenRangeMatcher.has_chunk(token_hash)` to prevent overwriting existing
compact-ID assignments when the range matcher already has entries for the same
token sequence.

`prefix_hits` counts the chunks found exclusively through the prefix probe
(after deduplication against fingerprint results).  When `fingerprint_results`
is non-empty and prefix candidates fill in additional positions,
`prefix_hits` reflects only the prefix-probe portion of the total
`storage_hits`.

## Request Scenarios

### Scenario 1 — Full Cache Hit

Path: `lookup_prefetch → retrieve → store`

```
CPU  ─[REQUEST_START]─[LP_START]─[LP_END]──[RETR_SUBMITTED]──[STORE_SUBMITTED]─[REQUEST_END]─►
GPU  ──────────────────────────────[RETR_START]─[vLLM_IPC]─[RETR_END]──[STORE_START]─[STORE_END]─►

root "request"  [═══════════════════════════════════════════════════════════════════════════════]
  mp.lookup_prefetch    [══════════]
  mp.retrieve                          [══════════════════]
  mp.store                                                        [══════════════════════]
```

Root closes at `REQUEST_END` (deferred until both retrieve and store complete).

---

### Scenario 2 — Cache Miss (no retrieve)

Path: `lookup_prefetch → store`, no retrieve

```
CPU  ─[REQUEST_START]─[LP_START]─[LP_END]──────────[STORE_SUBMITTED]─[REQUEST_END]─►
GPU  ───────────────────────────────────────────────────────[STORE_START]─[STORE_END]─►

root "request"  [═══════════════════════════════════════════════════════════════════]
  mp.lookup_prefetch    [══════════]
  mp.store                                                    [══════════════════════]
```

No retrieve occurred, so `mp.retrieve` is absent.

---

### Scenario 3 — Lookup Only

Path: `lookup_prefetch` only, no store

```
CPU  ─[REQUEST_START]─[LP_START]─[LP_END]─[REQUEST_END]─►

root "request"  [════════════════════════════════════════]
  mp.lookup_prefetch    [══════════]
```

Root closes immediately at `REQUEST_END`.

---

### Scenario 4 — Store Only (no lookup)

Path: `store` with no prior `lookup_prefetch_start()` call

```
CPU  ─(no REQUEST_START)──────[STORE_SUBMITTED]─[REQUEST_END]─►
GPU  ──────────────────────────────────[STORE_START]─[STORE_END]─►

root "request" (lazy, created at MP_STORE_START)
                                       [═════════════════════════]
  mp.store                             [══════════════]
```

`MP_REQUEST_START` is only emitted from `lookup_prefetch_start()`. If that path
was not taken, `_get_or_create_request_span()` is called lazily on the first child
`_on_start()`. Root start time equals `STORE_START` timestamp.

---

### Scenario 5 — REQUEST_END Races GPU Store

`end_session()` called before the GPU store callback fires.

```
CPU  ─[REQUEST_START]─[LP_START]─[LP_END]─[STORE_SUBMITTED]─[REQUEST_END]────────────────────►
GPU  ──────────────────────────────────────────────[STORE_START]──────────────[STORE_END]─────►
                                                                       ▲
                                              REQUEST_END arrives here─┘ (before STORE_END)

root "request"  [═══════════════════════════════════════════════════════════════════════════]
  mp.lookup_prefetch    [══════════]
  mp.store                                                    [═══════════════════]
                                                                                  ▲
                   STORE_SUBMITTED → count=1                                      │
                   REQUEST_END → count>0 → defer (save ts)                        │
                   STORE_END → count=0 → _close_request_span(deferred_ts) ────────────────┘
```

---

### Scenario 6 — Multiple Stores, Deferred Close

Two concurrent stores; root stays open until both complete.

```
CPU  ─[REQUEST_START]─[LP_START]─[LP_END]─[SUBMITTED×2]─[REQUEST_END]──────────────────────────────────►
GPU  ────────────────────────────────────────────────────[S1_START]─[S1_END]─[S2_START]─[S2_END]────────►

root "request"  [═══════════════════════════════════════════════════════════════════════════════════════]
  mp.lookup_prefetch    [══════════]
  mp.store (1)                                                       [══════════]
  mp.store (2)                                                                    [══════════]
                                                                                            ▲
                   count=2 at REQUEST_END → defer                                           │
                   S1_END → count=1 → still open                                            │
                   S2_END → count=0 → _close_request_span(deferred_ts) ─────────────────────────────┘
```

## Summary

| Scenario | Root opens | Root closes |
|----------|-----------|-------------|
| Full hit | `MP_REQUEST_START` | last `MP_STORE_END` / `MP_RETRIEVE_END` (stamped at `REQUEST_END` time) |
| Cache miss | `MP_REQUEST_START` | last `MP_STORE_END` (stamped at `REQUEST_END` time) |
| Lookup only | `MP_REQUEST_START` | `REQUEST_END` (immediate) |
| Store only | `MP_STORE_START` (lazy) | `REQUEST_END` (immediate) |
| REQUEST_END races store | `MP_REQUEST_START` | last `MP_STORE_END` (stamped at `REQUEST_END` time) |
| REQUEST_END races retrieve | `MP_REQUEST_START` | last `MP_RETRIEVE_END` (stamped at `REQUEST_END` time) |
| Multiple stores | `MP_REQUEST_START` | last `MP_STORE_END` (stamped at `REQUEST_END` time) |

## Implementation

| File | Change |
|------|--------|
| `lmcache/v1/mp_observability/event.py` | Add `MP_REQUEST_START`, `MP_STORE_SUBMITTED`, `MP_RETRIEVE_SUBMITTED`, `MP_REQUEST_END` |
| `lmcache/v1/multiprocess/server.py` | Emit the 4 events at `lookup_prefetch_start()`, `store()`, `retrieve()`, `end_session()` |
| `lmcache/v1/mp_observability/subscribers/tracing/mp_server.py` | Root span logic: `_pending_store_count`, `_pending_retrieve_count`, `_deferred_session_end_ts`; handlers `_on_request_start`, `_on_store_submitted`, `_on_retrieve_submitted`, `_on_session_end`; helpers `_get_or_create_request_span`, `_close_request_span` |
| `lmcache/v1/mp_observability/subscribers/tracing/span_registry.py` | `SpanRegistry`: shared dict of open spans keyed by `(session_id, span_name)` for cross-subscriber parent lookup |
| `tests/v1/mp_observability/subscribers/tracing/test_mp_server.py` | Tests for all scenarios including retrieve deferral |
| `lmcache/v1/multiprocess/modules/blend.py` | Prefix probe in `cb_lookup_pre_computed`; lazy registration; `has_chunk` on `BlendTokenRangeMatcher`; `prefix_hits` in `CB_LOOKUP_END` metadata |
| `lmcache/v1/mp_observability/subscribers/tracing/cb_server.py` | Stamp `prefix_hits` on `"cb.request"` root span from `CB_LOOKUP_END` |
| `tests/v1/multiprocess/test_blend_server_v2.py` | `has_chunk` unit tests |
| `tests/v1/mp_observability/subscribers/tracing/test_cb_server.py` | `prefix_hits` attribute tests |

---

## Extending the Span Hierarchy

### How the registry works

`MPServerTracingSubscriber` writes every open span into a shared
`SpanRegistry` while it is live:

```
registry[(session_id, "request")]       → (root_span, root_ctx)       # open: REQUEST_START → REQUEST_END
registry[(session_id, "retrieve")]      → (retrieve_span, ctx)         # open: RETRIEVE_START → RETRIEVE_END
registry[(session_id, "store")]         → (store_span, ctx)            # open: STORE_START → STORE_END
registry[(session_id, "lookup_prefetch")] → (lp_span, ctx)            # open: LP_START → LP_END
```

Any subscriber that receives the same `SpanRegistry` instance can call
`registry.get_context(session_id, "request")` (or any other name) to obtain
the OTel context needed to nest a new span.

---

### Example 1 — new span at the same level

To add an `l1.read` span nested directly under the root `"request"` span,
create a new subscriber file and register it with the shared registry.
No existing files need to change.

**`subscribers/tracing/l1.py`**:

```python
# SPDX-License-Identifier: Apache-2.0
from opentelemetry import trace

from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber
from lmcache.v1.mp_observability.subscribers.tracing.span_registry import SpanRegistry

_tracer = trace.get_tracer("lmcache_mp.l1")

class L1TracingSubscriber(EventSubscriber):
    def __init__(self, registry: SpanRegistry) -> None:
        self._registry = registry
        self._pending: dict[str, object] = {}

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {
            EventType.L1_READ_RESERVED: self._on_start,
            EventType.L1_READ_FINISHED: self._on_end,
        }

    def _on_start(self, event: Event) -> None:
        parent_ctx = self._registry.get_context(event.session_id, "request")
        span = _tracer.start_span(
            "l1.read", context=parent_ctx, start_time=int(event.timestamp * 1e9)
        )
        self._pending[event.session_id] = span

    def _on_end(self, event: Event) -> None:
        span = self._pending.pop(event.session_id, None)
        if span:
            span.end(end_time=int(event.timestamp * 1e9))
```

**`config.py`** (the only change needed):

```python
registry = SpanRegistry()
bus.register_subscriber(MPServerTracingSubscriber(registry))
bus.register_subscriber(L1TracingSubscriber(registry))   # ← add this line
```

This produces: `request → l1.read` (alongside `mp.retrieve`, `mp.store`, etc.)

---

### Example 2 — sub-span nested under an existing child span

To nest a span *inside* `mp.retrieve` (e.g. an L2 disk load that happens
during a retrieve), look up `"retrieve"` as the parent instead of `"request"`.
The `"retrieve"` entry is live in the registry from `MP_RETRIEVE_START` to
`MP_RETRIEVE_END`.

```python
    def _on_detail_start(self, event: Event) -> None:
        sid = event.session_id
        # Prefer the immediate parent; fall back to root if retrieve has ended.
        parent_ctx = (
            self._registry.get_context(sid, "retrieve")
            or self._registry.get_context(sid, "request")
        )
        span = _tracer.start_span(
            "l2.disk_load", context=parent_ctx, start_time=int(event.timestamp * 1e9)
        )
        self._pending[sid] = span
```

This produces a three-level trace: `request → mp.retrieve → l2.disk_load`.
