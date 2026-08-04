# CacheBlend V3 Observability — Design

**Status:** Proposal · **Scope:** unify CB V3 tracing across the vLLM plugin
(scheduler + worker) and the LMCache blend server into one distributed trace,
plus the metrics each side exposes.

## 1. Goal

A single CB request touches **three processes**:

```
vLLM scheduler ──CB_UNIFIED_LOOKUP──▶ LMCache blend server
vLLM worker    ──CB_RETRIEVE_V3─────▶ LMCache blend server
vLLM worker    ── model forward (FULL_RECOMP → CHECK → PARTIAL)
```

Today these are observed by **two disjoint systems**:

| side | mechanism | output |
|---|---|---|
| LMCache blend server | EventBus → `BlendTracingSubscriber` (`subscribers/tracing/cb_server.py`) | **OTel spans** (`cb.request` + children), OTLP export |
| vLLM plugin (connector/shim/attn) | `_cb_span` / `_cb_stats_emit` (`lmcache_cacheblend/connector.py`) | **ad-hoc JSONL** (`CB_PROFILE=1`) |

They never share a trace: you cannot see, in one view, that a slow request's
time went into the server-side L2 load vs the worker-side scatter vs the PARTIAL
forward. **Goal: one `cb.request` trace spanning all three processes**, with
sub-spans owned by whichever process did the work, plus aligned metrics.

## 2. The unified trace model

One trace per `request_id`. Process owner in brackets; cross-process children
linked by trace-context propagation (§5).

```
cb.request                         [scheduler — root]  request_id, model, world_size, n_prompt_tokens
│
├─ cb.schedule                     [scheduler]  the get_num_new_matched_tokens defer loop
│  ├─ cb.lookup.rpc                [scheduler]  CB_UNIFIED_LOOKUP incl. N poll re-issues; attr: n_polls
│  │  └─ cb.lookup                 [SERVER]  ← cross-process child; attr prefix_chunks
│  │     ├─ cb.fingerprint_match   [server]  n_probes, table_hits, matches (token-stride=1, any offset)
│  │     │  (cb.coordinator_match instead, when a coordinator is configured)  matches, timed_out
│  │     │  (no cb.prefix_lookup span — prefix is traced by mp.lookup_prefetch)
│  │     ├─ cb.sparse_prefetch     [server]  n_keys, l1_hits, l2_misses
│  │     │  └─ cb.l2_load          [server·IO]  chunks, bytes, ms        (coalesced L2→L1)
│  │     └─ cb.classify            [server]  found, stale, per_rank_ok
│  │        ↳ on end: stamp hit_rate / prefix_coverage_tokens / n_non_prefix_tokens on cb.request
│  └─ cb.build_meta                [scheduler]  broadcast metadata to workers
│
└─ cb.execute                      [worker]
   ├─ cb.start_load_kv             [worker]  submit + stream-wait (may fire 2×: partial→full block alloc)
   │  └─ cb.retrieve               [SERVER]  ← cross-process child
   │     └─ cb.scatter             [server·GPU]  scattered_tokens, n_prefix, n_shifted (re-RoPE'd), dropped
   │        (re-RoPE folded in — interleaved per-batch, not a separate span)
   └─ cb.model_forward             [worker]  the sliced forward
      ├─ cb.full_recomp            [worker·GPU]  layers 0..cl-1
      ├─ cb.check                  [worker·GPU]  layer cl; imp_count, recomp_ratio
      └─ cb.partial                [worker·GPU]  layers cl+1..L; dispatch=flex|unified, imp_empty
```

This is the contract both sides implement against. The server owns `cb.lookup`
and `cb.retrieve` subtrees; the plugin owns everything else.

### 2.1 V3 reuse is token-granular (#3582)

As of #3582, CB matches and scatters at **token** granularity, not vLLM-block /
chunk granularity — which is what the spans/attrs below must reflect:

- **Matching** runs at `probe_stride=1`, so the shared body is found at *any*
  token offset (`cb_unified_lookup` no longer filters non-prefix matches to a
  chunk-aligned `cur_st`). `cb.fingerprint_match` reports token-offset matches,
  not aligned chunks.
- **Scatter** writes per-token via `multi_layer_kv_transfer` with
  `slot_mapping = block_id[pos // bs] * bs + pos % bs`. The reused token range is
  written slot-by-slot, so a **partial vLLM block** holding both matched and
  recomputed tokens is written correctly with **no block-alignment trim on the
  write** — the old whole-block scatter path and block-aligned drop checks are
  gone. `cb.scatter`'s unit is `scattered_tokens` / `slot_writes`.
- **L2 storage stays chunk-granular** (256-token chunks): a non-block-aligned
  match still fetches whole chunks (`cb.l2_load` = chunks/bytes), then
  `cb.scatter` writes only the matched token sub-range. So `cb.l2_load` is
  measured in chunks while `cb.scatter` is measured in tokens/slots.
- **`cb.start_load_kv` may fire twice** (vLLM allocates the request's blocks
  partial-then-full). The first pass writes only slots inside the
  already-allocated block table — a slot-bound guard (`cur_ed > num_slots`),
  *not* a block-alignment trim; the second writes the rest. Expect two
  `cb.retrieve` children, the first with `scattered_tokens` < total.

## 3. LMCache-server side — what to expose

The server already emits `cb.request` / `cb.lookup` / `cb.retrieve` via the
EventBus. Two changes:

**(a) Finer V3 events for the lookup/retrieve subtrees.** V3 currently emits only
`CB_LOOKUP_START/END` and `CB_RETRIEVE_START/END` — too coarse for the subtree in
§2. Add paired events (CPU-sync for compute, `publish_on_stream` for GPU ops so
timing is GPU-accurate):

| new event pair | span | timing source |
|---|---|---|
| `CB_FINGERPRINT_MATCH_*` | `cb.fingerprint_match` | CPU |
| `CB_COORDINATOR_MATCH_*` | `cb.coordinator_match` (fleet directory leg; mutually exclusive with the local matcher) | CPU + IO |
| (prefix lookup) | `mp.lookup_prefetch` (reused; `prefix_chunks` attr on `cb.lookup`) | CPU |
| `CB_SPARSE_PREFETCH_*` | `cb.sparse_prefetch` (+ existing L2 prefetch span as `cb.l2_load`) | CPU + IO |
| `CB_SCATTER_*` | `cb.scatter` (re-RoPE folded in via `n_shifted`) | `publish_on_stream` (GPU) |

`BlendTracingSubscriber.SPAN_DEFS` gains the matching entries; all nest under
`cb.lookup` / `cb.retrieve` via the existing `SpanRegistry`.

**(b) Simplify the deferral logic for the V3 model.** The current root-close
deferral (`_waiting_for_store_final`, the `STORE_FINAL_SUBMITTED` bridge) is V2-only
and **inert under V3** (V3 never emits those). Under V3 the request ends at
`CB_RETRIEVE_END` (no async store-final after inference). Gate the `cb.request`
close on `_pending_gpu_ops[sid] == 0` only, and drop the V2 store-final bridge
from the V3 path. (The V2-only event handlers stay for `blend_legacy`.)

**Span attributes (server):** `request_id`, `prefix_coverage_tokens`,
`fingerprint_hits`, `storage_hits`, `stale_chunks`, `hit_tokens`,
`requested_tokens`, `hit_rate`, `prefix_hit_tokens`, `non_prefix_hit_tokens`,
`scatter_ms`, `scattered_tokens`, `slot_writes`, `partial_blocks`,
`n_shifted_tokens`, `n_prefix_tokens` (token-granular per §2.1 — not chunks).

**V3 hit rate.** `hit_rate = hit_tokens / requested_tokens`, where the numerator
counts **both reuse paths**: `hit_tokens = prefix_hit_tokens +
non_prefix_hit_tokens`. The two ranges are disjoint (the non-prefix complement
is `cur_st >= prefix coverage`), so they sum without double-counting. Both
components are also recorded individually on `cb.request` so a dashboard can
split prefix-reuse vs re-RoPE'd non-prefix reuse.

**Metrics (already present, keep):** the `lmcache_blend.*` counters
(`lookup_requests`, `lookup_hit_tokens`, `lookup_storage_hits`,
`lookup_stale_chunks`, `retrieve_requests`, `retrieve_failures`,
`chunks_evicted`, …). Note the V2-only store counters won't populate under V3 —
document, or recompute the "stored" notion from the unified path.

## 4. vLLM-plugin side — what to expose

The plugin's `_cb_span` spans (`sched.gnnmt`, `gnnmt.cb_unified_submit/poll`,
`sched.build_meta`, `slk.*`, `shim.wrapper.{prepare,fwd}`, `flex.*`,
`cb_admission_check`) already cover the §2 plugin subtree — but as **JSONL, not
OTel**. Make `_cb_span` dual-mode:

- when an OTel tracer is available → emit an **OTel span** (start/end, attributes);
- always (under `CB_PROFILE`) → keep the JSONL line (cheap local profiling).

Mapping plugin span → unified name: `sched.gnnmt`→`cb.schedule`,
`gnnmt.cb_unified_*`→`cb.lookup.rpc`, `sched.build_meta`→`cb.build_meta`,
`slk.*`→`cb.start_load_kv`, `shim.wrapper.fwd`→`cb.model_forward`,
`flex.*` + the layer hooks → `cb.full_recomp`/`cb.check`/`cb.partial`.

**Tracer source.** vLLM has its own OTel (`--otlp-traces-endpoint`) and creates a
per-request span. Prefer to **reuse vLLM's tracer** so CB spans nest under vLLM's
request span intra-process; if vLLM tracing is off, the plugin owns a tracer
pointed at the same OTLP endpoint as the LMCache server. Gate on a single
`CB_TRACING=1` (or reuse `--enable-tracing` semantics) so it's off by default.

## 5. Unification — linking the three processes

The blocker (from the surface map): **the RPC envelope carries no trace-context**
— `IPCCacheServerKey` and `CBUnifiedLookupResult` have no `traceparent` field. Two
ways to bridge:

**Option A — propagate W3C trace-context through the RPC (recommended).**
Add an optional `trace_context: str | None` (W3C `traceparent`) to the CB RPC
payloads (the lookup key + the retrieve args). The scheduler/worker **inject**
the current span's context; the server **extracts** it and starts `cb.lookup` /
`cb.retrieve` as remote children of it. Result: a *true* parent→child distributed
trace across processes. Cost: one optional protocol field (backward-compatible —
`None` when tracing off), an `inject`/`extract` at the two RPC boundaries.

**Option B — deterministic trace-id from `request_id` (zero protocol change).**
Both sides derive a 128-bit trace-id `= hash(request_id)` and tag every span with
it (+ `request_id` attribute). Backends group by trace-id, so the spans land in
one trace — but there are **no cross-process parent links** (sibling spans, not
nested). Use if the protocol field is undesirable short-term.

**Recommendation:** Option A. The field is tiny, optional, and gives real
parent/child causality (e.g. "the 90 ms gnnmt was 50 ms server L2-load + 40 ms
poll-wait"). Keep `request_id` as a span attribute regardless, so Option B is a
trivial fallback. The `SpanRegistry` already handles intra-process nesting on
each side; Option A only adds the *cross*-process edge.

## 6. Phasing

1. **Plugin OTel** — make `_cb_span` dual-mode (OTel + JSONL); reuse vLLM's tracer;
   gate with `CB_TRACING`. (plugin repo)
2. **V3 server sub-spans** — add the §3(a) events + `SPAN_DEFS`; simplify the
   §3(b) V3 deferral. (LMCache) — **DONE**: `cb.fingerprint_match` /
   `cb.sparse_prefetch` nest under `cb.lookup` (prefix lookup reuses
   `mp.lookup_prefetch`; `prefix_chunks` is a `cb.lookup` attr); `cb.scatter`
   (re-RoPE folded) nests under `cb.retrieve`; `hit_rate` = prefix + non-prefix.
   `cb.l2_load` GB/s is already covered by the existing `L2ThroughputSubscriber`
   (`L2_LOAD_TASK_*`), correlated by request; nesting that span under
   `cb.sparse_prefetch` is a cross-subsystem follow-up.
3. **Cross-process link** — add the optional `trace_context` RPC field; inject on
   the plugin side, extract on the server side. (both repos, in lockstep)
4. **Dashboards** — one trace view + the `lmcache_blend.*` / plugin latency metrics
   aligned on `request_id`.

Each phase is independently useful (1 and 2 give per-process traces; 3 unifies).

## 7. Open questions

- Reuse vLLM's tracer/provider, or a CB-owned one? (affects nesting under vLLM's
  request span vs a standalone `cb.request` root)
- Is the protocol field (Option A) acceptable for upstream, or start with the
  deterministic-trace-id fallback (Option B)?
- Sampling: per-request tracing is expensive at scale — head sampling at the
  scheduler (propagated via the same `trace_context`) so a sampled-out request is
  cheap on all three processes.
