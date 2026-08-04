# Continuous Usage Metrics for MP Mode

Status: PR 1 shipped (#4098). Extends init-time telemetry
([README.md](README.md)) with runtime metrics. Receiver: InfluxDB → Grafana.

## Goals

1. **Parity**: MP emits the single-process continuous metrics
   (`ContinuousContextMessage`). *(done)*
2. **Fleet dashboards**: e.g. total KV volume per week, hit rate over time.
3. **Hybrid-model attribution**: KV volume by attention architecture
   (full / full+SWA / full+linear / DSA).
4. **Reuse insights**: idealized (infinite-storage) hit rate, chunk
   lifecycle time, reuse patterns (bursty-short vs. sustained-long).

## Architecture

Metrics are map-reduce `MetricSpec`s over EventBus events (contract and
default registry in `metric_specs.py`): drain-thread callbacks
buffer samples, a flush thread reduces and sends every
`LMCACHE_USAGE_TRACK_INTERVAL` (600 s); idle intervals are heartbeats.
Spec fields must cover the message's metric fields exactly, so
`messages.py` stays the schema source of truth. Stateful metrics (chunk
tracker) get their own subscriber.

| Metric | Event source | PR |
|---|---|---|
| retrieved (hit) tokens; stored tokens/bytes | `MP_RETRIEVE_END`, `MP_STORE_END` (lmcache-driven only) | 1 (done) |
| eviction counts | `L1_KEYS_EVICTED`, `L2_KEYS_EVICTED` | 2 |
| attention architecture | KV-cache registration (`AttnWindowDesc`; needs a new event or registry hook) | 2 |
| chunk identity stream | `MP_LOOKUP` (`chunk_hashes`; emission already subscriber-gated) | 3–4 |

The non-MP lifespan histogram is not ported (store→reuse, ~3.5-day bucket
ceiling); the chunk tracker supersedes it. MP messages reuse the shared
types + endpoints, distinguished by `deployment_mode`; the MP/non-MP
code+endpoint split (`/mp/` prefix) is a separate future PR. MP-only
additions are new message types, not new fields on shared classes.

## InfluxDB schema rules

- **Tags** (low-cardinality only): `deployment_mode`, `message_type`,
  `model_name`, `attn_arch`, `lmcache_version`. `session_id` is a
  **field**, never a tag (unbounded series growth).
- **Interval deltas**, not cumulative counters: weekly volume =
  `SUM(...) GROUP BY time(1w)`; restart-safe.
- **Histograms** stay dict fields; ingest explodes buckets into
  `le=<bound>`-tagged points (Grafana heatmap format).
- Send numerators/denominators/sample rates — never precomputed ratios.
  `sequence_number` gaps = lost intervals.

## Hybrid-architecture attribution (PR 2)

Derive `attn_arch` from `AttnWindowDesc.num_chunks_in_sw` at registration:
all `-1` → `full`; window `> 1` chunk → `full+swa`; `== 1` → `full+linear`;
`use_mla` as its own bit. Sent via re-landed `MPInstanceMessage` and
tagged on per-model counters. **Gap**: DSA is invisible in the KV layout —
needs the connector to pass HF `model_type` (PR 5).

## Chunk reuse tracker (PRs 3–4)

Deterministic hash-sample (`hash % R == 0`, `R=64`) in a bounded table:
`first_seen`, `first_reuse`, `last_access`, `reuse_count`, `was_stored`.
Hashes never leave the process; only bucketed aggregates are sent.

- **Ideal hit rate**: fraction of sampled accesses seen before; gap vs.
  actual hit rate = capacity+policy miss headroom. Horizon = tracker
  retention; restarts reset it.
- **Lifespan**: `last_access − first_reuse`, emitted at retirement
  (idle > TTL ≈ 3 days); log buckets to ~1 month. Cap-forced retirements
  counted separately.
- **Reuse pattern**: 2D histogram `reuse_count × lifespan` at retirement
  (separates multi-turn-burst / daily-sustained / shared-prefix), plus a
  1D inter-reuse-gap histogram.

Env knobs: sample denominator `R`, idle TTL, table cap.

## PR plan

| PR | Content |
|---|---|
| 1 | **(done)** map-reduce reporter: parity counters + `uptime_seconds` |
| 2 | evictions + `MPInstanceMessage` re-land + `attn_arch` + per-model tags |
| 3 | `ChunkReuseTracker` pure class + unit tests |
| 4 | tracker wiring + `ReusePatternMessage` |
| 5 | connector passes `model_type` (DSA, exact arch) |
| — | backend (parallel): JSON→Influx ingest + starter dashboard |

## Open decisions

1. Confirm tag/field split against the actual Influx ingest.
2. PR 2 registration signal: new `MP_KV_REGISTERED` EventType vs. direct
   registry hook (lean: the event).
