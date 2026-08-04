# LMCache Observability Example

Minimal example showing per-request OTel tracing and metrics for LMCache + vLLM,
visualized in Grafana.

## Stack

```
LMCache / vLLM
  └─ OTLP gRPC → OTel Collector (:4320)
                   ├─ traces  → Tempo (:3200)
                   └─ metrics → Prometheus (:9091)
                                  └─ Grafana (:3000)
```

## Step 1 — Start the observability stack

```bash
cd examples/observability
docker compose up -d
```

## Step 2 — Start LMCache + vLLM

```bash
MODEL=/your/model/path bash start-server.sh
```
 

## Step 3 — Send requests to populate traces

```bash
# Run a short long-doc-qa benchmark: first query is a miss, subsequent
# queries against the same document are cache hits.
lmcache bench engine \
  --engine-url http://localhost:8100 \
  --workload long-doc-qa \
  --kv-cache-volume 1 \
  --ldqa-query-per-document 10
```

## Step 4 — Visualize in Grafana

Open **http://localhost:3000** → **Explore** → datasource **Tempo**.

```
# All request root spans
{ name = "request" }

# Filter to a specific session
{ name = "request" && span.session_id = "<request_id>" }

# Only cache-hit requests (had a retrieve)
{ name = "request" } >> { name = "mp.retrieve" }

# Requests with less than 50 % cache hit rate
{ name = "request" && span.hit_rate < 0.5 }

# Full cache hits only
{ name = "request" && span.hit_rate = 1.0 }

# Complete misses (lookup ran but nothing was cached)
{ name = "request" && span.requested_tokens > 0 && span.hit_tokens = 0 }
```

Click any trace to open the waterfall. Each root `request` span carries three
per-request cache hit rate attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `hit_tokens` | int | tokens served from L1+L2 cache |
| `requested_tokens` | int | total chunk-aligned tokens submitted for lookup |
| `hit_rate` | float | `hit_tokens / requested_tokens` (0.0 on a total miss) |

```
request  [══════════════════════════════════════]  hit_rate=0.75
  mp.lookup_prefetch  [════]
  mp.retrieve               [════════]
  mp.store                            [══════]
```

Store-only requests (no lookup phase) do not carry these attributes.

The pre-provisioned **LMCache** dashboard under **Dashboards** shows cache hit
rate, StorageManager read/write rates, and the live trace panel. The collapsed
**CacheBlend** row adds blend-server panels (see below).

## CacheBlend (blend server) traces

When LMCache runs the **blend** engine (`lmcache server --engine-type blend`),
CacheBlend V3 emits its own span tree to Tempo alongside the standard spans.
Expand the collapsed **CacheBlend** row on the dashboard, or query Tempo:

```
# All CacheBlend request traces
{ name = "cb.request" }

# Requests that actually blended non-prefix (shifted) KV
{ name = "cb.request" && span.non_prefix_hit_tokens > 0 }

# The token-scatter GPU step
{ name = "cb.scatter" }
```

Click a `cb.request` row to open the waterfall:

```
cb.request
  cb.lookup                (attr prefix_chunks; prefix timing is in mp.lookup_prefetch)
    cb.fingerprint_match   match probe hashes vs stored fingerprints
    cb.sparse_prefetch     non-prefix (shifted) chunks, sparse L2->L1
                           (emitted only on an actual L2 load; carries l2_keys)
  cb.retrieve
    cb.scatter             L1 -> paged KV per-token slot-scatter + re-RoPE
  cb.store_pre_computed
  cb.store_final
```

The root `cb.request` span carries the V3 hit-rate breakdown
(`hit_rate = prefix + non-prefix`):

| Attribute | Type | Description |
|-----------|------|-------------|
| `prefix_hit_tokens` | int | tokens reused from the prefix (L1+L2) |
| `non_prefix_hit_tokens` | int | tokens reused from sparse non-prefix chunks |
| `hit_tokens` | int | `prefix_hit_tokens + non_prefix_hit_tokens` |
| `requested_tokens` | int | total chunk-aligned tokens submitted |
| `hit_rate` | float | `hit_tokens / requested_tokens` |
| `prefix_hit_rate` | float | `prefix_hit_tokens / requested_tokens` |
| `non_prefix_hit_rate` | float | `non_prefix_hit_tokens / requested_tokens` (sums to `hit_rate`) |

The **CacheBlend Hit Rate & Chunks** panel overlays the overall token hit rate
(Prometheus) with the per-request prefix/non-prefix breakdown via
[TraceQL metrics](https://grafana.com/docs/tempo/latest/metrics-from-traces/),
served by Tempo's `local-blocks` metrics generator (enabled in `tempo.yml`):

```
# prefix vs non-prefix hit rate over time
{ name = "cb.request" } | avg_over_time(span.prefix_hit_rate)
{ name = "cb.request" } | avg_over_time(span.non_prefix_hit_rate)
```

## Files

```
docker-compose.yml          — 4-service stack (collector, tempo, prometheus, grafana)
otel-collector.yml          — OTLP receiver → Tempo + Prometheus fan-out
tempo.yml                   — local trace storage + local-blocks TraceQL metrics
prometheus.yml              — scrapes lmcache metrics from collector
grafana/provisioning/       — auto-provisioned datasources + dashboard
start-server.sh             — launches LMCache server + vLLM with OTLP enabled
```
