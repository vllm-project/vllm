# vLLM Kubernetes Monitoring Stack

Production monitoring for a multi-replica vLLM deployment (e.g. Qwen3-235B on 40 x 4-GPU H100 pods).

## Prerequisites

- Kubernetes cluster with GPU nodes
- [Prometheus Operator](https://github.com/prometheus-operator/prometheus-operator) installed (e.g. via `kube-prometheus-stack` Helm chart)
- Grafana available (typically bundled with `kube-prometheus-stack`)
- `kubectl` and `kustomize` CLI tools

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Kubernetes Cluster                                              │
│                                                                  │
│  ┌──────────┐ ┌──────────┐     ┌──────────┐                     │
│  │ vllm-0   │ │ vllm-1   │ ... │ vllm-39  │  (40 replicas)     │
│  │ 4x H100  │ │ 4x H100  │     │ 4x H100  │                    │
│  │ :8000    ─┼─┼──:8000  ─┼─────┼──:8000   │                    │
│  └──┬───┬───┘ └──┬───┬───┘     └──┬───┬───┘                     │
│     │   │        │   │             │   │                         │
│  /metrics │   /metrics │        /metrics │                       │
│     │  OTel│      │  OTel│          │  OTel│                     │
│     │   │        │   │             │   │                         │
│     ▼   │        ▼   │             ▼   │                         │
│  ┌──────┴────────────┴─────────────────┴───┐                     │
│  │        ServiceMonitor (Prometheus)       │                     │
│  └──────────────┬──────────────────────────┘                     │
│                 ▼                                                 │
│  ┌──────────────────────┐    ┌──────────────────────┐            │
│  │  Prometheus          │───▶│  Grafana Dashboard    │            │
│  │  (PrometheusRule     │    │  (fleet overview +    │            │
│  │   alerts)            │    │   per-pod drill-down) │            │
│  └──────────────────────┘    └──────────────────────┘            │
│                                                                  │
│  ┌──────────────────────┐                                        │
│  │  OTel Collector      │◀── traces from all 40 pods             │
│  │  (tail_sampling)     │                                        │
│  │  Keeps only slow     │──▶ stdout logs / Jaeger / Tempo        │
│  │  requests            │                                        │
│  └──────────────────────┘                                        │
└──────────────────────────────────────────────────────────────────┘
```

## What You Get

### 1. Prometheus Metrics (aggregate + per-pod)

| Metric | Description |
|--------|-------------|
| `vllm:e2e_request_latency_seconds` | End-to-end request latency histogram |
| `vllm:time_to_first_token_seconds` | Time to first token (TTFT) histogram |
| `vllm:inter_token_latency_seconds` | Inter-token latency / TPOT histogram |
| `vllm:num_requests_running` | Requests currently executing (gauge) |
| `vllm:num_requests_waiting` | Requests queued (gauge) |
| `vllm:kv_cache_usage_perc` | KV cache memory utilization |
| `vllm:prompt_tokens` / `vllm:generation_tokens` | Token throughput counters |
| `vllm:request_queue_time_seconds` | Queue wait time histogram |
| `vllm:request_prefill_time_seconds` | Prefill phase duration histogram |
| `vllm:request_decode_time_seconds` | Decode phase duration histogram |
| `vllm:num_preemptions` | Preemption count |
| `vllm:prefix_cache_hits` / `vllm:prefix_cache_queries` | Prefix cache effectiveness |

### 2. Alerts (PrometheusRule)

| Alert | Condition | Severity |
|-------|-----------|----------|
| `VllmE2ELatencyP99TooHigh` | p99 E2E > 2 min | warning |
| `VllmTTFTP99TooHigh` | p99 TTFT > 30 s | warning |
| `VllmITLP99TooHigh` | p99 ITL > 1 s | warning |
| `VllmSlowRequestsPresent` | Requests exceeding 120 s detected | info |
| `VllmKVCacheNearFull` | KV cache > 95% for 5 min | warning |
| `VllmHighQueueDepth` | > 50 waiting requests per pod | warning |
| `VllmPreemptionSpike` | > 1 preemption/s sustained | warning |
| `VllmReplicaDown` | < 40 replicas reporting | critical |

### 3. Slow-Request Tracing (OpenTelemetry)

The OTel Collector uses `tail_sampling` to capture only requests matching:

- E2E duration > 120 s (the "2-minute" threshold)
- TTFT > 10 s (configurable, adjust to your p99 threshold)
- Decode time > 60 s (configurable, catches long-decode outliers)

Each captured trace includes:
- `gen_ai.request.id` - unique request ID
- `gen_ai.latency.time_to_first_token` - TTFT value
- `gen_ai.latency.e2e` - end-to-end latency
- `gen_ai.usage.prompt_tokens` / `gen_ai.usage.completion_tokens` - token counts
- `gen_ai.request.temperature`, `gen_ai.request.top_p`, etc.

### 4. Grafana Dashboard

Fleet-level overview with six rows:
1. **Fleet Overview** - stat panels (running/waiting requests, req/s, tok/s, KV cache, replicas up)
2. **Latency (Aggregate)** - E2E, TTFT, ITL percentile time series
3. **Throughput & Queue** - token throughput, request rate by finish reason, phase latency
4. **Per-Pod Breakdown** - running/waiting/KV cache per replica
5. **Request Characteristics** - prompt/generation length heatmaps, prefix cache hit rate
6. **Engine Internals** - preemption rate, batch size heatmap, per-pod p99 latency

## Deployment

### Step 1: Create the namespace

```bash
kubectl create namespace vllm
```

### Step 2: Deploy the vLLM stack and monitoring

```bash
# Deploy vLLM + ServiceMonitor + PrometheusRule + OTel Collector
kubectl apply -k examples/online_serving/prometheus_grafana/k8s-monitoring/
```

### Step 3: Import the Grafana dashboard

**Option A: ConfigMap sidecar (automatic)**

If your Grafana is deployed via `kube-prometheus-stack` with the dashboard sidecar enabled:

```bash
# Create a ConfigMap with the dashboard JSON in Grafana's namespace
kubectl -n monitoring create configmap vllm-fleet-dashboard \
  --from-file=vllm-fleet-dashboard.json=examples/online_serving/prometheus_grafana/k8s-monitoring/grafana-dashboard.json

# Label it so the sidecar picks it up
kubectl -n monitoring label configmap vllm-fleet-dashboard grafana_dashboard=1
```

**Option B: Manual import**

1. Open Grafana UI
2. Go to Dashboards > Import
3. Upload `grafana-dashboard.json`
4. Select your Prometheus datasource

### Step 4: Verify

```bash
# Check vLLM pods are running
kubectl -n vllm get pods -l app=vllm

# Check metrics are being scraped (look for vllm targets)
kubectl -n monitoring port-forward svc/prometheus-operated 9090 &
open http://localhost:9090/targets

# Check OTel Collector is running
kubectl -n vllm logs -l app=otel-collector --tail=50

# Verify a pod's metrics endpoint
kubectl -n vllm port-forward deploy/vllm-qwen3-235b 8000 &
curl http://localhost:8000/metrics | head -50
```

## Customization

### Adjust alert thresholds

Edit `prometheusrule.yaml` and modify the threshold values:

```yaml
# Example: change E2E threshold to 3 minutes
- alert: VllmE2ELatencyP99TooHigh
  expr: |
    histogram_quantile(0.99, ...) > 180   # was 120
```

### Adjust OTel slow-request thresholds

Edit `otel-collector.yaml` and modify the `tail_sampling` policies:

```yaml
# Example: capture requests with TTFT > 20s instead of 10s
- name: high-ttft
  type: numeric_attribute
  numeric_attribute:
    key: gen_ai.latency.time_to_first_token
    min_value: 20    # was 10
```

### Export traces to Jaeger or Tempo

Uncomment the OTLP exporter in `otel-collector.yaml`:

```yaml
exporters:
  otlp/jaeger:
    endpoint: jaeger-collector.observability.svc.cluster.local:4317
    tls:
      insecure: true

service:
  pipelines:
    traces:
      exporters: [debug, otlp/jaeger]
```

### View slow-request traces from logs

```bash
# Stream OTel Collector logs to see slow-request traces
kubectl -n vllm logs -f -l app=otel-collector | grep -A 20 "gen_ai.request.id"
```

### Adapt labels to your deployment

The manifests assume these labels on your vLLM Service/Pods:
- `app: vllm`
- `model: qwen3-235b`

If your labels differ, update:
- `servicemonitor.yaml` → `spec.selector.matchLabels`
- `prometheusrule.yaml` → alert expressions (the `job` label)
- `vllm-deployment.yaml` → pod/service labels

## Useful PromQL Queries

```promql
# Fleet-wide request throughput
sum(rate(vllm:request_success[5m]))

# Per-pod E2E p99
histogram_quantile(0.99, sum by(le, pod) (rate(vllm:e2e_request_latency_seconds_bucket[5m])))

# Fraction of requests exceeding 2 min
1 - (
  sum(rate(vllm:e2e_request_latency_seconds_bucket{le="120"}[5m]))
  /
  sum(rate(vllm:e2e_request_latency_seconds_count[5m]))
)

# Fleet-wide generation throughput (tokens/s)
sum(rate(vllm:generation_tokens[5m]))

# Prefix cache hit rate
sum(rate(vllm:prefix_cache_hits[5m])) / sum(rate(vllm:prefix_cache_queries[5m]))

# Top 5 busiest pods by running requests
topk(5, vllm:num_requests_running)
```
