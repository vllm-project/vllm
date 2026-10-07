# Grafana Dashboards for vLLM Monitoring

This directory contains Grafana dashboards (as JSON) for the Prometheus metrics
exported by vLLM's `/metrics` endpoint.

## Dashboard Descriptions

| Dashboard | File | Use it to |
| --------- | ---- | --------- |
| **vLLM / Service Status** | `vllm_service_status.json` | Show the users of a service, at a glance and from across the room, whether each model is up, how fast it answers and how busy it is. One strip of large color-coded numbers per model, with their history underneath. |
| **vLLM / Overview** | `vllm_overview.json` | Check service health and latency SLOs: traffic, errors, TTFT, inter-token latency, TPOT and end-to-end latency, throughput, saturation (scheduler queue, KV cache, preemptions), caching, workload shape and the HTTP API. Includes a per-model summary. |
| **vLLM / Instances** | `vllm_instances.json` | Find the instance or data-parallel engine that misbehaves: per-instance table with drill-down, load balance, latency by instance, scheduler activity, API server process, cache configuration, LoRA adapters, sleep mode and model FLOPs utilization. |
| **vLLM / KV Cache** | `vllm_kv_cache.json` | Tell whether the KV cache is the bottleneck and whether caching pays off: KV cache usage, preemptions, prefix caching by cache tier, KV block residency, KV connector transfers (NIXL) and CPU/disk offloading. |
| **vLLM / Speculative Decoding** | `vllm_speculative_decoding.json` | Judge speculative decoding: draft acceptance rate, acceptance length, acceptance by draft position and draft token flow. |
| **Performance Statistics** | `performance_statistics.json` | Track latency and throughput. |
| **Query Statistics** | `query_statistics.json` | Track query performance, request volume and key performance indicators. |

The `vllm_*` dashboards link to each other through the *vLLM dashboards* links
in the top-right corner, keeping the time range and filters. Clicking a model in the
Overview's per-model summary filters the Overview to that model, and clicking an
instance in the Instances table focuses the drill-down on it.

## Requirements

- Grafana 10.4 or newer for the `vllm_*` dashboards (tested with 10.4, 11.6 and
  13.2)
- A Prometheus data source (Prometheus 2.x or 3.x, or a compatible backend such
  as Thanos or Mimir) that scrapes the `/metrics` endpoint of every vLLM server
- vLLM metrics enabled, which is the default (do not pass `--disable-log-stats`)

Set the **Scrape interval** of the Grafana data source to the Prometheus scrape
interval: the dashboards compute rates over `$__rate_interval`, which Grafana
derives from it.

The dashboards query vLLM's native Prometheus metrics. Metrics exported through
Ray (`RayPrometheusStatLogger`) use different names and are not covered.

## Service Status Dashboard

**vLLM / Service Status** is meant for the people who use a vLLM service rather
than the team running it, for example on a wall display. Add the `kiosk`
parameter to its URL (for example `/d/vllm-status?kiosk`) to hide the Grafana
menus.

| Tile | Meaning | Green | Orange | Red |
| ---- | ------- | ----- | ------ | --- |
| Status | *Online*, *Sleeping* (blue: all engines put to sleep) or *Down* (no server reporting) | Online | | Down |
| Response time | Median time until the first token arrives | < 1 s | 1-3 s | > 3 s |
| Output speed | Tokens generated per second for each request being answered | ≥ 20 tok/s | 10-20 tok/s | < 10 tok/s |
| Success rate | Share of requests finished without a server error | ≥ 99% | 95-99% | < 95% |
| Requests / min | Requests finished per minute | | | |

The numbers are averaged over the last 5 minutes. Adjust the color thresholds
in the panel settings to match the expectations of your users.

## Variables

The `vllm_*` dashboards share these variables (the Service Status dashboard
only shows Data source and Model):

| Variable | Description |
| -------- | ----------- |
| Data source | The Prometheus data source to query. |
| Job, Model, Instance | Scope the dashboards to Prometheus jobs, served model names (`--served-model-name`) and scrape targets. |
| Engine | Data-parallel engine index (all dashboards except the Overview). |
| TTFT SLO, TPOT SLO | Latency targets of the Overview's SLO attainment tiles. A target must be a bucket boundary of its histogram; add your own with `--custom-histogram-buckets`. |
| Filters | Ad hoc label filters, for example `namespace` or `cluster` on Kubernetes. |

Things to keep in mind:

- The HTTP and process metrics of the API server are not labelled by model, so
  the panels using them follow the Job and Instance filters only.
- Process metrics (CPU, memory, file descriptors, uptime) are not exported with
  `--api-server-count` > 1.
- Rows for optional features are collapsed and show no data unless the feature
  is enabled: `--kv-cache-metrics`, `--enable-mfu-metrics`, sleep mode, LoRA,
  speculative decoding, and `--kv-transfer-config` with the `NixlConnector`,
  `OffloadingConnector` or `SimpleCPUOffloadConnector`.

## Deployment Options

### Manual Import

1. In Grafana, open **Dashboards** > **New** > **Import**.
2. Upload a JSON file from this directory, or paste its content, and click
   **Import**.
3. The `vllm_*` dashboards use the default Prometheus data source; pick another
   one with the *Data source* variable if needed.

### HTTP API

The dashboard API expects the dashboard wrapped in a request object:

```bash
for f in vllm_*.json; do
  jq '{dashboard: ., overwrite: true}' "$f" |
    curl -X POST -H "Content-Type: application/json" -u admin:admin \
      --data @- http://localhost:3000/api/dashboards/db
done
```
