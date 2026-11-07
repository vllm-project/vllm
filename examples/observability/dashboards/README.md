# Monitoring Dashboards

This directory contains monitoring dashboard configurations for vLLM, providing
comprehensive observability for your vLLM deployments.

## Dashboard Platforms

We provide dashboards for two popular observability platforms:

- **[Grafana](https://grafana.com)**
- **[Perses](https://perses.dev)**

## Dashboard Format Approach

All dashboards are provided in **native formats** that work across different
deployment methods:

### Grafana (JSON)

- ✅ Works with any Grafana instance (cloud, self-hosted, Docker)
- ✅ Direct import via Grafana UI or API
- ✅ Can be wrapped in Kubernetes operators when needed
- ✅ No vendor lock-in or deployment dependencies

### Perses (YAML)

- ✅ Works with standalone Perses instances
- ✅ Compatible with Perses API and CLI
- ✅ Supports Dashboard-as-Code workflows
- ✅ Can be wrapped in Kubernetes operators when needed

## Dashboard Contents

### Grafana

| Dashboard | Description |
| --------- | ----------- |
| **vLLM / Service Status** | Simple status board for end users: availability, response time, output speed, success rate and load per model, readable from a distance |
| **vLLM / Overview** | Service health and latency SLOs: traffic, errors, latency, throughput, saturation and caching |
| **vLLM / Instances** | Per-instance and per-engine drill-down: load balance, latency outliers, scheduler, process and configuration |
| **vLLM / KV Cache** | KV cache pressure, prefix caching, KV block residency, and KV transfer and offloading |
| **vLLM / Speculative Decoding** | Draft acceptance rate, acceptance length and draft token flow |
| **Performance Statistics** | Tracks latency, throughput, and performance metrics |
| **Query Statistics** | Monitors request volume, query performance, and KPIs |

### Perses

| Dashboard | Description |
| --------- | ----------- |
| **Performance Statistics** | Tracks latency, throughput, and performance metrics |
| **Query Statistics** | Monitors request volume, query performance, and KPIs |

## Quick Start

First, navigate to this example's directory:

```bash
cd examples/observability/dashboards
```

### Grafana

Import the JSON directly into the Grafana UI, or use the API:

```bash
jq '{dashboard: ., overwrite: true}' grafana/vllm_overview.json |
  curl -X POST http://grafana/api/dashboards/db \
    -H "Content-Type: application/json" \
    --data @-
```

### Perses

Import via the Perses CLI:

```bash
percli apply -f perses/performance_statistics.yaml
```

## Requirements

- **Prometheus** metrics from your vLLM deployment
- **Data source** configured in your monitoring platform
- **vLLM metrics** enabled and accessible

## Platform-Specific Documentation

For detailed deployment instructions and platform-specific options, see:

- **[Grafana Documentation](grafana)** - JSON dashboards, operator usage, manual import
- **[Perses Documentation](perses)** - YAML specs, CLI usage, operator wrapping

## Contributing

When adding new dashboards, please:

1. Provide native formats (JSON for Grafana, YAML specs for Perses)
2. Update platform-specific README files
3. Ensure dashboards work across deployment methods
4. Test with the latest platform versions
