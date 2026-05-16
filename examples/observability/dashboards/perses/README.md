# Perses Dashboards for vLLM Monitoring

This directory contains Perses dashboard configurations designed to monitor vLLM
performance and metrics.

## Requirements

- Perses instance (standalone or via operator)
- Prometheus data source configured in Perses
- vLLM deployment with Prometheus metrics enabled

## Dashboard Format

We provide dashboards in the **native Perses YAML format** that works across all
deployment methods:

- **Files**: `*.yaml` (native Perses dashboard specifications)
- **Format**: Pure dashboard specifications that work everywhere
- **Usage**: Works with standalone Perses, API imports, CLI, and file provisioning
- **Kubernetes**: Directly compatible with Perses Operator

## Dashboard Descriptions

- **performance_statistics.yaml**: Performance metrics with aggregated latency
  statistics
- **query_statistics.yaml**: Query performance and deployment metrics

## Deployment Options

### Direct Import to Perses

Import the dashboard specifications via Perses API or CLI:

```bash
percli apply -f performance_statistics.yaml
```

### Perses Operator (Kubernetes)

The native YAML format works directly with the Perses Operator:

```bash
kubectl apply -f performance_statistics.yaml -n <namespace>
```

### File Provisioning

Place the YAML files in a Perses provisioning folder for automatic loading.
