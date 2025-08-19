# Perces Dashboards for vLLM Monitoring

This directory contains Perces dashboard configurations designed to monitor vLLM performance and metrics in a Kubernetes environment using the Perces Operator.

## Deployment

These dashboards are designed to be deployed using the Perces Operator in a Kubernetes environment.

1. Navigate to this directory:

   ```bash
   cd perces-dashboards
   ```

2. Apply the configuration to your cluster:

   ```bash
   kubectl apply -f . -n <namespace>
   ```

## Dashboard Descriptions

- **Query Statistics**: Tracks query performance, latency, and throughput metrics for your vLLM service.
- **Performance Statistics**: Tracks performance metrics for your vLLM service.

## Requirements

- Perces Operator installed in your cluster
- Prometheus data source configured in Perces
