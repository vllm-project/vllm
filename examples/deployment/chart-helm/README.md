# Helm Charts

This directory contains a Helm chart for deploying the vllm application. The chart includes configurations for deployment, autoscaling, resource management, and more.

## Files

- Chart.yaml: Defines the chart metadata including name, version, and maintainers.
- ct.yaml: Configuration for chart testing.
- lintconf.yaml: Linting rules for YAML files.
- values.schema.json: JSON schema for validating values.yaml.
- values.yaml: Default values for the Helm chart.
- templates/_helpers.tpl: Helper templates for defining common configurations.
- templates/configmap.yaml: Template for creating ConfigMaps.
- templates/custom-objects.yaml: Template for custom Kubernetes objects.
- templates/deployment.yaml: Template for creating Deployments.
- templates/hpa.yaml: Template for Horizontal Pod Autoscaler.
- templates/job.yaml: Template for Kubernetes Jobs.
- templates/poddisruptionbudget.yaml: Template for Pod Disruption Budget.
- templates/pvc.yaml: Template for Persistent Volume Claims.
- templates/secrets.yaml: Template for Kubernetes Secrets.
- templates/service.yaml: Template for creating Services.
- templates/servicemonitor.yaml: Optional metrics collection and filtering.

## Metrics collection

Set `serviceMonitor.enabled: true` to create a Prometheus Operator
`ServiceMonitor`. Install the ServiceMonitor CRD first, and configure
`serviceMonitor.additionalLabels` to match your Prometheus
`serviceMonitorSelector`. The monitor selects this release's Service in the
same namespace. It is disabled by default.

For example, keep only request counts and end-to-end latency metrics:

```yaml
serviceMonitor:
  enabled: true
  additionalLabels:
    release: prometheus
  port: service-port
  path: /metrics
  interval: 30s
  metricFilter:
    action: keep
    regex: 'vllm:(request_success_total|e2e_request_latency_seconds_(bucket|sum|count))'
```

Use `action: drop` with `regex: '(python|process)_.*'` to exclude Python and
process metrics instead. An empty regex disables filtering. Expressions use
Prometheus RE2 syntax and match the entire metric name; include `_bucket`,
`_sum`, and `_count` explicitly when selecting histogram families.

Filtering uses the ServiceMonitor's
[`metricRelabelings`](https://prometheus-operator.dev/docs/api-reference/api/#endpoint)
before ingestion into Prometheus. vLLM still generates and
serves the full `/metrics` payload. Choose metric names appropriate to your
vLLM version rather than relying on a built-in allowlist.

## Running Tests

This chart includes unit tests using [helm-unittest](https://github.com/helm-unittest/helm-unittest). Install the plugin and run tests:

```bash
# Install plugin
helm plugin install https://github.com/helm-unittest/helm-unittest

# Run tests
helm unittest .
```
