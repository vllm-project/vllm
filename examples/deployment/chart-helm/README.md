# Helm Charts

This directory contains a Helm chart for deploying the vllm application. The chart includes configurations for deployment, autoscaling, resource management, and more.

## Scheduling

Set `affinity` to configure the Pod affinity rules directly. When it is non-empty,
it is rendered as-is and takes precedence over `gpuModels`. `gpuModels` remains
available as a legacy fallback: when `affinity` is empty and both NVIDIA GPU
requests and limits are greater than zero, it creates a required node affinity
rule for `nvidia.com/gpu.product`. Both settings default to empty.

```yaml
affinity:
  podAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector:
          matchLabels:
            app: model-cache
        topologyKey: kubernetes.io/hostname
```

GPU-requesting deployments (where both NVIDIA GPU requests and limits are
greater than zero) continue to set `runtimeClassName: nvidia` independently of
these affinity settings.

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

## Running Tests

This chart includes unit tests using [helm-unittest](https://github.com/helm-unittest/helm-unittest). Install the plugin and run tests:

```bash
# Install plugin
helm plugin install https://github.com/helm-unittest/helm-unittest

# Run tests
helm unittest .
```
