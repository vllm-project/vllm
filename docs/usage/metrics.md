# Production Metrics

vLLM exposes a number of metrics that can be used to monitor the health of the
system. These metrics are exposed via the `/metrics` endpoint on the vLLM
OpenAI compatible API server.

You can start the server using Python, or using [Docker](../deployment/docker.md):

```bash
vllm serve unsloth/Llama-3.2-1B-Instruct
```

Then query the endpoint to get the latest metrics from the server:

??? console "Output"

    ```console
    $ curl http://0.0.0.0:8000/metrics

    # HELP vllm:iteration_tokens_total Histogram of number of tokens per engine_step.
    # TYPE vllm:iteration_tokens_total histogram
    vllm:iteration_tokens_total_sum{model_name="unsloth/Llama-3.2-1B-Instruct"} 0.0
    vllm:iteration_tokens_total_bucket{le="1.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="8.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="16.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="32.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="64.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="128.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="256.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="512.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    ...
    ```

The following metrics are exposed:

## General Metrics

--8<-- "docs/generated/metrics/general.inc.md"

## Speculative Decoding Metrics

--8<-- "docs/generated/metrics/spec_decode.inc.md"

## NIXL KV Connector Metrics

--8<-- "docs/generated/metrics/nixl_connector.inc.md"

### NIXL metrics aggregation semantics

NIXL connector metrics are emitted from transfer events on worker processes and
are then aggregated before they are exposed on `/metrics`.

- **Aggregation scope:** For each scheduler update, worker-level NIXL transfer
  stats are merged into a single engine-level payload.
- **Histograms:** Every transfer observation is recorded; histogram buckets are
  cumulative by definition in Prometheus.
- **Counters:** Failure counters (`*_failed_*`, `*_kv_expired_reqs`) are
  increment-only and accumulate over the process lifetime.
- **Metric labels:** NIXL metrics are tracked per engine label set. To compute a
  service-wide value, sum across engine label values in your PromQL query.
- **Time window behavior:** The in-memory transfer stats buffers used for
  periodic logging are reset after they are consumed, but Prometheus counters
  and histograms remain cumulative across scrapes.

Example PromQL (sum across all engines):

```promql
sum by (model_name) (rate(vllm:nixl_num_failed_transfers[5m]))
```

## Model Flops Utilization (MFU) Performance Metrics

These metrics are available via `--enable-mfu-metrics`:

--8<-- "docs/generated/metrics/perf.inc.md"

## Deprecation Policy

Note: when metrics are deprecated in version `X.Y`, they are hidden in version `X.Y+1`
but can be re-enabled using the `--show-hidden-metrics-for-version=X.Y` escape hatch,
and are then removed in version `X.Y+2`.
