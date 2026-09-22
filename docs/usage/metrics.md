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

--8<-- "gen:metrics-general"

## Speculative Decoding Metrics

--8<-- "gen:metrics-spec-decode"

## NIXL KV Connector Metrics

--8<-- "gen:metrics-nixl"

## Model Flops Utilization (MFU) Performance Metrics

These metrics are available via `--enable-mfu-metrics`:

--8<-- "gen:metrics-mfu"

## Priority Scheduling Metrics

When the server is started with `--scheduling-policy priority`, additional
priority-aware metrics become available. Under the default `fcfs` policy these
behaviors are inactive and the metrics below are unchanged.

### The `priority` label

In priority mode, every finished-request metric (for example
`vllm:e2e_request_latency_seconds`, `vllm:time_to_first_token_seconds`,
`vllm:request_queue_time_seconds`,
`vllm:request_success`, and the other per-request histograms) gains a
`priority` label so latency and throughput can be broken down per priority
tier.

Request priorities are arbitrary integers (lower value means higher priority).
To keep Prometheus cardinality bounded, the label value is **bucketed** rather
than emitted raw. The three conventional values are preserved exactly because
they carry semantic meaning:

| Priority value | `priority` label | Meaning           |
|----------------|------------------|-------------------|
| less than -1   | `<-1`            | Above high        |
| -1             | `-1`             | High priority     |
| 0              | `0`              | Default priority  |
| 1              | `1`              | Low priority      |
| greater than 1 | `>1`             | Below low         |

This caps the `priority` label to five values regardless of how many distinct
priorities clients submit, avoiding a cardinality explosion (for example when
deadline-style integer priorities are used).

### Additional metrics

- `vllm:request_priority` (Histogram) - Distribution of finished-request
  priorities using the raw (un-bucketed) priority numbers. Only registered
  under priority scheduling.
- `vllm:scheduler_policy_info` (Gauge) - Enum-style indicator of the active
  scheduler policy: one series per known policy (`fcfs`, `priority`), with the
  active policy set to `1` and the rest to `0`. Always exposed.

## Deprecation Policy

Note: when metrics are deprecated in version `X.Y`, they are hidden in version `X.Y+1`
but can be re-enabled using the `--show-hidden-metrics-for-version=X.Y` escape hatch,
and are then removed in version `X.Y+2`.
