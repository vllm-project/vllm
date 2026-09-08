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

## Fault Tolerance Metrics

With `--enable-fault-tolerance`, each API server exposes
`vllm:engine_healthy{engine="<rank>"}` for its managed engines. The value is `1`
when the cached FT state is healthy and the client has not failed, and `0`
when the engine is unhealthy, dead, or the client has failed. Recovery back to
healthy restores the value to `1`.

The metric is collected from the API server's local state on every scrape,
including while inference is halted. It does not run GPU probes or wait for
engine RPCs. In multi-port external load-balancer deployments, scrape each rank's
API port separately; a healthy Pod can contain an unhealthy rank endpoint.
The metric is not aggregated across API processes through Prometheus shared
files. It is absent when FT is disabled or engine status is unavailable.

Routers should exclude endpoints with a value of `0`, a missing metric, or a
failed scrape. The cached state reflects detected faults, so this does not
eliminate detection delay or requests racing a failure. Use this signal for
traffic eligibility, not container liveness during recovery.

## NIXL KV Connector Metrics

--8<-- "gen:metrics-nixl"

## Model Flops Utilization (MFU) Performance Metrics

These metrics are available via `--enable-mfu-metrics`:

--8<-- "gen:metrics-mfu"

## Deprecation Policy

Note: when metrics are deprecated in version `X.Y`, they are hidden in version `X.Y+1`
but can be re-enabled using the `--show-hidden-metrics-for-version=X.Y` escape hatch,
and are then removed in version `X.Y+2`.
