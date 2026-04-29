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

When NixlConnector logs a line such as `KV Transfer metrics: Num successful
transfers=4, Avg xfer time (ms)=1.381, ...`, the summary is computed from the
combined pool of rank-level transfer observations collected during the logging
interval. In tensor-parallel deployments, each TP rank records its own NIXL
transfer telemetry and the observations from all ranks are concatenated before
averages, percentiles, and throughput are computed.

Interpret the log fields as follows:

- `Num successful transfers` is the total count of successful rank-level
  transfers across all ranks.
- `Avg xfer time (ms)`, `P90 xfer time (ms)`, `Avg post time (ms)`, and
  `P90 post time (ms)` are computed over the combined distribution of
  rank-level timings.
- `Avg MB per transfer` is the average transferred size of an individual
  rank-level transfer, not the total bytes moved by one engine-level KV cache
  operation.
- `Throughput (MB/s)` is total transferred MB divided by the sum of transfer
  durations across the combined rank-level observations. This is an average
  rank-level transfer rate, not aggregate system throughput over wall-clock
  time.
- `Avg number of descriptors` is averaged over individual rank-level
  transfers.

## Model Flops Utilization (MFU) Performance Metrics

These metrics are available via `--enable-mfu-metrics`:

--8<-- "docs/generated/metrics/perf.inc.md"

## Deprecation Policy

Note: when metrics are deprecated in version `X.Y`, they are hidden in version `X.Y+1`
but can be re-enabled using the `--show-hidden-metrics-for-version=X.Y` escape hatch,
and are then removed in version `X.Y+2`.
