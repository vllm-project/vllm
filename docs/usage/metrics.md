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

## SimpleCPU KV Offload Metrics

When `SimpleCPUOffloadConnector` is enabled, vLLM exposes KV transfer metrics
through the existing `vllm:kv_offload_*` metric family with
`transfer_type="GPU_to_CPU"` for stores into CPU memory and
`transfer_type="CPU_to_GPU"` for loads back into GPU memory.

The connector also reports CPU pool state:

| Metric Name | Type | Description |
|-------------|------|-------------|
| `vllm:simple_cpu_offload_total_blocks` | Gauge | Total usable CPU KV cache blocks managed by `SimpleCPUOffloadConnector`. |
| `vllm:simple_cpu_offload_free_blocks` | Gauge | Free usable CPU KV cache blocks managed by `SimpleCPUOffloadConnector`. |
| `vllm:simple_cpu_offload_used_blocks` | Gauge | Used usable CPU KV cache blocks managed by `SimpleCPUOffloadConnector`. |
| `vllm:simple_cpu_offload_usage_perc` | Gauge | CPU KV cache usage for `SimpleCPUOffloadConnector`; `1` means 100 percent usage. |
| `vllm:simple_cpu_offload_pending_loads` | Gauge | Requests with pending CPU-to-GPU loads. |
| `vllm:simple_cpu_offload_pending_stores` | Gauge | Store events pending worker completion. |

`CPU_to_GPU` transfer samples appear only when the workload forces replay after
GPU cache eviction. If the GPU KV cache can hold the workload, only
`GPU_to_CPU` stores may be observed.

## Model Flops Utilization (MFU) Performance Metrics

These metrics are available via `--enable-mfu-metrics`:

--8<-- "docs/generated/metrics/perf.inc.md"

## Deprecation Policy

Note: when metrics are deprecated in version `X.Y`, they are hidden in version `X.Y+1`
but can be re-enabled using the `--show-hidden-metrics-for-version=X.Y` escape hatch,
and are then removed in version `X.Y+2`.
