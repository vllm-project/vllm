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

## Model Flops Utilization (MFU) Performance Metrics

These metrics are available via `--enable-mfu-metrics`:

--8<-- "docs/generated/metrics/perf.inc.md"

## SimpleCPU Offload Metrics

These metrics are available when `--kv-transfer-config` selects `SimpleCPUOffloadConnector`, e.g.:

```bash
vllm serve <model> \
  --kv-transfer-config '{
    "kv_connector": "SimpleCPUOffloadConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "cpu_bytes_to_use": 1000000000
    }
  }'
```

### Transfer Metrics

SimpleCPU offload emits the same flat transfer metrics as the generic `OffloadingConnector`. Direction is CPU pinned memory → GPU HBM for load, GPU HBM → CPU pinned memory for store.

| Metric | Type | Description |
| --- | --- | --- |
| `vllm:kv_offload_load_bytes` | Counter | Total bytes loaded from CPU to GPU. |
| `vllm:kv_offload_load_time` | Counter | Total load time, in seconds. DMA execution time only; excludes host preparation and the GPU compute barrier wait. |
| `vllm:kv_offload_load_size` | Histogram | Size of each load operation, in bytes. |
| `vllm:kv_offload_store_bytes` | Counter | Total bytes stored from GPU to CPU. |
| `vllm:kv_offload_store_time` | Counter | Total store time, in seconds. DMA execution time only; excludes host preparation and the GPU compute barrier wait. |
| `vllm:kv_offload_store_size` | Histogram | Size of each store operation, in bytes. |

The deprecated `vllm:kv_offload_total_bytes`/`vllm:kv_offload_total_time`/`vllm:kv_offload_size` series (labeled with `transfer_type="CPU_to_GPU"` for load and `"GPU_to_CPU"` for store) are still mirrored during the migration to the flat names above. The flat names are canonical; the labeled series is compatibility-only.

!!! note
    `CPU_to_GPU`/load samples only appear once previously-stored KV blocks are actually reloaded (a GPU eviction followed by a request that hits the same prefix). Their absence under a light workload is expected, not a bug.

### Pool and Pending-Work Gauges

| Metric | Description |
| --- | --- |
| `vllm:simple_cpu_offload_total_blocks` | Total usable CPU KV cache blocks (excludes the null block). |
| `vllm:simple_cpu_offload_free_blocks` | Free CPU KV cache blocks. |
| `vllm:simple_cpu_offload_used_blocks` | Used CPU KV cache blocks. |
| `vllm:simple_cpu_offload_usage_perc` | Fraction of CPU KV cache blocks in use, between `0.0` and `1.0` (`1.0` = 100% used). |
| `vllm:simple_cpu_offload_pending_loads` | Requests with an outstanding CPU-to-GPU load. |
| `vllm:simple_cpu_offload_pending_stores` | In-flight GPU-to-CPU store events, including abandoned ones still draining. |

These metrics are observability only; they do not change offload behavior or improve performance on their own. The copy path (`cuMemcpyBatchAsync`/`hipMemcpyBatchAsync`) is supported on both CUDA and ROCm.

## Deprecation Policy

Note: when metrics are deprecated in version `X.Y`, they are hidden in version `X.Y+1`
but can be re-enabled using the `--show-hidden-metrics-for-version=X.Y` escape hatch,
and are then removed in version `X.Y+2`.
