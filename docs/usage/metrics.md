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

## HiSparse KV Connector Metrics

These metrics are exposed when the `HiSparseConnector` KV connector is
configured (with a `--attention-hisparse-config` sparse-MLA model, e.g.
DeepSeek V3.2). The host-tier gauges are reported by the scheduler process
once per engine step and reflect the logical host KV pool shared by all
tensor-parallel ranks; the hot-buffer counters are sampled from the device
every 2000 worker steps. Note that `vllm:hisparse_host_blocks_used` counts
blocks backing live or cached prefixes — evictable cached blocks are
included, since eviction reclaims them on demand.
With `--kv-cache-metrics`, the `vllm:hisparse_host_block_*_seconds`
histograms sample host-pool block residency (lifetime, idle time before
eviction, reuse gaps), mirroring the device `vllm:kv_block_*_seconds`
histograms, which cover only the GPU pool.

--8<-- "gen:metrics-hisparse"

## Simple CPU Offload Connector Metrics

These metrics are exposed when the `SimpleCPUOffloadConnector` KV connector
is configured (e.g. `--kv-transfer-config='{"kv_connector":
"SimpleCPUOffloadConnector", "kv_role": "kv_both", "kv_connector_extra_config":
{"kv_offload_backend": "disk", "disk_path": "/mnt/nvme/kv"}}'`). They are
updated once per engine step.

Caveats to keep in mind when interpreting them:

- A "completed" store means the write syscalls returned; it is **not**
  fsync-durable, and the disk backend's file is process-lifetime scratch,
  unlinked at startup and shutdown.
- `save_outcomes_total` classifies eager-mode boundary hand-off stores;
  lazy-mode stores are not classified.
- `used_blocks` counts blocks pinned by in-flight transfers or cache hits;
  warm cached blocks that are evictable are not counted. Use the
  `capacity_blocks` label of `simple_kv_offload_info` as the denominator.
- Counters and gauges are quantized to engine steps. They are reported by the
  scheduler process and reflect engine-wide logical block counts, not values
  pooled from individual tensor-parallel workers.

--8<-- "gen:metrics-simple-kv-offload"

## Model Flops Utilization (MFU) Performance Metrics

These metrics are available via `--enable-mfu-metrics`:

--8<-- "gen:metrics-mfu"

## Deprecation Policy

Note: when metrics are deprecated in version `X.Y`, they are hidden in version `X.Y+1`
but can be re-enabled using the `--show-hidden-metrics-for-version=X.Y` escape hatch,
and are then removed in version `X.Y+2`.
