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

### API attribution

By default, request metrics retain their existing labels and aggregate traffic
across APIs. To distinguish Messages, Chat Completions, and Completions traffic,
start the server with `--enable-api-surface-metrics`:

```console
vllm serve <model> --enable-api-surface-metrics
```

This opt-in setting adds an `api_surface` label to the following request metrics:

- `vllm:request_success_total` (with the existing `finished_reason` label)
- `vllm:time_to_first_token_seconds`
- `vllm:inter_token_latency_seconds`
- `vllm:request_time_per_output_token_seconds`
- `vllm:e2e_request_latency_seconds`
- `vllm:request_queue_time_seconds`
- `vllm:request_inference_time_seconds`
- `vllm:request_prefill_time_seconds`
- `vllm:request_decode_time_seconds`
- `vllm:prompt_tokens_total`
- `vllm:generation_tokens_total`
- `vllm:prompt_tokens_by_source_total` (with the existing `source` label)
- `vllm:prompt_tokens_cached_total`
- `vllm:request_prompt_tokens`
- `vllm:request_generation_tokens`
- `vllm:request_prefill_kv_computed_tokens`

Its values are `messages` for `/v1/messages`, `chat_completions` for
`/v1/chat/completions`, `completions` for `/v1/completions`, and `unknown`
for callers without one of these API identities, including offline generation.
When enabled, the label applies to streaming and non-streaming generation,
including requests without tools, and does not require tracing. Inference traces
include the `api_surface` attribute for recognized APIs whenever tracing is
enabled, independently of `--enable-api-surface-metrics`.

Histogram buckets and counting semantics are unchanged. These metrics describe
engine requests: multiple generated choices contribute separately, and errors
rejected before engine admission are excluded. Enabling the setting replaces
the aggregate series with four sets of series per model and engine, one per API
value; it does not export both aggregate and per-API copies. Disabling it restores
the original labels. Changing the setting requires a server restart.

Token counters update as engine outputs arrive, including before a request
finishes. Prompt tokens are counted at prefill completion, with their existing
local-compute, local-cache, and external-transfer breakdown. Generation tokens
count all emitted token IDs, including multiple tokens in one engine output.
The scheduler's `vllm:prompt_tokens_cached_by_source_total` counter tracks
cache-tier hits at admission and remains engine-wide, as do scheduler gauges
and the per-iteration token histogram.

For example, request throughput by API can be queried with:

```promql
sum by (api_surface) (rate(vllm:request_success_total[5m]))
```

For output token throughput by API:

```promql
sum by (api_surface) (rate(vllm:generation_tokens_total[5m]))
```

To query p95 end-to-end latency by API:

```promql
histogram_quantile(0.95,
  sum by (api_surface, le) (rate(vllm:e2e_request_latency_seconds_bucket[5m])))
```

Before enabling the setting, update dashboard aggregations and vector joins
that assume a single series per model, engine, and finish reason. For totals
across APIs, aggregate with `sum without (api_surface)`. Each label combination
adds series, including histogram buckets, so account for the increased scrape
and storage volume. Existing queries retain their original behavior while the
setting remains disabled.

## Speculative Decoding Metrics

--8<-- "gen:metrics-spec-decode"

## NIXL KV Connector Metrics

--8<-- "gen:metrics-nixl"

## HiSparse KV Connector Metrics

These metrics are exposed when the `HiSparseConnector` KV connector is
configured. The host-pool gauges are reported by the scheduler once per engine
step, for the host pool shared by all tensor-parallel ranks; evictable cached
blocks count as free. With `--kv-cache-metrics`, the
`vllm:hisparse_host_block_*_seconds` histograms sample host-pool block
residency, like the device `vllm:kv_block_*_seconds` histograms, which cover
only the GPU pool.

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

## Custom Histogram Buckets

The core engine histograms ship with default bucket boundaries tuned for
typical serving workloads. The `--custom-histogram-buckets` option replaces
the boundaries of one or more *bucket families* — exactly the histograms
listed in the table below — with your own list; histograms owned by other
subsystems (for example, the NIXL connector metrics) are not affected. Use it,
for example, to track sub-300ms latency SLOs with the request-phase
histograms, whose smallest default boundary is 0.3s:

```bash
vllm serve Qwen/Qwen3-0.6B \
    --custom-histogram-buckets '{"request_latency": [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0, 30.0]}'
```

Each family key overrides a group of related histograms:

| Family key | Histograms |
| --- | --- |
| `request_latency` | `vllm:e2e_request_latency_seconds`, `vllm:request_queue_time_seconds`, `vllm:request_inference_time_seconds`, `vllm:request_prefill_time_seconds`, `vllm:request_decode_time_seconds` |
| `time_to_first_token` | `vllm:time_to_first_token_seconds` |
| `inter_token_latency` | `vllm:inter_token_latency_seconds`, `vllm:request_time_per_output_token_seconds` |
| `iteration_tokens` | `vllm:iteration_tokens_total` |
| `request_params_n` | `vllm:request_params_n` |
| `request_num_preemptions` | `vllm:request_num_preemptions` |
| `request_tokens` | `vllm:request_prompt_tokens`, `vllm:request_generation_tokens`, `vllm:request_max_num_generation_tokens`, `vllm:request_params_max_tokens`, `vllm:request_prefill_kv_computed_tokens` |
| `kv_cache_residency` | `vllm:kv_block_lifetime_seconds`, `vllm:kv_block_idle_before_evict_seconds`, `vllm:kv_block_reuse_gap_seconds` |

Bucket values must be positive, finite, and strictly increasing; unknown
family keys are rejected at startup. Families you do not list keep their
default boundaries. The `request_tokens` defaults normally scale with
`--max-model-len`; an override replaces that computed list. The
`kv_cache_residency` family only takes effect when `--kv-cache-metrics` is
enabled.

!!! warning "Bucket cardinality"
    Every bucket boundary creates one extra time series per metric and per
    label combination (model and engine index, multiplied under data-parallel
    deployments). Long bucket lists inflate Prometheus storage, scrape sizes,
    and query costs. Keep custom lists short, and only override the families
    you actively monitor.

## Deprecation Policy

Note: when metrics are deprecated in version `X.Y`, they are hidden in version `X.Y+1`
but can be re-enabled using the `--show-hidden-metrics-for-version=X.Y` escape hatch,
and are then removed in version `X.Y+2`.
