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

## Expert Parallel Load Balancer (EPLB) Metrics

These metrics are exposed when EPLB is enabled with
`--enable-eplb --eplb-config '{"enable_expert_metrics": true}'`. They report
one series per MoE layer and physical expert:

- `vllm:eplb_expert_topology` is the logical expert currently held by each
  physical slot. Physical experts are assigned to EP ranks in contiguous
  blocks, so `physical_expert // num_local_experts` identifies the owning GPU.
- `vllm:eplb_expert_tokens_total` counts tokens routed to each physical expert
  over the last metric sample. The `phase` label is `prefill`, `decode`, or
  `mixed` when a single step contains both. Each engine reports its own
  contribution; aggregate across engines for global load.
- `vllm:eplb_expert_load_window_steps` is the number of engine steps in the
  sample, and `vllm:eplb_expert_load_timestamp_seconds` is when the sample was
  taken.
- `vllm:eplb_rearrangements_total` and
  `vllm:eplb_rearrangement_timestamp_seconds` mark actual rearrangements.
- `vllm:eplb_rearrangement_pause_seconds` is a histogram of the blocking time
  spent rearranging experts, and
  `vllm:eplb_rearrangement_blocked_request_seconds_total` /
  `vllm:eplb_rearrangement_blocked_token_seconds_total` weight that pause by
  the in-flight batch. These are the EPLB metrics correlated with tail latency.
- `vllm:eplb_load_rank_skew` and `vllm:eplb_critical_rank_excess_tokens` are
  per-layer histograms of EP-rank imbalance, useful as guardrails.
- `vllm:eplb_load_expert_skew` and `vllm:eplb_critical_expert_excess_tokens`
  are per-layer histograms of physical-expert imbalance. They include
  zero-load experts and ranks, unlike max/mean computed from
  `vllm:eplb_expert_tokens_total` in PromQL.
- `vllm:eplb_expert_token_gini` and `vllm:eplb_expert_topk_token_share` measure
  per-layer expert concentration. The top-k value is configured with
  `expert_metrics_top_k` (default `8`).
- `vllm:eplb_active_experts_per_rank` reports the number of experts with
  nonzero load on each rank in each layer.
- `vllm:eplb_expert_topology_timestamp_seconds` marks when the current
  placement was committed.

Samples are taken every `expert_metrics_interval` steps (default `100`). Set it
to `1` for per-step samples; each sample then costs one EP-group all-reduce per
step. Rearrangement stalls are recorded per event, so they are not thinned by
the sampling interval or by Prometheus scrape loss.

```promql
# Global tokens per logical expert and phase, aggregated across replicas and
# engines.
sum by (model_name, layer, logical_expert, phase) (
  rate(vllm:eplb_expert_tokens_total[1m])
)

# Age of the last expert-load sample.
time() - max by (model_name, engine, model, phase) (
  vllm:eplb_expert_load_timestamp_seconds
)

# Current logical expert placement for layer 0 on EP rank 2.
vllm:eplb_expert_topology{layer="0", ep_rank="2"}

# p95 per-layer expert skew.
histogram_quantile(
  0.95,
  sum by (le, model_name, layer, phase) (
    rate(vllm:eplb_load_expert_skew_bucket[5m])
  )
)

# Experts touched per rank and layer.
vllm:eplb_active_experts_per_rank{model_name="my-model"}

# p99 rearrangement pause.
histogram_quantile(
  0.99,
  sum by (le, model_name, phase, async) (
    rate(vllm:eplb_rearrangement_pause_seconds_bucket[5m])
  )
)

# Fraction of request service time exposed to EPLB rearrangement stalls.
sum(increase(vllm:eplb_rearrangement_blocked_request_seconds_total[5m]))
/
(sum(avg_over_time(vllm:num_requests_running[5m])) * 300)
```

--8<-- "gen:metrics-eplb"

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
