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

## Parser Metrics

Parser calls run in the API server after the engine produces a token, so the
engine latency metrics above do not include them. Two histograms and a counter
expose parser cost:

- `vllm:parser_duration_seconds` (Histogram) - One observation per
  `Parser.parse` / `Parser.parse_delta` call. Non-streaming records one sample
  per parsed output; streaming records one sample per streamed delta. The
  number of deltas depends on scheduling and chunking, so compare per-delta
  cost only within a workload, or normalize with the sample count.
- `vllm:request_parser_duration_seconds` (Histogram) - One observation per
  request, summing every parse call the request made across its choices. The
  batch endpoint records one sample per conversation, because one batch
  request carries many conversations. Streaming requests record the time
  accumulated before a failure or cancellation; non-streaming requests record
  it only if parsing itself raises, so per-request percentiles are slightly
  biased downward for failed traffic.
- `vllm:tool_call_parser_invocations_total` (Counter) - Delegated tool parser
  invocations, by outcome. See [Tool-call invocation counter](#tool-call-invocation-counter).

Both histograms carry a long tail, out to 960s per call and 1920s per request,
so degenerate parses stay distinguishable instead of collapsing into `+Inf`.
When studying those outliers, use per-bucket rates rather than
`histogram_quantile`, which cannot resolve values above the last finite
boundary.

The two histograms share these labels:

- `model_name` - served model name.
- `parser` - the enabled parser pair, for example `deepseek_r1+hermes`.
  A single parser appears alone. A configured tool parser appears only with
  `--enable-auto-tool-choice`; without it, metrics are registered only when a
  reasoning parser is enabled.
- `mode` - `streaming`, `non_streaming`, or `streaming_replay`. Replay is the
  stateless scale-out derender path, which rebuilds parser state by replaying
  the whole token history through `parse_delta` once per HTTP chunk. Replay
  records samples only in `vllm:parser_duration_seconds`, never in the
  per-request histogram.
- `endpoint` - the API surface the parse serves: `chat_completions`,
  `responses`, `anthropic_messages`, `cohere_chat`, or `other` (batch and the
  token-in-token-out derender endpoint). Anthropic and Cohere requests are
  translated to chat-completion requests internally but keep their own value.

Every parser that runs through `parse` / `parse_delta` is timed at the call
site, whether the deployment uses a split parser pair or a unified parser that
extracts reasoning and tool calls in one pass; the histograms do not attribute
time to reasoning versus tool extraction. The Harmony `/v1/responses` path
drives `HarmonyParser.process_chunk` and `flush` directly and is timed too,
with the `mode` label following whether the request streams.

The histograms are registered when a reasoning parser or an effective tool
parser is enabled. Label combinations bind on first use, so a series appears
after a parse has used it rather than as zeros at startup. After an
elastic-EP scale-up, parser series disappear until the server restarts.

These metrics are emitted by the Python frontend only. The Rust frontend
(`VLLM_USE_RUST_FRONTEND=1`) has its own parser implementation and does not
export them yet.

### Example queries

Aggregate buckets across pods before applying `histogram_quantile`; never
average per-pod quantiles.

```promql
# P99 parser time per invocation, all API-server pods.
histogram_quantile(
  0.99,
  sum by (le, parser, mode) (rate(vllm:parser_duration_seconds_bucket[5m]))
)

# P90 parser time per request.
histogram_quantile(
  0.90,
  sum by (le, parser, endpoint)
    (rate(vllm:request_parser_duration_seconds_bucket[5m]))
)

# Mean parser time per request.
sum(rate(vllm:request_parser_duration_seconds_sum[5m]))
  / sum(rate(vllm:request_parser_duration_seconds_count[5m]))

# Parser CPU seconds per second, per API server process.
sum by (instance) (rate(vllm:parser_duration_seconds_sum[5m]))
```

Compare each value against that API-server container's CPU usage to get the
share of frontend CPU spent parsing. In a disaggregated deployment scrape the
API server, not the engine or the prefill/decode workers.

### Slow parse warnings

Parser calls run synchronously in the API server, so a call that takes seconds
stalls the event loop for every request on that process. Set
`VLLM_PARSER_SLOW_LOG=1` to log two structured `WARNING` events that point at
individual outliers; the histograms show distributions but cannot name the
request behind one. They default to off and never include
prompt or generated text, token IDs, or sampling parameters. Both events skip
replay calls (`mode=streaming_replay`) because replay is not one request.

- `event=parser_slow_call` is emitted for one call at or above
  `VLLM_PARSER_SLOW_CALL_SECONDS` (default 0.25; `0` disables). Fields:
  `duration_s`, `threshold_s`, `parser`, `mode`, `endpoint`, `call_index`,
  `tokens` (tokens parsed by this call, omitted when the call site does not
  report it), and `tokens_total` (parsed so far by that streamed choice,
  non-streaming request, or batch conversation). Healthy calls are
  microseconds to a few milliseconds; a quarter second is already a visible
  event-loop stall. Per-call events are capped per streamed choice,
  non-streaming request, or batch conversation, so a degraded stream cannot
  flood the log; the request summary's `slow_calls` reports the total.
- `event=parser_slow_request` is emitted once per request whose summed parser
  time reaches `VLLM_PARSER_SLOW_REQUEST_SECONDS` (default 5; `0` disables).
  It catches gradual quadratic accumulation when no single call is slow.
  Fields: `duration_s`, `threshold_s`, `parser`, `mode`, `endpoint`, `calls`,
  `max_call_s`, `slow_calls` (calls above the per-call threshold),
  `tokens_total`, and `finish_reason`. On the batch endpoint the unit is one
  conversation, and its `request_id` is `<request_id>_<prompt index>`, so the
  line can be tied back to the batch input. On multi-choice non-streaming
  requests the summary covers every choice. A request with a single call is
  not reported again if that call already produced a `parser_slow_call` event.

Both events are space-separated `key=value` pairs after the event name, and
append `request_id` only when `--enable-log-requests` is set, matching the
existing policy for request-ID logging. `calls` together with `max_call_s`
separates many small calls (history rescans) from one slow call;
`tokens_total` shows how much output the request had generated. Raise the
thresholds if long generations with re-scanning parsers are noisy; lower them
when diagnosing a known regression.

### Tool-call invocation counter

The `vllm:tool_call_parser_invocations_total` counter measures volume, not
time: one increment per non-streaming choice or per streaming delta, with
`model_name`, `mode`, `outcome` (`tool_call` / `no_tool_call`) and
`request_type` (`chat_completions`, `responses`, or `other`) labels. It counts
invocations of a delegated tool parser, so unified parsers that extract tool
calls in a single pass (the engine-based parsers and Harmony) do not increment
it. It is registered whenever parser metrics are enabled, including
reasoning-only deployments, and its zero-valued series are exported from
startup so existing queries keep working. Its `request_type` label predates the
histogram `endpoint` label and is kept for compatibility: it comes from the
request object, so translated Anthropic and Cohere requests, batch requests,
and token-in-token-out derender all report `chat_completions`, while replay
deltas report `mode="streaming_replay"`.

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
