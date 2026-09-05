# Per-Request Metrics

vLLM can return per-request timing metrics directly in API responses.
This is useful for billing, SLA monitoring, and latency analysis at the
individual request level, as a complement to the server-aggregated Prometheus
metrics exposed at `/metrics`.

## Enabling

Start the server with `--enable-per-request-metrics`:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --enable-per-request-metrics
```

When this flag is set, supported API responses include metrics for each
attributable request.

!!! warning "Security: prefix-cache state"
    Prefix-cache metrics expose exact per-request cache-hit and eviction state.
    In a shared deployment, this can reveal whether another tenant populated a
    guessed prefix without relying on timing analysis. Use this flag only in a
    trusted single-tenant deployment, or ensure that every request includes an
    unpredictable secret `cache_salt` scoped to the intended tenant isolation
    boundary. A shared or predictable salt does not provide cross-tenant
    isolation. See the
    [cache-salting guidance](../usage/security.md#prefix-cache-timing-side-channel-mitigation-cache-salting)
    and [CVE-2025-46570](https://github.com/vllm-project/vllm/security/advisories/GHSA-4qjh-9fv9-r85r).

!!! note
    At high concurrency, enabling per-request metrics computation may introduce
    non-negligible CPU overhead. Benchmark your specific workload to evaluate the
    impact before enabling in production.

## Response Format

When per-request metrics are enabled, the response includes a `metrics` object:

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "choices": [ ... ],
  "usage": {
    "prompt_tokens": 42,
    "completion_tokens": 128,
    "total_tokens": 170
  },
  "metrics": {
    "time_to_first_token_ms": 85.2,
    "generation_time_ms": 1240.5,
    "queue_time_ms": 12.3,
    "mean_itl_ms": 9.1,
    "tokens_per_second": 103.2,
    "prefix_cache": {
      "num_computed_tokens": 26,
      "num_cached_tokens": 16,
      "num_local_cached_tokens": 16,
      "num_external_cached_tokens": 0,
      "num_cache_creation_tokens": 26,
      "num_new_full_blocks": 2,
      "num_block_allocations": 3,
      "num_block_evictions": 1,
      "num_prefill_chunks": 1,
      "prefill_time_ms": 85.2
    }
  }
}
```

| Field | Description |
| --- | --- |
| `time_to_first_token_ms` | Time from when the request was scheduled until the first output token was generated (TTFT). |
| `generation_time_ms` | Decode time: time from the first output token to the last output token. Excludes both queue wait and prefill/TTFT. |
| `queue_time_ms` | Time the request spent waiting in the scheduler queue before processing began. |
| `mean_itl_ms` | Mean inter-token latency (average time between successive output tokens) during the decode phase. `null` for single-token responses. |
| `tokens_per_second` | Overall output token throughput: all generated tokens over the inference interval (scheduling to last output token). Unlike `generation_time_ms`, this includes the prefill phase, so it reflects end-to-end generation speed rather than pure decode speed. |

All timing fields are `null` if the underlying timestamp data is not available
for that request.

The experimental `metrics.prefix_cache` object provides request-attributed
prompt and KV-cache telemetry:

| Field | Description |
| --- | --- |
| `num_computed_tokens` | Logical prompt tokens assigned to local model computation when each input chunk is admitted. Recomputation after preemption is not double-counted. |
| `num_cached_tokens` | Prompt tokens skipped during local computation (`num_local_cached_tokens + num_external_cached_tokens`). |
| `num_local_cached_tokens` | Prompt tokens supplied by the local prefix cache. |
| `num_external_cached_tokens` | Prompt tokens supplied through an external KV transfer. This describes the scheduler source, not whether every transferred block was a cache hit in an upstream deployment. |
| `num_cache_creation_tokens` | Prompt tokens counted as local prefix-cache creation for the request. |
| `num_new_full_blocks` | Physical KV blocks newly inserted or promoted in local prefix-cache hash maps during prefill. |
| `num_block_allocations` | Physical KV blocks allocated during prefill. |
| `num_block_evictions` | Cached physical KV blocks evicted by allocations for this request. This attributes the eviction trigger; it does not claim that the request owned the evicted block. |
| `num_prefill_chunks` | Scheduler iterations that processed at least one prompt token, including recomputation after preemption. |
| `prefill_time_ms` | Time from first scheduling to the first output token, including prefill-time preemptions. It currently has the same measurement boundaries as `time_to_first_token_ms`. |

Block counts are physical counts summed across KV cache groups. Consequently,
one logical token block can contribute more than one physical block on hybrid
models. The response's top-level `id` correlates the telemetry with the request;
the same ID is present on the final streaming chunk.

For streaming-input sessions, prefix-cache telemetry is cumulative across all
admitted input chunks. Each continuation is measured independently inside the
scheduler and then added to the prior chunks. Generated tokens retained as the
next chunk's context are not counted again as prompt input or cache creation;
physical block activity and prefill-chunk counts still include work attributable
to every input chunk. Previously emitted `RequestOutput` snapshots remain
unchanged when a later chunk completes.

These engine-level fields intentionally live under `metrics`, rather than
OpenAI `usage`: block allocation and eviction are implementation details, and
external-transfer and local-cache sources are not billing-token categories.

Offline generation exposes the same engine values as
`RequestOutput.prefill_stats`, alongside `RequestOutput.request_id`. Custom stat
logger plugins receive it as `FinishedRequestStats.prefill_stats`, correlated by
`FinishedRequestStats.request_id`. As documented by the stat-logger interface,
the plugin-side stats classes are not stable APIs and can change between
versions.

!!! note
    Timing metrics describe a single generation stream, so they are only
    returned when the request maps to exactly one. They are suppressed (the
    `metrics` object is `null`) for requests with `n > 1`, because the
    underlying timing data reflects only one of the `n` sequences and cannot be
    accurately attributed to the request as a whole. Token usage
    (`prompt_tokens`, `completion_tokens`) remains accurate in these cases.
    Per-request metrics also require server-side statistics logging, which is
    on by default. vLLM rejects `--enable-per-request-metrics` when
    `--disable-log-stats` is also set.

## Example Request

=== "Non-streaming"

    ```python
    from openai import OpenAI

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="token")

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )

    print(response.usage)
    print(response.model_extra.get("metrics"))
    ```

=== "Streaming"

    In streaming responses, metrics are attached to the final usage chunk (the
    chunk sent after all content chunks). That chunk is only emitted when usage
    reporting is enabled with `stream_options.include_usage: true` or forced
    server-side with `--enable-force-include-usage`. Without forced usage, a
    streaming client must set `stream_options.include_usage: true` to receive
    metrics.

    ```python
    from openai import OpenAI

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="token")

    stream = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        stream=True,
        stream_options={"include_usage": True},
    )

    for chunk in stream:
        if chunk.usage:
            print("Usage:", chunk.usage)
            print("Metrics:", chunk.model_extra.get("metrics"))
    ```

## Completions API

Per-request metrics are also available on the `/v1/completions` endpoint using
the same `metrics` response field. As with `n > 1`, metrics are omitted for
requests with multiple prompts, because the timing data cannot be attributed to
a single prompt's generation.

## Relationship to Prometheus Metrics

The `metrics` response field provides per-request values for a single request.
The `/metrics` Prometheus endpoint exposes server-level histograms (e.g.
`vllm:time_to_first_token_seconds`) that aggregate across all requests.

## Speculative Decoding Acceptance

When speculative decoding is enabled, per-request acceptance metrics
(mean acceptance length and the accepted-draft-length distribution) can be
returned via `--per-request-spec-decode-metrics`. They share this `metrics`
object as `metrics.speculative_decoding`, and — like the timing fields — are
reported only for single-sequence (`n == 1`) requests. See
[Per-Request Acceptance Metrics](speculative_decoding/acceptance_metrics.md).
