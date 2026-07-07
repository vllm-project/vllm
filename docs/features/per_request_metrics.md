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
    "tokens_per_second": 103.2
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

All fields are `null` if the underlying timing data is not available for that
request.

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
