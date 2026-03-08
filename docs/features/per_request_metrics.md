# Per-Request Metrics

vLLM can return per-request performance metrics and token usage details directly in API responses.
This is useful for billing, SLA monitoring, and latency analysis at the individual request level,
as a complement to the server-aggregated Prometheus metrics exposed at `/metrics`.

## Enabling

Start the server with `--enable-per-request-metrics`:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --enable-per-request-metrics
```

Then set `include_metrics: true` in the request body to receive metrics for that request.
Metrics are only computed when both the server flag and the per-request parameter are set,
which avoids throughput overhead for clients that do not need them.

!!! note
    At high concurrency, enabling per-request metrics computation introduces additional
    CPU overhead. Benchmark your specific workload to evaluate the impact before enabling
    in production.

## Response Format

### Timing Metrics

When `include_metrics: true` is sent, the response includes a `metrics` object:

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
    "time_to_first_content_token_ms": null,
    "generation_time_ms": 1240.5,
    "queue_time_ms": 12.3,
    "mean_itl_ms": 9.1,
    "tokens_per_second": 103.2
  }
}
```

| Field | Description |
|-------|-------------|
| `time_to_first_token_ms` | Time from when the request was scheduled until the first output token was generated (TTFT). For reasoning models, this is the time to the first reasoning token. |
| `time_to_first_content_token_ms` | Time to the first non-reasoning (content) token. Only populated for reasoning models; `null` otherwise. |
| `generation_time_ms` | Total time spent generating output tokens, excluding queue wait time. |
| `queue_time_ms` | Time the request spent waiting in the scheduler queue before processing began. |
| `mean_itl_ms` | Mean inter-token latency (average time between successive output tokens) during the decode phase. `null` for single-token responses. |
| `tokens_per_second` | Output token throughput: `completion_tokens / generation_time_ms * 1000`. |

All fields are `null` if the underlying timing data is not available for that request.

### Reasoning Token Details

For reasoning models, the `usage` object includes a
`completion_tokens_details` field that separates reasoning tokens from content tokens:

```json
{
  "usage": {
    "prompt_tokens": 55,
    "completion_tokens": 512,
    "total_tokens": 567,
    "completion_tokens_details": {
      "reasoning_tokens": 384,
      "accepted_prediction_tokens": null,
      "rejected_prediction_tokens": null
    }
  }
}
```

The number of content (non-reasoning) tokens is `completion_tokens - reasoning_tokens`.

`completion_tokens_details` is populated automatically whenever a `--reasoning-parser` is
configured on the server. It does not require `include_metrics: true`.

## Example Request

=== "Non-streaming"

    ```python
    from openai import OpenAI

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="token")

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        extra_body={"include_metrics": True},
    )

    print(response.usage)
    print(response.model_extra.get("metrics"))
    ```

=== "Streaming"

    Metrics are attached to the final usage chunk (the chunk sent after all content chunks,
    when `stream_options.include_usage` is `true`):

    ```python
    from openai import OpenAI

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="token")

    stream = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"include_metrics": True},
    )

    for chunk in stream:
        if chunk.usage:
            print("Usage:", chunk.usage)
            print("Metrics:", chunk.model_extra.get("metrics"))
    ```

## Completions API

Per-request metrics are also available on the `/v1/completions` endpoint using the same
`include_metrics` request parameter and the same `metrics` response field.

## Relationship to Prometheus Metrics

The `metrics` response field provides per-request values for a single request.
The `/metrics` Prometheus endpoint exposes server-level histograms (e.g.
`vllm:time_to_first_token_seconds`) that aggregate across all requests.
Both are complementary: use Prometheus for fleet-level SLO monitoring and dashboards,
and `include_metrics` for per-user billing, per-request debugging, or client-side
latency attribution.
