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
    on by default. vLLM rejects `--enable-per-request-metrics` or
    `--enable-per-request-output-token-metrics` when `--disable-log-stats` is also
    set.

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

## Responses API

Per-request metrics are available on the `/v1/responses` endpoint using the
common `metrics` object described above. Non-streaming responses include it at
the top level. Streaming responses include it in the final response carried by
the `response.completed` event; intermediate events do not include metrics.

Metrics are omitted for Responses requests that perform multiple
model-generation turns, such as built-in tool-call workflows, because the
response retains timing data for only one generation turn while token usage is
accumulated across all turns.

## Output Token Metrics

Output-token metrics are experimental and subject to change. Enable them for a
reasoning model with a compatible output parser:

```bash
vllm serve openai/gpt-oss-20b \
  --enable-per-request-output-token-metrics
```

GPT-OSS models automatically configure the built-in `openai_gptoss` reasoning
parser and Harmony output parser. Other reasoning models must be started with a
compatible `--reasoning-parser`, such as `gemma4` or `nemotron_v3`. When
output-token metrics are enabled, vLLM verifies at startup that the resolved
parser supports output-token classification and rejects unsupported parser
configurations.

This option does not require `--enable-per-request-metrics`; it includes the
aggregate timing metrics and adds a nested `output_token_metrics` object to
Responses and Chat Completions responses:

```json
{
  "metrics": {
    "time_to_first_token_ms": 85.2,
    "generation_time_ms": 1240.5,
    "queue_time_ms": 12.3,
    "mean_itl_ms": 9.1,
    "tokens_per_second": 103.2,
    "output_token_metrics": {
      "reasoning": {
        "token_count": 36,
        "time_to_first_token_ms": 108.22,
        "generation_time_ms": 160.0,
        "mean_itl_ms": 4.57,
        "tokens_per_second": 218.75
      },
      "content": {
        "token_count": 20,
        "time_to_first_token_ms": 268.22,
        "generation_time_ms": 210.0,
        "mean_itl_ms": 11.05,
        "tokens_per_second": 90.5
      },
      "unclassified_token_count": 2
    }
  }
}
```

Each category reports timing scoped to its classified tokens. Both category
TTFT values use the request's scheduled time as their common origin. Category
generation time spans its first and last observed token batches; mean ITL
divides that interval by `token_count - 1`, and category throughput is its
reciprocal. Unlike aggregate `tokens_per_second`, category throughput excludes
the category's TTFT. A zero-token category remains present with `token_count: 0`
and `null` timing fields.

Token timing has engine output-batch resolution. When one output batch contains
multiple tokens, including tokens on both sides of a reasoning/content
boundary, those tokens share a timestamp. Category mean ITL and throughput are
averages over the interval between the first and last observed batches; vLLM
does not infer individual timestamps for tokens within a batch. These fields
remain `null` when there are fewer than two category tokens or no measurable
interval between the first and last observations. Multiple segments of the same
category are aggregated, so their generation interval includes time between
segments.

`reasoning.token_count` uses the same parser classification as the endpoint's
existing reasoning-token usage field. It therefore equals
`usage.output_tokens_details.reasoning_tokens` for Responses and
`usage.completion_tokens_details.reasoning_tokens` for Chat Completions. For
Harmony, this follows the existing usage convention in which analysis and
addressed commentary or tool-call tokens count as reasoning.

`content.token_count` represents final-answer payload tokens.
`unclassified_token_count` contains remaining boundary, framing, and other
parser-control tokens. For every classified response, the three counts
reconcile to the existing total output-token count:

```text
reasoning.token_count + content.token_count + unclassified_token_count
    = total generated tokens
```

The total is `usage.output_tokens` for Responses and
`usage.completion_tokens` for Chat Completions.

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
