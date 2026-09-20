# Per-Request Acceptance Metrics

When speculative decoding is enabled, vLLM can report per-request acceptance
metrics in the response, under `metrics.speculative_decoding`. This lets a
client compute the mean acceptance length and the accepted-draft-length
distribution for an individual request, as a complement to the server-aggregated
spec-decode metrics exposed at `/metrics`.

!!! warning "Experimental"
    `metrics.speculative_decoding` is experimental and its shape may change in a
    future release. Pin to a vLLM version if you depend on it.

## Enabling

Start the server with `--per-request-spec-decode-metrics` set to `summary` or
`detailed` (default `none`):

```bash
vllm serve <target-model> \
  --speculative-config '{"method": "ngram", "num_speculative_tokens": 3, "prompt_lookup_min": 1, "prompt_lookup_max": 3}' \
  --per-request-spec-decode-metrics summary
```

| Level | Behavior |
| --- | --- |
| `none` (default) | No collection; responses are unchanged. |
| `summary` | Acceptance metrics per request. |
| `detailed` | `summary` plus ordered per-step arrays. |

Collection is gated at the source: with `none`, nothing is accumulated.

## Response Format

Acceptance metrics share the top-level `metrics` object with the timing
[per-request metrics](../per_request_metrics.md) — `metrics.speculative_decoding`
sits alongside the timing fields. Like timing, they describe a single generation
stream, so they are reported only for single-sequence requests and are `null`
for `n > 1`.

A `summary` response's `metrics` looks like:

```json
{
  "choices": [ ... ],
  "usage": { ... },
  "metrics": {
    "speculative_decoding": {
      "mean_acceptance_length": 1.2325581395348837,
      "draft_acceptance_rate": 0.07751937984496124,
      "acceptance_histogram": [39, 1, 0, 3],
      "num_spec_steps": 43,
      "num_accepted_draft_tokens": 10,
      "num_draft_tokens": 129,
      "num_spec_tokens": 3
    }
  }
}
```

| Field | Description |
| --- | --- |
| `mean_acceptance_length` | Mean tokens emitted per verification step, including the bonus token: `1 + num_accepted_draft_tokens / num_spec_steps`. Ranges from `1.0` (nothing accepted) to `num_spec_tokens + 1`. |
| `draft_acceptance_rate` | Fraction of proposed draft tokens accepted: `num_accepted_draft_tokens / num_draft_tokens`. |
| `acceptance_histogram` | Dense list of length `num_spec_tokens + 1`; index `j` is the number of steps that accepted exactly `j` draft tokens. Excludes the always-accepted bonus token. |
| `num_spec_steps` | Number of verification steps for this request (the sum of the histogram). |
| `num_accepted_draft_tokens` | Total accepted draft tokens, excluding bonus tokens. |
| `num_draft_tokens` | Total proposed draft tokens, after subtracting drafts invalidated by structured-output constraints. |
| `num_spec_tokens` | Configured `num_speculative_tokens` (`k`), i.e. the maximum draft length per step. |

With `detailed`, two ordered arrays are added, one entry per verification step:

| Field | Description |
| --- | --- |
| `per_step_accepted` | Accepted draft count at each step. |
| `per_step_drafted` | Proposed draft count at each step. Records the effective proposal length per step, so variable-length drafting (e.g. adaptive speculation) is represented without a schema change. |

`metrics.speculative_decoding` is present whenever `--per-request-spec-decode-metrics`
is `summary`/`detailed`, speculative decoding is enabled, and `n == 1` (with an
all-zero histogram if the request drafted nothing). It is `null` otherwise.

## Streaming

In streaming responses, `metrics` (including `speculative_decoding`) rides the
final usage chunk, which is only emitted when usage reporting is enabled — set
`stream_options.include_usage: true` or start the server with
`--enable-force-include-usage`.

## Relationship to Prometheus metrics

The per-request fields are the individual-request counterpart of the
server-aggregated spec-decode counters at `/metrics`. Summed across the
single-sequence requests that report them, they reconcile with the aggregate
counters (which also count `n > 1` requests, so the totals match only for
all-`n == 1` workloads):

| Per-request field (summed) | Prometheus counter |
| --- | --- |
| `num_spec_steps` | `vllm:spec_decode_num_drafts_total` |
| `num_draft_tokens` | `vllm:spec_decode_num_draft_tokens_total` |
| `num_accepted_draft_tokens` | `vllm:spec_decode_num_accepted_tokens_total` |
