# Per-Request Acceptance Metrics

When speculative decoding is enabled, vLLM can return per-request acceptance
statistics directly on each API response choice. This lets a client compute the
mean acceptance length and the accepted-draft-length distribution for an
individual request, as a complement to the server-aggregated spec-decode
metrics exposed at `/metrics`.

## Enabling

Start the server with `--per-request-spec-decode-stats` set to `summary` or
`detailed` (default `none`):

```bash
vllm serve <target-model> \
  --speculative-config '{"method": "ngram", "num_speculative_tokens": 3, "prompt_lookup_min": 1, "prompt_lookup_max": 3}' \
  --per-request-spec-decode-stats summary
```

| Level | Behavior |
| --- | --- |
| `none` (default) | No collection; responses are unchanged. |
| `summary` | Aggregate acceptance stats per request. |
| `detailed` | `summary` plus ordered per-step arrays. |

Collection is gated at the source: with `none`, nothing is accumulated and the
response shape is identical to running without the flag.

## Response Format

Stats are attached **per choice**, under `speculative_decoding_stats`. Unlike
the top-level timing [per-request metrics](../per_request_metrics.md) — which
are suppressed for `n > 1` because they cannot be attributed to a single
sequence — acceptance stats are per choice by construction, so each of the `n`
sequences reports its own acceptance independently.

A `summary` choice looks like:

```json
{
  "index": 0,
  "message": { "role": "assistant", "content": "..." },
  "finish_reason": "stop",
  "speculative_decoding_stats": {
    "mean_acceptance_length": 1.2325581395348837,
    "draft_acceptance_rate": 0.07751937984496124,
    "acceptance_histogram": {"0": 39, "1": 1, "3": 3},
    "num_spec_steps": 43,
    "num_accepted_draft_tokens": 10,
    "num_draft_tokens": 129,
    "num_spec_tokens": 3
  }
}
```

| Field | Description |
| --- | --- |
| `mean_acceptance_length` | Mean number of tokens emitted per verification step, including the bonus token: `1 + num_accepted_draft_tokens / num_spec_steps`. Ranges from `1.0` (nothing accepted) to `num_spec_tokens + 1`. |
| `draft_acceptance_rate` | Fraction of proposed draft tokens that were accepted: `num_accepted_draft_tokens / num_draft_tokens`. |
| `acceptance_histogram` | Sparse map from accepted draft count `j` to the number of steps that accepted exactly `j` draft tokens. Keys with a zero count are omitted. Excludes the always-accepted bonus token. |
| `num_spec_steps` | Number of verification steps for this request (equals the sum of the histogram counts). |
| `num_accepted_draft_tokens` | Total accepted draft tokens, excluding bonus tokens. |
| `num_draft_tokens` | Total proposed draft tokens, after subtracting drafts invalidated by structured-output constraints. |
| `num_spec_tokens` | Configured `num_speculative_tokens` (`k`), i.e. the maximum draft length per step. |

With `detailed`, two ordered arrays are added, one entry per verification step:

| Field | Description |
| --- | --- |
| `per_step_accepted` | Accepted draft count at each step. |
| `per_step_drafted` | Proposed draft count at each step. This records the effective proposal length per step, so variable-length drafting (e.g. adaptive speculation) is represented without a schema change. |

All choices carry `speculative_decoding_stats: null` when the request did not go
through speculative decoding, or when the flag is `none`.

## Streaming

In streaming responses, acceptance stats are attached to the **terminal chunk's
choice** (the chunk carrying the finish reason), mirroring the non-streaming
contract.

## Relationship to Prometheus metrics

The per-request fields are the individual-request counterpart of the
server-aggregated spec-decode counters at `/metrics`. Summed across every choice
of every request served, they reconcile with the aggregate counters:

| Per-request field (summed) | Prometheus counter |
| --- | --- |
| `num_spec_steps` | `vllm:spec_decode_num_drafts_total` |
| `num_draft_tokens` | `vllm:spec_decode_num_draft_tokens_total` |
| `num_accepted_draft_tokens` | `vllm:spec_decode_num_accepted_tokens_total` |
