# Prefill Progress Reporting

The engine can report how much prefill work each request has left, on that request's own gRPC stream, so a router can price a running request by what remains instead of by the charge it took at admission, which goes stale while a long prompt is computed.

## Enabling it

```bash
vllm serve <model> --enable-prefill-progress
```

Off by default. The flag is read once at startup.

## How to use it

Read `GenerateResponse.prefill_progress.remaining_prefill_tokens` on the request's `GenerateStream`: the tokens of that request's prefill the worker has not computed yet. A request sends at most 32 reports, spaced evenly over the work it had left, plus a final zero when prefill finishes. It covers scheduled prefill over gRPC; queued work and the HTTP path are not covered.
