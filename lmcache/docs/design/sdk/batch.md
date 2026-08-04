# Batched Stream SDK

> Wrapper over the [Request Stream SDK](request.md) for batching many requests

## Goal

`LMCacheRequestStream` drives one request. `LMCacheBatchedStream` groups many
of them and runs each phase — prefill, modify, decode — across the whole batch
at once (one thread per stream), then aggregates the per-stream 
`StreamPerfMetrics` into a single `Metrics` report.

```py
import lmcache.sdk as lmc_sdk

batch = lmc_sdk.batch.LMCacheBatchedStream()
for toks in prompts:
    # contexts is an iterable, e.g. [kv_ctx] or [kv_ctx, q_ctx]
    batch.add(lmc_sdk.request.create_request([ctx], post_completion, toks))

batch.prefill({"temperature": 1.0}) # max_tokens forced to 1
batch.modify(drop_tokens) # edit every stream's KV concurrently
results = batch.decode({"max_tokens": 256})  # returns Metrics
results.emit() # print the Metric in a table on terminal
data = results.to_dict() # to get the Metric as a dictionary
```

See example in [token-dropping example](../../../examples/token_dropping/).

## Pipeline

The three high-level methods mirror the stream phases and each return a
`Metrics`:

- **`prefill(sampling_params, fmt="terminal", width=80, stream_ids=None)`**:
  forces `max_tokens=1`, runs every stream, reports prefill metrics.
- **`modify(fn, fmt="terminal", width=80, timeout=30.0, poll_interval=0.2)`**: applies
  `fn` to every stream's KV (`modify_kv`), polling until done. Only reports duration (s).
- **`decode(sampling_params, fmt="terminal", width=80, stream_ids=None)`**:
  runs every stream and reports decode metrics.

Lower-level pieces:

- **`add(request_stream)`** / **`get_request_stream(stream_id)`**: register / fetch a request stream
  (keyed by `request_stream.request_stream_id`).
- **`run_request_streams(sampling_params, stream_ids=None)`** returns duration (s).
  Batches `request_stream.generate(sampling_params)` across thread pool and stores each
  result in `perf_metrics`. Used by `prefill`/`decode`.
- **`modify_request_streams(fn, stream_ids=None, timeout=30.0, poll_interval=0.2`** returns duration
  (s). Batches `request_stream.modify_kv(fn)` across the thread pool.
- **`get_perf_metrics(duration, fmt, width, mode, stream_ids=None)`** returns
  the aggregated Metrics. `mode` can be `"prefill"` or `"decode"`.

`stream_ids=None` means "all streams in the batch".
