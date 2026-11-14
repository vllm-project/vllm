# Prefill Progress Reporting

A router that balances prefill load needs to know how much prefill work a worker still owes. Without a signal from the engine it has to remember what it charged when it admitted the request, and that goes stale while a long prompt is being computed: a worker whose requests are nearly finished looks as loaded as one that just accepted them.

## What is reported

Each scheduled chunk of prefill work produces one `PrefillProgress` message on that request's own gRPC `GenerateStream`, carrying `remaining_prefill_tokens`: the tokens of that request's prefill the worker has not computed yet. A request sends at most 32 reports, spaced evenly over the work it had left when it first reported, plus a final zero when its prefill finishes. A preempted request reports the work it puts back on the queue.

The field is optional on `GenerateResponse`, so an older client ignores it and an older worker never sets it. There is no sequence number: the frames arrive on the request's own ordered stream.

Two limits are worth knowing before building on it:

- It describes prefill only, and only for requests the engine has scheduled. Queued work is not covered, so a consumer still has to estimate it.
- It travels over gRPC only. A consumer on the HTTP path does not receive it.

## Disabling it

Reporting is on by default and adds a small amount of bookkeeping per scheduled chunk. A deployment whose router does not consume the value can turn it off:

```bash
vllm serve <model> --no-enable-prefill-progress
```

The flag is read once at startup and does not change the compiled graph, so toggling it does not trigger recompilation.
