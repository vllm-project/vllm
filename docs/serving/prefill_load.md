# Prefill load for routing

A worker reports how much prefill work it still owes, so a router can price a worker from the
worker's own accounting instead of a guess made when the request arrived.

The value is token work per worker — not a latency estimate and not a per-request report.

## Read it

Two surfaces publish the same snapshot; use whichever fits your poll rate.

**`GET /load`** — for frequent polling (a few requests per second per worker):

```console
$ curl -s localhost:8000/load
{"server_load": 7, "prefill_tokens_remaining": 412000, "sample_seq": 128}
```

**`GET /metrics`** — for dashboards, alerting, and slow consumers:

```text
vllm:prefill_tokens_remaining
vllm:prefill_sample_seq
```

Both are enabled by default. A worker that predates these fields simply omits them.

## Use it

| Field | Meaning |
|---|---|
| `prefill_tokens_remaining` | Prefill tokens the worker still owes across every request it knows about |
| `sample_seq` | Increases with each new sample; use it to detect stale data |

A typical routing loop:

1. Poll `/load` every 200–500 ms per worker.
2. Ignore a sample whose `sample_seq` is not newer than the last one you applied.
3. Replace your estimate with `prefill_tokens_remaining`, then add requests you dispatched after the
   sample was received and subtract the ones you finished.
4. On a failed poll or a missing field, keep your previous estimate.

The value covers requests that are still queued, using their prompt length as an upper bound — their
cache reuse is not known until they are scheduled. It can also rise: preemption and cache invalidation
send work back to the queue.

## Limits

- Worker-level only: it cannot tell you which request the work belongs to.
- Freshness is bounded by your poll interval; `sample_seq` makes staleness visible, not impossible.
- It describes prefill. Decode load, KV transfer time, and batching remain separate inputs.
- Do not extend a request's deadline because of this signal.
