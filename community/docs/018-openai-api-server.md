# OpenAI-compatible API server behaviour

> Community sprint note 018/120 · dependency-free design note · vLLM community working group

## Scope

The HTTP layer is where concurrency limits, queueing and JSON overhead become user-visible latency. Tuning the model is pointless if the front end saturates first.

## Mechanism

1. Requests are parsed and validated, then enqueued with a priority derived from the endpoint.
2. The async engine loop admits them under the same token budget the scheduler uses.
3. Responses stream as server-sent events, one chunk per generated token group.
4. Connection limits and keep-alive settings decide whether the front end or the model is the bottleneck.

## Knobs

| knob | default | what it changes |
| --- | --- | --- |
| `max_concurrency` | `512.0` | In-flight requests the server accepts. |
| `queue_depth` | `2048.0` | Pending requests before the server returns 429. |
| `json_overhead_us` | `180.0` | Serialization cost per response chunk. |
| `keepalive_s` | `75.0` | Idle connection timeout. |

## Failure modes

- **Benchmarking the engine directly and shipping an HTTP front end that caps throughput.**
- **Unbounded queues that convert overload into memory growth instead of fast rejection.**
- **Streaming with per-token flushes, which multiplies syscalls.**

## Instrumentation

Watch `http_rps`, `queue_rejections` and `serialization_time_pct` before and after every change in this area. If a change
moves throughput without moving one of those three, the measurement is probably noise.

## Reproduction shape

```bash
# illustrative only: substitute your own model, hardware and workload shape
vllm serve MODEL --max_concurrency 512.0 --gpu-memory-utilization 0.90
```

## Discussion

The model side of this note lives in `community/src/vllm_community/openai-api-server.py`, its tests in
`community/tests/test_openai-api-server.py`, and a micro-benchmark in
`community/benchmarks/bench_openai-api-server.py`. Treat the numbers there as ratios to be re-measured
on your own hardware, never as absolute throughput claims.
