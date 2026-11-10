# Request routing and load balancing

> Community sprint note 026/120 · dependency-free design note · vLLM community working group

## Scope

With several replicas, the router decides whether a request lands on a warm cache or a busy GPU. Least-outstanding is a decent default; prefix locality is worth more when prompts repeat.

## Mechanism

1. The router scores replicas by queue depth, KV usage and recent latency.
2. Prefix-aware routing hashes the prompt prefix and prefers the replica that already holds it.
3. Health checks remove replicas that fail a readiness probe or exceed an error budget.
4. Retries must be sticky-free, or a failing replica receives the same request repeatedly.

## Knobs

| knob | default | what it changes |
| --- | --- | --- |
| `queue_weight` | `1.0` | Weight of queue depth in the routing score. |
| `prefix_locality` | `0.4` | Bonus for a replica holding a matching prefix. |
| `health_window_s` | `30.0` | Window used to compute replica error rates. |
| `replica_count` | `4.0` | Fleet size the router balances over. |

## Failure modes

- **Round-robin over replicas with uneven cache state, which doubles prefill work.**
- **Health checks that only test the port, not the engine loop.**
- **Retry storms that amplify the original overload.**

## Instrumentation

Watch `routing_skew_pct`, `cache_locality_hit_rate` and `replica_error_rate` before and after every change in this area. If a change
moves throughput without moving one of those three, the measurement is probably noise.

## Reproduction shape

```bash
# illustrative only: substitute your own model, hardware and workload shape
vllm serve MODEL --queue_weight 1.0 --gpu-memory-utilization 0.90
```

## Discussion

The model side of this note lives in `community/src/vllm_community/request-routing.py`, its tests in
`community/tests/test_request-routing.py`, and a micro-benchmark in
`community/benchmarks/bench_request-routing.py`. Treat the numbers there as ratios to be re-measured
on your own hardware, never as absolute throughput claims.
