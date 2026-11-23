# Worker restarts and health checks

> Community sprint note 031/120 · dependency-free design note · vLLM community working group

## Scope

A worker that hangs is worse than one that crashes. Watchdogs, bounded restarts and health endpoints decide whether a deployment degrades or disappears.

## Mechanism

1. A watchdog thread timestamps each engine step; a stalled step count triggers a restart.
2. The supervisor restarts the worker with exponential backoff and a cap.
3. Readiness probes check the engine loop, not just the listening socket.
4. In-flight requests during a restart fail fast rather than hanging in the queue.

## Knobs

| knob | default | what it changes |
| --- | --- | --- |
| `watchdog_s` | `30.0` | Time without a completed step before the worker is considered stalled. |
| `max_restarts` | `3.0` | Restarts allowed inside the backoff window. |
| `health_timeout_s` | `5.0` | Probe timeout. |
| `backoff_s` | `10.0` | Base delay between restart attempts. |

## Failure modes

- **Readiness probes that pass while the engine loop is deadlocked.**
- **Unbounded restarts that hide a deterministic crash loop.**
- **Requests queued across a restart without an idempotency story.**

## Instrumentation

Watch `restarts_per_day`, `stalled_steps` and `failed_requests_during_restart` before and after every change in this area. If a change
moves throughput without moving one of those three, the measurement is probably noise.

## Reproduction shape

```bash
# illustrative only: substitute your own model, hardware and workload shape
vllm serve MODEL --watchdog_s 30.0 --gpu-memory-utilization 0.90
```

## Discussion

The model side of this note lives in `community/src/vllm_community/worker-restart-health.py`, its tests in
`community/tests/test_worker-restart-health.py`, and a micro-benchmark in
`community/benchmarks/bench_worker-restart-health.py`. Treat the numbers there as ratios to be re-measured
on your own hardware, never as absolute throughput claims.
