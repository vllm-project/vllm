# Scheduler policy: FCFS, priority and fairness

> Community sprint note 013/120 · dependency-free design note · vLLM community working group

## Scope

Admission order decides who waits. FCFS is predictable, priority maps to business value, and fair-share keeps one tenant from owning the GPU. All three need a starvation guard.

## Mechanism

1. Waiting requests are ordered by arrival, priority class, or a weighted fair-share key.
2. Each step admits as many waiting requests as the token and block budgets allow.
3. Aging raises the effective priority of long-waiting requests so nothing starves.
4. Preemption demotes or recomputes running sequences when a higher class needs the blocks.

## Knobs

| knob | default | what it changes |
| --- | --- | --- |
| `priority_levels` | `4.0` | Distinct priority classes the policy understands. |
| `aging_window_ms` | `5000.0` | Wait time after which a request gains one class. |
| `fair_share_weight` | `1.0` | Tenant weight in the fair-share key. |
| `preempt_cost_ms` | `8.0` | Cost of preempting one running sequence. |

## Failure modes

- **Priority without aging produces permanent starvation under sustained high-class load.**
- **Recompute-based preemption that ignores prefill cost makes demotion more expensive than waiting.**
- **Fair-share keys recomputed per step, which makes ordering unstable.**

## Instrumentation

Watch `queue_wait_ms_p99`, `starvation_events` and `preemptions_per_min` before and after every change in this area. If a change
moves throughput without moving one of those three, the measurement is probably noise.

## Reproduction shape

```bash
# illustrative only: substitute your own model, hardware and workload shape
vllm serve MODEL --priority_levels 4.0 --gpu-memory-utilization 0.90
```

## Discussion

The model side of this note lives in `community/src/vllm_community/scheduler-policy-fairness.py`, its tests in
`community/tests/test_scheduler-policy-fairness.py`, and a micro-benchmark in
`community/benchmarks/bench_scheduler-policy-fairness.py`. Treat the numbers there as ratios to be re-measured
on your own hardware, never as absolute throughput claims.
