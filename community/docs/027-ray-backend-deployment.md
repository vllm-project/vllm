# Ray-based multi-node deployment

> Community sprint note 027/120 · dependency-free design note · vLLM community working group

## Scope

Ray supplies placement groups, actors and a control plane for multi-node serving. Its overhead is small per operation but shows up in startup, heartbeat and failure paths.

## Mechanism

1. The head node coordinates placement groups that pin GPU bundles to nodes.
2. Each engine replica is a Ray actor holding its own worker processes.
3. Requests are forwarded to actors through the Ray object store or a direct RPC path.
4. Failure detection runs on heartbeats, and a dead actor triggers respawn and re-registration.

## Knobs

| knob | default | what it changes |
| --- | --- | --- |
| `actors` | `8.0` | Engine actors across the cluster. |
| `placement_groups` | `1.0` | Placement groups used to pin bundles. |
| `head_overhead_ms` | `6.0` | Per-request control-plane overhead. |
| `heartbeat_s` | `5.0` | Heartbeat interval used for failure detection. |

## Failure modes

- **Serializing large payloads through the object store on the hot path.**
- **Heartbeats longer than the failover SLO.**
- **Placement groups that fit the GPUs but not the host memory the engine needs.**

## Instrumentation

Watch `actor_restarts`, `rpc_overhead_pct` and `startup_time_s` before and after every change in this area. If a change
moves throughput without moving one of those three, the measurement is probably noise.

## Reproduction shape

```bash
# illustrative only: substitute your own model, hardware and workload shape
vllm serve MODEL --actors 8.0 --gpu-memory-utilization 0.90
```

## Discussion

The model side of this note lives in `community/src/vllm_community/ray-backend-deployment.py`, its tests in
`community/tests/test_ray-backend-deployment.py`, and a micro-benchmark in
`community/benchmarks/bench_ray-backend-deployment.py`. Treat the numbers there as ratios to be re-measured
on your own hardware, never as absolute throughput claims.
