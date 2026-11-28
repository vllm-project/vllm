# Pipeline parallelism and bubble fraction

> Community sprint note 008/120 · dependency-free design note · vLLM community working group

## Scope

Pipeline stages trade latency for memory: each device holds fewer layers but the pipeline drains and fills, leaving idle bubbles whose share shrinks as micro-batches grow.

## Mechanism

1. Layers are partitioned into pp_size contiguous stages, one per device or device group.
2. The batch is cut into micro-batches that flow through the stages asynchronously.
3. Steady state keeps every stage busy; the fill and drain phases do not.
4. Bubble fraction is approximately (pp_size - 1) / (micro_batches + pp_size - 1).

## Knobs

| knob | default | what it changes |
| --- | --- | --- |
| `pp_size` | `4.0` | Number of pipeline stages. |
| `micro_batches` | `8.0` | Micro-batches per step; the main lever on bubble fraction. |
| `stage_ms` | `12.0` | Time one stage needs per micro-batch. |
| `activation_mb` | `512.0` | Activations held per stage, which bounds how many micro-batches fit. |

## Failure modes

- **Raising micro-batches past the activation memory budget to shrink the bubble.**
- **Unbalanced stage partitioning, where the slowest stage sets the whole step time.**
- **Mixing pp with tp without checking which dimension limits the interconnect.**

## Instrumentation

Watch `bubble_fraction`, `step_time_ms` and `stage_imbalance_pct` before and after every change in this area. If a change
moves throughput without moving one of those three, the measurement is probably noise.

## Reproduction shape

```bash
# illustrative only: substitute your own model, hardware and workload shape
vllm serve MODEL --pp_size 4.0 --gpu-memory-utilization 0.90
```

## Discussion

The model side of this note lives in `community/src/vllm_community/pipeline-parallelism-bubble.py`, its tests in
`community/tests/test_pipeline-parallelism-bubble.py`, and a micro-benchmark in
`community/benchmarks/bench_pipeline-parallelism-bubble.py`. Treat the numbers there as ratios to be re-measured
on your own hardware, never as absolute throughput claims.
