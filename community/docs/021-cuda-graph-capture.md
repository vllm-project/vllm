# CUDA graph capture and replay

> Community sprint note 021/120 · dependency-free design note · vLLM community working group

## Scope

Capturing decode steps into CUDA graphs removes launch overhead that dominates small-batch decode. The catch is that batch shapes must match a captured graph, so a family of graphs is needed.

## Mechanism

1. At warm-up the engine captures graphs for a set of batch sizes.
2. At runtime the batch is padded to the next captured size and replayed.
3. Replay skips Python and kernel launch overhead entirely for the model forward.
4. Memory for graphs is reserved up front, reducing the pool available to KV blocks.

## Knobs

| knob | default | what it changes |
| --- | --- | --- |
| `capture_sizes` | `32.0` | Number of distinct batch sizes with a captured graph. |
| `graph_pool_mb` | `2048.0` | Memory reserved for graph-private allocations. |
| `replay_us` | `35.0` | Per-step replay overhead at the target batch. |
| `eager_us` | `220.0` | Eager-mode per-step overhead the graph replaces. |

## Failure modes

- **Padding every batch to the largest captured size, which wastes compute.**
- **Capturing after warm-up allocations, so the graph pins stale pointers.**
- **Ignoring the graph pool when computing the KV block budget.**

## Instrumentation

Watch `graph_hit_rate`, `step_overhead_us` and `graph_pool_usage_mb` before and after every change in this area. If a change
moves throughput without moving one of those three, the measurement is probably noise.

## Reproduction shape

```bash
# illustrative only: substitute your own model, hardware and workload shape
vllm serve MODEL --capture_sizes 32.0 --gpu-memory-utilization 0.90
```

## Discussion

The model side of this note lives in `community/src/vllm_community/cuda-graph-capture.py`, its tests in
`community/tests/test_cuda-graph-capture.py`, and a micro-benchmark in
`community/benchmarks/bench_cuda-graph-capture.py`. Treat the numbers there as ratios to be re-measured
on your own hardware, never as absolute throughput claims.
