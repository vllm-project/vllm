# torch.compile integration for the engine

> Community sprint note 028/120 · dependency-free design note · vLLM community working group

## Scope

Compiling the model forward buys fusion and removes Python overhead, but dynamic batch shapes trigger recompilation and guard failures. The serving envelope must be enumerable.

## Mechanism

1. The forward is traced with symbolic dimensions and specialised for the shapes it sees.
2. Each new shape combination or guard failure costs a recompile.
3. Piecewise compilation splits the graph at graph breaks, keeping unfused regions in eager mode.
4. Compile time is paid at warm-up, which must be budgeted in the deployment.

## Knobs

| knob | default | what it changes |
| --- | --- | --- |
| `dynamic_shapes` | `1.0` | Whether shapes stay symbolic (1) or are specialised (0). |
| `recompile_limit` | `8.0` | Recompiles before compilation is disabled for the process. |
| `compile_s` | `90.0` | Warm-up cost of the first compilation. |
| `guard_hits` | `64.0` | Guard evaluations per step before a recompile is considered. |

## Failure modes

- **Dynamic shapes in decode, where every batch size is a new graph.**
- **Ignoring graph breaks, which leave the hot loop in eager mode.**
- **Skipping the recompile limit, so a pathological client throttles the server.**

## Instrumentation

Watch `recompiles_per_hour`, `graph_breaks` and `compile_time_pct` before and after every change in this area. If a change
moves throughput without moving one of those three, the measurement is probably noise.

## Reproduction shape

```bash
# illustrative only: substitute your own model, hardware and workload shape
vllm serve MODEL --dynamic_shapes 1.0 --gpu-memory-utilization 0.90
```

## Discussion

The model side of this note lives in `community/src/vllm_community/torch-compile-integration.py`, its tests in
`community/tests/test_torch-compile-integration.py`, and a micro-benchmark in
`community/benchmarks/bench_torch-compile-integration.py`. Treat the numbers there as ratios to be re-measured
on your own hardware, never as absolute throughput claims.
