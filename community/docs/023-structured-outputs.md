# Structured outputs from JSON schema

> Community sprint note 023/120 · dependency-free design note · vLLM community working group

## Scope

Constraining generation to a schema means compiling it into a state machine, masking invalid tokens each step, and accepting compilation cost in exchange for never post-validating.

## Mechanism

1. The JSON schema is compiled into an automaton whose states allow only valid next tokens.
2. At each decode step the automaton contributes a token bitmask applied to the logits.
3. On a token advance the automaton transitions; accepting states release the request.
4. Compilation is cached per schema hash, since it costs far more than a single decode step.

## Knobs

| knob | default | what it changes |
| --- | --- | --- |
| `schema_depth` | `6.0` | Nesting depth that drives automaton size. |
| `fsm_states` | `1200.0` | Compiled automaton size. |
| `compile_ms` | `90.0` | Compilation cost per unique schema. |
| `mask_overhead_us` | `70.0` | Per-step bitmask application cost. |

## Failure modes

- **Recompiling an identical schema per request.**
- **Schemas with open-ended recursion, which blow up automaton size.**
- **Bitmask application in Python instead of fused into the sampler.**

## Instrumentation

Watch `compile_cache_hit_rate`, `constraint_overhead_pct` and `schema_rejections` before and after every change in this area. If a change
moves throughput without moving one of those three, the measurement is probably noise.

## Reproduction shape

```bash
# illustrative only: substitute your own model, hardware and workload shape
vllm serve MODEL --schema_depth 6.0 --gpu-memory-utilization 0.90
```

## Discussion

The model side of this note lives in `community/src/vllm_community/structured-outputs.py`, its tests in
`community/tests/test_structured-outputs.py`, and a micro-benchmark in
`community/benchmarks/bench_structured-outputs.py`. Treat the numbers there as ratios to be re-measured
on your own hardware, never as absolute throughput claims.
