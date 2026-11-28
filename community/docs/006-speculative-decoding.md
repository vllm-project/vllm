# Speculative decoding economics

> Community sprint note 006/120 · dependency-free design note · vLLM community working group

## Scope

A small draft model proposes k tokens, the target model verifies them in one pass, and accepted prefixes are free. Whether that helps depends on acceptance rate, draft cost and batch size.

## Mechanism

1. The draft model autoregressively proposes k candidates for the current position.
2. The target model scores all k+1 positions in a single forward pass.
3. A rejection-sampling rule accepts the longest prefix that matches the target distribution, guaranteeing output equivalence.
4. Accepted tokens advance the sequence by more than one position per target step.

## Knobs

| knob | default | what it changes |
| --- | --- | --- |
| `draft_len` | `5.0` | Proposed tokens per step; larger values amplify any acceptance-rate drop. |
| `accept_rate` | `0.7` | Expected fraction of proposals kept. |
| `draft_cost_ratio` | `0.1` | Draft forward cost as a fraction of the target forward. |
| `target_batch` | `32.0` | Batch size at verification; large batches shrink the relative win. |

## Failure modes

- **Measuring speed-up at batch 1 and shipping it at batch 64, where the target model is already saturated.**
- **Draft models with a different tokenizer silently break the equivalence guarantee.**
- **Ignoring the extra KV blocks the draft model needs when sizing the pool.**

## Instrumentation

Watch `accepted_tokens_per_step`, `speedup_vs_baseline` and `draft_overhead_pct` before and after every change in this area. If a change
moves throughput without moving one of those three, the measurement is probably noise.

## Reproduction shape

```bash
# illustrative only: substitute your own model, hardware and workload shape
vllm serve MODEL --draft_len 5.0 --gpu-memory-utilization 0.90
```

## Discussion

The model side of this note lives in `community/src/vllm_community/speculative-decoding.py`, its tests in
`community/tests/test_speculative-decoding.py`, and a micro-benchmark in
`community/benchmarks/bench_speculative-decoding.py`. Treat the numbers there as ratios to be re-measured
on your own hardware, never as absolute throughput claims.
