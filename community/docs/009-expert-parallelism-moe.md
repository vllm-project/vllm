# Expert parallelism for mixture-of-experts models

> Community sprint note 009/120 · dependency-free design note · vLLM community working group

## Scope

MoE layers route each token to a few experts, so capacity planning shifts from FLOPs per parameter to tokens per expert plus the two all-to-all collectives that move them.

## Mechanism

1. The router scores every token against all experts and keeps the top-k.
2. Tokens are permuted to their experts with an all-to-all, computed locally, then permuted back.
3. Expert parallelism spreads experts across ranks; tensor parallelism shards each expert internally.
4. Load imbalance shows up as a few hot experts stalling the collective for everyone.

## Knobs

| knob | default | what it changes |
| --- | --- | --- |
| `num_experts` | `64.0` | Total experts in the layer. |
| `top_k` | `2.0` | Experts each token visits. |
| `ep_size` | `8.0` | Ranks the experts are spread over. |
| `tokens_per_step` | `16384.0` | Token volume that determines the all-to-all message size. |

## Failure modes

- **Sizing ep for average load and ignoring the hot-expert tail.**
- **Dropped tokens from a capacity factor that is too small, which quietly changes model quality.**
- **Counting experts as dense parameters when estimating memory.**

## Instrumentation

Watch `expert_load_imbalance`, `all_to_all_time_pct` and `dropped_token_rate` before and after every change in this area. If a change
moves throughput without moving one of those three, the measurement is probably noise.

## Reproduction shape

```bash
# illustrative only: substitute your own model, hardware and workload shape
vllm serve MODEL --num_experts 64.0 --gpu-memory-utilization 0.90
```

## Discussion

The model side of this note lives in `community/src/vllm_community/expert-parallelism-moe.py`, its tests in
`community/tests/test_expert-parallelism-moe.py`, and a micro-benchmark in
`community/benchmarks/bench_expert-parallelism-moe.py`. Treat the numbers there as ratios to be re-measured
on your own hardware, never as absolute throughput claims.
