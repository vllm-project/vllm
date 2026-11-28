# Preemption: recompute versus swap

> Community sprint note 015/120 · dependency-free design note · vLLM community working group

## Scope

When the block pool runs dry the engine must give something up: recompute a victim's KV later or page it to host memory. Both policies are correct; their costs differ by an order of magnitude.

## Mechanism

1. The allocator signals pressure when free blocks fall under the watermark.
2. Recompute policy drops a victim's blocks and replays its tokens when it is rescheduled.
3. Swap policy copies blocks to host memory over PCIe and restores them later.
4. Choice depends on sequence length, prefill cost and available host bandwidth.

## Knobs

| knob | default | what it changes |
| --- | --- | --- |
| `watermark_pct` | `5.0` | Free-block threshold that triggers preemption. |
| `swap_bandwidth_gbps` | `24.0` | Effective host-device copy bandwidth. |
| `swap_space_gb` | `64.0` | Host memory reserved for swapped blocks. |
| `recompute_tokens_per_s` | `20000.0` | Prefill throughput used to price recompute. |

## Failure modes

- **Swapping long sequences over PCIe when recompute would have been cheaper.**
- **Recompute storms where preempted requests immediately preempt each other.**
- **Host swap space sized without accounting for the OS page cache.**

## Instrumentation

Watch `preemption_rate`, `swap_bytes_per_s` and `recompute_tokens_total` before and after every change in this area. If a change
moves throughput without moving one of those three, the measurement is probably noise.

## Reproduction shape

```bash
# illustrative only: substitute your own model, hardware and workload shape
vllm serve MODEL --watermark_pct 5.0 --gpu-memory-utilization 0.90
```

## Discussion

The model side of this note lives in `community/src/vllm_community/preemption-recompute-swap.py`, its tests in
`community/tests/test_preemption-recompute-swap.py`, and a micro-benchmark in
`community/benchmarks/bench_preemption-recompute-swap.py`. Treat the numbers there as ratios to be re-measured
on your own hardware, never as absolute throughput claims.
