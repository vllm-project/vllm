# Tensor parallelism communication cost

> Community sprint note 007/120 · dependency-free design note · vLLM community working group

## Scope

Splitting attention heads and MLP columns across GPUs adds two all-reduces per layer. Past a point, the communication rings eat the compute that extra devices were supposed to buy.

## Mechanism

1. Column-parallel linear layers shard the output dimension, row-parallel layers shard the input.
2. Each transformer block performs an all-reduce after the attention output projection and after the MLP down projection.
3. Ring all-reduce time scales with message size over link bandwidth, and inversely with ring size.
4. Small per-GPU batches turn those collectives into latency-bound operations rather than bandwidth-bound ones.

## Knobs

| knob | default | what it changes |
| --- | --- | --- |
| `tp_size` | `4.0` | Number of tensor-parallel ranks. |
| `allreduce_us` | `180.0` | Measured per-collective latency at the target message size. |
| `hidden_dim` | `4096.0` | Hidden width; drives message size per collective. |
| `seq_tokens` | `4096.0` | Tokens resident per step, which scales the message volume. |

## Failure modes

- **Scaling tp without re-measuring collectives on the actual interconnect.**
- **Forgetting that NVLink and PCIe differ by roughly an order of magnitude.**
- **Assuming tp always lowers latency; it usually lowers per-token compute but raises the floor.**

## Instrumentation

Watch `allreduce_time_pct`, `tokens_per_second_per_gpu` and `scaling_efficiency` before and after every change in this area. If a change
moves throughput without moving one of those three, the measurement is probably noise.

## Reproduction shape

```bash
# illustrative only: substitute your own model, hardware and workload shape
vllm serve MODEL --tp_size 4.0 --gpu-memory-utilization 0.90
```

## Discussion

The model side of this note lives in `community/src/vllm_community/tensor-parallelism-communication.py`, its tests in
`community/tests/test_tensor-parallelism-communication.py`, and a micro-benchmark in
`community/benchmarks/bench_tensor-parallelism-communication.py`. Treat the numbers there as ratios to be re-measured
on your own hardware, never as absolute throughput claims.
