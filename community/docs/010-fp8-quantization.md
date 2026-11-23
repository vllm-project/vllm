# FP8 quantization for weights and KV cache

> Community sprint note 010/120 · dependency-free design note · vLLM community working group

## Scope

FP8 keeps the dynamic range of fp16 in eight bits per element, halving weight and KV bandwidth. The win is bandwidth-bound, so it only shows up where the kernel is not compute-bound.

## Mechanism

1. Weights are stored as fp8 with per-tensor or per-block scales computed during calibration or on load.
2. Activations are quantized on the fly with a dynamically chosen scale per token or per block.
3. The GEMM runs in fp8 with fp32 accumulation, then dequantizes into the residual stream.
4. KV cache in fp8 halves the block footprint, roughly doubling the sequence capacity of the pool.

## Knobs

| knob | default | what it changes |
| --- | --- | --- |
| `weight_bits` | `8.0` | Storage width of the weights. |
| `act_bits` | `8.0` | Storage width used for the activation quantizer. |
| `scale_block` | `128.0` | Elements sharing one scale; smaller blocks reduce quantization error. |
| `compute_ratio` | `2.0` | FP8 throughput relative to bf16 on the target device. |

## Failure modes

- **Expecting a speed-up on a memory-bound decode when the kernel is already bandwidth-saturated by weights only.**
- **Per-tensor scales on long-tail activations clip outliers and degrade accuracy measurably.**
- **Mixing fp8 KV cache with a backend that lacks the kernel, which silently falls back to bf16 math.**

## Instrumentation

Watch `tokens_per_second_delta`, `accuracy_delta` and `kv_bytes_per_token` before and after every change in this area. If a change
moves throughput without moving one of those three, the measurement is probably noise.

## Reproduction shape

```bash
# illustrative only: substitute your own model, hardware and workload shape
vllm serve MODEL --weight_bits 8.0 --gpu-memory-utilization 0.90
```

## Discussion

The model side of this note lives in `community/src/vllm_community/fp8-quantization.py`, its tests in
`community/tests/test_fp8-quantization.py`, and a micro-benchmark in
`community/benchmarks/bench_fp8-quantization.py`. Treat the numbers there as ratios to be re-measured
on your own hardware, never as absolute throughput claims.
