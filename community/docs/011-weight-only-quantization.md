# Weight-only quantization with AWQ and GPTQ

> Community sprint note 011/120 · dependency-free design note · vLLM community working group

## Scope

4-bit weight-only formats cut the memory footprint enough to fit larger models, at the cost of dequantization work inside the GEMM and sensitivity to group size.

## Mechanism

1. Weights are grouped into blocks whose shared scale (and optional zero point) is stored alongside them.
2. The kernel loads packed 4-bit weights, unpacks into registers and dequantizes before the tensor core multiply.
3. Activation-aware search chooses scales that protect the channels with the largest activation magnitudes.
4. Group size trades accuracy against the size of the metadata and the efficiency of the unpack path.

## Knobs

| knob | default | what it changes |
| --- | --- | --- |
| `bits` | `4.0` | Weight width; 8 halves compression and is nearly lossless. |
| `group_size` | `128.0` | Weights sharing a scale. |
| `zero_points` | `0.0` | Whether asymmetric quantization is stored (1) or not (0). |
| `kernel_efficiency` | `0.75` | Fraction of bf16 GEMM throughput the dequant path reaches. |

## Failure modes

- **Comparing memory savings without measuring the decode overhead of the unpack path.**
- **Group size 32 chosen for accuracy while the kernel only vectorizes at 128.**
- **Calibration data that does not resemble production traffic, hiding accuracy loss.**

## Instrumentation

Watch `weight_mb`, `tokens_per_second_delta` and `perplexity_delta` before and after every change in this area. If a change
moves throughput without moving one of those three, the measurement is probably noise.

## Reproduction shape

```bash
# illustrative only: substitute your own model, hardware and workload shape
vllm serve MODEL --bits 4.0 --gpu-memory-utilization 0.90
```

## Discussion

The model side of this note lives in `community/src/vllm_community/weight-only-quantization.py`, its tests in
`community/tests/test_weight-only-quantization.py`, and a micro-benchmark in
`community/benchmarks/bench_weight-only-quantization.py`. Treat the numbers there as ratios to be re-measured
on your own hardware, never as absolute throughput claims.
