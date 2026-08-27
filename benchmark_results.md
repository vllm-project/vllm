# Benchmark Results

Hardware: gfx942/gfx950, TP4, vLLM 0.27.1+rocm723
Model: amd/MiniMax-M3-MXFP4
Drafter: Inferact/MiniMax-M3-EAGLE3 (num_speculative_tokens=3)
KV cache: fp8

## Speculative Decode: ROCM_AITER_FA vs TRITON_ATTN

Workload: random, input_len=8192, output_len=1024, concurrency=8, 80 prompts

| Metric | TRITON_ATTN | ROCM_AITER_FA | Delta |
|--------|-------------|---------------|-------|
| Total token throughput (tok/s) | 4790 | **5614** | **+17%** |
| Output token throughput (tok/s) | 527 | **618** | **+17%** |
| Request throughput (req/s) | 0.50 | **0.58** | **+16%** |
| Mean TPOT (ms) | 11.93 | **10.21** | **-14%** |
| Median TPOT (ms) | 10.70 | **9.66** | **-10%** |
| P99 TPOT (ms) | 25.38 | **18.11** | **-29%** |
| Mean TTFT (ms) | 3226 | **2652** | **-18%** |
| Median TTFT (ms) | 532 | **524** | -2% |
| P99 TTFT (ms) | 27109 | **22845** | **-16%** |
| Mean ITL (ms) | 31.86 | **27.62** | **-13%** |
| P99 ITL (ms) | 311 | **308** | -1% |
| Acceptance rate (%) | 56.4 | **59.6** | **+3.2pp** |
| Acceptance length | 2.69 | **2.79** | +4% |

## Non-Speculative Decode: ROCM_AITER_FA (no Eagle3)

Workload: random, input_len=8192, output_len=1024, concurrency=8, 80 prompts

| Metric | No spec decode | With Eagle3 spec decode | Delta |
|--------|---------------|------------------------|-------|
| Output token throughput (tok/s) | 460 | **618** | **+34%** |
| Total token throughput (tok/s) | 4180 | **5614** | **+34%** |
| Request throughput (req/s) | 0.43 | **0.58** | **+35%** |
| Mean TPOT (ms) | 15.11 | **10.21** | **-32%** |
| Mean TTFT (ms) | 1724 | 2652 | +54% |
| Mean ITL (ms) | 15.70 | 27.62 | +76% |

Eagle3 speculative decoding improves output throughput by +34% at the cost of
higher TTFT and ITL (due to draft model overhead and verification batching).

## Long Context: ROCM_AITER_FA + Eagle3

Workload: random, input_len=80000, output_len=2048, concurrency=1, 80 prompts

| Metric | Value |
|--------|-------|
| Output token throughput (tok/s) | 73.4 |
| Total token throughput (tok/s) | 2919 |
| Request throughput (req/s) | 0.03 |
| Mean TPOT (ms) | 11.14 |
| Median TPOT (ms) | 10.75 |
| P99 TPOT (ms) | 25.72 |
| Mean TTFT (ms) | 5049 |
| Median TTFT (ms) | 5015 |
| P99 TTFT (ms) | 9857 |
| Mean ITL (ms) | 32.24 |
| P99 ITL (ms) | 54.09 |
| Acceptance rate (%) | 61.5 |
| Acceptance length | 2.85 |

Long-context acceptance rate (61.5%) is higher than short-context (59.6%),
indicating the drafter benefits from longer context for prediction accuracy.
