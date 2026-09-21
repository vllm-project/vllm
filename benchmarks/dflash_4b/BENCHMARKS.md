# Qwen3.5-4B benchmark progress

Investigation of [vLLM #49730](https://github.com/vllm-project/vllm/issues/49730), measured on September 21, 2026.
One H100 per run, TP1, FP8 target weights, BF16 compute, concurrency 1, AIPerf GSM8K, 10 warmup requests, and at most 256 output tokens with EOS enabled.
vLLM uses MRV2 and PyTorch 2.13; SGLang uses PyTorch 2.11.

## 100-request comparison

“Original” uses the PyTorch Mamba block-table gather; “Triton” replaces it with one kernel while retaining the original as `mamba_get_block_table_tensor_reference()`.
All 79 equivalence tests passed after switching callers.

| Engine / implementation | Mode | ITL p50 (ms) | Output tok/s | AL |
| --- | --- | ---: | ---: | ---: |
| vLLM original | Baseline | 3.18 | 301.6 | — |
| vLLM original | DFlash | 8.74 | 565.6 | 5.6523 |
| vLLM Triton | Baseline | 3.25 | 298.2 | — |
| vLLM Triton | DFlash | 8.19 | 608.6 | 5.6523 |
| SGLang | Baseline | 3.25 | 299.8 | — |
| SGLang | DFlash | 5.99 | 847.6 | 5.7091 |

ITL is the interval between streamed chunks, not TPOT.
AL includes the bonus token; vLLM uses counter deltas, whereas SGLang uses a sampled gauge with different weighting.
The Triton change improved DFlash throughput by 7.6% and reduced ITL by 6.3%, closing approximately 20% of the ITL gap to SGLang.
These are single runs on separate GPUs; the small baseline change is inconclusive.

## Wall-clock runtime

| Engine / implementation | Mode | Server startup (s) | Benchmark total (s) | Warmup (s) | 100 measured requests (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| vLLM original | Baseline | 50.0 | 109.3 | 8.8 | 84.9 |
| vLLM original | DFlash | 46.0 | 66.1 | 5.2 | 45.3 |
| vLLM Triton | Baseline | 46.0 | 110.4 | 8.6 | 85.8 |
| vLLM Triton | DFlash | 46.0 | 61.6 | 4.7 | 42.1 |
| SGLang | Baseline | 39.0 | 109.4 | 8.6 | 85.4 |
| SGLang | DFlash | 41.0 | 49.2 | 3.7 | 30.2 |

Startup includes model loading, compilation, and graph capture, with readiness polled every two seconds.
Benchmark total includes AIPerf setup, warmup, measured requests, and export.
Compilation cache state affects startup time.

## Earlier 30-request check

| vLLM original | ITL p50 (ms) | Output tok/s | AL | Startup (s) | Benchmark total (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 3.15 | 307.1 | — | 44.0 | 48.3 |
| DFlash | 9.21 | 578.4 | 6.05 | 114.0 | 34.5 |

These wall times were reconstructed from file timestamps and exclude a failed initial startup.
The 100-request runs above provide the current comparison.

Local logs and per-run summaries: [original and SGLang](results/quick-100-10/), [Triton](results/quick-100-10-triton/), and [30-request check](results/quick-30-10/).
Results directories are gitignored.
