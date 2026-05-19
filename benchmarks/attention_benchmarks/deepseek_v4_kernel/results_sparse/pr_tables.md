# DSv4 sparse dequant kernel — PR-style benchmark

- GPU: **NVIDIA B200** (sm_100)
- block_size: 64, iters: 200
- Bytes/row = 584 (FP8+BF16+scale read) + 1024 (BF16 written) = 1608
- `*_us` columns are median µs; `*_p99_us` is the 99th percentile; `*_min_us` is the fastest sample (no-interference lower bound). All over `iters` cuda-event samples after 5 warmups. Median is the primary signal; p99 and min bound the noise envelope.

## Table A — Sparse Triton vs Sparse CuteDSL

Identical inputs (all slot ids of a freshly populated cache). Mirrors PR #42236 Table 1's `k_len` shapes.

| k_len | triton_us | triton_p99_us | triton_min_us | cutedsl_us | cutedsl_p99_us | cutedsl_min_us | triton_GB/s | cutedsl_GB/s | speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 20.22 | 37.70 | 16.64 | 11.58 | 23.46 | 8.93 | 0.1 | 0.1 | 1.75x |
| 8 | 17.63 | 46.82 | 14.43 | 10.75 | 21.50 | 8.48 | 0.7 | 1.2 | 1.64x |
| 32 | 23.49 | 37.34 | 15.84 | 11.17 | 16.80 | 8.70 | 2.2 | 4.6 | 2.10x |
| 128 | 18.62 | 27.55 | 16.35 | 12.42 | 16.54 | 9.02 | 11.1 | 16.6 | 1.50x |
| 512 | 20.00 | 35.62 | 16.80 | 12.90 | 19.90 | 10.30 | 41.2 | 63.8 | 1.55x |
| 2048 | 20.10 | 25.92 | 17.15 | 13.92 | 26.21 | 10.78 | 163.9 | 236.6 | 1.44x |
| 8192 | 21.12 | 29.18 | 18.85 | 12.80 | 28.10 | 11.65 | 623.7 | 1029.1 | 1.65x |
| 16384 | 23.17 | 36.32 | 22.59 | 16.19 | 17.70 | 14.37 | 1137.1 | 1627.1 | 1.43x |
| 32000 | 36.58 | 37.70 | 34.69 | 20.96 | 23.10 | 19.62 | 1406.8 | 2455.0 | 1.75x |
| 262144 | 227.36 | 230.11 | 226.72 | 110.46 | 112.61 | 108.06 | 1854.0 | 3816.0 | 2.06x |
| batched [97, 1024, 8192, 16384] | 30.69 | 31.84 | 29.34 | 18.37 | 20.64 | 17.38 | 1346.5 | 2249.6 | 1.67x |

## Table B — Sparse CuteDSL vs Dense CuteDSL across sparsity

Same K-cache; sparse path samples a random subset of slot ids at fractions `f ∈ {3, 6, 12, 25, 50, 100}%`. The dense column is `dequantize_and_gather_k_cache_cutedsl(seq_lens=[N_dense])`. `speedup = dense_us / sparse_us`; values >1 mean the sparse kernel saves wall time over the dense baseline at that sparsity.

| N_dense | fraction | rows | dense_us | dense_p99_us | dense_min_us | sparse_us | sparse_p99_us | sparse_min_us | dense_GB/s | sparse_GB/s | speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 3% | 246 | 12.42 | 13.82 | 11.58 | 10.37 | 14.59 | 8.61 | 1060.9 | 38.2 | 1.20x |
| 8192 | 6% | 492 | 12.70 | 15.65 | 11.33 | 10.85 | 15.01 | 8.70 | 1036.9 | 72.9 | 1.17x |
| 8192 | 12% | 983 | 12.32 | 13.38 | 11.62 | 10.53 | 14.78 | 8.93 | 1069.2 | 150.1 | 1.17x |
| 8192 | 25% | 2048 | 12.42 | 13.82 | 11.39 | 11.14 | 15.33 | 9.25 | 1060.9 | 295.7 | 1.11x |
| 8192 | 50% | 4096 | 12.93 | 17.22 | 11.33 | 12.16 | 16.51 | 10.02 | 1018.9 | 541.6 | 1.06x |
| 8192 | 100% | 8192 | 13.02 | 19.36 | 11.23 | 12.38 | 13.47 | 11.46 | 1011.4 | 1063.7 | 1.05x |
| 16384 | 3% | 492 | 14.91 | 17.22 | 13.82 | 10.59 | 13.25 | 8.77 | 1766.7 | 74.7 | 1.41x |
| 16384 | 6% | 983 | 15.78 | 17.09 | 13.86 | 10.53 | 12.90 | 8.80 | 1670.0 | 150.1 | 1.50x |
| 16384 | 12% | 1966 | 14.59 | 17.02 | 13.82 | 10.85 | 14.34 | 9.12 | 1805.5 | 291.4 | 1.35x |
| 16384 | 25% | 4096 | 14.66 | 17.15 | 13.76 | 10.72 | 15.62 | 9.60 | 1797.6 | 614.4 | 1.37x |
| 16384 | 50% | 8192 | 15.10 | 16.96 | 13.82 | 12.32 | 13.57 | 11.46 | 1744.3 | 1069.2 | 1.23x |
| 16384 | 100% | 16384 | 14.50 | 17.86 | 13.82 | 16.06 | 17.18 | 13.92 | 1817.4 | 1640.0 | 0.90x |
| 32768 | 3% | 983 | 20.35 | 22.56 | 19.52 | 11.65 | 16.26 | 9.12 | 2589.0 | 135.7 | 1.75x |
| 32768 | 6% | 1966 | 20.32 | 22.53 | 19.58 | 11.81 | 16.67 | 9.82 | 2593.1 | 267.7 | 1.72x |
| 32768 | 12% | 3932 | 20.29 | 22.72 | 19.74 | 11.97 | 16.26 | 9.89 | 2597.1 | 528.3 | 1.70x |
| 32768 | 25% | 8192 | 20.35 | 22.27 | 19.78 | 12.80 | 17.50 | 11.26 | 2589.0 | 1029.1 | 1.59x |
| 32768 | 50% | 16384 | 20.29 | 22.40 | 19.74 | 16.13 | 17.41 | 14.24 | 2597.1 | 1633.5 | 1.26x |
| 32768 | 100% | 32768 | 20.35 | 23.04 | 19.74 | 20.80 | 23.36 | 19.90 | 2589.0 | 2533.2 | 0.98x |

## Table C — End-to-end TTFT

_Pending — kernel-only PR. End-to-end TTFT numbers depend on the C4A prefill pipeline integration (dedup of `topk_indices` per request, sparse dequant call, mapping into the compact `kv` buffer), which is scheduled as a follow-up PR._
