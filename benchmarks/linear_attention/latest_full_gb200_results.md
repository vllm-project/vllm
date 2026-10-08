# Latest GDN/KDA full serving on GB200

Pins: frontend `7f85a1f74183e7205551c5795943d86956b346c6`, flashinfer `73b63f88ce24a38f6b2addc9a6f051b3e50cd659`, vllm `7c3c1a3f569c400e270f99ce498fdb0d5a499e9b`. Same vLLM PR #1.

Correctness/build gate: job 3269647; COMPLETED. Full measurements: partial.

Kimi and GLM use KDA; their vLLM-auto baseline is FlashKDA. Qwen uses GDN and its native vLLM-auto baseline. Each model compares vLLM auto, FlashInfer auto and FlashInfer cuDNN.

Prefill serving uses 8192 input/1 output. Full serving uses 8192 input/1024 output. C1/C8/C32/C128, six fresh ABC-CBA launches, three repeats per launch and a full warmup before every repeat. All timings are unprofiled. KDA state is BF16; GDN state is FP32. Kimi TP16, GLM TP4, Qwen3-Next TP2 and Qwen3.5 TP1.

Each server must pass finite generation and its own GSM64 gate before full GSM1319 and timing. A failed baseline score gate does not block independent FI arms. Each FI arm retains its own gate. Safety floors are not accuracy equivalence. Sequential variation limits statistical and causal claims.

## Prefill serving TTFT (8192/1, ms)

| Model | C | vLLM auto | FlashInfer auto | FlashInfer cuDNN |
| --- | ---: | ---: | ---: | ---: |
| Kimi-K3 (KDA) | 1 | Pending | Pending | Pending |
| Kimi-K3 (KDA) | 8 | Pending | Pending | Pending |
| Kimi-K3 (KDA) | 32 | Pending | Pending | Pending |
| Kimi-K3 (KDA) | 128 | Pending | Pending | Pending |
| GLM-5.3 (KDA) | 1 | Unavailable (failed run) | Unavailable (failed run) | Unavailable (failed run) |
| GLM-5.3 (KDA) | 8 | Unavailable (failed run) | Unavailable (failed run) | Unavailable (failed run) |
| GLM-5.3 (KDA) | 32 | Unavailable (failed run) | Unavailable (failed run) | Unavailable (failed run) |
| GLM-5.3 (KDA) | 128 | Unavailable (failed run) | Unavailable (failed run) | Unavailable (failed run) |
| Qwen3-Next (GDN) | 1 | 123.921 | 130.757 | 142.009 |
| Qwen3-Next (GDN) | 8 | 602.456 | 576.145 | 657.617 |
| Qwen3-Next (GDN) | 32 | 2315.848 | 2167.592 | 2420.889 |
| Qwen3-Next (GDN) | 128 | 7875.742 | 7706.646 | 8416.586 |
| Qwen3.5 (GDN) | 1 | Unavailable (failed run) | Unavailable (failed run) | Unavailable (failed run) |
| Qwen3.5 (GDN) | 8 | Unavailable (failed run) | Unavailable (failed run) | Unavailable (failed run) |
| Qwen3.5 (GDN) | 32 | Unavailable (failed run) | Unavailable (failed run) | Unavailable (failed run) |
| Qwen3.5 (GDN) | 128 | Unavailable (failed run) | Unavailable (failed run) | Unavailable (failed run) |

## Full serving TTFT (8192/1024, ms)

| Model | C | vLLM auto | FlashInfer auto | FlashInfer cuDNN |
| --- | ---: | ---: | ---: | ---: |
| Kimi-K3 (KDA) | 1 | Pending | Pending | Pending |
| Kimi-K3 (KDA) | 8 | Pending | Pending | Pending |
| Kimi-K3 (KDA) | 32 | Pending | Pending | Pending |
| Kimi-K3 (KDA) | 128 | Pending | Pending | Pending |
| GLM-5.3 (KDA) | 1 | Unavailable (failed run) | Unavailable (failed run) | Unavailable (failed run) |
| GLM-5.3 (KDA) | 8 | Unavailable (failed run) | Unavailable (failed run) | Unavailable (failed run) |
| GLM-5.3 (KDA) | 32 | Unavailable (failed run) | Unavailable (failed run) | Unavailable (failed run) |
| GLM-5.3 (KDA) | 128 | Unavailable (failed run) | Unavailable (failed run) | Unavailable (failed run) |
| Qwen3-Next (GDN) | 1 | 146.154 | 155.786 | 141.959 |
| Qwen3-Next (GDN) | 8 | 936.445 | 965.869 | 739.841 |
| Qwen3-Next (GDN) | 32 | 2834.217 | 3119.366 | 2683.102 |
| Qwen3-Next (GDN) | 128 | 11350.579 | 11139.986 | 9068.219 |
| Qwen3.5 (GDN) | 1 | Unavailable (failed run) | Unavailable (failed run) | Unavailable (failed run) |
| Qwen3.5 (GDN) | 8 | Unavailable (failed run) | Unavailable (failed run) | Unavailable (failed run) |
| Qwen3.5 (GDN) | 32 | Unavailable (failed run) | Unavailable (failed run) | Unavailable (failed run) |
| Qwen3.5 (GDN) | 128 | Unavailable (failed run) | Unavailable (failed run) | Unavailable (failed run) |

## Full serving TPOT (ms)

| Model | C | vLLM auto | FlashInfer auto | FlashInfer cuDNN |
| --- | ---: | ---: | ---: | ---: |
| Kimi-K3 (KDA) | 1 | Pending | Pending | Pending |
| Kimi-K3 (KDA) | 8 | Pending | Pending | Pending |
| Kimi-K3 (KDA) | 32 | Pending | Pending | Pending |
| Kimi-K3 (KDA) | 128 | Pending | Pending | Pending |
| GLM-5.3 (KDA) | 1 | Unavailable (failed run) | Unavailable (failed run) | Unavailable (failed run) |
| GLM-5.3 (KDA) | 8 | Unavailable (failed run) | Unavailable (failed run) | Unavailable (failed run) |
| GLM-5.3 (KDA) | 32 | Unavailable (failed run) | Unavailable (failed run) | Unavailable (failed run) |
| GLM-5.3 (KDA) | 128 | Unavailable (failed run) | Unavailable (failed run) | Unavailable (failed run) |
| Qwen3-Next (GDN) | 1 | 3.482 | 3.508 | 3.489 |
| Qwen3-Next (GDN) | 8 | 5.325 | 5.336 | 5.179 |
| Qwen3-Next (GDN) | 32 | 9.474 | 9.577 | 9.194 |
| Qwen3-Next (GDN) | 128 | 22.428 | 22.346 | 20.289 |
| Qwen3.5 (GDN) | 1 | Unavailable (failed run) | Unavailable (failed run) | Unavailable (failed run) |
| Qwen3.5 (GDN) | 8 | Unavailable (failed run) | Unavailable (failed run) | Unavailable (failed run) |
| Qwen3.5 (GDN) | 32 | Unavailable (failed run) | Unavailable (failed run) | Unavailable (failed run) |
| Qwen3.5 (GDN) | 128 | Unavailable (failed run) | Unavailable (failed run) | Unavailable (failed run) |

## Full output throughput (tokens/s)

| Model | C | vLLM auto | FlashInfer auto | FlashInfer cuDNN |
| --- | ---: | ---: | ---: | ---: |
| Kimi-K3 (KDA) | 1 | Pending | Pending | Pending |
| Kimi-K3 (KDA) | 8 | Pending | Pending | Pending |
| Kimi-K3 (KDA) | 32 | Pending | Pending | Pending |
| Kimi-K3 (KDA) | 128 | Pending | Pending | Pending |
| GLM-5.3 (KDA) | 1 | Unavailable (failed run) | Unavailable (failed run) | Unavailable (failed run) |
| GLM-5.3 (KDA) | 8 | Unavailable (failed run) | Unavailable (failed run) | Unavailable (failed run) |
| GLM-5.3 (KDA) | 32 | Unavailable (failed run) | Unavailable (failed run) | Unavailable (failed run) |
| GLM-5.3 (KDA) | 128 | Unavailable (failed run) | Unavailable (failed run) | Unavailable (failed run) |
| Qwen3-Next (GDN) | 1 | 276.122 | 273.643 | 276.130 |
| Qwen3-Next (GDN) | 8 | 1280.978 | 1272.342 | 1356.299 |
| Qwen3-Next (GDN) | 32 | 2619.147 | 2521.039 | 2699.890 |
| Qwen3-Next (GDN) | 128 | 3793.067 | 3822.434 | 4360.263 |
| Qwen3.5 (GDN) | 1 | Unavailable (failed run) | Unavailable (failed run) | Unavailable (failed run) |
| Qwen3.5 (GDN) | 8 | Unavailable (failed run) | Unavailable (failed run) | Unavailable (failed run) |
| Qwen3.5 (GDN) | 32 | Unavailable (failed run) | Unavailable (failed run) | Unavailable (failed run) |
| Qwen3.5 (GDN) | 128 | Unavailable (failed run) | Unavailable (failed run) | Unavailable (failed run) |

## Model correctness

| Model | Launch | GSM64 correct / 64 | Full GSM correct / 1319 | Full invalid |
| --- | --- | ---: | ---: | ---: |
| GLM-5.3 (KDA) | 00-candidate-vllm_auto | 57 | 1208 | 0 |

GLM-5.3 (KDA) runtime failures: `[{"arm": "01-candidate-flashinfer_auto", "error": "RuntimeError('Incomplete or nonfinite results: /lustre/fsw/coreai_libraries_cudnn/agopal/vllm-gdn-kda-20261001/kda-cpu-parity-20261005/serving-v27-latest-full/results/glm-GB200-3270870/01-candidate-flashinfer_auto/bench/8k1k-c128-r1.json')", "excluded_from_comparison": true}]`

| Qwen3.5 (GDN) | 00-candidate-vllm_auto | 59 | 1149 | 0 |
| Qwen3.5 (GDN) | 01-candidate-flashinfer_auto | 59 | 1132 | 0 |

Qwen3.5 (GDN) failures: `[{"arm": "02-candidate-flashinfer_cudnn", "reason": "GSM64 fails the declared minimum or protocol; timing stopped", "details": {"accuracy": 0.859375, "invalid_rate": 0.0, "latency": 15.682142404955812, "questions_per_second": 4.081075043660805, "total_output_tokens": 9243, "tokens_per_second": 589.3965098212003, "num_questions": 64, "num_shots": 5, "max_tokens": 256, "timestamp": 1791447369.895552}, "excluded_from_performance": true}]`

| Qwen3-Next (GDN) | 00-candidate-vllm_auto | 55 | 1142 | 1 |
| Qwen3-Next (GDN) | 01-candidate-flashinfer_auto | 56 | 1149 | 0 |
| Qwen3-Next (GDN) | 02-candidate-flashinfer_cudnn | 55 | 1136 | 0 |
| Qwen3-Next (GDN) | 03-candidate-flashinfer_cudnn | 55 | 1141 | 0 |
| Qwen3-Next (GDN) | 04-candidate-flashinfer_auto | 55 | 1149 | 1 |
| Qwen3-Next (GDN) | 05-candidate-vllm_auto | 55 | 1142 | 0 |

## Every repeat

| Model | Launch | C | Repeat | Prefill TTFT (ms) | Full TTFT (ms) | TPOT (ms) | Output tokens/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GLM-5.3 (KDA) | 00-candidate-vllm_auto | 1 | 1 | 160.945 | 175.227 | 5.338 | 181.676 |
| GLM-5.3 (KDA) | 00-candidate-vllm_auto | 1 | 2 | 159.313 | 179.769 | 5.339 | 181.475 |
| GLM-5.3 (KDA) | 00-candidate-vllm_auto | 1 | 3 | 157.872 | 175.800 | 5.351 | 181.240 |
| GLM-5.3 (KDA) | 00-candidate-vllm_auto | 8 | 1 | 677.913 | 866.755 | 8.077 | 894.877 |
| GLM-5.3 (KDA) | 00-candidate-vllm_auto | 8 | 2 | 677.275 | 864.650 | 8.144 | 888.431 |
| GLM-5.3 (KDA) | 00-candidate-vllm_auto | 8 | 3 | 677.632 | 865.702 | 8.145 | 888.125 |
| GLM-5.3 (KDA) | 00-candidate-vllm_auto | 32 | 1 | 2436.412 | 2848.592 | 13.667 | 1931.412 |
| GLM-5.3 (KDA) | 00-candidate-vllm_auto | 32 | 2 | 2437.162 | 2844.902 | 13.869 | 1908.477 |
| GLM-5.3 (KDA) | 00-candidate-vllm_auto | 32 | 3 | 2436.730 | 2835.013 | 13.776 | 1920.166 |
| GLM-5.3 (KDA) | 00-candidate-vllm_auto | 128 | 1 | 9474.298 | 10935.101 | 26.350 | 3389.545 |
| GLM-5.3 (KDA) | 00-candidate-vllm_auto | 128 | 2 | 9474.862 | 10806.593 | 26.308 | 3406.096 |
| GLM-5.3 (KDA) | 00-candidate-vllm_auto | 128 | 3 | 9483.177 | 10805.806 | 26.294 | 3406.627 |
| Qwen3.5 (GDN) | 00-candidate-vllm_auto | 1 | 1 | 103.529 | 118.121 | 3.303 | 292.777 |
| Qwen3.5 (GDN) | 00-candidate-vllm_auto | 1 | 2 | 102.866 | 119.858 | 3.302 | 292.703 |
| Qwen3.5 (GDN) | 00-candidate-vllm_auto | 1 | 3 | 102.324 | 121.985 | 3.303 | 292.447 |
| Qwen3.5 (GDN) | 00-candidate-vllm_auto | 8 | 1 | 439.842 | 585.751 | 5.689 | 1275.428 |
| Qwen3.5 (GDN) | 00-candidate-vllm_auto | 8 | 2 | 439.447 | 561.302 | 5.718 | 1274.374 |
| Qwen3.5 (GDN) | 00-candidate-vllm_auto | 8 | 3 | 439.455 | 583.792 | 5.609 | 1292.201 |
| Qwen3.5 (GDN) | 00-candidate-vllm_auto | 32 | 1 | 1557.537 | 1881.127 | 10.359 | 2604.929 |
| Qwen3.5 (GDN) | 00-candidate-vllm_auto | 32 | 2 | 1550.440 | 1838.545 | 10.522 | 2579.491 |
| Qwen3.5 (GDN) | 00-candidate-vllm_auto | 32 | 3 | 1555.966 | 1854.200 | 10.438 | 2594.171 |
| Qwen3.5 (GDN) | 00-candidate-vllm_auto | 128 | 1 | 6094.413 | 6532.055 | 21.748 | 4451.956 |
| Qwen3.5 (GDN) | 00-candidate-vllm_auto | 128 | 2 | 6127.046 | 7042.013 | 22.281 | 4298.144 |
| Qwen3.5 (GDN) | 00-candidate-vllm_auto | 128 | 3 | 6100.629 | 6875.600 | 22.026 | 4361.016 |
| Qwen3.5 (GDN) | 01-candidate-flashinfer_auto | 1 | 1 | 100.024 | 115.822 | 3.302 | 293.069 |
| Qwen3.5 (GDN) | 01-candidate-flashinfer_auto | 1 | 2 | 99.363 | 116.463 | 3.300 | 293.140 |
| Qwen3.5 (GDN) | 01-candidate-flashinfer_auto | 1 | 3 | 99.387 | 116.856 | 3.301 | 293.095 |
| Qwen3.5 (GDN) | 01-candidate-flashinfer_auto | 8 | 1 | 430.952 | 559.131 | 5.674 | 1283.871 |
| Qwen3.5 (GDN) | 01-candidate-flashinfer_auto | 8 | 2 | 431.946 | 552.700 | 5.683 | 1283.180 |
| Qwen3.5 (GDN) | 01-candidate-flashinfer_auto | 8 | 3 | 431.208 | 553.369 | 5.604 | 1299.642 |
| Qwen3.5 (GDN) | 01-candidate-flashinfer_auto | 32 | 1 | 1540.665 | 1716.000 | 10.464 | 2616.808 |
| Qwen3.5 (GDN) | 01-candidate-flashinfer_auto | 32 | 2 | 1544.243 | 1733.767 | 10.437 | 2619.566 |
| Qwen3.5 (GDN) | 01-candidate-flashinfer_auto | 32 | 3 | 1542.661 | 1889.901 | 10.605 | 2552.300 |
| Qwen3.5 (GDN) | 01-candidate-flashinfer_auto | 128 | 1 | 6042.239 | 6808.663 | 21.686 | 4421.343 |
| Qwen3.5 (GDN) | 01-candidate-flashinfer_auto | 128 | 2 | 6039.840 | 6381.395 | 21.558 | 4505.589 |
| Qwen3.5 (GDN) | 01-candidate-flashinfer_auto | 128 | 3 | 6146.243 | 6459.093 | 21.646 | 4481.571 |
| Qwen3-Next (GDN) | 00-candidate-vllm_auto | 1 | 1 | 131.809 | 131.492 | 3.456 | 279.192 |
| Qwen3-Next (GDN) | 00-candidate-vllm_auto | 1 | 2 | 143.208 | 133.319 | 3.456 | 279.073 |
| Qwen3-Next (GDN) | 00-candidate-vllm_auto | 1 | 3 | 147.857 | 164.149 | 3.522 | 271.811 |
| Qwen3-Next (GDN) | 00-candidate-vllm_auto | 8 | 1 | 711.533 | 1021.157 | 5.082 | 1313.586 |
| Qwen3-Next (GDN) | 00-candidate-vllm_auto | 8 | 2 | 741.679 | 1018.102 | 5.483 | 1232.992 |
| Qwen3-Next (GDN) | 00-candidate-vllm_auto | 8 | 3 | 745.937 | 1044.813 | 5.467 | 1231.122 |
| Qwen3-Next (GDN) | 00-candidate-vllm_auto | 32 | 1 | 2624.680 | 2183.167 | 8.700 | 2933.644 |
| Qwen3-Next (GDN) | 00-candidate-vllm_auto | 32 | 2 | 2724.606 | 3604.120 | 10.394 | 2286.967 |
| Qwen3-Next (GDN) | 00-candidate-vllm_auto | 32 | 3 | 2709.390 | 3453.695 | 9.900 | 2396.779 |
| Qwen3-Next (GDN) | 00-candidate-vllm_auto | 128 | 1 | 10735.872 | 13315.870 | 23.509 | 3462.694 |
| Qwen3-Next (GDN) | 00-candidate-vllm_auto | 128 | 2 | 7851.136 | 12774.337 | 23.774 | 3485.918 |
| Qwen3-Next (GDN) | 00-candidate-vllm_auto | 128 | 3 | 7358.408 | 12457.409 | 23.802 | 3509.798 |
| Qwen3-Next (GDN) | 01-candidate-flashinfer_auto | 1 | 1 | 120.587 | 130.375 | 3.442 | 280.355 |
| Qwen3-Next (GDN) | 01-candidate-flashinfer_auto | 1 | 2 | 129.580 | 134.446 | 3.443 | 279.992 |
| Qwen3-Next (GDN) | 01-candidate-flashinfer_auto | 1 | 3 | 117.005 | 187.722 | 3.695 | 258.035 |
| Qwen3-Next (GDN) | 01-candidate-flashinfer_auto | 8 | 1 | 534.839 | 1007.031 | 5.278 | 1275.441 |
| Qwen3-Next (GDN) | 01-candidate-flashinfer_auto | 8 | 2 | 548.743 | 980.725 | 5.414 | 1253.454 |
| Qwen3-Next (GDN) | 01-candidate-flashinfer_auto | 8 | 3 | 556.810 | 958.701 | 5.488 | 1243.173 |
| Qwen3-Next (GDN) | 01-candidate-flashinfer_auto | 32 | 1 | 2060.507 | 3137.307 | 9.303 | 2571.778 |
| Qwen3-Next (GDN) | 01-candidate-flashinfer_auto | 32 | 2 | 2076.113 | 3177.550 | 10.009 | 2425.990 |
| Qwen3-Next (GDN) | 01-candidate-flashinfer_auto | 32 | 3 | 2076.641 | 3302.642 | 9.697 | 2461.732 |
| Qwen3-Next (GDN) | 01-candidate-flashinfer_auto | 128 | 1 | 7943.339 | 12488.623 | 23.220 | 3565.216 |
| Qwen3-Next (GDN) | 01-candidate-flashinfer_auto | 128 | 2 | 7708.500 | 12031.591 | 22.731 | 3662.361 |
| Qwen3-Next (GDN) | 01-candidate-flashinfer_auto | 128 | 3 | 7493.742 | 11838.988 | 23.226 | 3629.755 |
| Qwen3-Next (GDN) | 02-candidate-flashinfer_cudnn | 1 | 1 | 142.335 | 127.884 | 3.426 | 281.832 |
| Qwen3-Next (GDN) | 02-candidate-flashinfer_cudnn | 1 | 2 | 147.331 | 126.500 | 3.426 | 281.954 |
| Qwen3-Next (GDN) | 02-candidate-flashinfer_cudnn | 1 | 3 | 149.041 | 135.058 | 3.429 | 281.064 |
| Qwen3-Next (GDN) | 02-candidate-flashinfer_cudnn | 8 | 1 | 671.356 | 639.046 | 4.970 | 1427.308 |
| Qwen3-Next (GDN) | 02-candidate-flashinfer_cudnn | 8 | 2 | 680.920 | 626.593 | 5.044 | 1411.642 |
| Qwen3-Next (GDN) | 02-candidate-flashinfer_cudnn | 8 | 3 | 677.618 | 604.364 | 5.021 | 1422.759 |
| Qwen3-Next (GDN) | 02-candidate-flashinfer_cudnn | 32 | 1 | 2461.820 | 2820.089 | 9.170 | 2667.499 |
| Qwen3-Next (GDN) | 02-candidate-flashinfer_cudnn | 32 | 2 | 2473.743 | 2318.490 | 9.105 | 2796.021 |
| Qwen3-Next (GDN) | 02-candidate-flashinfer_cudnn | 32 | 3 | 2471.904 | 2991.292 | 9.263 | 2609.922 |
| Qwen3-Next (GDN) | 02-candidate-flashinfer_cudnn | 128 | 1 | 9643.546 | 11211.294 | 22.540 | 3766.924 |
| Qwen3-Next (GDN) | 02-candidate-flashinfer_cudnn | 128 | 2 | 6565.128 | 10927.178 | 22.159 | 3842.216 |
| Qwen3-Next (GDN) | 02-candidate-flashinfer_cudnn | 128 | 3 | 8243.946 | 7966.404 | 19.271 | 4649.204 |
| Qwen3-Next (GDN) | 03-candidate-flashinfer_cudnn | 1 | 1 | 137.356 | 134.041 | 3.467 | 278.157 |
| Qwen3-Next (GDN) | 03-candidate-flashinfer_cudnn | 1 | 2 | 136.020 | 157.226 | 3.471 | 276.128 |
| Qwen3-Next (GDN) | 03-candidate-flashinfer_cudnn | 1 | 3 | 139.972 | 171.043 | 3.717 | 257.643 |
| Qwen3-Next (GDN) | 03-candidate-flashinfer_cudnn | 8 | 1 | 631.794 | 820.936 | 5.272 | 1314.755 |
| Qwen3-Next (GDN) | 03-candidate-flashinfer_cudnn | 8 | 2 | 641.313 | 856.927 | 5.387 | 1283.185 |
| Qwen3-Next (GDN) | 03-candidate-flashinfer_cudnn | 8 | 3 | 642.705 | 891.181 | 5.378 | 1278.144 |
| Qwen3-Next (GDN) | 03-candidate-flashinfer_cudnn | 32 | 1 | 2360.998 | 2974.003 | 9.285 | 2610.300 |
| Qwen3-Next (GDN) | 03-candidate-flashinfer_cudnn | 32 | 2 | 2374.722 | 2952.899 | 9.715 | 2524.887 |
| Qwen3-Next (GDN) | 03-candidate-flashinfer_cudnn | 32 | 3 | 2382.149 | 2041.841 | 8.629 | 2990.713 |
| Qwen3-Next (GDN) | 03-candidate-flashinfer_cudnn | 128 | 1 | 9278.267 | 7975.746 | 19.469 | 4613.259 |
| Qwen3-Next (GDN) | 03-candidate-flashinfer_cudnn | 128 | 2 | 9332.269 | 7597.872 | 18.679 | 4816.560 |
| Qwen3-Next (GDN) | 03-candidate-flashinfer_cudnn | 128 | 3 | 7436.362 | 8730.818 | 19.618 | 4473.416 |
| Qwen3-Next (GDN) | 04-candidate-flashinfer_auto | 1 | 1 | 139.296 | 138.259 | 3.478 | 277.032 |
| Qwen3-Next (GDN) | 04-candidate-flashinfer_auto | 1 | 2 | 139.567 | 175.667 | 3.488 | 273.453 |
| Qwen3-Next (GDN) | 04-candidate-flashinfer_auto | 1 | 3 | 138.508 | 168.246 | 3.502 | 272.989 |
| Qwen3-Next (GDN) | 04-candidate-flashinfer_auto | 8 | 1 | 599.960 | 962.007 | 5.126 | 1316.610 |
| Qwen3-Next (GDN) | 04-candidate-flashinfer_auto | 8 | 2 | 599.768 | 948.397 | 5.398 | 1262.759 |
| Qwen3-Next (GDN) | 04-candidate-flashinfer_auto | 8 | 3 | 616.747 | 938.353 | 5.310 | 1282.616 |
| Qwen3-Next (GDN) | 04-candidate-flashinfer_auto | 32 | 1 | 2262.581 | 3098.511 | 9.422 | 2555.538 |
| Qwen3-Next (GDN) | 04-candidate-flashinfer_auto | 32 | 2 | 2273.933 | 2965.953 | 9.438 | 2578.092 |
| Qwen3-Next (GDN) | 04-candidate-flashinfer_auto | 32 | 3 | 2255.777 | 3034.232 | 9.593 | 2533.105 |
| Qwen3-Next (GDN) | 04-candidate-flashinfer_auto | 128 | 1 | 7331.892 | 11259.574 | 23.050 | 3705.842 |
| Qwen3-Next (GDN) | 04-candidate-flashinfer_auto | 128 | 2 | 6985.453 | 10923.848 | 22.271 | 3829.869 |
| Qwen3-Next (GDN) | 04-candidate-flashinfer_auto | 128 | 3 | 8776.949 | 8297.290 | 19.581 | 4541.564 |
| Qwen3-Next (GDN) | 05-candidate-vllm_auto | 1 | 1 | 106.849 | 123.363 | 3.491 | 277.122 |
| Qwen3-Next (GDN) | 05-candidate-vllm_auto | 1 | 2 | 107.533 | 163.073 | 3.482 | 274.861 |
| Qwen3-Next (GDN) | 05-candidate-vllm_auto | 1 | 3 | 106.272 | 161.531 | 3.486 | 274.673 |
| Qwen3-Next (GDN) | 05-candidate-vllm_auto | 8 | 1 | 471.048 | 850.153 | 5.320 | 1298.478 |
| Qwen3-Next (GDN) | 05-candidate-vllm_auto | 8 | 2 | 472.757 | 841.493 | 5.290 | 1306.632 |
| Qwen3-Next (GDN) | 05-candidate-vllm_auto | 8 | 3 | 471.783 | 842.950 | 5.305 | 1303.056 |
| Qwen3-Next (GDN) | 05-candidate-vllm_auto | 32 | 1 | 1667.653 | 2781.087 | 9.365 | 2631.350 |
| Qwen3-Next (GDN) | 05-candidate-vllm_auto | 32 | 2 | 2253.928 | 2814.882 | 9.647 | 2566.063 |
| Qwen3-Next (GDN) | 05-candidate-vllm_auto | 32 | 3 | 1914.830 | 2168.349 | 8.837 | 2900.078 |
| Qwen3-Next (GDN) | 05-candidate-vllm_auto | 128 | 1 | 7573.132 | 8932.334 | 20.282 | 4339.227 |
| Qwen3-Next (GDN) | 05-candidate-vllm_auto | 128 | 2 | 7222.476 | 10338.309 | 21.545 | 3985.089 |
| Qwen3-Next (GDN) | 05-candidate-vllm_auto | 128 | 3 | 6513.425 | 10285.214 | 21.657 | 3975.678 |

All prompts, tokens, warmups, ranges, sources, model scores and failures remain in the machine-readable audit. No historical samples are substituted.
