# Latest GDN/KDA full serving on GB200

Pins: frontend `7f85a1f74183e7205551c5795943d86956b346c6`, flashinfer `73b63f88ce24a38f6b2addc9a6f051b3e50cd659`, vllm `7c3c1a3f569c400e270f99ce498fdb0d5a499e9b`. Same vLLM PR #1.

Correctness/build gate: job 3269647; COMPLETED. Full measurements: partial.

Kimi and GLM use KDA; their vLLM-auto baseline is FlashKDA. Qwen uses GDN and its native vLLM-auto baseline. Each model compares vLLM auto, FlashInfer auto and FlashInfer cuDNN.

Prefill serving uses 8192 input/1 output. Full serving uses 8192 input/1024 output. C1/C8/C32/C128, six fresh ABC-CBA launches, three repeats per launch and a full warmup before every repeat. All timings are unprofiled. KDA state is BF16; GDN state is FP32. Kimi TP16, GLM TP4, Qwen3-Next TP2 and Qwen3.5 TP1.

Each server must pass finite generation and its own GSM64 gate before full GSM1319 and timing. A failed baseline score gate does not block independent FI arms. Each FI arm retains its own gate. Safety floors are not accuracy equivalence. Sequential variation limits statistical and causal claims.

## Prefill serving TTFT (8192/1, ms)

| Model | C | vLLM auto | FlashInfer auto | FlashInfer cuDNN |
| --- | ---: | ---: | ---: | ---: |
| Kimi-K3 (KDA) | 1 | 289.221 | 333.196 | 274.394 |
| Kimi-K3 (KDA) | 8 | 1279.490 | 1443.067 | 1210.756 |
| Kimi-K3 (KDA) | 32 | 4742.299 | 5351.686 | 4419.455 |
| Kimi-K3 (KDA) | 128 | 18412.276 | 20009.054 | 20906.159 |
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
| Kimi-K3 (KDA) | 1 | 313.641 | 335.635 | 348.196 |
| Kimi-K3 (KDA) | 8 | 1769.013 | 1917.690 | 1890.517 |
| Kimi-K3 (KDA) | 32 | 5769.200 | 5934.229 | 6353.972 |
| Kimi-K3 (KDA) | 128 | 22531.251 | 20600.418 | 22893.252 |
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
| Kimi-K3 (KDA) | 1 | 8.096 | 8.037 | 8.073 |
| Kimi-K3 (KDA) | 8 | 13.010 | 11.688 | 13.131 |
| Kimi-K3 (KDA) | 32 | 21.887 | 21.054 | 22.405 |
| Kimi-K3 (KDA) | 128 | 58.247 | 55.325 | 58.737 |
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
| Kimi-K3 (KDA) | 1 | 119.116 | 119.649 | 118.966 |
| Kimi-K3 (KDA) | 8 | 542.008 | 588.946 | 533.271 |
| Kimi-K3 (KDA) | 32 | 1155.787 | 1183.941 | 1111.507 |
| Kimi-K3 (KDA) | 128 | 1443.019 | 1536.263 | 1437.274 |
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
| Kimi-K3 (KDA) | 00-candidate-vllm_auto | 64 | 1259 | 1 |
| Kimi-K3 (KDA) | 01-candidate-flashinfer_auto | 64 | 1254 | 1 |
| Kimi-K3 (KDA) | 02-candidate-flashinfer_cudnn | 64 | 1251 | 1 |
| Kimi-K3 (KDA) | 03-candidate-flashinfer_cudnn | 64 | 1247 | 1 |
| Kimi-K3 (KDA) | 04-candidate-flashinfer_auto | 64 | 1246 | 2 |
| Kimi-K3 (KDA) | 05-candidate-vllm_auto | 64 | 1248 | 1 |
| Qwen3.5 (GDN) | 00-candidate-vllm_auto | 59 | 1149 | 0 |
| Qwen3.5 (GDN) | 01-candidate-flashinfer_auto | 59 | 1132 | 0 |
| Qwen3-Next (GDN) | 00-candidate-vllm_auto | 55 | 1142 | 1 |
| Qwen3-Next (GDN) | 01-candidate-flashinfer_auto | 56 | 1149 | 0 |
| Qwen3-Next (GDN) | 02-candidate-flashinfer_cudnn | 55 | 1136 | 0 |
| Qwen3-Next (GDN) | 03-candidate-flashinfer_cudnn | 55 | 1141 | 0 |
| Qwen3-Next (GDN) | 04-candidate-flashinfer_auto | 55 | 1149 | 1 |
| Qwen3-Next (GDN) | 05-candidate-vllm_auto | 55 | 1142 | 0 |

GLM-5.3 (KDA) runtime failures: `[{"arm": "01-candidate-flashinfer_auto", "error": "RuntimeError('Incomplete or nonfinite results: /lustre/fsw/coreai_libraries_cudnn/agopal/vllm-gdn-kda-20261001/kda-cpu-parity-20261005/serving-v27-latest-full/results/glm-GB200-3270870/01-candidate-flashinfer_auto/bench/8k1k-c128-r1.json')", "excluded_from_comparison": true}]`

Qwen3.5 (GDN) failures: `[{"arm": "02-candidate-flashinfer_cudnn", "reason": "GSM64 fails the declared minimum or protocol; timing stopped", "details": {"accuracy": 0.859375, "invalid_rate": 0.0, "latency": 15.682142404955812, "questions_per_second": 4.081075043660805, "total_output_tokens": 9243, "tokens_per_second": 589.3965098212003, "num_questions": 64, "num_shots": 5, "max_tokens": 256, "timestamp": 1791447369.895552}, "excluded_from_performance": true}]`

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
| Kimi-K3 (KDA) | 00-candidate-vllm_auto | 1 | 1 | 290.506 | 311.993 | 8.114 | 118.893 |
| Kimi-K3 (KDA) | 00-candidate-vllm_auto | 1 | 2 | 289.585 | 310.836 | 8.072 | 119.506 |
| Kimi-K3 (KDA) | 00-candidate-vllm_auto | 1 | 3 | 288.536 | 310.398 | 8.072 | 119.513 |
| Kimi-K3 (KDA) | 00-candidate-vllm_auto | 8 | 1 | 1276.877 | 1564.744 | 12.817 | 556.713 |
| Kimi-K3 (KDA) | 00-candidate-vllm_auto | 8 | 2 | 1278.116 | 1748.953 | 12.991 | 543.331 |
| Kimi-K3 (KDA) | 00-candidate-vllm_auto | 8 | 3 | 1283.121 | 1884.940 | 13.173 | 531.942 |
| Kimi-K3 (KDA) | 00-candidate-vllm_auto | 32 | 1 | 4668.324 | 6091.466 | 22.245 | 1127.757 |
| Kimi-K3 (KDA) | 00-candidate-vllm_auto | 32 | 2 | 4658.207 | 5781.604 | 21.935 | 1152.691 |
| Kimi-K3 (KDA) | 00-candidate-vllm_auto | 32 | 3 | 4714.690 | 6330.195 | 22.282 | 1117.110 |
| Kimi-K3 (KDA) | 00-candidate-vllm_auto | 128 | 1 | 18135.231 | 23030.108 | 58.807 | 1426.145 |
| Kimi-K3 (KDA) | 00-candidate-vllm_auto | 128 | 2 | 18161.593 | 23235.774 | 59.064 | 1418.765 |
| Kimi-K3 (KDA) | 00-candidate-vllm_auto | 128 | 3 | 18164.665 | 22109.773 | 58.009 | 1452.652 |
| Kimi-K3 (KDA) | 01-candidate-flashinfer_auto | 1 | 1 | 334.116 | 332.517 | 8.026 | 119.862 |
| Kimi-K3 (KDA) | 01-candidate-flashinfer_auto | 1 | 2 | 334.239 | 337.250 | 8.053 | 119.396 |
| Kimi-K3 (KDA) | 01-candidate-flashinfer_auto | 1 | 3 | 335.415 | 350.791 | 8.054 | 119.207 |
| Kimi-K3 (KDA) | 01-candidate-flashinfer_auto | 8 | 1 | 1457.997 | 1982.283 | 11.812 | 580.886 |
| Kimi-K3 (KDA) | 01-candidate-flashinfer_auto | 8 | 2 | 1481.681 | 1889.677 | 11.664 | 591.132 |
| Kimi-K3 (KDA) | 01-candidate-flashinfer_auto | 8 | 3 | 1512.211 | 1842.122 | 11.608 | 595.622 |
| Kimi-K3 (KDA) | 01-candidate-flashinfer_auto | 32 | 1 | 5468.069 | 5933.960 | 21.266 | 1174.510 |
| Kimi-K3 (KDA) | 01-candidate-flashinfer_auto | 32 | 2 | 5486.649 | 5875.036 | 20.857 | 1195.124 |
| Kimi-K3 (KDA) | 01-candidate-flashinfer_auto | 32 | 3 | 5398.187 | 5717.822 | 20.814 | 1203.923 |
| Kimi-K3 (KDA) | 01-candidate-flashinfer_auto | 128 | 1 | 20750.781 | 23397.189 | 58.338 | 1435.969 |
| Kimi-K3 (KDA) | 01-candidate-flashinfer_auto | 128 | 2 | 20001.968 | 21005.157 | 55.720 | 1520.779 |
| Kimi-K3 (KDA) | 01-candidate-flashinfer_auto | 128 | 3 | 19443.992 | 19945.522 | 54.602 | 1559.533 |
| Kimi-K3 (KDA) | 02-candidate-flashinfer_cudnn | 1 | 1 | 275.649 | 348.851 | 8.077 | 118.899 |
| Kimi-K3 (KDA) | 02-candidate-flashinfer_cudnn | 1 | 2 | 273.866 | 349.338 | 8.060 | 119.137 |
| Kimi-K3 (KDA) | 02-candidate-flashinfer_cudnn | 1 | 3 | 273.160 | 344.336 | 8.046 | 119.408 |
| Kimi-K3 (KDA) | 02-candidate-flashinfer_cudnn | 8 | 1 | 1212.555 | 1885.813 | 13.061 | 535.928 |
| Kimi-K3 (KDA) | 02-candidate-flashinfer_cudnn | 8 | 2 | 1209.650 | 1817.413 | 13.037 | 539.209 |
| Kimi-K3 (KDA) | 02-candidate-flashinfer_cudnn | 8 | 3 | 1209.994 | 1920.487 | 13.179 | 530.526 |
| Kimi-K3 (KDA) | 02-candidate-flashinfer_cudnn | 32 | 1 | 4409.508 | 6498.363 | 22.390 | 1106.569 |
| Kimi-K3 (KDA) | 02-candidate-flashinfer_cudnn | 32 | 2 | 4409.817 | 6382.111 | 22.348 | 1112.591 |
| Kimi-K3 (KDA) | 02-candidate-flashinfer_cudnn | 32 | 3 | 4471.935 | 6161.560 | 22.343 | 1121.214 |
| Kimi-K3 (KDA) | 02-candidate-flashinfer_cudnn | 128 | 1 | 17178.413 | 23528.595 | 59.031 | 1421.205 |
| Kimi-K3 (KDA) | 02-candidate-flashinfer_cudnn | 128 | 2 | 21948.984 | 23840.441 | 60.398 | 1395.785 |
| Kimi-K3 (KDA) | 02-candidate-flashinfer_cudnn | 128 | 3 | 21495.367 | 22462.023 | 58.152 | 1452.905 |
| Kimi-K3 (KDA) | 03-candidate-flashinfer_cudnn | 1 | 1 | 275.832 | 361.369 | 8.051 | 119.092 |
| Kimi-K3 (KDA) | 03-candidate-flashinfer_cudnn | 1 | 2 | 274.041 | 348.143 | 8.094 | 118.671 |
| Kimi-K3 (KDA) | 03-candidate-flashinfer_cudnn | 1 | 3 | 273.818 | 337.137 | 8.111 | 118.589 |
| Kimi-K3 (KDA) | 03-candidate-flashinfer_cudnn | 8 | 1 | 1210.697 | 1840.629 | 13.106 | 535.885 |
| Kimi-K3 (KDA) | 03-candidate-flashinfer_cudnn | 8 | 2 | 1210.627 | 1933.426 | 13.159 | 530.798 |
| Kimi-K3 (KDA) | 03-candidate-flashinfer_cudnn | 8 | 3 | 1211.011 | 1945.336 | 13.247 | 527.280 |
| Kimi-K3 (KDA) | 03-candidate-flashinfer_cudnn | 32 | 1 | 4406.183 | 6400.222 | 22.568 | 1103.413 |
| Kimi-K3 (KDA) | 03-candidate-flashinfer_cudnn | 32 | 2 | 4410.863 | 6158.716 | 22.256 | 1124.706 |
| Kimi-K3 (KDA) | 03-candidate-flashinfer_cudnn | 32 | 3 | 4408.425 | 6522.862 | 22.524 | 1100.547 |
| Kimi-K3 (KDA) | 03-candidate-flashinfer_cudnn | 128 | 1 | 21163.074 | 22822.880 | 58.341 | 1443.734 |
| Kimi-K3 (KDA) | 03-candidate-flashinfer_cudnn | 128 | 2 | 21615.389 | 21092.885 | 56.362 | 1506.721 |
| Kimi-K3 (KDA) | 03-candidate-flashinfer_cudnn | 128 | 3 | 22035.728 | 23612.688 | 60.139 | 1403.295 |
| Kimi-K3 (KDA) | 04-candidate-flashinfer_auto | 1 | 1 | 332.638 | 333.334 | 8.044 | 119.586 |
| Kimi-K3 (KDA) | 04-candidate-flashinfer_auto | 1 | 2 | 331.066 | 331.553 | 8.024 | 119.903 |
| Kimi-K3 (KDA) | 04-candidate-flashinfer_auto | 1 | 3 | 331.701 | 328.365 | 8.024 | 119.939 |
| Kimi-K3 (KDA) | 04-candidate-flashinfer_auto | 8 | 1 | 1390.283 | 2009.778 | 11.721 | 583.635 |
| Kimi-K3 (KDA) | 04-candidate-flashinfer_auto | 8 | 2 | 1395.407 | 1892.000 | 11.642 | 591.991 |
| Kimi-K3 (KDA) | 04-candidate-flashinfer_auto | 8 | 3 | 1420.822 | 1890.279 | 11.680 | 590.411 |
| Kimi-K3 (KDA) | 04-candidate-flashinfer_auto | 32 | 1 | 5309.820 | 6203.264 | 21.372 | 1158.439 |
| Kimi-K3 (KDA) | 04-candidate-flashinfer_auto | 32 | 2 | 5336.491 | 6025.304 | 21.160 | 1175.366 |
| Kimi-K3 (KDA) | 04-candidate-flashinfer_auto | 32 | 3 | 5110.903 | 5849.988 | 20.855 | 1196.285 |
| Kimi-K3 (KDA) | 04-candidate-flashinfer_auto | 128 | 1 | 20280.879 | 19905.052 | 54.393 | 1564.288 |
| Kimi-K3 (KDA) | 04-candidate-flashinfer_auto | 128 | 2 | 19928.256 | 19689.919 | 54.419 | 1568.048 |
| Kimi-K3 (KDA) | 04-candidate-flashinfer_auto | 128 | 3 | 19648.446 | 19659.665 | 54.478 | 1568.959 |
| Kimi-K3 (KDA) | 05-candidate-vllm_auto | 1 | 1 | 289.640 | 314.895 | 8.098 | 119.075 |
| Kimi-K3 (KDA) | 05-candidate-vllm_auto | 1 | 2 | 288.851 | 312.629 | 8.121 | 118.782 |
| Kimi-K3 (KDA) | 05-candidate-vllm_auto | 1 | 3 | 288.209 | 321.097 | 8.103 | 118.924 |
| Kimi-K3 (KDA) | 05-candidate-vllm_auto | 8 | 1 | 1276.377 | 1830.072 | 13.026 | 539.161 |
| Kimi-K3 (KDA) | 05-candidate-vllm_auto | 8 | 2 | 1286.188 | 1768.667 | 13.007 | 542.042 |
| Kimi-K3 (KDA) | 05-candidate-vllm_auto | 8 | 3 | 1276.260 | 1816.701 | 13.045 | 538.858 |
| Kimi-K3 (KDA) | 05-candidate-vllm_auto | 32 | 1 | 4648.942 | 5616.716 | 21.919 | 1159.992 |
| Kimi-K3 (KDA) | 05-candidate-vllm_auto | 32 | 2 | 4652.942 | 5454.274 | 21.550 | 1182.684 |
| Kimi-K3 (KDA) | 05-candidate-vllm_auto | 32 | 3 | 5110.690 | 5340.943 | 21.392 | 1194.490 |
| Kimi-K3 (KDA) | 05-candidate-vllm_auto | 128 | 1 | 18622.272 | 23414.682 | 58.889 | 1417.606 |
| Kimi-K3 (KDA) | 05-candidate-vllm_auto | 128 | 2 | 18696.964 | 21107.019 | 56.804 | 1490.175 |
| Kimi-K3 (KDA) | 05-candidate-vllm_auto | 128 | 3 | 18692.931 | 22290.146 | 57.909 | 1452.770 |
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
