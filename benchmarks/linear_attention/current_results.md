# Current GDN/KDA measurements

## Current full-serving measurements

All three choices use vLLM `c871d07d6e955303a2033b15b90aeeb955f4c73a`, FlashInfer `5146335b10ee45d8c0c3760a4cf7bbffb9084eeb`, frontend `734e22b8e7fdaae650cf68651919beab9263dcbb`, matching native frontend 1.31.0 and cuDNN 9.20. These are current-adapter results; October 2 timings are excluded.

Six ABC-CBA launches per model/GPU on fixed nodes; 8192 input / 1024 output, C1/C8/C32/C128, three measured repeats and a full-burst warmup before each repeat. Generation and GSM64 gates precede full GSM1319 and timing. Floors: Kimi 64, GLM 55, QwenNext 54, Qwen3.5 56 correct; zero invalid. These floors are safety gates, not accuracy equivalence. Full-evaluation scores and invalid counts are retained separately.

TP: QwenNext 2, Qwen3.5 1, GLM 4; full Kimi 16/four nodes on GB200 and 8/two nodes on GB300. GDN state is FP32; KDA uses controlled BF16 state. No profiler or OS CPU observer. Means average six burst means; throughput averages measured output-token rates. Sequential order, repeat variability and topology limit causal or statistical claims.

| Model / GPU | Job | State | Audited launches |
| --- | --- | --- | ---: |
| Kimi-K3 / GB200 | 3257663 | partial | 3/6 |
| Kimi-K3 / GB300 | 3257664 | PENDING | 0/6 |
| GLM-5.3-Flash / GB200 | 3257665 | model_validation_failed | 0/6 |
| GLM-5.3-Flash / GB300 | 3257666 | PENDING | 0/6 |
| Qwen3-Next-80B-A3B-Instruct / GB200 | 3257667 | passed | 6/6 |
| Qwen3-Next-80B-A3B-Instruct / GB300 | 3257668 | PENDING | 0/6 |
| Qwen3.5-35B-A3B / GB200 | 3257669 | model_validation_failed | 3/6 |
| Qwen3.5-35B-A3B / GB300 | 3257670 | PENDING | 0/6 |
| GLM-5.3-Flash / GB200 | 3258192 | partial | 0/6 |
| Qwen3.5-35B-A3B / GB200 | 3258435 | partial | 0/6 |

Original GLM GB200 job3257665 stopped at FlashKDA GSM64 54/64 (floor55) before any timing. Qwen3.5 job3257669 stopped at reverse FI cuDNN 55/64 (floor56); its three forward launches are incomplete controls. Both failures remain excluded from balanced comparisons. Fixed-count diagnostics found GLM scores54–58/64 and Qwen3.5 scores55–58/64 on unchanged code; causes remain unknown. Each has one fresh comparison with unchanged gates (GLM3258192, Qwen3.5 3258435). No old partial timings are combined with these runs, and another gate failure stops the comparison.

### Qwen3-Next-80B-A3B-Instruct / GB200 — job 3257667

| Choice | C | TTFT ms | TPOT ms | Output tokens/s |
| --- | ---: | ---: | ---: | ---: |
| vLLM auto | 1 | 127.235 | 3.510 | 275.342 |
| vLLM auto | 8 | 640.637 | 5.177 | 1377.366 |
| vLLM auto | 32 | 2030.986 | 8.811 | 2945.927 |
| vLLM auto | 128 | 7409.171 | 18.735 | 4838.663 |
| FI auto | 1 | 125.591 | 3.555 | 272.119 |
| FI auto | 8 | 620.289 | 5.165 | 1384.244 |
| FI auto | 32 | 1980.933 | 8.737 | 2978.392 |
| FI auto | 128 | 7377.497 | 18.767 | 4838.289 |
| FI cuDNN | 1 | 122.895 | 3.507 | 275.951 |
| FI cuDNN | 8 | 600.428 | 5.142 | 1394.099 |
| FI cuDNN | 32 | 1951.025 | 8.744 | 2985.083 |
| FI cuDNN | 128 | 7298.862 | 18.755 | 4853.612 |

<details><summary>Ranges across six measured bursts</summary>

| Choice | C | TTFT range ms | TPOT range ms | Output tokens/s range |
| --- | ---: | ---: | ---: | ---: |
| vLLM auto | 1 | 120.676–133.901 | 3.508–3.513 | 275.053–275.802 |
| vLLM auto | 8 | 600.993–671.195 | 4.899–5.347 | 1330.217–1438.575 |
| vLLM auto | 32 | 1938.886–2152.754 | 8.422–9.249 | 2799.160–3079.075 |
| vLLM auto | 128 | 7359.174–7464.881 | 18.400–18.899 | 4798.053–4911.190 |
| FI auto | 1 | 123.818–127.310 | 3.551–3.565 | 271.354–272.347 |
| FI auto | 8 | 606.263–632.198 | 5.031–5.338 | 1343.085–1414.971 |
| FI auto | 32 | 1947.334–2026.686 | 8.304–8.985 | 2914.131–3105.133 |
| FI auto | 128 | 7277.243–7502.362 | 18.508–19.234 | 4742.226–4882.683 |
| FI cuDNN | 1 | 120.395–126.991 | 3.497–3.520 | 275.137–276.493 |
| FI cuDNN | 8 | 590.670–612.235 | 5.043–5.232 | 1371.191–1415.549 |
| FI cuDNN | 32 | 1864.522–2091.671 | 8.449–9.085 | 2855.293–3081.071 |
| FI cuDNN | 128 | 7179.996–7416.515 | 18.592–18.953 | 4796.843–4898.237 |

</details>

| Launch | GSM64 correct /64 | GSM1319 correct /1319 | Invalid /1319 |
| --- | ---: | ---: | ---: |
| 00-candidate-vllm_auto | 55 | 1140 | 0 |
| 01-candidate-flashinfer_auto | 56 | 1144 | 0 |
| 02-candidate-flashinfer_cudnn | 57 | 1142 | 0 |
| 03-candidate-flashinfer_cudnn | 56 | 1134 | 0 |
| 04-candidate-flashinfer_auto | 58 | 1141 | 1 |
| 05-candidate-vllm_auto | 55 | 1134 | 0 |

Every burst is retained below and in current_results.json. Equal score floors do not establish accuracy equivalence or an isolated prefill-backend effect.

## Completed current-stack prefill measurements

Measured vLLM `2bc6cd95d3f02dac347c64e1c86ecc932a47dd4d`; FlashInfer `5146335b10ee45d8c0c3760a4cf7bbffb9084eeb` (#6078), frontend `734e22b8e7fdaae650cf68651919beab9263dcbb` (#1418), matching rebuilt native frontend 1.31.0, cuDNN 9.20, Torch 2.13.0+cu132. The cleaned PR preserves the measured runtime Python files byte for byte. The original-caller control uses vLLM `b163158` on these same current libraries; it is not an October 2 run.

Eight ABCD-DCBA launches on fixed nodes; 3 repeats per concurrency per launch; warmup before every repeat. Unprofiled 8192-input/1-output C1/C8/C32. Generation and GSM64 gates precede every timing client: Kimi 64/64, GLM at least 55/64, zero invalid. The GLM floor is a safety gate, not accuracy equivalence. No profiler or OS CPU observer. This workload has **no TPOT or decode-throughput measurement**.

| Model / GPU | Choice | C1 TTFT ms | C8 TTFT ms | C32 TTFT ms | GSM64 forward / reverse |
| --- | --- | ---: | ---: | ---: | ---: |
| glm / GB200 | FlashKDA | 157.521 | 675.460 | 2422.969 | 55 / 55 |
| glm / GB200 | FI auto | 198.981 | 877.545 | 3178.831 | 56 / 59 |
| glm / GB200 | Original cuDNN caller | 156.772 | 687.745 | 2492.506 | 58 / 57 |
| glm / GB200 | Adapted cuDNN caller | 150.206 | 660.017 | 2385.376 | 55 / 56 |
| kimi / GB200 | FlashKDA | 319.982 | 1392.436 | 5133.784 | 64 / 64 |
| kimi / GB200 | FI auto | 278.932 | 1239.726 | 4579.768 | 64 / 64 |
| kimi / GB200 | Original cuDNN caller | 280.211 | 1233.823 | 4518.401 | 64 / 64 |
| kimi / GB200 | Adapted cuDNN caller | 358.048 | 1229.080 | 4525.605 | 64 / 64 |

GLM GB200 adapted/original TTFT improves 4.19% / 4.03% / 4.30%; GSM64 is 55/56 versus 58/57. Exact generation continuations match 1/3 per direction. This does not establish accuracy equivalence.

Kimi GB200 adapted/original TTFT changes are +27.78% / -0.38% / +0.16%. The C1 mean includes a 759.915 ms first repeat; the other five are 276.352–280.989 ms. No samples are removed; cause remains unknown. FlashKDA C1 launch means also vary (349.517/290.448 ms). All eight GSM scores are 64/64, but original/adapted exact continuations match 0/3 per direction.

Current GB300 prefill status: kimi / GB300 job3255983: RUNNING; glm / GB300 job3255985: FAILED. GLM uses TP4; Kimi uses TP16/four nodes on GB200 and TP8/two nodes on GB300. GLM/Kimi BF16 state is a controlled comparison, not stock FP32. Sequential order and topology prevent a GPU-architecture or CPU-boundness conclusion.

GLM GB300 prefill job 3255985 failed in FI auto's first measured C8 burst, after its warmup. Generation and GSM 56/64 passed with zero invalid; three C1 repeats were saved. Worker TP3 reported a CUDA illegal instruction at `cu_seqlens.tolist()` in `flashinfer.kda_prefill._cached_packed_task_metadata`, through the small-BH path. The error may be asynchronous, so the faulting kernel remains unknown. Neither cuDNN arm started. All 55 files / 996,510 bytes are archived and verified; no balanced result or automatic retry. This does not establish the cause of the earlier RPC timeout.

## Active CPU per KDA adapter call

Microbenchmarks on the same current libraries; mean of opposite-order process medians, µs/call. Fresh CPU-only processes, one core/one Torch thread, no Torch-profiler or graph-capture history. CPU excludes state reset and final synchronization. These cumulative caller changes are not isolated PR effects or distributed model latency.

| Model layout / GPU | FlashKDA | Adapted FI auto | Original cuDNN | Adapted cuDNN |
| --- | ---: | ---: | ---: | ---: |
| GLM H16/8192 / GB200 | 125.416 | 218.344 | 354.576 | 182.336 |
| GLM H16/8192 / GB300 | 108.024 | 177.784 | 277.120 | 140.776 |
| Kimi H6/8192 / GB200 | 88.200 | 176.984 | 316.560 | 180.816 |
| Kimi H12/8192 / GB300 | 76.720 | 145.768 | 246.896 | 143.000 |

## Every prefill repeat

### glm / GB200

| Launch | C1 ms | C8 ms | C32 ms |
| --- | --- | --- | --- |
| 00-candidate-vllm_auto | 159.615 / 156.436 / 156.301 | 678.201 / 672.848 / 673.411 | 2422.465 / 2423.202 / 2422.909 |
| 01-candidate-flashinfer_auto | 202.379 / 198.353 / 199.033 | 880.843 / 881.055 / 883.128 | 3193.053 / 3195.674 / 3193.530 |
| 02-base-flashinfer_cudnn | 157.393 / 155.237 / 156.355 | 685.416 / 688.024 / 689.487 | 2489.642 / 2505.200 / 2506.874 |
| 03-candidate-flashinfer_cudnn | 150.161 / 151.507 / 148.228 | 651.768 / 654.239 / 656.782 | 2368.351 / 2366.580 / 2365.497 |
| 04-candidate-flashinfer_cudnn | 152.702 / 149.151 / 149.490 | 662.087 / 666.138 / 669.087 | 2404.877 / 2403.777 / 2403.172 |
| 05-base-flashinfer_cudnn | 156.487 / 157.312 / 157.847 | 685.917 / 688.994 / 688.629 | 2483.052 / 2485.488 / 2484.779 |
| 06-candidate-flashinfer_auto | 200.113 / 197.207 / 196.800 | 871.208 / 874.235 / 874.798 | 3162.951 / 3163.471 / 3164.309 |
| 07-candidate-vllm_auto | 158.758 / 157.709 / 156.309 | 678.140 / 677.814 / 672.348 | 2424.467 / 2424.241 / 2420.529 |

### kimi / GB200

| Launch | C1 ms | C8 ms | C32 ms |
| --- | --- | --- | --- |
| 00-candidate-vllm_auto | 357.518 / 353.556 / 337.476 | 1486.321 / 1492.682 / 1519.591 | 5466.683 / 5466.429 / 5539.562 |
| 01-candidate-flashinfer_auto | 284.280 / 278.866 / 278.944 | 1244.240 / 1244.266 / 1244.546 | 4531.810 / 4803.990 / 4561.178 |
| 02-base-flashinfer_cudnn | 283.641 / 279.374 / 280.518 | 1234.406 / 1235.402 / 1233.933 | 4505.141 / 4510.346 / 4555.320 |
| 03-candidate-flashinfer_cudnn | 759.915 / 276.772 / 276.837 | 1227.726 / 1228.625 / 1230.216 | 4495.815 / 4502.001 / 4505.154 |
| 04-candidate-flashinfer_cudnn | 280.989 / 276.352 / 277.424 | 1225.568 / 1231.960 / 1230.386 | 4647.410 / 4488.636 / 4514.615 |
| 05-base-flashinfer_cudnn | 283.029 / 277.002 / 277.700 | 1230.426 / 1232.614 / 1236.156 | 4513.688 / 4514.617 / 4511.294 |
| 06-candidate-flashinfer_auto | 277.460 / 277.158 / 276.884 | 1234.697 / 1235.053 / 1235.553 | 4520.231 / 4515.938 / 4545.464 |
| 07-candidate-vllm_auto | 292.430 / 289.545 / 289.369 | 1285.004 / 1285.308 / 1285.714 | 4705.538 / 4700.179 / 4924.313 |

## Every full-serving repeat

### Qwen3-Next-80B-A3B-Instruct / GB200 — job 3257667

| Launch | C | Repeat | TTFT ms | TPOT ms | Output tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 00-candidate-vllm_auto | 1 | 1 | 120.676 | 3.511 | 275.802 |
| 00-candidate-vllm_auto | 1 | 2 | 124.990 | 3.512 | 275.390 |
| 00-candidate-vllm_auto | 1 | 3 | 123.690 | 3.513 | 275.389 |
| 00-candidate-vllm_auto | 8 | 1 | 633.615 | 4.941 | 1436.004 |
| 00-candidate-vllm_auto | 8 | 2 | 615.906 | 5.297 | 1353.816 |
| 00-candidate-vllm_auto | 8 | 3 | 600.993 | 5.297 | 1357.025 |
| 00-candidate-vllm_auto | 32 | 1 | 1965.626 | 8.746 | 2978.451 |
| 00-candidate-vllm_auto | 32 | 2 | 1943.061 | 8.726 | 2990.050 |
| 00-candidate-vllm_auto | 32 | 3 | 1938.886 | 8.422 | 3079.075 |
| 00-candidate-vllm_auto | 128 | 1 | 7464.881 | 18.899 | 4798.053 |
| 00-candidate-vllm_auto | 128 | 2 | 7366.878 | 18.844 | 4826.065 |
| 00-candidate-vllm_auto | 128 | 3 | 7446.258 | 18.781 | 4822.687 |
| 01-candidate-flashinfer_auto | 1 | 1 | 126.612 | 3.552 | 272.294 |
| 01-candidate-flashinfer_auto | 1 | 2 | 126.046 | 3.565 | 271.354 |
| 01-candidate-flashinfer_auto | 1 | 3 | 124.751 | 3.556 | 272.119 |
| 01-candidate-flashinfer_auto | 8 | 1 | 632.198 | 5.031 | 1413.646 |
| 01-candidate-flashinfer_auto | 8 | 2 | 621.352 | 5.338 | 1343.085 |
| 01-candidate-flashinfer_auto | 8 | 3 | 626.013 | 5.296 | 1351.690 |
| 01-candidate-flashinfer_auto | 32 | 1 | 2026.686 | 8.861 | 2931.272 |
| 01-candidate-flashinfer_auto | 32 | 2 | 1960.775 | 8.985 | 2914.131 |
| 01-candidate-flashinfer_auto | 32 | 3 | 1982.217 | 8.616 | 3009.738 |
| 01-candidate-flashinfer_auto | 128 | 1 | 7426.133 | 18.508 | 4878.871 |
| 01-candidate-flashinfer_auto | 128 | 2 | 7502.362 | 18.827 | 4805.203 |
| 01-candidate-flashinfer_auto | 128 | 3 | 7413.862 | 19.234 | 4742.226 |
| 02-candidate-flashinfer_cudnn | 1 | 1 | 126.991 | 3.497 | 276.360 |
| 02-candidate-flashinfer_cudnn | 1 | 2 | 124.411 | 3.498 | 276.493 |
| 02-candidate-flashinfer_cudnn | 1 | 3 | 124.577 | 3.506 | 275.854 |
| 02-candidate-flashinfer_cudnn | 8 | 1 | 597.122 | 5.057 | 1415.549 |
| 02-candidate-flashinfer_cudnn | 8 | 2 | 605.309 | 5.232 | 1371.191 |
| 02-candidate-flashinfer_cudnn | 8 | 3 | 605.020 | 5.086 | 1406.402 |
| 02-candidate-flashinfer_cudnn | 32 | 1 | 1922.317 | 8.717 | 2999.937 |
| 02-candidate-flashinfer_cudnn | 32 | 2 | 2091.671 | 9.085 | 2855.293 |
| 02-candidate-flashinfer_cudnn | 32 | 3 | 2002.403 | 8.819 | 2949.031 |
| 02-candidate-flashinfer_cudnn | 128 | 1 | 7342.017 | 18.714 | 4853.038 |
| 02-candidate-flashinfer_cudnn | 128 | 2 | 7416.515 | 18.953 | 4796.843 |
| 02-candidate-flashinfer_cudnn | 128 | 3 | 7348.637 | 18.592 | 4874.639 |
| 03-candidate-flashinfer_cudnn | 1 | 1 | 120.535 | 3.507 | 276.084 |
| 03-candidate-flashinfer_cudnn | 1 | 2 | 120.462 | 3.511 | 275.779 |
| 03-candidate-flashinfer_cudnn | 1 | 3 | 120.395 | 3.520 | 275.137 |
| 03-candidate-flashinfer_cudnn | 8 | 1 | 612.235 | 5.043 | 1415.336 |
| 03-candidate-flashinfer_cudnn | 8 | 2 | 592.214 | 5.219 | 1377.389 |
| 03-candidate-flashinfer_cudnn | 8 | 3 | 590.670 | 5.214 | 1378.726 |
| 03-candidate-flashinfer_cudnn | 32 | 1 | 1908.614 | 8.449 | 3081.071 |
| 03-candidate-flashinfer_cudnn | 32 | 2 | 1916.622 | 8.724 | 2998.075 |
| 03-candidate-flashinfer_cudnn | 32 | 3 | 1864.522 | 8.672 | 3027.092 |
| 03-candidate-flashinfer_cudnn | 128 | 1 | 7308.516 | 18.718 | 4856.651 |
| 03-candidate-flashinfer_cudnn | 128 | 2 | 7197.493 | 18.905 | 4842.262 |
| 03-candidate-flashinfer_cudnn | 128 | 3 | 7179.996 | 18.645 | 4898.237 |
| 04-candidate-flashinfer_auto | 1 | 1 | 125.008 | 3.553 | 272.319 |
| 04-candidate-flashinfer_auto | 1 | 2 | 127.310 | 3.551 | 272.282 |
| 04-candidate-flashinfer_auto | 1 | 3 | 123.818 | 3.554 | 272.347 |
| 04-candidate-flashinfer_auto | 8 | 1 | 613.391 | 5.043 | 1414.971 |
| 04-candidate-flashinfer_auto | 8 | 2 | 622.515 | 5.166 | 1382.705 |
| 04-candidate-flashinfer_auto | 8 | 3 | 606.263 | 5.114 | 1399.368 |
| 04-candidate-flashinfer_auto | 32 | 1 | 1997.569 | 8.873 | 2936.097 |
| 04-candidate-flashinfer_auto | 32 | 2 | 1947.334 | 8.780 | 2973.981 |
| 04-candidate-flashinfer_auto | 32 | 3 | 1971.015 | 8.304 | 3105.133 |
| 04-candidate-flashinfer_auto | 128 | 1 | 7319.455 | 18.604 | 4882.683 |
| 04-candidate-flashinfer_auto | 128 | 2 | 7277.243 | 18.690 | 4869.884 |
| 04-candidate-flashinfer_auto | 128 | 3 | 7325.925 | 18.738 | 4850.865 |
| 05-candidate-vllm_auto | 1 | 1 | 127.831 | 3.511 | 275.265 |
| 05-candidate-vllm_auto | 1 | 2 | 132.323 | 3.508 | 275.152 |
| 05-candidate-vllm_auto | 1 | 3 | 133.901 | 3.508 | 275.053 |
| 05-candidate-vllm_auto | 8 | 1 | 666.505 | 4.899 | 1438.575 |
| 05-candidate-vllm_auto | 8 | 2 | 671.195 | 5.347 | 1330.217 |
| 05-candidate-vllm_auto | 8 | 3 | 655.610 | 5.281 | 1348.559 |
| 05-candidate-vllm_auto | 32 | 1 | 2131.179 | 8.998 | 2868.066 |
| 05-candidate-vllm_auto | 32 | 2 | 2152.754 | 9.249 | 2799.160 |
| 05-candidate-vllm_auto | 32 | 3 | 2054.408 | 8.724 | 2960.762 |
| 05-candidate-vllm_auto | 128 | 1 | 7439.345 | 18.738 | 4832.019 |
| 05-candidate-vllm_auto | 128 | 2 | 7378.488 | 18.748 | 4841.964 |
| 05-candidate-vllm_auto | 128 | 3 | 7359.174 | 18.400 | 4911.190 |
