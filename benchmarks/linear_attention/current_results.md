# Current GDN/KDA measurements

## Current full-serving measurements

All three choices use vLLM `c871d07d6e955303a2033b15b90aeeb955f4c73a`, FlashInfer `5146335b10ee45d8c0c3760a4cf7bbffb9084eeb`, frontend `734e22b8e7fdaae650cf68651919beab9263dcbb`, matching native frontend 1.31.0 and cuDNN 9.20. These are current-adapter results; October 2 timings are excluded.

Six ABC-CBA launches per model/GPU on fixed nodes; 8192 input / 1024 output, C1/C8/C32/C128, three measured repeats and a full-burst warmup before each repeat. Generation and GSM64 gates precede full GSM1319 and timing. Floors: Kimi 64, GLM 55, QwenNext 54, Qwen3.5 56 correct; zero invalid. These floors are safety gates, not accuracy equivalence. Full-evaluation scores and invalid counts are retained separately.

TP: QwenNext 2, Qwen3.5 1, GLM 4; full Kimi 16/four nodes on GB200 and 8/two nodes on GB300. GDN state is FP32; KDA uses controlled BF16 state. No profiler or OS CPU observer. Means average six burst means; throughput averages measured output-token rates. Sequential order, repeat variability and topology limit causal or statistical claims.

| Model / GPU | Job | State | Audited launches |
| --- | --- | --- | ---: |
| Kimi-K3 / GB200 | 3257663 | passed | 6/6 |
| Kimi-K3 / GB300 | 3257664 | partial | 2/6 |
| GLM-5.3-Flash / GB200 | 3257665 | model_validation_failed | 0/6 |
| GLM-5.3-Flash / GB300 | 3257666 | run_failed | 1/6 |
| Qwen3-Next-80B-A3B-Instruct / GB200 | 3257667 | passed | 6/6 |
| Qwen3-Next-80B-A3B-Instruct / GB300 | 3257668 | passed | 6/6 |
| Qwen3.5-35B-A3B / GB200 | 3257669 | model_validation_failed | 3/6 |
| Qwen3.5-35B-A3B / GB300 | 3257670 | passed | 6/6 |
| GLM-5.3-Flash / GB200 | 3258192 | model_validation_failed | 0/6 |
| Qwen3.5-35B-A3B / GB200 | 3258435 | passed | 6/6 |
| GLM-5.3-Flash / GB200 | 3263484 | run_failed | 0/4 |
| GLM-5.3-Flash / GB200 | 3263929 | passed | 2/2 |
| GLM-5.3-Flash / GB300 | 3264375 | PENDING | 0/2 |

Original GLM GB200 job3257665 stopped at FlashKDA GSM64 54/64 (floor55) before any timing. Qwen3.5 job3257669 stopped at reverse FI cuDNN 55/64 (floor56); its three forward launches are incomplete controls. Both failures remain excluded from balanced comparisons. Fixed-count diagnostics found GLM scores54–58/64 and Qwen3.5 scores55–58/64 on unchanged code; causes remain unknown. Each has one fresh comparison with unchanged gates (GLM3258192, Qwen3.5 3258435). No old partial timings are combined with these runs, and another gate failure stops the comparison.

**GLM GB300 full-serving FI-auto failure:** job3257666 failed at the first measured C8 burst after warmup. GSM64 was60/64 and full GSM1214/1319, zero invalid. A sample_tokens RPC timeout preceded CUDA_ERROR_ILLEGAL_ADDRESS during SymmDeviceMemory cleanup; the faulting kernel is unknown. Three FI-auto C1 bursts and one baseline launch are retained, with no balanced comparison. Neither cuDNN arm started. All133files/17,122,178bytes are archived and verified. No blind FI-auto retry or common-cause claim with earlier failures.

The bounded GLM GB200 retry (job 3258192) also stopped at its first FlashKDA GSM64 gate: 54/64, below the unchanged 55/64 floor, with zero invalid answers and normal finite generation. Exact runtime, checkpoint and server arguments match the original failed run apart from launcher paths. Neither FI arm, full GSM1319 nor timing ran. All 29 files / 391,949 bytes are retained. The cause remains unresolved. The gate remains unchanged, and no further retry is scheduled.

**GLM GB200 FI-auto runtime failure:** independent job3263484 passed generation, GSM64 55/64 and full GSM1319 1205/1319 with zero invalid. Nine C1/C8/C32 bursts completed. The first measured C128 burst, after warmup, hit a `sample_tokens` worker RPC timeout. No explicit CUDA error or faulting kernel was recorded. Its 128 responses contained only 7,854/131,072 required output tokens; the harness rejected the result despite the stock client reporting 128 successful requests. Neither cuDNN launch started. All 82 files / 31,844,386 bytes are archived and verified. Partial bursts remain separate; no FI-auto performance retry or cross-allocation comparison.

User-requested GLM GB200 comparison 3263484 runs FlashInfer auto and cuDNN independently of the failed FlashKDA baseline: four ABBA launches, unchanged generation/GSM64 gates for each FI arm, then full GSM1319 and the same 8192/1024 protocol. This protocol can provide two-backend results only after completion; no failed or historical FlashKDA timing is substituted.

Standalone GLM GB200 cuDNN job 3263929 measures the arm that never started before FI-auto failed. Two fresh launches keep all model gates, full GSM1319 and the same timing protocol. No baseline or FI-auto retry; no matched cross-allocation comparison.

Standalone GLM GB300 cuDNN job 3264375 measures the arm that never started before FI-auto failed. Two fresh launches keep all model gates, full GSM1319 and the same timing protocol. No baseline or FI-auto retry; no matched cross-allocation comparison.

### Kimi-K3 / GB200 — job 3257663

| Choice | C | TTFT ms | TPOT ms | Output tokens/s |
| --- | ---: | ---: | ---: | ---: |
| vLLM auto | 1 | 321.285 | 8.037 | 119.851 |
| vLLM auto | 8 | 1611.301 | 12.871 | 552.883 |
| vLLM auto | 32 | 5290.343 | 21.512 | 1191.355 |
| vLLM auto | 128 | 20874.704 | 56.457 | 1501.466 |
| FI auto | 1 | 326.171 | 8.047 | 119.643 |
| FI auto | 8 | 1874.273 | 11.612 | 594.099 |
| FI auto | 32 | 5799.617 | 20.877 | 1197.857 |
| FI auto | 128 | 20407.663 | 55.261 | 1539.686 |
| FI cuDNN | 1 | 300.708 | 8.064 | 119.760 |
| FI cuDNN | 8 | 1725.954 | 12.945 | 546.105 |
| FI cuDNN | 32 | 5366.799 | 21.569 | 1186.117 |
| FI cuDNN | 128 | 19398.390 | 54.629 | 1570.013 |

<details><summary>Ranges across six measured bursts</summary>

| Choice | C | TTFT range ms | TPOT range ms | Output tokens/s range |
| --- | ---: | ---: | ---: | ---: |
| vLLM auto | 1 | 310.792–336.165 | 8.037–8.038 | 119.638–120.008 |
| vLLM auto | 8 | 1560.151–1697.881 | 12.752–12.929 | 547.471–559.086 |
| vLLM auto | 32 | 5250.268–5344.540 | 21.367–21.795 | 1176.535–1197.291 |
| vLLM auto | 128 | 20072.270–22566.511 | 55.389–58.593 | 1436.682–1534.302 |
| FI auto | 1 | 318.448–338.101 | 8.015–8.087 | 119.073–120.213 |
| FI auto | 8 | 1814.178–1939.376 | 11.561–11.656 | 590.363–598.953 |
| FI auto | 32 | 5487.859–6139.107 | 20.488–21.242 | 1166.985–1229.386 |
| FI auto | 128 | 19699.453–21629.976 | 54.455–56.522 | 1495.165–1566.879 |
| FI cuDNN | 1 | 297.310–306.970 | 8.039–8.102 | 119.159–120.155 |
| FI cuDNN | 8 | 1499.174–1986.865 | 12.749–13.103 | 532.712–561.869 |
| FI cuDNN | 32 | 5072.080–5928.585 | 21.196–22.162 | 1137.339–1215.283 |
| FI cuDNN | 128 | 19306.172–19553.269 | 54.445–54.798 | 1566.991–1575.591 |

</details>

| Launch | GSM64 correct /64 | GSM1319 correct /1319 | Invalid /1319 |
| --- | ---: | ---: | ---: |
| 00-candidate-vllm_auto | 64 | 1249 | 1 |
| 01-candidate-flashinfer_auto | 64 | 1245 | 1 |
| 02-candidate-flashinfer_cudnn | 64 | 1246 | 1 |
| 03-candidate-flashinfer_cudnn | 64 | 1255 | 1 |
| 04-candidate-flashinfer_auto | 64 | 1251 | 1 |
| 05-candidate-vllm_auto | 64 | 1245 | 1 |

Every burst is retained below and in current_results.json. Equal score floors do not establish accuracy equivalence or an isolated prefill-backend effect.

### Qwen3.5-35B-A3B / GB300 — job 3257670

| Choice | C | TTFT ms | TPOT ms | Output tokens/s |
| --- | ---: | ---: | ---: | ---: |
| vLLM auto | 1 | 116.926 | 3.272 | 295.558 |
| vLLM auto | 8 | 551.002 | 5.634 | 1293.797 |
| vLLM auto | 32 | 1711.797 | 10.521 | 2605.924 |
| vLLM auto | 128 | 6379.252 | 21.755 | 4474.889 |
| FI auto | 1 | 120.580 | 3.428 | 282.321 |
| FI auto | 8 | 576.623 | 5.855 | 1244.583 |
| FI auto | 32 | 1795.260 | 10.735 | 2544.321 |
| FI auto | 128 | 7889.560 | 23.526 | 4051.638 |
| FI cuDNN | 1 | 113.563 | 3.324 | 291.360 |
| FI cuDNN | 8 | 551.587 | 5.656 | 1289.123 |
| FI cuDNN | 32 | 1773.418 | 10.579 | 2581.152 |
| FI cuDNN | 128 | 6913.388 | 22.285 | 4314.904 |

<details><summary>Ranges across six measured bursts</summary>

| Choice | C | TTFT range ms | TPOT range ms | Output tokens/s range |
| --- | ---: | ---: | ---: | ---: |
| vLLM auto | 1 | 115.172–118.586 | 3.240–3.305 | 292.700–298.570 |
| vLLM auto | 8 | 527.470–566.440 | 5.552–5.699 | 1278.504–1316.102 |
| vLLM auto | 32 | 1664.299–1738.924 | 10.359–10.720 | 2562.867–2650.735 |
| vLLM auto | 128 | 6132.168–6610.187 | 21.340–22.096 | 4387.647–4578.467 |
| FI auto | 1 | 114.436–127.073 | 3.389–3.466 | 278.811–285.855 |
| FI auto | 8 | 530.135–621.831 | 5.711–5.995 | 1210.052–1280.591 |
| FI auto | 32 | 1701.688–1896.515 | 10.568–10.910 | 2493.035–2587.369 |
| FI auto | 128 | 6583.565–9614.991 | 21.932–26.972 | 3494.324–4392.787 |
| FI cuDNN | 1 | 110.368–115.851 | 3.298–3.349 | 289.045–293.738 |
| FI cuDNN | 8 | 542.448–557.990 | 5.571–5.729 | 1275.985–1305.658 |
| FI cuDNN | 32 | 1690.843–1884.228 | 10.347–10.685 | 2537.325–2631.433 |
| FI cuDNN | 128 | 6775.560–6980.330 | 22.159–22.546 | 4265.101–4345.284 |

</details>

| Launch | GSM64 correct /64 | GSM1319 correct /1319 | Invalid /1319 |
| --- | ---: | ---: | ---: |
| 00-candidate-vllm_auto | 59 | 1150 | 0 |
| 01-candidate-flashinfer_auto | 59 | 1125 | 0 |
| 02-candidate-flashinfer_cudnn | 59 | 1083 | 0 |
| 03-candidate-flashinfer_cudnn | 58 | 1106 | 0 |
| 04-candidate-flashinfer_auto | 60 | 1145 | 0 |
| 05-candidate-vllm_auto | 60 | 1125 | 1 |

Every burst is retained below and in current_results.json. Equal score floors do not establish accuracy equivalence or an isolated prefill-backend effect.

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

### Qwen3-Next-80B-A3B-Instruct / GB300 — job 3257668

| Choice | C | TTFT ms | TPOT ms | Output tokens/s |
| --- | ---: | ---: | ---: | ---: |
| vLLM auto | 1 | 128.611 | 3.421 | 282.185 |
| vLLM auto | 8 | 669.054 | 5.102 | 1387.733 |
| vLLM auto | 32 | 2164.614 | 8.849 | 2899.315 |
| vLLM auto | 128 | 7976.932 | 19.356 | 4633.455 |
| FI auto | 1 | 128.860 | 3.403 | 283.614 |
| FI auto | 8 | 666.332 | 5.106 | 1387.558 |
| FI auto | 32 | 2149.462 | 8.822 | 2910.183 |
| FI auto | 128 | 8004.152 | 19.425 | 4614.296 |
| FI cuDNN | 1 | 125.696 | 3.390 | 284.919 |
| FI cuDNN | 8 | 619.568 | 5.056 | 1410.721 |
| FI cuDNN | 32 | 1980.806 | 8.734 | 2979.969 |
| FI cuDNN | 128 | 7508.731 | 18.936 | 4787.258 |

<details><summary>Ranges across six measured bursts</summary>

| Choice | C | TTFT range ms | TPOT range ms | Output tokens/s range |
| --- | ---: | ---: | ---: | ---: |
| vLLM auto | 1 | 122.224–137.369 | 3.415–3.425 | 281.175–283.190 |
| vLLM auto | 8 | 655.583–683.311 | 4.956–5.220 | 1361.704–1419.822 |
| vLLM auto | 32 | 2143.126–2193.265 | 8.633–9.261 | 2797.942–2955.566 |
| vLLM auto | 128 | 7889.899–8060.140 | 19.168–19.622 | 4592.387–4679.932 |
| FI auto | 1 | 123.177–135.698 | 3.396–3.408 | 282.922–284.601 |
| FI auto | 8 | 653.774–677.276 | 4.985–5.299 | 1339.890–1414.026 |
| FI auto | 32 | 2115.825–2193.395 | 8.509–8.987 | 2855.854–2999.263 |
| FI auto | 128 | 7898.545–8136.190 | 18.897–19.801 | 4529.456–4726.077 |
| FI cuDNN | 1 | 120.796–131.564 | 3.377–3.405 | 284.074–285.608 |
| FI cuDNN | 8 | 585.865–655.381 | 4.926–5.175 | 1374.118–1452.219 |
| FI cuDNN | 32 | 1885.307–2087.603 | 8.441–8.975 | 2906.609–3089.937 |
| FI cuDNN | 128 | 7132.594–7922.024 | 18.298–19.453 | 4634.826–4963.378 |

</details>

| Launch | GSM64 correct /64 | GSM1319 correct /1319 | Invalid /1319 |
| --- | ---: | ---: | ---: |
| 00-candidate-vllm_auto | 55 | 1143 | 0 |
| 01-candidate-flashinfer_auto | 56 | 1143 | 0 |
| 02-candidate-flashinfer_cudnn | 57 | 1140 | 1 |
| 03-candidate-flashinfer_cudnn | 56 | 1137 | 0 |
| 04-candidate-flashinfer_auto | 57 | 1145 | 0 |
| 05-candidate-vllm_auto | 55 | 1142 | 0 |

Every burst is retained below and in current_results.json. Equal score floors do not establish accuracy equivalence or an isolated prefill-backend effect.

### Qwen3.5-35B-A3B / GB200 — job 3258435

| Choice | C | TTFT ms | TPOT ms | Output tokens/s |
| --- | ---: | ---: | ---: | ---: |
| vLLM auto | 1 | 115.346 | 3.317 | 291.854 |
| vLLM auto | 8 | 540.384 | 5.727 | 1276.693 |
| vLLM auto | 32 | 1679.026 | 10.383 | 2642.340 |
| vLLM auto | 128 | 6279.759 | 21.387 | 4549.819 |
| FI auto | 1 | 113.337 | 3.333 | 290.618 |
| FI auto | 8 | 533.532 | 5.721 | 1279.601 |
| FI auto | 32 | 1660.383 | 10.419 | 2638.080 |
| FI auto | 128 | 6194.609 | 21.316 | 4573.464 |
| FI cuDNN | 1 | 112.779 | 3.302 | 293.270 |
| FI cuDNN | 8 | 548.522 | 5.675 | 1285.871 |
| FI cuDNN | 32 | 1710.089 | 10.367 | 2639.429 |
| FI cuDNN | 128 | 6476.779 | 21.697 | 4469.175 |

<details><summary>Ranges across six measured bursts</summary>

| Choice | C | TTFT range ms | TPOT range ms | Output tokens/s range |
| --- | ---: | ---: | ---: | ---: |
| vLLM auto | 1 | 114.248–116.390 | 3.308–3.325 | 291.165–292.577 |
| vLLM auto | 8 | 533.020–545.246 | 5.668–5.764 | 1268.753–1288.395 |
| vLLM auto | 32 | 1643.619–1703.832 | 10.195–10.603 | 2589.515–2691.444 |
| vLLM auto | 128 | 6184.736–6402.195 | 21.282–21.535 | 4514.298–4580.472 |
| FI auto | 1 | 111.931–114.802 | 3.327–3.338 | 290.169–291.140 |
| FI auto | 8 | 528.728–544.817 | 5.569–5.806 | 1262.291–1308.887 |
| FI auto | 32 | 1641.643–1681.196 | 10.226–10.601 | 2602.608–2682.671 |
| FI auto | 128 | 6174.021–6232.205 | 21.159–21.461 | 4543.909–4600.563 |
| FI cuDNN | 1 | 112.336–114.181 | 3.294–3.314 | 292.139–293.975 |
| FI cuDNN | 8 | 526.697–561.692 | 5.635–5.738 | 1273.209–1294.464 |
| FI cuDNN | 32 | 1696.061–1719.380 | 10.046–10.502 | 2609.476–2712.494 |
| FI cuDNN | 128 | 6449.613–6531.719 | 21.579–21.822 | 4452.861–4489.668 |

</details>

| Launch | GSM64 correct /64 | GSM1319 correct /1319 | Invalid /1319 |
| --- | ---: | ---: | ---: |
| 00-candidate-vllm_auto | 56 | 1133 | 0 |
| 01-candidate-flashinfer_auto | 57 | 1107 | 0 |
| 02-candidate-flashinfer_cudnn | 59 | 1130 | 0 |
| 03-candidate-flashinfer_cudnn | 60 | 1130 | 0 |
| 04-candidate-flashinfer_auto | 59 | 1125 | 1 |
| 05-candidate-vllm_auto | 58 | 1123 | 0 |

Every burst is retained below and in current_results.json. Equal score floors do not establish accuracy equivalence or an isolated prefill-backend effect.

### GLM-5.3-Flash / GB200 — job 3263929

Standalone repeated measurement; other backends are unavailable, so this is not a matched comparison.

| Choice | C | TTFT ms | TPOT ms | Output tokens/s |
| --- | ---: | ---: | ---: | ---: |
| FI cuDNN | 1 | 187.547 | 5.328 | 181.594 |
| FI cuDNN | 8 | 1084.521 | 8.249 | 858.179 |
| FI cuDNN | 32 | 3143.571 | 13.979 | 1865.268 |
| FI cuDNN | 128 | 12332.411 | 27.644 | 3180.812 |

<details><summary>Ranges across six measured bursts</summary>

| Choice | C | TTFT range ms | TPOT range ms | Output tokens/s range |
| --- | ---: | ---: | ---: | ---: |
| FI cuDNN | 1 | 180.324–195.886 | 5.321–5.337 | 181.150–181.925 |
| FI cuDNN | 8 | 948.424–1159.885 | 8.078–8.401 | 837.651–886.760 |
| FI cuDNN | 32 | 2937.923–3646.471 | 13.691–14.488 | 1761.140–1910.256 |
| FI cuDNN | 128 | 11136.602–15167.594 | 26.671–30.559 | 2776.032–3341.542 |

</details>

| Launch | GSM64 correct /64 | GSM1319 correct /1319 | Invalid /1319 |
| --- | ---: | ---: | ---: |
| 00-candidate-flashinfer_cudnn | 59 | 1206 | 0 |
| 01-candidate-flashinfer_cudnn | 56 | 1210 | 0 |

Every burst is retained below and in current_results.json. Equal score floors do not establish accuracy equivalence or an isolated prefill-backend effect.

## Completed current-stack prefill measurements

Measured vLLM `2bc6cd95d3f02dac347c64e1c86ecc932a47dd4d`; FlashInfer `5146335b10ee45d8c0c3760a4cf7bbffb9084eeb` (#6078), frontend `734e22b8e7fdaae650cf68651919beab9263dcbb` (#1418), matching rebuilt native frontend 1.31.0, cuDNN 9.20, Torch 2.13.0+cu132. Measured NVIDIA source files are unchanged; the AMD model file was restored separately at the user’s request. The original-caller control uses vLLM `b163158` on these same current libraries; it is not an October 2 run.

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
| kimi / GB300 | FlashKDA | 317.084 | 1408.162 | 5160.946 | 64 / 64 |
| kimi / GB300 | FI auto | 310.191 | 1375.310 | 5040.579 | 64 / 64 |
| kimi / GB300 | Original cuDNN caller | 309.185 | 1362.106 | 4996.162 | 64 / 64 |
| kimi / GB300 | Adapted cuDNN caller | 319.412 | 1410.783 | 5137.881 | 64 / 64 |

GLM GB200 adapted/original TTFT improves 4.19% / 4.03% / 4.30%; GSM64 is 55/56 versus 58/57. Exact generation continuations match 1/3 per direction. This does not establish accuracy equivalence.

Kimi GB200 adapted/original TTFT changes are +27.78% / -0.38% / +0.16%. The C1 mean includes a 759.915 ms first repeat; the other five are 276.352–280.989 ms. No samples are removed; cause remains unknown. FlashKDA C1 launch means also vary (349.517/290.448 ms). All eight GSM scores are 64/64, but original/adapted exact continuations match 0/3 per direction.

Current GB300 prefill status: kimi / GB300 job3255983: COMPLETED; glm / GB300 job3255985: FAILED. GLM uses TP4; Kimi uses TP16/four nodes on GB200 and TP8/two nodes on GB300. GLM/Kimi BF16 state is a controlled comparison, not stock FP32. Sequential order and topology prevent a GPU-architecture or CPU-boundness conclusion.

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

### kimi / GB300

| Launch | C1 ms | C8 ms | C32 ms |
| --- | --- | --- | --- |
| 00-candidate-vllm_auto | 317.683 / 317.135 / 316.985 | 1407.832 / 1407.948 / 1407.919 | 5147.708 / 5147.815 / 5178.994 |
| 01-candidate-flashinfer_auto | 309.635 / 307.895 / 308.477 | 1376.291 / 1378.033 / 1375.991 | 5021.342 / 5027.599 / 5074.405 |
| 02-base-flashinfer_cudnn | 307.541 / 307.216 / 306.978 | 1361.877 / 1362.463 / 1362.844 | 4986.698 / 4983.322 / 5015.008 |
| 03-candidate-flashinfer_cudnn | 305.751 / 305.098 / 305.241 | 1353.535 / 1354.032 / 1354.484 | 4948.973 / 4966.142 / 4983.583 |
| 04-candidate-flashinfer_cudnn | 333.799 / 332.937 / 333.643 | 1461.654 / 1482.325 / 1458.667 | 5309.693 / 5299.807 / 5319.087 |
| 05-base-flashinfer_cudnn | 313.301 / 310.863 / 309.214 | 1361.187 / 1361.758 / 1362.506 | 4981.792 / 4981.373 / 5028.781 |
| 06-candidate-flashinfer_auto | 309.773 / 312.110 / 313.255 | 1371.992 / 1369.021 / 1380.530 | 5015.342 / 5047.429 / 5057.355 |
| 07-candidate-vllm_auto | 317.293 / 316.783 / 316.622 | 1408.004 / 1409.355 / 1407.916 | 5148.961 / 5161.834 / 5180.364 |

## Every full-serving repeat

### Kimi-K3 / GB200 — job 3257663

| Launch | C | Repeat | TTFT ms | TPOT ms | Output tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 00-candidate-vllm_auto | 1 | 1 | 336.165 | 8.038 | 119.638 |
| 00-candidate-vllm_auto | 1 | 2 | 334.293 | 8.038 | 119.661 |
| 00-candidate-vllm_auto | 1 | 3 | 324.341 | 8.038 | 119.796 |
| 00-candidate-vllm_auto | 8 | 1 | 1697.881 | 12.929 | 547.471 |
| 00-candidate-vllm_auto | 8 | 2 | 1659.135 | 12.899 | 550.046 |
| 00-candidate-vllm_auto | 8 | 3 | 1616.766 | 12.895 | 551.745 |
| 00-candidate-vllm_auto | 32 | 1 | 5344.540 | 21.795 | 1176.535 |
| 00-candidate-vllm_auto | 32 | 2 | 5314.776 | 21.367 | 1196.686 |
| 00-candidate-vllm_auto | 32 | 3 | 5259.089 | 21.436 | 1196.169 |
| 00-candidate-vllm_auto | 128 | 1 | 22566.511 | 58.593 | 1436.682 |
| 00-candidate-vllm_auto | 128 | 2 | 22141.000 | 57.704 | 1457.475 |
| 00-candidate-vllm_auto | 128 | 3 | 20284.523 | 55.880 | 1520.391 |
| 01-candidate-flashinfer_auto | 1 | 1 | 327.235 | 8.070 | 119.299 |
| 01-candidate-flashinfer_auto | 1 | 2 | 325.980 | 8.087 | 119.073 |
| 01-candidate-flashinfer_auto | 1 | 3 | 327.962 | 8.062 | 119.398 |
| 01-candidate-flashinfer_auto | 8 | 1 | 1885.903 | 11.612 | 593.562 |
| 01-candidate-flashinfer_auto | 8 | 2 | 1814.178 | 11.561 | 598.953 |
| 01-candidate-flashinfer_auto | 8 | 3 | 1821.329 | 11.606 | 596.604 |
| 01-candidate-flashinfer_auto | 32 | 1 | 5713.853 | 20.780 | 1205.641 |
| 01-candidate-flashinfer_auto | 32 | 2 | 5576.141 | 20.679 | 1216.435 |
| 01-candidate-flashinfer_auto | 32 | 3 | 5487.859 | 20.488 | 1229.386 |
| 01-candidate-flashinfer_auto | 128 | 1 | 21629.976 | 56.522 | 1495.165 |
| 01-candidate-flashinfer_auto | 128 | 2 | 20900.292 | 55.900 | 1518.838 |
| 01-candidate-flashinfer_auto | 128 | 3 | 19864.866 | 54.455 | 1565.819 |
| 02-candidate-flashinfer_cudnn | 1 | 1 | 304.695 | 8.102 | 119.159 |
| 02-candidate-flashinfer_cudnn | 1 | 2 | 298.830 | 8.080 | 119.558 |
| 02-candidate-flashinfer_cudnn | 1 | 3 | 306.970 | 8.080 | 119.433 |
| 02-candidate-flashinfer_cudnn | 8 | 1 | 1986.865 | 13.052 | 532.712 |
| 02-candidate-flashinfer_cudnn | 8 | 2 | 1931.014 | 13.103 | 532.821 |
| 02-candidate-flashinfer_cudnn | 8 | 3 | 1871.425 | 13.022 | 537.810 |
| 02-candidate-flashinfer_cudnn | 32 | 1 | 5928.585 | 22.162 | 1137.339 |
| 02-candidate-flashinfer_cudnn | 32 | 2 | 5565.901 | 21.851 | 1164.881 |
| 02-candidate-flashinfer_cudnn | 32 | 3 | 5465.988 | 21.446 | 1186.677 |
| 02-candidate-flashinfer_cudnn | 128 | 1 | 19553.269 | 54.539 | 1568.951 |
| 02-candidate-flashinfer_cudnn | 128 | 2 | 19413.846 | 54.690 | 1568.126 |
| 02-candidate-flashinfer_cudnn | 128 | 3 | 19383.842 | 54.798 | 1566.991 |
| 03-candidate-flashinfer_cudnn | 1 | 1 | 299.082 | 8.039 | 120.127 |
| 03-candidate-flashinfer_cudnn | 1 | 2 | 297.364 | 8.040 | 120.155 |
| 03-candidate-flashinfer_cudnn | 1 | 3 | 297.310 | 8.042 | 120.127 |
| 03-candidate-flashinfer_cudnn | 8 | 1 | 1531.340 | 12.910 | 554.363 |
| 03-candidate-flashinfer_cudnn | 8 | 2 | 1499.174 | 12.749 | 561.869 |
| 03-candidate-flashinfer_cudnn | 8 | 3 | 1535.906 | 12.836 | 557.056 |
| 03-candidate-flashinfer_cudnn | 32 | 1 | 5085.072 | 21.334 | 1208.267 |
| 03-candidate-flashinfer_cudnn | 32 | 2 | 5083.170 | 21.426 | 1204.252 |
| 03-candidate-flashinfer_cudnn | 32 | 3 | 5072.080 | 21.196 | 1215.283 |
| 03-candidate-flashinfer_cudnn | 128 | 1 | 19329.740 | 54.445 | 1575.591 |
| 03-candidate-flashinfer_cudnn | 128 | 2 | 19403.468 | 54.515 | 1572.121 |
| 03-candidate-flashinfer_cudnn | 128 | 3 | 19306.172 | 54.785 | 1568.296 |
| 04-candidate-flashinfer_auto | 1 | 1 | 318.448 | 8.015 | 120.213 |
| 04-candidate-flashinfer_auto | 1 | 2 | 319.301 | 8.015 | 120.201 |
| 04-candidate-flashinfer_auto | 1 | 3 | 338.101 | 8.033 | 119.672 |
| 04-candidate-flashinfer_auto | 8 | 1 | 1939.376 | 11.633 | 590.363 |
| 04-candidate-flashinfer_auto | 8 | 2 | 1896.935 | 11.656 | 591.183 |
| 04-candidate-flashinfer_auto | 8 | 3 | 1887.919 | 11.602 | 593.928 |
| 04-candidate-flashinfer_auto | 32 | 1 | 6139.107 | 21.242 | 1166.985 |
| 04-candidate-flashinfer_auto | 32 | 2 | 6026.014 | 21.182 | 1174.341 |
| 04-candidate-flashinfer_auto | 32 | 3 | 5854.726 | 20.894 | 1194.356 |
| 04-candidate-flashinfer_auto | 128 | 1 | 20505.119 | 55.429 | 1532.221 |
| 04-candidate-flashinfer_auto | 128 | 2 | 19846.269 | 54.716 | 1559.193 |
| 04-candidate-flashinfer_auto | 128 | 3 | 19699.453 | 54.543 | 1566.879 |
| 05-candidate-vllm_auto | 1 | 1 | 311.294 | 8.037 | 120.001 |
| 05-candidate-vllm_auto | 1 | 2 | 310.792 | 8.037 | 120.004 |
| 05-candidate-vllm_auto | 1 | 3 | 310.823 | 8.037 | 120.008 |
| 05-candidate-vllm_auto | 8 | 1 | 1565.556 | 12.826 | 556.340 |
| 05-candidate-vllm_auto | 8 | 2 | 1568.315 | 12.752 | 559.086 |
| 05-candidate-vllm_auto | 8 | 3 | 1560.151 | 12.924 | 552.609 |
| 05-candidate-vllm_auto | 32 | 1 | 5306.970 | 21.546 | 1189.003 |
| 05-candidate-vllm_auto | 32 | 2 | 5266.415 | 21.401 | 1197.291 |
| 05-candidate-vllm_auto | 32 | 3 | 5250.268 | 21.526 | 1192.444 |
| 05-candidate-vllm_auto | 128 | 1 | 20086.807 | 55.389 | 1534.302 |
| 05-candidate-vllm_auto | 128 | 2 | 20072.270 | 55.726 | 1526.648 |
| 05-candidate-vllm_auto | 128 | 3 | 20097.114 | 55.447 | 1533.299 |

### Qwen3.5-35B-A3B / GB300 — job 3257670

| Launch | C | Repeat | TTFT ms | TPOT ms | Output tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 00-candidate-vllm_auto | 1 | 1 | 115.172 | 3.240 | 298.570 |
| 00-candidate-vllm_auto | 1 | 2 | 118.586 | 3.241 | 298.177 |
| 00-candidate-vllm_auto | 1 | 3 | 118.519 | 3.240 | 298.237 |
| 00-candidate-vllm_auto | 8 | 1 | 566.440 | 5.639 | 1289.539 |
| 00-candidate-vllm_auto | 8 | 2 | 559.568 | 5.699 | 1278.504 |
| 00-candidate-vllm_auto | 8 | 3 | 527.470 | 5.552 | 1316.102 |
| 00-candidate-vllm_auto | 32 | 1 | 1707.789 | 10.523 | 2605.746 |
| 00-candidate-vllm_auto | 32 | 2 | 1726.994 | 10.585 | 2589.024 |
| 00-candidate-vllm_auto | 32 | 3 | 1719.265 | 10.720 | 2562.867 |
| 00-candidate-vllm_auto | 128 | 1 | 6372.608 | 21.485 | 4519.536 |
| 00-candidate-vllm_auto | 128 | 2 | 6137.281 | 21.650 | 4527.152 |
| 00-candidate-vllm_auto | 128 | 3 | 6132.168 | 21.340 | 4578.467 |
| 01-candidate-flashinfer_auto | 1 | 1 | 126.206 | 3.466 | 278.811 |
| 01-candidate-flashinfer_auto | 1 | 2 | 125.677 | 3.464 | 279.054 |
| 01-candidate-flashinfer_auto | 1 | 3 | 127.073 | 3.464 | 278.890 |
| 01-candidate-flashinfer_auto | 8 | 1 | 599.979 | 5.940 | 1223.745 |
| 01-candidate-flashinfer_auto | 8 | 2 | 621.831 | 5.833 | 1239.776 |
| 01-candidate-flashinfer_auto | 8 | 3 | 618.627 | 5.995 | 1210.052 |
| 01-candidate-flashinfer_auto | 32 | 1 | 1896.515 | 10.878 | 2495.846 |
| 01-candidate-flashinfer_auto | 32 | 2 | 1877.175 | 10.910 | 2493.035 |
| 01-candidate-flashinfer_auto | 32 | 3 | 1828.420 | 10.745 | 2534.953 |
| 01-candidate-flashinfer_auto | 128 | 1 | 6745.861 | 21.934 | 4392.246 |
| 01-candidate-flashinfer_auto | 128 | 2 | 6748.655 | 21.932 | 4392.787 |
| 01-candidate-flashinfer_auto | 128 | 3 | 6583.565 | 22.144 | 4381.638 |
| 02-candidate-flashinfer_cudnn | 1 | 1 | 115.625 | 3.349 | 289.062 |
| 02-candidate-flashinfer_cudnn | 1 | 2 | 115.125 | 3.349 | 289.112 |
| 02-candidate-flashinfer_cudnn | 1 | 3 | 115.851 | 3.349 | 289.045 |
| 02-candidate-flashinfer_cudnn | 8 | 1 | 557.990 | 5.667 | 1285.514 |
| 02-candidate-flashinfer_cudnn | 8 | 2 | 557.816 | 5.600 | 1299.474 |
| 02-candidate-flashinfer_cudnn | 8 | 3 | 550.598 | 5.648 | 1291.021 |
| 02-candidate-flashinfer_cudnn | 32 | 1 | 1734.069 | 10.671 | 2569.182 |
| 02-candidate-flashinfer_cudnn | 32 | 2 | 1785.583 | 10.675 | 2558.525 |
| 02-candidate-flashinfer_cudnn | 32 | 3 | 1779.543 | 10.505 | 2594.873 |
| 02-candidate-flashinfer_cudnn | 128 | 1 | 6960.974 | 22.217 | 4319.690 |
| 02-candidate-flashinfer_cudnn | 128 | 2 | 6980.330 | 22.546 | 4265.101 |
| 02-candidate-flashinfer_cudnn | 128 | 3 | 6890.071 | 22.266 | 4321.379 |
| 03-candidate-flashinfer_cudnn | 1 | 1 | 110.368 | 3.299 | 293.738 |
| 03-candidate-flashinfer_cudnn | 1 | 2 | 112.922 | 3.300 | 293.470 |
| 03-candidate-flashinfer_cudnn | 1 | 3 | 111.486 | 3.298 | 293.729 |
| 03-candidate-flashinfer_cudnn | 8 | 1 | 542.448 | 5.729 | 1275.985 |
| 03-candidate-flashinfer_cudnn | 8 | 2 | 557.779 | 5.571 | 1305.658 |
| 03-candidate-flashinfer_cudnn | 8 | 3 | 542.889 | 5.723 | 1277.085 |
| 03-candidate-flashinfer_cudnn | 32 | 1 | 1766.240 | 10.347 | 2631.433 |
| 03-candidate-flashinfer_cudnn | 32 | 2 | 1690.843 | 10.589 | 2595.573 |
| 03-candidate-flashinfer_cudnn | 32 | 3 | 1884.228 | 10.685 | 2537.325 |
| 03-candidate-flashinfer_cudnn | 128 | 1 | 6972.188 | 22.159 | 4325.536 |
| 03-candidate-flashinfer_cudnn | 128 | 2 | 6901.205 | 22.320 | 4312.432 |
| 03-candidate-flashinfer_cudnn | 128 | 3 | 6775.560 | 22.203 | 4345.284 |
| 04-candidate-flashinfer_auto | 1 | 1 | 115.632 | 3.391 | 285.587 |
| 04-candidate-flashinfer_auto | 1 | 2 | 114.436 | 3.389 | 285.855 |
| 04-candidate-flashinfer_auto | 1 | 3 | 114.458 | 3.391 | 285.729 |
| 04-candidate-flashinfer_auto | 8 | 1 | 530.135 | 5.830 | 1258.064 |
| 04-candidate-flashinfer_auto | 8 | 2 | 537.084 | 5.711 | 1280.591 |
| 04-candidate-flashinfer_auto | 8 | 3 | 552.085 | 5.822 | 1255.268 |
| 04-candidate-flashinfer_auto | 32 | 1 | 1701.688 | 10.615 | 2587.369 |
| 04-candidate-flashinfer_auto | 32 | 2 | 1708.991 | 10.695 | 2569.393 |
| 04-candidate-flashinfer_auto | 32 | 3 | 1758.771 | 10.568 | 2585.332 |
| 04-candidate-flashinfer_auto | 128 | 1 | 9247.860 | 26.972 | 3494.324 |
| 04-candidate-flashinfer_auto | 128 | 2 | 9614.991 | 24.651 | 3692.445 |
| 04-candidate-flashinfer_auto | 128 | 3 | 8396.425 | 23.520 | 3956.386 |
| 05-candidate-vllm_auto | 1 | 1 | 116.145 | 3.304 | 292.841 |
| 05-candidate-vllm_auto | 1 | 2 | 117.352 | 3.305 | 292.700 |
| 05-candidate-vllm_auto | 1 | 3 | 115.783 | 3.305 | 292.826 |
| 05-candidate-vllm_auto | 8 | 1 | 542.517 | 5.655 | 1291.136 |
| 05-candidate-vllm_auto | 8 | 2 | 547.489 | 5.656 | 1289.895 |
| 05-candidate-vllm_auto | 8 | 3 | 562.526 | 5.604 | 1297.607 |
| 05-candidate-vllm_auto | 32 | 1 | 1664.299 | 10.359 | 2650.735 |
| 05-candidate-vllm_auto | 32 | 2 | 1713.512 | 10.474 | 2615.151 |
| 05-candidate-vllm_auto | 32 | 3 | 1738.924 | 10.465 | 2612.022 |
| 05-candidate-vllm_auto | 128 | 1 | 6579.136 | 22.029 | 4399.724 |
| 05-candidate-vllm_auto | 128 | 2 | 6610.187 | 22.096 | 4387.647 |
| 05-candidate-vllm_auto | 128 | 3 | 6444.133 | 21.931 | 4436.811 |

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

### Qwen3-Next-80B-A3B-Instruct / GB300 — job 3257668

| Launch | C | Repeat | TTFT ms | TPOT ms | Output tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 00-candidate-vllm_auto | 1 | 1 | 132.412 | 3.422 | 281.838 |
| 00-candidate-vllm_auto | 1 | 2 | 132.445 | 3.423 | 281.734 |
| 00-candidate-vllm_auto | 1 | 3 | 137.369 | 3.425 | 281.175 |
| 00-candidate-vllm_auto | 8 | 1 | 683.311 | 4.956 | 1419.822 |
| 00-candidate-vllm_auto | 8 | 2 | 681.166 | 4.982 | 1413.806 |
| 00-candidate-vllm_auto | 8 | 3 | 675.685 | 5.205 | 1361.704 |
| 00-candidate-vllm_auto | 32 | 1 | 2171.766 | 8.633 | 2955.566 |
| 00-candidate-vllm_auto | 32 | 2 | 2187.063 | 8.843 | 2894.456 |
| 00-candidate-vllm_auto | 32 | 3 | 2193.265 | 8.869 | 2885.613 |
| 00-candidate-vllm_auto | 128 | 1 | 8023.025 | 19.278 | 4640.780 |
| 00-candidate-vllm_auto | 128 | 2 | 8015.129 | 19.168 | 4658.349 |
| 00-candidate-vllm_auto | 128 | 3 | 8060.140 | 19.322 | 4625.080 |
| 01-candidate-flashinfer_auto | 1 | 1 | 132.987 | 3.407 | 282.965 |
| 01-candidate-flashinfer_auto | 1 | 2 | 135.698 | 3.405 | 282.922 |
| 01-candidate-flashinfer_auto | 1 | 3 | 131.824 | 3.408 | 282.949 |
| 01-candidate-flashinfer_auto | 8 | 1 | 662.305 | 5.022 | 1408.484 |
| 01-candidate-flashinfer_auto | 8 | 2 | 669.529 | 5.029 | 1404.905 |
| 01-candidate-flashinfer_auto | 8 | 3 | 659.127 | 5.173 | 1372.641 |
| 01-candidate-flashinfer_auto | 32 | 1 | 2193.395 | 8.987 | 2855.854 |
| 01-candidate-flashinfer_auto | 32 | 2 | 2157.264 | 8.959 | 2871.602 |
| 01-candidate-flashinfer_auto | 32 | 3 | 2167.558 | 8.872 | 2891.304 |
| 01-candidate-flashinfer_auto | 128 | 1 | 8064.573 | 19.337 | 4622.363 |
| 01-candidate-flashinfer_auto | 128 | 2 | 8052.113 | 19.664 | 4569.333 |
| 01-candidate-flashinfer_auto | 128 | 3 | 8136.190 | 19.801 | 4529.456 |
| 02-candidate-flashinfer_cudnn | 1 | 1 | 121.910 | 3.399 | 284.438 |
| 02-candidate-flashinfer_cudnn | 1 | 2 | 120.872 | 3.401 | 284.421 |
| 02-candidate-flashinfer_cudnn | 1 | 3 | 120.796 | 3.405 | 284.074 |
| 02-candidate-flashinfer_cudnn | 8 | 1 | 585.865 | 4.926 | 1452.219 |
| 02-candidate-flashinfer_cudnn | 8 | 2 | 587.580 | 5.104 | 1406.110 |
| 02-candidate-flashinfer_cudnn | 8 | 3 | 592.137 | 5.101 | 1405.860 |
| 02-candidate-flashinfer_cudnn | 32 | 1 | 1903.349 | 8.480 | 3071.712 |
| 02-candidate-flashinfer_cudnn | 32 | 2 | 1885.307 | 8.441 | 3089.937 |
| 02-candidate-flashinfer_cudnn | 32 | 3 | 1931.079 | 8.790 | 2975.079 |
| 02-candidate-flashinfer_cudnn | 128 | 1 | 7132.594 | 18.980 | 4839.610 |
| 02-candidate-flashinfer_cudnn | 128 | 2 | 7174.018 | 18.298 | 4963.378 |
| 02-candidate-flashinfer_cudnn | 128 | 3 | 7147.675 | 18.493 | 4929.933 |
| 03-candidate-flashinfer_cudnn | 1 | 1 | 130.453 | 3.377 | 285.608 |
| 03-candidate-flashinfer_cudnn | 1 | 2 | 131.564 | 3.377 | 285.487 |
| 03-candidate-flashinfer_cudnn | 1 | 3 | 128.580 | 3.380 | 285.487 |
| 03-candidate-flashinfer_cudnn | 8 | 1 | 655.381 | 4.978 | 1421.213 |
| 03-candidate-flashinfer_cudnn | 8 | 2 | 651.687 | 5.175 | 1374.118 |
| 03-candidate-flashinfer_cudnn | 8 | 3 | 644.757 | 5.054 | 1404.804 |
| 03-candidate-flashinfer_cudnn | 32 | 1 | 2075.195 | 8.878 | 2915.077 |
| 03-candidate-flashinfer_cudnn | 32 | 2 | 2087.603 | 8.841 | 2921.400 |
| 03-candidate-flashinfer_cudnn | 32 | 3 | 2002.304 | 8.975 | 2906.609 |
| 03-candidate-flashinfer_cudnn | 128 | 1 | 7849.391 | 19.453 | 4634.826 |
| 03-candidate-flashinfer_cudnn | 128 | 2 | 7826.683 | 19.287 | 4671.914 |
| 03-candidate-flashinfer_cudnn | 128 | 3 | 7922.024 | 19.107 | 4683.889 |
| 04-candidate-flashinfer_auto | 1 | 1 | 123.541 | 3.396 | 284.601 |
| 04-candidate-flashinfer_auto | 1 | 2 | 123.177 | 3.400 | 284.310 |
| 04-candidate-flashinfer_auto | 1 | 3 | 125.931 | 3.402 | 283.936 |
| 04-candidate-flashinfer_auto | 8 | 1 | 677.276 | 4.985 | 1414.026 |
| 04-candidate-flashinfer_auto | 8 | 2 | 653.774 | 5.125 | 1385.400 |
| 04-candidate-flashinfer_auto | 8 | 3 | 675.981 | 5.299 | 1339.890 |
| 04-candidate-flashinfer_auto | 32 | 1 | 2136.617 | 8.509 | 2999.263 |
| 04-candidate-flashinfer_auto | 32 | 2 | 2126.116 | 8.655 | 2959.604 |
| 04-candidate-flashinfer_auto | 32 | 3 | 2115.825 | 8.952 | 2883.472 |
| 04-candidate-flashinfer_auto | 128 | 1 | 7898.545 | 18.897 | 4726.077 |
| 04-candidate-flashinfer_auto | 128 | 2 | 7940.163 | 19.534 | 4594.046 |
| 04-candidate-flashinfer_auto | 128 | 3 | 7933.325 | 19.314 | 4644.499 |
| 05-candidate-vllm_auto | 1 | 1 | 124.277 | 3.425 | 282.203 |
| 05-candidate-vllm_auto | 1 | 2 | 122.224 | 3.415 | 283.190 |
| 05-candidate-vllm_auto | 1 | 3 | 122.939 | 3.417 | 282.969 |
| 05-candidate-vllm_auto | 8 | 1 | 660.925 | 5.124 | 1383.991 |
| 05-candidate-vllm_auto | 8 | 2 | 655.583 | 5.126 | 1384.876 |
| 05-candidate-vllm_auto | 8 | 3 | 657.654 | 5.220 | 1362.198 |
| 05-candidate-vllm_auto | 32 | 1 | 2147.193 | 8.715 | 2938.506 |
| 05-candidate-vllm_auto | 32 | 2 | 2145.269 | 9.261 | 2797.942 |
| 05-candidate-vllm_auto | 32 | 3 | 2143.126 | 8.773 | 2923.805 |
| 05-candidate-vllm_auto | 128 | 1 | 7936.481 | 19.564 | 4604.203 |
| 05-candidate-vllm_auto | 128 | 2 | 7936.918 | 19.622 | 4592.387 |
| 05-candidate-vllm_auto | 128 | 3 | 7889.899 | 19.180 | 4679.932 |

### Qwen3.5-35B-A3B / GB200 — job 3258435

| Launch | C | Repeat | TTFT ms | TPOT ms | Output tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 00-candidate-vllm_auto | 1 | 1 | 116.315 | 3.309 | 292.448 |
| 00-candidate-vllm_auto | 1 | 2 | 116.390 | 3.309 | 292.389 |
| 00-candidate-vllm_auto | 1 | 3 | 115.685 | 3.308 | 292.577 |
| 00-candidate-vllm_auto | 8 | 1 | 542.813 | 5.680 | 1285.891 |
| 00-candidate-vllm_auto | 8 | 2 | 542.164 | 5.668 | 1288.395 |
| 00-candidate-vllm_auto | 8 | 3 | 545.246 | 5.750 | 1271.094 |
| 00-candidate-vllm_auto | 32 | 1 | 1698.070 | 10.332 | 2649.346 |
| 00-candidate-vllm_auto | 32 | 2 | 1692.094 | 10.430 | 2629.028 |
| 00-candidate-vllm_auto | 32 | 3 | 1703.832 | 10.603 | 2589.515 |
| 00-candidate-vllm_auto | 128 | 1 | 6357.790 | 21.316 | 4549.114 |
| 00-candidate-vllm_auto | 128 | 2 | 6345.613 | 21.535 | 4514.298 |
| 00-candidate-vllm_auto | 128 | 3 | 6402.195 | 21.414 | 4527.842 |
| 01-candidate-flashinfer_auto | 1 | 1 | 113.635 | 3.338 | 290.169 |
| 01-candidate-flashinfer_auto | 1 | 2 | 112.219 | 3.336 | 290.443 |
| 01-candidate-flashinfer_auto | 1 | 3 | 111.931 | 3.338 | 290.299 |
| 01-candidate-flashinfer_auto | 8 | 1 | 532.930 | 5.806 | 1262.291 |
| 01-candidate-flashinfer_auto | 8 | 2 | 544.817 | 5.569 | 1308.887 |
| 01-candidate-flashinfer_auto | 8 | 3 | 532.362 | 5.681 | 1287.846 |
| 01-candidate-flashinfer_auto | 32 | 1 | 1668.324 | 10.349 | 2651.169 |
| 01-candidate-flashinfer_auto | 32 | 2 | 1661.323 | 10.459 | 2629.365 |
| 01-candidate-flashinfer_auto | 32 | 3 | 1641.643 | 10.601 | 2602.608 |
| 01-candidate-flashinfer_auto | 128 | 1 | 6174.021 | 21.189 | 4598.019 |
| 01-candidate-flashinfer_auto | 128 | 2 | 6232.205 | 21.461 | 4543.909 |
| 01-candidate-flashinfer_auto | 128 | 3 | 6199.053 | 21.394 | 4558.886 |
| 02-candidate-flashinfer_cudnn | 1 | 1 | 112.500 | 3.296 | 293.855 |
| 02-candidate-flashinfer_cudnn | 1 | 2 | 112.336 | 3.297 | 293.794 |
| 02-candidate-flashinfer_cudnn | 1 | 3 | 112.854 | 3.294 | 293.975 |
| 02-candidate-flashinfer_cudnn | 8 | 1 | 526.697 | 5.654 | 1294.464 |
| 02-candidate-flashinfer_cudnn | 8 | 2 | 561.692 | 5.677 | 1282.630 |
| 02-candidate-flashinfer_cudnn | 8 | 3 | 558.409 | 5.635 | 1292.053 |
| 02-candidate-flashinfer_cudnn | 32 | 1 | 1703.715 | 10.046 | 2712.494 |
| 02-candidate-flashinfer_cudnn | 32 | 2 | 1719.380 | 10.480 | 2612.323 |
| 02-candidate-flashinfer_cudnn | 32 | 3 | 1714.262 | 10.448 | 2620.493 |
| 02-candidate-flashinfer_cudnn | 128 | 1 | 6449.613 | 21.601 | 4487.470 |
| 02-candidate-flashinfer_cudnn | 128 | 2 | 6461.431 | 21.579 | 4489.668 |
| 02-candidate-flashinfer_cudnn | 128 | 3 | 6498.694 | 21.657 | 4471.718 |
| 03-candidate-flashinfer_cudnn | 1 | 1 | 114.181 | 3.314 | 292.139 |
| 03-candidate-flashinfer_cudnn | 1 | 2 | 112.451 | 3.307 | 292.870 |
| 03-candidate-flashinfer_cudnn | 1 | 3 | 112.350 | 3.306 | 292.987 |
| 03-candidate-flashinfer_cudnn | 8 | 1 | 557.981 | 5.668 | 1285.400 |
| 03-candidate-flashinfer_cudnn | 8 | 2 | 539.451 | 5.676 | 1287.468 |
| 03-candidate-flashinfer_cudnn | 8 | 3 | 546.900 | 5.738 | 1273.209 |
| 03-candidate-flashinfer_cudnn | 32 | 1 | 1714.443 | 10.475 | 2614.993 |
| 03-candidate-flashinfer_cudnn | 32 | 2 | 1712.671 | 10.502 | 2609.476 |
| 03-candidate-flashinfer_cudnn | 32 | 3 | 1696.061 | 10.254 | 2666.797 |
| 03-candidate-flashinfer_cudnn | 128 | 1 | 6531.719 | 21.726 | 4457.391 |
| 03-candidate-flashinfer_cudnn | 128 | 2 | 6462.856 | 21.795 | 4455.940 |
| 03-candidate-flashinfer_cudnn | 128 | 3 | 6456.360 | 21.822 | 4452.861 |
| 04-candidate-flashinfer_auto | 1 | 1 | 113.265 | 3.327 | 291.140 |
| 04-candidate-flashinfer_auto | 1 | 2 | 114.802 | 3.331 | 290.666 |
| 04-candidate-flashinfer_auto | 1 | 3 | 114.172 | 3.328 | 290.989 |
| 04-candidate-flashinfer_auto | 8 | 1 | 530.353 | 5.776 | 1268.821 |
| 04-candidate-flashinfer_auto | 8 | 2 | 532.002 | 5.717 | 1280.403 |
| 04-candidate-flashinfer_auto | 8 | 3 | 528.728 | 5.775 | 1269.357 |
| 04-candidate-flashinfer_auto | 32 | 1 | 1650.964 | 10.226 | 2682.671 |
| 04-candidate-flashinfer_auto | 32 | 2 | 1658.851 | 10.431 | 2635.988 |
| 04-candidate-flashinfer_auto | 32 | 3 | 1681.196 | 10.450 | 2626.681 |
| 04-candidate-flashinfer_auto | 128 | 1 | 6178.262 | 21.300 | 4579.417 |
| 04-candidate-flashinfer_auto | 128 | 2 | 6200.102 | 21.392 | 4559.989 |
| 04-candidate-flashinfer_auto | 128 | 3 | 6184.013 | 21.159 | 4600.563 |
| 05-candidate-vllm_auto | 1 | 1 | 114.668 | 3.325 | 291.165 |
| 05-candidate-vllm_auto | 1 | 2 | 114.248 | 3.325 | 291.258 |
| 05-candidate-vllm_auto | 1 | 3 | 114.767 | 3.324 | 291.289 |
| 05-candidate-vllm_auto | 8 | 1 | 542.884 | 5.764 | 1268.753 |
| 05-candidate-vllm_auto | 8 | 2 | 533.020 | 5.759 | 1271.595 |
| 05-candidate-vllm_auto | 8 | 3 | 536.178 | 5.742 | 1274.433 |
| 05-candidate-vllm_auto | 32 | 1 | 1643.619 | 10.195 | 2691.444 |
| 05-candidate-vllm_auto | 32 | 2 | 1673.767 | 10.466 | 2625.185 |
| 05-candidate-vllm_auto | 32 | 3 | 1662.775 | 10.275 | 2669.523 |
| 05-candidate-vllm_auto | 128 | 1 | 6191.054 | 21.421 | 4558.608 |
| 05-candidate-vllm_auto | 128 | 2 | 6184.736 | 21.282 | 4580.472 |
| 05-candidate-vllm_auto | 128 | 3 | 6197.163 | 21.351 | 4568.582 |

### GLM-5.3-Flash / GB200 — job 3263929

| Launch | C | Repeat | TTFT ms | TPOT ms | Output tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 00-candidate-flashinfer_cudnn | 1 | 1 | 192.094 | 5.337 | 181.150 |
| 00-candidate-flashinfer_cudnn | 1 | 2 | 185.108 | 5.324 | 181.797 |
| 00-candidate-flashinfer_cudnn | 1 | 3 | 180.324 | 5.331 | 181.727 |
| 00-candidate-flashinfer_cudnn | 8 | 1 | 1141.588 | 8.249 | 852.800 |
| 00-candidate-flashinfer_cudnn | 8 | 2 | 1153.468 | 8.272 | 849.627 |
| 00-candidate-flashinfer_cudnn | 8 | 3 | 1159.885 | 8.401 | 837.651 |
| 00-candidate-flashinfer_cudnn | 32 | 1 | 3039.275 | 14.030 | 1869.298 |
| 00-candidate-flashinfer_cudnn | 32 | 2 | 3161.601 | 14.101 | 1848.706 |
| 00-candidate-flashinfer_cudnn | 32 | 3 | 3063.918 | 13.779 | 1894.518 |
| 00-candidate-flashinfer_cudnn | 128 | 1 | 11136.602 | 26.805 | 3332.054 |
| 00-candidate-flashinfer_cudnn | 128 | 2 | 11172.992 | 26.671 | 3341.542 |
| 00-candidate-flashinfer_cudnn | 128 | 3 | 13276.378 | 27.837 | 3082.240 |
| 01-candidate-flashinfer_cudnn | 1 | 1 | 184.204 | 5.321 | 181.925 |
| 01-candidate-flashinfer_cudnn | 1 | 2 | 195.886 | 5.326 | 181.386 |
| 01-candidate-flashinfer_cudnn | 1 | 3 | 187.663 | 5.329 | 181.580 |
| 01-candidate-flashinfer_cudnn | 8 | 1 | 948.424 | 8.078 | 886.760 |
| 01-candidate-flashinfer_cudnn | 8 | 2 | 949.913 | 8.209 | 873.943 |
| 01-candidate-flashinfer_cudnn | 8 | 3 | 1153.846 | 8.287 | 848.294 |
| 01-candidate-flashinfer_cudnn | 32 | 1 | 3012.239 | 13.691 | 1910.256 |
| 01-candidate-flashinfer_cudnn | 32 | 2 | 3646.471 | 14.488 | 1761.140 |
| 01-candidate-flashinfer_cudnn | 32 | 3 | 2937.923 | 13.785 | 1907.690 |
| 01-candidate-flashinfer_cudnn | 128 | 1 | 15167.594 | 30.559 | 2776.032 |
| 01-candidate-flashinfer_cudnn | 128 | 2 | 11445.346 | 26.856 | 3302.615 |
| 01-candidate-flashinfer_cudnn | 128 | 3 | 11795.554 | 27.137 | 3250.390 |
