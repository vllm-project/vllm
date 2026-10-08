# FE1454: KDA on GB200

Additive normalization is now built into cuDNN. The vLLM caller passes no additive-epsilon argument and no longer depends on FlashInfer #6078. AMD `linear.py` remains reverted.

**Numerical gate: passed.** vLLM 67/67, FlashInfer 75/75, serving-shape matrix 12/12; the same frontend/native binary passed 18/18 selected tests. All three backend pairs passed the existing output/final-state tolerance (relative RMSE <3% or maximum absolute error ≤1e-6) at H6/H16, C1/C8/C32 and 8192 tokens, including tiny Q/K. Separate small/zero-input, reference, ownership and replay tests passed. This is tolerance agreement on GB200, not bitwise equality or model accuracy equivalence.

Pins: vLLM `29ad8bdac62250209d0e7c694eb0246f1882efab`; frontend #1454 merge `db85ff42d6d8ce8ab05b1d94414620cfc451a0fa`; FlashInfer `13e9271f03b2d0e27881697c2d32c17d9705d10f` (main `b5e50c20` plus the native CuTe additive-formula fix). Native frontend 1.31.0, wheel SHA256 `48f6c5928bf78b82fbc5bc336d8ac77943895e23545d555d09094bc62130838d`, cuDNN 9.20. No configurable epsilon API was added.

## Side-by-side measurements

8192 input tokens / 1 output token, C1/C8/C32, three repeats with a full warmup burst before each repeat. Six fresh server launches use ABC-CBA order. TTFT is full-model serving latency measured before hooks. CPU and GPU columns cover only the KDA prefill caller and its correlated device kernels. CPU-only processes never enter Torch profiling; GPU-process CPU is excluded.

CPU and kernel burst times are averages of rank-local values, with rank ranges retained below. They are neither summed across ranks nor added to TTFT. Instrumented client latencies are excluded from TTFT. No earlier-pair timings are substituted.

Both models passed the full serving and trace audits. Compared with FlashKDA, cuDNN measured 8.4–10.0% lower Kimi TTFT and about 64% less KDA kernel time, while KDA active CPU was 2.4–2.7 times higher. GLM TTFT was 2.7–3.2% lower, KDA kernel time about 60% lower and KDA CPU about 37% higher. Sequential launch variation limits these comparisons; they do not establish a CPU bottleneck, statistical significance or an isolated PR effect.

**KIMI — job 3264917: COMPLETED; passed.**

**GLM — job 3264918: COMPLETED; passed.**

### Serving TTFT (ms)

| Model | C | vLLM auto (FlashKDA) | FlashInfer auto | FlashInfer cuDNN |
| --- | ---: | ---: | ---: | ---: |
| KIMI | 1 | 303.955 | 278.254 | 273.522 |
| KIMI | 8 | 1322.713 | 1242.950 | 1208.129 |
| KIMI | 32 | 4808.379 | 4545.975 | 4406.640 |
| GLM | 1 | 161.752 | 200.767 | 156.655 |
| GLM | 8 | 706.583 | 880.066 | 686.659 |
| GLM | 32 | 2533.197 | 3187.622 | 2464.486 |

### KDA active CPU per burst (ms/rank)

| Model | C | vLLM auto (FlashKDA) | FlashInfer auto | FlashInfer cuDNN |
| --- | ---: | ---: | ---: | ---: |
| KIMI | 1 | 7.089 | 73.727 | 17.060 |
| KIMI | 8 | 56.447 | 564.624 | 150.816 |
| KIMI | 32 | 225.111 | 2290.050 | 618.651 |
| GLM | 1 | 6.355 | 54.720 | 8.712 |
| GLM | 8 | 51.400 | 431.360 | 70.538 |
| GLM | 32 | 208.779 | 1739.589 | 286.053 |

### KDA kernel union per burst (ms/rank)

| Model | C | vLLM auto (FlashKDA) | FlashInfer auto | FlashInfer cuDNN |
| --- | ---: | ---: | ---: | ---: |
| KIMI | 1 | 22.018 | 13.745 | 7.942 |
| KIMI | 8 | 176.247 | 110.015 | 63.549 |
| KIMI | 32 | 704.966 | 440.117 | 254.198 |
| GLM | 1 | 15.595 | 11.532 | 6.301 |
| GLM | 8 | 124.747 | 92.268 | 50.466 |
| GLM | 32 | 499.065 | 369.086 | 201.836 |

The initial v24 jobs stopped before model startup because the launcher lacked executable permission. Their complete logs are retained; v25 fixes the launcher mode and uses the same verified sources, gates, client and hooks. No timing was produced by v24.

## Model checks and limits

Each launch must pass finite, nondegenerate generation and its own GSM64 gate (Kimi 64/64; GLM at least 55/64; zero invalid). Full GSM1319 accuracy and invalid counts are retained separately. A FlashKDA model-score failure does not prevent independent FI measurements; missing baselines remain unavailable. Kimi uses TP16/four GB200 nodes; GLM TP4/one node. State is BF16.

CPU measurements include host work within the model’s KDA caller: dispatch, validation, packing, state handling and launch submission. Kernel attribution uses matching GPU-launch correlations inside that caller. Different wrapper boundaries and layouts are retained for review. CPU savings alone do not establish a CPU-bound model. Device gaps do not prove CPU causality. Sequential launch variation limits causal and statistical claims.

## Receipts

- Numerical checks: `kda-cpu-parity-20261005/reviews/native-additive-numerics-20261007.json`
- Serving audit: `kda-cpu-parity-20261005/reviews/serving-v25-native-additive-live.json`
- Every request, warmup, CPU repeat, rank layout and source hash is retained under the new immutable `serving-v25-native-additive` bundle. Older results remain separate.

## Model accuracy by launch

| Model | Launch | GSM64 correct / 64 | Full correct / 1319 | Full invalid |
| --- | --- | ---: | ---: | ---: |
| KIMI | 00-cpu-vllm_auto | 64 | 1254 | 1 |
| KIMI | 01-cpu-flashinfer_auto | 64 | 1241 | 2 |
| KIMI | 02-cpu-flashinfer_cudnn | 64 | 1250 | 1 |
| KIMI | 03-gpu-flashinfer_cudnn | 64 | 1243 | 1 |
| KIMI | 04-gpu-flashinfer_auto | 64 | 1245 | 2 |
| KIMI | 05-gpu-vllm_auto | 64 | 1245 | 2 |
| GLM | 00-cpu-vllm_auto | 58 | 1211 | 0 |
| GLM | 01-cpu-flashinfer_auto | 60 | 1218 | 0 |
| GLM | 02-cpu-flashinfer_cudnn | 58 | 1209 | 0 |
| GLM | 03-gpu-flashinfer_cudnn | 58 | 1210 | 0 |
| GLM | 04-gpu-flashinfer_auto | 58 | 1216 | 0 |
| GLM | 05-gpu-vllm_auto | 55 | 1213 | 0 |

## Repeats and rank ranges

| Model | C | Mode | Metric | Repeat means (ms) | Rank-local range (ms) | Mean per call (µs) |
| --- | ---: | --- | --- | --- | --- | ---: |
| KIMI | 1 | vLLM auto (FlashKDA) | ttft | 324.723, 317.389, 318.732, 288.701, 287.175, 287.008 | — | — |
| KIMI | 1 | FlashInfer auto | ttft | 279.874, 278.907, 278.862, 278.653, 276.496, 276.733 | — | — |
| KIMI | 1 | FlashInfer cuDNN | ttft | 275.331, 273.022, 272.463, 274.759, 272.907, 272.650 | — | — |
| KIMI | 8 | vLLM auto (FlashKDA) | ttft | 1394.850, 1369.475, 1350.805, 1272.351, 1277.031, 1271.769 | — | — |
| KIMI | 8 | FlashInfer auto | ttft | 1246.371, 1252.049, 1249.784, 1236.957, 1236.642, 1235.898 | — | — |
| KIMI | 8 | FlashInfer cuDNN | ttft | 1207.267, 1206.832, 1206.827, 1214.399, 1206.806, 1206.645 | — | — |
| KIMI | 32 | vLLM auto (FlashKDA) | ttft | 5068.170, 4903.945, 4946.700, 4643.554, 4644.357, 4643.551 | — | — |
| KIMI | 32 | FlashInfer auto | ttft | 4530.748, 4571.312, 4592.543, 4506.957, 4510.958, 4563.329 | — | — |
| KIMI | 32 | FlashInfer cuDNN | ttft | 4389.894, 4391.469, 4394.485, 4409.429, 4407.536, 4447.028 | — | — |
| GLM | 1 | vLLM auto (FlashKDA) | ttft | 160.678, 159.228, 158.403, 163.819, 163.523, 164.861 | — | — |
| GLM | 1 | FlashInfer auto | ttft | 204.954, 199.829, 199.842, 201.215, 198.768, 199.992 | — | — |
| GLM | 1 | FlashInfer cuDNN | ttft | 157.040, 157.405, 156.316, 158.367, 154.301, 156.500 | — | — |
| GLM | 8 | vLLM auto (FlashKDA) | ttft | 679.477, 679.643, 687.433, 729.659, 731.036, 732.248 | — | — |
| GLM | 8 | FlashInfer auto | ttft | 882.569, 882.308, 882.796, 878.267, 879.377, 875.076 | — | — |
| GLM | 8 | FlashInfer cuDNN | ttft | 686.448, 686.548, 678.268, 690.371, 689.238, 689.081 | — | — |
| GLM | 32 | vLLM auto (FlashKDA) | ttft | 2444.056, 2444.135, 2444.205, 2636.749, 2595.188, 2634.846 | — | — |
| GLM | 32 | FlashInfer auto | ttft | 3202.952, 3205.738, 3208.218, 3174.551, 3170.647, 3163.627 | — | — |
| GLM | 32 | FlashInfer cuDNN | ttft | 2437.113, 2452.089, 2458.141, 2480.507, 2486.775, 2472.291 | — | — |
| KIMI | 1 | vLLM auto (FlashKDA) | cpu | 7.105, 7.075, 7.086 | 6.380–9.304 | 102.733 |
| KIMI | 1 | FlashInfer auto | cpu | 79.515, 71.093, 70.573 | 54.733–88.679 | 1068.507 |
| KIMI | 1 | FlashInfer cuDNN | cpu | 17.161, 17.103, 16.915 | 16.107–18.442 | 247.240 |
| KIMI | 8 | vLLM auto (FlashKDA) | cpu | 56.371, 56.495, 56.476 | 49.470–71.979 | 102.260 |
| KIMI | 8 | FlashInfer auto | cpu | 564.457, 563.191, 566.225 | 453.072–651.309 | 1022.870 |
| KIMI | 8 | FlashInfer cuDNN | cpu | 150.628, 150.944, 150.877 | 148.235–152.232 | 273.218 |
| KIMI | 32 | vLLM auto (FlashKDA) | cpu | 225.941, 224.817, 224.573 | 197.191–277.946 | 101.952 |
| KIMI | 32 | FlashInfer auto | cpu | 2299.654, 2275.167, 2295.330 | 1821.955–2675.793 | 1037.161 |
| KIMI | 32 | FlashInfer cuDNN | cpu | 618.651, 618.432, 618.868 | 610.406–625.989 | 280.186 |
| GLM | 1 | vLLM auto (FlashKDA) | cpu | 6.442, 6.299, 6.324 | 6.157–6.558 | 186.908 |
| GLM | 1 | FlashInfer auto | cpu | 53.908, 55.025, 55.228 | 51.840–57.423 | 1609.425 |
| GLM | 1 | FlashInfer cuDNN | cpu | 8.818, 8.665, 8.651 | 8.362–9.521 | 256.223 |
| GLM | 8 | vLLM auto (FlashKDA) | cpu | 51.285, 51.291, 51.624 | 50.397–52.892 | 188.971 |
| GLM | 8 | FlashInfer auto | cpu | 426.243, 438.546, 429.292 | 405.974–446.309 | 1585.884 |
| GLM | 8 | FlashInfer cuDNN | cpu | 70.114, 70.668, 70.832 | 68.817–73.784 | 259.332 |
| GLM | 32 | vLLM auto (FlashKDA) | cpu | 208.484, 208.812, 209.041 | 205.395–213.997 | 191.892 |
| GLM | 32 | FlashInfer auto | cpu | 1756.699, 1742.721, 1719.347 | 1673.413–1811.779 | 1598.887 |
| GLM | 32 | FlashInfer cuDNN | cpu | 284.938, 286.400, 286.821 | 282.347–291.347 | 262.916 |
| KIMI | 1 | vLLM auto (FlashKDA) | gpu | 22.007, 22.017, 22.031 | 21.906–22.112 | 319.105 |
| KIMI | 1 | FlashInfer auto | gpu | 13.741, 13.746, 13.748 | 13.687–13.833 | 199.199 |
| KIMI | 1 | FlashInfer cuDNN | gpu | 7.938, 7.942, 7.945 | 7.880–8.028 | 115.099 |
| KIMI | 8 | vLLM auto (FlashKDA) | gpu | 176.257, 176.234, 176.250 | 175.722–176.999 | 319.288 |
| KIMI | 8 | FlashInfer auto | gpu | 110.007, 110.042, 109.997 | 109.410–110.671 | 199.303 |
| KIMI | 8 | FlashInfer cuDNN | gpu | 63.538, 63.554, 63.554 | 63.115–64.551 | 115.124 |
| KIMI | 32 | vLLM auto (FlashKDA) | gpu | 704.949, 704.985, 704.965 | 702.498–707.790 | 319.278 |
| KIMI | 32 | FlashInfer auto | gpu | 440.133, 440.108, 440.112 | 437.648–442.410 | 199.329 |
| KIMI | 32 | FlashInfer cuDNN | gpu | 254.208, 254.196, 254.190 | 252.379–257.800 | 115.126 |
| GLM | 1 | vLLM auto (FlashKDA) | gpu | 15.609, 15.587, 15.590 | 15.572–15.645 | 458.690 |
| GLM | 1 | FlashInfer auto | gpu | 11.532, 11.534, 11.528 | 11.514–11.551 | 339.166 |
| GLM | 1 | FlashInfer cuDNN | gpu | 6.301, 6.303, 6.300 | 6.254–6.341 | 185.338 |
| GLM | 8 | vLLM auto (FlashKDA) | gpu | 124.749, 124.749, 124.745 | 124.592–124.931 | 458.630 |
| GLM | 8 | FlashInfer auto | gpu | 92.269, 92.264, 92.273 | 92.121–92.466 | 339.222 |
| GLM | 8 | FlashInfer cuDNN | gpu | 50.546, 50.416, 50.435 | 50.115–50.870 | 185.536 |
| GLM | 32 | vLLM auto (FlashKDA) | gpu | 499.093, 499.070, 499.033 | 498.412–499.723 | 458.700 |
| GLM | 32 | FlashInfer auto | gpu | 369.097, 369.090, 369.071 | 368.565–369.787 | 339.234 |
| GLM | 32 | FlashInfer cuDNN | gpu | 201.833, 201.801, 201.873 | 200.372–202.829 | 185.511 |
