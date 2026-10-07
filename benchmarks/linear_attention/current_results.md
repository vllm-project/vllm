# Current GDN/KDA measurements

## Full serving rerun: pending

The requested current-stack rerun covers Qwen3-Next-80B-A3B-Instruct, Qwen3.5-35B-A3B, GLM-5.3-Flash and full Kimi-K3 on GB200 and GB300. It will compare vLLM auto, FI auto and FI cuDNN at 8192 input / 1024 output tokens, C1/C8/C32/C128, with correctness checks before timing. TTFT, TPOT and generated-token throughput will be reported only after validation. October 2 numbers are intentionally excluded.

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

Current GB300 prefill comparisons remain queued. GLM uses TP4; Kimi uses TP16/four nodes on GB200 and TP8/two nodes on GB300. GLM/Kimi BF16 state is a controlled comparison, not stock FP32. Sequential order and topology prevent a GPU-architecture or CPU-boundness conclusion.

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
