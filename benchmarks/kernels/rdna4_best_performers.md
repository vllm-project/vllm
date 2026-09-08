# RDNA4 all-reduce best-performer matrix

Measured on 4× AMD Radeon AI PRO R9700 GPUs with `NCCL_PROTO=Simple`, 20 warmups, 9 median samples, and exact BF16 correctness rechecked with changed inputs after timing/graph replay. The route admits FlyDSL when `PyNCCL latency / FlyDSL latency >= 0.98`; otherwise it retains PyNCCL.

| TP | Execution | Size (bytes) | Routed best | Routed ms | PyNCCL ms | Speedup | Routed bus BW (GB/s) |
|---|---|---:|---|---:|---:|---:|---:|
| TP2 | eager | 8192 | pynccl | 0.015829 | 0.016055 | 1.0143× | 0.518 |
| TP2 | eager | 32768 | pynccl | 0.016297 | 0.016216 | 0.9950× | 2.011 |
| TP2 | eager | 65520 | pynccl | 0.016845 | 0.016770 | 0.9956× | 3.890 |
| TP2 | eager | 65536 | pynccl | 0.017393 | 0.017318 | 0.9957× | 3.768 |
| TP2 | eager | 65552 | pynccl | 0.016622 | 0.016604 | 0.9989× | 3.944 |
| TP2 | eager | 131072 | pynccl | 0.018930 | 0.018943 | 1.0007× | 6.924 |
| TP2 | eager | 262144 | pynccl | 0.022332 | 0.022597 | 1.0119× | 11.738 |
| TP2 | eager | 524288 | pynccl | 0.029683 | 0.029585 | 0.9967× | 17.663 |
| TP2 | eager | 1048576 | rdna4 | 0.036870 | 0.041790 | 1.1334× | 28.439 |
| TP2 | eager | 2097152 | rdna4 | 0.059219 | 0.067818 | 1.1452× | 35.414 |
| TP2 | eager | 4194304 | rdna4 | 0.105781 | 0.120190 | 1.1362× | 39.651 |
| TP2 | eager | 8388608 | rdna4 | 0.203752 | 0.231049 | 1.1340× | 41.171 |
| TP2 | eager | 16777216 | rdna4 | 0.393297 | 0.450502 | 1.1455× | 42.658 |
| TP2 | eager | 67108864 | rdna4 | 1.508800 | 1.770569 | 1.1735× | 44.478 |
| TP2 | eager | 134217728 | rdna4 | 2.985636 | 3.514967 | 1.1773× | 44.954 |
| TP2 | graph | 8192 | rdna4 | 0.007984 | 0.017691 | 2.2158× | 1.026 |
| TP2 | graph | 32768 | rdna4 | 0.009557 | 0.018216 | 1.9060× | 3.429 |
| TP2 | graph | 65520 | rdna4 | 0.011887 | 0.018909 | 1.5907× | 5.512 |
| TP2 | graph | 65536 | rdna4 | 0.011893 | 0.019156 | 1.6107× | 5.510 |
| TP2 | graph | 65552 | rdna4 | 0.018002 | 0.018928 | 1.0515× | 3.641 |
| TP2 | graph | 131072 | rdna4 | 0.018534 | 0.020848 | 1.1249× | 7.072 |
| TP2 | graph | 262144 | rdna4 | 0.021465 | 0.023895 | 1.1132× | 12.213 |
| TP2 | graph | 524288 | rdna4 | 0.027380 | 0.031022 | 1.1330× | 19.148 |
| TP2 | graph | 1048576 | rdna4 | 0.039435 | 0.043543 | 1.1042× | 26.590 |
| TP2 | graph | 2097152 | rdna4 | 0.062818 | 0.069646 | 1.1087× | 33.384 |
| TP2 | graph | 4194304 | rdna4 | 0.107925 | 0.121977 | 1.1302× | 38.863 |
| TP2 | graph | 8388608 | rdna4 | 0.206638 | 0.232984 | 1.1275× | 40.596 |
| TP2 | graph | 16777216 | rdna4 | 0.393318 | 0.452415 | 1.1503× | 42.656 |
| TP2 | graph | 67108864 | rdna4 | 1.504952 | 1.771313 | 1.1770× | 44.592 |
| TP2 | graph | 134217728 | rdna4 | 2.985023 | 3.527894 | 1.1819× | 44.964 |
| TP4 | eager | 8192 | rdna4 | 0.026908 | 0.040258 | 1.4961× | 0.457 |
| TP4 | eager | 32768 | pynccl | 0.030884 | 0.030822 | 0.9980× | 1.592 |
| TP4 | eager | 65536 | rdna4 | 0.031672 | 0.031804 | 1.0042× | 3.104 |
| TP4 | eager | 131072 | rdna4 | 0.030790 | 0.033609 | 1.0916× | 6.386 |
| TP4 | eager | 196592 | rdna4 | 0.032781 | 0.036075 | 1.1005× | 8.996 |
| TP4 | eager | 196608 | rdna4 | 0.032723 | 0.035893 | 1.0969× | 9.012 |
| TP4 | eager | 196624 | rdna4 | 0.033006 | 0.036113 | 1.0941× | 8.936 |
| TP4 | eager | 262144 | pynccl | 0.039592 | 0.039433 | 0.9960× | 9.932 |
| TP4 | eager | 393216 | pynccl | 0.041971 | 0.041991 | 1.0005× | 14.053 |
| TP4 | eager | 524288 | pynccl | 0.047075 | 0.047073 | 1.0000× | 16.706 |
| TP4 | eager | 786432 | pynccl | 0.058653 | 0.058593 | 0.9990× | 20.112 |
| TP4 | eager | 1048512 | pynccl | 0.066943 | 0.066994 | 1.0008× | 23.494 |
| TP4 | eager | 1048576 | rdna4 | 0.061435 | 0.066670 | 1.0852× | 25.602 |
| TP4 | eager | 1048640 | rdna4 | 0.061743 | 0.065740 | 1.0647× | 25.476 |
| TP4 | eager | 2097152 | rdna4 | 0.097656 | 0.102551 | 1.0501× | 32.212 |
| TP4 | eager | 4194304 | rdna4 | 0.171512 | 0.176238 | 1.0275× | 36.682 |
| TP4 | eager | 8388608 | rdna4 | 0.319030 | 0.324242 | 1.0163× | 39.441 |
| TP4 | eager | 16777216 | rdna4 | 0.616008 | 0.620828 | 1.0078× | 40.853 |
| TP4 | eager | 33554368 | rdna4 | 1.256110 | 1.241430 | 0.9883× | 40.069 |
| TP4 | eager | 33554432 | rdna4 | 1.219715 | 1.241390 | 1.0178× | 41.265 |
| TP4 | eager | 33554496 | rdna4 | 1.221324 | 1.261235 | 1.0327× | 41.211 |
| TP4 | eager | 50331584 | rdna4 | 1.832342 | 1.860383 | 1.0153× | 41.203 |
| TP4 | eager | 50331648 | rdna4 | 1.811878 | 1.860006 | 1.0266× | 41.668 |
| TP4 | eager | 50331712 | rdna4 | 1.810087 | 1.861189 | 1.0282× | 41.709 |
| TP4 | eager | 67108864 | rdna4 | 2.388987 | 2.471528 | 1.0346× | 42.136 |
| TP4 | eager | 134217728 | rdna4 | 4.899824 | 4.924982 | 1.0051× | 41.089 |
| TP4 | graph | 8192 | rdna4 | 0.009611 | 0.031691 | 3.2972× | 1.278 |
| TP4 | graph | 32768 | rdna4 | 0.012697 | 0.032622 | 2.5693× | 3.871 |
| TP4 | graph | 65536 | rdna4 | 0.016796 | 0.033573 | 1.9988× | 5.853 |
| TP4 | graph | 131072 | rdna4 | 0.026811 | 0.035568 | 1.3266× | 7.333 |
| TP4 | graph | 196592 | rdna4 | 0.035745 | 0.037754 | 1.0562× | 8.250 |
| TP4 | graph | 196608 | rdna4 | 0.035424 | 0.037801 | 1.0671× | 8.325 |
| TP4 | graph | 196624 | rdna4 | 0.035585 | 0.037900 | 1.0651× | 8.288 |
| TP4 | graph | 262144 | pynccl | 0.041018 | 0.040938 | 0.9981× | 9.586 |
| TP4 | graph | 393216 | pynccl | 0.043700 | 0.043771 | 1.0016× | 13.497 |
| TP4 | graph | 524288 | pynccl | 0.048937 | 0.049003 | 1.0013× | 16.070 |
| TP4 | graph | 786432 | pynccl | 0.060257 | 0.060350 | 1.0015× | 19.577 |
| TP4 | graph | 1048512 | pynccl | 0.068091 | 0.068247 | 1.0023× | 23.098 |
| TP4 | graph | 1048576 | rdna4 | 0.063752 | 0.068331 | 1.0718× | 24.671 |
| TP4 | graph | 1048640 | rdna4 | 0.064269 | 0.067405 | 1.0488× | 24.475 |
| TP4 | graph | 2097152 | rdna4 | 0.099922 | 0.103753 | 1.0383× | 31.482 |
| TP4 | graph | 4194304 | rdna4 | 0.174419 | 0.176632 | 1.0127× | 36.071 |
| TP4 | graph | 8388608 | rdna4 | 0.320367 | 0.326221 | 1.0183× | 39.277 |
| TP4 | graph | 16777216 | rdna4 | 0.611658 | 0.621400 | 1.0159× | 41.144 |
| TP4 | graph | 33554368 | rdna4 | 1.258794 | 1.241662 | 0.9864× | 39.984 |
| TP4 | graph | 33554432 | rdna4 | 1.220371 | 1.243524 | 1.0190× | 41.243 |
| TP4 | graph | 33554496 | rdna4 | 1.228878 | 1.260712 | 1.0259× | 40.957 |
| TP4 | graph | 50331584 | rdna4 | 1.833901 | 1.855136 | 1.0116× | 41.168 |
| TP4 | graph | 50331648 | rdna4 | 1.797228 | 1.858183 | 1.0339× | 42.008 |
| TP4 | graph | 50331712 | rdna4 | 1.803292 | 1.857307 | 1.0300× | 41.867 |
| TP4 | graph | 67108864 | rdna4 | 2.383466 | 2.470583 | 1.0366× | 42.234 |
| TP4 | graph | 134217728 | rdna4 | 4.882761 | 4.924318 | 1.0085× | 41.232 |

## Current routing summary

| Region | FlyDSL cases | PyNCCL fallback cases |
|---|---:|---:|
| TP2 eager | 7 | 8 |
| TP2 graph | 15 | 0 |
| TP4 eager | 20 | 6 |
| TP4 graph | 21 | 5 |
| **Total** | **63** | **19** |

The earlier 35 fallback cases are reduced to 19. The recovered 16 cases are TP2 graph at 65,552/131,072/262,144/524,288/1,048,576 bytes; TP2 eager at 1,048,576 bytes; TP4 eager and graph at 196,624 bytes; TP4 eager at 1,048,576/1,048,640/2,097,152/4,194,304/8,388,608/16,777,216/33,554,368 bytes; and TP4 graph at 33,554,368 bytes.

Remaining fallbacks:

- TP2 eager: 8192, 32768, 65520, 65536, 65552, 131072, 262144, 524288
- TP2 graph: none
- TP4 eager: 32768, 262144, 393216, 524288, 786432, 1048512
- TP4 graph: 262144, 393216, 524288, 786432, 1048512

## Geomean and weighted analysis

The accepted independent GEAK optimization result remains **1.716492×** versus its frozen original 74-case routed baseline. Against same-run PyNCCL, the updated 82-case hybrid routes 63 cases to FlyDSL and has a **1.122639×** uniform-case geomean. The FlyDSL-selected subset alone is **1.162357×** versus PyNCCL. All admitted FlyDSL cases pass the 0.98× gate; the lowest single-sweep ratio is **0.986390×**.

| Weighting view | Hybrid speedup vs PyNCCL | Equivalent latency reduction |
|---|---:|---:|
| Uniform across 82 cases | 1.122639× | 10.92% |
| Equal 25% per TP/execution quadrant | 1.132238× | 11.68% |
| Weight proportional to message bytes | 1.065666× | 6.16% |
| Aggregate sequential suite time | 1.055154× | 5.23% |

No serving-workload frequency distribution was supplied, so these weighting views are sensitivity analyses rather than a production traffic estimate.

## Porting finding

The accepted GPU implementations in the RDNA4 path are Python-authored FlyDSL. A literal FlyDSL port of the optimized one-stage HIP structure was exact but measured only about 0.53–0.93× PyNCCL in eager mode and 0.82–0.91× in the tested graph region, so it was rejected and removed. The installed AITER HIP path was also rejected as an oracle on gfx1201 because changed-input checks exposed stale-peer results after timing; TP4 additionally showed a deadlock/incorrectness case. PyNCCL remains only as the guarded fallback for regions where no FlyDSL candidate clears the gate.
