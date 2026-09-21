# Attention group size findings

Qwen3.5 with DFlash, vLLM MRV2 + Triton Mamba gather, H100, TP1, concurrency 1, 100 requests + 10 warmup, up to 256 output tokens. Sizes 1/2/4/8 ran once on separate GPUs; 27B size 16 ran separately afterward.

| Model | Group size | ITL p50 (ms) | Output tok/s | % slower than SGLANG | Reported KV capacity (tokens) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4B | 1 | 9.049 | 555.5 | 34.5% | 838,554 |
| 4B | 2 | 6.220 | 785.5 | 7.3% | 773,369 |
| 4B | 4 | 6.032 | 824.1 | 2.8% | 669,348 |
| 4B | 8 | 5.889 | 844.4 | 0.4% | 575,168 |
| 27B | 1 | 19.649 | 330.3 | 9.4% | 215,934 |
| 27B | 2 | 18.908 | 348.7 | 4.3% | 207,649 |
| 27B | 4 | 18.585 | 354.8 | 2.7% | 192,917 |
| 27B | 8 | 18.430 | 361.0 | 1.0% | 177,708 |
| 27B | 16 | 18.165 | 367.9 | −0.9% | 140,773 |

“% slower” means lower throughput: 100 × (1 − vLLM tok/s ÷ SGLang tok/s). SGLang references: 4B 847.6 tok/s; 27B 364.5 tok/s. The 27B reference used 200 requests + 20 warmup, the older GSM8K prompt format, and different server limits, so that comparison is approximate.

- A singleton draft bucket makes the default heuristic choose one layer per group, multiplying target metadata builds. Larger groups substantially improve speed, supporting repeated per-group preparation as a bottleneck.
- **4B, size 8 versus 1:** throughput +52.0%, ITL −34.9%, reported cache capacity −31.4%. The 844.4 tok/s result nearly matches our earlier SGLang reference of 847.6 tok/s; SGLang was not rerun in this sweep.
- **27B, size 8 versus 1:** throughput +9.3%, ITL −6.2%, reported cache capacity −17.7%. The smaller relative gain is consistent with model computation taking a larger share of runtime.
- **27B, size 16 versus 8:** throughput +1.9%, ITL −1.4%, reported cache capacity −20.8%. This restores 3 GDN groups (6 total groups). Throughput is 0.9% above the historical SGLang reference, subject to the comparison caveat above.
- Acceptance length was identical across group sizes: **5.6523 for 4B, 7.4068 for 27B**. This does not establish full output correctness.
- Size 4 captures most of the speed gain with less capacity loss. Keep grouping opt-in; sharing metadata preparation across groups could recover speed without changing allocation.

These are single-run results, not a universal grouping policy. The 4B size-1 run was slower than the previous 608.6 tok/s measurement, so the exact gain needs repeated runs. Startup included fresh compilation and was excluded from throughput measurements.

Detailed results and log locations: [4B](results/group-size-100-10/comparison.md), [27B](results/group-size-27b-100-10/comparison.md). Raw artifacts are local and ignored by Git.
