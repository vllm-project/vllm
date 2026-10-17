# Qwen3.8-27B dense mono decode (MI355X)

Opt-in FlyDSL decode for `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` (`qwen3_5_text`,
TP1 or TP2, gfx950). Set `VLLM_ROCM_MONO_DECODE=1`. The K1 and K2 kernels are adapted from
SGLang #43558. This pull request is their vLLM integration and the TP1
and TP2 dispatch.

On a pure decode inside the row cap, K1 replaces the linear-attention mixer
and K2 replaces the MXFP4 feed-forward. Norms, full attention, and the
tensor-parallel all-reduces stay outside those launches. Prefill, mixed
batches, spec decode, and wider decode call the compiled stock model with
its module methods left unwrapped.

## Enable

```bash
VLLM_ROCM_MONO_DECODE=1
```

The flag applies only to `qwen3_5_text`. Other model types that share the
environment variable stay on their own path. A set flag on a deployment
these kernels cannot run raises and lists the reason: TP other than 1 or 2,
a GPU other than gfx950, pipeline parallel, LoRA, a KV connector, sequence
parallel, speculative tokens, or a shape or quant method other than this
Quark MXFP4 27B dense MLP.

## When it runs

A step enters the mono graph only when it is a pure decode
(`num_prefills == 0`, at least one decode row, no spec-decode mask) and
the row count is inside the kernel build. The builds cover 1–8 rows at
TP1 and 1–16 rows at TP2.

Serving defaults are narrower than the builds. K1 and K2 each default to
4 rows at TP1 and 8 rows at TP2. Override either cap with
`VLLM_ROCM_QWEN38_DENSE_MONO_K1_MAX_ROWS` or
`VLLM_ROCM_QWEN38_DENSE_MONO_K2_MAX_ROWS`. A value outside 1..8 at TP1 or
1..16 at TP2 is clamped and a warning is logged.

TP1 compiles one whole-step graph and folds the norms into K1 and K2.
TP2 compiles a split graph: stock norms and the full-attention block stay
visible, and each rank's K1 and K2 partial is all-reduced after the launch.
`VLLM_ROCM_QWEN38_DENSE_MONO_OPROJ` (default on) folds `out_proj` into the
TP1 graph and into the TP2 K1 partial. `VLLM_ROCM_QWEN38_DENSE_MONO_K2`
(default on) selects the K2 launch on the TP2 graph.

## Measurements

TP1 serving image:
`nightly-aiter-5350-5669-5598-flydsl-qr4970-gdn57560-dense-mono-layer-graphs`.
TP2 serving image:
`nightly-aiter-5350-5669-5598-flydsl-qr4970-gdn57560-aiter6218-dense-mono-layer-graphs-tp2-split`.
The non-mono arm is the same stack on `...-gdn57560-aiter6218`. The TP1
cells below set both row caps to 8. TP2 uses the default cap of 8.

8k input / 1k output, `sglang.bench_serving`, random dataset, seed and
prompt counts from the OOB sweep. Output throughput is total output tokens
per second. TTFT, ITL, and TPOT are medians. A positive percent is an
improvement: more output tokens per second, or a lower median TTFT, ITL,
or TPOT.

**Where the mono kernels help (concurrency 4 and 8).** K1 and K2 run.
Output throughput, median ITL, and median TPOT improve on TP1 and TP2.
Median TTFT improves only at TP1 concurrency 4.

| TP | Concurrency | Output tok/s, non-mono → mono | Median TTFT (ms) | Median ITL (ms) | Median TPOT (ms) |
| --- | --- | --- | --- | --- | --- |
| 1 | 4 | 430.1 → 576.4 (+34.0%) | 202.4 → 201.0 (+0.7%) | 8.33 → 5.94 (+28.7%) | 8.89 → 6.51 (+26.7%) |
| 1 | 8 | 751.7 → 911.4 (+21.2%) | 206.7 → 208.4 (−0.8%) | 8.91 → 7.10 (+20.3%) | 10.09 → 8.30 (+17.8%) |
| 2 | 4 | 579.6 → 655.8 (+13.1%) | 180.2 → 183.3 (−1.7%) | 6.05 → 5.15 (+14.9%) | 6.57 → 5.66 (+13.9%) |
| 2 | 8 | 987.1 → 1080.7 (+9.5%) | 186.3 → 205.6 (−10.3%) | 6.56 → 5.79 (+11.8%) | 7.66 → 6.99 (+8.7%) |

**Where the mono kernels do not help (concurrency 16 and above).** The row
count is past the caps, so the mono kernels do not launch. These cells
replay the stock CUDA graph, including FlyDSL QuickReduce.

| TP | Concurrency | Output tok/s, non-mono → mono | Median TTFT (ms) | Median ITL (ms) | Median TPOT (ms) |
| --- | --- | --- | --- | --- | --- |
| 1 | 16 | 1223.0 → 1232.4 (+0.8%) | 206.3 → 207.3 (−0.5%) | 9.88 → 9.89 (−0.1%) | 12.36 → 12.42 (−0.5%) |
| 1 | 32 | 1742.4 → 1759.0 (+1.0%) | 344.3 → 230.6 (+33.0%) | 12.00 → 12.00 (+0.0%) | 16.96 → 16.97 (+0.0%) |
| 1 | 64 | 2337.9 → 2349.6 (+0.5%) | 375.2 → 376.6 (−0.4%) | 15.70 → 15.67 (+0.2%) | 25.66 → 25.61 (+0.2%) |
| 1 | 128 | 2799.3 → 2801.5 (+0.1%) | 560.5 → 577.4 (−3.0%) | 22.80 → 22.87 (−0.3%) | 41.91 → 41.81 (+0.2%) |
| 1 | 256 | 3066.0 → 3064.3 (−0.1%) | 918.3 → 934.0 (−1.7%) | 38.20 → 38.39 (−0.5%) | 77.23 → 77.35 (−0.2%) |
| 2 | 16 | 1547.3 → 1546.0 (−0.1%) | 189.5 → 189.6 (−0.1%) | 7.46 → 7.46 (+0.0%) | 9.78 → 9.82 (−0.4%) |
| 2 | 32 | 1987.2 → 1977.0 (−0.5%) | 206.9 → 208.1 (−0.6%) | 10.39 → 10.39 (+0.0%) | 14.92 → 15.10 (−1.2%) |
| 2 | 64 | 2759.7 → 2757.5 (−0.1%) | 340.4 → 341.8 (−0.4%) | 12.59 → 12.62 (−0.3%) | 21.54 → 21.64 (−0.5%) |
| 2 | 128 | 3349.2 → 3363.5 (+0.4%) | 531.9 → 617.7 (−16.1%) | 17.17 → 17.18 (+0.0%) | 34.59 → 34.62 (−0.1%) |
| 2 | 256 | 3728.1 → 3762.8 (+0.9%) | 885.5 → 900.7 (−1.7%) | 26.67 → 26.61 (+0.2%) | 63.01 → 62.05 (+1.5%) |

**Accuracy.** GPQA Diamond, thinking on, 32 threads, 198 questions. NIAH
is RULER `niah_single_2`. MMMU is `mmmu_val`, 900 questions, seed 4321.

| TP | Eval | non-mono → mono |
| --- | --- | --- |
| 1 | GPQA Diamond | 0.8232 → 0.8434 |
| 1 | NIAH 64k | 0.990 → 0.988 |
| 1 | NIAH 128k | 0.964 → 0.972 |
| 2 | GPQA Diamond | 0.8232 → 0.8333 |
| 2 | NIAH 64k | 0.988 → 0.988 |
| 2 | NIAH 128k | 0.972 → 0.984 |
| 2 | MMMU val | 0.5211 → 0.5256 |

TP2 GPQA mean answer length is 43,452 characters versus 41,440 for
non-mono (165/198 versus 163/198). An earlier TP2 build that wrapped the
stock modules scored 0.1818 on the same eval.

## Tests

```bash
pytest tests/models/qwen3_5/test_amd_dense_mono.py
```

`DenseFfnTest` checks K2 against the stock MXFP4 GEMM, Gemma norm, and
SiLU-mul, including the TP2 down partial before the all-reduce.
`DenseGdnTest` checks K1 against the stock conv update, gated delta rule,
and RMSNormGated, including pad rows and the TP2 `out_proj` partial.
`RowCapsTest`, `StockAccumulationTest`, `StockSplitTest`, and
`StockChainTest` check the row caps and that K2 follows the stock GEMM's
split-K order. The Buildkite job `qwen38-27b-dense-mono-numerics-amd`
runs this file on one MI355.

## Layout

```text
amd/dense_mono.py    eligibility, the compiled mono graph, vLLM tensors
amd/mono/
    dense_gdn.py     K1, the GDN mixer
    dense_ffn.py     K2, the MXFP4 FFN
    helpers.py       step tag, sigmoid, e8m0
    sources.py       kernel source digest
    common/          device ops, mailbox sync, MX helpers, plan / build keys
```
