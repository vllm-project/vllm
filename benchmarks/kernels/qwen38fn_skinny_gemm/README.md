# Qwen3.8-Flash-Next skinny GEMM tuning on GB10 (sm_121)

`benchmark_skinny_gemm.py` searches `SkinnyGemmConfig` space for the unquantized
projections of Qwen3.8-Flash-Next and keeps only the configs that beat
`F.linear`. Its output is `QWEN4_EXP_SM121_GEMM_PLANS` in
`vllm/models/qwen4_exp/nvidia/low_latency_gemm.py`, the plan table that file
consults at model build time.

## Why

`_gemm_plans()` returned `{}` on anything that was not sm_103 or sm_90, so the
CuTe skinny GEMM was dead on GB10. The shipped plans are also keyed by the
TP-local `(N, K)`, and every entry was measured at TP=4 — at TP=1 the same
projections have four times the rows and miss the table entirely.

This tuner is adapted from `benchmark_k3_cutedsl_residual.py`, which produced
the Kimi-K3 tables. Differences: `(N, K)` comes from the CLI instead of module
constants, there is no SM gate, it drives the installed
`shape_dynamic_skinny_gemm` rather than a kernel loaded from a path, and it
scores against `F.linear` — the retention criterion upstream applies by hand.

## Usage

Needs `cutlass`/`quack`, so run it inside the vLLM container. No server, model
or weights required.

```bash
# default shape set: Qwen3.8-FN at TP=1 and TP=2
python benchmark_skinny_gemm.py --mode sweep --tp 1 --m 1 4 --out sweep.jsonl

# one shape, quick check
python benchmark_skinny_gemm.py --shape 248320,2560 --m 1 --limit 5

# split across nodes (SLURM array); --out defaults to a per-shard name
python benchmark_skinny_gemm.py --mode sweep \
  --num-config-shards "$SLURM_ARRAY_TASK_COUNT" --config-shard "$SLURM_ARRAY_TASK_ID"

# then merge the shards into the global plan
python benchmark_skinny_gemm.py --merge 'skinny_sweep.shard*.jsonl'
```

Writes one JSON line per measurement and prints a paste-ready plan dict.

Sharding needs both halves. Each shard measures a disjoint slice of the search
space, so the plan it prints is the best over that slice alone, and the merge
step is what turns those into global winners. Each shard also truncates its own
output file, so they must not share a path -- hence the per-shard default.

## Method

- Candidates are pre-filtered by the two constraints `_compile()` enforces:
  `N % outputs_per_block == 0` and `K % (block_size * vector_width) == 0`.
- Every timing goes through CUDA graph replay. At 17 us per call, eager launch
  overhead would dominate the measurement, and vLLM decodes inside graphs anyway.
- Weights rotate through a working set larger than L2, so the cold regime is the
  one being measured. Each row records `working_set_bytes` and `truly_cold`, so a
  run where the buffer cap kept everything in L2 is visible rather than mislabelled.
- Replay counts are sized per shape to a fixed wall-time budget: one constant
  cannot serve both a 17 us and a 7.5 ms kernel.
- Correctness is scored against an FP32 reference, and the bar is "no worse than
  `F.linear`". Comparing the two BF16 paths directly fails on the near-zero
  outputs that cancellation produces, even when both are accurate.
- A config is retained only if it wins both hot and cold.

## How the table was produced

Four sharded sweeps, 38 JSONL files, merged with `--merge`:

| sweep | shapes | M | shards |
| --- | --- | --- | --- |
| TP=1 | 8 | 1, 4 | 8 |
| TP=2 | 8 | 1, 4 | 8 + 6 reruns |
| gap fill | `(336, 10240)`, `(640, 2560)` | 1, 4 | 4 |
| wider M | all | 2, 8, 16 | 8 |

**25911 valid measurements**, 92 rejected at compile or launch, covering 13
shapes x M in {1, 2, 4, 8, 16} = 65 points. 62 points have a config that beats
`F.linear`; the remaining three keep the standard implementation.

The gap fill exists because the first run scored candidates against cuBLAS BF16
with an elementwise `rtol`, which silently discarded three whole shape/M
combinations. The two BF16 paths differ only in summation order, but on the
near-zero outputs that cancellation produces a relative tolerance explodes, and
`(336, 10240)` -- K is four times longer, so there is four times the
cancellation -- failed at every single config. Scoring against FP32 instead
showed the skinny kernel was in fact *more* accurate than `F.linear` on exactly
those points.

Validation on top of the sweep: one nsys profile at batch size 1, k=3; a paired
patched/baseline bench over bs=1..32 for k in {1, 3}; and three repetitions per
arm of the bs=1,2,4 points, after a single run proved too noisy to separate the
effect from run-to-run variation.

The whole campaign was 149 SLURM jobs and 62 node-hours of exclusive GB10. A
third of the jobs failed, almost all to two scheduling mistakes rather than to
anything in this benchmark: billing an account whose reservation carries the
MAGNETIC flag, which pins every job to a handful of nodes that are mostly down,
and requesting `--gres=gpu:2` on a one-GPU node. Budget for the 62 node-hours,
not the job count.

## Measured on GB10, 2026-10-01

25911 measurements over 62 shape/M points, covering the whole M range the kernel
accepts at TP=1 and TP=2. Speedup over `F.linear`, cold; a dash means no
candidate beat `F.linear` and that point keeps the standard implementation.

| (N, K) | layer | M=1 | M=2 | M=4 | M=8 | M=16 |
| --- | --- | --- | --- | --- | --- | --- |
| (248320, 2560) | LM head | 1.42x | 1.03x | 1.04x | 1.04x | 1.05x |
| (124160, 2560) | LM head, TP=2 | 1.43x | 1.02x | 1.03x | 1.04x | 1.04x |
| (16384, 2560) | GDN fused QKVZ | 1.74x | 1.22x | 1.23x | 1.23x | 1.21x |
| (14336, 2560) | QSA fused QKV/gate | 1.57x | 1.13x | 1.14x | 1.15x | 1.14x |
| (8192, 2560) | GDN fused QKVZ, TP=2 | 1.70x | 1.28x | 1.28x | 1.30x | 1.28x |
| (7168, 2560) | QSA fused QKV/gate, TP=2 | 1.55x | 1.15x | 1.21x | 1.17x | 1.17x |
| (2560, 3072) | GDN and QSA output, TP=2 | 1.72x | 1.26x | 1.26x | 1.26x | - |
| (2560, 6144) | GDN and QSA output | 1.19x | 1.23x | 1.22x | 1.23x | 1.17x |
| (1280, 2560) | shared-expert gate/up | 1.82x | 1.49x | 1.43x | 1.46x | 1.27x |
| (640, 2560) | QSA indexer Q/K | 1.28x | 1.50x | 1.47x | 1.44x | 1.28x |
| (336, 10240) | HC merged down/injection | 1.40x | 1.35x | 1.35x | - | - |
| (96, 2560) | GDN fused B/A | 1.87x | 7.89x | 8.08x | 6.43x | 1.61x |
| (48, 2560) | GDN fused B/A, TP=2 | 2.01x | 8.77x | 7.50x | 6.50x | 4.31x |

M=1 is where the kernel earns its keep, which is the regime it was designed for:
it holds the activation rows in registers and streams the weight with scalar
FMAs, so a tensor-core MMA tile would be mostly padding. By M=2 most shapes fall
to 1.03-1.3x, and `__call__` refuses M > 16 outright.

The 7-8x on the B/A projections is cuBLAS falling over, not the kernel
excelling: `F.linear` on `(96, 2560)` takes 0.0176 ms at M=4 against 0.0039 ms
at M=1, getting slower with more work. The skinny kernel is flat at 0.0022 ms.

## End-to-end on Qwen3.8-Flash-Next-NVFP4

Measured with nsys at batch size 1, k=3, against the same recipe without the
table: **111.9 -> 104.4 ms/token, -6.7%**. A paired aiperf sweep over three
repetitions per arm puts the same point at **-6.8% +- 2.2%** (3.1 sigma), so GPU
time per token and client-side latency agree to within a tenth of a point.

The effect is significant at batch size 1 for both k (-2.9% at k=1, 6.4 sigma)
and falls to the noise floor from batch size 2 upward, as the M <= 16 ceiling
requires.

Of that, 84% is the LM head alone:

| | baseline | patched | saving |
| --- | --- | --- | --- |
| LM head M=1, x3 per token | 21.92 | 15.62 | 6.30 ms/token |
| LM head M=4, x1 per token | 5.64 | 5.61 | 0.03 |
| everything else | | | 1.21 |
| total | 111.9 | 104.4 | 7.54 |

Only five of the thirteen tuned shapes appear in the profile at all. On this
checkpoint `group_mxfp8_attention` covers `in_proj_qkv`, `in_proj_a` and
`in_proj_b`, and `group_mxfp8_shared_experts` covers the shared-expert
projections, so those layers are not `UnquantizedLinearMethod` and
`enable_qwen4_exp_low_latency_gemm` skips them whatever the table says. The GDN
QKVZ projection, which a layer-count weighting would make the largest single
contributor at 2.29 ms/token, contributes nothing: that kernel is not in the
profile.

What does run on all 48 layers is the HC projection. The other live shapes are
the MTP layer's own copies, which run three times per token on one layer rather
than once on 48.

Half the profile -- 20.5 s of 40.5 -- sits in the quantized CUTLASS and
FlashInfer paths, structurally out of reach for this table. The entries for the
shapes that stay dark are still correct measurements and would apply to a
checkpoint quantized differently, but on this one they are dead weight.

## Scope

The plan table is keyed by the TP-local `(N, K)` and selected by compute
capability, not by checkpoint. `enable_qwen4_exp_low_latency_gemm` is called
from four places -- `nvidia/model.py`, `nvidia/mtp.py` and their AMD
counterparts -- so any `qwen4_exp` checkpoint running on sm_121 picks these
configs up wherever its shapes match, not just Qwen3.8-Flash-Next at TP=1 or
TP=2. The measurements are shape-based and model-independent, so they stay
valid under that reuse, but the shapes were chosen from one model: a different
`qwen4_exp` checkpoint may hit some of these keys and miss others.

## Caveats

The sweep measures isolated kernels. Before weighting them by layer count,
check which layers actually reach the unquantized path -- a quantization config
can make most of the table unreachable, which is what happened here.

The verification GEMM runs at M = (k + 1) * batch_size, so it leaves the tuned
range above batch size 8 at k=1 and above 4 at k=3. The gain is a low-batch
latency effect by construction, not a throughput one.

The FP32 correctness bar catches gross breakage, not numerical quality.
