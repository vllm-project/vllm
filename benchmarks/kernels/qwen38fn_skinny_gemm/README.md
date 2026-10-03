# Qwen3.8-Flash-Next skinny GEMM tuning on RTX PRO 6000 (sm_120)

`benchmark_skinny_gemm.py` searches `SkinnyGemmConfig` space for the unquantized
projections of Qwen3.8-Flash-Next and keeps only the configs that beat
`F.linear`. Its output is `QWEN4_EXP_SM120_GEMM_PLANS` in
`vllm/models/qwen4_exp/nvidia/low_latency_gemm.py`, the plan table that file
consults at model build time.

## Why

`_gemm_plans()` returns `{}` on anything that is not sm_103, sm_100 or sm_90,
so the CuTe skinny GEMM is dead on RTX PRO 6000. The shipped plans are also keyed
by the TP-local `(N, K)`, and every entry was measured at TP=4 — at TP=1 the same
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
# default shape set: Qwen3.8-FN at TP=1 and TP=2 (shapes follow the model, not the GPU)
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

## Measured on RTX 5090 (sm_120), 2026-10-03

31046 measurements, 58 shape/M points, TP=1 and TP=2, every one with the weights
rotated past the 96 MiB L2. Speedup over `F.linear`; a dash means no candidate
beat it and that point keeps the standard implementation.

| (N, K) | layer | M=1 | M=2 | M=4 | M=8 | M=16 |
| --- | --- | --- | --- | --- | --- | --- |
| (248320, 2560) | LM head | 1.12x | 1.06x | 1.06x | 1.04x | - |
| (124160, 2560) | LM head, TP=2 | 1.18x | 1.13x | 1.13x | 1.12x | 1.03x |
| (16384, 2560) | GDN fused QKVZ | 1.07x | 1.18x | 1.18x | 1.13x | - |
| (14336, 2560) | QSA fused QKV/gate | 1.07x | 1.36x | 1.36x | 1.23x | - |
| (8192, 2560) | GDN fused QKVZ, TP=2 | 1.34x | 1.31x | 1.30x | 1.25x | - |
| (7168, 2560) | QSA fused QKV/gate, TP=2 | 1.36x | 1.33x | 1.32x | 1.28x | - |
| (2560, 3072) | GDN and QSA output, TP=2 | 1.35x | 1.95x | 1.89x | 1.79x | 1.51x |
| (2560, 6144) | GDN and QSA output | 1.36x | 1.95x | 1.93x | 1.84x | 1.57x |
| (1280, 2560) | shared-expert gate/up | 1.59x | 3.36x | 3.26x | 2.92x | 1.93x |
| (640, 2560) | QSA indexer Q/K | 1.62x | 2.05x | 1.95x | 1.51x | - |
| (336, 10240) | HC merged down/injection | 1.65x | 1.88x | 1.76x | 1.40x | - |
| (96, 2560) | GDN fused B/A | 2.32x | 12.99x | 11.41x | 8.94x | 1.55x |
| (48, 2560) | GDN fused B/A, TP=2 | 2.60x | 14.48x | 13.43x | 10.34x | 6.96x |

The gain peaks at M=2 and M=4 and is modest at M=1. That shape comes from the
baseline, not from this kernel: at M=1 cuBLAS dispatches a dedicated GEMV, and
from M=2 it must use a tiled GEMM whose minimum tile height it then pads. On the
shared-expert projection `F.linear` jumps from 0.0068 to 0.0145 ms between M=1
and M=2 and stays flat to M=16, while the skinny kernel climbs smoothly from
0.0042 to 0.0076. With MTP the verification GEMM runs at M = (k + 1) *
batch_size, so batch size 1 lands exactly in that window at both k=1 and k=3.

The LM head is the exception: 1.06x, flat across M. It is bandwidth-bound, so
cuBLAS wastes nothing there and there is nothing to win. That is the opposite of
GB10, where the LM head was 84% of the end-to-end gain -- this card has 6.5x the
bandwidth, which removes bandwidth as the binding constraint on every shape
except the largest.

Measured on an RTX 5090 because it is the sm_120 part this cluster offers with
the scratch mounted. Qwen3.8-Flash-Next itself does not fit in 32 GB and needs an
RTX PRO 6000; the sweep does not care, since it measures kernels on synthetic
tensors and never instantiates the model.

## Caveats

These are isolated kernels. Which of them a deployment actually reaches depends
on the checkpoint: on GB10 only five of thirteen shapes appeared in a profile,
because the quantization config covered the per-layer projections and they never
reached `UnquantizedLinearMethod`. **These numbers are per-kernel ratios, not
weighted end-to-end gains**, and weighting them by layer count without checking
the profile first is how the GB10 estimate came out a factor of two high.

The verification GEMM leaves the tuned range above batch size 8 at k=1 and above
4 at k=3, since the kernel refuses M > 16. The gain is a low-batch latency
effect by construction.

The FP32 correctness bar catches gross breakage, not numerical quality.
