# Experiment log

Every run, verbatim. Commands are self-contained — all workload parameters are
passed on the command line rather than baked into `scripts/workload.env`, so
each experiment is reproducible from this file alone.

Prefix for every command:

```bash
R=/path/to/vllm-inco          # repo root
M="$R/.venv/bin/modal"        # modal CLI (see Setup)
```

Artifacts land on the `inco-results` Modal volume under the `--label` given,
and are pulled locally with:

```bash
$M volume get inco-results <label> ./inco/results
```

---

## Setup (once)

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install modal && modal setup          # browser OAuth

# cache the 61GB of weights on a CPU container (minutes, cents)
$M run $R/inco/modal/modal_baseline.py::prefetch
```

Pinned environment: `nvidia/cuda:13.0.3-devel-ubuntu24.04`, vLLM installed from
this fork via `VLLM_USE_PRECOMPILED=1` with the wheel pinned to the checkout's
merge-base with `origin/main`. A CUDA *devel* base is required: FlashInfer JIT-
compiles its sampling kernels with `nvcc` during engine init.

---

## Standing configuration

Every measured run below uses:

| flag | value | why |
|---|---|---|
| `--kv-cache-memory-bytes` | 12 GiB (131,072 tokens) | pinned, because Modal's `gpu="H100"` returns either an 80GB HBM3 or a 94GB NVL, which at `--gpu-memory-utilization 0.90` leaves ~12.5 vs ~25 GiB of KV and doubles servable concurrency between runs |
| `--gpu-memory-utilization` | 0.90 | |
| `--max-model-len` | 4096 | |
| `--max-num-batched-tokens` | 8192 | |
| `--async-scheduling`, `--enable-prefix-caching` | on | audited before measuring |
| `ignore_eos:true` | on | pins OSL exactly; without it OSL is model-dependent |
| `enable_thinking` | false | via `--default-chat-template-kwargs` |

Client side: 8 waves per point (`request_count = 8 x concurrency`), warmup =
`max(16, concurrency)`, prefix cache flushed between points, concurrencies
chosen from vLLM's `cudagraph_capture_sizes` so no point pays graph padding.

---

## Baseline — 1024 in / 256 out

**`baseline-v2`** — the reference baseline. 15 points, single server instance.

```bash
$M run $R/inco/modal/modal_baseline.py \
  --label baseline-v2 --isl 1024 --osl 256 \
  --concurrencies "1,2,4,8,16,24,32,40,48,56,64,72,80,88,96" \
  --max-num-seqs 96
```

Result: 202 -> 3825 tok/s/gpu as tok/s/user falls 210 -> 43. Never plateaus;
still +27 tok/s/gpu per added request at c=96. KV ceiling is 102 requests.

**`baseline-dense-96`** — same sweep, run before the warmup/cap fixes below.
Kept because it agrees with `baseline-v2` within -1.9%..+2.9% at every point,
which is what established that those fixes do not matter at c<=96.

```bash
$M run $R/inco/modal/modal_baseline.py \
  --label baseline-dense-96 --isl 1024 --osl 256 \
  --concurrencies "1,2,4,8,16,24,32,40,48,56,64,72,80,88,96" \
  --max-num-seqs 96
```

---

## Saturation demonstration — 128 in / 128 out

At 259 tokens/request the KV ceiling rises to 506, far enough past the
crossover `B* = W/c ~= 235` for the plateau to be visible. This is the only
workload shape on this GPU where saturation is reachable.

**`short-128-128-v2`**

```bash
$M run $R/inco/modal/modal_baseline.py \
  --label short-128-128-v2 --isl 128 --osl 128 \
  --concurrencies "1,2,4,8,16,32,64,128,192,256,320,384,448" \
  --max-num-seqs 448
```

Result: rises to ~11,000 tok/s/gpu and flattens from c~320. Marginal gain per
added request decays 39.5 -> 13.5 -> 7.9 -> 4.2 -> 0.2.

**`short-128-128-top`** — top region with the batch cap raised above the sweep.

```bash
$M run $R/inco/modal/modal_baseline.py \
  --label short-128-128-top --isl 128 --osl 128 \
  --concurrencies "192,256,288,320,352,384,416,448" \
  --max-num-seqs 512
```

Run because `short-128-128-v2` showed a 2% dip at its last point, where
`concurrency == max_num_seqs` left the scheduler zero slack. With the cap at
512 the dip is gone (c=448 -> 11,106), so it was an artifact of the cap, not
over-saturation.

**`top-r1` / `top-r2` / `top-r3`** — three repeats for error bars, because the
two runs above disagreed about which point in the plateau was the peak.

```bash
for i in 1 2 3; do
  $M run $R/inco/modal/modal_baseline.py \
    --label top-r$i --isl 128 --osl 128 \
    --concurrencies "320,352,384,416,448" --max-num-seqs 512 &
done
```

---

## Realistic chat — 1024 in / 512 out

A ~380-word answer to a ~1K-token prompt: the most representative single shape
for interactive chat. 1539 tokens/request gives a KV ceiling of **85
requests**, so 80 is the largest captured graph size that fits (93.5% of KV).
Crossover is at `B* ~= 77`, so this reaches the knee and stops — same story as
the 1024/256 baseline, marginally earlier.

```bash
$M run $R/inco/modal/modal_baseline.py \
  --label chat-1024-512 --isl 1024 --osl 512 \
  --concurrencies "1,2,4,8,16,24,32,40,48,56,64,72,80" \
  --max-num-seqs 80
```

~19 min.

## Long generation — 1024 in / 1024 out

More realistic output length. At 2051 tokens/request the KV ceiling drops to
**63.9 requests**, so 64 would exceed it by 192 tokens; 56 is the largest
captured graph size that fits (87.6% of KV).

```bash
$M run $R/inco/modal/modal_baseline.py \
  --label long-1024-1024 --isl 1024 --osl 1024 \
  --concurrencies "1,2,4,8,16,24,32,40,48,56" \
  --max-num-seqs 56
```

Takes ~30 min: OSL=1024 is 4x the output tokens of the baseline, and the
low-concurrency points are sequential (c=1 alone is ~7 min).

---

## Diagnostics (not measurements)

**`jit-probe`** — enumerate in-inference Triton compilations.

```bash
$M run $R/inco/modal/modal_baseline.py \
  --label jit-probe --isl 128 --osl 128 --concurrencies "1,32" \
  --extra-serve-args "--jit-monitor-verbose"
```

**`moe-shapes`** — logs which MoE tile config gets selected, and by which token
count. Needs a one-off instrumentation hook in
`try_get_optimal_moe_config()` (`vllm/model_executor/layers/fused_moe/fused_moe.py`)
that logs `(M, top_k) -> BLOCK_M/N/K, stages, warps` once per distinct config,
gated on `INCO_MOE_SHAPE_LOG`. Not committed — `inco/patches/` is gitignored,
since it is diagnostic scaffolding rather than part of the harness.

```bash
$M run $R/inco/modal/modal_baseline.py \
  --label moe-shapes --isl 128 --osl 128 --concurrencies "1,32" \
  --moe-shape-log --extra-serve-args "--jit-monitor-verbose"
```

What it found — the MoE tile config is selected by *nearest token count*, so
each distinct M bucket compiles a separate Triton kernel:

| M (tokens) | BLOCK_M | BLOCK_N | BLOCK_K | warps |
|---|---|---|---|---|
| 2 | 16 | 64 | 128 | 4 |
| 64 | 32 | 64 | 128 | 4 |
| 96 | 32 | 128 | 64 | 4 |
| 128 | 64 | 128 | 64 | 4 |
| 8192 | 128 | 128 | 64 | 8 |

Startup warms the `warps=4` variants; every in-inference compile was `warps=8`,
i.e. a config startup never touched. Each compile costs ~350ms and they come in
pairs (the MoE's two GEMMs), so ~700ms lands in TTFT. `enable_jit_warmup`
defaults on but is only wired into attention paths, never MoE. One config
compiled twice with identical visible parameters, which is still unexplained.

Only reproduces at short ISL: at ISL=1024 the prefill lands in an
already-warmed bucket and TTFT p99/mean is 1.2x instead of 11x. So it does not
affect the baseline workload.

Both perturb the measurement: together they cost ~13% throughput
(2305 -> 2012 tok/s/gpu at c=32). Never leave them on for a measured run.

---

## Superseded / invalid — do not use

| label | why |
|---|---|
| `smoke-128` | plumbing validation only (2 points, `--isl 128 --osl 128 --concurrencies 1,32`) |
| `baseline-1024-256` | 7-point version of the baseline, superseded by the 15-point `baseline-dense-96` |
| `short-128-128` | **invalid.** Ran with `max_requests=1024`, which bound for every concurrency >= 128, so waves decayed as `1024/c` — 2.3 waves at c=448 with an unwarmed first wave at 44% of the sample. Produced four points where tok/s/user *rose* with concurrency, which is physically impossible, and understated throughput by up to 25%. `integrity_warnings()` now catches this automatically. |

---

## Useful invocations

```bash
# print the aiperf commands without running anything
cd $R/inco && python -m bench.sweep --dry-run

# re-render CSV/plot/summary from artifacts already on disk
cd $R/inco && python -m bench.sweep --analyze-only --label baseline-v2

# add points to an existing curve (results are collected from disk by label,
# so a later run merges into the same curve)
$M run $R/inco/modal/modal_baseline.py --label baseline-v2 --concurrencies "104,112"

# before/after comparison: per-concurrency ratio table plus overlaid curves
cd $R/inco && python -m bench.compare baseline-v2 <optimized-label>

# bare-metal GPU host instead of Modal (needs the server up separately)
bash $R/inco/scripts/serve_baseline.sh     # terminal A
bash $R/inco/scripts/run_baseline.sh       # terminal B
```

**For any A/B comparison, run baseline and variant in the same container.**
Measured cross-instance variation is 2-5%, plus the H100 variant lottery on
top. `gpu_model` is recorded in every `manifest.json` — check it matches before
comparing two runs.
