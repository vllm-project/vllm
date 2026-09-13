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

## REAP expert pruning

`inco/reap/` is a clone of [CerebrasResearch/reap](https://github.com/CerebrasResearch/reap),
gitignored because it is a separate repository with its own submodules and
virtualenv. To recreate it:

```bash
cd $R/inco && git clone https://github.com/CerebrasResearch/reap.git
```

**Environment is separate and must stay that way.** `reap/scripts/build.sh`
runs `VLLM_USE_PRECOMPILED=1 uv pip install -e .`, which installs *its own*
vLLM — sharing an environment with this fork would overwrite the engine under
test. It also needs `cp .env.template .env` with `USER_ID` / `GROUP_ID`
(`id -u` / `id -g`) and optionally `HF_TOKEN`.

It cannot be built on macOS: the install needs Linux + CUDA. Run it on the GPU
host, or in a Modal image of its own.

The six git submodules under `third-party/` are all evaluation harnesses
(evalplus, LiveCodeBench, helm, evalscope, creative-writing-bench,
llm-compressor). Only needed if you turn the evals on.

### Model support

`MODEL_ATTRS` in `src/reap/model_util.py` is keyed by architecture class and
contains `Qwen3MoeForCausalLM`, so `Qwen3-30B-A3B-Instruct-2507` is supported.
`Qwen/Qwen3-30B-A3B` is the default model in their own scripts.

### Pruning command

Both entry points take the same 13 positional arguments:

```
1  CUDA_VISIBLE_DEVICES      7  run_lm_eval
2  model_name                8  run_evalplus
3  pruning_method            9  run_livecodebench
4  seed                     10  run_math
5  compression_ratio        11  run_wildbench
6  calibration dataset      12  singleton_super_experts
                            13  singleton_outlier_experts
```

`pruning-layerwise-cli.sh` uses a block-wise calibration observer intended for
pruning large models on a single GPU (`num_batches=128`, `batch_size=8`, so
1024 calibration samples). Evals are switched off here so it only produces the
checkpoint — the defaults run lm_eval + evalplus + LiveCodeBench, which take
hours.

```bash
cd $R/inco/reap
experiments/pruning-layerwise-cli.sh 0 Qwen/Qwen3-30B-A3B-Instruct-2507 reap 42 0.5 \
  theblackcat102/evol-codealpaca-v1 false false false false false
```

Output: `artifacts/{model_hash}/{dataset_hash}/pruned_models/{method_config}/`,
a HuggingFace-loadable checkpoint that vLLM can serve directly.

After pruning, **re-pin the KV cache upward** — the freed weight memory does
not become KV automatically:

| prune | weights | freed | KV | max batch @ 1024/512 |
|---|---|---|---|---|
| 0% | 58.3 GiB | - | 12 GiB | 85 |
| 38% | 37.6 GiB | 20.7 | 32.7 GiB | 232 |
| 50% | 31.3 GiB | 27.0 | 39.0 GiB | 277 |

```bash
$M run $R/inco/modal/modal_baseline.py \
  --label reap50-1024-512 --model <pruned-checkpoint> \
  --isl 1024 --osl 512 --kv-cache-gib 39 --max-num-seqs 256 \
  --concurrencies "1,2,4,8,16,32,64,96,128,160,192,224,256"
```

### Results: 50% expert pruning at 1024/512

`reap50-1024-512` vs the `chat-1024-512` baseline, same GPU model
(H100 80GB HBM3), zero integrity warnings, zero preemption in either.

```bash
$M run $R/inco/modal/modal_baseline.py \
  --label reap50-1024-512 --served-model-name reap50 \
  --model /reap/pruned/Qwen3-30B-A3B-Instruct-2507/evol-codealpaca-v1/pruned_models/layerwise_reap-renorm_true-seed_42-0.50 \
  --isl 1024 --osl 512 --kv-cache-gib 38 --max-num-seqs 256 \
  --concurrencies "1,2,4,8,16,24,32,40,48,56,64,72,80,96,128,160,192,224,256"
```

|  | baseline | REAP 50% |
|---|---|---|
| weights | 58.3 GiB | 31.3 GiB |
| KV cache (pinned) | 12 GiB | 38 GiB |
| max resident requests | 85 | **270** |
| peak tokens/s/gpu | 3779 @ c=80 | **7738 @ c=256** |

**2.05x peak throughput.** SLO-matched:

| interactivity floor | baseline | REAP | gain |
|---|---|---|---|
| >= 50 tok/s/user | 3517 | 4756 | +35% |
| >= 40 tok/s/user | 3779 | 6326 | +67% |
| >= 35 tok/s/user | 3779 | 6861 | +82% |

The gain separates into two effects:

* **Faster steps: ~+12%, only at high concurrency.** At c=1 and c=2 it is
  -1% (noise): a token routes to top-8 experts whether the model has 128 or
  64, so the weight read per step is identical. The gain appears as batch
  size grows and more of the expert set gets touched, plateauing near +12.5%
  by c=64. This follows from the step-time decomposition -- pruning halves
  `W` but leaves `c` untouched.
* **More batch: 3.2x ceiling.** The dominant effect. 27 GiB freed from
  weights became KV cache, so every point from c=96 to c=256 is throughput
  the unpruned model cannot produce on this hardware at any latency.

So the 2x is mostly capacity, not speed.

**This checkpoint came from a 16-sample calibration.** Throughput is
determined by the architecture (64 experts), so the numbers above are valid.
Quality is *not* measurable from it -- any accuracy claim needs the
1024-sample checkpoint plus an eval.

### Calibration cost

Measured at `--batches-per-category 128 --batch-size 8` (1024 samples) on one
H100: **~109 s per decoder block, 48 blocks, so ~87 min of calibration**, plus
a `device_map="auto"` model reload and a ~60 s checkpoint write. Scales
linearly in `batches_per_category`, so the bare module default of 1024
batches/category would be roughly 12 hours.

Observations are cached and reused: `layerwise_prune` loads
`observations_{samples}_{distance}-seed_{seed}.pt` when it exists unless
`--overwrite_observations` is set. The filename does **not** include the
compression ratio, so a second prune at a different ratio with the same sample
count, seed and distance measure skips calibration entirely (~2 min):

```bash
$M run $R/inco/modal/modal_reap.py --compression-ratio 0.38 \
  --batches-per-category 128 --batch-size 8      # cache hit, prune only
```

Note the cache key is sample *count*, not batching, so `32 x 32` and `128 x 8`
both map to `observations_1024_...` -- same data either way, but do not rely on
the filename to distinguish batching experiments.

### Calibration memory

`batch_size` is **not** a free speed knob. `pruning_metrics.py` materialises
activations shaped `(num_experts, total_tokens, hidden_dim)` -- for *all* 128
experts, not just the routed top-8:

```
activation bytes ~= num_experts x (batch_size x model_max_length) x hidden_dim x 4
  128 x ( 8 x 2048) x 2048 x 4 = 17.2 GB   <- default, fits
  128 x (32 x 2048) x 2048 x 4 = 68.7 GB   <- OOM, observed
```

So the default `batch_size 8` is already near the limit on an 80GB card. The
product `batch_size x model_max_length` is what matters: `bs=16 @ seq=1024`
costs the same as `bs=8 @ seq=2048`.

### Published checkpoints, for reference

Cerebras has not released a pruned `Qwen3-30B-A3B`, but
`cerebras/Qwen3-Coder-REAP-25B-A3B` is a REAP-pruned `Qwen3-Coder-30B-A3B` —
architecturally identical (same `Qwen3MoeForCausalLM`, 128 experts, same
dimensions). Benchmarking it against `Qwen3-Coder-30B-A3B` gives a measured
before/after in about an hour with no calibration run, at the cost of swapping
the base model from Instruct to Coder.

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
