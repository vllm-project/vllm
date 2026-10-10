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
Quality is *not* measurable from it. The checkpoint was later rebuilt from the
1024-sample calibration and evaluated -- see
[Quality evals](#quality-evals--lm-eval). Throughput was not re-measured,
because 64 experts is 64 experts either way.

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

### The checkpoint path aliases across calibration sizes

The two artifacts disagree about what identifies a run:

| artifact | name | encodes sample count? |
|---|---|---|
| observations | `observations_1024_cosine-seed_42.pt` | yes |
| checkpoint dir | `layerwise_reap-renorm_true-seed_42-0.50` | no |

`get_pruned_model_dir` builds the directory name from method, renorm, seed and
ratio only, and `layerwise_prune` skips pruning whenever that directory already
holds a `*.safetensors` (`src/reap/layerwise_prune.py:350`). So a 1024-sample
run launched after an earlier 16-sample run at the same ratio recalibrates for
87 minutes, writes the new observations, then logs

```
INFO:__main__:Pruned model already exists at .../layerwise_reap-renorm_true-seed_42-0.50. Skipping pruning.
```

and exits `rc=0` with the *old* checkpoint still in place. **A clean exit does
not mean the checkpoint matches the calibration you just paid for.** Verify via
`reap_args.yaml` in the checkpoint dir, which records `batches_per_category`,
`batch_size` and `output_file_name`.

`--overwrite_pruned_model` forces the write. With observations already cached
this costs ~3 min, not 87:

```bash
$M run $R/inco/modal/modal_reap.py \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 --compression-ratio 0.5 \
  --batches-per-category 128 --batch-size 8 --model-max-length 2048 \
  --extra-args "--overwrite_pruned_model True"
```

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

### Saliency heatmaps from the calibration dump

The layerwise observer writes every metric it accumulated -- `reap`,
`ean_sum`, `expert_frequency` and the rest -- next to the checkpoint, one
tensor of 128 experts per layer, 6.4 MB total. Pruning consumes only the
per-layer ranking, so the dump is the whole decision and can be plotted
without touching a GPU:

```bash
$M volume get inco-reap \
  pruned/Qwen3-30B-A3B-Instruct-2507/evol-codealpaca-v1/layerwise/observations_1024_cosine-seed_42.pt \
  ./inco/results/saliency/
$R/.venv-inco/bin/python $R/inco/scripts/plot_reap_saliency.py
# -> inco/results/saliency/reap-saliency-heatmap.png
```

Four panels: saliency by expert index, the retained/pruned mask at 50%, the
same rows sorted within each layer, and absolute saliency per layer. Note the
dump is keyed by sample count (`observations_1024_...` vs
`observations_16_...`), so unlike the checkpoint directory it does *not* alias
across calibration sizes.

**Absolute saliency is not comparable across layers.** The median expert grows
105x from layer 0 (0.103) to layer 47 (10.79), because `reap` is
`mean(||expert_output|| * router_weight)` and activation norms grow with depth.
Layers 1-3 each hold one expert worth 81x / 38x / 16x their layer median, so
normalizing by the layer *max* -- the obvious choice -- flattens those layers
to near-white and hides everything else. The plot divides by the layer median
on a log colour scale instead.

**What the heatmap shows:**

- **Pruning is not index-structured.** Every index quartile retains 49.6-50.2%
  of its experts, and no expert index is kept in all 48 layers or dropped in
  all 48 (retention per index ranges 14-39 of 48 layers). The mask panel looks
  like noise because it is: saliency is a per-layer property, not a property of
  an expert slot.
- **The kept half carries ~2/3 of the saliency mass** (mean 0.669, range 0.608
  at layer 18 to 0.835 at layer 1). Deleting half the experts removes about a
  third of the measured saliency, not half -- which is the premise REAP trades
  on.
- **Deep layers are more concentrated.** The p90/p10 spread within a layer
  widens from ~3.2 in layer 0 to ~5.2 in layer 47, and 15 of the 23
  experts above 4x their layer median live in layers 42-47. Uniform per-layer
  ratios therefore cut deeper into the useful mass in early layers than late
  ones; `--perserve_super_experts` exists for exactly the late-layer outliers.

This is a picture of the ranking, not of quality. It explains *which* experts
went and how much saliency mass the cut left behind; what that costs on task
accuracy is the lm-eval section below.

### Published checkpoints, for reference

Cerebras has not released a pruned `Qwen3-30B-A3B`, but
`cerebras/Qwen3-Coder-REAP-25B-A3B` is a REAP-pruned `Qwen3-Coder-30B-A3B` —
architecturally identical (same `Qwen3MoeForCausalLM`, 128 experts, same
dimensions). Benchmarking it against `Qwen3-Coder-30B-A3B` gives a measured
before/after in about an hour with no calibration run, at the cost of swapping
the base model from Instruct to Coder.

---

## Quality evals — lm-eval

`modal_reap.py::evaluate` runs lm-eval tasks in-process via vLLM on one H100
and writes results to the `inco-reap` volume under `evals/{model_basename}/`.
OpenBookQA is 500 questions x 4 choices = 2000 loglikelihood requests, ~4 s of
inference; wall clock is almost entirely weight load.

```bash
T=rte,openbookqa,winogrande,arc_challenge,humaneval

# baseline
$M run --detach $R/inco/modal/modal_reap.py::evaluate \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 --tasks $T \
  --temperature 0.7 --top-p 0.8 --top-k 20 --min-p 0.0

# pruned checkpoint (path is inside the container, on the artifacts volume)
$M run --detach $R/inco/modal/modal_reap.py::evaluate \
  --model /artifacts/pruned/Qwen3-30B-A3B-Instruct-2507/evol-codealpaca-v1/pruned_models/layerwise_reap-renorm_true-seed_42-0.50 \
  --tasks $T \
  --temperature 0.7 --top-p 0.8 --top-k 20 --min-p 0.0

$M volume get inco-reap evals ./inco/results/evals

# per-task deltas, with a 1.96-combined-stderr significance gate
cd $R/inco && python -m bench.eval_compare \
  Qwen3-30B-A3B-Instruct-2507 layerwise_reap-renorm_true-seed_42-0.50 \
  --eval-root results/evals/evals
```

All five tasks go in one call so the 61GB weight load is paid once (~5 min of
the ~18 min wall clock). Both models run concurrently on separate H100s.

With `--detach` the client prints `Timed out waiting for final app logs` and
exits before the function's return value arrives, so the `saved to ...` line
never reaches the local log even on a clean run. Check `$M app list` for
`App completed`, or just pull the volume — do not read a missing marker as a
failure.

Protocol: 0-shot, `apply_chat_template=False`, seed 42 — matching
`reap/src/reap/eval.py`, so the numbers are comparable to the REAP paper's.

The sampling flags are the Qwen3 non-thinking defaults from the model card.
They are **inert for multiple-choice tasks**: OpenBookQA, ARC, WinoGrande and
friends are scored by loglikelihood over the candidate answers and never
sample a token. They matter only once a generative task (gsm8k, humaneval) is
added to `--tasks`.

### Three things humaneval needs that the MC tasks do not

All three were found the hard way; each cost a full weight load.

1. **`do_sample=True` must be passed explicitly, or the sampling flags above
   are silently discarded.** lm-eval *merges* CLI `gen_kwargs` into each
   task's own `generation_kwargs` rather than replacing them, and
   `humaneval.yaml` ships `do_sample: false`. The vLLM backend's
   `modify_gen_kwargs()` then does `if do_sample is False or "temperature" not
   in kwargs: kwargs["temperature"] = 0.0`. Without it the run is greedy and
   the reported temperature is a fiction. `evaluate()` now sets it.
2. **Two independent code-execution gates, not one.** `confirm_run_unsafe_code
   =True` satisfies lm-eval's. HuggingFace `evaluate`'s `code_eval` metric has
   its own: `if os.getenv("HF_ALLOW_CODE_EVAL", 0) != "1": raise ValueError`.
   `humaneval/utils.py` calls that metric at *import* time, so the failure
   lands in `get_task_dict` and takes the multiple-choice tasks down with it.
3. **Task configs resolve *after* the model loads.** `simple_evaluate`
   constructs the LM first, so a bad task name or an ungated metric costs
   ~5 min of weight load before it reports. `evaluate()` now calls
   `get_task_dict(task_list)` up front as a preflight; it fails in seconds.

Report **`acc_norm`**, not `acc`. Raw `acc` compares unnormalized logprobs and
is length-biased toward short options; `acc_norm` divides by byte length and is
what the REAP paper reports.

### Results: five benchmarks at 50% expert pruning

`Qwen3-30B-A3B-Instruct-2507` vs the REAP 50% checkpoint (1024-sample
evol-codealpaca calibration, seed 42). Headline metric is `acc_norm` where the
task reports one, else `acc`, else `pass@1`. `1.96se` is 1.96 x the stderr of
the difference; a delta inside it is not resolvable at these sample sizes.

| task | n | metric | baseline | REAP 50% | delta | rel | 1.96se | verdict |
|---|---|---|---|---|---|---|---|---|
| humaneval | 164 | pass@1 | 0.7317 | **0.7378** | +0.0061 | +0.8% | 0.0958 | **noise** |
| rte | 277 | acc | 0.7726 | 0.6895 | -0.0830 | -10.7% | 0.0737 | significant |
| winogrande | 1267 | acc | 0.7388 | 0.6298 | -0.1089 | -14.7% | 0.0360 | significant |
| openbookqa | 500 | acc_norm | 0.4480 | 0.3140 | -0.1340 | -29.9% | 0.0597 | significant |
| arc_challenge | 1172 | acc_norm | 0.6263 | 0.3976 | -0.2287 | -36.5% | 0.0394 | significant |

Mean headline delta over the five: **-0.1097**. Over the four MC tasks alone:
-0.1387.

**The split is the result.** Every multiple-choice task loses significantly —
ARC-Challenge by 36% relative — while HumanEval does not move at all. That is
the calibration corpus showing through: the checkpoint was calibrated on
`evol-codealpaca-v1`, so REAP retained the experts most salient on code, and
code generation survives deleting half of them. Commonsense and science QA are
out-of-domain for that corpus and collapse.

This is the paper's own domain-calibration claim reproduced from the opposite
direction: they showed C4 calibration destroys code (Table A8, several methods
at exactly 0.000 Eval+) while *improving* MC. Here code-calibration preserves
code and destroys MC. Two experiments, one conclusion — what you calibrate on
is what you keep.

**Do not read the HumanEval row as "lossless".** At n=164 with `repeats: 1`,
the combined 1.96se is 9.6pp, so the test cannot resolve anything smaller than
a ~10pp change. The honest claim is *no detectable degradation*, not zero
degradation. Tightening it means adding MBPP+ (378 problems, which also yields
the paper's `Eval+` column) or raising `repeats`. Sampling at temperature 0.7
rather than the task's default greedy widens this further, and lm-eval emits no
stderr for `pass@1`, which is why `bench.eval_compare` refuses to judge that
row on stderr alone and reports the interval instead.

Comparison against the paper's Table A6, which used the *original*
`Qwen3-30B-A3B` rather than `Instruct-2507`:

| task | ours base | A6 base | ours REAP | A6 REAP |
|---|---|---|---|---|
| openbookqa | 0.448 | 0.454 | 0.314 | 0.309 ± 0.001 |
| arc_challenge | 0.626 | 0.563 | 0.398 | 0.354 ± 0.006 |
| rte | 0.773 | 0.816 | 0.690 | 0.561 ± 0.020 |
| winogrande | 0.739 | 0.702 | 0.630 | 0.584 ± 0.004 |

OpenBookQA lands on top of the published pair. The other three differ on both
ends, which is expected from a different base checkpoint — `Instruct-2507` is
stronger on ARC-c and WinoGrande and weaker on RTE — and our pruned model is
above theirs on all four, consistent with starting from a stronger model. Only
OpenBookQA should be quoted as a reproduction; the rest are a different
experiment that happens to agree in direction and rough magnitude.

OpenBookQA also reproduced **exactly** (0.448 / 0.314) against the earlier
single-task run, which is the expected behaviour for loglikelihood scoring and
confirms the harness is deterministic across containers.

### Results: OpenBookQA at 50% expert pruning

| | acc | acc_norm |
|---|---|---|
| `Qwen3-30B-A3B-Instruct-2507` baseline | 0.324 | **0.448** |
| REAP 50%, 1024-sample evol-codealpaca calibration | 0.212 | **0.314** |

**-0.134 acc_norm, -30% relative.** Both ends reproduce the REAP paper's
Table A6 (`Qwen3-30B-A3B`, "Detailed benchmark results for multiple-choice QA
tasks") to within noise:

| | ours | Table A6 | delta |
|---|---|---|---|
| baseline | 0.448 | 0.454 | 0.006 |
| REAP 50% | 0.314 | 0.309 ± 0.001 | 0.005 |

Both deltas are well inside our ±0.021 stderr, so the harness config and the
pruning run are validated together.

**Compare against A6, not A8.** Table A8 is captioned "C4 calibrated" and
reports 0.454 -> 0.360 for the same model and ratio. A6 carries no such
qualifier and REAP's own `pruning-layerwise-cli.sh` defaults to
`theblackcat102/evol-codealpaca-v1`, which is what we used. Reading the two
tables together isolates the effect of the calibration corpus, holding model
and ratio fixed:

| calibration | OBQA @ 50% REAP | MC Avg |
|---|---|---|
| C4 (A8) | 0.360 | 0.580 |
| default / evol-codealpaca (A6) | 0.309 | 0.503 |

So the corpus alone is worth ~0.05 on OpenBookQA and ~0.08 on MC average, in
the direction you would expect: C4 is general web text, evol-codealpaca is
narrow code-instruction data, and REAP retains the experts most salient on the
calibration distribution. A code-calibrated checkpoint pays for it on
out-of-domain commonsense QA. (A6's calibration set is inferred from the
caption contrast and the match with our run, not from an explicit statement —
confirm against the paper body before citing.)

The pruned model's raw `acc` of 0.212 sits *below* the 0.25 random floor for a
4-way choice. That is a length-bias artifact of unnormalized scoring rather
than evidence of a broken checkpoint — `acc_norm` is well above chance, and it
matches the published number — but it does indicate 50% is a hard cut for this
calibration set.

**Throughput is unaffected by any of this.** The 2.05x result above is set by
the architecture (64 experts) and does not move with calibration quality or
sample count. Quote quality and throughput from different runs.

---

## Speculators (draft training) — DFlash2 smoke test

`inco/speculators/` is a clone of
[vllm-project/speculators](https://github.com/vllm-project/speculators),
gitignored for the same reason as `reap/`. To recreate it:

```bash
cd $R/inco && git clone https://github.com/vllm-project/speculators.git
```

`modal_speculators.py` runs the *offline* pipeline end-to-end on one H100:
serve target -> `prepare-data` -> `generate-offline-data` -> stop server ->
`train`. Offline rather than online because the server and the trainer then
never hold the GPU at once; the online recipe in `examples/train/` wants four.

```bash
$M run $R/inco/modal/modal_speculators.py            # full pipeline
$M run $R/inco/modal/modal_speculators.py --train-only  # reuse volume data
$M volume get inco-spec dflash2-reap50-smoke ./inco/results/spec
```

**Two interpreters, deliberately.** System python holds this fork's vLLM
(which supplies `extract_hidden_states` and `ExampleHiddenStatesConnector`);
`/opt/spec` is a venv for speculators, which pins `transformers>=5.0` and
`torch>=2.9` and would otherwise replace the engine. They talk over localhost.
`/opt/spec/bin` must stay **off** `PATH`: Modal resolves the function's own
interpreter from it, and `launch_vllm.py` spawns the engine with
`sys.executable` -- put the venv first and the server dies with
`No module named 'vllm'`.

### Result: 64 samples, 1 epoch, seq 2048

| | |
|---|---|
| train loss | 10.330 -> 5.916 |
| `val/loss_epoch` | 5.911 |
| `val/eal_epoch` | 1.109 |
| `val/accept_rate_epoch` | 0.020 |

Expected acceptance length above 1.0 means the draft is producing accepted
tokens rather than noise -- the pipeline works against a REAP-pruned target.
It is *not* a usable drafter: 64 samples, and the data is off-policy.

**`--max-model-len` must exceed `--seq-length`.** The first run set both to
2048 and `prepare-data` dropped 16 of 192 conversations with `max_tokens must
be at least 1, got 0`. Nothing sends that 0 -- it is computed. `prepare-data`
renders with `truncate_prompt_tokens=seq_length`
(`src/speculators/data_generation/preprocessing.py:231`), and vLLM's render
endpoint budgets `max_tokens = max_model_len - input_length`
(`vllm/entrypoints/serve/utils/api_utils.py:188`). Equal values leave zero
headroom for every conversation that reaches the truncation limit, and
`sampling_params.py:608` rejects a budget of 0. Short conversations had
headroom, which is why only the long ~8% failed.

The render endpoint budgets output tokens it never generates, so the fix is
just headroom: `max_model_len` now defaults to `2 * seq_length`. The upstream
offline example never passes `--max-model-len` at all, which is why this does
not show up there.

The failures are logged per-conversation and the step still exits 0, so with a
bad setting the loss is silent at scale.

What was checked before running, and held up:

* The checkpoint is a complete HF model — 7 safetensors shards + index,
  `config.json`, `generation_config.json`, and the full tokenizer set
  including `chat_template.jinja`.
* Speculators has no architecture allowlist for targets.
  `VerifierConfig.from_pretrained` (`src/speculators/config.py:79`) reads
  `config.json`, and `--verifier-name-or-path` accepts a local path.
* vLLM supports EAGLE3 aux hidden states for this architecture:
  `Qwen3MoeForCausalLM` declares `SupportsEagle3`
  (`vllm/model_executor/models/qwen3_moe.py:539`) and `Qwen3MoeModel` collects
  `aux_hidden_states` in its forward.
* `launch_vllm.py:446` derives `target_layer_ids` from `num_hidden_layers`,
  still 48, giving the same `[2, 24, 45, 48]` as the unpruned model.
* Pruning changed only `num_experts` (128 -> 64). `hidden_size` (2048),
  `vocab_size` (151936), `num_hidden_layers` (48) and `num_experts_per_tok`
  (8) are unchanged, so draft dimensions and the borrowed embedding/lm_head
  tensors keep their shapes.

Two things to get right for a real run:

**Data must be on-policy against *this* checkpoint.** `docs/user_guide/
tutorials/train.md:107` requires responses produced by the target model. The
`DATASET` in every `examples/train/*.sh` is
`hf:inference-optimization/speculators-ci-datasets:tutorial_regen`, which was
regenerated by **Qwen3-8B** — fine as a pipeline smoke test, wrong as training
data for this target. Regenerate first:

```bash
speculators regenerate-responses --dataset ultrachat \
  --model <pruned-checkpoint-path>
```

Prompt-set presets live in `src/speculators/data_generation/configs.py:106`
(`ultrachat`, `sharegpt`, `magpie`, `nemotron`, `open-perfectblend`, `gsm8k`,
`hermes-fc`). The repo's examples all use UltraChat.

**Consider a code-weighted prompt set instead.** REAP was calibrated on
`evol-codealpaca-v1`, so the retained experts are code-biased — the A6-vs-A8
comparison above puts that effect at ~0.05 OBQA / ~0.08 MC average. `--dataset` takes a local JSONL or
an `hf:` spec, so the calibration corpus can be reused as the regeneration
prompt set to match the drafter to the target's retained strengths.

### Real run: 5000 on-policy evol-codealpaca samples

Both of the above are now implemented in `modal_speculators.py`, whose defaults
are the upstream dflash2 recipe (5000 samples, 5 epochs, from
`examples/train/dflash2_qwen3_8b_ultrachat_online_5k.sh`). The function is
`pipeline`, not `smoke`; `--skip-regen` restores the old off-policy path.

```bash
$M run --detach $R/inco/modal/modal_speculators.py --label dflash2-reap50-code-5k
```

The pipeline gained two steps ahead of `prepare-data`:

* **Step 0 — prompts.** 5000 `instruction` fields from
  `theblackcat102/evol-codealpaca-v1` written as `{"prompt": ...}` rows, which
  is the schema `load_input_dataset` assumes for an unregistered dataset
  (`prompt_field="prompt"`). Shuffled with seed 42: file order in these corpora
  tracks their source subsets, so a bare first-N samples one subset rather than
  the corpus. Runs *before* the server, under `/opt/spec` because `datasets`
  lives there — a wrong dataset id then costs seconds instead of a 5-minute
  weight load.
* **Step 1b — `regenerate-responses`** against the target on localhost, seed
  42, Qwen3 non-thinking sampling, `--max-tokens 1024`. Nonzero rc raises:
  without that, `prepare-data` silently falls through to whatever `dataset`
  points at and the draft trains on the wrong distribution.

Its output is already `input_ids` + `loss_mask`, so `prepare-data` takes the
pretokenized passthrough (`preprocessing.py:498`) and never renders.

~65 min end to end on one H100: ~19 min regeneration (4.5 samples/s, 3.8k
output tok/s), a few minutes for prepare-data plus extraction, 41.6 min to
train.

| stage | count |
|---|---|
| prompts written | 5000 |
| prepared examples | 4985 (`dataset_info.json`) |
| hidden-state shards | 4938 (`hs_0`..`hs_4937`) |

The ~15 lost rows are in `regen.errors.jsonl`; the 47 further lost between
prepare-data and extraction are `--validate-outputs` rejections.

**67% of generations hit the 1024-token cap.** evol-codealpaca answers are long
code-plus-explanation, so most samples end at the cap rather than at EOS, and
the draft sees few EOS tokens. Arguably aligned with the serving workload,
which runs `ignore_eos:true` and never lets the model finish either — but it is
a property of this checkpoint, not a neutral default.

### Result: val metrics per epoch

`checkpoints/{epoch}/val_metrics.json`, 4985 samples, block_size 8:

| epoch | val loss | accept_rate | accept_len | eal |
|---|---|---|---|---|
| 0 | 2.572 | 0.209 | 1.936 | 2.542 |
| 1 | 2.226 | 0.269 | 2.274 | 2.953 |
| 2 | 2.028 | 0.318 | 2.571 | 3.245 |
| 3 | 1.885 | 0.358 | 2.813 | 3.476 |
| 4 | 1.821 | **0.391** | **3.022** | **3.637** |

Against the 64-sample off-policy smoke run (`val/eal` 1.109, `val/accept_rate`
0.020), on-policy regeneration is the whole difference between a draft that
barely lands a token and one accepting 39% at depth.

**Quote the val numbers, not the train numbers.** At epoch 4 `train/eal` was
6.519 and `train/accept_rate` 0.701, against 3.637 / 0.391 on validation — five
epochs over 4985 samples leaves a large fit gap. The number that belongs in a
throughput claim is neither: it is measured acceptance in vLLM at 1024/512,
which is lower again because acceptance falls with batch size.

**It had not converged.** Every metric was still improving monotonically at
epoch 4 and the per-epoch `eal` gain was decaying but positive (+0.411, +0.292,
+0.231, +0.161). Stopping at 5 epochs is the upstream recipe, not a plateau, so
more epochs or more samples are the obvious next lever.

**The run dir holds ~2x the hidden-state bytes it needs.** 14908 files =
4938 `hs_*.safetensors` + 4985 `chatcmpl-*.safetensors` + 4985 `.lock`: the
connector's per-request intermediates are not cleaned up. At 16 KB/token
measured (`[904, 4, 2048]` bf16 for a 904-token sample) that is worth deleting
before the next run.

---

## Serving the drafter — acceptance and throughput

`modal_speculators.py::acceptance` serves a target with and without its drafter
in one container and sweeps concurrency against both. vLLM wants the *drafter*
checkpoint as the served model: a speculators config makes it swap in the
config's own `verifier.name_or_path` for model and tokenizer and derive
`method=dflash` with 7 speculative tokens
(`transformers_utils/config.py:740`). Passing the target plus
`--speculative-config` instead would need `trust_remote_code` for the draft's
`auto_map`.

```bash
$M run --detach $R/inco/modal/modal_speculators.py::acceptance --label specdec-reap50-v5

$M run --detach $R/inco/modal/modal_speculators.py::acceptance \
  --label specdec-dense-v5 --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --drafter /spec/dflash2-dense-code-5k/checkpoints/4 \
  --kv-cache-gib 10 --gpu-memory-utilization 0.92
```

Workload: 64 HumanEval+ and 64 MBPP+ prompts, shuffled seed 42. ISL 66 mean
(HumanEval+ 107, MBPP+ 24; bimodal, median 39, max 263), OSL ~375 mean capped
at 512, greedy, natural stopping. **This is not the 1024/512 shape** -- prefill
is ~15% of the tokens here, so these tok/s do not belong on the same axes as
the aiperf sweeps.

### Four settings this measurement does not work without

Each was found by getting a wrong answer first.

* **`--no-async-scheduling` in *both* phases.** `async_scheduling` defaults to
  None, which resolves to True unless something blocks it, and
  `config/vllm.py:1345` blocks it only for `method="dflash"`. Omitting the flag
  therefore gives the no-draft phase a scheduler the drafted phase is not
  allowed to have. Worth 1.51x to the baseline at c=1.
* **A graph ceiling sized for the drafted batch.** The verify pass carries
  `concurrency x (1 + num_speculative_tokens)` rows, and
  `max_cudagraph_capture_size` defaults to 512 = exactly 64 x 8. Every point
  above c=64 ran the draft eager while the baseline kept its graphs, which
  manufactured a throughput peak at c=48 and a crossover at c=80. Now sized
  from `max_num_seqs x (1 + spec_tokens)` and passed to both phases.
* **`--no-enable-prefix-caching`.** The prompt pool repeats 2-6x within a
  point at high concurrency, so with caching those repeats read their prefills
  out of the cache.
* **Whole-pool request counts.** `requests = pool x ceil(max(24, 8c) / pool)`,
  giving >=8 waves everywhere and an identical workload at every concurrency.
  A fixed 128 prompts gave c=64 two waves, and two runs of that config
  disagreed by 46%; a bare `8 x c` made the prompt mix a function of
  concurrency, since a prefix of a source-grouped pool is all HumanEval+.

`_integrity_warnings` re-checks both of `bench/report.py`'s rules (>=4 waves,
tok/s/user never rising with concurrency) and stores them in `results.json`.

### Results: REAP-50% and unpruned, drafted vs not

Both runs audited `async_scheduling=False`, `spec_method=dflash`,
`num_speculative_tokens=7`, no integrity warnings. KV pinned 32 GiB (pruned)
and 10 GiB (unpruned -- 58.3 GiB of weights leaves no more); neither model is
KV-bound at these lengths, c=96 needs 74k tokens of a 99k-305k pool.

| c | REAP draft | REAP plain | ratio | acc_len | unpr draft | unpr plain | ratio | acc_len |
|---|---|---|---|---|---|---|---|---|
| 1 | 400 | 168 | 2.38x | 3.894 | 382 | 165 | 2.32x | 4.008 |
| 2 | 664 | 282 | 2.36x | 3.926 | 597 | 274 | 2.17x | 4.006 |
| 4 | 1105 | 440 | 2.51x | 3.900 | 945 | 413 | 2.29x | 4.012 |
| 8 | 1940 | 726 | 2.67x | 3.956 | 1530 | 660 | 2.32x | 4.014 |
| 16 | 3207 | 1202 | 2.67x | 3.927 | 2457 | 1022 | 2.40x | 4.016 |
| 32 | 5920 | 2062 | 2.87x | 3.924 | 4261 | 1625 | 2.62x | 3.990 |
| 48 | 7861 | 2846 | 2.76x | 3.918 | 6001 | 2198 | 2.73x | 4.004 |
| 64 | 9457 | 3623 | 2.61x | 3.921 | 7218 | 2770 | 2.61x | 4.006 |
| 80 | 10659 | 4285 | 2.49x | 3.921 | 8353 | 3282 | 2.55x | 4.006 |
| 96 | 11580 | 5004 | 2.31x | 3.922 | 9314 | 3776 | 2.47x | 4.006 |

* **The drafter is worth 2.2-2.9x on both models**, peaking at c=32-48. Below
  that the draft's fixed per-step cost is unamortised; above it, rejected
  tokens start costing real compute. Ceiling is the accepted length, so the
  peak captures 73% of ideal and the ends 58-61%.
* **Every curve rises monotonically to c=96.** No saturation, no crossover.
  Both earlier claims to the contrary were the graph-ceiling artifact.
* **Pruning is worth ~1.25x on top, and the two stack**: pruned+drafted 11,580
  against unpruned undrafted 3,777 is 3.1x.
* **Acceptance is flat in load and near-identical between models** --
  3.89-3.96 pruned, 3.99-4.02 unpruned, across a 96x range of concurrency.
  Validation had the pruned drafter 3% *ahead* (3.637 vs 3.524); serving puts
  it 2% behind. The two disagree in sign, so the honest claim is that pruning
  does not measurably change draftability.
* Serving acceptance exceeds validation for both (3.92 vs 3.64, 4.01 vs 3.52):
  serving is greedy, while the training data was regenerated at temperature
  0.7, and a greedier target is easier to predict.

Plot: `python -m bench.plot_specdec results/spec/results-reap-v5.json
results/spec/results-unpruned-v5.json --labels "REAP-50%,unpruned"`.

**No repeats.** One measurement per point, so 0.1-0.2x differences between the
two models' speedup curves are inside the run-to-run variation seen elsewhere
here (11% at a matched config). The acceptance gap, 2% in the same direction
at all ten points, is the more trustworthy of the two.

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
