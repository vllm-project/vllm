# Inco take-home — Track A (Engine & Model): baseline benchmark

Reproducible Pareto-curve benchmark for **Qwen3-30B-A3B-Instruct (MoE) on one
80GB H100 under vLLM**, measured with **NVIDIA AIPerf**. This is step 2 of the
assignment workflow — the baseline you have to trust before changing anything.

## Workload (stated precisely)

| | |
|---|---|
| Model | `Qwen/Qwen3-30B-A3B-Instruct-2507` — 30.5B total / 3.3B active MoE, bf16, TP=1 |
| GPU | one H100 80GB, `--gpu-memory-utilization 0.90` |
| Task shape | single-turn chat, `/v1/chat/completions`, streaming |
| Input length | 1024 tokens (synthetic, stddev 0, fixed seed 100) |
| Output length | 256 tokens, **exactly** — `ignore_eos:true` |
| Thinking mode | off (`--default-chat-template-kwargs '{"enable_thinking": false}'`) |
| Max context | 4096 tokens |
| Load | client concurrency swept over `1,2,4,8,16,32,48` |
| Axes | x = tokens/s/user (interactivity), y = tokens/s/gpu (throughput) |

### Why the sweep stops at 48

The MoE is sparse in FLOPs but not in memory — all 128 experts per layer are
resident even though only 8 run per token:

```
weights       30.5B params x 2 bytes (bf16)              = 61 GB
budget        80 GB x 0.90                               = 72 GB
activations + CUDA graphs                                ~  4 GB
                                                          -------
KV cache                                                 ~  7 GB

KV per token  48 layers x 4 KV heads x 128 head_dim
              x 2 (K+V) x 2 bytes                        = 96 KiB
capacity      7 GB / 96 KiB                              ~ 76k tokens
              76k / (1024 + 256)                         ~ 60 resident requests
```

So ~60 requests is the ceiling on this GPU, and the sweep stops at 48 to stay
inside it. Driving 128 or 256 would still produce numbers, but they would
measure how long requests sit in the scheduler queue, not how fast the engine
runs. **The harness does not trust this arithmetic** — it reads the engine's own
post-profiling `kv_cache_size_tokens` from `/server_info` and prints the real
capacity, warning about any sweep point that exceeds it.

On an H200 (141GB) or with TP=2 there is ~60GB of KV cache, enough for a
256-wide sweep: raise `INCO_MAX_NUM_SEQS` and `INCO_CONCURRENCIES` to 256.

### What makes the numbers comparable

1. **The server launches once, at max batch size** (`--max-num-seqs 48`). Only
   *client* concurrency is swept, so one engine configuration — one set of CUDA
   graphs, one KV cache — backs the entire curve.
2. **Output length is pinned** with `ignore_eos:true`. Without it OSL is a
   model-dependent variable and tokens/s stops being a fixed denominator.
   Qwen3's thinking mode is disabled for the same reason.
3. **The prefix cache is flushed between points** via `/reset_prefix_cache`, so
   a warm cache from a previous point cannot inflate a later one.
4. **The server is audited before it is measured.** The client reads the
   effective `VllmConfig` from `/server_info` and *refuses to run* if CUDA
   graphs or async scheduling are off, or if `max_num_seqs` is below the top of
   the sweep (`--allow-degraded-server` to override). A curve from a
   silently-eager server is not a baseline.

## Experiment log

Every run we made, with verbatim commands and what each showed, is in
[`EXPERIMENTS.md`](EXPERIMENTS.md) — including which runs are superseded or
invalid and why. Commands there pass all workload parameters explicitly rather
than relying on the defaults in `scripts/workload.env`, so each is reproducible
on its own.

## Quick start

### On a GPU host (Lambda / RunPod / EC2 / Vast)

```bash
# 1. install this fork + aiperf  (~5 min with precompiled kernels)
bash inco/scripts/install.sh
source .venv/bin/activate

# 2. terminal A: serve
bash inco/scripts/serve_baseline.sh

# 3. terminal B: sweep  (~20 min for the 7-point curve)
bash inco/scripts/run_baseline.sh
```

Outputs land in `inco/results/baseline/`:

| file | contents |
|---|---|
| `pareto.csv` | one row per concurrency: both axes, TTFT/ITL/e2e, error rate |
| `pareto.png` | the Pareto curve, annotated with concurrency |
| `summary.md` | markdown table + throughput at each interactivity SLO |
| `manifest.json` | workload, sweep config, every aiperf command, effective `VllmConfig` (including the profiled KV-cache size) |
| `concurrency*/` | raw aiperf artifacts (per-request records included) |

### From a laptop, on Modal

```bash
uv pip install modal && modal setup          # one-time, browser OAuth

# 1. cache the ~61GB of weights on a CPU container (minutes, cents)
modal run inco/modal/modal_baseline.py::prefetch

# 2. smoke test: builds the image, runs two points (~5 min of H100)
modal run inco/modal/modal_baseline.py --concurrencies 1,32

# 3. the real thing (~20 min of H100)
modal run inco/modal/modal_baseline.py

# 4. pull the artifacts back
modal volume get inco-results baseline ./inco/results --force
```

Server and client share one container, so aiperf measures the engine rather
than the network. `INCO_MODAL_GPU=H200` switches GPU.

The image installs **this fork** via `VLLM_USE_PRECOMPILED=1`, not a published
wheel — HEAD here is months ahead of any vLLM release, and the harness depends
on config fields and server flags that only exist in this tree. That also means
the baseline is measured on the same engine you will later modify.

## Useful invocations

```bash
# see the exact aiperf commands without running anything
python -m bench.sweep --dry-run

# re-render CSV/plot/summary from artifacts already on disk
python -m bench.sweep --analyze-only

# a different point in workload space (prefill-heavy)
python -m bench.sweep --label prefill-heavy --isl 8192 --osl 32

# faster, noisier sweep
python -m bench.sweep --concurrency 1 16 48 --requests-per-concurrency 4

# on an H200 / TP=2, where the KV cache can back a 256-wide sweep
INCO_MAX_NUM_SEQS=256 bash inco/scripts/serve_baseline.sh
python -m bench.sweep --concurrency 1 2 4 8 16 32 64 128 256

# before/after: overlay two labelled runs and print the speedup at each SLO
python -m bench.compare baseline my-optimization
```

Every knob is also an env var (`INCO_MODEL`, `INCO_OSL`, `INCO_CONCURRENCIES`,
…); `scripts/workload.env` is the single place where the server flags and the
aiperf flags are kept in sync.

## How to read the output

A Pareto curve is not one number, so two scalars are reported:

- **Throughput at an interactivity SLO** — "at ≥30 tok/s/user, how many
  tok/s does one GPU deliver?" This is the comparison that matters; it prevents
  claiming a win that was really just trading latency for batch size.
- **Interactivity at a fixed load** — tok/s/user at a given concurrency.

`bench/compare.py` computes both for any pair of runs, plus the per-concurrency
ratio table, and overlays the curves. An optimization is real when it moves the
frontier up and/or right — not when it moves one point along the existing curve.

## Sanity checks baked in

- `/server_info` audit gates the run (see above), including the engine's real
  KV-cache capacity versus the concurrency the sweep will drive.
- `output_sequence_length` is reported per point; it must equal 256. If it
  drifts, `ignore_eos` was not honoured and the run is invalid.
- `error_request_count` / `error_rate` per point; any nonzero value is printed
  to stderr. High-concurrency points that quietly 500 look like great
  throughput otherwise.
- `input_sequence_length` confirms the tokenizer agreed with the server.
- A point whose aiperf run exits nonzero, or exits clean but writes no export,
  is recorded as a failure rather than silently dropped; the sweep exits 1.

## Layout

```
inco/
├── bench/
│   ├── config.py     workload + sweep config, the single source of truth
│   ├── server.py     readiness, /server_info capture, perf-feature audit
│   ├── aiperf.py     aiperf command construction (flag names resolved
│   │                 against the installed binary, which renames them)
│   ├── collect.py    profile_export_aiperf.json -> tidy sweep points
│   ├── report.py     CSV / markdown / Pareto plot / throughput-at-SLO
│   ├── sweep.py      the driver:  python -m bench.sweep
│   └── compare.py    before/after overlay:  python -m bench.compare a b
├── scripts/          workload.env, serve_baseline.sh, run_baseline.sh, install.sh
├── modal/            modal_baseline.py — run the whole thing on a Modal GPU
├── tests/            pytest suite, no GPU required
└── results/          generated artifacts (gitignored)
```

## Tests

The harness is tested without a GPU — the measurement loop runs against a fake
aiperf and a fake server, so parsing, the audit gate, failure handling and the
Pareto math are all covered.

```bash
cd inco && python -m pytest tests -q --cov=bench --cov-report=term-missing
# 195 passed, 100% statement coverage on bench/
```

## Cost

One H100 for ~45 min covers the 7-point curve plus the ~61GB model download:
roughly $3 on Modal's per-second pricing, well inside the $100 budget. Start
with `--concurrencies 1,32` (~5 min) to confirm the plumbing before spending on
the full sweep. The first run pays the download; it is cached on a volume after
that.

## Next steps (assignment steps 3–5)

Baseline in hand, the analysis phase: `nsys` traces at a low-concurrency point
(decode-bound, hunting bubbles/syncs/dispatch overhead) and at concurrency 48
(where this model is KV- and memory-bandwidth-bound), then pick an optimization
and re-measure with `python -m bench.compare baseline <label>`.

The memory arithmetic above points at the obvious first targets for this
workload: 61GB of weights for 3.3B active params per token means the
concurrency ceiling is set by weight residency, not compute — so FP8 weights,
FP8 KV cache, and the MoE kernel path are where the headroom is.

## Tools used

Written with Claude Code (Opus 5). aiperf flag names, metric tags and JSON
export schema taken from the NVIDIA AIPerf docs; vLLM flags verified against
this fork's source rather than assumed.
