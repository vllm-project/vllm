# CI wiring and perf/accuracy artifacts

This is a proposal for the Buildkite side; step fields and agent pools must be
rechecked against `.buildkite/test_areas/*.yaml` and `.buildkite/test-amd.yaml`
at the time of the change.

## Steps per backend

| Step | Trigger | Hardware | Contents |
| --- | --- | --- | --- |
| `<Backend> unit` | PR + main, `source_file_dependencies` = backend file, its op modules (`vllm/_aiter_ops.py`, `vllm/v1/attention/ops/...`), `vllm/v1/attention/backend.py`, `backends/utils.py`, the selector for its platform, and its test dir | 1 GPU of the target arch | L0 + L2 covering set |
| `<Backend> runtime` | same | 1-2 GPUs | L3 dummy-weight runs |
| `<Backend> nightly matrix` | nightly on main, `optional: true` for PRs | 1 GPU | L2 full Cartesian product |
| `<Backend> perf` | PR when backend/op files change, every main commit (post-merge), nightly | 1 dedicated GPU, same pool every run | benchmark config, artifacts below |
| `<Backend> accuracy` | main post-merge + PR with `ready` label | 1-8 GPUs | GSM8K configs |

Also add the AITER / kernel-library version pin files (for ROCm,
`requirements/rocm*.txt` and the ROCm base Dockerfile) to the perf and unit
steps' dependencies: a library bump changes every dispatch target without
touching vLLM code.

## Branch coverage as a gate

Run the unit step with
`pytest --cov=vllm/v1/attention/backends/<backend>.py --cov-branch --cov-report=json:cov.json`
and upload `cov.json`. A small script fails the step when a branch inside
`forward`, `build`, `build_for_drafting`, `do_kv_cache_update` is uncovered and
not marked `# pragma: no cover` with a reason. This turns "every dispatch path
is tested" into a mechanical check that reviewers can read off the artifact.

## Artifact format

Every perf and accuracy run writes one JSON file per run:

```json
{
  "schema": 1,
  "backend": "ROCM_AITER_FA",
  "commit": "<sha>", "merge_base": "<sha or null>", "pr": 12345,
  "hardware": {"gpu": "MI355X", "arch": "gfx950", "count": 1},
  "versions": {"torch": "...", "rocm": "...", "aiter": "...", "triton": "..."},
  "env": {"VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT": "0", "...": "..."},
  "results": [
    {"key": "decode/h64kv16d64/bs128/ctx8192/bs16/fp8",
     "metric": "latency_us", "median": 41.2, "p10": 40.9, "p90": 41.8, "n": 50},
    {"key": "gsm8k/Qwen3-235B-A22B/tp8/kv-fp8",
     "metric": "accuracy", "value": 0.948, "stderr": 0.006},
    {"key": "correctness/decode/h128kv4d128/bf16",
     "metric": "max_abs_err", "value": 0.0031}
  ]
}
```

The result `key` is the join key between runs; never reuse a key for a
different shape. Perf numbers come from
`benchmarks/attention_benchmarks/benchmark.py --config <backend>.yaml
--output-json`; accuracy numbers from the GSM8K harness; correctness error
statistics from a pytest fixture that records max abs/rel error per case
(cheap, and shows numerical drift before a tolerance trips).

## Storage

Buildkite artifacts expire and cannot be queried across builds, so use them
only for the per-build copy (`buildkite-agent artifact upload "results/*.json"`)
and also write each file to long-lived storage keyed by commit:

```
s3://<ci-results-bucket>/attention/<backend>/<gpu>/<commit>.json
s3://<ci-results-bucket>/attention/<backend>/<gpu>/main-index.jsonl   # commit, timestamp, path
```

Only post-merge main runs append to `main-index.jsonl`. PR runs write under
`.../pr/<pr>/<commit>.json` and never update the index.

## Comparing a PR with main

1. The PR perf step computes `git merge-base HEAD origin/main`.
2. It fetches the baseline for that commit; if missing, the nearest ancestor
   in `main-index.jsonl` (and says which commit it used).
3. A compare script (extend
   `.buildkite/performance-benchmarks/scripts/compare-json-results.py` rather
   than adding a new one) joins on `key` and reports ratio per key.
4. Noise handling: each key runs `n >= 30` timed iterations after warmup with
   HIP/CUDA-graph replay; the regression threshold per key is
   `max(3%, 3 x MAD of the last 20 main runs for that key)`.
5. Output: `buildkite-agent annotate --context <backend>-perf` with the table of
   regressions and improvements, and the JSON as an artifact. Regressions
   soft-fail the step (`soft_fail: true`) until the noise model is trusted,
   then become blocking for the perf step only.
6. Accuracy keys compare against the main baseline with the GSM8K stderr; a
   drop beyond 2 stderr fails.

## Per-main-commit trend

A nightly job (or a static page built from `main-index.jsonl`) plots each key
over main history, so a slow drift that never crosses a per-PR threshold is
still visible. Keep the run hardware fixed per series; a pool change starts a
new series.
