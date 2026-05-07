# Code Notes: Tests, Benchmarks, and Data Assets

## 1) Cohere Test Suite Is an Integration Layer, Not Just Unit Tests

`tests/cohere/` is large because it encodes product-level coverage across:

- guided generation and structured output paths,
- thinking token budget behavior,
- speculative decoding + multimodal scenarios,
- reward and cancellation behavior,
- benchmark/eval orchestration scripts.

Interpretation:

- this suite acts as a compatibility harness for Cohere-specific runtime contracts that upstream does not guarantee.

## 2) Config-Driven Hardware and Model Routing

Core configs:

- `runner_map.json`: maps `(gpu, test_group)` -> runner labels.
- `tp_model_map.json`: per `(gpu, model)` minimum/recommended TP constraints.
- `model_eval_map.json` and related benchmark config files.
- `hardware_profiles.yaml`: declares per-GPU engine args (memory utilization, chunked prefill, cudagraph capture sizes, etc.) and env vars. Applied by `apply_hardware_profiles.py` during `setup_tests.sh`, exported as `VLLM_HARDWARE_PROFILE_ARGS`.

Why this is central:

- CI matrix logic depends on these files for both validity checks and workload partitioning.
- incorrect updates here can invalidate CI behavior even if workflow YAML is unchanged.
- hardware profile changes affect every test that uses `test_utils_engine_args.py` helpers (`get_engine_kwargs_with_overrides`, `get_async_engine_args_with_overrides`), which parse `VLLM_HARDWARE_PROFILE_ARGS` and merge profile defaults with test-specific kwargs. Changing a profile value effectively changes the engine config for all tests unless a test explicitly overrides that key.

## 2b) Model-Conditional Eval Config Generation

`generate-serving-config.py` produces `serving-cohere-tests.json` with model-conditional behavior:

- **C5 models** (`"c5" in model_name`) receive additional server flags: `--tool-call-parser cohere_command4`, `--reasoning-parser cohere_command4`, and `--enable-auto-tool-choice`. These enable Melody tool-calling and reasoning parse support at the vLLM server level.
- **C5 models** also receive `thinking_token_budget` (4096 for non-reasoning, 20480 for reasoning) in `eval_parameters`. `run-bee-eval.sh` reads these values and conditionally builds `--extra_body` args for the bee eval command; non-C5 models omit `--extra_body` entirely.
- **All models with reasoning tests** (any test path containing `"reasoning"`) receive a `--reasoning-config` server flag with `think_start_str`/`think_end_str` markers.

Why this matters:

- Thinking token budgets and tool-call/reasoning parsers are now config-driven from `generate-serving-config.py`, not hardcoded in shell scripts. To change budgets or extend to new model families, only the Python config generator needs updating.

## 3) Benchmark Reporting Contract

Performance pipeline expects `benchmark_results_summary.json`:

- produced by benchmark conversion scripts,
- copied to output dir by `run_tests.sh`,
- appended to reporting branch via `upload-results` action.

Consequence:

- naming/path compatibility (`benchmark_results_summary.json`) is part of CI API contract between test scripts and workflow upload steps.

`model_arch` GPU jobs now also emit
`unit_results_summary.json`:

- produced by the `tests/cohere/test_logits_processor.py` portion of the
  `model_arch` bucket,
- copied to the runner temp output directory by the shared Docker mount,
- appended to reporting branches through `test-pipeline.yaml` using a separate
  upload path from serving/perf summaries.

Consequence:

- `unit_results_summary.json` is a separate CI artifact contract for
  pytest-backed microbenchmarks; keep it distinct from
  `benchmark_results_summary.json`, which remains the performance-suite format.
- the fp32 LM-head microbenchmark intentionally keeps the isolated
  `compute_logits()` projection eager; end-to-end `torch.compile` and CUDA graph
  behavior for fp32 logits is covered by
  `tests/cohere/test_c5_fp32_logits.py`, where `compute_logits()` runs
  after the compiled/cudagraph-managed model forward.

## 4) Dataset Extensions for Multimodal Workloads

`vllm/benchmarks/datasets.py` adds:

- `custom_mm` dataset type,
- `--enable-multimodal-chat` option,
- sampling path for multimodal custom datasets.

Combined with spec decode script changes, this enables:

- local or custom JSONL multimodal benchmarking beyond static canned prompts.

## 5) Fixture and Asset Additions

Large fixture set additions include:

- long prompts and needle data,
- multimodal image fixtures,
- architecture/test diagrams.

These are not incidental; they exercise long-context, multimodal, and schema-heavy paths that frequently regress in rebases.

## 6) Practical Maintenance Notes

`run_tests.sh` explicitly notes it must stay in sync with `test-pipeline.yaml`.

Recommended maintenance pattern:

1. Treat workflow + script updates as paired changes.
2. Keep config JSON/YAML updates in same PR when changing matrix logic.
3. Verify output artifact names whenever benchmark script changes.

When running Cohere tests from a synced detached worktree on a remote workspace:

- use the synced worktree path for checkpoint downloads and Docker mounts, not
  the base checkout,
- expect `setup_tests.sh` to sometimes log a `setuptools-scm was unable to
  detect version for /vllm-workspace` warning during `uv pip install -e .
  --no-deps`,
- treat that warning as non-fatal only if the script continues by exporting
  `PYTHONPATH` and enters `/vllm-workspace/tests`; if setup exits there, the
  editable-install contract between the mounted checkout and the image likely
  drifted.

## 7) Change Hotspots and Validation

High-conflict files:

- `tests/cohere/scripts/run_tests.sh`
- `tests/cohere/configs/*.json` and `*.yaml`
- `.buildkite/lm-eval-harness/configs/*.yaml`
- `.buildkite/performance-benchmarks/scripts/convert-results-json-to-markdown.py`
- `vllm/benchmarks/datasets.py`

Validation checklist:

1. Run one feature test group and one benchmark group locally/in CI.
2. Confirm model/TP validation fails fast for bad combos.
3. Confirm benchmark summary JSON is generated and uploadable.
4. Verify multimodal custom dataset path with `custom_mm` end-to-end.
