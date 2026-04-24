# Upstream Diff Deep Dive: Tests, Benchmarks, and Data Assets

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

## 3) Benchmark Reporting Contract

Performance pipeline expects `benchmark_results_summary.json`:

- produced by benchmark conversion scripts,
- copied to output dir by `run_tests.sh`,
- appended to reporting branch via `upload-results` action.

Consequence:

- naming/path compatibility (`benchmark_results_summary.json`) is part of CI API contract between test scripts and workflow upload steps.

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

## 7) Rebase Hotspots and Validation

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
