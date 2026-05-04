# Observability Matrix

This document is the entry point for feature-level test planning in
`docs/cohere/tests/`.

## Related Documents

This matrix is part of a three-layer documentation structure:

| Layer | Document | Purpose |
| --- | --- | --- |
| **Registry** | This file (`observability_matrix.md`) | Central index of every test entry and benchmark metric, organized by category. Each entry gets a unique `<cat>.<feat>.<seq>` ID. |
| **Compatibility** | [`feature_matrix.md`](./feature_matrix.md) | Cross-feature compatibility tables. Cells reference this matrix via `T.<cat>.<feat>.<seq>` to record which test case verified compatibility. |
| **Detail** | [`features/*.md`](./features/) (e.g. [`c5_arch.md`](./features/c5_arch.md), [`fp32_logits.md`](./features/fp32_logits.md)) | Per-feature docs with full test case details: How it runs, Checks, Measurements, Compatibility, and Implementation. |

**How they connect:**

- A test entry here (e.g. `1.1.1`) links to its feature doc (`features/c5_arch.md`) via the `####` sub-heading.
- The feature doc's `## Compatibility` section classifies input types, hardware, etc.
- Those classifications are propagated to `feature_matrix.md` as `T.1.1.1` cell values.
- The feature doc's `## Measurements` section lists only CI-uploaded artifacts, traceable back to the benchmark entries here.

## Tests

### 1. Model Architecture

#### 1.1 [C5 Arch](./features/c5_arch.md)

- 1.1.1 `test_bee_samples` (c5-3a30t_fp8)

#### 1.2 [C5 LoRA Serving](./features/c5_lora.md)

- 1.2.1 `test_c5_lora_sanity_check`

### 2. Quantization

#### 2.1 [FP32 Logits](./features/fp32_logits.md)

- 2.1.1 `test_c5_fp32_logits_consistency`
- 2.1.2 `test_lm_head_fp32_projection_diff_is_small_but_nonzero`
- 2.1.3 `test_lm_head_fp32_projection_benchmark_writes_summary`

### 3. Multimodal

#### 3.1 [MM + GG + TB](./features/mm_gg_tb.md)

- 3.1.1 `test_gg_vision_spec_async` with `--thinking-budgets 500 1000 5000` (c4-25a218t_fp8_eagle_l5, speculative)

### 4. GG + TB + Melody

#### 4.1 [Thinking Budget — C5 bee samples](./features/c5_arch.md)

- 4.1.1 `test_bee_task` with `ENABLE_THINKING_BUDGET=1` (c5-3a30t_fp8)

### 5. Speculative Decoding

### 6. vLLM

## Benchmarks

### 1. Model Architecture

#### 1.1 [C5 Arch](./features/c5_arch.md)

- 1.1.1 `avg_score` (per task) -> `PRESENT`

### 2. Quantization

#### 2.1 [FP32 Logits](./features/fp32_logits.md)

- 2.1.1 `bf16_median_ms` -> `LOWER-ANCHOR3+5%`
- 2.1.2 `fp32_median_ms` -> `LOWER-ANCHOR3+5%`

### 3. Multimodal

### 4. GG + TB + Melody

#### 4.1 [Thinking Budget — C5 bee samples](./features/c5_arch.md)

- 4.1.1 `avg_score` (per task, with thinking budget) -> `PRESENT`

### 5. Speculative Decoding

### 6. vLLM

## Evaluations

## Entry Format

The `## Tests`, `## Benchmarks`, and `## Evaluations` sections above are the
central registry. The hierarchy is:

- `###` category headings (`### 1. Model Architecture`, `### 2. Quantization`, etc.)
- `####` feature sub-headings within each category (`#### 2.1 FP32 Logits`)
- Bullet-list entries under each feature sub-heading

### Numbering scheme

Each entry ID has the form `<category>.<feature>.<seq>`:

- `<category>` is the number of the `###` heading (e.g. `1` for Model
  Architecture, `2` for Quantization).
- `<feature>` is the sequential number of the `####` feature sub-heading
  within that category (e.g. `1` for the first feature under Quantization).
- `<seq>` is a sequential counter within the feature, continuing from the last
  entry. If FP32 Logits already has `2.1.1`, `2.1.2`, `2.1.3`, the next
  entry is `2.1.4`.
- The same ID is referenced as `T.<category>.<feature>.<seq>` in
  [`feature_matrix.md`](./feature_matrix.md) (e.g. entry `2.1.1` maps to
  `T.2.1.1`).

### Entry templates

**Feature sub-heading** -- one `####` per feature, linked to the feature doc:

```markdown
#### <cat>.<feat> [Feature Name](./features/<feature>.md)
```

**Tests** -- one bullet per pytest function under the feature sub-heading:

```markdown
- <cat>.<feat>.<seq> `test_function_name`
```

**Benchmarks** -- one bullet per metric under the feature sub-heading:

```markdown
- <cat>.<feat>.<seq> `metric_name` -> `PATTERN-CODE`
```

Pattern codes are defined in [Metric Pattern Codes](#metric-pattern-codes)
below.

### Picking a category

How profiles are applied:

1. [`/tests/cohere/scripts/setup_tests.sh`](../../../tests/cohere/scripts/setup_tests.sh)
   sources
   [`/tests/cohere/scripts/apply_hardware_profiles.py`](../../../tests/cohere/scripts/apply_hardware_profiles.py)
   at container startup. The script matches profiles by `GPU_TYPE` and exports
   env vars directly and CLI args via `VLLM_HARDWARE_PROFILE_ARGS`.
2. Benchmark and serving scripts (`run-performance-benchmarks.sh`,
   `run-bee-eval.sh`) append `${VLLM_HARDWARE_PROFILE_ARGS:-}` to `vllm serve`
   / `vllm bench` commands.
3. Profiles are applied in order; later profiles override earlier ones. The
   `vllm-default` profile is always applied as a baseline.

Rules for new or modified tests:

- **Respect hardware profiles by default.** Tests that use the vLLM engine for
  serving or benchmarking should rely on the env vars and args exported by the
  profile, not hard-code GPU-specific settings.
- Prefer enabling `torch.compile` and CUDA graphs when compatible with the
  test. Call out any intentional deviation in the test doc's `## Setup` section.
- If a test needs to override a profile value (e.g. a specific attention
  backend), document the override and the reason in the test doc.
- When adding a new GPU profile, update `hardware_profiles.yaml` and verify
  that `apply_hardware_profiles.py` emits the expected exports.

### Nightly Metric Emission

[`/.github/workflows/nightly-benchmark.yaml`](../../../.github/workflows/nightly-benchmark.yaml)
is the top-level nightly CI entry. It triggers image builds, then fans out
eval/perf/feature jobs per GPU via
[`dispatcher.yaml`](../../../.github/workflows/dispatcher.yaml) and
[`test-pipeline.yaml`](../../../.github/workflows/test-pipeline.yaml).

Metric upload path:

1. Nightly **scheduled** runs use `result_upload_branch: gh-pages` (fallback when no input is provided).
2. Nightly **manual dispatch** defaults to `ci_dump` (overridable via the `result_upload_branch` input).
3. Ad-hoc `build-and-eval` / `build-and-bench` runs default to `ci_dump`.
4. [`test-pipeline.yaml`](../../../.github/workflows/test-pipeline.yaml) uses
   the [`.github/actions/upload-results`](../../../.github/actions/upload-results)
   action to append JSON records to per-path, per-GPU aggregate lists on the
   target branch.

Current upload triggers in `test-pipeline.yaml`:

| `test_group` | Artifact | Upload path on target branch |
| --- | --- | --- |
| `performance` | `benchmark_results_summary.json` | `data/summary` |
| `bee_eval` | `eval_results_summary.json` | `eval_data/summary` |
| `model_arch_logits` | `unit_results_summary.json` | `unit_data/summary` |

Rules for new or modified tests:

- **Tests that emit benchmark metrics must write a summary JSON** to
  `$OUTPUT_DIR` so the upload step can find it.
- When adding a new uploadable `TEST_GROUP`, add a corresponding upload step
  in `test-pipeline.yaml` and document the artifact name and upload path here.
- Feature docs should reference the nightly reporting path (`gh-pages`) in
  their `## Measurements` section for any metric tracked over time.
- Prefer the shared [metric pattern codes](#metric-pattern-codes) when
  defining expected values for benchmark metrics.
- See [`docs/cohere/code_notes/ci-and-automation.md`](../code_notes/ci-and-automation.md)
  for the full pipeline topology and reporting semantics.

## Metric Pattern Codes

Use these codes in feature docs to keep benchmark expectations short and
consistent.

`Cohort`
: A cohort is one stable benchmark definition and execution shape. A new cohort
starts whenever the metric meaning or run shape changes, such as benchmark
name, GPU, input shape, compile mode, or CUDA-graph mode.

| Code | Meaning | Programmatic definition |
| --- | --- | --- |
| `LOWER-ANCHOR3+5%` | Fixed-anchor, lower-is-better | For one cohort, set the anchor to the median of the first 3 successful values. Current value must be `<= 1.05 x anchor`. Do not auto-update the anchor after it is established. |
| `HIGHER-ANCHOR3-5%` | Fixed-anchor, higher-is-better | For one cohort, set the anchor to the median of the first 3 successful values. Current value must be `>= 0.95 x anchor`. Do not auto-update the anchor after it is established. |
| `PRESENT` | Metric must exist | The metric field must be present in the emitted artifact for the cohort. |
| `NONZERO` | Metric must be non-zero | The metric field must be present and must not be zero. |
