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

- 3.1.1 `test_gg_vision_spec_async` with `--thinking-budgets 500 1000 5000` (c5-3a30t_fp8, speculative)

### 4. GG + TB + Melody

#### 4.1 [Thinking Budget — C5 bee samples](./features/c5_arch.md)

- 4.1.1 `test_bee_task` with `ENABLE_THINKING_BUDGET=1` (c5-3a30t_fp8)

#### 4.2 Thinking Budget — SD/non-SD (BLS)

- 4.2.1 `test_thinking_budget` non-SD (c5-3a30t_fp8)
- 4.2.2 `test_thinking_budget` SD (c5-3a30t_fp8 + eagle)

#### 4.5 [Thinking Budget TPOT Overhead](./features/thinking_budget_overhead.md)

- 4.5.1 `test_thinking_budget_overhead` (c5-3a30t_fp8)

#### 4.3 Guided Generation — Text (BLS)

- 4.3.1 `test_guided_generation --suite merged` non-SD (c5-3a30t_fp8, JSON + tools + long-context with thinking)
- 4.3.2 `test_guided_generation_melody` SD (c5-3a30t_fp8 + eagle)
- 4.3.3 `test_guided_generation_tools_melody` SD (c5-3a30t_fp8 + eagle)

#### 4.4 [Template / Tokenizer / Parser Check](./features/template_tokenizer_parser.md)

- *Diagnostic / investigation tool — no asserted entries (does not gate CI on rendered prompt, tokens, or parser output). Logs the chat-template + tokenization + parser pipeline end-to-end for `c5-3a30t_fp8` across `no_parsers` and `with_parsers` passes, with and without thinking budget.*

### 5. Speculative Decoding

#### 5.1 [Request Cancellation](./features/speculative_decoding_test.md)

- 5.1.1 `test_request_cancellation` SD sweep (c5-3a30t_fp8 + eagle, `--num-requests 32 64`)
- 5.1.2 `test_request_cancellation` non-SD sweep (c5-3a30t_fp8, `--num-requests 32 64`)

### 6. vLLM

#### 6.1 [Cohere Auto-Config](./features/auto_config.md)

- 6.1.1 `test_post_init_applies_for_cohere`
- 6.1.2 `test_post_init_disabled_by_default`
- 6.1.3 `test_post_init_no_op_for_non_cohere`

### 7. ASR

#### 7.1 [Cohere ASR](./features/asr.md)

- 7.1.1 `test_cohere_transcribe_wer_correctness`
- 7.1.2 `test_asr_long_audio_with_output_streaming`
- 7.1.3 `test_create_transcription_non_streaming_joins_chunks_by_language`
- 7.1.4 `test_non_streaming_cancel_aborts_engine_requests`

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

1. The canonical YAML lives at
   [`/vllm/cohere/hardware_profiles.yaml`](../../../vllm/cohere/hardware_profiles.yaml)
   (bundled into the wheel via `setup.py` `package_data`). Profile selection
   uses CEL `when:` clauses evaluated against `{server.type, gpu.name}` --
   `gpu.name` is bound lowercased so YAML patterns like `b200` and `mi300x`
   match real device names. The `vllm-default` profile is always applied
   as a baseline; later profiles override earlier ones on conflicting keys.
2. [`/vllm/cohere/auto_config.py`](../../../vllm/cohere/auto_config.py)
   implements `apply_cohere_auto_config(engine_args)`, which is invoked from
   [`/vllm/engine/arg_utils.py`](../../../vllm/engine/arg_utils.py) at the
   end of `EngineArgs.__post_init__` whenever
   `VLLM_ENABLE_COHERE_AUTO_CONFIG` is truthy (`1` / `true`). User-supplied
   field values are detected by comparing each live field to its dataclass
   default and never overwritten; profile env vars only land when the var
   is unset.
3. CI shells (`run_tests.sh`, `run-bee-eval.sh`,
   `run-performance-benchmarks.sh`) `export VLLM_ENABLE_COHERE_AUTO_CONFIG=1`
   so every spawned `vllm serve` / `vllm bench` / pytest process picks the
   profiles up automatically. Python entry points that build `LLM(...)` /
   `AsyncEngineArgs(...)` directly call
   `os.environ.setdefault("VLLM_ENABLE_COHERE_AUTO_CONFIG", "1")` in their
   `main()` / `__main__` block (or the pytest entry function). See
   [Cohere Auto-Config](../code_notes/runtime-and-scheduling.md#8-cohere-auto-config-hardware-profile-application)
   for the runtime-side contract and
   [Hardware Profiles](../code_notes/ci-and-automation.md#hardware-profiles)
   for the CI-side contract.

Rules for new or modified tests:

- **Respect hardware profiles by default.** Tests that use the vLLM engine for
  serving or benchmarking should opt in to auto-config (env var) rather than
  hard-coding GPU-specific settings; profile values fill in dataclass-default
  fields automatically.
- Prefer enabling `torch.compile` and CUDA graphs when compatible with the
  test. Call out any intentional deviation in the test doc's `## Setup` section.
- If a test needs to override a profile value (e.g. a specific attention
  backend), pass it explicitly to `LLM(...)` / `AsyncEngineArgs(...)` --
  user-set fields are preserved by `apply_cohere_auto_config`. Document the
  override and reason in the test doc.
- When adding a new GPU profile, update
  [`/vllm/cohere/hardware_profiles.yaml`](../../../vllm/cohere/hardware_profiles.yaml)
  and add coverage in
  [`tests/cohere/cpu/test_auto_config.py`](../../../tests/cohere/cpu/test_auto_config.py)
  for any non-trivial CEL `when:` clause or new keys.

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
