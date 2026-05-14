<!-- markdownlint-disable MD024 -->
# C5 Arch

> **Registry**: [`observability_matrix.md`](../observability_matrix.md) entries 1.1.1, 4.1.1 |
> **Compatibility**: [`feature_matrix.md`](../feature_matrix.md) sections C5 Arch, Thinking Budget

<details>
<summary>Test case 1: C5 bee samples</summary>

## How it runs

1. `run_model_arch_c5_3a30t_checks()` calls
   `run_bee_samples` with `MODELS=c5-3a30t_fp8 TP_SIZE=1`.
   - [`tests/cohere/scripts/run_tests.sh` L599](../../../../tests/cohere/scripts/run_tests.sh)
2. Starts `vllm serve` with reasoning config (thinking tokens), hardware
   profile args, `--mm-processor-cache-type shm`, `--disable-log-stats`, and
   parsers (`--reasoning-parser cohere_command4 --enable-auto-tool-choice --tool-call-parser cohere_command4`)
   - [`tests/cohere/scripts/run_tests.sh` L276-330](../../../../tests/cohere/scripts/run_tests.sh)
3. Invokes `pytest tests/cohere/test_bee_samples.py` with configuration
   passed via env vars (`BEE_MODEL`, `BEE_DATA_DIR`, etc.). The pytest
   integration fires 16 samples per task concurrently against 8 tasks:
   mmlupro, mgsm, mbpp_plus, ocrbench, infovqa, mathvista, aime, niah.
   Each task is a separate parametrized test case, producing per-task
   JUnit XML reporting.
   - [`tests/cohere/test_bee_samples.py`](../../../../tests/cohere/test_bee_samples.py)
   - [`tests/cohere/bee_eval_checker.py`](../../../../tests/cohere/bee_eval_checker.py)

## Checks

1. Each task's **avg_score >= min_score** (per-task threshold defined in
   `TASK_CONFIG`; default 0.5, mmlupro 0.6). Fails if any task is below its
   threshold.
   - `test_bee_samples.py` (`test_bee_task` parametrized by task name via pytest)

## Measurements

1. Uploads **`bee_samples_c5-3a30t_fp8.json`** (per-task summary, no
   per-sample generations) to `bee_samples_data/model_arch_c5_3a30t` on
   `ci_dump` via `upload-results` action.
   - `avg_score`, `passed`, `total`, `wall_time_s` per task -- `PRESENT`
   - [`.github/workflows/test-pipeline.yaml`](../../../../.github/workflows/test-pipeline.yaml) -- "Upload bee samples summary" step

## Compatibility

Features from [Feature Matrix](../feature_matrix.md)
([Compatibility Sources](../feature_matrix.md#compatibility-sources)):

1. **Input**: Basic (compatible -- mmlupro, aime, mbpp_plus), Long Context (compatible -- niah), Multilingual (compatible -- mgsm), Image (compatible -- ocrbench, infovqa, mathvista)
2. **Cohere Feature**:
3. **Model Architecture**: C5 Arch (compatible)
4. **Quantization**: FP8 (compatible)
5. **Hardware**: H100, B200, GB200, MI300x (compatible); A100 (not compatible)
   - [`tests/cohere/configs/runner_map.json`](../../../../tests/cohere/configs/runner_map.json) -- `model_arch_c5_3a30t` runners for H100/B200/GB200/MI300x; none for A100
6. **vLLM Feature**: Chunked Prefill (compatible), CUDA Graphs (compatible)
   - [`vllm/cohere/hardware_profiles.yaml`](../../../../vllm/cohere/hardware_profiles.yaml) -- default profile enables both

## Implementation

Primary test: [`tests/cohere/test_bee_samples.py`](../../../../tests/cohere/test_bee_samples.py)
Eval checker: [`tests/cohere/bee_eval_checker.py`](../../../../tests/cohere/bee_eval_checker.py)
CI entry: [`tests/cohere/scripts/run_tests.sh` L276](../../../../tests/cohere/scripts/run_tests.sh)

### Setup

1. `vllm serve` with `--tensor-parallel-size 1`, `--reasoning-config`
   (thinking start/end tokens), `--disable-log-stats`,
   `--mm-processor-cache-type shm` and parsers
   (`--reasoning-parser cohere_command4 --enable-auto-tool-choice --tool-call-parser cohere_command4`)
2. Hardware profile args applied automatically inside the spawned `vllm serve`
   process via `apply_cohere_auto_config` (`run_tests.sh` exports
   `VLLM_ENABLE_COHERE_AUTO_CONFIG=1`). See
   [Hardware Profiles](../../code_notes/ci-and-automation.md#hardware-profiles).
3. Thinking budget disabled (`ENABLE_THINKING_BUDGET=0`); no
   `thinking_token_budget` sent to the server.
4. Invoked via `pytest` with env vars (`BEE_MODEL`, `BEE_DATA_DIR`,
   `BEE_OUTPUT_JSON`, `ENABLE_THINKING_BUDGET`). Each task is a separate
   parametrized test producing per-task JUnit XML reporting.
5. 16 samples per task across 8 evaluation tasks (text + image).

</details>

<details>
<summary>Test case 2: C5 bee samples with thinking budget</summary>

## How it runs

1. `run_bee_sample_tb_check()` calls `run_bee_samples` with
   `MODELS=c5-3a30t_fp8 TP_SIZE=1 ENABLE_THINKING_BUDGET=1`.
   - [`tests/cohere/scripts/run_tests.sh` L614-621](../../../../tests/cohere/scripts/run_tests.sh)
2. Starts `vllm serve` with reasoning config (thinking tokens), hardware
   profile args, `--mm-processor-cache-type shm`, and
   `--disable-log-stats` (same server as test case 1).
   - [`tests/cohere/scripts/run_tests.sh` L276-330](../../../../tests/cohere/scripts/run_tests.sh)
3. `pytest tests/cohere/test_bee_samples.py` is invoked with
   `ENABLE_THINKING_BUDGET=1` env var, which sends per-task
   `thinking_token_budget` values from `TASK_CONFIG` (default 2048, aime
   16384, mbpp_plus 4096). Each task is a separate parametrized test case.
   - [`tests/cohere/test_bee_samples.py`](../../../../tests/cohere/test_bee_samples.py)

## Checks

1. Each task's **avg_score >= min_score** (per-task threshold defined in
   `TASK_CONFIG`; default 0.5, mmlupro 0.6). Fails if any task is below its
   threshold.
   - `test_bee_samples.py` (`test_bee_task` parametrized by task name via pytest)

## Measurements

1. Uploads **`bee_samples_c5-3a30t_fp8.json`** (per-task summary, no
   per-sample generations) to `bee_samples_data/bee_sample_tb_check` on
   `ci_dump` via `upload-results` action.
   - `avg_score`, `passed`, `total`, `wall_time_s` per task -- `PRESENT`
   - [`.github/workflows/test-pipeline.yaml`](../../../../.github/workflows/test-pipeline.yaml) -- "Upload bee samples summary" step

## Compatibility

Features from [Feature Matrix](../feature_matrix.md)
([Compatibility Sources](../feature_matrix.md#compatibility-sources)):

1. **Input**: Basic (compatible -- mmlupro, aime, mbpp_plus), Long Context (compatible -- niah), Multilingual (compatible -- mgsm), Image (compatible -- ocrbench, infovqa, mathvista)
2. **Cohere Feature**: Thinking Budget (compatible)
3. **Model Architecture**: C5 Arch (compatible)
4. **Quantization**: FP8 (compatible)
5. **Hardware**: H100, B200, GB200, MI300x (compatible); A100 (not compatible)
   - [`tests/cohere/configs/runner_map.json`](../../../../tests/cohere/configs/runner_map.json) -- `bee_sample_tb_check` runners for H100/B200/GB200/MI300x; none for A100
6. **vLLM Feature**: Chunked Prefill (compatible), CUDA Graphs (compatible)
   - [`vllm/cohere/hardware_profiles.yaml`](../../../../vllm/cohere/hardware_profiles.yaml) -- default profile enables both

## Implementation

Primary test: [`tests/cohere/test_bee_samples.py`](../../../../tests/cohere/test_bee_samples.py)
Eval checker: [`tests/cohere/bee_eval_checker.py`](../../../../tests/cohere/bee_eval_checker.py)
CI entry: `run_bee_sample_tb_check()` in
[`tests/cohere/scripts/run_tests.sh` L614](../../../../tests/cohere/scripts/run_tests.sh)
Dispatcher: `bee_sample_tb_check` test group, expanded from `thinking_budget`.
Runner map: [`tests/cohere/configs/runner_map.json`](../../../../tests/cohere/configs/runner_map.json)

### Setup

1. `vllm serve` with `--tensor-parallel-size 1`, `--reasoning-config`
   (thinking start/end tokens), `--disable-log-stats`,
   `--mm-processor-cache-type shm` and parsers
   (`--reasoning-parser cohere_command4 --enable-auto-tool-choice --tool-call-parser cohere_command4`)
2. Hardware profile args applied automatically inside the spawned `vllm serve`
   process via `apply_cohere_auto_config` (`run_tests.sh` exports
   `VLLM_ENABLE_COHERE_AUTO_CONFIG=1`). See
   [Hardware Profiles](../../code_notes/ci-and-automation.md#hardware-profiles).
3. `ENABLE_THINKING_BUDGET=1` env var passed to pytest, activating per-task
   `thinking_token_budget` from `TASK_CONFIG` (default 2048, aime 16384,
   mbpp_plus 4096).
4. Invoked via `pytest` with env vars (`BEE_MODEL`, `BEE_DATA_DIR`,
   `BEE_OUTPUT_JSON`, `ENABLE_THINKING_BUDGET`). Each task is a separate
   parametrized test producing per-task JUnit XML reporting.
5. 16 samples per task across 8 evaluation tasks (text + image).

</details>
