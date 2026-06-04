<!-- markdownlint-disable MD024 -->
# C5 LoRA Serving Tests

> **Registry**: [`observability_matrix.md`](../observability_matrix.md) entry 1.2.1 |
> **Compatibility**: [`feature_matrix.md`](../feature_matrix.md) section LoRA

Validates that the c5 3a30t (`cohere2moe`) model can load a LoRA adapter and
produce valid outputs both with and without the adapter active.

<details>
<summary>Test case 1: C5 LoRA serving sanity check</summary>

## How it runs

1. `run_model_arch_c5_lora_checks()` resolves `MODEL_DIR` to
   `${ENGINES_DIR}/c5-3a30t-petfatt-bf16` and, when `C5_LORA_DIR` is unset,
   generates a zero-weight dummy adapter at `${OUTPUT_DIR}/c5_dummy_lora` so
   the LoRA loading path is always exercised even without a trained adapter.
   - [`tests/cohere/scripts/run_tests.sh`](../../../../tests/cohere/scripts/run_tests.sh) -- `run_model_arch_c5_lora_checks` function
   - [`tests/cohere/scripts/create_dummy_lora.py`](../../../../tests/cohere/scripts/create_dummy_lora.py) -- builds zero-init `lora_A` / `lora_B` for `q_proj`, `k_proj`, `v_proj`, `o_proj` from `config.json`
2. The bash entrypoint exports `C5_MODEL_DIR` and `C5_LORA_DIR` and invokes
   `pytest -v -s tests/cohere/test_c5_lora.py` so the run produces JUnit XML
   reporting under the `model_arch_c5_lora` test group.
   - [`tests/cohere/scripts/run_tests.sh`](../../../../tests/cohere/scripts/run_tests.sh) -- pytest invocation in `run_model_arch_c5_lora_checks`
   - [`tests/cohere/test_c5_lora.py`](../../../../tests/cohere/test_c5_lora.py) -- `test_c5_lora_sanity_check` reads env and delegates
3. The pytest entry calls `run_c5_lora_sanity_check_test`, which builds one
   `LLM` with `enable_lora=True, max_loras=1, max_lora_rank=64` via
   `build_c5_llm`, then runs `_generate` twice over `C5_SANITY_PROMPTS` --
   first without a `LoRARequest`, then with the loaded adapter.
   - [`tests/cohere/test_c5_lora.py`](../../../../tests/cohere/test_c5_lora.py) -- `_build_lora_llm`, `_generate`, `run_c5_lora_sanity_check_test`
   - [`tests/cohere/test_utils_c5.py`](../../../../tests/cohere/test_utils_c5.py) -- `build_c5_llm` and `C5_SANITY_PROMPTS`
4. CI shape: dispatched as the dedicated `model_arch_c5_lora` test group via
   the `model_arch` feature expansion; runs on H100, B200, and GB200 runners
   per [`runner_map.json`](../../../../tests/cohere/configs/runner_map.json).
   See [Test Pipeline Integration](../../code_notes/ci-and-automation.md#7-test-pipeline-integration).
   - [`.github/scripts/dispatcher-set-matrix.js`](../../../../.github/scripts/dispatcher-set-matrix.js) -- `model_arch` feature expands to include `model_arch_c5_lora`
   - [`.github/workflows/test-pipeline.yaml`](../../../../.github/workflows/test-pipeline.yaml) -- `dorny/test-reporter` condition includes `model_arch_c5_lora`

## Checks

1. **Without LoRA**: every prompt from `C5_SANITY_PROMPTS` produces a
   generation that contains at least one expected keyword from
   `C5_SANITY_EXPECTED` (paris / nba / a+b / 巴黎). Asserts via
   `_check_outputs(..., "no-lora")`.
   - `test_c5_lora_sanity_check`
2. **With LoRA**: same keyword check after attaching `LoRARequest(lora_name="c5-adapter", lora_int_id=1, lora_path=lora_path)`
   to `llm.generate(...)`, confirming the adapter is loaded without breaking
   generation. Asserts via `_check_outputs(..., "lora")`.
   - `test_c5_lora_sanity_check`

## Measurements

1. Pytest run produces a **JUnit XML report** in `${OUTPUT_DIR}` via the
   `tests/conftest.py` `pytest_configure` hook; `dorny/test-reporter@v2`
   surfaces it as a "model_arch_c5_lora Test Report" check on the GitHub run.
   - `test_c5_lora_sanity_check` -- `PRESENT` (pass/fail status)
   - [`.github/workflows/test-pipeline.yaml`](../../../../.github/workflows/test-pipeline.yaml) -- "Test Report" step (`model_arch_c5_lora` branch)
   - See [JUnit XML Reporting (pytest)](../../code_notes/ci-and-automation.md#junit-xml-reporting-pytest)

No `upload-results` artifact is emitted -- the test group has no
benchmark/summary JSON upload step in `test-pipeline.yaml`.

## Compatibility

Features from [Feature Matrix](../feature_matrix.md)
([Compatibility Sources](../feature_matrix.md#compatibility-sources)):

1. **Input**: Basic (compatible), Multilingual (compatible)
   - [`tests/cohere/test_utils_c5.py`](../../../../tests/cohere/test_utils_c5.py) -- `C5_SANITY_PROMPTS` mixes English (capitals, NBA, Python `add`) with Chinese (`中国的首都是哪里?`)
2. **Cohere Feature**:
3. **Model Architecture**: C5 Arch (compatible)
   - [`vllm/model_executor/models/cohere2_moe.py`](../../../../vllm/model_executor/models/cohere2_moe.py) -- `Cohere2MoeForCausalLM` declares `SupportsLoRA` and `packed_modules_mapping` for `qkv_proj` / `gate_up_proj`
4. **Quantization**: BF16 (compatible)
   - [`tests/cohere/scripts/download_checkpoints.sh`](../../../../tests/cohere/scripts/download_checkpoints.sh) -- `download_model_arch_c5_lora_assets` pulls `c5_3a30t_petfatt-141_hf_export_bf16` into `c5-3a30t-petfatt-bf16`
5. **Hardware**: H100, B200, GB200 (compatible); A100, MI300x (not compatible)
   - [`tests/cohere/configs/runner_map.json`](../../../../tests/cohere/configs/runner_map.json) -- `model_arch_c5_lora` runners for H100/B200/GB200; none for A100/MI300x
6. **vLLM Feature**: Chunked Prefill (compatible), CUDA Graphs (compatible)
   - [`vllm/cohere/hardware_profiles.yaml`](../../../../vllm/cohere/hardware_profiles.yaml) -- `vllm-default` profile sets `enable-chunked-prefill` and `max-cudagraph-capture-size: 128`

## Implementation

Primary test:
[`tests/cohere/test_c5_lora.py`](../../../../tests/cohere/test_c5_lora.py)
Shared helpers:
[`tests/cohere/test_utils_c5.py`](../../../../tests/cohere/test_utils_c5.py)
Dummy LoRA generator:
[`tests/cohere/scripts/create_dummy_lora.py`](../../../../tests/cohere/scripts/create_dummy_lora.py)
CI entry:
[`tests/cohere/scripts/run_tests.sh`](../../../../tests/cohere/scripts/run_tests.sh) -- `run_model_arch_c5_lora_checks`
Runtime paths:
[`vllm/model_executor/models/cohere2_moe.py`](../../../../vllm/model_executor/models/cohere2_moe.py),
[`vllm/lora/worker_manager.py`](../../../../vllm/lora/worker_manager.py)

### Setup

1. **Env vars**: `C5_MODEL_DIR` (path to c5 3a30t bf16 base checkpoint --
   `c5-3a30t-petfatt-bf16`), `C5_LORA_DIR` (optional path to a real LoRA
   adapter). When `C5_LORA_DIR` is unset, `run_tests.sh` calls
   `create_dummy_lora.py` to write a zero-weight bf16 adapter targeting
   `q_proj`, `k_proj`, `v_proj`, `o_proj` for every layer (rank 8) so the
   LoRA loading path is always exercised.
2. **Engine flags**: `enable_lora=True`, `max_loras=1`, `max_lora_rank=64`,
   `max_model_len=32768`, `tensor_parallel_size=1` are passed explicitly to
   `LLM(...)` in `_build_lora_llm`. Profile-derived defaults (memory
   utilization, attention backend, cudagraph capture sizes, etc.) are filled
   in by `apply_cohere_auto_config` from `EngineArgs.__post_init__` because
   `run_tests.sh` exports `VLLM_ENABLE_COHERE_AUTO_CONFIG=1`; the pytest
   entry also calls `os.environ.setdefault(...)` so standalone runs
   self-configure. See
   [Hardware Profiles](../../code_notes/ci-and-automation.md#hardware-profiles).
3. **Sampling**: `SamplingParams(temperature=0.0, max_tokens=32)` over the
   four fixed prompts in `C5_SANITY_PROMPTS` (English capitals Q&A, NBA,
   Python `add`, Chinese capitals Q&A) with matching `C5_SANITY_EXPECTED`
   keyword sets.
4. **Generation pattern**: same `LLM` instance is used for both runs --
   first `llm.generate(C5_SANITY_PROMPTS, sampling_params)` without
   `lora_request`, then again with
   `LoRARequest(lora_name="c5-adapter", lora_int_id=1, lora_path=...)`.
   `shutdown_llm` is called from `finally` to release the engine.

</details>
