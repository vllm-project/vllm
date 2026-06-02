<!-- markdownlint-disable MD024 -->
# Online Quantization Config Parsing

> **Registry**: [`observability_matrix.md`](../observability_matrix.md) entries 2.2.1 - 2.2.8 |
> **Compatibility**: [`feature_matrix.md`](../feature_matrix.md) section Online Quantization Config Parsing

Validates the cohere-specific `OnlineQuantizationConfig.from_config` parsing
path that loads online quantization config from a checkpoint's
`config.json` `quantization_config` block. Distinct from the upstream
CLI / `OnlineQuantizationConfigArgs` path covered by
[`tests/quantization/test_online.py`](../../../../tests/quantization/test_online.py).

<details>
<summary>Test case 1: Checkpoint config.json parsing</summary>

## How it runs

1. Calls `OnlineQuantizationConfig.from_config(...)` directly on a
   plain `dict` mimicking the `quantization_config` block of a checkpoint
   `config.json`. No model load, no GPU.
   - [`tests/cohere/cpu/test_online_quant_from_config.py`](../../../../tests/cohere/cpu/test_online_quant_from_config.py)
   - [`vllm/model_executor/layers/quantization/online/base.py`](../../../../vllm/model_executor/layers/quantization/online/base.py)
2. Exercises the four scheme shorthands (`fp8_per_tensor`, `fp8_per_block`,
   `mxfp8`, `int8_per_channel_weight_only`) via `quant_method`, plus the
   explicit `quant_method: "online"` form with `linear_scheme_override` /
   `moe_scheme_override`.
   - [`tests/cohere/cpu/test_online_quant_from_config.py`](../../../../tests/cohere/cpu/test_online_quant_from_config.py)
3. Discovered by `pytest -v -s cohere/cpu` inside `run_cpu_tests` and run
   on `ubuntu-latest` via the `cpu_check` group on every PR to `cohere`.
   - [`tests/cohere/scripts/run_tests.sh` L91](../../../../tests/cohere/scripts/run_tests.sh)
   - [`.github/workflows/pr-cpu-tests.yaml`](../../../../.github/workflows/pr-cpu-tests.yaml)

## Checks

1. **Scheme shorthand** in `quant_method` populates `args.global_scheme` to
   the matching `OnlineQuantScheme` and leaves `linear_scheme_override` /
   `moe_scheme_override` / `ignored_layers` empty.
   - `test_shorthand_quant_method_populates_global_scheme`
2. `quant_method: "online"` with explicit overrides keeps `global_scheme`
   `None` and routes the **linear / MoE schemes** through the override
   fields.
   - `test_online_quant_method_with_explicit_overrides`
3. **Explicit `global_scheme` wins** over a `quant_method` shorthand
   instead of being clobbered.
   - `test_explicit_global_scheme_not_overwritten_by_shorthand`
4. `ignore`, `ignored_layers`, and `modules_to_not_convert` aliases are
   **merged in declared order** into `ignored_layers` (including `re:` regex
   patterns).
   - `test_ignore_aliases_are_merged`
   - `test_ignore_only_from_alias_when_primary_missing`
5. `activation_scheme: "dynamic"` is **accepted**; any other value
   (e.g. `"static"`) raises `ValueError("activation_scheme...")`.
   - `test_activation_scheme_dynamic_is_accepted`
   - `test_activation_scheme_static_raises`
6. Configs with no `global_scheme` / `linear_scheme_override` /
   `moe_scheme_override` raise `ValueError("global_scheme...")`.
   - `test_no_scheme_raises`

## Measurements

N/A -- pure parsing test, no CI-uploaded artifacts.

## Compatibility

Features from [Feature Matrix](../feature_matrix.md)
([Compatibility Sources](../feature_matrix.md#compatibility-sources)):

1. **Input**:
2. **Cohere Feature**:
3. **Model Architecture**:
4. **Quantization**: FP8, MXFP8 (compatible -- shorthand parsing covers
   `fp8_per_tensor`, `fp8_per_block`, `mxfp8`, `int8_per_channel_weight_only`)
   - [`tests/cohere/cpu/test_online_quant_from_config.py`](../../../../tests/cohere/cpu/test_online_quant_from_config.py)
5. **Hardware**: A100, H100, B200, GB200, MI300x (not compatible -- CPU-only,
   runs on `ubuntu-latest`)
   - [`.github/workflows/pr-cpu-tests.yaml`](../../../../.github/workflows/pr-cpu-tests.yaml)
6. **vLLM Feature**: Torch Compile, CUDA Graphs (not compatible -- no engine
   load)

## Implementation

Primary test: [`tests/cohere/cpu/test_online_quant_from_config.py`](../../../../tests/cohere/cpu/test_online_quant_from_config.py)
Runtime path: [`vllm/model_executor/layers/quantization/online/base.py`](../../../../vllm/model_executor/layers/quantization/online/base.py) (`OnlineQuantizationConfig.from_config`)
CI entry: [`tests/cohere/scripts/run_tests.sh` L91](../../../../tests/cohere/scripts/run_tests.sh) (`run_cpu_tests`)

### Setup

1. CPU-only; gated by `pytest.importorskip("torch")` and
   `pytest.importorskip("vllm")` so the file collects cleanly in any env.
2. No vLLM engine, no model checkpoint, no GPU. The test only constructs
   `OnlineQuantizationConfig` from in-memory `dict` payloads.
3. Lives under `tests/cohere/cpu/` so it is picked up by
   `pytest -v -s cohere/cpu` in `run_cpu_tests` and routed through the
   `cpu_check` group on `ubuntu-latest`.

</details>
