<!-- markdownlint-disable MD024 -->
# Cohere Auto-Config Tests

> **Registry**: [`observability_matrix.md`](../observability_matrix.md) entries 6.1.1, 6.1.2, 6.1.3 |
> **Compatibility**: [`feature_matrix.md`](../feature_matrix.md) section Auto-Config

CPU unit suite for [`vllm/cohere/auto_config.py`](../../../../vllm/cohere/auto_config.py) — the
opt-in hook that applies
[`vllm/cohere/hardware_profiles.yaml`](../../../../vllm/cohere/hardware_profiles.yaml)
to `EngineArgs` from inside `EngineArgs.__post_init__` when
`VLLM_ENABLE_COHERE_AUTO_CONFIG=1`. Background:
[Cohere Auto-Config](../../code_notes/runtime-and-scheduling.md#8-cohere-auto-config-hardware-profile-application).

<details>
<summary>Test case 1: Cohere auto-config CPU unit suite</summary>

## How it runs

1. `tests/cohere/cpu/test_auto_config.py` is collected by the `cpu` test group
   inside the prebuilt `vllm-cpu` image and runs without GPUs, model
   downloads, or real `current_platform` queries (`_gpu_name` is monkeypatched
   in every test that needs a device name).
   - [`tests/cohere/cpu/test_auto_config.py`](../../../../tests/cohere/cpu/test_auto_config.py)
   - [`tests/cohere/scripts/run_tests.sh`](../../../../tests/cohere/scripts/run_tests.sh) -- `cpu` group dispatch
2. The suite uses an inline YAML fixture (`SAMPLE_YAML`) written to
   `tmp_path/hardware_profiles.yaml` and a `_clear_caches` autouse fixture that
   resets `detect_cohere_from_model_id`, `_gpu_name`, and `_load_profiles_doc`
   between tests so cached YAML / GPU-name reads don't leak across cases.
   - [`tests/cohere/cpu/test_auto_config.py`](../../../../tests/cohere/cpu/test_auto_config.py)
3. Tests that exercise `apply_cohere_auto_config(engine_args)` end-to-end set
   `VLLM_ENABLE_COHERE_AUTO_CONFIG=1` via `monkeypatch.setenv` and point
   `_DEFAULT_PROFILES_PATH` at the fixture path via `monkeypatch.setattr` so
   `EngineArgs.__post_init__` triggers the real opt-in branch and applies the
   fixture YAML rather than the package YAML.
   - [`vllm/engine/arg_utils.py`](../../../../vllm/engine/arg_utils.py) -- `EngineArgs.__post_init__` opt-in gate
   - [`vllm/cohere/auto_config.py`](../../../../vllm/cohere/auto_config.py) -- `apply_cohere_auto_config`
4. CI shape: dispatched as the `cpu` test group on `ubuntu-latest` (no GPU);
   triggered automatically on every PR via `pr-cpu-tests.yaml` and from
   `fast_check` / `all` feature buckets.
   - [`.github/workflows/pr-cpu-tests.yaml`](../../../../.github/workflows/pr-cpu-tests.yaml)
   - [`.github/workflows/test-pipeline.yaml`](../../../../.github/workflows/test-pipeline.yaml)

## Checks

1. **Architecture drift guard**: every entry in the module-level
   `_COHERE_ARCHITECTURES` frozenset still resolves to a registered model
   architecture in `ModelRegistry.get_supported_archs()` -- catches upstream
   renames or removals of Cohere model classes.
   - `test_cohere_archs_subset_of_registry`
2. **CEL `when:` evaluation**: matching against lowercased `gpu.name` returns
   the expected boolean for `b200`, `mi300x`, `gb200`, `h100|h200` regex
   alternation, etc. The namespace shape (`{server: {type: "vllm"}, gpu: {name: ...}}`)
   is exercised explicitly. Parse errors and runtime errors both log a single
   `WARNING` and skip the offending profile (no exception leaks out).
   - `test_evaluate_when` (parametrized)
   - `test_evaluate_when_namespace_shape`
   - `test_evaluate_when_failure` (parametrized)
3. **YAML-string -> dataclass-field coercion**: `_coerce` converts profile
   string values to the declared `EngineArgs` field type for the four shapes
   that appear in the YAML -- `bool` / `bool | None` (empty string -> `True`),
   `int` / `int | None`, `float` / `float | None`, and `Enum` / `Enum | str`
   (lookup by name, falls back to raw string when not an Enum member).
   - `test_coerce` (parametrized)
   - `test_coerce_enum`
   - `test_coerce_enum_or_str`
4. **Profile resolution**: `vllm-default` is always applied as a baseline;
   the GPU-specific profile overlays it (later wins on conflicting keys);
   when no GPU profile matches, only the default applies. Substring matches
   are intentional (`b200` matches `gb200`) so both profiles overlay -- this
   is asserted explicitly. Missing YAML returns empty `(args, env, applied)`
   without raising.
   - `test_resolve_profiles_default_only`
   - `test_resolve_profiles_b200_overlays`
   - `test_resolve_profiles_mi300x`
   - `test_resolve_profiles_missing_yaml`
   - `test_resolve_profiles_b200_substring_in_gb200`
5. **`__post_init__` opt-in semantics**: with the env var set and a Cohere
   model id, profile values land on otherwise-default `EngineArgs` fields;
   user-supplied overrides (fields whose live value differs from the dataclass
   default) are preserved unchanged. Without the env var the call site is
   never entered, even for Cohere models. Non-Cohere models bypass the body
   even when the env var is set. Internal errors inside
   `apply_cohere_auto_config` are swallowed so a malformed YAML or buggy CEL
   clause cannot break engine startup.
   - `test_post_init_applies_for_cohere`
   - `test_post_init_respects_user_override`
   - `test_post_init_no_op_for_non_cohere`
   - `test_post_init_disabled_by_default`
   - `test_post_init_unknown_gpu_only_default`
   - `test_post_init_swallows_internal_error`
6. **Env-var application**: profile `env:` entries land in `os.environ` only
   when not already set; pre-existing values win and trigger an `INFO` log so
   user overrides are visible.
   - `test_post_init_env_vars_applied`
   - `test_post_init_env_var_user_set_wins`
7. **Drift warning**: a profile arg whose key does not match any `EngineArgs`
   field name logs a `WARNING` listing the unrecognized key (catches
   upstream renames without breaking startup).
   - `test_unknown_yaml_field_logs_drift_warning`

## Measurements

1. Pytest run produces a **JUnit XML report** in `${OUTPUT_DIR}` via the
   `tests/conftest.py` `pytest_configure` hook; `dorny/test-reporter@v2`
   surfaces it as a "cpu Test Report" check on the GitHub run.
   - `test_cohere_auto_config` suite -- `PRESENT` (pass/fail status)
   - See [JUnit XML Reporting (pytest)](../../code_notes/ci-and-automation.md#junit-xml-reporting-pytest)

No `upload-results` artifact is emitted -- `cpu` has no
benchmark/summary JSON upload step in `test-pipeline.yaml`.

## Compatibility

Features from [Feature Matrix](../feature_matrix.md)
([Compatibility Sources](../feature_matrix.md#compatibility-sources)):

1. **Input**: not applicable (CPU unit suite, no model inputs).
2. **Cohere Feature**: not applicable.
3. **Model Architecture**: drift-tested against every entry in
   `_COHERE_ARCHITECTURES` (`CohereForCausalLM`, `Cohere2ForCausalLM`,
   `Cohere2MoeForCausalLM`, `Cohere2VisionForConditionalGeneration`).
   - [`vllm/cohere/auto_config.py`](../../../../vllm/cohere/auto_config.py) -- `_COHERE_ARCHITECTURES`
4. **Quantization**: not applicable.
5. **Hardware**: CPU-only -- runs on the `vllm-cpu` Docker image on
   `ubuntu-latest`. No GPU runner involvement.
   - [`.github/workflows/pr-cpu-tests.yaml`](../../../../.github/workflows/pr-cpu-tests.yaml)
6. **vLLM Feature**: validates the auto-config integration with `EngineArgs`
   (covers all four GPU profiles: `b200`, `gb200`, `mi300x`, default).

## Implementation

Primary test:
[`tests/cohere/cpu/test_auto_config.py`](../../../../tests/cohere/cpu/test_auto_config.py)
Runtime paths:
[`vllm/cohere/auto_config.py`](../../../../vllm/cohere/auto_config.py),
[`vllm/engine/arg_utils.py`](../../../../vllm/engine/arg_utils.py) -- opt-in gate in `EngineArgs.__post_init__`,
[`vllm/cohere/hardware_profiles.yaml`](../../../../vllm/cohere/hardware_profiles.yaml) -- canonical profile data
CI entry: `cpu` group via
[`.github/workflows/pr-cpu-tests.yaml`](../../../../.github/workflows/pr-cpu-tests.yaml)
and `fast_check` / `all` expansion in
[`.github/workflows/test-pipeline.yaml`](../../../../.github/workflows/test-pipeline.yaml)

### Setup

1. **Inline YAML fixture**: `SAMPLE_YAML` declares one default profile plus
   per-GPU profiles for `b200`, `gb200`, and `mi300x`. Each test that needs
   profiles writes it to `tmp_path / "hardware_profiles.yaml"` via the
   `yaml_path` fixture and points `_DEFAULT_PROFILES_PATH` at that path with
   `monkeypatch.setattr(ac, "_DEFAULT_PROFILES_PATH", yaml_path)`.
2. **Cache invalidation**: an autouse `_clear_caches` fixture clears
   `detect_cohere_from_model_id.cache_clear()`, `_gpu_name.cache_clear()`,
   and `_load_profiles_doc.cache_clear()` before every test so previous
   YAML/GPU values don't leak.
3. **Log capture**: tests that assert on log output use the upstream
   `caplog_vllm` fixture from `tests/conftest.py` (vLLM sets
   `propagate=False` on its logger, so plain `caplog` won't see records).
4. **Env-var gating**: tests that exercise `EngineArgs.__post_init__` end to
   end use `monkeypatch.setenv("VLLM_ENABLE_COHERE_AUTO_CONFIG", "1")`. The
   gate in `arg_utils.py` reads `os.environ.get(...)` directly (not
   `envs.VLLM_ENABLE_COHERE_AUTO_CONFIG`) to avoid priming the `vllm.envs`
   cache before profile env vars are applied; tests must therefore use
   `monkeypatch.setenv` and not `envs.disable_envs_cache()`.
5. **GPU-name stubbing**: every test that drives profile resolution patches
   `_gpu_name` via `monkeypatch.setattr(ac, "_gpu_name", lambda: "...")`
   instead of importing `current_platform`, keeping the suite GPU-free.

</details>
