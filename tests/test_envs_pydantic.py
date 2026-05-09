# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Targeted tests for tricky pydantic-settings behaviors in vllm/envs.py.

Covers the special-cased fields called out in the refactor design:
- Cross-var defaults (VLLM_DP_RANK_LOCAL -> VLLM_DP_RANK)
- Alias fallbacks (VLLM_DO_NOT_TRACK -> DO_NOT_TRACK)
- Side-effect defaults (VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME)
- Tri-state semantics (VLLM_PLUGINS)
- Choice validation (VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS)
- Case-insensitive literals (VLLM_FLOAT32_MATMUL_PRECISION, VLLM_MM_HASHER_ALGORITHM)
- Clamping (VLLM_LOG_STATS_INTERVAL)
- Widened bool accept set (yes/no/on/off)
- Cache semantics and validate_environ
"""

from __future__ import annotations

import importlib
import os
import sys

import pytest

_EXTERNAL_VARS = {
    "DO_NOT_TRACK",
    "VLLM_PRECOMPILED_WHEEL_LOCATION",
    "JAX_PLATFORMS",
    "MAX_JOBS",
    "CMAKE_BUILD_TYPE",
    "CUDA_HOME",
}


def _reload_envs():
    if "vllm.envs" in sys.modules:
        del sys.modules["vllm.envs"]
    return importlib.import_module("vllm.envs")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Strip all VLLM_ and relevant external vars for test isolation."""
    for k in list(os.environ.keys()):
        if k.startswith("VLLM_") or k in _EXTERNAL_VARS:
            monkeypatch.delenv(k, raising=False)


def test_port_uri_error_k8s(monkeypatch):
    monkeypatch.setenv("VLLM_PORT", "http://my-service.default.svc.cluster.local")
    envs = _reload_envs()
    with pytest.raises(ValueError, match="appears to be a URI"):
        _ = envs.VLLM_PORT


def test_port_invalid_integer(monkeypatch):
    monkeypatch.setenv("VLLM_PORT", "not-a-number")
    envs = _reload_envs()
    with pytest.raises(ValueError, match="must be a valid integer"):
        _ = envs.VLLM_PORT


def test_port_valid(monkeypatch):
    monkeypatch.setenv("VLLM_PORT", "8000")
    envs = _reload_envs()
    assert envs.VLLM_PORT == 8000


def test_port_unset(monkeypatch):
    envs = _reload_envs()
    assert envs.VLLM_PORT is None


def test_do_not_track_fallback(monkeypatch):
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    envs = _reload_envs()
    assert envs.VLLM_DO_NOT_TRACK is True


def test_do_not_track_vllm_wins_over_fallback(monkeypatch):
    # The VLLM_-prefixed name is the canonical one; it should be preferred.
    monkeypatch.setenv("VLLM_DO_NOT_TRACK", "0")
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    envs = _reload_envs()
    assert envs.VLLM_DO_NOT_TRACK is False


def test_dp_rank_local_fallback_to_dp_rank(monkeypatch):
    monkeypatch.setenv("VLLM_DP_RANK", "3")
    envs = _reload_envs()
    assert envs.VLLM_DP_RANK_LOCAL == 3


def test_dp_rank_local_explicit_wins(monkeypatch):
    monkeypatch.setenv("VLLM_DP_RANK", "3")
    monkeypatch.setenv("VLLM_DP_RANK_LOCAL", "7")
    envs = _reload_envs()
    assert envs.VLLM_DP_RANK_LOCAL == 7


def test_object_storage_shm_buffer_autogen():
    # Unset -> UUID generated AND written back to os.environ.
    envs = _reload_envs()
    name = envs.VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME
    assert name.startswith("VLLM_OBJECT_STORAGE_SHM_BUFFER_")
    assert os.environ["VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME"] == name


def test_object_storage_shm_buffer_explicit(monkeypatch):
    monkeypatch.setenv("VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME", "my_name")
    envs = _reload_envs()
    assert envs.VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME == "my_name"


def test_use_precompiled_via_wheel_location(monkeypatch):
    monkeypatch.setenv("VLLM_PRECOMPILED_WHEEL_LOCATION", "/tmp/some.whl")
    envs = _reload_envs()
    assert envs.VLLM_USE_PRECOMPILED is True


def test_use_precompiled_unset():
    envs = _reload_envs()
    assert envs.VLLM_USE_PRECOMPILED is False


def test_plugins_unset_is_none():
    envs = _reload_envs()
    assert envs.VLLM_PLUGINS is None


def test_plugins_empty_string(monkeypatch):
    # Preserves pre-refactor behavior: "".split(",") yields [""].
    monkeypatch.setenv("VLLM_PLUGINS", "")
    envs = _reload_envs()
    assert envs.VLLM_PLUGINS == [""]


def test_plugins_csv(monkeypatch):
    monkeypatch.setenv("VLLM_PLUGINS", "a,b,c")
    envs = _reload_envs()
    assert envs.VLLM_PLUGINS == ["a", "b", "c"]


def test_gpt_oss_labels_valid(monkeypatch):
    monkeypatch.setenv(
        "VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS", "container,code_interpreter"
    )
    envs = _reload_envs()
    assert {
        "container",
        "code_interpreter",
    } == envs.VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS


def test_gpt_oss_labels_invalid(monkeypatch):
    monkeypatch.setenv("VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS", "container,bogus")
    envs = _reload_envs()
    with pytest.raises(ValueError, match="bogus|Invalid"):
        _ = envs.VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS


def test_log_stats_interval_clamps_nonpositive(monkeypatch):
    monkeypatch.setenv("VLLM_LOG_STATS_INTERVAL", "-1")
    envs = _reload_envs()
    assert envs.VLLM_LOG_STATS_INTERVAL == 10.0


def test_log_stats_interval_zero_clamps(monkeypatch):
    monkeypatch.setenv("VLLM_LOG_STATS_INTERVAL", "0")
    envs = _reload_envs()
    assert envs.VLLM_LOG_STATS_INTERVAL == 10.0


def test_log_stats_interval_positive_kept(monkeypatch):
    monkeypatch.setenv("VLLM_LOG_STATS_INTERVAL", "5.5")
    envs = _reload_envs()
    assert envs.VLLM_LOG_STATS_INTERVAL == 5.5


def test_logging_level_uppercased(monkeypatch):
    monkeypatch.setenv("VLLM_LOGGING_LEVEL", "debug")
    envs = _reload_envs()
    assert envs.VLLM_LOGGING_LEVEL == "DEBUG"


def test_mm_hasher_case_insensitive_coerced(monkeypatch):
    # Pre-refactor: env_with_choices(case_sensitive=False) accepted upper-case
    # but returned the value unchanged. Now coerced to canonical lower-case.
    # The only consumer (vllm/multimodal/hasher.py) already calls .lower(),
    # so this is a behavior-equivalent normalization.
    monkeypatch.setenv("VLLM_MM_HASHER_ALGORITHM", "SHA256")
    envs = _reload_envs()
    assert envs.VLLM_MM_HASHER_ALGORITHM == "sha256"


def test_float32_precision_case_insensitive_coerced(monkeypatch):
    # Pre-refactor: returned the user-typed casing. torch.set_float32_matmul_precision
    # is case-sensitive and silently warns + no-ops on upper-case input, so the
    # old behavior was a latent miconfiguration. Coercing to lower-case fixes it.
    monkeypatch.setenv("VLLM_FLOAT32_MATMUL_PRECISION", "HIGH")
    envs = _reload_envs()
    assert envs.VLLM_FLOAT32_MATMUL_PRECISION == "high"


def test_bool_widened_accepts_yes(monkeypatch):
    # Pydantic accepts yes/no/on/off in addition to 1/0/true/false.
    # Widened accept set relative to pre-refactor (documented in PR).
    monkeypatch.setenv("VLLM_SERVER_DEV_MODE", "yes")
    envs = _reload_envs()
    assert envs.VLLM_SERVER_DEV_MODE is True


def test_bool_accepts_true(monkeypatch):
    monkeypatch.setenv("VLLM_SERVER_DEV_MODE", "true")
    envs = _reload_envs()
    assert envs.VLLM_SERVER_DEV_MODE is True


def test_bool_accepts_1(monkeypatch):
    monkeypatch.setenv("VLLM_SERVER_DEV_MODE", "1")
    envs = _reload_envs()
    assert envs.VLLM_SERVER_DEV_MODE is True


def test_is_set_true(monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "secret")
    envs = _reload_envs()
    assert envs.is_set("VLLM_API_KEY") is True


def test_is_set_false():
    envs = _reload_envs()
    assert envs.is_set("VLLM_API_KEY") is False


def test_is_set_unknown_raises():
    envs = _reload_envs()
    with pytest.raises(AttributeError):
        envs.is_set("VLLM_BOGUS_NOT_A_REAL_VAR")


def test_validate_environ_unknown_var_warns(monkeypatch, caplog):
    import logging as _logging

    monkeypatch.setenv("VLLM_DEFINITELY_NOT_REAL", "1")
    envs = _reload_envs()
    with caplog.at_level(_logging.WARNING, logger="vllm.envs"):
        envs.validate_environ(hard_fail=False)
    assert any("VLLM_DEFINITELY_NOT_REAL" in r.message for r in caplog.records)


def test_validate_environ_unknown_var_raises(monkeypatch):
    monkeypatch.setenv("VLLM_DEFINITELY_NOT_REAL", "1")
    envs = _reload_envs()
    with pytest.raises(ValueError, match="VLLM_DEFINITELY_NOT_REAL"):
        envs.validate_environ(hard_fail=True)


def test_dir_exposes_known_vars():
    envs = _reload_envs()
    names = dir(envs)
    for expected in [
        "VLLM_PORT",
        "VLLM_LOGGING_LEVEL",
        "MAX_JOBS",
        "CUDA_HOME",
        "VLLM_DO_NOT_TRACK",
        "Q_SCALE_CONSTANT",
    ]:
        assert expected in names, f"{expected} missing from dir(envs)"


def test_environment_variables_shim_present():
    envs = _reload_envs()
    # The legacy dict-of-lambdas shim is kept for external callers.
    assert isinstance(envs.environment_variables, dict)
    assert "VLLM_PORT" in envs.environment_variables
    # Calling the lambda returns the resolved value.
    assert envs.environment_variables["VLLM_PORT"]() is None


def test_cache_hides_env_mutation(monkeypatch):
    envs = _reload_envs()
    monkeypatch.setenv("VLLM_LOGGING_LEVEL", "debug")
    assert envs.VLLM_LOGGING_LEVEL == "DEBUG"

    envs.enable_envs_cache()
    try:
        monkeypatch.setenv("VLLM_LOGGING_LEVEL", "error")
        # Cached - ignores env mutation.
        assert envs.VLLM_LOGGING_LEVEL == "DEBUG"
    finally:
        envs.disable_envs_cache()

    # Fresh read after disable.
    assert envs.VLLM_LOGGING_LEVEL == "ERROR"


def test_disable_envs_cache_resets_singleton(monkeypatch):
    envs = _reload_envs()
    monkeypatch.setenv("VLLM_LOGGING_LEVEL", "warning")
    assert envs.VLLM_LOGGING_LEVEL == "WARNING"

    envs.enable_envs_cache()
    envs.disable_envs_cache()

    monkeypatch.setenv("VLLM_LOGGING_LEVEL", "critical")
    # After disable, the singleton is cleared, so the new value is read.
    assert envs.VLLM_LOGGING_LEVEL == "CRITICAL"
