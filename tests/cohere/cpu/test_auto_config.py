# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for Cohere auto-config (vllm/cohere/auto_config.py).

These tests must not require a GPU, must not download a model, and must not
invoke `current_platform` for real device queries — everything that would
touch a GPU is patched.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import textwrap
import typing
from pathlib import Path

import pytest

from vllm.cohere import auto_config as ac
from vllm.engine.arg_utils import EngineArgs

SAMPLE_YAML = textwrap.dedent(
    """
    profiles:
      - name: vllm-default
        when: server.type == "vllm"
        args:
          gpu-memory-utilization: "0.95"
          enable-chunked-prefill: ""
          max-num-batched-tokens: "8192"
          max-num-seqs: "128"
          max-model-len: "256000"
          reasoning-config: >-
            {"reasoning_start_str":"<|START_THINKING|>",
            "reasoning_end_str":"<|END_THINKING|>"}
        env:
          VLLM_USE_V1: "1"
      - name: vllm-b200
        when: >-
          server.type == "vllm" &&
          matches(gpu.name, "b200") &&
          !matches(gpu.name, "gb200")
        args:
          attention-backend: "FLASHINFER"
        env:
          VLLM_ZERO_NULL_KV_BLOCK_AFTER_CUDA_GRAPH_CAPTURE: "true"
      - name: vllm-gb200
        when: server.type == "vllm" && matches(gpu.name, "gb200")
        args:
          attention-backend: "FLASHINFER"
      - name: vllm-mi300x
        when: server.type == "vllm" && matches(gpu.name, "mi300x")
        env:
          VLLM_ROCM_USE_AITER: "1"
    """
)


@pytest.fixture
def yaml_path(tmp_path: Path) -> Path:
    p = tmp_path / "hardware_profiles.yaml"
    p.write_text(SAMPLE_YAML)
    return p


@pytest.fixture(autouse=True)
def _isolate_globals():
    """Isolate process-global state across tests.

    `_apply_env` writes directly to `os.environ` (not via monkeypatch),
    so env mutations from one test would otherwise leak into later tests
    in the same session. Snapshot + restore covers that gap.
    """
    env_snapshot = os.environ.copy()
    ac.detect_cohere_from_model_id.cache_clear()
    ac._gpu_name.cache_clear()
    ac._load_profiles_doc.cache_clear()
    yield
    ac.detect_cohere_from_model_id.cache_clear()
    ac._gpu_name.cache_clear()
    ac._load_profiles_doc.cache_clear()
    os.environ.clear()
    os.environ.update(env_snapshot)


# ---------- Pure helpers ----------


def test_cohere_archs_subset_of_registry():
    """`_COHERE_ARCHITECTURES` must be a subset of the registered model archs.

    If upstream renames a Cohere arch, this test fails before the helper
    silently stops detecting it.
    """
    from vllm.model_executor.models import ModelRegistry

    supported = ModelRegistry.get_supported_archs()
    missing = ac._COHERE_ARCHITECTURES - supported
    assert not missing, (
        f"Cohere archs in _COHERE_ARCHITECTURES are no longer in the model "
        f"registry: {sorted(missing)}"
    )


# ---------- CEL `when:` evaluation ----------


@pytest.mark.parametrize(
    "when,gpu,expected",
    [
        ("", "anything", True),
        ('server.type == "vllm"', "B200", True),
        (
            'server.type == "vllm" && matches(gpu.name, "b200")',
            "NVIDIA B200 80GB HBM3",
            True,
        ),
        ('server.type == "vllm" && matches(gpu.name, "b200")', "NVIDIA H100", False),
        ('server.type == "vllm" && matches(gpu.name, "gb200")', "NVIDIA GB200", True),
        (
            'server.type == "vllm" && matches(gpu.name, "mi300x")',
            "AMD Instinct MI300X",
            True,
        ),
        ('server.type == "vllm" && matches(gpu.name, "mi300x")', "NVIDIA H100", False),
        ('matches(gpu.name, "b200")', "nvidia B200", True),
        ('matches(gpu.name, "h100|h200")', "NVIDIA H200", True),
        ('matches(gpu.name, "h100|h200")', "NVIDIA L40S", False),
    ],
)
def test_evaluate_when(when: str, gpu: str, expected: bool):
    assert ac._evaluate_when(when, gpu) is expected


def test_evaluate_when_namespace_shape():
    """Pin the variable namespace bound into CEL evaluation."""
    assert ac._evaluate_when('server.type == "vllm"', "x") is True
    assert ac._evaluate_when("has(gpu.name)", "x") is True


@pytest.mark.parametrize(
    "bad_clause",
    [
        "this is not (((CEL",  # parse error
        'cuda.version == "12.4"',  # unbound var at eval time
    ],
)
def test_evaluate_when_failure(bad_clause: str, caplog_vllm: pytest.LogCaptureFixture):
    with caplog_vllm.at_level(logging.WARNING, logger="vllm.cohere.auto_config"):
        assert ac._evaluate_when(bad_clause, "b200") is False
    assert any(
        "when-clause" in r.message and "failed" in r.message
        for r in caplog_vllm.records
    )


# ---------- Type coercion ----------


@pytest.mark.parametrize(
    "value,field_name,expected",
    [
        # primitive string -> typed value (delegated to vLLM's argparse fns)
        ("0.95", "gpu_memory_utilization", 0.95),
        ("8192", "max_num_seqs", 8192),
        ("FLASHINFER", "attention_backend", "FLASHINFER"),
        # native YAML values stringify-then-delegate
        (0.95, "gpu_memory_utilization", 0.95),
        (8192, "max_num_seqs", 8192),
        # human_readable_int_or_auto handles "auto" sentinel and "256k" suffixes
        ("auto", "max_model_len", -1),
        ("256k", "max_model_len", 256000),
        # BooleanOptionalAction fields: coerce bool/truthy/falsy strings inline
        ("", "enable_chunked_prefill", True),
        ("true", "enable_chunked_prefill", True),
        ("false", "enable_chunked_prefill", False),
        (True, "enable_chunked_prefill", True),
    ],
)
def test_coerce(value: object, field_name: str, expected: object):
    assert ac._coerce(value, field_name) == expected


def test_coerce_unknown_field_raises():
    """Unknown EngineArgs field names raise TypeError so _apply_args skips them."""
    with pytest.raises(TypeError):
        ac._coerce("0.95", "no_such_field_anywhere")


def test_coerce_invalid_value_raises():
    """Values that the underlying type fn rejects surface as TypeError."""
    with pytest.raises(TypeError):
        ac._coerce("not-a-number", "max_num_seqs")
    with pytest.raises(TypeError):
        ac._coerce("garbage", "enable_chunked_prefill")


def test_coerce_survives_forward_ref_annotation():
    """String-literal forward-ref `Field.type` doesn't break unrelated fields.

    When `typing.get_type_hints(EngineArgs)` raises (e.g. an unresolvable
    TYPE_CHECKING-only name), string-annotated fields fall back to
    `Any | None`. Non-string-annotated fields are untouched and continue to
    coerce correctly.
    """
    from vllm.engine.arg_utils import _compute_kwargs

    fields_by_name = {f.name: f for f in dataclasses.fields(EngineArgs)}
    target = fields_by_name["gpu_memory_utilization"]
    original_type = target.type
    original_annotation = EngineArgs.__annotations__["gpu_memory_utilization"]
    # `typing.get_type_hints` reads __annotations__, and `_compute_kwargs`
    # reads Field.type; mutate both so the failure path is exercised
    # deterministically, independent of any other forward-refs on EngineArgs.
    bad = "SomeUnimportedName | None"
    target.type = bad
    EngineArgs.__annotations__["gpu_memory_utilization"] = bad
    ac._engine_arg_kwargs.cache_clear()
    _compute_kwargs.cache_clear()
    try:
        # Unrelated (non-string-annotated) field still coerces correctly.
        assert ac._coerce("8192", "max_num_seqs") == 8192
        # The string-annotated field falls back to Any | None
        assert ac._coerce("0.95", "gpu_memory_utilization") == "0.95"
        assert ac._coerce(None, "gpu_memory_utilization") is None
        # Field.type restored after get_kwargs returned.
        assert target.type == bad
    finally:
        target.type = original_type
        EngineArgs.__annotations__["gpu_memory_utilization"] = original_annotation
        ac._engine_arg_kwargs.cache_clear()
        _compute_kwargs.cache_clear()


# ---------- Profile resolution ----------


def test_resolve_profiles_default_only(yaml_path: Path):
    args, env, applied = ac.resolve_profiles(
        gpu_name="NVIDIA H100", profiles_path=yaml_path
    )
    assert applied == ["vllm-default"]
    assert args["gpu-memory-utilization"] == "0.95"
    assert env["VLLM_USE_V1"] == "1"
    assert "attention-backend" not in args


def test_resolve_profiles_b200_overlays(yaml_path: Path):
    args, env, applied = ac.resolve_profiles(
        gpu_name="NVIDIA B200 80GB HBM3", profiles_path=yaml_path
    )
    assert applied == ["vllm-default", "vllm-b200"]
    assert args["attention-backend"] == "FLASHINFER"
    assert env["VLLM_ZERO_NULL_KV_BLOCK_AFTER_CUDA_GRAPH_CAPTURE"] == "true"


def test_resolve_profiles_mi300x(yaml_path: Path):
    args, env, applied = ac.resolve_profiles(
        gpu_name="AMD Instinct MI300X", profiles_path=yaml_path
    )
    assert "vllm-mi300x" in applied
    assert env["VLLM_ROCM_USE_AITER"] == "1"


def test_resolve_profiles_missing_yaml(
    tmp_path: Path, caplog_vllm: pytest.LogCaptureFixture
):
    with caplog_vllm.at_level(logging.WARNING, logger="vllm.cohere.auto_config"):
        args, env, applied = ac.resolve_profiles(
            gpu_name="x", profiles_path=tmp_path / "nope.yaml"
        )
    assert (args, env, applied) == ({}, {}, [])
    assert any("not found" in r.message for r in caplog_vllm.records)


def test_resolve_profiles_gb200_does_not_match_b200(yaml_path: Path):
    """`vllm-b200` excludes GB200 via `&& !matches(gpu.name, "gb200")`.

    Without that guard, GB200 would silently inherit B200-only settings
    (e.g. `mm-encoder-attn-backend`) and B200-only env vars. Pinning the
    selector keeps the two profiles isolated.
    """
    args, env, applied = ac.resolve_profiles(
        gpu_name="NVIDIA GB200 480GB", profiles_path=yaml_path
    )
    assert applied == ["vllm-default", "vllm-gb200"]
    assert args.get("attention-backend") == "FLASHINFER"


# ---------- End-to-end via EngineArgs.__post_init__ ----------


@pytest.fixture
def opt_in(monkeypatch: pytest.MonkeyPatch, yaml_path: Path):
    monkeypatch.setenv("VLLM_ENABLE_COHERE_AUTO_CONFIG", "1")
    monkeypatch.setattr(ac, "_DEFAULT_PROFILES_PATH", yaml_path)


@pytest.fixture
def fake_b200(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ac, "_gpu_name", lambda: "NVIDIA B200 80GB HBM3")


@pytest.fixture
def cohere_detected(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ac, "detect_cohere_from_model_id", lambda model, **_: True)


@pytest.fixture
def not_cohere(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ac, "detect_cohere_from_model_id", lambda model, **_: False)


def test_post_init_applies_for_cohere(
    opt_in: None,
    fake_b200: None,
    cohere_detected: None,
    caplog_vllm: pytest.LogCaptureFixture,
):
    with caplog_vllm.at_level(logging.INFO, logger="vllm.cohere.auto_config"):
        ea = EngineArgs(model="CohereLabs/c4ai-command-a")
    assert ea.gpu_memory_utilization == 0.95
    assert ea.enable_chunked_prefill is True
    assert ea.max_num_batched_tokens == 8192
    assert ea.max_num_seqs == 128
    assert ea.max_model_len == 256000
    assert ea.reasoning_config is not None
    assert ea.reasoning_config.reasoning_start_str == "<|START_THINKING|>"
    assert ea.reasoning_config.reasoning_end_str == "<|END_THINKING|>"
    # `attention_backend` lands as a string (matches `vllm serve` behavior;
    # downstream config validation resolves the string -> AttentionBackendEnum
    # the same way it does for CLI users).
    assert ea.attention_backend == "FLASHINFER"

    msgs = [r.message for r in caplog_vllm.records]
    assert any("profiles applied=['vllm-default', 'vllm-b200']" in m for m in msgs)
    assert any("--gpu-memory-utilization" in m and "0.95" in m for m in msgs)


def test_post_init_respects_user_override(
    opt_in: None,
    fake_b200: None,
    cohere_detected: None,
    caplog_vllm: pytest.LogCaptureFixture,
):
    with caplog_vllm.at_level(logging.INFO, logger="vllm.cohere.auto_config"):
        ea = EngineArgs(
            model="CohereLabs/c4ai-command-a",
            gpu_memory_utilization=0.80,
        )
    assert ea.gpu_memory_utilization == 0.80
    msgs = [r.message for r in caplog_vllm.records]
    assert any("--gpu-memory-utilization already set to 0.8 by user" in m for m in msgs)


def test_post_init_no_op_for_non_cohere(
    opt_in: None, fake_b200: None, not_cohere: None
):
    ea = EngineArgs(model="meta-llama/Llama-3-8B")
    field_map = {f.name: f for f in dataclasses.fields(EngineArgs)}
    assert ea.gpu_memory_utilization == field_map["gpu_memory_utilization"].default
    assert ea.enable_chunked_prefill is None
    assert ea.attention_backend is None


def test_post_init_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
    yaml_path: Path,
    fake_b200: None,
    cohere_detected: None,
):
    """Without VLLM_ENABLE_COHERE_AUTO_CONFIG set, the call site never
    invokes the helper, so even a Cohere model gets stock defaults.
    """
    monkeypatch.delenv("VLLM_ENABLE_COHERE_AUTO_CONFIG", raising=False)
    monkeypatch.setattr(ac, "_DEFAULT_PROFILES_PATH", yaml_path)
    ea = EngineArgs(model="CohereLabs/c4ai-command-a")
    assert ea.attention_backend is None
    assert ea.gpu_memory_utilization != 0.95


def test_post_init_unknown_gpu_only_default(
    opt_in: None, monkeypatch: pytest.MonkeyPatch, cohere_detected: None
):
    monkeypatch.setattr(ac, "_gpu_name", lambda: "NVIDIA RTX 4090")
    ea = EngineArgs(model="CohereLabs/c4ai-command-a")
    assert ea.gpu_memory_utilization == 0.95
    assert ea.attention_backend is None


def test_post_init_env_vars_applied(
    opt_in: None,
    fake_b200: None,
    cohere_detected: None,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("VLLM_USE_V1", raising=False)
    monkeypatch.delenv(
        "VLLM_ZERO_NULL_KV_BLOCK_AFTER_CUDA_GRAPH_CAPTURE", raising=False
    )
    EngineArgs(model="CohereLabs/c4ai-command-a")

    assert os.environ["VLLM_USE_V1"] == "1"
    assert os.environ["VLLM_ZERO_NULL_KV_BLOCK_AFTER_CUDA_GRAPH_CAPTURE"] == "true"


def test_post_init_env_var_user_set_wins(
    opt_in: None,
    fake_b200: None,
    cohere_detected: None,
    monkeypatch: pytest.MonkeyPatch,
    caplog_vllm: pytest.LogCaptureFixture,
):
    monkeypatch.setenv("VLLM_USE_V1", "0")
    with caplog_vllm.at_level(logging.INFO, logger="vllm.cohere.auto_config"):
        EngineArgs(model="CohereLabs/c4ai-command-a")

    assert os.environ["VLLM_USE_V1"] == "0"
    assert any("VLLM_USE_V1 already set" in r.message for r in caplog_vllm.records)


def test_post_init_swallows_internal_error(
    monkeypatch: pytest.MonkeyPatch,
    opt_in: None,
    cohere_detected: None,
    caplog_vllm: pytest.LogCaptureFixture,
):
    """Auto-config bug must never break a real launch."""

    def boom(*_: object, **__: object) -> typing.NoReturn:
        raise RuntimeError("synthetic")

    monkeypatch.setattr(ac, "resolve_profiles", boom)
    with caplog_vllm.at_level(logging.WARNING, logger="vllm.cohere.auto_config"):
        ea = EngineArgs(model="CohereLabs/c4ai-command-a")
    assert ea.model == "CohereLabs/c4ai-command-a"
    assert any("unexpected error" in r.message for r in caplog_vllm.records)


def test_unknown_yaml_field_logs_drift_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_b200: None,
    cohere_detected: None,
    caplog_vllm: pytest.LogCaptureFixture,
):
    yaml = tmp_path / "p.yaml"
    yaml.write_text(
        textwrap.dedent(
            """
            profiles:
              - name: vllm-default
                when: server.type == "vllm"
                args:
                  this-field-does-not-exist: "42"
            """
        )
    )
    monkeypatch.setenv("VLLM_ENABLE_COHERE_AUTO_CONFIG", "1")
    monkeypatch.setattr(ac, "_DEFAULT_PROFILES_PATH", yaml)
    with caplog_vllm.at_level(logging.WARNING, logger="vllm.cohere.auto_config"):
        EngineArgs(model="CohereLabs/c4ai-command-a")
    assert any(
        "not recognized as EngineArgs fields" in r.message for r in caplog_vllm.records
    )
