# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for AITERConfig: env-var defaults, JSON round-trip, hashing, and the
VllmConfig / worker-process sync of the rocm_aiter_ops AITER class vars."""

import multiprocessing

import pytest

from vllm.config import AITERConfig, VllmConfig
from vllm.platforms import current_platform

# Class vars init_from_config writes; snapshot/restore so a test that mutates the
# process-global rocm_aiter_ops class vars do not leak into the next one.
_AITER_CLASS_VARS = (
    "_AITER_ENABLED",
    "_CUSTOM_ALL_REDUCE_ENABLED",
    "_LINEAR_ENABLED",
    "_FMOE_ENABLED",
    "_MLA_ENABLED",
    "_MHA_ENABLED",
    "_SHUFFLE_KV_CACHE_ENABLED",
    "_TRITON_UNIFIED_ATTN_ENABLED",
    "_FP8BMM_ENABLED",
    "_FP4BMM_ENABLED",
    "_LINEAR_HIPBMM_ENABLED",
    "_TRITON_ROTARY_EMBED",
    "_MOE_SHARED_EXPERTS_ENABLED",
    "_MOE_SITUV2_A8W4",
    "_TRITON_UNQUANT_GEMM",
    "_MOE_DISPATCH_POLICY",
)


@pytest.fixture
def restore_aiter_class_vars():
    from vllm._aiter_ops import rocm_aiter_ops

    saved = {v: getattr(rocm_aiter_ops, v) for v in _AITER_CLASS_VARS}
    try:
        yield rocm_aiter_ops
    finally:
        for v, val in saved.items():
            setattr(rocm_aiter_ops, v, val)


def test_defaults_match_env_vars(monkeypatch: pytest.MonkeyPatch):
    """Every field defaults from its VLLM_ROCM_USE_AITER* env var, so an
    unset config reproduces prior behaviour."""
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_MOE", "0")
    monkeypatch.setenv("VLLM_ROCM_AITER_MOE_DISPATCH_POLICY", "2")

    cfg = AITERConfig()

    assert cfg.enabled is True
    assert cfg.moe is False
    assert cfg.moe_dispatch_policy == 2
    # untouched field keeps its env default
    assert cfg.mha is True


def test_explicit_values_override_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "0")

    cfg = AITERConfig(enabled=True, moe=False)

    assert cfg.enabled is True
    assert cfg.moe is False


def test_json_dict_round_trip():
    """The CLI passes --aiter-config as a JSON dict; EngineArgs then builds
    AITERConfig(**dict)."""
    payload = {"enabled": True, "moe": True, "mla": False}
    cfg = AITERConfig(**payload)

    assert (cfg.enabled, cfg.moe, cfg.mla) == (True, True, False)


def test_compute_hash_reflects_fields():
    base = AITERConfig(enabled=False)
    same = AITERConfig(enabled=False)
    flipped = AITERConfig(enabled=True)

    assert base.compute_hash() == same.compute_hash()
    assert base.compute_hash() != flipped.compute_hash()


def test_init_from_config_syncs_class_vars(restore_aiter_class_vars):
    """rocm_aiter_ops holds the config values in class vars for the hot path."""
    rocm_aiter_ops = restore_aiter_class_vars

    rocm_aiter_ops.init_from_config(
        AITERConfig(enabled=True, moe=False, mla=True, moe_dispatch_policy=3)
    )
    assert rocm_aiter_ops._AITER_ENABLED is True
    assert rocm_aiter_ops._FMOE_ENABLED is False
    assert rocm_aiter_ops._MLA_ENABLED is True
    assert rocm_aiter_ops._MOE_DISPATCH_POLICY == 3


def test_refresh_env_variables_restores_every_synced_field(
    restore_aiter_class_vars, monkeypatch: pytest.MonkeyPatch
):
    """refresh_env_variables() must reset every class var init_from_config()
    writes, so a test that monkeypatches an env var and then calls it does not
    leak state. Regression guard for _MOE_DISPATCH_POLICY, which init_from_config
    writes but refresh_env_variables historically skipped."""
    rocm_aiter_ops = restore_aiter_class_vars

    rocm_aiter_ops.init_from_config(
        AITERConfig(enabled=True, moe=False, moe_dispatch_policy=7)
    )
    assert rocm_aiter_ops._MOE_DISPATCH_POLICY == 7

    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "0")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_MOE", "1")
    monkeypatch.setenv("VLLM_ROCM_AITER_MOE_DISPATCH_POLICY", "0")
    rocm_aiter_ops.refresh_env_variables()

    assert rocm_aiter_ops._AITER_ENABLED is False
    assert rocm_aiter_ops._FMOE_ENABLED is True
    assert rocm_aiter_ops._MOE_DISPATCH_POLICY == 0


# Override that differs from every env default it touches
# (VLLM_ROCM_USE_AITER=False, _MOE=True, _MLA=True, dispatch policy 0).
_EXPECTED = (True, False, False, 2)


def _override_config() -> AITERConfig:
    return AITERConfig(enabled=True, moe=False, mla=False, moe_dispatch_policy=2)


def _class_var_tuple(ops) -> tuple:
    return (
        ops._AITER_ENABLED,
        ops._FMOE_ENABLED,
        ops._MLA_ENABLED,
        ops._MOE_DISPATCH_POLICY,
    )


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-specific")
def test_vllm_config_post_init_syncs_class_vars(restore_aiter_class_vars):
    """VllmConfig.__post_init__ sets the AITER class vars from aiter_config for the
    front-end / single-GPU path."""
    rocm_aiter_ops = restore_aiter_class_vars

    VllmConfig(aiter_config=_override_config())

    assert _class_var_tuple(rocm_aiter_ops) == _EXPECTED


def _worker_class_var_probe(vllm_config, q):
    """Runs in a spawned process: mimics a worker receiving VllmConfig by value."""
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.platforms import current_platform

    # Fresh process, VllmConfig arrived by (multiprocessing) serialization but
    # __post_init__ did not re-run -> class vars still at import-time env defaults.
    before = (
        rocm_aiter_ops._AITER_ENABLED,
        rocm_aiter_ops._FMOE_ENABLED,
        rocm_aiter_ops._MLA_ENABLED,
    )
    # What WorkerBase.__init__ does:
    current_platform.sync_process_config_state(vllm_config)
    after = _class_var_tuple(rocm_aiter_ops)
    q.put({"before": before, "after": after})


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-specific")
def test_worker_process_resyncs_class_vars(restore_aiter_class_vars):
    """A worker gets VllmConfig by value but __post_init__ is not re-run there,
    so WorkerBase.__init__ must call init_from_config to pick up overrides."""
    vllm_config = VllmConfig(aiter_config=_override_config())

    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_worker_class_var_probe, args=(vllm_config, q))
    p.start()
    result = q.get(timeout=180)
    p.join(timeout=30)

    # Child started from env defaults (VLLM_ROCM_USE_AITER=False, _MOE/_MLA=True),
    # not the override -> proves the boundary drops the __post_init__ sync.
    assert result["before"] == (False, True, True)
    # init_from_config in the worker recovers the configured values.
    assert result["after"] == _EXPECTED


def test_vllm_config_hash_includes_aiter_config():
    """aiter_config feeds VllmConfig.compute_hash so a field change busts the
    compilation cache."""
    h_moe_on = VllmConfig(aiter_config=AITERConfig(moe=True)).compute_hash()
    h_moe_off = VllmConfig(aiter_config=AITERConfig(moe=False)).compute_hash()
    assert h_moe_on != h_moe_off
