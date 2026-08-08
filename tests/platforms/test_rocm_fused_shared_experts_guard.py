# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ``RocmPlatform._maybe_disable_aiter_fused_shared_experts``.

The guard only reads ``enable_eplb`` and ``use_ubatching``, so we use a stub
instead of a real ``ParallelConfig`` (which validates the platform on init).
"""

from types import SimpleNamespace

import pytest

import vllm._aiter_ops as aiter_ops_mod
import vllm.platforms.rocm as rocm_mod
from vllm.platforms.rocm import RocmPlatform

_FSE_ENV = "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS"


def _parallel_config(*, enable_eplb=False, use_ubatching=False):
    return SimpleNamespace(enable_eplb=enable_eplb, use_ubatching=use_ubatching)


@pytest.fixture
def fused_se(monkeypatch):
    """Control the fused shared-experts flag and record warnings.

    Returns a state dict: ``set(bool)`` forces the reported capability flag and
    ``warnings`` captures ``logger.warning_once`` messages. ``refresh_env_variables``
    is stubbed to mirror the env var into the capability flag.
    """
    state = {"enabled": False, "warnings": []}

    monkeypatch.setenv(_FSE_ENV, "1")

    def _is_enabled(cls):
        return state["enabled"]

    def _refresh(cls):
        import os

        state["enabled"] = os.environ.get(_FSE_ENV, "0").lower() in ("true", "1")

    monkeypatch.setattr(
        aiter_ops_mod.rocm_aiter_ops,
        "is_fusion_moe_shared_experts_enabled",
        classmethod(_is_enabled),
    )
    monkeypatch.setattr(
        aiter_ops_mod.rocm_aiter_ops,
        "refresh_env_variables",
        classmethod(_refresh),
    )
    monkeypatch.setattr(
        rocm_mod.logger,
        "warning_once",
        lambda msg, *a, **k: state["warnings"].append(msg % a if a else msg),
    )

    def _set(enabled: bool):
        state["enabled"] = enabled

    state["set"] = _set
    return state


def test_noop_when_fused_se_disabled(fused_se):
    fused_se["set"](False)
    # Nothing to disable even if both conflicting features are on.
    RocmPlatform._maybe_disable_aiter_fused_shared_experts(
        _parallel_config(enable_eplb=True, use_ubatching=True)
    )
    import os

    assert os.environ[_FSE_ENV] == "1"
    assert fused_se["warnings"] == []


def test_fused_se_kept_when_no_conflict(fused_se):
    fused_se["set"](True)
    RocmPlatform._maybe_disable_aiter_fused_shared_experts(_parallel_config())
    import os

    assert os.environ[_FSE_ENV] == "1"
    assert fused_se["enabled"] is True
    assert fused_se["warnings"] == []


def test_fused_se_disabled_on_eplb(fused_se):
    fused_se["set"](True)
    RocmPlatform._maybe_disable_aiter_fused_shared_experts(
        _parallel_config(enable_eplb=True)
    )
    import os

    assert os.environ[_FSE_ENV] == "0"
    assert fused_se["enabled"] is False
    assert len(fused_se["warnings"]) == 1
    assert "EPLB" in fused_se["warnings"][0]


def test_fused_se_disabled_on_dbo(fused_se):
    fused_se["set"](True)
    RocmPlatform._maybe_disable_aiter_fused_shared_experts(
        _parallel_config(use_ubatching=True)
    )
    import os

    assert os.environ[_FSE_ENV] == "0"
    assert fused_se["enabled"] is False
    assert "DBO" in fused_se["warnings"][0]


def test_warning_lists_all_conflicts(fused_se):
    fused_se["set"](True)
    RocmPlatform._maybe_disable_aiter_fused_shared_experts(
        _parallel_config(enable_eplb=True, use_ubatching=True)
    )
    msg = fused_se["warnings"][0]
    assert "EPLB" in msg and "DBO" in msg
