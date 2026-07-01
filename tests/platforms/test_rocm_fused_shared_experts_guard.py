# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``RocmPlatform._verify_aiter_fused_shared_experts``.

These verify that enabling AITER fused shared-experts
(``VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS``) together with features that
reason about the per-rank routed-expert layout (EPLB, dual/two batch overlap)
is rejected up front with a clear error, and that otherwise-valid combinations
are left untouched.

The guard only reads ``enable_eplb`` and ``use_ubatching`` off the parallel
config, so we use a lightweight stub instead of a real ``ParallelConfig``
(constructing one with ``enable_eplb=True`` triggers a CUDA-platform check that
is unrelated to what we exercise here).
"""

from types import SimpleNamespace

import pytest

import vllm._aiter_ops as aiter_ops_mod
from vllm.platforms.rocm import RocmPlatform


def _parallel_config(*, enable_eplb=False, use_ubatching=False):
    return SimpleNamespace(enable_eplb=enable_eplb, use_ubatching=use_ubatching)


@pytest.fixture
def set_fused_se(monkeypatch):
    """Return a setter that forces the fused shared-experts capability flag."""

    def _set(enabled: bool):
        monkeypatch.setattr(
            aiter_ops_mod.rocm_aiter_ops,
            "is_fusion_moe_shared_experts_enabled",
            classmethod(lambda cls: enabled),
        )

    return _set


def test_no_conflict_when_fused_se_disabled(set_fused_se):
    set_fused_se(False)
    # Even with both features on, a disabled fused-SE must not raise.
    parallel_config = _parallel_config(enable_eplb=True, use_ubatching=True)
    RocmPlatform._verify_aiter_fused_shared_experts(parallel_config)


def test_fused_se_alone_is_allowed(set_fused_se):
    set_fused_se(True)
    parallel_config = _parallel_config(enable_eplb=False, use_ubatching=False)
    RocmPlatform._verify_aiter_fused_shared_experts(parallel_config)


def test_fused_se_with_eplb_raises(set_fused_se):
    set_fused_se(True)
    parallel_config = _parallel_config(enable_eplb=True)
    with pytest.raises(ValueError, match="EPLB"):
        RocmPlatform._verify_aiter_fused_shared_experts(parallel_config)


def test_fused_se_with_ubatching_raises(set_fused_se):
    set_fused_se(True)
    parallel_config = _parallel_config(use_ubatching=True)
    with pytest.raises(ValueError, match="batch overlap"):
        RocmPlatform._verify_aiter_fused_shared_experts(parallel_config)


def test_eplb_conflict_checked_before_ubatching(set_fused_se):
    # When both conflict, EPLB is reported first (deterministic message).
    set_fused_se(True)
    parallel_config = _parallel_config(enable_eplb=True, use_ubatching=True)
    with pytest.raises(ValueError, match="EPLB"):
        RocmPlatform._verify_aiter_fused_shared_experts(parallel_config)
