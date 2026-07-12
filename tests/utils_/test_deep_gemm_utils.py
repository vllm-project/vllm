# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Toolchain gating for DeepGEMM backend selection.

DeepGEMM JIT-compiles kernels at first use and its compiler asserts on
nvcc older than 12.3 — long after backend selection, mid-weight-load.
``is_deep_gemm_supported`` must therefore reject a positively identified
too-old toolchain up front, and only then.
"""

import pytest

from vllm.utils import deep_gemm as dg_utils

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def deep_gemm_otherwise_supported(monkeypatch):
    monkeypatch.setattr(dg_utils.envs, "VLLM_USE_DEEP_GEMM", True)
    monkeypatch.setattr(dg_utils, "has_deep_gemm", lambda: True)
    monkeypatch.setattr(dg_utils.current_platform, "support_deep_gemm", lambda: True)


@pytest.mark.parametrize(
    ("nvcc_version", "expected"),
    [
        ((12, 1), False),  # positively too old -> rejected
        ((12, 3), True),  # exact minimum -> allowed
        ((13, 2), True),  # newer -> allowed
        (None, True),  # unknown toolchain must not disqualify
    ],
)
def test_nvcc_gate(deep_gemm_otherwise_supported, monkeypatch, nvcc_version, expected):
    monkeypatch.setattr(dg_utils, "_nvcc_version", lambda: nvcc_version)
    assert dg_utils.is_deep_gemm_supported() is expected


def test_arch_and_env_gates_still_apply(monkeypatch):
    monkeypatch.setattr(dg_utils, "_nvcc_version", lambda: (13, 2))
    monkeypatch.setattr(dg_utils.envs, "VLLM_USE_DEEP_GEMM", True)
    monkeypatch.setattr(dg_utils, "has_deep_gemm", lambda: True)
    monkeypatch.setattr(dg_utils.current_platform, "support_deep_gemm", lambda: False)
    assert dg_utils.is_deep_gemm_supported() is False


def test_nvcc_version_parses_or_returns_none():
    version = dg_utils._nvcc_version.__wrapped__()
    assert version is None or (
        isinstance(version, tuple)
        and len(version) == 2
        and all(isinstance(part, int) for part in version)
    )
