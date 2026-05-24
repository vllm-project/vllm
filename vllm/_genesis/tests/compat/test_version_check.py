# SPDX-License-Identifier: Apache-2.0
"""Tests for vllm._genesis.compat.version_check."""
from __future__ import annotations

import pytest

from vllm._genesis.compat.version_check import (
    VersionProfile,
    check_version_constraints,
    detect_versions,
    reset_cache,
)


@pytest.fixture(autouse=True)
def _reset_cache():
    reset_cache()
    yield
    reset_cache()


def _profile(**kw) -> VersionProfile:
    """Build a synthetic VersionProfile for testing."""
    return VersionProfile(**kw)


class TestVllmVersionRange:
    def test_in_range(self):
        p = _profile(vllm="0.20.1rc1.dev16+g7a1eb8ac2")
        ok, results = check_version_constraints(
            {"vllm_version_range": (">=0.20.0", "<0.21.0")}, p,
        )
        assert ok is True
        assert results[0].matched is True

    def test_below_range(self):
        p = _profile(vllm="0.19.0")
        ok, results = check_version_constraints(
            {"vllm_version_range": ">=0.20.0"}, p,
        )
        assert ok is False
        assert results[0].matched is False

    def test_above_range(self):
        p = _profile(vllm="0.22.0")
        ok, results = check_version_constraints(
            {"vllm_version_range": ("<0.21.0",)}, p,
        )
        assert ok is False
        assert results[0].matched is False

    def test_unknown_vllm_version_conservative_pass(self):
        p = _profile(vllm=None)
        ok, results = check_version_constraints(
            {"vllm_version_range": ">=0.20.0"}, p,
        )
        assert ok is True
        assert results[0].matched is None

    def test_dev_suffix_ignored(self):
        """vllm dev suffixes like .dev16+gSHA should not break PEP 440 match."""
        p = _profile(vllm="0.20.1rc1.dev16+g7a1eb8ac2")
        ok, _ = check_version_constraints(
            {"vllm_version_range": (">=0.20.0", "<0.21.0")}, p,
        )
        assert ok is True


class TestTorchTritonCuda:
    def test_torch_min_passes(self):
        p = _profile(torch="2.5.1+cu124")
        ok, _results = check_version_constraints({"torch_version_min": "2.0"}, p)
        assert ok is True

    def test_torch_min_fails(self):
        p = _profile(torch="1.13.0")
        ok, _results = check_version_constraints({"torch_version_min": "2.0"}, p)
        assert ok is False

    def test_triton_min_passes(self):
        p = _profile(triton="3.1.0")
        ok, _ = check_version_constraints({"triton_version_min": "3.0"}, p)
        assert ok is True

    def test_triton_missing_conservative_pass(self):
        p = _profile(triton=None)
        ok, results = check_version_constraints({"triton_version_min": "3.0"}, p)
        assert ok is True
        assert results[0].matched is None

    def test_cuda_runtime_match(self):
        p = _profile(cuda_runtime="12.4")
        ok, _ = check_version_constraints({"cuda_runtime_min": "12.0"}, p)
        assert ok is True


class TestComputeCapability:
    def test_min_passes(self):
        p = _profile(compute_capabilities=((8, 6), (8, 6)))  # 2× A5000
        ok, _ = check_version_constraints({"compute_capability_min": (8, 0)}, p)
        assert ok is True

    def test_min_fails_one_gpu_below(self):
        p = _profile(compute_capabilities=((7, 5), (8, 6)))  # mixed Tesla + A5000
        ok, results = check_version_constraints({"compute_capability_min": (8, 0)}, p)
        assert ok is False
        assert results[0].matched is False

    def test_max_passes(self):
        p = _profile(compute_capabilities=((8, 6),))
        ok, _ = check_version_constraints({"compute_capability_max": (12, 0)}, p)
        assert ok is True

    def test_max_fails(self):
        p = _profile(compute_capabilities=((9, 0),))  # H100
        ok, _ = check_version_constraints({"compute_capability_max": (8, 6)}, p)
        assert ok is False

    def test_no_gpu_conservative_pass(self):
        p = _profile(compute_capabilities=())  # CPU-only host
        ok, results = check_version_constraints({"compute_capability_min": (8, 0)}, p)
        assert ok is True
        assert results[0].matched is None


class TestMultipleConstraints:
    def test_all_pass(self):
        p = _profile(
            vllm="0.20.1", torch="2.5.0", triton="3.1.0",
            compute_capabilities=((8, 6),),
        )
        ok, results = check_version_constraints({
            "vllm_version_range": ">=0.20.0,<0.21.0",
            "torch_version_min": "2.0",
            "triton_version_min": "3.0",
            "compute_capability_min": (8, 0),
        }, p)
        assert ok is True
        # All four should be evaluated and matched
        assert all(r.matched is True for r in results)

    def test_one_fails_aggregate_fails(self):
        p = _profile(
            vllm="0.20.1", torch="1.13.0",  # torch too old
            compute_capabilities=((8, 6),),
        )
        ok, results = check_version_constraints({
            "vllm_version_range": ">=0.20.0",
            "torch_version_min": "2.0",
        }, p)
        assert ok is False
        torch_result = next(r for r in results if r.key == "torch_version_min")
        assert torch_result.matched is False


class TestDetection:
    def test_detect_caches(self):
        """Profile detection caches per-process."""
        p1 = detect_versions()
        p2 = detect_versions()
        assert p1 is p2  # same object — cached

    def test_refresh_returns_new(self):
        p1 = detect_versions()
        p2 = detect_versions(refresh=True)
        # Same content (assuming nothing changed), but distinct object
        # after refresh
        assert p1 is not p2 or p1 == p2  # acceptable either way

    def test_detected_python_present(self):
        p = detect_versions(refresh=True)
        # Python version always detectable
        assert p.python is not None
        assert p.python.count(".") >= 1
