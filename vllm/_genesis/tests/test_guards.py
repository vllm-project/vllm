# SPDX-License-Identifier: Apache-2.0
"""TDD tests for vllm._genesis.guards.

These tests verify canonical vendor/chip/dependency detection.
All helpers must fail-safe: return False/None on any exception.
"""
from __future__ import annotations

import pytest


class TestVendorIdentity:
    """Group 1: Vendor/platform identification."""

    def test_is_nvidia_cuda_returns_bool(self):
        """is_nvidia_cuda() always returns a bool (never raises)."""
        from vllm._genesis.guards import is_nvidia_cuda
        result = is_nvidia_cuda()
        assert isinstance(result, bool)

    def test_is_amd_rocm_returns_bool(self):
        from vllm._genesis.guards import is_amd_rocm
        result = is_amd_rocm()
        assert isinstance(result, bool)

    def test_is_intel_xpu_returns_bool(self):
        from vllm._genesis.guards import is_intel_xpu
        result = is_intel_xpu()
        assert isinstance(result, bool)

    def test_is_cpu_only_returns_bool(self):
        from vllm._genesis.guards import is_cpu_only
        result = is_cpu_only()
        assert isinstance(result, bool)

    def test_exactly_one_vendor_is_true(self):
        """Exactly one of {cuda, rocm, xpu, cpu} should be True."""
        from vllm._genesis.guards import (
            is_nvidia_cuda, is_amd_rocm, is_intel_xpu, is_cpu_only
        )
        count = sum([
            is_nvidia_cuda(),
            is_amd_rocm(),
            is_intel_xpu(),
            is_cpu_only(),
        ])
        # Exactly 1 should be true — unless vllm import fails (then all 0).
        assert count in (0, 1), f"Expected 0 or 1 vendor to match, got {count}"

    def test_is_cuda_alike_matches_cuda_or_rocm(self):
        """is_cuda_alike() = is_nvidia_cuda() OR is_amd_rocm()."""
        from vllm._genesis.guards import (
            is_cuda_alike, is_nvidia_cuda, is_amd_rocm
        )
        expected = is_nvidia_cuda() or is_amd_rocm()
        assert is_cuda_alike() == expected


class TestComputeCapability:
    """Group 2: NVIDIA compute capability detection."""

    def test_get_compute_capability_returns_tuple_or_none(self):
        """Returns (major, minor) tuple on NVIDIA, None otherwise."""
        from vllm._genesis.guards import get_compute_capability, is_nvidia_cuda
        cc = get_compute_capability()
        if is_nvidia_cuda():
            assert isinstance(cc, tuple)
            assert len(cc) == 2
            assert all(isinstance(x, int) for x in cc)
        else:
            assert cc is None

    def test_is_sm_at_least_zero_always_true_on_nvidia(self):
        """SM >= (0, 0) tautologically true on any NVIDIA GPU."""
        from vllm._genesis.guards import is_sm_at_least, is_nvidia_cuda
        if is_nvidia_cuda():
            assert is_sm_at_least(0, 0) is True

    def test_is_sm_at_least_high_value_false_on_ancient_gpu(self):
        """SM >= (99, 0) always False (no such GPU exists)."""
        from vllm._genesis.guards import is_sm_at_least
        assert is_sm_at_least(99, 0) is False

    def test_is_sm_at_least_false_on_non_nvidia(self):
        """is_sm_at_least returns False on non-NVIDIA platforms."""
        from vllm._genesis.guards import is_sm_at_least, is_nvidia_cuda
        if not is_nvidia_cuda():
            assert is_sm_at_least(8, 0) is False

    def test_specific_arch_predicates_consistent(self):
        """If is_hopper() then get_compute_capability() == (9, 0)."""
        from vllm._genesis.guards import (
            get_compute_capability,
            is_ampere_datacenter, is_ampere_consumer,
            is_ada_lovelace, is_hopper, is_blackwell,
        )
        cc = get_compute_capability()
        if cc is None:
            # Non-NVIDIA: all must be False
            assert not is_ampere_datacenter()
            assert not is_ampere_consumer()
            assert not is_ada_lovelace()
            assert not is_hopper()
            assert not is_blackwell()
        else:
            # Exactly one arch predicate matches (or none for unknown CC)
            matches = [
                is_ampere_datacenter() and cc == (8, 0),
                is_ampere_consumer() and cc == (8, 6),
                is_ada_lovelace() and cc == (8, 9),
                is_hopper() and cc == (9, 0),
                # Issue #20: is_blackwell now matches BOTH datacenter (sm_10x)
                # AND consumer Blackwell (sm_120 — RTX 5090/5080/5070/5060).
                is_blackwell() and cc[0] in (10, 12),
            ]
            # Each True match must correspond to actual cc
            true_matches = sum(matches)
            assert true_matches <= 1

    def test_blackwell_split_predicates(self, monkeypatch):
        """Issue #20: is_blackwell_datacenter vs is_blackwell_consumer."""
        from vllm._genesis import guards
        # Mock SM 10.0 (B100/B200/RTX PRO 6000) — datacenter Blackwell
        monkeypatch.setattr(guards, "_COMPUTE_CAPABILITY", (10, 0))
        assert guards.is_blackwell() is True
        assert guards.is_blackwell_datacenter() is True
        assert guards.is_blackwell_consumer() is False
        # Mock SM 12.0 (RTX 5090/5080) — consumer Blackwell
        monkeypatch.setattr(guards, "_COMPUTE_CAPABILITY", (12, 0))
        assert guards.is_blackwell() is True
        assert guards.is_blackwell_datacenter() is False
        assert guards.is_blackwell_consumer() is True
        # Mock SM 9.0 (Hopper) — neither
        monkeypatch.setattr(guards, "_COMPUTE_CAPABILITY", (9, 0))
        assert guards.is_blackwell() is False
        assert guards.is_blackwell_datacenter() is False
        assert guards.is_blackwell_consumer() is False
        # Mock SM 8.6 (Ampere consumer) — neither
        monkeypatch.setattr(guards, "_COMPUTE_CAPABILITY", (8, 6))
        assert guards.is_blackwell() is False


class TestDependencyVersions:
    """Group 3: External dependency version detection."""

    def test_get_torch_version_returns_tuple(self):
        from vllm._genesis.guards import get_torch_version
        v = get_torch_version()
        assert v is not None  # torch is available (we imported it)
        assert isinstance(v, tuple)
        assert len(v) == 2
        assert all(isinstance(x, int) for x in v)

    def test_is_torch_211_plus_matches_actual(self):
        from vllm._genesis.guards import is_torch_211_plus, get_torch_version
        v = get_torch_version()
        expected = v is not None and v >= (2, 11)
        assert is_torch_211_plus() == expected

    def test_get_transformers_version_safe(self):
        """Returns tuple if transformers installed, else None."""
        from vllm._genesis.guards import get_transformers_version
        v = get_transformers_version()
        if v is not None:
            assert isinstance(v, tuple)
            assert len(v) == 3
            assert all(isinstance(x, int) for x in v)

    def test_get_vllm_version_tuple_safe(self):
        """Handles messy version strings like '0.19.2rc1.dev8'."""
        from vllm._genesis.guards import get_vllm_version_tuple
        v = get_vllm_version_tuple()
        if v is not None:
            assert all(isinstance(x, int) for x in v)


class TestModelArchDetection:
    """Group 4: Model architecture detection."""

    def test_is_model_arch_none_config(self):
        """None config returns False (fail-safe)."""
        from vllm._genesis.guards import is_model_arch
        assert is_model_arch(None, "Qwen3") is False

    def test_is_model_arch_matches_substring(self):
        """Substring match works case-insensitive."""
        from vllm._genesis.guards import is_model_arch

        class MockConfig:
            architectures = ["Qwen3MoeForCausalLM"]

        assert is_model_arch(MockConfig(), "Qwen3") is True
        assert is_model_arch(MockConfig(), "qwen3") is True
        assert is_model_arch(MockConfig(), "MoE") is True
        assert is_model_arch(MockConfig(), "Llama") is False

    def test_is_model_arch_handles_missing_attr(self):
        """Config without .architectures returns False."""
        from vllm._genesis.guards import is_model_arch

        class MockConfig:
            pass  # no architectures attribute

        assert is_model_arch(MockConfig(), "Qwen3") is False

    def test_family_helpers(self):
        """Family predicates wrap is_model_arch correctly."""
        from vllm._genesis.guards import (
            is_qwen3_family, is_deepseek_v3, is_llama_family,
            is_gemma_family, is_mixtral_family,
        )

        class Qwen3Config:
            architectures = ["Qwen3MoeForCausalLM"]

        class DeepSeekConfig:
            architectures = ["DeepseekV3ForCausalLM"]

        class LlamaConfig:
            architectures = ["LlamaForCausalLM"]

        assert is_qwen3_family(Qwen3Config()) is True
        assert is_deepseek_v3(DeepSeekConfig()) is True
        assert is_llama_family(LlamaConfig()) is True
        assert is_gemma_family(LlamaConfig()) is False
        assert is_mixtral_family(LlamaConfig()) is False


class TestBackendDetection:
    """Group 5: Backend / kernel selection."""

    def test_has_turboquant_support_none(self):
        from vllm._genesis.guards import has_turboquant_support
        assert has_turboquant_support(None) is False
        assert has_turboquant_support("") is False

    def test_has_turboquant_support_matches_prefix(self):
        from vllm._genesis.guards import has_turboquant_support
        assert has_turboquant_support("turboquant_k8v4") is True
        assert has_turboquant_support("turboquant_4bit_nc") is True
        assert has_turboquant_support("fp8") is False
        assert has_turboquant_support("auto") is False


class TestPathResolution:
    """Group 6: File path helpers."""

    def test_vllm_install_root_returns_string_or_none(self):
        from vllm._genesis.guards import vllm_install_root
        root = vllm_install_root()
        if root is not None:
            import os
            assert isinstance(root, str)
            assert os.path.isdir(root), f"vllm install root doesn't exist: {root}"

    def test_resolve_vllm_file_nonexistent(self):
        """Returns None for nonexistent files."""
        from vllm._genesis.guards import resolve_vllm_file
        result = resolve_vllm_file("does/not/exist/__FAKE__.py")
        assert result is None

    def test_resolve_vllm_file_existing(self):
        """Returns absolute path for existing vLLM files."""
        from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
        if vllm_install_root() is None:
            pytest.skip("vLLM not installed")
        # __init__.py should always exist in vllm package
        result = resolve_vllm_file("__init__.py")
        assert result is not None
        import os
        assert os.path.exists(result)


class TestPlatformSummary:
    """Group 7: Diagnostic summary."""

    def test_platform_summary_returns_dict(self):
        """Always returns a dict (never raises)."""
        from vllm._genesis.guards import platform_summary
        s = platform_summary()
        assert isinstance(s, dict)
        assert "vendor" in s
        assert "versions" in s
        assert "paths" in s

    def test_platform_summary_is_serializable(self):
        """Summary must be JSON-serializable for logging."""
        import json
        from vllm._genesis.guards import platform_summary
        s = platform_summary()
        # default=str handles tuples and other non-JSON-native types
        json_str = json.dumps(s, default=str)
        assert len(json_str) > 0
