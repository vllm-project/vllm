# SPDX-License-Identifier: Apache-2.0
"""Integration tests — verify platform matrix behavior for all Genesis kernels.

Each test simulates a specific platform (NVIDIA Ampere/Ada/Hopper/Blackwell,
AMD ROCm, Intel XPU, CPU) and verifies that all kernels behave correctly
(apply OR gracefully skip).

This validates the "МЫ ЧИНИМ, НЕ ЛОМАЕМ" guarantee: no kernel ever crashes
the engine regardless of platform.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations


import pytest


# ═══════════════════════════════════════════════════════════════════════════
#                      PLATFORM MOCK FACTORIES
# ═══════════════════════════════════════════════════════════════════════════

def _mock_platform(
    monkeypatch,
    is_cuda=False,
    is_rocm=False,
    is_xpu=False,
    is_cpu=False,
    compute_capability=None,
    gcn_arch="",
):
    """Helper: mock all platform detection helpers consistently.

    Note: as of v7.62, the platform detectors are snapshot-at-load
    constants (no @functools.cache), so we just monkeypatch the public
    attribute directly. The earlier `.cache_clear()` calls are no longer
    needed — the functions are plain returns of module-level constants.
    """
    from vllm._genesis import guards

    monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: is_cuda)
    monkeypatch.setattr(guards, "is_amd_rocm", lambda: is_rocm)
    monkeypatch.setattr(guards, "is_intel_xpu", lambda: is_xpu)
    monkeypatch.setattr(guards, "is_cpu_only", lambda: is_cpu)
    monkeypatch.setattr(guards, "is_cuda_alike",
                        lambda: is_cuda or is_rocm)

    monkeypatch.setattr(guards, "get_compute_capability",
                        lambda: compute_capability)

    def _is_sm_at_least(major, minor=0):
        if compute_capability is None:
            return False
        return compute_capability >= (major, minor)
    monkeypatch.setattr(guards, "is_sm_at_least", _is_sm_at_least)

    monkeypatch.setattr(guards, "_gcn_arch", lambda: gcn_arch)


# ═══════════════════════════════════════════════════════════════════════════
#                  PLATFORM MATRIX TESTS
# ═══════════════════════════════════════════════════════════════════════════

PLATFORMS = [
    ("nvidia_ampere_consumer_a5000", dict(is_cuda=True, compute_capability=(8, 6))),
    ("nvidia_ampere_datacenter_a100", dict(is_cuda=True, compute_capability=(8, 0))),
    ("nvidia_ada_4090", dict(is_cuda=True, compute_capability=(8, 9))),
    ("nvidia_hopper_h100", dict(is_cuda=True, compute_capability=(9, 0))),
    ("nvidia_blackwell_r6000", dict(is_cuda=True, compute_capability=(10, 0))),
    ("amd_rocm_mi210_cdna2", dict(is_rocm=True, gcn_arch="gfx90a")),
    ("amd_rocm_mi300x_cdna3", dict(is_rocm=True, gcn_arch="gfx942")),
    ("intel_xpu_arc", dict(is_xpu=True)),
    ("cpu_only", dict(is_cpu=True)),
]


@pytest.mark.parametrize("platform_name,platform_mocks", PLATFORMS)
class TestPlatformMatrix:
    """Verify Genesis kernels never crash on any platform."""

    def test_guards_never_raise(self, monkeypatch, platform_name, platform_mocks):
        """All guard functions return safely regardless of platform."""
        _mock_platform(monkeypatch, **platform_mocks)

        from vllm._genesis import guards

        # All these should return a bool without raising
        assert isinstance(guards.is_nvidia_cuda(), bool)
        assert isinstance(guards.is_amd_rocm(), bool)
        assert isinstance(guards.is_intel_xpu(), bool)
        assert isinstance(guards.is_cpu_only(), bool)
        assert isinstance(guards.is_cuda_alike(), bool)
        assert isinstance(guards.has_native_fp8(), bool)

    def test_router_softmax_works_on_all_platforms(
        self, monkeypatch, platform_name, platform_mocks
    ):
        """router_softmax doesn't care about platform (pure torch op)."""
        import torch
        _mock_platform(monkeypatch, **platform_mocks)

        from vllm._genesis.kernels.router_softmax import router_softmax

        gating = torch.randn(2, 64, dtype=torch.bfloat16)
        result = router_softmax(gating)

        assert result.shape == gating.shape
        assert result.dtype == gating.dtype

    def test_dequant_buffer_guard_respects_platform(
        self, monkeypatch, platform_name, platform_mocks, reset_genesis_prealloc
    ):
        """TurboQuantBufferManager.should_apply matches platform requirements.

        Should be True only on NVIDIA CUDA with SM >= 8.0.
        """
        _mock_platform(monkeypatch, **platform_mocks)

        from vllm._genesis.kernels.dequant_buffer import TurboQuantBufferManager

        is_cuda = platform_mocks.get("is_cuda", False)
        cc = platform_mocks.get("compute_capability")
        expected = is_cuda and cc is not None and cc >= (8, 0)

        assert TurboQuantBufferManager.should_apply() == expected

    def test_dequant_buffer_returns_none_gracefully(
        self, monkeypatch, platform_name, platform_mocks, reset_genesis_prealloc
    ):
        """On unsupported platforms, returns (None, None) — never crashes."""
        _mock_platform(monkeypatch, **platform_mocks)

        from vllm._genesis.kernels.dequant_buffer import TurboQuantBufferManager

        # Call with reasonable args — should never crash
        k, v = TurboQuantBufferManager.get_or_create_kv_buffers(
            num_kv_heads=2, head_size=128, max_alloc_len=1024,
            device="cpu",  # always usable for test
        )
        import torch
        assert k is None or isinstance(k, torch.Tensor)
        assert v is None or isinstance(v, torch.Tensor)

    def test_gdn_dual_stream_init_doesnt_crash(
        self, monkeypatch, platform_name, platform_mocks
    ):
        """DualStreamDispatcher.init_once() safe on all platforms."""
        _mock_platform(monkeypatch, **platform_mocks)

        from vllm._genesis.kernels.gdn_dual_stream import DualStreamDispatcher

        # Reset state
        DualStreamDispatcher._initialized = False
        DualStreamDispatcher._aux_stream = None

        result = DualStreamDispatcher.init_once()
        assert isinstance(result, bool)

    def test_marlin_tuning_safe_on_all_platforms(
        self, monkeypatch, platform_name, platform_mocks
    ):
        """get_optimal_block_size_m returns None on non-NVIDIA."""
        _mock_platform(monkeypatch, **platform_mocks)
        monkeypatch.delenv("VLLM_MARLIN_MOE_BLOCK_SIZE_M", raising=False)

        from vllm._genesis.kernels.marlin_tuning import get_optimal_block_size_m

        result = get_optimal_block_size_m()

        is_cuda = platform_mocks.get("is_cuda", False)
        if not is_cuda:
            assert result is None, (
                f"Expected None on non-NVIDIA ({platform_name}), got {result}"
            )

    def test_fp8_dispatcher_safe_on_all_platforms(
        self, monkeypatch, platform_name, platform_mocks
    ):
        """FP8 dispatcher returns safe defaults on non-NVIDIA."""
        _mock_platform(monkeypatch, **platform_mocks)

        from vllm._genesis.kernels.fp8_dispatcher import (
            requires_marlin_fp8_fallback,
            fp8_triton_kernel_supported,
        )

        # Both should return False on non-NVIDIA (no FP8 path)
        is_cuda = platform_mocks.get("is_cuda", False)
        if not is_cuda:
            assert requires_marlin_fp8_fallback() is False
            assert fp8_triton_kernel_supported() is False


# ═══════════════════════════════════════════════════════════════════════════
#                    "МЫ ЧИНИМ, НЕ ЛОМАЕМ" invariants
# ═══════════════════════════════════════════════════════════════════════════

class TestNeverBreakInvariants:
    """High-level sanity checks: kernels never crash, always predictable."""

    def test_platform_summary_serializable_everywhere(self):
        """platform_summary() must always return JSON-serializable dict."""
        import json
        from vllm._genesis.guards import platform_summary

        summary = platform_summary()
        json_str = json.dumps(summary, default=str)
        assert len(json_str) > 0

    def test_all_kernels_importable(self):
        """All kernel modules must import without error, on any platform.

        Use importlib.import_module because kernels/__init__.py re-exports
        the `router_softmax` function, which would otherwise shadow the
        `router_softmax` submodule name under `from ... import`.
        """
        import importlib
        modules_and_symbols = [
            ("vllm._genesis.kernels.router_softmax", "router_softmax"),
            ("vllm._genesis.kernels.dequant_buffer", "TurboQuantBufferManager"),
            ("vllm._genesis.kernels.gdn_dual_stream", "DualStreamDispatcher"),
            ("vllm._genesis.kernels.marlin_tuning", "get_optimal_block_size_m"),
            ("vllm._genesis.kernels.fp8_dispatcher", "requires_marlin_fp8_fallback"),
            ("vllm._genesis.kernels.block_table_zero", "zero_block_table_tail"),
            ("vllm._genesis.kernels.tq_decode_tune", "resolve_decode_tune"),
            ("vllm._genesis.kernels.tq_continuation_prefill",
             "continuation_prefill_fp16_rotate"),
        ]
        for mod_name, symbol in modules_and_symbols:
            mod = importlib.import_module(mod_name)
            assert hasattr(mod, symbol), (
                f"Module {mod_name} missing expected symbol {symbol}"
            )

    def test_apply_all_orchestrator_importable(self):
        """patches/apply_all must import on any platform."""
        from vllm._genesis.patches import apply_all
        assert hasattr(apply_all, "run")
        assert hasattr(apply_all, "PATCH_REGISTRY")
