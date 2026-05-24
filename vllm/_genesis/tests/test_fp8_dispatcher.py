# SPDX-License-Identifier: Apache-2.0
"""TDD tests for vllm._genesis.kernels.fp8_dispatcher.

Patches 1 + 2 migration: FP8 kernel routing decisions based on SM version.
SM < 8.9 (Ampere) requires Marlin fallback (Triton FP8 unsupported).
SM >= 8.9 (Ada/Hopper/Blackwell) use native Triton FP8.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations



class TestRequiresMarlinFp8Fallback:
    """Group 1: SM-based fallback decision."""

    def test_false_on_non_nvidia(self, monkeypatch):
        from vllm._genesis.kernels import fp8_dispatcher as fd
        from vllm._genesis import guards

        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: False)
        assert fd.requires_marlin_fp8_fallback() is False

    def test_true_on_ampere_datacenter(self, monkeypatch):
        """SM 8.0 (A100) → Marlin fallback required."""
        from vllm._genesis.kernels import fp8_dispatcher as fd
        from vllm._genesis import guards

        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(guards, "is_sm_at_least",
                            lambda major, minor=0: False)
        assert fd.requires_marlin_fp8_fallback() is True

    def test_true_on_ampere_consumer(self, monkeypatch):
        """SM 8.6 (A5000) → Marlin fallback required (primary Genesis target)."""
        from vllm._genesis.kernels import fp8_dispatcher as fd
        from vllm._genesis import guards

        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)

        def is_sm_at_least(major, minor=0):
            return (major, minor) <= (8, 6)

        monkeypatch.setattr(guards, "is_sm_at_least", is_sm_at_least)
        assert fd.requires_marlin_fp8_fallback() is True

    def test_false_on_ada(self, monkeypatch):
        """SM 8.9 (Ada/4090) → native FP8, no fallback."""
        from vllm._genesis.kernels import fp8_dispatcher as fd
        from vllm._genesis import guards

        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(guards, "is_sm_at_least",
                            lambda major, minor=0: True)
        assert fd.requires_marlin_fp8_fallback() is False

    def test_false_on_hopper(self, monkeypatch):
        """SM 9.0 (H100) → native FP8."""
        from vllm._genesis.kernels import fp8_dispatcher as fd
        from vllm._genesis import guards

        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(guards, "is_sm_at_least",
                            lambda major, minor=0: True)
        assert fd.requires_marlin_fp8_fallback() is False

    def test_false_on_blackwell(self, monkeypatch):
        """SM 10.0 (B100/R6000 Blackwell) → native FP8."""
        from vllm._genesis.kernels import fp8_dispatcher as fd
        from vllm._genesis import guards

        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(guards, "is_sm_at_least",
                            lambda major, minor=0: True)
        assert fd.requires_marlin_fp8_fallback() is False


class TestFp8TritonKernelSupported:
    """Group 2: Native FP8 capability check."""

    def test_false_on_non_nvidia(self, monkeypatch):
        from vllm._genesis.kernels import fp8_dispatcher as fd
        from vllm._genesis import guards

        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: False)
        assert fd.fp8_triton_kernel_supported() is False

    def test_false_on_ampere(self, monkeypatch):
        from vllm._genesis.kernels import fp8_dispatcher as fd
        from vllm._genesis import guards

        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(guards, "is_sm_at_least",
                            lambda major, minor=0: False)
        assert fd.fp8_triton_kernel_supported() is False

    def test_true_on_ada_plus(self, monkeypatch):
        from vllm._genesis.kernels import fp8_dispatcher as fd
        from vllm._genesis import guards

        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(guards, "is_sm_at_least",
                            lambda major, minor=0: True)
        assert fd.fp8_triton_kernel_supported() is True

    def test_inverse_relationship(self, monkeypatch):
        """requires_marlin_fp8_fallback() == not fp8_triton_kernel_supported()
        on NVIDIA.
        """
        from vllm._genesis.kernels import fp8_dispatcher as fd
        from vllm._genesis import guards

        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)

        # Test both SM <8.9 and SM >=8.9
        for sm_supported in [False, True]:
            monkeypatch.setattr(
                guards, "is_sm_at_least",
                lambda major, minor=0, s=sm_supported: s
            )
            assert (
                fd.requires_marlin_fp8_fallback()
                == (not fd.fp8_triton_kernel_supported())
            )


class TestShouldSkipTritonFp8:
    """Group 3: Explicit CC-based skip."""

    def test_skip_on_ampere_consumer(self):
        from vllm._genesis.kernels.fp8_dispatcher import should_skip_triton_fp8
        # Explicit CC argument
        assert should_skip_triton_fp8((8, 6)) is True

    def test_skip_on_ampere_datacenter(self):
        from vllm._genesis.kernels.fp8_dispatcher import should_skip_triton_fp8
        assert should_skip_triton_fp8((8, 0)) is True

    def test_no_skip_on_ada(self):
        from vllm._genesis.kernels.fp8_dispatcher import should_skip_triton_fp8
        assert should_skip_triton_fp8((8, 9)) is False

    def test_no_skip_on_hopper(self):
        from vllm._genesis.kernels.fp8_dispatcher import should_skip_triton_fp8
        assert should_skip_triton_fp8((9, 0)) is False

    def test_none_cc_returns_false(self, monkeypatch):
        """None (non-NVIDIA) → no skip (different path).

        Must monkey-patch `get_compute_capability` because passing None
        makes the function fall through to the live platform query, which
        on real CUDA hardware (e.g. inside the integration container) would
        return a real SM tuple and flip the decision. The test's intent is
        "when the platform reports None (non-NVIDIA), return False".
        """
        from vllm._genesis.kernels import fp8_dispatcher as fd
        from vllm._genesis import guards
        monkeypatch.setattr(guards, "get_compute_capability", lambda: None)
        assert fd.should_skip_triton_fp8(None) is False

    def test_uses_platform_when_no_arg(self, monkeypatch):
        """When no explicit CC, queries current platform."""
        from vllm._genesis.kernels import fp8_dispatcher as fd
        from vllm._genesis import guards

        monkeypatch.setattr(guards, "get_compute_capability", lambda: (8, 6))
        assert fd.should_skip_triton_fp8() is True

        monkeypatch.setattr(guards, "get_compute_capability", lambda: (9, 0))
        assert fd.should_skip_triton_fp8() is False


class TestLogDispatcherDecision:
    """Group 4: Observability."""

    def test_log_does_not_raise_on_non_nvidia(self, caplog, monkeypatch):
        from vllm._genesis.kernels.fp8_dispatcher import log_dispatcher_decision
        from vllm._genesis import guards

        monkeypatch.setattr(guards, "get_compute_capability", lambda: None)
        log_dispatcher_decision()  # should not raise

    def test_log_does_not_raise_on_nvidia(self, caplog, monkeypatch):
        from vllm._genesis.kernels.fp8_dispatcher import log_dispatcher_decision
        from vllm._genesis import guards

        monkeypatch.setattr(guards, "get_compute_capability", lambda: (8, 6))
        log_dispatcher_decision()  # should not raise
