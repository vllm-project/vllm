# SPDX-License-Identifier: Apache-2.0
"""TDD tests for vllm._genesis.kernels.gdn_dual_stream.DualStreamDispatcher.

Patch 7 migration target: parallelize in_proj_qkvz + in_proj_ba GEMMs in
GatedDeltaNet via CUDA aux stream, with platform-aware graceful fallback.

Prior art: vllm-project/vllm#39748 (jhsmith409).

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import pytest
import torch


class TestInitOnce:
    """Group 1: init_once() platform gating."""

    def test_init_returns_bool(self):
        from vllm._genesis.kernels.gdn_dual_stream import DualStreamDispatcher
        # Reset state for test
        DualStreamDispatcher._initialized = False
        DualStreamDispatcher._aux_stream = None

        result = DualStreamDispatcher.init_once()
        assert isinstance(result, bool)

    def test_init_idempotent(self):
        """Calling init_once multiple times returns consistent result."""
        from vllm._genesis.kernels.gdn_dual_stream import DualStreamDispatcher
        DualStreamDispatcher._initialized = False
        DualStreamDispatcher._aux_stream = None

        r1 = DualStreamDispatcher.init_once()
        r2 = DualStreamDispatcher.init_once()
        r3 = DualStreamDispatcher.init_once()

        assert r1 == r2 == r3

    def test_init_false_on_cpu_only(self, monkeypatch):
        """CPU-only platform → init returns False."""
        from vllm._genesis.kernels import gdn_dual_stream as gds
        from vllm._genesis import guards

        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: False)
        monkeypatch.setattr(guards, "is_amd_rocm", lambda: False)

        # Reset state
        gds.DualStreamDispatcher._initialized = False
        gds.DualStreamDispatcher._aux_stream = None

        assert gds.DualStreamDispatcher.init_once() is False

    def test_init_false_on_ancient_sm(self, monkeypatch):
        """SM < 8.0 → init returns False (streams weaker on older arches)."""
        from vllm._genesis.kernels import gdn_dual_stream as gds
        from vllm._genesis import guards

        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(guards, "is_amd_rocm", lambda: False)
        monkeypatch.setattr(guards, "is_sm_at_least",
                            lambda major, minor=0: False)

        gds.DualStreamDispatcher._initialized = False
        gds.DualStreamDispatcher._aux_stream = None

        assert gds.DualStreamDispatcher.init_once() is False


class TestMaybeParallel:
    """Group 2: maybe_parallel() executes both fns correctly."""

    def test_sequential_fallback_on_no_aux_stream(self):
        """When _aux_stream is None → sequential execution returns both results."""
        from vllm._genesis.kernels.gdn_dual_stream import DualStreamDispatcher

        # Force sequential path
        DualStreamDispatcher._initialized = True
        DualStreamDispatcher._aux_stream = None

        def fn_a():
            return torch.tensor([1.0, 2.0])

        def fn_b():
            return torch.tensor([3.0, 4.0])

        result_a, result_b = DualStreamDispatcher.maybe_parallel(fn_a, fn_b)

        assert torch.equal(result_a, torch.tensor([1.0, 2.0]))
        assert torch.equal(result_b, torch.tensor([3.0, 4.0]))

    def test_both_functions_called_exactly_once(self):
        from vllm._genesis.kernels.gdn_dual_stream import DualStreamDispatcher

        DualStreamDispatcher._initialized = True
        DualStreamDispatcher._aux_stream = None

        call_count_a = [0]
        call_count_b = [0]

        def fn_a():
            call_count_a[0] += 1
            return 42

        def fn_b():
            call_count_b[0] += 1
            return 100

        DualStreamDispatcher.maybe_parallel(fn_a, fn_b)

        assert call_count_a[0] == 1
        assert call_count_b[0] == 1

    def test_returns_tuple_of_two(self):
        from vllm._genesis.kernels.gdn_dual_stream import DualStreamDispatcher

        DualStreamDispatcher._initialized = True
        DualStreamDispatcher._aux_stream = None

        result = DualStreamDispatcher.maybe_parallel(
            lambda: "a", lambda: "b"
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result == ("a", "b")

    def test_exception_in_fn_a_propagates(self):
        """Exception from fn_a propagates to caller (no silent swallow)."""
        from vllm._genesis.kernels.gdn_dual_stream import DualStreamDispatcher

        DualStreamDispatcher._initialized = True
        DualStreamDispatcher._aux_stream = None

        def bad_fn():
            raise ValueError("deliberate test error")

        with pytest.raises(ValueError, match="deliberate"):
            DualStreamDispatcher.maybe_parallel(bad_fn, lambda: None)


@pytest.mark.cuda_required
class TestCUDAParallelism:
    """Group 3: Real CUDA parallel execution."""

    def test_cuda_aux_stream_created(self, cuda_available):
        if not cuda_available:
            pytest.skip("CUDA not available")

        from vllm._genesis.kernels.gdn_dual_stream import DualStreamDispatcher

        DualStreamDispatcher._initialized = False
        DualStreamDispatcher._aux_stream = None

        ok = DualStreamDispatcher.init_once()
        # On CUDA this should succeed
        if ok:
            assert DualStreamDispatcher._aux_stream is not None
            assert isinstance(
                DualStreamDispatcher._aux_stream, torch.cuda.Stream
            )

    def test_cuda_parallel_executes_both_gemms(self, cuda_available):
        """Real CUDA parallel execution of two GEMMs."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        from vllm._genesis.kernels.gdn_dual_stream import DualStreamDispatcher

        DualStreamDispatcher._initialized = False
        DualStreamDispatcher._aux_stream = None
        DualStreamDispatcher.init_once()

        device = "cuda"
        M, K, N = 64, 128, 256
        w1 = torch.randn(K, N, device=device, dtype=torch.bfloat16)
        w2 = torch.randn(K, N, device=device, dtype=torch.bfloat16)
        x1 = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        x2 = torch.randn(M, K, device=device, dtype=torch.bfloat16)

        y1, y2 = DualStreamDispatcher.maybe_parallel(
            lambda: x1 @ w1,
            lambda: x2 @ w2,
        )

        assert y1.shape == (M, N)
        assert y2.shape == (M, N)
        assert y1.device.type == "cuda"
        assert y2.device.type == "cuda"
