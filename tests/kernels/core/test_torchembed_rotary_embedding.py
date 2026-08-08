# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the torchembed optional RoPE backend."""

from importlib.util import find_spec

import pytest
import torch

from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.platforms import current_platform
from vllm.utils.torchembed import is_torchembed_available

_has_torchembed = find_spec("torchembed") is not None
_has_triton = find_spec("triton") is not None


# ── utility ──────────────────────────────────────────────────────────────


def test_is_torchembed_available():
    """The utility correctly reflects whether the package is installed."""
    assert is_torchembed_available() is (_has_torchembed and _has_triton)


# ── flag behaviour (no CUDA needed) ─────────────────────────────────────


class TestUseTorchEmbedFlag:

    def test_false_when_package_missing(self):
        """Flag is False when torchembed is not available."""
        rope = RotaryEmbedding(128, 128, 4096, 10000.0, True, torch.float32)
        assert rope.use_torchembed is False

    def test_false_when_not_neox(self):
        """Flag is False for GPT-J style RoPE."""
        rope = RotaryEmbedding(128, 128, 4096, 10000.0, False, torch.float32)
        assert rope.use_torchembed is False

    def test_false_with_nonstandard_cache(self):
        """Flag is False when cos/sin cache shape differs from standard."""
        rope = RotaryEmbedding(128, 64, 4096, 10000.0, True, torch.float32)
        assert rope.use_torchembed is False


# ── CUDA-based tests ────────────────────────────────────────────────────


@pytest.mark.skipif(not current_platform.is_cuda_alike(),
                    reason="CUDA / ROCm required")
class TestForwardCUDABehaviour:

    def test_fallback_to_default_kernel(self):
        """forward_cuda works via the default C++ kernel when torchembed
        is not installed."""
        rope = RotaryEmbedding(
            128, 128, 4096, 10000.0, True, torch.bfloat16,
        ).to("cuda")
        n, nq, nkv = 7, 32, 8
        pos = torch.randint(0, 1024, (n,), device="cuda")
        q = torch.randn(n, nq * 128, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(n, nkv * 128, dtype=torch.bfloat16, device="cuda")

        q_out, k_out = rope.forward_cuda(pos, q, k)

        assert q_out.shape == q.shape
        assert k_out.shape == k.shape
        assert not torch.isnan(q_out).any()
        assert not torch.isnan(k_out).any()

    def test_fallback_with_key_none(self):
        """forward_cuda works when key is None (cross-layer KV sharing)."""
        rope = RotaryEmbedding(
            128, 128, 4096, 10000.0, True, torch.bfloat16,
        ).to("cuda")
        n, nq = 7, 32
        pos = torch.randint(0, 1024, (n,), device="cuda")
        q = torch.randn(n, nq * 128, dtype=torch.bfloat16, device="cuda")

        q_out, k_out = rope.forward_cuda(pos, q, None)

        assert q_out.shape == q.shape
        assert k_out is None

    def test_native_matches_cuda(self):
        """forward_native and forward_cuda produce the same result when
        torchembed is not available (both delegate to the same logic)."""
        torch.manual_seed(42)
        rope = RotaryEmbedding(
            128, 128, 4096, 10000.0, True, torch.bfloat16,
        ).to("cuda")
        n, nq, nkv = 11, 32, 8
        pos = torch.randint(0, 1024, (n,), device="cuda")
        q = torch.randn(n, nq * 128, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(n, nkv * 128, dtype=torch.bfloat16, device="cuda")

        q_nat, k_nat = rope.forward_native(pos, q.clone(), k.clone())
        q_cud, k_cud = rope.forward_cuda(pos, q.clone(), k.clone())

        torch.testing.assert_close(q_nat, q_cud, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(k_nat, k_cud, atol=1e-2, rtol=1e-2)

    @pytest.mark.skipif(not _has_torchembed or not _has_triton,
                        reason="torchembed + triton required")
    def test_use_torchembed_flag_true_when_available(self):
        """Flag is True when torchembed and triton are installed."""
        rope = RotaryEmbedding(
            128, 128, 4096, 10000.0, True, torch.float32,
        ).to("cuda")
        assert rope.use_torchembed is True


# ── numerical correctness (torchembed installed) ────────────────────────


@pytest.mark.skipif(
    not current_platform.is_cuda_alike() or not _has_torchembed or not _has_triton,
    reason="CUDA + torchembed + triton required",
)
class TestTorchEmbedNumerics:

    ROPE_CONFIGS = [
        pytest.param(32, 32, id="dim32"),
        pytest.param(64, 64, id="dim64"),
        pytest.param(128, 128, id="dim128"),
    ]

    @pytest.mark.parametrize("head_size,rotary_dim", ROPE_CONFIGS)
    def test_matches_native(self, head_size, rotary_dim):
        """torchembed forward_cuda matches forward_native."""
        torch.manual_seed(42)
        rope = RotaryEmbedding(
            head_size, rotary_dim, 4096, 10000.0, True, torch.bfloat16,
        ).to("cuda")
        n, nq, nkv = 13, 16, 4
        pos = torch.randint(0, 1024, (n,), device="cuda")
        q = torch.randn(n, nq * head_size, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(n, nkv * head_size, dtype=torch.bfloat16, device="cuda")

        q_nat, k_nat = rope.forward_native(pos, q.clone(), k.clone())
        q_opt, k_opt = rope.forward_cuda(pos, q.clone(), k.clone())

        torch.testing.assert_close(q_nat, q_opt, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(k_nat, k_opt, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("head_size,rotary_dim", ROPE_CONFIGS)
    def test_gradient_flow(self, head_size, rotary_dim):
        """Gradients flow correctly through the torchembed path."""
        rope = RotaryEmbedding(
            head_size, rotary_dim, 4096, 10000.0, True, torch.float32,
        ).to("cuda")
        n, nq = 5, 8
        pos = torch.randint(0, 1024, (n,), device="cuda")

        q = torch.randn(n, nq * head_size, dtype=torch.float32,
                        device="cuda", requires_grad=True)
        k = torch.randn(n, nq * head_size, dtype=torch.float32,
                        device="cuda", requires_grad=True)

        q_out, k_out = rope.forward_cuda(pos, q, k)
        loss = (q_out**2).sum() + (k_out**2).sum()
        loss.backward()

        assert q.grad is not None
        assert k.grad is not None
        assert not torch.isnan(q.grad).any()
        assert not torch.isnan(k.grad).any()

    @pytest.mark.parametrize("head_size,rotary_dim", ROPE_CONFIGS)
    def test_key_can_be_none(self, head_size, rotary_dim):
        """forward_cuda handles key=None gracefully."""
        rope = RotaryEmbedding(
            head_size, rotary_dim, 4096, 10000.0, True, torch.bfloat16,
        ).to("cuda")
        n, nq = 7, 16
        pos = torch.randint(0, 1024, (n,), device="cuda")
        q = torch.randn(n, nq * head_size, dtype=torch.bfloat16, device="cuda")

        q_out, k_out = rope.forward_cuda(pos, q, None)

        assert q_out.shape == q.shape
        assert k_out is None

    @pytest.mark.parametrize("head_size,rotary_dim", ROPE_CONFIGS)
    def test_partial_rotary_dim(self, head_size, rotary_dim):
        """RoPE applied only to rotary_dim < head_size leaves pass-through
        dimensions untouched."""
        rope = RotaryEmbedding(
            head_size, rotary_dim, 4096, 10000.0, True, torch.bfloat16,
        ).to("cuda")
        n, nq, nkv = 11, 8, 4
        pos = torch.randint(0, 1024, (n,), device="cuda")
        q = torch.randn(n, nq * head_size, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(n, nkv * head_size, dtype=torch.bfloat16, device="cuda")

        q_out, k_out = rope.forward_cuda(pos, q.clone(), k.clone())
        q_ref, k_ref = rope.forward_native(pos, q.clone(), k.clone())

        torch.testing.assert_close(q_out, q_ref, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(k_out, k_ref, atol=1e-2, rtol=1e-2)
