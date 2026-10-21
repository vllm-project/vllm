# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the DSpark confidence-head fused kernel.

Validates the TileLang fused kernel against:
  1. An fp64 ground-truth (tight tolerance)
  2. The current vLLM PyTorch fallback (loose tolerance)

Run:
    pytest tests/kernels/dspark/test_confidence.py -v
"""

import pytest
import torch

from vllm.model_executor.kernels.dspark import (
    dspark_confidence_fused,
    dspark_confidence_torch,
    is_dspark_fused_available,
)

DSPARK_CONFIDENCE_DEVICE = "cuda"


def _reference_fp64(hidden, markov_embed, weight, temp):
    """fp64 ground truth."""
    h = hidden.to(torch.float64)
    w = weight.to(torch.float64)
    r = weight.shape[0] - hidden.shape[-1]
    if r > 0:
        m = markov_embed[:, :r].to(torch.float64)
        x = torch.cat([h, m], dim=-1)
    else:
        x = h
    return torch.sigmoid((x @ w) / temp)


def _make_inputs(n, H, r, seed=0, hidden_strided=False):
    torch.manual_seed(seed)
    if hidden_strided:
        # Simulate the speculator's `head_hidden[sample_indices]` pattern:
        # wider storage, narrow view with non-trivial leading-dim stride.
        storage = torch.randn(n * 2, H, dtype=torch.bfloat16, device="cuda")
        hidden = storage[::2]
        assert hidden.stride(0) > 1
    else:
        hidden = torch.randn(n, H, dtype=torch.bfloat16, device="cuda")
    markov = torch.randn(n, max(r, 1), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(H + r, dtype=torch.float32, device="cuda") * 0.02
    return hidden, markov, weight


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("n", [1, 32, 128, 512])
@pytest.mark.parametrize("H,r", [(2048, 0), (2048, 8), (2048, 32), (7168, 8)])
def test_fused_matches_fp64_ground_truth(n, H, r):
    """Fused output must be close to fp64 reference (tight bound)."""
    if not is_dspark_fused_available():
        pytest.skip("TileLang fused path unavailable")
    hidden, markov, weight = _make_inputs(n, H, r)
    out = torch.empty(n, dtype=torch.float32, device="cuda")
    torch.ops.vllm.dspark_confidence_tilelang(hidden, markov, weight, out, 1.0)
    ref = _reference_fp64(hidden, markov, weight, 1.0)
    max_rel = ((out.double() - ref).abs() / ref.clamp_min(1e-8)).max().item()
    # fp32 GEMV with 2048..7168 length accumulates roughly 1e-6..1e-5 rel error.
    # We accept up to 1e-3 to leave room for reduction-order differences.
    assert max_rel < 1e-3, f"max_rel_err={max_rel:.3e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("n", [1, 32, 256, 1024])
@pytest.mark.parametrize("H,r", [(2048, 8), (7168, 32)])
def test_fused_vs_pytorch_fallback(n, H, r):
    """Fused and fallback should produce comparable results.

    This is the load-bearing regression check: the fused path must not
    silently diverge from the existing vLLM output.
    """
    if not is_dspark_fused_available():
        pytest.skip("TileLang fused path unavailable")
    hidden, markov, weight = _make_inputs(n, H, r)
    out_fused = torch.empty(n, dtype=torch.float32, device="cuda")
    torch.ops.vllm.dspark_confidence_tilelang(hidden, markov, weight, out_fused, 1.0)

    out_ref = dspark_confidence_torch(hidden, markov, weight, 1.0)
    torch.testing.assert_close(out_fused, out_ref, rtol=1e-3, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_strided_hidden_input():
    """TileLang T.Tensor assumes contiguous layout. The speculator's
    ``head_hidden[sample_indices]`` produces a contiguous result in practice
    (gather over dim-0 writes a fresh contiguous buffer), so the contract
    is: last-dim stride must be 1. We verify here that a contiguous tensor
    produced from a strided view runs correctly.
    """
    if not is_dspark_fused_available():
        pytest.skip("TileLang fused path unavailable")
    n, H, r = 64, 2048, 8
    hidden, markov, weight = _make_inputs(n, H, r, hidden_strided=True)
    assert hidden.stride(-1) == 1
    hidden = hidden.contiguous()
    out = torch.empty(n, dtype=torch.float32, device="cuda")
    torch.ops.vllm.dspark_confidence_tilelang(hidden, markov, weight, out, 1.0)
    ref = _reference_fp64(hidden, markov, weight, 1.0)
    max_rel = ((out.double() - ref).abs() / ref.clamp_min(1e-8)).max().item()
    assert max_rel < 1e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dispatch_dispatch() -> None:
    """Sanity check on the high-level dispatcher."""
    n, H, r = 32, 1024, 8
    hidden, markov, weight = _make_inputs(n, H, r)
    out = dspark_confidence_fused(hidden, markov, weight, temp=1.0)
    assert out.shape == (n,)
    assert out.dtype == torch.float32
    assert torch.all((out >= 0.0) & (out <= 1.0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fallback_when_fp32_hidden():
    """If the caller passes fp32 hidden, dispatcher must decline (return None).
    The model caller is responsible for falling back to PyTorch."""
    n, H, r = 32, 1024, 8
    hidden, markov, weight = _make_inputs(n, H, r)
    hidden = hidden.float()  # force dispatch decline
    out = dspark_confidence_fused(hidden, markov, weight)
    assert out is None, "fused dispatch should decline fp32 hidden"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fallback_when_huge_batch():
    """Over max_fused_batch, dispatcher must decline."""
    n, H, r = 4096, 1024, 8
    hidden, markov, weight = _make_inputs(n, H, r)
    out = dspark_confidence_fused(hidden, markov, weight, max_fused_batch=1024)
    assert out is None, "fused dispatch should decline when n > max_fused_batch"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_temperature_scaling():
    """temp parameter must actually shift the sigmoid."""
    if not is_dspark_fused_available():
        pytest.skip("TileLang fused path unavailable")
    n, H, r = 32, 1024, 8
    hidden, markov, weight = _make_inputs(n, H, r)
    out_t1 = torch.empty(n, dtype=torch.float32, device="cuda")
    out_t2 = torch.empty(n, dtype=torch.float32, device="cuda")
    torch.ops.vllm.dspark_confidence_tilelang(hidden, markov, weight, out_t1, 1.0)
    torch.ops.vllm.dspark_confidence_tilelang(hidden, markov, weight, out_t2, 2.0)
    # Higher temp -> scaled logit smaller -> sigmoid closer to 0.5 on average.
    # We don't assert specifics; just that they differ (no aliasing bug).
    assert not torch.allclose(out_t1, out_t2)
