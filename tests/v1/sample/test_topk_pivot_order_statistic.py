# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the order-statistic gather pivot in the combined top-k/top-p
Triton kernel (#61998).

The kernel's zeroth pass used to estimate the gather pivot as a Gaussian
quantile (avg + std * sigma) computed from the first 8192 logits only. On
real logits that pivot is frequently too high, the gather returns fewer than
k candidates, and the row falls back to a full-vocab ternary search. The
pivot is now the m-th largest value of a strided sample taken across the
whole row, which keeps the gathered set small on every distribution tried
here while never under-covering k.

These tests exercise the distributions where the old pivot fails and compare
the Triton mask against an exact PyTorch reference.
"""

import pytest
import torch

from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton

VOCAB = 152576
DEVICES = ["cuda"]


def _reference_topk_mask(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Exact top-k mask: keep the k largest logits (stable tie order)."""
    out = logits.clone()
    order = torch.argsort(-logits, dim=-1, stable=True)
    keep = torch.zeros_like(logits, dtype=torch.bool)
    keep.scatter_(1, order[:, :k], True)
    out[~keep] = float("-inf")
    return out


def _make_rows(kind: str, batch: int, vocab: int, device: str, seed: int = 0):
    g = torch.Generator(device=device).manual_seed(seed)
    if kind == "zipf_sorted":
        # High logits concentrated at low token ids: the head-biased sample
        # statistics of the old pivot are wrong here.
        x = -torch.log1p(torch.arange(vocab, device=device).float())
        return (x + 2.0 * torch.randn(batch, vocab, device=device, generator=g))
    if kind == "two_modes":
        # Bimodal: the inflated std pushes the Gaussian pivot above the row
        # max, so the old gather collects nothing.
        x = torch.randn(batch, vocab, device=device, generator=g)
        return x - 10.0 * (torch.rand(batch, vocab, device=device, generator=g) < 0.2)
    if kind == "uniform":
        return torch.rand(batch, vocab, device=device, generator=g)
    if kind == "grammar_sparse":
        # 99% of the vocab masked to -inf, like a grammar bitmask: the strided
        # sample may contain no finite logits at all.
        x = torch.randn(batch, vocab, device=device, generator=g)
        keep = torch.rand(batch, vocab, device=device, generator=g) > 0.99
        return torch.where(keep, x, torch.full_like(x, float("-inf")))
    raise ValueError(kind)


@pytest.mark.parametrize("kind", ["zipf_sorted", "two_modes", "uniform"])
@pytest.mark.parametrize("k", [50, 200])
def test_topk_pivot_covers_k(kind: str, k: int):
    """The gather must cover k candidates: exactly k tokens survive."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = "cuda"
    x = _make_rows(kind, 16, VOCAB, device)
    kt = torch.full((16,), k, device=device, dtype=torch.int32)
    out = apply_top_k_top_p_triton(x.clone(), kt, None)
    finite = (x > float("-inf")).sum(-1)
    kept = (out > float("-inf")).sum(-1)
    assert (kept == torch.minimum(finite, torch.full_like(finite, k))).all()
    # The kept set must be exactly the k largest values.
    ref = _reference_topk_mask(x, k)
    assert torch.equal(out > float("-inf"), ref > float("-inf"))


def test_topk_pivot_all_masked_sample():
    """Rows where the strided sample sees no finite logit must still produce
    an exact top-k mask via the full-row fallback search."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = "cuda"
    batch = 8
    x = torch.full((batch, VOCAB), float("-inf"), device=device)
    # Only 5 finite tokens per row, at positions unlikely to be sampled.
    g = torch.Generator(device=device).manual_seed(0)
    for i in range(batch):
        idx = torch.randperm(VOCAB, generator=g, device=device)[:5]
        x[i, idx] = torch.randn(5, generator=g, device=device)
    kt = torch.full((batch,), 3, device=device, dtype=torch.int32)
    out = apply_top_k_top_p_triton(x.clone(), kt, None)
    kept = (out > float("-inf")).sum(-1)
    assert (kept == 3).all()
    ref = _reference_topk_mask(x, 3)
    assert torch.equal(out > float("-inf"), ref > float("-inf"))


def test_topk_pivot_grammar_masked_keeps_k():
    """A grammar-style mask (99% masked) with more finite tokens than k."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = "cuda"
    x = _make_rows("grammar_sparse", 8, VOCAB, device)
    kt = torch.full((8,), 200, device=device, dtype=torch.int32)
    out = apply_top_k_top_p_triton(x.clone(), kt, None)
    finite = (x > float("-inf")).sum(-1)
    kept = (out > float("-inf")).sum(-1)
    # k=200 vs ~1% of 152576 = ~1525 finite: some rows may have fewer than
    # k finite logits; those must keep every finite token.
    assert (kept == torch.minimum(finite, torch.full_like(finite, 200))).all()
