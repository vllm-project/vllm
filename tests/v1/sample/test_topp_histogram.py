# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the histogram-bracket top-p pipeline used for large p-only batches."""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("CUDA-only kernel", allow_module_level=True)

from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton  # noqa: E402
from vllm.v1.sample.ops.topp_histogram_triton import (  # noqa: E402
    MIN_BATCH_SIZE,
    apply_top_p_histogram,
)

DEVICE = "cuda"


def _within_rounding(
    logits: torch.Tensor, p: torch.Tensor, keep: torch.Tensor, tol: float = 1e-6
) -> torch.Tensor:
    """Per row: ``keep`` is a prefix of the (logit desc, index asc) order and
    differs from the float64 reference only by tokens whose cumulative mass is
    within ``tol`` of p, where fp32 sums can flip the comparison with p."""
    x = logits.double()
    order = torch.argsort(-x, dim=-1, stable=True)
    cum = torch.softmax(x, dim=-1).gather(1, order).cumsum(-1)
    kept = keep.gather(1, order)
    n = kept.sum(-1, keepdim=True)
    ranks = torch.arange(x.shape[1], device=x.device)
    prefix = (kept == (ranks < n)).all(-1)
    pd = p.double().unsqueeze(1)
    n_ref = ((cum < pd).sum(-1, keepdim=True) + 1).clamp(max=x.shape[1])
    lo, hi = torch.minimum(n, n_ref), torch.maximum(n, n_ref)
    reached = cum.gather(1, (lo - 1).clamp(min=0)) >= pd - tol
    not_past = torch.where(hi >= 2, cum.gather(1, (hi - 2).clamp(min=0)), 0.0)
    return prefix & (reached & (not_past < pd + tol)).squeeze(1)


def _reference_keep(logits: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Exact float64 top-p: the smallest prefix by (logit desc, index asc)
    whose probability mass reaches p, plus the row max(es)."""
    batch, vocab = logits.shape
    x = logits.double()
    probs = torch.softmax(x, dim=-1)
    # Stable descending order on value, ascending index on ties.
    order = torch.argsort(-x, dim=-1, stable=True)
    sorted_probs = probs.gather(1, order)
    cum = sorted_probs.cumsum(-1)
    # Count of tokens needed: first position where cum >= p (inclusive).
    need = (cum < p.double().unsqueeze(1)).sum(-1) + 1
    need = need.clamp(max=vocab)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(vocab, device=logits.device).expand(batch, -1))
    keep = rank < need.unsqueeze(1)
    keep |= x == x.max(-1, keepdim=True).values
    keep &= torch.isfinite(x)
    keep[p >= 1.0] = torch.isfinite(x[p >= 1.0])
    return keep


def _logits(batch: int, vocab: int, regime: str, seed: int) -> torch.Tensor:
    g = torch.Generator(device=DEVICE)
    g.manual_seed(seed)
    if regime == "flat":
        return torch.randn(batch, vocab, device=DEVICE, generator=g)
    if regime in ("bf16", "fp16"):
        # LM-like: a few hot tokens over a broad tail, bf16 or fp16 values
        # upcast to fp32 (the sampler's copy of a half-precision head's
        # logits), so the boundary region holds exactly tied tokens.
        x = torch.randn(batch, vocab, device=DEVICE, generator=g) * 2.5 - 3.0
        hot = torch.randint(0, vocab, (batch, 16), device=DEVICE, generator=g)
        x.scatter_(1, hot, torch.rand(batch, 16, device=DEVICE, generator=g) * 6 + 8)
        half = torch.bfloat16 if regime == "bf16" else torch.float16
        return x.to(half).float().contiguous()
    if regime == "wide":
        # max - min beyond the coarse range cap: a far tail shares the last bin.
        x = torch.randn(batch, vocab, device=DEVICE, generator=g) * 3.0
        x[:, ::7] -= 60.0
        return x.contiguous()
    if regime == "shifted":
        # Large logits: fp32 spacing near the max is coarser than a fine bin.
        x = torch.randn(batch, vocab, device=DEVICE, generator=g) * 2.0 + 300.0
        hot = torch.randint(0, vocab, (batch, 8), device=DEVICE, generator=g)
        x.scatter_(1, hot, torch.rand(batch, 8, device=DEVICE, generator=g) * 10 + 304)
        return x.contiguous()
    x = torch.randn(batch, vocab, device=DEVICE, generator=g) * 2.0
    n_hot = 8 if regime == "peaked" else 64
    hot = torch.randint(0, vocab, (batch, n_hot), device=DEVICE, generator=g)
    x.scatter_(1, hot, torch.rand(batch, n_hot, device=DEVICE, generator=g) * 10 + 4)
    if regime == "ties":
        # Quantize so many tokens share a value, including at the boundary.
        x = (x * 4).round() / 4
    if regime == "masked":
        drop = torch.rand(batch, vocab, device=DEVICE, generator=g) < 0.3
        x = x.masked_fill(drop, float("-inf"))
    return x.contiguous()


def _check(
    logits: torch.Tensor,
    p: torch.Tensor,
    out: torch.Tensor,
    mask_value: float = float("-inf"),
):
    ref_keep = _reference_keep(logits, p)
    keep = (out != mask_value) & torch.isfinite(out)
    # Kept tokens keep their value; the rest are masked.
    assert torch.equal(out[keep], logits[keep])
    probs = torch.softmax(logits.double(), dim=-1)
    mass = (probs * keep).sum(-1)
    active = p < 1.0
    # The kept mass reaches p (up to fp32 summation rounding) ...
    assert bool((mass[active] >= p[active].double() - 1e-5).all())
    # ... and the set matches the float64 reference exactly, except for
    # boundary tokens whose cumulative mass is within fp32 rounding of p.
    diff = (keep != ref_keep).sum(-1)
    ok = (diff == 0) | _within_rounding(logits, p, keep)
    assert bool(ok[active].all()), diff[active & ~ok]
    # Rows with p >= 1 are untouched.
    assert torch.equal(out[~active], logits[~active])


@pytest.mark.parametrize("batch", [MIN_BATCH_SIZE, 64, 257])
@pytest.mark.parametrize("vocab", [1024, 32000, 200064])
@pytest.mark.parametrize(
    "regime",
    ["peaked", "mixed", "flat", "ties", "masked", "bf16", "fp16", "wide", "shifted"],
)
def test_top_p_histogram_matches_reference(batch: int, vocab: int, regime: str):
    logits = _logits(batch, vocab, regime, seed=batch * 7 + vocab)
    g = torch.Generator(device=DEVICE)
    g.manual_seed(1)
    p = torch.rand(batch, device=DEVICE, generator=g) * 0.98 + 0.01
    p[::5] = 0.95
    p[1::5] = 0.99
    p[2::5] = 0.995
    p[4::10] = 0.999
    p[3::11] = 1.0
    out = apply_top_p_histogram(logits.clone(), p)
    _check(logits, p, out)


def test_top_p_histogram_edge_rows():
    batch, vocab = MIN_BATCH_SIZE, 4096
    logits = torch.randn(batch, vocab, device=DEVICE)
    logits[0] = 1.0  # uniform row: keep everything
    logits[1] = float("-inf")
    logits[1, 17] = 3.0  # a single finite logit
    logits[2, :] = float("-inf")  # nothing finite: untouched
    logits[3, 5] = 1e4  # dominant token, tiny p
    p = torch.full((batch,), 0.9, device=DEVICE)
    p[3] = 1e-9
    out = apply_top_p_histogram(logits.clone(), p)
    assert torch.equal(out[0], logits[0])
    assert torch.equal(out[1], logits[1])
    assert torch.equal(out[2], logits[2])
    assert int(torch.isfinite(out[3]).sum()) == 1 and torch.isfinite(out[3, 5])
    _check(logits[4:], p[4:], out[4:])


def test_top_p_histogram_padded_stride_and_mask_value():
    batch, vocab, mask_value = 64, 32000, -1e9
    logits = _logits(batch, vocab, "bf16", seed=5)
    p = torch.full((batch,), 0.99, device=DEVICE)
    padded = torch.randn(batch, vocab + 13, device=DEVICE)
    padded[:, :vocab] = logits
    tail = padded[:, vocab:].clone()
    apply_top_p_histogram(padded[:, :vocab], p, mask_value)
    _check(logits, p, padded[:, :vocab], mask_value)
    assert torch.equal(padded[:, vocab:], tail)


def test_top_p_histogram_signed_zero_ties():
    batch, vocab = MIN_BATCH_SIZE, 4096
    logits = torch.full((batch, vocab), float("-inf"), device=DEVICE)
    logits[:, 0] = 2.0
    logits[:, 10:210:2] = -0.0
    logits[:, 11:210:2] = 0.0
    # The boundary is the 101st zero in index order.
    e2 = torch.tensor(2.0).exp().item()
    p = torch.full((batch,), (e2 + 100.5) / (e2 + 200), device=DEVICE)
    out = apply_top_p_histogram(logits.clone(), p)
    _check(logits, p, out)


@pytest.mark.parametrize("distinct", [False, True])
def test_top_p_histogram_gather_overflow(distinct: bool):
    """More bracket tokens than the gather scratch holds: exact on ties, and
    a superset within the coarse bracket past the distinct-value budget."""
    batch, vocab = MIN_BATCH_SIZE, 65536
    g = torch.Generator(device=DEVICE)
    g.manual_seed(0)
    logits = torch.randn(batch, vocab, device=DEVICE, generator=g) - 8.0
    logits[:, 0] = 4.0
    cluster = torch.zeros(vocab // 2, device=DEVICE)
    if distinct:
        cluster = torch.linspace(0.0, 1e-3, vocab // 2, device=DEVICE).flip(0)
    logits[:, 1::2] = cluster
    p = torch.full((batch,), 0.5, device=DEVICE)
    out = apply_top_p_histogram(logits.clone(), p)
    if not distinct:
        _check(logits, p, out)
        return
    keep = torch.isfinite(out)
    ref_keep = _reference_keep(logits, p)
    assert bool((keep | ~ref_keep).all())
    boundary = logits.masked_fill(~ref_keep, float("inf")).min(-1).values
    extra = logits.masked_fill(~(keep & ~ref_keep), float("inf")).min(-1).values
    assert bool((extra > boundary - 0.05).all())


def test_top_p_histogram_dispatch_and_cuda_graph():
    batch, vocab = 64, 200064
    logits = _logits(batch, vocab, "mixed", seed=3)
    p = torch.full((batch,), 0.95, device=DEVICE)
    expected = apply_top_p_histogram(logits.clone(), p)
    _check(logits, p, expected)
    # apply_top_k_top_p_triton routes p-only batches >= MIN_BATCH_SIZE here.
    routed = apply_top_k_top_p_triton(logits.clone(), None, p)
    assert torch.equal(torch.isfinite(routed), torch.isfinite(expected))
    # Deterministic and capturable: replaying a graph reproduces the eager mask.
    buf = logits.clone()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        apply_top_p_histogram(buf, p)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    buf.copy_(logits)
    with torch.cuda.graph(graph, stream=stream):
        apply_top_p_histogram(buf, p)
    buf.copy_(logits)
    graph.replay()
    torch.accelerator.synchronize()
    _check(logits, p, buf)
