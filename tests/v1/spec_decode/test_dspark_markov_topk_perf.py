# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Performance benchmarks for the DSpark Markov topk optimizations.

Each test isolates one optimization and compares the NEW (optimized) path
against the OLD (baseline) path, asserting positive impact.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode.dspark.topk_markov import (
    cache_markov_candidates,
    compute_markov_bias_top_ids,
    markov_walk_topk,
)

NEG_INF = -float("inf")

requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Triton kernels need CUDA"
)

_WARMUP = 3
_REPEATS = 10


def _cuda_time(fn, *, warmup=_WARMUP, repeats=_REPEATS) -> float:
    """Median GPU time in ms for *fn*, measured with CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times_ms: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
    times_ms.sort()
    return times_ms[len(times_ms) // 2]


# ---------------------------------------------------------------------------
# Shared synthetic data builder (mirrors _case in the correctness test file)
# ---------------------------------------------------------------------------


def _case(
    *,
    num_reqs: int,
    num_steps: int,
    vocab: int,
    rank: int,
    top_k: int,
    dtype: torch.dtype = torch.bfloat16,
    scale: float = 1.0,
    temperature: float = 0.0,
    seed: int = 0,
) -> SimpleNamespace:
    draft_vocab = vocab
    device = torch.device("cuda")
    gen = torch.Generator().manual_seed(seed)

    def rnd(*shape, scale_=1.0):
        return (torch.randn(shape, generator=gen) * scale_).to(dtype)

    base = rnd(num_reqs, num_steps, draft_vocab, scale_=2.0).to(device)
    w1 = rnd(vocab, rank, scale_=rank**-0.5).to(device)
    w2 = rnd(draft_vocab, rank, scale_=rank**-0.5).to(device)

    anchor = torch.randint(
        0, vocab, (num_reqs,), generator=gen, dtype=torch.int32
    ).to(device)
    rows = torch.arange(num_reqs, device=device).unsqueeze(-1)
    steps = torch.arange(1, num_steps + 1, device=device).unsqueeze(0)
    sample_pos = (rows * 1000 + steps).expand(num_reqs, num_steps).reshape(-1)
    idx_mapping = (
        torch.arange(num_reqs, device=device)
        .repeat_interleave(num_steps)
        .to(torch.int32)
    )
    temperature_t = torch.full(
        (num_reqs,), temperature, dtype=torch.float32, device=device
    )
    seeds = torch.arange(num_reqs, dtype=torch.int64, device=device) * 104729 + 7
    return SimpleNamespace(
        base=base,
        w1=w1,
        w2=w2,
        d2t=None,
        anchor=anchor,
        sample_pos=sample_pos,
        idx_mapping=idx_mapping,
        temperature=temperature_t,
        seeds=seeds,
        num_reqs=num_reqs,
        num_steps=num_steps,
        vocab=vocab,
        draft_vocab=draft_vocab,
        rank=rank,
        top_k=top_k,
        scale=scale,
        device=device,
    )


def _run_walk(
    case: SimpleNamespace,
    *,
    probabilistic: bool = False,
    base_logits: torch.Tensor | None = None,
    static_ids: torch.Tensor | None = None,
    static_biases: torch.Tensor | None = None,
) -> SimpleNamespace:
    """Select the candidates and run the fused walk (as the speculator does)."""
    base_logits = case.base if base_logits is None else base_logits
    cand_values = torch.empty(
        (case.num_reqs, case.num_steps, case.top_k),
        dtype=base_logits.dtype,
        device=case.device,
    )
    cand_ids = torch.empty(
        (case.num_reqs, case.num_steps, case.top_k),
        dtype=torch.int64,
        device=case.device,
    )
    torch.topk(
        base_logits, case.top_k, dim=-1, sorted=False, out=(cand_values, cand_ids)
    )

    draft_tokens = torch.zeros(
        (case.num_reqs, case.num_steps), dtype=torch.int64, device=case.device
    )
    static_m = 0 if static_ids is None else int(static_ids.shape[1])
    union_k = case.top_k + static_m
    realized = (
        torch.zeros(
            (case.num_reqs, case.num_steps, union_k),
            dtype=torch.float32,
            device=case.device,
        )
        if probabilistic
        else None
    )
    union_ids = (
        torch.empty(
            (case.num_reqs, case.num_steps, union_k),
            dtype=torch.int64,
            device=case.device,
        )
        if static_ids is not None
        else None
    )
    markov_walk_topk(
        num_reqs=case.num_reqs,
        cand_values=cand_values,
        cand_ids=cand_ids,
        w1=case.w1,
        w2=case.w2,
        scale=case.scale,
        draft_tokens=draft_tokens,
        input_ids=case.anchor,
        anchor_indices=torch.arange(
            case.num_reqs, dtype=torch.int64, device=case.device
        ),
        sample_pos=case.sample_pos,
        sample_idx_mapping=case.idx_mapping,
        temperature=case.temperature,
        seeds=case.seeds,
        d2t=case.d2t,
        static_ids=static_ids,
        static_biases=static_biases,
        base_logits=base_logits if static_ids is not None else None,
        union_ids=union_ids,
        realized_scores=realized,
        probabilistic=probabilistic,
    )
    return SimpleNamespace(
        cand_values=cand_values,
        cand_ids=cand_ids,
        draft_tokens=draft_tokens,
        realized=realized,
        union_ids=union_ids,
        union_k=union_k,
        static_m=static_m,
    )


def _dense_reference(case: SimpleNamespace) -> tuple[torch.Tensor, torch.Tensor]:
    """Full-vocab Markov walk: tokens and the dense per-step logits."""
    prev = case.anchor.to(torch.int64)
    tokens = torch.empty(
        (case.num_reqs, case.num_steps), dtype=torch.int64, device=case.device
    )
    dense_logits = torch.empty(
        (case.num_reqs, case.num_steps, case.draft_vocab),
        dtype=torch.float32,
        device=case.device,
    )
    for step in range(case.num_steps):
        bias = (case.w1[prev].float() @ case.w2.float().T) * case.scale
        logits = case.base[:, step].float() + bias
        dense_logits[:, step] = logits
        picked = logits.argmax(dim=-1)
        tokens[:, step] = picked
        prev = picked
    return tokens, dense_logits


def _old_python_loop(case: SimpleNamespace) -> torch.Tensor:
    """Reconstructed OLD _sample_sequential_topk: Python for-loop with gather +
    baddbmm + scatter + argmax per step (greedy). Mirrors the code on main."""
    prev = case.anchor.to(torch.int64)
    draft_tokens = torch.zeros(
        (case.num_reqs, case.num_steps), dtype=torch.int64, device=case.device
    )
    # Topk selection (same as the old code).
    base_values, base_indices = case.base.topk(case.top_k, dim=-1)
    for step in range(case.num_steps):
        embed = case.w1[prev].float()  # [num_reqs, rank]
        weight = case.w2[base_indices[:, step]].float()  # [num_reqs, k, rank]
        bias = (weight @ embed.unsqueeze(-1)).squeeze(-1)  # [num_reqs, k]
        scores = base_values[:, step].float() + bias * case.scale
        picked = scores.argmax(dim=-1)
        draft_tokens[:, step] = picked
        prev = picked
    return draft_tokens


# ===========================================================================
# Opt A: Fused walk kernel vs old Python loop (the biggest single change)
# ===========================================================================


@requires_cuda
@pytest.mark.parametrize("top_k", [16, 32])
def test_opt_a_fused_walk_faster_than_python_loop(top_k):
    """A single fused Triton kernel must beat the per-step Python loop that
    calls embed + gather + baddbmm + scatter + argmax for each draft step."""
    case = _case(
        num_reqs=32,
        num_steps=8,
        vocab=152064,
        rank=256,
        top_k=top_k,
        seed=100,
    )
    fused_ms = _cuda_time(lambda: _run_walk(case))
    loop_ms = _cuda_time(lambda: _old_python_loop(case))
    print(
        f"\n[opt-a topk={top_k}]  fused walk: {fused_ms:.2f} ms, "
        f"python loop: {loop_ms:.2f} ms  (loop/fused = {loop_ms / fused_ms:.1f}x)"
    )
    torch.cuda.empty_cache()
    assert fused_ms < loop_ms


# ===========================================================================
# Opt B: Bigram union (markov_bias_topk) improves acceptance and is not slow
# ===========================================================================


@requires_cuda
def test_opt_b_union_walk_recovers_bias_driven_winners():
    """Base-logit-only candidates miss what the Markov head predicts; the
    precomputed bigram top-m union recovers those winners."""
    m = 16
    case = _case(
        num_reqs=32,
        num_steps=8,
        vocab=1024,
        rank=64,
        top_k=8,
        scale=4.0,
        seed=101,
    )
    dense_tokens, _ = _dense_reference(case)

    base_only = _run_walk(case)
    static_ids, static_biases = compute_markov_bias_top_ids(
        case.w1, case.w2, m, case.scale
    )
    union = _run_walk(case, static_ids=static_ids, static_biases=static_biases)

    base_misses = int((base_only.draft_tokens != dense_tokens).sum())
    union_misses = int((union.draft_tokens != dense_tokens).sum())
    print(
        f"\n[opt-b]  base-only misses: {base_misses}, "
        f"union misses: {union_misses}  (recovered {base_misses - union_misses})"
    )
    assert union_misses <= base_misses


@requires_cuda
def test_opt_b_union_walk_latency():
    """Union walk with precomputed bigram biases should not be catastrophically
    slower than base-only at the same total candidate count."""
    m = 16
    case_union = _case(
        num_reqs=32,
        num_steps=8,
        vocab=152064,
        rank=256,
        top_k=16,
        seed=102,
    )
    case_base = _case(
        num_reqs=32,
        num_steps=8,
        vocab=152064,
        rank=256,
        top_k=32,  # same total k = 16 base + 16 static
        seed=102,
    )
    static_ids, static_biases = compute_markov_bias_top_ids(
        case_union.w1, case_union.w2, m, case_union.scale
    )
    union_ms = _cuda_time(
        lambda: _run_walk(
            case_union, static_ids=static_ids, static_biases=static_biases
        )
    )
    base_ms = _cuda_time(lambda: _run_walk(case_base))
    print(
        f"\n[opt-b latency]  union (16+16): {union_ms:.2f} ms, "
        f"base-only (32): {base_ms:.2f} ms  (union/base = {union_ms / base_ms:.2f}x)"
    )
    torch.cuda.empty_cache()
    assert union_ms <= base_ms * 1.5




# ===========================================================================
# Opt C: flashinfer top_k vs torch.topk on the base-logit candidate selection
# ===========================================================================


@requires_cuda
@pytest.mark.parametrize("batch", [1, 4, 32, 128])
def test_opt_c_flashinfer_topk_faster_than_torch_topk(batch):
    """flashinfer's radix top_k should beat torch.topk for the large-vocab
    candidate-selection step in _sample_sequential_topk."""
    from flashinfer import top_k as flashinfer_topk

    n_spec = 8
    vocab = 152064
    k = 32
    device = torch.device("cuda")
    dtype = torch.bfloat16

    base_logits = torch.randn(batch, n_spec, vocab, dtype=dtype, device=device)
    cand_values = torch.empty(batch, n_spec, k, dtype=dtype, device=device)
    cand_ids = torch.empty(batch, n_spec, k, dtype=torch.int64, device=device)

    def torch_topk():
        torch.topk(base_logits, k, dim=-1, sorted=False,
                   out=(cand_values, cand_ids))

    def fi_topk():
        flat = base_logits.view(-1, vocab)
        fi_v, fi_i = flashinfer_topk(flat, k)
        cand_values.copy_(fi_v.view(batch, n_spec, -1))
        cand_ids.copy_(fi_i.view(batch, n_spec, -1))

    torch_ms = _cuda_time(torch_topk, repeats=50, warmup=20)
    fi_ms = _cuda_time(fi_topk, repeats=50, warmup=20)
    print(
        f"\n[opt-c batch={batch}]  torch.topk: {torch_ms:.3f} ms, "
        f"flashinfer: {fi_ms:.3f} ms  (torch/fi = {torch_ms / fi_ms:.1f}x)"
    )
    torch.cuda.empty_cache()
    assert fi_ms < torch_ms


# ===========================================================================
# Opt D: O(k) cache kernel vs full-vocab cache write
# ===========================================================================


@requires_cuda
@pytest.mark.parametrize("top_k", [16, 32])
def test_opt_d_cache_kernel_faster_than_dense_write(top_k):
    """The O(k) cache kernel must be faster than writing the full-vocab
    draft-logit cache (what the non-topk probabilistic path does)."""
    case = _case(
        num_reqs=32,
        num_steps=8,
        vocab=152064,
        rank=256,
        top_k=top_k,
        temperature=1.0,
        seed=104,
    )
    out = _run_walk(case, probabilistic=True)

    draft_logits = torch.full(
        (case.num_reqs, case.num_steps, case.draft_vocab),
        NEG_INF,
        dtype=torch.float32,
        device=case.device,
    )
    cached_ids = torch.zeros(
        (case.num_reqs, case.num_steps, case.top_k),
        dtype=torch.int64,
        device=case.device,
    )

    def cache_fn():
        cache_markov_candidates(
            draft_logits=draft_logits,
            cached_ids=cached_ids,
            cand_ids=out.cand_ids,
            realized_scores=out.realized,
            sample_idx_mapping=case.idx_mapping,
            d2t=case.d2t,
        )

    cache_ms = _cuda_time(cache_fn)

    # Dense baseline: what the non-topk probabilistic path must do — write the
    # full [num_reqs, num_steps, V] draft_logits tensor.
    _, dense_logits = _dense_reference(case)

    def dense_fn(ref=dense_logits):
        fresh = torch.full(
            (case.num_reqs, case.num_steps, case.draft_vocab),
            NEG_INF,
            dtype=torch.float32,
            device=case.device,
        )
        fresh.copy_(ref)

    dense_ms = _cuda_time(dense_fn)
    print(
        f"\n[opt-d topk={top_k}]  cache kernel: {cache_ms:.2f} ms, "
        f"dense write: {dense_ms:.2f} ms  (dense/cache = {dense_ms / cache_ms:.1f}x)"
    )
    del dense_logits
    torch.cuda.empty_cache()
    assert cache_ms < dense_ms


# ===========================================================================
# Opt E: Full pipeline — topk draft pipeline vs dense
# ===========================================================================


@requires_cuda
@pytest.mark.parametrize("top_k", [16, 32])
def test_opt_e_topk_draft_pipeline_faster_than_dense(top_k):
    """End-to-end: the topk draft pipeline (topk + walk + cache) must beat
    the dense pipeline (full-vocab Markov loop + full-vocab cache write)."""
    case = _case(
        num_reqs=32,
        num_steps=8,
        vocab=152064,
        rank=256,
        top_k=top_k,
        temperature=1.0,
        seed=105,
    )

    # --- Dense pipeline: base_logits → Python Markov loop → copy full cache ---
    def dense_pipeline():
        _, dense_logits = _dense_reference(case)
        _ = torch.full(
            (case.num_reqs, case.num_steps, case.draft_vocab),
            NEG_INF,
            dtype=torch.float32,
            device=case.device,
        ).copy_(dense_logits)
        del dense_logits

    # --- Topk pipeline: topk → fused walk → cache kernel ---
    draft_logits = torch.full(
        (case.num_reqs, case.num_steps, case.draft_vocab),
        NEG_INF,
        dtype=torch.float32,
        device=case.device,
    )
    cached_ids = torch.zeros(
        (case.num_reqs, case.num_steps, case.top_k),
        dtype=torch.int64,
        device=case.device,
    )

    def topk_pipeline():
        out = _run_walk(case, probabilistic=True)
        cache_markov_candidates(
            draft_logits=draft_logits,
            cached_ids=cached_ids,
            cand_ids=out.cand_ids,
            realized_scores=out.realized,
            sample_idx_mapping=case.idx_mapping,
            d2t=case.d2t,
        )

    dense_ms = _cuda_time(dense_pipeline)
    topk_ms = _cuda_time(topk_pipeline)
    print(
        f"\n[opt-e topk={top_k}]  dense pipeline: {dense_ms:.2f} ms, "
        f"topk pipeline: {topk_ms:.2f} ms  (dense/topk = {dense_ms / topk_ms:.1f}x)"
    )
    torch.cuda.empty_cache()
    assert topk_ms < dense_ms