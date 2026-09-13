# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the candidate-pruned (top-k) DSpark Markov head.

The pruned head must produce
  * the same candidate logits as the original full-vocab Markov projection
    (``base + scale * (W1[prev] @ W2^T)`` restricted to the candidates),
  * the same draft tokens as the dense path, greedy and probabilistic,
  * the same truncated draft distribution for the verifier (zero probability
    outside the candidate set),
  * real (target-vocab) token ids for reduced-vocabulary drafts.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.config.speculative import (
    DEFAULT_MARKOV_TOPK,
    resolve_markov_bias_topk,
    resolve_markov_topk,
)
from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.dspark.topk_markov import (
    cache_markov_candidates,
    compute_markov_bias_top_ids,
    markov_walk_topk,
    walk_is_supported,
)

NEG_INF = -float("inf")

requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Triton walk kernel needs CUDA"
)


def _head(w1: torch.Tensor, w2: torch.Tensor) -> DSparkMarkovHead:
    head = DSparkMarkovHead.__new__(DSparkMarkovHead)
    nn.Module.__init__(head)
    head.markov_w1 = nn.Embedding.from_pretrained(w1.cpu().clone())
    head.markov_w2 = nn.Linear(w1.shape[1], w2.shape[0], bias=False, dtype=w2.dtype)
    head.markov_w2.weight.data.copy_(w2.cpu())
    head.markov_w2.tp_size = 1
    return head.cuda()


def _case(
    *,
    num_reqs: int,
    num_steps: int,
    vocab: int,
    rank: int,
    top_k: int,
    dtype: torch.dtype = torch.float32,
    draft_vocab: int | None = None,
    scale: float = 1.0,
    temperature: float = 0.0,
    seed: int = 0,
) -> SimpleNamespace:
    """A synthetic DSpark draft step: base logits + Markov head + bookkeeping."""
    draft_vocab = draft_vocab or vocab
    device = torch.device("cuda")
    gen = torch.Generator().manual_seed(seed)

    def rnd(*shape, scale_=1.0):
        return (torch.randn(shape, generator=gen) * scale_).to(dtype)

    base = rnd(num_reqs, num_steps, draft_vocab, scale_=2.0).to(device)
    # W1 is indexed by the *target* (verifier) vocabulary, W2 by the draft one.
    w1 = rnd(vocab, rank, scale_=rank**-0.5).to(device)
    w2 = rnd(draft_vocab, rank, scale_=rank**-0.5).to(device)

    d2t = None
    if draft_vocab != vocab:
        # Injective draft -> target map, as produced by a pruned draft vocab.
        target_ids = torch.randperm(vocab, generator=gen)[:draft_vocab].sort().values
        d2t = (target_ids - torch.arange(draft_vocab)).to(device)

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
    temperature = torch.full((num_reqs,), temperature, dtype=torch.float32, device=device)
    seeds = (torch.arange(num_reqs, dtype=torch.int64, device=device) * 104729 + 7)
    return SimpleNamespace(
        base=base,
        w1=w1,
        w2=w2,
        d2t=d2t,
        anchor=anchor,
        sample_pos=sample_pos,
        idx_mapping=idx_mapping,
        temperature=temperature,
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


def _dense_reference(case: SimpleNamespace) -> tuple[torch.Tensor, torch.Tensor]:
    """Original full-vocab Markov walk: tokens and the dense per-step logits."""
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
        if case.d2t is not None:
            picked = picked + case.d2t[picked]
        tokens[:, step] = picked
        prev = picked
    return tokens, dense_logits


def _run_walk(
    case: SimpleNamespace,
    *,
    probabilistic: bool = False,
    store_embeds: bool = False,
    base_logits: torch.Tensor | None = None,
    static_ids: torch.Tensor | None = None,
) -> SimpleNamespace:
    """Select the candidates and run the fused walk (as the speculator does)."""
    base_logits = case.base if base_logits is None else base_logits
    cand_values = torch.empty(
        (case.num_reqs, case.num_steps, case.top_k),
        dtype=base_logits.dtype,
        device=case.device,
    )
    cand_ids = torch.empty(
        (case.num_reqs, case.num_steps, case.top_k), dtype=torch.int64, device=case.device
    )
    # Candidate order is irrelevant, mirroring sorted=False at runtime.
    torch.topk(base_logits, case.top_k, dim=-1, sorted=False, out=(cand_values, cand_ids))

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
    embeds = (
        torch.empty(
            (case.num_reqs, case.num_steps, case.rank), dtype=case.w1.dtype,
            device=case.device,
        )
        if store_embeds
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
        base_logits=base_logits if static_ids is not None else None,
        union_ids=union_ids,
        realized_scores=realized,
        markov_embeds=embeds,
        probabilistic=probabilistic,
    )
    return SimpleNamespace(
        cand_values=cand_values,
        cand_ids=cand_ids,
        draft_tokens=draft_tokens,
        realized=realized,
        embeds=embeds,
        union_ids=union_ids,
        union_k=union_k,
        static_m=static_m,
    )


def _dense_scores_on_chain(
    case: SimpleNamespace, out: SimpleNamespace
) -> torch.Tensor:
    """Full-vocab Markov projection evaluated on the walk's own chain.

    Pruning can legitimately pick a different token than the dense argmax once
    ``k < V`` (the bias may promote tokens outside the base top-k), so candidate
    logits have to be compared *given* the previous token the walk used, which
    is exactly the quantity the pruned head is supposed to reproduce.
    """
    assert case.d2t is None, "target and draft vocab coincide in this test"
    prev = case.anchor.to(torch.int64)
    scores = torch.empty_like(out.realized)
    for step in range(case.num_steps):
        bias = (case.w1[prev].float() @ case.w2.float().T) * case.scale
        ids = (
            out.cand_ids[:, step]
            if getattr(out, "union_ids", None) is None
            else out.union_ids[:, step]
        )
        scores[:, step] = (case.base[:, step].float() + bias).gather(1, ids)
        prev = out.draft_tokens[:, step]
    return scores


def _dense_gumbel_tokens(
    case: SimpleNamespace, truncated_logits: torch.Tensor
) -> torch.Tensor:
    """Tokens from the existing dense sampler over the same (truncated) logits."""
    tokens = torch.empty(
        (case.num_reqs, case.num_steps), dtype=torch.int64, device=case.device
    )
    pos = case.sample_pos.view(case.num_reqs, case.num_steps) - 1
    idx_map = case.idx_mapping.view(case.num_reqs, case.num_steps)
    for step in range(case.num_steps):
        tokens[:, step] = gumbel_sample(
            truncated_logits[:, step],
            idx_map[:, step],
            case.temperature,
            case.seeds,
            pos[:, step],
            apply_temperature=True,
        )
    return tokens


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("top_k", [4, 16, 64])
@pytest.mark.parametrize("num_steps,rank", [(3, 8), (8, 192)])
def test_candidate_logits_match_full_vocab_head(
    dtype, top_k, num_steps, rank
):
    """Pruned-head candidate logits == full-vocab projection at those candidates."""
    case = _case(
        num_reqs=5,
        num_steps=num_steps,
        vocab=257,
        rank=rank,
        top_k=top_k,
        dtype=dtype,
        seed=1,
    )
    out = _run_walk(case, probabilistic=True)
    expected = _dense_scores_on_chain(case, out)
    assert out.realized.dtype is torch.float32
    # The walk accumulates in fp32 while the dense head rounds its bias to the
    # head dtype before adding, so bf16 needs the matching tolerance.
    rtol, atol = (1e-5, 1e-4) if dtype is torch.float32 else (4e-2, 1e-1)
    torch.testing.assert_close(out.realized, expected, rtol=rtol, atol=atol)


@requires_cuda
@pytest.mark.parametrize("scale", [1.0, 0.5])
def test_eager_candidate_scores_match_dense_bias(scale):
    """``candidate_scores`` == the dense Markov bias gathered at the candidates."""
    vocab, rank, num_reqs, top_k = 64, 8, 3, 5
    gen = torch.Generator().manual_seed(1)
    w1 = torch.randn((vocab, rank), generator=gen).cuda()
    w2 = torch.randn((vocab, rank), generator=gen).cuda()
    head = _head(w1, w2)
    prev = torch.randint(0, vocab, (num_reqs,), generator=gen).cuda()
    values = torch.randn((num_reqs, top_k), generator=gen).cuda()
    index = torch.randint(0, vocab, (num_reqs, top_k), generator=gen).cuda()

    dense_bias = head.embed(prev).float() @ head.markov_w2.weight.float().T
    expected = values + dense_bias.gather(1, index) * scale
    got = head.candidate_scores(head.embed(prev).float(), values, index, scale)
    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)


@requires_cuda
@pytest.mark.parametrize(
    "num_reqs,num_steps,vocab,rank",
    [(2, 3, 64, 8), (3, 8, 128, 192), (5, 4, 96, 256)],
)
def test_full_candidate_set_matches_dense_reference(num_reqs, num_steps, vocab, rank):
    """With k == V the pruned walk is the original full-vocab Markov head."""
    case = _case(
        num_reqs=num_reqs, num_steps=num_steps, vocab=vocab, rank=rank,
        top_k=vocab, seed=2,
    )
    ref_tokens, dense_logits = _dense_reference(case)
    out = _run_walk(case, probabilistic=True)
    torch.testing.assert_close(out.draft_tokens, ref_tokens)
    # Same scores, permuted by the candidate order.
    torch.testing.assert_close(
        out.realized, dense_logits.gather(2, out.cand_ids), rtol=1e-5, atol=1e-4
    )


@requires_cuda
@pytest.mark.parametrize("top_k", [1, 4, 16])
def test_greedy_walk_is_argmax_within_candidates(top_k):
    """The walk's selection is exactly the argmax over the corrected candidates."""
    case = _case(
        num_reqs=4, num_steps=8, vocab=512, rank=64, top_k=top_k, seed=9
    )
    out = _run_walk(case)
    head = _head(case.w1, case.w2)
    prev = case.anchor.long()
    for step in range(case.num_steps):
        values = case.base[:, step].gather(1, out.cand_ids[:, step])
        scores = head.candidate_scores(head.embed(prev), values, out.cand_ids[:, step])
        chosen = out.cand_ids[:, step].gather(
            1, scores.argmax(dim=-1, keepdim=True)
        ).squeeze(-1)
        torch.testing.assert_close(chosen, out.draft_tokens[:, step])
        prev = chosen


@requires_cuda
def test_reduced_vocab_walk_returns_target_ids():
    """Candidates are draft ids; the walk must return real target token ids."""
    case = _case(
        num_reqs=3, num_steps=4, vocab=96, draft_vocab=37, rank=16, top_k=37, seed=3
    )
    ref_tokens, _ = _dense_reference(case)
    out = _run_walk(case)
    torch.testing.assert_close(out.draft_tokens, ref_tokens)
    assert out.cand_ids.max().item() < case.draft_vocab
    # Sampled ids live in the target vocabulary, i.e. they exceed the draft
    # vocabulary for the tokens the d2t map relocates.
    assert out.draft_tokens.max().item() >= case.draft_vocab
    expected_targets = out.cand_ids + case.d2t[out.cand_ids]
    assert torch.equal(expected_targets.sort().values.unique().max(),
                       case.d2t.max() + case.draft_vocab - 1)


@requires_cuda
def test_probabilistic_walk_matches_dense_gumbel_sampler():
    """Sampling agrees bitwise with the dense Gumbel sampler on the same set."""
    case = _case(
        num_reqs=4, num_steps=8, vocab=256, rank=64, top_k=16,
        temperature=0.9, seed=4,
    )
    case.seeds = torch.arange(4, dtype=torch.int64, device="cuda") * 1234 + 7
    out = _run_walk(case, probabilistic=True)

    # The distribution the walk sampled from: candidates only, -inf elsewhere.
    truncated = torch.full(
        (case.num_reqs, case.num_steps, case.draft_vocab),
        NEG_INF,
        dtype=torch.float32,
        device=case.device,
    )
    truncated.scatter_(2, out.cand_ids, out.realized)
    torch.testing.assert_close(
        out.draft_tokens, _dense_gumbel_tokens(case, truncated)
    )

    # The cache published for the verifier is exactly that distribution, so
    # acceptance testing and full-vocab rejection correction are unchanged.
    draft_logits = torch.full(
        (case.num_reqs, case.num_steps, case.draft_vocab),
        NEG_INF,
        dtype=torch.float32,
        device=case.device,
    )
    cached_ids = torch.zeros(
        (case.num_reqs, case.num_steps, case.top_k), dtype=torch.int64, device=case.device
    )
    cache_markov_candidates(
        draft_logits=draft_logits,
        cached_ids=cached_ids,
        cand_ids=out.cand_ids,
        realized_scores=out.realized,
        sample_idx_mapping=case.idx_mapping,
        d2t=case.d2t,
    )
    torch.testing.assert_close(draft_logits, truncated)
    torch.testing.assert_close(cached_ids, out.cand_ids)


@requires_cuda
def test_probabilistic_walk_reduced_vocab_cache_is_in_target_space():
    """With d2t, cached probabilities sit at the real target ids."""
    case = _case(
        num_reqs=2, num_steps=3, vocab=200, draft_vocab=101, rank=16, top_k=8,
        temperature=1.0, seed=5,
    )
    out = _run_walk(case, probabilistic=True)
    draft_logits = torch.full(
        (case.num_reqs, case.num_steps, case.vocab),
        NEG_INF,
        dtype=torch.float32,
        device=case.device,
    )
    cached_ids = torch.zeros(
        (case.num_reqs, case.num_steps, case.top_k), dtype=torch.int64, device=case.device
    )
    cache_markov_candidates(
        draft_logits=draft_logits,
        cached_ids=cached_ids,
        cand_ids=out.cand_ids,
        realized_scores=out.realized,
        sample_idx_mapping=case.idx_mapping,
        d2t=case.d2t,
    )
    target_ids = out.cand_ids + case.d2t[out.cand_ids]
    torch.testing.assert_close(cached_ids, target_ids)
    expected = torch.full_like(draft_logits, NEG_INF)
    expected.scatter_(2, target_ids, out.realized)
    torch.testing.assert_close(draft_logits, expected)


@requires_cuda
def test_cache_rewrite_leaves_only_current_candidates():
    """Rewriting a column clears the previous candidates (no stale mass)."""
    case = _case(
        num_reqs=2, num_steps=2, vocab=37, rank=8, top_k=5, temperature=1.0, seed=6
    )
    case.seeds = torch.zeros(2, dtype=torch.int64, device=case.device)
    draft_logits = torch.full(
        (case.num_reqs, case.num_steps, case.vocab),
        NEG_INF,
        dtype=torch.float32,
        device=case.device,
    )
    cached_ids = torch.zeros(
        (case.num_reqs, case.num_steps, case.top_k), dtype=torch.int64, device=case.device
    )
    gen = torch.Generator(device="cuda").manual_seed(0)
    for trial in range(3):
        base = torch.randn_like(case.base, generator=gen) * (trial + 1)
        out = _run_walk(case, probabilistic=True, base_logits=base)
        cache_markov_candidates(
            draft_logits=draft_logits,
            cached_ids=cached_ids,
            cand_ids=out.cand_ids,
            realized_scores=out.realized,
            sample_idx_mapping=case.idx_mapping,
            d2t=case.d2t,
        )
        expected = torch.full_like(draft_logits, NEG_INF)
        expected.scatter_(2, out.cand_ids, out.realized)
        torch.testing.assert_close(draft_logits, expected)

        probs = torch.softmax(draft_logits / 0.7, dim=-1)
        outside = torch.ones_like(probs, dtype=torch.bool)
        outside.scatter_(2, out.cand_ids, False)
        assert torch.equal(probs[outside], torch.zeros_like(probs[outside]))
        torch.testing.assert_close(
            probs.sum(dim=-1),
            torch.ones((case.num_reqs, case.num_steps), device=case.device),
            rtol=1e-5,
            atol=1e-6,
        )


@requires_cuda
def test_inert_rows_are_neither_sampled_nor_cached():
    """Padded rows (idx_mapping == -1) must not touch the cache."""
    case = _case(
        num_reqs=3, num_steps=2, vocab=40, rank=8, top_k=6, temperature=1.0, seed=7
    )
    case.idx_mapping = torch.tensor([0, 0, -1, -1, 2, 2], dtype=torch.int32, device="cuda")
    out = _run_walk(case, probabilistic=True)
    draft_logits = torch.full(
        (case.num_reqs, case.num_steps, case.vocab),
        NEG_INF,
        dtype=torch.float32,
        device=case.device,
    )
    cached_ids = torch.zeros(
        (case.num_reqs, case.num_steps, case.top_k), dtype=torch.int64, device=case.device
    )
    cache_markov_candidates(
        draft_logits=draft_logits,
        cached_ids=cached_ids,
        cand_ids=out.cand_ids,
        realized_scores=out.realized,
        sample_idx_mapping=case.idx_mapping,
        d2t=case.d2t,
    )
    assert torch.isneginf(draft_logits[1]).all()
    assert torch.equal(cached_ids[1], torch.zeros_like(cached_ids[1]))
    assert torch.isfinite(draft_logits[[0, 2]]).any()


@requires_cuda
def test_walk_stores_markov_embeds_for_confidence_head():
    """Adaptive verification still receives the chained previous-token embeddings."""
    case = _case(num_reqs=2, num_steps=3, vocab=64, rank=16, top_k=4, seed=8)
    out = _run_walk(case, store_embeds=True)
    expected = torch.stack(
        [
            case.w1[case.anchor.long()],
            case.w1[out.draft_tokens[:, 0]],
            case.w1[out.draft_tokens[:, 1]],
        ],
        dim=1,
    )
    torch.testing.assert_close(out.embeds, expected)


@requires_cuda
def test_walk_is_supported_rejects_sharded_and_odd_weights():
    w1 = torch.randn(64, 8, device="cuda")
    w2 = torch.randn(64, 8, device="cuda")
    assert walk_is_supported(w1, w2)
    assert not walk_is_supported(w1, w2, tp_size=2)
    assert not walk_is_supported(w1.t().contiguous().t(), w2)  # row stride != 1
    assert not walk_is_supported(w1, w2[:, :4])  # rank mismatch
    assert not walk_is_supported(w1.to(torch.float8_e4m3fn), w2)


@pytest.mark.parametrize(
    ("markov_topk", "legacy", "hf_markov_topk", "archs", "expected"),
    [
        (None, None, None, ["Qwen3DSparkModel"], DEFAULT_MARKOV_TOPK),
        (16, None, None, ["Qwen3DSparkModel"], 16),
        (0, None, None, ["Qwen3DSparkModel"], 0),
        (32, 8, None, ["Qwen3DSparkModel"], 32),
        (None, 8, None, ["Qwen3DSparkModel"], 8),
        (None, None, 64, ["Qwen3DSparkModel"], 64),
        (None, None, 0, ["Qwen3DSparkModel"], 0),
        (None, None, None, ["Qwen3OmniDSparkModel"], DEFAULT_MARKOV_TOPK),
        # Architectures without the fused walk keep the full-vocab head.
        (None, None, None, ["Gemma4DSparkModel"], 0),
    ],
)
def test_resolve_markov_topk(markov_topk, legacy, hf_markov_topk, archs, expected):
    """0 falls back to the full-vocab projection; 16 is the default budget."""
    hf_config = SimpleNamespace(
        **({"markov_topk": hf_markov_topk} if hf_markov_topk is not None else {})
    )
    config = SimpleNamespace(
        markov_topk=markov_topk,
        dspark_draft_topk=legacy,
        draft_model_config=SimpleNamespace(hf_config=hf_config, architectures=archs),
    )
    assert resolve_markov_topk(config) == expected


def _dense_bias_top_ids(case: SimpleNamespace, m: int) -> torch.Tensor:
    """Reference bigram table: top-m of the unscaled projection ``W1 @ W2^T``.

    A negative logit scale reverses the ranking, which is what
    :func:`compute_markov_bias_top_ids` reproduces with ``largest``.
    """
    rows = case.w1.float() @ case.w2.float().T
    return torch.topk(rows, m, dim=-1, largest=case.scale >= 0).indices


def test_compute_markov_bias_top_ids_matches_dense_projection(tmp_path):
    """The precomputed table is the dense Markov projection's own top-m."""
    case = _case(
        num_reqs=2, num_steps=4, vocab=512, rank=64, top_k=8, scale=1.0, seed=3
    )
    m = 12
    rows = case.w1.float() @ case.w2.float().T
    expected_values = torch.topk(rows, m, dim=-1).values

    ids = compute_markov_bias_top_ids(
        case.w1, case.w2, m, case.scale, chunk=97, cache_dir=str(tmp_path)
    )
    assert ids.shape == (case.vocab, m)
    assert ids.dtype == torch.int32
    rows_cpu = rows.cpu()
    got_values = torch.gather(rows_cpu, 1, ids.to(torch.int64).cpu())
    torch.testing.assert_close(got_values, expected_values.cpu())
    # Highest-scoring continuation is reproduced exactly (ties cannot move it).
    torch.testing.assert_close(ids[:, 0].cpu(), rows_cpu.argmax(dim=-1).int().cpu())

    # A second call must reuse the cached artifact bit-for-bit.
    again = compute_markov_bias_top_ids(
        case.w1, case.w2, m, case.scale, chunk=97, cache_dir=str(tmp_path)
    )
    assert torch.equal(ids, again)


def test_compute_markov_bias_top_ids_follows_negative_scale(tmp_path):
    """A negative logit scale makes the head *penalize*, so the table flips."""
    case = _case(
        num_reqs=2, num_steps=4, vocab=256, rank=16, top_k=8, scale=-1.0, seed=4
    )
    m = 8
    rows = case.w1.float() @ case.w2.float().T
    ids = compute_markov_bias_top_ids(
        case.w1, case.w2, m, case.scale, chunk=64, cache_dir=str(tmp_path)
    )
    reference = _dense_bias_top_ids(case, m).to(torch.int32)
    torch.testing.assert_close(ids.cpu(), reference.cpu())
    # Sanity: with scale < 0 these are the *smallest* projection rows.
    rows_cpu = rows.cpu()
    torch.testing.assert_close(
        torch.gather(rows_cpu, 1, reference.to(torch.int64).cpu()),
        torch.topk(rows_cpu, m, dim=-1, largest=False).values.cpu(),
    )


@pytest.mark.parametrize("num_steps,rank,vocab", [(4, 16, 128), (8, 64, 512)])
def test_union_with_full_bigram_table_equals_dense_reference(
    num_steps, rank, vocab
):
    """A union covering the vocabulary must reproduce the dense chain exactly.

    Exercises the bigram half end to end: the ``[V, m]`` lookup keyed by the
    chained ``prev``, the ``base_logits`` gather at those ids, the combined
    scoring and the argmax over the union.
    """
    case = _case(
        num_reqs=6,
        num_steps=num_steps,
        vocab=vocab,
        rank=rank,
        top_k=8,
        scale=1.0,
        seed=7,
    )
    dense_tokens, _ = _dense_reference(case)

    out = _run_walk(case, static_ids=_dense_bias_top_ids(case, vocab).to(torch.int32))
    assert out.union_k == 8 + vocab
    torch.testing.assert_close(out.draft_tokens, dense_tokens)


def test_union_covers_what_logit_only_candidates_miss():
    """The production failure mode: base top-k misses the bias-driven winner.

    The backbone fills every draft slot with a mask token, so its logits rank a
    slot in isolation while the chain direction comes from the Markov bias. A
    logit-only candidate set therefore drops the dense head's own choice; adding
    the precomputed bigram top-m recovers it.

    We use a large bias scale so the Markov head can promote tokens well outside
    the base top-k.  If a particular seed does not produce such a mismatch we
    try a few more until we find one that does -- the property is probabilistic
    but overwhelmingly likely for ``scale >> 1``.
    """
    found = False
    for seed in range(11, 50):
        case = _case(
            num_reqs=16,
            num_steps=8,
            vocab=1024,
            rank=64,
            top_k=8,
            scale=4.0,
            seed=seed,
        )
        dense_tokens, _ = _dense_reference(case)
        base_rank = case.base.float().argsort(dim=-1, descending=True).argsort(dim=-1)
        winner_base_rank = base_rank.gather(2, dense_tokens[..., None]).squeeze(-1)
        if int((winner_base_rank >= case.top_k).sum()) > 0:
            found = True
            break
    assert found, "Could not find a seed where the bias pushes a winner outside base top-k"

    base_only = _run_walk(case)
    base_misses = int((base_only.draft_tokens != dense_tokens).sum())
    assert base_misses > 0

    for m in (16, 64):
        static_ids = _dense_bias_top_ids(case, m).to(torch.int32)
        union = _run_walk(case, static_ids=static_ids)
        union_misses = int((union.draft_tokens != dense_tokens).sum())
        assert union_misses <= base_misses
        if m == 64:
            # The bias side dominates here, so a wide bigram table recovers all.
            assert union_misses == 0


def test_union_walk_keeps_candidate_scores_identical_to_dense_head():
    """Both halves of the union are scored exactly as the dense head would."""
    case = _case(
        num_reqs=4, num_steps=5, vocab=256, rank=32, top_k=6, scale=0.75, seed=13
    )
    m = 16
    static_ids = _dense_bias_top_ids(case, m).to(torch.int32)
    out = _run_walk(case, probabilistic=True, static_ids=static_ids)

    dense_scores = _dense_scores_on_chain(case, out)
    # out.realized is indexed by candidate position, not by vocab id.
    # The dense_scores are already gathered at the union ids, so compare
    # directly position-by-position.
    torch.testing.assert_close(out.realized, dense_scores, rtol=2e-2, atol=2e-2)


def test_probabilistic_union_publishes_truncated_distribution():
    """The verifier must read the union-truncated, renormalized draft law."""
    case = _case(
        num_reqs=4,
        num_steps=5,
        vocab=128,
        rank=16,
        top_k=6,
        scale=1.0,
        temperature=0.9,
        seed=17,
    )
    static_ids = _dense_bias_top_ids(case, 10).to(torch.int32)
    out = _run_walk(case, probabilistic=True, static_ids=static_ids)

    cache = torch.full(
        (case.num_reqs, case.num_steps, case.draft_vocab),
        float("-inf"),
        dtype=torch.float32,
        device=case.device,
    )
    cached_ids = torch.zeros_like(out.union_ids)
    cache_markov_candidates(
        draft_logits=cache,
        cached_ids=cached_ids,
        cand_ids=out.union_ids,
        realized_scores=out.realized,
        sample_idx_mapping=case.idx_mapping,
        d2t=case.d2t,
    )

    target_ids = out.union_ids
    if case.d2t is not None:
        target_ids = target_ids + case.d2t[target_ids]
    # Every published entry equals the pre-temperature score the walk used.
    torch.testing.assert_close(
        cache.gather(2, target_ids), out.realized, rtol=1e-5, atol=1e-5
    )
    # Nothing outside the candidate union carries probability mass.
    finite = torch.isfinite(cache).sum(dim=-1)
    assert int(finite.max()) <= out.union_k
    # A token can land in both halves of the union. The cache is keyed by token
    # id, so duplicates collapse into one column; the drafter's Gumbel-max noise
    # is keyed by token id too, so its duplicate columns draw the *same* noise
    # and behave as a single outcome. Both sides therefore see the deduplicated
    # law, which renormalizes to 1 over the union and to 0 outside it.
    dedup = torch.full_like(cache, float("-inf"))
    dedup.scatter_(2, target_ids, out.realized)
    torch.testing.assert_close(cache, dedup)

    published = torch.softmax(cache / 0.9, dim=-1)
    expected = torch.softmax(dedup / 0.9, dim=-1)
    torch.testing.assert_close(published, expected, rtol=1e-6, atol=1e-6)
    # The deduplicated cache (which is what the verifier actually reads)
    # normalizes to 1 over its finite entries.
    torch.testing.assert_close(
        published.sum(dim=-1), torch.ones_like(published.sum(dim=-1)), rtol=1e-5, atol=1e-5
    )
    # Every non-candidate column holds exactly zero probability.
    outside = published.clone()
    outside.scatter_(2, target_ids, 0.0)
    assert float(outside.sum()) == 0.0


@pytest.mark.parametrize(
    "markov_topk,markov_bias_topk,hf_bias_topk,expected",
    [
        (16, None, None, 16),       # default mirrors the base budget
        (16, 0, None, 0),           # explicit 0 = logit-only candidates
        (16, 64, 32, 64),           # the knob beats the checkpoint
        (16, None, 32, 32),         # checkpoint value is honored
        (0, 64, None, 0),           # full-vocab head has no candidate set
        (256, None, None, 16),
    ],
)
def test_resolve_markov_bias_topk(markov_topk, markov_bias_topk, hf_bias_topk, expected):
    hf_config = SimpleNamespace(architectures=["Qwen3DSparkModel"])
    if hf_bias_topk is not None:
        hf_config.markov_bias_topk = hf_bias_topk
    draft_model_config = SimpleNamespace(
        hf_config=hf_config, architectures=["Qwen3DSparkModel"], model=None
    )
    config = SimpleNamespace(
        method="dspark",
        markov_topk=markov_topk,
        markov_bias_topk=markov_bias_topk,
        dspark_draft_topk=None,
        draft_model_config=draft_model_config,
    )
    assert resolve_markov_bias_topk(config) == expected
