# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark candidate scoring, sampling, and verifier cache correctness."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config.speculative import (
    resolve_markov_bias_topk,
    resolve_markov_topk,
)
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

    anchor = torch.randint(0, vocab, (num_reqs,), generator=gen, dtype=torch.int32).to(
        device
    )
    rows = torch.arange(num_reqs, device=device).unsqueeze(-1)
    steps = torch.arange(1, num_steps + 1, device=device).unsqueeze(0)
    sample_pos = (rows * 1000 + steps).expand(num_reqs, num_steps).reshape(-1)
    idx_mapping = (
        torch.arange(num_reqs, device=device)
        .repeat_interleave(num_steps)
        .to(torch.int32)
    )
    temperature = torch.full(
        (num_reqs,), temperature, dtype=torch.float32, device=device
    )
    seeds = torch.arange(num_reqs, dtype=torch.int64, device=device) * 104729 + 7
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
    embeds = (
        torch.empty(
            (case.num_reqs, case.num_steps, case.rank),
            dtype=case.w1.dtype,
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
        static_biases=static_biases,
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


def _dense_scores_on_chain(case: SimpleNamespace, out: SimpleNamespace) -> torch.Tensor:
    """Score the pruned walk's chain, which can differ from the dense chain."""
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
            is_drafting=True,
        )
    return tokens


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("top_k", [4, 16, 64])
@pytest.mark.parametrize("num_steps,rank", [(3, 8), (8, 192)])
def test_candidate_logits_match_full_vocab_head(dtype, top_k, num_steps, rank):
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
    rtol, atol = (1e-5, 1e-4) if dtype is torch.float32 else (4e-2, 1e-1)
    torch.testing.assert_close(out.realized, expected, rtol=rtol, atol=atol)


@requires_cuda
@pytest.mark.parametrize(
    "num_reqs,num_steps,vocab,rank",
    [(2, 3, 64, 8), (3, 8, 128, 192), (5, 4, 96, 256)],
)
def test_full_candidate_set_matches_dense_reference(num_reqs, num_steps, vocab, rank):
    """With k == V the pruned walk is the original full-vocab Markov head."""
    case = _case(
        num_reqs=num_reqs,
        num_steps=num_steps,
        vocab=vocab,
        rank=rank,
        top_k=vocab,
        seed=2,
    )
    ref_tokens, dense_logits = _dense_reference(case)
    out = _run_walk(case, probabilistic=True)
    torch.testing.assert_close(out.draft_tokens, ref_tokens)
    # Same scores, permuted by the candidate order.
    torch.testing.assert_close(
        out.realized, dense_logits.gather(2, out.cand_ids), rtol=1e-5, atol=1e-4
    )


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
    assert out.draft_tokens.max().item() >= case.draft_vocab


@requires_cuda
def test_probabilistic_walk_matches_dense_gumbel_sampler():
    """Sampling agrees bitwise with the dense Gumbel sampler on the same set."""
    case = _case(
        num_reqs=4,
        num_steps=8,
        vocab=256,
        rank=64,
        top_k=16,
        temperature=0.9,
        seed=4,
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
    torch.testing.assert_close(out.draft_tokens, _dense_gumbel_tokens(case, truncated))

    # The verifier must receive the distribution used by the sampler.
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
        num_reqs=2,
        num_steps=3,
        vocab=200,
        draft_vocab=101,
        rank=16,
        top_k=8,
        temperature=1.0,
        seed=5,
    )
    out = _run_walk(case, probabilistic=True)
    draft_logits = torch.full(
        (case.num_reqs, case.num_steps, case.vocab),
        NEG_INF,
        dtype=torch.float32,
        device=case.device,
    )
    cached_ids = torch.zeros(
        (case.num_reqs, case.num_steps, case.top_k),
        dtype=torch.int64,
        device=case.device,
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
        (case.num_reqs, case.num_steps, case.top_k),
        dtype=torch.int64,
        device=case.device,
    )
    gen = torch.Generator(device="cuda").manual_seed(0)
    for _ in range(2):
        base = torch.randn(
            case.base.shape, dtype=case.base.dtype, device=case.device, generator=gen
        )
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


@requires_cuda
def test_inert_rows_are_neither_sampled_nor_cached():
    """Padded rows (idx_mapping == -1) must not touch the cache."""
    case = _case(
        num_reqs=3, num_steps=2, vocab=40, rank=8, top_k=6, temperature=1.0, seed=7
    )
    case.idx_mapping = torch.tensor(
        [0, 0, -1, -1, 2, 2], dtype=torch.int32, device="cuda"
    )
    out = _run_walk(case, probabilistic=True)
    draft_logits = torch.full(
        (case.num_reqs, case.num_steps, case.vocab),
        NEG_INF,
        dtype=torch.float32,
        device=case.device,
    )
    cached_ids = torch.zeros(
        (case.num_reqs, case.num_steps, case.top_k),
        dtype=torch.int64,
        device=case.device,
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
    assert not walk_is_supported(w1.t().contiguous().t(), w2)
    assert not walk_is_supported(w1, w2[:, :4])  # rank mismatch
    assert not walk_is_supported(w1.to(torch.float8_e4m3fn), w2)


@pytest.mark.parametrize(
    ("markov_topk", "legacy", "hf_markov_topk", "archs", "expected"),
    [
        # Unset resolves to 0 (full-vocab head): candidate pruning is opt-in.
        (None, None, None, ["Qwen3DSparkModel"], 0),
        (16, None, None, ["Qwen3DSparkModel"], 16),
        (0, None, None, ["Qwen3DSparkModel"], 0),
        (32, 8, None, ["Qwen3DSparkModel"], 32),
        (None, 8, None, ["Qwen3DSparkModel"], 8),
        (None, None, 64, ["Qwen3DSparkModel"], 64),
        (None, None, 0, ["Qwen3DSparkModel"], 0),
        (None, None, None, ["Qwen3OmniDSparkModel"], 0),
        (None, None, None, ["Gemma4DSparkModel"], 0),
    ],
)
def test_resolve_markov_topk(markov_topk, legacy, hf_markov_topk, archs, expected):
    """Unset or explicit 0 keeps the full-vocab projection; pruning is opt-in."""
    hf_config = SimpleNamespace(
        **({"markov_topk": hf_markov_topk} if hf_markov_topk is not None else {})
    )
    config = SimpleNamespace(
        markov_topk=markov_topk,
        dspark_draft_topk=legacy,
        draft_model_config=SimpleNamespace(hf_config=hf_config, architectures=archs),
    )
    assert resolve_markov_topk(config) == expected


def _dense_bias_top_table(
    case: SimpleNamespace, m: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference bigram table with fp32 bias values: (top-m ids, top-m values)."""
    rows = case.w1.float() @ case.w2.float().T
    topk = torch.topk(rows, m, dim=-1, largest=case.scale >= 0)
    return topk.indices, topk.values


@requires_cuda
@pytest.mark.parametrize("scale", [1.0, -1.0])
def test_compute_markov_bias_top_ids_matches_dense_projection(scale):
    """Chunked precomputation follows the scaled bias ranking."""
    case = _case(
        num_reqs=2, num_steps=4, vocab=512, rank=64, top_k=8, scale=scale, seed=3
    )
    m = 12
    expected_ids, expected_values = _dense_bias_top_table(case, m)

    ids, bias_values = compute_markov_bias_top_ids(
        case.w1, case.w2, m, case.scale, chunk=97
    )
    assert ids.shape == (case.vocab, m)
    assert ids.dtype == torch.int32
    assert bias_values.shape == (case.vocab, m)
    assert bias_values.dtype == torch.float32
    torch.testing.assert_close(ids, expected_ids.to(torch.int32))
    torch.testing.assert_close(bias_values, expected_values)


@requires_cuda
@pytest.mark.parametrize("num_steps,rank,vocab", [(4, 16, 128), (8, 64, 512)])
def test_union_with_full_bigram_table_equals_dense_reference(num_steps, rank, vocab):
    """A union covering the vocabulary must reproduce the dense chain exactly."""
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

    static_ids, static_biases = _dense_bias_top_table(case, vocab)
    out = _run_walk(
        case, static_ids=static_ids.to(torch.int32), static_biases=static_biases
    )
    assert out.union_k == 8 + vocab
    torch.testing.assert_close(out.draft_tokens, dense_tokens)


@requires_cuda
def test_union_covers_what_logit_only_candidates_miss():
    """A bigram candidate can win even when its base logit is outside top-k."""
    case = _case(num_reqs=1, num_steps=3, vocab=4, rank=1, top_k=1)
    case.base.zero_()
    case.base[..., 0] = 1.0
    case.w1.fill_(2.0)
    case.w2.zero_()
    case.w2[2] = 3.0
    dense_tokens, _ = _dense_reference(case)
    base_only = _run_walk(case)
    assert torch.all(base_only.draft_tokens == 0)
    assert torch.all(dense_tokens == 2)
    static_ids, static_biases = _dense_bias_top_table(case, 1)
    union = _run_walk(
        case, static_ids=static_ids.to(torch.int32), static_biases=static_biases
    )
    torch.testing.assert_close(union.draft_tokens, dense_tokens)


@requires_cuda
def test_union_walk_keeps_candidate_scores_identical_to_dense_head():
    """Both halves of the union are scored exactly as the dense head would."""
    case = _case(
        num_reqs=4, num_steps=5, vocab=256, rank=32, top_k=6, scale=0.75, seed=13
    )
    m = 16
    static_ids, static_biases = _dense_bias_top_table(case, m)
    out = _run_walk(
        case,
        probabilistic=True,
        static_ids=static_ids.to(torch.int32),
        static_biases=static_biases,
    )

    dense_scores = _dense_scores_on_chain(case, out)
    torch.testing.assert_close(out.realized, dense_scores, rtol=2e-2, atol=2e-2)


@requires_cuda
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
    static_ids, static_biases = _dense_bias_top_table(case, 10)
    out = _run_walk(
        case,
        probabilistic=True,
        static_ids=static_ids.to(torch.int32),
        static_biases=static_biases,
    )

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

    # Duplicate ids collapse into one cache column.
    expected = torch.full_like(cache, NEG_INF)
    expected.scatter_(2, out.union_ids, out.realized)
    torch.testing.assert_close(cache, expected)


@pytest.mark.parametrize(
    "markov_topk,markov_bias_topk,hf_bias_topk,expected",
    [
        (16, None, None, 16),  # default bigram budget
        (16, 0, None, 0),  # explicit 0 = logit-only candidates
        (16, 64, 32, 64),  # the knob beats the checkpoint
        (16, None, 32, 32),  # checkpoint value is honored
        (0, 64, None, 0),  # full-vocab head has no candidate set
        (256, None, None, 16),
    ],
)
def test_resolve_markov_bias_topk(
    markov_topk, markov_bias_topk, hf_bias_topk, expected
):
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
