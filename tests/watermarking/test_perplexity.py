# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import numpy as np
import pytest
import torch
from scipy import optimize

from vllm.platforms import current_platform
from vllm.v1.watermarking import PPLHardWatermarker, PPLSoftWatermarker
from vllm.v1.watermarking.binomial import (
    _least_probable_top_token,
    _level_best,
    binomial_scores,
    binomial_thresholds,
)
from vllm.v1.watermarking.perplexity import (
    _MONTE_CARLO_KEY,
    binomial_level_probs,
    convex_hull_algorithm_vectorized,
    first_score_ranks,
    hull_candidates,
)
from vllm.v1.watermarking.watermarker import RandomSampler


def _random_log_probs(num_rows: int, vocab_size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    logits = 3 * torch.randn(num_rows, vocab_size, generator=generator)
    logits[torch.rand(num_rows, vocab_size, generator=generator) < 0.3] = -math.inf
    logits[:, 0] = 0.0
    return logits.double().log_softmax(dim=-1)


def test_level_best_keeps_the_most_probable_token_per_score():
    log_probs = torch.tensor([[-1.0, -2.0, -0.5, -math.inf, -3.0, -0.5]]).log_softmax(
        -1
    )
    scores = torch.tensor([[2, 0, 2, 1, 0, 2]])

    level_log_probs, level_token_ids = _level_best(log_probs, scores, 4)

    # Level 1 holds only a filtered token and level 3 none at all.
    assert level_log_probs[0].tolist() == [
        log_probs[0, 1].item(),
        -math.inf,
        log_probs[0, 2].item(),
        -math.inf,
    ]
    # Ties go to the lowest token id; levels without tokens report token 0.
    assert level_token_ids[0].tolist() == [1, 0, 2, 0]


def test_least_probable_top_token():
    log_probs = torch.tensor([[-1.0, -2.0, -0.5, -math.inf, -3.0, -3.0]]).log_softmax(
        -1
    )
    scores = torch.tensor([[2, 0, 2, 3, 2, 2]])

    least, token_ids = _least_probable_top_token(log_probs, scores)

    # Score 3 holds only a filtered token; of the score-2 tokens, 4 and 5 are
    # the least probable and the lower id wins.
    assert least.tolist() == [log_probs[0, 4].item()]
    assert token_ids.tolist() == [4]


def _budget(log_probs: torch.Tensor, epsilon: float) -> torch.Tensor:
    probs_safe = log_probs.exp().clamp(min=1e-30)
    return -(probs_safe * probs_safe.log()).sum(dim=1) + epsilon


@pytest.mark.parametrize("epsilon", [0.0, 0.5, 3.0])
def test_hard_ppl_solves_the_linear_program(epsilon: float):
    watermarker = PPLHardWatermarker(key=42, context_width=2, epsilon=epsilon)
    log_probs = _random_log_probs(24, 60, seed=0).float()
    contexts = torch.randint(0, 60, (24, 2), generator=torch.Generator().manual_seed(1))

    watermarked = watermarker.convex_hull_algorithm(log_probs, contexts).double()

    scores = watermarker.scores(contexts, 60)
    budget = _budget(log_probs, epsilon)
    for row in range(24):
        support = log_probs[row].isfinite().numpy()
        log_p = log_probs[row].double().numpy()[support]
        g = scores[row].numpy()[support].astype(np.float64)
        q = watermarked[row].numpy()
        program = optimize.linprog(
            -g,
            A_ub=[-log_p],
            b_ub=[budget[row].item()],
            A_eq=[np.ones_like(g)],
            b_eq=[1.0],
            method="highs",
        )
        assert program.status == 0
        assert q.sum() == pytest.approx(1.0)
        assert q[~support].sum() == 0
        assert (q > 0).sum() <= 2
        assert -(q[support] * log_p).sum() <= budget[row].item() + 1e-5
        assert (q[support] * g).sum() == pytest.approx(-program.fun, abs=1e-5)


@pytest.mark.parametrize("epsilon", [0.0, 1.0, 5.0])
def test_hard_ppl_spends_the_budget_on_tied_top_scores(epsilon: float):
    """Tokens 1 and 3 share the top score; like the reference implementation,
    the leftover budget goes to the less probable one."""
    log_probs = torch.tensor([[0.6, 0.3, 0.07, 0.03]], dtype=torch.float64).log()
    scores = torch.tensor([[10, 20, 12, 20]])
    budget = _budget(log_probs, epsilon).item()
    cost = (-log_probs[0]).tolist()

    probs, candidate_scores, token_ids = hull_candidates(
        *_level_best(log_probs, scores, 31),
        *_least_probable_top_token(log_probs, scores),
    )
    best_q = convex_hull_algorithm_vectorized(
        probs, candidate_scores, epsilon, _budget(log_probs, 0.0)
    )
    q = torch.zeros(1, 4, dtype=torch.float64).scatter_add_(1, token_ids, best_q)[0]

    if budget < cost[1]:
        # Below token 1's surprisal: mix tokens 0 and 1 to spend the budget.
        weight = (budget - cost[0]) / (cost[1] - cost[0])
        expected = [1 - weight, weight, 0.0, 0.0]
    elif budget < cost[3]:
        weight = (budget - cost[1]) / (cost[3] - cost[1])
        expected = [0.0, 1 - weight, 0.0, weight]
    else:
        expected = [0.0, 0.0, 0.0, 1.0]
    torch.testing.assert_close(q, torch.tensor(expected, dtype=torch.float64))


def _selected(nodes: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """The log-probability each draw selects, for one ``beta`` per row."""
    levels = torch.arange(nodes.shape[-1], dtype=nodes.dtype)
    keys = torch.where(
        nodes.isfinite(), levels + beta[:, None, None] * nodes, -math.inf
    )
    return nodes.gather(-1, keys.argmax(dim=-1, keepdim=True)).squeeze(-1)


def _mean_selected(nodes: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    return _selected(nodes, beta).mean(-1)


def test_first_score_ranks_match_explicit_draws():
    """Drawn directly, the ranks where the scores first appear have their
    distribution under i.i.d. scores, so the Monte Carlo estimate is the same."""
    num_draws, width = 20_000, 64
    G = binomial_scores(
        torch.rand(num_draws, width, generator=torch.Generator().manual_seed(0)),
        binomial_thresholds(30),
    )
    explicit = torch.full((num_draws, 31), width).scatter_reduce_(
        -1, G, torch.arange(width).expand_as(G), "amin"
    )
    uniforms = torch.rand(num_draws, 62, generator=torch.Generator().manual_seed(1))
    direct = first_score_ranks(uniforms, binomial_level_probs(30), width)

    def assert_same_mean(a: torch.Tensor, b: torch.Tensor) -> None:
        a, b = a.double(), b.double()
        se = ((a.var(0) + b.var(0)) / num_draws).sqrt()
        assert ((a.mean(0) - b.mean(0)).abs() <= 5 * se + 1e-9).all()

    # Each score's rank, and whether it appears among the first width tokens.
    assert_same_mean(explicit, direct)
    assert_same_mean(explicit < width, direct < width)
    # The quantity the bisection estimates: the log-probability selected.
    ranked_logp = torch.from_numpy(
        np.sort(np.log(np.random.default_rng(2).dirichlet(np.full(width, 0.3))))[
            ::-1
        ].copy()
    )
    nodes = [
        ranked_logp[ranks.clamp(max=width - 1)]
        .masked_fill(ranks >= width, -math.inf)
        .unsqueeze(0)
        for ranks in (explicit, direct)
    ]
    for beta in (0.1, 0.5, 3.0):
        beta = torch.tensor([beta], dtype=torch.float64)
        assert_same_mean(_selected(nodes[0], beta)[0], _selected(nodes[1], beta)[0])


def test_monte_carlo_draws_are_new_for_every_new_context():
    """A context repeats its draws, whatever the watermark key."""
    contexts = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 3, 4]])

    uniforms = PPLSoftWatermarker(key=42).monte_carlo_uniforms(contexts)

    assert uniforms.shape == (3, 128, 62)
    assert torch.equal(uniforms[0], uniforms[2])
    assert not torch.equal(uniforms[0], uniforms[1])
    assert torch.equal(
        PPLSoftWatermarker(key=7).monte_carlo_uniforms(contexts), uniforms
    )


def _nodes_and_target(
    watermarker: PPLSoftWatermarker, log_probs: torch.Tensor, contexts: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The bisection inputs for ``contexts``, for vocabularies of at most 4096
    tokens."""
    support = log_probs.isfinite()
    p_safe = log_probs.exp().clamp_min(1e-12)
    logp = torch.where(support, p_safe.log(), -math.inf)
    vocab_size = log_probs.shape[-1]
    uniforms = watermarker.monte_carlo_uniforms(contexts)
    ranks = first_score_ranks(uniforms, watermarker.level_probs, vocab_size)
    ranked_logp = logp.sort(dim=-1, descending=True).values
    nodes = ranked_logp.gather(1, ranks.clamp(max=vocab_size - 1).flatten(1))
    nodes = nodes.view(ranks.shape).masked_fill(ranks >= vocab_size, -math.inf)
    target = torch.where(support, p_safe * logp, 0.0).sum(dim=-1)
    return nodes, target


def test_find_beta_crosses_the_target():
    log_probs = _random_log_probs(8, 500, seed=3).float()
    contexts = torch.randint(0, 500, (8, 4), generator=torch.Generator().manual_seed(4))
    watermarker = PPLSoftWatermarker(key=42)

    beta = watermarker.find_beta_for_logp_constraint(log_probs, contexts)

    nodes, target = _nodes_and_target(watermarker, log_probs, contexts)
    assert (_mean_selected(nodes, beta * (1 + 1e-4)) > target).all()
    assert (_mean_selected(nodes, beta * (1 - 1e-4)) < target).all()


def test_soft_ppl_selects_the_best_score_plus_scaled_log_probability():
    watermarker = PPLSoftWatermarker(key=42, context_width=2)
    logits = 2 * torch.randn(4, 300, generator=torch.Generator().manual_seed(5))
    contexts = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])

    sample = watermarker.sample(logits, contexts)

    beta = watermarker.find_beta_for_logp_constraint(
        logits.log_softmax(dim=-1), contexts
    )
    probs = logits.softmax(dim=-1)
    scores = watermarker.scores(contexts, 300)
    keys = scores + beta.unsqueeze(-1) * torch.log(probs.clamp_min(1e-12))
    assert torch.equal(sample.token_ids, keys.argmax(dim=-1))
    # A deterministic selection leaves the model's logits as the output.
    assert sample.logits is logits


def test_soft_ppl_selects_among_the_ranked_tokens():
    """Only the 4096 most probable tokens can be selected, as only they are
    drawn. Here half of the tokens are barely less probable; for such a flat
    distribution beta is about 0, so without the cap the highest score would
    often be among them."""
    logits = torch.zeros(16, 8192)
    logits[:, 4096:] = -1e-3
    contexts = torch.randint(
        0, 8192, (16, 2), generator=torch.Generator().manual_seed(3)
    )

    sample = PPLSoftWatermarker(key=42, context_width=2).sample(logits, contexts)

    assert (sample.token_ids < 4096).all()


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)
def test_monte_carlo_bisection_kernel_matches_reference():
    from vllm.v1.worker.gpu.sample.binomial_watermark import monte_carlo_bisection

    log_probs = _random_log_probs(64, 2_000, seed=6).float()
    contexts = torch.randint(
        -1, 2_000, (64, 4), generator=torch.Generator().manual_seed(7)
    )
    watermarker = PPLSoftWatermarker(key=42)
    nodes, target = _nodes_and_target(watermarker, log_probs, contexts)

    beta, kernel_nodes = monte_carlo_bisection(
        log_probs.topk(2_000, dim=-1).values.cuda(),
        contexts.cuda(),
        _MONTE_CARLO_KEY,
        watermarker.n_mc,
        watermarker.level_probs.cuda(),
        target.cuda(),
        0.0,
        1e2,
        60,
    )

    # The kernel draws in float32: rounding can move the odd draw across a
    # boundary of the float64 reference.
    kernel_nodes = kernel_nodes.cpu()
    same = torch.isclose(kernel_nodes, nodes, rtol=1e-5, atol=1e-6)
    assert same.double().mean() > 0.999
    # beta sits where a draw's selection flips, which rounding can move, so
    # check that it brackets the target rather than comparing it exactly.
    beta = beta.cpu()
    assert (_mean_selected(kernel_nodes, beta * (1 + 1e-4)) > target).all()
    assert (_mean_selected(kernel_nodes, beta * (1 - 1e-4)) < target).all()


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)
@pytest.mark.parametrize("vocab_size", [2_000, 20_000])
@pytest.mark.parametrize("epsilon", [0.0, 0.3, 2.0])
def test_hard_ppl_kernels_match_reference(vocab_size: int, epsilon: float):
    watermarker = PPLHardWatermarker(key=42, context_width=2, epsilon=epsilon)
    logits = _random_log_probs(32, vocab_size, seed=8).float()
    contexts = torch.randint(
        0, vocab_size, (32, 2), generator=torch.Generator().manual_seed(9)
    )
    skip_mask = torch.arange(32) < 4

    actual = watermarker.watermark_logits(
        logits.cuda(), contexts.cuda(), skip_mask.cuda()
    ).cpu()

    expected = watermarker.convex_hull_algorithm(logits.log_softmax(dim=-1), contexts)
    torch.testing.assert_close(actual[4:].exp(), expected[4:], atol=1e-6, rtol=1e-5)
    assert torch.equal(actual[:4], logits[:4])


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)
@pytest.mark.parametrize("vocab_size", [2_000, 20_000])
def test_soft_ppl_kernels_match_reference(vocab_size: int):
    """The kernels select the token with the highest key, and sample the rows
    without a full context as the plain sampler."""
    logits = _random_log_probs(16, vocab_size, seed=10).float().cuda()
    contexts = torch.randint(
        0, vocab_size, (16, 2), generator=torch.Generator().manual_seed(11)
    ).cuda()
    contexts[:3, 0] = -1
    random_sampler = RandomSampler(
        expanded_idx_mapping=torch.arange(16, device="cuda"),
        temperatures=torch.ones(16, device="cuda"),
        seeds=torch.arange(16, device="cuda"),
        positions=torch.full((16,), 7, device="cuda"),
    )

    watermarker = PPLSoftWatermarker(key=42, context_width=2, epsilon=0.2)

    sample = watermarker.sample(logits, contexts, random_sampler)

    skipped = (contexts < 0).any(dim=-1)
    assert torch.equal(sample.token_ids[skipped], random_sampler(logits)[skipped])
    # Rounding may break a near tie differently, so compare keys, not tokens.
    log_probs = logits.log_softmax(dim=-1)
    ranked = log_probs.topk(min(vocab_size, 4096), dim=-1).values
    beta = watermarker.find_beta_for_logp_constraint(log_probs, contexts, ranked=ranked)
    keys = watermarker.scores(contexts, vocab_size) + beta.unsqueeze(-1) * torch.log(
        log_probs.exp().clamp_min(1e-12)
    )
    keys = keys.masked_fill(
        log_probs.isneginf() | (log_probs < ranked[:, -1:]), -math.inf
    )
    selected = keys.gather(1, sample.token_ids.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(
        selected[~skipped], keys.amax(dim=-1)[~skipped], rtol=1e-5, atol=1e-4
    )
    assert sample.logits is logits


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)
@pytest.mark.parametrize("vocab_size", [2_000, 20_000])
def test_hard_ppl_sample_kernels_match_reference(vocab_size: int):
    """The kernels draw the token random_sampler draws from watermark_logits,
    and return the logits as they came."""
    watermarker = PPLHardWatermarker(key=42, context_width=2, epsilon=0.3)
    logits = _random_log_probs(16, vocab_size, seed=13).float().cuda()
    contexts = torch.randint(
        0, vocab_size, (16, 2), generator=torch.Generator().manual_seed(14)
    ).cuda()
    contexts[:3, 0] = -1
    random_sampler = RandomSampler(
        expanded_idx_mapping=torch.arange(16, device="cuda"),
        temperatures=torch.ones(16, device="cuda"),
        seeds=torch.arange(16, device="cuda"),
        positions=torch.full((16,), 7, device="cuda"),
    )

    sample = watermarker.sample(logits, contexts, random_sampler)

    skip_mask = (contexts < 0).any(dim=-1)
    expected = random_sampler(watermarker.watermark_logits(logits, contexts, skip_mask))
    assert torch.equal(sample.token_ids, expected)
    assert sample.logits is logits
