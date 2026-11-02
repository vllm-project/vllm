# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.watermarking import ChiSquaredWatermarker
from vllm.v1.watermarking.binomial import _level_mass
from vllm.v1.watermarking.watermarker import RandomSampler


def _token_level_solution(probs: np.ndarray, scores: np.ndarray, delta: float):
    """Solve sum_u p_u * max(0, 1 + delta * (g_u + mu)) = 1 token by token,
    activating tokens in decreasing score order."""
    order = np.argsort(-scores, kind="stable")
    mass = np.cumsum(probs[order])
    weighted = np.cumsum(probs[order] * (1 + delta * scores[order]))
    for count in range(1, len(order) + 1):
        if mass[count - 1] == 0:
            continue
        shift = (1 - weighted[count - 1]) / mass[count - 1]
        next_score = scores[order[count]] if count < len(order) else -np.inf
        if 1 + delta * next_score + shift <= 1e-12:
            break
    return probs * np.maximum(0, 1 + delta * scores + shift)


def _random_inputs(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    logits = 3 * torch.randn(16, 200, generator=generator, dtype=torch.float64)
    logits[torch.rand(16, 200, generator=generator) < 0.3] = -float("inf")
    scores = torch.from_numpy(np.random.default_rng(seed).binomial(30, 0.5, (16, 200)))
    return logits.log_softmax(dim=-1), scores


@pytest.mark.parametrize("delta", [0.05, 0.5, 2.0])
def test_adjusted_scores_match_the_token_level_solution(delta: float):
    log_probs, scores = _random_inputs(seed=0)
    adjusted_scores = ChiSquaredWatermarker(key=0, delta=delta).adjusted_scores(
        _level_mass(log_probs, scores, 31)
    )

    watermarked = (log_probs + adjusted_scores.double().gather(1, scores)).exp()
    for row in range(16):
        expected = _token_level_solution(
            log_probs[row].exp().numpy(), scores[row].numpy(), delta
        )
        np.testing.assert_allclose(watermarked[row].numpy(), expected, atol=1e-6)


def test_bernoulli_scores_reduce_to_a_tournament_layer():
    """With one trial and delta 1, the watermark is a SynthID tournament layer:
    q = p * (1 + g - p . g)."""
    generator = torch.Generator().manual_seed(1)
    log_probs = torch.randn(8, 50, generator=generator).log_softmax(dim=-1)
    scores = torch.randint(0, 2, (8, 50), generator=generator)
    adjusted_scores = ChiSquaredWatermarker(key=0, delta=1.0).adjusted_scores(
        _level_mass(log_probs, scores, 2)
    )

    probs = log_probs.exp()
    torch.testing.assert_close(
        (log_probs + adjusted_scores.gather(1, scores)).exp(),
        probs * (1 + scores - (probs * scores).sum(dim=-1, keepdim=True)),
    )


class _ArgmaxRandomSampler(RandomSampler):
    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1)


def test_sample_reweights_the_tempered_distribution():
    """Tokens are drawn from q = p * w (Theorem D.6 of the paper), where p is
    the distribution after temperature scaling, whatever the temperature."""
    watermarker = ChiSquaredWatermarker(key=42, context_width=2, delta=0.5)
    logits = torch.randn(3, 64, generator=torch.Generator().manual_seed(2))
    contexts = torch.tensor([[1, 2], [3, 4], [5, 6]])
    random_sampler = _ArgmaxRandomSampler(
        expanded_idx_mapping=torch.tensor([2, 0, 1]),
        temperatures=torch.tensor([0.5, 0.7, 2.0]),
        seeds=torch.zeros(3, dtype=torch.int64),
        positions=torch.zeros(3, dtype=torch.int64),
    )

    sample = watermarker.sample(logits, contexts, random_sampler)

    scores = watermarker.scores(contexts, 64)
    log_w = watermarker.adjusted_scores(
        _level_mass(logits.log_softmax(dim=-1), scores, 31)
    ).gather(1, scores)
    assert torch.equal(sample.token_ids, (logits + log_w).argmax(dim=-1))
    assert sample.logits is logits


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)
@pytest.mark.parametrize(
    ("vocab_size", "context_width", "num_trials", "delta"),
    [
        (2_000, 2, 30, 0.1),
        (20_000, 2, 30, 2.0),
        # Not a multiple of the kernels' block sizes.
        (50_257, 1, 1, 1.0),
        (50_257, 5, 30, 0.5),
    ],
)
def test_watermark_logits_kernels_match_reference(
    vocab_size: int, context_width: int, num_trials: int, delta: float
):
    watermarker = ChiSquaredWatermarker(
        key=2**40 + 7, context_width=context_width, delta=delta, num_trials=num_trials
    )
    generator = torch.Generator().manual_seed(12)
    logits = 3 * torch.randn(16, vocab_size, generator=generator)
    logits[torch.rand(16, vocab_size, generator=generator) < 0.3] = -math.inf
    contexts = torch.randint(-1, vocab_size, (16, context_width), generator=generator)
    skip_mask = torch.arange(16) < 2

    actual = watermarker.watermark_logits(
        logits.cuda(), contexts.cuda(), skip_mask.cuda()
    ).cpu()

    expected = watermarker.watermark_logits(logits, contexts, skip_mask)
    assert torch.equal(actual[:2], logits[:2])
    # Rounding moves mu by ~1e-7, which log w amplifies near its clamp; compare
    # what sampling uses.
    torch.testing.assert_close(
        actual.softmax(dim=-1), expected.softmax(dim=-1), atol=1e-6, rtol=1e-3
    )
