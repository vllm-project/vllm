# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch

from vllm.v1.watermarking import (
    SynthIDTextWatermarkDetector,
    SynthIDTextWatermarker,
)
from vllm.v1.watermarking.synthid_text import _binomial_survival, synthid_text_logits


def _argmax_sampler(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1)


def _tournament_winner_distribution(
    probs: list[float], g_values: list[list[int]]
) -> list[float]:
    """Play every two-candidate match of each layer, breaking ties evenly."""
    vocabulary = range(len(probs))
    for layer_g in g_values:
        winner = [0.0] * len(probs)
        for first in vocabulary:
            for second in vocabulary:
                match = probs[first] * probs[second]
                if layer_g[first] == layer_g[second]:
                    winner[first] += match / 2
                    winner[second] += match / 2
                elif layer_g[first] > layer_g[second]:
                    winner[first] += match
                else:
                    winner[second] += match
        probs = winner
    return probs


@pytest.mark.parametrize("depth", [1, 3])
def test_reference_matches_brute_force_tournament(depth: int):
    logits = torch.tensor([[0.5, -1.0, 2.0, 0.0, -torch.inf, 1.0]])
    contexts = torch.tensor([[11, 12, 13, 14]])
    watermarker = SynthIDTextWatermarker(key=42, depth=depth)
    words = watermarker.prf.words(contexts, torch.arange(logits.shape[-1]))[0]
    g_values = [[int(word) >> layer & 1 for word in words] for layer in range(depth)]

    expected = _tournament_winner_distribution(
        logits.softmax(dim=-1)[0].tolist(), g_values
    )
    actual = synthid_text_logits(logits, contexts, watermarker.prf, depth).exp()[0]

    torch.testing.assert_close(actual, torch.tensor(expected), rtol=1e-5, atol=1e-7)


def test_skipped_rows_use_the_unwatermarked_logits():
    torch.manual_seed(0)
    logits = torch.randn(4, 64)
    contexts = torch.arange(16).reshape(4, 4)
    skip_mask = torch.tensor([True, False, True, False])
    watermarker = SynthIDTextWatermarker(key=42)
    watermarked = synthid_text_logits(logits, contexts, watermarker.prf, 30)

    sample = watermarker.sample(logits, contexts, _argmax_sampler, skip_mask)

    expected = torch.where(skip_mask, logits.argmax(-1), watermarked.argmax(-1))
    assert torch.equal(sample.token_ids, expected)
    assert sample.logits is logits


def test_detector_separates_watermarked_from_unwatermarked_text():
    vocab_size, context_width, num_tokens = 64, 4, 48
    generator = torch.Generator().manual_seed(0)
    watermarker = SynthIDTextWatermarker(key=42, context_width=context_width)

    def sampler(logits: torch.Tensor) -> torch.Tensor:
        uniforms = torch.rand(logits.shape, generator=generator)
        return (logits - torch.log(-torch.log(uniforms))).argmax(dim=-1)

    generated: list[int] = []
    for _ in range(num_tokens):
        context = torch.tensor([([-1] * context_width + generated)[-context_width:]])
        sample = watermarker.sample(
            torch.zeros(1, vocab_size), context, sampler, (context < 0).any(-1)
        )
        generated.append(int(sample.token_ids))
    unwatermarked = torch.randint(vocab_size, (num_tokens,), generator=generator)

    detector = SynthIDTextWatermarkDetector(key=42, context_width=context_width)

    assert detector.detect(generated).p_value < 1e-6
    assert detector.detect(unwatermarked.tolist()).p_value > 0.05
    assert SynthIDTextWatermarkDetector(key=43).detect(generated).p_value > 0.05


@pytest.mark.parametrize(("successes", "trials"), [(0, 5), (3, 3), (40, 60)])
def test_binomial_survival_is_exact(successes: int, trials: int):
    expected = sum(math.comb(trials, count) for count in range(successes, trials + 1))

    assert _binomial_survival(successes, trials) == pytest.approx(
        expected / 2**trials, rel=1e-12
    )


@pytest.mark.parametrize("depth", [0, 33])
def test_depth_is_bounded(depth: int):
    with pytest.raises(ValueError, match="depth must be between 1 and 32"):
        SynthIDTextWatermarker(key=42, depth=depth)
    with pytest.raises(ValueError, match="depth must be between 1 and 32"):
        SynthIDTextWatermarkDetector(key=42, depth=depth)
