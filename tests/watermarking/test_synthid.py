# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch
from pydantic import ValidationError

from vllm.config.watermarking import WatermarkConfig
from vllm.v1.watermarking.factory import create_watermarker
from vllm.v1.watermarking.prfs import PhiloxPRF
from vllm.v1.watermarking.synthid import (
    SynthIDWatermarkDetector,
    SynthIDWatermarker,
)


def test_synthid_config_and_factory():
    config = WatermarkConfig(key=42, algorithm="synthid_text", context_width=3, depth=5)
    watermarker = create_watermarker(config)

    assert isinstance(watermarker, SynthIDWatermarker)
    assert watermarker.context_width == 3
    assert watermarker.depth == 5
    assert watermarker.prf.key == 42
    assert not config.supports_speculative_decoding
    assert WatermarkConfig(key=42, algorithm="gumbel").depth == 32


@pytest.mark.parametrize("depth", [0, 33])
def test_synthid_config_rejects_invalid_depth(depth):
    with pytest.raises(ValidationError):
        WatermarkConfig(key=42, algorithm="synthid_text", depth=depth)
    assert WatermarkConfig(key=42, algorithm="gumbel", depth=depth).depth == depth


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"context_width": 0}, "context_width must be positive"),
        ({"depth": 0}, "SynthID-Text depth must be between 1 and 32"),
        ({"depth": 33}, "SynthID-Text depth must be between 1 and 32"),
        ({"key": -1}, "Philox keys must fit in 64 bits"),
    ],
)
def test_synthid_rejects_invalid_constructor_args(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SynthIDWatermarker(**({"key": 42} | kwargs))


@pytest.mark.parametrize("key, depth", [(0, 1), (42, 5), (2**64 - 1, 32)])
def test_synthid_logits_follow_philox_bits_and_reweighting(key, depth):
    contexts = torch.tensor([[-1, -1, 7], [11, 12, 13]])
    logits = torch.tensor([[0.2, -0.3, 0.7, -1.1], [0.4, 1.3, -0.8, 0.1]])
    words = PhiloxPRF(key).uint32(contexts, torch.arange(logits.shape[1]))

    expected_probs = torch.softmax(logits, dim=1)
    for bit in range(depth):
        g = ((words >> bit) & 1).to(logits.dtype)
        mass = (expected_probs * g).sum(dim=1, keepdim=True)
        expected_probs *= 1 + g - mass

    actual = SynthIDWatermarker(key, context_width=3, depth=depth).watermark_logits(
        logits, contexts
    )
    torch.testing.assert_close(actual.exp(), expected_probs)
    torch.testing.assert_close(actual.exp().sum(dim=1), torch.ones(2))
    assert torch.equal(
        actual,
        SynthIDWatermarker(key, context_width=3, depth=depth).watermark_logits(
            logits, contexts
        ),
    )


def test_synthid_native_partial_context_changes_stream():
    prf = PhiloxPRF(42)
    candidates = torch.arange(32)
    partial = torch.tensor([[-1, -1, 7]])
    zero_filled = torch.tensor([[0, 0, 7]])

    assert torch.equal(prf.uint32(partial, candidates), prf.uint32(partial, candidates))
    assert not torch.equal(
        prf.uint32(partial, candidates), prf.uint32(zero_filled, candidates)
    )


def test_synthid_sample_uses_watermarked_logits_and_skip_mask():
    watermarker = SynthIDWatermarker(42, context_width=3, depth=5)
    logits = torch.tensor([[0.2, 0.4, -0.1], [0.3, -0.2, 0.5]])
    contexts = torch.tensor([[-1, -1, 7], [11, 12, 13]])
    skip_mask = torch.tensor([False, True])
    sampled_logits = []

    def random_sampler(scores):
        sampled_logits.append(scores)
        return scores.argmax(dim=-1)

    result = watermarker.sample(logits, contexts, random_sampler, skip_mask)
    expected = watermarker.watermark_logits(logits, contexts)

    assert len(sampled_logits) == 1
    torch.testing.assert_close(result.logits[0], expected[0])
    torch.testing.assert_close(result.logits[1], logits[1])
    torch.testing.assert_close(sampled_logits[0], result.logits)
    assert torch.equal(result.token_ids, result.logits.argmax(dim=-1))


def test_synthid_requires_random_sampler():
    with pytest.raises(ValueError, match="SynthID-Text requires a random sampler"):
        SynthIDWatermarker(42).sample(torch.zeros(1, 3), torch.zeros(1, 4))


@pytest.mark.parametrize(
    "key, context_width, depth",
    [(0, 1, 1), (42, 2, 3), (2**64 - 1, 4, 32)],
)
def test_synthid_detector_matches_independent_bit_count(key, context_width, depth):
    tokens = [3, 5, 7, 11, 13, 17]
    contexts = torch.tensor(
        [
            ([-1] * context_width + tokens[:position])[-context_width:]
            for position in range(len(tokens))
        ]
    )
    words = PhiloxPRF(key).uint32(contexts, torch.tensor(tokens)[:, None]).flatten()
    trials = len(tokens) * depth
    hits = sum((int(word) // (2**bit)) % 2 for word in words for bit in range(depth))
    expected_p_value = sum(math.comb(trials, k) for k in range(hits, trials + 1)) / (
        2**trials
    )

    result = SynthIDWatermarkDetector(key, context_width, depth).detect(tokens)

    assert result.num_scored_tokens == len(tokens)
    assert result.score == hits / trials
    assert result.p_value == pytest.approx(expected_p_value)


def test_synthid_detector_empty_and_repeated_contexts():
    detector = SynthIDWatermarkDetector(42, context_width=1, depth=4)

    empty = detector.detect([])
    repeated = detector.detect([1, 1, 1, 1, 1])

    assert empty.num_scored_tokens == 0
    assert empty.p_value == 1
    assert not empty.is_watermarked
    assert repeated.num_scored_tokens == 2


def test_synthid_detector_recognizes_generated_tokens():
    key = 42
    width = 4
    depth = 4
    watermarker = SynthIDWatermarker(key, width, depth)
    generator = torch.Generator().manual_seed(123)
    logits = torch.zeros(1, 32)
    tokens: list[int] = []

    def random_sampler(scores):
        return torch.multinomial(
            scores.softmax(dim=-1), 1, generator=generator
        ).flatten()

    for _ in range(96):
        context = ([-1] * width + tokens)[-width:]
        sample = watermarker.sample(logits, torch.tensor([context]), random_sampler)
        tokens.append(int(sample.token_ids.item()))

    matched = SynthIDWatermarkDetector(key, width, depth).detect(tokens)

    assert matched.is_watermarked
    assert matched.p_value < 0.01
