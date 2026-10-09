# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch
from pydantic import ValidationError

from vllm.config.watermarking import WatermarkConfig
from vllm.platforms import current_platform
from vllm.v1.watermarking.factory import create_watermarker
from vllm.v1.watermarking.gumbel import GumbelWatermarker
from vllm.v1.watermarking.prfs import PhiloxPRF
from vllm.v1.watermarking.synthid import (
    _STREAM_DOMAIN,
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


def test_synthid_config_rejects_invalid_depth():
    with pytest.raises(ValidationError):
        WatermarkConfig(
            key=42,
            algorithm="synthid_text",
            depth=0,
        )


@pytest.mark.parametrize("depth", [1, 32, 33, 64, 65])
def test_synthid_accepts_positive_depth(depth):
    config = WatermarkConfig(
        key=42,
        algorithm="synthid_text",
        depth=depth,
    )
    assert config.depth == depth


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"context_width": 0}, "context_width must be positive"),
        ({"depth": 0}, "SynthID-Text depth must be positive"),
        ({"key": -1}, "Philox keys must fit in 64 bits"),
    ],
)
def test_synthid_rejects_invalid_constructor_args(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SynthIDWatermarker(**({"key": 42} | kwargs))


@pytest.mark.parametrize(
    "key, depth",
    [
        (0, 1),
        (42, 5),
        (2**64 - 1, 32),
        (42, 33),
        (42, 64),
        (42, 65),
    ],
)
def test_synthid_logits_follow_philox_bits_and_reweighting(key, depth):
    contexts = torch.tensor([[-1, -1, 7], [11, 12, 13]])
    logits = torch.tensor([[0.2, -0.3, 0.7, -1.1], [0.4, 1.3, -0.8, 0.1]])
    prf = PhiloxPRF(key)

    expected_probs = torch.softmax(logits, dim=1)

    for depth_index in range(depth):
        stream = _STREAM_DOMAIN | depth_index // 32
        bit = depth_index % 32

        words = prf.uint32(
            contexts,
            torch.arange(logits.shape[1]),
            stream=stream,
        )

        g = ((words >> bit) & 1).to(logits.dtype)
        mass = (expected_probs * g).sum(dim=1, keepdim=True)
        expected_probs *= 1 + g - mass

    actual = SynthIDWatermarker(
        key,
        context_width=3,
        depth=depth,
    ).watermark_logits(logits, contexts)

    torch.testing.assert_close(actual.exp(), expected_probs)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)
@pytest.mark.parametrize("depth", [1, 32, 65])
def test_synthid_accelerator_matches_cpu(depth):
    logits = torch.randn(4, 3000, generator=torch.Generator().manual_seed(0))
    logits[0] *= 30  # peaked row
    logits[1, 100:] = float("-inf")  # masked tokens
    contexts = torch.tensor([[-1, -1, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    watermarker = SynthIDWatermarker(42, context_width=3, depth=depth)

    expected = watermarker.watermark_logits(logits.double(), contexts)
    actual = watermarker.watermark_logits(logits.cuda(), contexts.cuda()).cpu()

    assert actual.dtype == torch.float32
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))
    torch.testing.assert_close(
        actual.exp().double(), expected.exp().double(), rtol=1e-5, atol=1e-7
    )


@pytest.mark.parametrize(
    "constructor",
    [SynthIDWatermarker, SynthIDWatermarkDetector],
)
def test_synthid_warns_for_large_context_width(constructor):
    with pytest.warns(
        UserWarning,
        match="context_width values greater than 16 reduce robustness to edits",
    ):
        constructor(42, context_width=17)


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
    torch.testing.assert_close(sampled_logits[0][0], expected[0])
    torch.testing.assert_close(sampled_logits[0][1], logits[1])
    assert torch.equal(result.token_ids, sampled_logits[0].argmax(dim=-1))
    assert result.logits is logits

    # A peaked row underflows in-support tokens; only the input's -inf stay -inf.
    peaked = torch.tensor([[20.0, 0.0, 0.0, float("-inf")]])
    actual = SynthIDWatermarker(42, context_width=3, depth=32).watermark_logits(
        peaked, contexts[:1]
    )
    assert torch.equal(actual.isneginf(), peaked.isneginf())


def test_synthid_requires_random_sampler():
    with pytest.raises(ValueError, match="SynthID-Text requires a random sampler"):
        SynthIDWatermarker(42).sample(torch.zeros(1, 3), torch.zeros(1, 4))


@pytest.mark.parametrize(
    "key, context_width, depth",
    [
        (0, 1, 1),
        (42, 2, 3),
        (2**64 - 1, 4, 32),
        (42, 4, 33),
        (42, 4, 64),
        (42, 4, 65),
    ],
)
def test_synthid_detector_matches_independent_bit_count(
    key,
    context_width,
    depth,
):
    tokens = [3, 5, 7, 11, 13, 17]
    contexts = torch.tensor(
        [
            ([-1] * context_width + tokens[:position])[-context_width:]
            for position in range(len(tokens))
        ]
    )
    targets = torch.tensor(tokens)[:, None]
    prf = PhiloxPRF(key)

    hits = 0
    for depth_index in range(depth):
        stream = _STREAM_DOMAIN | depth_index // 32
        bit = depth_index % 32

        # [N, C] and [N, 1] -> [N, 1] -> [N]
        words = prf.uint32(
            contexts,
            targets,
            stream=stream,
        ).flatten()

        hits += sum((int(word) >> bit) & 1 for word in words)

    trials = len(tokens) * depth
    expected_p_value = sum(math.comb(trials, k) for k in range(hits, trials + 1)) / (
        2**trials
    )

    result = SynthIDWatermarkDetector(
        key,
        context_width,
        depth,
    ).detect(tokens)

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


def test_synthid_runs_tournament_in_fp32_for_bf16_logits():
    logits = torch.randn(4, 1000, generator=torch.Generator().manual_seed(0))
    logits = logits.to(torch.bfloat16)
    contexts = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    watermarker = SynthIDWatermarker(42, context_width=3, depth=32)

    actual = watermarker.watermark_logits(logits, contexts)
    expected = watermarker.watermark_logits(logits.float(), contexts)

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)


def test_synthid_detector_ignores_gumbel_text_with_same_key():
    key = 42
    width = 4
    generator = torch.Generator().manual_seed(0)
    watermarker = GumbelWatermarker(key, width)
    tokens: list[int] = []

    for _ in range(400):
        logits = torch.randn(1, 4096, generator=generator) * 2
        context = ([-1] * width + tokens)[-width:]
        sample = watermarker.sample(logits, torch.tensor([context]))
        tokens.append(int(sample.token_ids.item()))

    result = SynthIDWatermarkDetector(key, width, 32).detect(tokens)

    assert not result.is_watermarked
    assert result.p_value > 1e-4
