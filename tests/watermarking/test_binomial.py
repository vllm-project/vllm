# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch
from scipy import stats

from vllm.platforms import current_platform
from vllm.v1.watermarking import (
    BinomialScoreWatermarkDetector,
    ChiSquaredWatermarker,
    PPLHardWatermarker,
    PPLSoftWatermarker,
)
from vllm.v1.watermarking.binomial import (
    _binomial_survival,
    binomial_scores,
    binomial_thresholds,
)


def test_scores_of_every_prf_uniform_are_binomial():
    thresholds = binomial_thresholds(30)
    counts = torch.zeros(31, dtype=torch.int64)
    # The PRF's uniforms are (m + 1) / (2**24 + 1) for m below 2**24; scored
    # in chunks to bound the memory of the comparisons.
    for start in range(0, 2**24, 2**20):
        m = torch.arange(start, start + 2**20, dtype=torch.float64)
        scores = binomial_scores(((m + 1) / (2**24 + 1)).float(), thresholds)
        counts += torch.bincount(scores, minlength=31)

    frequencies = counts.numpy() / 2**24
    np.testing.assert_allclose(
        frequencies, stats.binom.pmf(np.arange(31), 30, 0.5), rtol=0, atol=2**-23
    )


@pytest.mark.parametrize("trials", [1, 30, 6000])
def test_binomial_survival(trials: int):
    for successes in (0, 1, trials // 2, trials // 2 + 40, trials, trials + 1):
        assert _binomial_survival(successes, trials, 0.5) == pytest.approx(
            stats.binom.sf(successes - 1, trials, 0.5), rel=1e-9, abs=1e-300
        )


def test_detector_sums_token_scores():
    detector = BinomialScoreWatermarkDetector(
        key=42, context_width=1, deduplicate_contexts=False
    )
    token_ids = [3, 1, 4, 1, 5, 9, 2, 6]
    scores = ChiSquaredWatermarker(key=42, context_width=1).scores(
        torch.tensor([[token] for token in token_ids[:-1]]), 10
    )

    detection = detector.detect(token_ids)

    # The first token has no full context, so it is not scored.
    assert detection.num_scored_tokens == len(token_ids) - 1
    assert detection.score == scores[torch.arange(7), token_ids[1:]].sum().item()
    assert detection.p_value == pytest.approx(
        stats.binom.sf(detection.score - 1, 30 * (len(token_ids) - 1), 0.5)
    )


def _argmax_sampler(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1)


@pytest.mark.parametrize(
    "watermarker_cls", [ChiSquaredWatermarker, PPLHardWatermarker, PPLSoftWatermarker]
)
def test_tokens_without_a_full_context_are_not_watermarked(watermarker_cls):
    """As in the reference, the watermark starts once the output gives a token
    a full context."""
    watermarker = watermarker_cls(key=42, context_width=2)
    logits = 2 * torch.randn(3, 64, generator=torch.Generator().manual_seed(0))
    contexts = torch.tensor([[-1, -1], [-1, 5], [4, 5]])

    sample = watermarker.sample(logits, contexts, _argmax_sampler)

    assert torch.equal(sample.token_ids[:2], logits[:2].argmax(dim=-1))
    assert torch.equal(sample.logits[:2], logits[:2])


def test_detector_recognizes_watermarked_tokens():
    watermarker = ChiSquaredWatermarker(key=42, context_width=2, delta=0.5)
    generator = torch.Generator().manual_seed(0)
    token_ids: list[int] = []
    for _ in range(128):
        context = ([-1, -1] + token_ids)[-2:]
        logits = watermarker.watermark_logits(
            torch.randn(1, 1000, generator=generator), torch.tensor([context])
        )
        token_ids.append(
            int(torch.multinomial(logits.softmax(-1), 1, generator=generator))
        )
    unwatermarked = torch.randint(0, 1000, (128,), generator=generator).tolist()

    detection = BinomialScoreWatermarkDetector(key=42, context_width=2).detect(
        token_ids
    )

    assert detection.is_watermarked
    assert detection.p_value < 1e-10
    assert (
        not BinomialScoreWatermarkDetector(key=42, context_width=2)
        .detect(unwatermarked)
        .is_watermarked
    )
    assert (
        not BinomialScoreWatermarkDetector(key=43, context_width=2)
        .detect(token_ids)
        .is_watermarked
    )


def _kernel_inputs(num_rows: int, vocab_size: int, context_width: int, dtype):
    generator = torch.Generator(device="cuda").manual_seed(0)
    logits = torch.randn(num_rows, vocab_size, device="cuda", generator=generator)
    filtered = torch.rand(num_rows, vocab_size, device="cuda", generator=generator)
    logits = logits.masked_fill(filtered < 0.5, -float("inf"))
    logits[:, 0] = 0.0
    contexts = torch.randint(
        -1, vocab_size, (num_rows, context_width), device="cuda", generator=generator
    )
    return (3 * logits).to(dtype), contexts


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_level_bias_kernel_matches_reference(dtype: torch.dtype):
    from vllm.v1.worker.gpu.sample.binomial_watermark import philox_level_bias

    watermarker = ChiSquaredWatermarker(key=2**40 + 7, context_width=4)
    vocab_size = 50_257
    logits, contexts = _kernel_inputs(6, vocab_size, 4, dtype)
    generator = torch.Generator(device="cuda").manual_seed(1)
    level_bias = torch.randn(6, 31, device="cuda", generator=generator)
    level_bias[:, :12] = -float("inf")
    skip_mask = torch.tensor([False, True, False, False, True, False], device="cuda")

    output = philox_level_bias(
        logits,
        contexts,
        watermarker.prf.key,
        watermarker.thresholds.cuda(),
        level_bias,
        skip_mask=skip_mask,
    )

    bias = level_bias.gather(1, watermarker.scores(contexts, vocab_size))
    assert output.dtype == torch.float32
    assert torch.equal(
        output, logits.float() + bias.masked_fill(skip_mask.unsqueeze(-1), 0.0)
    )
