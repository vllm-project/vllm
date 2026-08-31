# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.watermarking import GumbelWatermarkDetector, GumbelWatermarker
from vllm.v1.watermarking.gumbel import _gamma_survival_integer_shape


def test_gamma_survival_integer_shape():
    assert _gamma_survival_integer_shape(0.0, 4) == 1.0
    assert _gamma_survival_integer_shape(2.0, 1) == pytest.approx(0.1353352832)


def test_detector_deduplicates_repeated_prf_inputs_by_default():
    detection = GumbelWatermarkDetector(key=42, context_width=1).detect([1, 1, 1, 1, 1])

    assert detection.num_scored_tokens == 2


def test_detector_can_score_repeated_prf_inputs():
    detection = GumbelWatermarkDetector(
        key=42, context_width=1, deduplicate_contexts=False
    ).detect([1, 1, 1, 1, 1])

    assert detection.num_scored_tokens == 5


def test_detector_deduplicates_context_even_when_target_differs():
    detection = GumbelWatermarkDetector(key=42, context_width=1).detect([1, 2, 1, 3])

    assert detection.num_scored_tokens == 3


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)
@pytest.mark.parametrize("key", [42, 15726070495360670683])
@pytest.mark.parametrize("context_width", [1, 4, 16])
def test_fused_watermarker_matches_cpu(key: int, context_width: int):
    torch.manual_seed(0)
    contexts = torch.randint(0, 248320, (32, context_width), dtype=torch.int64)
    logits = torch.randn(32, 8193)
    watermarker = GumbelWatermarker(key, context_width)

    expected = watermarker.sample(logits, contexts, lambda values: None).token_ids
    actual = watermarker.sample(
        logits.cuda(), contexts.cuda(), lambda values: None
    ).token_ids.cpu()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
