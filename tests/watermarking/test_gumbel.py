# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.watermarking import GumbelWatermarkDetector
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
