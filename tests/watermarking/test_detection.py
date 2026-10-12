# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.watermarking import (
    DualKeyGumbelWatermarkDetector,
    GumbelWatermarkDetector,
)

_NON_WATERMARKED_TOKEN_IDS = list(range(64))
_WATERMARKED_TOKEN_IDS = [
    85,
    56,
    123,
    77,
    105,
    62,
    9,
    104,
    16,
    98,
    123,
    22,
    16,
    35,
    47,
    127,
    77,
    67,
    19,
    2,
    10,
    8,
    75,
    22,
    127,
    30,
    95,
    9,
    101,
    14,
    113,
    122,
    85,
    66,
    122,
    50,
    38,
    119,
    15,
    58,
    68,
    62,
    24,
    126,
    80,
    126,
    68,
    83,
    81,
    126,
    62,
    31,
    18,
    81,
    64,
    47,
    22,
    70,
    47,
    109,
    54,
    21,
    65,
    38,
]


def test_detection_on_known_sequence():
    detector = GumbelWatermarkDetector(key=42, context_width=4, prf="philox")

    assert detector.detect(_WATERMARKED_TOKEN_IDS).is_watermarked
    assert not detector.detect(_NON_WATERMARKED_TOKEN_IDS).is_watermarked


@pytest.mark.parametrize("token_ids", [[], [1], [1, 2, 3, 4]])
def test_detector_handles_input_without_complete_context(token_ids):
    detector = GumbelWatermarkDetector(key=42, context_width=4, prf="philox")
    contexts, targets = detector._prepare_inputs(token_ids)
    detection = detector.detect(token_ids)

    assert contexts.shape == (0, 4)
    assert targets.shape == (0,)
    assert detection.score == 0
    assert detection.p_value == 1
    assert not detection.is_watermarked


@pytest.mark.parametrize(
    "detector_type,key,token_ids,threshold,score,p_value",
    [
        (
            GumbelWatermarkDetector,
            42,
            [7, 2, 13, 2, 5, 1, 11, 14],
            0.0015,
            19.672172609864035,
            0.0009697880157637389,
        ),
        (
            DualKeyGumbelWatermarkDetector,
            7,
            [7, 3, 13, 2, 5, 1, 4, 14],
            0.003,
            17.2829982488662,
            0.002130807652336366,
        ),
    ],
)
def test_legacy_opening_context_scores_remain_available(
    detector_type, key, token_ids, threshold, score, p_value
):
    detection = detector_type(
        key=key, p_value_threshold=threshold, skip_partial_contexts=False
    ).detect(token_ids)

    assert detection.num_scored_tokens == 8
    assert detection.score == pytest.approx(score, rel=1e-9)
    assert detection.p_value == pytest.approx(p_value, rel=1e-9)
    assert detection.is_watermarked
    assert (
        not detector_type(key=key, p_value_threshold=threshold)
        .detect(token_ids)
        .is_watermarked
    )
