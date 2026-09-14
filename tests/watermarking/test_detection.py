# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.watermarking import GumbelWatermarkDetector

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


def test_detector_handles_empty_input():
    detector = GumbelWatermarkDetector(key=42, context_width=4, prf="philox")
    contexts, targets = detector._prepare_inputs([])
    detection = detector.detect([])

    assert contexts.shape == (0, 4)
    assert targets.shape == (0,)
    assert detection.score == 0
    assert detection.p_value == 1
    assert not detection.is_watermarked
