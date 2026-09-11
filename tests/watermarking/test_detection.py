# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

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


def _detector(**kwargs) -> GumbelWatermarkDetector:
    return GumbelWatermarkDetector(key=42, context_width=4, prf="philox", **kwargs)


def test_context_prefix_positions_are_not_scored():
    prefix = list(range(100, 110))
    suffix = list(range(200, 208))

    detection = _detector(history_scope="all").detect(suffix, context_prefix=prefix)

    assert detection.num_scored_tokens == len(suffix)


def test_context_prefix_seeds_the_contexts_of_the_first_tokens():
    prefix = list(range(100, 110))
    suffix = list(range(200, 208))
    detector = _detector(deduplicate_contexts=False, history_scope="all")

    contexts, targets = detector._prepare_inputs(suffix, prefix)
    full_contexts, full_targets = detector._prepare_inputs(prefix + suffix)

    assert contexts.tolist() == full_contexts[len(prefix) :].tolist()
    assert targets.tolist() == full_targets[len(prefix) :].tolist()


def test_context_prefix_deduplicates_against_the_prefix():
    # The suffix token 9 follows the context (1, 2, 3, 4), which the prefix
    # already contains, so generation with "all" left it unwatermarked.
    prefix = [1, 2, 3, 4, 5, 6, 7, 8]
    suffix = [1, 2, 3, 4, 9]

    with_dedup = _detector(history_scope="all")
    without_dedup = _detector(deduplicate_contexts=False, history_scope="all")

    _, deduplicated = with_dedup._prepare_inputs(suffix, prefix)
    _, kept = without_dedup._prepare_inputs(suffix, prefix)

    assert 9 not in deduplicated.tolist()
    assert kept.tolist() == suffix


@pytest.mark.parametrize("history_scope", ["single_turn", "none"])
def test_context_prefix_is_rejected_outside_the_all_scope(history_scope):
    with pytest.raises(ValueError):
        _detector(history_scope=history_scope).detect([5, 6, 7], [1, 2])


def test_all_scope_without_prefix_scores_like_the_completion_only_detector():
    tokens = _WATERMARKED_TOKEN_IDS

    without_prefix = _detector(history_scope="all").detect(tokens)
    completion_only = _detector().detect(tokens)

    assert without_prefix == completion_only
