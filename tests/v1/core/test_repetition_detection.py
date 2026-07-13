# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import Any

import pytest

from vllm.sampling_params import RepetitionDetectionParams, SamplingParams
from vllm.v1.core.sched.utils import check_sequence_repetition, check_stop
from vllm.v1.request import Request, RequestStatus

pytestmark = pytest.mark.cpu_test

CONSECUTIVE = {
    "max_pattern_size": 5,
    "min_pattern_size": 2,
    "min_count": 3,
}
OCCURRENCE_16 = {
    "max_pattern_size": 16,
    "min_pattern_size": 16,
    "min_count": 6,
    "mode": "occurrence",
}


def _separated_occurrences(ngram: list[int], count: int) -> list[int]:
    tokens: list[int] = []
    for index in range(count):
        tokens.extend(ngram)
        if index + 1 < count:
            tokens.append(1000 + index)
    return tokens


@pytest.mark.parametrize(
    ("token_ids", "config", "detected"),
    [
        pytest.param(
            [1, 2, 3] * 3,
            {"max_pattern_size": 3, "min_pattern_size": 2, "min_count": 3},
            True,
            id="threshold-reached",
        ),
        pytest.param(
            [1, 2, 3] * 2,
            {"max_pattern_size": 3, "min_pattern_size": 2, "min_count": 3},
            False,
            id="below-count-threshold",
        ),
        pytest.param(
            [1, 2] * 4,
            {"max_pattern_size": 5, "min_pattern_size": 2, "min_count": 4},
            True,
            id="two-token-pattern",
        ),
        pytest.param(list(range(1, 10)), CONSECUTIVE, False, id="varied-sequence"),
        pytest.param(
            [1, 2, 3, 1, 2, 3, 1, 2, 4],
            {"max_pattern_size": 3, "min_pattern_size": 2, "min_count": 3},
            False,
            id="incomplete-tail",
        ),
        pytest.param([], CONSECUTIVE, False, id="empty"),
        pytest.param([1, 2] * 3, {}, False, id="disabled"),
        pytest.param(
            [1, 2, 3, 4, 5, 6, 5, 6, 5, 6],
            {"max_pattern_size": 3, "min_pattern_size": 2, "min_count": 3},
            True,
            id="repeated-tail-after-prefix",
        ),
        pytest.param(
            list(range(8)) * 5,
            {"max_pattern_size": 10, "min_pattern_size": 2, "min_count": 3},
            True,
            id="large-pattern",
        ),
        pytest.param(
            [10, 20, 30] * 2,
            {"max_pattern_size": 3, "min_pattern_size": 3, "min_count": 2},
            True,
            id="exact-size-boundary",
        ),
        pytest.param(
            [7, 8, 9, 10] * 3,
            CONSECUTIVE,
            True,
            id="pattern-inside-size-range",
        ),
        pytest.param(
            [10, 20] * 3,
            {"max_pattern_size": 5, "min_pattern_size": 3, "min_count": 3},
            False,
            id="below-size-boundary",
        ),
        pytest.param(
            [10, 20] * 3,
            {"max_pattern_size": 5, "min_pattern_size": 2, "min_count": 5},
            False,
            id="high-count-threshold",
        ),
    ],
)
def test_consecutive_repetition_contract(
    token_ids: list[int], config: dict[str, Any], detected: bool
) -> None:
    params = RepetitionDetectionParams(**config)

    assert check_sequence_repetition(token_ids, params) is detected


@pytest.mark.parametrize(
    ("token_ids", "config", "detected"),
    [
        pytest.param(
            _separated_occurrences(list(range(16)), 6),
            OCCURRENCE_16,
            True,
            id="nonconsecutive-threshold-reached",
        ),
        pytest.param(
            _separated_occurrences(list(range(16)), 5),
            OCCURRENCE_16,
            False,
            id="nonconsecutive-below-threshold",
        ),
        pytest.param([7] * 21, OCCURRENCE_16, True, id="overlapping-ngrams"),
        pytest.param(
            [11, 12, 13, 14] * 19,
            {"mode": "occurrence", "occurrence_rules": [(4, 20)]},
            False,
            id="four-gram-below-threshold",
        ),
        pytest.param(
            [11, 12, 13, 14] * 20,
            {"mode": "occurrence", "occurrence_rules": [(4, 20)]},
            True,
            id="four-gram-threshold-reached",
        ),
        pytest.param(
            list(range(8)) * 9,
            {"mode": "occurrence", "occurrence_rules": [(8, 10)]},
            False,
            id="eight-gram-below-threshold",
        ),
        pytest.param(
            list(range(8)) * 10,
            {"mode": "occurrence", "occurrence_rules": [(8, 10)]},
            True,
            id="eight-gram-threshold-reached",
        ),
    ],
)
def test_occurrence_repetition_contract(
    token_ids: list[int], config: dict[str, Any], detected: bool
) -> None:
    params = RepetitionDetectionParams(**config)

    assert check_sequence_repetition(token_ids, params) is detected


@pytest.mark.parametrize(
    ("config", "message"),
    [
        pytest.param(
            {"mode": "consecutive", "occurrence_rules": [(8, 10)]},
            "only be used with mode='occurrence'",
            id="wrong-mode",
        ),
        pytest.param(
            {"mode": "occurrence", "occurrence_rules": [(0, 2)]},
            "ngram_size must be positive",
            id="nonpositive-size",
        ),
        pytest.param(
            {"mode": "occurrence", "occurrence_rules": [(4, 1)]},
            "min_count must be >= 2",
            id="count-below-two",
        ),
        pytest.param(
            {"mode": "occurrence", "occurrence_rules": []},
            "must contain at least one rule",
            id="empty-rules",
        ),
    ],
)
def test_occurrence_rule_validation(config: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        RepetitionDetectionParams(**config)


def _request(
    token_ids: Sequence[int] = (),
    repetition_detection: RepetitionDetectionParams | None = None,
    **sampling: Any,
) -> Request:
    request = Request(
        request_id="test",
        prompt_token_ids=[1],
        sampling_params=SamplingParams(
            max_tokens=sampling.pop("max_tokens", 100),
            repetition_detection=repetition_detection,
            **sampling,
        ),
        pooling_params=None,
    )
    if token_ids:
        request.append_output_token_ids(token_ids)
    return request


@pytest.mark.parametrize(
    ("token_ids", "detection", "sampling", "expected"),
    [
        pytest.param(
            [10, 20] * 3,
            RepetitionDetectionParams(**CONSECUTIVE),
            {},
            (True, RequestStatus.FINISHED_REPETITION, "repetition_detected"),
            id="repetition-stops",
        ),
        pytest.param(
            [10, 20] * 3,
            None,
            {},
            (False, RequestStatus.WAITING, None),
            id="disabled-continues",
        ),
        pytest.param(
            [10, 20] * 3,
            RepetitionDetectionParams(**CONSECUTIVE),
            {"min_tokens": 10},
            (False, RequestStatus.WAITING, None),
            id="minimum-output-precedes-detection",
        ),
        pytest.param(
            [999] * 3,
            RepetitionDetectionParams(
                max_pattern_size=1, min_pattern_size=1, min_count=3
            ),
            {"stop_token_ids": [999]},
            (True, RequestStatus.FINISHED_STOPPED, 999),
            id="stop-token-precedes-detection",
        ),
        pytest.param(
            [10, 20, 30, 40, 50, 60],
            RepetitionDetectionParams(**CONSECUTIVE),
            {},
            (False, RequestStatus.WAITING, None),
            id="nonrepetition-continues",
        ),
    ],
)
def test_check_stop_contract(
    token_ids: list[int],
    detection: RepetitionDetectionParams | None,
    sampling: dict[str, Any],
    expected: tuple[bool, RequestStatus, int | str | None],
) -> None:
    request = _request(token_ids, detection, **sampling)

    result = check_stop(request, max_model_len=1024)

    assert (result, request.status, request.stop_reason) == expected


@pytest.mark.parametrize(
    ("detection", "chunks"),
    [
        pytest.param(
            RepetitionDetectionParams(**OCCURRENCE_16),
            [
                [*range(16), 1000 + index]
                for index in range(5)
            ]
            + [list(range(16))],
            id="default-sixteen-gram-rule",
        ),
        pytest.param(
            RepetitionDetectionParams(
                mode="occurrence", occurrence_rules=[(8, 10), (4, 20)]
            ),
            [list(range(8)) for _ in range(10)],
            id="custom-eight-gram-rule",
        ),
    ],
)
def test_occurrences_accumulate_across_decode_steps(
    detection: RepetitionDetectionParams, chunks: list[list[int]]
) -> None:
    request = _request(repetition_detection=detection, max_tokens=200)

    for chunk in chunks[:-1]:
        request.append_output_token_ids(chunk)
        assert not check_stop(request, max_model_len=1024)

    request.append_output_token_ids(chunks[-1])
    assert check_stop(request, max_model_len=1024)
    assert (request.status, request.stop_reason) == (
        RequestStatus.FINISHED_REPETITION,
        "repetition_detected",
    )
