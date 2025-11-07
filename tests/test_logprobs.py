# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest

from vllm.logprobs import (
    FlattenLogprobs,
    Logprob,
    LogprobsOnePosition,
    append_logprobs_for_next_position,
    create_prompt_logprobs,
    create_sample_logprobs,
)


def test_create_logprobs_non_flatten(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_FLATTEN_LOGPROBS", "0")

    prompt_logprobs = create_prompt_logprobs()
    assert isinstance(prompt_logprobs, list)
    # Ensure first prompt position logprobs is None
    assert len(prompt_logprobs) == 1
    assert prompt_logprobs[0] is None

    sample_logprobs = create_sample_logprobs()
    assert isinstance(sample_logprobs, list)
    assert len(sample_logprobs) == 0


def test_create_logprobs_flatten(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_FLATTEN_LOGPROBS", "1")

    prompt_logprobs = create_prompt_logprobs()
    assert isinstance(prompt_logprobs, FlattenLogprobs)
    assert prompt_logprobs.start_indices == [0]
    assert prompt_logprobs.end_indices == [0]
    assert len(prompt_logprobs.token_ids) == 0
    assert len(prompt_logprobs.logprobs) == 0
    assert len(prompt_logprobs.ranks) == 0
    assert len(prompt_logprobs.decoded_tokens) == 0
    # Ensure first prompt position logprobs is empty
    assert len(prompt_logprobs) == 1
    assert prompt_logprobs[0] == dict()

    sample_logprobs = create_sample_logprobs()
    assert isinstance(sample_logprobs, FlattenLogprobs)
    assert len(sample_logprobs.start_indices) == 0
    assert len(sample_logprobs.end_indices) == 0
    assert len(sample_logprobs.token_ids) == 0
    assert len(sample_logprobs.logprobs) == 0
    assert len(sample_logprobs.ranks) == 0
    assert len(sample_logprobs.decoded_tokens) == 0
    assert len(sample_logprobs) == 0


def test_append_logprobs_for_next_position_none_flatten(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_FLATTEN_LOGPROBS", "0")
    logprobs = create_sample_logprobs()
    append_logprobs_for_next_position(
        logprobs,
        token_ids=[1],
        logprobs=[0.1],
        decoded_tokens=["1"],
        rank=10,
        num_logprobs=-1,
    )
    append_logprobs_for_next_position(
        logprobs,
        token_ids=[2, 3],
        logprobs=[0.2, 0.3],
        decoded_tokens=["2", "3"],
        rank=11,
        num_logprobs=-1,
    )
    assert isinstance(logprobs, list)
    assert logprobs == [
        {1: Logprob(logprob=0.1, rank=10, decoded_token="1")},
        {
            2: Logprob(logprob=0.2, rank=11, decoded_token="2"),
            3: Logprob(logprob=0.3, rank=1, decoded_token="3"),
        },
    ]


def test_append_logprobs_for_next_position_flatten(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_FLATTEN_LOGPROBS", "1")
    logprobs = create_sample_logprobs()
    append_logprobs_for_next_position(
        logprobs,
        token_ids=[1],
        logprobs=[0.1],
        decoded_tokens=["1"],
        rank=10,
        num_logprobs=-1,
    )
    append_logprobs_for_next_position(
        logprobs,
        token_ids=[2, 3],
        logprobs=[0.2, 0.3],
        decoded_tokens=["2", "3"],
        rank=11,
        num_logprobs=-1,
    )
    assert isinstance(logprobs, FlattenLogprobs)
    assert logprobs.start_indices == [0, 1]
    assert logprobs.end_indices == [1, 3]
    assert logprobs.token_ids == [1, 2, 3]
    assert logprobs.logprobs == [0.1, 0.2, 0.3]
    assert logprobs.ranks == [10, 11, 1]
    assert logprobs.decoded_tokens == ["1", "2", "3"]


LOGPROBS_ONE_POSITION_0: LogprobsOnePosition = {
    1: Logprob(logprob=0.1, rank=10, decoded_token="10")
}
LOGPROBS_ONE_POSITION_1: LogprobsOnePosition = {
    2: Logprob(logprob=0.2, rank=20, decoded_token="20"),
    3: Logprob(logprob=0.3, rank=30, decoded_token="30"),
}
LOGPROBS_ONE_POSITION_2: LogprobsOnePosition = {
    4: Logprob(logprob=0.4, rank=40, decoded_token="40"),
    5: Logprob(logprob=0.5, rank=50, decoded_token="50"),
    6: Logprob(logprob=0.6, rank=60, decoded_token="60"),
}


def test_flatten_logprobs_append() -> None:
    logprobs = FlattenLogprobs()
    logprobs.append(LOGPROBS_ONE_POSITION_0)
    logprobs.append(LOGPROBS_ONE_POSITION_1)
    assert logprobs.start_indices == [0, 1]
    assert logprobs.end_indices == [1, 3]
    assert logprobs.token_ids == [1, 2, 3]
    assert logprobs.logprobs == [0.1, 0.2, 0.3]
    assert logprobs.ranks == [10, 20, 30]
    assert logprobs.decoded_tokens == ["10", "20", "30"]

    logprobs.append(LOGPROBS_ONE_POSITION_2)
    assert logprobs.start_indices == [0, 1, 3]
    assert logprobs.end_indices == [1, 3, 6]
    assert logprobs.token_ids == [1, 2, 3, 4, 5, 6]
    assert logprobs.logprobs == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert logprobs.ranks == [10, 20, 30, 40, 50, 60]
    assert logprobs.decoded_tokens == ["10", "20", "30", "40", "50", "60"]


def test_flatten_logprobs_extend() -> None:
    logprobs = FlattenLogprobs()
    # Extend with list[LogprobsOnePosition]
    logprobs.extend([LOGPROBS_ONE_POSITION_2, LOGPROBS_ONE_POSITION_0])
    assert logprobs.start_indices == [0, 3]
    assert logprobs.end_indices == [3, 4]
    assert logprobs.token_ids == [4, 5, 6, 1]
    assert logprobs.logprobs == [0.4, 0.5, 0.6, 0.1]
    assert logprobs.ranks == [40, 50, 60, 10]
    assert logprobs.decoded_tokens == ["40", "50", "60", "10"]

    other_logprobs = FlattenLogprobs()
    other_logprobs.extend([LOGPROBS_ONE_POSITION_1, LOGPROBS_ONE_POSITION_0])
    # Extend with another FlattenLogprobs
    logprobs.extend(other_logprobs)
    assert logprobs.start_indices == [0, 3, 4, 6]
    assert logprobs.end_indices == [3, 4, 6, 7]
    assert logprobs.token_ids == [4, 5, 6, 1, 2, 3, 1]
    assert logprobs.logprobs == [0.4, 0.5, 0.6, 0.1, 0.2, 0.3, 0.1]
    assert logprobs.ranks == [40, 50, 60, 10, 20, 30, 10]
    assert logprobs.decoded_tokens == ["40", "50", "60", "10", "20", "30", "10"]


def test_flatten_logprobs_access() -> None:
    logprobs = FlattenLogprobs()
    logprobs.extend(
        [LOGPROBS_ONE_POSITION_1, LOGPROBS_ONE_POSITION_2, LOGPROBS_ONE_POSITION_0]
    )
    assert logprobs.start_indices == [0, 2, 5]
    assert logprobs.end_indices == [2, 5, 6]
    assert logprobs.token_ids == [2, 3, 4, 5, 6, 1]
    assert logprobs.logprobs == [0.2, 0.3, 0.4, 0.5, 0.6, 0.1]
    assert logprobs.ranks == [20, 30, 40, 50, 60, 10]
    assert logprobs.decoded_tokens == ["20", "30", "40", "50", "60", "10"]

    # Test __len__
    assert len(logprobs) == 3

    # Test __iter__
    for actual_logprobs, expected_logprobs in zip(
        logprobs,
        [LOGPROBS_ONE_POSITION_1, LOGPROBS_ONE_POSITION_2, LOGPROBS_ONE_POSITION_0],
    ):
        assert actual_logprobs == expected_logprobs

    # Test __getitem__ : single item
    assert logprobs[0] == LOGPROBS_ONE_POSITION_1
    assert logprobs[1] == LOGPROBS_ONE_POSITION_2
    assert logprobs[2] == LOGPROBS_ONE_POSITION_0

    # Test __getitem__ : slice
    logprobs02 = logprobs[:2]
    assert len(logprobs02) == 2
    assert logprobs02[0] == LOGPROBS_ONE_POSITION_1
    assert logprobs02[1] == LOGPROBS_ONE_POSITION_2
    assert logprobs02.start_indices == [0, 2]
    assert logprobs02.end_indices == [2, 5]
    assert logprobs02.token_ids == [2, 3, 4, 5, 6]
    assert logprobs02.logprobs == [0.2, 0.3, 0.4, 0.5, 0.6]
    assert logprobs02.ranks == [20, 30, 40, 50, 60]
    assert logprobs02.decoded_tokens == ["20", "30", "40", "50", "60"]
    logprobs_last2 = logprobs[-2:]
    assert len(logprobs_last2) == 2
    assert logprobs_last2[0] == LOGPROBS_ONE_POSITION_2
    assert logprobs_last2[1] == LOGPROBS_ONE_POSITION_0
    assert logprobs_last2.start_indices == [0, 3]
    assert logprobs_last2.end_indices == [3, 4]
    assert logprobs_last2.token_ids == [4, 5, 6, 1]
    assert logprobs_last2.logprobs == [0.4, 0.5, 0.6, 0.1]
    assert logprobs_last2.ranks == [40, 50, 60, 10]
    assert logprobs_last2.decoded_tokens == ["40", "50", "60", "10"]
