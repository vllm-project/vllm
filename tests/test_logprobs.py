# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np

from vllm.logprobs import (
    FlatLogprobs,
    Logprob,
    LogprobsOnePosition,
    append_logprobs_for_next_position,
    create_prompt_logprobs,
    create_sample_logprobs,
)


def test_create_logprobs_non_flat() -> None:
    prompt_logprobs = create_prompt_logprobs(flat_logprobs=False)
    assert isinstance(prompt_logprobs, list)
    # Ensure first prompt position logprobs is None
    assert len(prompt_logprobs) == 1
    assert prompt_logprobs[0] is None

    sample_logprobs = create_sample_logprobs(flat_logprobs=False)
    assert isinstance(sample_logprobs, list)
    assert len(sample_logprobs) == 0


def test_create_logprobs_flat() -> None:
    prompt_logprobs = create_prompt_logprobs(flat_logprobs=True)
    assert isinstance(prompt_logprobs, FlatLogprobs)
    assert prompt_logprobs.start_indices == [0]
    assert prompt_logprobs.end_indices == [0]
    assert len(prompt_logprobs.token_ids) == 0
    assert len(prompt_logprobs.logprobs) == 0
    assert len(prompt_logprobs.ranks) == 0
    assert len(prompt_logprobs.decoded_tokens) == 0
    # Ensure first prompt position logprobs is empty
    assert len(prompt_logprobs) == 1
    assert prompt_logprobs[0] == dict()

    sample_logprobs = create_sample_logprobs(flat_logprobs=True)
    assert isinstance(sample_logprobs, FlatLogprobs)
    assert len(sample_logprobs.start_indices) == 0
    assert len(sample_logprobs.end_indices) == 0
    assert len(sample_logprobs.token_ids) == 0
    assert len(sample_logprobs.logprobs) == 0
    assert len(sample_logprobs.ranks) == 0
    assert len(sample_logprobs.decoded_tokens) == 0
    assert len(sample_logprobs) == 0


def test_append_logprobs_for_next_position_none_flat() -> None:
    logprobs = create_sample_logprobs(flat_logprobs=False)
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


def test_append_logprobs_for_next_position_flat() -> None:
    logprobs = create_sample_logprobs(flat_logprobs=True)
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
    assert isinstance(logprobs, FlatLogprobs)
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


def test_flat_logprobs_append() -> None:
    logprobs = FlatLogprobs()
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


def test_flat_logprobs_extend() -> None:
    logprobs = FlatLogprobs()
    # Extend with list[LogprobsOnePosition]
    logprobs.extend([LOGPROBS_ONE_POSITION_2, LOGPROBS_ONE_POSITION_0])
    assert logprobs.start_indices == [0, 3]
    assert logprobs.end_indices == [3, 4]
    assert logprobs.token_ids == [4, 5, 6, 1]
    assert logprobs.logprobs == [0.4, 0.5, 0.6, 0.1]
    assert logprobs.ranks == [40, 50, 60, 10]
    assert logprobs.decoded_tokens == ["40", "50", "60", "10"]

    other_logprobs = FlatLogprobs()
    other_logprobs.extend([LOGPROBS_ONE_POSITION_1, LOGPROBS_ONE_POSITION_0])
    # Extend with another FlatLogprobs
    logprobs.extend(other_logprobs)
    assert logprobs.start_indices == [0, 3, 4, 6]
    assert logprobs.end_indices == [3, 4, 6, 7]
    assert logprobs.token_ids == [4, 5, 6, 1, 2, 3, 1]
    assert logprobs.logprobs == [0.4, 0.5, 0.6, 0.1, 0.2, 0.3, 0.1]
    assert logprobs.ranks == [40, 50, 60, 10, 20, 30, 10]
    assert logprobs.decoded_tokens == ["40", "50", "60", "10", "20", "30", "10"]


def test_flat_logprobs_access() -> None:
    logprobs = FlatLogprobs()
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

    for empty_slice in (slice(0, 0), slice(3, 3), slice(10, 10)):
        empty = logprobs[empty_slice]
        assert isinstance(empty, FlatLogprobs)
        assert len(empty) == 0
        assert empty.start_indices == []
        assert empty.end_indices == []


def test_flat_logprobs_reads_like_list_across_storage_changes() -> None:
    """Random mixes of every append path read back like list[dict], across
    buffer growth, mixed widths, None positions and ranks, and values that
    widen the int32 / float32 columns; checked before and after a rank that
    is not the engine's (0, 1..k) layout."""
    rng = np.random.default_rng(0)
    flat = FlatLogprobs()
    expected: list[LogprobsOnePosition] = []

    def position(width: int, engine_ranks: bool) -> LogprobsOnePosition:
        ranks = [int(rng.integers(0, 50)), *range(1, width)][:width]
        if not engine_ranks:
            ranks = rng.choice([None, *range(width)], width).tolist()
        return {
            int(rng.integers(0, 1000)): Logprob(
                float(np.float32(-rng.random())), rank, rng.choice([None, "t"])
            )
            for rank in ranks
        }

    def check() -> None:
        assert len(flat) == len(expected)
        assert list(flat) == expected
        assert [flat[i] for i in range(-len(flat), 0)] == expected
        for index in (slice(5, 77), slice(None, None, 3), slice(-40, None)):
            assert list(flat[index]) == expected[index]
        assert len(flat.token_ids) == flat.num_entries == sum(map(len, expected))

    for step in range(300):
        if step == 150:
            check()
        engine_ranks = step < 150
        op = rng.integers(0, 4)
        if op == 0:
            width = int(rng.integers(0, 4))
            entry = position(width, engine_ranks) if rng.random() < 0.9 else None
            flat.append(entry)
            expected.append(entry or {})
        elif op == 1:
            n, width = int(rng.integers(1, 4)), int(rng.integers(1, 4))
            ids = rng.integers(0, 1000, (n, width)).astype(np.int32)
            lps = -rng.random((n, width)).astype(np.float32)
            first = rng.integers(0, 50, n)
            flat.append_rows(ids, lps, first)
            for i in range(n):
                ranks = [int(first[i]), *range(1, width)]
                expected.append(
                    {
                        t: Logprob(lp, r)
                        for t, lp, r in zip(ids[i].tolist(), lps[i].tolist(), ranks)
                    }
                )
        elif op == 2:
            entries = [position(2, engine_ranks) for _ in range(rng.integers(1, 3))]
            source = FlatLogprobs()
            source.extend(entries)
            flat.extend(source if rng.random() < 0.5 else entries)
            expected.extend(entries)
        else:
            entry = {2**40: Logprob(0.1, 3, None)} if rng.random() < 0.1 else {}
            flat.append(entry)
            expected.append(entry)

    check()
