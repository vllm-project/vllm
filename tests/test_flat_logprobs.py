# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for FlatLogprobs.append_fast with list.extend optimization."""

import itertools

import pytest

from vllm.logprobs import (
    FlatLogprobs,
    append_logprobs_for_next_position,
    create_prompt_logprobs,
    create_sample_logprobs,
)


class TestFlatLogprobsAppendFast:
    def test_basic_extend(self):
        flat = FlatLogprobs()
        token_ids = [100, 200, 300, 400, 500, 600]
        logprobs = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6]
        ranks = [3, 1, 2, 3, 4, 5]
        decoded = ["a", "b", "c", "d", "e", "f"]

        flat.append_fast(token_ids, logprobs, ranks, decoded)

        assert len(flat) == 1
        assert flat.token_ids == token_ids
        assert flat.logprobs == logprobs
        assert flat.ranks == ranks
        assert flat.decoded_tokens == decoded
        assert flat.start_indices == [0]
        assert flat.end_indices == [6]

    def test_multiple_positions(self):
        flat = FlatLogprobs()

        for pos in range(3):
            flat.append_fast(
                [pos * 10 + i for i in range(4)],
                [-0.1 * (i + 1) for i in range(4)],
                [pos + 1, 1, 2, 3],
                [f"t{pos}_{i}" for i in range(4)],
            )

        assert len(flat) == 3
        assert flat.start_indices == [0, 4, 8]
        assert flat.end_indices == [4, 8, 12]
        assert len(flat.token_ids) == 12

        item0 = flat[0]
        assert 0 in item0
        assert item0[0].rank == 1

    def test_getitem_after_extend(self):
        flat = FlatLogprobs()
        flat.append_fast(
            [1000, 1001, 1002],
            [-0.5, -1.0, -1.5],
            [3, 1, 2],
            ["hello", "world", "foo"],
        )

        item = flat[0]
        assert item[1000].logprob == -0.5
        assert item[1000].rank == 3
        assert item[1000].decoded_token == "hello"
        assert item[1001].rank == 1
        assert item[1002].rank == 2


class TestAppendLogprobsForNextPosition:
    def test_flat_oversized_input(self):
        flat = create_sample_logprobs(flat_logprobs=True)
        assert isinstance(flat, FlatLogprobs)

        # Simulate max_num_logprobs=10 but request only wants 5
        token_ids = list(range(100, 111))  # 11 elements
        logprobs = [-0.1 * i for i in range(11)]
        decoded = [f"tok{i}" for i in range(11)]

        append_logprobs_for_next_position(
            flat, token_ids, logprobs, decoded, rank=3, num_logprobs=5
        )

        assert len(flat) == 1
        assert len(flat.token_ids) == 6
        assert flat.token_ids == [100, 101, 102, 103, 104, 105]
        assert flat.ranks == [3, 1, 2, 3, 4, 5]

    def test_flat_exact_size_input(self):
        flat = create_sample_logprobs(flat_logprobs=True)

        token_ids = list(range(100, 106))  # exactly 6 elements
        logprobs = [-0.1 * i for i in range(6)]
        decoded = [f"tok{i}" for i in range(6)]

        append_logprobs_for_next_position(
            flat, token_ids, logprobs, decoded, rank=2, num_logprobs=5
        )

        assert len(flat) == 1
        assert len(flat.token_ids) == 6
        assert flat.ranks == [2, 1, 2, 3, 4, 5]

    def test_flat_nones_decoded(self):
        flat = create_sample_logprobs(flat_logprobs=True)
        NONES = itertools.repeat(None)

        token_ids = list(range(100, 112))
        logprobs = [-0.1 * i for i in range(12)]

        append_logprobs_for_next_position(
            flat, token_ids, logprobs, NONES, rank=1, num_logprobs=5
        )

        assert len(flat) == 1
        assert len(flat.token_ids) == 6
        assert flat.decoded_tokens == [None] * 6

    def test_flat_num_logprobs_minus_one(self):
        flat = create_sample_logprobs(flat_logprobs=True)

        token_ids = [100, 101, 102]
        logprobs = [-0.1, -0.2, -0.3]
        decoded = ["a", "b", "c"]

        append_logprobs_for_next_position(
            flat, token_ids, logprobs, decoded, rank=1, num_logprobs=-1
        )

        assert len(flat) == 1
        assert len(flat.token_ids) == 3
        assert flat.ranks == [1, 1, 2]

    def test_dict_path_unchanged(self):
        logprobs_list = create_sample_logprobs(flat_logprobs=False)
        assert isinstance(logprobs_list, list)

        token_ids = [100, 101, 102, 103, 104, 105]
        logprobs = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6]
        decoded = ["a", "b", "c", "d", "e", "f"]

        append_logprobs_for_next_position(
            logprobs_list, token_ids, logprobs, decoded, rank=2, num_logprobs=3
        )

        assert len(logprobs_list) == 1
        item = logprobs_list[0]
        assert len(item) == 4  # sampled + 3 top-k
        assert item[100].rank == 2
        assert item[101].rank == 1

    def test_prompt_logprobs_flat(self):
        prompt_lps = create_prompt_logprobs(flat_logprobs=True)
        assert isinstance(prompt_lps, FlatLogprobs)
        assert len(prompt_lps) == 1  # starts with None entry

        append_logprobs_for_next_position(
            prompt_lps,
            [50, 51, 52, 53],
            [-1.0, -2.0, -3.0, -4.0],
            ["x", "y", "z", "w"],
            rank=1,
            num_logprobs=3,
        )

        assert len(prompt_lps) == 2
        item = prompt_lps[1]
        assert len(item) == 4
        assert item[50].logprob == -1.0

    def test_flat_empty_token_ids(self):
        flat = create_sample_logprobs(flat_logprobs=True)

        append_logprobs_for_next_position(flat, [], [], [], rank=1, num_logprobs=5)

        assert len(flat) == 1
        assert len(flat.token_ids) == 0
        assert len(flat.ranks) == 0
        assert flat.start_indices == [0]
        assert flat.end_indices == [0]

    def test_flat_generator_decoded_tokens(self):
        flat = create_sample_logprobs(flat_logprobs=True)

        def decoded_gen():
            yield from ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]

        append_logprobs_for_next_position(
            flat,
            list(range(100, 106)),
            [-0.1 * i for i in range(6)],
            decoded_gen(),
            rank=2,
            num_logprobs=3,
        )

        assert len(flat) == 1
        assert len(flat.token_ids) == 4
        assert flat.decoded_tokens == ["alpha", "beta", "gamma", "delta"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
