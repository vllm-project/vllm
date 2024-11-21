import pytest

from vllm.multimodal.processing import iter_token_matches, iter_token_runs


# yapf: disable
@pytest.mark.parametrize(
    ("token_ids", "expected"),
    [
        ([], []),
        (
            [32000, 32000, 32000],
            [{ "token_id": 32000, "start_idx": 0, "length": 3 }],
        ),
        (
            [9833, 28747, 32000, 32000, 32000, 9833, 28747, 32000, 32000, 918],
            [
                { "token_id": 9833, "start_idx": 0, "length": 1 },
                { "token_id": 28747, "start_idx": 1, "length": 1 },
                { "token_id": 32000, "start_idx": 2, "length": 3 },
                { "token_id": 9833, "start_idx": 5, "length": 1 },
                { "token_id": 28747, "start_idx": 6, "length": 1 },
                { "token_id": 32000, "start_idx": 7, "length": 2 },
                { "token_id": 918, "start_idx": 9, "length": 1 },
            ],
        ),
    ],
)
# yapf: enable
def test_iter_token_runs(token_ids, expected):
    result = list(iter_token_runs(token_ids))

    # Manually constructed results
    assert result == expected

    # Invariants
    assert sum(run_info["length"] for run_info in result) == len(token_ids)


# yapf: disable
@pytest.mark.parametrize(
    ("token_ids", "match_ids", "expected"),
    [
        ([], [], [{ "start_idx": 0, "end_idx": 0 }]),
        ([], [32000], []),
        (
            [32000, 32000, 32000],
            [32000],
            [
                { "start_idx": 0, "end_idx": 1 },
                { "start_idx": 1, "end_idx": 2 },
                { "start_idx": 2, "end_idx": 3 },
            ],
        ),
        (
            [32000, 32000, 32000],
            [32000, 32000],
            [
                { "start_idx": 0, "end_idx": 2 },
                { "start_idx": 1, "end_idx": 3 },
            ],
        ),
        (
            [32000, 32000, 32000],
            [32000, 32000, 32000],
            [{ "start_idx": 0, "end_idx": 3 }],
        ),
        (
            [9833, 28747, 32000, 32000, 32000, 9833, 28747, 32000, 32000, 918],
            [28747, 32000],
            [
                { "start_idx": 1, "end_idx": 3 },
                { "start_idx": 6, "end_idx": 8 },
            ],
        ),
        (
            [9833, 28747, 32000, 32000, 32000, 9833, 28747, 32000, 32000, 918],
            [28747, 32000, 32000, 32000],
            [
                { "start_idx": 1, "end_idx": 5 },
            ],
        ),
        (
            [9833, 28747, 32000, 32000, 32000, 9833, 28747, 32000, 32000, 918],
            [28747, 0, 32000],
            [],
        ),
    ],
)
# yapf: enable
def test_iter_token_matches(token_ids, match_ids, expected):
    result = list(iter_token_matches(token_ids, match_ids))

    # Manually constructed results
    assert [item._asdict() for item in result] == expected

    # Invariants
    match_lens = [end - start for start, end in result]
    print("match_lens:", match_lens)  # Only displayed on error
    assert all(match_len == len(match_ids) for match_len in match_lens)
