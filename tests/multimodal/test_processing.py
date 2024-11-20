import pytest

from vllm.multimodal.processing import apply_placeholders, iter_token_runs


# yapf: disable
@pytest.mark.parametrize(
    ("token_ids", "expected"),
    [
        ([], []),
        ([32000, 32000, 32000], [(32000, { "offset": 0, "length": 3 })]),
        (
            [9833, 28747, 32000, 32000, 32000, 9833, 28747, 32000, 32000, 918],
            [
                (9833, { "offset": 0, "length": 1 }),
                (28747, { "offset": 1, "length": 1 }),
                (32000, { "offset": 2, "length": 3 }),
                (9833, { "offset": 5, "length": 1 }),
                (28747, { "offset": 6, "length": 1 }),
                (32000, { "offset": 7, "length": 2 }),
                (918, { "offset": 9, "length": 1 }),
            ],
        ),
    ],
)
# yapf: enable
def test_iter_token_runs(token_ids, expected):
    result = list(iter_token_runs(token_ids))

    # Invariants
    assert sum(run_info["length"] for _, run_info in result) == len(token_ids)

    # Manually constructed results
    assert result == expected


# yapf: disable
@pytest.mark.parametrize(
    (
        "token_ids", "match_ids", "replacement_id", "replacement_count",
        "expected_new_token_ids", "expected_range",
    ),
    [
        # Empty
        (
            [], [-1], +1, 0,
            [], None,
        ),
        # No match
        (
            [32000, 32000, 32000], [-1], +1, 0,
            [32000, 32000, 32000], None,
        ),
        # Match first
        (
            [-1, 32000, 32000], [-1], +1, 0,
            [32000, 32000], { "offset": 0, "length": 0 },
        ),
        (
            [-1, 32000, 32000], [-1], +1, 1,
            [+1, 32000, 32000], { "offset": 0, "length": 1 },
        ),
        (
            [-1, 32000, 32000], [-1], +1, 2,
            [+1, +1, 32000, 32000], { "offset": 0, "length": 2 },
        ),
        # Match middle
        (
            [32000, -1, 32000], [-1], +1, 0,
            [32000, 32000], { "offset": 1, "length": 0 },
        ),
        (
            [32000, -1, 32000], [-1], +1, 1,
            [32000, +1, 32000], { "offset": 1, "length": 1 },
        ),
        (
            [32000, -1, 32000], [-1], +1, 2,
            [32000, +1, +1, 32000], { "offset": 1, "length": 2},
        ),
        # Match last
        (
            [32000, 32000, -1], [-1], +1, 0,
            [32000, 32000], { "offset": 2, "length": 0 },
        ),
        (
            [32000, 32000, -1], [-1], +1, 1,
            [32000, 32000, +1], { "offset": 2, "length": 1 },
        ),
        (
            [32000, 32000, -1], [-1], +1, 2,
            [32000, 32000, +1, +1], { "offset": 2, "length": 2},
        ),
        # Match all
        (
            [32000, 32000, 32000], [32000], +1, 0,
            [32000, 32000], { "offset": 0, "length": 0 },
        ),
        (
            [32000, 32000, 32000], [32000], +1, 1,
            [+1, 32000, 32000], { "offset": 0, "length": 1 },
        ),
        (
            [32000, 32000, 32000], [32000], +1, 2,
            [+1, +1, 32000, 32000],  { "offset": 0, "length": 2 },
        ),
    ],
)
# yapf: enable
def test_apply_placeholders(
    token_ids,
    match_ids,
    replacement_id,
    replacement_count,
    expected_new_token_ids,
    expected_range,
):
    orig_token_ids = token_ids[:]

    placeholder_range = apply_placeholders(
        token_ids,
        match_ids,
        replacement_id,
        replacement_count,
    )

    # Invariants
    if placeholder_range is None:
        assert orig_token_ids == token_ids
    else:
        offset = placeholder_range["offset"]
        match_len = len(match_ids)
        repl_len = placeholder_range["length"]

        assert orig_token_ids[offset:offset + match_len] == match_ids

        repl_ids = [replacement_id] * replacement_count
        assert token_ids[offset:offset + repl_len] == repl_ids

    # Manually constructed results
    assert token_ids == expected_new_token_ids
    assert placeholder_range == expected_range
