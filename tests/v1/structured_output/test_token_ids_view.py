# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.structured_output.utils import TokenIdsView

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    ("prefix", "suffix", "length"),
    [
        ([1, 2, 3, 4], [5, 6], 6),
        ([1, 2, 3, 4], [5, 6], 5),
        ([1, 2, 3, 4], [5, 6], 4),
        ([1, 2, 3, 4], [5, 6], 2),
        ([1, 2, 3, 4], [], 4),
        ([], [5, 6], 2),
        ([1, 2, 1], [2, 1], 5),
        ([1, 2], [3], 0),
    ],
)
def test_token_ids_view_matches_list(prefix, suffix, length):
    expected = (prefix + suffix)[:length]
    view = TokenIdsView(prefix, suffix, length)

    assert len(view) == len(expected)
    assert list(view) == expected
    for i in range(-len(expected), len(expected)):
        assert view[i] == expected[i]
    for i in (len(expected), -len(expected) - 1):
        with pytest.raises(IndexError):
            view[i]
    bounds = (None, *range(-8, 9))
    for start in bounds:
        for stop in bounds:
            for step in (None, 2, -1):
                assert view[start:stop:step] == expected[start:stop:step]
    for value in range(8):
        assert (value in view) == (value in expected)
        for start, stop in ((0, None), (1, None), (0, -1), (2, 4), (-3, None)):
            try:
                want = expected.index(
                    value, start, len(expected) if stop is None else stop
                )
            except ValueError:
                with pytest.raises(ValueError):
                    view.index(value, start, stop)
            else:
                assert view.index(value, start, stop) == want


def test_token_ids_view_rejects_bad_length():
    with pytest.raises(ValueError):
        TokenIdsView([1, 2], [3], 4)
