# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.structured_output.utils import strip_speculative_padding


def test_strips_trailing_padding():
    assert strip_speculative_padding([3, 4, -1, -1]) == [3, 4]


def test_empty_when_leading_padding():
    assert strip_speculative_padding([-1, -1]) == []


def test_passthrough_without_padding():
    assert strip_speculative_padding([3, 4]) == [3, 4]


def test_truncates_at_first_sentinel():
    assert strip_speculative_padding([3, -1, 4]) == [3]
