# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.reasoning.gemma4_utils import parse_thinking_output

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("4", {"thinking": None, "answer": "4"}),
        (
            "<|channel>thought\n2+2<channel|>4",
            {"thinking": "2+2", "answer": "4"},
        ),
        (
            "<|channel>thought\n2+2",
            {"thinking": "2+2", "answer": ""},
        ),
        (
            "<|channel>thought\n",
            {"thinking": "", "answer": ""},
        ),
        (
            "prefix\n<|channel>thought\n2+2",
            {"thinking": "2+2", "answer": "prefix"},
        ),
        (
            "thought\n4<turn|>",
            {"thinking": None, "answer": "4"},
        ),
    ],
)
def test_parse_thinking_output(text: str, expected: dict[str, str | None]):
    assert parse_thinking_output(text) == expected
