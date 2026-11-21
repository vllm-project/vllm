# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.reasoning.gemma4_utils import parse_thinking_output

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("4", {"thinking": None, "answer": "4"}),
        ("4<turn|><eos>", {"thinking": None, "answer": "4"}),
        (
            "<|channel>thought\n2+2<channel|>4",
            {"thinking": "2+2", "answer": "4"},
        ),
        (
            "<|channel>thought<channel|>4",
            {"thinking": "", "answer": "4"},
        ),
        (
            "<|channel>internal thought<channel|>4",
            {"thinking": "internal thought", "answer": "4"},
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
            "<|channel>thought",
            {"thinking": "", "answer": ""},
        ),
        (
            "<|channel>tho",
            {"thinking": "", "answer": ""},
        ),
        (
            "<|channel>",
            {"thinking": "", "answer": ""},
        ),
        (
            "prefix\n<|channel>thought\n2+2",
            {"thinking": "2+2", "answer": "prefix"},
        ),
        (
            "<|channel>thought\n2+2<turn|>",
            {"thinking": "2+2", "answer": ""},
        ),
        (
            "<|channel>thought\n2+2<eos>",
            {"thinking": "2+2", "answer": ""},
        ),
        (
            "<|channel>thought\n2+2<turn|><eos>",
            {"thinking": "2+2", "answer": ""},
        ),
        (
            "<|channel>thought<turn|><eos>",
            {"thinking": "", "answer": ""},
        ),
        (
            "answer mentions <|channel> literally",
            {"thinking": None, "answer": "answer mentions <|channel> literally"},
        ),
        (
            "answer mentions <|channel> thought",
            {"thinking": None, "answer": "answer mentions <|channel> thought"},
        ),
        (
            "answer mentions <|channel>thought in prose",
            {"thinking": None, "answer": "answer mentions <|channel>thought in prose"},
        ),
        (
            "mention <|channel> literally\n<|channel>thought\n2+2",
            {"thinking": "2+2", "answer": "mention <|channel> literally"},
        ),
        (
            "<|channel>thought\nA<channel|>ans <|channel>thought\nB",
            {"thinking": "A\nB", "answer": "ans"},
        ),
        (
            "<|channel>thought\nA<channel|>ans <|channel>thought\nB<turn|><eos>",
            {"thinking": "A\nB", "answer": "ans"},
        ),
        (
            "<|channel>thought\nA<channel|>ans <|channel> literally",
            {"thinking": "A", "answer": "ans <|channel> literally"},
        ),
        (
            "<|channel>thought\nA<channel|>ans <|channel>thought\n",
            {"thinking": "A", "answer": "ans"},
        ),
        (
            "thought\n4<turn|>",
            {"thinking": None, "answer": "4"},
        ),
    ],
)
def test_parse_thinking_output(text: str, expected: dict[str, str | None]):
    assert parse_thinking_output(text) == expected
