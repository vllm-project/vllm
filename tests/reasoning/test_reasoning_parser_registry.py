# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The structured-output manager applies the grammar once `is_reasoning_end` or
`is_reasoning_end_streaming` reports the end of reasoning. ReasoningParser is
not an ABC, so a parser that keeps the abstract `is_reasoning_end` still
instantiates, returns None, and the grammar is never applied."""

import pytest

from vllm.reasoning import ReasoningParserManager


@pytest.mark.parametrize("name", ReasoningParserManager.list_registered())
def test_parser_implements_is_reasoning_end(name: str):
    parser_cls = ReasoningParserManager.get_reasoning_parser(name)
    assert not getattr(parser_cls.is_reasoning_end, "__isabstractmethod__", False)
