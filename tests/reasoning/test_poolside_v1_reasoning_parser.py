# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.reasoning.poolside_v1_reasoning_parser import PoolsideV1ReasoningParser

pytestmark = pytest.mark.skip_global_cleanup

START_ID = 1
END_ID = 2
TOOL_START_ID = 4
TOOL_END_ID = 5


class _StubTokenizer:
    def get_vocab(self) -> dict[str, int]:
        return {
            "<think>": START_ID,
            "</think>": END_ID,
            "<assistant>": 3,
            "<tool_call>": TOOL_START_ID,
            "</tool_call>": TOOL_END_ID,
        }


def _make_parser(*, enable_thinking: bool) -> PoolsideV1ReasoningParser:
    return PoolsideV1ReasoningParser(
        _StubTokenizer(),
        chat_template_kwargs={"enable_thinking": enable_thinking},
    )


@pytest.mark.parametrize(
    "enable_thinking,token_ids,expected",
    [
        pytest.param(True, [10, 11, END_ID, 20], 2, id="prompt_opened_span"),
        pytest.param(
            True,
            [10, 11, END_ID, 20, START_ID, 21],
            2,
            id="prompt_opened_span_with_late_start",
        ),
        pytest.param(True, [10, 11], 2, id="prompt_opened_truncation"),
        pytest.param(True, [9, START_ID, 10, 11, END_ID, 20], 2, id="explicit_span"),
        pytest.param(False, [20, 21], 0, id="disabled_message"),
        pytest.param(False, [TOOL_START_ID, 30, TOOL_END_ID], 0, id="disabled_tool"),
        pytest.param(True, [END_ID, 20], 0, id="prompt_opened_empty_span"),
    ],
)
def test_count_reasoning_tokens(
    enable_thinking: bool,
    token_ids: list[int],
    expected: int,
) -> None:
    parser = _make_parser(enable_thinking=enable_thinking)

    assert parser.count_reasoning_tokens(token_ids) == expected
