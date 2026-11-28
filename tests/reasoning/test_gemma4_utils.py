# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.reasoning.gemma4_utils import parse_thinking_output


@pytest.mark.parametrize(
    ("text", "expected_thinking", "expected_answer"),
    [
        pytest.param("4", None, "4", id="plain_answer"),
        pytest.param("", None, "", id="empty_string"),
        pytest.param("4<turn|>", None, "4", id="plain_turn_sentinel"),
        pytest.param("4<eos>", None, "4", id="plain_eos_sentinel"),
        pytest.param("4<turn|><eos>", None, "4", id="plain_stacked_sentinels"),
        pytest.param("4<eos><turn|>", None, "4", id="plain_reversed_sentinels"),
        pytest.param(
            "4 <turn|> <eos> ", None, "4", id="plain_spaced_stacked_sentinels"
        ),
        pytest.param("4<turn|><turn|>", None, "4", id="plain_repeated_turn_sentinel"),
        pytest.param(
            "4 <turn|> mid", None, "4 <turn|> mid", id="interior_sentinel_preserved"
        ),
        pytest.param("thought\n4", None, "4", id="spurious_thought_label"),
        pytest.param(
            "thought about it", None, "thought about it", id="thought_word_preserved"
        ),
        pytest.param(
            "<|channel>thought\n2+2<channel|>4",
            "2+2",
            "4",
            id="closed_thought",
        ),
        pytest.param(
            "<|channel>thought\n2+2<channel|>4<turn|><eos>",
            "2+2",
            "4",
            id="closed_thought_stacked_sentinels",
        ),
        pytest.param(
            "<|channel>thought\n<channel|>4",
            "",
            "4",
            id="closed_empty_thought",
        ),
        pytest.param(
            "<|channel>thought\nthought<channel|>4",
            "thought",
            "4",
            id="closed_thought_body_is_word_thought",
        ),
        pytest.param(
            "2+2<channel|>4",
            "2+2",
            "4",
            id="closed_missing_start_tag",
        ),
        pytest.param(
            "<|channel>thought\n2+2<channel|>The closing tag is <channel|> here",
            "2+2",
            "The closing tag is <channel|> here",
            id="literal_end_tag_in_answer",
        ),
        pytest.param(
            "<|channel>thought\n2+2",
            "2+2",
            "",
            id="unfinished_thought",
        ),
        pytest.param(
            "<|channel>thought\r\n2+2",
            "2+2",
            "",
            id="unfinished_thought_crlf",
        ),
        pytest.param(
            "<|channel>thought\nstep 1\nstep 2",
            "step 1\nstep 2",
            "",
            id="unfinished_multiline_thought",
        ),
        pytest.param(
            "<|channel>thought\n2+2<turn|>",
            "2+2",
            "",
            id="unfinished_thought_turn_sentinel",
        ),
        pytest.param(
            "<|channel>thought\n2+2<eos>",
            "2+2",
            "",
            id="unfinished_thought_eos_sentinel",
        ),
        pytest.param(
            "<|channel>thought\n2+2<turn|><eos>",
            "2+2",
            "",
            id="unfinished_thought_stacked_sentinels",
        ),
        pytest.param(
            "<|channel>thought\n  \n  <turn|><eos>",
            "",
            "",
            id="unfinished_whitespace_thought_with_sentinels",
        ),
        pytest.param(
            "<|channel>thought\n",
            "",
            "",
            id="unfinished_empty_thought_newline",
        ),
        pytest.param(
            "<|channel>thought",
            "",
            "",
            id="unfinished_truncated_thought_role",
        ),
        pytest.param(
            "<|channel>tho",
            "",
            "",
            id="unfinished_truncated_partial_role",
        ),
        pytest.param(
            "<|channel>",
            "",
            "",
            id="unfinished_bare_start_tag",
        ),
        pytest.param(
            "<|channel>thought\nthought",
            "thought",
            "",
            id="unfinished_thought_body_is_word_thought",
        ),
        pytest.param(
            "answer mentions <|channel> literally",
            None,
            "answer mentions <|channel> literally",
            id="literal_start_tag_in_plain_text",
        ),
        pytest.param(
            "Use <|channel> to open a channel",
            None,
            "Use <|channel> to open a channel",
            id="literal_start_tag_with_prose",
        ),
        pytest.param(
            "<|channel>thought\nA<channel|>ans <|channel>thought\nB",
            "A\nB",
            "ans",
            id="closed_then_unfinished_thought",
        ),
        pytest.param(
            "<|channel>thought\nA<channel|><|channel>thought\n",
            "A",
            "",
            id="closed_then_empty_unfinished_thought",
        ),
        pytest.param(
            "<|channel>thought\nA<channel|>mid <|channel>thought\nB<channel|>end",
            "A\nB",
            "mid end",
            id="two_closed_thoughts",
        ),
        pytest.param(
            "mentions <|channel> literally then <|channel>thought\n2+2",
            "2+2",
            "mentions <|channel> literally then",
            id="literal_mention_before_unfinished_thought",
        ),
    ],
)
def test_parse_thinking_output(
    text: str, expected_thinking: str | None, expected_answer: str
) -> None:
    assert parse_thinking_output(text) == {
        "thinking": expected_thinking,
        "answer": expected_answer,
    }
