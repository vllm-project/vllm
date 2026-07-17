# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.reasoning.utils import run_reasoning_extraction_mistral
from vllm.reasoning import ReasoningParser, ReasoningParserManager
from vllm.tokenizers.mistral import MistralTokenizer

_PARSER_NAME = "mistral"
_MODEL_V13 = "mistralai/Magistral-Small-2509"
_MODEL_V11 = "mistralai/Magistral-Small-2506"


@pytest.fixture(scope="module")
def mistral_tokenizer() -> MistralTokenizer:
    """v13 Magistral tokenizer with `[THINK]`/`[/THINK]` special tokens."""
    return MistralTokenizer.from_pretrained(_MODEL_V13)


@pytest.fixture(scope="module")
def mistral_v11_tokenizer() -> MistralTokenizer:
    """v11 Magistral tokenizer using plain-text `<think>`/`</think>` tags."""
    return MistralTokenizer.from_pretrained(_MODEL_V11)


def _encode_v13(tokenizer: MistralTokenizer, output: str) -> list[int]:
    """Encode output string with `[THINK]`/`[/THINK]` markers into token IDs.

    `[THINK]` and `[/THINK]` placeholders in `output` are replaced by their
    special-token IDs; all surrounding text is encoded with the base tokenizer.
    Mirrors the encoding helper from the deleted
    `tests/reasoning/test_mistral_reasoning_parser.py`.
    """
    think_start = "[THINK]"
    think_end = "[/THINK]"
    len_start = len(think_start)
    len_end = len(think_end)

    index_start = output.find(think_start)
    index_end = output.find(think_end)

    out: list[int] = []
    if index_start != -1:
        out += tokenizer.tokenizer.encode(output[:index_start], False, False)
        out += [tokenizer.instruct.BEGIN_THINK]

        if index_end != -1:
            middle = output[index_start + len_start : index_end]
            suffix = output[index_end + len_end :]
            out += tokenizer.tokenizer.encode(middle, False, False)
            out += [tokenizer.instruct.END_THINK]
            out += tokenizer.tokenizer.encode(suffix, False, False)
        else:
            out += tokenizer.tokenizer.encode(
                output[index_start + len_start :], False, False
            )
    elif index_end != -1:
        out += tokenizer.tokenizer.encode(output[:index_end], False, False)
        out += [tokenizer.instruct.END_THINK]
        out += tokenizer.tokenizer.encode(output[index_end + len_end :], False, False)
    else:
        out += tokenizer.tokenizer.encode(output, False, False)
    return out


_TEST_CASES = [
    # v13 special-token encoding: [THINK]…[/THINK] with trailing content.
    pytest.param(
        False,
        "[THINK]r[/THINK]c",
        "r",
        "c",
        id="v13_basic",
    ),
    pytest.param(
        True,
        "[THINK]r[/THINK]c",
        "r",
        "c",
        id="v13_basic_streaming",
    ),
    # No trailing content after reasoning ends.
    pytest.param(
        False,
        "[THINK]r[/THINK]",
        "r",
        None,
        id="v13_no_trailing_content",
    ),
    pytest.param(
        True,
        "[THINK]r[/THINK]",
        "r",
        None,
        id="v13_no_trailing_content_streaming",
    ),
    # Multi-line reasoning and multi-line content.
    pytest.param(
        False,
        "[THINK]a\nb[/THINK]c\nd",
        "a\nb",
        "c\nd",
        id="v13_multiline",
    ),
    pytest.param(
        True,
        "[THINK]a\nb[/THINK]c\nd",
        "a\nb",
        "c\nd",
        id="v13_multiline_streaming",
    ),
    # No reasoning at all: output is pure content.
    pytest.param(
        False,
        "hello world",
        None,
        "hello world",
        id="v13_no_reasoning",
    ),
    pytest.param(
        True,
        "hello world",
        None,
        "hello world",
        id="v13_no_reasoning_streaming",
    ),
    # [THINK] opened but never closed; reasoning accumulates, content is None.
    pytest.param(
        False,
        "[THINK]r",
        "r",
        None,
        id="v13_think_no_end",
    ),
    pytest.param(
        True,
        "[THINK]r",
        "r",
        None,
        id="v13_think_no_end_streaming",
    ),
    # Stray [/THINK] with no matching [THINK]: no reasoning, content follows.
    pytest.param(
        False,
        "[/THINK]This is the rest",
        None,
        "This is the rest",
        id="v13_stray_think_end",
    ),
    pytest.param(
        True,
        "[/THINK]This is the rest",
        None,
        "This is the rest",
        id="v13_stray_think_end_streaming",
    ),
    # Content before [THINK]: leading segment joins with trailing segment.
    pytest.param(
        False,
        "Before\n[THINK]r[/THINK]\nAfter",
        "r",
        "Before\n\nAfter",
        id="v13_leading_content",
    ),
    pytest.param(
        True,
        "Before\n[THINK]r[/THINK]\nAfter",
        "r",
        "Before\n\nAfter",
        id="v13_leading_content_streaming",
    ),
    # Empty output: no reasoning, no content.
    pytest.param(
        False,
        "",
        None,
        None,
        id="v13_empty",
    ),
    pytest.param(
        True,
        "",
        None,
        None,
        id="v13_empty_streaming",
    ),
]


@pytest.mark.parametrize(
    "streaming, output, expected_reasoning, expected_content",
    _TEST_CASES,
)
def test_mistral_reasoning_v13(
    streaming: bool,
    output: str,
    expected_reasoning: str | None,
    expected_content: str | None,
    mistral_tokenizer: MistralTokenizer,
) -> None:
    output_tokens = _encode_v13(mistral_tokenizer, output)
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(_PARSER_NAME)(
        mistral_tokenizer
    )
    reasoning, content = run_reasoning_extraction_mistral(
        parser, output_tokens, streaming=streaming
    )
    assert reasoning == expected_reasoning
    assert content == expected_content


def test_system_prompt_think_ignored(
    mistral_tokenizer: MistralTokenizer,
) -> None:
    """Streaming content extraction must not
    include `[THINK]`/`[/THINK]` examples from the system prompt.
    """
    output = "[THINK]real[/THINK]answer"
    output_tokens = _encode_v13(mistral_tokenizer, output)
    for streaming in (False, True):
        parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(
            _PARSER_NAME
        )(mistral_tokenizer)
        reasoning, content = run_reasoning_extraction_mistral(
            parser, output_tokens, streaming=streaming
        )
        assert reasoning == "real"
        assert content == "answer"


def test_reasoning_with_tool_calls(
    mistral_tokenizer: MistralTokenizer,
) -> None:
    """Reasoning extraction with a tool call in the generated output."""
    tool_calls_token_id = mistral_tokenizer.get_vocab().get("[TOOL_CALLS]")
    if tool_calls_token_id is None:
        pytest.skip("Tokenizer does not have [TOOL_CALLS] in vocab")

    # Token sequence representing: [THINK]r[/THINK][TOOL_CALLS]f{"a":1}
    output_tokens = (
        [mistral_tokenizer.instruct.BEGIN_THINK]
        + mistral_tokenizer.tokenizer.encode("r", False, False)
        + [mistral_tokenizer.instruct.END_THINK]
        + [tool_calls_token_id]
        + mistral_tokenizer.tokenizer.encode('f{"a":1}', False, False)
    )
    for streaming in (False, True):
        parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(
            _PARSER_NAME
        )(mistral_tokenizer)
        reasoning, _content = run_reasoning_extraction_mistral(
            parser, output_tokens, streaming=streaming
        )
        assert reasoning == "r"
        assert "[TOOL_CALLS]" not in (reasoning or "")


@pytest.mark.parametrize("streaming", [False, True], ids=["nonstream", "stream"])
def test_reasoning_v11_plain_text_think(
    mistral_v11_tokenizer: MistralTokenizer, streaming: bool
) -> None:
    """v11 tokenizers use plain-text `<think>`/`</think>` (no special tokens)."""
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(_PARSER_NAME)(
        mistral_v11_tokenizer
    )

    tokens = mistral_v11_tokenizer.tokenizer.encode("<think>r</think>c", False, False)
    reasoning, content = run_reasoning_extraction_mistral(
        parser, tokens, streaming=streaming
    )
    assert reasoning == "r"
    assert content == "c"


def test_is_reasoning_end(
    mistral_tokenizer: MistralTokenizer,
) -> None:
    """is_reasoning_end returns True iff reasoning has finished."""
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(_PARSER_NAME)(
        mistral_tokenizer
    )

    # Complete reasoning: END_THINK present → True.
    complete_ids = _encode_v13(mistral_tokenizer, "[THINK]r[/THINK]c")
    assert parser.is_reasoning_end(complete_ids)

    # Open-only: no END_THINK, BEGIN_THINK present → False.
    open_ids = _encode_v13(mistral_tokenizer, "[THINK]r")
    assert not parser.is_reasoning_end(open_ids)

    # No reasoning markers at all → False.
    no_reasoning_ids = _encode_v13(mistral_tokenizer, "hello")
    assert not parser.is_reasoning_end(no_reasoning_ids)

    # Explicit END_THINK closes reasoning → True.
    explicit_end_ids = (
        [mistral_tokenizer.instruct.BEGIN_THINK]
        + mistral_tokenizer.tokenizer.encode("r", False, False)
        + [mistral_tokenizer.instruct.END_THINK]
    )
    assert parser.is_reasoning_end(explicit_end_ids)

    # [TOOL_CALLS] acts as implicit reasoning-end marker.
    tool_calls_token_id = mistral_tokenizer.get_vocab().get("[TOOL_CALLS]")
    if tool_calls_token_id is None:
        pytest.skip("Tokenizer does not have [TOOL_CALLS] in vocab")

    implicit_end_ids = (
        [mistral_tokenizer.instruct.BEGIN_THINK]
        + mistral_tokenizer.tokenizer.encode("r", False, False)
        + [tool_calls_token_id]
    )
    assert parser.is_reasoning_end(implicit_end_ids)
