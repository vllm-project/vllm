# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import AutoTokenizer

from tests.reasoning.utils import run_reasoning_extraction
from vllm.reasoning import ReasoningParser, ReasoningParserManager
from vllm.reasoning.olmo3_reasoning_parser import (
    Olmo3ReasoningBuffer,
    Olmo3ReasoningParser,
)

parser_name = "olmo3"
START_REASONING = "<think>"
END_REASONING = "</think>"

NO_REASONING = {
    "output": f"{START_REASONING}{END_REASONING}No thoughts, head empty!",
    "reasoning": None,
    "content": "No thoughts, head empty!",
}

NO_REASONING_WITH_NEWLINE = {
    "output": f"{START_REASONING}\n{END_REASONING}\n\nNo thoughts, head empty!",
    "reasoning": "\n",
    "content": "\n\nNo thoughts, head empty!",
}

SIMPLE_REASONING = {
    "output": f"{START_REASONING}This is a reasoning section{END_REASONING}This is the rest",  # noqa: E501
    "reasoning": "This is a reasoning section",
    "content": "This is the rest",
}

SIMPLE_REASONING_WITH_NEWLINE = {
    "output": f"{START_REASONING} Look!\n\nI'm thinking...{END_REASONING}\nThis is the rest",  # noqa: E501
    "reasoning": " Look!\n\nI'm thinking...",
    "content": "\nThis is the rest",
}

SIMPLE_REASONING_WITH_MULTIPLE_NEWLINES = {
    "output": f"{START_REASONING}\nLook!\nI'm thinking...\n\n{END_REASONING}\n\n\nThis is the rest",  # noqa: E501
    "reasoning": "\nLook!\nI'm thinking...\n\n",
    "content": "\n\n\nThis is the rest",
}

SIMPLE_REASONING_WITH_TRAILING_SPACE = {
    "output": f"{START_REASONING}\nLook!\nI'm thinking... {END_REASONING}\nThis is the rest",  # noqa: E501
    "reasoning": "\nLook!\nI'm thinking... ",
    "content": "\nThis is the rest",
}

NO_REASONING_ONLY_END_THINK = {
    "output": f"{END_REASONING}\n\nNo thoughts, head empty!",
    "reasoning": None,
    "content": "\n\nNo thoughts, head empty!",
}

REASONING_ONLY_END_THINK = {
    "output": f"The user is asking me not to think.{END_REASONING}No thoughts!",
    "reasoning": "The user is asking me not to think.",
    "content": "No thoughts!",
}

TEST_CASES = [
    pytest.param(
        False,  # not streaming
        NO_REASONING,
        id="no_reasoning",
    ),
    pytest.param(
        False,  # not streaming
        NO_REASONING_WITH_NEWLINE,
        id="no_reasoning_with_newline",
    ),
    pytest.param(
        False,  # not streaming
        SIMPLE_REASONING,
        id="simple_reasoning",
    ),
    pytest.param(
        False,  # not streaming
        SIMPLE_REASONING_WITH_NEWLINE,
        id="simple_reasoning_with_newline",
    ),
    pytest.param(
        True,  # enable streaming
        SIMPLE_REASONING_WITH_MULTIPLE_NEWLINES,
        id="simple_reasoning_with_multiple_newlines",
    ),
    pytest.param(
        False,  # not streaming
        NO_REASONING_ONLY_END_THINK,
        id="no_reasoning_only_end_think",
    ),
    pytest.param(
        False,  # not streaming
        REASONING_ONLY_END_THINK,
        id="yes_reasoning_only_end_think",
    ),
    pytest.param(
        True,  # enable streaming
        NO_REASONING,
        id="no_reasoning_streaming",
    ),
    pytest.param(
        True,  # enable streaming
        NO_REASONING_WITH_NEWLINE,
        id="no_reasoning_with_newline_streaming",
    ),
    pytest.param(
        True,  # enable streaming
        SIMPLE_REASONING,
        id="simple_reasoning_streaming",
    ),
    pytest.param(
        True,  # enable streaming
        SIMPLE_REASONING_WITH_NEWLINE,
        id="simple_reasoning_with_newline_streaming",
    ),
    pytest.param(
        True,  # enable streaming
        SIMPLE_REASONING_WITH_MULTIPLE_NEWLINES,
        id="simple_reasoning_with_multiple_newlines_streaming",
    ),
    pytest.param(
        True,  # enable streaming
        SIMPLE_REASONING_WITH_TRAILING_SPACE,
        id="simple_reasoning_with_trailing_space_streaming",
    ),
    pytest.param(
        True,  # enable streaming
        NO_REASONING_ONLY_END_THINK,
        id="no_reasoning_only_end_think_streaming",
    ),
    pytest.param(
        True,  # enable streaming
        REASONING_ONLY_END_THINK,
        id="yes_reasoning_only_end_think_streaming",
    ),
]

# Global tokenizer initialization to avoid repeated loading
tokenizer = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Think")


@pytest.mark.parametrize("streaming, param_dict", TEST_CASES)
def test_reasoning(
    streaming: bool,
    param_dict: dict[str, str],
):
    output = tokenizer.tokenize(param_dict["output"])

    # decode everything to tokens
    model_output: list[str] = [
        tokenizer.convert_tokens_to_string([token]) for token in output
    ]
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    parser: ReasoningParser = parser_cls(tokenizer)

    reasoning, content = run_reasoning_extraction(
        reasoning_parser=parser, model_output=model_output, streaming=streaming
    )

    assert reasoning == param_dict["reasoning"]
    assert content == param_dict["content"]


class _TinyTokenizer:
    """Just enough vocab for Olmo3ReasoningParser's think_end splits."""

    def get_vocab(self):
        return {"Ġ</": 1, "</": 2, "think": 3, ">": 4}


def _stream(parser, chunks):
    reasoning, content, prev = "", "", ""
    for chunk in chunks:
        message = parser.extract_reasoning_streaming(
            prev, prev + chunk, chunk, [], [], []
        )
        prev += chunk
        if message is not None:
            reasoning += message.reasoning or ""
            content += message.content or ""
    return reasoning, content


def test_streaming_does_not_hold_a_delta_that_merely_occurs_in_a_marker():
    # #60340: "think" is a substring of "<think>"/"</think>" but cannot grow
    # into one here, so it must stream like any other text.
    parser = Olmo3ReasoningParser(_TinyTokenizer())
    chunks = ["<think>", "R", "</", "think", ">", "That is what I ", "think"]
    reasoning, content = _stream(parser, chunks)
    assert reasoning == "R"
    assert content == "That is what I think"
    assert parser.finish_streaming() is None


def test_finish_streaming_flushes_a_tail_still_awaiting_a_marker():
    # The stream can genuinely end on a marker prefix ("</"); at that point
    # the held tail is emitted as content, like the non-streaming path.
    parser = Olmo3ReasoningParser(_TinyTokenizer())
    chunks = ["<think>", "R", "</", "think", ">", "That is what I ", "</"]
    reasoning, content = _stream(parser, chunks)
    assert (reasoning, content) == ("R", "That is what I ")
    flushed = parser.finish_streaming()
    assert flushed is not None
    assert flushed.content == "</"
    assert flushed.reasoning is None
    # draining is idempotent: a second finish has nothing left to say
    assert parser.finish_streaming() is None


def test_finish_streaming_flushes_unterminated_reasoning_as_reasoning():
    # If the stream ends mid-reasoning, earlier deltas already went out as
    # reasoning, so the held tail stays reasoning for stream consistency.
    parser = Olmo3ReasoningParser(_TinyTokenizer())
    reasoning, content = _stream(parser, ["<think>", "R", "</"])
    assert (reasoning, content) == ("R", "")
    flushed = parser.finish_streaming()
    assert flushed is not None
    assert flushed.reasoning == "</"
    assert flushed.content is None


def test_buffer_holds_only_a_genuine_marker_prefix():
    buffer = Olmo3ReasoningBuffer()
    # "think" occurs inside "<think>" but is not its start: no hold
    assert buffer.add_text("think") is not None
    assert len(buffer) == 0
    # "</" can still complete into "</think>": held
    assert buffer.add_text("</") is None
    assert len(buffer) == 2
    # completing the marker flushes the buffer
    buffer.add_text("think>")
    assert len(buffer) == 0


def test_char_level_streaming_holds_only_marker_characters():
    # One character per delta. A character of the reasoning or the content
    # must surface in the call that fed it; only the markers' own characters
    # are allowed to produce no delta while they are still ambiguous.
    text = "<think>R</think>That is what I think"
    # "<think>" is text[0:6], "R" is text[6:7], "</think>" is text[7:15]
    marker_spans = set(range(0, 6)) | set(range(7, 15))

    parser = Olmo3ReasoningParser(_TinyTokenizer())
    reasoning, content, prev = "", "", ""
    silent: list[int] = []
    for i, ch in enumerate(text):
        message = parser.extract_reasoning_streaming(prev, prev + ch, ch, [], [], [])
        prev += ch
        if message is None:
            silent.append(i)
            continue
        reasoning += message.reasoning or ""
        content += message.content or ""
        # never behind: what has streamed is a prefix of the final text, so
        # no character is carried over into a later chunk
        assert "R".startswith(reasoning)
        assert "That is what I think".startswith(content)

    assert (reasoning, content) == ("R", "That is what I think")
    assert set(silent) <= marker_spans
    assert not set(silent) & set(range(15, len(text)))
    assert parser.finish_streaming() is None


def test_contained_delta_streams_in_the_call_that_fed_it():
    # A delta that occurs inside a marker but cannot grow into one ("think"
    # after "</think" already closed, or in plain content) must come back in
    # its own call, not be merged into whatever chunk follows it.
    parser = Olmo3ReasoningParser(_TinyTokenizer())
    prev = ""

    def step(chunk):
        nonlocal prev
        message = parser.extract_reasoning_streaming(
            prev, prev + chunk, chunk, [], [], []
        )
        prev += chunk
        return message

    step("<think>")
    assert step("R").reasoning == "R"
    step("</")
    step("think")
    step(">")

    contained = step("think")
    assert contained is not None, "contained delta was held back"
    assert contained.content == "think"

    follow_up = step("ing done")
    assert follow_up is not None
    assert follow_up.content == "ing done"
