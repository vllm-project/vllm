# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
from transformers import AutoTokenizer

from tests.reasoning.utils import DeltaMessage, run_reasoning_extraction
from vllm.reasoning import ReasoningParser, ReasoningParserManager

parser_name = "granite"
START_REASONING = "Here is my thought process:"
START_RESPONSE = "Here is my response:"

SIMPLE_REASONING = {
    "output": f"{START_REASONING}This is a reasoning section{START_RESPONSE}This is the rest",  # noqa: E501
    "reasoning": "This is a reasoning section",
    "content": "This is the rest",
}
COMPLETE_REASONING = {
    "output": f"{START_REASONING}This is a reasoning section{START_RESPONSE}",
    "reasoning": "This is a reasoning section",
    "content": None,
}
NO_REASONING = {
    "output": "This is content",
    "reasoning": None,
    "content": "This is content",
}
MULTIPLE_LINES = {
    "output": f"{START_REASONING}This\nThat{START_RESPONSE}This is the rest\nThat",
    "reasoning": "This\nThat",
    "content": "This is the rest\nThat",
}
REASONING_WITH_THINK = {
    "output": f"{START_REASONING}This is a reasoning section{START_RESPONSE}This is the rest",  # noqa: E501
    "reasoning": "This is a reasoning section",
    "content": "This is the rest",
}
COMPLETE_REASONING_WITH_THINK = {
    "output": f"{START_REASONING}This is a reasoning section{START_RESPONSE}",
    "reasoning": "This is a reasoning section",
    "content": None,
}
MULTIPLE_LINES_WITH_THINK = {
    "output": f"{START_REASONING}This\nThat{START_RESPONSE}This is the rest\nThat",
    "reasoning": "This\nThat",
    "content": "This is the rest\nThat",
}

TEST_CASES = [
    pytest.param(
        False,
        SIMPLE_REASONING,
        id="simple_reasoning",
    ),
    pytest.param(
        False,
        COMPLETE_REASONING,
        id="complete_reasoning",
    ),
    pytest.param(
        False,
        NO_REASONING,
        id="no_reasoning",
    ),
    pytest.param(
        False,
        MULTIPLE_LINES,
        id="multiple_lines",
    ),
    pytest.param(
        False,
        REASONING_WITH_THINK,
        id="reasoning_with_think",
    ),
    pytest.param(
        False,
        COMPLETE_REASONING_WITH_THINK,
        id="complete_reasoning_with_think",
    ),
    pytest.param(
        False,
        MULTIPLE_LINES_WITH_THINK,
        id="multiple_lines_with_think",
    ),
    pytest.param(
        True,
        SIMPLE_REASONING,
        id="simple_reasoning_streaming",
    ),
    pytest.param(
        True,
        COMPLETE_REASONING,
        id="complete_reasoning_streaming",
    ),
    pytest.param(
        True,
        NO_REASONING,
        id="no_reasoning_streaming",
    ),
    pytest.param(
        True,
        MULTIPLE_LINES,
        id="multiple_lines_streaming",
    ),
    pytest.param(
        True,
        REASONING_WITH_THINK,
        id="reasoning_with_think_streaming",
    ),
    pytest.param(
        True,
        COMPLETE_REASONING_WITH_THINK,
        id="complete_reasoning_with_think_streaming",
    ),
    pytest.param(
        True,
        MULTIPLE_LINES_WITH_THINK,
        id="multiple_lines_with_think_streaming",
    ),
]

# Global tokenizer initialization to avoid repeated loading
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")


@pytest.mark.parametrize("streaming, param_dict", TEST_CASES)
def test_reasoning(
    streaming: bool,
    param_dict: dict,
):
    output = tokenizer.tokenize(param_dict["output"])
    # decode everything to tokens
    output_tokens: list[str] = [
        tokenizer.convert_tokens_to_string([token]) for token in output
    ]
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(parser_name)(
        tokenizer
    )

    reasoning, content = run_reasoning_extraction(
        parser, output_tokens, streaming=streaming
    )

    assert reasoning == param_dict["reasoning"]
    assert content == param_dict["content"]


# Additional tests for verifying the correctness of granite streaming; this
# is complicated because granite uses multiple tokens to indicate when thinking
# is starting / when it's starting its response, so skipping special tokens
# is awkward.

### Handling the start of reasoning
STREAMING_1 = {
    "previous_text": None,
    "current_text": "Here",
    "delta_text": "Here",
    "reasoning": None,
    "content": None,
}
# When we fail, we should give what was previously being silenced first
STREAMING_2 = {
    "previous_text": "Here is my thought",
    "current_text": "Here is my thought failure",
    "delta_text": " failure",
    "reasoning": None,
    "content": "Here is my thought failure",
}
# But then after the first one, we should only add the delta text to content
STREAMING_3 = {
    "previous_text": "Here wrong",
    "current_text": " words",
    "delta_text": " Here wrong words",
    "reasoning": None,
    "content": " words",
}
# But then after the first one, we should only add the delta text to content
STREAMING_4 = {
    "previous_text": "Here is my thought",
    "current_text": "Here is my thought process:",
    "delta_text": " process:",
    "reasoning": None,
    "content": None,
}
# Reasoning started successfully; parse reasoning content
STREAMING_5 = {
    "previous_text": "Here is my thought process:",
    "current_text": "Here is my thought process: foo",
    "delta_text": " foo",
    "reasoning": " foo",
    "content": None,
}
# Response special sequence has started, but not finished.
STREAMING_6 = {
    "previous_text": "Here is my thought process: foo",
    "current_text": "Here is my thought process: foo Here is",
    "delta_text": " Here is",
    "reasoning": " ",
    "content": None,
}
# Response special sequence started, but was broken; the reasoning
# content should be the content that was previously unused.
STREAMING_7 = {
    "previous_text": "Here is my thought process: foo Here is",
    "current_text": "Here is my thought process: foo Here is Here",
    "delta_text": " Here",
    "reasoning": "Here is ",
    "content": None,
}
# Response special sequence is ongoing
STREAMING_8 = {
    "previous_text": "Here is my thought process: foo Here is my response:",
    "current_text": "Here is my thought process: foo Here is my response: bar",
    "delta_text": " bar",
    "reasoning": None,
    "content": " bar",
}
# The delta text has everything; we should be able to correctly parse both
STREAMING_9 = {
    "previous_text": None,
    "current_text": "Here is my thought process: foo Here is my response: bar",
    "delta_text": "Here is my thought process: foo Here is my response: bar",
    "reasoning": " foo ",
    "content": " bar",
}
## The Response is ongoing, and the delta mixes reasoning content / content
STREAMING_10 = {
    "previous_text": "Here is my thought process: foo",
    "current_text": "Here is my thought process: foo bar Here is my response: baz",
    "delta_text": " bar Here is my response: baz",
    "reasoning": " bar ",
    "content": " baz",
}
# The delta text starts a new substring that might be a response special seq
STREAMING_11 = {
    "previous_text": "Here is my thought process: This is a reasoning section ",
    "current_text": "Here is my thought process: This is a reasoning section Here",
    "delta_text": "Here",
    "reasoning": None,
    "content": None,
}
# The delta text is finishing the response special seq
STREAMING_12 = {
    "previous_text": "Here is my thought process: foo Here is my response",
    "current_text": "Here is my thought process: foo Here is my response:",
    "delta_text": ":",
    "reasoning": None,
    "content": None,
}
STREAMING_13 = {
    "previous_text": "Here is my thought process: foo Here",
    "current_text": "Here is my thought process: foo Here was",
    "delta_text": " was",
    "reasoning": "Here was",
    "content": None,
}

STREAMING_SUBCASES = [
    pytest.param(
        STREAMING_1,
        id="Starting reasoning special sequence",
    ),
    pytest.param(
        STREAMING_2,
        id="Unexpected start reasoning sequence",
    ),
    pytest.param(
        STREAMING_3,
        id="Continuing unexpected start reasoning sequence",
    ),
    pytest.param(
        STREAMING_4,
        id="Only start reasoning sequence and nothing else",
    ),
    pytest.param(
        STREAMING_5,
        id="Reasoning content has started",
    ),
    pytest.param(
        STREAMING_6,
        id="Response special sequence has started",
    ),
    pytest.param(
        STREAMING_7,
        id="Response special sequence reset",
    ),
    pytest.param(
        STREAMING_8,
        id="Response text has started",
    ),
    pytest.param(
        STREAMING_9,
        id="Delta contains everything",
    ),
    pytest.param(
        STREAMING_10,
        id="Delta contains some reasoning and response",
    ),
    pytest.param(
        STREAMING_11,
        id="Delta starts response sequence",
    ),
    pytest.param(
        STREAMING_12,
        id="Delta finishes response sequence",
    ),
    pytest.param(
        STREAMING_13,
        id="Delta breaks potential responise sequence",
    ),
]


@pytest.mark.parametrize("param_dict", STREAMING_SUBCASES)
def test_streaming_subcases(param_dict):
    # Get all of the token IDs
    previous_token_ids = (
        tokenizer.encode(param_dict["previous_text"])
        if param_dict["previous_text"] is not None
        else []
    )
    current_token_ids = tokenizer.encode(param_dict["current_text"])
    delta_token_ids = tokenizer.encode(param_dict["delta_text"])

    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(parser_name)(
        tokenizer
    )

    response = parser.extract_reasoning_streaming(
        previous_text=param_dict["previous_text"],
        current_text=param_dict["current_text"],
        delta_text=param_dict["delta_text"],
        previous_token_ids=previous_token_ids,
        current_token_ids=current_token_ids,
        delta_token_ids=delta_token_ids,
    )
    # Streaming currently expects at least one of reasoning content / content,
    # so the response should return None in that case.
    if param_dict["reasoning"] is None and param_dict["content"] is None:
        assert response is None
    else:
        assert isinstance(response, DeltaMessage)
        assert param_dict["reasoning"] == response.reasoning
        assert param_dict["content"] == response.content
