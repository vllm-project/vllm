# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
from transformers import AutoTokenizer

from tests.reasoning.utils import DeltaMessage, run_reasoning_extraction
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.abstract_parser import DelegatingParser
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


def _thinking_parser() -> ReasoningParser:
    return ReasoningParserManager.get_reasoning_parser(parser_name)(
        tokenizer, chat_template_kwargs={"thinking": True}
    )


def _ids(text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


END_DETECTION_OUTPUTS = [
    pytest.param(SIMPLE_REASONING["output"], id="simple_reasoning"),
    pytest.param(MULTIPLE_LINES["output"], id="multiple_lines"),
    pytest.param(
        "Here's my thought process:\nThink.\n\nHere's my response:\nDone.",
        id="heres_with_newlines",
    ),
    pytest.param(
        "Here is my thought process: a: b. Here's my response: c",
        id="mixed_spellings",
    ),
]


@pytest.mark.parametrize("output", END_DETECTION_OUTPUTS)
def test_reasoning_end_matches_extraction(output: str):
    """The structured-output manager sees only token ids, so the reasoning end
    must fire on the token where extraction finds the response marker."""
    ids = _ids(output)
    parser = _thinking_parser()
    extraction_end = next(
        i
        for i in range(1, len(ids) + 1)
        if parser.extract_reasoning(tokenizer.decode(ids[:i]), None)[0] is not None
    )

    gate_steps = [
        i
        for i in range(1, len(ids) + 1)
        if parser.is_reasoning_end_streaming(ids[:i], ids[i - 1 : i])
    ]
    assert gate_steps == [extraction_end]
    assert parser.is_reasoning_end(ids[: extraction_end - 1]) is False
    assert parser.is_reasoning_end(ids[:extraction_end]) is True


def test_is_reasoning_end_streaming_multi_token_delta():
    ids = _ids(SIMPLE_REASONING["output"])
    end = len(_ids(COMPLETE_REASONING["output"]))
    assert ids[:end] == _ids(COMPLETE_REASONING["output"])
    parser = _thinking_parser()

    # Several tokens accepted at once, e.g. speculative decoding.
    assert parser.is_reasoning_end_streaming(ids[:end], iter(ids[end - 3 : end]))
    assert not parser.is_reasoning_end_streaming(ids[: end + 2], ids[end : end + 2])

    # A delta holding the whole marker.
    marker = _ids(START_RESPONSE)
    assert parser.is_reasoning_end_streaming(marker, marker)

    # The marker started in an earlier step and ends early in a long delta.
    ids = _ids(f"{COMPLETE_REASONING['output']}{' and then' * 12}")
    delta = ids[end - 2 : end + 12]
    assert parser.is_reasoning_end_streaming(ids[: end + 12], delta)


def test_is_reasoning_end_ignores_earlier_turns_in_prompt():
    """Granite's chat template ends every turn with EOS. The thinking system
    prompt quotes both markers and earlier answers can carry them too."""
    eos = tokenizer.eos_token
    system = (
        "Write your thoughts after 'Here is my thought process:' and write your "
        "response after 'Here is my response:' for each user query."
    )
    prompt = (
        f"<|start_of_role|>system<|end_of_role|>{system}{eos}\n"
        f"<|start_of_role|>user<|end_of_role|>hi{eos}\n"
        f"<|start_of_role|>assistant<|end_of_role|>{SIMPLE_REASONING['output']}{eos}\n"
        f"<|start_of_role|>user<|end_of_role|>again{eos}\n"
        "<|start_of_role|>assistant<|end_of_role|>"
    )
    prompt_ids = _ids(prompt)
    assert prompt_ids.count(tokenizer.eos_token_id) == 4
    parser = _thinking_parser()

    assert parser.is_reasoning_end(prompt_ids) is False
    assert parser.is_reasoning_end(prompt_ids + _ids(START_REASONING)) is False
    assert (
        parser.is_reasoning_end(prompt_ids + _ids(COMPLETE_REASONING["output"])) is True
    )


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"chat_template_kwargs": None}, {"chat_template_kwargs": {"thinking": False}}],
)
def test_reasoning_ended_from_start_without_thinking(kwargs: dict):
    """Without `thinking`, Granite's chat template asks for no thought section,
    so the whole output is the response."""
    parser = ReasoningParserManager.get_reasoning_parser(parser_name)(
        tokenizer, **kwargs
    )
    ids = _ids("{}")

    assert parser.is_reasoning_end([]) is True
    assert parser.is_reasoning_end_streaming(ids, ids) is True


@pytest.mark.parametrize("tokens_per_delta", [1, 3])
def test_parse_delta_switches_to_content_at_response(tokens_per_delta: int):
    """The chat frontend leaves the reasoning phase when
    `is_reasoning_end_streaming` fires and keeps streaming the content."""

    class ReasoningOnlyParser(DelegatingParser):
        reasoning_parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)

    parser = ReasoningOnlyParser(tokenizer, chat_template_kwargs={"thinking": True})
    request = ChatCompletionRequest(
        model="test-model", messages=[{"role": "user", "content": "hi"}]
    )
    ids = _ids(SIMPLE_REASONING["output"])
    deltas = [
        ids[i : i + tokens_per_delta] for i in range(0, len(ids), tokens_per_delta)
    ]

    reasoning = content = ""
    prompt_token_ids: list[int] | None = []
    for delta_ids in deltas:
        message = parser.parse_delta(
            tokenizer.decode(delta_ids),
            delta_ids,
            request,
            prompt_token_ids=prompt_token_ids,
            finished=False,
        )
        prompt_token_ids = None
        reasoning += (message.reasoning or "") if message else ""
        content += (message.content or "") if message else ""

    assert reasoning == SIMPLE_REASONING["reasoning"]
    assert content == SIMPLE_REASONING["content"]
