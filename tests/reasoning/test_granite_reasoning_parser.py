# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from functools import cache

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
    parser = _parser()

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

    parser = _parser()

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


# Test data: what Granite 3.2's template writes into the system turn with
# `thinking`. It quotes both markers.
TEMPLATE_INSTRUCTION = (
    "Write your thoughts after 'Here is my thought process:' and write your "
    "response after 'Here is my response:' for each user query."
)


def _prompt(system: str) -> str:
    eos = tokenizer.eos_token
    return (
        f"<|start_of_role|>system<|end_of_role|>{system}{eos}\n"
        f"<|start_of_role|>user<|end_of_role|>hi{eos}\n"
        "<|start_of_role|>assistant<|end_of_role|>"
    )


THINKING_PROMPT = _prompt(TEMPLATE_INSTRUCTION)


def _parser() -> ReasoningParser:
    return ReasoningParserManager.get_reasoning_parser(parser_name)(tokenizer)


class _ReasoningOnlyParser(DelegatingParser):
    reasoning_parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)


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
    must fire on the token after which extraction puts new text in content
    instead of the thought section."""
    ids = _ids(output)
    parser = _parser()
    extraction_end = next(
        i
        for i in range(1, len(ids) + 1)
        if all(parser.extract_reasoning(tokenizer.decode(ids[:i]) + " X", None))
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
    parser = _parser()

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
    prompt = (
        f"<|start_of_role|>system<|end_of_role|>{TEMPLATE_INSTRUCTION}{eos}\n"
        f"<|start_of_role|>user<|end_of_role|>hi{eos}\n"
        f"<|start_of_role|>assistant<|end_of_role|>{SIMPLE_REASONING['output']}{eos}\n"
        f"<|start_of_role|>user<|end_of_role|>again{eos}\n"
        "<|start_of_role|>assistant<|end_of_role|>"
    )
    prompt_ids = _ids(prompt)
    assert prompt_ids.count(tokenizer.eos_token_id) == 4
    # As the engine's reasoner: no prompt seen before the gate.
    parser = ReasoningParserManager.get_reasoning_parser(parser_name)(tokenizer)

    assert parser.is_reasoning_end(prompt_ids) is False
    assert parser.is_reasoning_end(prompt_ids + _ids(START_REASONING)) is False
    assert parser.is_reasoning_end(prompt_ids + _ids(START_RESPONSE)) is True


@pytest.mark.parametrize(
    "prompt", ["", _prompt("You are a helpful AI assistant.")], ids=["empty", "plain"]
)
def test_reasoning_ended_from_start_without_thinking(prompt: str):
    """Without a thought marker in the first turn, Granite's chat template
    asked for no thought section, so the grammar applies from the start."""
    parser = ReasoningParserManager.get_reasoning_parser(parser_name)(tokenizer)
    assert parser.is_reasoning_end(_ids(prompt)) is True


NO_THOUGHT = {
    "output": 'Here is my response: {"answer": 51}',
    "reasoning": None,
    "content": ' {"answer": 51}',
}


@pytest.mark.parametrize(
    "output", [NO_THOUGHT["output"], 'Here\'s my response: {"answer": 51}']
)
def test_response_without_thought_section(output: str):
    """With `thinking` on, the reasoning end also fires on a response marker
    that opens the output, so the content starts after that marker."""
    tokens = [
        tokenizer.convert_tokens_to_string([t]) for t in tokenizer.tokenize(output)
    ]
    for streaming in (False, True):
        extracted = run_reasoning_extraction(_parser(), tokens, streaming=streaming)
        assert extracted == (None, NO_THOUGHT["content"])


@pytest.mark.parametrize("case", [SIMPLE_REASONING, NO_THOUGHT])
@pytest.mark.parametrize("tokens_per_delta", [1, 3])
def test_parse_delta_switches_to_content_at_response(tokens_per_delta: int, case: dict):
    """The chat frontend leaves the reasoning phase when
    `is_reasoning_end_streaming` fires and keeps streaming the content."""
    parser = _ReasoningOnlyParser(tokenizer)
    request = ChatCompletionRequest(
        model="test-model", messages=[{"role": "user", "content": "hi"}]
    )
    ids = _ids(case["output"])
    deltas = [
        ids[i : i + tokens_per_delta] for i in range(0, len(ids), tokens_per_delta)
    ]

    reasoning = content = ""
    prompt_token_ids: list[int] | None = _ids(THINKING_PROMPT)
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

    assert reasoning == (case["reasoning"] or "")
    assert content == case["content"]


SPLIT_CASES = [
    pytest.param(
        '\nHere is my response: {"a": 1}', None, '\n {"a": 1}', id="newline_first"
    ),
    pytest.param(
        '{"text": "Here is my response: hi"}',
        None,
        '{"text": "Here is my response: hi"}',
        id="marker_inside_a_json_value",
    ),
    pytest.param(
        f"Sure. {START_RESPONSE} ok",
        None,
        f"Sure. {START_RESPONSE} ok",
        id="no_thought_preamble",
    ),
    pytest.param(
        f"Sure. {START_REASONING} a {START_RESPONSE} b",
        None,
        f"Sure. {START_REASONING} a {START_RESPONSE} b",
        id="thought_after_preamble",
    ),
    pytest.param(
        f"{START_RESPONSE} quote '{START_REASONING} a {START_RESPONSE} b'",
        None,
        f" quote '{START_REASONING} a {START_RESPONSE} b'",
        id="no_thought_quotes_both_markers",
    ),
    pytest.param(
        f"{START_REASONING} still thinking",
        " still thinking",
        None,
        id="cut_in_thought",
    ),
    pytest.param("Here is my respon", None, "Here is my respon", id="cut_in_marker"),
    pytest.param(
        "Here is my thou", None, "Here is my thou", id="cut_in_thought_marker"
    ),
    pytest.param(
        f"{START_REASONING} a {START_RESPONSE} b {START_RESPONSE} c",
        " a ",
        f" b {START_RESPONSE} c",
        id="first_response_marker_wins",
    ),
]


@pytest.mark.parametrize("output, reasoning, content", SPLIT_CASES)
@pytest.mark.parametrize("tokens_per_delta", [1, 3])
def test_streaming_and_non_streaming_split_alike(
    output: str, reasoning: str | None, content: str | None, tokens_per_delta: int
):
    """Both paths split where the reasoning-end gate fires: at the first
    response marker of the turn. Streaming flushes held text on the last delta."""
    assert _parser().extract_reasoning(output, None) == (reasoning, content)

    parser = _ReasoningOnlyParser(tokenizer)
    request = ChatCompletionRequest(
        model="test-model", messages=[{"role": "user", "content": "hi"}]
    )
    ids = _ids(output)
    streamed_reasoning = streamed_content = ""
    prompt_token_ids: list[int] | None = _ids(THINKING_PROMPT)
    for i in range(0, len(ids), tokens_per_delta):
        delta_ids = ids[i : i + tokens_per_delta]
        message = parser.parse_delta(
            tokenizer.decode(delta_ids),
            delta_ids,
            request,
            prompt_token_ids=prompt_token_ids,
            finished=i + tokens_per_delta >= len(ids),
        )
        prompt_token_ids = None
        if message:
            streamed_reasoning += message.reasoning or ""
            streamed_content += message.content or ""
    assert (streamed_reasoning or None, streamed_content or None) == (
        reasoning,
        content,
    )


@cache
def _granite_tokenizer():
    return AutoTokenizer.from_pretrained("ibm-granite/granite-3.2-2b-instruct")


def _granite_prompt(messages: list[dict], **template_kwargs) -> list[int]:
    tok = _granite_tokenizer()
    text = tok.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, **template_kwargs
    )
    return tok.encode(text, add_special_tokens=False)


THOUGHT_OUTPUT = f"{START_REASONING} x {START_RESPONSE} y"


def test_template_ignores_thinking_with_system_message():
    """Granite's template writes the thought instruction only without a system
    message, tools or documents, so `thinking=True` alone does not start a
    thought section: the grammar applies from the first token."""
    prompt = _granite_prompt(
        [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}],
        thinking=True,
    )
    parser = ReasoningParserManager.get_reasoning_parser(parser_name)(
        _granite_tokenizer()
    )
    assert parser.is_reasoning_end(prompt) is True


def test_thinking_follows_the_prompt_without_kwargs():
    """Offline `LLM.chat(..., chat_template_kwargs={"thinking": True})` renders
    the instruction but passes no parser kwargs; the prompt alone decides."""
    tok = _granite_tokenizer()
    prompt = _granite_prompt([{"role": "user", "content": "hi"}], thinking=True)
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    output = tok.encode(THOUGHT_OUTPUT, add_special_tokens=False)

    assert parser_cls(tok).is_reasoning_end(prompt) is False
    # The frontend may report the prompt check itself, so the engine's
    # reasoner gets the prompt only through set_prompt_token_ids.
    gate_steps = _gate_steps(parser_cls(tok), prompt, output, 1)
    assert len(gate_steps) == 1
    assert tok.decode(output[: gate_steps[0]]).endswith(START_RESPONSE)

    assert parser_cls(tok).extract_reasoning(THOUGHT_OUTPUT, None) == (" x ", " y")


@pytest.mark.parametrize("tokens_per_delta", [1, 3])
def test_whitespace_before_thought_marker(tokens_per_delta: int):
    output = f"\n{START_REASONING} X {START_RESPONSE} Y"
    assert _parser().extract_reasoning(output, None) == (" X ", " Y")
    tokens = [
        tokenizer.convert_tokens_to_string([t]) for t in tokenizer.tokenize(output)
    ]
    deltas = [
        "".join(tokens[i : i + tokens_per_delta])
        for i in range(0, len(tokens), tokens_per_delta)
    ]
    assert run_reasoning_extraction(_parser(), deltas, streaming=True) == (
        " X ",
        " Y",
    )


def _gate_steps(parser, prompt: list[int], output: list[int], k: int) -> list[int]:
    """Feed `output` to the engine's gate in deltas of `k` tokens, as with
    speculative decoding; return the ends of the deltas where it fires."""
    # As the structured-output manager does before its first gate call.
    parser.set_prompt_token_ids(prompt)
    return [
        j
        for i in range(0, len(output), k)
        if parser.is_reasoning_end_streaming(
            prompt + output[: (j := min(i + k, len(output)))], output[i:j]
        )
    ]


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "user", "content": ""}],
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": f"{START_REASONING} x {START_RESPONSE}"},
            {"role": "user", "content": "again"},
        ],
    ],
    ids=["empty_user_turn", "earlier_answer_ends_with_marker"],
)
@pytest.mark.parametrize("k", [1, 5, 8])
def test_gate_window_stays_in_the_current_turn(messages: list[dict], k: int):
    """The system turn quotes the response marker, and so can an earlier
    answer; a multi-token delta must still fire on the marker it holds."""
    tok = _granite_tokenizer()
    prompt = _granite_prompt(messages, thinking=True)
    output = tok.encode(f'{START_RESPONSE} {{"a": 1}}', add_special_tokens=False)
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    assert parser_cls(tok).is_reasoning_end(prompt) is False

    steps = _gate_steps(parser_cls(tok), prompt, output, k)
    marker_end = next(
        j for j in range(1, len(output) + 1) if START_RESPONSE in tok.decode(output[:j])
    )
    assert steps == [min(len(output), -(-marker_end // k) * k)]


def test_extraction_without_a_prompt_splits_like_main():
    """Some frontends never pass the prompt (e.g. a derenderer without prompt
    ids); extraction reads the output alone and splits at the markers."""
    output = SIMPLE_REASONING["output"]
    expected = (SIMPLE_REASONING["reasoning"], SIMPLE_REASONING["content"])
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    assert parser_cls(tokenizer).extract_reasoning(output, None) == expected

    parser = _ReasoningOnlyParser(tokenizer)
    request = ChatCompletionRequest(
        model="test-model", messages=[{"role": "user", "content": "hi"}]
    )
    reasoning = content = ""
    ids = _ids(output)
    for i, token_id in enumerate(ids):
        message = parser.parse_delta(
            tokenizer.decode([token_id]),
            [token_id],
            request,
            prompt_token_ids=None,
            finished=i == len(ids) - 1,
        )
        reasoning += (message.reasoning or "") if message else ""
        content += (message.content or "") if message else ""
    assert (reasoning, content) == expected


@pytest.mark.parametrize(
    "system, thinking",
    [
        (
            "Think first. Put your notes after 'Here's my thought process:' "
            "and the answer after 'Here's my response:'.",
            True,
        ),
        ("Be brief.", False),
        (None, True),
    ],
    ids=["reworded_instruction", "plain_system_message", "template_instruction"],
)
@pytest.mark.parametrize("bos", [False, True])
def test_gate_follows_the_system_turn(system: str | None, thinking: bool, bos: bool):
    """A system turn that names a thought marker asks for thoughts in any
    wording. Granite's BOS is its EOS, so a leading BOS is skipped."""
    tok = _granite_tokenizer()
    messages = [{"role": "user", "content": "hi"}]
    if system is not None:
        messages.insert(0, {"role": "system", "content": system})
    prompt = _granite_prompt(messages, thinking=True)
    if bos:
        prompt = [tok.bos_token_id] + prompt
    parser = ReasoningParserManager.get_reasoning_parser(parser_name)(tok)
    assert parser.is_reasoning_end(prompt) is (not thinking)


def test_empty_output_is_empty_content():
    assert _parser().extract_reasoning("", None) == (None, "")


def test_prompt_check_decodes_a_bounded_slice(monkeypatch):
    """A raw prompt has no EOS, so neither the first turn nor the current turn
    ends; both checks still decode only a bounded number of tokens."""
    parser = _parser()
    filler = _ids(" word") * 100_000
    decoded: list[int] = []
    decode = tokenizer.decode

    def counting_decode(ids, **kwargs):
        decoded.append(len(ids))
        return decode(ids, **kwargs)

    monkeypatch.setattr(tokenizer, "decode", counting_decode)
    assert parser.is_reasoning_end(_ids(TEMPLATE_INSTRUCTION) + filler) is False
    assert decoded and max(decoded) <= 1024


@pytest.mark.parametrize("lag", [0, 1, 3])
def test_parse_delta_follows_lagging_text(lag: int):
    """A stop string makes the detokenizer hold back the last few characters,
    so the text can trail the token that ends the response marker. The switch
    to content follows the text, and the content keeps its first characters."""
    output = f'{START_REASONING} think.{START_RESPONSE} {{"answer": 51}}'
    reasoning, content = _parser().extract_reasoning(output, None)
    assert content == ' {"answer": 51}'

    parser = _ReasoningOnlyParser(tokenizer)
    request = ChatCompletionRequest(
        model="test-model", messages=[{"role": "user", "content": "hi"}]
    )
    ids = _ids(output)
    streamed_reasoning = streamed_content = sent = ""
    prompt_token_ids: list[int] | None = _ids(THINKING_PROMPT)
    for i in range(1, len(ids) + 1):
        text = tokenizer.decode(ids[:i])
        finished = i == len(ids)
        visible = text if finished else text[: max(len(sent), len(text) - lag)]
        message = parser.parse_delta(
            visible[len(sent) :],
            ids[i - 1 : i],
            request,
            prompt_token_ids=prompt_token_ids,
            finished=finished,
        )
        sent, prompt_token_ids = visible, None
        if message:
            streamed_reasoning += message.reasoning or ""
            streamed_content += message.content or ""
    assert (streamed_reasoning, streamed_content) == (reasoning, content)


@pytest.mark.parametrize(
    "output",
    [output for output, *_ in (case.values for case in SPLIT_CASES)]
    + [f"{START_RESPONSE} yes", '{"direct": true}'],
)
def test_thinking_off_prompt_keeps_the_whole_output(output: str):
    """With thinking off the engine applies the grammar from the first token
    and streaming passes the output through, so non-streaming extraction
    given the same prompt keeps the output whole too, even when the model
    writes a thought section unasked."""
    tok = _granite_tokenizer()
    prompt = _granite_prompt([{"role": "user", "content": "hi"}])
    request = ChatCompletionRequest(
        model="test-model", messages=[{"role": "user", "content": "hi"}]
    )

    parser = _ReasoningOnlyParser.reasoning_parser_cls(tok)
    parser.set_prompt_token_ids(prompt)
    assert parser.extract_reasoning(output, request) == (None, output)

    # serving.py sets the prompt before DelegatingParser.parse.
    delegating = _ReasoningOnlyParser(tok)
    delegating.set_prompt_token_ids(prompt)
    assert delegating.parse(output, request)[:2] == (None, output)

    streaming = _ReasoningOnlyParser(tok)
    ids = tok.encode(output, add_special_tokens=False)
    streamed_reasoning = streamed_content = ""
    prompt_token_ids: list[int] | None = prompt
    for i, token_id in enumerate(ids):
        message = streaming.parse_delta(
            tok.decode(ids[: i + 1])[len(tok.decode(ids[:i])) :],
            [token_id],
            request,
            prompt_token_ids=prompt_token_ids,
            finished=i == len(ids) - 1,
        )
        prompt_token_ids = None
        if message:
            streamed_reasoning += message.reasoning or ""
            streamed_content += message.content or ""
    assert (streamed_reasoning, streamed_content) == ("", output)


def _extract_after_prompt(prompt: list[int], output: str) -> tuple:
    """The production order: serving.py sets the prompt, then parses."""
    tok = _granite_tokenizer()
    request = ChatCompletionRequest(
        model="test-model", messages=[{"role": "user", "content": "hi"}]
    )
    parser = _ReasoningOnlyParser.reasoning_parser_cls(tok)
    parser.set_prompt_token_ids(prompt)
    delegating = _ReasoningOnlyParser(tok)
    delegating.set_prompt_token_ids(prompt)
    split = parser.extract_reasoning(output, request)
    assert delegating.parse(output, request)[:2] == split
    return split


@pytest.mark.parametrize("output, reasoning, content", SPLIT_CASES)
def test_thinking_on_prompt_still_splits(
    output: str, reasoning: str | None, content: str | None
):
    prompt = _granite_prompt([{"role": "user", "content": "hi"}], thinking=True)
    assert _extract_after_prompt(prompt, output) == (reasoning, content)


GATE_CASES = [
    pytest.param(f"{START_REASONING} a {START_RESPONSE} b", START_RESPONSE, id="split"),
    pytest.param(f"\n{START_RESPONSE} b", START_RESPONSE, id="response_first"),
    pytest.param(
        f"{START_REASONING} a {START_RESPONSE} b {START_RESPONSE} c",
        START_RESPONSE,
        id="first_response_marker_wins",
    ),
    pytest.param('{"text": "Here is my response: hi"}', None, id="inside_json"),
    pytest.param(f"Sure. {START_RESPONSE} ok", None, id="no_thought_preamble"),
    pytest.param(
        f"Sure. {START_REASONING} a {START_RESPONSE} b",
        None,
        id="thought_after_preamble",
    ),
]


@pytest.mark.parametrize("output, marker", GATE_CASES)
@pytest.mark.parametrize("k", [1, 3])
def test_engine_gate_ends_reasoning_where_extraction_does(
    output: str, marker: str | None, k: int
):
    """The grammar starts where extraction's content starts after the marker,
    and never on a marker that extraction keeps in the content."""
    tok = _granite_tokenizer()
    prompt = _granite_prompt([{"role": "user", "content": "hi"}], thinking=True)
    ids = tok.encode(output, add_special_tokens=False)
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    steps = _gate_steps(parser_cls(tok), prompt, ids, k)
    reasoning, content = _extract_after_prompt(prompt, output)
    if marker is None:
        assert steps == [] and (reasoning, content) == (None, output)
        return
    marker_end = next(
        j for j in range(1, len(ids) + 1) if marker in tok.decode(ids[:j])
    )
    assert steps == [min(len(ids), -(-marker_end // k) * k)]
    assert content.lstrip() == output.split(marker, 1)[1].lstrip()


def test_raw_prompt_quoting_markers_does_not_end_reasoning():
    """Without a role header, only a response marker that ends the prompt
    ends reasoning there."""
    tok = _granite_tokenizer()
    prompt = tok.encode(
        f"Answer in this format.\n{START_REASONING} steps\n{START_RESPONSE} 4"
        "\n\nQ: 3+3?",
        add_special_tokens=False,
    )
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    assert parser_cls(tok).is_reasoning_end(prompt) is False


def test_raw_prompt_ending_with_response_marker_ends_reasoning():
    tok = _granite_tokenizer()
    prompt = tok.encode(
        f"Q: 3+3?\n{START_REASONING} add\n{START_RESPONSE}", add_special_tokens=False
    )
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    assert parser_cls(tok).is_reasoning_end(prompt) is True


def test_engine_gate_after_a_raw_prompt_fires_at_the_response_marker():
    """Without a role header the output start is unknown, so the gate ends
    reasoning at the first response marker, where extraction splits an
    output that opens with a thought."""
    tok = _granite_tokenizer()
    prompt = tok.encode(
        f"Answer in this format.\n{START_REASONING} steps\n{START_RESPONSE} 4\n\n"
        "Q: 3+3?\n",
        add_special_tokens=False,
    )
    output = tok.encode(THOUGHT_OUTPUT, add_special_tokens=False)
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    marker_end = next(
        j for j in range(1, len(output) + 1) if START_RESPONSE in tok.decode(output[:j])
    )
    assert _gate_steps(parser_cls(tok), prompt, output, 1) == [marker_end]


@pytest.mark.parametrize(
    "prefill",
    [
        "Sure.",
        f"{START_RESPONSE} {{",
        f"{START_REASONING} x {START_RESPONSE}",
    ],
    ids=["plain_text", "response_started", "closed_thought"],
)
def test_assistant_prefill_ends_reasoning(prefill: str):
    """A continue_final_message prompt already holds the start of the answer,
    so the engine constrains the output from its first token and extraction
    returns it whole."""
    tok = _granite_tokenizer()
    prompt = _granite_prompt(
        [{"role": "user", "content": "hi"}], thinking=True
    ) + tok.encode(prefill, add_special_tokens=False)
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    assert parser_cls(tok).is_reasoning_end(prompt) is True
    assert _extract_after_prompt(prompt, THOUGHT_OUTPUT) == (None, THOUGHT_OUTPUT)


@pytest.mark.parametrize(
    "prefill",
    [f"{START_REASONING} Let me", "Here's my thought process: Let me"],
    ids=["open_thought", "heres_spelling"],
)
@pytest.mark.parametrize("k", [1, 3])
def test_prefill_with_an_open_thought_puts_the_output_inside_it(prefill, k):
    """The model still has to finish its thought and write the response
    marker: reasoning runs to that marker, where the grammar starts."""
    tok = _granite_tokenizer()
    prompt = _granite_prompt(
        [{"role": "user", "content": "hi"}], thinking=True
    ) + tok.encode(prefill, add_special_tokens=False)
    output = tok.encode(f" think. {START_RESPONSE} y", add_special_tokens=False)
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    assert parser_cls(tok).is_reasoning_end(prompt) is False
    steps = _gate_steps(parser_cls(tok), prompt, output, k)
    assert len(steps) == 1
    assert START_RESPONSE in tok.decode(output[: steps[0]])
    assert START_RESPONSE not in tok.decode(output[: max(0, steps[0] - k)])
    assert _extract_after_prompt(prompt, tok.decode(output)) == (" think. ", " y")


def test_response_marker_begun_in_the_prefill_does_not_end_reasoning():
    """Extraction reads the output alone and never sees a marker that the
    prefill began, so the gate must not see one either."""
    tok = _granite_tokenizer()
    prompt = _granite_prompt(
        [{"role": "user", "content": "hi"}], thinking=True
    ) + tok.encode(f"{START_REASONING} a. Here is my resp", add_special_tokens=False)
    output = tok.encode('onse: {"a": 1}', add_special_tokens=False)
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    assert _gate_steps(parser_cls(tok), prompt, output, 1) == []
    text = tok.decode(output)
    assert _extract_after_prompt(prompt, text) == (text, None)


def test_engine_gate_reads_the_turn_past_an_eos_in_the_output():
    """With ignore_eos the output can hold EOS; the turn still starts at its
    role header, so a marker after a preamble stays out of the grammar."""
    tok = _granite_tokenizer()
    prompt = _granite_prompt([{"role": "user", "content": "hi"}], thinking=True)
    output = (
        tok.encode("Sure.", add_special_tokens=False)
        + [tok.eos_token_id]
        + tok.encode(f" {START_RESPONSE} b", add_special_tokens=False)
    )
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    assert _gate_steps(parser_cls(tok), prompt, output, 1) == []


def test_closed_last_turn_is_not_a_prefill():
    """A prompt rendered without a generation prompt ends with a closed
    turn: the output opens a new one, so reasoning has not ended."""
    tok = _granite_tokenizer()
    prompt = _granite_prompt([{"role": "user", "content": "hi"}], thinking=True)
    closed = prompt[: len(prompt) - prompt[::-1].index(tok.eos_token_id)]
    assert tok.decode(closed).endswith(f"hi{tok.eos_token}")
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    assert parser_cls(tok).is_reasoning_end(closed) is False


def _marker_end(tok, output: list[int]) -> int:
    return next(
        j for j in range(1, len(output) + 1) if START_RESPONSE in tok.decode(output[:j])
    )


def test_engine_gate_after_a_raw_prompt_follows_extraction():
    """The output start comes from the prompt, so a raw prompt gets the same
    rule as a chat prompt: a marker after a preamble stays in the content."""
    tok = _granite_tokenizer()
    prompt = tok.encode(f"Use '{START_REASONING}'.\nQ: hi\n", add_special_tokens=False)
    output = tok.encode(f"Sure. {START_RESPONSE} b", add_special_tokens=False)
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    assert parser_cls(tok).is_reasoning_end(prompt) is False
    assert _gate_steps(parser_cls(tok), prompt, output, 1) == []


def test_raw_prompt_ending_with_a_thought_marker_puts_the_output_inside_it():
    tok = _granite_tokenizer()
    prompt = tok.encode(f"Q: hi\n{START_REASONING}", add_special_tokens=False)
    output = tok.encode(f" x. {START_RESPONSE} b", add_special_tokens=False)
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    assert parser_cls(tok).is_reasoning_end(prompt) is False
    assert _gate_steps(parser_cls(tok), prompt, output, 1) == [_marker_end(tok, output)]
    assert _extract_after_prompt(prompt, tok.decode(output)) == (" x. ", " b")


@pytest.mark.parametrize("lead", ["eos", "own_turn"])
def test_engine_gate_after_a_closed_turn_follows_extraction(lead: str):
    """Rendered without a generation prompt, the output opens the next turn.
    Extraction reads detokenized text, which drops special tokens, so the
    gate fires where the text opens with a thought section."""
    tok = _granite_tokenizer()
    prompt = _granite_prompt([{"role": "user", "content": "hi"}], thinking=True)
    closed = prompt[: len(prompt) - prompt[::-1].index(tok.eos_token_id)]
    output = tok.encode(THOUGHT_OUTPUT, add_special_tokens=False)
    if lead == "eos":
        output = [tok.eos_token_id] + output
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    assert parser_cls(tok).is_reasoning_end(closed) is False
    assert _gate_steps(parser_cls(tok), closed, output, 1) == [_marker_end(tok, output)]


def test_engine_gate_ignores_a_role_header_in_the_output():
    """With ignore_eos the model can write a new turn; its text after the
    header is not the start of the output, as extraction sees it."""
    tok = _granite_tokenizer()
    prompt = _granite_prompt([{"role": "user", "content": "hi"}], thinking=True)
    output = (
        tok.encode("Sure.", add_special_tokens=False)
        + [tok.eos_token_id]
        + tok.encode(
            f"<|start_of_role|>assistant<|end_of_role|>{THOUGHT_OUTPUT}",
            add_special_tokens=False,
        )
    )
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    assert _gate_steps(parser_cls(tok), prompt, output, 1) == []
    text = tok.decode(output, skip_special_tokens=True)
    assert parser_cls(tok).extract_reasoning(text, None) == (None, text)


def test_user_turn_continuation_is_not_a_prefill():
    """continue_final_message on a user turn: the model still writes its own
    assistant turn, so reasoning has not ended."""
    tok = _granite_tokenizer()
    text = tok.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        tokenize=False,
        continue_final_message=True,
        thinking=True,
    )
    assert text.endswith("user<|end_of_role|>hi")
    prompt = tok.encode(text, add_special_tokens=False)
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    assert parser_cls(tok).is_reasoning_end(prompt) is False


def test_response_marker_prefill_keeps_the_whole_output():
    """A continue_final_message prompt that already wrote the response marker
    ends reasoning, so the output is content, as the engine and streaming
    treat it."""
    prompt = _granite_prompt(
        [{"role": "user", "content": "hi"}], thinking=True
    ) + _granite_tokenizer().encode(
        f"{START_REASONING} x {START_RESPONSE}", add_special_tokens=False
    )
    output = THOUGHT_OUTPUT
    assert _extract_after_prompt(prompt, output) == (None, output)
