# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from tests.reasoning.utils import (
    run_reasoning_extraction_nonstreaming,
    run_reasoning_extraction_streaming,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.abstract_parser import DelegatingParser
from vllm.reasoning.plamo3_reasoning_parser import (
    BEGIN_THINK_TAG,
    BEGIN_TOOL_REQUESTS_TAG,
    END_THINK_TAG,
    EOT_TAG,
    Plamo3ReasoningParser,
    compute_safe_until,
)
from vllm.tool_parsers.plamo3_tool_parser import Plamo3ToolParser


class _DummyTokenizer:
    """Minimal tokenizer with PLaMo-3 special-token ID mappings for unit tests."""

    def __init__(self):
        self._vocab: dict[str, int] = {}
        self.bos_token_id: int | None = 1

    def get_vocab(self) -> dict[str, int]:
        return self._vocab

    def tokenize(self, text: str) -> list[str]:
        return [text] if text else []

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(tokens)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text == "<|plamo:begin_":
            return [256]
        if text == "<|plamo:end_":
            return [257]
        if text == ":plamo|>":
            return [258]
        if text == "<|plamo:begin_think:plamo|>":
            return [256, 21279, 258]
        if text == "<|plamo:end_think:plamo|>":
            return [257, 21279, 258]
        if text == "<|plamo:begin_tool_requests:plamo|>":
            return [256, 13672, 95, 31026, 258]
        if text == "<|plamo:end_tool_requests:plamo|>":
            return [257, 13672, 95, 31026, 258]
        if text == "<|plamo:begin_tool_request:plamo|>":
            return [256, 13672, 95, 2475, 258]
        if text == "<|plamo:end_tool_request:plamo|>":
            return [257, 13672, 95, 2475, 258]
        if text == "<|plamo:begin_tool_name:plamo|>":
            return [256, 13672, 50416, 258]
        if text == "<|plamo:end_tool_name:plamo|>":
            return [257, 13672, 50416, 258]
        if text == (
            "<|plamo:begin_tool_arguments:plamo|><|plamo:constrain|>json<|plamo:msg|>"
        ):
            return [256, 13672, 95, 19868, 258, 31, 349, 19]
        if text == "<|plamo:end_tool_arguments:plamo|>":
            return [257, 13672, 95, 19868, 258]
        if text == "<|plamo:constrain|>":
            return [31]
        if text == "<|plamo:msg|>":
            return [19]
        if text == "<|plamo:tag|>":
            return [16]
        # For streaming tests, unknown text (including tag fragments) is
        # treated as generating no token ids.
        return []

    def decode(self, token_ids: list[int], **kwargs) -> str:
        return "".join(str(token_id) for token_id in token_ids)


# Keep token-id expectations tied to the dummy tokenizer's mapping rather than
# duplicating its implementation details in parametrized tests.
_BEGIN_IDS = _DummyTokenizer().encode(BEGIN_THINK_TAG, add_special_tokens=False)
_END_IDS = _DummyTokenizer().encode(END_THINK_TAG, add_special_tokens=False)
_END0, _END1, _END2 = _END_IDS  # END_THINK_TAG must be three tokens


@pytest.fixture
def tokenizer():
    return _DummyTokenizer()


@pytest.fixture
def parser(tokenizer):
    return Plamo3ReasoningParser(tokenizer)


def test_reasoning_delimiter_properties(parser):
    assert parser.reasoning_start_str == BEGIN_THINK_TAG
    assert parser.reasoning_end_str == END_THINK_TAG


@pytest.mark.parametrize(
    "enable_thinking, input_ids, expected",
    [
        (False, [100, 200], [100, 200]),
        (True, [42], []),
        (True, _BEGIN_IDS, []),
        (True, _END_IDS + [99, 100], [99, 100]),
        (True, _BEGIN_IDS + [42] + _END_IDS, []),
        (True, _BEGIN_IDS + [42] + _END_IDS + [99], [99]),
        (True, _END_IDS + [99] + _END_IDS + [100], [100]),
    ],
)
def test_extract_content_ids(enable_thinking, input_ids, expected, tokenizer):
    kwargs = (
        {} if enable_thinking else {"chat_template_kwargs": {"enable_thinking": False}}
    )
    parser = Plamo3ReasoningParser(tokenizer, **kwargs)
    assert parser.extract_content_ids(input_ids) == expected


@pytest.mark.parametrize(
    "enable_thinking, model_output, expected_reasoning, expected_content",
    [
        (
            False,
            f"{BEGIN_THINK_TAG}reasoning{END_THINK_TAG}answer",
            None,
            f"{BEGIN_THINK_TAG}reasoning{END_THINK_TAG}answer",
        ),
        (False, f"answer{EOT_TAG}ignored", None, "answer"),
        (True, "reasoning in progress", "reasoning in progress", None),
        (True, f"reasoning{END_THINK_TAG}answer", "reasoning", "answer"),
        (True, f"reasoning{EOT_TAG}ignored", "reasoning", None),
        (True, BEGIN_THINK_TAG, None, None),
        (True, BEGIN_THINK_TAG[:-1], None, None),
        (
            True,
            f"{BEGIN_THINK_TAG}reasoning in progress",
            "reasoning in progress",
            None,
        ),
        (
            True,
            f"{BEGIN_THINK_TAG}reasoning{END_THINK_TAG}answer",
            "reasoning",
            "answer",
        ),
        (
            True,
            f"{BEGIN_THINK_TAG}reasoning{END_THINK_TAG}answer{EOT_TAG}ignored",
            "reasoning",
            "answer",
        ),
        (True, f"{BEGIN_THINK_TAG}reasoning{EOT_TAG}ignored", "reasoning", None),
    ],
)
def test_non_streaming_extract_reasoning(
    enable_thinking, model_output, expected_reasoning, expected_content, tokenizer
):
    kwargs = (
        {} if enable_thinking else {"chat_template_kwargs": {"enable_thinking": False}}
    )
    parser = Plamo3ReasoningParser(tokenizer, **kwargs)
    reasoning, content = run_reasoning_extraction_nonstreaming(parser, [model_output])
    assert reasoning == expected_reasoning
    assert content == expected_content


@pytest.mark.parametrize(
    ("enable_thinking", "deltas", "expected_reasoning", "expected_content"),
    [
        (True, [BEGIN_THINK_TAG], None, None),
        (True, [BEGIN_THINK_TAG[:-1]], None, None),
        (
            True,
            [BEGIN_THINK_TAG, "reasoning", END_THINK_TAG, "answer"],
            "reasoning",
            "answer",
        ),
        (
            True,
            [BEGIN_THINK_TAG, "reasoning", END_THINK_TAG, "answer", EOT_TAG, "ignored"],
            "reasoning",
            "answer",
        ),
        (
            True,
            [
                BEGIN_THINK_TAG,
                "reasoning",
                END_THINK_TAG + "answer" + EOT_TAG + "ignored",
            ],
            "reasoning",
            "answer",
        ),
        (
            True,
            [BEGIN_THINK_TAG, "reasoning", EOT_TAG, "ignored"],
            "reasoning",
            None,
        ),
        (True, ["reasoning", EOT_TAG, "ignored"], "reasoning", None),
        (
            True,
            [
                "<|plamo:begin_",
                "think",
                ":plamo|>",
                "reasoning",
                "<|plamo:end_",
                "think",
                ":plamo|>",
                "answer",
            ],
            "reasoning",
            "answer",
        ),
        (
            True,
            ["reasoning", END_THINK_TAG, "answer"],
            "reasoning",
            "answer",
        ),
        (
            False,
            [
                "prefix",
                BEGIN_THINK_TAG,
                "ignored reasoning",
                END_THINK_TAG,
                "suffix",
            ],
            None,
            f"prefix{BEGIN_THINK_TAG}ignored reasoning{END_THINK_TAG}suffix",
        ),
        (False, ["answer", EOT_TAG, "ignored"], None, "answer"),
        (False, [f"answer{EOT_TAG}ignored", "later"], None, "answer"),
    ],
)
def test_streaming_reasoning_extraction(
    enable_thinking, deltas, expected_reasoning, expected_content, tokenizer
):
    kwargs = (
        {} if enable_thinking else {"chat_template_kwargs": {"enable_thinking": False}}
    )
    parser = Plamo3ReasoningParser(tokenizer, **kwargs)
    reconstructor = run_reasoning_extraction_streaming(parser, deltas)
    assert reconstructor.reasoning == expected_reasoning
    assert reconstructor.other_content == expected_content


@pytest.mark.parametrize(
    "steps",
    [
        [
            (BEGIN_THINK_TAG, None, None),
            ("abc", "abc", None),
            ("thk<|plamo:end_thin", "thk", None),
            ("k:plamo|>", None, None),
            ("X", None, "X"),
        ],
        [
            ("", None, None),
            (BEGIN_THINK_TAG + END_THINK_TAG, None, None),
            ("answer", None, "answer"),
        ],
        [("reasoning in progress", "reasoning in progress", None)],
        [(BEGIN_THINK_TAG + "reasoning" + EOT_TAG + "ignored", "reasoning", None)],
    ],
)
def test_streaming_reasoning_edge_deltas(tokenizer, steps):
    parser = Plamo3ReasoningParser(tokenizer)
    current_text = ""
    for delta_text, expected_reasoning, expected_content in steps:
        previous_text = current_text
        current_text += delta_text
        message = parser.extract_reasoning_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
        )

        actual_reasoning = message.reasoning if message else None
        actual_content = message.content if message else None
        assert actual_reasoning == expected_reasoning
        assert actual_content == expected_content


@pytest.mark.parametrize(
    ("enable_thinking", "input_ids", "expected"),
    [
        (True, [], False),
        (True, [100, 200, 300], False),
        (True, _BEGIN_IDS + [42], False),
        (True, _BEGIN_IDS + [42] + _END_IDS[:2], False),
        (True, _BEGIN_IDS + [42] + _END_IDS, True),
        (True, [100] + _END_IDS + [42], True),
        (True, _END_IDS + _BEGIN_IDS, False),
        (False, [100, 200], True),
    ],
)
def test_is_reasoning_end(enable_thinking, input_ids, expected, tokenizer):
    kwargs = (
        {} if enable_thinking else {"chat_template_kwargs": {"enable_thinking": False}}
    )
    parser = Plamo3ReasoningParser(tokenizer, **kwargs)
    assert parser.is_reasoning_end(input_ids) is expected


@pytest.mark.parametrize(
    ("enable_thinking", "input_ids", "delta_ids", "expected"),
    [
        (True, [100], [100], False),
        (True, _BEGIN_IDS + [42], [42], False),
        (True, [1] + _BEGIN_IDS + [42], [42], False),
        (False, [100, 200], [200], True),
        (True, _BEGIN_IDS + [42, _END0], [_END0], False),
        (True, _BEGIN_IDS + [42, _END0, _END1], [_END1], False),
        (True, _BEGIN_IDS + [42] + _END_IDS, [_END2], True),
        (True, _BEGIN_IDS + [42] + _END_IDS, _END_IDS, True),
        (True, _BEGIN_IDS + [42, _END0, _END1], [_END0, _END1], False),
        (True, _BEGIN_IDS + [42] + _END_IDS + [99], [99], False),
    ],
)
def test_is_reasoning_end_streaming(
    enable_thinking, input_ids, delta_ids, expected, tokenizer
):
    kwargs = (
        {} if enable_thinking else {"chat_template_kwargs": {"enable_thinking": False}}
    )
    parser = Plamo3ReasoningParser(tokenizer, **kwargs)
    assert parser.is_reasoning_end_streaming(input_ids, delta_ids) is expected


def test_is_reasoning_end_does_not_depend_on_streaming_state(tokenizer):
    parser = Plamo3ReasoningParser(tokenizer)
    streamed_ids = _BEGIN_IDS + [42] + _END_IDS + [99]
    parser.extract_reasoning_streaming(
        previous_text="",
        current_text=BEGIN_THINK_TAG + "reasoning" + END_THINK_TAG + "answer",
        delta_text=BEGIN_THINK_TAG + "reasoning" + END_THINK_TAG + "answer",
        previous_token_ids=[],
        current_token_ids=streamed_ids,
        delta_token_ids=streamed_ids,
    )

    assert parser.is_reasoning_end([100]) is False
    assert parser.is_reasoning_end([100] + _END_IDS + [42]) is True


def test_extract_content_ids_uses_accumulated_stream_tokens(tokenizer):
    parser = Plamo3ReasoningParser(tokenizer)
    streamed_ids = _BEGIN_IDS + [42] + _END_IDS + [99]
    parser.extract_reasoning_streaming(
        previous_text="",
        current_text=BEGIN_THINK_TAG + "reasoning" + END_THINK_TAG + "answer",
        delta_text=BEGIN_THINK_TAG + "reasoning" + END_THINK_TAG + "answer",
        previous_token_ids=[],
        current_token_ids=streamed_ids,
        delta_token_ids=streamed_ids,
    )

    assert parser.extract_content_ids([0]) == [99]


class _Plamo3DelegatingParser(DelegatingParser):
    reasoning_parser_cls = Plamo3ReasoningParser
    tool_parser_cls = Plamo3ToolParser


def _named_tool_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="test-model",
        messages=[],
        tools=[
            {
                "type": "function",
                "function": {"name": "noop", "parameters": {"type": "object"}},
            }
        ],
        tool_choice={"type": "function", "function": {"name": "noop"}},
        include_reasoning=True,
    )


def _run_named_tool_stream(tokenizer, chunks):
    parser = _Plamo3DelegatingParser(tokenizer)
    request = _named_tool_request()
    messages = []
    for index, (delta_text, delta_ids) in enumerate(chunks):
        message = parser.parse_delta(
            delta_text,
            delta_ids,
            request,
            prompt_token_ids=[] if index == 0 else None,
            finished=index == len(chunks) - 1,
        )
        if message is not None:
            messages.append(message)
    reasoning = "".join(message.reasoning or "" for message in messages)
    arguments = "".join(
        tool_call.function.arguments or ""
        for message in messages
        for tool_call in message.tool_calls or []
        if tool_call.function is not None
    )
    return reasoning, arguments


@pytest.mark.parametrize(
    "chunks",
    [
        [
            (BEGIN_THINK_TAG, _BEGIN_IDS),
            ("reasoning", [100]),
            (END_THINK_TAG, _END_IDS),
            ("answer-1", [200]),
            ("answer-2", [201]),
        ],
        [
            (BEGIN_THINK_TAG, _BEGIN_IDS),
            ("reasoning", [100]),
            (END_THINK_TAG + "answer-1", _END_IDS + [200]),
            ("answer-2", [201]),
        ],
        [
            (BEGIN_THINK_TAG, _BEGIN_IDS),
            ("reasoning", [100]),
            ("<|plamo:end_", [_END0]),
            ("", [_END1]),
            ("think:plamo|>answer-1", [_END2, 200]),
            ("answer-2", [201]),
        ],
    ],
)
def test_named_tool_streaming_content_survives_reasoning_end(tokenizer, chunks):
    reasoning, arguments = _run_named_tool_stream(tokenizer, chunks)
    assert reasoning == "reasoning"
    assert arguments == "answer-1answer-2"


def test_init_rejects_empty_think_tag_tokenization():
    class EmptyThinkTagTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return []

    with pytest.raises(ValueError, match="failed to tokenize think tags"):
        Plamo3ReasoningParser(EmptyThinkTagTokenizer())


@pytest.mark.parametrize(
    ("buf", "floor", "additional_tags", "expected"),
    [
        ("abc", 0, (), 3),
        ("reasoning text<|pla", 0, (), 14),
        ("reasoning text<|plamo:end_thin", 0, (), 14),
        (f"reasoning text{END_THINK_TAG}", 0, (), 39),
        ("abc<|pla", 7, (), 7),
        ("x<|plamo:end_ztext<|pla", 0, (), 18),
        ("content<|plamo:b", 0, (BEGIN_TOOL_REQUESTS_TAG,), 7),
    ],
)
def test_compute_safe_until(buf, floor, additional_tags, expected):
    tags = (END_THINK_TAG, *additional_tags)
    assert compute_safe_until(buf, floor=floor, tags=tags) == expected
