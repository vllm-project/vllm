# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence
from typing import Any, Literal

import pytest
from openai_harmony import (
    Conversation,
    Message,
    RenderConversationConfig,
    Role,
)
from transformers import AutoTokenizer, GenerationConfig

from vllm.config import StructuredOutputsConfig, VllmConfig
from vllm.entrypoints.generate.base.protocol import FunctionCall
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.parser.harmony_utils import (
    get_encoding,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.parser.harmony import HarmonyParser
from vllm.parser.parser_manager import ParserManager
from vllm.sampling_params import StructuredOutputsParams
from vllm.v1.structured_output.backend_types import StructuredOutputOptions
from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend

REASONING_MODEL_NAME = "openai/gpt-oss-20b"


@pytest.fixture(scope="module")
def gpt_oss_tokenizer():
    return AutoTokenizer.from_pretrained(REASONING_MODEL_NAME)


@pytest.fixture(scope="module")
def gpt_oss_stop_token_ids() -> set[int]:
    eos_token_id = GenerationConfig.from_pretrained(REASONING_MODEL_NAME).eos_token_id
    if isinstance(eos_token_id, int):
        return {eos_token_id}
    return set(eos_token_id)


@pytest.fixture(scope="module")
def xgrammar_backend(gpt_oss_tokenizer) -> XgrammarBackend:
    vllm_config = VllmConfig(
        structured_outputs_config=StructuredOutputsConfig(backend="xgrammar")
    )
    return XgrammarBackend(
        vllm_config,
        tokenizer=gpt_oss_tokenizer,
        vocab_size=len(gpt_oss_tokenizer),
    )


@pytest.fixture
def harmony_parser(gpt_oss_tokenizer):
    parser_cls = ParserManager.get_parser(
        tool_parser_name="openai",
        reasoning_parser_name="openai_gptoss",
        enable_auto_tools=True,
        model_name=REASONING_MODEL_NAME,
        is_harmony=True,
    )
    assert parser_cls is HarmonyParser
    return parser_cls(gpt_oss_tokenizer)


@pytest.fixture
def chat_request():
    return ChatCompletionRequest(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": "Hello"}],
    )


@pytest.fixture
def malformed_msgs_str() -> list[str]:
    return [
        "<|channel|>analysis<|message|>thinking<|end|>",
        "<|start|>assistant<|channel|>commentary<|message|>thinking<|end|>",
        '<|start|>assistant<|channel|>final {"answer": "hi"}<|return|>',
    ]


def encode_output(harmony_str: str) -> list[int]:
    return get_encoding().encode(harmony_str, allowed_special="all")


def assistant(content: str, channel: str, content_type: str | None = None) -> Message:
    message = Message.from_role_and_content(Role.ASSISTANT, content).with_channel(
        channel
    )
    return message if content_type is None else message.with_content_type(content_type)


def tool_call(
    recipient: str,
    content: str,
    channel: str = "commentary",
    content_type: str | None = "json",
) -> Message:
    message = assistant(content, channel).with_recipient(recipient)
    return message if content_type is None else message.with_content_type(content_type)


def get_model_output_tokens(response_messages: Sequence[Message]) -> list[int]:
    enc = get_encoding()
    prompt_messages = [Message.from_role_and_content(Role.USER, "x")]
    # Keep analysis messages when synthesizing model-output-only token sequences
    # for parser tests; the default render path drops them after a later final turn.
    config = RenderConversationConfig(auto_drop_analysis=False)
    prompt_ids = enc.render_conversation_for_completion(
        Conversation.from_messages(prompt_messages),
        Role.ASSISTANT,
        config=config,
    )
    full_ids = enc.render_conversation(
        Conversation.from_messages([*prompt_messages, *response_messages]),
        config=config,
    )
    assert full_ids[: len(prompt_ids)] == prompt_ids
    return full_ids[len(prompt_ids) :]


def get_model_output_str(response_messages: Sequence[Message]) -> str:
    return get_encoding().decode_utf8(get_model_output_tokens(response_messages))


def get_text(msg: Message) -> str:
    return msg.content[0].text if msg.content else ""


def completed_message_rows(
    segments: Sequence[Any],
) -> list[tuple[str | None, str | None, str | None, str]]:
    return [
        (msg.channel, msg.recipient, msg.content_type, get_text(msg))
        for segment in segments
        if (msg := segment.completed_message) is not None
    ]


def tool_call_tuples(tool_calls: list[FunctionCall] | None) -> list[tuple[str, str]]:
    return [] if tool_calls is None else [(tc.name, tc.arguments) for tc in tool_calls]


def tool_call_headers(delta_message) -> list:
    if delta_message is None or not delta_message.tool_calls:
        return []
    return [
        tool_call
        for tool_call in delta_message.tool_calls
        if tool_call.function and tool_call.function.name
    ]


def tool_call_payloads(delta_message) -> list:
    if delta_message is None or not delta_message.tool_calls:
        return []
    return [
        tool_call
        for tool_call in delta_message.tool_calls
        if tool_call.function and tool_call.function.arguments
    ]


def tool_call_entries(delta_message) -> list[tuple[int, str | None, str | None]]:
    if delta_message is None or not delta_message.tool_calls:
        return []
    return [
        (
            tool_call.index,
            tool_call.function.name if tool_call.function else None,
            tool_call.function.arguments if tool_call.function else None,
        )
        for tool_call in delta_message.tool_calls
    ]


def collect_streamed_tool_calls(delta_messages) -> list[tuple[int, str, str]]:
    calls: dict[int, list[str]] = {}
    for delta_message in delta_messages:
        for tool_call in delta_message.tool_calls or []:
            function = tool_call.function
            assert function is not None
            if function.name is not None:
                assert tool_call.index not in calls
                calls[tool_call.index] = [function.name, function.arguments or ""]
            else:
                assert tool_call.index in calls
                calls[tool_call.index][1] += function.arguments or ""
    return [(index, name, arguments) for index, (name, arguments) in calls.items()]


def parse_delta_stream(
    harmony_parser, chat_request, output, chunk_size, monkeypatch
) -> tuple[list[Message], list[Any]]:
    completed_messages = []
    poll_completed_message = harmony_parser._poll_completed_message

    def capture_completed_message():
        message = poll_completed_message()
        if message is not None:
            completed_messages.append(message)
        return message

    monkeypatch.setattr(
        harmony_parser, "_poll_completed_message", capture_completed_message
    )
    delta_messages = []
    for start in range(0, len(output), chunk_size):
        delta = harmony_parser.parse_delta(
            delta_text="",
            delta_token_ids=output[start : start + chunk_size],
            request=chat_request,
            finished=False,
        )
        if delta is not None:
            delta_messages.append(delta)
    final_delta = harmony_parser.parse_delta(
        delta_text="",
        delta_token_ids=[],
        request=chat_request,
        finished=True,
    )
    if final_delta is not None:
        delta_messages.append(final_delta)
    return completed_messages, delta_messages


def streamed_delta_text(delta_messages) -> str:
    emitted_parts = []
    for delta in delta_messages:
        parts = [delta.reasoning or "", delta.content or ""]
        parts.extend(
            tool_call.function.arguments or ""
            for tool_call in delta.tool_calls or []
            if tool_call.function is not None
        )
        nonempty_parts = [part for part in parts if part]
        assert len(nonempty_parts) <= 1
        emitted_parts.extend(nonempty_parts)
    return "".join(emitted_parts)


def message_rows(messages: Sequence[Message]):
    return [
        (message.channel, message.recipient, message.content_type, get_text(message))
        for message in messages
    ]


def assert_parser_is_reset(harmony_parser: HarmonyParser):
    assert harmony_parser._parser is None
    assert harmony_parser._num_processed_messages == 0
    assert harmony_parser._current_message_tokens == []


class TestFlush:
    def test_flush(self, harmony_parser):
        harmony_parser.process_chunk(
            encode_output("<|channel|>analysis<|message|>Think")
        )

        flushed_segments = harmony_parser.flush()
        assert flushed_segments is not None
        assert len(flushed_segments) == 1
        flushed = flushed_segments[0]

        assert flushed is not None
        assert flushed.channel == "analysis"
        assert flushed.recipient is None
        assert flushed.delta == ""
        assert flushed.completed_message is not None
        assert get_text(flushed.completed_message) == "Think"
        assert_parser_is_reset(harmony_parser)

    def test_flush_recovers_invalid_output(self, harmony_parser, malformed_msgs_str):
        for msg_str in malformed_msgs_str[:-1]:
            chunk = harmony_parser.process_chunk(encode_output(msg_str))
            assert "".join(segment.delta for segment in chunk.segments) == "thinking"

        last_msg_str = malformed_msgs_str[-1]
        harmony_parser.process_chunk(encode_output(last_msg_str))
        flushed_segments = harmony_parser.flush()
        assert len(flushed_segments) == 2
        delta_segment = flushed_segments[0]
        message_segment = flushed_segments[1]

        assert delta_segment.channel == "final"
        assert delta_segment.recipient is None
        assert delta_segment.delta == last_msg_str
        assert message_segment.channel == "final"
        assert message_segment.recipient is None
        assert get_text(message_segment.completed_message) == last_msg_str
        assert_parser_is_reset(harmony_parser)


class TestParse:
    # Rendered conversation outputs.

    def test_reasoning_only(self, harmony_parser, chat_request):
        response = [assistant("This is reasoning", "analysis")]

        reasoning, content, tool_calls = harmony_parser.parse(
            "",
            chat_request,
            model_output_token_ids=get_model_output_tokens(response),
        )

        assert reasoning == "This is reasoning"
        assert content is None
        assert tool_calls is None

    def test_content_only(self, harmony_parser, chat_request):
        response = [assistant("This is a test", "final")]

        reasoning, content, tool_calls = harmony_parser.parse(
            "",
            chat_request,
            model_output_token_ids=get_model_output_tokens(response),
        )

        assert reasoning is None
        assert content == "This is a test"
        assert tool_calls is None

    def test_reasoning_and_content(self, harmony_parser, chat_request):
        response = [
            assistant("I should think first.", "analysis"),
            assistant("The answer is 4.", "final"),
        ]

        reasoning, content, tool_calls = harmony_parser.parse(
            "",
            chat_request,
            model_output_token_ids=get_model_output_tokens(response),
        )

        assert reasoning == "I should think first."
        assert content == "The answer is 4."
        assert tool_calls is None

    @pytest.mark.parametrize(
        "tool_args",
        [
            '{"location": "Tokyo"}',
            '{\n"location": "Tokyo"\n}',
        ],
    )
    @pytest.mark.parametrize("tool_channel", ["commentary", "analysis"])
    def test_single_tool_call(
        self, harmony_parser, chat_request, tool_args, tool_channel
    ):
        response = [tool_call("functions.get_current_weather", tool_args, tool_channel)]

        reasoning, content, tool_calls = harmony_parser.parse(
            "",
            chat_request,
            model_output_token_ids=get_model_output_tokens(response),
        )

        assert reasoning is None
        assert content is None
        assert tool_call_tuples(tool_calls) == [
            ("get_current_weather", json.dumps({"location": "Tokyo"}))
        ]

    def test_multiple_tool_calls_varied_formats(self, harmony_parser, chat_request):
        response = [
            tool_call("functions.get_current_weather", '{"location": "Tokyo"}'),
            tool_call("functions.get_user_location", '{"location": "Tokyo"}'),
            tool_call(
                "functions.no_content_type",
                '{"location": "Tokyo"}',
                content_type=None,
            ),
            tool_call("functions.not_json_no_content_type", "foo", content_type=None),
            tool_call("functions.empty_args", "{}"),
            tool_call("functions.no_args", ""),
        ]

        _, content, tool_calls = harmony_parser.parse(
            "",
            chat_request,
            model_output_token_ids=get_model_output_tokens(response),
        )

        assert content is None
        assert tool_call_tuples(tool_calls) == [
            ("get_current_weather", json.dumps({"location": "Tokyo"})),
            ("get_user_location", json.dumps({"location": "Tokyo"})),
            ("no_content_type", json.dumps({"location": "Tokyo"})),
            ("not_json_no_content_type", "foo"),
            ("empty_args", json.dumps({})),
            ("no_args", ""),
        ]

    def test_alternating_valid_and_repaired_messages(
        self, harmony_parser, chat_request
    ):
        output = encode_output(
            "<|channel|>analysis<|message|>think-1<|end|>"
            "<|start|>assistant<|channel|>commentary to=functions.one "
            '<|constrain|>analysis json<|message|>{"value":1}<|call|>'
            "<|start|>assistant<|channel|>final<|message|>answer-1<|end|>"
            "<|start|>assistant<|channel|>commentary to=functions.two "
            "<|constrain|>final code<|message|>raw-two<|call|>"
            "<|start|>assistant<|channel|>analysis<|message|>think-2<|end|>"
            "<|start|>assistant<|channel|>commentary to=functions.three "
            '<|constrain|>commentary json<|message|>{"value":3}<|call|>'
            "<|start|>assistant<|channel|>final<|message|>answer-2<|end|>"
        )

        reasoning, content, tool_calls = harmony_parser.parse(
            "", chat_request, model_output_token_ids=output
        )

        assert reasoning == "think-1\nthink-2"
        assert content == "answer-1\nanswer-2"
        assert tool_call_tuples(tool_calls) == [
            ("one", json.dumps({"value": 1})),
            ("two", "raw-two"),
            ("three", json.dumps({"value": 3})),
        ]

    def test_tool_call_bare_recipient(self, harmony_parser, chat_request):
        response = [tool_call("get_current_weather", '{"location": "Tokyo"}')]

        _, _, tool_calls = harmony_parser.parse(
            "",
            chat_request,
            model_output_token_ids=get_model_output_tokens(response),
        )

        assert tool_call_tuples(tool_calls) == [
            ("get_current_weather", json.dumps({"location": "Tokyo"}))
        ]

    def test_multiple_tool_calls_bare_recipients(self, harmony_parser, chat_request):
        response = [
            tool_call("get_current_weather", '{"location": "Tokyo"}'),
            tool_call("get_user_location", "{}"),
        ]

        _, _, tool_calls = harmony_parser.parse(
            "",
            chat_request,
            model_output_token_ids=get_model_output_tokens(response),
        )

        assert tool_call_tuples(tool_calls) == [
            ("get_current_weather", json.dumps({"location": "Tokyo"})),
            ("get_user_location", json.dumps({})),
        ]

    def test_assistant_recipient_not_tool(self, harmony_parser, chat_request):
        response = [
            tool_call("assistant", "Some tool response", content_type=None),
            assistant("Here is the answer", "final"),
        ]

        reasoning, content, tool_calls = harmony_parser.parse(
            "",
            chat_request,
            model_output_token_ids=get_model_output_tokens(response),
        )

        assert reasoning is None
        assert content == "Here is the answer"
        assert tool_calls is None

    def test_tool_call_dotted_name(self, harmony_parser, chat_request):
        response = [tool_call("math.sum", '{"a": 2, "b": 3}')]

        _, _, tool_calls = harmony_parser.parse(
            "",
            chat_request,
            model_output_token_ids=get_model_output_tokens(response),
        )

        assert tool_call_tuples(tool_calls) == [
            ("math.sum", json.dumps({"a": 2, "b": 3}))
        ]

    def test_tool_calls_with_final_content(self, harmony_parser, chat_request):
        response = [
            assistant("User asked about the weather.", "analysis"),
            tool_call("functions.get_current_weather", '{"location": "Tokyo"}'),
            assistant("This tool call will get the weather.", "final"),
        ]

        reasoning, content, tool_calls = harmony_parser.parse(
            "",
            chat_request,
            model_output_token_ids=get_model_output_tokens(response),
        )

        assert reasoning == "User asked about the weather."
        assert content == "This tool call will get the weather."
        assert tool_call_tuples(tool_calls) == [
            ("get_current_weather", json.dumps({"location": "Tokyo"}))
        ]

    # Raw/truncated Harmony output streams.

    def test_interrupted_first_message(self, harmony_parser, chat_request):
        reasoning, content, tool_calls = harmony_parser.parse(
            "",
            chat_request,
            model_output_token_ids=encode_output(
                "<|channel|>final<|message|>I'm in the middle of answering"
            ),
        )

        assert reasoning is None
        assert content == "I'm in the middle of answering"
        assert tool_calls is None
        assert_parser_is_reset(harmony_parser)

    def test_interrupted_reasoning_first_message(self, harmony_parser, chat_request):
        reasoning, content, tool_calls = harmony_parser.parse(
            "",
            chat_request,
            model_output_token_ids=encode_output(
                "<|channel|>analysis<|message|>I'm in the middle of thinking"
            ),
        )

        assert reasoning == "I'm in the middle of thinking"
        assert content is None
        assert tool_calls is None
        assert_parser_is_reset(harmony_parser)

    def test_truncated_output(self, harmony_parser, chat_request):
        reasoning, content, tool_calls = harmony_parser.parse(
            "",
            chat_request,
            model_output_token_ids=encode_output(
                "<|channel|>analysis<|message|>I'm thinking.<|end|>"
                "<|start|>assistant<|channel|>final<|message|>"
                "I'm in the middle of answering"
            ),
        )

        assert reasoning == "I'm thinking."
        assert content == "I'm in the middle of answering"
        assert tool_calls is None
        assert_parser_is_reset(harmony_parser)

    def test_malformed_msgs_recovers_raw_content(
        self, harmony_parser, chat_request, malformed_msgs_str
    ):
        combined_output = "".join(malformed_msgs_str)

        reasoning, content, tool_calls = harmony_parser.parse(
            "",
            chat_request,
            model_output_token_ids=encode_output(combined_output),
        )

        assert reasoning == "thinking"
        assert content == "thinking\n" + malformed_msgs_str[-1]
        assert tool_calls is None
        assert_parser_is_reset(harmony_parser)

    @pytest.mark.parametrize(
        ("harmony_str", "expected_content"),
        [
            (
                "<|channel|>commentary<|message|>I'll search for that",
                "I'll search for that",
            ),
            (
                "<|channel|>commentary<|message|>Let me look that up.<|end|>"
                "<|start|>assistant<|channel|>final<|message|>The answer is 42.<|end|>",
                "Let me look that up.\nThe answer is 42.",
            ),
        ],
    )
    def test_commentary_preambles(
        self,
        harmony_parser,
        chat_request,
        harmony_str,
        expected_content,
    ):
        reasoning, content, tool_calls = harmony_parser.parse(
            "",
            chat_request,
            model_output_token_ids=encode_output(harmony_str),
        )

        assert reasoning is None
        assert content == expected_content
        assert tool_calls is None

    def test_commentary_with_recipient_excluded(self, harmony_parser, chat_request):
        reasoning, content, tool_calls = harmony_parser.parse(
            "",
            chat_request,
            model_output_token_ids=encode_output(
                "<|channel|>commentary"
                "<|message|>Let me check the weather.<|end|>"
                "<|start|>assistant to=functions.get_weather"
                "<|channel|>commentary"
                '<|message|>{"location": "SF"}<|end|>'
            ),
        )

        assert reasoning is None
        assert content == "Let me check the weather."
        assert tool_call_tuples(tool_calls) == [
            ("get_weather", json.dumps({"location": "SF"}))
        ]


class TestParseDelta:
    def test_basic(self, gpt_oss_tokenizer, chat_request):
        parser = HarmonyParser(gpt_oss_tokenizer)

        first_delta = parser.parse_delta(
            delta_text="",
            delta_token_ids=encode_output("<|channel|>analysis<|message|>Thinking"),
            request=chat_request,
            finished=False,
        )
        second_delta = parser.parse_delta(
            delta_text="",
            delta_token_ids=encode_output(
                "<|end|><|start|>assistant<|channel|>final<|message|>Answer"
            ),
            request=chat_request,
            finished=True,
        )

        assert first_delta is not None
        assert first_delta.reasoning == "Thinking"
        assert first_delta.content is None
        assert second_delta is not None
        assert second_delta.content == "Answer"
        assert second_delta.reasoning is None
        assert_parser_is_reset(parser)

    def test_multi_token(self, gpt_oss_tokenizer, chat_request):
        parser = HarmonyParser(gpt_oss_tokenizer)

        delta = parser.parse_delta(
            delta_text="",
            delta_token_ids=encode_output("<|channel|>final<|message|>Hello, world!"),
            request=chat_request,
            finished=False,
        )

        assert delta is not None
        assert delta.content == "Hello, world!"
        assert delta.reasoning is None
        assert not delta.tool_calls

    def test_malformed_msgs_recovers_raw_content(
        self, gpt_oss_tokenizer, chat_request, malformed_msgs_str
    ):
        parser = HarmonyParser(gpt_oss_tokenizer)

        for msg_str in malformed_msgs_str[:-1]:
            delta = parser.parse_delta(
                delta_text="",
                delta_token_ids=encode_output(msg_str),
                request=chat_request,
                finished=False,
            )
            assert delta.reasoning or delta.content == "thinking"
            assert not delta.tool_calls

        last_delta = parser.parse_delta(
            delta_text="",
            delta_token_ids=encode_output(malformed_msgs_str[-1]),
            request=chat_request,
            finished=True,
        )

        assert last_delta is not None
        assert last_delta.content == malformed_msgs_str[-1]
        assert last_delta.reasoning is None
        assert not last_delta.tool_calls
        assert_parser_is_reset(parser)

    @pytest.mark.parametrize("tool_channel", ["commentary", "analysis"])
    def test_tool_call_split_across_deltas(
        self, gpt_oss_tokenizer, chat_request, tool_channel
    ):
        parser = HarmonyParser(gpt_oss_tokenizer)

        first_delta = parser.parse_delta(
            delta_text="",
            delta_token_ids=encode_output(
                "<|channel|>analysis<|message|>Thinking<|end|>"
                f"<|start|>assistant to=functions.get_weather<|channel|>{tool_channel}"
                '<|constrain|>json<|message|>{"location": '
            ),
            request=chat_request,
            finished=False,
        )
        second_delta = parser.parse_delta(
            delta_text="",
            delta_token_ids=encode_output('"Paris"}<|call|>'),
            request=chat_request,
            finished=False,
        )

        assert first_delta is not None
        assert first_delta.reasoning == "Thinking"
        assert first_delta.content is None
        assert tool_call_entries(first_delta) == [
            (0, "get_weather", '{"location": '),
        ]

        assert second_delta is not None
        assert second_delta.reasoning is None
        assert second_delta.content is None
        assert tool_call_entries(second_delta) == [(0, None, '"Paris"}')]

    def test_commentary_preamble_streaming(self, gpt_oss_tokenizer, chat_request):
        parser = HarmonyParser(gpt_oss_tokenizer)

        delta = parser.parse_delta(
            delta_text="",
            delta_token_ids=encode_output(
                "<|channel|>commentary<|message|>I'll search for that"
            ),
            request=chat_request,
            finished=False,
        )

        assert delta is not None
        assert delta.content == "I'll search for that"
        assert delta.reasoning is None
        assert not delta.tool_calls

    def test_multiple_choices(self, gpt_oss_tokenizer, chat_request):
        parser_a = HarmonyParser(gpt_oss_tokenizer)
        parser_b = HarmonyParser(gpt_oss_tokenizer)

        delta_a = parser_a.parse_delta(
            delta_text="",
            delta_token_ids=encode_output(
                "<|channel|>analysis<|message|>Check weather<|end|>"
                "<|start|>assistant to=functions.get_weather<|channel|>commentary"
                '<|constrain|>json<|message|>{"location": "Paris"}'
            ),
            request=chat_request,
            finished=False,
        )
        delta_b = parser_b.parse_delta(
            delta_text="",
            delta_token_ids=encode_output(
                "<|channel|>analysis<|message|>Check time<|end|>"
                "<|start|>assistant to=functions.get_time<|channel|>commentary"
                '<|constrain|>json<|message|>{"timezone": "UTC"}'
            ),
            request=chat_request,
            finished=False,
        )

        assert [tool.function.name for tool in tool_call_headers(delta_a)] == [
            "get_weather"
        ]
        assert [tool.function.name for tool in tool_call_headers(delta_b)] == [
            "get_time"
        ]
        assert {tool.index for tool in delta_a.tool_calls} == {0}
        assert {tool.index for tool in delta_b.tool_calls} == {0}

    def test_dotted_function_name(self, gpt_oss_tokenizer, chat_request):
        parser = HarmonyParser(gpt_oss_tokenizer)

        delta = parser.parse_delta(
            delta_text="",
            delta_token_ids=encode_output(
                "<|channel|>analysis<|message|>Compute this<|end|>"
                "<|start|>assistant to=math.sum<|channel|>commentary"
                '<|constrain|>json<|message|>{"a": 2, "b": 3}'
            ),
            request=chat_request,
            finished=False,
        )

        assert delta is not None
        assert [tool.function.name for tool in tool_call_headers(delta)] == ["math.sum"]
        assert {tool.index for tool in delta.tool_calls} == {0}

    @pytest.mark.parametrize("recipient", ["assistant", "browser"])
    def test_builtin_recipient_skipped(
        self,
        gpt_oss_tokenizer,
        chat_request,
        recipient,
    ):
        parser = HarmonyParser(gpt_oss_tokenizer)
        response = [tool_call(recipient, "Ignore this", content_type=None)]

        delta = parser.parse_delta(
            delta_text="",
            delta_token_ids=get_model_output_tokens(response),
            request=chat_request,
            finished=False,
        )

        assert delta is None

    def test_cross_channel_with_tool(self, gpt_oss_tokenizer, chat_request):
        parser = HarmonyParser(gpt_oss_tokenizer)

        delta = parser.parse_delta(
            delta_text="",
            delta_token_ids=encode_output(
                "<|channel|>analysis<|message|>Reasoning about query...<|end|>"
                "<|start|>assistant to=functions.search<|channel|>commentary"
                '<|constrain|>json<|message|>{"query": "vllm"}<|call|>'
                "<|start|>assistant<|channel|>final<|message|>Done"
            ),
            request=chat_request,
            finished=False,
        )

        assert delta is not None
        assert delta.reasoning == "Reasoning about query..."
        assert delta.content == "Done"
        assert tool_call_entries(delta) == [(0, "search", '{"query": "vllm"}')]

    def test_tool_index_across_calls(self, gpt_oss_tokenizer, chat_request):
        parser = HarmonyParser(gpt_oss_tokenizer)

        first_delta = parser.parse_delta(
            delta_text="",
            delta_token_ids=encode_output(
                "<|channel|>analysis<|message|>Thinking<|end|>"
                "<|start|>assistant to=functions.get_weather<|channel|>commentary"
                '<|constrain|>json<|message|>{"location": "Paris"}<|call|>'
            ),
            request=chat_request,
            finished=False,
        )
        second_delta = parser.parse_delta(
            delta_text="",
            delta_token_ids=encode_output(
                "<|start|>assistant to=functions.get_time<|channel|>commentary"
                '<|constrain|>json<|message|>{"timezone": "UTC"}<|call|>'
            ),
            request=chat_request,
            finished=False,
        )

        assert [tool.index for tool in tool_call_headers(first_delta)] == [0]
        assert [tool.index for tool in tool_call_headers(second_delta)] == [1]
        assert [tool.function.name for tool in tool_call_headers(second_delta)] == [
            "get_time"
        ]

    def test_multi_tool_interleaved(self, gpt_oss_tokenizer, chat_request):
        parser = HarmonyParser(gpt_oss_tokenizer)

        first_delta = parser.parse_delta(
            delta_text="",
            delta_token_ids=encode_output(
                "<|channel|>analysis<|message|>Plan<|end|>"
                "<|start|>assistant to=functions.tool_a<|channel|>commentary"
                '<|constrain|>json<|message|>{"a": 1}<|call|>'
                "<|start|>assistant to=functions.tool_b<|channel|>commentary"
                '<|constrain|>json<|message|>{"b": '
            ),
            request=chat_request,
            finished=False,
        )
        second_delta = parser.parse_delta(
            delta_text="",
            delta_token_ids=encode_output("2"),
            request=chat_request,
            finished=False,
        )
        third_delta = parser.parse_delta(
            delta_text="",
            delta_token_ids=encode_output(
                "}<|call|><|start|>assistant<|channel|>final<|message|>Done<|end|>"
                "<|start|>assistant to=functions.tool_c<|channel|>commentary"
                '<|constrain|>json<|message|>{"c": 3}'
            ),
            request=chat_request,
            finished=False,
        )

        assert tool_call_entries(first_delta) == [
            (0, "tool_a", '{"a": 1}'),
            (1, "tool_b", '{"b": '),
        ]
        assert [tool.index for tool in tool_call_headers(first_delta)] == [0, 1]

        assert second_delta is not None
        assert tool_call_entries(second_delta) == [(1, None, "2")]
        assert [tool.index for tool in tool_call_payloads(second_delta)] == [1]

        assert third_delta is not None
        assert third_delta.content == "Done"
        assert tool_call_entries(third_delta) == [
            (1, None, "}"),
            (2, "tool_c", '{"c": 3}'),
        ]
        assert [tool.index for tool in tool_call_headers(third_delta)] == [2]

    @pytest.mark.parametrize("chunk_size", [1, 4, 11])
    def test_alternating_valid_and_repaired_messages_stream_across_chunks(
        self,
        harmony_parser,
        gpt_oss_tokenizer,
        chat_request,
        monkeypatch,
        chunk_size,
    ):
        output = encode_output(
            "<|channel|>analysis<|message|>valid-1<|end|>"
            "<|start|>assistant<|channel|>commentary to=functions.first "
            "<|constrain|>analysis code<|message|>repair-1<|call|>"
            "<|start|>assistant<|channel|>final<|message|>valid-2<|end|>"
            "<|start|>assistant<|channel|>commentary to=functions.second "
            '<|constrain|>final json<|message|>{"repair":2}<|call|>'
            "<|start|>assistant<|channel|>commentary<|message|>valid-3<|end|>"
            "<|start|>assistant<|channel|>commentary to=functions.third "
            "<|constrain|>commentary to=assistant "
            "<|constrain|>analysis to=functions.third code"
            "<|message|>repair-3<|call|>"
            "<|start|>assistant<|channel|>final<|message|>valid-4"
        )
        bulk_reasoning_count = (
            HarmonyParser(gpt_oss_tokenizer).process_chunk(output).reasoning_token_count
        )
        completed_messages, delta_messages = parse_delta_stream(
            harmony_parser, chat_request, output, chunk_size, monkeypatch
        )

        assert message_rows(completed_messages) == [
            ("analysis", None, None, "valid-1"),
            (
                "commentary",
                "functions.first",
                "<|constrain|>code",
                "repair-1",
            ),
            ("final", None, None, "valid-2"),
            (
                "commentary",
                "functions.second",
                "<|constrain|>json",
                '{"repair":2}',
            ),
            ("commentary", None, None, "valid-3"),
            (
                "commentary",
                "functions.third",
                "<|constrain|>code",
                "repair-3",
            ),
            ("final", None, None, "valid-4"),
        ]
        assert "".join(delta.reasoning or "" for delta in delta_messages) == "valid-1"
        assert "".join(delta.content or "" for delta in delta_messages) == (
            "valid-2valid-3valid-4"
        )
        assert collect_streamed_tool_calls(delta_messages) == [
            (0, "first", "repair-1"),
            (1, "second", '{"repair":2}'),
            (2, "third", "repair-3"),
        ]
        if chunk_size == 1:
            assert streamed_delta_text(delta_messages) == (
                'valid-1repair-1valid-2{"repair":2}valid-3repair-3valid-4'
            )
        assert harmony_parser._num_counted_tokens == len(output)
        assert harmony_parser._num_reasoning_tokens == bulk_reasoning_count
        assert harmony_parser.count_reasoning_tokens(output) == bulk_reasoning_count
        assert harmony_parser._next_tool_call_index == 0
        assert_parser_is_reset(harmony_parser)

    def test_unrepairable_headers_stream_across_repair_and_flush(
        self, harmony_parser, chat_request, monkeypatch
    ):
        output = encode_output(
            "<|channel|>analysis<|message|>valid-before<|end|>"
            "<|start|>assistant<|channel|>commentary to=functions.before "
            "<|constrain|>analysis code<|message|>repair-before<|call|>"
            "<|start|>assistant<|channel|>commentary to=functions.ignored "
            "<|constrain|>analysis authored code<|message|>ignored-1<|call|>"
            "<|start|>assistant<|channel|>final<|message|>valid-middle<|end|>"
            "<|start|>assistant<|channel|>commentary to=functions.after "
            '<|constrain|>final json<|message|>{"repair":"after"}<|call|>'
            "<|start|>assistant<|channel|>commentary to=functions.ignored "
            "<|constrain|>analysis yaml<|message|>ignored-2<|call|>"
            "<|start|>assistant<|channel|>final<|message|>flush-tail"
        )
        completed_messages, delta_messages = parse_delta_stream(
            harmony_parser, chat_request, output, 1, monkeypatch
        )

        assert message_rows(completed_messages) == [
            ("analysis", None, None, "valid-before"),
            (
                "commentary",
                "functions.before",
                "<|constrain|>code",
                "repair-before",
            ),
            ("final", None, None, "valid-middle"),
            (
                "commentary",
                "functions.after",
                "<|constrain|>json",
                '{"repair":"after"}',
            ),
            ("final", None, None, "flush-tail"),
        ]
        assert streamed_delta_text(delta_messages) == (
            'valid-beforerepair-beforevalid-middle{"repair":"after"}flush-tail'
        )
        assert collect_streamed_tool_calls(delta_messages) == [
            (0, "before", "repair-before"),
            (1, "after", '{"repair":"after"}'),
        ]
        assert harmony_parser._num_counted_tokens == len(output)
        assert harmony_parser._next_tool_call_index == 0
        assert_parser_is_reset(harmony_parser)


class TestProcessChunk:
    def test_empty(self, harmony_parser):
        result = harmony_parser.process_chunk([])
        assert result.segments == []
        assert result.reasoning_token_count == 0

    def test_single_channel(self, harmony_parser):
        result = harmony_parser.process_chunk(
            encode_output("<|channel|>final<|message|>Hello")
        )

        assert [
            (s.channel, s.recipient, s.delta) for s in result.segments if s.delta
        ] == [("final", None, "Hello")]

    def test_constrained_output_segment_recipient_normalized(self, harmony_parser):
        result = harmony_parser.process_chunk(
            encode_output(
                '<|channel|>final <|constrain|>json<|message|>{"result":true}<|end|>'
            )
        )

        content_segments = [segment for segment in result.segments if segment.delta]
        assert all(segment.channel == "final" for segment in content_segments)
        assert all(segment.recipient is None for segment in content_segments)
        assert (
            "".join(segment.delta for segment in content_segments) == '{"result":true}'
        )
        completed_messages = [
            segment.completed_message
            for segment in result.segments
            if segment.completed_message is not None
        ]
        assert len(completed_messages) == 1
        assert completed_messages[0].recipient is None

    def test_cross_channel(self, harmony_parser):
        result = harmony_parser.process_chunk(
            encode_output(
                "<|channel|>analysis<|message|>Think<|end|>"
                "<|start|>assistant<|channel|>final<|message|>Answer"
            )
        )

        assert [
            (s.channel, s.recipient, s.delta) for s in result.segments if s.delta
        ] == [
            ("analysis", None, "Think"),
            ("final", None, "Answer"),
        ]

    def test_multi_boundary(self, harmony_parser):
        result = harmony_parser.process_chunk(
            encode_output(
                "<|channel|>analysis<|message|>One<|end|>"
                "<|start|>assistant<|channel|>final<|message|>Two<|end|>"
            )
        )

        boundary_segments = [
            segment
            for segment in result.segments
            if segment.completed_message is not None
        ]
        assert [
            (segment.completed_message.channel, get_text(segment.completed_message))
            for segment in boundary_segments
        ] == [
            ("analysis", "One"),
            ("final", "Two"),
        ]

    def test_malformed_token_does_not_raise(self, harmony_parser):
        malformed = encode_output(
            "<|channel|>analysis<|message|>think<|end|><|return|>"
            "<|start|>assistant<|channel|>final<|message|>answer<|return|>"
        )

        result = harmony_parser.process_chunk(malformed)
        assert "".join(s.delta for s in result.segments if s.delta) == "thinkanswer"

    @pytest.mark.parametrize("recipient", ["python", "assistant"])
    def test_malformed_header_drops_channel_from_content_type(
        self, harmony_parser, recipient
    ):
        malformed = encode_output(
            f"<|channel|>commentary to={recipient} "
            "<|constrain|>analysis code<|message|>print(6 * 7)<|call|>"
            "<|start|>assistant<|channel|>final<|message|>Done<|end|>"
        )

        result = harmony_parser.process_chunk(malformed)
        messages = [
            segment.completed_message
            for segment in result.segments
            if segment.completed_message is not None
        ]

        assert [(msg.channel, msg.recipient, msg.content_type) for msg in messages] == [
            ("commentary", recipient, "<|constrain|>code"),
            ("final", None, None),
        ]
        assert get_text(messages[0]) == "print(6 * 7)"
        assert get_text(messages[1]) == "Done"

    def test_malformed_header_drops_recipient_from_content_type(self, harmony_parser):
        malformed = encode_output(
            "<|channel|>commentary to=python "
            "<|constrain|>python code<|message|>print(6 * 7)<|call|>"
        )

        result = harmony_parser.process_chunk(malformed)
        messages = [
            segment.completed_message
            for segment in result.segments
            if segment.completed_message is not None
        ]

        assert [(msg.channel, msg.recipient, msg.content_type) for msg in messages] == [
            ("commentary", "python", "<|constrain|>code")
        ]
        assert get_text(messages[0]) == "print(6 * 7)"

    def test_malformed_header_drops_spliced_metadata_from_content_type(
        self, harmony_parser
    ):
        malformed = encode_output(
            "<|start|>assistant<|channel|>commentary to=python "
            "<|constrain|>commentary to=assistant "
            "<|constrain|>analysis to=python code"
            "<|message|>print(6 * 7)<|call|>"
        )

        result = harmony_parser.process_chunk(malformed)
        messages = [
            segment.completed_message
            for segment in result.segments
            if segment.completed_message is not None
        ]

        assert [(msg.channel, msg.recipient, msg.content_type) for msg in messages] == [
            ("commentary", "python", "<|constrain|>code")
        ]
        assert get_text(messages[0]) == "print(6 * 7)"

    @pytest.mark.parametrize(
        "duplicate_markers",
        [
            "<|constrain|>",
            "<|channel|><|constrain|>",
            "<|constrain|><|channel|><|constrain|>",
        ],
    )
    def test_malformed_header_drops_repeated_bare_markers(
        self, harmony_parser, duplicate_markers
    ):
        malformed = encode_output(
            "<|channel|>commentary to=python "
            f"<|constrain|>{duplicate_markers}code"
            "<|message|>print(6 * 7)<|call|>"
        )

        result = harmony_parser.process_chunk(malformed)

        assert completed_message_rows(result.segments) == [
            ("commentary", "python", "<|constrain|>code", "print(6 * 7)")
        ]

    def test_alternating_valid_and_repaired_messages_are_delivered_once(
        self, harmony_parser
    ):
        output = encode_output(
            "<|channel|>analysis<|message|>valid-1<|end|>"
            "<|start|>assistant<|channel|>commentary to=python "
            "<|constrain|>analysis code<|message|>repair-1<|call|>"
            "<|start|>assistant<|channel|>final<|message|>valid-2<|end|>"
            "<|start|>assistant<|channel|>commentary to=functions.second "
            '<|constrain|>final json<|message|>{"repair":2}<|call|>'
            "<|start|>assistant<|channel|>commentary<|message|>valid-3<|end|>"
            "<|start|>assistant<|channel|>commentary to=assistant "
            "<|constrain|>commentary to=assistant "
            "<|constrain|>analysis to=assistant code"
            "<|message|>repair-3<|call|>"
            "<|start|>assistant<|channel|>final<|message|>valid-4<|end|>"
        )

        result = harmony_parser.process_chunk(output)
        messages = completed_message_rows(result.segments)

        assert len(messages) == 7
        assert messages == [
            ("analysis", None, None, "valid-1"),
            ("commentary", "python", "<|constrain|>code", "repair-1"),
            ("final", None, None, "valid-2"),
            (
                "commentary",
                "functions.second",
                "<|constrain|>json",
                '{"repair":2}',
            ),
            ("commentary", None, None, "valid-3"),
            ("commentary", "assistant", "<|constrain|>code", "repair-3"),
            ("final", None, None, "valid-4"),
        ]

    def test_unrepairable_headers_preserve_cursor_across_repair_and_flush(
        self, harmony_parser
    ):
        output = encode_output(
            "<|channel|>analysis<|message|>valid-before<|end|>"
            "<|start|>assistant<|channel|>commentary to=python "
            "<|constrain|>analysis code<|message|>repair-before<|call|>"
            "<|start|>assistant<|channel|>commentary to=python "
            "<|constrain|>analysis authored code<|message|>ignored-1<|call|>"
            "<|start|>assistant<|channel|>final<|message|>valid-middle<|end|>"
            "<|start|>assistant<|channel|>commentary to=functions.after "
            '<|constrain|>final json<|message|>{"repair":"after"}<|call|>'
            "<|start|>assistant<|channel|>commentary to=python "
            "<|constrain|>analysis yaml<|message|>ignored-2<|call|>"
            "<|start|>assistant<|channel|>final<|message|>flush-tail"
        )

        result = harmony_parser.process_chunk(output)
        flushed = harmony_parser.flush()
        messages = completed_message_rows([*result.segments, *flushed])

        assert len(messages) == 5
        assert messages == [
            ("analysis", None, None, "valid-before"),
            ("commentary", "python", "<|constrain|>code", "repair-before"),
            ("final", None, None, "valid-middle"),
            (
                "commentary",
                "functions.after",
                "<|constrain|>json",
                '{"repair":"after"}',
            ),
            ("final", None, None, "flush-tail"),
        ]

    def test_malformed_header_on_second_message_preserves_both(self, harmony_parser):
        output = encode_output(
            "<|channel|>analysis<|message|>First<|end|>"
            "<|start|>assistant<|channel|>commentary to=python "
            "<|constrain|>commentary to=assistant "
            "<|constrain|>analysis to=python code"
            "<|message|>print(6 * 7)<|call|>"
        )

        result = harmony_parser.process_chunk(output)
        messages = [
            segment.completed_message
            for segment in result.segments
            if segment.completed_message is not None
        ]

        assert [
            (msg.channel, msg.recipient, msg.content_type, get_text(msg))
            for msg in messages
        ] == [
            ("analysis", None, None, "First"),
            ("commentary", "python", "<|constrain|>code", "print(6 * 7)"),
        ]

    def test_repair_uses_latest_start_after_rejected_header(self, harmony_parser):
        output = encode_output(
            "<|channel|>analysis<|message|>First<|end|>"
            "<|start|>assistant<|channel|>commentary to=python "
            "<|constrain|>analysis authored code<|message|>ignored<|call|>"
            "<|start|>assistant<|channel|>commentary to=python "
            "<|constrain|>analysis code<|message|>print(42)<|call|>"
        )

        result = harmony_parser.process_chunk(output)
        messages = [
            segment.completed_message
            for segment in result.segments
            if segment.completed_message is not None
        ]

        assert [get_text(msg) for msg in messages] == ["First", "print(42)"]

    def test_malformed_initial_header_uses_first_channel(self, harmony_parser):
        output = encode_output(
            "<|channel|>commentary to=python "
            "<|constrain|><|channel|>analysis code"
            "<|message|>print(42)<|call|>"
        )

        result = harmony_parser.process_chunk(output)
        messages = [
            segment.completed_message
            for segment in result.segments
            if segment.completed_message is not None
        ]

        assert [
            (msg.channel, msg.recipient, msg.content_type, get_text(msg))
            for msg in messages
        ] == [
            ("commentary", "python", "<|constrain|>code", "print(42)"),
        ]

    @pytest.mark.parametrize(
        "malformed_section",
        ["<|constrain|>", "analysis yaml", "analysis authored code"],
    )
    def test_repair_does_not_drop_unrecognized_header_text(
        self, harmony_parser, malformed_section
    ):
        output = encode_output(
            "<|channel|>commentary to=python "
            f"<|constrain|>{malformed_section}<|message|>ignored<|call|>"
            "<|start|>assistant<|channel|>final<|message|>Kept<|end|>"
        )

        result = harmony_parser.process_chunk(output)
        messages = [
            segment.completed_message
            for segment in result.segments
            if segment.completed_message is not None
        ]

        assert [(msg.channel, get_text(msg)) for msg in messages] == [("final", "Kept")]


class TestCountReasoningTokens:
    def test_matches_process_chunk(self, harmony_parser, gpt_oss_tokenizer):
        """Chat usage must report the same count as the Responses API path."""
        token_ids = get_model_output_tokens(
            [
                assistant("Let me check the weather.", "analysis"),
                tool_call("functions.get_weather", '{"location": "SF"}'),
            ]
        )
        count = harmony_parser.count_reasoning_tokens(token_ids)

        fresh = HarmonyParser(gpt_oss_tokenizer)
        assert count == fresh.process_chunk(token_ids).reasoning_token_count
        assert count > 0

    def test_excludes_harmony_control_tokens(self, harmony_parser):
        text = "Let me think about this."
        token_ids = get_model_output_tokens(
            [assistant(text, "analysis"), assistant("Hi", "final")]
        )
        expected = len(get_encoding().encode(text))
        assert harmony_parser.count_reasoning_tokens(token_ids) == expected

    def test_final_only_has_no_reasoning(self, harmony_parser):
        token_ids = get_model_output_tokens([assistant("Hello", "final")])
        assert harmony_parser.count_reasoning_tokens(token_ids) == 0

    def test_streaming_reuses_running_total(
        self, harmony_parser, gpt_oss_tokenizer, chat_request, monkeypatch
    ):
        """Called per streamed chunk with all tokens so far; must not replay."""
        token_ids = get_model_output_tokens(
            [assistant("Think", "analysis"), assistant("Answer", "final")]
        )
        expected = HarmonyParser(gpt_oss_tokenizer).count_reasoning_tokens(token_ids)
        harmony_parser.parse_delta("", token_ids[:3], chat_request, finished=False)

        def fail():
            raise AssertionError("count_reasoning_tokens replayed the output")

        monkeypatch.setattr(
            "vllm.parser.harmony.get_streamable_parser_for_assistant", fail
        )
        for end in range(3, len(token_ids)):
            harmony_parser.process_chunk(token_ids[end : end + 1])
            harmony_parser.count_reasoning_tokens(token_ids[: end + 1])
        assert harmony_parser.count_reasoning_tokens(token_ids) == expected

    def test_does_not_disturb_stream_state(self, harmony_parser, chat_request):
        """Streaming counts after each chunk, so it must not consume parser state."""
        token_ids = get_model_output_tokens(
            [assistant("Think", "analysis"), assistant("Answer", "final")]
        )
        harmony_parser.count_reasoning_tokens(token_ids)

        reasoning, content, _ = harmony_parser.parse(
            "", chat_request, model_output_token_ids=token_ids
        )
        assert (reasoning, content) == ("Think", "Answer")


class TestAdjustRequest:
    REQUEST_TEXT = "Hello"
    TOOL_TYPE = "function"
    TOOL_1_NAME = "get_user_location"
    TOOL_2_NAME = "get_weather"
    TOOLS = [
        {
            "name": TOOL_1_NAME,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        {
            "name": TOOL_2_NAME,
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    ]
    OUTPUT_SCHEMA = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }

    ANALYSIS = get_model_output_str([assistant("reasoning", "analysis")])
    COMMENTARY = get_model_output_str([assistant("commentary", "commentary")])
    TOOL_CALL_1_CHANNEL_FIRST = (
        ANALYSIS + f"<|start|>assistant<|channel|>commentary to=functions.{TOOL_1_NAME}"
        " <|constrain|>json<|message|>{}<|call|>"
    )
    TOOL_CALL_1_FUNCTION_FIRST = get_model_output_str(
        [
            assistant("reasoning", "analysis"),
            tool_call(f"functions.{TOOL_1_NAME}", "{}"),
        ]
    )
    TOOL_CALL_2_CHANNEL_FIRST = (
        ANALYSIS + f"<|start|>assistant<|channel|>commentary to=functions.{TOOL_2_NAME}"
        ' <|constrain|>json<|message|>{"city": "Tokyo"}<|call|>'
    )
    TOOL_CALL_2_FUNCTION_FIRST = get_model_output_str(
        [
            assistant("reasoning", "analysis"),
            tool_call(f"functions.{TOOL_2_NAME}", '{"city": "Tokyo"}'),
        ]
    )
    FINAL_JSON_SCHEMA = get_model_output_str(
        [
            assistant("reasoning", "analysis"),
            assistant('{"answer": "Tokyo"}', "final", "json"),
        ]
    )
    FINAL_JSON_OBJECT = get_model_output_str(
        [
            assistant("reasoning", "analysis"),
            assistant('{"city": "Tokyo"}', "final", "<|constrain|>json"),
        ]
    )
    FINAL_TEXT_ONLY = get_model_output_str(
        [assistant("reasoning", "analysis"), assistant("any", "final")]
    )
    FINAL_REGEX = get_model_output_str(
        [assistant("reasoning", "analysis"), assistant("regex", "final")]
    )
    FINAL_CHOICE = get_model_output_str(
        [assistant("reasoning", "analysis"), assistant("choice1", "final")]
    )
    FINAL_GRAMMAR = get_model_output_str(
        [assistant("reasoning", "analysis"), assistant("grammar", "final")]
    )
    FINAL_STRUCTURAL_TAG = get_model_output_str(
        [assistant("reasoning", "analysis"), assistant("tag content", "final")]
    )
    ADMISSION_SAMPLES = (
        "COMMENTARY",
        "TOOL_CALL_1_CHANNEL_FIRST",
        "TOOL_CALL_1_FUNCTION_FIRST",
        "TOOL_CALL_2_CHANNEL_FIRST",
        "TOOL_CALL_2_FUNCTION_FIRST",
        "FINAL_JSON_SCHEMA",
        "FINAL_JSON_OBJECT",
        "FINAL_TEXT_ONLY",
        "FINAL_REGEX",
        "FINAL_CHOICE",
        "FINAL_GRAMMAR",
        "FINAL_STRUCTURAL_TAG",
    )

    @staticmethod
    def _build_request(
        request_kind: Literal["chat", "responses"],
        tool_choice: str = "none",
        strict_tools: bool = False,
        response_format_type: str | None = None,
        structured_outputs: StructuredOutputsParams | None = None,
    ) -> ChatCompletionRequest | ResponsesRequest:
        data: dict[str, Any] = {
            "model": REASONING_MODEL_NAME,
        }
        if request_kind == "chat":
            data["messages"] = [
                {
                    "role": "user",
                    "content": TestAdjustRequest.REQUEST_TEXT,
                }
            ]
        else:
            data["input"] = TestAdjustRequest.REQUEST_TEXT

        if request_kind == "chat":
            data["tools"] = [
                {
                    "type": TestAdjustRequest.TOOL_TYPE,
                    "function": {"strict": strict_tools, **tool_def},
                }
                for tool_def in TestAdjustRequest.TOOLS
            ]
            data["tool_choice"] = (
                {
                    "type": TestAdjustRequest.TOOL_TYPE,
                    "function": {"name": TestAdjustRequest.TOOL_2_NAME},
                }
                if tool_choice == "named"
                else tool_choice
            )
        else:
            data["tools"] = [
                {
                    "type": TestAdjustRequest.TOOL_TYPE,
                    "strict": strict_tools,
                    **tool_def,
                }
                for tool_def in TestAdjustRequest.TOOLS
            ]
            data["tool_choice"] = (
                {
                    "type": TestAdjustRequest.TOOL_TYPE,
                    "name": TestAdjustRequest.TOOL_2_NAME,
                }
                if tool_choice == "named"
                else tool_choice
            )

        if response_format_type == "json_schema":
            schema_format = {
                "name": "answer_format",
                "schema": TestAdjustRequest.OUTPUT_SCHEMA,
                "strict": True,
            }
            if request_kind == "chat":
                data["response_format"] = {
                    "type": "json_schema",
                    "json_schema": schema_format,
                }
            else:
                data["text"] = {
                    "format": {
                        "type": "json_schema",
                        **schema_format,
                    }
                }
        elif response_format_type == "json_object":
            if request_kind == "chat":
                data["response_format"] = {"type": "json_object"}
            else:
                data["text"] = {"format": {"type": "json_object"}}

        if structured_outputs is not None:
            data["structured_outputs"] = structured_outputs

        if request_kind == "chat":
            return ChatCompletionRequest.model_validate(data)
        return ResponsesRequest.model_validate(data)

    @staticmethod
    def _assert_format_cleared(
        adjusted_request: ChatCompletionRequest | ResponsesRequest,
    ) -> None:
        if isinstance(adjusted_request, ResponsesRequest):
            assert adjusted_request.text is None or adjusted_request.text.format is None
        else:
            assert adjusted_request.response_format is None

        structured_outputs = adjusted_request.structured_outputs
        assert structured_outputs is not None
        assert structured_outputs.structural_tag is not None
        assert structured_outputs.all_non_structural_tag_constraints_none()

    @classmethod
    def _assert_structured_outputs_admission(
        cls,
        adjusted_request: ChatCompletionRequest | ResponsesRequest,
        expected_admission: Sequence[str],
        xgrammar_backend: XgrammarBackend,
        stop_token_ids: set[int],
    ) -> None:
        structured_outputs = adjusted_request.structured_outputs
        assert structured_outputs is not None
        assert structured_outputs.structural_tag is not None
        assert structured_outputs.all_non_structural_tag_constraints_none()

        grammar = xgrammar_backend.compile_grammar(
            StructuredOutputOptions.STRUCTURAL_TAG,
            structured_outputs.structural_tag,
            stop_token_ids=stop_token_ids,
        )
        expected_admission_set = set(expected_admission)

        for sample_name in cls.ADMISSION_SAMPLES:
            tokens = encode_output(getattr(cls, sample_name))
            accepted = grammar.validate_tokens(tokens)
            admitted = accepted == tokens
            should_admit = sample_name in expected_admission_set
            assert admitted is should_admit, (
                f"Expected structured_outputs admission for {sample_name} "
                f"to be {should_admit}, got {admitted}."
            )

    @pytest.mark.parametrize("request_kind", ["chat", "responses"])
    @pytest.mark.parametrize(
        ("request_kwargs", "expected_admission"),
        [
            (
                {"tool_choice": "auto", "strict_tools": True},
                [
                    "COMMENTARY",
                    "TOOL_CALL_1_CHANNEL_FIRST",
                    "TOOL_CALL_1_FUNCTION_FIRST",
                    "TOOL_CALL_2_CHANNEL_FIRST",
                    "TOOL_CALL_2_FUNCTION_FIRST",
                    "FINAL_JSON_SCHEMA",
                    "FINAL_JSON_OBJECT",
                    "FINAL_TEXT_ONLY",
                    "FINAL_REGEX",
                    "FINAL_CHOICE",
                    "FINAL_GRAMMAR",
                    "FINAL_STRUCTURAL_TAG",
                ],
            ),
            (
                {"tool_choice": "required"},
                [
                    "COMMENTARY",
                    "TOOL_CALL_1_CHANNEL_FIRST",
                    "TOOL_CALL_1_FUNCTION_FIRST",
                    "TOOL_CALL_2_CHANNEL_FIRST",
                    "TOOL_CALL_2_FUNCTION_FIRST",
                ],
            ),
            (
                {"tool_choice": "named"},
                [
                    "COMMENTARY",
                    "TOOL_CALL_2_CHANNEL_FIRST",
                    "TOOL_CALL_2_FUNCTION_FIRST",
                ],
            ),
            (
                {
                    "tool_choice": "auto",
                    "strict_tools": True,
                    "response_format_type": "json_schema",
                },
                [
                    "COMMENTARY",
                    "TOOL_CALL_1_CHANNEL_FIRST",
                    "TOOL_CALL_1_FUNCTION_FIRST",
                    "TOOL_CALL_2_CHANNEL_FIRST",
                    "TOOL_CALL_2_FUNCTION_FIRST",
                    "FINAL_JSON_SCHEMA",
                ],
            ),
            (
                {"response_format_type": "json_schema"},
                ["FINAL_JSON_SCHEMA"],
            ),
            (
                {"response_format_type": "json_object"},
                ["FINAL_JSON_SCHEMA", "FINAL_JSON_OBJECT"],
            ),
            (
                {"structured_outputs": StructuredOutputsParams(json=OUTPUT_SCHEMA)},
                ["FINAL_JSON_SCHEMA"],
            ),
            (
                {"structured_outputs": StructuredOutputsParams(json_object=True)},
                ["FINAL_JSON_SCHEMA", "FINAL_JSON_OBJECT"],
            ),
            (
                {"structured_outputs": StructuredOutputsParams(regex=r"regex")},
                ["FINAL_REGEX"],
            ),
            (
                {
                    "structured_outputs": StructuredOutputsParams(
                        choice=["choice1", "choice2"]
                    )
                },
                ["FINAL_CHOICE"],
            ),
            (
                {
                    "structured_outputs": StructuredOutputsParams(
                        grammar='root ::= "grammar"'
                    )
                },
                ["FINAL_GRAMMAR"],
            ),
            (
                {
                    "structured_outputs": StructuredOutputsParams(
                        structural_tag=json.dumps(
                            {
                                "type": "structural_tag",
                                "format": {
                                    "type": "json_schema",
                                    "json_schema": OUTPUT_SCHEMA,
                                },
                            }
                        )
                    )
                },
                ["FINAL_JSON_SCHEMA"],
            ),
            (
                {
                    "structured_outputs": StructuredOutputsParams(
                        structural_tag=json.dumps(
                            {
                                "type": "structural_tag",
                                "structures": [
                                    {
                                        "begin": "<tag>",
                                        "schema": {"type": "object"},
                                        "end": "</tag>",
                                    }
                                ],
                                "triggers": ["<tag>"],
                            }
                        )
                    )
                },
                [
                    # Legacy triggered tags allow free text until a trigger, so
                    # unconstrained final-channel payloads are also admitted.
                    "FINAL_TEXT_ONLY",
                    "FINAL_REGEX",
                    "FINAL_CHOICE",
                    "FINAL_GRAMMAR",
                    "FINAL_STRUCTURAL_TAG",
                ],
            ),
        ],
        ids=[
            "tool_auto_strict",
            "tool_required",
            "tool_named",
            "pr56086_tool_auto_strict_response_format_json_schema",
            "response_format_json_schema",
            "response_format_json_object",
            "structured_outputs_json",
            "structured_outputs_json_object",
            "structured_outputs_regex",
            "structured_outputs_choice",
            "structured_outputs_grammar",
            "structured_outputs_structural_tag_modern",
            "structured_outputs_structural_tag_legacy",
        ],
    )
    def test_adjust_request(
        self,
        harmony_parser,
        xgrammar_backend,
        gpt_oss_stop_token_ids,
        request_kind,
        request_kwargs,
        expected_admission,
    ):
        request = self._build_request(request_kind, **request_kwargs)
        adjusted_request = harmony_parser.adjust_request(request)
        self._assert_format_cleared(adjusted_request)
        self._assert_structured_outputs_admission(
            adjusted_request,
            expected_admission,
            xgrammar_backend,
            gpt_oss_stop_token_ids,
        )
