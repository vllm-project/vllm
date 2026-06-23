# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence

import pytest
from openai_harmony import (
    Conversation,
    Message,
    RenderConversationConfig,
    Role,
)
from transformers import AutoTokenizer

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import FunctionCall
from vllm.entrypoints.openai.parser.harmony_utils import (
    get_encoding,
)
from vllm.parser.harmony import HarmonyParser, HarmonyParserError
from vllm.parser.parser_manager import ParserManager

REASONING_MODEL_NAME = "openai/gpt-oss-20b"


@pytest.fixture(scope="module")
def gpt_oss_tokenizer():
    return AutoTokenizer.from_pretrained(REASONING_MODEL_NAME)


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


def encode_output(harmony_str: str) -> list[int]:
    return get_encoding().encode(harmony_str, allowed_special="all")


def assistant(content: str, channel: str) -> Message:
    return Message.from_role_and_content(Role.ASSISTANT, content).with_channel(channel)


def tool_call(
    recipient: str,
    content: str,
    channel: str = "commentary",
    content_type: str | None = "json",
) -> Message:
    message = assistant(content, channel).with_recipient(recipient)
    return message if content_type is None else message.with_content_type(content_type)


def get_model_output_tokens(
    prompt_messages: Sequence[Message],
    response_messages: Sequence[Message],
) -> list[int]:
    enc = get_encoding()
    # Keep analysis messages when synthesizing model-output-only token sequences
    # for parser tests; the default render path drops them after a later final turn.
    config = RenderConversationConfig(auto_drop_analysis=False)
    prompt_ids = enc.render_conversation_for_completion(
        Conversation.from_messages(list(prompt_messages)),
        Role.ASSISTANT,
        config=config,
    )
    full_ids = enc.render_conversation_for_completion(
        Conversation.from_messages([*prompt_messages, *response_messages]),
        Role.ASSISTANT,
        config=config,
    )
    assert full_ids[: len(prompt_ids)] == prompt_ids
    return full_ids[len(prompt_ids) :]


def get_text(msg: Message) -> str:
    return msg.content[0].text if msg.content else ""


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


class TestFlush:
    def test_flush(self, harmony_parser):
        harmony_parser.process_chunk(
            encode_output("<|channel|>analysis<|message|>Think")
        )

        flushed = harmony_parser.flush()

        assert flushed is not None
        assert flushed.channel == "analysis"
        assert flushed.recipient is None
        assert flushed.delta == ""
        assert flushed.completed_message is not None
        assert get_text(flushed.completed_message) == "Think"
        assert harmony_parser._parser is None

    def test_flush_resets_after_eos_error(self, harmony_parser):
        harmony_parser.process_chunk(encode_output("<|channel|>analysis"))

        flushed = harmony_parser.flush()

        assert flushed is None
        assert harmony_parser._parser is None


class TestParse:
    # Rendered conversation outputs.

    def test_reasoning_only(self, harmony_parser, chat_request):
        prompt = [Message.from_role_and_content(Role.USER, "Why?")]
        response = [assistant("This is reasoning", "analysis")]

        reasoning, content, tool_calls = harmony_parser.parse(
            "",
            chat_request,
            model_output_token_ids=get_model_output_tokens(prompt, response),
        )

        assert reasoning == "This is reasoning"
        assert content is None
        assert tool_calls is None

    def test_content_only(self, harmony_parser, chat_request):
        prompt = [Message.from_role_and_content(Role.USER, "Hello")]
        response = [assistant("This is a test", "final")]

        reasoning, content, tool_calls = harmony_parser.parse(
            "",
            chat_request,
            model_output_token_ids=get_model_output_tokens(prompt, response),
        )

        assert reasoning is None
        assert content == "This is a test"
        assert tool_calls is None

    def test_reasoning_and_content(self, harmony_parser, chat_request):
        prompt = [Message.from_role_and_content(Role.USER, "What is 2+2?")]
        response = [
            assistant("I should think first.", "analysis"),
            assistant("The answer is 4.", "final"),
        ]

        reasoning, content, tool_calls = harmony_parser.parse(
            "",
            chat_request,
            model_output_token_ids=get_model_output_tokens(prompt, response),
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
        prompt = [
            Message.from_role_and_content(Role.USER, "What is the weather in Tokyo?")
        ]
        response = [tool_call("functions.get_current_weather", tool_args, tool_channel)]

        reasoning, content, tool_calls = harmony_parser.parse(
            "",
            chat_request,
            model_output_token_ids=get_model_output_tokens(prompt, response),
        )

        assert reasoning is None
        assert content is None
        assert tool_call_tuples(tool_calls) == [
            ("get_current_weather", json.dumps({"location": "Tokyo"}))
        ]

    def test_multiple_tool_calls_varied_formats(self, harmony_parser, chat_request):
        prompt = [
            Message.from_role_and_content(
                Role.USER, "What is the weather in Tokyo based on where I'm at?"
            )
        ]
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
            model_output_token_ids=get_model_output_tokens(prompt, response),
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

    def test_tool_call_bare_recipient(self, harmony_parser, chat_request):
        prompt = [Message.from_role_and_content(Role.USER, "Weather?")]
        response = [tool_call("get_current_weather", '{"location": "Tokyo"}')]

        _, _, tool_calls = harmony_parser.parse(
            "",
            chat_request,
            model_output_token_ids=get_model_output_tokens(prompt, response),
        )

        assert tool_call_tuples(tool_calls) == [
            ("get_current_weather", json.dumps({"location": "Tokyo"}))
        ]

    def test_multiple_tool_calls_bare_recipients(self, harmony_parser, chat_request):
        prompt = [Message.from_role_and_content(Role.USER, "Use both tools.")]
        response = [
            tool_call("get_current_weather", '{"location": "Tokyo"}'),
            tool_call("get_user_location", "{}"),
        ]

        _, _, tool_calls = harmony_parser.parse(
            "",
            chat_request,
            model_output_token_ids=get_model_output_tokens(prompt, response),
        )

        assert tool_call_tuples(tool_calls) == [
            ("get_current_weather", json.dumps({"location": "Tokyo"})),
            ("get_user_location", json.dumps({})),
        ]

    def test_assistant_recipient_not_tool(self, harmony_parser, chat_request):
        prompt = [Message.from_role_and_content(Role.USER, "Hello")]
        response = [
            tool_call("assistant", "Some tool response", content_type=None),
            assistant("Here is the answer", "final"),
        ]

        reasoning, content, tool_calls = harmony_parser.parse(
            "",
            chat_request,
            model_output_token_ids=get_model_output_tokens(prompt, response),
        )

        assert reasoning is None
        assert content == "Here is the answer"
        assert tool_calls is None

    def test_tool_call_dotted_name(self, harmony_parser, chat_request):
        prompt = [Message.from_role_and_content(Role.USER, "Compute 2+3")]
        response = [tool_call("math.sum", '{"a": 2, "b": 3}')]

        _, _, tool_calls = harmony_parser.parse(
            "",
            chat_request,
            model_output_token_ids=get_model_output_tokens(prompt, response),
        )

        assert tool_call_tuples(tool_calls) == [
            ("math.sum", json.dumps({"a": 2, "b": 3}))
        ]

    def test_tool_calls_with_final_content(self, harmony_parser, chat_request):
        prompt = [Message.from_role_and_content(Role.USER, "What is the weather?")]
        response = [
            assistant("User asked about the weather.", "analysis"),
            tool_call("functions.get_current_weather", '{"location": "Tokyo"}'),
            assistant("This tool call will get the weather.", "final"),
        ]

        reasoning, content, tool_calls = harmony_parser.parse(
            "",
            chat_request,
            model_output_token_ids=get_model_output_tokens(prompt, response),
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
        assert harmony_parser._parser is None

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
        assert harmony_parser._parser is None

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
        assert harmony_parser._parser is None

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
        assert parser._parser is None

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
        prompt = [Message.from_role_and_content(Role.USER, "Hello")]
        response = [tool_call(recipient, "Ignore this", content_type=None)]

        delta = parser.parse_delta(
            delta_text="",
            delta_token_ids=get_model_output_tokens(prompt, response),
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

    def test_invalid_token_sequence_raises_harmony_parser_error(self, harmony_parser):
        """
        Test that HarmonyParserError is raised when the parser encounters
        an unexpected token sequence (e.g., consecutive stop tokens).
        """
        # Simulate the error case from issue #46464:
        # After a message ends with <|return|> (200002), the parser expects
        # <|start|> (200006) to begin the next message, but receives another
        # <|return|> instead.
        invalid_sequence = encode_output(
            "<|channel|>final<|message|>Some content<|return|>"
            # Missing <|start|> here - sending <|return|> again instead
            "<|return|>"
        )

        with pytest.raises(HarmonyParserError) as exc_info:
            harmony_parser.process_chunk(invalid_sequence)

        # Verify the error message contains useful debugging info
        error_msg = str(exc_info.value)
        assert "Harmony parser failed" in error_msg
        assert "token" in error_msg.lower()
