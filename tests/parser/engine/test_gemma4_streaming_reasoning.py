# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the unified Gemma4 parser engine."""

import json
from unittest.mock import MagicMock

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from tests.parser.engine.streaming_helpers import (
    collect_content,
    collect_function_name,
    collect_tool_arguments,
    simulate_tool_streaming,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.parser.gemma4 import Gemma4Parser

# ── Special token IDs (arbitrary but consistent) ─────────────────────
CHANNEL_START_ID = 50  # <|channel>
CHANNEL_END_ID = 51  # <channel|>
TOOL_CALL_START_ID = 48  # <|tool_call>
TOOL_CALL_END_ID = 49  # <tool_call|>
QUOTED_ID = 52  # <|"|>
NEW_TURN_ID = 53  # <|turn>
SPECIAL_TOKEN_MAP = {
    CHANNEL_START_ID: "<|channel>",
    CHANNEL_END_ID: "<channel|>",
    TOOL_CALL_START_ID: "<|tool_call>",
    TOOL_CALL_END_ID: "<tool_call|>",
    QUOTED_ID: '<|"|>',
    NEW_TURN_ID: "<|turn>",
}

SPECIAL_TEXT_TO_ID = {v: k for k, v in SPECIAL_TOKEN_MAP.items()}


def _make_tokenizer(sequence: list[tuple[int, str]]) -> MagicMock:
    decode_map: dict[int, str] = dict(SPECIAL_TOKEN_MAP)
    for tid, text in sequence:
        decode_map[tid] = text

    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = dict(SPECIAL_TEXT_TO_ID)
    tokenizer.encode.return_value = [tid for tid, _ in sequence]

    def decode(ids, skip_special_tokens=False):
        parts = []
        for tid in ids:
            if skip_special_tokens and tid in SPECIAL_TOKEN_MAP:
                continue
            text = decode_map.get(tid, f"?{tid}?")
            parts.append(text)
        return "".join(parts)

    tokenizer.decode.side_effect = decode
    tokenizer.all_special_tokens = list(SPECIAL_TOKEN_MAP.values())
    tokenizer.all_special_ids = list(SPECIAL_TOKEN_MAP.keys())
    return tokenizer


# ── Model output ────────────────────────────────────────────────────

REASONING_TEXT = (
    "The user is asking for the current weather in Dallas, Texas, "
    "and specifically requests the temperature in Fahrenheit. "
    "I have a tool `get_current_weather` that can provide this "
    "information. I should call this tool with `city='Dallas'`, "
    "`state='TX'`, and `unit='fahrenheit'`."
)

# Break reasoning into word-level tokens
_reasoning_words = REASONING_TEXT.split(" ")
_REGULAR_TOKEN_START = 1000
REASONING_TOKENS: list[tuple[int, str]] = []
for i, word in enumerate(_reasoning_words):
    prefix = " " if i > 0 else ""
    REASONING_TOKENS.append((_REGULAR_TOKEN_START + i, prefix + word))

# Tool call body tokens
TOOL_BODY_TOKENS: list[tuple[int, str]] = [
    (2000, "call"),
    (2001, ":"),
    (2002, "get_current_weather"),
    (2003, "{"),
    (2004, "city"),
    (2005, ":"),
    (2006, "Dallas"),
    (2007, ","),
    (2008, "state"),
    (2009, ":"),
    (2010, "TX"),
    (2011, ","),
    (2012, "unit"),
    (2013, ":"),
    (2014, "fahrenheit"),
    (2015, "}"),
]

FULL_TOKEN_SEQUENCE: list[tuple[int, str]] = []
FULL_TOKEN_SEQUENCE.append((CHANNEL_START_ID, "<|channel>"))
FULL_TOKEN_SEQUENCE.append((3000, "thought"))
FULL_TOKEN_SEQUENCE.append((3001, "\n"))
FULL_TOKEN_SEQUENCE.extend(REASONING_TOKENS)
FULL_TOKEN_SEQUENCE.append((CHANNEL_END_ID, "<channel|>"))
FULL_TOKEN_SEQUENCE.append((TOOL_CALL_START_ID, "<|tool_call>"))
FULL_TOKEN_SEQUENCE.extend(TOOL_BODY_TOKENS[:4])
FULL_TOKEN_SEQUENCE.extend(TOOL_BODY_TOKENS[4:6])
FULL_TOKEN_SEQUENCE.append((QUOTED_ID, '<|"|>'))
FULL_TOKEN_SEQUENCE.append(TOOL_BODY_TOKENS[6])
FULL_TOKEN_SEQUENCE.append((QUOTED_ID, '<|"|>'))
FULL_TOKEN_SEQUENCE.extend(TOOL_BODY_TOKENS[7:10])
FULL_TOKEN_SEQUENCE.append((QUOTED_ID, '<|"|>'))
FULL_TOKEN_SEQUENCE.append(TOOL_BODY_TOKENS[10])
FULL_TOKEN_SEQUENCE.append((QUOTED_ID, '<|"|>'))
FULL_TOKEN_SEQUENCE.extend(TOOL_BODY_TOKENS[11:14])
FULL_TOKEN_SEQUENCE.append((QUOTED_ID, '<|"|>'))
FULL_TOKEN_SEQUENCE.append(TOOL_BODY_TOKENS[14])
FULL_TOKEN_SEQUENCE.append((QUOTED_ID, '<|"|>'))
FULL_TOKEN_SEQUENCE.append(TOOL_BODY_TOKENS[15])
FULL_TOKEN_SEQUENCE.append((TOOL_CALL_END_ID, "<tool_call|>"))

# Full model output as a single string
FULL_MODEL_OUTPUT = "".join(text for _, text in FULL_TOKEN_SEQUENCE)

# ── Helpers ──────────────────────────────────────────────────────────


def _stream_tokens_batched(
    parser, tokenizer, request, batch_size=10, prompt_token_ids=None
) -> list[DeltaMessage | None]:
    """Feed tokens in batches through parse_delta."""
    token_ids = tokenizer.encode("", add_special_tokens=False)
    results: list[DeltaMessage | None] = []
    n = len(token_ids)

    for start in range(0, n, batch_size):
        batch_ids = token_ids[start : start + batch_size]
        delta_text = tokenizer.decode(batch_ids)
        result = parser.parse_delta(
            delta_text,
            batch_ids,
            request,
            prompt_token_ids=prompt_token_ids,
            finished=(start + batch_size >= n),
        )
        prompt_token_ids = None
        results.append(result)
    return results


def _collect_fields(results):
    reasoning = "".join(r.reasoning for r in results if r and r.reasoning)
    content = "".join(r.content for r in results if r and r.content)
    tool_calls = [tc for r in results if r and r.tool_calls for tc in r.tool_calls]
    return reasoning, content, tool_calls


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def mock_tokenizer():
    return _make_tokenizer(FULL_TOKEN_SEQUENCE)


@pytest.fixture
def parser(mock_tokenizer):
    return Gemma4Parser(mock_tokenizer)


@pytest.fixture
def request_obj():
    return ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
    )


# ── Tests ────────────────────────────────────────────────────────────


class TestGemma4StreamingReasoningThenToolCall:
    """Streaming: reasoning followed by a tool call."""

    def test_tool_call_extracted(self, parser, mock_tokenizer, request_obj):
        """Tool calls must be extracted from streaming output."""
        results = _stream_tokens_batched(
            parser,
            mock_tokenizer,
            request_obj,
            batch_size=10,
            prompt_token_ids=[],
        )

        reasoning, content, tool_calls = _collect_fields(results)

        assert len(tool_calls) > 0, (
            f"Expected tool_calls but got none. "
            f"content={content!r}, reasoning={reasoning[:80]!r}..."
        )

        names = [
            tc.function.name for tc in tool_calls if tc.function and tc.function.name
        ]
        assert "get_current_weather" in names, (
            f"Expected get_current_weather, got {names}"
        )

        args_text = "".join(
            tc.function.arguments
            for tc in tool_calls
            if tc.function and tc.function.arguments
        )
        if args_text:
            parsed_args = json.loads(args_text)
            assert parsed_args.get("city") == "Dallas"
            assert parsed_args.get("state") == "TX"
            assert parsed_args.get("unit") == "fahrenheit"

    def test_tool_call_text_not_in_content(self, parser, mock_tokenizer, request_obj):
        """Tool call body must not leak into content."""
        results = _stream_tokens_batched(
            parser,
            mock_tokenizer,
            request_obj,
            batch_size=10,
            prompt_token_ids=[],
        )

        _, content, _ = _collect_fields(results)

        assert "call:" not in content, (
            f"Tool call text leaked into content: {content!r}"
        )
        assert "get_current_weather" not in content, (
            f"Function name leaked into content: {content!r}"
        )

    def test_reasoning_extracted(self, parser, mock_tokenizer, request_obj):
        """Reasoning content should be captured."""
        results = _stream_tokens_batched(
            parser,
            mock_tokenizer,
            request_obj,
            batch_size=10,
            prompt_token_ids=[],
        )

        reasoning, _, _ = _collect_fields(results)

        assert "weather" in reasoning.lower(), (
            f"Expected reasoning about weather, got: {reasoning[:100]!r}"
        )


# ── Prompt ends inside an open <|channel>thought\n block ─────────────

_OPEN_REASONING_GEN_SEQUENCE: list[tuple[int, str]] = [
    (7001, "Sure"),
    (7002, ","),
    (7003, " the"),
    (7004, " answer"),
    (7005, " is"),
    (7006, " 42"),
    (CHANNEL_END_ID, "<channel|>"),
    (7007, "Hello"),
    (7008, " world"),
]


class TestGemma4PromptOpenReasoning:
    """When ``add_generation_prompt=True`` after a final tool response with
    ``enable_thinking=True``, the Gemma4 chat template leaves the prompt
    ending with ``<|channel>thought\\n`` — i.e. inside an open reasoning
    channel. Tokens generated before ``<channel|>`` must be classified as
    ``reasoning``, not visible ``content``.

    Regression test for vllm-project/vllm#45834.
    """

    @pytest.fixture
    def open_reasoning_tokenizer(self):
        return _make_tokenizer(_OPEN_REASONING_GEN_SEQUENCE)

    @pytest.fixture
    def open_reasoning_parser(self, open_reasoning_tokenizer):
        return Gemma4Parser(open_reasoning_tokenizer)

    @staticmethod
    def _prompt_ids_open_channel() -> list[int]:
        # Mimics a prompt that ends with ``...<|channel>thought\n``. The
        # specific token ids for ``thought`` and ``\n`` are arbitrary — only
        # the trailing ``<|channel>`` start token matters for detection.
        return [CHANNEL_START_ID, 3000, 3001]

    def test_reasoning_not_leaked_into_content(
        self, open_reasoning_parser, open_reasoning_tokenizer, request_obj
    ):
        results = _stream_tokens_batched(
            open_reasoning_parser,
            open_reasoning_tokenizer,
            request_obj,
            batch_size=1,
            prompt_token_ids=self._prompt_ids_open_channel(),
        )

        reasoning, content, _ = _collect_fields(results)

        assert "Sure, the answer is 42" in reasoning, (
            f"Expected pre-<channel|> tokens in reasoning, got "
            f"reasoning={reasoning!r} content={content!r}"
        )
        for leaked in ("Sure", "answer", "42"):
            assert leaked not in content, (
                f"Reasoning text leaked into content: {content!r}"
            )

    def test_post_reasoning_text_in_content(
        self, open_reasoning_parser, open_reasoning_tokenizer, request_obj
    ):
        results = _stream_tokens_batched(
            open_reasoning_parser,
            open_reasoning_tokenizer,
            request_obj,
            batch_size=1,
            prompt_token_ids=self._prompt_ids_open_channel(),
        )

        _, content, _ = _collect_fields(results)

        assert "Hello world" in content, (
            f"Post-<channel|> text missing from content: {content!r}"
        )

    def test_new_turn_prompt_unchanged(self, parser, mock_tokenizer, request_obj):
        """When the prompt does NOT end in an open reasoning channel (e.g. a
        new turn that ends with ``<|turn>model\\n``), behaviour must match
        the existing flow — the model itself opens ``<|channel>``.
        """
        results = _stream_tokens_batched(
            parser,
            mock_tokenizer,
            request_obj,
            batch_size=10,
            # No <|channel> in the prompt tail.
            prompt_token_ids=[9000, 9001],
        )

        reasoning, content, tool_calls = _collect_fields(results)

        assert "weather" in reasoning.lower(), (
            f"Expected reasoning about weather, got: {reasoning[:100]!r}"
        )
        assert len(tool_calls) > 0, f"Tool calls missing — content={content!r}"


# ── Engine pre-initialised to REASONING + model still emits channel open ──

_PRE_INIT_THOUGHT_GEN_SEQUENCE: list[tuple[int, str]] = [
    # Model naively emits the full reasoning opener even though the engine
    # was pre-initialised to REASONING from the prompt.
    (CHANNEL_START_ID, "<|channel>"),
    (8000, "thought"),
    (8001, "\n"),
    (8002, "Reason"),
    (8003, "ing"),
    (8004, " body"),
    (CHANNEL_END_ID, "<channel|>"),
    (8005, "Final"),
    (8006, " content"),
]


class TestGemma4PreInitReasoningRobustness:
    """Tests for the ``(REASONING, THINK_START)`` no-op transition and
    cooperating ``thought\\n`` prefix stripping when the engine has been
    pre-initialised to ``REASONING`` from the prompt.

    These cover the case the reviewer raised: prompt ends with
    ``<|turn>model\\n`` (``is_reasoning_end`` returns ``False`` because
    thinking is enabled, so the engine is pre-initialised), but the model
    still emits its own ``<|channel>thought\\n…<channel|>content``. The
    ``thought\\n`` prefix must be stripped, the ``<|channel>`` must not
    leak as text, and the post-``<channel|>`` text must appear as content.
    """

    @pytest.fixture
    def pre_init_tokenizer(self):
        return _make_tokenizer(_PRE_INIT_THOUGHT_GEN_SEQUENCE)

    @pytest.fixture
    def pre_init_parser(self, pre_init_tokenizer):
        return Gemma4Parser(pre_init_tokenizer)

    def test_redundant_channel_open_swallowed_after_new_turn(
        self, pre_init_parser, pre_init_tokenizer, request_obj
    ):
        # Prompt ends with ``<|turn>model\n``-style sentinel. With
        # ``enable_thinking=True`` (the default), ``is_reasoning_end``
        # returns ``False`` for a ``<|turn>`` tail, so the engine is
        # pre-initialised to ``REASONING``.
        results = _stream_tokens_batched(
            pre_init_parser,
            pre_init_tokenizer,
            request_obj,
            batch_size=1,
            prompt_token_ids=[NEW_TURN_ID, 9100, 9101],
        )

        reasoning, content, _ = _collect_fields(results)

        # ``thought\n`` prefix must be stripped from reasoning even though
        # the engine was pre-initialised to REASONING.
        assert reasoning.startswith("Reason"), (
            f"thought\\n prefix leaked into reasoning: {reasoning!r}"
        )
        assert "thought\n" not in reasoning, (
            f"thought\\n prefix leaked into reasoning: {reasoning!r}"
        )
        assert "Reasoning body" in reasoning, f"Reasoning body missing: {reasoning!r}"

        # The redundant ``<|channel>`` opener must not appear as text.
        assert "<|channel>" not in content, (
            f"<|channel> leaked into content: {content!r}"
        )
        assert "<|channel>" not in reasoning, (
            f"<|channel> leaked into reasoning: {reasoning!r}"
        )

        # Post-``<channel|>`` text must appear as content.
        assert "Final content" in content, (
            f"Post-<channel|> text missing from content: {content!r}"
        )

    def test_redundant_channel_open_swallowed_after_open_channel_prompt(
        self, pre_init_parser, pre_init_tokenizer, request_obj
    ):
        # Prompt already ends inside an open ``<|channel>`` block. Engine
        # is pre-initialised to ``REASONING`` via the start-token check.
        # Even if the model redundantly re-emits ``<|channel>thought\n``,
        # the no-op transition + prefix stripping must keep output clean.
        results = _stream_tokens_batched(
            pre_init_parser,
            pre_init_tokenizer,
            request_obj,
            batch_size=1,
            prompt_token_ids=[CHANNEL_START_ID, 3000, 3001],
        )

        reasoning, content, _ = _collect_fields(results)

        assert "<|channel>" not in content, (
            f"<|channel> leaked into content: {content!r}"
        )
        assert "thought\n" not in reasoning, (
            f"thought\\n prefix leaked into reasoning: {reasoning!r}"
        )
        assert "Reasoning body" in reasoning
        assert "Final content" in content


# ── Second model output: two tool calls with holdback ────────────────

REASONING_TEXT_2 = (
    "The user wants me to:\n"
    "1. Perform some reasoning.\n"
    "2. Call a tool to fetch the hostname.\n"
    "3. Call a tool to fetch the current date.\n"
    "\n"
    "Since I am an AI assistant (opencode), I can use the "
    "`bash` tool to execute commands.\n"
    "To get the hostname, I can run `hostname`.\n"
    "To get the current date, I can run `date`.\n"
    "\n"
    "I should do this in a single response with "
    "multiple tool calls for efficiency."
)

_reasoning_words_2 = REASONING_TEXT_2.split(" ")
_R2_TOKEN_START = 4000
REASONING_TOKENS_2: list[tuple[int, str]] = []
for i, word in enumerate(_reasoning_words_2):
    prefix = " " if i > 0 else ""
    REASONING_TOKENS_2.append((_R2_TOKEN_START + i, prefix + word))

TOOL_BODY_TOKENS_2A: list[tuple[int, str]] = [
    (5000, "call"),
    (5001, ":"),
    (5002, "bash"),
    (5003, "{"),
    (5004, "command"),
    (5005, ":"),
    (5006, "hostname"),
    (5007, ","),
    (5008, "description"),
    (5009, ":"),
    (5010, "Fetch the hostname of the system."),
    (5011, "}"),
]

TOOL_BODY_TOKENS_2B: list[tuple[int, str]] = [
    (6000, "call"),
    (6001, ":"),
    (6002, "bash"),
    (6003, "{"),
    (6004, "command"),
    (6005, ":"),
    (6006, "date"),
    (6007, ","),
    (6008, "description"),
    (6009, ":"),
    (6010, "Fetch the current system date and time."),
    (6011, "}"),
]

FULL_TOKEN_SEQUENCE_2: list[tuple[int, str]] = []
FULL_TOKEN_SEQUENCE_2.append((CHANNEL_START_ID, "<|channel>"))
FULL_TOKEN_SEQUENCE_2.append((3000, "thought"))
FULL_TOKEN_SEQUENCE_2.append((3001, "\n"))
FULL_TOKEN_SEQUENCE_2.extend(REASONING_TOKENS_2)
FULL_TOKEN_SEQUENCE_2.append((CHANNEL_END_ID, "<channel|>"))
FULL_TOKEN_SEQUENCE_2.append((TOOL_CALL_START_ID, "<|tool_call>"))
FULL_TOKEN_SEQUENCE_2.extend(TOOL_BODY_TOKENS_2A[:6])
FULL_TOKEN_SEQUENCE_2.append((QUOTED_ID, '<|"|>'))
FULL_TOKEN_SEQUENCE_2.append(TOOL_BODY_TOKENS_2A[6])
FULL_TOKEN_SEQUENCE_2.append((QUOTED_ID, '<|"|>'))
FULL_TOKEN_SEQUENCE_2.extend(TOOL_BODY_TOKENS_2A[7:10])
FULL_TOKEN_SEQUENCE_2.append((QUOTED_ID, '<|"|>'))
FULL_TOKEN_SEQUENCE_2.append(TOOL_BODY_TOKENS_2A[10])
FULL_TOKEN_SEQUENCE_2.append((QUOTED_ID, '<|"|>'))
FULL_TOKEN_SEQUENCE_2.append(TOOL_BODY_TOKENS_2A[11])
FULL_TOKEN_SEQUENCE_2.append((TOOL_CALL_END_ID, "<tool_call|>"))
FULL_TOKEN_SEQUENCE_2.append((TOOL_CALL_START_ID, "<|tool_call>"))
FULL_TOKEN_SEQUENCE_2.extend(TOOL_BODY_TOKENS_2B[:6])
FULL_TOKEN_SEQUENCE_2.append((QUOTED_ID, '<|"|>'))
FULL_TOKEN_SEQUENCE_2.append(TOOL_BODY_TOKENS_2B[6])
FULL_TOKEN_SEQUENCE_2.append((QUOTED_ID, '<|"|>'))
FULL_TOKEN_SEQUENCE_2.extend(TOOL_BODY_TOKENS_2B[7:10])
FULL_TOKEN_SEQUENCE_2.append((QUOTED_ID, '<|"|>'))
FULL_TOKEN_SEQUENCE_2.append(TOOL_BODY_TOKENS_2B[10])
FULL_TOKEN_SEQUENCE_2.append((QUOTED_ID, '<|"|>'))
FULL_TOKEN_SEQUENCE_2.append(TOOL_BODY_TOKENS_2B[11])
FULL_TOKEN_SEQUENCE_2.append((TOOL_CALL_END_ID, "<tool_call|>"))


def _stream_tokens_with_holdback(
    parser,
    tokenizer,
    request,
    batch_size=10,
    holdback_chars=12,
    prompt_token_ids=None,
) -> list[DeltaMessage | None]:
    """Feed tokens in batches with simulated detokenizer holdback."""
    token_ids = tokenizer.encode("", add_special_tokens=False)
    results: list[DeltaMessage | None] = []
    prev_safe_text = ""

    for start in range(0, len(token_ids), batch_size):
        batch_end = min(start + batch_size, len(token_ids))
        batch_ids = token_ids[start:batch_end]

        full_decoded = tokenizer.decode(token_ids[:batch_end])

        if batch_end < len(token_ids):
            safe_len = max(0, len(full_decoded) - holdback_chars)
            safe_text = full_decoded[:safe_len]
        else:
            safe_text = full_decoded

        delta_text = safe_text[len(prev_safe_text) :]
        prev_safe_text = safe_text

        result = parser.parse_delta(
            delta_text,
            batch_ids,
            request,
            prompt_token_ids=prompt_token_ids,
            finished=False,
        )
        prompt_token_ids = None
        results.append(result)
    return results


class TestGemma4ReasoningTruncationWithHoldback:
    """Reasoning text must not be truncated when detokenizer holds back text."""

    @pytest.fixture
    def tokenizer_2(self):
        return _make_tokenizer(FULL_TOKEN_SEQUENCE_2)

    @pytest.fixture
    def parser_2(self, tokenizer_2):
        return Gemma4Parser(tokenizer_2)

    def test_reasoning_not_truncated(self, parser_2, tokenizer_2, request_obj):
        """Reasoning must include the full text up to <channel|>."""
        results = _stream_tokens_with_holdback(
            parser_2,
            tokenizer_2,
            request_obj,
            batch_size=10,
            holdback_chars=12,
            prompt_token_ids=[],
        )

        reasoning, content, tool_calls = _collect_fields(results)

        assert "efficiency" in reasoning, (
            f"Reasoning truncated — missing 'efficiency'. "
            f"Reasoning ends with: {reasoning[-60:]!r}"
        )

    def test_both_tool_calls_extracted(self, parser_2, tokenizer_2, request_obj):
        """Both bash tool calls must be extracted."""
        results = _stream_tokens_with_holdback(
            parser_2,
            tokenizer_2,
            request_obj,
            batch_size=10,
            holdback_chars=12,
            prompt_token_ids=[],
        )

        _, _, tool_calls = _collect_fields(results)

        names = [
            tc.function.name for tc in tool_calls if tc.function and tc.function.name
        ]
        assert len(names) >= 2, f"Expected 2 tool calls, got {len(names)}: {names}"
        assert names.count("bash") >= 2, f"Expected 2 bash tool calls, got {names}"

    def test_tool_call_text_not_in_content(self, parser_2, tokenizer_2, request_obj):
        """Tool call body must not leak into content."""
        results = _stream_tokens_with_holdback(
            parser_2,
            tokenizer_2,
            request_obj,
            batch_size=10,
            holdback_chars=12,
            prompt_token_ids=[],
        )

        _, content, _ = _collect_fields(results)

        assert "call:" not in content, (
            f"Tool call text leaked into content: {content!r}"
        )


# ── Simple mock tokenizer for tool-only tests ────────────────────────


@pytest.fixture
def tool_call_tokenizer():
    """Mock tokenizer with Gemma4 special token vocab."""
    return make_mock_tokenizer(
        vocab={
            "<|tool_call>": TOOL_CALL_START_ID,
            "<tool_call|>": TOOL_CALL_END_ID,
            "<|channel>": CHANNEL_START_ID,
            "<channel|>": CHANNEL_END_ID,
            '<|"|>': QUOTED_ID,
        },
    )


@pytest.fixture
def tool_call_parser(tool_call_tokenizer):
    return Gemma4Parser(tool_call_tokenizer)


# ── Non-streaming tool call extraction tests ─────────────────────────


class TestNonStreamingToolCalls:
    """Non-streaming tool call extraction via extract_tool_calls()."""

    def test_no_tool_calls(self, tool_call_parser, mock_request):
        result = tool_call_parser.extract_tool_calls(
            "Hello, how can I help you today?",
            mock_request,
        )
        assert result.tools_called is False
        assert result.tool_calls == []
        assert result.content == "Hello, how can I help you today?"

    def test_single_tool_call(self, tool_call_parser, mock_request):
        text = '<|tool_call>call:get_weather{location:<|"|>London<|"|>}<tool_call|>'
        result = tool_call_parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"location": "London"}

    def test_multiple_arguments(self, tool_call_parser, mock_request):
        text = (
            "<|tool_call>call:get_weather{"
            'location:<|"|>San Francisco<|"|>,'
            'unit:<|"|>celsius<|"|>}'
            "<tool_call|>"
        )
        result = tool_call_parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert result.tool_calls[0].function.name == "get_weather"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"location": "San Francisco", "unit": "celsius"}

    def test_text_before_tool_call(self, tool_call_parser, mock_request):
        text = (
            "Let me check the weather for you. "
            '<|tool_call>call:get_weather{location:<|"|>Paris<|"|>}'
            "<tool_call|>"
        )
        result = tool_call_parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert result.content is not None
        assert "Let me check the weather" in result.content
        assert result.tool_calls[0].function.name == "get_weather"

    def test_multiple_tool_calls(self, tool_call_parser, mock_request):
        text = (
            '<|tool_call>call:get_weather{location:<|"|>London<|"|>}'
            "<tool_call|>"
            '<|tool_call>call:get_time{location:<|"|>London<|"|>}'
            "<tool_call|>"
        )
        result = tool_call_parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].function.name == "get_weather"
        assert result.tool_calls[1].function.name == "get_time"

    def test_nested_arguments(self, tool_call_parser, mock_request):
        text = (
            "<|tool_call>call:complex_function{"
            'nested:{inner:<|"|>value<|"|>},'
            'list:[<|"|>a<|"|>,<|"|>b<|"|>]}'
            "<tool_call|>"
        )
        result = tool_call_parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert result.tool_calls[0].function.name == "complex_function"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"nested": {"inner": "value"}, "list": ["a", "b"]}

    def test_number_and_boolean(self, tool_call_parser, mock_request):
        text = (
            "<|tool_call>call:set_status{"
            "is_active:true,"
            "count:42,"
            "score:3.14}"
            "<tool_call|>"
        )
        result = tool_call_parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"is_active": "true", "count": "42", "score": "3.14"}

    def test_no_arguments(self, tool_call_parser, mock_request):
        text = "<|tool_call>call:get_status{}<tool_call|>"
        result = tool_call_parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert result.tool_calls[0].function.name == "get_status"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {}

    def test_hyphenated_function_name(self, tool_call_parser, mock_request):
        text = '<|tool_call>call:get-weather{location:<|"|>London<|"|>}<tool_call|>'
        result = tool_call_parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert result.tool_calls[0].function.name == "get-weather"

    def test_dotted_function_name(self, tool_call_parser, mock_request):
        text = '<|tool_call>call:weather.get{location:<|"|>London<|"|>}<tool_call|>'
        result = tool_call_parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert result.tool_calls[0].function.name == "weather.get"


# ── Streaming tool call edge-case tests ──────────────────────────────


class TestStreamingToolCallEdgeCases:
    """Streaming tool call extraction via extract_tool_calls_streaming()."""

    def test_basic_streaming(self, tool_call_parser, mock_request):
        chunks = [
            "<|tool_call>",
            "call:get_weather{",
            'location:<|"|>Paris',
            ", France",
            '<|"|>}',
            "<tool_call|>",
        ]

        results = simulate_tool_streaming(tool_call_parser, mock_request, chunks)

        name = collect_function_name(results)
        assert name == "get_weather"

        args_text = collect_tool_arguments(results)
        assert args_text
        parsed = json.loads(args_text)
        assert parsed == {"location": "Paris, France"}

    def test_streaming_multi_arg(self, tool_call_parser, mock_request):
        chunks = [
            "<|tool_call>",
            "call:get_weather{",
            'location:<|"|>Tokyo<|"|>,',
            'unit:<|"|>celsius<|"|>}',
            "<tool_call|>",
        ]

        results = simulate_tool_streaming(tool_call_parser, mock_request, chunks)

        name = collect_function_name(results)
        assert name == "get_weather"

        args_text = collect_tool_arguments(results)
        assert args_text
        parsed = json.loads(args_text)
        assert parsed == {"location": "Tokyo", "unit": "celsius"}

    def test_streaming_no_extra_brace(self, tool_call_parser, mock_request):
        chunks = [
            "<|tool_call>",
            "call:get_weather{",
            'location:<|"|>London<|"|>}',
            "<tool_call|>",
        ]

        results = simulate_tool_streaming(tool_call_parser, mock_request, chunks)
        args_text = collect_tool_arguments(results)
        assert args_text

        parsed = json.loads(args_text)
        assert parsed == {"location": "London"}
        assert args_text.count("}") <= 1

    def test_streaming_text_before_tool(self, tool_call_parser, mock_request):
        chunks = [
            "Let me check ",
            "the weather. ",
            "<|tool_call>",
            "call:get_weather{",
            'location:<|"|>London<|"|>}',
            "<tool_call|>",
        ]

        results = simulate_tool_streaming(tool_call_parser, mock_request, chunks)
        assert collect_content(results).strip().startswith("Let me check")

    def test_streaming_numeric_args(self, tool_call_parser, mock_request):
        chunks = [
            "<|tool_call>",
            "call:set_config{",
            "count:42,",
            "active:true}",
            "<tool_call|>",
        ]

        results = simulate_tool_streaming(tool_call_parser, mock_request, chunks)
        args_text = collect_tool_arguments(results)
        if args_text:
            parsed = json.loads(args_text)
            assert parsed["count"] == "42"
            assert parsed["active"] == "true"

    def test_streaming_empty_args(self, tool_call_parser, mock_request):
        chunks = [
            "<|tool_call>",
            "call:get_status{}",
            "<tool_call|>",
        ]

        results = simulate_tool_streaming(tool_call_parser, mock_request, chunks)
        name = collect_function_name(results)
        assert name == "get_status"

    def test_streaming_split_delimiter(self, tool_call_parser, mock_request):
        """Partial <|"|> delimiter must not leak into JSON."""
        chunks = [
            "<|tool_call>",
            "call:todowrite{",
            'content:<|"|>Buy milk<|',
            '"|>}',
            "<tool_call|>",
        ]

        results = simulate_tool_streaming(tool_call_parser, mock_request, chunks)
        args_text = collect_tool_arguments(results)
        assert args_text
        parsed = json.loads(args_text)
        assert parsed["content"] == "Buy milk"
        assert "<|" not in args_text

    def test_streaming_bool_split(self, tool_call_parser, mock_request):
        chunks = [
            "<|tool_call>",
            "call:search{input:{all:t",
            "rue}}",
            "<tool_call|>",
        ]

        results = simulate_tool_streaming(tool_call_parser, mock_request, chunks)
        args_text = collect_tool_arguments(results)
        assert args_text
        parsed = json.loads(args_text)
        assert parsed["input"]["all"] == "true"

    def test_streaming_number_split(self, tool_call_parser, mock_request):
        chunks = [
            "<|tool_call>",
            "call:set{count:4",
            "2}",
            "<tool_call|>",
        ]

        results = simulate_tool_streaming(tool_call_parser, mock_request, chunks)
        args_text = collect_tool_arguments(results)
        assert args_text
        parsed = json.loads(args_text)
        assert parsed["count"] == "42"

    def test_streaming_trailing_bare_bool(self, tool_call_parser, mock_request):
        chunks = [
            "<|tool_call>",
            "call:Edit{",
            'file_path:<|"|>src/env.py<|"|>,',
            'old_string:<|"|>old_val<|"|>,',
            'new_string:<|"|>new_val<|"|>,',
            "replace_all:",
            "false}",
            "<tool_call|>",
        ]

        results = simulate_tool_streaming(tool_call_parser, mock_request, chunks)
        args_text = collect_tool_arguments(results)
        assert args_text

        parsed = json.loads(args_text)
        assert parsed == {
            "file_path": "src/env.py",
            "old_string": "old_val",
            "new_string": "new_val",
            "replace_all": "false",
        }

        assert args_text.count("replace_all") == 1


# ── Non-streaming reasoning + tool call extraction tests ──────────


class TestNonStreamingReasoningPlusToolCalls:
    """Non-streaming extraction with reasoning + tool calls."""

    def test_extract_tool_calls_from_full_text(self, parser, request_obj):
        """extract_tool_calls on full model output must find tools."""
        model_output = FULL_MODEL_OUTPUT
        result = parser.extract_tool_calls(model_output, request_obj)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_current_weather"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["city"] == "Dallas"
        assert args["state"] == "TX"
        assert args["unit"] == "fahrenheit"

    def test_extract_reasoning_from_full_text(self, parser, request_obj):
        """extract_reasoning on full model output must find reasoning."""
        model_output = FULL_MODEL_OUTPUT
        reasoning, content = parser.extract_reasoning(model_output, request_obj)

        assert reasoning is not None
        assert "weather" in reasoning.lower()
        assert not reasoning.startswith("thought")

    def test_bug_report_scenario(self, tool_call_parser, mock_request):
        """Exact scenario from the bug report: get_weather for Raleigh."""
        model_output = (
            "<|channel>thought\n"
            'The user wants to get the weather for "Raleigh". '
            "I should use the `get_weather` tool and pass "
            '"Raleigh" as the `city` argument.'
            "<channel|>"
            '<|tool_call>call:get_weather{city:<|"|>Raleigh<|"|>}'
            "<tool_call|>"
        )
        result = tool_call_parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True, (
            f"No tool calls found. content={result.content!r}"
        )
        assert result.tool_calls[0].function.name == "get_weather"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["city"] == "Raleigh"

    def test_both_extractions_independent(self, parser, request_obj):
        """Calling extract_reasoning then extract_tool_calls on the same
        parser instance should both work (each resets the engine)."""
        model_output = FULL_MODEL_OUTPUT

        reasoning, _ = parser.extract_reasoning(model_output, request_obj)
        result = parser.extract_tool_calls(model_output, request_obj)

        assert reasoning is not None
        assert "weather" in reasoning.lower()
        assert result.tools_called is True
        assert result.tool_calls[0].function.name == "get_current_weather"


class TestAdapterExtractReasoning:
    """The reasoning adapter's extract_reasoning uses skip_tool_parsing
    so tool call text is preserved as content for the tool adapter."""

    @pytest.fixture
    def adapter(self, mock_tokenizer):
        from vllm.parser.engine.adapters import make_adapters

        reasoning_cls, _ = make_adapters(Gemma4Parser)
        return reasoning_cls(mock_tokenizer)

    def test_preserves_tool_text_in_content(self, adapter, request_obj):
        """Tool call markers must appear in content after extraction."""
        reasoning, content = adapter.extract_reasoning(FULL_MODEL_OUTPUT, request_obj)

        assert reasoning is not None
        assert "weather" in reasoning.lower()
        assert content is not None
        assert "<|tool_call>" in content
        assert "<tool_call|>" in content
        assert "get_current_weather" in content

    def test_skip_tool_parsing_restored_after_extraction(self, adapter, request_obj):
        """skip_tool_parsing must be restored to its prior value."""
        engine = adapter._parser_engine._engine
        assert engine.skip_tool_parsing is False
        adapter.extract_reasoning(FULL_MODEL_OUTPUT, request_obj)
        assert engine.skip_tool_parsing is False

    def test_no_reasoning_returns_none(self, adapter, request_obj):
        """Content-only text returns (None, content)."""
        text = "Hello world, no thinking here."
        reasoning, content = adapter.extract_reasoning(text, request_obj)
        assert reasoning is None
        assert content == text


# ── Schema-aware type coercion during streaming ────────────────────


class TestGemma4SchemaAwareTypeCoercion:
    """Verify that streaming and non-streaming produce identical
    type-fixed arguments when tool schemas declare string parameters
    but the model outputs bare numbers/booleans."""

    @pytest.fixture
    def tools(self):
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionToolsParam,
        )

        return [
            ChatCompletionToolsParam(
                type="function",
                function={
                    "name": "update_record",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "zipcode": {"type": "string"},
                            "count": {"type": "integer"},
                        },
                    },
                },
            )
        ]

    @pytest.fixture
    def parser_with_tools(self, tool_call_tokenizer, tools):
        return Gemma4Parser(tool_call_tokenizer, tools=tools)

    def test_streaming_string_param_not_coerced(self, parser_with_tools, mock_request):
        """A numeric value for a string-typed param must remain a string
        in the streamed output, matching the non-streaming result."""
        chunks = [
            "<|tool_call>",
            "call:update_record{",
            "zipcode:12345}",
            "<tool_call|>",
        ]

        results = simulate_tool_streaming(parser_with_tools, mock_request, chunks)
        args_text = collect_tool_arguments(results)
        parsed = json.loads(args_text)
        assert parsed["zipcode"] == "12345"

    def test_streaming_mixed_types(self, parser_with_tools, mock_request):
        """String params get type-fixed, integer params stay integers."""
        chunks = [
            "<|tool_call>",
            "call:update_record{",
            "zipcode:90210,",
            "count:42}",
            "<tool_call|>",
        ]

        results = simulate_tool_streaming(parser_with_tools, mock_request, chunks)
        args_text = collect_tool_arguments(results)
        parsed = json.loads(args_text)
        assert parsed["zipcode"] == "90210"
        assert parsed["count"] == 42

    def test_streaming_matches_non_streaming(self, parser_with_tools, mock_request):
        """Concatenated streaming deltas must produce the same arguments
        as non-streaming extraction."""
        text = "<|tool_call>call:update_record{zipcode:12345}<tool_call|>"

        non_streaming = parser_with_tools.extract_tool_calls(text, mock_request)
        ns_args = json.loads(non_streaming.tool_calls[0].function.arguments)

        chunks = [
            "<|tool_call>",
            "call:update_record{",
            "zipcode:1234",
            "5}",
            "<tool_call|>",
        ]
        results = simulate_tool_streaming(parser_with_tools, mock_request, chunks)
        s_args = json.loads(collect_tool_arguments(results))

        assert s_args == ns_args


class TestGemma4SchemaCoercionBoolNumberNull:
    """Verify that _fix_arg_types coerces string values to non-string
    schema types for the Gemma4 parser."""

    @pytest.fixture
    def tools(self):
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionToolsParam,
        )

        return [
            ChatCompletionToolsParam(
                type="function",
                function={
                    "name": "configure",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "enabled": {"type": "boolean"},
                            "ratio": {"type": "number"},
                            "label": {"type": "string"},
                            "value": {"type": ["string", "null"]},
                        },
                    },
                },
            )
        ]

    @pytest.fixture
    def parser_with_tools(self, tool_call_tokenizer, tools):
        return Gemma4Parser(tool_call_tokenizer, tools=tools)

    def test_bool_param_coerced(self, parser_with_tools, mock_request):
        text = "<|tool_call>call:configure{enabled:true}<tool_call|>"
        result = parser_with_tools.extract_tool_calls(text, mock_request)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["enabled"] is True
        assert isinstance(args["enabled"], bool)

    def test_number_whole_normalized(self, parser_with_tools, mock_request):
        text = "<|tool_call>call:configure{ratio:5.0}<tool_call|>"
        result = parser_with_tools.extract_tool_calls(text, mock_request)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["ratio"] == 5
        assert isinstance(args["ratio"], int)

    def test_null_coerced_when_nullable(self, parser_with_tools, mock_request):
        text = "<|tool_call>call:configure{value:null}<tool_call|>"
        result = parser_with_tools.extract_tool_calls(text, mock_request)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["value"] is None

    def test_null_stays_string_without_null_schema(
        self, parser_with_tools, mock_request
    ):
        text = "<|tool_call>call:configure{label:null}<tool_call|>"
        result = parser_with_tools.extract_tool_calls(text, mock_request)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["label"] == "null"
        assert isinstance(args["label"], str)

    def test_streaming_type_stability(self, parser_with_tools, mock_request):
        """Values streamed incrementally must not cause prefix
        incompatibility when types are coerced."""
        text = (
            "<|tool_call>call:configure{"
            "enabled:true,"
            "ratio:3.14,"
            "label:hello}"
            "<tool_call|>"
        )
        non_stream = parser_with_tools.extract_tool_calls(text, mock_request)
        ns_args = json.loads(non_stream.tool_calls[0].function.arguments)

        chunks = [
            "<|tool_call>",
            "call:configure{",
            "enabled:true,",
            "ratio:3.14,",
            "label:hello}",
            "<tool_call|>",
        ]
        results = simulate_tool_streaming(parser_with_tools, mock_request, chunks)
        s_args = json.loads(collect_tool_arguments(results))

        assert s_args == ns_args
        assert ns_args == {
            "enabled": True,
            "ratio": pytest.approx(3.14),
            "label": "hello",
        }


class TestGemma4NestedSchemaCoercion:
    """Verify that _fix_arg_types recurses into nested Gemma4 objects."""

    @pytest.fixture
    def tools(self):
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionToolsParam,
        )

        return [
            ChatCompletionToolsParam(
                type="function",
                function={
                    "name": "search",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "filters": {
                                "type": "object",
                                "properties": {
                                    "language": {"type": "string"},
                                    "min_stars": {"type": "integer"},
                                },
                            },
                        },
                    },
                },
            )
        ]

    @pytest.fixture
    def parser_with_tools(self, tool_call_tokenizer, tools):
        return Gemma4Parser(tool_call_tokenizer, tools=tools)

    def test_nested_object_coerced(self, parser_with_tools, mock_request):
        text = (
            "<|tool_call>call:search{"
            'query:<|"|>vllm<|"|>,'
            "filters:{language:python,min_stars:100}}"
            "<tool_call|>"
        )
        result = parser_with_tools.extract_tool_calls(text, mock_request)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["query"] == "vllm"
        assert args["filters"]["language"] == "python"
        assert args["filters"]["min_stars"] == 100
        assert isinstance(args["filters"]["min_stars"], int)


# ── Tests for bare "thought" without channel opener ──────────────────

BARE_THOUGHT_SEQUENCE: list[tuple[int, str]] = []
BARE_THOUGHT_SEQUENCE.append((3000, "thought"))
BARE_THOUGHT_SEQUENCE.append((3001, "\n"))
BARE_THOUGHT_SEQUENCE.extend(REASONING_TOKENS)
BARE_THOUGHT_SEQUENCE.append((CHANNEL_END_ID, "<channel|>"))
BARE_THOUGHT_SEQUENCE.append((TOOL_CALL_START_ID, "<|tool_call>"))
BARE_THOUGHT_SEQUENCE.extend(TOOL_BODY_TOKENS[:4])
BARE_THOUGHT_SEQUENCE.extend(TOOL_BODY_TOKENS[4:6])
BARE_THOUGHT_SEQUENCE.append((QUOTED_ID, '<|"|>'))
BARE_THOUGHT_SEQUENCE.append(TOOL_BODY_TOKENS[6])  # Dallas
BARE_THOUGHT_SEQUENCE.append((QUOTED_ID, '<|"|>'))
BARE_THOUGHT_SEQUENCE.append(TOOL_BODY_TOKENS[15])  # }
BARE_THOUGHT_SEQUENCE.append((TOOL_CALL_END_ID, "<tool_call|>"))


class TestBareThoughtWithoutChannelOpener:
    """When the model omits <|channel> and starts with bare ``thought``,
    the parser should auto-inject the channel opener so reasoning is
    captured correctly."""

    @pytest.fixture
    def bare_thought_tokenizer(self):
        return _make_tokenizer(BARE_THOUGHT_SEQUENCE)

    @pytest.fixture
    def bare_thought_parser(self, bare_thought_tokenizer):
        return Gemma4Parser(bare_thought_tokenizer)

    def test_bare_thought_reasoning_then_tool_call(
        self, bare_thought_parser, bare_thought_tokenizer, request_obj
    ):
        results = _stream_tokens_batched(
            bare_thought_parser,
            bare_thought_tokenizer,
            request_obj,
            batch_size=1,
            prompt_token_ids=[],
        )
        reasoning, content, tool_calls = _collect_fields(results)

        assert reasoning == REASONING_TEXT
        assert content == ""
        assert len(tool_calls) > 0
        names = [
            tc.function.name for tc in tool_calls if tc.function and tc.function.name
        ]
        assert "get_current_weather" in names

    def test_bare_thought_larger_batches(
        self, bare_thought_parser, bare_thought_tokenizer, request_obj
    ):
        results = _stream_tokens_batched(
            bare_thought_parser,
            bare_thought_tokenizer,
            request_obj,
            batch_size=10,
            prompt_token_ids=[],
        )
        reasoning, content, tool_calls = _collect_fields(results)

        assert reasoning == REASONING_TEXT
        assert content == ""
        assert len(tool_calls) > 0

    def test_normal_content_not_classified_as_reasoning(self, request_obj):
        content_seq: list[tuple[int, str]] = [
            (6000, "The"),
            (6001, " answer"),
            (6002, " is"),
            (6003, " 42."),
        ]
        tokenizer = _make_tokenizer(content_seq)
        parser = Gemma4Parser(tokenizer)

        results = _stream_tokens_batched(
            parser,
            tokenizer,
            request_obj,
            batch_size=2,
            prompt_token_ids=[],
        )
        reasoning, content, tool_calls = _collect_fields(results)

        assert reasoning == ""
        assert content == "The answer is 42."
        assert len(tool_calls) == 0

    def test_bare_thought_token_at_end_of_stream(self, request_obj):
        """When the stream ends with just "thought" (no \\n), the parser
        should treat it as the thought prefix token, not real reasoning."""
        seq: list[tuple[int, str]] = [
            (CHANNEL_START_ID, "<|channel>"),
            (3000, "thought"),
        ]
        tokenizer = _make_tokenizer(seq)
        parser = Gemma4Parser(tokenizer)

        results = _stream_tokens_batched(
            parser,
            tokenizer,
            request_obj,
            batch_size=1,
            prompt_token_ids=[],
        )
        reasoning, content, tool_calls = _collect_fields(results)

        assert reasoning == ""
        assert content == ""
        assert len(tool_calls) == 0


# ── Regression: commas inside <|"|>-delimited string values ─────────
#
# _make_tokenizer sets all_special_tokens, which activates the auto-drop
# mechanism in _build_drop_info. If <|"|> is not in configured_texts,
# it gets silently dropped and commas inside string values become field
# separators, e.g. "San Francisco, CA" → {"location": "San Francisco"}.


COMMA_TOKEN_SEQUENCE: list[tuple[int, str]] = [
    (TOOL_CALL_START_ID, "<|tool_call>"),
    (4000, "call"),
    (4001, ":"),
    (4002, "get_weather"),
    (4003, "{"),
    (4004, "location"),
    (4005, ":"),
    (QUOTED_ID, '<|"|>'),
    (4006, "San Francisco"),
    (4007, ", CA"),
    (QUOTED_ID, '<|"|>'),
    (4008, ","),
    (4009, "unit"),
    (4010, ":"),
    (QUOTED_ID, '<|"|>'),
    (4011, "celsius"),
    (QUOTED_ID, '<|"|>'),
    (4012, "}"),
    (TOOL_CALL_END_ID, "<tool_call|>"),
]

MULTI_COMMA_TOKEN_SEQUENCE: list[tuple[int, str]] = [
    (TOOL_CALL_START_ID, "<|tool_call>"),
    (4000, "call"),
    (4001, ":"),
    (4020, "send_message"),
    (4003, "{"),
    (4021, "destination"),
    (4005, ":"),
    (QUOTED_ID, '<|"|>'),
    (4022, "456 Oakwood Avenue"),
    (4023, ", Rivermist"),
    (4024, ", 83214"),
    (QUOTED_ID, '<|"|>'),
    (4012, "}"),
    (TOOL_CALL_END_ID, "<tool_call|>"),
]


class TestCommaInStringValueRegression:
    """Regression: <|"|> delimiters must not be auto-dropped.

    When _build_drop_info discovers <|"|> as a special token and it is
    not in configured_texts, the delimiter is silently removed. Without
    it, _parse_gemma4_args treats commas inside string values as field
    separators.
    """

    @pytest.fixture
    def comma_tokenizer(self):
        return _make_tokenizer(COMMA_TOKEN_SEQUENCE)

    @pytest.fixture
    def comma_parser(self, comma_tokenizer):
        return Gemma4Parser(comma_tokenizer)

    @pytest.fixture
    def multi_comma_tokenizer(self):
        return _make_tokenizer(MULTI_COMMA_TOKEN_SEQUENCE)

    @pytest.fixture
    def multi_comma_parser(self, multi_comma_tokenizer):
        return Gemma4Parser(multi_comma_tokenizer)

    def test_batched_streaming_comma_in_value(
        self, comma_parser, comma_tokenizer, request_obj
    ):
        results = _stream_tokens_batched(
            comma_parser,
            comma_tokenizer,
            request_obj,
            batch_size=1,
            prompt_token_ids=[],
        )
        _, _, tool_calls = _collect_fields(results)
        assert len(tool_calls) > 0
        args_text = "".join(
            tc.function.arguments
            for tc in tool_calls
            if tc.function and tc.function.arguments
        )
        parsed = json.loads(args_text)
        assert parsed["location"] == "San Francisco, CA"
        assert parsed["unit"] == "celsius"

    def test_batched_streaming_multiple_commas(
        self, multi_comma_parser, multi_comma_tokenizer, request_obj
    ):
        results = _stream_tokens_batched(
            multi_comma_parser,
            multi_comma_tokenizer,
            request_obj,
            batch_size=1,
            prompt_token_ids=[],
        )
        _, _, tool_calls = _collect_fields(results)
        assert len(tool_calls) > 0
        args_text = "".join(
            tc.function.arguments
            for tc in tool_calls
            if tc.function and tc.function.arguments
        )
        parsed = json.loads(args_text)
        assert parsed["destination"] == "456 Oakwood Avenue, Rivermist, 83214"

    def test_non_streaming_comma_in_value(self, comma_parser, request_obj):
        text = "".join(text for _, text in COMMA_TOKEN_SEQUENCE)
        result = comma_parser.extract_tool_calls(text, request_obj)
        assert result.tools_called is True
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["location"] == "San Francisco, CA"
        assert args["unit"] == "celsius"

    def test_non_streaming_multiple_commas(self, multi_comma_parser, request_obj):
        text = "".join(text for _, text in MULTI_COMMA_TOKEN_SEQUENCE)
        result = multi_comma_parser.extract_tool_calls(text, request_obj)
        assert result.tools_called is True
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["destination"] == "456 Oakwood Avenue, Rivermist, 83214"
