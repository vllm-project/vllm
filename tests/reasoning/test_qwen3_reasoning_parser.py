# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

import pytest
from transformers import AutoTokenizer

from tests.reasoning.utils import (
    StreamingReasoningReconstructor,
    run_reasoning_extraction,
    run_reasoning_extraction_streaming,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    ExtractedToolCallInformation,
)
from vllm.parser.abstract_parser import DelegatingParser
from vllm.parser.parser_manager import ParserManager
from vllm.reasoning import ReasoningParser, ReasoningParserManager
from vllm.tool_parsers.abstract_tool_parser import ToolParser

parser_name = "qwen3"
start_token = "<think>"
end_token = "</think>"

REASONING_MODEL_NAMES = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3.5-397B-A17B",
    "Qwen/Qwen3-4B-Thinking-2507",
]


@pytest.fixture(scope="module", params=REASONING_MODEL_NAMES)
def qwen3_tokenizer(request):
    return AutoTokenizer.from_pretrained(request.param)


# --- <think> in prompt, only </think> in output (typical) ---

WITHOUT_START_TOKEN = {
    "output": "This is a reasoning section</think>This is the rest",
    "reasoning": "This is a reasoning section",
    "content": "This is the rest",
}
WITHOUT_START_TOKEN_STREAM = {
    "output": "This is a reasoning section</think>This is the rest",
    "reasoning": "This is a reasoning section",
    "content": "This is the rest",
}
WITHOUT_START_TOKEN_COMPLETE_REASONING = {
    "output": "This is a reasoning section</think>",
    "reasoning": "This is a reasoning section",
    "content": None,
}

# --- <think> present in output (old template / edge case) ---

WITH_THINK = {
    "output": "<think>This is a reasoning section</think>This is the rest",
    "reasoning": "This is a reasoning section",
    "content": "This is the rest",
}
WITH_THINK_STREAM = {
    "output": "<think>This is a reasoning section</think>This is the rest",
    "reasoning": "This is a reasoning section",
    "content": "This is the rest",
}

# --- No think tokens at all (thinking enabled, truncated) ---

# With thinking enabled (default), no think tokens means the output was
# truncated before </think> could be generated. All output is reasoning.
WITHOUT_THINK = {
    "output": "This is the rest",
    "reasoning": "This is the rest",
    "content": None,
}
# In streaming, the parser cannot distinguish "thinking disabled" from
# "reasoning in progress" when no think tokens have appeared yet.
# It assumes reasoning. The serving layer handles the "thinking disabled"
# case by checking prompt_is_reasoning_end_arr before calling the parser.
WITHOUT_THINK_STREAM = {
    "output": "This is the rest",
    "reasoning": "This is the rest",
    "content": None,
}

# --- <tool_call> without </think> (implicit reasoning end) ---

TOOL_CALL_BODY = (
    "<tool_call>\n<function=bash>\n<parameter=command>"
    "\ncat /etc/hosts\n</parameter>\n</function>\n</tool_call>"
)

TOOL_CALL_NO_THINK_END = {
    "output": "I need to read the file.\n\n" + TOOL_CALL_BODY,
    "reasoning": "I need to read the file.\n\n",
    "content": TOOL_CALL_BODY,
}

TOOL_CALL_WITH_THINK_NO_END = {
    "output": "<think>I need to read the file.\n\n" + TOOL_CALL_BODY,
    "reasoning": "I need to read the file.\n\n",
    "content": TOOL_CALL_BODY,
}

# --- Edge cases ---

COMPLETE_REASONING = {
    "output": "<think>This is a reasoning section</think>",
    "reasoning": "This is a reasoning section",
    "content": None,
}
MULTILINE_REASONING = {
    "output": "<think>This is a reasoning\nsection</think>This is the rest\nThat",
    "reasoning": "This is a reasoning\nsection",
    "content": "This is the rest\nThat",
}
# Truncated output: <think> present but no </think> (thinking enabled).
# Everything is reasoning because the output was cut off mid-thought.
ONLY_OPEN_TAG = {
    "output": "<think>This is a reasoning section",
    "reasoning": "This is a reasoning section",
    "content": None,
}

ONLY_OPEN_TAG_STREAM = {
    "output": "<think>This is a reasoning section",
    "reasoning": "This is a reasoning section",
    "content": None,
}

# Truncated output without <think> prefix (Qwen3.5 style where <think>
# is in the prompt). No </think> means truncation — all is reasoning.
TRUNCATED_NO_START_TOKEN = {
    "output": "This is a reasoning section",
    "reasoning": "This is a reasoning section",
    "content": None,
}

TRUNCATED_NO_START_TOKEN_STREAM = {
    "output": "This is a reasoning section",
    "reasoning": "This is a reasoning section",
    "content": None,
}

TEST_CASES = [
    pytest.param(
        False,
        WITHOUT_START_TOKEN,
        id="without_start_token",
    ),
    pytest.param(
        True,
        WITHOUT_START_TOKEN_STREAM,
        id="without_start_token_stream",
    ),
    pytest.param(
        False,
        WITHOUT_START_TOKEN_COMPLETE_REASONING,
        id="without_start_token_complete_reasoning",
    ),
    pytest.param(
        True,
        WITHOUT_START_TOKEN_COMPLETE_REASONING,
        id="without_start_token_complete_reasoning_stream",
    ),
    pytest.param(
        False,
        WITH_THINK,
        id="with_think",
    ),
    pytest.param(
        True,
        WITH_THINK_STREAM,
        id="with_think_stream",
    ),
    pytest.param(
        False,
        WITHOUT_THINK,
        id="without_think",
    ),
    pytest.param(
        True,
        WITHOUT_THINK_STREAM,
        id="without_think_stream",
    ),
    pytest.param(
        False,
        COMPLETE_REASONING,
        id="complete_reasoning",
    ),
    pytest.param(
        True,
        COMPLETE_REASONING,
        id="complete_reasoning_stream",
    ),
    pytest.param(
        False,
        MULTILINE_REASONING,
        id="multiline_reasoning",
    ),
    pytest.param(
        True,
        MULTILINE_REASONING,
        id="multiline_reasoning_stream",
    ),
    pytest.param(
        False,
        ONLY_OPEN_TAG,
        id="only_open_tag",
    ),
    pytest.param(
        True,
        ONLY_OPEN_TAG_STREAM,
        id="only_open_tag_stream",
    ),
    pytest.param(
        False,
        TRUNCATED_NO_START_TOKEN,
        id="truncated_no_start_token",
    ),
    pytest.param(
        True,
        TRUNCATED_NO_START_TOKEN_STREAM,
        id="truncated_no_start_token_stream",
    ),
    pytest.param(
        False,
        TOOL_CALL_NO_THINK_END,
        id="tool_call_no_think_end",
    ),
    pytest.param(
        True,
        TOOL_CALL_NO_THINK_END,
        id="tool_call_no_think_end_stream",
    ),
    pytest.param(
        False,
        TOOL_CALL_WITH_THINK_NO_END,
        id="tool_call_with_think_no_end",
    ),
    pytest.param(
        True,
        TOOL_CALL_WITH_THINK_NO_END,
        id="tool_call_with_think_no_end_stream",
    ),
]


@pytest.mark.parametrize("streaming, param_dict", TEST_CASES)
def test_reasoning(
    streaming: bool,
    param_dict: dict,
    qwen3_tokenizer,
):
    output = qwen3_tokenizer.tokenize(param_dict["output"])
    output_tokens: list[str] = [
        qwen3_tokenizer.convert_tokens_to_string([token]) for token in output
    ]
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(parser_name)(
        qwen3_tokenizer
    )

    reasoning, content = run_reasoning_extraction(
        parser, output_tokens, streaming=streaming
    )

    assert reasoning == param_dict["reasoning"]
    assert content == param_dict["content"]


# Multi-token delta tests: simulate real-world streaming where a single
# delta can contain multiple tokens (e.g., speculative decoding).
MULTI_TOKEN_DELTA_CASES = [
    pytest.param(
        # <think> grouped with following text in one delta
        ["<think>This is a reasoning section", "</think>", "This is the rest"],
        "This is a reasoning section",
        "This is the rest",
        id="start_token_grouped_with_text",
    ),
    pytest.param(
        # </think> grouped with following content in one delta
        ["reasoning section", "</think>This is the rest"],
        "reasoning section",
        "This is the rest",
        id="end_token_grouped_with_content",
    ),
    pytest.param(
        # <think> and </think> in the same delta, no content after
        ["<think>reasoning</think>"],
        "reasoning",
        None,
        id="start_and_end_in_one_delta_no_content",
    ),
    pytest.param(
        # No start token, end grouped with content (Qwen3.5 style)
        ["reasoning section", "</think>content"],
        "reasoning section",
        "content",
        id="no_start_end_grouped_with_content",
    ),
    pytest.param(
        # <tool_call> arrives in a separate delta after reasoning text
        ["I need to read the file.\n\n", "<tool_call>\n<function=bash>"],
        "I need to read the file.\n\n",
        "<tool_call>\n<function=bash>",
        id="tool_call_implicit_reasoning_end",
    ),
]


@pytest.mark.parametrize(
    "deltas, expected_reasoning, expected_content", MULTI_TOKEN_DELTA_CASES
)
def test_reasoning_streaming_multi_token_deltas(
    deltas: list[str],
    expected_reasoning: str | None,
    expected_content: str | None,
    qwen3_tokenizer,
):
    """Test that multi-token deltas don't leak <think> into reasoning."""
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(parser_name)(
        qwen3_tokenizer
    )

    reconstructor: StreamingReasoningReconstructor = run_reasoning_extraction_streaming(
        parser, deltas
    )

    assert reconstructor.reasoning == expected_reasoning
    assert (reconstructor.other_content or None) == expected_content


# --- Tests for enable_thinking=False (thinking explicitly disabled) ---


THINKING_DISABLED_CASES = [
    pytest.param(
        "This is plain content",
        None,
        "This is plain content",
        id="thinking_disabled_plain_content",
    ),
    pytest.param(
        "Some output without think tokens",
        None,
        "Some output without think tokens",
        id="thinking_disabled_no_think_tokens",
    ),
    pytest.param(
        "I need to read the file.\n\n" + TOOL_CALL_BODY,
        None,
        "I need to read the file.\n\n" + TOOL_CALL_BODY,
        id="thinking_disabled_with_tool_call",
    ),
]


@pytest.mark.parametrize(
    "output, expected_reasoning, expected_content", THINKING_DISABLED_CASES
)
def test_reasoning_thinking_disabled(
    output: str,
    expected_reasoning: str | None,
    expected_content: str | None,
    qwen3_tokenizer,
):
    """When enable_thinking=False, output without </think> is all content."""
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(parser_name)(
        qwen3_tokenizer,
        chat_template_kwargs={"enable_thinking": False},
    )

    reasoning, content = parser.extract_reasoning(
        model_output=output,
        request=ChatCompletionRequest(messages=[], model="test-model"),
    )

    assert reasoning == expected_reasoning
    assert content == expected_content


def test_is_reasoning_end_complete_tool_call_in_single_delta(
    qwen3_tokenizer,
) -> None:
    """
    Regression – is_reasoning_end_streaming must return True when a complete
    <tool_call>…</tool_call> arrives in a single delta (MTP / speculative-
    decoding scenario).

    is_reasoning_end uses a paired-token guard so it returns False when
    <tool_call> is followed by </tool_call> (prompt examples). For streaming
    delta checks, is_reasoning_end_streaming is used instead — it inspects
    only the delta_ids and always returns True when <tool_call> is present.
    """
    vocab = qwen3_tokenizer.get_vocab()
    tool_call_id = vocab["<tool_call>"]
    tool_call_end_id = vocab["</tool_call>"]
    content_token = qwen3_tokenizer.encode("hello", add_special_tokens=False)[0]

    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(
        parser_name
    )(qwen3_tokenizer)

    # Simulate accumulated token IDs before this delta
    prior_ids = [content_token]
    # Single delta that contains the complete <tool_call>…</tool_call>
    delta_ids = [tool_call_id, content_token, tool_call_end_id]

    assert parser.is_reasoning_end_streaming(prior_ids, delta_ids), (
        "is_reasoning_end_streaming returned False for a complete "
        "<tool_call>…</tool_call> pair in a single delta. "
        "Speculative-decoding flushes can deliver both the opening and closing "
        "tokens at once; the streaming check must still signal reasoning end."
    )


def test_is_reasoning_end_streaming_detects_tool_call_in_delta(
    qwen3_tokenizer,
) -> None:
    """
    is_reasoning_end_streaming must return True when the current delta
    contains <tool_call>, regardless of whether </tool_call> is also present.

    This covers the real-world case where the model emits <tool_call> as its
    first implicit-end-of-reasoning signal.  abstract_parser uses
    is_reasoning_end_streaming (not is_reasoning_end) for per-delta checks so
    that the paired-token guard in is_reasoning_end does not prevent the
    streaming transition.
    """
    vocab = qwen3_tokenizer.get_vocab()
    tool_call_id = vocab["<tool_call>"]
    tool_call_end_id = vocab["</tool_call>"]
    reasoning_token = qwen3_tokenizer.encode("thinking", add_special_tokens=False)[0]
    content_token = qwen3_tokenizer.encode("hello", add_special_tokens=False)[0]

    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(
        parser_name
    )(qwen3_tokenizer)

    # Accumulated token IDs up to (but not including) the current delta
    prior_ids = [reasoning_token, reasoning_token]

    # Delta that begins the tool call (speculative decoding may include the
    # entire <tool_call>…</tool_call> sequence in one flush)
    delta_ids = [tool_call_id, content_token, tool_call_end_id]

    assert parser.is_reasoning_end_streaming(prior_ids, delta_ids), (
        "is_reasoning_end_streaming returned False when the delta contains "
        "<tool_call>. abstract_parser will miss the reasoning→tool-call "
        "transition and route all subsequent output to the reasoning parser."
    )


def test_extract_reasoning_streaming_fragmented_tool_call(qwen3_tokenizer):
    """
    Test streaming reasoning extraction when the <tool_call> tag is fragmented
    across multiple deltas and delta_token_ids does not contain the special token ID.
    """
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(parser_name)(
        qwen3_tokenizer
    )

    # Force the parser to have a valid tool call tag
    # (Qwen3 uses <tool_call>)
    assert parser._tool_call_tag == "<tool_call>"
    
    # Delta 1: "<to"
    previous_text = "Here is my reasoning. "
    current_text = "Here is my reasoning. <to"
    msg1 = parser.extract_reasoning_streaming(
        previous_text=previous_text,
        current_text=current_text,
        delta_text="<to",
        previous_token_ids=[1, 2],
        current_token_ids=[1, 2, 3],
        delta_token_ids=[3],
    )
    assert msg1 is not None
    # Tag fragments should NOT be reasoning
    assert msg1.reasoning is None
    assert msg1.content is None

    # Delta 2: "ol_"
    previous_text = current_text
    current_text = current_text + "ol_"
    msg2 = parser.extract_reasoning_streaming(
        previous_text=previous_text,
        current_text=current_text,
        delta_text="ol_",
        previous_token_ids=[1, 2, 3],
        current_token_ids=[1, 2, 3, 4],
        delta_token_ids=[4],
    )
    assert msg2 is not None
    # Tag fragments should NOT be reasoning
    assert msg2.reasoning is None
    assert msg2.content is None

    # Delta 3: "call>\n<function=bash>"
    previous_text = current_text
    current_text = current_text + "call>\n<function=bash>"
    msg3 = parser.extract_reasoning_streaming(
        previous_text=previous_text,
        current_text=current_text,
        delta_text="call>\n<function=bash>",
        previous_token_ids=[1, 2, 3, 4],
        current_token_ids=[1, 2, 3, 4, 5, 6],
        delta_token_ids=[5, 6],
    )
    assert msg3 is not None
    assert msg3.reasoning is None
    assert msg3.content == "<tool_call>\n<function=bash>"


class _StubToolParser(ToolParser):
    """Always returns DeltaMessage(content="[tool]") to detect overwrites."""

    def extract_tool_calls(
        self, model_output: str, _request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        return ExtractedToolCallInformation(
            tools_called=False, tool_calls=[], content=model_output
        )

    def extract_tool_calls_streaming(
        self,
        _previous_text: str,
        _current_text: str,
        _delta_text: str,
        _previous_token_ids: Sequence[int],
        _current_token_ids: Sequence[int],
        _delta_token_ids: Sequence[int],
        _request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        return DeltaMessage(content="[tool]")


def test_reasoning_preserved_when_transition_and_tool_call_in_same_delta(
    qwen3_tokenizer,
) -> None:
    """
    When a delta contains both the last reasoning fragment AND the <tool_call>
    token (implicit reasoning end), parse_delta must not lose the reasoning
    text by overwriting delta_message with the result of
    extract_tool_calls_streaming.

    Sequence inside parse_delta (before fix):
      1. extract_reasoning_streaming  → DeltaMessage(reasoning="last text",
                                                      content="<tool_call>…")
      2. Transition: reasoning_ended = True, current_text = "<tool_call>…"
      3. extract_tool_calls_streaming → DeltaMessage(content="[tool]")
      4. delta_message = step-3 result  ← reasoning "last text" silently lost
    """
    vocab = qwen3_tokenizer.get_vocab()
    tool_call_id = vocab["<tool_call>"]
    reasoning_token = qwen3_tokenizer.encode("reasoning", add_special_tokens=False)[0]

    parser_cls = ParserManager.get_parser(
        reasoning_parser_name="qwen3",
        tool_parser_name="qwen3_coder",
    )
    # Swap in the stub so the test doesn't depend on qwen3_coder streaming logic
    parser = parser_cls(qwen3_tokenizer)
    parser._tool_parser = _StubToolParser(qwen3_tokenizer)

    request = ChatCompletionRequest(messages=[], model="test-model")
    delta_text = "last reasoning text\n<tool_call>\n<function=test_func>"

    result = parser.parse_delta(
        delta_text=delta_text,
        delta_token_ids=[reasoning_token, tool_call_id],
        request=request,
    )

    assert result is not None, (
        "parse_delta returned None — both reasoning and tool content were lost."
    )
    assert result.reasoning == "last reasoning text\n", (
        f"last reasoning fragment lost in transition delta. "
        f"Got reasoning={result.reasoning!r}, expected 'last reasoning text\\n'. "
        f"delta_message was overwritten by extract_tool_calls_streaming."
    )


def test_reasoning_swallows_post_think_content(qwen3_tokenizer):
    """
    Regression test: If </think> was ALREADY emitted, and a later delta contains <tool_call>,
    the parser incorrectly sends the text BEFORE <tool_call> as reasoning instead of content.
    """
    from vllm.reasoning import ReasoningParserManager
    parser = ReasoningParserManager.get_reasoning_parser("qwen3")(qwen3_tokenizer)
    
    # Delta 1: ends thinking
    end_token_id = qwen3_tokenizer.get_vocab()["</think>"]
    msg1 = parser.extract_reasoning_streaming(
        previous_text="<think> thoughts",
        current_text="<think> thoughts</think>",
        delta_text="</think>",
        previous_token_ids=[1, 2],
        current_token_ids=[1, 2, end_token_id], # end_token_id is </think>
        delta_token_ids=[end_token_id]
    )
    if msg1 is not None:
        assert msg1.content is None
    
    # Delta 2: content + tool call in same delta (e.g. speculative decoding or fast generation)
    # The text "I will use a tool\n" is clearly AFTER </think>, so it must be content!
    vocab = qwen3_tokenizer.get_vocab()
    tool_call_id = vocab["<tool_call>"]
    
    prev_text = "<think> thoughts</think>"
    delta_text = "I will use a tool.\n<tool_call>"
    curr_text = prev_text + delta_text
    
    msg2 = parser.extract_reasoning_streaming(
        previous_text=prev_text,
        current_text=curr_text,
        delta_text=delta_text,
        previous_token_ids=[1, 2, end_token_id],
        current_token_ids=[1, 2, end_token_id, 3, 4, tool_call_id],
        delta_token_ids=[3, 4, tool_call_id]
    )
    
    assert msg2 is not None
    # "I will use a tool.\n" should NOT be reasoning!
    assert msg2.reasoning is None, f"Parser incorrectly swallowed post-think text into reasoning: {msg2.reasoning}"
    assert "I will use a tool" in msg2.content


def test_extract_content_ids_multiple_tool_calls(qwen3_tokenizer):
    """
    Regression test: extract_content_ids should retain ALL tool calls when
    reasoning implicitly ends with <tool_call>.
    Previously, it searched backwards and discarded all but the LAST tool call.
    """
    parser = ReasoningParserManager.get_reasoning_parser("qwen3")(qwen3_tokenizer)
    vocab = qwen3_tokenizer.get_vocab()
    tool_call_id = vocab["<tool_call>"]
    
    # input_ids: [..., tool_call, ..., tool_call, ...] without </think>
    input_ids = [1, 2, tool_call_id, 3, 4, tool_call_id, 5, 6]
    
    content_ids = parser.extract_content_ids(input_ids)
    
    # Content should start at the FIRST tool call, retaining both.
    assert content_ids == [tool_call_id, 3, 4, tool_call_id, 5, 6], \
        "extract_content_ids deleted the first tool call by searching backwards."


def test_extract_reasoning_streaming_fragmented_end_and_tool_call(qwen3_tokenizer):
    """
    Regression test: When the end token (</think>) or tool call tag is fragmented
    across deltas, the parser incorrectly swallows parts of the tag into reasoning,
    corrupting the content sent to the tool parser.
    """
    parser = ReasoningParserManager.get_reasoning_parser("qwen3")(qwen3_tokenizer)
    
    # Delta 1: ends thinking but incomplete
    prev_text_1 = "<think> reasoning "
    delta_text_1 = "</think>\n<to"
    curr_text_1 = prev_text_1 + delta_text_1
    
    msg1 = parser.extract_reasoning_streaming(
        previous_text=prev_text_1,
        current_text=curr_text_1,
        delta_text=delta_text_1,
        previous_token_ids=[1, 2],
        current_token_ids=[1, 2, 3],
        delta_token_ids=[3],
    )
    
    # Since parser cannot know if <to will complete, it has to be handled carefully,
    # but the bug manifests when the tag COMPLETES in the next delta.
    
    # Delta 2: completes tool call
    prev_text_2 = curr_text_1
    delta_text_2 = "ol_call>\n<function="
    curr_text_2 = prev_text_2 + delta_text_2
    
    msg2 = parser.extract_reasoning_streaming(
        previous_text=prev_text_2,
        current_text=curr_text_2,
        delta_text=delta_text_2,
        previous_token_ids=[1, 2, 3],
        current_token_ids=[1, 2, 3, 4],
        delta_token_ids=[4],
    )
    
    # If the parser split blindly at the point the tag completed in current_text,
    # it would return 'ol_call>\n<function=' as content, missing the '<to' part.
    if msg2 and msg2.content:
        assert msg2.content != "ol_call>\n<function=", \
            "Parser corrupted the tool call tag by splitting it across reasoning and content."


def test_streaming_thinking_disabled_treats_output_as_content(qwen3_tokenizer):
    """When ``enable_thinking=False`` the chat template injects
    ``<think>\\n\\n</think>\\n\\n`` into the prompt, so the model output
    starts with content, not reasoning.

    The non-streaming ``extract_reasoning`` honours ``self.thinking_enabled``
    and returns the output as content.  The streaming path was relying on
    the serving layer to detect this via ``prompt_is_reasoning_end`` and
    bypass the parser entirely — but if the parser is invoked directly
    (or if that bypass ever regresses) the streaming path silently routes
    the model output to the reasoning channel.

    Streaming should be self-consistent with the non-streaming path:
    when thinking is explicitly disabled and no ``</think>`` arrives in
    the output, every delta is content.
    """
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(parser_name)(
        qwen3_tokenizer,
        chat_template_kwargs={"enable_thinking": False},
    )

    delta_text = "Hello world, no thinking."
    msg = parser.extract_reasoning_streaming(
        previous_text="",
        current_text=delta_text,
        delta_text=delta_text,
        previous_token_ids=[],
        current_token_ids=[1, 2],
        delta_token_ids=[1, 2],
    )
    assert msg is not None
    assert msg.content == delta_text, (
        f"With enable_thinking=False, streaming output must be content. "
        f"Got: content={msg.content!r}, reasoning={msg.reasoning!r}"
    )
    assert msg.reasoning is None, (
        f"With enable_thinking=False, no token should be marked reasoning. "
        f"Got reasoning={msg.reasoning!r}"
    )


def test_count_reasoning_tokens_qwen35_template(qwen3_tokenizer):
    """For the Qwen3.5+ chat template ``<think>`` lives in the prompt,
    so the model output starts with reasoning tokens and only ``</think>``
    ever appears.  The inherited depth-counter starts at depth=0 and is
    never incremented (no ``<think>`` seen in output), so it returns 0
    even when the output begins with a long reasoning span.

    The Qwen3 parser must override this to "treat the start of the
    output as the reasoning region until ``</think>`` (or implicit
    ``<tool_call>``) is seen".
    """
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(parser_name)(
        qwen3_tokenizer
    )
    vocab = qwen3_tokenizer.get_vocab()
    end_token_id = vocab["</think>"]
    reason_token = qwen3_tokenizer.encode("hello", add_special_tokens=False)[0]
    content_token = qwen3_tokenizer.encode("world", add_special_tokens=False)[0]

    # Output: 3 reasoning tokens, then </think>, then 2 content tokens.
    output_ids = [reason_token, reason_token, reason_token, end_token_id,
                  content_token, content_token]

    n = parser.count_reasoning_tokens(output_ids)
    assert n == 3, (
        f"count_reasoning_tokens returned {n} for a Qwen3.5-style output "
        f"with 3 reasoning tokens before </think>; the inherited counter "
        f"never sees <think> (it lives in the prompt) so depth stays at 0."
    )


def test_streaming_partial_tool_call_prefix_is_not_lost(qwen3_tokenizer):
    """Held-back partial-``<tool_call>`` overlap must be re-emitted as
    reasoning when the next deltas reveal it was NOT actually a tool call.

    The parser holds back the tail of the current text whenever it looks
    like the prefix of ``<tool_call>``.  If the model then emits unrelated
    text (e.g. ``<tool_use>`` or just ``<tool_belt``), the held-back
    characters must be flushed as reasoning rather than silently dropped.
    """
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(parser_name)(
        qwen3_tokenizer
    )

    # Delta 1 ends with "<tool" — looks like the start of "<tool_call>",
    # so the parser holds those 5 chars back.
    prev_text = "thinking "
    delta_text_1 = "<tool"
    curr_text_1 = prev_text + delta_text_1
    msg1 = parser.extract_reasoning_streaming(
        previous_text=prev_text,
        current_text=curr_text_1,
        delta_text=delta_text_1,
        previous_token_ids=[1],
        current_token_ids=[1, 2],
        delta_token_ids=[2],
    )

    # Delta 2 turns out to NOT be "<tool_call>": the model emitted
    # "<tool_use>" instead. The held-back "<tool" must not be lost.
    prev_text_2 = curr_text_1
    delta_text_2 = "_use>"
    curr_text_2 = prev_text_2 + delta_text_2
    msg2 = parser.extract_reasoning_streaming(
        previous_text=prev_text_2,
        current_text=curr_text_2,
        delta_text=delta_text_2,
        previous_token_ids=[1, 2],
        current_token_ids=[1, 2, 3],
        delta_token_ids=[3],
    )

    reasoning_emitted = ""
    for msg in (msg1, msg2):
        if msg and msg.reasoning:
            reasoning_emitted += msg.reasoning

    assert "<tool" in reasoning_emitted, (
        f"Held-back '<tool' was silently dropped when the next delta "
        f"revealed it was not a tool_call. Reasoning emitted: "
        f"{reasoning_emitted!r}"
    )
    assert "_use>" in reasoning_emitted, (
        f"Second delta '_use>' missing from reasoning: {reasoning_emitted!r}"
    )


def test_is_reasoning_end_false_for_prompt_with_paired_tool_call_examples(
    qwen3_tokenizer,
) -> None:
    vocab = qwen3_tokenizer.get_vocab()
    tool_call_id = vocab["<tool_call>"]
    tool_call_end_id = vocab["</tool_call>"]
    content_token = qwen3_tokenizer.encode("hello", add_special_tokens=False)[0]

    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(
        parser_name
    )(qwen3_tokenizer)

    # Simulate a prompt whose token_ids contain a paired tool-call example:
    #   ... system_text ... <tool_call> func_body </tool_call> ... user_text ...
    # (No <think> or </think> after the pair, as is the case for non-3.5 models.)
    prompt_token_ids = [
        content_token,    # "You are a helpful assistant."
        content_token,    # "Example:"
        tool_call_id,     # <tool_call>  ← example in system prompt
        content_token,    # function body text
        tool_call_end_id, # </tool_call>  ← closes the example
        content_token,    # user turn content
    ]

    result = parser.is_reasoning_end(prompt_token_ids)
    assert not result, (
        "is_reasoning_end returned True for a prompt that contains a PAIRED "
        "<tool_call>…</tool_call> example. "
        "This makes abstract_parser set state.reasoning_ended=True before the "
        "model generates any token, routing all output (including reasoning) to "
        "the tool parser and silently discarding the <reasoning> field. "
        "The paired-token guard must be reinstated for this case."
    )


def test_parse_delta_reasoning_not_bypassed_when_prompt_has_tool_examples(
    qwen3_tokenizer,
) -> None:

    

    vocab = qwen3_tokenizer.get_vocab()
    tool_call_id = vocab["<tool_call>"]
    tool_call_end_id = vocab["</tool_call>"]
    content_token = qwen3_tokenizer.encode("hello", add_special_tokens=False)[0]
    end_token_id = vocab["</think>"]

    # Build a DelegatingParser with qwen3 reasoning + stub tool parser.
    class _TestParser(DelegatingParser):
        pass

    _TestParser._reasoning_parser = None
    _TestParser._tool_parser = None

    reasoning_parser = ReasoningParserManager.get_reasoning_parser(parser_name)(
        qwen3_tokenizer
    )
    stub_tool_parser = _StubToolParser(qwen3_tokenizer)

    parser = _TestParser(qwen3_tokenizer)
    parser._reasoning_parser = reasoning_parser
    parser._tool_parser = stub_tool_parser

    request = ChatCompletionRequest(messages=[], model="test-model")

    # prompt_token_ids: system message with a <tool_call> example, then user turn.
    # No <think> token at the end (Qwen3 non-3.5 style).
    prompt_token_ids = [
        content_token,    # system text
        tool_call_id,     # <tool_call> in system example
        content_token,    # function body
        tool_call_end_id, # </tool_call> closing the example
        content_token,    # user message
    ]

    # First delta: a reasoning token (model starts generating its reasoning).
    reasoning_text = "I need to think about this."
    reasoning_token = qwen3_tokenizer.encode(
        reasoning_text, add_special_tokens=False
    )[0]

    result = parser.parse_delta(
        delta_text=reasoning_text,
        delta_token_ids=[reasoning_token],
        request=request,
        prompt_token_ids=prompt_token_ids,
    )

    assert result is not None, (
        "parse_delta returned None for the first reasoning delta — "
        "the reasoning parser was bypassed entirely."
    )
    assert result.reasoning is not None, (
        f"First delta was routed to the tool parser instead of the reasoning "
        f"parser. Got result={result!r}. "
        f"is_reasoning_end(prompt_token_ids) returned True because the prompt "
        f"contains a <tool_call> token, causing state.reasoning_ended=True "
        f"before any model output was generated."
    )
    assert result.tool_calls is None or result.tool_calls == [], (
        "Tool calls were incorrectly populated during a reasoning-phase delta."
    )


