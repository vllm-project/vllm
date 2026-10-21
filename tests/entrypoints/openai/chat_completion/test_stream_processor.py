# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`ChatStreamProcessor` applies the per choice rules that chat streaming and
streaming chat derender must share which means both emit the same deltas."""

from unittest.mock import MagicMock

from vllm.entrypoints.generate.base.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.stream_processor import (
    ChatStreamProcessor,
)


def _request(**kwargs) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="m", messages=[{"role": "user", "content": "hi"}], **kwargs
    )


def _tool_call(index: int) -> DeltaToolCall:
    return DeltaToolCall(
        id=f"call_{index}",
        index=index,
        function=DeltaFunctionCall(name="f", arguments="{}"),
    )


def test_tool_calls_set_finish_reason_and_respect_parallel_tool_calls():
    """A streamed tool call turns a later `stop` into `tool_calls` and
    `parallel_tool_calls=False` keeps only the first call."""
    parser = MagicMock()
    parser.parse_delta.side_effect = [
        DeltaMessage(tool_calls=[_tool_call(0), _tool_call(1)]),
        None,
    ]
    processor = ChatStreamProcessor(_request(parallel_tool_calls=False), parser)

    delta, finish_reason = processor.step("x", [1], None, [0])
    assert delta is not None
    assert [tc.index for tc in delta.tool_calls] == [0]
    assert finish_reason is None

    delta, finish_reason = processor.step("", [2], "stop", [0])
    assert delta is None
    assert finish_reason == "tool_calls"
    assert processor.num_output_tokens == 2


def test_finished_without_output_skips_parser():
    """A choice that finishes before generating anything sends an empty
    delta without asking the parser."""
    parser = MagicMock()
    processor = ChatStreamProcessor(_request(), parser)

    delta, finish_reason = processor.step("", [], "length", [0])

    assert delta == DeltaMessage()
    assert finish_reason == "length"
    parser.parse_delta.assert_not_called()


def test_hidden_reasoning_hides_metadata_only_with_a_parser():
    """Logprobs and token IDs are hidden only when there is reasoning to hide."""
    request = _request(include_reasoning=False)
    assert ChatStreamProcessor(request, MagicMock()).hide_metadata
    assert not ChatStreamProcessor(request, None).hide_metadata
