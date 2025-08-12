from typing import Iterable, List, Tuple, Union

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaMessage,
                                              ExtractedToolCallInformation,
                                              FunctionCall, ToolCall)
from vllm.entrypoints.openai.tool_parsers import ToolParser


class StreamingToolReconstructor:

    def __init__(self, assert_one_tool_per_delta: bool = True):
        self.tool_calls: List[ToolCall] = []
        self.other_content: str = ""
        self._assert_one_tool_per_delta = assert_one_tool_per_delta

    def append_delta(self, delta: DeltaMessage):
        if delta.content is not None:
            self.other_content += delta.content
        else:
            assert delta.tool_calls, (
                "Streaming results should have either content or tool calls "
                "(or both)")
        if self._assert_one_tool_per_delta:
            # Note: This isn't strictly required by the API and may not be
            # possible to adhere to depending on the token space and number of
            # tokens per streamed response from the model, but it is required
            # by tool_use tests, so we enforce it here by default also.
            assert len(delta.tool_calls) < 2, (
                "Streaming should include only one tool call per update.")
        for call_delta in delta.tool_calls:
            assert call_delta.type == "function", (
                "Streaming tool calls should only emit function calls. Got "
                f"{call_delta.type}")
            current_tool_call = self.tool_calls[
                call_delta.index] if call_delta.index < len(
                    self.tool_calls) else None
            if current_tool_call:
                assert (not call_delta.function.name), (
                    "Streaming tool calls should emit the full function name "
                    f"exactly once. Got {call_delta.function.name}")
                assert (not call_delta.id), (
                    "Streaming tool calls must emit function id only once. Got "
                    f"{call_delta.id}")
                assert (call_delta.index == len(self.tool_calls) - 1), (
                    f"Incorrect index for tool delta. Got {call_delta.index}, "
                    f"expected {len(self.tool_calls) - 1}")
                current_tool_call.function.arguments += (
                    call_delta.function.arguments)
            else:
                assert call_delta.id is not None, (
                    "Streaming tool calls must have an id on first appearance")
                assert call_delta.function.name is not None, (
                    "Streaming tool calls must have a function name on first "
                    "appearance")
                assert call_delta.index == len(self.tool_calls), (
                    f"Incorrect index for tool delta. Got {call_delta.index}, "
                    f"expected {len(self.tool_calls)}")
                self.tool_calls.append(
                    ToolCall(id=call_delta.id,
                             function=FunctionCall(
                                 name=call_delta.function.name,
                                 arguments=call_delta.function.arguments
                                 or "")))


def run_tool_extraction(
    tool_parser: ToolParser,
    model_output: str,
    request: Union[ChatCompletionRequest, None] = None,
    streaming: bool = False,
    assert_one_tool_per_delta: bool = True,
) -> Tuple[Union[str, None], List[ToolCall]]:
    if streaming:
        reconstructor = run_tool_extraction_streaming(
            tool_parser,
            model_output,
            request,
            assert_one_tool_per_delta=assert_one_tool_per_delta)
        return reconstructor.other_content or None, reconstructor.tool_calls
    else:
        extracted = run_tool_extraction_nonstreaming(tool_parser, model_output,
                                                     request)
        assert extracted.tools_called == bool(extracted.tool_calls)
        return extracted.content, extracted.tool_calls


def run_tool_extraction_nonstreaming(
    tool_parser: ToolParser,
    model_output: str,
    request: Union[ChatCompletionRequest, None] = None
) -> ExtractedToolCallInformation:
    request = request or ChatCompletionRequest(messages=[], model="test-model")
    return tool_parser.extract_tool_calls(model_output, request)


def run_tool_extraction_streaming(
    tool_parser: ToolParser,
    model_deltas: Iterable[str],
    request: Union[ChatCompletionRequest, None] = None,
    assert_one_tool_per_delta: bool = True,
) -> StreamingToolReconstructor:
    request = request or ChatCompletionRequest(messages=[], model="test-model")
    reconstructor = StreamingToolReconstructor(
        assert_one_tool_per_delta=assert_one_tool_per_delta)
    previous_text = ""
    previous_tokens: List[int] = []
    for delta in model_deltas:
        token_delta = [
            tool_parser.vocab.get(token)
            for token in tool_parser.model_tokenizer.tokenize(delta)
            if token in tool_parser.vocab
        ]
        current_text = previous_text + delta
        current_tokens = previous_tokens + token_delta
        delta_message = tool_parser.extract_tool_calls_streaming(
            previous_text, current_text, delta, previous_tokens,
            current_tokens, token_delta, request)
        if delta_message is not None:
            reconstructor.append_delta(delta_message)
        previous_text = current_text
        previous_tokens = current_tokens
    return reconstructor
