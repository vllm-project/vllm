# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

try:
    from cohere_melody import PyFilter, PyFilterOptions
except ImportError as e:
    raise ImportError(
        "The Cohere tool parser requires the `cohere_melody` "
        "package, which is not installed. Install it with:\n"
        "    pip install cohere_melody"
    ) from e

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
)
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers import ToolParser
from vllm.tool_parsers.utils import Tool


class BaseCohereCommandToolParser(ToolParser):
    def __init__(
        self,
        tokenizer: TokenizerLike,
        streaming_opts: PyFilterOptions,
        unary_opts: PyFilterOptions,
    ):
        super().__init__(tokenizer)
        self.melody_streaming = PyFilter(streaming_opts)
        self.melody_unary = PyFilter(unary_opts)

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        request = super().adjust_request(request)
        request.skip_special_tokens = False
        return request

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        r = self.melody_streaming.write_decoded(delta_text)
        if r.content is not None:
            return DeltaMessage(content=r.content)
        if r.reasoning is not None:
            return DeltaMessage(reasoning=r.reasoning)
        if r.tool_calls:
            return DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        id=tc.id,
                        index=tc.index,
                        type="function",
                        function=DeltaFunctionCall(
                            name=tc.name, arguments=tc.arguments
                        ),
                    )
                    for tc in r.tool_calls
                ]
            )
        return None

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        result = self.melody_unary.process_full_text(model_output)
        tool_calls = [
            ToolCall(
                id=tc.id,
                type="function",
                function=FunctionCall(name=tc.name, arguments=tc.arguments),
            )
            for tc in result.tool_calls
        ]
        return ExtractedToolCallInformation(
            tools_called=len(tool_calls) > 0,
            tool_calls=tool_calls,
            content=result.content,
        )


class CohereCommand3ToolParser(BaseCohereCommandToolParser):
    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
    ):
        super().__init__(
            tokenizer,
            streaming_opts=PyFilterOptions().cmd3(),
            unary_opts=PyFilterOptions().cmd3(),
        )


class CohereCommand4ToolParser(BaseCohereCommandToolParser):
    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
    ):
        super().__init__(
            tokenizer,
            streaming_opts=PyFilterOptions().cmd4(),
            unary_opts=PyFilterOptions().cmd4(),
        )
