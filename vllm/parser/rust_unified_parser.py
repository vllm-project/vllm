# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from collections.abc import Sequence
from contextlib import suppress
from typing import Any
from weakref import WeakKeyDictionary

from openai.types.responses.function_tool import FunctionTool

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.logger import init_logger
from vllm.parser.abstract_parser import Parser
from vllm.parser.metrics import record_tool_parser_invocation
from vllm.reasoning.rust_unified_reasoning_parser import RustUnifiedReasoningParser
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import Tool
from vllm.tool_parsers.rust_unified_tool_parser import RustUnifiedToolParser

logger = init_logger(__name__)

_TOKENIZER_METADATA_CACHE: WeakKeyDictionary[Any, Any] = WeakKeyDictionary()


def rust_unified_parser_module() -> Any:
    try:
        return importlib.import_module("vllm._rust_tool_parser")
    except ImportError as exc:
        raise RuntimeError(
            "Rust unified parsing requires the vllm._rust_tool_parser PyO3 "
            "extension. Rebuild vLLM with Rust frontend/extensions enabled."
        ) from exc


def _tokenizer_metadata(tokenizer: TokenizerLike) -> Any:
    try:
        metadata = _TOKENIZER_METADATA_CACHE.get(tokenizer)
    except TypeError:
        metadata = None
    if metadata is not None:
        return metadata

    token_to_id = {
        token: int(token_id) for token, token_id in tokenizer.get_vocab().items()
    }
    special_ids = {
        int(token_id) for token_id in getattr(tokenizer, "all_special_ids", ())
    }
    metadata = rust_unified_parser_module().TokenizerMetadata(
        token_to_id,
        special_ids,
    )
    with suppress(TypeError):
        _TOKENIZER_METADATA_CACHE[tokenizer] = metadata
    return metadata


def _extract_content_ids(
    tokenizer: TokenizerLike,
    end_marker: str | None,
    input_ids: list[int],
) -> list[int]:
    if end_marker:
        try:
            end_ids = list(tokenizer.encode(end_marker, add_special_tokens=False))
        except TypeError:
            end_ids = list(tokenizer.encode(end_marker))
        marker_len = len(end_ids)
        for index in range(len(input_ids) - marker_len, -1, -1):
            if input_ids[index : index + marker_len] == end_ids:
                return input_ids[index + marker_len :]
    return input_ids


class RustUnifiedParser(Parser):
    """Adapter from the Rust unified event stream to vLLM protocol objects."""

    reasoning_parser_cls = RustUnifiedReasoningParser
    tool_parser_cls = RustUnifiedToolParser
    rust_parser_name: str
    server_auto_tools_enabled: bool = False

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        *args,
        model_config=None,
        **kwargs,
    ):
        self.model_tokenizer = tokenizer
        self._reasoning_parser = None
        self._tool_parser = None
        self.tools = tools
        self._parser: Any | None = None
        self._stream_initialized = False
        self._stream_error: Exception | None = None
        self._tool_call_ids: dict[int, str] = {}

    def _tokenizer_metadata(self) -> Any:
        return _tokenizer_metadata(self.model_tokenizer)

    def _rust_tools(self) -> list[Any]:
        if not self.tools:
            return []

        tools: list[Any] = []
        for tool in self.tools:
            if isinstance(tool, FunctionTool):
                name = tool.name
                description = tool.description
                parameters = tool.parameters or {}
                strict = getattr(tool, "strict", None)
            elif isinstance(tool, ChatCompletionToolsParam):
                name = tool.function.name
                description = tool.function.description
                parameters = tool.function.parameters or {}
                strict = getattr(tool.function, "strict", None)
            else:
                continue
            tools.append(
                rust_unified_parser_module().Tool(
                    name,
                    description,
                    parameters,
                    strict,
                )
            )
        return tools

    def _new_parser(self) -> Any:
        return rust_unified_parser_module().UnifiedParser(
            self.rust_parser_name,
            self._rust_tools(),
            self._tokenizer_metadata(),
        )

    def _get_parser(self) -> Any:
        if self._parser is None:
            self._parser = self._new_parser()
        return self._parser

    def adjust_request(
        self,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> ChatCompletionRequest | ResponsesRequest:
        if self._get_parser().preserve_special_tokens():
            request.skip_special_tokens = False
        return request

    @property
    def reasoning_start_str(self) -> str | None:
        return self._get_parser().reasoning_start_str()

    @property
    def reasoning_end_str(self) -> str | None:
        return self._get_parser().reasoning_end_str()

    @staticmethod
    def _tool_calls_enabled(
        request: ChatCompletionRequest | ResponsesRequest,
        enable_auto_tools: bool = True,
    ) -> bool:
        return request.tool_choice != "none" and (
            enable_auto_tools or request.tool_choice not in (None, "auto")
        )

    def _tool_call_id(self, parser: Any, index: int) -> str:
        if index not in self._tool_call_ids:
            self._tool_call_ids[index] = (
                parser.tool_call_id(index) or make_tool_call_id()
            )
        return self._tool_call_ids[index]

    def _delta_from_output(
        self,
        parser: Any,
        output: Any,
        *,
        include_reasoning: bool,
        include_tool_calls: bool,
    ) -> DeltaMessage | None:
        content: list[str] = []
        reasoning: list[str] = []
        tool_calls: list[DeltaToolCall] = []

        for event in output.events:
            if event.kind == "text":
                content.append(event.text)
            elif event.kind == "reasoning":
                if include_reasoning:
                    reasoning.append(event.text)
            elif event.kind == "tool_call" and include_tool_calls:
                call = event.tool_call
                name = call.name
                tool_calls.append(
                    DeltaToolCall(
                        index=call.tool_index,
                        id=(
                            self._tool_call_id(parser, call.tool_index)
                            if name is not None
                            else None
                        ),
                        type="function" if name is not None else None,
                        function=DeltaFunctionCall(
                            name=name,
                            arguments=call.arguments,
                        ),
                    )
                )

        if not content and not reasoning and not tool_calls:
            return None
        return DeltaMessage(
            content="".join(content) or None,
            reasoning="".join(reasoning) or None,
            tool_calls=tool_calls,
        )

    def _parse_complete(
        self,
        model_output: str,
        prompt_token_ids: Sequence[int],
    ) -> tuple[Any, Any]:
        parser = self._new_parser()
        output = rust_unified_parser_module().UnifiedParserOutput()
        parser.initialize(list(prompt_token_ids))
        parser.parse_into(model_output, output)
        return parser, output

    def parse(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
        enable_auto_tools: bool = False,
        model_output_token_ids: Sequence[int] = (),
        prompt_token_ids: Sequence[int] = (),
    ) -> tuple[str | None, str | None, list[FunctionCall] | None]:
        del model_output_token_ids
        include_tools = self._tool_calls_enabled(request, enable_auto_tools)
        is_tool_called: bool | Exception = False
        try:
            parser, output = self._parse_complete(model_output, prompt_token_ids)
            self._parser = parser
            output.append(parser.finish())

            reasoning: list[str] = []
            content: list[str] = []
            calls: dict[int, dict[str, str | None]] = {}
            call_ids: dict[int, str] = {}
            for event in output.events:
                if event.kind == "reasoning":
                    reasoning.append(event.text)
                elif event.kind == "text":
                    content.append(event.text)
                elif event.kind == "tool_call" and include_tools:
                    call = event.tool_call
                    current = calls.setdefault(
                        call.tool_index,
                        {"name": None, "arguments": ""},
                    )
                    if call.name is not None:
                        current["name"] = call.name
                        call_ids[call.tool_index] = (
                            parser.tool_call_id(call.tool_index) or make_tool_call_id()
                        )
                    current["arguments"] = (current["arguments"] or "") + call.arguments

            tool_calls: list[FunctionCall] = []
            for index, call in calls.items():
                name = call["name"]
                if name is None:
                    continue
                tool_calls.append(
                    FunctionCall(
                        id=call_ids.get(index),
                        name=name,
                        arguments=call["arguments"] or "",
                    )
                )
            is_tool_called = bool(tool_calls)
            return (
                "".join(reasoning) or None,
                "".join(content) or None,
                tool_calls if include_tools else [],
            )
        except Exception as error:
            is_tool_called = error
            logger.exception(
                "Error parsing complete %s output.",
                self.rust_parser_name,
            )
            return None, model_output, []
        finally:
            record_tool_parser_invocation(
                is_tool_called=is_tool_called,
                is_streaming=False,
                request=request,
            )

    def parse_delta(
        self,
        delta_text: str,
        delta_token_ids: list[int],
        request: ChatCompletionRequest | ResponsesRequest,
        prompt_token_ids: list[int] | None = None,
        *,
        finished: bool,
    ) -> DeltaMessage | None:
        del delta_token_ids
        parser = self._get_parser()

        include_tools = self._tool_calls_enabled(
            request,
            self.server_auto_tools_enabled,
        )
        output = rust_unified_parser_module().UnifiedParserOutput()
        is_tool_called: bool | Exception = False
        try:
            if self._stream_error is not None:
                return DeltaMessage(content=delta_text) if delta_text else None
            if not self._stream_initialized:
                parser.initialize(prompt_token_ids or [])
                self._stream_initialized = True
            parser.parse_into(delta_text, output)
            if finished:
                output.append(parser.finish())
            delta = self._delta_from_output(
                parser,
                output,
                include_reasoning=request.include_reasoning,
                include_tool_calls=include_tools,
            )
            is_tool_called = bool(delta and delta.tool_calls)
            return delta
        except Exception as error:
            self._stream_error = error
            is_tool_called = error
            logger.exception(
                "Error parsing streaming %s output.",
                self.rust_parser_name,
            )
            delta = self._delta_from_output(
                parser,
                output,
                include_reasoning=request.include_reasoning,
                include_tool_calls=include_tools,
            )
            recovered = parser.reset()
            if not recovered and not output.events:
                recovered = delta_text
            if recovered:
                if delta is None:
                    delta = DeltaMessage(content=recovered)
                else:
                    delta.content = (delta.content or "") + recovered
            return delta
        finally:
            record_tool_parser_invocation(
                is_tool_called=is_tool_called,
                is_streaming=True,
                request=request,
            )

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        return self._get_parser().is_reasoning_end(input_ids)

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        return self._get_parser().count_reasoning_tokens(list(token_ids))

    def prepare_structured_tag(
        self,
        original_tag: str | None,
        tool_server: Any,
    ) -> str | None:
        del tool_server
        return original_tag

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        return _extract_content_ids(
            self.model_tokenizer,
            self.reasoning_end_str,
            input_ids,
        )

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        reasoning, content, _ = self.parse(model_output, request)
        return reasoning, content

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        return DeltaMessage(content=delta_text)

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> ExtractedToolCallInformation:
        _, content, calls = self.parse(
            model_output,
            request,
            enable_auto_tools=True,
        )
        tool_calls = [
            ToolCall(
                id=call.id or make_tool_call_id(),
                function=call,
            )
            for call in calls or []
        ]
        return ExtractedToolCallInformation(
            tools_called=bool(tool_calls),
            tool_calls=tool_calls,
            content=content,
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> DeltaMessage | None:
        return self.parse_delta(
            delta_text,
            list(delta_token_ids),
            request,
            finished=False,
        )


def configured_rust_parser(
    parser_name: str,
    *,
    enable_auto_tools: bool,
) -> type[RustUnifiedParser]:
    class _RustUnifiedParser(RustUnifiedParser):
        rust_parser_name = parser_name
        server_auto_tools_enabled = enable_auto_tools

    return _RustUnifiedParser
