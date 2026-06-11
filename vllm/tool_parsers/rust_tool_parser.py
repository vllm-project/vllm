# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from collections.abc import Sequence
from typing import Any

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
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser

logger = init_logger(__name__)


def _rust_tool_parser_module() -> Any:
    try:
        return importlib.import_module("vllm._rust_tool_parser")
    except ImportError as exc:
        raise RuntimeError(
            "Rust tool parsing requires the vllm._rust_tool_parser PyO3 "
            "extension. Rebuild vLLM with Rust frontend/extensions enabled."
        ) from exc


class RustToolParser(ToolParser):
    """Adapter from an opaque Rust parser to the vLLM ToolParser API.

    Subclasses provide only model-specific configuration: the exact Rust parser
    name and an optional tool-call start marker for fast complete-output
    rejection.

    This class keeps the vLLM-specific bridge work:
    - convert vLLM tool definitions into the Rust ``Tool`` shape;
    - translate typed Rust parser outputs into vLLM protocol objects; and
    - maintain vLLM streaming bookkeeping used by finish-reason handling.

    The parser grammar and incremental parser state stay in Rust.
    """

    # Rust-backed parsers are opaque to Python by default. Do not use vLLM's
    # standard JSON required/named handling; let the Rust parser consume the
    # model's native tool-call syntax.
    supports_required_and_named = False

    rust_parser_name: str
    tool_call_start_token: str | None = None

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)
        self._parser: Any | None = None
        self._error: Exception | None = None

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )

        logger.debug(
            "vLLM successfully imported tool parser %s", self.__class__.__name__
        )

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        """Adjust request options without installing Python-side constraints.

        Rust-backed parsers are treated as source-of-truth opaque parsers.  The
        bridge intentionally avoids ``super().adjust_request()`` so Python does
        not install JSON schema guidance or structural-tag constraints that may
        conflict with the Rust parser's native grammar.
        """
        if self._get_parser().preserve_special_tokens():
            request.skip_special_tokens = False
        return request

    def _rust_tools(self) -> list[Any]:
        """Build Rust ``Tool`` objects from vLLM tool definitions."""
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
                _rust_tool_parser_module().Tool(name, description, parameters, strict)
            )
        return tools

    def _new_parser(self) -> Any:
        """Create a fresh Rust parser with the current tool schemas."""
        return _rust_tool_parser_module().ToolParser(
            self.rust_parser_name, self._rust_tools()
        )

    def _get_parser(self) -> Any:
        if self._parser is None:
            self._parser = self._new_parser()
        return self._parser

    def _reset_streaming_state(self) -> None:
        """Reset parser state for a new request on a reused parser instance."""
        self._parser = self._new_parser()
        self._error = None
        self.prev_tool_call_arr.clear()
        self.streamed_args_for_tool.clear()
        self.current_tool_id = -1
        self.current_tool_name_sent = False

    def _ensure_tool_state(self, index: int) -> None:
        """Grow vLLM streaming state arrays to contain ``index``."""
        while len(self.prev_tool_call_arr) <= index:
            self.prev_tool_call_arr.append({})
        while len(self.streamed_args_for_tool) <= index:
            self.streamed_args_for_tool.append("")

    def _record_delta(
        self, index: int, name: str | None, arguments: str | None
    ) -> str | None:
        """Mirror a Rust parser delta into vLLM streaming bookkeeping.

        ``prev_tool_call_arr`` and ``streamed_args_for_tool`` are read later by
        the chat serving layer to decide the final ``tool_calls`` finish reason
        and to flush any remaining argument bytes.
        """
        tool_call_id = None
        self._ensure_tool_state(index)

        if name is not None:
            # Prefer the model-emitted ID surfaced by the Rust parser (e.g.
            # Kimi K2) over a randomly generated one.
            tool_call_id = self._get_parser().tool_call_id(index) or make_tool_call_id()
            self.prev_tool_call_arr[index] = {"name": name, "arguments": {}}
            self.current_tool_name_sent = True

        if arguments is not None:
            self.streamed_args_for_tool[index] += arguments
            self.prev_tool_call_arr[index]["arguments"] = self.streamed_args_for_tool[
                index
            ]
            self.current_tool_id = index

        return tool_call_id

    def _delta_message_from_parser_output(
        self, parser_output: Any | None
    ) -> DeltaMessage | None:
        """Translate one Rust parser output into a vLLM ``DeltaMessage``."""
        if parser_output is None:
            return None

        normal_text = parser_output.normal_text or None
        tool_calls: list[DeltaToolCall] = []
        for tool_call in parser_output.calls:
            index = tool_call.tool_index
            name = tool_call.name
            arguments: str | None = tool_call.arguments
            if name is None and arguments is None:
                continue

            tool_call_id = self._record_delta(index, name, arguments)
            tool_calls.append(
                DeltaToolCall(
                    index=index,
                    id=tool_call_id,
                    type="function" if name is not None else None,
                    function=DeltaFunctionCall(
                        name=name,
                        arguments=arguments,
                    ),
                )
            )

        if normal_text is None and not tool_calls:
            return None
        return DeltaMessage(content=normal_text, tool_calls=tool_calls)

    def _parse_complete(self, model_output: str) -> tuple[Any, dict[int, str]] | None:
        """Parse complete model output with a throwaway Rust parser instance.

        Returns the coalesced parser output along with any model-emitted tool
        call IDs keyed by tool index.
        """
        parser = self._new_parser()
        output = _rust_tool_parser_module().ToolParserOutput()
        try:
            parser.parse_into(model_output, output)
            # finish() clears parser state, so snapshot model-emitted IDs first.
            tool_call_ids = {
                call.tool_index: tool_call_id
                for call in output.calls
                if (tool_call_id := parser.tool_call_id(call.tool_index)) is not None
            }
            output.append(parser.finish())
        except Exception:
            logger.exception(
                "Error parsing %s tool call output.", self.rust_parser_name
            )
            return None
        return output.coalesce_calls(), tool_call_ids

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from complete model output (non-streaming)."""
        if (
            self.tool_call_start_token is not None
            and self.tool_call_start_token not in model_output
        ):
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        parse_result = self._parse_complete(model_output)
        if parse_result is None:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )
        parsed, tool_call_ids = parse_result

        tool_calls: list[ToolCall] = []
        self.prev_tool_call_arr.clear()
        for parsed_tool_call in parsed.calls:
            name = parsed_tool_call.name
            arguments = parsed_tool_call.arguments or "{}"
            if name is None:
                continue
            tool_calls.append(
                ToolCall(
                    id=tool_call_ids.get(parsed_tool_call.tool_index)
                    or make_tool_call_id(),
                    type="function",
                    function=FunctionCall(name=name, arguments=arguments),
                )
            )
            self.prev_tool_call_arr.append({"name": name, "arguments": arguments})

        if not tool_calls:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        content = parsed.normal_text or None
        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=tool_calls,
            content=content,
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],  # pylint: disable=unused-argument
        current_token_ids: Sequence[int],  # pylint: disable=unused-argument
        delta_token_ids: Sequence[int],  # pylint: disable=unused-argument
        request: ChatCompletionRequest,  # pylint: disable=unused-argument
    ) -> DeltaMessage | None:
        """Extract tool calls from streaming model output.

        The Rust parser owns the incremental buffer, so this adapter feeds only
        the newest text delta and lets the serving layer handle final empty
        chunks.
        """
        # TODO: Add a final-chunk hook if streaming needs to call Rust finish().
        if not previous_text:
            self._reset_streaming_state()

        if self._error is not None:
            return None

        parser_output = _rust_tool_parser_module().ToolParserOutput()
        try:
            self._get_parser().parse_into(delta_text, parser_output)
        except Exception as error:
            self._error = error
            logger.exception(
                "Error parsing %s streaming tool call output.",
                self.rust_parser_name,
            )

        delta_message = self._delta_message_from_parser_output(parser_output)
        if delta_message is not None:
            return delta_message

        return None
