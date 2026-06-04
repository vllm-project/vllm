# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence
from contextlib import suppress
from typing import Any

from openai.types.responses.function_tool import FunctionTool

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
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
from vllm.tool_parsers._llm_nom.m3_text import M3TextParser
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser

logger = init_logger(__name__)


class MinimaxM3ToolParser(ToolParser):
    """Adapter from the vendored MiniMax M3 parser to vLLM ToolParser.

    The real M3 grammar lives in ``_llm_nom.m3_text.M3TextParser``. This
    class keeps only the vLLM-specific bridge work:
    - convert vLLM tool definitions into the function schema shape expected by
      the vendored parser;
    - translate parser ``content`` / ``tool_calls`` deltas into vLLM protocol
      objects; and
    - maintain vLLM streaming bookkeeping used by finish-reason handling.

    M3 is not M2 with renamed tags: it prefixes each structural tag with the
    MiniMax namespace marker, allows multiple ``<invoke>`` tags in one wrapper,
    and represents nested arguments with parameter-name XML tags.
    """

    # M3 emits its own XML-like tool-call format from the chat template. For
    # required/named tool_choice, do not let the serving layer force JSON guided
    # output; parse the M3 syntax through this parser instead.
    supports_required_and_named = False

    tool_call_start_token = "]<]minimax[>[<tool_call>"

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)
        self._parser: M3TextParser | None = None
        self._error: Exception | None = None
        self._tool_call_ids: dict[int, str] = {}

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
        """Adjust generation options for MiniMax M3 tool-call syntax.

        Required/named tool choice must skip ``super().adjust_request()``
        because the base implementation would install JSON structured output
        constraints. M3 needs to preserve and generate its namespace-tagged
        syntax, so we only ensure special-token text is not stripped.
        """
        if request.tools:
            tool_choice = getattr(request, "tool_choice", None)
            if tool_choice == "required" or isinstance(
                tool_choice, ChatCompletionNamedToolChoiceParam
            ):
                if hasattr(request, "skip_special_tokens"):
                    request.skip_special_tokens = False
                return request

        request = super().adjust_request(request)
        if (
            request.tools
            and getattr(request, "tool_choice", None) != "none"
            and hasattr(request, "skip_special_tokens")
        ):
            request.skip_special_tokens = False
        return request

    def _functions(self) -> dict[str, dict[str, Any]] | None:
        """Build the function map consumed by ``M3TextParser``."""
        if not self.tools:
            return None

        functions: dict[str, dict[str, Any]] = {}
        for tool in self.tools:
            if isinstance(tool, FunctionTool):
                name = tool.name
                parameters = tool.parameters
            elif isinstance(tool, ChatCompletionToolsParam):
                name = tool.function.name
                parameters = tool.function.parameters
            else:
                continue
            functions[name] = {"parameters": parameters}
        return functions

    def _new_parser(self) -> M3TextParser:
        """Create a fresh vendored parser with the current tool schemas."""
        return M3TextParser(
            with_reasoning=False,
            reasoning_prefix="",
            functions=self._functions(),
        )

    def _get_parser(self) -> M3TextParser:
        if self._parser is None:
            self._parser = self._new_parser()
        return self._parser

    def _reset_streaming_state(self) -> None:
        """Reset parser state for a new request on a reused parser instance."""
        self._parser = self._new_parser()
        self._error = None
        self._tool_call_ids.clear()
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
        """Mirror a vendored-parser delta into vLLM streaming bookkeeping.

        ``prev_tool_call_arr`` and ``streamed_args_for_tool`` are read later by
        the chat serving layer to decide the final ``tool_calls`` finish reason
        and to flush any remaining argument bytes.
        """
        tool_call_id = None
        self._ensure_tool_state(index)

        if name is not None:
            tool_call_id = make_tool_call_id()
            self._tool_call_ids[index] = tool_call_id
            self.prev_tool_call_arr[index] = {"name": name, "arguments": {}}
            self.current_tool_name_sent = True

        if arguments is not None:
            self.streamed_args_for_tool[index] += arguments
            with suppress(json.JSONDecodeError):
                self.prev_tool_call_arr[index]["arguments"] = json.loads(
                    self.streamed_args_for_tool[index]
                )
            self.current_tool_id = index

        return tool_call_id

    def _delta_message_from_parser_delta(
        self, parser_delta: dict[str, Any] | None
    ) -> DeltaMessage | None:
        """Translate one ``M3TextParser`` delta into a vLLM ``DeltaMessage``."""
        if parser_delta is None:
            return None

        normal_text = parser_delta.get("content") or None
        tool_calls: list[DeltaToolCall] = []
        for tool_call in parser_delta.get("tool_calls", []):
            func = tool_call.get("function", {})
            index = tool_call.get("index", 0)
            name = func.get("name")
            arguments = func.get("arguments")
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

    def _parse_complete(self, model_output: str) -> dict[str, Any] | None:
        """Parse complete model output with a throwaway parser instance."""
        parser = self._new_parser()
        try:
            parser.update(model_output)
        except Exception:
            logger.exception("Error parsing MiniMax M3 tool call output.")
            return None
        return parser.get_delta() or parser.get_final()

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from complete model output (non-streaming)."""
        if self.tool_call_start_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        parsed = self._parse_complete(model_output)
        if parsed is None:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        tool_calls: list[ToolCall] = []
        self.prev_tool_call_arr.clear()
        for parsed_tool_call in parsed.get("tool_calls", []):
            func = parsed_tool_call.get("function", {})
            name = func.get("name")
            arguments = func.get("arguments", "{}")
            if name is None:
                continue
            tool_calls.append(
                ToolCall(
                    type="function",
                    function=FunctionCall(name=name, arguments=arguments),
                )
            )
            try:
                args = json.loads(arguments)
            except json.JSONDecodeError:
                args = arguments
            self.prev_tool_call_arr.append({"name": name, "arguments": args})

        if not tool_calls:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        content = parsed.get("content") or None
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
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,  # pylint: disable=unused-argument
    ) -> DeltaMessage | None:
        """Extract tool calls from streaming model output.

        ``M3TextParser`` owns the incremental buffer, so this adapter feeds only
        the newest text delta. It returns an empty final content delta on EOS
        after a tool call so the serving layer reaches its finish-reason path.
        """
        if not previous_text:
            self._reset_streaming_state()

        if self._error is not None:
            return None

        try:
            self._get_parser().update(delta_text)
        except Exception as error:
            self._error = error
            logger.exception("Error parsing MiniMax M3 streaming tool call output.")

        parser_delta = self._get_parser().get_delta()
        delta_message = self._delta_message_from_parser_delta(parser_delta)
        if delta_message is not None:
            return delta_message

        if not delta_text and delta_token_ids and self.prev_tool_call_arr:
            return DeltaMessage(content="")
        return None
