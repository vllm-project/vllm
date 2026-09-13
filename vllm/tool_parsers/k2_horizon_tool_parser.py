# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence
from typing import Any

import regex as re
from openai.types.responses import ToolChoiceFunction

from vllm.entrypoints.generate.base.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser
from vllm.tool_parsers.utils import (
    coerce_to_schema_type,
    extract_types_from_schema,
    find_tool_name,
    find_tool_properties,
    partial_tag_overlap,
)

logger = init_logger(__name__)

ToolParserRequest = ChatCompletionRequest | ResponsesRequest


class K2HorizonToolParser(ToolParser):
    """Parser for K2 Horizon's IFM tool-call formats."""

    supports_required_and_named = False
    engine_based_streaming = True

    _SUPPORTED_TOOL_FORMATS = frozenset({"json", "xml", "xml_typed"})

    _TOOL_CALLS_START = "<ifm|tool_calls>"
    _TOOL_CALLS_END = "</ifm|tool_calls>"
    _TOOL_CALL_START = "<ifm|tool_call>"
    _TOOL_CALL_END = "</ifm|tool_call>"
    _ARG_KEY_START = "<ifm|arg_key>"
    _ARG_KEY_END = "</ifm|arg_key>"
    _ARG_TYPE_START = "<ifm|arg_type>"
    _ARG_TYPE_END = "</ifm|arg_type>"
    _ARG_VALUE_START = "<ifm|arg_value>"
    _ARG_VALUE_END = "</ifm|arg_value>"

    _TOOL_CALL_RE = re.compile(
        re.escape(_TOOL_CALL_START) + r"(.*?)" + re.escape(_TOOL_CALL_END),
        re.DOTALL,
    )
    _ARG_RE = re.compile(
        re.escape(_ARG_KEY_START)
        + r"(.*?)"
        + re.escape(_ARG_KEY_END)
        + r"\s*(?:"
        + re.escape(_ARG_TYPE_START)
        + r"(.*?)"
        + re.escape(_ARG_TYPE_END)
        + r"\s*)?"
        + re.escape(_ARG_VALUE_START)
        + r"(.*?)"
        + re.escape(_ARG_VALUE_END),
        re.DOTALL,
    )
    _ARG_MARKERS = (
        _ARG_KEY_START,
        _ARG_KEY_END,
        _ARG_TYPE_START,
        _ARG_TYPE_END,
        _ARG_VALUE_START,
        _ARG_VALUE_END,
    )

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
    ) -> None:
        super().__init__(tokenizer, tools)
        self.tool_format = "xml"
        self._stream_text = ""
        self._stream_cursor = 0
        self._stream_group_start: int | None = None
        self._stream_resolved = False
        self._pending_content = ""
        self._content_seen = False

    @classmethod
    def _request_tool_format(cls, request: ToolParserRequest) -> str:
        chat_template_kwargs = request.chat_template_kwargs or {}
        tool_format = chat_template_kwargs.get("tool_call_format", "xml")
        if (
            not isinstance(tool_format, str)
            or tool_format not in cls._SUPPORTED_TOOL_FORMATS
        ):
            supported = ", ".join(sorted(cls._SUPPORTED_TOOL_FORMATS))
            raise ValueError(
                f"Unsupported tool_call_format {tool_format!r}. "
                f"Supported values: {supported}."
            )
        return tool_format

    def adjust_request(self, request: ToolParserRequest) -> ToolParserRequest:
        self._request_tool_format(request)
        return request

    @staticmethod
    def _content_or_none(content: str) -> str | None:
        return content if content.strip() else None

    @classmethod
    def _no_tool_calls(cls, content: str) -> ExtractedToolCallInformation:
        return ExtractedToolCallInformation(
            tools_called=False,
            tool_calls=[],
            content=cls._content_or_none(content),
        )

    @staticmethod
    def _make_tool_call(name: str, arguments: dict[str, Any]) -> ToolCall:
        return ToolCall(
            type="function",
            function=FunctionCall(
                name=name,
                arguments=json.dumps(
                    arguments,
                    ensure_ascii=False,
                    allow_nan=False,
                ),
            ),
        )

    @staticmethod
    def _base_arg_type(arg_type: str | None) -> str | None:
        if not arg_type:
            return None
        return arg_type.split("[", 1)[0].strip().lower()

    @staticmethod
    def _coerce_json_value(value: Any, schema: Any) -> Any:
        if not isinstance(schema, dict):
            return value
        raw_value = (
            value
            if isinstance(value, str)
            else json.dumps(value, ensure_ascii=False, allow_nan=False)
        )
        return coerce_to_schema_type(raw_value, extract_types_from_schema(schema))

    @classmethod
    def _coerce_xml_value(
        cls,
        value: str,
        schema: Any,
        explicit_type: str | None,
    ) -> Any:
        explicit_base = cls._base_arg_type(explicit_type)
        target_types = extract_types_from_schema(schema)
        if isinstance(schema, dict):
            if explicit_base in target_types and len(set(target_types)) > 1:
                target_types = [explicit_base]
        elif explicit_base == "any":
            return value
        elif explicit_base:
            target_types = [explicit_base]

        raw_value = value if "string" in target_types else value.strip()
        return coerce_to_schema_type(raw_value, target_types)

    @staticmethod
    def _named_tool(request: ToolParserRequest) -> str | None:
        tool_choice = request.tool_choice
        if isinstance(tool_choice, ChatCompletionNamedToolChoiceParam):
            return tool_choice.function.name
        if isinstance(tool_choice, ToolChoiceFunction):
            return tool_choice.name
        return None

    @classmethod
    def _validate_tool_name(
        cls,
        name: Any,
        request: ToolParserRequest,
    ) -> str:
        if (
            not isinstance(name, str)
            or not name
            or any(char.isspace() for char in name)
        ):
            raise ValueError("IFM tool call has an invalid function name.")
        if request.tools and not find_tool_name(request.tools, name):
            raise ValueError(f"Unknown IFM tool name: {name}.")
        named_tool = cls._named_tool(request)
        if named_tool is not None and name != named_tool:
            raise ValueError(f"Unexpected IFM tool name: {name}.")
        return name

    @classmethod
    def _parse_json_call(
        cls,
        body: str,
        request: ToolParserRequest,
    ) -> tuple[str, dict[str, Any]]:
        raw_call = json.loads(body.strip())
        if not isinstance(raw_call, dict):
            raise ValueError("IFM JSON tool call must be an object.")

        name = cls._validate_tool_name(raw_call.get("name"), request)
        arguments = raw_call.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ValueError("IFM JSON tool-call arguments must be an object.")

        properties = find_tool_properties(request.tools, name)
        arguments = {
            arg_name: cls._coerce_json_value(arg_value, properties.get(arg_name))
            for arg_name, arg_value in arguments.items()
        }
        return name, arguments

    @classmethod
    def _parse_xml_call(
        cls,
        body: str,
        request: ToolParserRequest,
        tool_format: str,
    ) -> tuple[str, dict[str, Any]]:
        first_arg = body.find(cls._ARG_KEY_START)
        if first_arg == -1:
            if any(marker in body for marker in cls._ARG_MARKERS):
                raise ValueError("Malformed IFM XML argument tags.")
            name = cls._validate_tool_name(body.strip(), request)
            return name, {}

        name = cls._validate_tool_name(body[:first_arg].strip(), request)
        properties = find_tool_properties(request.tools, name)
        arguments: dict[str, Any] = {}
        position = first_arg
        for match in cls._ARG_RE.finditer(body, first_arg):
            if body[position : match.start()].strip():
                raise ValueError("Malformed IFM XML argument tags.")

            arg_name = match.group(1).strip()
            if not arg_name:
                raise ValueError("IFM XML argument is missing a name.")
            if arg_name in arguments:
                raise ValueError(f"Duplicate IFM XML argument: {arg_name}.")

            explicit_type = match.group(2)
            if explicit_type is not None:
                explicit_type = explicit_type.strip()
            if tool_format == "xml" and explicit_type is not None:
                raise ValueError("Unexpected type tag in IFM XML tool call.")
            if tool_format == "xml_typed" and not explicit_type:
                raise ValueError("IFM typed XML argument is missing a type.")
            arguments[arg_name] = cls._coerce_xml_value(
                match.group(3),
                properties.get(arg_name),
                explicit_type,
            )
            position = match.end()

        if position == first_arg or body[position:].strip():
            raise ValueError("Malformed IFM XML argument tags.")
        return name, arguments

    @classmethod
    def _parse_call(
        cls,
        body: str,
        request: ToolParserRequest,
        tool_format: str,
    ) -> tuple[str, dict[str, Any]]:
        if tool_format == "json":
            return cls._parse_json_call(body, request)
        return cls._parse_xml_call(body, request, tool_format)

    def extract_tool_calls(
        self,
        model_output: str,
        request: ToolParserRequest,
    ) -> ExtractedToolCallInformation:
        tool_format = self._request_tool_format(request)
        if request.tool_choice == "none":
            return self._no_tool_calls(model_output)

        group_start = model_output.find(self._TOOL_CALLS_START)
        if group_start == -1:
            return self._no_tool_calls(model_output)

        body_start = group_start + len(self._TOOL_CALLS_START)
        group_end = model_output.find(self._TOOL_CALLS_END, body_start)
        if group_end == -1:
            return self._no_tool_calls(model_output)

        group_body = model_output[body_start:group_end]
        matches = list(self._TOOL_CALL_RE.finditer(group_body))
        if not matches or self._TOOL_CALL_RE.sub("", group_body).strip():
            logger.warning("Malformed K2 Horizon IFM tool-call group.")
            return self._no_tool_calls(model_output)

        try:
            tool_calls = [
                self._make_tool_call(
                    *self._parse_call(match.group(1), request, tool_format)
                )
                for match in matches
            ]
        except (json.JSONDecodeError, TypeError, ValueError):
            logger.warning("Failed to parse K2 Horizon IFM tool calls.", exc_info=True)
            return self._no_tool_calls(model_output)

        suffix = model_output[group_end + len(self._TOOL_CALLS_END) :]
        content = model_output[:group_start] + suffix
        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=tool_calls,
            content=self._content_or_none(content),
        )

    def _emit_content(self, addition: str, *, final: bool = False) -> str | None:
        if addition:
            if self._content_seen:
                return addition
            self._pending_content += addition
            if self._pending_content.strip():
                self._content_seen = True
                content = self._pending_content
                self._pending_content = ""
                return content
        if final:
            self._pending_content = ""
        return None

    def _streaming_tool_calls(
        self,
        tool_calls: list[ToolCall],
    ) -> list[DeltaToolCall]:
        deltas: list[DeltaToolCall] = []
        for index, tool_call in enumerate(tool_calls):
            arguments_json = tool_call.function.arguments
            arguments = json.loads(arguments_json)
            self.prev_tool_call_arr.append(
                {"name": tool_call.function.name, "arguments": arguments}
            )
            self.streamed_args_for_tool.append(arguments_json)
            deltas.append(
                DeltaToolCall(
                    index=index,
                    id=tool_call.id,
                    type="function",
                    function=DeltaFunctionCall(
                        name=tool_call.function.name,
                        arguments=arguments_json,
                    ),
                )
            )
        return deltas

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ToolParserRequest,
    ) -> DeltaMessage | None:
        del (
            previous_text,
            current_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )

        self._request_tool_format(request)
        if request.tool_choice == "none":
            emitted_content = self._emit_content(delta_text)
            return (
                DeltaMessage(content=emitted_content)
                if emitted_content is not None
                else None
            )

        self._stream_text += delta_text
        if self._stream_resolved:
            emitted_content = self._emit_content(delta_text)
            return (
                DeltaMessage(content=emitted_content)
                if emitted_content is not None
                else None
            )

        content: str | None = None
        if self._stream_group_start is None:
            group_start = self._stream_text.find(
                self._TOOL_CALLS_START, self._stream_cursor
            )
            if group_start == -1:
                pending = self._stream_text[self._stream_cursor :]
                overlap = partial_tag_overlap(pending, self._TOOL_CALLS_START)
                sendable_end = len(self._stream_text) - overlap
                content = self._emit_content(
                    self._stream_text[self._stream_cursor : sendable_end]
                )
                self._stream_cursor = sendable_end
                return DeltaMessage(content=content) if content is not None else None

            self._stream_group_start = group_start
            content = self._emit_content(
                self._stream_text[self._stream_cursor : group_start]
            )
            self._stream_cursor = group_start

        group_start = self._stream_group_start
        body_start = group_start + len(self._TOOL_CALLS_START)
        group_end = self._stream_text.find(self._TOOL_CALLS_END, body_start)
        if group_end == -1:
            return DeltaMessage(content=content) if content is not None else None

        group_end += len(self._TOOL_CALLS_END)
        group_text = self._stream_text[group_start:group_end]
        parsed = self.extract_tool_calls(group_text, request)
        self._stream_resolved = True

        if not parsed.tools_called:
            failed_content = self._emit_content(
                self._stream_text[self._stream_cursor :]
            )
            self._stream_cursor = len(self._stream_text)
            combined = (content or "") + (failed_content or "")
            return DeltaMessage(content=combined) if combined else None

        tool_calls = self._streaming_tool_calls(parsed.tool_calls)
        suffix = self._emit_content(self._stream_text[group_end:])
        self._stream_cursor = len(self._stream_text)
        combined = (content or "") + (suffix or "")
        return DeltaMessage(content=combined or None, tool_calls=tool_calls)

    def finish_streaming(self) -> DeltaMessage | None:
        if not self._stream_resolved:
            content = self._emit_content(
                self._stream_text[self._stream_cursor :], final=True
            )
            self._stream_cursor = len(self._stream_text)
            self._stream_resolved = True
        else:
            content = self._emit_content("", final=True)
        return DeltaMessage(content=content) if content is not None else None
