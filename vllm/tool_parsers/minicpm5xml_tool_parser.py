# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import html
import json
from collections.abc import Sequence
from typing import Any

import regex as re
from openai.types.responses import FunctionTool

from vllm.entrypoints.generate.base.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.renderers.chat_utils import make_tool_call_id
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    Tool,
    ToolParser,
)
from vllm.tool_parsers.schema_types import (
    ParameterTypes,
    convert_value,
    parameter_types,
)
from vllm.tool_parsers.utils import (
    consume_space,
    get_function_tools,
    partial_tag_overlap,
    safe_literal_eval,
)

_FUNCTION_START = "<function"
_FUNCTION_END = "</function>"
_FUNCTION_NAME = re.compile(r"""<function\s*name=['"]([^'"]+)['"][^<>]*>""")
_PARAM = re.compile(r"""<param(?:\s*name=['"]([^'"]+)['"]|(?=\s|>))[^<>]*>""")
_PARAM_END = "</param>"
_CDATA_START = "<![CDATA["
_CDATA_END = "]]>"
_CDATA_PARAM_START = re.compile(_PARAM.pattern + r"\s*" + re.escape(_CDATA_START))
_XML_ENTITY = re.compile(r"&(?:amp|lt|gt|quot|apos|#[0-9]{1,7}|#x[0-9a-fA-F]{1,6});")

# SentencePiece/GPT-style decoders may emit U+0120 (Ġ) / U+010A (Ċ).
_TOKENIZER_SPACE = "\u0120"
_TOKENIZER_NEWLINE = "\u010a"


def _normalize_model_output(text: str) -> str:
    # SentencePiece may emit U+0120 / U+010A. Some checkpoints also collapse
    # the attribute onto the tag: <functionname="foo"> / <paramname="bar">.
    return (
        text.replace(_TOKENIZER_SPACE, " ")
        .replace(_TOKENIZER_NEWLINE, "\n")
        .replace("<functionname=", "<function name=")
        .replace("<paramname=", "<param name=")
    )


def _strip_thinking_content(text: str) -> str:
    """Return only user-visible content after MiniCPM5 thinking, when present."""
    if "</think>" not in text:
        return text
    visible = text.rsplit("</think>", 1)[-1].lstrip()
    return visible if visible else text


def _parse_arguments(json_value: str) -> tuple[Any, bool]:
    try:
        try:
            parsed_value = json.loads(json_value)
        except json.JSONDecodeError:
            parsed_value = safe_literal_eval(json_value.strip())
        return parsed_value, True
    except (ValueError, SyntaxError, TypeError, RecursionError):
        return json_value, False


def _coerce_argument_value(
    func_name: str,
    arg_key: str,
    value: Any,
    param_types: dict[str, ParameterTypes],
) -> Any:
    props, extra = param_types.get(func_name, ({}, None))
    allowed = props.get(arg_key, extra)
    if isinstance(value, str):
        return convert_value(value, allowed, lambda v: _parse_arguments(v)[0])
    # Already decoded (wrapped arguments): serialize it where a string is meant.
    text = json.dumps(value, ensure_ascii=False)
    return convert_value(text, allowed, lambda _: value)


def _add_argument(
    func_name: str,
    key: str,
    val_text: str,
    arguments: dict[str, Any],
    seen_keys: set[str],
    allowed_props: set[str],
    param_types: dict[str, ParameterTypes],
) -> None:
    # Accept wrappers emitted by existing MiniCPM5 checkpoints.
    if (
        key in {"properties", "arguments"}
        and allowed_props
        and key not in allowed_props
    ):
        parsed_val, parsed_ok = _parse_arguments(val_text)
        if parsed_ok and isinstance(parsed_val, dict):
            for wrapped_key, wrapped_value in parsed_val.items():
                if wrapped_key in allowed_props and wrapped_key not in seen_keys:
                    seen_keys.add(wrapped_key)
                    arguments[wrapped_key] = _coerce_argument_value(
                        func_name, wrapped_key, wrapped_value, param_types
                    )
        return

    if key in seen_keys or (allowed_props and key not in allowed_props):
        return
    seen_keys.add(key)
    arguments[key] = _coerce_argument_value(func_name, key, val_text, param_types)


def _normalize_alias_tool_call(
    func_name: str,
    arguments: dict[str, Any],
    tool_names: set[str],
) -> tuple[str, dict[str, Any]]:
    """Map common MiniCPM5 alias tool names to the exposed tool schema."""
    if func_name in tool_names:
        return func_name, arguments

    if func_name == "get_details_by_phone" and "get_customer_by_phone" in tool_names:
        phone_number = arguments.get("phone_number") or arguments.get("phone")
        if phone_number is not None:
            return "get_customer_by_phone", {"phone_number": phone_number}

    if func_name == "get_details_by_name" and "get_customer_by_name" in tool_names:
        full_name = arguments.get("full_name") or arguments.get("name")
        dob = arguments.get("date_of_birth") or arguments.get("dob")
        mapped = {}
        if full_name is not None:
            mapped["full_name"] = full_name
        if dob is not None:
            mapped["date_of_birth"] = dob
            mapped["dob"] = dob
        if mapped:
            return "get_customer_by_name", mapped

    if (
        func_name
        in {
            "get_line_details",
            "get_line_status",
            "get_roaming_status",
        }
        and "get_details_by_id" in tool_names
    ):
        detail_id = arguments.get("line_id") or arguments.get("id")
        if detail_id is not None:
            return "get_details_by_id", {"id": detail_id}

    if func_name == "get_plan_details" and "get_details_by_id" in tool_names:
        plan_id = arguments.get("plan_id") or arguments.get("id")
        if plan_id is not None:
            return "get_details_by_id", {"id": plan_id}

    if func_name == "enable_roaming" and "toggle_roaming" in tool_names:
        line_id = arguments.get("line_id") or arguments.get("id")
        if line_id is not None:
            return "toggle_roaming", {"line_id": line_id, "enabled": True}

    if func_name == "disable_roaming" and "toggle_roaming" in tool_names:
        line_id = arguments.get("line_id") or arguments.get("id")
        if line_id is not None:
            return "toggle_roaming", {"line_id": line_id, "enabled": False}

    if (
        func_name in {"add_refueled_data", "add_data_to_line"}
        and "refuel_data" in tool_names
    ):
        line_id = arguments.get("line_id") or arguments.get("id")
        amount = (
            arguments.get("amount_gb")
            or arguments.get("gb")
            or arguments.get("gb_amount")
            or arguments.get("amount")
        )
        mapped = {}
        if "customer_id" in arguments:
            mapped["customer_id"] = arguments["customer_id"]
        if line_id is not None:
            mapped["line_id"] = line_id
        if amount is not None:
            mapped["amount_gb"] = amount
        if mapped:
            return "refuel_data", mapped

    return func_name, arguments


def _build_tool_maps(
    tools: Sequence[ChatCompletionToolsParam | FunctionTool],
) -> tuple[set[str], dict[str, set[str]], dict[str, set[str]]]:
    name_to_allowed_props: dict[str, set[str]] = {}
    name_to_required: dict[str, set[str]] = {}

    for tool in tools:
        function = tool.function if isinstance(tool, ChatCompletionToolsParam) else tool
        params = function.parameters or {}
        properties = params.get("properties")
        required = params.get("required")
        name_to_allowed_props[function.name] = (
            set(properties) if isinstance(properties, dict) else set()
        )
        name_to_required[function.name] = (
            {key for key in required if isinstance(key, str)}
            if isinstance(required, list)
            else set()
        )

    return set(name_to_allowed_props), name_to_allowed_props, name_to_required


def _find_function_end(text: str, start: int, *, final: bool) -> int:
    """Find the close tag outside CDATA, including across streaming chunks."""
    while (end := text.find(_FUNCTION_END, start)) != -1:
        cdata = text.find(_CDATA_START, start, end)
        if cdata == -1:
            return end
        cdata_end = text.find(_CDATA_END, cdata + len(_CDATA_START))
        if cdata_end == -1:
            return end if final else -1

        # A later function with its own CDATA parameter owns this terminator.
        after_close = end + len(_FUNCTION_END)
        recovery_end = None
        for opener in _FUNCTION_NAME.finditer(text, after_close, cdata_end):
            if recovery_end is None:
                recovery_end = text.rfind(_FUNCTION_END, end, opener.start())
            next_end = text.find(_FUNCTION_END, opener.end(), cdata_end)
            next_end = cdata_end if next_end == -1 else next_end
            if _CDATA_PARAM_START.search(text, opener.end(), next_end):
                return recovery_end

        start = cdata_end + len(_CDATA_END)
    return -1


def _read_param(text: str, start: int, end: int) -> tuple[str, int]:
    cdata = consume_space(start, text)
    if text.startswith(_CDATA_START, cdata, end):
        start = cdata + len(_CDATA_START)
        cdata_end = text.find(_CDATA_END, start, end)
        if cdata_end != -1:
            after = cdata_end + len(_CDATA_END)
            if text.startswith(_PARAM_END, after):
                after += len(_PARAM_END)
            return text[start:cdata_end], after

    close = text.find(_PARAM_END, start, end)
    next_param = _PARAM.search(text, start, end if close == -1 else close)
    if next_param is not None:
        stop = next_param.start()
        value, after = text[start:stop].strip(), stop
    elif close != -1:
        value, after = text[start:close], close + len(_PARAM_END)
    else:
        value, after = text[start:end].strip(), end
    return _XML_ENTITY.sub(lambda match: html.unescape(match[0]), value), after


class MiniCPM5XMLToolParser(ToolParser):
    """Read MiniCPM XML calls with CDATA-aware recovery of model output."""

    engine_based_streaming = True

    def __init__(
        self, tokenizer: TokenizerLike, tools: list[Tool] | None = None
    ) -> None:
        super().__init__(tokenizer, tools)
        self._remainder = ""
        self._request: ChatCompletionRequest | ResponsesRequest | None = None

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            # Keep the XML tags when the tokenizer marks them as special.
            request.skip_special_tokens = False
        return request

    def _scan(
        self,
        text: str,
        request: ChatCompletionRequest | ResponsesRequest,
        *,
        final: bool,
    ) -> tuple[str, list[FunctionCall], int]:
        tool_maps = None
        param_types = {}
        content: list[str] = []
        calls: list[FunctionCall] = []
        pos = 0
        while pos < len(text):
            start = text.find(_FUNCTION_START, pos)
            if start == -1:
                overlap = 0 if final else partial_tag_overlap(text, _FUNCTION_START)
                stop = len(text) - overlap
                content.append(text[pos:stop])
                pos = stop
                break
            content.append(text[pos:start])
            opener = _FUNCTION_NAME.match(text, start)
            if opener is None:
                tag_end = text.find(">", start)
                if tag_end == -1 and not final:
                    return "".join(content), calls, start
                # Advance past the literal marker, leaving later calls visible.
                pos = start + len(_FUNCTION_START)
                content.append(text[start:pos])
                continue

            body_end = _find_function_end(text, opener.end(), final=final)
            if body_end == -1:
                if final:
                    content.append(text[start:])
                    pos = len(text)
                else:
                    pos = start
                break
            if tool_maps is None:
                tools = get_function_tools(request.tools or ())
                tool_maps = _build_tool_maps(tools)
                param_types = parameter_types(tools)
            tool_names, allowed_props, required = tool_maps

            pos = body_end + len(_FUNCTION_END)
            name = opener[1].strip()
            args: dict[str, Any] = {}
            seen: set[str] = set()
            i = opener.end()
            while param := _PARAM.search(text, i, body_end):
                value, i = _read_param(text, param.end(), body_end)
                if param[1] is not None:
                    _add_argument(
                        name,
                        param[1].strip(),
                        value,
                        args,
                        seen,
                        allowed_props.get(name, set()),
                        param_types,
                    )

            name, args = _normalize_alias_tool_call(name, args, tool_names)
            if name in tool_names and required[name].issubset(args):
                calls.append(
                    FunctionCall(
                        name=name, arguments=json.dumps(args, ensure_ascii=False)
                    )
                )
            else:
                # Keep rejected blocks visible. Dropping them makes a failed
                # call disappear from both tool_calls and content.
                content.append(text[start:pos])
        return "".join(content), calls, pos

    def extract_tool_calls(
        self, model_output: str, request: ChatCompletionRequest | ResponsesRequest
    ) -> ExtractedToolCallInformation:
        text = _normalize_model_output(model_output)
        content, calls, _ = self._scan(text, request, final=True)
        if _FUNCTION_START in text:
            content = content.strip()
        return ExtractedToolCallInformation(
            tools_called=bool(calls),
            tool_calls=[
                ToolCall(id=make_tool_call_id(), type="function", function=call)
                for call in calls
            ],
            content=None if calls else _strip_thinking_content(content),
        )

    def _make_delta(
        self, content: str, calls: list[FunctionCall]
    ) -> DeltaMessage | None:
        deltas = []
        for call in calls:
            self.current_tool_id += 1
            self.prev_tool_call_arr.append(
                {"name": call.name, "arguments": call.arguments}
            )
            self.streamed_args_for_tool.append(call.arguments)
            deltas.append(
                DeltaToolCall(
                    index=self.current_tool_id,
                    id=make_tool_call_id(),
                    type="function",
                    function=DeltaFunctionCall(
                        name=call.name, arguments=call.arguments
                    ).model_dump(exclude_none=True),
                )
            )
        if content or deltas:
            return DeltaMessage(content=content or None, tool_calls=deltas)
        return None

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
        if request.tool_choice == "none" or not request.tools:
            return DeltaMessage(content=delta_text) if delta_text else None

        text = _normalize_model_output(self._remainder + delta_text)
        content, calls, consumed = self._scan(text, request, final=False)
        self._remainder = text[consumed:]
        self._request = request
        return self._make_delta(content, calls)

    def finish_streaming(self) -> DeltaMessage | None:
        if not self._remainder or self._request is None:
            return None
        content, calls, _ = self._scan(self._remainder, self._request, final=True)
        self._remainder = ""
        return self._make_delta(content, calls)
