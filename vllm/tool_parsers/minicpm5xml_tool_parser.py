# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import re
from collections.abc import Sequence
from typing import Any

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
from vllm.logger import init_logger
from vllm.renderers.chat_utils import make_tool_call_id
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    Tool,
    ToolParser,
)
from vllm.tool_parsers.schema_types import (
    ParameterTypes,
    ParameterTypesCache,
    convert_value,
)
from vllm.tool_parsers.utils import partial_tag_overlap, safe_literal_eval

logger = init_logger(__name__)

_FUNCTION_START = "<function"
_FUNCTION_END = "</function>"
_FUNCTION_NAME = re.compile(r"""<function\s+name=['"]([^'"]+)['"][^<>]*>""")
_PARAM = re.compile(r"""<param(?=\s|>)(?:\s+name=['"]([^'"]+)['"])?[^<>]*>""")
_PARAM_END = "</param>"
_CDATA_START = "<![CDATA["
_CDATA_END = "]]>"

# SentencePiece/GPT-style decoders may emit U+0120 (Ġ) / U+010A (Ċ).
_TOKENIZER_SPACE = "\u0120"
_TOKENIZER_NEWLINE = "\u010a"


def _normalize_model_output(text: str) -> str:
    if (
        _TOKENIZER_SPACE not in text
        and _TOKENIZER_NEWLINE not in text
        and "<functionname=" not in text
        and "<paramname=" not in text
    ):
        return text

    normalized = text.replace(_TOKENIZER_SPACE, " ")
    normalized = normalized.replace(_TOKENIZER_NEWLINE, "\n")
    # Some model outputs collapse tag names and attributes, e.g.
    # <functionname="foo"> or <paramname="bar">.
    normalized = normalized.replace("<functionname=", "<function name=")
    normalized = normalized.replace("<paramname=", "<param name=")
    return normalized


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
    except Exception:
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
    tools: list[ChatCompletionToolsParam] | None,
) -> tuple[set[str], dict[str, set[str]], dict[str, set[str]]]:
    name_to_allowed_props: dict[str, set[str]] = {}
    name_to_required: dict[str, set[str]] = {}

    for tool in tools or []:
        name = tool.function.name
        if not name:
            continue
        params = tool.function.parameters or {}
        props = params.get("properties", {}) if isinstance(params, dict) else {}
        name_to_allowed_props[name] = set(props.keys())
        required = params.get("required", []) if isinstance(params, dict) else []
        try:
            name_to_required[name] = set(required)
        except TypeError:
            logger.warning(
                "Failed to parse 'required' field for tool %s. "
                "It should be a list of strings. Got: %s",
                name,
                required,
            )
            name_to_required[name] = set()

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
        start = cdata_end + len(_CDATA_END)
    return -1


def _read_param(text: str, start: int, end: int) -> tuple[str, int]:
    if text.startswith(_CDATA_START, start):
        start += len(_CDATA_START)
        cdata_end = text.find(_CDATA_END, start, end)
        if cdata_end != -1:
            after = cdata_end + len(_CDATA_END)
            if text.startswith(_PARAM_END, after):
                after += len(_PARAM_END)
            return text[start:cdata_end], after

    close = text.find(_PARAM_END, start, end)
    next_param = _PARAM.search(text, start, end)
    if next_param and (close == -1 or next_param.start() < close):
        return text[start : next_param.start()], next_param.start()
    if close != -1:
        return text[start:close], close + len(_PARAM_END)
    return text[start:end], end


class MiniCPM5XMLToolParser(ToolParser):
    """Read MiniCPM XML calls with CDATA-aware recovery of model output."""

    engine_based_streaming = True

    def __init__(
        self, tokenizer: TokenizerLike, tools: list[Tool] | None = None
    ) -> None:
        super().__init__(tokenizer, tools)
        self._remainder = ""
        self._request: ChatCompletionRequest | None = None
        self._parameter_types_cache = ParameterTypesCache()

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            # Keep the XML tags when the tokenizer marks them as special.
            request.skip_special_tokens = False
        return request

    def _scan(
        self, text: str, request: ChatCompletionRequest, *, final: bool
    ) -> tuple[str, list[FunctionCall], int]:
        tool_names, allowed_props, required = _build_tool_maps(request.tools)
        param_types = self._parameter_types_cache.for_tools(request.tools)
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
            if name not in tool_names:
                content.append(text[start:pos])
            elif required[name].issubset(args):
                calls.append(
                    FunctionCall(
                        name=name, arguments=json.dumps(args, ensure_ascii=False)
                    )
                )
        return "".join(content), calls, pos

    def extract_tool_calls(
        self, model_output: str, request: ChatCompletionRequest
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
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        # Engine-based streaming supplies only the latest chunk.
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
