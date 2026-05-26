# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import json
import re
from collections.abc import Sequence
from typing import Any

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
from vllm.tool_parsers.abstract_tool_parser import (
    Tool,
    ToolParser,
)
from vllm.tool_parsers.utils import partial_tag_overlap
from vllm.utils import random_uuid

logger = init_logger(__name__)

try:
    from lxml import etree as ET  # type: ignore

    _HAS_LXML = True
except Exception:  # pragma: no cover
    import xml.etree.ElementTree as ET  # type: ignore

    _HAS_LXML = False

_FUNC_NAME_V1_REGEX = re.compile(r"<function\s+name=['\"]([^'\"]+)['\"][^>]*>")
_PARAM_WITH_NAME_REGEX = re.compile(
    r"<param\s+name=['\"]([^'\"]+)['\"]>([\s\S]*?)</param>", re.DOTALL
)
_PARAM_MISSING_NAME_REGEX = re.compile(r"<param(?![^>]*\bname=)[^>]*>", re.DOTALL)
_FUNC_BLOCK_REGEX = re.compile(r"<function.*?</function>", re.DOTALL)

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


def _streaming_args_snapshot(args_json: str, *, is_complete: bool) -> str:
    """Return the streamed arguments prefix, omitting the closing brace until done."""
    if is_complete or not args_json.endswith("}"):
        return args_json
    return args_json[:-1]


def _streaming_args_diff(
    prev_args: str, args_json: str, *, is_complete: bool
) -> str | None:
    """Compute the next arguments fragment for OpenAI-style streaming accumulation."""
    if prev_args == "{}":
        prev_args = ""

    target = _streaming_args_snapshot(args_json, is_complete=is_complete)
    if not prev_args:
        return target or None
    if target == prev_args:
        return None
    if target.startswith(prev_args):
        return target[len(prev_args) :] or None

    # Recover from a previously closed partial JSON snapshot.
    if prev_args.endswith("}"):
        prev_open = prev_args[:-1]
        if target.startswith(prev_open):
            return target[len(prev_open) :] or None

    return None


def _parse_arguments(json_value: str) -> tuple[Any, bool]:
    try:
        try:
            parsed_value = json.loads(json_value)
        except json.JSONDecodeError:
            parsed_value = ast.literal_eval(json_value)
        return parsed_value, True
    except Exception:
        return json_value, False


def _get_argument_type(
    func_name: str,
    arg_key: str,
    name_to_tool: dict[str, ChatCompletionToolsParam],
) -> str | None:
    tool = name_to_tool.get(func_name)
    if tool is None or tool.function.parameters is None:
        return None
    props = tool.function.parameters.get("properties", {})
    if arg_key not in props:
        return None
    return props[arg_key].get("type")


def _coerce_argument_value(
    func_name: str,
    arg_key: str,
    value: Any,
    name_to_tool: dict[str, ChatCompletionToolsParam],
) -> Any:
    arg_type = _get_argument_type(func_name, arg_key, name_to_tool)
    if arg_type == "string":
        return (
            value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        )
    if isinstance(value, str):
        parsed_val, _ = _parse_arguments(value)
        return parsed_val
    return value


def _add_argument(
    func_name: str,
    key: str,
    val_text: str,
    arguments: dict[str, Any],
    seen_keys: set[str],
    allowed_props: set[str],
    name_to_tool: dict[str, ChatCompletionToolsParam],
) -> bool:
    """Add one argument; return False if it violates the tool schema."""
    # MiniCPM5 can emit OpenAI-style wrappers, e.g.
    # <param name="properties">{'id': '...'}</param>. Keep known fields and
    # ignore extra fields so valid tool calls do not fall back to long text.
    if (
        key in {"properties", "arguments"}
        and allowed_props
        and key not in allowed_props
    ):
        parsed_val, parsed_ok = _parse_arguments(val_text)
        if not parsed_ok or not isinstance(parsed_val, dict):
            return False
        added = False
        for wrapped_key, wrapped_value in parsed_val.items():
            if wrapped_key not in allowed_props:
                continue
            if wrapped_key in seen_keys:
                return False
            seen_keys.add(wrapped_key)
            arguments[wrapped_key] = _coerce_argument_value(
                func_name, wrapped_key, wrapped_value, name_to_tool
            )
            added = True
        return added

    if allowed_props and key not in allowed_props:
        return True
    if key in seen_keys:
        return False
    seen_keys.add(key)
    arguments[key] = _coerce_argument_value(func_name, key, val_text, name_to_tool)
    return True


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
) -> tuple[
    set[str],
    dict[str, set[str]],
    dict[str, set[str]],
    dict[str, ChatCompletionToolsParam],
]:
    name_to_tool: dict[str, ChatCompletionToolsParam] = {}
    name_to_allowed_props: dict[str, set[str]] = {}
    name_to_required: dict[str, set[str]] = {}

    for tool in tools or []:
        name = tool.function.name
        if not name:
            continue
        name_to_tool[name] = tool
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

    return (
        set(name_to_tool.keys()),
        name_to_allowed_props,
        name_to_required,
        name_to_tool,
    )


def _parse_function_block(
    block: str,
    tool_names: set[str],
    name_to_allowed_props: dict[str, set[str]],
    name_to_required: dict[str, set[str]],
    name_to_tool: dict[str, ChatCompletionToolsParam],
) -> dict[str, Any] | None:
    """Return {name, parameters} if block is valid, else None."""
    func_name: str | None = None
    arguments: dict[str, Any] = {}
    parsed_ok = False
    param_invalid = False

    try:
        if _HAS_LXML:
            try:
                parser = ET.XMLParser(**{"strip_cdata": False})  # type: ignore[call-arg]
            except TypeError:
                parser = ET.XMLParser()
            root = ET.fromstring(block, parser=parser)
        else:
            root = ET.fromstring(block)

        if root.tag == "function":
            func_node = root
        else:
            func_node = root.find("function") if hasattr(root, "find") else None

        if func_node is not None:
            func_name = (func_node.attrib.get("name") or "").strip()

        args_node = func_node.find("arguments") if func_node is not None else None
        param_nodes: list[Any] = []
        if func_node is not None:
            param_nodes = list(func_node.findall("param"))
            if args_node is not None and not param_nodes:
                param_nodes = list(args_node.findall("param"))

        if func_node is not None:
            seen_keys: set[str] = set()
            allowed_props = name_to_allowed_props.get(func_name or "", set())
            has_invalid_param = False
            for param in param_nodes:
                key = param.attrib.get("name")
                if not key:
                    has_invalid_param = True
                    break
                val_text = (param.text or "").strip()
                if not _add_argument(
                    func_name or "",
                    key,
                    val_text,
                    arguments,
                    seen_keys,
                    allowed_props,
                    name_to_tool,
                ):
                    has_invalid_param = True
                    break
            if has_invalid_param:
                arguments.clear()
                param_invalid = True
        parsed_ok = bool(func_name)
    except Exception:
        parsed_ok = False

    if not parsed_ok:
        try:
            m_fn = _FUNC_NAME_V1_REGEX.search(block)
            if m_fn:
                func_name = (m_fn.group(1) or "").strip()
            has_invalid_param = bool(_PARAM_MISSING_NAME_REGEX.search(block))
            seen_keys = set()
            allowed_props = name_to_allowed_props.get(func_name or "", set())
            for pm in _PARAM_WITH_NAME_REGEX.finditer(block):
                key = pm.group(1).strip()
                val_text = pm.group(2) or ""
                if val_text.startswith("<![CDATA[") and val_text.endswith("]]>"):
                    val_text = val_text[len("<![CDATA[") : -len("]]>")]
                val_text = val_text.strip()
                if not _add_argument(
                    func_name or "",
                    key,
                    val_text,
                    arguments,
                    seen_keys,
                    allowed_props,
                    name_to_tool,
                ):
                    has_invalid_param = True
                    break
            if has_invalid_param:
                arguments.clear()
                param_invalid = True
            parsed_ok = bool(func_name)
        except Exception:
            parsed_ok = False

    if not func_name or param_invalid:
        return None

    func_name, arguments = _normalize_alias_tool_call(func_name, arguments, tool_names)

    if func_name not in tool_names:
        return None

    req_props = name_to_required.get(func_name, set())
    if req_props and not req_props.issubset(arguments.keys()):
        return None

    if not parsed_ok:
        return None

    return {"name": func_name, "parameters": arguments}


def _parse_partial_params(
    block: str,
    func_name: str,
    name_to_allowed_props: dict[str, set[str]],
    name_to_tool: dict[str, ChatCompletionToolsParam],
) -> dict[str, Any]:
    arguments: dict[str, Any] = {}
    seen_keys: set[str] = set()
    allowed_props = name_to_allowed_props.get(func_name, set())
    for pm in _PARAM_WITH_NAME_REGEX.finditer(block):
        key = pm.group(1).strip()
        if not key or key in seen_keys:
            continue
        val_text = pm.group(2) or ""
        if val_text.startswith("<![CDATA[") and val_text.endswith("]]>"):
            val_text = val_text[len("<![CDATA[") : -len("]]>")]
        val_text = val_text.strip()
        _add_argument(
            func_name,
            key,
            val_text,
            arguments,
            seen_keys,
            allowed_props,
            name_to_tool,
        )
    return arguments


class MiniCPM5XMLToolParser(ToolParser):
    """MiniCPM5 XML tool parser."""

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
    ):
        super().__init__(tokenizer, tools)
        self.tool_call_start_token = "<function"
        self.tool_call_end_token = "</function>"
        self._processed_len = 0

    def _reset_stream_state(self) -> None:
        self._processed_len = 0
        self.current_tool_id = -1
        self.current_tool_name_sent = False
        self.prev_tool_call_arr = []
        self.streamed_args_for_tool = []

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            # Tool XML tags are special tokens in MiniCPM5; must not strip them
            # before tool parsing (see internlm2/mistral vLLM tool parsers).
            request.skip_special_tokens = False
        return request

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        model_output = _normalize_model_output(model_output)
        if self.tool_call_start_token not in model_output:
            logger.debug("[MiniCPM5XMLToolParser] no <function token in output")
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=_strip_thinking_content(model_output),
            )

        tool_names, name_to_allowed_props, name_to_required, name_to_tool = (
            _build_tool_maps(request.tools)
        )

        tool_calls: list[ToolCall] = []
        normal_parts: list[str] = []
        last_end = 0

        try:
            for match in _FUNC_BLOCK_REGEX.finditer(model_output):
                if match.start() > last_end:
                    normal_parts.append(model_output[last_end : match.start()])

                block = match.group(0)
                parsed = _parse_function_block(
                    block,
                    tool_names,
                    name_to_allowed_props,
                    name_to_required,
                    name_to_tool,
                )
                if parsed is not None:
                    tool_calls.append(
                        ToolCall(
                            id=f"call_{random_uuid()}",
                            type="function",
                            function=FunctionCall(
                                name=parsed["name"],
                                arguments=json.dumps(
                                    parsed["parameters"],
                                    ensure_ascii=False,
                                ),
                            ),
                        )
                    )
                else:
                    normal_parts.append(block)

                last_end = match.end()

            if last_end < len(model_output):
                normal_parts.append(model_output[last_end:])

            content = _strip_thinking_content("".join(normal_parts).strip())

            logger.debug(
                "[MiniCPM5XMLToolParser] extracted %d tool calls",
                len(tool_calls),
            )

            tools_called = len(tool_calls) > 0
            return ExtractedToolCallInformation(
                tools_called=tools_called,
                tool_calls=tool_calls,
                content=None if tools_called else content,
            )
        except Exception as e:
            logger.error("Error in MiniCPM5XMLToolParser.extract_tool_calls: %s", e)
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

    def _emit_tool_args_delta(
        self,
        tool_index: int,
        args_json: str,
        *,
        is_complete: bool = False,
    ) -> DeltaMessage | None:
        prev_args = (
            self.streamed_args_for_tool[tool_index]
            if tool_index < len(self.streamed_args_for_tool)
            else ""
        )
        arg_diff = _streaming_args_diff(
            prev_args,
            args_json,
            is_complete=is_complete,
        )
        if not arg_diff:
            return None
        while len(self.streamed_args_for_tool) <= tool_index:
            self.streamed_args_for_tool.append("")
        self.streamed_args_for_tool[tool_index] = prev_args + arg_diff
        return DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=tool_index,
                    function=DeltaFunctionCall(
                        arguments=arg_diff,
                    ).model_dump(exclude_none=True),
                )
            ],
        )

    def _start_tool_call(self, func_name: str) -> DeltaMessage:
        self.current_tool_id += 1
        self.current_tool_name_sent = True
        while len(self.streamed_args_for_tool) <= self.current_tool_id:
            self.streamed_args_for_tool.append("")
        while len(self.prev_tool_call_arr) <= self.current_tool_id:
            self.prev_tool_call_arr.append({})
        self.prev_tool_call_arr[self.current_tool_id] = {"name": func_name}
        return DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=self.current_tool_id,
                    id=make_tool_call_id(),
                    type="function",
                    function=DeltaFunctionCall(
                        name=func_name,
                    ).model_dump(exclude_none=True),
                )
            ],
        )

    def _process_complete_block_streaming(
        self,
        block: str,
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        tool_names, name_to_allowed_props, name_to_required, name_to_tool = (
            _build_tool_maps(request.tools)
        )
        parsed = _parse_function_block(
            block,
            tool_names,
            name_to_allowed_props,
            name_to_required,
            name_to_tool,
        )
        if parsed is None:
            return DeltaMessage(content=block)

        args_json = json.dumps(parsed["parameters"], ensure_ascii=False)
        func_name = parsed["name"]

        if not self.current_tool_name_sent:
            self.current_tool_id += 1
            tool_index = self.current_tool_id
            while len(self.streamed_args_for_tool) <= tool_index:
                self.streamed_args_for_tool.append("")
            while len(self.prev_tool_call_arr) <= tool_index:
                self.prev_tool_call_arr.append({})
            self.streamed_args_for_tool[tool_index] = args_json
            self.prev_tool_call_arr[tool_index] = {
                "name": func_name,
                "arguments": parsed["parameters"],
            }
            if not self.prev_tool_call_arr:
                self.prev_tool_call_arr = [{"arguments": {}}]
            return DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=tool_index,
                        id=make_tool_call_id(),
                        type="function",
                        function=DeltaFunctionCall(
                            name=func_name,
                            arguments=args_json,
                        ).model_dump(exclude_none=True),
                    )
                ],
            )

        tool_index = self.current_tool_id
        self.prev_tool_call_arr[tool_index]["arguments"] = parsed["parameters"]
        delta = self._emit_tool_args_delta(
            tool_index,
            args_json,
            is_complete=True,
        )
        self.current_tool_name_sent = False
        if delta:
            if not self.prev_tool_call_arr:
                self.prev_tool_call_arr = [{"arguments": {}}]
            return delta
        return DeltaMessage(content="")

    def _process_partial_block_streaming(
        self,
        block: str,
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        tool_names, name_to_allowed_props, _, name_to_tool = _build_tool_maps(
            request.tools
        )
        if _PARAM_MISSING_NAME_REGEX.search(block):
            return None

        match = _FUNC_NAME_V1_REGEX.search(block)
        if not match:
            return None
        func_name = (match.group(1) or "").strip()
        if func_name not in tool_names:
            return None

        if not self.current_tool_name_sent:
            return self._start_tool_call(func_name)

        arguments = _parse_partial_params(
            block,
            func_name,
            name_to_allowed_props,
            name_to_tool,
        )
        if not arguments:
            return None
        args_json = json.dumps(arguments, ensure_ascii=False)
        self.prev_tool_call_arr[self.current_tool_id]["arguments"] = arguments
        delta = self._emit_tool_args_delta(
            self.current_tool_id,
            args_json,
            is_complete=False,
        )
        if delta and not self.prev_tool_call_arr:
            self.prev_tool_call_arr = [{"arguments": {}}]
        return delta

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
        del previous_token_ids, current_token_ids, delta_token_ids
        try:
            current_text = _normalize_model_output(current_text)
            if not previous_text:
                self._reset_stream_state()

            if self.tool_call_start_token not in current_text:
                if self._processed_len < len(current_text):
                    content = current_text[self._processed_len :]
                    self._processed_len = len(current_text)
                    return DeltaMessage(content=content) if content else None
                return None

            for match in _FUNC_BLOCK_REGEX.finditer(current_text):
                if match.end() <= self._processed_len:
                    continue

                if match.start() > self._processed_len:
                    gap = current_text[self._processed_len : match.start()]
                    self._processed_len = match.start()
                    return DeltaMessage(content=gap) if gap else None

                block = match.group(0)
                delta = self._process_complete_block_streaming(block, request)
                self._processed_len = match.end()
                if delta is not None:
                    return delta

            remainder = current_text[self._processed_len :]
            if not remainder:
                if (
                    not delta_text
                    and self.current_tool_id >= 0
                    and not self.current_tool_name_sent
                ):
                    return DeltaMessage(content="")
                return None

            func_idx = remainder.find(self.tool_call_start_token)
            if func_idx > 0:
                gap = remainder[:func_idx]
                self._processed_len += func_idx
                return DeltaMessage(content=gap)

            if func_idx == -1:
                overlap = partial_tag_overlap(remainder, self.tool_call_start_token)
                if overlap:
                    return None
                self._processed_len = len(current_text)
                return DeltaMessage(content=remainder) if remainder else None

            partial_block = remainder[func_idx:]
            if self.tool_call_end_token in partial_block:
                end_idx = partial_block.rfind(self.tool_call_end_token)
                complete_block = partial_block[
                    : end_idx + len(self.tool_call_end_token)
                ]
                delta = self._process_complete_block_streaming(complete_block, request)
                self._processed_len += func_idx + len(complete_block)
                if delta is not None:
                    return delta
                partial_block = partial_block[end_idx + len(self.tool_call_end_token) :]
                if not partial_block.strip():
                    return None
                func_idx = partial_block.find(self.tool_call_start_token)
                if func_idx == -1:
                    self._processed_len = len(current_text)
                    if partial_block:
                        return DeltaMessage(content=partial_block)
                    return None
                partial_block = partial_block[func_idx:]

            return self._process_partial_block_streaming(partial_block, request)
        except Exception:
            logger.exception(
                "Error in MiniCPM5XMLToolParser.extract_tool_calls_streaming"
            )
            return None
