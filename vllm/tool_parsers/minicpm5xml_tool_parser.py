# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.engine.protocol import (
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    ToolParser,
    ToolParserManager,
)
from vllm.utils import random_uuid

logger = init_logger(__name__)

try:
    from lxml import etree as ET  # type: ignore

    _HAS_LXML = True
except Exception:  # pragma: no cover
    import xml.etree.ElementTree as ET  # type: ignore

    _HAS_LXML = False

_FUNC_NAME_V1_REGEX = re.compile(
    r"<function\s+name=['\"]([^'\"]+)['\"][^>]*>")
_PARAM_WITH_NAME_REGEX = re.compile(
    r"<param\s+name=['\"]([^'\"]+)['\"]>([\s\S]*?)</param>", re.DOTALL)
_PARAM_MISSING_NAME_REGEX = re.compile(r"<param(?![^>]*\bname=)[^>]*>", re.DOTALL)
_FUNC_BLOCK_REGEX = re.compile(r"<function.*?</function>", re.DOTALL)


def _parse_arguments(json_value: str) -> Tuple[Any, bool]:
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
    name_to_tool: Dict[str, ChatCompletionToolsParam],
) -> Optional[str]:
    tool = name_to_tool.get(func_name)
    if tool is None or tool.function.parameters is None:
        return None
    props = tool.function.parameters.get("properties", {})
    if arg_key not in props:
        return None
    return props[arg_key].get("type")


def _build_tool_maps(
    tools: Optional[List[ChatCompletionToolsParam]],
) -> Tuple[
    Set[str],
    Dict[str, Set[str]],
    Dict[str, Set[str]],
    Dict[str, ChatCompletionToolsParam],
]:
    name_to_tool: Dict[str, ChatCompletionToolsParam] = {}
    name_to_allowed_props: Dict[str, Set[str]] = {}
    name_to_required: Dict[str, Set[str]] = {}

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
        except Exception:
            name_to_required[name] = set()

    return set(name_to_tool.keys()), name_to_allowed_props, name_to_required, name_to_tool


def _parse_function_block(
    block: str,
    tool_names: Set[str],
    name_to_allowed_props: Dict[str, Set[str]],
    name_to_required: Dict[str, Set[str]],
    name_to_tool: Dict[str, ChatCompletionToolsParam],
) -> Optional[Dict[str, Any]]:
    """Return {name, parameters} if block is valid, else None."""
    func_name: Optional[str] = None
    arguments: Dict[str, Any] = {}
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
        param_nodes: List[Any] = []
        if func_node is not None:
            param_nodes = list(func_node.findall("param"))
            if args_node is not None and not param_nodes:
                param_nodes = list(args_node.findall("param"))

        if func_node is not None:
            seen_keys: Set[str] = set()
            allowed_props = name_to_allowed_props.get(func_name or "", set())
            has_invalid_param = False
            for param in param_nodes:
                key = param.attrib.get("name")
                if not key:
                    has_invalid_param = True
                    break
                if allowed_props and key not in allowed_props:
                    has_invalid_param = True
                    break
                if key in seen_keys:
                    has_invalid_param = True
                    break
                seen_keys.add(key)
                val_text = (param.text or "").strip()
                arg_type = _get_argument_type(func_name or "", key, name_to_tool)
                if arg_type != "string":
                    parsed_val, _ = _parse_arguments(val_text)
                    arguments[key] = parsed_val
                else:
                    arguments[key] = val_text
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
                if allowed_props and key not in allowed_props:
                    has_invalid_param = True
                    break
                if key in seen_keys:
                    has_invalid_param = True
                    break
                seen_keys.add(key)
                val_text = (pm.group(2) or "")
                if val_text.startswith("<![CDATA[") and val_text.endswith("]]>"):
                    val_text = val_text[len("<![CDATA["):-len("]]>")]
                val_text = val_text.strip()
                arg_type = _get_argument_type(func_name or "", key, name_to_tool)
                if arg_type != "string":
                    parsed_val, _ = _parse_arguments(val_text)
                    arguments[key] = parsed_val
                else:
                    arguments[key] = val_text
            if has_invalid_param:
                arguments.clear()
                param_invalid = True
            parsed_ok = bool(func_name)
        except Exception:
            parsed_ok = False

    if not func_name or func_name not in tool_names or param_invalid:
        return None

    req_props = name_to_required.get(func_name, set())
    if req_props and not req_props.issubset(arguments.keys()):
        return None

    if not parsed_ok:
        return None

    return {"name": func_name, "parameters": arguments}


@ToolParserManager.register_module("minicpm5")
class MiniCPM5XMLToolParser(ToolParser):
    """MiniCPM5 XML tool parser (SGLang minicpm5_detector compatible)."""

    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)
        self.tool_call_start_token = "<function"
        self.tool_call_end_token = "</function>"

    def adjust_request(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionRequest:
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
        if self.tool_call_start_token not in model_output:
            logger.debug("[MiniCPM5XMLToolParser] no <function token in output")
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        tool_names, name_to_allowed_props, name_to_required, name_to_tool = (
            _build_tool_maps(request.tools))

        tool_calls: List[ToolCall] = []
        normal_parts: List[str] = []
        last_end = 0

        try:
            for match in _FUNC_BLOCK_REGEX.finditer(model_output):
                if match.start() > last_end:
                    normal_parts.append(model_output[last_end:match.start()])

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
                        ))
                else:
                    normal_parts.append(block)

                last_end = match.end()

            if last_end < len(model_output):
                normal_parts.append(model_output[last_end:])

            content = "".join(normal_parts).strip()

            logger.debug(
                "[MiniCPM5XMLToolParser] extracted %d tool calls",
                len(tool_calls),
            )

            return ExtractedToolCallInformation(
                tools_called=len(tool_calls) > 0,
                tool_calls=tool_calls,
                content=content,
            )
        except Exception as e:
            logger.error("Error in MiniCPM5XMLToolParser.extract_tool_calls: %s", e)
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )
