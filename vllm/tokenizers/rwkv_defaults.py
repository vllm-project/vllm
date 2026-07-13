# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path
from typing import Any

import regex as re

from vllm.transformers_utils.configs.rwkv7 import build_rwkv7_config_from_pth

RWKV_NATIVE_CHAT_TEMPLATE = "{# RWKV native chat template #}"
RWKV_TOOL_CALL_PARSER = "rwkv"
RWKV_DEFAULT_STOPS = ("\nUser:", "\n### User")
RWKV_DEFAULT_STOP = RWKV_DEFAULT_STOPS[0]
RWKV_DEFAULT_STOP_TOKEN_IDS = (0,)
RWKV_GENERATION_PROMPT_OPEN_THINK = "open_think"
RWKV_GENERATION_PROMPT_FAKE_THINK = "fake_think"
RWKV_GENERATION_PROMPT_MODES = (
    RWKV_GENERATION_PROMPT_OPEN_THINK,
    RWKV_GENERATION_PROMPT_FAKE_THINK,
)

_BLANK_LINES_RE = re.compile(r"\n{2,}")
_DAPO_MATH_PROMPT_PREFIX_RE = re.compile(
    r"\A\s*Solve the following math problem step by step\.\s*"
    r"The last line of your response should be of the form Answer: \$Answer "
    r"\(without quotes\) where \$Answer is the answer to the problem\.\s*\n",
)
_DAPO_MATH_PROMPT_SUFFIX_RE = re.compile(
    r'\n\s*Remember to put your answer on its own line after "Answer:"\.?\s*\Z',
)


def normalize_rwkv_message_content(content: Any) -> str:
    if content is None:
        text = ""
    elif isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        text = "\n".join(part for part in parts if part)
    else:
        text = str(content)

    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    return _BLANK_LINES_RE.sub("\n", text)


def simplify_rwkv_math_prompt(text: str) -> str:
    text = _DAPO_MATH_PROMPT_PREFIX_RE.sub("", text)
    text = _DAPO_MATH_PROMPT_SUFFIX_RE.sub("", text)
    return text.strip()


def render_rwkv_chat_template(
    messages: list[Any],
    tools: list[dict[str, Any]] | None = None,
    *,
    add_generation_prompt: bool,
    rwkv_generation_prompt: str = RWKV_GENERATION_PROMPT_OPEN_THINK,
) -> str:
    _check_generation_prompt(rwkv_generation_prompt)
    has_tool_history = any(
        _field(message, "role", "") == "tool"
        or bool(_field(message, "tool_calls", None))
        for message in messages
    )
    if tools or has_tool_history:
        return _render_tool_chat(
            messages,
            tools or [],
            add_generation_prompt=add_generation_prompt,
            rwkv_generation_prompt=rwkv_generation_prompt,
        )
    return _render_basic_chat(
        messages,
        add_generation_prompt=add_generation_prompt,
        rwkv_generation_prompt=rwkv_generation_prompt,
    )


def is_rwkv_model_config(model_config: Any) -> bool:
    if _is_rwkv_tokenizer_mode(getattr(model_config, "tokenizer_mode", None)):
        return True
    hf_config = getattr(model_config, "hf_config", None)
    return getattr(hf_config, "model_type", None) == "rwkv7"


def is_rwkv_model_arg(model: str | Path | None) -> bool:
    if model is None:
        return False
    try:
        return build_rwkv7_config_from_pth(model) is not None
    except ValueError:
        return False


def resolve_rwkv_tool_parser(
    *,
    tool_parser: str | None,
    enable_auto_tools: bool,
    model_config: Any | None = None,
    tokenizer_mode: str | None = None,
    model: str | Path | None = None,
) -> str | None:
    if tool_parser is not None or not enable_auto_tools:
        return tool_parser
    if model_config is not None and is_rwkv_model_config(model_config):
        return RWKV_TOOL_CALL_PARSER
    if _is_rwkv_tokenizer_mode(tokenizer_mode) or is_rwkv_model_arg(model):
        return RWKV_TOOL_CALL_PARSER
    return tool_parser


def apply_rwkv_default_sampling_params(
    default_sampling_params: dict[str, Any],
    model_config: Any,
) -> None:
    if not is_rwkv_model_config(model_config):
        return
    if "stop" not in default_sampling_params:
        default_sampling_params["stop"] = list(RWKV_DEFAULT_STOPS)
    if "stop_token_ids" not in default_sampling_params:
        default_sampling_params["stop_token_ids"] = list(RWKV_DEFAULT_STOP_TOKEN_IDS)


def _render_basic_chat(
    messages: list[Any],
    *,
    add_generation_prompt: bool,
    rwkv_generation_prompt: str,
) -> str:
    rendered: list[str] = []
    for message in messages:
        role = _field(message, "role", "")
        content = normalize_rwkv_message_content(_field(message, "content", ""))
        if role == "user":
            content = simplify_rwkv_math_prompt(content)
        if role == "system":
            label = "System"
        elif role == "user":
            label = "User"
        elif role == "assistant":
            label = "Assistant"
        else:
            raise ValueError(f"Unsupported RWKV chat message role: {role!r}")
        rendered.append(f"{label}: {content}" if content else f"{label}:")

    if add_generation_prompt:
        rendered.append(f"Assistant: {_generation_prompt_text(rwkv_generation_prompt)}")

    return "\n\n".join(rendered)


def _render_tool_chat(
    messages: list[Any],
    tools: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
    rwkv_generation_prompt: str,
) -> str:
    lines: list[str] = []
    pending_system: list[str] = []

    for message in messages:
        role = _field(message, "role", "")
        content = normalize_rwkv_message_content(_field(message, "content", ""))
        if role == "user":
            content = simplify_rwkv_math_prompt(content)

        if role == "system":
            pending_system.append(content)
            continue

        if pending_system or (tools and not lines):
            lines.append("### System")
            lines.extend(item for item in pending_system if item)
            lines.extend(_render_tool_definitions(tools))
            pending_system.clear()

        if role == "user":
            lines.extend(["### User", content])
        elif role == "assistant":
            lines.append("### Assistant")
            if content:
                lines.append(content)
            for tool_call in _field(message, "tool_calls", []) or []:
                function = _tool_function(tool_call)
                name = _field(function, "name", "")
                arguments = _field(function, "arguments", {}) or {}
                payload = {
                    "name": name,
                    "arguments": _json_value(arguments),
                }
                lines.extend(["**Tool Call:**", "```json", _json_text(payload), "```"])
        elif role == "tool":
            payload = _json_value(content)
            lines.extend(["### Tool Output", "```json", _json_text(payload), "```"])
        else:
            raise ValueError(f"Unsupported RWKV chat message role: {role!r}")

    if pending_system:
        lines.append("### System")
        lines.extend(item for item in pending_system if item)
        lines.extend(_render_tool_definitions(tools))

    if add_generation_prompt:
        lines.append("### Assistant")
        lines.append(_generation_prompt_text(rwkv_generation_prompt))

    return "\n".join(lines)


def _check_generation_prompt(mode: str) -> None:
    if mode not in RWKV_GENERATION_PROMPT_MODES:
        raise ValueError(
            "Unsupported RWKV generation prompt mode: "
            f"{mode!r}. Expected one of {RWKV_GENERATION_PROMPT_MODES!r}."
        )


def _generation_prompt_text(mode: str) -> str:
    if mode == RWKV_GENERATION_PROMPT_OPEN_THINK:
        return "<think"
    if mode == RWKV_GENERATION_PROMPT_FAKE_THINK:
        return "<think></think"
    _check_generation_prompt(mode)
    raise AssertionError("unreachable")


def _render_tool_definitions(tools: list[dict[str, Any]]) -> list[str]:
    rendered: list[str] = []
    for tool in tools:
        function = _tool_function(tool)
        name = _field(function, "name", "")
        description = _field(function, "description", "") or ""
        parameters = _field(function, "parameters", {}) or {}
        rendered.append(f"### `{name}`")
        if description:
            rendered.append(
                f"**Description:** {normalize_rwkv_message_content(description)}"
            )
        rendered.append("**Parameters:**")
        rendered.append("```json")
        rendered.append(_json_text(parameters))
        rendered.append("```")
    if rendered:
        rendered.extend(
            [
                "To call one of these tools, write exactly this format:",
                "**Tool Call:**",
                "```json",
                '{"name": "tool_name", "arguments": {"key": "value"}}',
                "```",
                "Do not invent tool call IDs or write tool outputs yourself.",
            ]
        )
    return rendered


def _json_text(value: Any) -> str:
    return json.dumps(_json_value(value), ensure_ascii=False, indent=2)


def _json_value(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return normalize_rwkv_message_content(value)
    return value


def _tool_function(tool: Any) -> Any:
    if isinstance(tool, dict):
        return tool.get("function", tool)
    return getattr(tool, "function", tool)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _is_rwkv_tokenizer_mode(tokenizer_mode: Any) -> bool:
    return isinstance(tokenizer_mode, str) and tokenizer_mode.lower() == "rwkv"
