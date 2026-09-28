# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling chat-encoding core.

Pure implementation of Inkling chat rendering, kept deliberately free of
vLLM imports: it depends only on the :class:`InklingTextTokenizer` protocol
and speaks OpenAI-style message dicts. If a standalone Inkling
input-processing library becomes available, this module is the unit to
swap out — the swap point is marked in ``vllm/renderers/inkling.py``.

Multimodal parts follow the contract of vLLM's Inkling multimodal processor
(``InklingMultiModalProcessor`` anchors prompt updates on the bare
content-kind marker and inserts the per-patch placeholder run itself):

* image parts emit only ``<|content_image|>`` — no seed placeholder id;
* audio parts emit ``<|content_audio_input|><|audio_end|>`` — no seed
  placeholder id between them.

``reasoning_effort`` (a float in [0, 0.99], sourced from
``chat_template_kwargs`` only) renders the ``Thinking effort level:``
system control block after the tool declarations and initial system messages.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Protocol

END_OF_TEXT = "<|endoftext|>"
MESSAGE_USER = "<|message_user|>"
MESSAGE_MODEL = "<|message_model|>"
MESSAGE_SYSTEM = "<|message_system|>"
MESSAGE_TOOL = "<|message_tool|>"
CONTENT_TEXT = "<|content_text|>"
CONTENT_IMAGE = "<|content_image|>"
CONTENT_MODEL_END_SAMPLING = "<|content_model_end_sampling|>"
CONTENT_THINKING = "<|content_thinking|>"
CONTENT_AUDIO_INPUT = "<|content_audio_input|>"
CONTENT_TOOL_ERROR = "<|content_tool_error|>"
CONTENT_XML = "<|content_xml|>"
CONTENT_INVOKE_TOOL_JSON = "<|content_invoke_tool_json|>"
CONTENT_INVOKE_TOOL_TEXT = "<|content_invoke_tool_text|>"
END_MESSAGE = "<|end_message|>"
AUDIO_END = "<|audio_end|>"

ROLE_MESSAGE_TOKENS: dict[str, str] = {
    "user": MESSAGE_USER,
    "assistant": MESSAGE_MODEL,
    "system": MESSAGE_SYSTEM,
    # The developer role folds into the system role on the Inkling wire.
    "developer": MESSAGE_SYSTEM,
    "tool": MESSAGE_TOOL,
}

# Alternate vocab spellings per semantic token. The Inkling HF tokenizer
# exposes some semantic slots as ``<|unused_NNNNNN|>`` tokens (notably
# CONTENT_XML = <|unused_200024|>), so ID resolution must try these
# spellings in order.
SPECIAL_TOKEN_SPELLINGS: dict[str, tuple[str, ...]] = {
    MESSAGE_USER: (MESSAGE_USER,),
    MESSAGE_MODEL: (MESSAGE_MODEL,),
    MESSAGE_SYSTEM: (MESSAGE_SYSTEM,),
    MESSAGE_TOOL: (MESSAGE_TOOL,),
    CONTENT_TEXT: (CONTENT_TEXT,),
    CONTENT_IMAGE: (CONTENT_IMAGE,),
    CONTENT_MODEL_END_SAMPLING: (CONTENT_MODEL_END_SAMPLING,),
    CONTENT_THINKING: (CONTENT_THINKING,),
    CONTENT_AUDIO_INPUT: (CONTENT_AUDIO_INPUT,),
    CONTENT_XML: (CONTENT_XML, "<|unused_200024|>"),
    CONTENT_INVOKE_TOOL_JSON: (CONTENT_INVOKE_TOOL_JSON,),
    END_MESSAGE: (END_MESSAGE,),
    AUDIO_END: (AUDIO_END,),
}


class InklingTextTokenizer(Protocol):
    """Structural tokenizer contract required by the renderer."""

    def encode_text(self, text: str) -> list[int]: ...

    def encode_special(self, token: str) -> int: ...


# OpenAI content-part type spellings that mean image / audio (rendering only
# needs the kind, not the bytes — the bytes are handled by the MM processor).
_IMAGE_PART_TYPES = frozenset({"image", "input_image", "image_url"})
_AUDIO_PART_TYPES = frozenset({"audio", "input_audio", "audio_url"})

_MAX_REASONING_EFFORT = 0.99


def render_inkling_messages(
    messages: Sequence[Mapping[str, Any]],
    tokenizer: InklingTextTokenizer,
    *,
    add_generation_prompt: bool = True,
    tools: Sequence[Mapping[str, Any]] | None = None,
    reasoning_effort: float | None = None,
) -> list[int]:
    """Render chat messages to Inkling input ids.

    PURE renderer: emits Inkling framing plus bare media markers; media
    encoding and placeholder expansion happen later in the MM processor.
    ``add_generation_prompt`` appends the assistant turn opener so the
    model continues into the response.
    """
    input_ids: list[int] = []
    tool_call_id_to_name: dict[str, str] = {}

    # Request-level tools plus per-developer-message tools (Rust renderer
    # semantics) are declared in a single leading system block.
    all_tools = list(tools or [])
    for message in messages:
        if message.get("role") == "developer":
            all_tools.extend(message.get("tools") or [])

    if all_tools:
        _append_message(
            input_ids,
            tokenizer,
            "system",
            "xml",
            _tool_declare_json(all_tools),
            author_name="tool_declare",
        )

    for message in messages:
        role = _expect_role(message)
        if role not in {"system", "developer"} and reasoning_effort is not None:
            _append_reasoning_effort(input_ids, tokenizer, reasoning_effort)
            reasoning_effort = None

        if role == "tool":
            tool_name = message.get("name") or tool_call_id_to_name.get(
                str(message.get("tool_call_id") or ""), ""
            )
            _append_message(
                input_ids,
                tokenizer,
                "tool",
                "text",
                _flatten_text_content(message.get("content")),
                author_name=str(tool_name),
            )
            continue

        if role == "assistant":
            reasoning_content = message.get("reasoning")
            if reasoning_content is None:
                reasoning_content = message.get("reasoning_content")
            if reasoning_content:
                if not isinstance(reasoning_content, str):
                    raise TypeError(
                        "assistant reasoning_content must be a string for "
                        "Inkling rendering"
                    )
                _append_message(
                    input_ids,
                    tokenizer,
                    "assistant",
                    "thinking",
                    reasoning_content,
                )

        for kind, text in _iter_render_parts(message.get("content")):
            _append_message(input_ids, tokenizer, role, kind, text)

        if role == "assistant":
            for tool_call in message.get("tool_calls") or []:
                name, args = _tool_call_name_and_args(tool_call)
                tool_call_id = _as_mapping(tool_call).get("id")
                if tool_call_id:
                    tool_call_id_to_name[str(tool_call_id)] = name
                _append_message(
                    input_ids,
                    tokenizer,
                    "assistant",
                    "invoke_tool_json",
                    _tool_call_json(name, args),
                    author_name=name,
                )

            input_ids.append(tokenizer.encode_special(CONTENT_MODEL_END_SAMPLING))

    if reasoning_effort is not None:
        _append_reasoning_effort(input_ids, tokenizer, reasoning_effort)

    if add_generation_prompt:
        input_ids.append(tokenizer.encode_special(MESSAGE_MODEL))
    return input_ids


def _append_reasoning_effort(
    input_ids: list[int],
    tokenizer: InklingTextTokenizer,
    reasoning_effort: float,
) -> None:
    _append_message(
        input_ids,
        tokenizer,
        "system",
        "text",
        _thinking_effort_text(reasoning_effort),
    )


def _thinking_effort_text(reasoning_effort: Any) -> str:
    if isinstance(reasoning_effort, bool) or not isinstance(
        reasoning_effort, (int, float)
    ):
        raise TypeError(
            "Inkling reasoning_effort must be a number in [0.0, 0.99], "
            f"got {type(reasoning_effort).__name__}"
        )
    value = float(reasoning_effort)
    if not 0.0 <= value <= _MAX_REASONING_EFFORT:
        raise ValueError(
            f"Inkling reasoning_effort must be in [0.0, 0.99], got {value}"
        )

    effort_text = f"{value:.2f}".rstrip("0").rstrip(".")
    if effort_text in {"0", "-0"}:
        effort_text = "0.0"
    return f"Thinking effort level: {effort_text}"


def _append_message(
    input_ids: list[int],
    tokenizer: InklingTextTokenizer,
    role: str,
    kind: str,
    text: str,
    *,
    author_name: str | None = None,
) -> None:
    input_ids.append(tokenizer.encode_special(ROLE_MESSAGE_TOKENS[role]))
    if author_name:
        input_ids.extend(tokenizer.encode_text(author_name))

    if kind == "text":
        input_ids.append(tokenizer.encode_special(CONTENT_TEXT))
        input_ids.extend(tokenizer.encode_text(text))
    elif kind == "image":
        # Bare marker only: the MM processor replaces the marker with
        # `marker + placeholder * num_patches` (see mm_preprocess.py).
        input_ids.append(tokenizer.encode_special(CONTENT_IMAGE))
    elif kind == "audio":
        input_ids.append(tokenizer.encode_special(CONTENT_AUDIO_INPUT))
        input_ids.append(tokenizer.encode_special(AUDIO_END))
    elif kind == "thinking":
        input_ids.append(tokenizer.encode_special(CONTENT_THINKING))
        input_ids.extend(tokenizer.encode_text(text))
    elif kind == "xml":
        input_ids.append(tokenizer.encode_special(CONTENT_XML))
        input_ids.extend(tokenizer.encode_text(text))
    elif kind == "invoke_tool_json":
        input_ids.append(tokenizer.encode_special(CONTENT_INVOKE_TOOL_JSON))
        input_ids.extend(tokenizer.encode_text(text))
    else:
        raise ValueError(f"unsupported Inkling render part kind: {kind!r}")

    input_ids.append(tokenizer.encode_special(END_MESSAGE))


def _iter_render_parts(content: Any) -> Iterator[tuple[str, str]]:
    """Yield (kind, text) per content part: kind in {text, image, audio}."""
    if content is None:
        return
    if isinstance(content, str):
        if content:
            yield ("text", content)
        return
    if not isinstance(content, Sequence) or isinstance(content, (bytes, bytearray)):
        raise TypeError("message content must be a string or a sequence of parts")
    for part in content:
        if isinstance(part, str):
            yield ("text", part)
            continue
        if not isinstance(part, Mapping):
            raise TypeError(f"content part must be mapping, got {type(part).__name__}")
        ptype = part.get("type")
        if ptype in (None, "text", "input_text"):
            text = part.get("text", "")
            yield ("text", text if isinstance(text, str) else "")
        elif ptype in _IMAGE_PART_TYPES:
            yield ("image", "")
        elif ptype in _AUDIO_PART_TYPES:
            yield ("audio", "")
        else:
            raise ValueError(f"unsupported content part type: {ptype!r}")


def _flatten_text_content(content: Any) -> str:
    """Flatten a tool-response content (string or text parts) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for kind, text in _iter_render_parts(content):
        if kind != "text":
            raise ValueError(
                "Inkling tool response content must be text, "
                f"got a part of kind {kind!r}"
            )
        parts.append(text)
    return "".join(parts)


def _expect_role(message: Mapping[str, Any]) -> str:
    role = message.get("role")
    if role not in ROLE_MESSAGE_TOKENS:
        raise ValueError(
            f"unsupported Inkling message role {role!r}; "
            f"expected one of {sorted(ROLE_MESSAGE_TOKENS)}"
        )
    return str(role)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    raise TypeError(f"expected mapping, got {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _sort_json(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _sort_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _sort_json(value[key]) for key in sorted(value)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_sort_json(item) for item in value]
    return value


def _tool_declare_json(tools: Sequence[Mapping[str, Any]]) -> str:
    tool_specs = []
    for tool_value in tools:
        tool = _as_mapping(tool_value)
        function = _as_mapping(tool.get("function", {}))
        tool_specs.append(
            {
                "description": function.get("description") or "",
                "name": function["name"],
                "parameters": function.get("parameters") or {},
                "type": tool.get("type", "function"),
            }
        )
    return _canonical_json(tool_specs)


def _tool_call_name_and_args(tool_call_value: Any) -> tuple[str, Mapping[str, Any]]:
    tool_call = _as_mapping(tool_call_value)
    function = _as_mapping(tool_call.get("function", {}))
    name = function.get("name")
    if not isinstance(name, str):
        raise TypeError("tool call function name must be a string")

    raw_args = function.get("arguments") or {}
    if isinstance(raw_args, str):
        args = json.loads(raw_args) if raw_args.strip() else {}
    else:
        args = raw_args
    if not isinstance(args, Mapping):
        raise TypeError("tool call function arguments must decode to an object")
    return name, args


def _tool_call_json(name: str, args: Mapping[str, Any]) -> str:
    name_json = json.dumps(name, ensure_ascii=False, allow_nan=False)
    return f'{{"name":{name_json},"args":{_canonical_json(args)}}}'
