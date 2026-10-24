# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import json
from typing import Any

from transformers import TokenizersBackend

from vllm.renderers.chat_utils import ChatCompletionMessageParam

from .deepseek_v41_encoding import (
    IMAGE_PLACEHOLDER,
    REASONING_EFFORT_MAPPINGS,
    encode_messages,
    find_reserved_token,
    resolve_tool_call_arguments,
    to_json,
)
from .hf import HfTokenizer, get_cached_tokenizer
from .protocol import TokenizerLike


def _reject_reserved_tokens(text: Any, where: str) -> None:
    """Reject text that spells a reserved special token.

    The encoder writes these tokens as prompt structure. A Hugging Face
    tokenizer splits added-vocabulary strings wherever they appear, so a
    message that spells one would otherwise be promoted to the token itself
    and could forge a turn boundary, an end of reasoning, or an image slot.
    """
    if not isinstance(text, str):
        return
    token = find_reserved_token(text)
    if token is None:
        return
    if token == IMAGE_PLACEHOLDER:
        raise ValueError(
            f"{where} contains image special token {IMAGE_PLACEHOLDER!r}; "
            "provide images as image content parts."
        )
    raise ValueError(
        f"{where} contains reserved special token {token!r}. Reserved tokens "
        "are written by the prompt encoder and must not appear in request text."
    )


def _reject_reserved_tokens_in_tools(tools: list[Any]) -> None:
    """Reject tool definitions whose text the encoder renders verbatim."""
    for index, tool in enumerate(tools):
        function = (tool.get("function") or {}) if isinstance(tool, dict) else {}
        _reject_reserved_tokens(function.get("name"), f"tools[{index}].function.name")
        _reject_reserved_tokens(
            function.get("description"),
            f"tools[{index}].function.description",
        )
        parameters = function.get("parameters")
        if parameters is not None:
            _reject_reserved_tokens(
                json.dumps(parameters, ensure_ascii=False),
                f"tools[{index}].function.parameters",
            )


def _reject_reserved_tokens_in_content_blocks(content_blocks: Any) -> None:
    """Check the block form, which the encoder renders alongside content.

    A tool result block carries its own content, either a string or a list of
    text blocks, so the check follows the same nesting the encoder renders.
    """
    if isinstance(content_blocks, str):
        _reject_reserved_tokens(content_blocks, "message content")
        return
    if not isinstance(content_blocks, list):
        return
    for block in content_blocks:
        if not isinstance(block, dict):
            continue
        _reject_reserved_tokens(block.get("text"), "message content")
        if block.get("type") == "tool_result":
            _reject_reserved_tokens_in_content_blocks(block.get("content"))


def _reject_reserved_tokens_in_tool_calls(message: dict[str, Any]) -> None:
    """Reject reserved tokens in the values the encoder renders as DSML.

    The encoder decodes ``arguments`` (JSON strings, double-encoded strings,
    objects) before writing them, so scanning the raw string is not enough: a
    caller could spell a token with JSON escapes and have it decoded into the
    prompt. Validate the decoded name, keys, and rendered values instead.
    """
    for index, tool_call in enumerate(message.get("tool_calls") or []):
        function = tool_call.get("function") or {}
        _reject_reserved_tokens(
            function.get("name"),
            f"tool_calls[{index}].function.name",
        )
        arguments = resolve_tool_call_arguments(
            {"arguments": function.get("arguments")}
        )
        for key, value in arguments.items():
            where = f"tool_calls[{index}].function.arguments"
            _reject_reserved_tokens(key, f"{where} key")
            rendered = value if isinstance(value, str) else to_json(value)
            _reject_reserved_tokens(rendered, f"{where}[{key!r}]")


def _normalize_messages(
    messages: list[ChatCompletionMessageParam],
) -> list[dict[str, Any]]:
    # vLLM additions to the reference: Responses API part types, image_pil,
    # and ValueError (400) instead of the reference's asserts.
    result = [dict(message) for message in copy.deepcopy(messages)]
    image_idx = 0
    for message in result:
        image_indices = message["_image_indices"] = []
        role = message.get("role")
        if role not in ("system", "developer", "user", "assistant", "tool"):
            raise ValueError(f"Invalid role: {role}")
        if "reasoning" in message:
            message["reasoning_content"] = message["reasoning"]
        _reject_reserved_tokens(message.get("reasoning_content"), "reasoning_content")
        _reject_reserved_tokens_in_tool_calls(message)
        _reject_reserved_tokens_in_content_blocks(message.get("content_blocks"))
        response_format = message.get("response_format")
        if response_format is not None:
            _reject_reserved_tokens(
                json.dumps(response_format, ensure_ascii=False),
                "response_format",
            )
        content = message.get("content")
        if isinstance(content, str):
            _reject_reserved_tokens(content, "message content")
        elif isinstance(content, list):
            parts = []
            for block in content:
                part_type = block.get("type")
                if part_type in ("text", "input_text", "output_text"):
                    _reject_reserved_tokens(block.get("text"), "message content")
                    parts.append(block.get("text", ""))
                elif part_type in ("image_url", "input_image", "image_pil"):
                    parts.append(IMAGE_PLACEHOLDER)
                    image_indices.append(image_idx)
                    image_idx += 1
                else:
                    raise ValueError(
                        "DeepSeek V4.1 supports text and image content only; "
                        f"got {part_type!r}"
                    )
            message["content"] = "\n\n".join(parts)
    return result


def get_deepseek_v41_tokenizer(tokenizer: HfTokenizer) -> HfTokenizer:
    """Wrap an HF tokenizer with the V4.1 prompt encoder."""
    wrapped = copy.copy(tokenizer)
    added_vocab = tokenizer.get_added_vocab()

    class _DeepseekV41Tokenizer(tokenizer.__class__):  # type: ignore
        def apply_chat_template(
            self,
            messages: list[ChatCompletionMessageParam],
            tools: list[dict[str, Any]] | None = None,
            image_order: list[int] | None = None,
            **kwargs,
        ) -> str | list[int]:
            # The generic renderer's conversation has already flattened text
            # parts with '\n'. V4.1 encodes the original messages with '\n\n'.
            conversation = _normalize_messages(messages)
            if tools:
                _reject_reserved_tokens_in_tools(tools)
                system = next((m for m in conversation if m["role"] == "system"), None)
                if system is None:
                    system = {"role": "system", "content": ""}
                    conversation.insert(0, system)
                system["tools"] = tools

            thinking = bool(kwargs.get("thinking") or kwargs.get("enable_thinking"))
            if "thinking" not in kwargs and "enable_thinking" not in kwargs:
                thinking = True
            effort = kwargs.get("reasoning_effort")
            if effort == "none":
                thinking = False
                effort = None
            if effort is None:
                effort = "high"
            if not (
                (type(effort) is int and 1 <= effort <= 100)
                or (isinstance(effort, str) and effort in REASONING_EFFORT_MAPPINGS)
            ):
                raise ValueError(
                    "DeepSeek V4.1 reasoning_effort must be low, high, xhigh, max, "
                    "or an integer within [1, 100] in chat_template_kwargs"
                )

            prompt = encode_messages(
                conversation,
                thinking_mode="thinking" if thinking else "chat",
                drop_thinking=kwargs.get("drop_thinking", True),
                reasoning_effort=effort,
                image_order=image_order,
            )
            if kwargs.get("tokenize", True):
                tokenizer_kwargs = {
                    key: kwargs[key]
                    for key in ("truncation", "max_length")
                    if key in kwargs
                }
                return self.encode(prompt, add_special_tokens=False, **tokenizer_kwargs)
            return prompt

        def num_special_tokens_to_add(self) -> int:
            return len(self.encode(""))

        def get_added_vocab(self) -> dict[str, int]:
            return added_vocab.copy()

        def __reduce__(self):
            return get_deepseek_v41_tokenizer, (tokenizer,)

    _DeepseekV41Tokenizer.__name__ = f"DSV41{tokenizer.__class__.__name__}"
    wrapped.__class__ = _DeepseekV41Tokenizer
    return wrapped


class DeepseekV41Tokenizer(TokenizerLike):
    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> HfTokenizer:
        tokenizer = TokenizersBackend.from_pretrained(*args, **kwargs)
        return get_cached_tokenizer(get_deepseek_v41_tokenizer(tokenizer))
