# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
from collections.abc import Iterator
from typing import Any

from transformers import TokenizersBackend

from vllm.renderers.chat_utils import ChatCompletionMessageParam

from .deepseek_v41_encoding import (
    IMAGE_PLACEHOLDER,
    REASONING_EFFORT_MAPPINGS,
    encode_messages,
    find_last_user_index,
)
from .hf import HfTokenizer, get_cached_tokenizer
from .protocol import TokenizerLike


def _reject_image_placeholder(text: Any) -> None:
    if isinstance(text, str) and IMAGE_PLACEHOLDER in text:
        raise ValueError(
            f"Message text contains image special token {IMAGE_PLACEHOLDER!r}; "
            "provide images as image content parts."
        )


def _normalize_messages(
    messages: list[ChatCompletionMessageParam],
) -> list[dict[str, Any]]:
    # vLLM additions to the reference: Responses API part types, image_pil,
    # tool_reference, and ValueError (400) instead of the reference's asserts.
    # A tool_reference renders as the tool name; `_add_updated_tool_schemas`
    # lists the referenced tools after the user turn.
    result: list[dict[str, Any]] = []
    for original_message in messages:
        content = original_message.get("content")
        if isinstance(content, Iterator):
            original_message["content"] = list(content)
        result.append(dict(copy.deepcopy(original_message)))
    image_idx = 0
    for message in result:
        image_indices = message["_image_indices"] = []
        role = message.get("role")
        if role not in ("system", "developer", "user", "assistant", "tool"):
            raise ValueError(f"Invalid role: {role}")
        if "reasoning" in message:
            message["reasoning_content"] = message["reasoning"]
        _reject_image_placeholder(message.get("reasoning_content"))
        content = message.get("content")
        if isinstance(content, str):
            _reject_image_placeholder(content)
        elif isinstance(content, list):
            parts = []
            tool_references = []
            for block in content:
                part_type = block.get("type")
                if part_type in ("text", "input_text", "output_text"):
                    _reject_image_placeholder(block.get("text"))
                    parts.append(block.get("text", ""))
                elif part_type in (
                    "image_url",
                    "input_image",
                    "image_pil",
                    "image_embeds",
                ):
                    parts.append(IMAGE_PLACEHOLDER)
                    image_indices.append(image_idx)
                    image_idx += 1
                elif part_type == "tool_reference":
                    _reject_image_placeholder(block.get("name"))
                    parts.append(block.get("name", ""))
                    tool_references.append(block.get("name", ""))
                else:
                    raise ValueError(
                        "DeepSeek V4.1 supports text and image content only; "
                        f"got {part_type!r}"
                    )
            message["content"] = "\n\n".join(parts)
            if tool_references:
                message["_tool_references"] = tool_references

    return result


def _is_deferred(tool: dict[str, Any]) -> bool:
    return bool(
        tool.get("defer_loading") or tool.get("function", {}).get("defer_loading")
    )


def _tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
    function = {
        k: v
        for k, v in tool["function"].items()
        if k != "defer_loading" and v is not None
    }
    return {"type": "function", "function": function}


def _add_updated_tool_schemas(
    conversation: list[dict[str, Any]], tools: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    """Like deepseek-recipe, follow each user turn whose tool results reference
    tools with a system turn listing those tools' schemas, in reference order."""
    tools_by_name = {tool["function"]["name"]: tool for tool in tools or []}
    result: list[dict[str, Any]] = []
    referenced: list[dict[str, Any]] = []
    for message, next_message in zip(conversation, [*conversation[1:], {}]):
        result.append(message)
        names = message.pop("_tool_references", [])
        # Without tools (e.g. tool_choice "none"), references stay unlisted.
        for name in names if tools else []:
            if name not in tools_by_name:
                raise ValueError(f"tool_reference {name!r} not found in tools")
            referenced.append(_tool_schema(tools_by_name[name]))
        if referenced and next_message.get("role") not in ("user", "tool"):
            result.append(
                {"role": "system", "content": "", "updated_tools": referenced}
            )
            referenced = []
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
            conversation = _add_updated_tool_schemas(conversation, tools)
            # As in deepseek-recipe, deferred tools load through tool search.
            listed = [_tool_schema(t) for t in tools or [] if not _is_deferred(t)]
            if tools and not listed:
                raise ValueError(
                    "At least one tool must have defer_loading=false; "
                    "all tools cannot be deferred."
                )
            if listed:
                system = next(
                    (
                        m
                        for m in conversation
                        if m["role"] == "system" and "updated_tools" not in m
                    ),
                    None,
                )
                if system is None:
                    system = {"role": "system", "content": ""}
                    conversation.insert(0, system)
                system["tools"] = listed

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

            if (
                thinking
                and kwargs.get("drop_thinking", True)
                and not any(message.get("tools") for message in conversation)
            ):
                last_user_index = find_last_user_index(conversation)
                for message in conversation[: max(last_user_index, 0)]:
                    if message["role"] not in ("system", "developer"):
                        break
                    if message["role"] == "developer" and message.get("content"):
                        message["role"] = "system"

            prompt = encode_messages(
                conversation,
                thinking_mode="thinking" if thinking else "chat",
                drop_thinking=kwargs.get("drop_thinking", True),
                reasoning_effort=effort,
                image_order=image_order,
                add_generation_prompt=kwargs.get("add_generation_prompt", True),
                continue_final_message=kwargs.get("continue_final_message", False),
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
    def from_pretrained(cls, *args, **kwargs) -> HfTokenizer:  # type: ignore[override]
        tokenizer = TokenizersBackend.from_pretrained(*args, **kwargs)
        return get_cached_tokenizer(get_deepseek_v41_tokenizer(tokenizer))
