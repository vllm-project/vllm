# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import uuid
from collections.abc import Callable, Iterator
from typing import Any

import regex as re
from transformers import TokenizersBackend

from vllm.renderers.chat_utils import ChatCompletionMessageParam

from .deepseek_v41_encoding import (
    IMAGE_PLACEHOLDER,
    REASONING_EFFORT_MAPPINGS,
    encode_messages,
)
from .hf import HfTokenizer, get_cached_tokenizer
from .protocol import TokenizerLike

# Message text is sanitized by default: every added token in it (control tokens
# such as '<think>', '</think>', '<｜User｜>', the image placeholder, ...) is
# encoded as plain text, never as the special token, for every role and replayed
# assistant turns. The model's own delimiters are never part of replayed text,
# so a match is a mention (or an injection), not structure. Images must be
# provided as image content parts.
#
# Added tokens are matched in plain text by the tokenizer, so escaped text is
# replaced by per-call sentinels before the prompt is rendered, and each
# sentinel is encoded as ordinary text afterwards.
_ESCAPE_OPEN = "\ue000"
_ESCAPE_CLOSE = "\ue001"


def _make_escape(
    added_tokens_re: re.Pattern[str] | None, nonce: str
) -> Callable[[Any], Any]:
    prefix = f"{_ESCAPE_OPEN}{nonce}:"

    def escape(text: Any) -> Any:
        if added_tokens_re is None or not isinstance(text, str):
            return text
        return added_tokens_re.sub(
            lambda m: f"{prefix}{m.group(0)}{_ESCAPE_CLOSE}", text
        )

    return escape


def _normalize_messages(
    messages: list[ChatCompletionMessageParam],
    escape: Callable[[Any], Any] | None = None,
) -> list[dict[str, Any]]:
    # vLLM additions to the reference: Responses API part types, image_pil,
    # tool_reference, and ValueError (400) instead of the reference's asserts.
    escape = escape or (lambda text: text)
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

        if "reasoning_content" in message:
            message["reasoning_content"] = escape(message["reasoning_content"])
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = escape(content)
        elif isinstance(content, list):
            parts = []
            for block in content:
                part_type = block.get("type")
                if part_type in ("text", "input_text", "output_text"):
                    parts.append(escape(block.get("text", "")))
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
                    # The system prompt already lists every tool, deferred
                    # ones included, so the name is all the model needs.
                    parts.append(escape(block.get("name", "")))
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
    added_tokens_re = (
        re.compile(
            "|".join(re.escape(t) for t in sorted(added_vocab, key=len, reverse=True))
        )
        if added_vocab
        else None
    )
    plain_token_ids: dict[str, list[int]] = {}

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
            nonce = uuid.uuid4().hex
            conversation = _normalize_messages(
                messages, _make_escape(added_tokens_re, nonce)
            )
            if tools:
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
                add_generation_prompt=kwargs.get("add_generation_prompt", True),
                continue_final_message=kwargs.get("continue_final_message", False),
            )
            sentinel_re = re.compile(
                f"{_ESCAPE_OPEN}{nonce}:(.*?){_ESCAPE_CLOSE}", re.DOTALL
            )
            if not kwargs.get("tokenize", True):
                return sentinel_re.sub(lambda m: m.group(1), prompt)

            tokenizer_kwargs = {
                key: kwargs[key]
                for key in ("truncation", "max_length")
                if key in kwargs
            }
            if not sentinel_re.search(prompt):
                return self.encode(prompt, add_special_tokens=False, **tokenizer_kwargs)

            def encode_plain(token: str) -> list[int]:
                # Encode an added-token string as ordinary text by splitting it
                # so the tokenizer cannot match the added token.
                if token not in plain_token_ids:
                    ids = self.encode(
                        token[:1], add_special_tokens=False
                    ) + self.encode(token[1:], add_special_tokens=False)
                    if added_vocab.get(token) in ids:
                        ids = [
                            i
                            for c in token
                            for i in self.encode(c, add_special_tokens=False)
                        ]
                    plain_token_ids[token] = ids
                return plain_token_ids[token]

            ids: list[int] = []
            pos = 0
            for m in sentinel_re.finditer(prompt):
                if m.start() > pos:
                    ids += self.encode(
                        prompt[pos : m.start()], add_special_tokens=False
                    )
                ids += encode_plain(m.group(1))
                pos = m.end()
            if pos < len(prompt):
                ids += self.encode(prompt[pos:], add_special_tokens=False)
            max_length = tokenizer_kwargs.get("max_length")
            if tokenizer_kwargs.get("truncation") and max_length is not None:
                ids = ids[:max_length]
            return ids

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
