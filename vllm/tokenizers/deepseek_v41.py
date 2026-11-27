# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import functools
import json
import uuid
from collections.abc import Callable, Iterator
from typing import Any

import regex as re
import tokenizers
from transformers import TokenizersBackend

from vllm.renderers.chat_utils import ChatCompletionMessageParam

from .deepseek_v41_encoding import (
    IMAGE_PLACEHOLDER,
    REASONING_EFFORT_MAPPINGS,
    encode_messages,
    parse_tool_call_arguments,
)
from .hf import HfTokenizer, get_cached_tokenizer
from .protocol import TokenizerLike

# Client-supplied text is sanitized by default, as in deepseek-recipe: every
# added token in it (control tokens such as '<think>', '</think>',
# '<｜User｜>', the image placeholder, ...) is encoded as plain text, never as the
# special token. This covers message text of every role (replayed assistant
# turns included), tool call names and arguments, tool definitions and the
# response format schema. The model's own delimiters are never part of replayed
# text, so a match is a mention (or an injection), not structure. Images must
# be provided as image content parts.
#
# Added tokens are matched in plain text by the tokenizer, so escaped text is
# replaced by per-call sentinels before the prompt is rendered, and each
# sentinel is encoded as ordinary text afterwards.
_ESCAPE_OPEN = "\ue000"
_ESCAPE_CLOSE = "\ue001"
# Message keys the encoder reads that no client may set.
_ENCODER_KEYS = ("content_blocks", "updated_tools", "wo_eos", "_tool_references")

# As in deepseek-recipe, the only special tokens that assistant content and
# reasoning content can contain are the grounding tokens; every other added
# token there is plain text.
_ASSISTANT_SPECIAL_TOKENS = frozenset(
    {
        "<｜ref｜>",
        "<｜/ref｜>",
        "<｜box｜>",
        "<｜/box｜>",
        "<｜point｜>",
        "<｜/point｜>",
        "<｜polygon｜>",
        "<｜/polygon｜>",
    }
)


def _make_escape(
    added_tokens_re: re.Pattern[str] | None, nonce: str
) -> Callable[[Any], Any]:
    prefix = f"{_ESCAPE_OPEN}{nonce}:"

    def escape(text: Any) -> Any:
        if text is not None and not isinstance(text, str):
            raise ValueError(f"DeepSeek V4.1 expects text, got {type(text).__name__}")
        if added_tokens_re is None or text is None:
            return text
        return added_tokens_re.sub(
            lambda m: f"{prefix}{m.group(0)}{_ESCAPE_CLOSE}", text
        )

    return escape


def _escape_deep(value: Any, escape: Callable[[Any], Any]) -> Any:
    """Escape every key and string of a JSON-like value."""
    if isinstance(value, dict):
        return {
            _escape_deep(k, escape): _escape_deep(v, escape) for k, v in value.items()
        }
    if isinstance(value, list):
        return [_escape_deep(v, escape) for v in value]
    return escape(value) if isinstance(value, str) else value


def _normalize_messages(
    messages: list[ChatCompletionMessageParam],
    escape: Callable[[Any], Any],
    escape_assistant: Callable[[Any], Any],
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
        for key in _ENCODER_KEYS:
            message.pop(key, None)
        image_indices = message["_image_indices"] = []
        role = message.get("role")
        if role not in ("system", "developer", "user", "assistant", "tool"):
            raise ValueError(f"Invalid role: {role}")
        if "reasoning" in message:
            message["reasoning_content"] = message["reasoning"]

        # Assistant text keeps the grounding tokens, as in deepseek-recipe.
        text_escape = escape_assistant if role == "assistant" else escape
        if "reasoning_content" in message:
            message["reasoning_content"] = text_escape(message["reasoning_content"])
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            tool_references = []
            for block in content:
                part_type = block.get("type")
                if part_type in ("text", "input_text", "output_text"):
                    parts.append(text_escape(block.get("text", "")))
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
                    parts.append(escape(block.get("name", "")))
                    tool_references.append(block.get("name", ""))
                else:
                    raise ValueError(
                        "DeepSeek V4.1 supports text and image content only; "
                        f"got {part_type!r}"
                    )
            message["content"] = "\n\n".join(parts)
            if tool_references:
                message["_tool_references"] = tool_references
        elif content is not None:
            message["content"] = text_escape(content)
        # Tool call names and arguments are plain text for every token.
        for tool_call in message.get("tool_calls") or []:
            function = tool_call["function"]
            function["name"] = escape(function.get("name"))
            # The same parsing as the encoder, so what it renders is escaped.
            function["arguments"] = _escape_deep(
                parse_tool_call_arguments(function.get("arguments")), escape
            )

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


def _compile_added_tokens_pattern(tokens: list[str]) -> re.Pattern[str] | None:
    """Match the longest added token at each position using shared prefixes."""
    if not tokens:
        return None

    trie: dict[str, Any] = {}
    for token in tokens:
        node = trie
        for char in token:
            node = node.setdefault(char, {})
        node[""] = {}  # End of a token, even if the node has children.

    def build(node: dict[str, Any]) -> str:
        branches = [
            re.escape(char) + build(child) for char, child in node.items() if char
        ]
        if not branches:
            return ""
        if "" in node:
            # Prefer a longer token before falling back to this prefix.
            return f"(?:{'|'.join(branches)})?"
        return branches[0] if len(branches) == 1 else f"(?:{'|'.join(branches)})"

    return re.compile(build(trie))


def get_deepseek_v41_tokenizer(tokenizer: HfTokenizer) -> HfTokenizer:
    """Wrap an HF tokenizer with the V4.1 prompt encoder."""
    wrapped = copy.copy(tokenizer)
    added_vocab = tokenizer.get_added_vocab()

    added_tokens_re = _compile_added_tokens_pattern(list(added_vocab))
    assistant_tokens_re = _compile_added_tokens_pattern(
        [t for t in added_vocab if t not in _ASSISTANT_SPECIAL_TOKENS]
    )

    @functools.cache
    def plain_tokenizer() -> tokenizers.Tokenizer:
        # A copy of the backend without the added vocabulary, so reserved
        # tokens in a text are encoded as the ordinary text they spell, as in
        # deepseek-recipe.
        backend = getattr(tokenizer, "_tokenizer", None)
        if backend is None:
            raise RuntimeError(
                "DeepSeek V4.1 needs a tokenizers-backed tokenizer to encode "
                "reserved tokens in text as plain text"
            )
        spec = json.loads(backend.to_str())
        spec["added_tokens"] = []
        spec["truncation"] = None
        spec["padding"] = None
        return tokenizers.Tokenizer.from_str(json.dumps(spec))

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
            escape = _make_escape(added_tokens_re, nonce)
            conversation = _normalize_messages(
                messages, escape, _make_escape(assistant_tokens_re, nonce)
            )
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
            # Tool definitions and the response format schema are plain text.
            for message in conversation:
                for key in ("tools", "updated_tools", "response_format"):
                    if key in message:
                        message[key] = _escape_deep(message[key], escape)

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
            tokenizer_kwargs = {
                key: kwargs[key]
                for key in ("truncation", "max_length")
                if key in kwargs
            }
            if nonce not in prompt:
                if not kwargs.get("tokenize", True):
                    return prompt
                return self.encode(prompt, add_special_tokens=False, **tokenizer_kwargs)

            sentinel_re = re.compile(
                f"{_ESCAPE_OPEN}{nonce}:(.*?){_ESCAPE_CLOSE}", re.DOTALL
            )
            if nonce in sentinel_re.sub("", prompt):
                # Defensive: the escape markers must reach the rendered prompt
                # verbatim. If rendering altered them (e.g. the ensure_ascii
                # fallback of to_json writes them as text), the reserved token
                # would be special again, so reject the prompt.
                raise ValueError(
                    "DeepSeek V4.1 could not render the escaped text of a "
                    "message, tool or response format as plain text"
                )
            # A text prompt spells the escaped tokens, so whatever tokenizes it
            # again matches them as special tokens: ask for ids to keep them
            # plain text.
            if not kwargs.get("tokenize", True):
                return sentinel_re.sub(lambda m: m.group(1), prompt)

            # As in deepseek-recipe, only the added tokens outside a sentinel
            # are special tokens; every stretch of text between two of them,
            # escaped tokens included, is encoded as a whole without the added
            # vocabulary.
            assert added_tokens_re is not None
            regions: list[str] = []
            special_ids: list[int] = []
            # The plain text pieces since the last special token.
            run: list[str] = []
            # Splitting on the sentinel pattern alternates the text outside a
            # sentinel (even indices) with the escaped text (odd indices).
            for index, part in enumerate(sentinel_re.split(prompt)):
                if index % 2:
                    # Escaped client text: the reserved token it holds is
                    # plain text, so it joins the run as is.
                    run.append(part)
                    continue
                # Text outside the sentinels (the chat template, image
                # placeholders): an added token found here is a real special
                # token, as the client ones are inside a sentinel.
                pos = 0
                for m in added_tokens_re.finditer(part):
                    # The text before the special token ends the run.
                    run.append(part[pos : m.start()])
                    regions.append("".join(run))
                    run.clear()
                    special_ids.append(added_vocab[m.group(0)])
                    pos = m.end()
                # The rest of the part starts or continues a run, which goes on
                # in the next odd part (an escaped token) if there is one.
                run.append(part[pos:])
            # The text after the last special token.
            regions.append("".join(run))
            # Encode every run as a whole, so that a merge across the pieces
            # (e.g. the text next to an escaped token) is kept.
            encodings = plain_tokenizer().encode_batch_fast(
                regions, add_special_tokens=False
            )
            ids: list[int] = []
            for encoding, special_id in zip(encodings, special_ids):
                ids.extend(encoding.ids)
                ids.append(special_id)
            ids.extend(encodings[-1].ids)
            max_length = tokenizer_kwargs.get("max_length")
            if tokenizer_kwargs.get("truncation") and max_length is not None:
                if getattr(self, "truncation_side", "right") == "left":
                    ids = ids[max(len(ids) - max_length, 0) :]
                else:
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
