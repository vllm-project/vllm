# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 / V4.1 prompt encoding through the official deepseek-recipe
package.

deepseek-recipe ships the reference renderer and the segmented parallel
tokenizer that DeepSeek's checkpoints are trained with. The ``recipe``
tokenize backend prefers it for DeepSeek models and keeps vLLM's own
renderer only as a fallback: the two do not always agree (the reference
renderer emits its own system/tool boundaries and injects the V4.1
``Reasoning Effort`` system turn in thinking mode).

A rendered prompt interleaves control tokens with plain text, and prompt
encoding tokenizes the two differently:

* a control token (one ``RenderedPrompt.special_ranges`` span) is a single
  added token of the full vocabulary;
* plain text is encoded *without* the added vocabulary, so a control-token
  spelling that a user typed inside a message stays ordinary text.

``Tokenizer.encode(..., add_special_tokens=False)`` only disables the
post-processor; added tokens are still matched greedily, and they occur in
message text. :func:`_plain_tokenizer` therefore rebuilds the same
``tokenizer.json`` with an empty ``added_tokens`` list: it drops the added
tokens' matching rules only, while ``model.vocab`` -- and with it every
token id -- stays byte-identical.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class SegmentedEncodingError(RuntimeError):
    """A rendered prompt cannot be tokenized span by span.

    The condition is recoverable: the reference encoder tokenizes the whole
    conversation in one pass, which is the authoritative result.
    """


def load_deepseek_recipe():
    """Import the optional deepseek-recipe package, or None when absent."""
    try:
        import deepseek_recipe
    except ImportError:
        return None
    return deepseek_recipe


def _plain_tokenizer(raw_json: str):
    """A tokenizer that encodes plain text without the added vocabulary.

    ``model.vocab`` is preserved on purpose, so every token id is unchanged;
    only the added tokens' matching rules are removed.
    """
    from tokenizers import Tokenizer

    data = json.loads(raw_json)
    added = data.get("added_tokens")
    if not isinstance(added, list):
        raise ValueError("tokenizer.json has no `added_tokens` list")
    data["added_tokens"] = []
    return Tokenizer.from_str(json.dumps(data, ensure_ascii=False))


def _validate_ranges(prompt: str, ranges: list[tuple[int, int]]) -> None:
    """Reject ranges that cannot describe code-point spans of ``prompt``."""
    cursor = 0
    for start, end in ranges:
        if not cursor <= start < end <= len(prompt):
            raise SegmentedEncodingError(
                f"special range {(start, end)} is out of bounds or out of order"
            )
        cursor = end


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    # Flatten OpenAI content parts to text; media parts are handled by the
    # caller's fallback path.
    return "".join(
        part.get("text", "") if isinstance(part, dict) else str(part)
        for part in content
    )


def _to_recipe_messages(dr: Any, messages: list[dict[str, Any]]):
    """Convert vLLM OpenAI messages to deepseek-recipe message objects."""
    converted = []
    for message in messages:
        role = message.get("role")
        content = _content_to_text(message.get("content"))
        if role == "system":
            converted.append(dr.SystemMessage(content))
        elif role == "assistant":
            tool_calls = [
                dr.ToolCall(
                    call.get("id", ""),
                    call["function"]["name"],
                    call["function"]["arguments"]
                    if isinstance(call["function"]["arguments"], str)
                    else json.dumps(call["function"]["arguments"], ensure_ascii=False),
                )
                for call in (message.get("tool_calls") or [])
            ]
            converted.append(
                dr.AssistantMessage(content, tool_calls=tool_calls or None)
            )
        elif role == "tool":
            converted.append(
                dr.ToolMessage(content, tool_call_id=message.get("tool_call_id"))
            )
        else:
            converted.append(dr.UserMessage(content))
    return converted


def _to_recipe_tools(dr: Any, tools: list[dict[str, Any]] | None):
    """Convert OpenAI tool definitions to deepseek-recipe tool definitions."""
    converted = []
    for tool in tools or []:
        function = tool.get("function") or {}
        if not function.get("name"):
            continue
        converted.append(
            dr.ToolDefinition(
                function["name"],
                function.get("description"),
                function.get("parameters"),
                function.get("strict"),
            )
        )
    return converted


class DeepSeekRecipeEncoder:
    """Official deepseek-recipe renderer + segmented parallel tokenizer."""

    _cache: dict[int, DeepSeekRecipeEncoder] = {}

    def __init__(self, backend_tokenizer: Any, spec: str) -> None:
        dr = load_deepseek_recipe()
        if dr is None:
            raise ImportError("deepseek-recipe is not installed")
        self._dr = dr
        self._backend_tokenizer = backend_tokenizer
        raw_json = backend_tokenizer.to_str()
        # The reference tokenizer: full vocabulary, used for control tokens
        # and for the authoritative whole-conversation encoding.
        self._tokenizer = dr.Tokenizer.from_str(raw_json)
        # Plain text regions: same vocabulary, no added-token matching.
        self._text_tokenizer = _plain_tokenizer(raw_json)
        # The added tokens' matching rules are gone, their ids are not: the
        # plain tokenizer must still agree with the original on every token of
        # the base vocabulary.
        self._added_tokens = backend_tokenizer.get_added_tokens_decoder()
        added_contents = {token.content for token in self._added_tokens.values()}
        plain_vocab = self._text_tokenizer.get_vocab()
        for token, token_id in backend_tokenizer.get_vocab().items():
            if token in added_contents:
                continue
            # Every base-vocabulary token must keep its id: only the added
            # tokens' matching rules may have been dropped.
            if plain_vocab.get(token) != token_id:
                raise ValueError("plain tokenizer changed the base vocabulary")
        encoding = (
            dr.DeepseekV4Encoding() if spec == "dsv4" else dr.DeepseekV41Encoding()
        )
        self._encoding = encoding.with_tokenizer(self._tokenizer)

    @classmethod
    def get(cls, backend_tokenizer: Any, spec: str) -> DeepSeekRecipeEncoder | None:
        if load_deepseek_recipe() is None:
            logger.warning_once(
                "--tokenize-backend recipe needs the deepseek-recipe package for "
                "DeepSeek models; install it with `pip install deepseek-recipe`."
            )
            return None
        key = id(backend_tokenizer)
        encoder = cls._cache.get(key)
        if encoder is None:
            try:
                encoder = cls(backend_tokenizer, spec)
            except Exception as exc:  # pragma: no cover - package drift
                logger.warning_once("deepseek-recipe encoder unavailable: %r", exc)
                return None
            cls._cache[key] = encoder
        return encoder

    def render(
        self,
        messages: list[dict[str, Any]],
        *,
        thinking: bool,
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[str, list[tuple[int, int]], Any]:
        """Render a conversation with the official recipe encoder.

        Returns the prompt, its special-token ranges (*code-point* offsets,
        the unit ``RenderedPrompt`` exposes to Python) and the recipe
        ``Conversation`` that produced them. The conversation is per-request
        data: it is only used by :meth:`encode` as the authoritative fallback.
        """
        conversation = self._dr.Conversation(
            messages=_to_recipe_messages(self._dr, messages),
            thinking_mode=thinking,
            tools=_to_recipe_tools(self._dr, tools) or None,
        )
        rendered = self._encoding.render_conversation(conversation)
        return rendered.prompt, list(rendered.special_ranges), conversation

    def encode(
        self,
        prompt: str,
        ranges: list[tuple[int, int]],
        conversation: Any,
    ) -> list[int]:
        """Token ids for an officially rendered prompt.

        The special ranges are interleaved with the plain regions, which are
        encoded as one ``encode_batch_fast`` batch (parallel across CPU
        threads) and each control span becomes exactly one token id. When the
        spans cannot be separated faithfully -- for example a control token the
        served tokenizer does not know -- the reference encoder handles the
        whole conversation instead.
        """
        try:
            return self._encode_segmented(prompt, ranges)
        except SegmentedEncodingError as exc:
            logger.warning_once(
                "deepseek-recipe prompt cannot be encoded span by span (%s); "
                "falling back to the reference encoder.",
                exc,
            )
        return self._encoding.encode(conversation)

    def _encode_segmented(
        self, prompt: str, ranges: list[tuple[int, int]]
    ) -> list[int]:
        _validate_ranges(prompt, ranges)
        regions: list[str] = []
        cursor = 0
        for start, end in ranges:
            regions.append(prompt[cursor:start])
            cursor = end
        regions.append(prompt[cursor:])
        encodings = self._text_tokenizer.encode_batch_fast(
            regions, add_special_tokens=False
        )
        token_ids: list[int] = []
        for index, (start, end) in enumerate(ranges):
            token_ids.extend(encodings[index].ids)
            token_ids.append(self._control_token_id(prompt[start:end]))
        token_ids.extend(encodings[len(ranges)].ids)
        return token_ids

    def _control_token_id(self, spelling: str) -> int:
        """Id of one control-token spelling, or raise to fall back."""
        try:
            ids = self._tokenizer.encode(spelling)
        except Exception as exc:  # pragma: no cover - package drift
            raise SegmentedEncodingError(f"cannot encode {spelling!r}") from exc
        if len(ids) != 1:
            raise SegmentedEncodingError(f"{spelling!r} is not a single token")
        if ids[0] not in self._added_tokens:
            # Not an added-vocabulary token: the served tokenizer does not know
            # this control token, so the reference encoder must decide.
            raise SegmentedEncodingError(f"{spelling!r} is not an added token")
        return ids[0]
