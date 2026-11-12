# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 / V4.1 prompt encoding through the official deepseek-recipe
package.

deepseek-recipe ships the reference renderer and the segmented parallel
tokenizer that DeepSeek's checkpoints are trained with. The ``recipe``
tokenize backend prefers it for DeepSeek models and keeps vLLM's own
renderer only as a fallback: the two do not always agree (the reference
renderer wraps system messages in ``<?System?>`` and injects the V4.1
``Reasoning Effort`` system turn in thinking mode).
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# deepseek-recipe's own names for the two supported checkpoints.
MODEL_NAMES = {"dsv4": "DeepSeek-V4-Pro", "dsv41": "DeepSeek-V4.1-Flash"}


def load_deepseek_recipe():
    """Import the optional deepseek-recipe package, or None when absent."""
    try:
        import deepseek_recipe
    except ImportError:
        return None
    return deepseek_recipe


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
    """vLLM OpenAI messages -> deepseek-recipe message objects."""
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
    """vLLM OpenAI tool definitions -> deepseek-recipe tool definitions."""
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

    _cache: dict[int, "DeepSeekRecipeEncoder"] = {}

    def __init__(self, backend_tokenizer: Any, spec: str) -> None:
        dr = load_deepseek_recipe()
        if dr is None:
            raise ImportError("deepseek-recipe is not installed")
        self._dr = dr
        self._spec = spec
        self._backend_tokenizer = backend_tokenizer
        self._tokenizer = dr.Tokenizer.from_str(backend_tokenizer.to_str())
        encoding = (
            dr.DeepseekV4Encoding() if spec == "dsv4" else dr.DeepseekV41Encoding()
        )
        self._encoding = encoding.with_tokenizer(self._tokenizer)
        self._added_ids = {
            token.content: token_id
            for token_id, token in backend_tokenizer.get_added_tokens_decoder().items()
            if token.content
        }

    @classmethod
    def get(
        cls, backend_tokenizer: Any, spec: str
    ) -> "DeepSeekRecipeEncoder | None":
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
    ) -> tuple[str, list[tuple[int, int]]]:
        """Render a conversation with the official recipe encoder.

        Returns the prompt string and its special-token ranges (code-point
        offsets, the same offsets ``RenderedPrompt`` exposes in Python).
        """
        conversation = self._dr.Conversation(
            messages=_to_recipe_messages(self._dr, messages),
            thinking_mode=thinking,
            tools=_to_recipe_tools(self._dr, tools) or None,
        )
        rendered = self._encoding.render_conversation(conversation)
        return rendered.prompt, list(rendered.special_ranges)

    def encode(self, prompt: str, ranges: list[tuple[int, int]]) -> list[int] | None:
        """Token ids for an officially rendered prompt.

        The special ranges are interleaved with the plain regions encoded as
        one ``encode_batch_fast`` batch (parallel across CPU threads) -- the
        semantics of deepseek-recipe's ``TokenizerEncoder::encode_ids``.
        """
        backend = self._backend_tokenizer
        regions: list[str] = []
        cursor = 0
        for start, end in ranges:
            regions.append(prompt[cursor:start])
            cursor = end
        regions.append(prompt[cursor:])
        encodings = backend.encode_batch_fast(regions, add_special_tokens=False)
        token_ids: list[int] = []
        for index, (start, end) in enumerate(ranges):
            token_ids.extend(encodings[index].ids)
            spelling = prompt[start:end]
            token_id = self._added_ids.get(spelling)
            if token_id is None:
                logger.warning_once(
                    "deepseek-recipe special spelling %r is not in the added "
                    "vocabulary; falling back to builtin encoding.",
                    spelling,
                )
                return None
            token_ids.append(token_id)
        token_ids.extend(encodings[len(ranges)].ids)
        return token_ids
