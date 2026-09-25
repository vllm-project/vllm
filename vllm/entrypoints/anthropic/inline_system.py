# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Placement of inline ``role: "system"`` messages from the Messages API.

Clients such as Claude Code append a new system message to ``messages`` on
almost every turn. Hoisting it into the leading system block changes the head
of the prompt each turn and defeats prefix caching, so inline system messages
are either kept in place (when the renderer marks them as a distinct system
turn) or folded into a neighbouring message. Both modes only ever change the
end of the conversation, so each request renders as an extension of the
previous one.

Folding never adds a user turn after tool results, because many templates keep
reasoning only after the last user turn, and never places a message between
an assistant's ``tool_use`` and the ``tool_result`` blocks answering it.
"""

import asyncio
import copy
from typing import TYPE_CHECKING, Any

from vllm.entrypoints.anthropic.protocol import (
    AnthropicContentBlock,
    AnthropicCountTokensRequest,
    AnthropicInlineSystemMode,
    AnthropicInlineSystemOption,
    AnthropicMessage,
    AnthropicMessagesRequest,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.logger import init_logger
from vllm.renderers.inputs.preprocess import extract_prompt_components

if TYPE_CHECKING:
    from vllm.renderers.online_renderer import OnlineRenderer

logger = init_logger(__name__)

# Claude Code's per-request attribution hash, which would defeat prefix caching.
_BILLING_HEADER = "x-anthropic-billing-header"


def _join(*texts: str) -> str:
    return "\n\n".join(t for t in texts if t)


def _blocks_text(blocks: list[AnthropicContentBlock]) -> str:
    return "".join(
        b.text
        for b in blocks
        if b.type == "text" and b.text and not b.text.startswith(_BILLING_HEADER)
    )


def system_text(message: AnthropicMessage) -> str:
    """Text of a system message, without Claude Code's billing header."""
    if isinstance(message.content, str):
        return "" if message.content.startswith(_BILLING_HEADER) else message.content
    return _blocks_text(message.content)


def system_prompt_text(
    request: AnthropicMessagesRequest | AnthropicCountTokensRequest,
) -> str:
    """Text of the top-level ``system`` field, without the billing header."""
    system = request.system
    if system is None or isinstance(system, str):
        return system or ""
    return _blocks_text(system)


def _get(item: Any, key: str) -> Any:
    return item.get(key) if isinstance(item, dict) else getattr(item, key)


def _add_text(content: Any, text: str, *, at_end: bool, make_item) -> Any:
    """Join ``text`` onto the text at one end of ``content``.

    ``content`` is a string, None, or a list of blocks (dicts or
    ``AnthropicContentBlock``). Text is merged into an adjacent text block
    rather than added as a new block, because some templates concatenate
    text blocks without a separator.
    """
    if content is None or isinstance(content, str):
        return _join(content or "", text) if at_end else _join(text, content or "")
    items = list(content)
    end = -1 if at_end else 0
    if items and _get(items[end], "type") == "text":
        old = _get(items[end], "text") or ""
        new = _join(old, text) if at_end else _join(text, old)
        item = items[end]
        items[end] = (
            {**item, "text": new}
            if isinstance(item, dict)
            else item.model_copy(update={"text": new})
        )
    else:
        items.insert(len(items) if at_end else 0, make_item(text))
    return items


def _text_block(text: str) -> AnthropicContentBlock:
    return AnthropicContentBlock(type="text", text=text)


def _append_to_user(message: AnthropicMessage, text: str) -> AnthropicMessage:
    """Append to the user's own text or, failing that, to the last tool result.

    Blocks other than ``tool_result`` become the user message that follows the
    tool messages, so appending there never adds a turn.
    """
    blocks = message.content
    if (
        isinstance(blocks, list)
        and blocks
        and all(b.type == "tool_result" for b in blocks)
    ):
        last = blocks[-1]
        folded = last.model_copy(
            update={
                "content": _add_text(
                    last.content,
                    text,
                    at_end=True,
                    make_item=lambda t: {"type": "text", "text": t},
                )
            }
        )
        content: Any = [*blocks[:-1], folded]
    else:
        own = (
            blocks
            if isinstance(blocks, str)
            else [b for b in blocks if b.type != "tool_result"]
        )
        content = _add_text(own, text, at_end=True, make_item=_text_block)
        if not isinstance(blocks, str):
            content = [b for b in blocks if b.type == "tool_result"] + content
    return message.model_copy(update={"content": content})


def _move_out_of_tool_runs(
    messages: list[AnthropicMessage],
) -> list[AnthropicMessage]:
    """Move system messages between a ``tool_use`` and its results after them."""
    out: list[AnthropicMessage] = []
    held: list[AnthropicMessage] = []
    awaiting_results = False
    for message in messages:
        if message.role == "system":
            (held if awaiting_results else out).append(message)
            continue
        if message.role == "assistant":
            out.extend(held)
            held.clear()
        out.append(message)
        if message.role == "user":
            out.extend(held)
            held.clear()
        awaiting_results = (
            message.role == "assistant"
            and isinstance(message.content, list)
            and any(b.type == "tool_use" for b in message.content)
        )
    out.extend(held)
    return out


def normalize_inline_system(
    request: AnthropicMessagesRequest | AnthropicCountTokensRequest,
    *,
    mode: AnthropicInlineSystemMode,
) -> tuple[str, list[AnthropicMessage]]:
    """Place inline system messages without rewriting earlier turns.

    Args:
        request: The Anthropic request.
        mode: ``"preserve"`` keeps inline system messages as system turns;
            ``"fold"`` merges their text into a neighbouring message.
            ``"preserve"`` folds too when the request has no system prompt,
            because some renderers (e.g. DeepSeek V4) render request-level
            tools at the first system message wherever it is.

    Returns:
        Text to append to the system prompt (from system messages that lead
        ``messages``), and the remaining messages.

    """
    messages = request.messages
    if all(m.role != "system" for m in messages):
        return "", messages

    num_leading = 0
    while num_leading < len(messages) and messages[num_leading].role == "system":
        num_leading += 1
    leading = _join(*(system_text(m) for m in messages[:num_leading]))
    rest = _move_out_of_tool_runs(messages[num_leading:])
    if mode == "preserve" and (system_prompt_text(request) or leading):
        return leading, rest

    out: list[AnthropicMessage] = []
    pending: list[str] = []  # system text after an assistant turn
    for message in rest:
        if message.role == "system":
            if not (text := system_text(message)):
                continue
            if out and out[-1].role == "user" and not pending:
                out[-1] = _append_to_user(out[-1], text)
            else:
                pending.append(text)
            continue
        if pending:
            text = _join(*pending)
            pending.clear()
            if message.role == "user":
                message = message.model_copy(
                    update={
                        "content": _add_text(
                            message.content, text, at_end=False, make_item=_text_block
                        )
                    }
                )
            else:
                out.append(AnthropicMessage(role="user", content=text))
        out.append(message)
    if pending:
        out.append(AnthropicMessage(role="user", content=_join(*pending)))
    return leading, out


# ---------------------------------------------------------------------------
# Renderer probe
# ---------------------------------------------------------------------------

# Each text starts and ends with a distinct word so that the inserted segment
# found by prefix/suffix matching starts exactly at the inserted message.
_SYS0 = "Alpha leading instructions"
_USER1 = "Bravo first question"
_ASSISTANT1 = "Charlie first answer"
_USER2 = "Delta second question"
_INLINE = "Echo inline note"
_REASONING = "Foxtrot prior reasoning"
_TOOL_CALL_ID = "probe0001"
_TOOL = {
    "type": "function",
    "function": {
        "name": "probe_tool",
        "description": "Probe tool.",
        "parameters": {"type": "object", "properties": {}},
    },
}
_SYSTEM = {"role": "system", "content": _SYS0}
_USER = {"role": "user", "content": _USER1}

# (name, conversation, index to insert the inline message at)
_PROBE_CASES: tuple[tuple[str, list[dict[str, Any]], int], ...] = (
    (
        "mid",
        [
            _SYSTEM,
            _USER,
            {"role": "assistant", "content": _ASSISTANT1},
            {"role": "user", "content": _USER2},
        ],
        3,
    ),
    (
        "after_tool",
        [
            _SYSTEM,
            _USER,
            {
                "role": "assistant",
                "reasoning": _REASONING,
                "tool_calls": [
                    {
                        "id": _TOOL_CALL_ID,
                        "type": "function",
                        "function": {"name": "probe_tool", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": _TOOL_CALL_ID, "content": "Golf output"},
        ],
        4,
    ),
    ("trailing", [_SYSTEM, _USER], 2),
)


def _insertion(base: list[int], new: list[int]) -> tuple[int, list[int]] | None:
    """Return ``(p, seg)`` if ``new == base[:p] + seg + base[p:]``."""
    p = 0
    while p < min(len(base), len(new)) and base[p] == new[p]:
        p += 1
    s = 0
    while s < min(len(base), len(new)) - p and base[-1 - s] == new[-1 - s]:
        s += 1
    if p + s != len(base):
        return None
    return p, new[p : len(new) - s]


def _without_reasoning(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in m.items() if k != "reasoning"} for m in messages]


class _Prober:
    def __init__(self, online_renderer: "OnlineRenderer") -> None:
        self.online_renderer = online_renderer
        self.tokenizer = online_renderer.renderer.get_tokenizer()
        self.special_ids = {
            *self.tokenizer.all_special_ids,
            *self.tokenizer.get_added_vocab().values(),
        }

    async def render(
        self, messages: list[dict[str, Any]], reasoning_effort: str | None
    ) -> list[int] | None:
        try:
            # Renderers may rewrite messages in place (e.g. tool call ids).
            request = ChatCompletionRequest(
                messages=copy.deepcopy(messages),  # type: ignore[arg-type]
                tools=[_TOOL],  # type: ignore[list-item]
                tool_choice=None,
                reasoning_effort=reasoning_effort,  # type: ignore[arg-type]
            )
            result = await self.online_renderer.render_chat(request)
        except Exception:
            logger.debug("Inline system probe render failed", exc_info=True)
            return None
        if isinstance(result, ErrorResponse):
            logger.debug("Inline system probe render failed: %s", result)
            return None
        _, engine_inputs = result
        if len(engine_inputs) != 1:
            return None
        components = extract_prompt_components(
            self.online_renderer.model_config, engine_inputs[0]
        )
        return components.token_ids

    async def insert(
        self,
        messages: list[dict[str, Any]],
        base: list[int],
        index: int,
        message: dict[str, Any],
        reasoning_effort: str | None,
    ) -> tuple[list[int], list[int] | None, tuple[int, list[int]] | None]:
        """Render ``message`` inserted at ``index``; return base, new, insertion."""
        with_message = messages[:index] + [message] + messages[index:]
        new = await self.render(with_message, reasoning_effort)
        found = None if new is None else _insertion(base, new)
        if new is not None and found is None:
            # Re-render once in case the renderer injects the current date.
            retry_base = await self.render(messages, reasoning_effort)
            if retry_base is not None and retry_base != base:
                base = retry_base
                new = await self.render(with_message, reasoning_effort)
                found = None if new is None else _insertion(base, new)
        return base, new, found

    async def check_case(
        self,
        name: str,
        messages: list[dict[str, Any]],
        base: list[int],
        index: int,
        reasoning_effort: str | None,
    ) -> str | None:
        """Return why the inline system message is unsafe, or None."""
        inline = {"role": "system", "content": _INLINE}
        base, new, found = await self.insert(
            messages, base, index, inline, reasoning_effort
        )
        if new is None:
            return f"{name}: renderer rejects it"
        new_text = self.tokenizer.decode(new)
        if _INLINE not in new_text:
            return f"{name}: renderer drops it"
        if found is None:
            return f"{name}: rendering it changes other parts of the prompt"
        position, segment = found
        if index == len(messages) and position >= len(base):
            return f"{name}: it lands after the generation prompt"
        if self.tokenizer.decode(segment).count(_INLINE) != 1:
            return f"{name}: it is not rendered as its own segment"
        if not self.special_ids.intersection(segment):
            return f"{name}: it has no role marker"
        if name == "mid":
            as_user = {"role": "user", "content": _INLINE}
            *_, user_found = await self.insert(
                messages, base, index, as_user, reasoning_effort
            )
            if user_found is not None and user_found[1] == segment:
                return f"{name}: it renders the same as a user message"
        if (_REASONING in self.tokenizer.decode(base)) != (_REASONING in new_text):
            return f"{name}: it changes which reasoning is kept"
        return None

    async def probe(self) -> tuple[AnthropicInlineSystemMode, str]:
        evaluated: list[list[int]] | None = None
        # Claude Code sets an effort, and some renderers keep reasoning in
        # history only when thinking is enabled; also check the defaults.
        for reasoning_effort in ("high", None):
            cases, bases = [], []
            for name, messages, index in _PROBE_CASES:
                base = await self.render(messages, reasoning_effort)
                if base is None and any("reasoning" in m for m in messages):
                    # Some renderers reject reasoning in history (e.g. Mistral
                    # tokenizers without think tokens); probe without it.
                    messages = _without_reasoning(messages)
                    base = await self.render(messages, reasoning_effort)
                if base is None:
                    break
                cases.append((name, messages, index))
                bases.append(base)
            if len(bases) != len(_PROBE_CASES) or bases == evaluated:
                continue
            evaluated = bases
            for (name, messages, index), base in zip(cases, bases):
                reason = await self.check_case(
                    name, messages, base, index, reasoning_effort
                )
                if reason is not None:
                    return "fold", f"{reason} (reasoning_effort={reasoning_effort})"
        if evaluated is None:
            return "fold", "the probe conversations could not be rendered"
        return "preserve", "the renderer marks them as distinct system turns"


async def _probe_mode(online_renderer: "OnlineRenderer") -> AnthropicInlineSystemMode:
    try:
        mode, reason = await _Prober(online_renderer).probe()
    except Exception:
        logger.warning("Inline system probe failed; folding", exc_info=True)
        return "fold"
    logger.info(
        "Anthropic inline system messages will be %s: %s",
        "kept in place" if mode == "preserve" else "folded into adjacent turns",
        reason,
    )
    return mode


class InlineSystemModeResolver:
    """Resolves ``auto`` by probing the renderer once, on first use."""

    def __init__(
        self, online_renderer: "OnlineRenderer", option: AnthropicInlineSystemOption
    ) -> None:
        self.online_renderer = online_renderer
        self.option = option
        self._probe: asyncio.Future[AnthropicInlineSystemMode] | None = None

    async def resolve(self) -> AnthropicInlineSystemMode:
        if self.option != "auto":
            return self.option
        if self._probe is None:
            self._probe = asyncio.ensure_future(_probe_mode(self.online_renderer))
        # Shield so a cancelled request does not cancel the shared probe.
        return await asyncio.shield(self._probe)
