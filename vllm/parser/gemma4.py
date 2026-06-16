# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma4 parser.

Handles channel-based reasoning plus custom tool call format in a single
state machine::

    <|channel>thought
    ...reasoning...<channel|>
    <|tool_call>call:func_name{key:<|"|>value<|"|>,num:42}<tool_call|>
"""

from __future__ import annotations

import functools
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.logger import init_logger
from vllm.parser.engine.events import EventType, SemanticEvent
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

# Tokens the model generates that must not leak into response content.
_GEMMA4_MODEL_DROP_TOKENS: set[str] = {
    # Turn boundaries
    "<|turn>",
    "<turn|>",
    # Channel / reasoning
    "<|channel>",
    "<channel|>",
    # Tool protocol tokens
    "<|tool>",
    "<tool|>",
    "<|tool_call>",
    "<tool_call|>",
    "<|tool_response>",
    "<tool_response|>",
    '<|"|>',
    # Thinking
    "<|think|>",
    # Multi-modal (defensive — not expected during text completion)
    "<|image>",
    "<|image|>",
    "<image|>",
    "<|audio>",
    "<|audio|>",
    "<audio|>",
    "<|video|>",
}

CHANNEL_START = "<|channel>"
CHANNEL_END = "<channel|>"
TOOL_CALL_START = "<|tool_call>"
TOOL_CALL_END = "<tool_call|>"
STRING_DELIM = '<|"|>'
_DELIM_LEN = len(STRING_DELIM)

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Gemma4 argument parser
# ---------------------------------------------------------------------------

_PARTIAL_DELIM_SUFFIXES = tuple(
    STRING_DELIM[:k] for k in range(len(STRING_DELIM), 0, -1)
)


def _strip_partial_delim(value: str) -> str:
    """Strip a trailing partial ``STRING_DELIM`` prefix from *value*.

    Prevents partial delimiters from leaking into the streamed JSON diff.
    """
    for suffix in _PARTIAL_DELIM_SUFFIXES:
        if value.endswith(suffix):
            return value[: -len(suffix)]
    return value


def _parse_gemma4_args(args_str: str, *, partial: bool = False) -> dict:
    """Parse Gemma4's custom key:value format into a Python dict.

    Format examples::

        location:<|"|>Tokyo<|"|>
        location:<|"|>San Francisco<|"|>,unit:<|"|>celsius<|"|>
        count:42,flag:true
        nested:{inner_key:<|"|>val<|"|>}
        items:[<|"|>a<|"|>,<|"|>b<|"|>]

    Args:
        args_str: The raw Gemma4 argument string.
        partial: When True (streaming), bare values at end of string are
            omitted because they may be incomplete and type-unstable
            (e.g. partial boolean parsed as bare string).

    Returns a dict ready for ``json.dumps()``.
    """
    if not args_str or not args_str.strip():
        return {}

    result: dict = {}
    i = 0
    n = len(args_str)

    while i < n:
        while i < n and args_str[i] in (" ", ",", "\n", "\t"):
            i += 1
        if i >= n:
            break

        key_start = i
        while i < n and args_str[i] != ":":
            i += 1
        if i >= n:
            break
        key = args_str[key_start:i].strip()
        if key.startswith(STRING_DELIM) and key.endswith(STRING_DELIM):
            key = key[_DELIM_LEN:-_DELIM_LEN]
        i += 1

        if i >= n:
            if not partial:
                result[key] = ""
            break

        while i < n and args_str[i] in (" ", "\n", "\t"):
            i += 1
        if i >= n:
            if not partial:
                result[key] = ""
            break

        if args_str[i : i + _DELIM_LEN] == STRING_DELIM:
            i += _DELIM_LEN
            val_start = i
            end_pos = args_str.find(STRING_DELIM, i)
            if end_pos == -1:
                # Unterminated string — take rest, strip partial delimiter.
                value = args_str[val_start:]
                if partial:
                    value = _strip_partial_delim(value)
                result[key] = value
                break
            result[key] = args_str[val_start:end_pos]
            i = end_pos + _DELIM_LEN

        elif args_str[i] == "{":
            depth = 1
            obj_start = i + 1
            i += 1
            while i < n and depth > 0:
                if args_str[i : i + _DELIM_LEN] == STRING_DELIM:
                    # Skip over string contents to avoid counting { inside strings
                    i += _DELIM_LEN
                    next_delim = args_str.find(STRING_DELIM, i)
                    i = n if next_delim == -1 else next_delim + _DELIM_LEN
                    continue
                if args_str[i] == "{":
                    depth += 1
                elif args_str[i] == "}":
                    depth -= 1
                i += 1
            if depth > 0:
                # Incomplete nested object — use i (not i-1) to avoid
                # dropping the last char, and recurse as partial.
                result[key] = _parse_gemma4_args(args_str[obj_start:i], partial=True)
            else:
                result[key] = _parse_gemma4_args(args_str[obj_start : i - 1])

        elif args_str[i] == "[":
            depth = 1
            arr_start = i + 1
            i += 1
            while i < n and depth > 0:
                if args_str[i : i + _DELIM_LEN] == STRING_DELIM:
                    i += _DELIM_LEN
                    next_delim = args_str.find(STRING_DELIM, i)
                    i = n if next_delim == -1 else next_delim + _DELIM_LEN
                    continue
                if args_str[i] == "[":
                    depth += 1
                elif args_str[i] == "]":
                    depth -= 1
                i += 1
            if depth > 0:
                result[key] = _parse_gemma4_array(args_str[arr_start:i], partial=True)
            else:
                result[key] = _parse_gemma4_array(args_str[arr_start : i - 1])

        else:
            val_start = i
            while i < n and args_str[i] not in (",", "}", "]"):
                i += 1
            if partial and i >= n:
                # Value may be incomplete (e.g. partial boolean) —
                # withhold to avoid type instability during streaming.
                break
            if i == val_start:
                logger.warning(
                    "Gemma4 args parser made no progress at position %d; "
                    "aborting on malformed input.",
                    i,
                )
                break
            raw_val = args_str[val_start:i].strip()
            if partial and raw_val.endswith("."):
                # Digits may still arrive (e.g. "108." -> "108.2");
                # withhold to avoid corrupting the streaming diff.
                break
            result[key] = raw_val

    return result


def _parse_gemma4_array(arr_str: str, *, partial: bool = False) -> list:
    items: list = []
    i = 0
    n = len(arr_str)

    while i < n:
        while i < n and arr_str[i] in (" ", ",", "\n", "\t"):
            i += 1
        if i >= n:
            break

        if arr_str[i : i + _DELIM_LEN] == STRING_DELIM:
            i += _DELIM_LEN
            end_pos = arr_str.find(STRING_DELIM, i)
            if end_pos == -1:
                items.append(arr_str[i:])
                break
            items.append(arr_str[i:end_pos])
            i = end_pos + _DELIM_LEN

        elif arr_str[i] == "{":
            depth = 1
            obj_start = i + 1
            i += 1
            while i < n and depth > 0:
                if arr_str[i : i + _DELIM_LEN] == STRING_DELIM:
                    i += _DELIM_LEN
                    nd = arr_str.find(STRING_DELIM, i)
                    i = nd + _DELIM_LEN if nd != -1 else n
                    continue
                if arr_str[i] == "{":
                    depth += 1
                elif arr_str[i] == "}":
                    depth -= 1
                i += 1
            if depth > 0:
                items.append(_parse_gemma4_args(arr_str[obj_start:i], partial=True))
            else:
                items.append(_parse_gemma4_args(arr_str[obj_start : i - 1]))

        elif arr_str[i] == "[":
            depth = 1
            sub_start = i + 1
            i += 1
            while i < n and depth > 0:
                if arr_str[i : i + _DELIM_LEN] == STRING_DELIM:
                    i += _DELIM_LEN
                    nd = arr_str.find(STRING_DELIM, i)
                    i = nd + _DELIM_LEN if nd != -1 else n
                    continue
                if arr_str[i] == "[":
                    depth += 1
                elif arr_str[i] == "]":
                    depth -= 1
                i += 1
            if depth > 0:
                items.append(_parse_gemma4_array(arr_str[sub_start:i], partial=True))
            else:
                items.append(_parse_gemma4_array(arr_str[sub_start : i - 1]))

        else:
            val_start = i
            while i < n and arr_str[i] not in (",", "]"):
                i += 1
            if partial and i >= n:
                break
            if i == val_start:
                logger.warning(
                    "Gemma4 array parser made no progress at position %d; "
                    "aborting on malformed input.",
                    i,
                )
                break
            raw_val = arr_str[val_start:i].strip()
            if partial and raw_val.endswith("."):
                break
            items.append(raw_val)

    return items


def _gemma4_arg_converter(raw_args: str, partial: bool) -> str:
    """Convert Gemma4 custom arg format to a JSON string."""
    text = raw_args.strip()
    if text.endswith("}"):
        text = text[:-1]

    parsed = _parse_gemma4_args(text, partial=partial)
    return json.dumps(parsed, ensure_ascii=False)


@functools.cache
def gemma4_config() -> ParserEngineConfig:
    used_tokens = {
        CHANNEL_START,
        CHANNEL_END,
        TOOL_CALL_START,
        TOOL_CALL_END,
        '<|"|>',
    }

    return ParserEngineConfig(
        name="gemma4",
        initial_state=ParserState.CONTENT,
        terminals={
            "THINK_START": CHANNEL_START,
            "THINK_END": CHANNEL_END,
            "TOOL_START": TOOL_CALL_START,
            "TOOL_END": TOOL_CALL_END,
            "CALL_PREFIX": "call:",
            "OPEN_BRACE": "{",
        },
        token_id_terminals={
            "THINK_START": CHANNEL_START,
            "THINK_END": CHANNEL_END,
            "TOOL_START": TOOL_CALL_START,
            "TOOL_END": TOOL_CALL_END,
        },
        transitions={
            # -- Reasoning transitions --
            (ParserState.CONTENT, "THINK_START"): Transition(
                ParserState.REASONING,
                (EventType.REASONING_START,),
            ),
            (ParserState.REASONING, "THINK_END"): Transition(
                ParserState.CONTENT,
                (EventType.REASONING_END,),
            ),
            # Tool call directly from reasoning (no explicit <channel|>)
            (ParserState.REASONING, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (EventType.REASONING_END, EventType.TOOL_CALL_START),
            ),
            # -- Tool call transitions --
            (ParserState.CONTENT, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (EventType.REASONING_END, EventType.TOOL_CALL_START),
            ),
            (ParserState.TOOL_PREAMBLE, "CALL_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (),
            ),
            (ParserState.TOOL_NAME, "OPEN_BRACE"): Transition(
                ParserState.TOOL_ARGS,
                (),
            ),
            (ParserState.TOOL_ARGS, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (EventType.TOOL_CALL_END,),
            ),
            # Back-to-back tool calls
            (ParserState.CONTENT, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (),
            ),
            # Absorb a bare <channel|> that arrives after we already
            # returned to CONTENT; prevents leaking it as TEXT_CHUNK.
            (ParserState.CONTENT, "THINK_END"): Transition(
                ParserState.CONTENT,
                (),
            ),
        },
        content_events={
            ParserState.CONTENT: EventType.TEXT_CHUNK,
            ParserState.REASONING: EventType.REASONING_CHUNK,
            ParserState.TOOL_NAME: EventType.TOOL_NAME,
            ParserState.TOOL_ARGS: EventType.ARG_VALUE_CHUNK,
        },
        arg_converter=_gemma4_arg_converter,
        tool_args_json=False,
        arg_structural_chars=frozenset(",:{}[]<"),
        drop_tokens=frozenset(_GEMMA4_MODEL_DROP_TOKENS - used_tokens),
    )


_GEMMA4_THOUGHT_PREFIX = "thought\n"
_GEMMA4_THOUGHT_TOKEN = "thought"


class Gemma4Parser(ParserEngine):
    """Gemma4 parser: ``<|channel>`` reasoning + ``<|tool_call>``
    tool calls in a single engine.

    - Strips the ``thought\\n`` prefix from reasoning content
    - Sets ``skip_special_tokens=False`` so boundary tokens are visible
    - Detects ``<|tool_call>`` token as implicit reasoning end
    """

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        self._thinking_enabled = chat_kwargs.get("enable_thinking", True)
        super().__init__(
            tokenizer,
            tools,
            parser_engine_config=gemma4_config(),
            **kwargs,
        )
        vocab = self.vocab
        self._tool_call_token_id: int | None = vocab.get("<|tool_call>")
        self._new_turn_token_id: int | None = vocab.get("<|turn>")
        self._tool_response_token_id: int | None = vocab.get("<|tool_response>")
        self._reasoning_text: str = ""
        self._prefix_stripped: bool = False
        self._is_first_feed: bool = True

    def adjust_request(
        self,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> ChatCompletionRequest | ResponsesRequest:
        """Skip ``skip_special_tokens=False`` when thinking is disabled.

        When there are no reasoning channel tokens to preserve,
        keeping the default prevents tool-call delimiter tokens
        from leaking into content (e.g. with ``tool_choice="none"``).
        """
        chat_template_kwargs = getattr(request, "chat_template_kwargs", None) or {}
        if not chat_template_kwargs.get("enable_thinking", True):
            return request
        return super().adjust_request(request)

    def _reset(self, initial_state=None) -> None:
        super()._reset(initial_state=initial_state)
        self._reasoning_text = ""
        self._prefix_stripped = False
        self._is_first_feed = True

    def _preprocess_feed(
        self,
        delta_text: str,
        delta_token_ids: Sequence[int],
    ) -> tuple[str, Sequence[int]]:
        if not self._is_first_feed:
            return delta_text, delta_token_ids
        self._is_first_feed = False

        if (
            not delta_text
            or self._engine.state != ParserState.CONTENT
            or self._reasoning_start_token_id is None
            or self._reasoning_end_token_id is None
        ):
            return delta_text, delta_token_ids

        if CHANNEL_START in delta_text:
            return delta_text, delta_token_ids

        needs_injection = (
            CHANNEL_END in delta_text
            or delta_text.startswith(_GEMMA4_THOUGHT_PREFIX)
            or delta_text == _GEMMA4_THOUGHT_TOKEN
        )
        if not needs_injection:
            return delta_text, delta_token_ids

        delta_text = CHANNEL_START + delta_text
        if delta_token_ids:
            delta_token_ids = [self._reasoning_start_token_id, *delta_token_ids]

        return delta_text, delta_token_ids

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        end_id = self._reasoning_end_token_id
        start_id = self._reasoning_start_token_id
        tool_call_id = self._tool_call_token_id
        new_turn_id = self._new_turn_token_id
        tool_response_id = self._tool_response_token_id

        if end_id is not None and not input_ids:
            return self.parser_engine_config.initial_state != ParserState.REASONING

        for i in range(len(input_ids) - 1, -1, -1):
            tid = input_ids[i]
            if start_id is not None and tid == start_id:
                return False
            if tool_call_id is not None and tid == tool_call_id:
                return True
            if new_turn_id is not None and tid == new_turn_id:
                return not self._thinking_enabled
            if tool_response_id is not None and tid == tool_response_id:
                return not self._thinking_enabled
            if end_id is not None and tid == end_id:
                return True
        return True

    def _events_to_delta(
        self,
        events: list[SemanticEvent],
        finished: bool = False,
    ) -> DeltaMessage | None:
        delta = super()._events_to_delta(events, finished=finished)
        if delta is None or delta.reasoning is None:
            return delta

        if self._prefix_stripped:
            return delta
        self._reasoning_text += delta.reasoning

        if self._reasoning_text.startswith(_GEMMA4_THOUGHT_PREFIX):
            prefix_len = len(_GEMMA4_THOUGHT_PREFIX)
            prev_reasoning_len = len(self._reasoning_text) - len(delta.reasoning)
            if prev_reasoning_len >= prefix_len:
                self._prefix_stripped = True
                return delta
            chars_of_prefix_in_delta = prefix_len - prev_reasoning_len
            stripped = delta.reasoning[chars_of_prefix_in_delta:]
            if stripped:
                self._prefix_stripped = True
                delta.reasoning = stripped
                return delta
            if len(self._reasoning_text) >= prefix_len:
                self._prefix_stripped = True
                delta.reasoning = None
                if delta.content is not None or delta.tool_calls:
                    return delta
                return None
            return None

        if _GEMMA4_THOUGHT_PREFIX.startswith(self._reasoning_text):
            if finished:
                self._prefix_stripped = True
            return None

        self._prefix_stripped = True
        delta.reasoning = self._reasoning_text
        return delta

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        reasoning, content = super().extract_reasoning(model_output, request)
        if reasoning:
            if reasoning.startswith(_GEMMA4_THOUGHT_PREFIX):
                reasoning = reasoning[len(_GEMMA4_THOUGHT_PREFIX) :]
            elif reasoning == _GEMMA4_THOUGHT_PREFIX.rstrip():
                reasoning = None
        return reasoning or None, content
