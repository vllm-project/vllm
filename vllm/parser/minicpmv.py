# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniCPM-V family parser.

Wraps the Qwen3 grammar (``<think>`` reasoning + XML tool calls) with two
MiniCPM-V specifics:

- ``<|im_end|>`` and the reserved marker tokens are passthrough terminals,
  so they never leak into ``content``.
- The tokenizer decodes a newline as the two characters ``\\`` and ``n``
  (e.g. token id 1639) instead of ``\\n``, so the decoded text is recovered
  before it reaches the client. Code blocks, inline code and LaTeX are left
  alone, as are tool call arguments.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

from vllm.entrypoints.generate.base.protocol import DeltaMessage
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)
from vllm.parser.qwen3 import Qwen3Parser, qwen3_config

if TYPE_CHECKING:
    from vllm.entrypoints.generate.base.protocol import FunctionCall
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool


_RESERVED_MARKER_IDS = range(12, 16)
_IM_END = "<|im_end|>"


@functools.cache
def minicpmv_config(thinking: bool) -> ParserEngineConfig:
    if thinking:
        base = qwen3_config(thinking=True, name="minicpmv")
    else:
        base = ParserEngineConfig(
            name="minicpmv_no_thinking",
            initial_state=ParserState.CONTENT,
            terminals={
                "THINK_START": "<think>",
                "THINK_END": "</think>",
            },
            token_id_terminals={
                "THINK_START": "<think>",
                "THINK_END": "</think>",
            },
            transitions={
                (ParserState.CONTENT, "THINK_START"): Transition(ParserState.REASONING),
                (ParserState.REASONING, "THINK_START"): Transition(
                    ParserState.REASONING
                ),
                (ParserState.REASONING, "THINK_END"): Transition(ParserState.CONTENT),
                (ParserState.CONTENT, "THINK_END"): Transition(ParserState.CONTENT),
            },
            strip_trailing_reasoning_whitespace=False,
        )

    terminals = dict(base.terminals)
    token_id_terminals = dict(base.token_id_terminals)
    transitions = dict(base.transitions)
    terminals["IM_END"] = _IM_END
    token_id_terminals["IM_END"] = _IM_END
    for state in (ParserState.CONTENT, ParserState.REASONING):
        transitions[(state, "IM_END")] = Transition(state)

    for index in _RESERVED_MARKER_IDS:
        text_name = f"RESERVED_TEXT_{index}"
        token_name = f"RESERVED_TOKEN_{index}"
        terminals[text_name] = f"<reserved_{index}>"
        terminals[token_name] = f"<|reserved_{index}|>"
        token_id_terminals[token_name] = f"<|reserved_{index}|>"
        for state in (ParserState.CONTENT, ParserState.REASONING):
            transitions[(state, text_name)] = Transition(state)
            transitions[(state, token_name)] = Transition(state)

    return replace(
        base,
        terminals=terminals,
        token_id_terminals=token_id_terminals,
        transitions=transitions,
    )


class EscapedNewlineNormalizer:
    """Recover literal escape sequences written by the model as real text.

    Scans character by character so that a backslash inside a code block,
    inline code, LaTeX span or a doubled escape is left untouched. A trailing
    half of an escape is held in ``_pending`` until the next chunk arrives,
    which keeps streaming output identical to the non-streaming result.
    """

    def __init__(self) -> None:
        self._closing_marker: str | None = None
        self._pending = ""

    def feed(self, text: str, *, final: bool = False) -> str:
        text = self._pending + text
        self._pending = ""
        output: list[str] = []
        index = 0

        while index < len(text):
            if self._closing_marker is not None:
                marker = self._closing_marker
                remaining = text[index:]
                if remaining.startswith(marker):
                    output.append(marker)
                    index += len(marker)
                    self._closing_marker = None
                    continue
                if not final and marker.startswith(remaining):
                    self._pending = remaining
                    break
                output.append(text[index])
                index += 1
                continue

            remaining = text[index:]
            if text[index] == "`":
                if not final and remaining in ("`", "``"):
                    self._pending = remaining
                    break
                if remaining.startswith("```"):
                    output.append("```")
                    index += 3
                    self._closing_marker = "```"
                else:
                    output.append("`")
                    index += 1
                    self._closing_marker = "`"
                continue

            if text[index] == "$":
                if not final and remaining == "$":
                    self._pending = remaining
                    break
                if remaining.startswith("$$"):
                    output.append("$$")
                    index += 2
                    self._closing_marker = "$$"
                else:
                    output.append("$")
                    index += 1
                    self._closing_marker = "$"
                continue

            if text[index] != "\\":
                output.append(text[index])
                index += 1
                continue

            slash_end = index + 1
            while slash_end < len(text) and text[slash_end] == "\\":
                slash_end += 1
            if slash_end - index > 1:
                output.append(text[index:slash_end])
                index = slash_end
                continue

            if not final and remaining in ("\\", "\\r", "\\r\\"):
                self._pending = remaining
                break
            if remaining.startswith("\\r\\n"):
                output.append("\n")
                index += 4
            elif remaining.startswith(("\\n", "\\r")):
                output.append("\n")
                index += 2
            elif remaining.startswith("\\("):
                output.append("\\(")
                index += 2
                self._closing_marker = "\\)"
            elif remaining.startswith("\\["):
                output.append("\\[")
                index += 2
                self._closing_marker = "\\]"
            else:
                output.append("\\")
                index += 1

        if final and self._pending:
            output.append(self._pending)
            self._pending = ""
        return "".join(output)


def recover_newlines(text: str | None) -> str | None:
    if text is None or "\\" not in text:
        return text
    return EscapedNewlineNormalizer().feed(text, final=True)


class MiniCPMVOutputNormalizer:
    """Apply :class:`EscapedNewlineNormalizer` to a whole stream.

    Reasoning and content are tracked separately so that a half-written escape
    at the end of a reasoning chunk does not swallow the first characters of
    the content that follows.
    """

    def __init__(self) -> None:
        self._reasoning = EscapedNewlineNormalizer()
        self._content = EscapedNewlineNormalizer()

    def normalize_delta(
        self,
        delta: DeltaMessage | None,
        *,
        finished: bool,
    ) -> DeltaMessage | None:
        reasoning_finished = finished or (
            delta is not None and (delta.content is not None or bool(delta.tool_calls))
        )
        reasoning = self._reasoning.feed(
            delta.reasoning if delta and delta.reasoning else "",
            final=reasoning_finished,
        )
        content = self._content.feed(
            delta.content if delta and delta.content else "",
            final=finished,
        )

        if delta is None:
            if not reasoning and not content:
                return None
            return DeltaMessage(
                reasoning=reasoning or None,
                content=content or None,
            )

        if reasoning:
            delta.reasoning = reasoning
        elif delta.reasoning is not None:
            delta.reasoning = None
        if content:
            delta.content = content
        elif delta.content is not None:
            delta.content = None
        return delta


class MiniCPMVParser(Qwen3Parser):
    """Parse MiniCPM-V family output in thinking and non-thinking modes."""

    CONFIG_NAME = "minicpmv"

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        thinking_enabled = chat_kwargs.get("enable_thinking", False)
        kwargs.setdefault("parser_engine_config", minicpmv_config(thinking_enabled))
        super().__init__(tokenizer, tools, **kwargs)
        self._output_normalizer = MiniCPMVOutputNormalizer()

    @property
    def reasoning_ended(self) -> bool:
        if not self.thinking_enabled:
            return False
        return super().reasoning_ended

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        if not self.thinking_enabled:
            return False
        return super().is_reasoning_end(input_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if not self.thinking_enabled:
            return input_ids
        return super().extract_content_ids(input_ids)

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        if self.thinking_enabled:
            return super().extract_reasoning(model_output, request)
        _, content = ParserEngine.extract_reasoning(self, model_output, request)
        return None, content

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        if self.thinking_enabled:
            return super().extract_reasoning_streaming(
                previous_text,
                current_text,
                delta_text,
                previous_token_ids,
                current_token_ids,
                delta_token_ids,
            )

        delta = ParserEngine.extract_reasoning_streaming(
            self,
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )
        if delta is None:
            return None
        delta.reasoning = None
        if delta.content is None and not delta.tool_calls:
            return None
        return delta

    def get_streaming_fallback_content(
        self,
        text: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> str | None:
        if self.thinking_enabled:
            return super().get_streaming_fallback_content(text, request)
        delta = ParserEngine.finish_streaming(self)
        return delta.content if delta is not None else None

    def parse(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
        enable_auto_tools: bool = False,
        model_output_token_ids: Sequence[int] = (),
    ) -> tuple[str | None, str | None, list[FunctionCall] | None]:
        reasoning, content, tool_calls = super().parse(
            model_output,
            request,
            enable_auto_tools,
            model_output_token_ids,
        )
        # Tool call arguments are JSON, where a literal `\n` is meaningful and
        # must survive as written.
        return (
            recover_newlines(reasoning),
            recover_newlines(content),
            tool_calls,
        )

    def parse_delta(
        self,
        delta_text: str,
        delta_token_ids: list[int],
        request: ChatCompletionRequest | ResponsesRequest,
        prompt_token_ids: list[int] | None = None,
        *,
        finished: bool,
    ) -> DeltaMessage | None:
        delta = super().parse_delta(
            delta_text,
            delta_token_ids,
            request,
            prompt_token_ids,
            finished=finished,
        )
        return self._output_normalizer.normalize_delta(delta, finished=finished)
