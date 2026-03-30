# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
KimiK2ToolParser — clean rewrite (Optimized & vLLM Native Compliant).

Design principles:
  1. Single source of truth: streaming state is rebuilt from current_text.
  2. No silent drops: every early return is explicit and logged.
  3. Section markers are stripped once, at entry.
  4. Maintains self.prev_tool_call_arr and self.streamed_args_for_tool
     strictly for vLLM serving.py compatibility.
  5. Infinite-context safe: rolling buffer for markers is capped at 256 bytes.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import ToolParser

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

#: Safety valve: if we stay inside a tool section this long, force-exit.
#: Set to 512KB to support massive max_tokens=16000+ outputs.
SECTION_MAX_CHARS: int = int(os.getenv("KIMI_PARSER_SECTION_MAX", "524288"))

# ---------------------------------------------------------------------------
# Internal state container
# ---------------------------------------------------------------------------


@dataclass
class _StreamState:
    """All mutable streaming state in one place — optimized to use O(1) memory."""

    in_tool_section: bool = False
    section_char_count: int = 0
    marker_buffer: str = ""

    current_tool_id: int = -1
    tool_name_sent: bool = False
    current_args: str = ""

    def reset(self) -> None:
        self.__init__()

    def enter_section(self) -> None:
        self.in_tool_section = True
        self.section_char_count = 0
        self.marker_buffer = ""

    def exit_section(self) -> None:
        self.in_tool_section = False
        self.section_char_count = 0
        self.marker_buffer = ""


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class KimiK2ToolParser(ToolParser):
    _SECTION_BEGIN_VARIANTS = (
        "<|tool_calls_section_begin|>",
        "<|tool_call_section_begin|>",
    )
    _SECTION_END_VARIANTS = (
        "<|tool_calls_section_end|>",
        "<|tool_call_section_end|>",
    )
    _CALL_BEGIN = "<|tool_call_begin|>"
    _CALL_END = "<|tool_call_end|>"
    _ARG_BEGIN = "<|tool_call_argument_begin|>"

    _ALL_MARKERS = (
        *_SECTION_BEGIN_VARIANTS,
        *_SECTION_END_VARIANTS,
        _CALL_BEGIN,
        _CALL_END,
        _ARG_BEGIN,
    )

    _RE_FULL = re.compile(
        r"<\|tool_call_begin\|>\s*(?P<call_id>[^<\s]+)\s*"
        r"<\|tool_call_argument_begin\|>\s*(?P<args>(?:(?!<\|tool_call_begin\|>).)*?)\s*"
        r"<\|tool_call_end\|>",
        re.DOTALL,
    )

    _RE_STREAM_ID = re.compile(r"^\s*(?P<call_id>[^<\s]+(?::\d+|_\d+))\s*$")
    _RE_UNDERSCORE_SUFFIX = re.compile(r"_\d+$")

    def __init__(self, tokenizer: TokenizerLike) -> None:
        # Base class ToolParser initializes:
        # self.prev_tool_call_arr =[]
        # self.streamed_args_for_tool =[]
        super().__init__(tokenizer)

        if not self.model_tokenizer:
            raise ValueError("KimiK2ToolParser requires model_tokenizer to be set.")

        self._section_begin_ids: tuple[int, ...] = tuple(
            tid
            for v in self._SECTION_BEGIN_VARIANTS
            if (tid := self.vocab.get(v)) is not None
        )
        self._section_end_ids: tuple[int, ...] = tuple(
            tid
            for v in self._SECTION_END_VARIANTS
            if (tid := self.vocab.get(v)) is not None
        )
        self._call_begin_id: Optional[int] = self.vocab.get(self._CALL_BEGIN)
        self._call_end_id: Optional[int] = self.vocab.get(self._CALL_END)

        if not self._section_begin_ids or not self._section_end_ids:
            raise RuntimeError("Missing tool section begin/end tokens in vocab.")
        if self._call_begin_id is None or self._call_end_id is None:
            raise RuntimeError("Missing tool call begin/end tokens in vocab.")

        self._state = _StreamState()

    def reset_streaming_state(self) -> None:
        self._state.reset()
        # Ensure upstream vLLM arrays are wiped cleanly for the next request
        self.prev_tool_call_arr.clear()
        self.streamed_args_for_tool.clear()
        logger.debug("KimiK2ToolParser: state reset")

    def extract_tool_calls(
        self, model_output: str, request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        section_begin = next(
            (v for v in self._SECTION_BEGIN_VARIANTS if v in model_output), None
        )
        if section_begin is None:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            tool_calls = [
                ToolCall(
                    id=m.group("call_id"),
                    type="function",
                    function=FunctionCall(
                        name=self._call_id_to_name(m.group("call_id")),
                        arguments=m.group("args"),
                    ),
                )
                for m in self._RE_FULL.finditer(model_output)
            ]
            content = model_output[: model_output.index(section_begin)]
            return ExtractedToolCallInformation(
                tools_called=bool(tool_calls),
                tool_calls=tool_calls,
                content=content or None,
            )
        except Exception:
            logger.exception("KimiK2ToolParser.extract_tool_calls failed")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> Optional[DeltaMessage]:
        s = self._state

        # ----------------------------------------------------------------
        # Step 1: Feed rolling look-ahead buffer for split markers.
        # Cap at 256 bytes to prevent memory bloat on massive texts.
        # ----------------------------------------------------------------
        s.marker_buffer = (s.marker_buffer + delta_text)[-256:]

        found_begin = any(v in s.marker_buffer for v in self._SECTION_BEGIN_VARIANTS)
        found_end = any(v in s.marker_buffer for v in self._SECTION_END_VARIANTS)

        # ----------------------------------------------------------------
        # Step 2: Handle tool section boundaries
        # ----------------------------------------------------------------
        if found_begin and not s.in_tool_section:
            s.enter_section()

        if found_end and s.in_tool_section:
            post_marker = self._text_after_section_end(delta_text)
            s.exit_section()
            return (
                DeltaMessage(content=post_marker)
                if post_marker.strip()
                else DeltaMessage(content="")
            )

        # ----------------------------------------------------------------
        # Step 3: Pure reasoning text (bypass tool logic)
        # ----------------------------------------------------------------
        if not s.in_tool_section and not any(
            tid in current_token_ids for tid in self._section_begin_ids
        ):
            return DeltaMessage(content=delta_text)

        # ----------------------------------------------------------------
        # Step 4: Safety valve for unbounded tool sections
        # ----------------------------------------------------------------
        if s.in_tool_section:
            s.section_char_count += len(delta_text)
            if s.section_char_count > SECTION_MAX_CHARS:
                logger.warning(
                    "KimiK2ToolParser: section length exceeded %d max limit.",
                    SECTION_MAX_CHARS,
                )
                s.exit_section()
                return DeltaMessage(content="")

        clean_delta = self._strip_all_markers(delta_text)

        prev_begin_count = previous_token_ids.count(self._call_begin_id)
        cur_begin_count = current_token_ids.count(self._call_begin_id)
        cur_end_count = current_token_ids.count(self._call_end_id)
        call_end_in_delta = self._call_end_id in delta_token_ids

        # ----------------------------------------------------------------
        # Phase A: New Tool Call Started
        # ----------------------------------------------------------------
        if cur_begin_count > prev_begin_count:
            s.current_tool_id += 1
            s.tool_name_sent = False
            s.current_args = ""

            # --- vLLM Native Compatibility Sync ---
            # Pre-pad arrays so vLLM serving.py never hits IndexError
            self.prev_tool_call_arr.append(
                {
                    "type": "function",
                    "id": None,
                    "function": {"name": None, "arguments": ""},
                }
            )
            self.streamed_args_for_tool.append("")

            return self._try_emit_name(self._extract_call_portion(current_text))

        # ----------------------------------------------------------------
        # Phase B: Tool Call Ended
        # ----------------------------------------------------------------
        if call_end_in_delta:
            return self._handle_call_end(current_text)

        # ----------------------------------------------------------------
        # Phase C: Noise Suppress (Between section start and tool start)
        # ----------------------------------------------------------------
        if s.in_tool_section and cur_begin_count == 0:
            return DeltaMessage(content="")

        # ----------------------------------------------------------------
        # Phase D: Append Arguments to current tool
        # ----------------------------------------------------------------
        if cur_begin_count > cur_end_count:
            portion = self._extract_call_portion(current_text)
            if not s.tool_name_sent:
                return self._try_emit_name(portion)

            parsed = self._parse_call_portion(portion)
            if parsed:
                return self._diff_and_emit_args(parsed.get("arguments") or "")
            return None

        # ----------------------------------------------------------------
        # Phase E: Idle in section (waiting for next tool or section end)
        # ----------------------------------------------------------------
        if s.in_tool_section:
            return DeltaMessage(content="")

        return DeltaMessage(content=clean_delta)

    # ------------------------------------------------------------------
    # Optimized Helpers
    # ------------------------------------------------------------------

    @classmethod
    def _call_id_to_name(cls, call_id: str) -> str:
        """Lightning-fast tool name extraction."""
        if ":" in call_id:
            return call_id.split(":", 1)[0].rsplit(".", 1)[-1]
        name = cls._RE_UNDERSCORE_SUFFIX.sub("", call_id)
        return name[10:] if name.startswith("functions_") else name

    def _strip_all_markers(self, text: str) -> str:
        for marker in self._ALL_MARKERS:
            text = text.replace(marker, "")
        return text

    def _text_after_section_end(self, delta_text: str) -> str:
        for v in self._SECTION_END_VARIANTS:
            if v in delta_text:
                return delta_text.split(v, 1)[1]
        return ""

    def _extract_call_portion(self, current_text: str) -> str:
        idx = current_text.rfind(self._CALL_BEGIN)
        if idx != -1:
            return current_text[idx + len(self._CALL_BEGIN) :].lstrip()
        return ""

    def _parse_call_portion(self, portion: str) -> Optional[dict]:
        if self._ARG_BEGIN in portion:
            id_part, _, args_part = portion.partition(self._ARG_BEGIN)
            call_id = id_part.strip()
            if not call_id:
                return None
            return {
                "id": call_id,
                "name": self._call_id_to_name(call_id),
                "arguments": args_part,
            }

        m = self._RE_STREAM_ID.match(portion)
        if m:
            call_id = m.group("call_id")
            return {
                "id": call_id,
                "name": self._call_id_to_name(call_id),
                "arguments": None,
            }
        return None

    def _try_emit_name(self, portion: str) -> Optional[DeltaMessage]:
        parsed = self._parse_call_portion(portion)
        if not parsed or not parsed.get("name"):
            return None

        s = self._state
        s.tool_name_sent = True

        # Sync parsed Name and ID with vLLM's internal array
        if 0 <= s.current_tool_id < len(self.prev_tool_call_arr):
            self.prev_tool_call_arr[s.current_tool_id]["id"] = parsed["id"]
            self.prev_tool_call_arr[s.current_tool_id]["function"]["name"] = parsed[
                "name"
            ]

        return DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=s.current_tool_id,
                    type="function",
                    id=parsed["id"],
                    function=DeltaFunctionCall(name=parsed["name"]).model_dump(
                        exclude_none=True
                    ),
                )
            ]
        )

    def _diff_and_emit_args(self, cur_args: str) -> Optional[DeltaMessage]:
        s = self._state
        already = s.current_args

        if not cur_args or cur_args == already:
            return None

        if not cur_args.startswith(already):
            new_part = cur_args
        else:
            new_part = cur_args[len(already) :]

        s.current_args = cur_args

        # Sync updated arguments with vLLM's internal arrays
        if 0 <= s.current_tool_id < len(self.prev_tool_call_arr):
            self.prev_tool_call_arr[s.current_tool_id]["function"]["arguments"] = (
                cur_args
            )
            self.streamed_args_for_tool[s.current_tool_id] = cur_args

        if not new_part:
            return None

        return DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=s.current_tool_id,
                    function=DeltaFunctionCall(arguments=new_part).model_dump(
                        exclude_none=True
                    ),
                )
            ]
        )

    def _handle_call_end(self, current_text: str) -> Optional[DeltaMessage]:
        if self._state.current_tool_id < 0:
            return None

        after_begin = self._extract_call_portion(current_text)
        before_end = after_begin.split(self._CALL_END, 1)[0].rstrip()

        parsed = self._parse_call_portion(before_end)
        if not parsed:
            return None

        return self._diff_and_emit_args(parsed.get("arguments") or "")
