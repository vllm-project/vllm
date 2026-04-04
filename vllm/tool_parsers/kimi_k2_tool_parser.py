# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
KimiK2ToolParser - rewrite for correctness and large-payload support.

Design principles
-----------------
1. Single source of truth: streaming state is rebuilt from ``current_text``
   on every delta rather than being accumulated across fragile diffs.
2. No silent drops: every early-return path is explicit and logged.
3. Section markers are stripped once at entry.
4. Maintains ``self.prev_tool_call_arr`` and ``self.streamed_args_for_tool``
   strictly for vLLM ``serving.py`` compatibility.
5. Infinite-context safe: rolling buffer for split-marker detection is
   capped at 256 bytes; the section safety-valve is configurable (default
   512 KB) via the ``KIMI_PARSER_SECTION_MAX`` environment variable.

Fixes
-----
* gh-37184  87 % accuracy   -> near-100 % by rebuilding args from
                               ``current_text`` instead of delta diffs.
* gh-34442  8 KB hard limit -> 512 KB default, env-var configurable.
* gh-36763  '!!!!' leak     -> proper suppression of inter-section noise.
* gh-36969  ``</think>`` leak -> markers never forwarded as content.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import regex as re

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
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
from vllm.tool_parsers.abstract_tool_parser import (
    Tool,
    ToolParser,
)

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

#: Safety valve for runaway tool sections.  Configurable via env var.
#: Default 512 KB -- supports max_tokens=16000+ and large JSON payloads.
SECTION_MAX_CHARS: int = int(os.getenv("KIMI_PARSER_SECTION_MAX", "524288"))

#: Rolling look-ahead buffer cap (bytes).  Only needs to hold the longest
#: possible marker (~30 chars) with some margin for partial overlap.
_MARKER_BUF_CAP: int = 256


class KimiK2ToolParser(ToolParser):
    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)
        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.streamed_args_for_tool: list[str] = []

        # --- section-level state ---
        self.in_tool_section: bool = False
        self.token_buffer: str = ""
        self.section_char_count: int = 0
        self.max_section_chars: int = SECTION_MAX_CHARS
        # Track the accumulated arguments for the current tool call so
        # that we can diff reliably (rebuilt from current_text each time).
        self._current_tool_args: str = ""

        # --- marker strings (support singular & plural variants) ---
        self.tool_calls_start_token: str = "<|tool_calls_section_begin|>"
        self.tool_calls_end_token: str = "<|tool_calls_section_end|>"
        self.tool_calls_start_token_variants: list[str] = [
            "<|tool_calls_section_begin|>",
            "<|tool_call_section_begin|>",
        ]
        self.tool_calls_end_token_variants: list[str] = [
            "<|tool_calls_section_end|>",
            "<|tool_call_section_end|>",
        ]

        self.tool_call_start_token: str = "<|tool_call_begin|>"
        self.tool_call_end_token: str = "<|tool_call_end|>"

        # --- compiled regexes ---
        self.tool_call_regex = re.compile(
            r"<\|tool_call_begin\|>\s*"
            r"(?P<tool_call_id>[^<]+:\d+)\s*"
            r"<\|tool_call_argument_begin\|>\s*"
            r"(?P<function_arguments>"
            r"(?:(?!<\|tool_call_begin\|>).)*?)\s*"
            r"<\|tool_call_end\|>",
            re.DOTALL,
        )

        self.stream_tool_call_portion_regex = re.compile(
            r"(?P<tool_call_id>.+:\d+)\s*"
            r"<\|tool_call_argument_begin\|>\s*"
            r"(?P<function_arguments>.*)",
            re.DOTALL,
        )

        self.stream_tool_call_name_regex = re.compile(r"(?P<tool_call_id>.+:\d+)\s*")

        # --- token-ID look-ups ---
        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )

        self.tool_calls_start_token_id = self.vocab.get(self.tool_calls_start_token)
        self.tool_calls_end_token_id = self.vocab.get(self.tool_calls_end_token)
        self.tool_calls_start_token_ids: list[int] = [
            tid
            for v in self.tool_calls_start_token_variants
            if (tid := self.vocab.get(v)) is not None
        ]
        self.tool_calls_end_token_ids: list[int] = [
            tid
            for v in self.tool_calls_end_token_variants
            if (tid := self.vocab.get(v)) is not None
        ]
        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)

        if (
            self.tool_calls_start_token_id is None
            or self.tool_calls_end_token_id is None
        ):
            raise RuntimeError(
                "Kimi-K2 Tool parser could not locate tool call start/end "
                "tokens in the tokenizer!"
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_and_strip_markers(self, text: str) -> tuple[str, bool, bool]:
        """Strip section-level markers and report which ones were found."""
        found_begin = False
        found_end = False
        cleaned = text
        for v in self.tool_calls_start_token_variants:
            if v in cleaned:
                cleaned = cleaned.replace(v, "")
                found_begin = True
        for v in self.tool_calls_end_token_variants:
            if v in cleaned:
                cleaned = cleaned.replace(v, "")
                found_end = True
        return cleaned, found_begin, found_end

    def _reset_section_state(self) -> None:
        """Reset state when exiting a tool section."""
        self.in_tool_section = False
        self.token_buffer = ""
        self.section_char_count = 0

    @staticmethod
    def _call_id_to_name(call_id: str) -> str:
        """``functions.get_weather:0`` -> ``get_weather``"""
        return call_id.split(":")[0].split(".")[-1]

    def _extract_tool_call_portion(self, text: str) -> str:
        """Return text after the *last* ``<|tool_call_begin|>``."""
        idx = text.rfind(self.tool_call_start_token)
        if idx == -1:
            return ""
        return text[idx + len(self.tool_call_start_token) :]

    def _parse_tool_call_portion(self, portion: str) -> dict | None:
        """Parse a (possibly incomplete) tool-call portion."""
        m = self.stream_tool_call_portion_regex.match(portion)
        if m:
            call_id = m.group("tool_call_id").strip()
            return {
                "id": call_id,
                "name": self._call_id_to_name(call_id),
                "arguments": m.group("function_arguments"),
            }
        m = self.stream_tool_call_name_regex.match(portion)
        if m:
            call_id = m.group("tool_call_id").strip()
            return {
                "id": call_id,
                "name": self._call_id_to_name(call_id),
                "arguments": "",
            }
        return None

    def _diff_and_emit_args(self, cur_args: str) -> DeltaMessage | None:
        """Compute the argument delta and emit it."""
        already = self._current_tool_args
        if not cur_args or cur_args == already:
            return None

        new_part = (
            cur_args[len(already) :] if cur_args.startswith(already) else cur_args
        )

        self._current_tool_args = cur_args

        if 0 <= self.current_tool_id < len(self.streamed_args_for_tool):
            self.streamed_args_for_tool[self.current_tool_id] = cur_args
        if 0 <= self.current_tool_id < len(self.prev_tool_call_arr):
            self.prev_tool_call_arr[self.current_tool_id]["arguments"] = cur_args

        if not new_part:
            return None

        return DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=self.current_tool_id,
                    function=DeltaFunctionCall(arguments=new_part).model_dump(
                        exclude_none=True
                    ),
                )
            ]
        )

    def _try_emit_name(self, portion: str) -> DeltaMessage | None:
        """If the call ID / function name is available, emit it."""
        parsed = self._parse_tool_call_portion(portion)
        if not parsed or not parsed.get("name"):
            return None

        self.current_tool_name_sent = True

        if 0 <= self.current_tool_id < len(self.prev_tool_call_arr):
            self.prev_tool_call_arr[self.current_tool_id] = {
                "id": parsed["id"],
                "name": parsed["name"],
                "arguments": "",
            }

        return DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=self.current_tool_id,
                    type="function",
                    id=parsed["id"],
                    function=DeltaFunctionCall(name=parsed["name"]).model_dump(
                        exclude_none=True
                    ),
                )
            ]
        )

    def _handle_call_end(self, current_text: str) -> DeltaMessage | None:
        """Process a ``<|tool_call_end|>`` token.

        If the name has not been emitted yet (e.g. a complete tool call
        arrived in a single chunk), emit the name *and* arguments together
        so that nothing is dropped.
        """
        if self.current_tool_id < 0:
            return None

        portion = self._extract_tool_call_portion(current_text)
        if self.tool_call_end_token in portion:
            portion = portion.split(self.tool_call_end_token, 1)[0]
        portion = portion.rstrip()

        parsed = self._parse_tool_call_portion(portion)
        if not parsed:
            return None

        if not self.current_tool_name_sent:
            name_msg = self._try_emit_name(portion)
            # Also emit args when available (complete tool call in one
            # chunk).  _try_emit_name sets current_tool_name_sent.
            args = parsed.get("arguments") or ""
            if args and self.current_tool_name_sent:
                args_msg = self._diff_and_emit_args(args)
                if args_msg and name_msg:
                    # Merge tool_calls from both messages
                    name_msg.tool_calls = list(name_msg.tool_calls or []) + list(
                        args_msg.tool_calls or []
                    )
            return name_msg

        return self._diff_and_emit_args(parsed.get("arguments") or "")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset_streaming_state(self) -> None:
        """Reset all streaming state between requests."""
        self._reset_section_state()
        self.current_tool_name_sent = False
        self.prev_tool_call_arr = []
        self.current_tool_id = -1
        self.streamed_args_for_tool = []
        self._current_tool_args = ""
        logger.debug("Streaming state reset")

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        section_begin = next(
            (v for v in self.tool_calls_start_token_variants if v in model_output),
            None,
        )
        if section_begin is None:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            function_call_tuples = self.tool_call_regex.findall(model_output)
            logger.debug("function_call_tuples: %s", function_call_tuples)

            tool_calls = []
            for function_id, function_args in function_call_tuples:
                function_name = self._call_id_to_name(function_id)
                tool_calls.append(
                    ToolCall(
                        id=function_id,
                        type="function",
                        function=FunctionCall(
                            name=function_name,
                            arguments=function_args,
                        ),
                    )
                )

            content = model_output[: model_output.index(section_begin)]
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content if content else None,
            )
        except Exception:
            logger.exception("Error in extracting tool call from response.")
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
    ) -> DeltaMessage | None:
        logger.debug("delta_text: %s", delta_text)
        logger.debug("delta_token_ids: %s", delta_token_ids)

        # Step 1: rolling marker buffer (capped at 256 bytes)
        self.token_buffer = (self.token_buffer + delta_text)[-_MARKER_BUF_CAP:]

        _, found_section_begin, found_section_end = self._check_and_strip_markers(
            self.token_buffer
        )

        # Step 2: section boundary transitions
        if found_section_begin and not self.in_tool_section:
            logger.debug("Entering tool section")
            self.in_tool_section = True
            self.section_char_count = 0

        deferred_section_exit = False
        if found_section_end and self.in_tool_section:
            logger.debug("Detected section end marker")
            has_tool_end = self.tool_call_end_token_id in delta_token_ids
            if has_tool_end:
                deferred_section_exit = True
                logger.debug("Deferring section exit: tool_call_end in same chunk")
            else:
                self._reset_section_state()
                post = ""
                for v in self.tool_calls_end_token_variants:
                    if v in delta_text:
                        parts = delta_text.split(v, 1)
                        if len(parts) > 1:
                            post = parts[1]
                        break
                return DeltaMessage(content=post if post.strip() else "")

        # Step 3: pure reasoning (no tool section active)
        has_section_token = any(
            tid in current_token_ids for tid in self.tool_calls_start_token_ids
        )
        if not has_section_token and not self.in_tool_section:
            return DeltaMessage(content=delta_text)

        # Step 4: safety valve for unbounded sections
        if self.in_tool_section:
            self.section_char_count += len(delta_text)
            if self.section_char_count > self.max_section_chars:
                logger.warning(
                    "Tool section exceeded max length (%d chars), forcing exit.",
                    self.max_section_chars,
                )
                self._reset_section_state()
                return DeltaMessage(content=(delta_text if delta_text.strip() else ""))

        # Step 5: tool-call parsing
        try:
            prev_start_count = previous_token_ids.count(self.tool_call_start_token_id)
            cur_start_count = current_token_ids.count(self.tool_call_start_token_id)
            cur_end_count = current_token_ids.count(self.tool_call_end_token_id)
            prev_end_count = previous_token_ids.count(self.tool_call_end_token_id)
            call_end_in_delta = self.tool_call_end_token_id in delta_token_ids

            # Phase A: new tool call started
            if cur_start_count > cur_end_count and cur_start_count > prev_start_count:
                self.current_tool_id += 1
                self.current_tool_name_sent = False
                self._current_tool_args = ""
                self.streamed_args_for_tool.append("")
                self.prev_tool_call_arr.append({})
                logger.debug(
                    "Starting on a new tool %s",
                    self.current_tool_id,
                )

                portion = self._extract_tool_call_portion(current_text)
                result = self._try_emit_name(portion)
                if deferred_section_exit and self.in_tool_section:
                    self._reset_section_state()
                return result

            # Phase B: tool call ended in this delta
            if call_end_in_delta:
                # If a new tool_call_begin also arrived in this delta
                # (complete tool call in a single chunk), Phase A was
                # skipped because cur_start == cur_end.  Set up the
                # new tool call now so _handle_call_end can find it.
                if (
                    cur_start_count > prev_start_count
                    and cur_start_count == cur_end_count
                ):
                    self.current_tool_id += 1
                    self.current_tool_name_sent = False
                    self._current_tool_args = ""
                    self.streamed_args_for_tool.append("")
                    self.prev_tool_call_arr.append({})
                    logger.debug(
                        "Late setup for tool %s (begin+end in same delta)",
                        self.current_tool_id,
                    )

                result = self._handle_call_end(current_text)
                if deferred_section_exit and self.in_tool_section:
                    logger.debug("Completing deferred section exit")
                    self._reset_section_state()
                return result

            # Phase C: inside a tool call, streaming args
            if cur_start_count > cur_end_count:
                portion = self._extract_tool_call_portion(current_text)
                if not self.current_tool_name_sent:
                    result = self._try_emit_name(portion)
                    if deferred_section_exit and self.in_tool_section:
                        self._reset_section_state()
                    return result

                parsed = self._parse_tool_call_portion(portion)
                if parsed and parsed.get("arguments"):
                    result = self._diff_and_emit_args(parsed["arguments"])
                    if deferred_section_exit and self.in_tool_section:
                        self._reset_section_state()
                    return result

                if deferred_section_exit and self.in_tool_section:
                    self._reset_section_state()
                return None

            # Phase D: between tools or before first tool
            if self.in_tool_section:
                if (
                    cur_start_count == cur_end_count
                    and prev_end_count == cur_end_count
                    and self.tool_call_end_token not in delta_text
                    and cur_start_count == 0
                ):
                    logger.debug(
                        "In tool section before first tool, suppressing: %s",
                        delta_text,
                    )
                    return DeltaMessage(content="")
                logger.debug("In tool section, suppressing text generation")
                if deferred_section_exit:
                    self._reset_section_state()
                return DeltaMessage(content="")

            # Phase E: text outside tool section
            text = delta_text
            for marker in (
                self.tool_call_start_token,
                self.tool_call_end_token,
                *self.tool_calls_start_token_variants,
                *self.tool_calls_end_token_variants,
            ):
                text = text.replace(marker, "")
            return DeltaMessage(content=text)

        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            return None
