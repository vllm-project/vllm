# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Tool call parser for "Cursor-style" tool tags (named "cwm" in vLLM).

Supported format (one or more tool blocks):

    <think>...</think>
    <tool: function>
    terminal find . -name "*.py" | xargs grep -n "def bulk_create"
    </tool>

The `<tool: ...>` tag label is treated as informational; the function name is
extracted from the first token inside the tool body ("terminal" in the example).

Arguments are extracted as:
  - If the remaining text parses as JSON object/array: use that as arguments.
  - Otherwise: wrap as {"command": "<remaining text>"}.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import regex as re

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.tool_parsers.abstract_tool_parser import ToolParser

logger = init_logger(__name__)


class CwmToolParser(ToolParser):
    """
    Parser for:
      - `<tool: function> ... </tool>`

    Used when `--enable-auto-tool-choice --tool-call-parser cwm` are set.
    """

    # <tool: function> ... </tool>
    _TOOL_BLOCK_RE = re.compile(
        r"<tool:\s*(?P<label>[^>\s]+)\s*>\s*(?P<body>[\s\S]*?)\s*</tool>",
        re.IGNORECASE,
    )

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        # Streaming bookkeeping.
        self._emitted_tool_calls: int = 0
        self._in_tool_block: bool = False

        # Compatibility fields (serving code may inspect these).
        self.prev_tool_call_arr: list[dict] = []
        self.streamed_args_for_tool: list[str] = []
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False

    @staticmethod
    def _parse_tool_body_to_name_and_args(body: str) -> tuple[str, str] | None:
        """
        Parse the tool body into (tool_name, arguments_json_string).

        The tool name is the first token of the body; everything after it is
        treated as the arguments payload.
        """
        text = (body or "").strip()
        if not text:
            return None

        # First non-empty line.
        lines = [ln for ln in text.splitlines() if ln.strip() != ""]
        if not lines:
            return None

        first = lines[0].strip()
        parts = first.split(maxsplit=1)
        tool_name = parts[0]
        rest_first_line = parts[1] if len(parts) > 1 else ""

        remaining_lines = lines[1:]
        if rest_first_line and remaining_lines:
            remaining = "\n".join([rest_first_line] + remaining_lines).strip()
        elif remaining_lines:
            remaining = "\n".join(remaining_lines).strip()
        else:
            remaining = rest_first_line.strip()

        # If the remaining payload looks like JSON, accept it as-is (canonicalized).
        if remaining:
            stripped = remaining.lstrip()
            if stripped.startswith("{") or stripped.startswith("["):
                try:
                    parsed = json.loads(remaining)
                    # Canonicalize to compact JSON string.
                    args_json = json.dumps(parsed, ensure_ascii=False)
                    return tool_name, args_json
                except json.JSONDecodeError:
                    pass

        # Otherwise, wrap plain text under "command".
        args_json = json.dumps({"command": remaining}, ensure_ascii=False)
        return tool_name, args_json

    @classmethod
    def _extract_all_tool_calls_from_text(cls, text: str) -> tuple[list[ToolCall], str]:
        tool_calls: list[ToolCall] = []

        # Content is the prefix before the first tool block.
        first_match = cls._TOOL_BLOCK_RE.search(text)
        content = text[: first_match.start()].rstrip() if first_match else text
        content = content if content.strip() else None

        for idx, match in enumerate(cls._TOOL_BLOCK_RE.finditer(text)):
            parsed = cls._parse_tool_body_to_name_and_args(match.group("body"))
            if parsed is None:
                continue
            tool_name, args_json = parsed
            tool_calls.append(
                ToolCall(
                    id=make_tool_call_id(func_name=tool_name, idx=idx),
                    type="function",
                    function=FunctionCall(name=tool_name, arguments=args_json),
                )
            )

        return tool_calls, content

    def extract_tool_calls(
        self, model_output: str, request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        if "<tool:" not in model_output.lower():
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            tool_calls, content = self._extract_all_tool_calls_from_text(model_output)
            return ExtractedToolCallInformation(
                tools_called=len(tool_calls) > 0,
                tool_calls=tool_calls,
                content=content,
            )
        except Exception:
            logger.exception("Error extracting Cwm-style tool calls.")
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
        """
        Minimal streaming implementation:
          - Stream regular content until `<tool:` appears.
          - Do not stream tool block text.
          - Once a complete `</tool>` is observed, emit any newly completed tool calls
            as a single DeltaMessage(tool_calls=[...]).
        """
        cur_lower = current_text.lower()

        # Detect entering a tool block (stop emitting content).
        if not self._in_tool_block and "<tool:" in cur_lower:
            self._in_tool_block = True

            # If the `<tool:` tag arrived mid-delta, only emit the prefix before it.
            tool_pos = cur_lower.find("<tool:")
            prev_len = len(previous_text)
            new_prefix = current_text[:tool_pos]
            if len(new_prefix) > prev_len:
                return DeltaMessage(content=new_prefix[prev_len:])
            return None

        # If we haven't started a tool block, keep streaming content.
        if not self._in_tool_block:
            return DeltaMessage(content=delta_text)

        # Inside a tool block: wait until a closing tag is present.
        if "</tool>" not in cur_lower:
            return None

        # We have at least one complete tool call; extract all and emit the new ones.
        try:
            tool_calls, _ = self._extract_all_tool_calls_from_text(current_text)
        except Exception:
            logger.exception("Error extracting streaming Cwm-style tool calls.")
            return None

        if self._emitted_tool_calls >= len(tool_calls):
            # Might have just closed a tool block but nothing new parsed.
            return None

        new_calls = tool_calls[self._emitted_tool_calls :]
        tool_deltas: list[DeltaToolCall] = []

        for i, call in enumerate(new_calls, start=self._emitted_tool_calls):
            # Update compatibility arrays.
            while len(self.prev_tool_call_arr) <= i:
                self.prev_tool_call_arr.append({"name": "", "arguments": ""})
            while len(self.streamed_args_for_tool) <= i:
                self.streamed_args_for_tool.append("")

            self.prev_tool_call_arr[i]["name"] = call.function.name
            self.prev_tool_call_arr[i]["arguments"] = call.function.arguments
            self.streamed_args_for_tool[i] = call.function.arguments or ""

            tool_deltas.append(
                DeltaToolCall(
                    index=i,
                    id=call.id,
                    type="function",
                    function=DeltaFunctionCall(
                        name=call.function.name, arguments=call.function.arguments
                    ),
                )
            )

        self._emitted_tool_calls = len(tool_calls)

        # If all tool blocks are closed, allow emitting content again.
        open_count = cur_lower.count("<tool:")
        close_count = cur_lower.count("</tool>")
        if close_count >= open_count:
            self._in_tool_block = False

        return DeltaMessage(tool_calls=tool_deltas)


