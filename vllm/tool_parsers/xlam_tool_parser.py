# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa
import json
from collections.abc import Sequence
from typing import Any, Optional, Union

import regex as re

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.tool_parsers.abstract_tool_parser import (
    Tool,
    ToolParser,
)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.utils import random_uuid

logger = init_logger(__name__)


class xLAMToolParser(ToolParser):
    # Streaming markers that can wrap the tool-call JSON array.
    _FENCE = "```"
    _FENCE_LANG = "json"
    _TOOL_CALLS_TAG = "[TOOL_CALLS]"
    _TOOL_CALL_XML = "<tool_call>"
    _TOOL_CALL_XML_END = "</tool_call>"

    _TOOL_NAME_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')
    _TOOL_ARGS_RE = re.compile(
        r'"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*'
        r"(\{(?:[^{}]|(?:\{[^{}]*\}))*\})"
    )

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)

        # Initialize state for streaming mode
        self.prev_tool_calls: list[dict] = []
        self.current_tool_id = -1
        self.current_tool_name_sent = False

        # For backward compatibility with tests
        self.current_tools_sent: list[bool] = []

        # For backward compatibility with serving code
        self.prev_tool_call_arr = []

        # Regex patterns for preprocessing
        self.json_code_block_patterns = [
            r"```(?:json)?\s*([\s\S]*?)```",
            r"\[TOOL_CALLS\]([\s\S]*?)(?=\n|$)",
            r"<tool_call>([\s\S]*?)</tool_call>",
        ]
        self.thinking_tag_pattern = r"</think>([\s\S]*)"

        # Streaming content scanner state: everything before _consumed has
        # been classified as content, whitespace, or tool block.
        self._consumed = 0
        self._pending_ws = ""
        self._content_emitted = False
        # Active tool block: {"json_start", "json_end", "close", "end"}.
        # "end" is the index just past the closing marker once seen.
        self._block: Optional[dict[str, Any]] = None

        self.streaming_state: dict[str, Any] = {
            "current_tool_index": -1,
            "tool_ids": [],
            "sent_tools": [],
        }

    def preprocess_model_output(
        self, model_output: str
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Preprocess the model output to extract content and potential tool calls.
        Returns:
            Tuple of (content, potential_tool_calls_json)
        """
        # Check for thinking tag
        thinking_match = re.search(self.thinking_tag_pattern, model_output)
        if thinking_match:
            content = model_output[: thinking_match.start() + len("</think>")].strip()
            thinking_content = thinking_match.group(1).strip()

            # Try to parse the thinking content as JSON
            try:
                json.loads(thinking_content)
                return content, thinking_content
            except json.JSONDecodeError:
                # If can't parse as JSON, look for JSON code blocks
                for json_pattern in self.json_code_block_patterns:
                    json_matches = re.findall(json_pattern, thinking_content)
                    if json_matches:
                        for json_str in json_matches:
                            try:
                                json.loads(json_str)
                                return content, json_str
                            except json.JSONDecodeError:
                                continue

        # Check for JSON code blocks in the entire output
        for json_pattern in self.json_code_block_patterns:
            json_matches = re.findall(json_pattern, model_output)
            if json_matches:
                for json_str in json_matches:
                    try:
                        json.loads(json_str)
                        # Extract content by removing the JSON code block
                        content = re.sub(json_pattern, "", model_output).strip()
                        return content, json_str
                    except json.JSONDecodeError:
                        continue

        # If the entire output is a valid JSON array or looks like one, treat it as tool calls
        if model_output.strip().startswith("["):
            try:
                json.loads(model_output)
                return None, model_output
            except json.JSONDecodeError:
                # Even if it's not valid JSON yet, it might be a tool call in progress
                if (
                    "{" in model_output
                    and "name" in model_output
                    and "arguments" in model_output
                ):
                    return None, model_output

        # If no tool calls found, return the original output as content
        return model_output, None

    def extract_tool_calls(
        self, model_output: str, request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from a complete model output.
        """
        try:
            # Preprocess the model output
            content, potential_tool_calls = self.preprocess_model_output(model_output)

            if not potential_tool_calls:
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=content
                )

            # Parse the potential tool calls as JSON
            tool_calls_data = json.loads(potential_tool_calls)

            # Ensure it's an array
            if not isinstance(tool_calls_data, list):
                logger.debug("Tool calls data is not an array")
                return ExtractedToolCallInformation(
                    tools_called=False,
                    tool_calls=[],
                    content=content or model_output,
                )

            tool_calls: list[ToolCall] = []

            for idx, call in enumerate(tool_calls_data):
                if (
                    not isinstance(call, dict)
                    or "name" not in call
                    or "arguments" not in call
                ):
                    logger.debug("Invalid tool call format at index %d", idx)
                    continue

                tool_call = ToolCall(
                    id=f"call_{idx}_{random_uuid()}",
                    type="function",
                    function=FunctionCall(
                        name=call["name"],
                        arguments=(
                            json.dumps(call["arguments"], ensure_ascii=False)
                            if isinstance(call["arguments"], dict)
                            else call["arguments"]
                        ),
                    ),
                )
                tool_calls.append(tool_call)

            return ExtractedToolCallInformation(
                tools_called=len(tool_calls) > 0,
                tool_calls=tool_calls,
                content=content,
            )

        except Exception as e:
            logger.exception("Error extracting tool calls: %s", str(e))
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
    ) -> Union[DeltaMessage, None]:
        """
        Extract tool calls for streaming mode.

        A single forward scan over ``current_text`` classifies each span as
        chat content, tool-block markup, or tool-call JSON. Content that
        could still turn out to be a wrapper marker is withheld until
        disambiguated, so markers and tool JSON never leak into
        ``delta.content``. Argument deltas are mirrored into
        ``self.streamed_args_for_tool`` so the serving layer's end-of-stream
        flush (``get_remaining_unstreamed_args``) sees what was actually
        sent and does not re-append the full argument string.
        """
        try:
            content = self._scan_content(current_text)

            tool_deltas: list[DeltaToolCall] = []
            if self._block is not None:
                region = self._block_region(current_text)
                self._try_parse_tools(region)

                legacy = self._handle_test_compatibility(current_text)
                if legacy is not None:
                    return legacy

                # Catch up on every state transition already visible in the
                # region, but emit at most one arguments delta per tool per
                # call so argument streaming stays incremental.
                emitted_args: set[int] = set()
                while True:
                    delta = self._step_tool_streaming(region, emitted_args)
                    if delta is None:
                        break
                    tool_deltas.append(delta)

            if content and tool_deltas:
                return DeltaMessage(content=content, tool_calls=tool_deltas)
            if tool_deltas:
                return DeltaMessage(tool_calls=tool_deltas)
            if content:
                return DeltaMessage(content=content)
            return None
        except Exception:
            logger.exception("Error in streaming tool call extraction")
            return None

    # ------------------------------------------------------------------
    # Content scanner
    # ------------------------------------------------------------------

    def _scan_content(self, current_text: str) -> str:
        """Advance over unclassified text, returning content safe to emit.

        Whitespace is withheld until more content follows, which mirrors the
        ``.strip()`` non-streaming extraction applies around the tool block.
        """
        out: list[str] = []

        def emit(text: str) -> None:
            if not text:
                return
            if self._content_emitted:
                out.append(self._pending_ws)
            self._pending_ws = ""
            out.append(text)
            self._content_emitted = True

        while self._consumed < len(current_text):
            if self._block is not None and self._block["end"] is None:
                self._locate_block_end(current_text)
                if self._block["end"] is None:
                    break
                self._consumed = self._block["end"]
                continue

            seg = current_text[self._consumed :]
            ch = seg[0]
            if ch.isspace():
                ws_len = len(seg) - len(seg.lstrip())
                self._pending_ws += seg[:ws_len]
                self._consumed += ws_len
                continue
            if ch in "`[<":
                verdict, advance = self._match_block_start(current_text)
                if verdict == "wait":
                    break
                if verdict == "open":
                    self._consumed += advance
                    continue
                emit(ch)
                self._consumed += 1
                continue
            safe = len(seg)
            for i, c in enumerate(seg):
                if c in "`[<" or c.isspace():
                    safe = i
                    break
            emit(seg[:safe])
            self._consumed += safe

        return "".join(out)

    def _match_block_start(self, current_text: str) -> tuple[str, int]:
        """Decide whether a tool block opens at ``self._consumed``.

        Returns ``("open", advance)`` once a block is confirmed,
        ``("wait", 0)`` while the tail could still become one, and
        ``("no", 0)`` when it is plain content.
        """
        if self._block is not None:
            # Only the first block is treated as tool calls, matching the
            # non-streaming path which parses the first valid JSON payload.
            return ("no", 0)

        pos = self._consumed
        seg = current_text[pos:]
        ch = seg[0]

        if ch == "`":
            if len(seg) < len(self._FENCE):
                return ("wait", 0) if self._FENCE.startswith(seg) else ("no", 0)
            if not seg.startswith(self._FENCE):
                return ("no", 0)
            body = seg[len(self._FENCE) :]
            if len(body) < len(self._FENCE_LANG) and self._FENCE_LANG.startswith(body):
                return ("wait", 0)
            if body.startswith(self._FENCE_LANG):
                body = body[len(self._FENCE_LANG) :]
            after_ws = body.lstrip()
            if not after_ws:
                return ("wait", 0)
            if after_ws[0] == "[":
                json_start = pos + len(seg) - len(after_ws)
                self._open_block(json_start, close=self._FENCE)
                return ("open", json_start - pos)
            return ("no", 0)

        if ch == "[":
            tag = self._TOOL_CALLS_TAG
            if seg.startswith(tag):
                after = seg[len(tag) :]
                if not after:
                    return ("wait", 0)
                if after[0] == "[":
                    self._open_block(pos + len(tag), close="\n")
                    return ("open", len(tag))
                return ("no", 0)
            if len(seg) < len(tag) and tag.startswith(seg):
                return ("wait", 0)
            # A bare JSON array counts only at the start of the output or
            # right after a </think> block (as in preprocess_model_output).
            before = current_text[:pos]
            if not self._content_emitted or before.rstrip().endswith("</think>"):
                self._open_block(pos, close=None)
                return ("open", 0)
            return ("no", 0)

        if ch == "<":
            tag = self._TOOL_CALL_XML
            if seg.startswith(tag):
                after_ws = seg[len(tag) :].lstrip()
                if not after_ws:
                    return ("wait", 0)
                if after_ws[0] == "[":
                    json_start = pos + len(seg) - len(after_ws)
                    self._open_block(json_start, close=self._TOOL_CALL_XML_END)
                    return ("open", json_start - pos)
                return ("no", 0)
            if len(seg) < len(tag) and tag.startswith(seg):
                return ("wait", 0)
            return ("no", 0)

        return ("no", 0)

    def _open_block(self, json_start: int, close: Optional[str]) -> None:
        self._block = {
            "json_start": json_start,
            "json_end": None,
            "close": close,
            "end": None,
        }

    def _locate_block_end(self, current_text: str) -> None:
        close = self._block["close"]
        if close is None:
            return
        idx = current_text.find(close, self._block["json_start"])
        if idx == -1:
            return
        self._block["json_end"] = idx
        # A newline closes a [TOOL_CALLS] block but stays part of content.
        self._block["end"] = idx if close == "\n" else idx + len(close)

    def _block_region(self, current_text: str) -> str:
        end = self._block["json_end"]
        if end is None:
            end = len(current_text)
        return current_text[self._block["json_start"] : end]

    # ------------------------------------------------------------------
    # Tool-call streaming
    # ------------------------------------------------------------------

    def _try_parse_tools(self, region: str) -> None:
        try:
            parsed, _ = json.JSONDecoder().raw_decode(region.strip())
        except (json.JSONDecodeError, ValueError):
            return
        if isinstance(parsed, list):
            self.prev_tool_call_arr = parsed

    def _handle_test_compatibility(self, current_text: str) -> Optional[DeltaMessage]:
        # Handles the case where tests manually set current_tools_sent.
        if not (
            len(self.current_tools_sent) == 1 and self.current_tools_sent[0] is False
        ):
            return None
        name_match = self._TOOL_NAME_RE.search(current_text)
        if not name_match:
            return None
        function_name = name_match.group(1)

        tool_id = make_tool_call_id()
        delta = DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=0,
                    type="function",
                    id=tool_id,
                    function=DeltaFunctionCall(name=function_name).model_dump(
                        exclude_none=True
                    ),  # type: ignore
                )
            ]
        )
        self.current_tools_sent = [True]
        self.current_tool_id = 0
        self.streaming_state["current_tool_index"] = 0
        if len(self.streaming_state["sent_tools"]) == 0:
            self.streaming_state["sent_tools"].append(
                {
                    "sent_name": True,
                    "sent_arguments_prefix": False,
                    "sent_arguments": "",
                }
            )
        else:
            self.streaming_state["sent_tools"][0]["sent_name"] = True
        self.current_tool_name_sent = True
        return delta

    def _step_tool_streaming(
        self, region: str, emitted_args: set[int]
    ) -> Optional[DeltaToolCall]:
        name_matches = list(self._TOOL_NAME_RE.finditer(region))
        tool_count = len(name_matches)
        if tool_count == 0:
            return None

        state = self.streaming_state
        while len(state["sent_tools"]) < tool_count:
            state["sent_tools"].append(
                {
                    "sent_name": False,
                    "sent_arguments_prefix": False,
                    "sent_arguments": "",
                }
            )
        while len(state["tool_ids"]) < tool_count:
            state["tool_ids"].append(None)
        while len(self.streamed_args_for_tool) < tool_count:
            self.streamed_args_for_tool.append("")

        idx = state["current_tool_index"]

        # Finish streaming the current tool's arguments before advancing.
        if 0 <= idx < tool_count:
            delta = self._stream_args_delta(region, idx, tool_count, emitted_args)
            if delta is not None:
                return delta

        if idx == -1 or (
            idx < tool_count - 1 and self._args_complete(region, idx, tool_count)
        ):
            next_idx = idx + 1
            if next_idx < tool_count and not state["sent_tools"][next_idx]["sent_name"]:
                state["current_tool_index"] = next_idx
                self.current_tool_id = next_idx
                tool_name = name_matches[next_idx].group(1)
                tool_id = f"call_{next_idx}_{random_uuid()}"
                state["tool_ids"][next_idx] = tool_id
                state["sent_tools"][next_idx]["sent_name"] = True
                self.current_tool_name_sent = True
                return DeltaToolCall(
                    index=next_idx,
                    type="function",
                    id=tool_id,
                    function=DeltaFunctionCall(name=tool_name).model_dump(
                        exclude_none=True
                    ),  # type: ignore
                )

        return None

    def _resolve_args_text(
        self, region: str, idx: int, tool_count: int
    ) -> Optional[str]:
        args_matches = list(self._TOOL_ARGS_RE.finditer(region))
        if idx >= len(args_matches):
            return None
        args_text = args_matches[idx].group(1)

        # For multiple tools, re-serialize from the parsed JSON so each
        # tool's arguments are extracted precisely.
        if tool_count > 1:
            try:
                parsed, _ = json.JSONDecoder().raw_decode(region.strip())
                if isinstance(parsed, list) and idx < len(parsed):
                    args = parsed[idx].get("arguments")
                    if isinstance(args, dict):
                        args_text = json.dumps(args, ensure_ascii=False)
                    elif args is not None:
                        args_text = str(args)
            except (json.JSONDecodeError, ValueError, KeyError):
                pass

        return args_text

    def _stream_args_delta(
        self, region: str, idx: int, tool_count: int, emitted_args: set[int]
    ) -> Optional[DeltaToolCall]:
        args_text = self._resolve_args_text(region, idx, tool_count)
        if args_text is None:
            return None
        tool_state = self.streaming_state["sent_tools"][idx]

        if not tool_state["sent_arguments_prefix"] and args_text.startswith("{"):
            if idx in emitted_args:
                return None
            emitted_args.add(idx)
            tool_state["sent_arguments_prefix"] = True
            tool_state["sent_arguments"] = "{"
            self.streamed_args_for_tool[idx] += "{"
            return DeltaToolCall(
                index=idx,
                function=DeltaFunctionCall(arguments="{").model_dump(exclude_none=True),  # type: ignore
            )

        sent = tool_state["sent_arguments"]
        if args_text.startswith(sent):
            args_diff = args_text[len(sent) :]
            if args_diff:
                if idx in emitted_args:
                    return None
                emitted_args.add(idx)
                tool_state["sent_arguments"] = args_text
                self.streamed_args_for_tool[idx] += args_diff
                return DeltaToolCall(
                    index=idx,
                    function=DeltaFunctionCall(arguments=args_diff).model_dump(
                        exclude_none=True
                    ),  # type: ignore
                )

        return None

    def _args_complete(self, region: str, idx: int, tool_count: int) -> bool:
        args_text = self._resolve_args_text(region, idx, tool_count)
        return (
            args_text is not None
            and args_text.endswith("}")
            and self.streaming_state["sent_tools"][idx]["sent_arguments"] == args_text
        )
