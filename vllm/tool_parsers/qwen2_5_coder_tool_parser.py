# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from collections.abc import Sequence
from typing import Any

import regex as re

from vllm.entrypoints.chat_utils import make_tool_call_id
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
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser

logger = init_logger(__name__)


def _partial_tag_overlap(text: str, tag: str) -> int:
    """Length of the longest prefix of `tag` that matches a suffix of `text`.

    E.g. text ending in "<too" returns 4 when tag is "<tools>".
    Returns 0 if there is no overlap.
    """
    max_check = min(len(tag) - 1, len(text))
    for k in range(max_check, 0, -1):
        if text.endswith(tag[:k]):
            return k
    return 0


class Qwen25CoderToolParser(ToolParser):
    """Parser for Qwen2.5-Coder <tools> tag format.

    Requires system prompt with few-shot <tools> examples.
    Not applicable to Qwen2.5 (non-Coder) which uses hermes natively.
    """

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)

        self.tool_call_start_token: str = "<tools>"
        self.tool_call_end_token: str = "</tools>"

        self.tool_call_regex = re.compile(
            r"<tools>\s*(.*?)\s*</tools>|<tools>\s*(.*)", re.DOTALL
        )
        self.tool_call_closed_regex = re.compile(
            r"<tools>\s*(.*?)\s*</tools>", re.DOTALL
        )

        self._sent_content_idx: int = 0

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from a complete model response."""
        if self.tool_call_start_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        try:
            tool_calls: list[ToolCall] = []
            text_parts: list[str] = []
            last_end = 0

            for match in self.tool_call_regex.finditer(model_output):
                if match.start() > last_end:
                    text_parts.append(model_output[last_end : match.start()])

                # group(1) = closed tag body, group(2) = unclosed tag body
                json_str = (match.group(1) or match.group(2) or "").strip()
                parsed = self._parse_tool_json(json_str)

                if parsed:
                    for tool_data in parsed:
                        tool_calls.append(
                            ToolCall(
                                id=make_tool_call_id(),
                                type="function",
                                function=FunctionCall(
                                    name=tool_data["name"],
                                    arguments=json.dumps(
                                        tool_data.get("arguments", {}),
                                        ensure_ascii=False,
                                    ),
                                ),
                            )
                        )

                last_end = match.end()

            if last_end < len(model_output):
                text_parts.append(model_output[last_end:])

            content = "".join(text_parts).strip() or None

            return ExtractedToolCallInformation(
                tools_called=len(tool_calls) > 0,
                tool_calls=tool_calls if tool_calls else [],
                content=content,
            )

        except Exception:
            logger.exception("Error in extracting tool call from response.")
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

    def _extract_content(self, current_text: str) -> str | None:
        """Return unsent non-tool-call text, or None.

        Holds back any suffix that could be a partial <tools> tag, so that
        token-level streaming does not leak partial prefixes (e.g. "<", "<t",
        "<to") to the client before the tag is fully formed.
        """
        start = self.tool_call_start_token
        end = self.tool_call_end_token

        if start not in current_text:
            overlap = _partial_tag_overlap(current_text, start)
            sendable_idx = len(current_text) - overlap
        elif current_text.count(start) > current_text.count(end):
            # Inside an unclosed tag — sendable up to the open tag.
            sendable_idx = current_text.index(start)
        else:
            # All tags closed — skip past last </tools>, then hold back any
            # partial <tools> prefix at the tail (could be a new call starting).
            after_pos = current_text.rfind(end) + len(end)
            if self._sent_content_idx < after_pos:
                self._sent_content_idx = after_pos
            overlap = _partial_tag_overlap(current_text[after_pos:], start)
            sendable_idx = len(current_text) - overlap

        if sendable_idx > self._sent_content_idx:
            content = current_text[self._sent_content_idx : sendable_idx]
            self._sent_content_idx = sendable_idx
            return content
        return None

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
        """Extract tool calls in streaming mode, holding back partial tags."""
        content = self._extract_content(current_text)

        if self.tool_call_start_token not in current_text:
            if content:
                return DeltaMessage(content=content)
            return None

        try:
            current_matches = list(self.tool_call_closed_regex.finditer(current_text))
            prev_matches = list(self.tool_call_closed_regex.finditer(previous_text))

            delta_tool_calls: list[DeltaToolCall] | None = None
            if len(current_matches) > len(prev_matches):
                delta_tool_calls = []
                for new_match in current_matches[len(prev_matches) :]:
                    json_str = new_match.group(1).strip()
                    parsed = self._parse_tool_json(json_str)

                    if parsed:
                        for tool_data in parsed:
                            self.current_tool_id += 1
                            delta_tool_calls.append(
                                DeltaToolCall(
                                    index=self.current_tool_id,
                                    id=make_tool_call_id(),
                                    type="function",
                                    function=DeltaFunctionCall(
                                        name=tool_data["name"],
                                        arguments=json.dumps(
                                            tool_data.get("arguments", {}),
                                            ensure_ascii=False,
                                        ),
                                    ),
                                )
                            )
                if delta_tool_calls:
                    end_of_last_tag = current_matches[-1].end()
                    if end_of_last_tag > self._sent_content_idx:
                        self._sent_content_idx = end_of_last_tag
                else:
                    delta_tool_calls = None

            # Inside an unclosed <tools> tag — buffer until closed.
            if current_text.count(self.tool_call_start_token) > current_text.count(
                self.tool_call_end_token
            ):
                if content or delta_tool_calls:
                    return DeltaMessage(content=content, tool_calls=delta_tool_calls)
                return None

            # All tags closed — pick up trailing text after </tools>, which may
            # have arrived in the same delta as the closing tag.
            trailing = self._extract_content(current_text)
            parts = [p for p in (content, trailing) if p]
            combined = "".join(parts) or None

            if combined or delta_tool_calls:
                return DeltaMessage(content=combined, tool_calls=delta_tool_calls)

            return None

        except Exception:
            logger.exception("Error in streaming tool call extraction.")
            return DeltaMessage(content=delta_text)

    def _parse_tool_json(self, json_str: str) -> list[dict[str, Any]]:
        """Parse <tools> body as a single object, array, or JSONL.

        Falls back to a double-escape-normalized retry for models that
        emit ``\\"`` instead of ``\"`` inside JSON strings.
        """
        result = self._try_parse_json(json_str)
        if result:
            return result

        # Retry after normalizing double-escaped quotes (\\" -> \")
        # Some models (e.g. 14B) double-escape quotes in JSON output
        if "\\\\" in json_str or '\\"' in json_str:
            normalized = json_str.replace('\\\\"', '\\"')
            result = self._try_parse_json(normalized)
            if result:
                return result

        logger.warning("Failed to parse tool JSON: %s...", json_str[:100])
        return []

    def _try_parse_json(self, json_str: str) -> list[dict[str, Any]]:
        """Try parsing JSON in multiple formats."""
        # 1. Try as single JSON (object or array)
        try:
            parsed = json.loads(json_str)

            if isinstance(parsed, list):
                result = []
                for item in parsed:
                    if self._is_valid_tool_call(item):
                        result.append(item)
                return result

            if self._is_valid_tool_call(parsed):
                return [parsed]

            return []

        except json.JSONDecodeError:
            pass

        # 2. Try as JSONL (newline-separated JSON objects)
        result = []
        for line in json_str.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            line = line.rstrip(",")
            if not line:
                continue
            try:
                parsed = json.loads(line)
                if self._is_valid_tool_call(parsed):
                    result.append(parsed)
            except json.JSONDecodeError:
                continue

        if result:
            return result

        # 3. Try as comma-separated objects (wrap in array)
        # e.g., "{...}, {...}" -> "[{...}, {...}]"
        wrapped = f"[{json_str.strip()}]"
        try:
            parsed = json.loads(wrapped)
            if isinstance(parsed, list):
                for item in parsed:
                    if self._is_valid_tool_call(item):
                        result.append(item)
                if result:
                    return result
        except json.JSONDecodeError:
            pass

        return []

    def _is_valid_tool_call(self, obj: Any) -> bool:
        """Check if obj is a valid tool call dict (has 'name' string)."""
        if not isinstance(obj, dict):
            return False
        if "name" not in obj:
            return False
        return isinstance(obj["name"], str)
