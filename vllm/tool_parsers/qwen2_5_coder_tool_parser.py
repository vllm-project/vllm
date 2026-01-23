# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import re
from collections.abc import Sequence
from typing import Any

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
from vllm.tool_parsers.abstract_tool_parser import ToolParser

logger = init_logger(__name__)


class Qwen25CoderToolParser(ToolParser):
    """Parser for Qwen2.5-Coder <tools> tag format.

    Requires system prompt with few-shot <tools> examples.
    Not applicable to Qwen2.5 (non-Coder) which uses hermes natively.
    """

    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)

        # <tools> tag tokens
        self.tool_call_start_token: str = "<tools>"
        self.tool_call_end_token: str = "</tools>"

        # Streaming state
        self.current_tool_id: int = -1

        # Pattern: closed tag or unclosed tag (streaming edge case)
        self.tool_call_regex = re.compile(
            r"<tools>\s*(.*?)\s*</tools>|<tools>\s*(.*)", re.DOTALL
        )

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from model output (non-streaming).

        Args:
            model_output: Full model response text
            request: Original chat completion request

        Returns:
            ExtractedToolCallInformation with parsed tool calls
        """
        # No <tools> tag found — return as plain text
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

                # Parse JSON (group(1)=closed tag, group(2)=unclosed tag)
                json_str = (match.group(1) or match.group(2) or "").strip()
                parsed = self._parse_tool_json(json_str)

                if parsed:
                    for tool_data in parsed:
                        tool_calls.append(
                            ToolCall(
                                id=f"tool_{len(tool_calls)}",
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
        Extract tool calls in streaming mode.

        Args:
            previous_text: Accumulated text up to the previous token
            current_text: Accumulated text up to the current token
            delta_text: Newly added text
            previous_token_ids: Previous token IDs
            current_token_ids: Current token IDs
            delta_token_ids: New token IDs
            request: Original chat completion request

        Returns:
            DeltaMessage or None
        """
        # No <tools> tag yet — send as regular content
        if self.tool_call_start_token not in current_text:
            return DeltaMessage(content=delta_text)

        try:
            pattern = re.compile(r"<tools>\s*(.*?)\s*</tools>", re.DOTALL)
            current_matches = list(pattern.finditer(current_text))
            prev_matches = list(pattern.finditer(previous_text))

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
                                    id=f"tool_{self.current_tool_id}",
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
                    return DeltaMessage(tool_calls=delta_tool_calls)

            # Inside an unclosed <tools> tag — buffer until closed
            if current_text.count(self.tool_call_start_token) > current_text.count(
                self.tool_call_end_token
            ):
                return None

            # Send delta text after the last </tools>
            last_end_pos = current_text.rfind(self.tool_call_end_token)
            if last_end_pos != -1:
                after_tag = current_text[last_end_pos + len(self.tool_call_end_token) :]
                if self.tool_call_end_token in previous_text:
                    prev_end = previous_text.rfind(self.tool_call_end_token) + len(
                        self.tool_call_end_token
                    )
                    prev_after = previous_text[prev_end:]
                else:
                    prev_after = ""

                new_content = after_tag[len(prev_after) :]
                if new_content:
                    return DeltaMessage(content=new_content)

            return None

        except Exception:
            logger.exception("Error in streaming tool call extraction.")
            return DeltaMessage(content=delta_text)

    def _parse_tool_json(self, json_str: str) -> list[dict[str, Any]]:
        """
        Parse a JSON string into a list of tool call dicts.

        Supported formats:
            1. Single object: {"name": "...", "arguments": {...}}
            2. Array: [{"name": "..."}, {"name": "..."}]
            3. JSONL (newline-separated): {"name": "..."}\n{"name": "..."}

        Args:
            json_str: JSON string extracted from <tools> tags

        Returns:
            List of tool call dicts [{"name": "...", "arguments": {...}}, ...]
        """
        result = self._try_parse_json(json_str)
        if result:
            return result

        # Retry after normalizing double-escaped quotes (\\" -> \")
        # Some models (e.g. 14B) double-escape quotes in JSON output
        if "\\\\" in json_str or '\\"' in json_str:
            # \\" -> \"
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

            # Array
            if isinstance(parsed, list):
                result = []
                for item in parsed:
                    if self._is_valid_tool_call(item):
                        result.append(item)
                return result

            # Single object
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
