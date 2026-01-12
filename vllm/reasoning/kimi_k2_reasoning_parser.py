# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from vllm.entrypoints.openai.protocol import DeltaMessage
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.protocol import (
        ChatCompletionRequest,
        ResponsesRequest,
    )
else:
    ChatCompletionRequest = Any
    ResponsesRequest = Any

# Tool call markers to detect when </think> is missing
TOOL_MARKERS = [
    "<|tool_calls_section_begin|>",
    "<|tool_call_section_begin|>",
    "<|tool_call_begin|>",
]


class KimiK2ReasoningParser(DeepSeekR1ReasoningParser):
    """
    Reasoning parser for Kimi K2 model.

    Kimi K2 uses <think>...</think> tokens like DeepSeek R1. However, it
    sometimes omits </think> before tool calls. This parser detects tool
    markers and splits the output at the first marker boundary.
    """

    def extract_reasoning(
        self, model_output: str, request: ChatCompletionRequest | ResponsesRequest
    ) -> tuple[str | None, str | None]:
        # Try standard parsing first (handles </think> case)
        reasoning, content = super().extract_reasoning(model_output, request)

        # If content was found, </think> was present - use parent's result
        if content is not None:
            return reasoning, content

        # No </think> found - check for tool markers
        # Parent returned the whole output as reasoning
        if reasoning is None:
            return None, None

        tool_positions = []
        for marker in TOOL_MARKERS:
            pos = reasoning.find(marker)
            if pos >= 0:
                tool_positions.append(pos)

        if tool_positions:
            first_marker_pos = min(tool_positions)
            new_reasoning = reasoning[:first_marker_pos].rstrip()
            content = reasoning[first_marker_pos:]
            return new_reasoning if new_reasoning else None, content

        return reasoning, None

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        # Try standard streaming parsing first
        ret = super().extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )

        # If parent found end token or returned content, use that result
        if ret is not None and (
            ret.content is not None or self.end_token_id in previous_token_ids
        ):
            return ret

        # Check if tool marker was already seen in previous text
        for marker in TOOL_MARKERS:
            if marker in previous_text:
                # Already in content mode - return delta as content
                return DeltaMessage(content=delta_text)

        # Check if tool markers appeared in delta (without </think>)
        if ret is not None and ret.reasoning is not None:
            # Find all marker positions in delta
            tool_positions = []
            for marker in TOOL_MARKERS:
                pos = delta_text.find(marker)
                if pos >= 0:
                    tool_positions.append(pos)

            if tool_positions:
                # Split at the first marker (by position)
                first_marker_pos = min(tool_positions)
                reasoning = delta_text[:first_marker_pos].rstrip()
                content = delta_text[first_marker_pos:]
                return DeltaMessage(
                    reasoning=reasoning if reasoning else None,
                    content=content if content else None,
                )

        return ret
