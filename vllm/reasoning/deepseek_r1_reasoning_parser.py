# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

from vllm.entrypoints.openai.protocol import DeltaMessage
from vllm.entrypoints.tool_server import ToolServer
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser


class DeepSeekR1ReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for DeepSeek R1 model.

    The DeepSeek R1 model uses <think>...</think> tokens to denote reasoning
    text. This parser extracts the reasoning content from the model output.
    """

    @property
    def start_token(self) -> str:
        """The token that starts reasoning content."""
        return "<think>"

    @property
    def end_token(self) -> str:
        """The token that ends reasoning content."""
        return "</think>"

    def prepare_structured_tag(
        self, original_tag: str | None, tool_server: ToolServer | None
    ) -> str:
        """
        Prepare structural tag for Kimi K2 / DeepSeek R1 models to enforce
        proper reasoning + tool call structure.

        This prevents early EOS sampling by constraining the model to always
        generate tool calls when tools are available.
        """
        if original_tag is None:
            if tool_server is None or not tool_server.has_builtin_tools():
                # No tools available: just reasoning structure
                structural_tag = {
                    "type": "structural_tag",
                    "format": {
                        "type": "triggered_tags",
                        "tags": [
                            {
                                "begin": "<think>",
                                "content": {"type": "any_text"},
                                "end": "</think>",
                            }
                        ],
                        "triggers": ["<think>"],
                        "stop_after_first": False,
                    },
                }
            else:
                # Tools available: enforce reasoning + tool calls structure
                structural_tag = {
                    "type": "structural_tag",
                    "format": {
                        "type": "triggered_tags",
                        "tags": [
                            {
                                "begin": "<think>",
                                "content": {"type": "any_text"},
                                "end": "</think>",
                            },
                            {
                                "begin": "<|tool_calls_section_begin|>",
                                "content": {"type": "any_text"},
                                "end": "<|tool_calls_section_end|>",
                            },
                        ],
                        "triggers": ["<think>", "<|tool_calls_section_begin|>"],
                        "stop_after_first": False,
                    },
                }
            import json
            return json.dumps(structural_tag)
        else:
            return original_tag

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        ret = super().extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )
        if (
            ret is not None
            and self.start_token_id not in previous_token_ids
            and self.start_token_id not in delta_token_ids
        ):
            if self.end_token_id in delta_token_ids:
                # end token in delta with more tokens,
                # extract reasoning content and content
                end_index = delta_text.find(self.end_token)
                reasoning = delta_text[:end_index]
                content = delta_text[end_index + len(self.end_token) :]
                return DeltaMessage(
                    reasoning=reasoning,
                    content=content if content else None,
                )
            elif self.end_token_id in previous_token_ids:
                # end token in previous, thinking content ends
                return DeltaMessage(content=delta_text)
            else:
                # no end token in previous or delta, reasoning content continues
                return DeltaMessage(reasoning=delta_text)

        return ret
