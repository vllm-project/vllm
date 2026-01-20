# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    ExtractedToolCallInformation,
)
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.hermes_tool_parser import Hermes2ProToolParser


class TrinityToolParser(Hermes2ProToolParser):
    """
    Tool parser for Trinity models using Qwen-style tool call format:

    <tool_call>
    {"name":"func1", "arguments":{...}}
    </tool_call>

    This is essentially a Hermes parser that strips <think>...</think> tags
    before parsing tool calls.
    """

    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)
        self.think_start_token = "<think>"
        self.think_end_token = "</think>"
        self._think_buffer = ""

    def _strip_think_tags(self, text: str) -> str:
        """Remove think tags from text (non-streaming)."""
        return text.replace(self.think_start_token, "").replace(
            self.think_end_token, ""
        )

    def _ends_with_partial_token(self, text: str, token: str) -> int:
        """Check if text ends with a partial match of token."""
        max_len = min(len(text), len(token) - 1)
        for i in range(max_len, 0, -1):
            if text.endswith(token[:i]):
                return i
        return 0

    def _strip_think_tags_streaming(self, text: str) -> str:
        """Remove think tags from text (streaming-aware, handles partial tags)."""
        if not text:
            return ""
        combined = self._think_buffer + text
        combined = combined.replace(self.think_start_token, "").replace(
            self.think_end_token, ""
        )
        pending_len = max(
            self._ends_with_partial_token(combined, self.think_start_token),
            self._ends_with_partial_token(combined, self.think_end_token),
        )
        if pending_len:
            self._think_buffer = combined[-pending_len:]
            return combined[:-pending_len]
        self._think_buffer = ""
        return combined

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        # Strip think tags before delegating to parent
        cleaned_output = self._strip_think_tags(model_output)
        return super().extract_tool_calls(cleaned_output, request)

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
        # Reset think buffer on first call
        if not previous_text:
            self._think_buffer = ""

        # Strip think tags from delta and reconstruct texts
        cleaned_delta = self._strip_think_tags_streaming(delta_text)
        cleaned_previous = self._strip_think_tags(previous_text)
        cleaned_current = cleaned_previous + cleaned_delta

        return super().extract_tool_calls_streaming(
            cleaned_previous,
            cleaned_current,
            cleaned_delta,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
            request,
        )
