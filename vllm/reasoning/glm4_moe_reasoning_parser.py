# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.protocol import ChatCompletionRequest, DeltaMessage
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParser

logger = init_logger(__name__)


class Glm4MoeModelReasoningParser(ReasoningParser):
    """
    Reasoning parser for the Glm4MoeModel model.

    The Glm4MoeModel model uses <think>...</think> tokens to denote reasoning
    text within its output. The model provides a strict switch to disable
    reasoning output via the 'enable_thinking=False' parameter. This parser
    extracts the reasoning content enclosed by <think> and </think> tokens
    from the model's output.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.think_start_token = "<think>"
        self.think_end_token = "</think>"
        self.assistant_token = "<|assistant|>"

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ReasoningParser "
                "constructor during construction."
            )

        self.think_start_token_id = self.vocab.get(self.think_start_token)
        self.think_end_token_id = self.vocab.get(self.think_end_token)
        self.assistant_token_id = self.vocab.get(self.assistant_token)
        if (
            self.think_start_token_id is None
            or self.think_end_token_id is None
            or self.assistant_token_id is None
        ):
            raise RuntimeError(
                "Glm4MoeModel reasoning parser could not locate "
                "think start/end or assistant tokens in the tokenizer!"
            )

    def _in_think_block(self, text: str) -> bool:
        """
        Return True if `text` ends while inside a <think>...</think> block.

        This is a purely text-based scan, which ensures correctness in
        multi-turn conversations, tool-call scenarios, and any case where
        tokenizer token IDs do not align with literal text layout.
        """
        last_start = text.rfind(self.think_start_token)
        last_end = text.rfind(self.think_end_token)

        if last_start == -1:
            return False
        if last_end == -1:
            return True
        return last_start > last_end

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """
        Extract reasoning content from a delta message.

        Handles streaming output where previous + delta = current.
        Uses a text-based parsing state machine that:
            • tracks whether we're "inside <think>" across turns
            • strips <think> / </think> tags from output
            • routes reasoning-only text to DeltaMessage.reasoning
            • routes post-reasoning text to DeltaMessage.content

        This fixes failures in multi-turn streaming with tool calls.
        """

        # Ignore chunks that are only special tags
        if delta_text in (self.think_start_token, self.think_end_token):
            return None
        if len(delta_token_ids) == 1 and delta_token_ids[0] in (
            self.think_start_token_id,
            self.think_end_token_id,
        ):
            return None

        # Determine whether we were already inside a reasoning block
        in_think = self._in_think_block(previous_text)

        reasoning_parts: list[str] = []
        content_parts: list[str] = []

        i = 0
        n = len(delta_text)

        while i < n:
            start_idx = delta_text.find(self.think_start_token, i)
            end_idx = delta_text.find(self.think_end_token, i)

            if start_idx == -1:
                start_idx = n + 1
            if end_idx == -1:
                end_idx = n + 1

            next_tag_idx = min(start_idx, end_idx)

            # No more tags: everything remaining is plain text
            if next_tag_idx > n:
                remainder = delta_text[i:]
                if remainder:
                    if in_think:
                        reasoning_parts.append(remainder)
                    else:
                        content_parts.append(remainder)
                break

            # Emit text prior to next tag
            if next_tag_idx > i:
                before = delta_text[i:next_tag_idx]
                if in_think:
                    reasoning_parts.append(before)
                else:
                    content_parts.append(before)

            # Process tag — update state but do not emit the tag itself
            if start_idx == next_tag_idx:
                in_think = True
                i = next_tag_idx + len(self.think_start_token)
            else:
                in_think = False
                i = next_tag_idx + len(self.think_end_token)

        reasoning = "".join(reasoning_parts) or None
        content = "".join(content_parts) or None

        if reasoning is None and content is None:
            return None

        return DeltaMessage(reasoning=reasoning, content=content)

    def extract_reasoning(
        self, model_output: str, request: ChatCompletionRequest
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning content from the model output.

        For text <think>abc</think>xyz:
        - 'abc' goes to reasoning
        - 'xyz' goes to content
        """

        if (
            self.think_start_token not in model_output
            or self.think_end_token not in model_output
        ):
            return None, model_output

        parts = model_output.partition(self.think_start_token)
        model_output = parts[2] if parts[1] else parts[0]

        if self.think_end_token not in model_output:
            return None, model_output

        reasoning, _, content = model_output.partition(self.think_end_token)
        return reasoning, (content or None)

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        """
        GLM's chat template has <think></think> tokens after every
        <|assistant|> token. Thus, we need to check if </think> is
        after the most recent <|assistant|> token (if present).
        """
        for token_id in reversed(input_ids):
            if token_id == self.think_end_token_id:
                return True
            if token_id == self.assistant_token_id:
                return False
        return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if self.think_end_token_id not in input_ids[:-1]:
            return []
        return input_ids[input_ids.index(self.think_end_token_id) + 1 :]
