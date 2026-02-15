# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
)
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser
from vllm.tokenizers import TokenizerLike


class Step3p5ReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for Step3p5 model.

    Step3p5 uses the <think>...</think> format, but it tends to emit an extra
    newline immediately before and/or after the </think> token. This parser trims:
      - the newline right before </think>
      - the newline right after </think>
    """

    @property
    def start_token(self) -> str:
        return "<think>"

    @property
    def end_token(self) -> str:
        return "</think>"

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        # Used to hold a trailing "\n" from reasoning content so we can decide
        # whether it is immediately before </think>.
        self._pending_reasoning_newline = False

        # Tracks whether we've seen </think> but are still waiting for one more
        # token to confirm the end.
        self._end_token_pending = False

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        return self._is_reasoning_end_from_ids(input_ids)

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Sequence[int]
    ) -> bool:
        # Only examine newly generated tokens; they may contain multiple ids.
        return self._is_reasoning_end_from_ids(delta_ids)

    def _is_reasoning_end_from_ids(self, input_ids: Sequence[int]) -> bool:
        # Scan backwards to find the last special token, <think> or </think>.
        last_special = None
        last_idx = -1
        for i in range(len(input_ids) - 1, -1, -1):
            token_id = input_ids[i]
            if token_id == self.start_token_id:
                last_special = "start"
                last_idx = i
                break
            if token_id == self.end_token_id:
                last_special = "end"
                last_idx = i
                break

        if last_special == "start":
            # If we're already waiting for one token after </think>, do not
            # clear the pending state just because the prompt contains <think>.
            # Streaming deltas should not include <think> for this model.
            if self._end_token_pending:
                return False
            # A start token after any end token means reasoning is ongoing.
            self._end_token_pending = False
            return False

        if last_special == "end":
            # Require at least one token after </think> before ending.
            if last_idx < len(input_ids) - 1:
                self._end_token_pending = False
                return True
            self._end_token_pending = True
            return False

        # No special tokens in this input. If we were waiting for one token
        # after </think>, any new token completes the end.
        if self._end_token_pending and input_ids:
            self._end_token_pending = False
            return True

        return False

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        reasoning, content = super().extract_reasoning(model_output, request)
        if reasoning is not None:
            reasoning = reasoning.removesuffix("\n")
        if content is not None:
            content = content.removeprefix("\n")
        return reasoning or None, content or None

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        # Drop the immediate newline that models often emit after </think>.
        if previous_text.endswith(self.end_token) and delta_text:
            if delta_text == "\n":
                return None
            elif delta_text.startswith("\n"):
                remaining = delta_text.removeprefix("\n")
                return DeltaMessage(content=remaining) if remaining else None

        ret = super().extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )

        if ret is None:
            return None

        # Compatibility path for models that don't generate the start token:
        # treat everything before </think> as reasoning and everything after
        # as content.
        if (
            self.start_token_id not in previous_token_ids
            and self.start_token_id not in delta_token_ids
        ):
            if self.end_token_id in delta_token_ids:
                end_index = delta_text.find(self.end_token)
                reasoning = delta_text[:end_index]
                content = delta_text[end_index + len(self.end_token) :]
                ret = DeltaMessage(reasoning=reasoning, content=content or None)
            elif self.end_token_id in previous_token_ids:
                ret = DeltaMessage(content=delta_text)
            else:
                ret = DeltaMessage(reasoning=delta_text)

        reasoning_to_output = ret.reasoning
        content_to_output = ret.content

        # Reasoning: handle the newline immediately before </think>.
        if reasoning_to_output is not None:
            if self._pending_reasoning_newline:
                reasoning_to_output = "\n" + reasoning_to_output
                self._pending_reasoning_newline = False

            if reasoning_to_output.endswith("\n"):
                reasoning_to_output = reasoning_to_output.removesuffix("\n")
                if self.end_token in delta_text:
                    # Trailing "\n" is right before </think>, drop it.
                    self._pending_reasoning_newline = False
                else:
                    # Hold the trailing "\n" until we know whether </think> follows.
                    self._pending_reasoning_newline = True

        # Content: handle the newline immediately after </think>.
        if content_to_output is not None:
            # If we have content, reasoning must have ended.
            self._pending_reasoning_newline = False

            if self.end_token in delta_text and content_to_output.startswith("\n"):
                content_to_output = content_to_output.removeprefix("\n")

        reasoning_to_output = reasoning_to_output or None
        content_to_output = content_to_output or None
        if reasoning_to_output is None and content_to_output is None:
            return None

        return DeltaMessage(reasoning=reasoning_to_output, content=content_to_output)
