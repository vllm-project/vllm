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
from vllm.logger import init_logger
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser
from vllm.tokenizers import TokenizerLike

logger = init_logger(__name__)


class Qwen3ReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for the Qwen3/Qwen3.5 model family.

    The Qwen3 model family uses <think>...</think> tokens to denote reasoning
    text. Starting with Qwen3.5, the chat template places <think> in the
    prompt so only </think> appears in the generated output. The model
    provides a strict switch to disable reasoning output via the
    'enable_thinking=False' parameter.

    When thinking is disabled, the template places <think>\\n\\n</think>\\n\\n
    in the prompt. The serving layer detects this via prompt_is_reasoning_end
    and routes deltas as content without calling the streaming parser.

    NOTE: Models up to the 2507 release (e.g., Qwen/Qwen3-235B-A22B-Instruct-2507)
    use an older chat template where the model generates <think> itself.
    This parser handles both styles: if <think> appears in the generated output
    it is stripped before extraction (non-streaming) or skipped (streaming).

    NOTE: When the model generates <tool_call> before </think> (a known
    failure mode), <tool_call> is treated as an implicit end-of-reasoning
    boundary so that tool calls are not swallowed into reasoning content.
    """

    TOOL_CALL_TOKEN = "<tool_call>"

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        # Qwen3 defaults to thinking enabled; only treat output as
        # pure content when the user explicitly disables it.
        self.thinking_enabled = chat_kwargs.get("enable_thinking", True)
        self.tool_call_start_token_id: int | None = self.vocab.get(
            self.TOOL_CALL_TOKEN
        )

    @property
    def start_token(self) -> str:
        """The token that starts reasoning content."""
        return "<think>"

    @property
    def end_token(self) -> str:
        """The token that ends reasoning content."""
        return "</think>"

    def _reasoning_end_index(self, input_ids: Sequence[int]) -> int | None:
        """Return the index of the first reasoning-end token (</think> or
        <tool_call>), or None if neither has appeared yet."""
        for i, tid in enumerate(input_ids):
            if tid == self.end_token_id or tid == self.tool_call_start_token_id:
                return i
        return None

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        return self._reasoning_end_index(input_ids) is not None

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        idx = self._reasoning_end_index(input_ids)
        if idx is None:
            return input_ids
        # For </think>: content starts after the token.
        # For <tool_call>: content starts AT the token (keep it for tool parser).
        if input_ids[idx] == self.end_token_id:
            return input_ids[idx + 1:]
        return input_ids[idx:]

    def extract_reasoning(
        self, model_output: str, request: ChatCompletionRequest | ResponsesRequest
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning content from the model output.

        The <think> token is placed in the prompt by the chat template,
        so typically only </think> appears in the generated output.
        If <think> is present (e.g. from a different template), it is
        stripped before extraction.

        When thinking is explicitly disabled and no </think> appears,
        returns (None, model_output) — all output is content.
        Otherwise (thinking enabled, default), a missing </think> means
        the output was truncated and everything is reasoning:
        returns (model_output, None).

        If <tool_call> appears before </think>, it is treated as an
        implicit end-of-reasoning boundary to avoid swallowing tool calls
        into reasoning content.

        Returns:
            tuple[Optional[str], Optional[str]]: reasoning content and content
        """

        # Strip <think> if present in the generated output.
        model_output_parts = model_output.partition(self.start_token)
        model_output = (
            model_output_parts[2] if model_output_parts[1] else model_output_parts[0]
        )

        think_end = model_output.find(self.end_token)
        tool_call_start = model_output.find(self.TOOL_CALL_TOKEN)

        if think_end == -1 and tool_call_start == -1:
            if not self.thinking_enabled:
                # Thinking explicitly disabled — treat everything as content.
                return None, model_output
            # Thinking enabled but no end marker: output was truncated.
            return model_output, None

        # Use whichever end marker comes first.
        if think_end != -1 and (tool_call_start == -1 or think_end <= tool_call_start):
            # Normal case: </think> is the boundary.
            reasoning, _, content = model_output.partition(self.end_token)
            return reasoning, content or None

        # <tool_call> appears before </think>: treat it as implicit end.
        # Keep <tool_call> in content so the tool parser can see it.
        logger.warning(
            "Model generated <tool_call> before </think> — tool call was "
            "inside the reasoning block. Treating <tool_call> as implicit "
            "end of reasoning to recover the tool call."
        )
        reasoning = model_output[:tool_call_start]
        content = model_output[tool_call_start:]
        return reasoning or None, content

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
        Extract reasoning content from a streaming delta.

        Since <think> is placed in the prompt by the chat template, all
        generated tokens before </think> are reasoning and tokens after
        are content.

        NOTE: When thinking is disabled, no think tokens appear in the
        generated output. The serving layer detects this via
        prompt_is_reasoning_end and routes deltas as content without
        calling this method.
        """
        # Strip <think> from delta if present (old template / edge case
        # where the model generates <think> itself).
        if self.start_token_id in delta_token_ids:
            start_idx = delta_text.find(self.start_token)
            if start_idx >= 0:
                delta_text = delta_text[start_idx + len(self.start_token) :]

        if self.end_token_id in delta_token_ids:
            # End token in this delta: split reasoning from content.
            end_index = delta_text.find(self.end_token)
            if end_index >= 0:
                reasoning = delta_text[:end_index]
                content = delta_text[end_index + len(self.end_token) :]
                if not reasoning and not content:
                    return None
                return DeltaMessage(
                    reasoning=reasoning if reasoning else None,
                    content=content if content else None,
                )
            # end_token_id in IDs but not in text (already stripped)
            return None

        # <tool_call> as implicit end of reasoning: keep it in content.
        if (
            self.tool_call_start_token_id is not None
            and self.tool_call_start_token_id in delta_token_ids
        ):
            tool_idx = delta_text.find(self.TOOL_CALL_TOKEN)
            if tool_idx >= 0:
                logger.warning(
                    "Model generated <tool_call> before </think> — tool call "
                    "was inside the reasoning block. Treating <tool_call> as "
                    "implicit end of reasoning to recover the tool call."
                )
                reasoning = delta_text[:tool_idx]
                content = delta_text[tool_idx:]
                return DeltaMessage(
                    reasoning=reasoning if reasoning else None,
                    content=content if content else None,
                )
            return None

        # No end token in this delta.
        if not delta_text:
            # Nothing left after stripping start token.
            return None
        elif (
            self.end_token_id in previous_token_ids
            or (
                self.tool_call_start_token_id is not None
                and self.tool_call_start_token_id in previous_token_ids
            )
        ):
            # End token already passed: everything is content now.
            return DeltaMessage(content=delta_text)
        else:
            # No end token yet: still in reasoning phase.
            return DeltaMessage(reasoning=delta_text)
