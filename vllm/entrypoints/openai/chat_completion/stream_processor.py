# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import NamedTuple

from vllm.entrypoints.generate.base.protocol import DeltaMessage
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.serve.utils.tool_calls_utils import (
    maybe_filter_parallel_tool_calls,
    resolve_finish_reason,
)
from vllm.parser.abstract_parser import Parser


class ChatStreamStep(NamedTuple):
    delta: DeltaMessage | None
    """None when the parser has nothing to emit for this chunk yet."""
    finish_reason: str | None
    """The chat `finish_reason` or None while the choice is still running."""


class ChatStreamProcessor:
    """Turn one chat choice generated chunks into stream deltas.

    Shared by `OpenAIServingChat` streaming and streaming chat derender, so
    both emit the same `reasoning` / `content` / `tool_calls` deltas and
    `finish_reason` for the same chunks. Use one instance per choice and call
    `step` once per chunk, in order and with the chunking the engine produced.
    """

    def __init__(self, request: ChatCompletionRequest, parser: Parser | None) -> None:
        self.request = request
        self.parser = parser
        self.num_output_tokens = 0
        self.tools_streamed = False
        # Logprobs and token IDs would leak hidden reasoning through decoded
        # logprob tokens or raw IDs, so callers drop both on every chunk.
        self.hide_metadata = parser is not None and not request.include_reasoning

    def step(
        self,
        delta_text: str,
        delta_token_ids: list[int],
        finish_reason: str | None,
        prompt_token_ids: list[int] | None,
    ) -> ChatStreamStep:
        """Parse one chunk and resolve its chat `finish_reason`.

        Args:
            delta_text: Text decoded from this chunk's tokens.
            delta_token_ids: This chunk's token IDs.
            finish_reason: The engine finish reason. None if not finished.
            prompt_token_ids: Prompt token IDs if known.

        Returns:
            The delta to stream, filtered for `parallel_tool_calls` and the
            chat `finish_reason`.

        """
        delta: DeltaMessage | None
        if not delta_text and not delta_token_ids and not self.num_output_tokens:
            # Finished without output. Send the final chunk unparsed.
            delta = DeltaMessage()
        elif self.parser is not None:
            delta = self.parser.parse_delta(
                delta_text=delta_text,
                delta_token_ids=delta_token_ids,
                request=self.request,
                prompt_token_ids=prompt_token_ids,
                finished=finish_reason is not None,
            )
            if delta is not None and delta.tool_calls:
                self.tools_streamed = True
        else:
            delta = DeltaMessage(content=delta_text)
        self.num_output_tokens += len(delta_token_ids)

        if delta is not None:
            delta = maybe_filter_parallel_tool_calls(delta, self.request)
        if finish_reason is not None:
            finish_reason = resolve_finish_reason(
                finish_reason, self.request, self.tools_streamed
            )
        return ChatStreamStep(delta, finish_reason)
