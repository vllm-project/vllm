# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Sequence

from vllm.entrypoints.generate.base.protocol import DeltaMessage
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.utils import partial_tag_overlap


class K2HorizonReasoningParser(DeepSeekR1ReasoningParser):
    """Reasoning parser for K2 Horizon."""

    _TOOL_CALLS_START = "<ifm|tool_calls>"
    _EFFORT_TOKENS: dict[str, tuple[str, str]] = {
        "high": ("<ifm|think>", "</ifm|think>"),
        "medium": ("<ifm|think_fast>", "</ifm|think_fast>"),
        "low": ("<ifm|think_faster>", "</ifm|think_faster>"),
    }

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs) -> None:
        chat_template_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        effort = chat_template_kwargs.get("reasoning_effort", "high")
        if not isinstance(effort, str) or effort not in self._EFFORT_TOKENS:
            supported = ", ".join(self._EFFORT_TOKENS)
            raise ValueError(
                f"Unsupported reasoning_effort {effort!r}. "
                f"Supported values: {supported}."
            )

        self._start_token, self._end_token = self._EFFORT_TOKENS[effort]
        super().__init__(tokenizer, *args, **kwargs)
        self._tool_calls_start_id = self.vocab.get(self._TOOL_CALLS_START)
        self._stream_buffer = ""
        self._stream_at_start = True
        self._stream_finished = False

    @property
    def start_token(self) -> str:
        return self._start_token

    @property
    def end_token(self) -> str:
        return self._end_token

    def _without_generated_start(self, model_output: str) -> str:
        return model_output.removeprefix(self.start_token)

    def _split_model_output(self, model_output: str) -> tuple[str, str | None]:
        output = self._without_generated_start(model_output)
        if self.end_token in output:
            reasoning, _, content = output.partition(self.end_token)
            return reasoning, content or None

        tool_start = output.find(self._TOOL_CALLS_START)
        if tool_start != -1:
            return output[:tool_start], output[tool_start:]

        return "", output or None

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str, str | None]:
        del request
        return self._split_model_output(model_output)

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        for token_id in reversed(input_ids):
            if token_id == self.start_token_id:
                return False
            if token_id == self.end_token_id or token_id == self._tool_calls_start_id:
                return True
        return False

    def is_reasoning_end_streaming(
        self,
        input_ids: Sequence[int],
        delta_ids: Iterable[int],
    ) -> bool:
        del input_ids
        return self._stream_finished or super().is_reasoning_end_streaming(
            (), delta_ids
        )

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if self.end_token_id in input_ids:
            return input_ids[input_ids.index(self.end_token_id) + 1 :]
        if (
            self._tool_calls_start_id is not None
            and self._tool_calls_start_id in input_ids
        ):
            return input_ids[input_ids.index(self._tool_calls_start_id) :]
        return input_ids if self._stream_finished else []

    def _strip_stream_start(self) -> bool:
        if not self._stream_at_start:
            return True
        if self.start_token.startswith(self._stream_buffer):
            if self._stream_buffer != self.start_token:
                return False
            self._stream_buffer = ""
        elif self._stream_buffer.startswith(self.start_token):
            self._stream_buffer = self._stream_buffer[len(self.start_token) :]
        self._stream_at_start = False
        return True

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        del (
            previous_text,
            current_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )

        if self._stream_finished:
            return DeltaMessage(content=delta_text) if delta_text else None

        self._stream_buffer += delta_text
        if not self._strip_stream_start():
            return None

        explicit_end = self._stream_buffer.find(self.end_token)
        tool_start = self._stream_buffer.find(self._TOOL_CALLS_START)
        if explicit_end != -1:
            reasoning = self._stream_buffer[:explicit_end]
            content = self._stream_buffer[explicit_end + len(self.end_token) :]
        elif tool_start != -1:
            reasoning = self._stream_buffer[:tool_start]
            content = self._stream_buffer[tool_start:]
        else:
            overlap = max(
                partial_tag_overlap(self._stream_buffer, self.end_token),
                partial_tag_overlap(self._stream_buffer, self._TOOL_CALLS_START),
            )
            sendable_end = len(self._stream_buffer) - overlap
            reasoning = self._stream_buffer[:sendable_end]
            self._stream_buffer = self._stream_buffer[sendable_end:]
            return DeltaMessage(reasoning=reasoning) if reasoning else None

        self._stream_buffer = ""
        self._stream_finished = True
        return DeltaMessage(
            reasoning=reasoning,
            content=content or None,
        )
