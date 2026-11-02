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
    # The model often closes medium/low thinking with </ifm|think>, so the end
    # token of any effort ends reasoning.
    _END_TOKENS = tuple(end for _, end in _EFFORT_TOKENS.values())
    _BOUNDARY_TOKENS = (*_END_TOKENS, _TOOL_CALLS_START)

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
        self._end_token_ids = {
            self.vocab[end] for end in self._END_TOKENS if end in self.vocab
        }
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

    def _find_boundary(self, text: str) -> tuple[int, int] | None:
        """Return (reasoning end, content start) of the earliest boundary."""
        hits = [
            (i, tag) for tag in self._BOUNDARY_TOKENS if (i := text.find(tag)) != -1
        ]
        if not hits:
            return None
        index, tag = min(hits)
        return index, index if tag == self._TOOL_CALLS_START else index + len(tag)

    def _split_model_output(self, model_output: str) -> tuple[str, str | None]:
        output = self._without_generated_start(model_output)
        boundary = self._find_boundary(output)
        if boundary is not None:
            reasoning_end, content_start = boundary
            return output[:reasoning_end], output[content_start:] or None

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
            if token_id in self._end_token_ids or token_id == self._tool_calls_start_id:
                return True
        return False

    def is_reasoning_end_streaming(
        self,
        input_ids: Sequence[int],
        delta_ids: Iterable[int],
    ) -> bool:
        del input_ids
        return self._stream_finished or any(
            token_id in self._end_token_ids for token_id in delta_ids
        )

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        for index, token_id in enumerate(input_ids):
            if token_id in self._end_token_ids:
                return input_ids[index + 1 :]
            if token_id == self._tool_calls_start_id:
                return input_ids[index:]
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

        boundary = self._find_boundary(self._stream_buffer)
        if boundary is not None:
            reasoning_end, content_start = boundary
            reasoning = self._stream_buffer[:reasoning_end]
            content = self._stream_buffer[content_start:]
        else:
            overlap = max(
                partial_tag_overlap(self._stream_buffer, tag)
                for tag in self._BOUNDARY_TOKENS
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
