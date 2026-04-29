# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from transformers import PreTrainedTokenizerBase

from vllm.logger import init_logger
from vllm.reasoning import ReasoningParser
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
from vllm.sampling_params import StructuredOutputsParams

from .identity_reasoning_parser import IdentityReasoningParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.engine.protocol import DeltaMessage
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

logger = init_logger(__name__)

# When thinking is enabled the chat template opens a <think> block before
# generation begins. Any structured-output constraint applied from token 0
# would block the closing </think>, so we wrap the constraint in a structural
# tag that only engages after the reasoning section is closed. The trigger is
# the bare closing tag (kicks in as soon as </think> appears) and the
# structure's `begin` forces the canonical \n\n separator before the schema-
# conforming output starts. See vllm-project/vllm#41132 / #33215.
_THINKING_END_TRIGGER = "</think>"
_THINKING_END_BEGIN = "</think>\n\n"


class DeepSeekV3ReasoningParser(ReasoningParser):
    """
    V3 parser that delegates to either DeepSeekR1ReasoningParser or
    IdentityReasoningParser based on `thinking` and `separate_reasoning`.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        thinking = bool(chat_kwargs.get("thinking", False))
        enable_thinking = bool(chat_kwargs.get("enable_thinking", False))
        thinking = thinking or enable_thinking

        self._parser: ReasoningParser
        if thinking:
            self._parser = DeepSeekR1ReasoningParser(tokenizer, *args, **kwargs)
        else:
            self._parser = IdentityReasoningParser(tokenizer, *args, **kwargs)

    @property
    def reasoning_start_str(self) -> str | None:
        return self._parser.reasoning_start_str

    @property
    def reasoning_end_str(self) -> str | None:
        return self._parser.reasoning_end_str

    def adjust_request(
        self, request: "ChatCompletionRequest | ResponsesRequest"
    ) -> "ChatCompletionRequest | ResponsesRequest":
        if not isinstance(self._parser, DeepSeekR1ReasoningParser):
            return request

        response_format = getattr(request, "response_format", None)
        if response_format is None:
            return request

        if response_format.type == "json_schema":
            json_schema = response_format.json_schema
            assert json_schema is not None
            schema = json_schema.json_schema
        elif response_format.type == "json_object":
            schema = {"type": "object"}
        else:
            return request

        request.structured_outputs = StructuredOutputsParams(
            structural_tag=json.dumps(
                {
                    "triggers": [_THINKING_END_TRIGGER],
                    "structures": [
                        {
                            "begin": _THINKING_END_BEGIN,
                            "schema": schema,
                            "end": "",
                        }
                    ],
                }
            )
        )
        request.response_format = None
        return request

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        return self._parser.is_reasoning_end(input_ids)

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        return self._parser.is_reasoning_end_streaming(input_ids, delta_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        return self._parser.extract_content_ids(input_ids)

    def extract_reasoning(
        self, model_output: str, request: "ChatCompletionRequest | ResponsesRequest"
    ) -> tuple[str | None, str | None]:
        return self._parser.extract_reasoning(model_output, request)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> "DeltaMessage | None":
        return self._parser.extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )


class DeepSeekV3ReasoningWithThinkingParser(DeepSeekV3ReasoningParser):
    """
    DeepSeekV3ReasoningParser that defaults to thinking mode.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        thinking = chat_kwargs.get("thinking", None)
        enable_thinking = chat_kwargs.get("enable_thinking", None)
        if thinking is None and enable_thinking is None:
            chat_kwargs["thinking"] = True
            chat_kwargs["enable_thinking"] = True
            kwargs["chat_template_kwargs"] = chat_kwargs
        super().__init__(tokenizer, *args, **kwargs)
