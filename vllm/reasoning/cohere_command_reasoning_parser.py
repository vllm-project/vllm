# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

from vllm.reasoning import ReasoningParser
from vllm.v1.token_buffer import rfind_tokens

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike


class BaseCohereCommandReasoningParser(ReasoningParser):
    """Expose is_reasoning_end for the structured-output engine.
    Everything else runs in
    :class:`vllm.parser.cohere_command.CohereCommandParser`.
    """

    melody_preset: ClassVar[str]

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.start_token_id = tokenizer.convert_tokens_to_ids("<|START_THINKING|>")
        self.end_token_id = tokenizer.convert_tokens_to_ids("<|END_THINKING|>")
        self.chatbot_token_id = tokenizer.convert_tokens_to_ids("<|CHATBOT_TOKEN|>")

    @property
    def reasoning_start_str(self) -> str | None:
        return "<|START_THINKING|>"

    @property
    def reasoning_end_str(self) -> str | None:
        return "<|END_THINKING|>"

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        # Reasoning ended if an end token follows the newest start token,
        # unless a chatbot token comes after that start.
        i = rfind_tokens(input_ids, (self.start_token_id, self.chatbot_token_id))
        if i >= 0 and input_ids[i] == self.chatbot_token_id:
            return False
        return rfind_tokens(input_ids, (self.end_token_id,), start=i + 1) >= 0


class CohereCommand3ReasoningParser(BaseCohereCommandReasoningParser):
    melody_preset = "cmd3"


class CohereCommand4ReasoningParser(BaseCohereCommandReasoningParser):
    melody_preset = "cmd4"
