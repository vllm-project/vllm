# SPDX-License-Identifier: Apache-2.0

import re
from collections.abc import Sequence
from typing import Optional, Union

from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaMessage)
from vllm.entrypoints.openai.reasoning_parsers.abs_reasoning_parsers import (
    ReasoningParser, ReasoningParserManager)
from vllm.logger import init_logger

logger = init_logger(__name__)


@ReasoningParserManager.register_module("deepseek_r1")
class DeepSeekR1ReasoningParser(ReasoningParser):
    """
    Reasoning parser for DeepSeek R1 model.

    The DeepSeek R1 model uses a chat_template that includes a starting
    <think> token as input, resulting in the output not having <think>
    token to denote thinking content.

    This parser extracts the reasoning content from the model output
    adjusted to this behaviour.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__(tokenizer)
        self.think_start_token = "<think>"
        self.think_end_token = "</think>"

        self.reasoning_regex = re.compile(
            rf'\n*{self.think_end_token}\n*', re.DOTALL)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ReasoningParser "
                "constructor during construction.")

        self.think_end_token_id = self.vocab.get(self.think_end_token)
        if self.think_end_token_id is None:
            raise RuntimeError(
                "DeepSeek R1 reasoning parser could not locate think start/end "
                "tokens in the tokenizer!")

    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> Union[DeltaMessage, None]:
        """
        Extract reasoning content from a delta message.
        Handles streaming output where previous + delta = current.
        Uses token IDs for faster processing.
        For text abc</think>xyz:
        - 'abc' goes to reasoning_content
        - 'xyz' goes to content
        """
        # Skip single special tokens
        if len(delta_token_ids) == 1 and (delta_token_ids[0] in [
                self.think_end_token_id
        ]):
            return None

        # <think> is effectively present in previous with new chat_template
        # Incompatible with models not include <think> token in chat_template
        if self.think_end_token_id in delta_token_ids:
            # </think> in delta,
            # extract reasoning content
            end_index = delta_text.find(self.think_end_token)
            reasoning_content = delta_text[:end_index]
            content = delta_text[end_index + len(self.think_end_token):]
            return DeltaMessage(reasoning_content=reasoning_content,
                                content=content if content else None)
        elif self.think_end_token_id in previous_token_ids:
            # </think> in previous,
            # content continues
            return DeltaMessage(reasoning_content=None, content=delta_text)
        else:
            # no </think> in previous or delta,
            # reasoning content continues
            return DeltaMessage(reasoning_content=delta_text, content=None)

    def extract_reasoning_content(
            self, model_output: str, request: ChatCompletionRequest
    ) -> tuple[Optional[str], Optional[str]]:

        # DeepSeek R1 doesn't generate <think> now.
        # Thus we assume the reasoning content is always at the start.
        # Ref https://huggingface.co/deepseek-ai/DeepSeek-R1/commit/8a58a132790c9935686eb97f042afa8013451c9f
        if search := self.reasoning_regex.search(model_output):
            reasoning_content = model_output[:search.start()]
            content = model_output[search.end():]
            return reasoning_content, content
        return model_output, None