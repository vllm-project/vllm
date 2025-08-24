# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from transformers import PreTrainedTokenizerBase

from vllm.logger import init_logger
from vllm.reasoning import ReasoningParser, ReasoningParserManager
from vllm.reasoning.deepseek_r1_reasoning_parser import (
    DeepSeekR1ReasoningParser)

logger = init_logger(__name__)


@ReasoningParserManager.register_module("seed_oss")
class SeedOssReasoningParser(DeepSeekR1ReasoningParser):
    """
    Reasoning parser for Seed Oss models.

    This model uses <seed:think>...</seed:think> tokens to denote reasoning
    text. This parser extracts the reasoning content from the model output.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        ReasoningParser.__init__(self, tokenizer)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ReasoningParser "
                "constructor during construction.")

        self.start_token = "<seed:think>"
        self.end_token = "</seed:think>"

        self.start_token_id = self.vocab.get(self.start_token)
        self.end_token_id = self.vocab.get(self.end_token)

        if self.start_token_id is None or self.end_token_id is None:
            raise RuntimeError(
                "Seed Oss reasoning parser could not locate think start/end "
                "tokens in the tokenizer!")
