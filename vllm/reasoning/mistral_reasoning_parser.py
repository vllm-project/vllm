# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from mistral_common.tokens.tokenizers.base import SpecialTokens

from vllm.logger import init_logger
from vllm.reasoning import ReasoningParser, ReasoningParserManager
from vllm.reasoning.deepseek_r1_reasoning_parser import (
    DeepSeekR1ReasoningParser)
from vllm.transformers_utils.tokenizers.mistral import MistralTokenizer

logger = init_logger(__name__)


@ReasoningParserManager.register_module("mistral")
class MistralReasoningParser(DeepSeekR1ReasoningParser):
    """
    Reasoning parser for Mistral model.
    """

    def __init__(self, tokenizer: MistralTokenizer):
        if not isinstance(tokenizer, MistralTokenizer):
            raise ValueError(
                "The tokenizer must be an instance of MistralTokenizer.")

        ReasoningParser.__init__(self, tokenizer)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ReasoningParser "
                "constructor during construction.")

        self.start_token = SpecialTokens.begin_think
        self.end_token = SpecialTokens.end_think

        self.start_token_id = tokenizer.tokenizer.get_control_token(
            self.start_token)
        self.end_token_id = tokenizer.tokenizer.get_control_token(
            self.end_token)

        if self.start_token_id is None or self.end_token_id is None:
            raise RuntimeError(
                "DeepSeek R1 reasoning parser could not locate think start/end "
                "tokens in the tokenizer!")
