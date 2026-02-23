# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import cached_property

from vllm.logger import init_logger
from vllm.reasoning import ReasoningParser
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
from vllm.transformers_utils.tokenizers.mistral import MistralTokenizer

logger = init_logger(__name__)


class MistralReasoningParser(DeepSeekR1ReasoningParser):
    """
    Reasoning parser for Mistral models.

    The Mistral models uses [THINK]...[/THINK] tokens to denote reasoning
    text. This parser extracts the reasoning content from the model output.
    """

    def __init__(self, tokenizer: MistralTokenizer, *args, **kwargs):
        if not isinstance(tokenizer, MistralTokenizer):
            raise ValueError("The tokenizer must be an instance of MistralTokenizer.")

        ReasoningParser.__init__(self, tokenizer, *args, **kwargs)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ReasoningParser "
                "constructor during construction."
            )

        self.start_token_id = tokenizer.tokenizer.get_control_token(self.start_token)
        self.end_token_id = tokenizer.tokenizer.get_control_token(self.end_token)

        if self.start_token_id is None or self.end_token_id is None:
            raise RuntimeError(
                "Mistral reasoning parser could not locate think start/end "
                "tokens in the tokenizer!"
            )

    @cached_property
    def start_token(self) -> str:
        """The token that starts reasoning content."""
        from mistral_common.tokens.tokenizers.base import SpecialTokens

        return SpecialTokens.begin_think

    @cached_property
    def end_token(self) -> str:
        """The token that ends reasoning content."""
        from mistral_common.tokens.tokenizers.base import SpecialTokens

        return SpecialTokens.end_think
