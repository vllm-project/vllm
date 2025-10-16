# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from transformers import PreTrainedTokenizerBase

from vllm.logger import init_logger
from vllm.reasoning import (
    DeepSeekR1ReasoningParser,
    DeepSeekV3ReasoningParser,
    ReasoningParserManager,
)

from .identity_reasoning_parser import IdentityReasoningParser

logger = init_logger(__name__)


@ReasoningParserManager.register_module("exaone4")
class Exaone4ReasoningParser(DeepSeekV3ReasoningParser):
    """
    Reasoning parser for EXAONE 4.0 model.

    The EXAONE 4.0 model uses <think>...</think> tokens to denote reasoning
    text. This parser extracts the reasoning content from the model output.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        print("Exaone4ReasoningParser init")

        chat_kwargs = kwargs.pop("chat_template_kwargs", {}) or {}
        enable_thinking = bool(chat_kwargs.pop("enable_thinking", False))

        if enable_thinking:
            self._parser = DeepSeekR1ReasoningParser(tokenizer, *args, **kwargs)
        else:
            self._parser = IdentityReasoningParser(tokenizer, *args, **kwargs)
