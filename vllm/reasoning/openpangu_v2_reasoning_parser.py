# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from transformers import PreTrainedTokenizerBase

from vllm.logger import init_logger
from vllm.reasoning import ReasoningParser
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
from vllm.reasoning.deepseek_v3_reasoning_parser import DeepSeekV3ReasoningParser

from .identity_reasoning_parser import IdentityReasoningParser

logger = init_logger(__name__)


class OpenPanguV2ReasoningParser(DeepSeekV3ReasoningParser):
    """
    OpenPanguV2 reasoning parser that delegates to either DeepSeekR1ReasoningParser
    or IdentityReasoningParser based on `think`.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        ReasoningParser.__init__(self, tokenizer, *args, **kwargs)

        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        think = bool(chat_kwargs.get("think", True))

        self._parser: ReasoningParser
        if think:
            self._parser = DeepSeekR1ReasoningParser(tokenizer, *args, **kwargs)
        else:
            self._parser = IdentityReasoningParser(tokenizer, *args, **kwargs)
