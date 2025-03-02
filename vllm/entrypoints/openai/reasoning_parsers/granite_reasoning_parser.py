# SPDX-License-Identifier: Apache-2.0

import re
from typing import Optional, Sequence, Tuple, Union

from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaMessage)
from vllm.entrypoints.openai.reasoning_parsers.abs_reasoning_parsers import (
    ReasoningParser, ReasoningParserManager)
from vllm.logger import init_logger

logger = init_logger(__name__)


@ReasoningParserManager.register_module("granite")
class GraniteReasoningParser(ReasoningParser):
    """
    Reasoning parser for IBM Granite.

    IBM granite models currently use "Here is my start process:"
    and "Here is my response:" to separate its thinking / response outputs.
    NOTE: There have been some observed occurrences of quantized instances of
    the current models using "Here's" instead of "Here is", so to be safe, we
    match on both
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__(tokenizer)
        self.think_start_expr = r"(?:Here's|Here is) my thought process:"
        self.think_end_expr = r"(?:Here's|Here is) my response:"

        self.reasoning_regex = re.compile(
            rf"{self.think_start_expr}(.*?){self.think_end_expr}(.*)",
            re.DOTALL)

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
        For text: Here is my thinking:abcHere is my response:xyz:
        - 'abc' goes to reasoning_content
        - 'xyz' goes to content
        """
        raise NotImplementedError("Streaming not implemented")

    def extract_reasoning_content(
            self, model_output: str, request: ChatCompletionRequest
    ) -> Tuple[Optional[str], Optional[str]]:
        re_match = self.reasoning_regex.findall(model_output)
        if not re_match:
            return model_output, None
        reasoning_content, response_content = re_match[0]
        if not response_content:
            return reasoning_content, None
        return reasoning_content, response_content
