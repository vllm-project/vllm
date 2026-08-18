# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.parser.engine.registered_adapters import (
    InklingParserReasoningAdapter as _InklingParserReasoningAdapter,
)
from vllm.parser.inkling import CONTENT_MODEL_END_SAMPLING, END_MESSAGE

_STRUCTURED_OUTPUT_STOP_TOKENS = (
    CONTENT_MODEL_END_SAMPLING,
    END_MESSAGE,
    "<|begin_of_text|>",
)


class InklingParserReasoningAdapter(_InklingParserReasoningAdapter):
    def structured_output_stop_token_ids(self) -> set[int]:
        return {
            token_id
            for token in _STRUCTURED_OUTPUT_STOP_TOKENS
            if (token_id := self.vocab.get(token)) is not None
        }


__all__ = ["InklingParserReasoningAdapter"]
