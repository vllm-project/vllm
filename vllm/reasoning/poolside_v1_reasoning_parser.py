# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Laguna reasoning parser.

``DeepSeekV3ReasoningParser.is_reasoning_end`` walks the entire
token sequence backwards and returns ``True`` on the first ``</think>`` it
sees. When called on ``prompt_token_ids`` that mistakes any stray
``</think>`` in conversation history, few-shot examples or tool descriptions
for a template-injected "thinking already ended" marker. In the streaming
path (see ``vllm/entrypoints/openai/chat_completion/serving.py``,
``prompt_is_reasoning_end_arr``) that false positive short-circuits the
reasoning parser for the whole response, so any ``<think>...</think>`` the
model emits itself ends up in the content field instead of the reasoning
field.

As we have more flexible templates, we instead scope
the backward search to the current assistant turn: the
walk terminates as soon as we hit the ``<assistant>`` start-of-message
token. A ``</think>`` in a prior user turn or few-shot example is no longer
visible.
"""

from collections.abc import Sequence

from transformers import PreTrainedTokenizerBase

from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
from vllm.reasoning.deepseek_v3_reasoning_parser import DeepSeekV3ReasoningParser
from vllm.reasoning.identity_reasoning_parser import IdentityReasoningParser


class PoolsideV1ReasoningParser(DeepSeekV3ReasoningParser):
    """Drop-in replacement for ``deepseek_v3`` that tolerates ``</think>``
    tokens appearing anywhere in the prompt other than the generation prefix.
    """

    _start_of_assistant_message = "<assistant>"

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        if self._start_of_assistant_message not in self.vocab:
            raise ValueError(
                f"Tokenizer must contain {self._start_of_assistant_message!r} token"
            )
        self._start_of_assistant_message_token_id = self.vocab[
            self._start_of_assistant_message
        ]

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        # IdentityReasoningParser always returns True: no reasoning to parse.
        if isinstance(self._parser, IdentityReasoningParser):
            return True

        assert isinstance(self._parser, DeepSeekR1ReasoningParser)
        for tok_id in reversed(input_ids):
            # <think>: reasoning is not yet ended.
            if tok_id == self._parser.start_token_id:
                return False
            # </think>: reasoning has ended.
            if tok_id == self._parser.end_token_id:
                return True
            # <assistant>: reached the start of the current assistant turn
            # without seeing either marker. Anything further back belongs to
            # the prior conversation and should be ignored.
            if tok_id == self._start_of_assistant_message_token_id:
                return False
        return False


__all__ = ["PoolsideV1ReasoningParser"]
