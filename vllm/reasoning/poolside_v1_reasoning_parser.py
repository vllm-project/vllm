# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Laguna reasoning parser.

Poolside Laguna force-opens the thinking block: the model emits chain-of-thought
then a literal ``</think>`` with no opening ``<think>``. That is the same
force-open contract as DeepSeek R1, so this parser subclasses
``DeepSeekR1ReasoningParser`` directly.

Previously this class subclassed ``DeepSeekV3ReasoningParser``, which only
installs the R1 force-open path when ``chat_template_kwargs.thinking`` /
``enable_thinking`` is True and otherwise falls back to
``IdentityReasoningParser`` (passthrough). With the Identity fallback, CoT and
a bare ``</think>`` land in ``content`` and the reasoning channel stays empty.

Separately, ``DeepSeekR1ReasoningParser.is_reasoning_end`` walks the entire
token sequence backwards and returns ``True`` on the first ``</think>`` it
sees. When called on ``prompt_token_ids`` that mistakes any stray
``</think>`` in conversation history, few-shot examples or tool descriptions
for a template-injected "thinking already ended" marker. In the streaming
path (see ``vllm/entrypoints/openai/chat_completion/serving.py``,
``prompt_is_reasoning_end_arr``) that false positive short-circuits the
reasoning parser for the whole response. We therefore scope the backward
search to the current assistant turn: the walk terminates as soon as we hit
the ``<assistant>`` start-of-message token. A ``</think>`` in a prior user
turn or few-shot example is no longer visible.
"""

from collections.abc import Sequence

from transformers import PreTrainedTokenizerBase

from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser


class PoolsideV1ReasoningParser(DeepSeekR1ReasoningParser):
    """Force-open ``</think>`` parser for Poolside Laguna.

    Drop-in for ``deepseek_r1`` force-open semantics that also tolerates
    ``</think>`` tokens appearing in the prompt outside the generation
    prefix (scoped to the current ``<assistant>`` turn).
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
        for tok_id in reversed(input_ids):
            # <think>: reasoning is not yet ended.
            if tok_id == self.start_token_id:
                return False
            # </think>: reasoning has ended.
            if tok_id == self.end_token_id:
                return True
            # <assistant>: reached the start of the current assistant turn
            # without seeing either marker. Anything further back belongs to
            # the prior conversation and should be ignored.
            if tok_id == self._start_of_assistant_message_token_id:
                return False
        return False


__all__ = ["PoolsideV1ReasoningParser"]
