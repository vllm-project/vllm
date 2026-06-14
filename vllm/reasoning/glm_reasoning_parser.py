# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
GLM reasoning parser (streaming tool-call routing).

GLM-4/5-series models use ``<think>`` blocks and GLM tool-call XML.
OpenAI-compatible clients (e.g. SWE-agent on SWE-bench) expect ``tool_calls`` in
the Chat Completions API. In long-context runs, vLLM can route post-think
``<tool_call>`` tokens to ``reasoning_content`` / ``delta.reasoning`` while
``tool_calls`` stays empty (``FunctionCallingFormatError``).

Root cause: ``<|observation|>`` may be stripped before the reasoning parser
sees streamed text, corrupting chunk boundaries so the ``<tool_call>`` token
(after ``</think>``) is misclassified. The ``<tool_call>`` id is read
from ``tokenizer.get_vocab()`` like
:class:`~vllm.reasoning.basic_parsers.BaseThinkingReasoningParser`.

``GLMReasoningParser.extract_reasoning_streaming`` forces those post-think
``<tool_call>`` deltas to ``content`` so the downstream tool parser can run.
Non-streaming extraction is unchanged from
:class:`DeepSeekV3ReasoningWithThinkingParser`.
"""

from collections.abc import Sequence

from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.chat_completion.protocol import DeltaMessage
from vllm.reasoning.deepseek_v3_reasoning_parser import (
    DeepSeekV3ReasoningWithThinkingParser,
)

_TOOL_CALL_TAG = "<tool_call>"


class GLMReasoningParser(DeepSeekV3ReasoningWithThinkingParser):
    """Reasoning parser for GLM-4/5-series models (streaming tool-call routing)."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        tool_call_token_id = self.vocab.get(_TOOL_CALL_TAG)
        if tool_call_token_id is None:
            raise RuntimeError(
                f"{self.__class__.__name__} could not locate {_TOOL_CALL_TAG!r} in "
                "the tokenizer vocabulary."
            )
        self._tool_call_token_id: int = tool_call_token_id

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """Route post-think ``<tool_call>`` deltas to ``content``."""
        end_token_id = getattr(self, "end_token_id", None)
        if end_token_id is None:
            delegated = getattr(self, "_parser", None)
            end_token_id = getattr(delegated, "end_token_id", None)

        if (
            self._tool_call_token_id in delta_token_ids
            and end_token_id is not None
            and end_token_id in previous_token_ids
        ):
            return DeltaMessage(content=delta_text)

        return super().extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )
