# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
GLM reasoning parser (streaming tool-call routing).

GLM-4/5-series models use ``<think>`` blocks and GLM tool-call XML.
OpenAI-compatible clients (e.g. SWE-agent on SWE-bench) expect ``tool_calls`` in
the Chat Completions API. In long-context runs, vLLM can route post-think
``<tool_call>`` tokens to ``reasoning_content`` / ``delta.reasoning`` while
``tool_calls`` stays empty (``FunctionCallingFormatError``).

Root cause: ``<|observation|>`` (token id 154829) may be stripped before the
reasoning parser sees streamed text, corrupting chunk boundaries so
``<tool_call>`` (id 154843) after ``</think>`` is misclassified.

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

# GLM-5 special token IDs (confirmed from tokenizer)
# 154841 = <think>
# 154842 = </think>
# 154843 = <tool_call>
# 154829 = <|observation|> (stripped on some paths; corrupts streaming boundaries)
_GLM_TOOL_CALL_TOKEN_ID = 154843


class GLMReasoningParser(DeepSeekV3ReasoningWithThinkingParser):
    """Reasoning parser for GLM-4/5-series models (streaming tool-call routing)."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

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
            _GLM_TOOL_CALL_TOKEN_ID in delta_token_ids
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
