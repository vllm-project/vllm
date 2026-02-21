# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning.deepseek_v3_reasoning_parser import (
    DeepSeekV3ReasoningWithThinkingParser,
)


class KimiK25ReasoningParser(DeepSeekV3ReasoningWithThinkingParser):
    def __init__(self, tokenizer, *args, **kwargs):
        chat_kwargs = dict(kwargs.get("chat_template_kwargs", {}) or {})
        chat_kwargs["thinking"] = True
        kwargs["chat_template_kwargs"] = chat_kwargs
        super().__init__(tokenizer, *args, **kwargs)
        self._reasoning_end_token_ids = {
            self.vocab.get("<|tool_calls_section_begin|>"),
            self.vocab.get("<|tool_call_begin|>"),
        }
        self._reasoning_end_token_ids = {
            tid for tid in self._reasoning_end_token_ids if tid is not None
        }
        self._tool_call_started = False

    def extract_reasoning_streaming(
        self,
        previous_text,
        current_text,
        delta_text,
        previous_token_ids,
        current_token_ids,
        delta_token_ids,
    ):
        if self._tool_call_started:
            return DeltaMessage(content=delta_text)
        # Only intercept tool tokens if we haven't seen </think> yet
        if self._parser.end_token_id not in previous_token_ids and any(
            tid in self._reasoning_end_token_ids for tid in delta_token_ids
        ):
            self._tool_call_started = True
            return DeltaMessage(content=delta_text)
        ret = self._parser.extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )
        return ret
