# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
from vllm.tokenizers import TokenizerLike


class NemotronV3ReasoningParser(DeepSeekR1ReasoningParser):
    """
    Reasoning parser for Nemotron V3 models.
    """

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        self._enable_thinking = chat_kwargs.get("enable_thinking", True)
        self._force_nonempty_content = chat_kwargs.get("force_nonempty_content", False)

    def extract_reasoning(self, model_output: str) -> tuple[str | None, str | None]:
        reasoning, final_content = super().extract_reasoning(model_output)

        if (
            self._enable_thinking is False or self._force_nonempty_content is True
        ) and final_content is None:
            reasoning, final_content = final_content, reasoning

        return reasoning, final_content
