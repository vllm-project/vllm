# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
)
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser


class NemotronV3ReasoningParser(DeepSeekR1ReasoningParser):
    """
    Reasoning parser for Nemotron V3 models.
    """

    def extract_reasoning(
        self, model_output: str, request: ChatCompletionRequest | ResponsesRequest
    ) -> tuple[str | None, str | None]:
        reasoning, final_content = super().extract_reasoning(model_output, request)
        chat_template_kwargs = getattr(request, "chat_template_kwargs", None)

        enable_thinking = (
            chat_template_kwargs.get("enable_thinking", None)
            if chat_template_kwargs
            else None
        )
        force_nonempty = (
            chat_template_kwargs.get("force_nonempty_content", False)
            if chat_template_kwargs
            else False
        )

        if final_content is None:
            if enable_thinking is False or force_nonempty:
                # Explicit non-thinking mode or force-nonempty: put output in
                # content.
                reasoning, final_content = final_content, reasoning
            elif enable_thinking is not True and self.start_token not in model_output:
                # No explicit enable_thinking=True and no start token in the
                # output: the model did not enter a thinking section, so the
                # entire output is regular content rather than reasoning.
                reasoning, final_content = final_content, reasoning

        return reasoning, final_content
