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
        reasoning_content, final_content = super().extract_reasoning(
            model_output, request
        )
        chat_template_kwargs = getattr(request, "chat_template_kwargs", None)

        if (
            chat_template_kwargs
            and chat_template_kwargs.get("enable_thinking") is False
            and final_content is None
        ):
            reasoning_content, final_content = final_content, reasoning_content

        return reasoning_content, final_content
