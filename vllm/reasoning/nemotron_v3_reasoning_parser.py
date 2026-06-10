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

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        # Translate the OpenAI-standard reasoning_effort field into the
        # Nemotron chat template's native kwargs (low_effort, enable_thinking)
        # so the template can actually adjust generation behavior.
        reasoning_effort = getattr(request, "reasoning_effort", None)
        if reasoning_effort is not None:
            chat_kwargs = request.chat_template_kwargs
            if chat_kwargs is None:
                chat_kwargs = {}
                request.chat_template_kwargs = chat_kwargs
            if reasoning_effort == "low" and "low_effort" not in chat_kwargs:
                chat_kwargs["low_effort"] = True
            if reasoning_effort == "none" and "enable_thinking" not in chat_kwargs:
                chat_kwargs["enable_thinking"] = False
        return request

    def extract_reasoning(
        self, model_output: str, request: ChatCompletionRequest | ResponsesRequest
    ) -> tuple[str | None, str | None]:
        reasoning, final_content = super().extract_reasoning(model_output, request)
        chat_template_kwargs = getattr(request, "chat_template_kwargs", None)

        if (
            chat_template_kwargs
            and (
                chat_template_kwargs.get("enable_thinking") is False
                or chat_template_kwargs.get("force_nonempty_content") is True
            )
            and (final_content is None or not final_content.strip())
        ):
            reasoning, final_content = final_content, reasoning

        return reasoning, final_content
