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
        # Translate OpenAI-standard reasoning_effort into the Nemotron chat
        # template's native kwargs so the template actually adjusts generation.
        # The Nemotron jinja template reads `low_effort` and `enable_thinking`;
        # it does not read `reasoning_effort` directly.
        reasoning_effort = getattr(request, "reasoning_effort", None)
        if reasoning_effort is not None:
            chat_kwargs = getattr(request, "chat_template_kwargs", None)
            if chat_kwargs is None and hasattr(request, "chat_template_kwargs"):
                chat_kwargs = {}
                request.chat_template_kwargs = chat_kwargs
            if chat_kwargs is not None:
                if reasoning_effort == "low" and "low_effort" not in chat_kwargs:
                    chat_kwargs["low_effort"] = True
                elif (
                    reasoning_effort == "none"
                    and "enable_thinking" not in chat_kwargs
                ):
                    chat_kwargs["enable_thinking"] = False
                # "medium" and "high" map to default full-thinking behavior;
                # the template has no finer-grained knob for those levels.
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
            and final_content is None
        ):
            reasoning, final_content = final_content, reasoning

        return reasoning, final_content
