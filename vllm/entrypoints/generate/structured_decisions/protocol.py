# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request and response models of the structured decisions API (/v1/systemone)."""

import time
from typing import Any, Literal

from pydantic import Field

from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import ChatTemplateContentFormatOption
from vllm.entrypoints.serve.engine.protocol import OpenAIBaseModel
from vllm.renderers import ChatParams, TokenizeParams, merge_kwargs
from vllm.utils import random_uuid


class QuestionSpec(OpenAIBaseModel):
    type: str
    instructions: Any = ""
    criteria: Any = None


class StructuredDecisionRequest(OpenAIBaseModel):
    model: str | None = None
    state: Any = Field(..., description="What the questions are about: text or JSON.")
    questions: dict[str, QuestionSpec] = Field(
        ..., description="Question id to question, asked in this order."
    )
    instructions: str | None = Field(
        default=None, description="Context placed ahead of the questions."
    )
    decision_template: str | None = Field(
        default=None,
        description="A Jinja decision template for this request. Needs the "
        "server to run with --trust-request-chat-template.",
    )
    chat_template_kwargs: dict[str, Any] | None = None
    seed: int | None = Field(
        default=None,
        description="Changes which label each option gets. Reading a question "
        "under several seeds and averaging the answers cancels the model's "
        "preference for particular labels.",
    )
    priority: int = Field(default=0, ge=-(2**63), le=2**63 - 1)
    request_id: str = Field(default_factory=random_uuid)


class DecisionUsage(OpenAIBaseModel):
    input_tokens: int
    output_tokens: int


class QuestionDiagnostics(OpenAIBaseModel):
    label_mass: float = Field(
        description="Probability the model put on the labels, over the full "
        "vocabulary. Low values mean it wanted to say something else."
    )
    argmax_is_label: bool


class StructuredDecisionResponse(OpenAIBaseModel):
    id: str
    object: Literal["structured_decision"] = "structured_decision"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    answers: dict[str, dict[str, Any]]
    usage: DecisionUsage
    diagnostics: dict[str, QuestionDiagnostics]


class ReadPromptRequest(OpenAIBaseModel):
    """Chat options for one read's prompt: system and state, then an assistant
    reply left open before the question's label."""

    chat_template_kwargs: dict[str, Any] | None = None

    def build_chat_params(
        self,
        default_template: str | None,
        default_template_content_format: ChatTemplateContentFormatOption,
    ) -> ChatParams:
        return ChatParams(
            chat_template=default_template,
            chat_template_content_format=default_template_content_format,
            chat_template_kwargs=merge_kwargs(
                self.chat_template_kwargs,
                dict(add_generation_prompt=False, continue_final_message=True),
            ),
        )

    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams:
        return TokenizeParams(
            max_total_tokens=model_config.max_model_len,
            max_output_tokens=1,
            add_special_tokens=False,
        )
