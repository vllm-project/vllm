# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request and response models of the structured decisions API (/v1/systemone)."""

import time
from typing import Any, Literal

from pydantic import Field, model_validator

from vllm.entrypoints.generate.base.protocol import validate_cache_salt
from vllm.entrypoints.serve.engine.protocol import OpenAIBaseModel
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
    chat_template_kwargs: dict[str, Any] | None = None
    cache_salt: str | None = Field(
        default=None,
        description=(
            "If specified, the prefix cache will be salted with the provided "
            "string to prevent an attacker to guess prompts in multi-user "
            "environments. The salt should be random, protected from "
            "access by 3rd parties, and long enough to be "
            "unpredictable (e.g., 43 characters base64-encoded, corresponding "
            "to 256 bit)."
        ),
    )
    priority: int = Field(default=0, ge=-(2**63), le=2**63 - 1)
    request_id: str = Field(default_factory=random_uuid)

    @model_validator(mode="before")
    @classmethod
    def check_cache_salt_support(cls, data: Any) -> Any:
        if isinstance(data, dict):
            validate_cache_salt(data.get("cache_salt"))
        return data


class DecisionUsage(OpenAIBaseModel):
    input_tokens: int
    output_tokens: int


class QuestionDiagnostics(OpenAIBaseModel):
    label_mass: float = Field(
        description="Probability the model put on the labels, over the full "
        "vocabulary. A low value means most of the probability went to tokens "
        "that are not labels."
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
