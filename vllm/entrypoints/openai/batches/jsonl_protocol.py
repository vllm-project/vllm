# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, TypeAlias

from pydantic import Field, TypeAdapter, field_validator
from pydantic_core.core_schema import ValidationInfo

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.pooling.embed.protocol import EmbeddingRequest
from vllm.entrypoints.pooling.scoring.protocol import RerankRequest, ScoreRequest
from vllm.entrypoints.serve.engine.protocol import ErrorResponse, OpenAIBaseModel
from vllm.utils import random_uuid

BatchRequestBody: TypeAlias = (
    ChatCompletionRequest | EmbeddingRequest | ScoreRequest | RerankRequest
)

_BODY_ADAPTERS: dict[str, TypeAdapter] = {
    "/v1/chat/completions": TypeAdapter(ChatCompletionRequest),
    "/v1/embeddings": TypeAdapter(EmbeddingRequest),
    "/v1/score": TypeAdapter(ScoreRequest),
    "/v1/rerank": TypeAdapter(RerankRequest),
}


class BatchRequestInput(OpenAIBaseModel):
    """A line of a batch input file."""

    custom_id: str = Field(
        description="A developer-provided id used to match outputs to inputs. "
        "Must be unique for each request in a batch."
    )
    method: str = Field(description="The HTTP method. Only POST is supported.")
    url: str = Field(description="The API endpoint, e.g. /v1/chat/completions.")
    body: BatchRequestBody = Field(description="The parameters of the request.")

    @field_validator("body", mode="plain")
    @classmethod
    def validate_body_for_url(cls, value: Any, info: ValidationInfo):
        url = info.data["url"]
        if url not in _BODY_ADAPTERS:
            raise ValueError(f"Unsupported url: {url}")
        return _BODY_ADAPTERS[url].validate_python(value)


class BatchResponseData(OpenAIBaseModel):
    status_code: int = 200
    request_id: str = Field(default_factory=lambda: f"vllm-batch-{random_uuid()}")
    body: Any | None = None


class BatchRequestOutput(OpenAIBaseModel):
    """A line of a batch output or error file."""

    id: str = Field(default_factory=lambda: f"vllm-{random_uuid()}")
    custom_id: str = Field(
        description="The custom_id of the request this output belongs to."
    )
    response: BatchResponseData | None
    error: ErrorResponse | None = None

    @classmethod
    def success(cls, custom_id: str, body: Any) -> "BatchRequestOutput":
        return cls(custom_id=custom_id, response=BatchResponseData(body=body))

    @classmethod
    def failure(cls, custom_id: str, error: ErrorResponse) -> "BatchRequestOutput":
        return cls(
            custom_id=custom_id,
            response=BatchResponseData(status_code=error.error.code),
            error=error,
        )
