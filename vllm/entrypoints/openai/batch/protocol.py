# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Protocol definitions for the Online Batch API."""
from typing import Any

from pydantic import TypeAdapter, field_validator
from pydantic_core.core_schema import ValidationInfo

from vllm.entrypoints.openai.engine.protocol import OpenAIBaseModel


class BatchRequestInput(OpenAIBaseModel):
    """
    The per-line object of the batch input file.
    """

    # A developer-provided per-request id that will be used to match outputs to
    # inputs. Must be unique for each request in a batch.
    custom_id: str

    # The HTTP method to be used for the request. Currently only POST is
    # supported.
    method: str

    # The OpenAI API relative URL to be used for the request.
    url: str

    # The parameters of the request.
    body: Any

    @field_validator('body', mode='plain')
    @classmethod
    def check_type_for_url(cls, value: Any, info: ValidationInfo):
        url = info.data['url']
        if url == "/v1/chat/completions":
            from vllm.entrypoints.openai.chat_completion.protocol import (
                ChatCompletionRequest,
            )
            return ChatCompletionRequest.model_validate(value)
        if url == "/v1/embeddings":
            from vllm.entrypoints.pooling.embed.protocol import EmbeddingRequest
            return TypeAdapter(EmbeddingRequest).validate_python(value)
        if url == "/v1/score":
            from vllm.entrypoints.pooling.scoring.protocol import ScoreRequest
            return ScoreRequest.model_validate(value)
        return value


class BatchResponseData(OpenAIBaseModel):
    # HTTP status code of the response.
    status_code: int = 200

    # An unique identifier for the API request.
    request_id: str

    # The body of the response.
    body: Any | None = None


class BatchRequestOutput(OpenAIBaseModel):
    """
    The per-line object of the batch output and error files
    """

    id: str

    # A developer-provided per-request id that will be used to match outputs to
    # inputs.
    custom_id: str

    response: BatchResponseData | None

    # For requests that failed with a non-HTTP error, this will contain more
    # information on the cause of the failure.
    error: Any | None


class FileObject(OpenAIBaseModel):
    """Represents an uploaded file."""
    id: str
    object: str = "file"
    bytes: int
    created_at: int
    filename: str
    purpose: str  # "batch", "batch_output", "batch_error"


class FileListResponse(OpenAIBaseModel):
    object: str = "list"
    data: list[FileObject]


class FileDeleteResponse(OpenAIBaseModel):
    """Response for a successful file deletion."""
    id: str
    object: str = "file"
    deleted: bool = True


class BatchRequestCounts(OpenAIBaseModel):
    total: int
    completed: int
    failed: int


class BatchError(OpenAIBaseModel):
    code: str
    message: str
    param: str | None = None
    line: int | None = None


class BatchErrors(OpenAIBaseModel):
    object: str = "list"
    data: list[BatchError]


class BatchObject(OpenAIBaseModel):
    """Represents a batch processing job."""
    id: str
    object: str = "batch"
    endpoint: str
    input_file_id: str
    output_file_id: str | None = None
    error_file_id: str | None = None
    status: str
    completion_window: str
    created_at: int
    in_progress_at: int | None = None
    finalizing_at: int | None = None
    completed_at: int | None = None
    failed_at: int | None = None
    cancelling_at: int | None = None
    cancelled_at: int | None = None
    expires_at: int | None = None
    request_counts: BatchRequestCounts
    errors: BatchErrors | None = None
    metadata: dict[str, str] | None = None


class BatchListResponse(OpenAIBaseModel):
    object: str = "list"
    data: list[BatchObject]
    has_more: bool
    first_id: str | None = None
    last_id: str | None = None
