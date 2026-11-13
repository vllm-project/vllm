# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Literal, TypeAlias

from pydantic import Field

from vllm.entrypoints.serve.engine.protocol import OpenAIBaseModel

BatchEndpoint: TypeAlias = Literal[
    "/v1/chat/completions", "/v1/embeddings", "/v1/score", "/v1/rerank"
]
BatchStatus: TypeAlias = Literal[
    "validating",
    "failed",
    "in_progress",
    "finalizing",
    "completed",
    "expired",
    "cancelling",
    "cancelled",
]
ACTIVE_BATCH_STATUSES: tuple[BatchStatus, ...] = (
    "validating",
    "in_progress",
    "cancelling",
)
TERMINAL_BATCH_STATUSES: tuple[BatchStatus, ...] = ("completed", "failed", "cancelled")


class BatchCreateRequest(OpenAIBaseModel):
    input_file_id: str = Field(
        description="The id of an uploaded JSONL file containing the requests."
    )
    endpoint: BatchEndpoint = Field(
        description="The endpoint used for all requests in the batch."
    )
    completion_window: Literal["24h"] = Field(
        default="24h", description="The time frame to process the batch in."
    )
    metadata: dict[str, str] | None = Field(
        default=None, description="Key-value pairs attached to the batch."
    )


class BatchRequestCounts(OpenAIBaseModel):
    total: int = Field(default=0, description="Total number of requests.")
    completed: int = Field(default=0, description="Requests that succeeded.")
    failed: int = Field(default=0, description="Requests that failed.")


class BatchError(OpenAIBaseModel):
    code: str = Field(description="An error code identifying the error type.")
    message: str = Field(description="A human-readable description of the error.")
    param: str | None = Field(
        default=None, description="The parameter that caused the error."
    )
    line: int | None = Field(
        default=None, description="The input file line number of the request."
    )


class BatchErrors(OpenAIBaseModel):
    object: Literal["list"] = "list"
    data: list[BatchError] = Field(
        default_factory=list, description="The errors of the batch."
    )


class BatchObject(OpenAIBaseModel):
    id: str = Field(description="The batch identifier.")
    object: Literal["batch"] = "batch"
    endpoint: BatchEndpoint = Field(
        description="The endpoint used for all requests in the batch."
    )
    input_file_id: str = Field(description="The id of the input file.")
    output_file_id: str | None = Field(
        default=None, description="The id of the file with successful results."
    )
    error_file_id: str | None = Field(
        default=None, description="The id of the file with failed results."
    )
    status: BatchStatus = Field(description="The current status of the batch.")
    completion_window: str = Field(
        description="The time frame to process the batch in."
    )
    created_at: int = Field(
        description="The Unix timestamp (in seconds) when the batch was created."
    )
    in_progress_at: int | None = Field(
        default=None, description="When the batch started processing."
    )
    finalizing_at: int | None = Field(
        default=None, description="When the batch started finalizing."
    )
    completed_at: int | None = Field(
        default=None, description="When the batch was completed."
    )
    failed_at: int | None = Field(default=None, description="When the batch failed.")
    cancelling_at: int | None = Field(
        default=None, description="When the batch started cancelling."
    )
    cancelled_at: int | None = Field(
        default=None, description="When the batch was cancelled."
    )
    expires_at: int | None = Field(
        default=None, description="When the batch and its files will be deleted."
    )
    request_counts: BatchRequestCounts = Field(
        default_factory=BatchRequestCounts,
        description="The request counts per status.",
    )
    errors: BatchErrors | None = Field(
        default=None, description="Errors that made the batch fail."
    )
    metadata: dict[str, str] | None = Field(
        default=None, description="Key-value pairs attached to the batch."
    )


class BatchListResponse(OpenAIBaseModel):
    object: Literal["list"] = "list"
    data: list[BatchObject] = Field(description="The batches, newest first.")
    has_more: bool = Field(description="Whether more batches are available.")
    first_id: str | None = Field(
        default=None, description="The id of the first batch in `data`."
    )
    last_id: str | None = Field(
        default=None, description="The id of the last batch in `data`."
    )
