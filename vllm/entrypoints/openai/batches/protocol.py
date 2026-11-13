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
    input_file_id: str
    endpoint: BatchEndpoint
    completion_window: Literal["24h"] = "24h"
    metadata: dict[str, str] | None = None


class BatchRequestCounts(OpenAIBaseModel):
    total: int = 0
    completed: int = 0
    failed: int = 0


class BatchError(OpenAIBaseModel):
    code: str
    message: str
    param: str | None = None
    line: int | None = Field(
        default=None, description="The input file line number of the request."
    )


class BatchErrors(OpenAIBaseModel):
    object: Literal["list"] = "list"
    data: list[BatchError] = Field(default_factory=list)


class BatchObject(OpenAIBaseModel):
    id: str
    object: Literal["batch"] = "batch"
    endpoint: BatchEndpoint
    input_file_id: str
    output_file_id: str | None = None
    error_file_id: str | None = None
    status: BatchStatus
    completion_window: str
    created_at: int
    in_progress_at: int | None = None
    finalizing_at: int | None = None
    completed_at: int | None = None
    failed_at: int | None = None
    cancelling_at: int | None = None
    cancelled_at: int | None = None
    expires_at: int | None = None
    request_counts: BatchRequestCounts = Field(default_factory=BatchRequestCounts)
    errors: BatchErrors | None = None
    metadata: dict[str, str] | None = None


class BatchListResponse(OpenAIBaseModel):
    object: Literal["list"] = "list"
    data: list[BatchObject]
    has_more: bool
    first_id: str | None = None
    last_id: str | None = None
