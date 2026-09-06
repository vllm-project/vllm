# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pydantic import BaseModel, Field


class CancelRequest(BaseModel):
    request_ids: list[str] = Field(
        default_factory=list,
        description=(
            "List of request IDs to cancel. If empty on /abort_requests, "
            "all in-flight requests are aborted."
        ),
    )


class CancelResponse(BaseModel):
    status: str = Field(
        default="cancelled",
        description="Status of the cancellation request.",
    )
    cancelled_request_ids: list[str] = Field(
        default_factory=list,
        description="List of request IDs that were cancelled.",
    )


class SingleCancelResponse(BaseModel):
    status: str = Field(
        default="cancelled",
        description="Status of the cancellation request.",
    )
    request_id: str = Field(
        description="The request ID that was cancelled.",
    )
