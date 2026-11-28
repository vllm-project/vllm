# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pydantic import Field, model_validator

from vllm.config.utils import config


@config
class MMTensorRefsConfig:
    """Configuration for `features.kwargs_refs` in `/inference/v1/generate`:
    processed multimodal tensors that clients upload to an S3-compatible store
    and reference by URI. Passing this config enables them."""

    s3_allowed_uri_prefix: str = ""
    """The `s3://bucket/prefix` that every reference must be under. Required.
    Every caller of the endpoint can reference every object the server can
    read under it, so use one prefix per trust domain."""

    max_object_bytes: int = Field(default=1 << 30, gt=0)
    """The largest object the server downloads. Default: 1 GiB."""

    max_request_bytes: int = Field(default=1 << 30, gt=0)
    """The largest total size of the objects one request references. A request
    over it fails with a 400 before anything is downloaded. Default: 1 GiB."""

    max_reserved_bytes: int = Field(default=8 << 30, gt=0)
    """The memory one API server reserves for referenced tensors. Each request
    reserves twice the declared size of its objects, for the downloaded bytes
    and the decoded tensors, and keeps the reservation until it finishes, even
    if the client disconnects. Requests wait in arrival order until their
    reservation fits. Must be at least twice `max_request_bytes`.
    Default: 8 GiB."""

    max_concurrent_reads: int = Field(default=32, ge=1)
    """How many requests per API server can download references at once.
    Default: 32."""

    @model_validator(mode="after")
    def _validate(self) -> "MMTensorRefsConfig":
        if not self.s3_allowed_uri_prefix.startswith("s3://"):
            raise ValueError(
                "s3_allowed_uri_prefix must be set to an s3://bucket/prefix"
            )
        if 2 * self.max_request_bytes > self.max_reserved_bytes:
            raise ValueError(
                "max_reserved_bytes must be at least twice max_request_bytes"
            )
        return self
