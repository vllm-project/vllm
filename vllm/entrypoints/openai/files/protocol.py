# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pydantic response models for the `/v1/files` endpoint.

Shapes are a conservative subset of OpenAI's Files API — enough to
satisfy the `openai-python` SDK and LiteLLM's passthrough without
committing to OpenAI-specific values like `purpose="assistants"`.
"""

from typing import Literal

from pydantic import Field

from vllm.entrypoints.openai.engine.protocol import OpenAIBaseModel

# Allowed `purpose` values for uploads. Matches a conservative subset of
# OpenAI's Files API. Values like `assistants`, `batch`, and `fine-tune` are
# specific to OpenAI's hosted services and are rejected here with 400.
FilePurpose = Literal["vision", "user_data"]

# Valid object statuses. Uploads land as `processed` once the bytes are
# streamed to disk and MIME-validated.
FileStatus = Literal["uploaded", "processed", "error"]


class FileObject(OpenAIBaseModel):
    """An uploaded file, shaped to be compatible with OpenAI's Files API.

    The `id` is a 128-bit capability handle (`file-<32 hex>`). Possession of
    the id implies authority to access the file, subject to scope-header
    checks when the server is configured with `--file-upload-scope-header`.
    """

    id: str
    object: Literal["file"] = "file"
    bytes: int
    created_at: int
    expires_at: int | None = None
    filename: str
    purpose: FilePurpose
    status: FileStatus = "processed"
    status_details: str | None = None


class FileList(OpenAIBaseModel):
    """Response shape for `GET /v1/files`.

    Returned only when `--file-upload-disable-listing` is NOT set. Scoping
    (if enabled) filters the returned data to files matching the caller's
    scope header value.
    """

    object: Literal["list"] = "list"
    data: list[FileObject] = Field(default_factory=list)


class FileDeleteResponse(OpenAIBaseModel):
    """Response shape for `DELETE /v1/files/{file_id}`."""

    id: str
    object: Literal["file"] = "file"
    deleted: bool = True
