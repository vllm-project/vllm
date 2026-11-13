# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Literal, TypeAlias

from pydantic import Field

from vllm.entrypoints.serve.engine.protocol import OpenAIBaseModel

FilePurpose: TypeAlias = Literal["batch", "batch_output", "batch_error"]


class FileObject(OpenAIBaseModel):
    id: str = Field(description="The file identifier.")
    object: Literal["file"] = "file"
    bytes: int = Field(description="The size of the file in bytes.")
    created_at: int = Field(
        description="The Unix timestamp (in seconds) when the file was created."
    )
    filename: str = Field(description="The name of the file.")
    purpose: FilePurpose = Field(description="The intended purpose of the file.")


class FileListResponse(OpenAIBaseModel):
    object: Literal["list"] = "list"
    data: list[FileObject] = Field(description="The files, newest first.")


class FileDeleteResponse(OpenAIBaseModel):
    id: str = Field(description="The identifier of the deleted file.")
    object: Literal["file"] = "file"
    deleted: bool = Field(default=True, description="Whether the file was deleted.")
