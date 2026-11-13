# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Literal, TypeAlias

from vllm.entrypoints.serve.engine.protocol import OpenAIBaseModel

FilePurpose: TypeAlias = Literal["batch", "batch_output", "batch_error"]


class FileObject(OpenAIBaseModel):
    id: str
    object: Literal["file"] = "file"
    bytes: int
    created_at: int
    filename: str
    purpose: FilePurpose


class FileListResponse(OpenAIBaseModel):
    object: Literal["list"] = "list"
    data: list[FileObject]


class FileDeleteResponse(OpenAIBaseModel):
    id: str
    object: Literal["file"] = "file"
    deleted: bool = True
