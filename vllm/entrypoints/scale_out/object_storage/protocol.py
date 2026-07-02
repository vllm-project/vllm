# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pydantic import BaseModel


class ObjectInfo(BaseModel):
    name: str
    size: int
    block_size: int
    blocks: list[int]
    last_modified: str
