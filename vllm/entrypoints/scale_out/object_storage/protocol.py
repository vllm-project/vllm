# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pydantic import BaseModel


class UUIDResponse(BaseModel):
    uuid: str
    size: int
