# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pydantic import BaseModel, Field


class LoadLoRAAdapterRequest(BaseModel):
    lora_name: str
    lora_path: str


class UnloadLoRAAdapterRequest(BaseModel):
    lora_name: str
    lora_int_id: int | None = Field(default=None)
