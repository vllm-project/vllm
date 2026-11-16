# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from dataclasses import dataclass


@dataclass(slots=True)
class BaseModelPath:
    name: str
    model_path: str


@dataclass(slots=True)
class LoRAModulePath:
    name: str
    path: str
    base_model_name: str | None = None
    is_3d_lora_weight: bool = False
