# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-tensor precision metadata of Inkling2 checkpoints (hf_quant_config.json).

Only the routed experts are quantized (NVFP4 or MXFP8); they are handled by
`Inkling2MoE`. Every other layer runs unquantized.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)

_COMPUTE_DTYPES = ("float32", "bfloat16", "mxfp8", "nvfp4")


class Inkling2QuantConfig(QuantizationConfig):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.mxfp8_weights: set[str] = set()
        for group in config["model"]["parameter_groups"]:
            compute = group["compute_dtype"]
            if compute not in _COMPUTE_DTYPES:
                raise ValueError(f"Unsupported Inkling2 compute dtype: {compute}")
            if compute == "mxfp8":
                self.mxfp8_weights.update(group["tensors"])

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "inkling2"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 100

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["hf_quant_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Inkling2QuantConfig:
        return cls(config)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        return None

    def is_mxfp8_layer(self, layer_id: int) -> bool:
        name = f"model.llm.layers.{layer_id}.mlp.experts.w13_weight"
        return name in self.mxfp8_weights
