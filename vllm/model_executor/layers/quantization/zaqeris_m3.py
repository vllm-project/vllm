# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# ZAQERIS-M3: a 3-stage octa-lattice vector-quantized weight format with sparse
# salient outlier preservation. Each quantized tensor is stored as a bundle of
# compact buffers (see ``ZaqerisAxiomM3Weight``); this config wires those buffers
# into vLLM's linear layers and dequantizes on-the-fly at compute time.

from typing import Any

import torch

from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)


@register_quantization_config("zaqeris_m3")
class ZaqerisM3QuantConfig(QuantizationConfig):
    """Config for the ZAQERIS-M3 quantization method."""

    def __init__(
        self,
        bits: float = 3.16,
        block_size: int = 8,
        stages: int = 3,
    ):
        super().__init__()
        self.bits = bits
        self.block_size = block_size
        self.stages = stages

    def get_name(self) -> str:
        return "zaqeris_m3"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ZaqerisM3QuantConfig":
        return cls(
            bits=float(config.get("bits", 3.16)),
            block_size=int(config.get("block_size", 8)),
            stages=int(config.get("stages", 3)),
        )

    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg: dict[str, Any],
        user_quant: str | None,
        hf_config: Any = None,
    ) -> str | None:
        if user_quant is not None:
            return user_quant
        if hf_quant_cfg.get("quant_method") == "zaqeris_m3":
            return "zaqeris_m3"
        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        from vllm.model_executor.models.zaqeris_axiom import (
            ZaqerisAxiomM3QuantMethod,
        )

        if isinstance(layer, LinearBase):
            return ZaqerisAxiomM3QuantMethod(self)
        return None

    def is_layer_quantized(self, prefix: str) -> bool:
        return True
