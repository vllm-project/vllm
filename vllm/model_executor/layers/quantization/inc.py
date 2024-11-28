from typing import Any, Dict, List, Optional

import torch

from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE, UnquantizedFusedMoEMethod)
from vllm.model_executor.layers.linear import (LinearBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)


class INCConfig(QuantizationConfig):
    """Config class for FP8 using Intel Neural Compressor."""

    @classmethod
    def get_name(cls) -> str:
        return "inc"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "INCConfig":
        raise AssertionError

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        elif isinstance(layer, FusedMoE):
            return UnquantizedFusedMoEMethod()
        return None

    @classmethod
    def get_min_capability(cls) -> int:
        raise AssertionError

    @staticmethod
    def get_config_filenames() -> List[str]:
        return []
