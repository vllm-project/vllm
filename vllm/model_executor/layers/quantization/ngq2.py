# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NGQ2 checkpoints: n-gram lookup tables stored as a 2-bit trellis code.

Only the tables in ``modules_to_quantize`` are packed; every other layer is
unquantized. Each model builds its own table method, so this config only parses
the checkpoint metadata and says which tables are packed.
"""

from typing import Any

import regex as re
import torch

from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.models.utils import WeightsMapper

# The format parameters the decode kernel implements.
_SUPPORTED_CODE = {
    "version": 1,
    "code": "qtip_hyb_trellis",
    "L": 16,
    "K": 2,
    "V": 2,
    "granularity": "row",
}


def _table_key(name: str) -> str:
    """Drop the wrapper prefix, which differs between HF and vLLM module names."""
    return name.rpartition("layers.")[2]


class Ngq2Config(QuantizationConfig):
    """Config class for NGQ2 packed n-gram tables."""

    def __init__(self, modules_to_quantize: list[str]) -> None:
        super().__init__()
        self.modules_to_quantize = modules_to_quantize

    def get_name(self) -> QuantizationMethods:
        return "ngq2"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Ngq2Config":
        for key, value in _SUPPORTED_CODE.items():
            if config.get(key) != value:
                raise NotImplementedError(
                    f"NGQ2 {key}={config.get(key)!r} is not supported, "
                    f"expected {value!r}"
                )
        if "base_quantization_config" in config:
            raise NotImplementedError(
                "NGQ2 on a quantized backbone (base_quantization_config) is not "
                "supported"
            )
        modules_to_quantize = cls.get_from_keys(config, ["modules_to_quantize"])
        if len(modules_to_quantize) != 1:
            raise NotImplementedError(
                "NGQ2 supports exactly one table, got "
                f"modules_to_quantize={modules_to_quantize!r}"
            )
        return cls(modules_to_quantize)

    def get_checkpoint_weight_mapper(self) -> WeightsMapper:  # type: ignore[override]
        """Give the top-level codebook tensor to the table that decodes with it."""
        (table,) = self.modules_to_quantize
        return super().get_checkpoint_weight_mapper() | WeightsMapper(
            orig_to_new_regex={
                re.compile(r"^ngq2\.codebook$"): f"{table}.ngq2_codebook"
            }
        )

    def is_table_quantized(self, prefix: str) -> bool:
        """Return whether the n-gram table at `prefix` is stored as NGQ2."""
        return _table_key(prefix) in map(_table_key, self.modules_to_quantize)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        if isinstance(layer, RoutedExperts):
            return UnquantizedFusedMoEMethod(layer.moe_config)
        return None
