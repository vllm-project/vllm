# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch

from vllm.config.quantization import (
    OnlineQuantizationConfigArgs,
    OnlineQuantScheme,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    should_ignore_layer,
)
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PerBlockOnlineLinearMethod,
    Fp8PerBlockOnlineMoEMethod,
    Fp8PerTensorOnlineLinearMethod,
    Fp8PerTensorOnlineMoEMethod,
)
from vllm.model_executor.layers.quantization.online.int8 import (
    Int8OnlineMoEMethod,
)

logger = init_logger(__name__)


class OnlineQuantizationConfig(QuantizationConfig):
    """Model-level config class for online quantization (quantize fp16/bf16 weights
    during model loading, without requiring a pre-quantized checkpoint)."""

    def __init__(
        self,
        args: OnlineQuantizationConfigArgs,
    ) -> None:
        super().__init__()
        if (
            args.global_scheme is None
            and args.linear_scheme_override is None
            and args.moe_scheme_override is None
        ):
            raise ValueError(
                "OnlineQuantizationConfig requires at least one of "
                "global_scheme, linear_scheme_override, or "
                "moe_scheme_override to be set."
            )
        self.args = args
        self.quant_scheme = args.global_scheme
        self.ignored_layers: list[str] = args.ignore

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "online"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        # Note: as more online quant schemes will be added, this
        # value will become the minimum across all supported schemes.
        return 75

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "OnlineQuantizationConfig":
        raise NotImplementedError(
            "OnlineQuantizationConfig does not support loading from a "
            "checkpoint config. Use quantization_config or "
            "quantization='fp8_per_tensor'/'fp8_per_block' instead."
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        if isinstance(layer, LinearBase):
            if should_ignore_layer(
                prefix,
                ignore=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()

            linear_scheme = self.args.linear_scheme_override or self.args.global_scheme
            if linear_scheme == OnlineQuantScheme.INT8_PER_CHANNEL_WEIGHT_ONLY:
                logger.warning_once(
                    "INT8 online quantization only quantizes MoE expert "
                    "weights. linear layers remain in full precision."
                )
                return UnquantizedLinearMethod()
            elif linear_scheme == OnlineQuantScheme.FP8_PER_BLOCK:
                return Fp8PerBlockOnlineLinearMethod()
            else:
                return Fp8PerTensorOnlineLinearMethod()
        elif isinstance(layer, FusedMoE):
            if should_ignore_layer(
                prefix,
                ignore=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedFusedMoEMethod(layer.moe_config)

            moe_scheme = self.args.moe_scheme_override or self.args.global_scheme
            if moe_scheme == OnlineQuantScheme.INT8_PER_CHANNEL_WEIGHT_ONLY:
                return Int8OnlineMoEMethod(layer=layer)
            elif moe_scheme == OnlineQuantScheme.FP8_PER_BLOCK:
                return Fp8PerBlockOnlineMoEMethod(layer=layer)
            else:
                return Fp8PerTensorOnlineMoEMethod(layer=layer)
        return None
