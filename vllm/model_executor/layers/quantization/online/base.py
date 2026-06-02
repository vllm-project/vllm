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
    Fp8PtpcOnlineLinearMethod,  # cohere
    Fp8PtpcOnlineMoEMethod,  # cohere
)
from vllm.model_executor.layers.quantization.online.int8 import (
    Int8OnlineMoEMethod,
)
from vllm.model_executor.layers.quantization.online.mxfp8 import (
    Mxfp8OnlineLinearMethod,
    Mxfp8OnlineMoEMethod,
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
        # cohere start
        """Load online quantization config from a checkpoint's
        ``config.json`` ``quantization_config`` block.

        Schema::

            "quantization_config": {
                # Picks the dispatch path. May be ``"online"`` (in which case
                # ``global_scheme`` / ``linear_scheme_override`` /
                # ``moe_scheme_override`` must be set in the same dict) or one
                # of the scheme shorthands ``"fp8_per_tensor"``,
                # ``"fp8_per_block"``, ``"mxfp8"``, or
                # ``"int8_per_channel_weight_only"`` -- the latter auto-populate
                # ``global_scheme``.
                "quant_method": "fp8_per_block",

                # Optional. Entries are matched as exact module names or
                # regex patterns when prefixed with ``re:``.
                "ignore": ["re:.*self_attn\\..*", "model.layers.0.mlp.experts"],

                # Optional aliases of ``ignore`` (for back-compat with
                # ``Mxfp8Config`` / HF / modelopt). Merged into ``ignore``.
                "ignored_layers": [...],
                "modules_to_not_convert": [...],

                # Optional explicit overrides if you want different schemes
                # for linear vs MoE layers.
                "linear_scheme_override": "fp8_per_block",
                "moe_scheme_override": "fp8_per_tensor",

                # Optional, must be ``"dynamic"`` (the only supported value).
                "activation_scheme": "dynamic"
            }
        """
        args_dict: dict[str, Any] = dict(config)

        quant_method = args_dict.pop("quant_method", None)

        ignore: list[str] = list(args_dict.pop("ignore", None) or [])
        for alias in ("ignored_layers", "modules_to_not_convert"):
            extra = args_dict.pop(alias, None)
            if extra:
                ignore.extend(extra)
        if ignore:
            args_dict["ignore"] = ignore

        activation_scheme = args_dict.pop("activation_scheme", None)
        if activation_scheme is not None and activation_scheme != "dynamic":
            raise ValueError(
                "online quantization only supports activation_scheme="
                f"'dynamic', got {activation_scheme!r}"
            )

        scheme_values = {s.value for s in OnlineQuantScheme}
        if quant_method in scheme_values and args_dict.get("global_scheme") is None:
            args_dict["global_scheme"] = quant_method

        return cls(args=OnlineQuantizationConfigArgs(**args_dict))
        # cohere end

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
            elif linear_scheme == OnlineQuantScheme.FP8_PER_CHANNEL:  # cohere
                return Fp8PtpcOnlineLinearMethod()  # cohere
            elif linear_scheme == OnlineQuantScheme.FP8_PER_BLOCK:
                return Fp8PerBlockOnlineLinearMethod()
            elif linear_scheme == OnlineQuantScheme.MXFP8:
                return Mxfp8OnlineLinearMethod()
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
            elif moe_scheme == OnlineQuantScheme.FP8_PER_CHANNEL:  # cohere
                return Fp8PtpcOnlineMoEMethod(layer=layer)  # cohere
            elif moe_scheme == OnlineQuantScheme.FP8_PER_BLOCK:
                return Fp8PerBlockOnlineMoEMethod(layer=layer)
            elif moe_scheme == OnlineQuantScheme.MXFP8:
                return Mxfp8OnlineMoEMethod(layer=layer)
            else:
                return Fp8PerTensorOnlineMoEMethod(layer=layer)
        return None
