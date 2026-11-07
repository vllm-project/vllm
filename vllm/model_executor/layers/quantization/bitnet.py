# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BitNet b1.58: ternary weights with dynamic per-token int8 activations.

Checkpoints pack four ternary values per uint8 along the output dim. They are
unpacked to int8 at load time and run with the int8 W8A8 scaled-mm kernels.
"""

from typing import Any

import torch

from vllm.model_executor.kernels.linear import init_int8_linear_kernel
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)
from vllm.model_executor.parameter import (
    ModelWeightParameter,
    PerTensorScaleParameter,
)


def unpack_ternary_weight(packed: torch.Tensor) -> torch.Tensor:
    """Unpack a `[N // 4, K]` uint8 tensor into `[N, K]` int8 in {-1, 0, 1}.

    Output row `i * N // 4 + r` is stored in bits `[2i, 2i + 2)` of packed row
    `r`, matching `transformers.integrations.bitnet.pack_weights`.
    """
    shifts = torch.arange(0, 8, 2, dtype=torch.uint8, device=packed.device)
    unpacked = (packed.unsqueeze(0) >> shifts.view(-1, 1, 1)) & 3
    return (unpacked.to(torch.int8) - 1).reshape(-1, packed.shape[-1])


class BitNetConfig(QuantizationConfig):
    """Config for BitNet b1.58 checkpoints (`quant_method: "bitnet"`)."""

    def __init__(
        self,
        linear_class: str = "bitlinear",
        quantization_mode: str = "offline",
        modules_to_not_convert: list[str] | None = None,
    ) -> None:
        super().__init__()
        if quantization_mode != "offline":
            raise ValueError(
                "BitNet only supports checkpoints with pre-quantized ternary "
                f"weights (quantization_mode='offline'), got {quantization_mode!r}."
            )
        self.linear_class = linear_class
        self.modules_to_not_convert = modules_to_not_convert or []

    def __repr__(self) -> str:
        return (
            f"BitNetConfig(linear_class={self.linear_class}, "
            f"modules_to_not_convert={self.modules_to_not_convert})"
        )

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "bitnet"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BitNetConfig":
        return cls(
            linear_class=cls.get_from_keys_or(config, ["linear_class"], "bitlinear"),
            quantization_mode=cls.get_from_keys_or(
                config, ["quantization_mode"], "offline"
            ),
            modules_to_not_convert=cls.get_from_keys_or(
                config, ["modules_to_not_convert"], None
            ),
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if not isinstance(layer, LinearBase):
            return None
        if is_layer_skipped(
            prefix,
            self.modules_to_not_convert,
            self.packed_modules_mapping,
            match_mode="substring",
        ):
            return UnquantizedLinearMethod()
        return BitNetLinearMethod(self)


@register_weight_loader_v2_supported_method
class BitNetLinearMethod(LinearMethodBase):
    def __init__(self, quant_config: BitNetConfig) -> None:
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        weight_loader = extra_weight_attrs["weight_loader"]
        layer.logical_widths = output_partition_sizes

        self.kernel = init_int8_linear_kernel(
            is_channelwise=False,
            is_static_input_scheme=False,
            input_symmetric=True,
            module_name=self.__class__.__name__,
        )

        # Unpack the full tensor before the layer loader shards it: packed rows
        # interleave output rows that belong to different TP ranks.
        def ternary_weight_loader(param, loaded_weight, *args, **kwargs):
            if loaded_weight.dtype == torch.uint8:
                loaded_weight = unpack_ternary_weight(loaded_weight)
            weight_loader(param, loaded_weight.to(torch.int8), *args, **kwargs)

        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes), input_size_per_partition, dtype=torch.int8
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=ternary_weight_loader,
        )
        layer.register_parameter("weight", weight)

        weight_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

        layer.register_parameter("input_scale", None)
        layer.register_parameter("input_zero_point", None)
        layer.register_parameter("azp_adj", None)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # `BitLinear` stores the inverse scale (y = x_q @ w_q.T / s_w), while
        # `AutoBitLinear` stores it directly (y = x @ w_q.T * s_w).
        if self.quant_config.linear_class != "autobitlinear":
            layer.weight_scale.data.reciprocal_()
        self.kernel.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.kernel.apply_weights(layer, x, bias)
