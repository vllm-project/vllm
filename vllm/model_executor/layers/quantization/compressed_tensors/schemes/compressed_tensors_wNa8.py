# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight N-bit INT scheme with symmetric INT8 activation quant via Humming.

Handles compressed-tensors pack-quantized INT weight checkpoints (2-8 bit)
with INT8 symmetric dynamic per-token/per-group input activation
quantization. Static, per-tensor, and asymmetric activation quantization
are not supported.
"""

import math
from collections.abc import Callable
from fractions import Fraction

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
)

from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import (
    MPLinearLayerConfig,
    choose_mp_linear_kernel,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import (  # noqa: E501
    WNA16_SUPPORTED_TYPES_MAP,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_repeat_scales_on_all_ranks,
)
from vllm.model_executor.parameter import (
    BasevLLMParameter,
    ChannelQuantScaleParameter,
    GroupQuantScaleParameter,
    PackedvLLMParameter,
)

logger = init_logger(__name__)

__all__ = ["CompressedTensorsWNA8Int"]


class CompressedTensorsWNA8Int(CompressedTensorsScheme):
    _kernel_backends_being_used: set[str] = set()

    def __init__(
        self,
        num_bits: int,
        strategy: str,
        group_size: int | None = None,
        input_quant: QuantizationArgs | None = None,
        layer_name: str | None = None,
        quant_format: str = "pack-quantized",
    ):
        self.num_bits = num_bits
        self.pack_factor = Fraction(32, num_bits)
        self.strategy = strategy
        self.group_size = -1 if group_size is None else group_size
        self.input_quant = input_quant
        self.layer_name = layer_name
        self.quant_format = quant_format

        if num_bits not in WNA16_SUPPORTED_TYPES_MAP:
            raise ValueError(
                f"Unsupported num_bits = {num_bits} for WNA8Int; "
                f"supported = {sorted(WNA16_SUPPORTED_TYPES_MAP)}"
            )
        self.quant_type = WNA16_SUPPORTED_TYPES_MAP[num_bits]

        if input_quant is not None:
            if not input_quant.symmetric:
                raise ValueError(
                    "WNA8Int requires symmetric activation quantization, "
                    f"got symmetric={input_quant.symmetric}"
                )
            if not input_quant.dynamic:
                raise ValueError(
                    "WNA8Int requires dynamic activation quantization, "
                    f"got dynamic={input_quant.dynamic}"
                )
            if input_quant.strategy not in (
                QuantizationStrategy.TOKEN.value,
                QuantizationStrategy.GROUP.value,
            ):
                raise ValueError(
                    "WNA8Int requires per-token or per-group activation "
                    f"quantization, got strategy={input_quant.strategy}"
                )

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    def _build_input_quant_config(self) -> dict | None:
        """Build the config dict that BaseInputSchema.from_config expects."""
        if self.input_quant is None:
            return None
        iq = self.input_quant
        type_val = iq.type.value if hasattr(iq.type, "value") else iq.type
        strategy_val = (
            iq.strategy.value if hasattr(iq.strategy, "value") else iq.strategy
        )
        return {
            "num_bits": iq.num_bits,
            "type": type_val,
            "strategy": strategy_val,
            "symmetric": iq.symmetric,
            "dynamic": iq.dynamic,
            "group_size": iq.group_size or 0,
            "quant_method": "compressed-tensors",
            "format": "int-quantized",
        }

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_size: int,
        input_size: int,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.output_partition_sizes = output_partition_sizes
        layer.params_dtype = params_dtype
        if not hasattr(layer, "has_bias"):
            layer.has_bias = False

        mp_config = MPLinearLayerConfig(
            full_weight_shape=(input_size, output_size),
            partition_weight_shape=(
                input_size_per_partition,
                output_size_per_partition,
            ),
            weight_type=self.quant_type,
            act_type=params_dtype,
            group_size=self.group_size,
            zero_points=False,
            has_g_idx=False,
        )

        kernel_type = choose_mp_linear_kernel(mp_config)
        if kernel_type.__name__ not in self._kernel_backends_being_used:
            logger.info("Using %s for CompressedTensorsWNA8Int", kernel_type.__name__)
            self._kernel_backends_being_used.add(kernel_type.__name__)

        self.kernel = kernel_type(
            mp_config,
            w_q_param_name="weight_packed",
            w_s_param_name="weight_scale",
        )

        input_quant_config = self._build_input_quant_config()
        if input_quant_config is not None:
            layer._humming_input_quant_config = input_quant_config

        # --- weight parameters ---
        group_size = self.group_size if self.group_size != -1 else input_size
        row_parallel = input_size != input_size_per_partition
        partition_scales = not marlin_repeat_scales_on_all_ranks(
            False, self.group_size, row_parallel
        )
        scales_size = input_size // group_size
        if partition_scales:
            assert input_size_per_partition % group_size == 0
            scales_size = input_size_per_partition // group_size

        packed_input_dim = math.ceil(input_size_per_partition * self.num_bits / 32)
        layer.register_parameter(
            "weight_packed",
            PackedvLLMParameter(
                input_dim=1,
                output_dim=0,
                packed_factor=self.pack_factor,
                packed_dim=1,
                weight_loader=weight_loader,
                data=torch.empty(
                    output_size_per_partition,
                    packed_input_dim,
                    dtype=torch.int32,
                ),
            ),
        )

        scale_data = torch.empty(
            output_size_per_partition, scales_size, dtype=params_dtype
        )
        if partition_scales:
            weight_scale = GroupQuantScaleParameter(
                data=scale_data,
                output_dim=0,
                input_dim=1,
                weight_loader=weight_loader,
            )
        else:
            weight_scale = ChannelQuantScaleParameter(
                data=scale_data,
                output_dim=0,
                weight_loader=weight_loader,
            )
        layer.register_parameter("weight_scale", weight_scale)

        layer.register_parameter(
            "weight_shape",
            BasevLLMParameter(
                data=torch.empty(2, dtype=torch.int64),
                weight_loader=weight_loader,
            ),
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.kernel.process_weights_after_loading(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.kernel.apply_weights(layer, x, bias)
