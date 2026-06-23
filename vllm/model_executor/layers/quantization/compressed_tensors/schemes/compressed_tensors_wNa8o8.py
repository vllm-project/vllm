# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight N-bit INT scheme with static INT8 input/output activation quant.

Handles compressed-tensors INT weight checkpoints that carry static per-tensor
INT8 ``input_activations`` and/or ``output_activations``. The activation quant is
reproduced as a float fake-quant on the layer input and output, around a
weight-only matmul, rather than a fused int8 GEMM.
"""

from collections.abc import Callable

import torch
from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32

from vllm.model_executor.kernels.linear import (
    MPLinearLayerConfig,
    choose_mp_linear_kernel,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_repeat_scales_on_all_ranks,
)
from vllm.model_executor.parameter import (
    BasevLLMParameter,
    ChannelQuantScaleParameter,
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PackedvLLMParameter,
)
from vllm.scalar_type import scalar_types

__all__ = ["CompressedTensorsWNA8O8Int", "fake_quant_static_int8"]

WNA8O8_SUPPORTED_TYPES_MAP = {
    2: scalar_types.uint2b2,
    4: scalar_types.uint4b8,
    8: scalar_types.uint8b128,
}


def fake_quant_static_int8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Static per-tensor symmetric INT8 quantize-dequantize, in x's dtype."""
    scale = scale.to(x.dtype)
    q = torch.clamp(torch.round(x / scale), -128.0, 127.0)
    return q * scale


class CompressedTensorsWNA8O8Int(CompressedTensorsScheme):
    def __init__(
        self,
        num_bits: int,
        strategy: str,
        group_size: int | None = None,
        has_input_act: bool = False,
        has_output_act: bool = False,
        layer_name: str | None = None,
        quant_format: str = "pack-quantized",
    ):
        self.num_bits = num_bits
        self.pack_factor = 32 // num_bits
        self.strategy = strategy
        self.group_size = -1 if group_size is None else group_size
        self.has_input_act = has_input_act
        self.has_output_act = has_output_act
        self.layer_name = layer_name
        # "pack-quantized" (sub-byte, int32-packed) or "int-quantized" (8-bit int8).
        self.quant_format = quant_format
        self.is_int_quantized = quant_format == "int-quantized"
        if num_bits not in WNA8O8_SUPPORTED_TYPES_MAP:
            raise ValueError(
                f"Unsupported num_bits = {num_bits} for WNA8O8Int; "
                f"supported = {sorted(WNA8O8_SUPPORTED_TYPES_MAP)}"
            )
        self.quant_type = WNA8O8_SUPPORTED_TYPES_MAP[num_bits]
        self._input_scale: torch.Tensor | None = None
        self._output_scale: torch.Tensor | None = None

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

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
        # Set for kernels' weight prep; also covers ParallelLMHead, which does
        # not set these in __init__.
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
            act_type=params_dtype,  # activation quant applied externally (SRQ)
            group_size=self.group_size,
            zero_points=False,
            has_g_idx=False,
        )
        self.kernel = choose_mp_linear_kernel(mp_config)(
            mp_config,
            w_q_param_name="weight_packed",
            w_s_param_name="weight_scale",
        )

        self._register_weight(
            layer, input_size, input_size_per_partition, params_dtype, weight_loader
        )

    def _register_weight(
        self, layer, input_size, input_size_per_partition, params_dtype, weight_loader
    ):
        out = layer.output_size_per_partition
        if self.is_int_quantized:
            # Plain int8 weight; packed to the canonical int32 layout after load.
            layer.register_parameter(
                "weight",
                ModelWeightParameter(
                    data=torch.empty(out, input_size_per_partition, dtype=torch.int8),
                    input_dim=1,
                    output_dim=0,
                    weight_loader=weight_loader,
                ),
            )
        else:
            layer.register_parameter(
                "weight_packed",
                PackedvLLMParameter(
                    input_dim=1,
                    output_dim=0,
                    packed_dim=1,
                    packed_factor=self.pack_factor,
                    weight_loader=weight_loader,
                    data=torch.empty(
                        out,
                        input_size_per_partition // self.pack_factor,
                        dtype=torch.int32,
                    ),
                ),
            )
            layer.register_parameter(
                "weight_shape",
                BasevLLMParameter(
                    data=torch.empty(2, dtype=torch.int64), weight_loader=weight_loader
                ),
            )

        # Scale: per-output-channel, or per group along the input dim under TP.
        group_size = self.group_size if self.group_size != -1 else input_size
        partitioned = not marlin_repeat_scales_on_all_ranks(
            False, self.group_size, input_size != input_size_per_partition
        )
        scales = (input_size_per_partition if partitioned else input_size) // group_size
        scale_data = torch.empty(out, scales, dtype=params_dtype)
        if partitioned:
            assert input_size_per_partition % group_size == 0
            weight_scale = GroupQuantScaleParameter(
                data=scale_data, output_dim=0, input_dim=1, weight_loader=weight_loader
            )
        else:
            weight_scale = ChannelQuantScaleParameter(
                data=scale_data, output_dim=0, weight_loader=weight_loader
            )
        layer.register_parameter("weight_scale", weight_scale)

        for name, present in (
            ("input_scale", self.has_input_act),
            ("output_scale", self.has_output_act),
        ):
            if present:
                layer.register_parameter(
                    name,
                    BasevLLMParameter(
                        data=torch.empty(1, dtype=torch.float32),
                        weight_loader=weight_loader,
                    ),
                )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Lift the static activation scales off the layer (applied externally) so
        # the kernel only sees weight tensors. Drop uncalibrated (zero) scales.
        self._input_scale = self._take_act_scale(layer, "input_scale")
        self._output_scale = self._take_act_scale(layer, "output_scale")
        self.has_input_act = self._input_scale is not None
        self.has_output_act = self._output_scale is not None

        if self.is_int_quantized:
            self._pack_int_quantized_weight(layer)

        self.kernel.process_weights_after_loading(layer)

    def _pack_int_quantized_weight(self, layer: torch.nn.Module) -> None:
        """Normalize an int-quantized (plain int8) weight to the canonical
        ``weight_packed`` int32 + ``weight_shape`` layout the MP kernels expect."""
        weight = layer.weight
        out_features, in_features = weight.shape
        packed = pack_to_int32(weight.data.contiguous(), self.num_bits)
        delattr(layer, "weight")

        def _noop_loader(*_, **__):
            return None

        layer.register_parameter(
            "weight_packed",
            PackedvLLMParameter(
                data=packed.contiguous(),
                input_dim=1,
                output_dim=0,
                packed_dim=1,
                packed_factor=self.pack_factor,
                weight_loader=_noop_loader,
            ),
        )
        layer.register_parameter(
            "weight_shape",
            BasevLLMParameter(
                data=torch.tensor([out_features, in_features], dtype=torch.int64),
                weight_loader=_noop_loader,
            ),
        )

    @staticmethod
    def _take_act_scale(layer, name: str) -> torch.Tensor | None:
        param = getattr(layer, name, None)
        if param is None:
            return None
        scale = param.data.clone()
        delattr(layer, name)
        return None if float(scale.reshape(-1)[0]) == 0.0 else scale

    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        if self.has_input_act:
            x = fake_quant_static_int8(x, self._input_scale)
        out = self.kernel.apply_weights(layer, x, bias)
        if self.has_output_act:
            out = fake_quant_static_int8(out, self._output_scale)
        return out
