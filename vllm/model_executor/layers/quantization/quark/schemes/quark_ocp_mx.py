# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from fractions import Fraction

import torch

from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import (
    MxFp4LinearKernel,
    MxFp6LinearKernel,
    Mxfp8LinearKernel,
    init_mxfp4_linear_kernel,
    init_mxfp6_linear_kernel,
    init_mxfp8_linear_kernel,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_SCALE_DTYPE,
    MXFP8_VALUE_DTYPE,
)
from vllm.model_executor.layers.quantization.utils.ocp_mx_utils import (
    _ACTIVATION_QUANT_KEY_MAP,
    _WEIGHT_QUANT_KEY_MAP,
    OCP_MX_BLOCK_SIZE,
    OCP_MX_UNPACKED_DTYPES,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Static,
    kMxfp6E2M3Static,
    kMxfp6E3M2Static,
    kMxfp8Static,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PackedvLLMParameter,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform

from .quark_scheme import QuarkScheme

logger = init_logger(__name__)


class QuarkOCP_MX(QuarkScheme):
    ocp_mx_linear: MxFp6LinearKernel | MxFp4LinearKernel | Mxfp8LinearKernel
    supported_activation_quant_keys = [*_ACTIVATION_QUANT_KEY_MAP.values(), None]
    supported_weight_quant_keys = [*_WEIGHT_QUANT_KEY_MAP.values()]

    def __init__(
        self,
        weight_quant_key: QuantKey,
        activation_quant_key: QuantKey | None,
        dynamic_mxfp4_quant: bool = False,
        scale_block_rows: int = 1,
    ):
        super().__init__(weight_quant_key, activation_quant_key)
        self.dynamic_mxfp4_quant = dynamic_mxfp4_quant
        self.scale_block_rows = scale_block_rows
        self.weight_dtype = next(
            dtype
            for dtype, quant_key in _WEIGHT_QUANT_KEY_MAP.items()
            if quant_key == weight_quant_key
        )
        self.input_dtype = (
            next(
                dtype
                for dtype, quant_key in _ACTIVATION_QUANT_KEY_MAP.items()
                if quant_key == activation_quant_key
            )
            if activation_quant_key is not None
            else None
        )

        self.is_unpacked = self.weight_dtype in OCP_MX_UNPACKED_DTYPES

        if self.is_unpacked:
            self.packed_factor: int | Fraction = 1
        elif self.weight_dtype == "mxfp4":
            self.packed_factor = 2
        else:
            self.packed_factor = Fraction(numerator=8, denominator=6)

        if self.is_unpacked:
            # MXFP8 runs on the shared kernel abstraction, which picks a native
            # backend where one exists and emulates otherwise. The MXFP4/MXFP6
            # warnings below do not apply.
            return

        if not current_platform.supports_mx():
            logger.warning_once(
                "The current platform does not support native MXFP4/MXFP6 "
                "computation. Simulated weight dequantization and activation "
                "QDQ (quantize and dequantize) will be used, with the linear "
                "layers computed in high precision."
            )

        if current_platform.supports_mx() and (
            self.input_dtype != "mxfp4" or self.weight_dtype != "mxfp4"
        ):
            logger.warning_once(
                "The current platform supports native MXFP4/MXFP6 "
                f"computation, but kernels for input_dtype={self.input_dtype} "
                f"and weight_dtype={self.weight_dtype} are not yet integrated "
                "in vLLM. Simulated weight dequantization and activation "
                "QDQ (quantize and dequantize) will be used, with the linear "
                "layers computed in high precision."
            )

    def get_packed_dim(self, dim: int, quant_dtype: str):
        if quant_dtype in OCP_MX_UNPACKED_DTYPES:
            return dim
        elif quant_dtype == "mxfp4":
            assert dim % 2 == 0
            return dim // 2
        elif quant_dtype in {"mxfp6_e3m2", "mxfp6_e2m3"}:
            # FP6 packs 4 * 6 = 24 bits on 3 bytes.
            assert (dim * 3) % 4 == 0
            return (dim * 3) // 4
        else:
            raise NotImplementedError(
                "Unsupported quant_dtype in QuarkOCP_MX.get_packed_dim, "
                f"got quant_dtype={quant_dtype}. Something is wrong, please "
                "open an issue."
            )

    def _expanding_scale_loader(self, weight_loader: Callable) -> Callable:
        """Expand a 2-D block scale to one row per weight row.

        A ``block_size=[R, 32]`` export carries one e8m0 value per R weight
        rows; the kernels want one per row. At ``R == 1`` the layouts already
        coincide, so the loader is passed through untouched.
        """
        if self.scale_block_rows == 1:
            return weight_loader

        rows = self.scale_block_rows

        def expanding_loader(param, loaded_weight, *args, **kwargs):
            loaded_weight = loaded_weight.view(torch.uint8).repeat_interleave(
                rows, dim=0
            )
            return weight_loader(param, loaded_weight, *args, **kwargs)

        return expanding_loader

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    def process_dynamic_mxfp4_weights_after_loading(
        self, layer: torch.nn.Module
    ) -> None:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant

        w_q, w_s = dynamic_mxfp4_quant(layer.weight)
        layer.weight_scale = torch.nn.Parameter(w_s, requires_grad=False)
        layer.weight = torch.nn.Parameter(w_q, requires_grad=False)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight = torch.nn.Parameter(layer.weight.data, requires_grad=False)

        if self.dynamic_mxfp4_quant:
            self.process_dynamic_mxfp4_weights_after_loading(layer)

        self.ocp_mx_linear.process_weights_after_loading(layer)

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        if input_size_per_partition % OCP_MX_BLOCK_SIZE != 0:
            layer_name = getattr(layer, "prefix", "") or type(layer).__name__
            raise ValueError(
                f"OCP MX linear layer {layer_name!r} has an input size per "
                f"partition of {input_size_per_partition}, which must be "
                f"divisible by the OCP MX group size {OCP_MX_BLOCK_SIZE}. "
                "Choose a compatible tensor-parallel size or avoid "
                "tensor-parallel sharding for this layer."
            )

        if self.dynamic_mxfp4_quant:
            weight = ModelWeightParameter(
                data=torch.empty(
                    sum(output_partition_sizes),
                    input_size_per_partition,
                    dtype=params_dtype,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            )

            layer.register_parameter("weight", weight)
            set_weight_attrs(weight, kwargs)
        elif self.is_unpacked:
            output_size_per_partition = sum(output_partition_sizes)
            layer.logical_widths = output_partition_sizes
            layer.input_size_per_partition = input_size_per_partition
            layer.output_size_per_partition = output_size_per_partition
            layer.params_dtype = params_dtype

            weight = ModelWeightParameter(
                data=torch.empty(
                    output_size_per_partition,
                    input_size_per_partition,
                    dtype=MXFP8_VALUE_DTYPE,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter("weight", weight)

            weight_scale = GroupQuantScaleParameter(
                data=torch.empty(
                    output_size_per_partition,
                    input_size_per_partition // OCP_MX_BLOCK_SIZE,
                    dtype=MXFP8_SCALE_DTYPE,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=self._expanding_scale_loader(weight_loader),
            )
            layer.register_parameter("weight_scale", weight_scale)
            layer.weight_block_size = [1, OCP_MX_BLOCK_SIZE]
        else:
            output_size_per_partition = sum(output_partition_sizes)
            layer.logical_widths = output_partition_sizes

            # WEIGHT
            weight = PackedvLLMParameter(
                data=torch.empty(
                    output_size_per_partition,
                    self.get_packed_dim(input_size_per_partition, self.weight_dtype),
                    dtype=torch.uint8,
                ),
                input_dim=1,
                output_dim=0,
                packed_dim=1,
                packed_factor=self.packed_factor,
                weight_loader=weight_loader,
            )
            layer.register_parameter("weight", weight)

            # WEIGHT SCALE
            weight_scale = GroupQuantScaleParameter(
                data=torch.empty(
                    output_size_per_partition,
                    input_size_per_partition // OCP_MX_BLOCK_SIZE,
                    dtype=torch.uint8,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter("weight_scale", weight_scale)

        if self.weight_quant_key == kMxfp4Static:
            self.ocp_mx_linear = init_mxfp4_linear_kernel(
                activation_quant_key=self.activation_quant_key,
            )
        elif self.weight_quant_key in [kMxfp6E2M3Static, kMxfp6E3M2Static]:
            self.ocp_mx_linear = init_mxfp6_linear_kernel(
                weight_quant_key=self.weight_quant_key,
                activation_quant_key=self.activation_quant_key,
            )
        elif self.weight_quant_key == kMxfp8Static:
            self.ocp_mx_linear = init_mxfp8_linear_kernel()
        else:
            raise NotImplementedError(
                "No OCP MX linear kernel for weight_quant_key="
                f"{self.weight_quant_key}. Something is wrong, please open an "
                "issue."
            )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.ocp_mx_linear.apply_weights(layer, x, bias)
