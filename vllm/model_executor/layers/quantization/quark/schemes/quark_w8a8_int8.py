# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.kernels.scaled_mm import (
    init_int8_linear_kernel,
)
from vllm.model_executor.layers.quantization.quark.schemes import QuarkScheme
from vllm.model_executor.layers.quantization.quark.transform import (
    OrthogonalTransform,
    rotation_weight_loader,
)
from vllm.model_executor.parameter import (
    BasevLLMParameter,
    ChannelQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)

logger = init_logger(__name__)


class QuarkW8A8Int8(QuarkScheme):
    def __init__(
        self,
        qscheme: str,
        is_static_input_scheme: bool | None,
        input_symmetric: bool | None,
        quant_config: dict | None = None,
        layer_names: list[str] | None = None,
    ):
        self.qscheme = qscheme
        self.is_static_input_scheme = is_static_input_scheme
        self.input_symmetric = input_symmetric

        # Setup optional online activation transform.
        if quant_config is not None and layer_names is not None:
            (
                self.use_online_rotation,
                self.rotation_config,
                self.rotation_size,
            ) = OrthogonalTransform.setup_transform(
                quant_config=quant_config, layer_names=layer_names
            )
        else:
            self.use_online_rotation = False
            self.rotation_config = None
            self.rotation_size = None

    @classmethod
    def get_min_capability(cls) -> int:
        # turing and up
        return 75

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        layer.logical_widths = output_partition_sizes

        self.kernel = init_int8_linear_kernel(
            is_channelwise=(self.qscheme == "per_channel"),
            is_static_input_scheme=(self.is_static_input_scheme is True),
            input_symmetric=(self.input_symmetric is True),
            module_name=self.__class__.__name__,
        )

        # WEIGHT
        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes), input_size_per_partition, dtype=torch.int8
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )

        layer.register_parameter("weight", weight)

        # WEIGHT SCALE
        if self.qscheme == "per_channel":
            weight_scale = ChannelQuantScaleParameter(
                data=torch.empty((sum(output_partition_sizes)), dtype=torch.float32),
                output_dim=0,
                weight_loader=weight_loader,
            )
            ChannelQuantZPParameter = ChannelQuantScaleParameter
            weight_zero_point = ChannelQuantZPParameter(
                data=torch.empty((sum(output_partition_sizes)), dtype=torch.int8),
                output_dim=0,
                weight_loader=weight_loader,
            )
        else:
            assert self.qscheme == "per_tensor"
            weight_scale = PerTensorScaleParameter(
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader,
            )
            PerTensorZPParameter = PerTensorScaleParameter
            weight_zero_point = PerTensorZPParameter(
                data=torch.empty(len(output_partition_sizes), dtype=torch.int8),
                weight_loader=weight_loader,
            )
        layer.register_parameter("weight_scale", weight_scale)
        layer.register_parameter("weight_zero_point", weight_zero_point)

        # INPUT SCALE
        input_zero_point = None
        input_scale = None
        if self.is_static_input_scheme:
            input_scale = BasevLLMParameter(
                data=torch.empty(1, dtype=torch.float32), weight_loader=weight_loader
            )

            input_zero_point = BasevLLMParameter(
                data=torch.empty(1, dtype=torch.int8), weight_loader=weight_loader
            )

        layer.register_parameter("input_scale", input_scale)
        layer.register_parameter("input_zero_point", input_zero_point)
        if not hasattr(layer, "azp_adj"):
            layer.register_parameter("azp_adj", None)

        if self.use_online_rotation:
            dtype = torch.float64 if self.rotation_config["trainable"] else torch.int8  # type: ignore[index]

            input_rotation = ModelWeightParameter(
                data=torch.empty(self.rotation_size, self.rotation_size, dtype=dtype),
                input_dim=1,
                output_dim=0,
                weight_loader=rotation_weight_loader,
            )
            layer.register_parameter("input_rotation", input_rotation)

            self.input_transform = OrthogonalTransform(
                layer.input_rotation, self.rotation_config
            )

    # Checkpoints are serialized in quark format, which is
    # different from the format the kernel may want. Handle repacking here.
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.register_parameter("weight_zero_point", None)
        delattr(layer, "weight_zero_point")
        if self.input_symmetric:
            layer.register_parameter("input_zero_point", None)
            delattr(layer, "input_zero_point")

        self.kernel.process_weights_after_loading(layer)

        if self.use_online_rotation:
            self.input_transform.post_process_transform()

    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        if self.use_online_rotation:
            x = self.input_transform(x)

        return self.kernel.apply_weights(layer, x, bias)
