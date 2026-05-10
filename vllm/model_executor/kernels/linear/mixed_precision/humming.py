# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from humming.layer import HummingMethod

from vllm import envs
from vllm.model_executor.layers.quantization.utils.humming_utils import (
    convert_linear_layer_to_humming_standard,
    prepare_humming_layer,
)
from vllm.platforms import current_platform

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig


class HummingLinearKernel(MPLinearKernel):
    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if envs.VLLM_USE_HUMMING_LINEAR_KERNEL is None:
            return False, "Humming is disabled by default"

        if envs.VLLM_USE_HUMMING_LINEAR_KERNEL is False:
            return False, "Humming is disabled by VLLM_USE_HUMMING_LINEAR_KERNEL=0"

        if not current_platform.is_cuda():
            return False, "Humming only supported on CUDA"

        if c.has_g_idx:
            return False, "Humming doesn't support actorder"

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        name_map = {"weight": self.w_q_name, "weight_scale": self.w_s_name}

        quant_config = {
            "quant_method": "humming",
            "dtype": "int" + str(self.config.weight_type.size_bits),
            "group_size": self.config.group_size,
        }

        if self.config.zero_points:
            assert self.w_zp_name is not None
            name_map["zero_point"] = self.w_zp_name
            quant_config["has_zero_point"] = True

        convert_linear_layer_to_humming_standard(layer=layer, name_map=name_map)
        prepare_humming_layer(layer, quant_config)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        flatten_inputs = x.view(-1, x.size(-1))
        output = HummingMethod.forward_layer(
            layer=layer,
            inputs=flatten_inputs,
            compute_config=layer.compute_config,
        )
        output = output.view(*x.shape[:-1], output.size(-1))
        return output
