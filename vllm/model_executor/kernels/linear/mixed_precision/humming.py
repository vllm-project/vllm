# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Humming GEMM as a mixed-precision WNA16Int linear kernel."""

import torch

from vllm.platforms import current_platform
from vllm.utils.import_utils import has_humming

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig


class HummingLinearKernel(MPLinearKernel):
    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_cuda():
            return False, "Humming is only supported on CUDA"
        if not has_humming():
            return False, "Humming is not installed"
        if c.has_g_idx:
            return False, "Humming does not support act-order (g_idx)"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from vllm.model_executor.layers.quantization.utils.humming_utils import (
            convert_linear_layer_to_humming_standard,
            prepare_humming_layer,
        )

        name_map = {"weight": self.w_q_name, "weight_scale": self.w_s_name}
        group_size = self.config.group_size
        quant_config = {
            "quant_method": "humming",
            "dtype": "int" + str(self.config.weight_type.size_bits),
            "group_size": 0 if group_size == -1 else group_size,
        }

        if self.config.zero_points:
            assert self.w_zp_name is not None
            name_map["zero_point"] = self.w_zp_name
            quant_config["has_zero_point"] = True

        convert_linear_layer_to_humming_standard(layer=layer, name_map=name_map)
        input_quant_config = getattr(layer, "_humming_input_quant_config", None)
        prepare_humming_layer(
            layer, quant_config, input_quant_config=input_quant_config
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm.utils.humming import HummingMethod

        flatten_inputs = x.view(-1, x.size(-1))
        output = HummingMethod.forward_layer(
            layer=layer,
            inputs=flatten_inputs,
            compute_config=layer.compute_config,
        )
        return output.view(*x.shape[:-1], output.size(-1))
