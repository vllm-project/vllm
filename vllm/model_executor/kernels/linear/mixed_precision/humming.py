# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Humming GEMM as a mixed-precision WNA16Int linear kernel."""

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTokenSym,
    kInt8DynamicTokenSym,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_humming

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

# Per-token activation quantization requested through `act_type`.
_ACT_QUANT_KEYS = {
    torch.int8: kInt8DynamicTokenSym,
    torch.float8_e4m3fn: kFp8DynamicTokenSym,
}


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
        if c.act_type.itemsize == 1 and c.act_type not in _ACT_QUANT_KEYS:
            return False, f"Activation type ({c.act_type}) not supported by Humming"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from vllm.model_executor.layers.quantization.utils.humming import (
            convert_linear_layer_to_humming_standard,
            get_humming_linear_compute_config,
            prepare_humming_linear_layer_config,
            quant_key_to_input_schema,
        )

        name_map = {"weight": self.w_q_name, "weight_scale": self.w_s_name}
        if self.w_zp_name is not None and hasattr(layer, self.w_zp_name):
            name_map["zero_point"] = self.w_zp_name
        group_size = self.config.group_size
        quant_config = {
            "quant_method": "humming",
            "dtype": "int" + str(self.config.weight_type.size_bits),
            "group_size": 0 if group_size == -1 else group_size,
            "has_zero_point": self.config.zero_points,
        }

        if self.config.zero_points:
            assert self.w_zp_name is not None
            name_map["zero_point"] = self.w_zp_name
            quant_config["has_zero_point"] = True

        convert_linear_layer_to_humming_standard(layer=layer, name_map=name_map)
        input_quant_config = getattr(layer, "_humming_input_quant_config", None)
        input_schema = None
        if (act_key := _ACT_QUANT_KEYS.get(self.config.act_type)) is not None:
            input_schema = quant_key_to_input_schema(act_key)
        self.layer_config = prepare_humming_linear_layer_config(
            layer,
            quant_config,
            input_quant_config=input_quant_config,
            input_schema=input_schema,
        )
        self.compute_config = get_humming_linear_compute_config()
        self.locks = torch.zeros(1024, dtype=torch.int32, device=layer.weight.device)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm.model_executor.layers.quantization.utils.humming import (
            apply_humming_linear,
        )

        return apply_humming_linear(
            layer,
            x,
            skip_bias_add=bias is None,
            layer_config=self.layer_config,
            compute_config=self.compute_config,
            locks=self.locks,
        )
