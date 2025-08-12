# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from importlib.util import find_spec
from typing import Final, Optional

import torch

from vllm.model_executor.parameter import (BasevLLMParameter,
                                           permute_param_layout_)
from vllm.scalar_type import scalar_types

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

_CONCH_SUPPORTED_WEIGHT_TYPES: Final = [
    scalar_types.uint4, scalar_types.uint8, scalar_types.uint4b8,
    scalar_types.uint8b128
]
_CONCH_SUPPORTED_GROUP_SIZES: Final = [-1, 128]


class ConchLinearKernel(MPLinearKernel):

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def can_implement(cls,
                      c: MPLinearLayerConfig) -> tuple[bool, Optional[str]]:
        if c.weight_type not in _CONCH_SUPPORTED_WEIGHT_TYPES:
            error_msg = f"Weight type ({c.weight_type}) not supported by "\
                        "ConchLinearKernel, supported types are: " \
                        f"{_CONCH_SUPPORTED_WEIGHT_TYPES}"
            return False, error_msg

        if c.group_size not in _CONCH_SUPPORTED_GROUP_SIZES:
            error_msg = f"Group size ({c.group_size}) not supported by "\
                        "ConchLinearKernel, supported group sizes are: " \
                        f"{_CONCH_SUPPORTED_GROUP_SIZES}"
            return False, error_msg

        if find_spec("conch") is None:
            error_msg = "conch-triton-kernels is not installed, please "\
                        "install it via `pip install conch-triton-kernels` "\
                        "and try again!"
            return False, error_msg

        return True, None

    # note assumes that
    #  `weight_packed` is: {input_dim = 0, output_dim = 1, packed_dim = 0}
    #  `weight_scale` is: {input_dim = 0, output_dim = 1}
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:

        def transform_w_q(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)
            x.data = x.data.contiguous()
            return x

        def transform_w_s(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1)
            x.data = x.data.contiguous()
            return x

        self._transform_param(layer, self.w_q_name, transform_w_q)
        self._transform_param(layer, self.w_s_name, transform_w_s)

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        from conch.ops.quantization.gemm import mixed_precision_gemm

        w_q, w_s, w_zp, _ = self._get_weight_params(layer)

        output = mixed_precision_gemm(
            x=x,
            w_q_packed=w_q.data,
            w_s=w_s.data,
            w_zp=w_zp.data if w_zp is not None else None,
            weight_size_bits=self.config.weight_type.size_bits,
            weight_bias=self.config.weight_type.bias,
            group_size=self.config.group_size,
        )

        if bias is not None:
            output.add_(bias)  # In-place add

        return output
