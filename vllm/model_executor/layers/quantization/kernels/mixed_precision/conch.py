# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from importlib.util import find_spec
from typing import Final

import torch

from vllm.model_executor.parameter import BasevLLMParameter, permute_param_layout_
from vllm.scalar_type import scalar_types

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

_CONCH_SUPPORTED_WEIGHT_TYPES: Final = [
    scalar_types.uint4,
    scalar_types.uint8,
    scalar_types.uint4b8,
    scalar_types.uint8b128,
]
_CONCH_SUPPORTED_GROUP_SIZES: Final = [-1, 128]


class ConchLinearKernel(MPLinearKernel):
    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if c.weight_type not in _CONCH_SUPPORTED_WEIGHT_TYPES:
            error_msg = (
                f"Weight type ({c.weight_type}) not supported by "
                "ConchLinearKernel, supported types are: "
                f"{_CONCH_SUPPORTED_WEIGHT_TYPES}"
            )
            return False, error_msg

        if c.group_size not in _CONCH_SUPPORTED_GROUP_SIZES:
            error_msg = (
                f"Group size ({c.group_size}) not supported by "
                "ConchLinearKernel, supported group sizes are: "
                f"{_CONCH_SUPPORTED_GROUP_SIZES}"
            )
            return False, error_msg

        if find_spec("conch") is None:
            error_msg = (
                "conch-triton-kernels is not installed, please "
                "install it via `pip install conch-triton-kernels` "
                "and try again!"
            )
            return False, error_msg

        return True, None

    # note assumes that
    #  `weight_packed` is: {input_dim = 0, output_dim = 1, packed_dim = 0}
    #  `weight_scale` is: {input_dim = 0, output_dim = 1}
    #  `weight_zero_point` is: {input_dim = 1, output_dim = 0, packed_dim = 0}
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

        def transform_w_zp(x):
            # Zero points are stored PACKED as [N//pack_factor, K//G]
            # The Conch kernel expects UNPACKED zeros: [K//G, N]
            # We need to unpack and reorder
            assert isinstance(x, BasevLLMParameter)
            packed = x.data  # shape: [N//pack_factor, K//G], dtype: int32

            # Determine packing based on weight bit width
            size_bits = self.config.weight_type.size_bits
            pack_factor = 32 // size_bits  # 8 for 4-bit, 4 for 8-bit
            mask = (1 << size_bits) - 1  # 0xF for 4-bit, 0xFF for 8-bit

            n_packed, k_groups = packed.shape
            n_full = n_packed * pack_factor

            # Unpack using vectorized bitwise ops
            # shifts = [0, size_bits, 2*size_bits, ...] for each packed position
            shifts = torch.arange(
                0, 32, size_bits, dtype=torch.int32, device=packed.device
            )
            # packed: [N//pack_factor, K//G] -> [N//pack_factor, K//G, 1]
            # shifts: [pack_factor] -> [1, 1, pack_factor]
            # Result: [N//pack_factor, K//G, pack_factor]
            unpacked = (packed.unsqueeze(-1) >> shifts) & mask

            # Permute to [K//G, N//pack_factor, pack_factor] then reshape to [K//G, N]
            unpacked = unpacked.permute(1, 0, 2).reshape(k_groups, n_full)

            x.data = unpacked.to(torch.uint8).contiguous()

            # Update metadata - zeros are no longer packed
            if hasattr(x, "_input_dim"):
                x._input_dim = 0
            if hasattr(x, "_output_dim"):
                x._output_dim = 1
            if hasattr(x, "_packed_factor"):
                x._packed_factor = 1
            return x

        self._transform_param(layer, self.w_q_name, transform_w_q)
        self._transform_param(layer, self.w_s_name, transform_w_s)
        if self.config.zero_points:
            self._transform_param(layer, self.w_zp_name, transform_w_zp)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
