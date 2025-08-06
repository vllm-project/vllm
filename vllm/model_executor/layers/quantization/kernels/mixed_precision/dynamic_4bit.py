# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import torch

from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.platforms import CpuArchEnum, current_platform
from vllm.scalar_type import scalar_types

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig


class Dynamic4bitLinearKernel(MPLinearKernel):
    SUPPORTED_QUANT_TYPES = [scalar_types.int4]

    @classmethod
    def get_min_capability(cls) -> int:
        return 1

    @classmethod
    def can_implement(cls,
                      c: MPLinearLayerConfig) -> tuple[bool, Optional[str]]:
        if not current_platform.is_cpu():
            return False, "Only CPU is supported"
        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return False, f"Unsupported quant type {c.weight_type}"
        if current_platform.get_cpu_architecture(
        ) == CpuArchEnum.ARM and c.act_type not in [
                torch.float32,
        ]:
            return False, "Dynamic4bitLinearKernel on Arm requires"\
                " Float32 activations"
        if c.full_weight_shape[0] % c.group_size != 0:
            return False, f"Group size ({c.group_size}) does not evenly divide"\
                " the number of input features "\
                f"({c.full_weight_shape[0]})"
        if current_platform.get_cpu_architecture() == CpuArchEnum.ARM:
            try:
                # Attempt to retrieve the operation
                _ = torch.ops.aten._dyn_quant_matmul_4bit
            except AttributeError:
                return False, f"PyTorch {torch.__version__} does not support"\
                    " _dyn_quant_matmul_4bit. Install a newer version"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module):
        c = self.config
        packed_weight = getattr(layer, self.w_q_name)
        packed_weight = packed_weight.add(8)
        uint8_packed = (packed_weight[::, 1::2] << 4
                        | packed_weight[::, ::2]).to(torch.uint8)

        scales = getattr(layer, self.w_s_name)
        block_size = c.group_size

        # Handle scaling factors for partitioned weights
        if block_size == c.partition_weight_shape[0]:
            scales = scales.to(
                torch.float32
            )  # Float32 & Bfloat16 variants requires float32 scales
            scales = scales.view(-1, 1)  # Channel-wise scales
            if layer.bias is not None:
                layer.bias = layer.bias.to(
                    torch.float32
                )  # Float32 & Bfloat16 variants requires float32 bias
        else:
            # KleidiAI kernel requires bfloat16 scales with groupwise scheme
            scales = scales.to(torch.bfloat16)

        # Repack weights as per kernel requirement
        w = torch.ops.aten._dyn_quant_pack_4bit_weight(
            uint8_packed, scales, layer.bias, block_size,
            c.partition_weight_shape[0], c.partition_weight_shape[1])
        replace_parameter(layer, self.w_q_name,
                          torch.nn.Parameter(w, requires_grad=False))
        setattr(layer, self.w_s_name, None)

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        c = self.config
        x_2d = x.reshape(-1, x.shape[-1])
        out_shape = x.shape[:-1] + (c.partition_weight_shape[1], )

        w_q = getattr(layer, self.w_q_name)
        output = torch.ops.aten._dyn_quant_matmul_4bit(
            x_2d, w_q, c.group_size, c.partition_weight_shape[0],
            c.partition_weight_shape[1])
        return output.reshape(out_shape)
