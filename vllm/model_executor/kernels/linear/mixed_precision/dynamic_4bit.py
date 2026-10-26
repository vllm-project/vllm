# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    unpack_quantized_values_into_int32,
)
from vllm.platforms import CpuArchEnum, current_platform
from vllm.scalar_type import scalar_types

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig


# This implementation is for the KleidiAI-accelerated w4a8int quantization
# scheme on Arm CPUs:
# torch.ops.aten._dyn_quant_matmul_4bit performs dynamic quantized matmul
#   it takes:
#       - int4 weights packed along with bias/scales by
#         torch.ops.aten._dyn_quant_pack_4bit_weight
#       - float32/bfloat16 activations
#   then it leverages KleidiAI ukernels that:
#       - dynamically quantize the activations to int8
#       - unpack the int4 weights to int8
#       - perform int8 x int8 -> int32 matmul
#       - dequantize the int32 output to float32/bfloat16 outputs
class Dynamic4bitLinearKernel(MPLinearKernel):
    SUPPORTED_QUANT_TYPES = [scalar_types.int4, scalar_types.uint4b8]

    @classmethod
    def get_min_capability(cls) -> int:
        return 1

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_cpu():
            return False, "Only CPU is supported"
        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return False, f"Unsupported quant type {c.weight_type}"
        cpu_arch = current_platform.get_cpu_architecture()
        if c.weight_type == scalar_types.uint4b8 and cpu_arch != CpuArchEnum.ARM:
            return False, "uint4b8 packed weights are only supported on Arm"
        if c.weight_type == scalar_types.uint4b8 and c.zero_points:
            return False, "Zero points are not supported for uint4b8"
        if cpu_arch == CpuArchEnum.ARM and c.act_type not in [
            torch.float32,
            torch.bfloat16,
            torch.float16,
        ]:
            return (
                False,
                "Dynamic4bitLinearKernel on Arm requires Float32 or"
                " BFloat16 or Float16 activations",
            )
        if c.group_size <= 0 or c.partition_weight_shape[0] % c.group_size != 0:
            return (
                False,
                f"Group size ({c.group_size}) does not evenly divide"
                " the number of input features "
                f"({c.partition_weight_shape[0]})",
            )
        if c.group_size != c.partition_weight_shape[0] and c.group_size % 32 != 0:
            return False, "Group size must be a multiple of 32"
        if cpu_arch == CpuArchEnum.ARM:
            try:
                # Attempt to retrieve the operation
                _ = torch.ops.aten._dyn_quant_pack_4bit_weight
                _ = torch.ops.aten._dyn_quant_matmul_4bit
            except AttributeError:
                return (
                    False,
                    f"PyTorch {torch.__version__} does not support"
                    " _dyn_quant_matmul_4bit. Install a newer version",
                )
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module):
        c = self.config
        packed_weight = getattr(layer, self.w_q_name)
        if c.weight_type == scalar_types.uint4b8:
            # Compressed-tensors W4A16 checkpoints store eight offset-binary
            # uint4 values in each int32 along the input dimension. Decode the
            # nibbles to [0, 15] before converting to the byte-packed layout
            # expected by the PyTorch/KleidiAI operator.
            packed_weight = unpack_quantized_values_into_int32(
                packed_weight, c.weight_type, packed_dim=1
            )
        else:
            # W4A8 checkpoints provide unpacked signed int4 values in int8
            # storage. Convert [-8, 7] to the same offset-binary [0, 15]
            # representation.
            packed_weight = packed_weight.add(8)

        uint8_packed = (packed_weight[:, 1::2] << 4 | packed_weight[:, ::2]).to(
            torch.uint8
        )

        scales = getattr(layer, self.w_s_name)
        block_size = c.group_size

        # Handle scaling factors for partitioned weights
        if block_size == c.partition_weight_shape[0]:
            scales = scales.to(
                torch.float32
            )  # Float32 & Bfloat16 variants requires float32 scales
            scales = scales.view(-1, 1)  # Channel-wise scales
            if layer.bias is not None:
                # Float32 & Bfloat16 variants requires float32 bias
                replace_parameter(
                    layer,
                    "bias",
                    torch.nn.Parameter(
                        layer.bias.to(torch.float32), requires_grad=False
                    ),
                )
        else:
            # KleidiAI kernel requires bfloat16 scales with groupwise scheme
            scales = scales.to(torch.bfloat16)

        # Repack weights as per kernel requirement
        w = torch.ops.aten._dyn_quant_pack_4bit_weight(
            uint8_packed,
            scales,
            layer.bias,
            block_size,
            c.partition_weight_shape[0],
            c.partition_weight_shape[1],
        )
        replace_parameter(
            layer, self.w_q_name, torch.nn.Parameter(w, requires_grad=False)
        )
        setattr(layer, self.w_s_name, None)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # PyTorch / KleidiAI kernels natively support the following configs:
        # - channelwise with bfloat16 / float32 activations
        # - groupwise with float32 activations
        # To support:
        # - groupwise with bfloat16/float16 activations: we need to upcast
        #   activations to float32 before matmul and downcast back to bfloat16/float16
        # - channelwise with float16 activations, we need to upcast activations to
        #   float32 before matmul and downcast back to float16
        # Note: these activations will be dynamically quantized to int8 by the kernel.

        c = self.config
        is_groupwise = c.group_size != c.partition_weight_shape[0]
        # dtype of activations before they get dynamically quantized to int8
        original_pre_quant_act_dtype = x.dtype
        pre_quant_act_dtype = original_pre_quant_act_dtype
        if (
            is_groupwise and pre_quant_act_dtype == torch.bfloat16
        ) or pre_quant_act_dtype == torch.float16:
            pre_quant_act_dtype = torch.float32

        x_2d = x.reshape(-1, x.shape[-1])
        if pre_quant_act_dtype != original_pre_quant_act_dtype:
            x_2d = x_2d.to(pre_quant_act_dtype)

        out_shape = x.shape[:-1] + (c.partition_weight_shape[1],)

        w_q = getattr(layer, self.w_q_name)
        output = torch.ops.aten._dyn_quant_matmul_4bit(
            x_2d,
            w_q,
            c.group_size,
            c.partition_weight_shape[0],
            c.partition_weight_shape[1],
        ).reshape(out_shape)

        if pre_quant_act_dtype != original_pre_quant_act_dtype:
            output = output.to(original_pre_quant_act_dtype)
        return output
