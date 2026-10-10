# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch
import torch.nn.functional as F
from torch.fx.experimental.symbolic_shapes import guard_or_false

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32,
)
from vllm.model_executor.parameter import BasevLLMParameter, permute_param_layout_
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.utils.math_utils import round_up

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig


class ExllamaLinearKernel(MPLinearKernel):
    SUPPORTED_QUANT_TYPES = [scalar_types.uint4b8, scalar_types.uint8b128]
    # In theory supports `scalar_types.uint2b2, scalar_types.uint3b4` too but
    # currently untested so not added to the list

    @classmethod
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_cuda_alike():
            return (
                False,
                "Exllama is only supported on CUDA and ROCm",
            )

        if c.partition_weight_shape[1] % (32 // c.weight_type.size_bits) != 0:
            return (
                False,
                "Output features must be a multiple of the pack "
                "factor (32 / num_bits) so that we can correctly "
                "pack the zero points",
            )

        if c.act_type != torch.float16:
            return False, "Exllama only supports float16 activations"

        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return (
                False,
                f"Quant type ({c.weight_type}) not supported by "
                "Exllama, supported types are: "
                f"{cls.SUPPORTED_QUANT_TYPES}",
            )

        if c.group_size <= 0:
            return (
                False,
                f"Group size ({c.group_size}) must be positive, "
                "Exllama does not support channelwise quantization",
            )

        if c.full_weight_shape[0] % c.group_size != 0:
            return (
                False,
                f"Group size ({c.group_size}) does not evenly divide"
                " the number of input features "
                f"({c.full_weight_shape[0]})",
            )

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module):
        c = self.config
        out_features = c.partition_weight_shape[1]

        # The exllama kernels walk each 128-row block of K in steps of 32 rows
        # and assume the next quantization group starts group_size rows after the
        # block start, which only holds for group sizes 32, 64, multiples of 128
        # and a single group. Other group sizes use the plain GPTQ kernels. These take
        # the weight unshuffled, and the small-batch one writes whole blocks of
        # 128 output features without a bounds check.
        self.use_exllama = (
            c.group_size in (32, 64)
            or c.group_size % 128 == 0
            or c.group_size == c.partition_weight_shape[0]
        )
        pad_n = 0 if self.use_exllama else round_up(out_features, 128) - out_features

        # For Exllama, we need to set a zero-point tensor if there is not one
        if not c.zero_points:
            self.w_zp_name = "qzeros"
            device = getattr(layer, self.w_q_name).device
            groups = c.partition_weight_shape[0] // c.group_size

            if c.weight_type.has_bias():
                # if the type has a bias we have to create a zeros tensor that
                # contains the bias values repeated for each group (-1 due to
                # a bug in the original GPTQ checkpoint format leading to
                # exllama kernel adding 1 to the zero points during inference)
                # Documentation of the bug can be found here:
                #  https://garden.danieldk.eu/GPTQ-Checkpoint-Format
                zeros = torch.full(
                    (groups, out_features),
                    c.weight_type.bias - 1,
                    dtype=torch.int32,
                    device=device,
                )
            else:
                raise NotImplementedError(
                    "A 0 zero-point is not supported by Exllama due to "
                    "a bug in the original GPTQ checkpoint format leading to "
                    "exllama kernel adding 1 to the zero points during "
                    "inference"
                )
            zeros = pack_quantized_values_into_int32(zeros, c.weight_type, packed_dim=1)
            setattr(
                layer, self.w_zp_name, torch.nn.Parameter(zeros, requires_grad=False)
            )

        def transform_w_q(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)
            x_cont = F.pad(x.data, (0, pad_n)).contiguous()
            if self.use_exllama:
                ops.gptq_shuffle(x_cont, c.weight_type.size_bits)
            return x_cont

        def transform_w_s(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1)
            x.data = F.pad(x.data, (0, pad_n)).contiguous()
            return x.to(dtype=c.act_type)

        def transform_w_zp(x):
            return F.pad(x.data, (0, pad_n // (32 // c.weight_type.size_bits)))

        # Repack weights and scales for Machete
        self._transform_param(layer, self.w_q_name, transform_w_q)
        self._transform_param(layer, self.w_s_name, transform_w_s)
        self._transform_param(layer, self.w_zp_name, transform_w_zp)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = self.config

        x_2d = x.reshape(-1, x.shape[-1])
        out_shape = x.shape[:-1] + (c.partition_weight_shape[1],)

        w_q, w_s, w_zp = self._get_weight_params(layer)
        # gptq_gemm supports GPTQv2 format by passing use_v2_format=True.
        # However, the MPLinearLayerConfig doesn't contain format info.
        # So hardcode GPTQv1 format here, to keep its behavior unchanged.
        use_v2_format = False

        assert w_zp is not None, "Zero points are required by Exllama"
        # The plain small-batch kernel cannot be launched with no rows.
        if not self.use_exllama and guard_or_false(x_2d.shape[0] == 0):
            return x.new_empty(out_shape)
        output = ops.gptq_gemm(
            x_2d,
            w_q,
            w_zp,
            w_s,
            self.use_exllama,
            use_v2_format,
            c.weight_type.size_bits,
        )
        if not self.use_exllama:
            output = output[:, : c.partition_weight_shape[1]].contiguous()

        if bias is not None:
            output.add_(bias)
        return output.reshape(out_shape)
