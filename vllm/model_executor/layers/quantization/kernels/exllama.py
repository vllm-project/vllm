from typing import Optional, Tuple

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32)
from vllm.model_executor.parameter import (BasevLLMParameter,
                                           permute_param_layout_)
from vllm.scalar_type import scalar_types

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig


class ExllamaLinearKernel(MPLinearKernel):
    SUPPORTED_QUANT_TYPES = [scalar_types.uint4b8, scalar_types.uint8b128]
    # In theory supports `scalar_types.uint2b2, scalar_types.uint3b4` too but
    # currently untested so not added to the list

    @classmethod
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def can_implement(cls,
                      c: MPLinearLayerConfig) -> Tuple[bool, Optional[str]]:
        if c.has_g_idx and\
            c.partition_weight_shape[0] != c.full_weight_shape[0]:
            return False, "Act reordering currently not supported by Exllama, "\
                          "when the input features are partitioned across "\
                          "devices"

        if c.partition_weight_shape[1] % (32 // c.weight_type.size_bits) != 0:
            return False, "Output features must be a multiple of the pack " \
                            "factor (32 / num_bits) so that we can correctly " \
                            "pack the zero points"

        if c.act_type != torch.float16:
            return False, "Exllama only supports float16 activations"

        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return False, f"Quant type ({c.weight_type}) not supported by "\
                           "Exllama, supported types are: "\
                           f"{cls.SUPPORTED_QUANT_TYPES}"

        if c.full_weight_shape[0] % c.group_size != 0:
            return False, f"Group size ({c.group_size}) does not evenly divide"\
                           " the number of input features "\
                           f"({c.full_weight_shape[0]})"

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module):
        c = self.config

        # For Exllama, we need to set a zero-point tensor if there is not one
        if not c.zero_points:
            self.w_zp_name = "qzeros"
            device = getattr(layer, self.w_q_name).device
            groups = c.partition_weight_shape[0] // c.group_size
            out_features = c.partition_weight_shape[1]

            if c.weight_type.has_bias():
                # if the type has a bias we have to create a zeros tensor that
                # contains the bias values repeated for each group (-1 due to
                # a bug in the original GPTQ checkpoint format leading to
                # exllama kernel adding 1 to the zero points during inference)
                # Documentation of the bug can be found here:
                #  https://garden.danieldk.eu/GPTQ-Checkpoint-Format
                zeros = torch.full((groups, out_features),
                                   c.weight_type.bias - 1,
                                   dtype=torch.int32,
                                   device=device)
            else:
                raise NotImplementedError(
                    "A 0 zero-point is not supported by Exllama due to "
                    "a bug in the original GPTQ checkpoint format leading to "
                    "exllama kernel adding 1 to the zero points during "
                    "inference")
            zeros = pack_quantized_values_into_int32(zeros,
                                                     c.weight_type,
                                                     packed_dim=1)
            setattr(layer, self.w_zp_name,
                    torch.nn.Parameter(zeros, requires_grad=False))

        if c.has_g_idx:

            def transform_w_g_idx(x):
                # Exllama wants the permutation array instead of the group
                # indices
                return torch.argsort(x).to(torch.int)

            self._transform_param(layer, self.w_gidx_name, transform_w_g_idx)
        else:
            self.w_gidx_name = "g_idx"
            empty_g_idx = torch.nn.Parameter(torch.empty((0, ),
                                                         dtype=torch.int,
                                                         device=device),
                                             requires_grad=False)
            setattr(layer, self.w_gidx_name, empty_g_idx)

        def transform_w_q(x):
            assert isinstance(x, BasevLLMParameter)
            assert self.w_gidx_name is not None
            g_idx = getattr(layer, self.w_gidx_name)

            permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)
            x_cont = x.data.contiguous()
            ops.gptq_shuffle(x_cont, g_idx, c.weight_type.size_bits)
            return x_cont

        def transform_w_s(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1)
            x.data = x.data.contiguous()
            return x.to(dtype=c.act_type)

        # Repack weights and scales for Machete
        self._transform_param(layer, self.w_q_name, transform_w_q)
        self._transform_param(layer, self.w_s_name, transform_w_s)

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        c = self.config

        x_2d = x.reshape(-1, x.shape[-1])
        out_shape = x.shape[:-1] + (c.partition_weight_shape[1], )

        w_q, w_s, w_zp, w_g_idx = self._get_weight_params(layer)

        assert w_zp is not None, "Zero points are required by Exllama"
        assert w_g_idx is not None, "Group index is required by Exllama"
        output = ops.gptq_gemm(x_2d, w_q, w_zp, w_s, w_g_idx, True,
                               c.weight_type.size_bits)

        if bias is not None:
            output.add_(bias)
        return output.reshape(out_shape)
