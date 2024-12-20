from typing import Any, Dict, List, Optional

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.utils import set_weight_attrs


def make_group_map(q_groups, num_qrows):
    gr = q_groups.tolist()
    group_map = []
    num_groups = len(gr) // 2

    for i in range(num_groups):
        bits = gr[i * 2]
        if i < num_groups - 1:
            qrows = gr[i * 2 + 3] - gr[i * 2 + 1]
        else:
            qrows = num_qrows - gr[i * 2 + 1]
        rows = qrows * 32 // bits
        for j in range(rows):
            group_map += [i]
            group_map += [rows - j]
    return torch.tensor(group_map, dtype=torch.short, device=q_groups.device)


class Exl2Config(QuantizationConfig):
    """Config class for Exl2."""

    def __repr__(self) -> str:
        return "Exl2Config()"

    @classmethod
    def get_name(cls) -> str:
        return "exl2"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half]

    @classmethod
    # Need to figure it out
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Exl2Config":
        return cls()

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["Exl2LinearMethod"]:
        if isinstance(layer, LinearBase):
            return Exl2LinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []

    def merge_weight(self) -> bool:
        return False

    def quant_vocab(self) -> List[bool]:
        return [False, True]

    def support_fused_moe(self) -> bool:
        return False

    def rope_style(self) -> Optional[bool]:
        return None


class Exl2LinearMethod(LinearMethodBase):
    """Linear method for Exl2.

    Args:
        quant_config: The Exl2 quantization config.
    """

    def __init__(self, quant_config: Exl2Config):
        self.quant_config = quant_config

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: List[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attr):
        # The shape of weight is unknown until load state dict
        # q_groups, q_invperm, q_scale, q_scale_max, q_weight, q_groups
        layer.exllama_state = 0
        qweight = torch.nn.parameter.UninitializedParameter(
            requires_grad=False)
        set_weight_attrs(qweight, {"output_dim": 1, "ignore_warning": True})
        layer.register_parameter("q_weight", qweight)
        qscale = torch.nn.parameter.UninitializedParameter(requires_grad=False)
        set_weight_attrs(
            qscale, {
                "output_dim": 1,
                "packed_dim": 1,
                "pack_factor": 8,
                "ignore_warning": True
            })
        layer.register_parameter("q_scale", qscale)
        for name in ["q_groups", "q_invperm", "q_scale_max"]:
            fake_weight = torch.nn.parameter.UninitializedParameter(
                requires_grad=False)
            set_weight_attrs(fake_weight, {"ignore_warning": True})
            layer.register_parameter(name, fake_weight)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        out_shape = x.shape[:-1] + (layer.q_weight.shape[-1], )
        reshaped_x = x.reshape(-1, x.shape[-1])

        if layer.exllama_state == 0:
            layer.q_scale_max /= 256
            layer.q_invperm = layer.q_invperm.short()
            if not hasattr(layer, 'q_perm'):
                layer.q_perm = torch.argsort(layer.q_invperm).to(torch.short)
            if not hasattr(layer, 'q_group_map'):
                layer.q_group_map = make_group_map(layer.q_groups,
                                                   layer.q_weight.shape[0])
            layer.q_matrix = ops.exl2_make_q_matrix(
                layer.q_weight,
                layer.q_perm,
                layer.q_invperm,
                layer.q_scale,
                layer.q_scale_max,
                layer.q_groups,
                layer.q_group_map,
            )
            layer.exllama_state = 1

        output = ops.exl2_gemm(reshaped_x, layer.q_matrix)

        if bias is not None:
            output.add_(bias)
        return output.reshape(out_shape)
