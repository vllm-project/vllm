"""
Common utilities for torchao.
"""

from typing import Dict, Set

import torch


def torchao_quantize_param_data(param: torch.Tensor, torchao_config: str):
    """Quantize a Tensor with torchao quantization specified by torchao_config

    Args:
       `param`: weight parameter of the linear module
       `torchao_config`: type of quantization and their arguments we want to use to
        quantize the Tensor, e.g. int4wo-128 means int4 weight only quantization with group_size
        128
    """
    # Lazy import to suppress some warnings
    from torchao.quantization import (
        float8_dynamic_activation_float8_weight,
        int4_weight_only,
        int8_dynamic_activation_int8_weight,
        int8_weight_only,
        quantize_,
    )
    from torchao.quantization.observer import PerRow, PerTensor

    dummy_linear = torch.nn.Linear(param.shape[1], param.shape[0], bias=False)
    dummy_linear.weight = param
    if "int8wo" in torchao_config:
        quantize_(dummy_linear, int8_weight_only())
    elif "int8dq" in torchao_config:
        quantize_(dummy_linear, int8_dynamic_activation_int8_weight())
    elif "int4wo" in torchao_config:
        group_size = int(torchao_config.split("-")[-1])
        assert group_size in [
            32,
            64,
            128,
            256,
        ], f"int4wo groupsize needs to be one of [32, 64, 128, 256] but got {group_size}"
        quantize_(dummy_linear, int4_weight_only(group_size=group_size))
    elif "fp8wo" in torchao_config:
        from torchao.quantization import float8_weight_only

        # this requires newer hardware
        # [rank0]: AssertionError: fp8e4nv data type is not supported on CUDA arch < 89
        quantize_(dummy_linear, float8_weight_only())
    elif "fp8dq" in torchao_config:
        granularity = torchao_config.split("-")[-1]
        GRANULARITY_MAP = {
            "per_row": PerRow(),
            "per_tensor": PerTensor(),
        }
        assert (
            granularity in GRANULARITY_MAP
        ), f"Supported granularity are: {GRANULARITY_MAP.keys()}, got {granularity}"
        quantize_(
            dummy_linear,
            float8_dynamic_activation_float8_weight(
                granularity=GRANULARITY_MAP[granularity]
            ),
        )

    return dummy_linear.weight


def apply_torchao_config_(
    self: torch.nn.Module,
    params_dict: Dict[str, torch.Tensor],
    param_suffixes: Set[str],
) -> None:
    """A util function used for quantizing the weight parameters after they are loaded if
       self.torchao_config is specified

    Args:
      `self`: the model we want to quantize
      `params_dict`: dictionary mapping from param_name to the parameter Tensor
      `param_suffixes`: a set of suffixes, we'll quantize the Tensor matching these suffixes

    Returns:
       None, the `params_dict` is modified inplace and the weights of `self` model are quantized
    """
    if self.torchao_config:
        for param_suffix in param_suffixes:
            for name in params_dict:
                param = params_dict[name]
                if param_suffix in name and param.ndim == 2:
                    params_dict[name] = torchao_quantize_param_data(
                        param, self.torchao_config
                    )
        self.load_state_dict(params_dict, assign=True)
