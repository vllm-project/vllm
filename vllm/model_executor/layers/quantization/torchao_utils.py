# SPDX-License-Identifier: Apache-2.0
"""
Common utilities for torchao.
"""

from typing import Any, Dict, Set

import torch


def torchao_quantize_param_data(param: torch.Tensor,
                                torchao_config: Any) -> torch.nn.Parameter:
    """Quantize a Tensor with torchao quantization specified by torchao_config

    Args:
       `param`: weight parameter of the linear module
       `torchao_config`: type of quantization and their arguments we want to
        use to quantize the Tensor
    """
    try:
        from torchao.core.config import AOBaseConfig
        from torchao.quantization import quantize_
    except ImportError as err:
        raise ImportError(
            "torchao required for this quantization method, "
            "Please install torchao with `pip install torchao>=0.10.0`."
        ) from err
    assert isinstance(torchao_config, AOBaseConfig)
    dummy_linear = torch.nn.Linear(param.shape[1], param.shape[0], bias=False)
    dummy_linear.weight = param
    quantize_(dummy_linear, torchao_config)
    return dummy_linear.weight


def apply_torchao_config_(
    self: torch.nn.Module,
    params_dict: Dict[str, torch.Tensor],
    param_suffixes: Set[str],
) -> None:
    """A util function used for quantizing the weight parameters after they are
       loaded if self.torchao_config is specified

    Args:
      `self`: the model we want to quantize
      `params_dict`: dictionary mapping from param_name to the parameter Tensor
      `param_suffixes`: a set of suffixes, we'll quantize the Tensor matching
       these suffixes

    Returns:
       None, the `params_dict` is modified inplace and the weights of `self`
       model are quantized
    """
    if self.torchao_config:
        for param_suffix in param_suffixes:
            for name in params_dict:
                param = params_dict[name]
                if param_suffix in name and param.ndim == 2:
                    params_dict[name] = torchao_quantize_param_data(
                        param, self.torchao_config)
        self.load_state_dict(params_dict, assign=True)
