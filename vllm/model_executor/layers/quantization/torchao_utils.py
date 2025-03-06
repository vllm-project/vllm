# SPDX-License-Identifier: Apache-2.0
"""
Common utilities for torchao.
"""

from typing import Any

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
