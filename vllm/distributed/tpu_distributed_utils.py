# SPDX-License-Identifier: Apache-2.0
from collections import OrderedDict

import torch
import torch_xla.distributed.spmd as xs

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)

logger = init_logger(__name__)


def wrap_column_parallel_linear(layer: torch.nn.Module,
                                mesh) -> torch.nn.Module:
    assert isinstance(layer, ColumnParallelLinear)
    xs.mark_sharding(layer.weight, mesh, ('x', None))
    logger.info(f"Applied column-parallel sharding to {layer}")
    return layer


def wrap_row_parallel_linear(layer: torch.nn.Module, mesh) -> torch.nn.Module:
    assert isinstance(
        layer,
        RowParallelLinear)  # Fixed: was checking for ColumnParallelLinear
    xs.mark_sharding(layer.weight, mesh, (None, 'x'))
    logger.info(f"Applied row-parallel sharding to {layer}")
    return layer


MODULE_TYPE_TO_WRAPPING_FUNC = OrderedDict([
    (ColumnParallelLinear, wrap_column_parallel_linear),
    (RowParallelLinear, wrap_row_parallel_linear),
])


def shard_model(model: torch.nn.Module, mesh: "xs.Mesh") -> None:
    """
    Recursively check a PyTorch model and apply appropriate sharding based on 
    the MODULE_TYPE_TO_WRAPPING_FUNC mapping.
    
    Args:
        model: The PyTorch model to process
        mesh: An XLA SPMD mesh object used for sharding
    """

    def _process_module(module, parent=None, name=None):
        for module_type, wrapping_func in MODULE_TYPE_TO_WRAPPING_FUNC.items():
            if isinstance(module, module_type):
                wrapped_module = wrapping_func(module, mesh)

                if parent is not None and name is not None:
                    setattr(parent, name, wrapped_module)

                module = wrapped_module
                break

        for child_name, child_module in list(module.named_children()):
            _process_module(child_module, module, child_name)

    _process_module(model)
