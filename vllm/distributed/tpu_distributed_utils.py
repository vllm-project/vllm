# SPDX-License-Identifier: Apache-2.0
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla.distributed.spmd as xs
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)

logger = init_logger(__name__)


class XlaQKVParallelLinear(nn.Module):

    def __init__(self,
                 qkv_linear: nn.Module,
                 mesh: Optional["xs.Mesh"] = None):
        super().__init__()
        assert isinstance(qkv_linear, QKVParallelLinear)
        assert qkv_linear.bias is None, "Bias is not supported for QKVLinear"
        assert qkv_linear.return_bias is False, \
            "Return bias is not supported for QKVLinear"
        assert qkv_linear.tp_size == 1, "TP > 1 is only supported under SPMD."
        self._load_and_shard_weight_from_qkv_linear(qkv_linear)
        if mesh is not None:
            self._shard_weight(mesh)

    def _shard_weight(self, mesh: "xs.Mesh"):
        self.q_weight = Parameter(self.q_weight.to('xla'), requires_grad=False)
        self.k_weight = Parameter(self.k_weight.to('xla'), requires_grad=False)
        self.v_weight = Parameter(self.v_weight.to('xla'), requires_grad=False)
        xs.mark_sharding(self.q_weight, mesh, ('x', None))
        xs.mark_sharding(self.k_weight, mesh, ('x', None))
        xs.mark_sharding(self.v_weight, mesh, ('x', None))

    def _load_and_shard_weight_from_qkv_linear(self, qkv_linear: nn.Module):
        q_proj_size, k_proj_size, _ = qkv_linear.output_sizes
        # The weight of qkv linear is a concatenation of q, k, and v weights
        # along the output dimension.
        qkv_weight = qkv_linear.weight.data
        q_weight = Parameter(qkv_weight[:q_proj_size], requires_grad=False)
        k_weight = Parameter(qkv_weight[q_proj_size:q_proj_size + k_proj_size],
                             requires_grad=False)
        v_weight = Parameter(qkv_weight[q_proj_size + k_proj_size:],
                             requires_grad=False)
        self.register_parameter("q_weight", q_weight)
        self.register_parameter("k_weight", k_weight)
        self.register_parameter("v_weight", v_weight)

    def forward(self, input):
        q_proj = F.linear(input, self.q_weight)
        k_proj = F.linear(input, self.k_weight)
        v_proj = F.linear(input, self.v_weight)
        # The q/k/v projections will be split outside of the QKVParallelLinear.
        # Because we are replacing XlaQKVParallelLinear with the
        # QKVParallelLinear, we need to concatenate q, k, and v projections to
        # match the output shape of the QKVParallelLinear implementation even if
        # it seems to be redundant.
        # The concat and the following split will be noop, and should be
        # optimized away by the compiler.
        qkv_proj = torch.cat([q_proj, k_proj, v_proj], dim=-1)
        return qkv_proj


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

    def _process_module(module, name=None, parent=None):
        for module_type, wrapping_func in MODULE_TYPE_TO_WRAPPING_FUNC.items():
            if isinstance(module, module_type):
                wrapped_module = wrapping_func(module, mesh)

                assert parent is not None and name is not None, (
                    "Top Level module is not expected to be wrapped.")
                if parent is not None and name is not None:
                    setattr(parent, name, wrapped_module)

                module = wrapped_module
                break

        for child_name, child_module in list(module.named_children()):
            _process_module(child_module, child_name, module)

    _process_module(model)
