# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sink fuser: hand an attention module's learnable sink to vLLM's attention layer."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import fx, nn

from vllm.model_executor.model_loader.weight_utils import sharded_weight_loader
from vllm.model_executor.models.transformers.fusers.base import BaseFuser
from vllm.model_executor.models.transformers.fx_utils import find_node, is_leaf_call
from vllm.model_executor.utils import set_weight_attrs

if TYPE_CHECKING:
    from vllm.config import VllmConfig

SINK_KWARG = "s_aux"
"""Keyword every Transformers model passes its learnable per-head sink with."""


def _owner(module: nn.Module, name: str) -> tuple[nn.Module, str]:
    """The module `name` is an attribute of, and the attribute's own name."""
    parent, _, attr = name.rpartition(".")
    return module.get_submodule(parent), attr


@dataclass
class SinkFuser(BaseFuser):
    """Fuser for attention modules with a learnable per-head sink.

    Only the attention impl can fold a sink into the softmax denominator, so the
    parameter has to reach `Attention` when it is constructed. `match` reads it off
    the `s_aux` argument of the traced attention interface call, so neither the name
    the model gives the parameter nor the shape of the module around it matters.
    """

    source_cls: str
    """Class of the HF module holding the sink (for logging)."""
    sink_name: str
    """Name of the parameter the sink is loaded into."""

    def info(self, name: str) -> str:
        return (
            f"Fused: {name}.{self.sink_name} ({self.source_cls}) -> Attention (sinks)"
        )

    @classmethod
    def match(cls, graph: fx.Graph, module: nn.Module) -> "SinkFuser | None":
        leaf = find_node(graph, is_leaf_call)
        if leaf is None:
            return None
        sink = leaf.kwargs.get(SINK_KWARG)
        if not isinstance(sink, fx.Node) or sink.op != "get_attr":
            return None
        return cls(source_cls=type(module).__name__, sink_name=str(sink.target))

    def sink(self, module: nn.Module) -> nn.Parameter | None:
        """`module`'s sink parameter, or `None` if this instance has no sink.

        Models that apply sinks on only some layers leave it unset on the others
        (e.g. MiMo-V2-Flash, which has them on its sliding window layers only).
        """
        owner, name = _owner(module, self.sink_name)
        return getattr(owner, name, None)

    def validate(self, module: nn.Module, vllm_config: "VllmConfig") -> bool:
        return self.sink(module) is not None

    def fuse(
        self, module: nn.Module, prefix: str, vllm_config: "VllmConfig"
    ) -> nn.Module:
        """Materialize the sink as this rank's slice of the model's heads.

        `Attention` holds a reference to the tensor the checkpoint is loaded into,
        so the parameter is replaced here rather than in `Base.init_parameters`,
        which runs after the attention instances have been created.
        """
        sink = self.sink(module)
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        param = nn.Parameter(
            torch.empty(
                sink.numel() // tp_size,
                dtype=vllm_config.model_config.dtype,
                device=vllm_config.device_config.device,
            ),
            requires_grad=False,
        )
        set_weight_attrs(param, {"weight_loader": sharded_weight_loader(0)})
        owner, name = _owner(module, self.sink_name)
        setattr(owner, name, param)
        return module
