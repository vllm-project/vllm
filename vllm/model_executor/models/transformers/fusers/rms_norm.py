# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RMSNorm fuser: detect the norm structurally and swap in vLLM's fused RMSNorm."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import fx, nn

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.distributed.parallel_state import model_parallel_is_initialized
from vllm.distributed.utils import split_tensor_along_last_dim
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
from vllm.model_executor.models.transformers.fusers.base import BaseFuser
from vllm.model_executor.models.transformers.fx_utils import find_node, is_op, peel

if TYPE_CHECKING:
    from vllm.config.model import ModelConfig
    from vllm.model_executor.layers.quantization import QuantizationConfig


def _is_squared(node: object, x: fx.Node) -> bool:
    """`x**2`, `x.square()` or `x * x`, through any dtype casts."""
    node = peel(node)
    if is_op(node, "pow"):
        base, exp = node.args
        return peel(base) is x and exp == 2
    if is_op(node, "square"):
        return peel(node.args[0]) is x
    if is_op(node, "mul"):
        a, b = node.args
        return peel(a) is x and peel(b) is x
    return False


def _variance_eps(rsqrt: fx.Node, x: fx.Node) -> float | None:
    """eps from `rsqrt(mean(x**2, -1) + eps)`, or `None` if not that shape."""
    add = peel(rsqrt.args[0])
    if not is_op(add, "add"):
        return None
    consts = [a for a in add.args if isinstance(a, (int, float))]
    nodes = [a for a in add.args if isinstance(a, fx.Node)]
    if len(consts) != 1 or len(nodes) != 1:
        return None
    mean = peel(nodes[0])
    if not is_op(mean, "mean"):
        return None
    if not _is_squared(mean.args[0], x):
        return None
    return float(consts[0])


def _is_one_plus(node: object) -> bool:
    """`1 + weight` in either operand order (marks a zero-centered weight)."""
    node = peel(node)
    if not is_op(node, "add"):
        return False
    return any(isinstance(a, (int, float)) and a == 1 for a in node.args)


class TPAwareNormMixin(nn.Module):
    """Mixin for RMSNorms that reconstructs a TP-sharded input before normalizing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if model_parallel_is_initialized():
            self.tp_size = get_tensor_model_parallel_world_size()
            self.tp_rank = get_tensor_model_parallel_rank()
        else:
            self.tp_size, self.tp_rank = 1, 0

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.tp_size > 1 and x.shape[-1] < (full := self.weight.shape[0]):
            if x.shape[-1] * self.tp_size != full:
                raise ValueError(
                    f"Cannot gather norm of width {full}: a TP-sharded input of "
                    f"width {x.shape[-1]} does not tile it evenly across "
                    f"{self.tp_size} ranks (replicated or uneven sharding)."
                )
            x = tensor_model_parallel_all_gather(x.contiguous())
            x = super().forward(x)
            splits = split_tensor_along_last_dim(x, num_partitions=self.tp_size)
            return splits[self.tp_rank]
        return super().forward(x, residual)


class TPAwareRMSNorm(TPAwareNormMixin, RMSNorm):
    """`RMSNorm` that reconstructs a TP-sharded input before normalizing."""


class TPAwareGemmaRMSNorm(TPAwareNormMixin, GemmaRMSNorm):
    """`GemmaRMSNorm` that reconstructs a TP-sharded input before normalizing."""


@dataclass
class RMSNormFuser(BaseFuser):
    """Fuser for RMSNorm patterns, including Gemma-style zero-centered weights."""

    eps: float | None
    """`None` only for a fused `rms_norm` op with default eps; resolved in `fuse`."""
    has_weight: bool
    """Does the norm have a weight?"""
    zero_centered: bool
    """Gemma-style `(1 + weight)` scaling (weight initialised at zero)."""
    source_cls: str
    """Class name of the norm this was matched from (for logging)."""

    def info(self, name: str) -> str:
        norm = "GemmaRMSNorm" if self.zero_centered else "RMSNorm"
        return f"Fused: {name} ({self.source_cls}) -> {norm} (CustomOp)"

    @classmethod
    def match(cls, graph: fx.Graph, module: nn.Module) -> "RMSNormFuser | None":
        """Match a graph to the RMSNorm pattern, returning a fuser if found."""
        x = find_node(graph, lambda n: n.op == "placeholder")
        if x is None:
            return None
        # Handle native torch `rms_norm` op.
        fused = find_node(graph, lambda n: is_op(n, "rms_norm"))
        if fused is not None and fused.args and peel(fused.args[0]) is x:
            args, kwargs = fused.args, fused.kwargs
            weight = args[2] if len(args) > 2 else kwargs.get("weight")
            eps = args[3] if len(args) > 3 else kwargs.get("eps")
            return cls(
                eps=eps if isinstance(eps, (int, float)) else None,
                has_weight=isinstance(weight, fx.Node),
                zero_centered=False,
                source_cls=type(module).__name__,
            )
        # Handle explicit `x * rsqrt(mean(x**2, -1) + eps)` pattern.
        # The rsqrt over the mean-square variance is the spine of the norm.
        eps = rsqrt = None
        for node in graph.nodes:
            if is_op(node, "rsqrt") and (eps := _variance_eps(node, x)) is not None:
                rsqrt = node
                break
        if rsqrt is None:
            return None
        # The `x * rsqrt(...)` normalize multiply.
        normalize = find_node(
            graph, lambda n: is_op(n, "mul") and rsqrt in map(peel, n.args)
        )
        if normalize is None:
            return None
        # An optional later `weight * normalized` (or `(1 + weight) * normalized`).
        has_weight = zero_centered = False
        for node in graph.nodes:
            if not is_op(node, "mul") or node is normalize:
                continue
            operands = [peel(a) for a in node.args if isinstance(a, fx.Node)]
            if len(operands) == 2 and normalize in operands:
                weight = next(o for o in operands if o is not normalize)
                has_weight = True
                zero_centered = _is_one_plus(weight)
                break
        return cls(
            eps=eps,
            has_weight=has_weight,
            zero_centered=zero_centered,
            source_cls=type(module).__name__,
        )

    def validate(self, module: nn.Module, model_config: "ModelConfig") -> bool:
        return True

    def fuse(
        self,
        module: nn.Module,
        prefix: str,
        model_config: "ModelConfig",
        quant_config: "QuantizationConfig",
    ) -> nn.Module:
        """Fuse the matched RMSNorm pattern into a vLLM fused RMSNorm CustomOp."""
        weight = getattr(module, "weight", None)
        hidden_size = (
            weight.size(0) if weight is not None else model_config.get_hidden_size()
        )
        eps = self.eps
        if eps is None:
            # Could be `None` for native torch `rms_norm`. Match torch behaviour.
            dtype = weight.dtype if weight is not None else model_config.dtype
            eps = torch.finfo(dtype).eps
        if self.zero_centered:
            return TPAwareGemmaRMSNorm(hidden_size=hidden_size, eps=eps)
        has_weight = self.has_weight and weight is not None
        return TPAwareRMSNorm(
            hidden_size=hidden_size,
            eps=eps,
            has_weight=has_weight,
            dtype=weight.dtype if has_weight else None,
        )
