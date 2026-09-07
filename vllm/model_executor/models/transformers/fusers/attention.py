# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention fuser: the module that dispatches to the attention interface."""

import ast
import inspect
import textwrap
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, ClassVar

from torch import fx, nn

from vllm.model_executor.models.transformers.fusers.base import BaseFuser

if TYPE_CHECKING:
    from vllm.config import VllmConfig

VLLM_ATTN_IMPL = "vllm"
VLLM_MLA_ATTN_IMPL = "vllm_mla"


def _is_interface_lookup(node: ast.expr | None) -> bool:
    """Whether `node` reads an entry out of `ALL_ATTENTION_FUNCTIONS`."""
    # ALL_ATTENTION_FUNCTIONS.get_interface(...) or ALL_ATTENTION_FUNCTIONS[...]
    if isinstance(node, ast.Call):
        node = node.func
    if not isinstance(node, (ast.Attribute, ast.Subscript)):
        return False
    return (
        isinstance(node.value, ast.Name) and node.value.id == "ALL_ATTENTION_FUNCTIONS"
    )


@cache
def interface_call(cls: type[nn.Module]) -> ast.Call | None:
    """The attention interface call in `cls.forward`, if it makes exactly one."""
    try:
        source = inspect.getsource(inspect.unwrap(cls.forward))
        tree = ast.parse(textwrap.dedent(source))
    except (AttributeError, OSError, SyntaxError, TypeError):
        return None

    # Find the names of the local variables that read from the interface lookup
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        if not _is_interface_lookup(node.value):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names.update(t.id for t in targets if isinstance(t, ast.Name))
    # Find the calls to those names, and return the one if there is exactly one
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in names
    ]
    return calls[0] if len(calls) == 1 else None


def _resolve(node: ast.expr, module: nn.Module) -> object:
    """The value of `node` on `module`, for literals and `self.<attr>`."""
    if isinstance(node, ast.Constant):
        return node.value
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    ):
        return getattr(module, node.attr, None)
    return None


@dataclass
class AttentionFuser(BaseFuser):
    """A module that dispatches through the Transformers attention interface."""

    redefines_forward: ClassVar[bool] = False

    source_cls: str
    """Class of the HF module that dispatches (for logging)."""
    scale_expr: ast.expr | None
    """Source of the `scaling=` the module hands the interface, if it hands one."""

    def info(self, name: str) -> str:
        return f"Found: {name} ({self.source_cls}) -> attention interface"

    @classmethod
    def match(
        cls, graph: fx.Graph | None, module: nn.Module
    ) -> "AttentionFuser | None":
        if (call := interface_call(type(module))) is None:
            return None
        scaling = [kw.value for kw in call.keywords if kw.arg == "scaling"]
        scale_expr = scaling[0] if len(scaling) == 1 else None
        return cls(source_cls=type(module).__name__, scale_expr=scale_expr)

    def validate(self, module: nn.Module, vllm_config: "VllmConfig") -> bool:
        """Whether `module` will actually dispatch to vLLM."""
        config = getattr(module, "config", None)
        # Only patched in the text config, this excludes attention based mm encoders
        vllm_attn_impls = {VLLM_ATTN_IMPL, VLLM_MLA_ATTN_IMPL}
        return getattr(config, "_attn_implementation", None) in vllm_attn_impls

    def fuse(
        self, module: nn.Module, prefix: str, vllm_config: "VllmConfig"
    ) -> nn.Module:
        return module

    def layer_index(self, module: nn.Module) -> int | None:
        """The layer `module` computes attention for, if it declares one."""
        layer_idx = getattr(module, "layer_idx", None)
        return layer_idx if isinstance(layer_idx, int) else None

    def scale(self, module: nn.Module) -> float | None:
        """The softmax scale `module` passes to the interface, or `None`."""
        if self.scale_expr is None:
            return None
        scale = _resolve(self.scale_expr, module)
        if not isinstance(scale, (int, float)) or isinstance(scale, bool):
            expression = ast.unparse(self.scale_expr)
            raise ValueError(
                f"Cannot resolve attention scaling expression {expression!r} in "
                f"{type(module).__name__}."
            )
        return float(scale)
