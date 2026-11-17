# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SSD fuser: a Mamba2 mixer whose conv and scan `kernelize` can replace."""

import ast
import inspect
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from torch import fx, nn

from vllm.model_executor.models.transformers.fusers.base import BaseFuser
from vllm.model_executor.models.transformers.linear_attention import (
    SSD_FORWARD,
    shard_mamba2_mixer,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def calls_function(forward: Callable, name: str) -> bool:
    """Whether `forward` calls the module-level function `name`."""
    try:
        source = inspect.getsource(inspect.unwrap(forward))
        tree = ast.parse(textwrap.dedent(source))
    except (AttributeError, OSError, SyntaxError, TypeError):
        return False
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
        for node in ast.walk(tree)
    )


@dataclass
class SSDFuser(BaseFuser):
    """A Mamba2 mixer that runs its causal conv and scan through the Transformers
    `causal_conv1d_forward` and `ssd_forward`, which vLLM replaces with
    `kernelize`. Its layers are sharded for tensor parallelism (see
    `shard_mamba2_mixer`)."""

    redefines_forward: ClassVar[bool] = False

    source_cls: str
    """Class of the HF mixer (for logging)."""

    def info(self, name: str) -> str:
        return f"Found: {name} ({self.source_cls}) -> {SSD_FORWARD}"

    @classmethod
    def match(cls, graph: fx.Graph | None, module: nn.Module) -> "SSDFuser | None":
        if not calls_function(type(module).forward, SSD_FORWARD):
            return None
        return cls(source_cls=type(module).__name__)

    def validate(self, module: nn.Module, vllm_config: "VllmConfig") -> bool:
        """Whether `kernelize` can replace the mixer's conv and scan, i.e. the
        mixer is decorated with `use_kernelized_func` (needs `kernels`)."""
        return SSD_FORWARD in getattr(module, "_kernel_funcs", {})

    def fuse(
        self, module: nn.Module, prefix: str, vllm_config: "VllmConfig"
    ) -> nn.Module:
        shard_mamba2_mixer(module, prefix, vllm_config)
        return module

    def layer_index(self, module: nn.Module) -> int | None:
        """The layer `module` computes the SSD for, if it declares one."""
        layer_idx = getattr(module, "layer_idx", None)
        return layer_idx if isinstance(layer_idx, int) else None
