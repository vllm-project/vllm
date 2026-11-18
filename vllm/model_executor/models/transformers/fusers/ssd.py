# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SSD fuser: a Mamba2 mixer whose conv and scan `kernelize` can replace."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from torch import fx, nn

from vllm.model_executor.models.transformers.fusers.base import BaseFuser
from vllm.model_executor.models.transformers.linear_attention import (
    CAUSAL_CONV1D_FORWARD,
    SSD_FORWARD,
    shard_mamba2_mixer,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


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
        kernel_funcs = getattr(module, "_kernel_funcs", {}).values()
        layer_names = {type(f).kernel_layer_name for f in kernel_funcs}
        if not {CAUSAL_CONV1D_FORWARD, SSD_FORWARD} <= layer_names:
            return None
        return cls(source_cls=type(module).__name__)

    def validate(self, module: nn.Module, vllm_config: "VllmConfig") -> bool:
        return True

    def fuse(
        self, module: nn.Module, prefix: str, vllm_config: "VllmConfig"
    ) -> nn.Module:
        shard_mamba2_mixer(module, prefix, vllm_config)
        return module

    def layer_index(self, module: nn.Module) -> int | None:
        """The layer `module` computes the SSD for, if it declares one."""
        layer_idx = getattr(module, "layer_idx", None)
        return layer_idx if isinstance(layer_idx, int) else None
