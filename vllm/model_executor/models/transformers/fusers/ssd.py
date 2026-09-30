# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SSD fuser: the module that dispatches to the Transformers SSD interface."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from torch import fx, nn

from vllm.model_executor.models.transformers.fusers.attention import interface_call
from vllm.model_executor.models.transformers.fusers.base import BaseFuser

if TYPE_CHECKING:
    from vllm.config import VllmConfig

VLLM_LINEAR_ATTN_IMPL = "vllm"
SSD_INTERFACE = "ALL_SSD_FUNCTIONS"


@dataclass
class SSDFuser(BaseFuser):
    """A Mamba2 mixer that dispatches its conv + scan through the SSD interface."""

    redefines_forward: ClassVar[bool] = False

    source_cls: str
    """Class of the HF module that dispatches (for logging)."""

    def info(self, name: str) -> str:
        return f"Found: {name} ({self.source_cls}) -> SSD interface"

    @classmethod
    def match(cls, graph: fx.Graph | None, module: nn.Module) -> "SSDFuser | None":
        if interface_call(type(module).forward, SSD_INTERFACE) is None:
            return None
        return cls(source_cls=type(module).__name__)

    def validate(self, module: nn.Module, vllm_config: "VllmConfig") -> bool:
        """Whether `module` will actually dispatch to vLLM."""
        config = getattr(module, "config", None)
        impl = getattr(config, "_linear_attn_implementation", None)
        return impl == VLLM_LINEAR_ATTN_IMPL

    def fuse(
        self, module: nn.Module, prefix: str, vllm_config: "VllmConfig"
    ) -> nn.Module:
        if vllm_config.parallel_config.tensor_parallel_size > 1:
            raise NotImplementedError(
                f"{self.source_cls} ({prefix}) does not support tensor parallelism "
                "in the Transformers modeling backend yet."
            )
        return module

    def layer_index(self, module: nn.Module) -> int | None:
        """The layer `module` computes the SSD for, if it declares one."""
        layer_idx = getattr(module, "layer_idx", None)
        return layer_idx if isinstance(layer_idx, int) else None
