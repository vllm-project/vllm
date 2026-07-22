# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base classes for the Transformers backend fusers."""

import types
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from torch import fx, nn

from vllm.model_executor.models.utils import ShardId, maybe_prefix

if TYPE_CHECKING:
    from vllm.config.model import ModelConfig
    from vllm.model_executor.layers.quantization import QuantizationConfig


@dataclass
class BaseFuser(ABC):
    """A detected fusion and how to apply it.

    `match` analyses the module *class* once (cached, see `get_fuser`); `fuse`
    then applies the fusion to an instance in `recursive_replace`, returning the
    module to install in its place.
    """

    @abstractmethod
    def info(self, name: str) -> str:
        """A human-readable description of the fusion at `name`, for logging."""

    @classmethod
    @abstractmethod
    def match(cls, graph: fx.Graph, module: nn.Module) -> "BaseFuser | None":
        """Match the pattern in `graph`, returning a fuser if found."""

    @abstractmethod
    def validate(self, module: nn.Module, model_config: "ModelConfig") -> bool:
        """Whether this fuser can be applied to this `module` instance."""

    @abstractmethod
    def fuse(
        self,
        module: nn.Module,
        prefix: str,
        model_config: "ModelConfig",
        quant_config: "QuantizationConfig",
    ) -> nn.Module:
        """Apply the fusion to an already-validated `module`, returning the
        module to install in its place (mutated in place, or freshly built)."""

    def orig_to_new_stacked(self, prefix: str) -> dict[str, tuple[str, ShardId]]:
        """`WeightsMapper.orig_to_new_stacked` entries this fuser contributes
        (none unless it stacks weights)."""
        return {}

    @property
    def packed_modules_mapping(self) -> dict[str, list[str]]:
        """`packed_modules_mapping` entries this fuser contributes (none unless
        it stacks weights)."""
        return {}


@dataclass
class StackedFuser(BaseFuser):
    """A fuser that merges sibling projections into one stacked linear and
    rewrites the forward to call it.

    `match` and `update_forward` analyse the class once; `fuse` builds the merged
    submodule and binds the compiled forward on an instance in place, so it keeps
    its class and any attribute the fusion does not consume.
    """

    merged_name: ClassVar[str]
    """Attribute name of the merged module created by `update_attrs`."""
    merged_cls: ClassVar[str]
    """Name of the vLLM class the merged projection becomes (for logging)."""

    source_cls: str
    """Class of the HF module the fused projections belonged to (for logging)."""

    fused_forward: Callable = field(init=False, repr=False)
    """The compiled rewritten forward, set by `update_forward`."""

    def info(self, name: str) -> str:
        sources = " + ".join(shard for shard, _ in self.shards)
        return (
            f"Fused: {sources} ({name}: {self.source_cls}) -> "
            f"{self.merged_name} ({self.merged_cls})"
        )

    @property
    @abstractmethod
    def shards(self) -> list[tuple[str, ShardId]]:
        """Each projection's original name and its shard id in the merged module.

        Source for both `orig_to_new_stacked` and `packed_modules_mapping`."""

    def orig_to_new_stacked(self, prefix: str) -> dict[str, tuple[str, ShardId]]:
        """`WeightsMapper.orig_to_new_stacked` entries for one fused instance.

        Maps each checkpoint name to `(merged_name, shard_id)`, keyed by qualname
        so only this exact layer is remapped, never a same-named projection
        elsewhere (e.g. an unfused MoE expert's `gate_proj`)."""
        merged = maybe_prefix(prefix, self.merged_name)
        return {
            maybe_prefix(prefix, name): (merged, shard) for name, shard in self.shards
        }

    @property
    def packed_modules_mapping(self) -> dict[str, list[str]]:
        """`{merged_name: [projection names]}` so quantization can unpack the
        fused layer into its per-shard configs."""
        return {self.merged_name: [name for name, _ in self.shards]}

    @abstractmethod
    def update_forward(self, module: nn.Module) -> None:
        """Rewrite and compile `type(module)`'s forward source.

        Raises if the source does not admit the rewrite (fusion is then skipped).
        """

    @abstractmethod
    def update_attrs(
        self,
        module: nn.Module,
        prefix: str,
        model_config: "ModelConfig",
        quant_config: "QuantizationConfig",
    ) -> None:
        """Replace `module`'s submodules with the merged module."""

    def fuse(
        self,
        module: nn.Module,
        prefix: str,
        model_config: "ModelConfig",
        quant_config: "QuantizationConfig",
    ) -> nn.Module:
        """Fuse an already-validated `module` in place (see `Fusers.__getitem__`).

        Builds the merged submodule and binds the compiled forward."""
        self.update_attrs(module, prefix, model_config, quant_config)
        module.forward = types.MethodType(self.fused_forward, module)
        return module
