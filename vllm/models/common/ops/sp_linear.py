# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sequence-parallel linear boundaries for eager model paths."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import torch
from torch.distributed._symmetric_memory import enable_symm_mem_for_group

from vllm.distributed import get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.custom_op import PluggableLayer, maybe_get_oot_by_class
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.models.common.ops.sequence_parallel import (
    can_sp_all_gather_matmul,
    can_sp_matmul_reduce_scatter,
    sp_all_gather_matmul,
    sp_matmul_reduce_scatter,
)

logger = init_logger(__name__)


class SPLinearBoundary(PluggableLayer, ABC):
    """Base for a fused communication and linear operator."""

    def __init__(
        self,
        layer: LinearBase,
        min_tokens: int = 128,
        max_tokens: int | None = None,
    ) -> None:
        super().__init__()
        if min_tokens < 0:
            raise ValueError("min_tokens must be non-negative")
        if max_tokens is not None and max_tokens < min_tokens:
            raise ValueError("max_tokens must be greater than or equal to min_tokens")
        object.__setattr__(self, "layer", layer)
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

    @classmethod
    def is_supported(cls, layer: LinearBase) -> tuple[bool, str | None]:
        return True, None

    @classmethod
    @abstractmethod
    def create(
        cls,
        layer: LinearBase,
        min_tokens: int = 128,
        max_tokens: int | None = None,
    ) -> "SPLinearBoundary":
        """Create the implementation appropriate for the layer's data type."""

    def can_fuse(self, x: torch.Tensor) -> bool:
        return False

    def fused(self, x: torch.Tensor, num_tokens: int) -> torch.Tensor:
        raise NotImplementedError

    def _add_bias(self, output: torch.Tensor) -> torch.Tensor:
        layer = self.layer
        if layer.bias is not None and not layer.skip_bias_add:
            output = output + layer.bias
        return output


@PluggableLayer.register("sp_all_gather_matmul")
class AGMM(SPLinearBoundary):
    """All-gather + matmul boundary placed before a column-parallel linear."""

    @classmethod
    def create(
        cls,
        layer: LinearBase,
        min_tokens: int = 128,
        max_tokens: int | None = None,
    ) -> SPLinearBoundary:
        for impl_cls in cls.get_implementations():
            supported, reason = impl_cls.is_supported(layer)
            if supported:
                return impl_cls(layer, min_tokens, max_tokens)
            logger.debug("%s not available: %s", impl_cls.__name__, reason)
        raise AssertionError("the base AGMM implementation supports every linear")

    @classmethod
    def get_implementations(cls) -> tuple[type["AGMM"], ...]:
        """Return data-type implementations in priority order."""
        if cls is not AGMM:
            return (cls,)
        return BF16AGMM, AGMM

class BF16AGMM(AGMM):
    """Symmetric-memory AGMM for unquantized BF16 linears."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        enable_symm_mem_for_group(get_tp_group().device_group.group_name)
        super().__init__(*args, **kwargs)

    @classmethod
    def is_supported(cls, layer: LinearBase) -> tuple[bool, str | None]:
        if getattr(layer, "gather_output", False):
            return False, "requires gather_output=False"
        if type(layer.quant_method) is not UnquantizedLinearMethod:
            return False, "requires UnquantizedLinearMethod"
        return True, None

    def can_fuse(self, x: torch.Tensor) -> bool:
        return can_sp_all_gather_matmul(
            x,
            self.layer.weight,
            self.min_tokens,
            self.max_tokens,
        )

    def fused(self, x: torch.Tensor, num_tokens: int) -> torch.Tensor:
        output = sp_all_gather_matmul(x, self.layer.weight)[:num_tokens]
        return self._add_bias(output)


@PluggableLayer.register("sp_matmul_reduce_scatter")
class MMRS(SPLinearBoundary):
    """Matmul + reduce-scatter boundary placed after a row-parallel linear."""

    @classmethod
    def create(
        cls,
        layer: LinearBase,
        min_tokens: int = 128,
        max_tokens: int | None = None,
    ) -> SPLinearBoundary:
        for impl_cls in cls.get_implementations():
            supported, reason = impl_cls.is_supported(layer)
            if supported:
                return impl_cls(layer, min_tokens, max_tokens)
            logger.debug("%s not available: %s", impl_cls.__name__, reason)
        raise AssertionError("the base MMRS implementation supports every linear")

    @classmethod
    def get_implementations(cls) -> tuple[type["MMRS"], ...]:
        """Return data-type implementations in priority order."""
        if cls is not MMRS:
            return (cls,)
        return BF16MMRS, MMRS

class BF16MMRS(MMRS):
    """Symmetric-memory MMRS for unquantized BF16 linears."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        enable_symm_mem_for_group(get_tp_group().device_group.group_name)
        super().__init__(*args, **kwargs)

    @classmethod
    def is_supported(cls, layer: LinearBase) -> tuple[bool, str | None]:
        if type(layer.quant_method) is not UnquantizedLinearMethod:
            return False, "requires UnquantizedLinearMethod"
        return True, None

    def can_fuse(self, x: torch.Tensor) -> bool:
        return can_sp_matmul_reduce_scatter(
            x,
            self.layer.weight,
            self.min_tokens,
            self.max_tokens,
        )

    def fused(self, x: torch.Tensor, num_tokens: int) -> torch.Tensor:
        assert isinstance(self.layer, RowParallelLinear)
        assert self.layer.input_is_parallel
        return self._add_bias(sp_matmul_reduce_scatter(x, self.layer.weight))


def _create_sp_linear_boundary(
    linear: ColumnParallelLinear | RowParallelLinear,
) -> SPLinearBoundary:
    boundary_cls: type[AGMM] | type[MMRS]
    boundary_cls = MMRS if isinstance(linear, RowParallelLinear) else AGMM
    oot_cls = maybe_get_oot_by_class(boundary_cls)
    return oot_cls.create(linear)


def mark_sp_region(
    entry_after: RowParallelLinear | None,
    exit_before: ColumnParallelLinear | None,
    sp_num_tokens: Callable[[], int | None],
) -> None:
    """Mark the region between two linear layers as sequence-parallel."""
    for linear in (exit_before, entry_after):
        if linear is None:
            continue
        boundary = _create_sp_linear_boundary(linear)
        logger.info_once(
            "SP: using %s with %s.",
            type(boundary).__name__,
            type(linear.quant_method).__name__,
        )
        linear.register_sp_hooks(boundary, sp_num_tokens)
