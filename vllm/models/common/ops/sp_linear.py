# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sequence-parallel linear boundaries for eager model paths: the
all-gather/reduce-scatter around a TP linear, fused with its GEMM via
symmetric memory when the shape and quant method allow, else plain
collectives.

The compile-mode analog is the AsyncTP pass; this covers models running
eager.
"""

from collections.abc import Callable
from typing import Any

import torch
from torch.distributed._symmetric_memory import enable_symm_mem_for_group

from vllm.distributed import get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.models.common.ops.sequence_parallel import (
    can_sp_all_gather_matmul,
    can_sp_matmul_reduce_scatter,
    sp_all_gather,
    sp_all_gather_matmul,
    sp_matmul_reduce_scatter,
    sp_reduce_scatter,
)

logger = init_logger(__name__)


class SPLinearKernel:
    """Comm+GEMM for a linear layer at an SP region boundary. It replaces
    the layer's `forward`: while SP is active it runs `fused` when
    `can_fuse` allows, else `unfused`; otherwise the plain `forward`.
    Subclasses per quant format provide `is_supported`, `can_fuse` and
    `fused`.
    """

    def __init__(
        self, layer: LinearBase, sp_num_tokens: Callable[[], int | None]
    ) -> None:
        self.layer = layer
        self.layer_forward = layer.forward
        self.sp_num_tokens = sp_num_tokens

    @classmethod
    def is_supported(cls, layer: LinearBase) -> tuple[bool, str | None]:
        return True, None

    def can_fuse(self, x: torch.Tensor) -> bool:
        """Whether `fused` applies to this call's input."""
        return False

    def fused(self, x: torch.Tensor, num_tokens: int) -> Any:
        raise NotImplementedError

    def unfused(self, x: torch.Tensor, num_tokens: int) -> Any:
        raise NotImplementedError

    def __call__(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        num_tokens = self.sp_num_tokens()
        if num_tokens is None:
            return self.layer_forward(x, *args, **kwargs)
        if self.can_fuse(x):
            return self.fused(x, num_tokens)
        return self.unfused(x, num_tokens)

    def _add_bias(self, output: torch.Tensor) -> Any:
        """Finish a GEMM output like the layer's `forward`. For row-parallel
        layers, each rank adds the bias to its own reduce-scattered shard."""
        layer = self.layer
        if layer.bias is not None and not layer.skip_bias_add:
            output = output + layer.bias
        if not layer.return_bias:
            return output
        return output, layer.bias if layer.skip_bias_add else None


class AllGatherGemm(SPLinearKernel):
    """AllGather + GEMM for the `ColumnParallelLinear` exiting an SP
    region. Used as is, it's the unfused fallback for any quant method."""

    def unfused(self, x: torch.Tensor, num_tokens: int) -> Any:
        return self.layer_forward(sp_all_gather(x)[:num_tokens])


class GemmReduceScatter(SPLinearKernel):
    """GEMM + ReduceScatter for the `RowParallelLinear` entering an SP
    region. Used as is, it's the unfused fallback for any quant method."""

    def unfused(self, x: torch.Tensor, num_tokens: int) -> Any:
        layer = self.layer
        assert layer.input_is_parallel
        return self._add_bias(sp_reduce_scatter(layer.quant_method.apply(layer, x)))


def _bf16_is_supported(layer: LinearBase) -> tuple[bool, str | None]:
    if type(layer.quant_method) is not UnquantizedLinearMethod:
        return False, "requires UnquantizedLinearMethod"
    return True, None


class BF16AllGatherGemm(AllGatherGemm):
    """Symmetric-memory AllGather + GEMM for unquantized linears."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        enable_symm_mem_for_group(get_tp_group().device_group.group_name)
        super().__init__(*args, **kwargs)

    @classmethod
    def is_supported(cls, layer: LinearBase) -> tuple[bool, str | None]:
        if getattr(layer, "gather_output", False):
            return False, "requires gather_output=False"
        return _bf16_is_supported(layer)

    def can_fuse(self, x: torch.Tensor) -> bool:
        return can_sp_all_gather_matmul(x, self.layer.weight)

    def fused(self, x: torch.Tensor, num_tokens: int) -> Any:
        output = sp_all_gather_matmul(x, self.layer.weight)[:num_tokens]
        return self._add_bias(output)


class BF16GemmReduceScatter(GemmReduceScatter):
    """Symmetric-memory GEMM + ReduceScatter for unquantized linears."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        enable_symm_mem_for_group(get_tp_group().device_group.group_name)
        super().__init__(*args, **kwargs)

    @classmethod
    def is_supported(cls, layer: LinearBase) -> tuple[bool, str | None]:
        return _bf16_is_supported(layer)

    def can_fuse(self, x: torch.Tensor) -> bool:
        return can_sp_matmul_reduce_scatter(x, self.layer.weight)

    def fused(self, x: torch.Tensor, num_tokens: int) -> Any:
        assert self.layer.input_is_parallel
        return self._add_bias(sp_matmul_reduce_scatter(x, self.layer.weight))


def _init_sp_linear_kernel(
    linear: ColumnParallelLinear | RowParallelLinear,
    sp_num_tokens: Callable[[], int | None],
) -> SPLinearKernel:
    """Pick the first supported kernel, in priority order. New quant
    formats add their kernels ahead of the generic ones."""
    candidates: list[type[SPLinearKernel]]
    if isinstance(linear, RowParallelLinear):
        candidates = [BF16GemmReduceScatter, GemmReduceScatter]
    else:
        candidates = [BF16AllGatherGemm, AllGatherGemm]
    for kernel_cls in candidates:
        supported, reason = kernel_cls.is_supported(linear)
        if supported:
            return kernel_cls(linear, sp_num_tokens)
        logger.debug("%s not available: %s", kernel_cls.__name__, reason)
    raise AssertionError("the base kernels support every linear")


def mark_sp_region(
    entry_after: RowParallelLinear | None,
    exit_before: ColumnParallelLinear | None,
    sp_num_tokens: Callable[[], int | None],
) -> None:
    """Mark the region between two linear layers as sequence-parallel.

    Args:
        entry_after: Row-parallel layer whose output enters the region --
            it's reduce-scattered down to a local shard. `None` for the
            first region, whose input embeddings the model shards.
        exit_before: Column-parallel layer whose input exits the region --
            it's all-gathered back to full tokens before its own GEMM.
            `None` for the last region, whose shard is gathered by the
            model after the final norm.
        sp_num_tokens: Token count of the in-flight forward if it runs SP,
            else `None`.

    Everything in between (typically a residual add + norm) runs on the
    local shard, padded to a multiple of the TP size. Each layer's
    `forward` is replaced by an `SPLinearKernel`; while SP is active it
    runs a fused (symmetric-memory) comm+GEMM kernel when the shape and
    quant method allow, otherwise plain collectives.

    """
    for linear in (exit_before, entry_after):
        if linear is None:
            continue
        kernel = _init_sp_linear_kernel(linear, sp_num_tokens)
        logger.info_once(
            "SP: using %s with %s.",
            type(kernel).__name__,
            type(linear.quant_method).__name__,
        )
        linear.forward = kernel  # type: ignore[method-assign,assignment]
