# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import contextmanager
from types import MethodType
from typing import Any

import torch

from vllm.model_executor.layers.linear import LinearBase
from vllm.models.common.ops.sequence_parallel import sp_all_gather, sp_reduce_scatter


class LinearCommKernel(ABC):
    """Comm+GEMM for a linear layer at an SP region boundary. ``apply``
    replaces the layer's ``forward`` while SP is active."""

    def __init__(self, layer: LinearBase) -> None:
        self.layer = layer
        self.layer_forward = layer.forward

    @classmethod
    @abstractmethod
    def is_supported(cls, layer: LinearBase) -> tuple[bool, str | None]:
        raise NotImplementedError

    @abstractmethod
    def apply(self, x: torch.Tensor) -> Any:
        raise NotImplementedError

    @staticmethod
    @contextmanager
    def _replace(source: MethodType, target: Callable):
        """Swap the bound method ``source`` for ``target`` for one call."""
        owner, name = source.__self__, source.__name__
        setattr(owner, name, target)
        try:
            yield
        finally:
            setattr(owner, name, source)

    def column_parallel_forward(self, x: torch.Tensor) -> Any:
        """``ColumnParallelLinear.forward``, called through the pre-patch
        bound method captured in ``__init__`` so it still goes through
        ``apply``'s swapped-out quant method GEMM."""
        return self.layer_forward(x)

    def row_parallel_forward(
        self,
        x: torch.Tensor,
        reduce_scatter: bool = False,
    ) -> Any:
        """``RowParallelLinear.forward`` with the all-reduce replaced by
        ``reduce_scatter``, or by nothing if the GEMM already scatters its
        output. Each rank then adds the bias to its own token shard."""
        layer = self.layer
        assert layer.input_is_parallel
        output = layer.quant_method.apply(layer, x)
        if reduce_scatter:
            output = sp_reduce_scatter(output)
        bias = layer.bias
        if bias is not None and not layer.skip_bias_add:
            output = output + bias
        if not layer.return_bias:
            return output
        return output, bias if layer.skip_bias_add else None


class FusedCommGemmKernel(LinearCommKernel):
    """A ``LinearCommKernel`` that works by swapping out the quant
    method's GEMM for one call, fusing a symmetric-memory collective into it.

    Subclasses implement ``source_gemm_method`` (the bound GEMM method to
    replace) and ``target_gemm_method`` (the fused comm+GEMM replacement,
    with the same signature as the source method).
    """

    @abstractmethod
    def source_gemm_method(self) -> MethodType:
        """The quant method's GEMM method this kernel fuses a collective
        into, e.g. ``self._fp8_kernel.apply_scaled_mm``."""
        raise NotImplementedError

    @abstractmethod
    def target_gemm_method(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Fused comm+GEMM replacement for ``source_gemm_method``, with the
        same signature."""
        raise NotImplementedError


class FusedAllGatherGemmKernel(FusedCommGemmKernel):
    """Symmetric-memory AllGather + GEMM for the ``ColumnParallelLinear``
    exiting an SP region."""

    def apply(self, x: torch.Tensor) -> Any:
        with self._replace(self.source_gemm_method(), self.target_gemm_method):
            return self.column_parallel_forward(x)


class FusedGemmReduceScatterKernel(FusedCommGemmKernel):
    """Symmetric-memory GEMM + ReduceScatter for the ``RowParallelLinear``
    entering an SP region. ``target_gemm_method`` already scatters its
    output, so no further reduce-scatter is applied."""

    def apply(self, x: torch.Tensor) -> Any:
        with self._replace(self.source_gemm_method(), self.target_gemm_method):
            return self.row_parallel_forward(x)


class UnfusedAllGatherGemm(LinearCommKernel):
    """All-gather the input shard, then the layer's plain ``forward``.

    ``all_gather_input`` controls whether ``apply`` all-gathers ``x`` itself
    before calling ``forward``. Subclasses that instead gather some
    derivative of ``x`` (e.g. a quantized activation) from within a swapped
    GEMM method should pass ``all_gather_input=False``.
    """

    def __init__(self, layer: LinearBase, all_gather_input: bool = True) -> None:
        self.all_gather_input = all_gather_input
        super().__init__(layer)

    @classmethod
    def is_supported(cls, layer: LinearBase) -> tuple[bool, str | None]:
        return True, None

    def apply(self, x: torch.Tensor) -> Any:
        if self.all_gather_input:
            x = sp_all_gather(x)
        return self.column_parallel_forward(x)


class UnfusedGemmReduceScatter(LinearCommKernel):
    """The layer's GEMM, then reduce-scatter."""

    @classmethod
    def is_supported(cls, layer: LinearBase) -> tuple[bool, str | None]:
        return True, None

    def apply(self, x: torch.Tensor) -> Any:
        return self.row_parallel_forward(x, reduce_scatter=True)
