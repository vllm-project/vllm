# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from typing import Any

import torch

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
    """Comm+GEMM for a linear layer at an SP region boundary, replacing its
    `forward`. While SP is active, a row-parallel layer reduce-scatters its
    output to a local token shard and a column-parallel one all-gathers its
    input back to full tokens. Used as is, it runs plain collectives and
    works for any quant method; subclasses fuse them with the GEMM.
    """

    def __init__(
        self, layer: LinearBase, sp_num_tokens: Callable[[], int | None]
    ) -> None:
        self.layer = layer
        self.layer_forward = layer.forward
        self.sp_num_tokens = sp_num_tokens
        self.is_row = isinstance(layer, RowParallelLinear)

    @classmethod
    def is_supported(cls, layer: LinearBase) -> bool:
        return True

    def __call__(self, x: torch.Tensor) -> Any:
        num_tokens = self.sp_num_tokens()
        if num_tokens is None:
            return self.layer_forward(x)
        if self.is_row:
            return self.gemm_reduce_scatter(x)
        return self.all_gather_gemm(x, num_tokens)

    def gemm_reduce_scatter(self, x: torch.Tensor) -> Any:
        layer = self.layer
        # As in `RowParallelLinear.forward`: only rank 0 adds the bias.
        bias = None if layer.tp_rank > 0 or layer.skip_bias_add else layer.bias
        output = sp_reduce_scatter(layer.quant_method.apply(layer, x, bias=bias))
        if not layer.return_bias:
            return output
        return output, layer.bias if layer.skip_bias_add else None

    def all_gather_gemm(self, x: torch.Tensor, num_tokens: int) -> Any:
        return self.layer_forward(sp_all_gather(x)[:num_tokens])


class BF16SPLinearKernel(SPLinearKernel):
    """Symmetric-memory fused comm+GEMM for unquantized linears."""

    # Overrides of the fusion thresholds in `can_sp_matmul_reduce_scatter`
    # and `can_sp_all_gather_matmul`.
    reduce_scatter_min_tokens: int | None = None
    all_gather_min_tokens: int | None = None

    @classmethod
    def is_supported(cls, layer: LinearBase) -> bool:
        # The fused symm-mem ops take no bias.
        return (
            type(layer.quant_method) is UnquantizedLinearMethod and layer.bias is None
        )

    def gemm_reduce_scatter(self, x: torch.Tensor) -> Any:
        weight = self.layer.weight
        if not can_sp_matmul_reduce_scatter(x, weight, self.reduce_scatter_min_tokens):
            return super().gemm_reduce_scatter(x)
        output = sp_matmul_reduce_scatter(x, weight)
        return (output, None) if self.layer.return_bias else output

    def all_gather_gemm(self, x: torch.Tensor, num_tokens: int) -> Any:
        weight = self.layer.weight
        if not can_sp_all_gather_matmul(x, weight, self.all_gather_min_tokens):
            return super().all_gather_gemm(x, num_tokens)
        output = sp_all_gather_matmul(x, weight)[:num_tokens]
        return (output, None) if self.layer.return_bias else output


# Priority order: new quant formats add their kernels ahead of the generic one.
DEFAULT_SP_LINEAR_KERNELS: tuple[type[SPLinearKernel], ...] = (
    BF16SPLinearKernel,
    SPLinearKernel,
)


def wrap_sp_linear(
    linear: ColumnParallelLinear | RowParallelLinear,
    sp_num_tokens: Callable[[], int | None],
    kernels: tuple[type[SPLinearKernel], ...] = DEFAULT_SP_LINEAR_KERNELS,
) -> None:
    """Replace `linear.forward` with the first supported `SPLinearKernel`.

    Args:
        linear: TP linear at an SP region boundary.
        sp_num_tokens: Token count of the in-flight forward if it runs SP,
            else `None`.
        kernels: Candidate kernels in priority order; the last one must
            support every layer.

    """
    if isinstance(linear, RowParallelLinear):
        assert linear.input_is_parallel
    else:
        assert not linear.gather_output, "SP boundary linears keep TP shards"
    kernel_cls = next(c for c in kernels if c.is_supported(linear))
    logger.info_once(
        "Sequence parallel: %s (%s) uses %s.",
        type(linear).__name__,
        type(linear.quant_method).__name__,
        kernel_cls.__name__,
    )
    linear.forward = kernel_cls(linear, sp_num_tokens)  # type: ignore[method-assign,assignment]
