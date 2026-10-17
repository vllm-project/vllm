# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.logger import init_logger
from vllm.model_executor.kernels.sequence_parallel.base import (
    FusedAllGatherGemmKernel,
    FusedGemmReduceScatterKernel,
    LinearCommKernel,
    UnfusedAllGatherGemm,
    UnfusedGemmReduceScatter,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)

logger = init_logger(__name__)


def _init_base_fused_kernel(
    linear: ColumnParallelLinear | RowParallelLinear,
) -> LinearCommKernel | None:
    if type(linear.quant_method) is not UnquantizedLinearMethod:
        return None
    from vllm.model_executor.kernels.sequence_parallel.unquantized import (
        FusedAllGatherGemm,
        FusedGemmReduceScatter,
    )

    kernel_cls = (
        FusedGemmReduceScatter
        if isinstance(linear, RowParallelLinear)
        else FusedAllGatherGemm
    )
    return kernel_cls(linear)


def init_fused_comm_kernel(
    linear: ColumnParallelLinear | RowParallelLinear,
) -> LinearCommKernel | None:
    """Return a fused (symmetric-memory) comm+GEMM kernel, or ``None``."""
    kernel = _init_base_fused_kernel(linear)
    if kernel is not None:
        return kernel
    logger.debug("Fused comm+GEMM not available for %s", type(linear).__name__)
    return None


def init_unfused_comm_kernel(
    linear: ColumnParallelLinear | RowParallelLinear,
) -> LinearCommKernel:
    """Return a comm+GEMM kernel built on plain collectives."""
    if isinstance(linear, RowParallelLinear):
        return UnfusedGemmReduceScatter(linear)

    if isinstance(linear, ColumnParallelLinear):
        return UnfusedAllGatherGemm(linear)
    raise TypeError(
        f"init_unfused_comm_kernel: expected ColumnParallelLinear or "
        f"RowParallelLinear, got {type(linear).__name__}."
    )


__all__ = [
    "LinearCommKernel",
    "FusedAllGatherGemmKernel",
    "FusedGemmReduceScatterKernel",
    "UnfusedAllGatherGemm",
    "UnfusedGemmReduceScatter",
    "init_fused_comm_kernel",
    "init_unfused_comm_kernel",
]
