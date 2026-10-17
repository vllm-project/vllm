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
from vllm.platforms import PlatformEnum, current_platform

logger = init_logger(__name__)


def _try_kernels(
    linear: ColumnParallelLinear | RowParallelLinear,
    kernels: list[type[LinearCommKernel]],
) -> LinearCommKernel | None:
    for kernel_cls in kernels:
        supported, reason = kernel_cls.is_supported(linear)
        if supported:
            return kernel_cls(linear)
        logger.debug("%s not available: %s", kernel_cls.__name__, reason)
    return None


def _try_fused_kernels(
    linear: ColumnParallelLinear | RowParallelLinear,
    ag_kernels: dict[PlatformEnum, list[type[FusedAllGatherGemmKernel]]],
    rs_kernels: dict[PlatformEnum, list[type[FusedGemmReduceScatterKernel]]],
) -> LinearCommKernel | None:
    kernels = rs_kernels if isinstance(linear, RowParallelLinear) else ag_kernels
    return _try_kernels(linear, list(kernels.get(current_platform._enum, [])))


def _init_unquantized_fused_kernel(
    linear: ColumnParallelLinear | RowParallelLinear,
) -> LinearCommKernel | None:
    if type(linear.quant_method) is not UnquantizedLinearMethod:
        return None
    from vllm.model_executor.kernels.sequence_parallel.unquantized import (
        FusedAllGatherGemm,
        FusedGemmReduceScatter,
    )

    if isinstance(linear, RowParallelLinear):
        return FusedGemmReduceScatter(linear)
    return FusedAllGatherGemm(linear)


def init_fused_comm_kernel(
    linear: ColumnParallelLinear | RowParallelLinear,
) -> LinearCommKernel | None:
    """Return a fused (symmetric-memory) comm+GEMM kernel, or ``None``."""
    for init_kernel_fn in (_init_unquantized_fused_kernel,):
        kernel = init_kernel_fn(linear)
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
