# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Triton-distributed custom operations for GEMM+AllReduce fusion.

This module provides custom ops for fusing GEMM operations with AllReduce
communication using Triton-distributed's GemmAR kernels with NVSHMEM.
"""

import torch
from typing import Dict, Optional, Tuple

from vllm.distributed import get_tp_group
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

# Global context registry for GemmAR layers
# Key: (rank, world_size, max_M, N, K, dtype, use_ll_kernel)
# Value: GemmARLayer instance
_GEMM_AR_CONTEXTS: Dict[Tuple, "GemmARLayer"] = {}

# Flag to track if NVSHMEM has been initialized
_NVSHMEM_INITIALIZED = False

# NCCL process group for NVSHMEM
_NVSHMEM_PG = None


def initialize_nvshmem_for_gemm_ar():
    """
    Initialize NVSHMEM for GemmAR operations.

    This should be called once before any GemmAR operations are performed.
    Creates a dedicated NCCL process group and initializes NVSHMEM.
    """
    global _NVSHMEM_INITIALIZED, _NVSHMEM_PG

    if _NVSHMEM_INITIALIZED:
        return _NVSHMEM_PG

    try:
        from triton_dist.utils import init_nvshmem_by_torch_process_group
    except ImportError:
        raise RuntimeError(
            "Triton-distributed is not installed. "
            "Please install with: pip install triton-dist"
        ) from None

    world_size = get_tensor_model_parallel_world_size()

    # Create NCCL process group for NVSHMEM
    _NVSHMEM_PG = torch.distributed.new_group(
        ranks=list(range(world_size)),
        backend="nccl"
    )

    # Initialize NVSHMEM
    init_nvshmem_by_torch_process_group(_NVSHMEM_PG)
    _NVSHMEM_INITIALIZED = True

    logger.info("NVSHMEM initialized for Triton-distributed GemmAR (world_size=%d)", world_size)
    return _NVSHMEM_PG


def create_or_get_gemm_ar_context(
    rank: int,
    world_size: int,
    max_M: int,
    N: int,
    K: int,
    dtype: torch.dtype,
    use_ll_kernel: bool = False,
    num_comm_sms: int = 16,
) -> "GemmARLayer":
    """
    Get or create a GemmAR context for given parameters.

    Lazily initializes NVSHMEM on first call if not already initialized.

    Args:
        rank: Current rank in TP group
        world_size: Total number of ranks in TP group
        max_M: Maximum sequence length for context allocation
        N: Output dimension
        K: Total input dimension (across all ranks)
        dtype: Data type
        use_ll_kernel: Whether to use low-latency cooperative kernel
        num_comm_sms: Number of SMs to dedicate to communication

    Returns:
        GemmARLayer instance for the given configuration
    """
    try:
        from triton_dist.layers.nvidia import GemmARLayer
    except ImportError:
        raise RuntimeError(
            "Triton-distributed is not installed. "
            "Please install with: pip install triton-dist"
        ) from None

    # Initialize NVSHMEM if not already done
    tp_group = initialize_nvshmem_for_gemm_ar()

    key = (rank, world_size, max_M, N, K, dtype, use_ll_kernel)

    if key not in _GEMM_AR_CONTEXTS:
        logger.debug(
            "Creating new GemmAR context: rank=%d, ws=%d, max_M=%d, "
            "N=%d, K=%d, dtype=%s, use_ll=%s, num_sms=%d",
            rank, world_size, max_M, N, K, dtype, use_ll_kernel, num_comm_sms
        )

        _GEMM_AR_CONTEXTS[key] = GemmARLayer(
            tp_group=tp_group,
            max_M=max_M,
            N=N,
            K=K,
            input_dtype=dtype,
            output_dtype=dtype,
            local_world_size=world_size,
            persistent=True,
            use_ll_kernel=use_ll_kernel,
            copy_to_local=False,
            NUM_COMM_SMS=num_comm_sms,
        )

    return _GEMM_AR_CONTEXTS[key]


@torch._dynamo.disable
def triton_dist_gemm_allreduce(
    input: torch.Tensor,
    weight: torch.Tensor,
    max_M: int,
    use_ll_kernel: bool = False,
    num_comm_sms: int = 16,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Fused GEMM + AllReduce using Triton-distributed.

    This operation performs a matrix multiplication followed by an all-reduce
    operation, with compute-communication overlap using NVSHMEM primitives.

    Args:
        input: [M, K_local] activation tensor (column-partitioned)
        weight: [N, K_local] weight tensor (column-partitioned)
        max_M: Maximum sequence length (for context allocation)
        use_ll_kernel: Use low-latency cooperative kernel variant
        num_comm_sms: Number of SMs to dedicate to communication
        bias: Optional [N] bias tensor

    Returns:
        output: [M, N] result after GEMM and AllReduce

    Note:
        This function is decorated with @torch._dynamo.disable because
        GemmAR uses NVSHMEM primitives and custom Triton kernels that
        are incompatible with dynamo graph tracing.
    """
    # Check if we're in fake mode (during compilation/tracing)
    # In fake mode, return fake output without actual execution
    from torch._subclasses.fake_tensor import FakeTensor

    if isinstance(input, FakeTensor) or isinstance(weight, FakeTensor):
        # Return fake tensor with correct shape
        M = input.shape[0]
        N = weight.shape[0]
        return torch.empty(M, N, dtype=input.dtype, device=input.device)

    M, K = input.shape
    N, _ = weight.shape

    world_size = get_tensor_model_parallel_world_size()
    rank = get_tensor_model_parallel_rank()

    # Check if dimensions are compatible with max_M constraint
    # If not, fall back to standard GEMM + AllReduce
    if M > max_M:
        logger.warning(
            "TritonDistGemmAR: Input M=%d exceeds max_M=%d, "
            "falling back to standard GEMM+AllReduce",
            M, max_M
        )
        # Fallback: standard linear + all_reduce
        from vllm.distributed import tensor_model_parallel_all_reduce
        output = torch.nn.functional.linear(input, weight, bias)
        return tensor_model_parallel_all_reduce(output)

    # Get total K dimension (across all ranks)
    K_total = K * world_size

    logger.debug(
        "TritonDistGemmAR: M=%d, N=%d, K_local=%d, K_total=%d, max_M=%d",
        M, N, K, K_total, max_M
    )

    # Look up or create context (this will lazily initialize NVSHMEM)
    ctx = create_or_get_gemm_ar_context(
        rank=rank,
        world_size=world_size,
        max_M=max_M,
        N=N,
        K=K_total,
        dtype=input.dtype,
        use_ll_kernel=use_ll_kernel,
        num_comm_sms=num_comm_sms,
    )

    # Verify context dimensions match before calling forward
    if hasattr(ctx, 'max_M') and hasattr(ctx, 'N'):
        if M > ctx.max_M or N != ctx.N:
            logger.warning(
                "TritonDistGemmAR: Dimension mismatch! "
                "Input M=%d vs ctx.max_M=%d, weight N=%d vs ctx.N=%d, "
                "falling back to standard GEMM+AllReduce",
                M, ctx.max_M, N, ctx.N
            )
            from vllm.distributed import tensor_model_parallel_all_reduce
            output = torch.nn.functional.linear(input, weight, bias)
            return tensor_model_parallel_all_reduce(output)

    # Execute fused GEMM+AllReduce
    return ctx.forward(input, weight, bias)


def triton_dist_gemm_allreduce_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    max_M: int,
    use_ll_kernel: bool = False,
    num_comm_sms: int = 16,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Fake implementation for compilation and shape inference.

    This is used by torch.compile for graph tracing without actually
    executing the kernel.
    """
    M = input.shape[0]
    N = weight.shape[0]
    return torch.empty(M, N, dtype=input.dtype, device=input.device)


# Register the custom op
direct_register_custom_op(
    op_name="triton_dist_gemm_allreduce",
    op_func=triton_dist_gemm_allreduce,
    mutates_args=[],
    fake_impl=triton_dist_gemm_allreduce_fake,
)


def clear_gemm_ar_contexts():
    """
    Clear all cached GemmAR contexts.

    This is useful for cleanup or when changing TP configuration.
    """
    global _GEMM_AR_CONTEXTS
    for ctx in _GEMM_AR_CONTEXTS.values():
        if hasattr(ctx, 'finalize'):
            ctx.finalize()
    _GEMM_AR_CONTEXTS.clear()
    logger.debug("Cleared all GemmAR contexts")
