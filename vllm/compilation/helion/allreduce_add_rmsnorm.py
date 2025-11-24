# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Helion implementation of fused AllReduce + Add + RMSNorm.

This module provides a Helion-based implementation of the AllReduceFusedAddRMSNorm
pattern that overlaps communication with computation using symmetric memory and
progress tracking.
"""

import helion
import helion.language as hl
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


def copy_engine_all_reduce_w_progress(
    output: torch.Tensor,
    inp: torch.Tensor,  # Must be symmetric tensor
    progress: torch.Tensor,
    splits_per_rank: int,
    backend_stream: torch.cuda.Stream | None = None,
) -> torch.cuda.Stream:
    """
    Performs an all-reduce (sum) operation with progress tracking using symmetric memory.

    This function splits the AllReduce into chunks and signals progress as each chunk
    completes, enabling the Helion kernel to overlap computation with communication.

    Args:
        output: The output tensor to store the reduced results [M, K]
        inp: The input tensor to be reduced (must be a symmetric tensor) [M, K]
        progress: Tensor used to track progress of the operation [splits_per_rank]
        splits_per_rank: Number of splits for progressive processing
        backend_stream: CUDA stream for backend operations

    Returns:
        The CUDA stream used for the operation
    """
    backend_stream = symm_mem._get_backend_stream(priority=-1)
    assert inp.is_contiguous(), "Input must be contiguous"

    # Get symmetric memory group (defaults to WORLD)
    symm_mem_group = dist.group.WORLD
    if symm_mem_group is None:
        raise RuntimeError(
            "Distributed group not initialized. Call dist.init_process_group() first."
        )

    # Rendezvous with symmetric memory
    try:
        symm_mem_hdl = symm_mem.rendezvous(inp, group=symm_mem_group)
    except Exception as e:
        raise RuntimeError(
            f"Failed to rendezvous with symmetric memory. "
            f"Ensure input tensor is created with symm_mem.empty() or symm_mem.zeros(). "
            f"Error: {e}"
        ) from e

    if symm_mem_hdl is None:
        raise RuntimeError(
            "Symmetric memory rendezvous returned None. "
            "Ensure input tensor is allocated with symmetric memory (symm_mem.empty/zeros)."
        )

    rank = symm_mem_hdl.rank
    world_size = symm_mem_hdl.world_size

    assert inp.numel() % splits_per_rank == 0, (
        f"Input size {inp.numel()} must be divisible by splits_per_rank {splits_per_rank}"
    )
    assert progress.numel() >= splits_per_rank, (
        f"Progress tensor size {progress.numel()} must be >= splits_per_rank {splits_per_rank}"
    )
    assert list(output.shape) == list(inp.shape), (
        f"Output shape {output.shape} must match input shape {inp.shape}"
    )

    chunks = output.chunk(splits_per_rank)
    inp_chunks = inp.chunk(splits_per_rank)
    symm_mem_hdl.barrier()
    backend_stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(backend_stream):
        for split_id in range(splits_per_rank):
            # Initialize output chunk with local data
            chunks[split_id].copy_(inp_chunks[split_id])

            # In-place accumulate data from other ranks
            # This is much faster than stack+sum as it avoids creating intermediate tensors
            for src_rank in range(world_size):
                if src_rank != rank:
                    src_buf = symm_mem_hdl.get_buffer(
                        src_rank,
                        chunks[0].shape,
                        inp.dtype,
                        chunks[0].numel() * split_id,
                    )
                    # In-place addition - no extra tensor allocation!
                    chunks[split_id].add_(src_buf)

            # Signal that this chunk is ready
            # cuStreamWriteValue32 issues a system level fence before the write
            symm_mem_hdl.stream_write_value32(
                progress,
                offset=split_id,
                val=1,
            )

        symm_mem_hdl.barrier()

    return backend_stream


@helion.kernel(
    config=helion.Config(
        block_sizes=[2],
        indexing=[
            "pointer",
            "pointer",
            "pointer",
            "tensor_descriptor",
            "pointer",
            "tensor_descriptor",
            "pointer",
            "pointer",
            "tensor_descriptor",
            "pointer",
            "pointer",
            "tensor_descriptor",
        ],
        load_eviction_policies=["last", "first", "", "", "first", "first", "first", ""],
        num_stages=1,
        num_warps=8,
        pid_type="persistent_interleaved",
        range_flattens=[False],
        range_multi_buffers=[None],
        range_num_stages=[3],
        range_unroll_factors=[1],
        range_warp_specializes=[],
        reduction_loops=[None],
    ),
    static_shapes=True,
)
def _allreduce_add_rmsnorm_helion_kernel(
    allreduce_buf: torch.Tensor,
    residual: torch.Tensor,
    rms_gamma: torch.Tensor,
    progress: torch.Tensor,
    rms_eps: float,
    SPLITS_PER_RANK: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Helion kernel for AllReduce + Add + RMSNorm with progress-based overlap.

    This kernel waits for AllReduce chunks to complete (via progress tensor)
    and processes them as they become available, overlapping computation with
    communication.

    Operation: RMSNorm(AllReduce(input) + residual), returns both normalized and residual

    Algorithm:
    1. Wait for chunk to be ready (via hl.wait on progress tensor)
    2. Add residual connection
    3. Compute variance: sum(x^2) / hidden_size
    4. Compute normalization: rsqrt(variance + epsilon)
    5. Apply normalization and weight scaling

    Args:
        allreduce_buf: Buffer being filled by AllReduce [M, K]
        residual: Residual tensor to add [M, K]
        rms_gamma: RMSNorm gamma weights [K]
        progress: Progress tracking tensor [SPLITS_PER_RANK]
        rms_eps: Epsilon for numerical stability
        SPLITS_PER_RANK: Number of splits per rank

    Returns:
        Tuple of (normalized_output, updated_residual) both [M, K]
    """
    M, K = allreduce_buf.size()
    out = torch.empty([M, K], dtype=allreduce_buf.dtype, device=allreduce_buf.device)
    residual_out = torch.empty(
        [M, K], dtype=allreduce_buf.dtype, device=allreduce_buf.device
    )

    # Process rows (M dimension) as they become available
    for tile_m in hl.tile(M):
        # Calculate which split this tile belongs to
        split_id = tile_m.begin // (M // SPLITS_PER_RANK)

        # Wait for this split to be ready
        hl.wait(progress, [split_id], signal=1)

        # Load the full row data
        allreduce_data = allreduce_buf[tile_m, :]  # [tile_m_size, K]
        residual_data = residual[tile_m, :]  # [tile_m_size, K]

        # Step 1: Add residual
        added = allreduce_data + residual_data

        # Store updated residual for output
        residual_out[tile_m, :] = added

        # Step 2: RMSNorm
        # Compute mean of squares for each row
        squared = added * added
        mean_sq = torch.mean(squared, dim=-1, keepdim=True)  # [tile_m_size, 1]

        # Compute normalization factor
        rsqrt_var = torch.rsqrt(mean_sq + rms_eps)  # [tile_m_size, 1]

        # Apply normalization and gamma scaling
        normalized = added * rsqrt_var * rms_gamma[None, :]

        # Write result
        out[tile_m, :] = normalized

    return out, residual_out


def helion_allreduce_add_rmsnorm(
    input_shared: torch.Tensor,
    residual: torch.Tensor,
    rms_gamma: torch.Tensor,
    rms_eps: float = 1e-6,
    splits_per_rank: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fuses AllReduce + Add + RMSNorm operations with communication/compute overlap.

    This is the main entry point for the fused operation. It:
    1. Starts an asynchronous AllReduce with progress tracking
    2. Launches a Helion kernel that processes chunks as they arrive
    3. Ensures proper stream synchronization

    Args:
        input_shared: Shared tensor across ranks to be reduced [M, K]
        residual: Residual tensor to add [M, K]
        rms_gamma: RMSNorm gamma weights [K]
        rms_eps: RMSNorm epsilon for numerical stability
        splits_per_rank: Number of splits for overlapping (higher = more overlap)

    Returns:
        Tuple of (normalized_output, updated_residual) both [M, K]
        - normalized_output: RMSNorm(AllReduce(input) + residual)
        - updated_residual: AllReduce(input) + residual

    Example:
        >>> # In a distributed setting with symmetric memory
        >>> input_shared = symm_mem.empty(
        ...     1024, 4096, dtype=torch.bfloat16, device="cuda"
        ... )
        >>> residual = torch.randn(1024, 4096, dtype=torch.bfloat16, device="cuda")
        >>> rms_gamma = torch.randn(4096, dtype=torch.bfloat16, device="cuda")
        >>> norm_out, residual_out = helion_allreduce_add_rmsnorm(
        ...     input_shared, residual, rms_gamma
        ... )
    """
    # Prepare buffers
    allreduce_out = torch.empty_like(input_shared)
    progress = torch.zeros(
        splits_per_rank,
        dtype=torch.uint32,
        device=input_shared.device,
    )

    # Start async AllReduce with progress tracking
    backend_stream = copy_engine_all_reduce_w_progress(
        allreduce_out, input_shared, progress, splits_per_rank
    )

    # Overlap: Helion kernel waits for chunks and computes Add + RMSNorm
    norm_out, residual_out = _allreduce_add_rmsnorm_helion_kernel(
        allreduce_out,
        residual,
        rms_gamma,
        progress,
        rms_eps,
        SPLITS_PER_RANK=splits_per_rank,
    )

    # Ensure backend stream completes
    torch.cuda.current_stream().wait_stream(backend_stream)

    return norm_out, residual_out
