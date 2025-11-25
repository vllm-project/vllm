# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Helion implementation of fused AllReduce + Add + RMSNorm.

This module provides a Helion-based implementation of the AllReduceFusedAddRMSNorm
pattern that overlaps communication with computation using symmetric memory and
progress tracking.
"""

import contextlib
from typing import Any

import helion
import helion.language as hl
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from vllm.compilation.helion.benchmark import DistributedKernelBenchmark

try:
    import flashinfer.comm as flashinfer_comm

    FLASHINFER_AVAILABLE = True
except ImportError:
    FLASHINFER_AVAILABLE = False


def copy_engine_all_reduce_w_progress(
    output: torch.Tensor,
    inp: torch.Tensor,
    progress: torch.Tensor,
    splits_per_rank: int,
    backend_stream: torch.cuda.Stream | None = None,
) -> torch.cuda.Stream:
    """
    Performs an allreduce (sum) operation with progress tracking using symmetric memory.

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
            f"Ensure input tensor is created with symm_mem.empty() "
            f"or symm_mem.zeros(). Error: {e}"
        ) from e

    if symm_mem_hdl is None:
        raise RuntimeError(
            "Symmetric memory rendezvous returned None. "
            "Ensure input tensor is allocated with symmetric memory"
            " (symm_mem.empty/zeros)."
        )

    rank = symm_mem_hdl.rank
    world_size = symm_mem_hdl.world_size

    assert inp.numel() % splits_per_rank == 0, (
        f"Input size {inp.numel()} must be divisible by "
        f"splits_per_rank {splits_per_rank}"
    )
    assert progress.numel() >= splits_per_rank, (
        f"Progress tensor size {progress.numel()} must be "
        f">= splits_per_rank {splits_per_rank}"
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
            # This is much faster than stack+sum as it avoids creating
            # intermediate tensors
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

    Operation: RMSNorm(AllReduce(input) + residual), returns both normalized
    and residual

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
        split_id = tile_m.begin // (M // SPLITS_PER_RANK)

        hl.wait(progress, [split_id], signal=1)

        allreduce_data = allreduce_buf[tile_m, :]
        residual_data = residual[tile_m, :]
        added = allreduce_data + residual_data
        residual_out[tile_m, :] = added

        # Use FP32 for all intermediate computations to match tight tolerance
        # requirements and match FlashInfer's fp32_acc=True behavior
        # TODO(gmagogsfm): Support fp32_acc=False
        added_fp32 = added.float()
        squared = added_fp32 * added_fp32
        mean_sq = torch.mean(squared, dim=-1, keepdim=True)

        rsqrt_var = torch.rsqrt(mean_sq + rms_eps)
        normalized = (added_fp32 * rsqrt_var * rms_gamma[None, :].float()).to(
            added.dtype
        )
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
    allreduce_out = torch.empty_like(input_shared)
    progress = torch.zeros(
        splits_per_rank,
        dtype=torch.uint32,
        device=input_shared.device,
    )

    backend_stream = copy_engine_all_reduce_w_progress(
        allreduce_out, input_shared, progress, splits_per_rank
    )

    norm_out, residual_out = _allreduce_add_rmsnorm_helion_kernel(
        allreduce_out,
        residual,
        rms_gamma,
        progress,
        rms_eps,
        SPLITS_PER_RANK=splits_per_rank,
    )

    torch.cuda.current_stream().wait_stream(backend_stream)

    return norm_out, residual_out


class AllReduceAddRMSNormBenchmark(DistributedKernelBenchmark):
    """
    Benchmark for Helion AllReduce + Add + RMSNorm kernel.

    This benchmark requires multi-GPU setup and compares the Helion fused kernel
    against FlashInfer's trtllm_allreduce_fusion baseline.

    The benchmark measures:
    - Numerical correctness against FlashInfer
    - Performance with and without CUDA graphs
    - Speedup across different tensor shapes and split configurations

    Note: CUDA graphs are disabled for the Helion kernel because it uses
    overlapped AllReduce with cross-stream synchronization (backend stream
    for AllReduce + main stream for compute). CUDA graphs can only capture
    operations on a single stream, causing deadlocks during graph replay.
    The baseline (FlashInfer) still uses CUDA graphs for fair comparison.
    """

    benchmark_name = "allreduce_add_rmsnorm"

    def __init__(self, num_gpus: int = 2):
        """
        Args:
            num_gpus: Number of GPUs to use for distributed benchmark (default: 2)
        """
        super().__init__(num_gpus=num_gpus, master_port=12348)

        if not FLASHINFER_AVAILABLE:
            raise RuntimeError(
                "FlashInfer is required for baseline comparison. "
                "Install with: pip install flashinfer"
            )

        # Will be initialized per-worker
        self._flashinfer_workspace = None
        self._flashinfer_ipc_handles = None
        self._buffer_cache: dict[tuple[int, int], dict[str, torch.Tensor]] = {}

    def supports_cudagraph(self) -> bool:
        """
        Enable CUDA graphs for Helion kernel.

        Note: Some configurations may hang during CUDA graph capture with
        overlapped AllReduce. If hangs occur, this can be overridden to return False.
        """
        return True

    def _setup_flashinfer_workspace(
        self, M: int, K: int, local_rank: int, world_size: int
    ):
        """
        Setup FlashInfer IPC workspace.

        Must be called once per worker before running FlashInfer baseline.
        """
        self._flashinfer_ipc_handles, self._flashinfer_workspace = (
            flashinfer_comm.trtllm_create_ipc_workspace_for_all_reduce_fusion(
                tp_rank=local_rank,
                tp_size=world_size,
                max_token_num=M,
                hidden_dim=K,
                group=dist.group.WORLD,
                use_fp32_lamport=False,
            )
        )

    def _cleanup_flashinfer_workspace(self):
        """Cleanup FlashInfer IPC workspace."""
        if self._flashinfer_ipc_handles is not None:
            with contextlib.suppress(Exception):
                flashinfer_comm.trtllm_destroy_ipc_workspace_for_all_reduce_fusion(
                    self._flashinfer_ipc_handles
                )
            self._flashinfer_workspace = None
            self._flashinfer_ipc_handles = None

    def _get_or_create_cached_buffer(
        self,
        M: int,
        K: int,
        dtype: torch.dtype,
        buffer_name: str,
        input_data: torch.Tensor,
        residual_data: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get or create a cached symmetric memory buffer for the given shape.

        This method manages buffer caching to enable CUDA graph capture by avoiding
        tensor allocations during graph replay. Buffers are cached per (M, K) shape.

        Args:
            M: Number of rows
            K: Number of columns
            dtype: Data type for the buffer
            buffer_name: Name to use for caching this buffer (e.g., 'baseline_input')
            input_data: Input tensor data to copy into the buffer
            residual_data: Residual tensor data to clone

        Returns:
            Tuple of (input_buffer, residual_buffer) where:
            - input_buffer is a cached symmetric memory buffer filled with input_data
            - residual_buffer is a fresh clone of residual_data (not cached)
        """
        cache_key = (M, K)

        # Get or create input buffer from cache
        if hasattr(self, "_buffer_cache") and cache_key in self._buffer_cache:
            buffers = self._buffer_cache[cache_key]
            if buffer_name in buffers:
                input_buffer = buffers[buffer_name]
            else:
                input_buffer = symm_mem.empty(M, K, dtype=dtype, device="cuda")
                buffers[buffer_name] = input_buffer
        else:
            input_buffer = symm_mem.empty(M, K, dtype=dtype, device="cuda")
            if hasattr(self, "_buffer_cache"):
                self._buffer_cache[cache_key] = {buffer_name: input_buffer}

        # Always clone residual (don't cache) to prevent mutation between runs
        residual_buffer = residual_data.clone()

        # Copy input data into cached buffer
        input_buffer.copy_(input_data)

        return input_buffer, residual_buffer

    def get_quick_test_shapes(
        self,
    ) -> list[tuple[list[tuple], torch.dtype, dict[str, list[Any]]]]:
        """Quick smoke test with 3 representative configurations."""
        shapes = [
            (128, 2048),
            (512, 4096),
            (1024, 4096),
        ]
        return [
            (shapes, torch.bfloat16, {"splits_per_rank": [4]}),
        ]

    def get_full_test_shapes(
        self,
    ) -> list[tuple[list[tuple], torch.dtype, dict[str, list[Any]]]]:
        """Comprehensive benchmark across multiple shapes and dtypes."""
        shapes = [
            (64, 1024),
            (128, 1024),
            (256, 1024),
            (512, 1024),
            (64, 4096),
            (128, 4096),
            (256, 4096),
            (512, 4096),
            (1024, 4096),
            (128, 8192),
            (256, 8192),
            (512, 8192),
            (1024, 8192),
        ]
        return [
            (shapes, torch.bfloat16, {"splits_per_rank": [2, 4, 8]}),
            (shapes, torch.float16, {"splits_per_rank": [2, 4, 8]}),
        ]

    def create_inputs(self, dtype: torch.dtype, **shape_params) -> tuple[Any, ...]:
        """
        Create inputs for the AllReduce + Add + RMSNorm kernel.

        This is called within a distributed worker process where CUDA device
        and distributed environment are already initialized.

        Args:
            dtype: Data type for tensors
            **shape_params: Must contain 'shape' key with (M, K) tuple

        Returns:
            Tuple of (input_data, residual_data, gamma, eps, splits_per_rank)
        """
        M, K = shape_params["shape"]
        splits_per_rank = shape_params.get("splits_per_rank", 4)

        # Set deterministic seed per rank FIRST
        torch.manual_seed(42 + dist.get_rank())

        # Create test data with the seeded random state
        input_data = torch.randn(M, K, dtype=dtype, device="cuda")
        residual_data = torch.randn(M, K, dtype=dtype, device="cuda")
        gamma = torch.ones(K, dtype=dtype, device="cuda")
        eps = 1e-6

        return input_data, residual_data, gamma, eps, splits_per_rank

    def run_baseline(self, *args, **kwargs) -> Any:
        """
        Run FlashInfer baseline kernel.

        Args are unpacked from create_inputs():
            input_data, residual_data, gamma, eps, splits_per_rank

        Returns:
            Tuple of (norm_out, residual_out)
        """
        input_data, residual_data, gamma, eps, splits_per_rank = args

        M, K = input_data.shape
        local_rank = dist.get_rank()
        world_size = dist.get_world_size()

        input_baseline, residual_baseline = self._get_or_create_cached_buffer(
            M, K, input_data.dtype, "baseline_input", input_data, residual_data
        )

        # FlashInfer operates in-place
        norm_out_baseline = input_baseline
        residual_out_baseline = residual_baseline

        flashinfer_comm.trtllm_allreduce_fusion(
            allreduce_in=input_baseline,
            token_num=M,
            residual_in=residual_baseline,
            residual_out=residual_out_baseline,
            norm_out=norm_out_baseline,
            rms_gamma=gamma,
            rms_eps=eps,
            hidden_dim=K,
            workspace_ptrs=self._flashinfer_workspace,
            pattern_code=flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNorm,
            allreduce_out=None,
            quant_out=None,
            scale_out=None,
            layout_code=flashinfer_comm.QuantizationSFLayout.SWIZZLED_128x4,
            scale_factor=None,
            use_oneshot=False,
            world_rank=local_rank,
            world_size=world_size,
            launch_with_pdl=True,
            trigger_completion_at_end=True,
            fp32_acc=True,
        )

        return norm_out_baseline.clone(), residual_out_baseline.clone()

    def run_helion(self, *args, **kwargs) -> Any:
        """
        Run Helion kernel.

        Args are unpacked from create_inputs():
            input_data, residual_data, gamma, eps, splits_per_rank

        Returns:
            Tuple of (norm_out, residual_out)
        """
        input_data, residual_data, gamma, eps, splits_per_rank = args

        M, K = input_data.shape

        # Get or create cached buffers
        input_helion, residual_helion = self._get_or_create_cached_buffer(
            M, K, input_data.dtype, "helion_input", input_data, residual_data
        )

        norm_out, residual_out = helion_allreduce_add_rmsnorm(
            input_helion,
            residual_helion,
            gamma,
            eps,
            splits_per_rank,
        )

        return norm_out, residual_out

    def get_shape_description(self, **shape_params) -> str:
        """Generate description for shape configuration."""
        M, K = shape_params["shape"]
        splits = shape_params.get("splits_per_rank", 4)
        return f"M={M}_K={K}_splits={splits}"

    def setup_config(self, local_rank: int, world_size: int, config):
        """
        Setup FlashInfer workspace for this specific config.

        Recreating the workspace per test config to ensure isolation
        """
        M, K = config.shape_params["shape"]
        self._setup_flashinfer_workspace(M, K, local_rank, world_size)

    def teardown_config(self):
        """
        Cleanup FlashInfer workspace and buffer cache after this config completes.

        This ensures fresh state for the next config by cleaning up:
        - FlashInfer IPC workspace (prevents cumulative state corruption)
        - Buffer cache (ensures fresh symmetric memory buffers for CUDA graph capture)
        """
        self._cleanup_flashinfer_workspace()
        self._buffer_cache.clear()
