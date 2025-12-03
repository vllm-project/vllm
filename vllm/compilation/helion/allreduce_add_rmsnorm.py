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

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from vllm.compilation.helion.benchmark import DistributedKernelBenchmark
from vllm.compilation.helion.custom_op import HelionCustomOp
from vllm.compilation.helion.register import register_kernel
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp

logger = init_logger(__name__)

# Try to import Helion - it's an optional dependency
try:
    import helion
    import helion.language as hl

    HELION_AVAILABLE = True
except ImportError:
    HELION_AVAILABLE = False

try:
    import flashinfer.comm as flashinfer_comm

    FLASHINFER_AVAILABLE = True
except ImportError:
    FLASHINFER_AVAILABLE = False


def _is_symmetric_memory_tensor(tensor: torch.Tensor) -> bool:
    """
    Check if a tensor is allocated with symmetric memory.

    Returns:
        True if the tensor is symmetric memory, False otherwise.
    """
    # Check if distributed is initialized
    if not dist.is_initialized():
        return False

    try:
        # The rendezvous function is the canonical way to check if a tensor
        # is symmetric memory. It returns a handle if successful, None if not.
        symm_mem_group = dist.group.WORLD
        handle = symm_mem.rendezvous(tensor, group=symm_mem_group)
        return handle is not None
    except Exception:
        # If rendezvous fails for any reason, it's not symmetric memory
        return False


def _ensure_symmetric_memory(tensor: torch.Tensor) -> torch.Tensor:
    """
    Ensure a tensor is allocated with symmetric memory, converting if necessary.

    Args:
        tensor: Input tensor that may or may not be symmetric memory

    Returns:
        A tensor that is guaranteed to be symmetric memory
    """
    # Fast path: if it's already symmetric memory, return as-is
    if _is_symmetric_memory_tensor(tensor):
        return tensor

    # Convert regular tensor to symmetric memory
    symm_tensor = symm_mem.empty(
        *tensor.shape,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    symm_tensor.copy_(tensor)
    return symm_tensor


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


# Create a custom op wrapper for fake tensor support
# TODO(gmagogsfm): remove this custom op registration when torch.compile
# and make_fx support it
@torch.library.custom_op(
    "vllm_helion::copy_engine_all_reduce_w_progress",
    mutates_args=("output", "progress"),  # output and progress tensors are mutated
    device_types="cuda",
)
def _copy_engine_all_reduce_w_progress_custom_op(
    output: torch.Tensor,
    inp: torch.Tensor,
    progress: torch.Tensor,
    splits_per_rank: int,
) -> None:
    """Custom op wrapper for copy_engine_all_reduce_w_progress with fake tensor support."""
    # Check if tensor is already symmetric memory and convert if needed
    input_symm = _ensure_symmetric_memory(inp)

    copy_engine_all_reduce_w_progress(output, input_symm, progress, splits_per_rank)


@_copy_engine_all_reduce_w_progress_custom_op.register_fake
def copy_engine_all_reduce_w_progress_fake(
    output: torch.Tensor,
    inp: torch.Tensor,
    progress: torch.Tensor,
    splits_per_rank: int,
) -> None:
    """
    Fake implementation for copy_engine_all_reduce_w_progress.

    During tracing/fake mode, we just ensure the output tensor has the right shape
    and the progress tensor gets filled with the expected values.
    """
    # For shape inference, just ensure output matches input
    assert output.shape == inp.shape, (
        f"Output shape {output.shape} != input shape {inp.shape}"
    )
    assert progress.numel() >= splits_per_rank, (
        f"Progress size {progress.numel()} < splits_per_rank {splits_per_rank}"
    )
    # In fake mode, we don't actually fill the tensors, just validate shapes


# Only define the Helion kernel if Helion is available
if HELION_AVAILABLE:

    def _allreduce_add_rmsnorm_fake(
        allreduce_buf: torch.Tensor,
        residual: torch.Tensor,
        rms_gamma: torch.Tensor,
        progress: torch.Tensor,
        rms_eps: float,
        SPLITS_PER_RANK: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Custom fake implementation for allreduce_add_rmsnorm.

        Shape contract:
        - allreduce_buf: [M, K]
        - residual: [M, K]
        - rms_gamma: [K]
        - progress: [SPLITS_PER_RANK]
        - returns: tuple of (normalized_output, updated_residual) both [M, K]
        """
        M, K = allreduce_buf.size()
        out = torch.empty(
            [M, K], dtype=allreduce_buf.dtype, device=allreduce_buf.device
        )
        residual_out = torch.empty(
            [M, K], dtype=allreduce_buf.dtype, device=allreduce_buf.device
        )
        return out, residual_out

    # Apply @register_kernel to the actual Helion kernel
    @register_kernel("allreduce_add_rmsnorm", fake_impl=_allreduce_add_rmsnorm_fake)
    @helion.kernel(
        autotune_baseline_atol=0.0,
        autotune_baseline_rtol=0.0,
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
            load_eviction_policies=[
                "last",
                "first",
                "",
                "",
                "first",
                "first",
                "first",
                "",
            ],
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
    def allreduce_add_rmsnorm(
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
        out = torch.empty(
            [M, K], dtype=allreduce_buf.dtype, device=allreduce_buf.device
        )
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
    1. Converts regular tensor to symmetric memory if needed
    2. Starts an asynchronous AllReduce with progress tracking
    3. Launches a Helion kernel that processes chunks as they arrive
    4. Ensures proper stream synchronization

    Args:
        input_shared: Shared tensor across ranks to be reduced [M, K]
                     Can be either a regular tensor or symmetric memory tensor
        residual: Residual tensor to add [M, K]
        rms_gamma: RMSNorm gamma weights [K]
        rms_eps: RMSNorm epsilon for numerical stability
        splits_per_rank: Number of splits for overlapping (higher = more overlap)

    Returns:
        Tuple of (normalized_output, updated_residual) both [M, K]
        - normalized_output: RMSNorm(AllReduce(input) + residual)
        - updated_residual: AllReduce(input) + residual

    Example:
        >>> # In a distributed setting
        >>> input_tensor = torch.randn(1024, 4096, dtype=torch.bfloat16, device="cuda")
        >>> residual = torch.randn(1024, 4096, dtype=torch.bfloat16, device="cuda")
        >>> rms_gamma = torch.randn(4096, dtype=torch.bfloat16, device="cuda")
        >>> norm_out, residual_out = helion_allreduce_add_rmsnorm(
        ...     input_tensor, residual, rms_gamma
        ... )
    """
    if not HELION_AVAILABLE:
        raise ImportError(
            "Helion is not installed. Please install Helion to use "
            "helion_allreduce_add_rmsnorm. Alternatively, use FlashInfer's "
            "trtllm_allreduce_fusion or call AllReduce and RMSNorm separately."
        )

    # Create output tensors
    allreduce_out = torch.empty_like(input_shared)
    progress = torch.zeros(
        splits_per_rank,
        dtype=torch.uint32,
        device=input_shared.device,
    )

    # Perform AllReduce with progress tracking (custom op handles fake mode and symmetric memory conversion)
    torch.ops.vllm_helion.copy_engine_all_reduce_w_progress(
        allreduce_out, input_shared, progress, splits_per_rank
    )

    # Call the Helion kernel for Add + RMSNorm
    norm_out, residual_out = allreduce_add_rmsnorm(
        allreduce_out,
        residual,
        rms_gamma,
        progress,
        rms_eps,
        splits_per_rank,
    )

    # Wait for the AllReduce backend stream to complete before returning
    backend_stream = symm_mem._get_backend_stream(priority=-1)
    torch.cuda.current_stream().wait_stream(backend_stream)

    return norm_out, residual_out


@CustomOp.register("allreduce_add_rmsnorm_helion")
class AllReduceAddRMSNormHelion(HelionCustomOp):
    """
    Fused AllReduce + Add + RMSNorm with communication/compute overlap using Helion.

    This is a distributed operation that requires multi-GPU setup and symmetric memory.
    It overlaps AllReduce communication with RMSNorm computation for improved performance.

    Operation: RMSNorm(AllReduce(input) + residual)

    The operation:
    1. Splits AllReduce into chunks with progress tracking
    2. Processes each chunk with Add + RMSNorm as it arrives
    3. Overlaps communication and computation across chunks

    Shapes:
        input_shared: (num_tokens, hidden_size) - symmetric memory tensor
        residual: (num_tokens, hidden_size)
        rms_gamma: (hidden_size,)
        output: tuple of (normalized, updated_residual) both (num_tokens, hidden_size)

    Requirements:
    - Multi-GPU distributed environment (torch.distributed initialized)
    - input_shared must be allocated with symm_mem.empty() or symm_mem.zeros()
    - Helion must be enabled

    Note: When Helion is disabled, use FlashInfer's trtllm_allreduce_fusion
    or separate AllReduce + RMSNorm operations instead.
    """

    def __init__(self, splits_per_rank: int = 4):
        """
        Initialize the AllReduceAddRMSNormHelion operation.

        Args:
            splits_per_rank: Number of splits for overlapping communication
                           and computation. Higher values provide more overlap
                           but may increase overhead. Default: 4
        """
        super().__init__()
        self.splits_per_rank = splits_per_rank

    def forward_helion(
        self,
        input_shared: torch.Tensor,
        residual: torch.Tensor,
        rms_gamma: torch.Tensor,
        rms_eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Helion implementation with overlapped AllReduce.

        Args:
            input_shared: Shared tensor across ranks to be reduced [M, K]
                         Must be allocated with symm_mem.empty()
            residual: Residual tensor to add [M, K]
            rms_gamma: RMSNorm gamma weights [K]
            rms_eps: RMSNorm epsilon for numerical stability

        Returns:
            Tuple of (normalized_output, updated_residual) both [M, K]
            - normalized_output: RMSNorm(AllReduce(input) + residual)
            - updated_residual: AllReduce(input) + residual
        """
        if not HELION_AVAILABLE:
            raise ImportError(
                "Helion is not installed. Please install Helion to use "
                "AllReduceAddRMSNormHelion. Alternatively, use FlashInfer's "
                "trtllm_allreduce_fusion or call AllReduce and RMSNorm separately."
            )
        return helion_allreduce_add_rmsnorm(
            input_shared, residual, rms_gamma, rms_eps, self.splits_per_rank
        )

    def get_autotune_inputs(self) -> dict[str, tuple]:
        """
        Generate autotune inputs for hidden_size + splits combinations.

        Returns:
            Dictionary mapping config keys to input tuples
        """
        inputs = {}
        hidden_sizes = [4096, 8192]  # Only larger models use distributed
        splits_per_rank_options = [4, 8]
        batch_size = 256

        for hidden_size in hidden_sizes:
            for splits in splits_per_rank_options:
                # Create symmetric memory tensors for distributed
                allreduce_buf = torch.randn(
                    batch_size, hidden_size, dtype=torch.bfloat16, device="cuda"
                )
                residual = torch.randn(
                    batch_size, hidden_size, dtype=torch.bfloat16, device="cuda"
                )
                rms_gamma = torch.randn(
                    hidden_size, dtype=torch.bfloat16, device="cuda"
                )

                # Progress tensor for tracking AllReduce completion
                progress = torch.zeros(splits, dtype=torch.uint32, device="cuda")
                rms_eps = 1e-6

                # Key includes both hidden_size and splits
                key = f"h{hidden_size}_s{splits}"
                inputs[key] = (
                    allreduce_buf,
                    residual,
                    rms_gamma,
                    progress,
                    rms_eps,
                    splits,
                )

        return inputs

    def get_best_config(
        self, model_config, available_configs: dict[str, "helion.Config"]
    ):
        """
        Select config using hidden_size + splits_per_rank with fallback.

        Args:
            model_config: vLLM ModelConfig instance
            available_configs: Dictionary mapping config keys to loaded Helion configs

        Returns:
            Best matching Helion config from available_configs, or None if no suitable match
        """
        if not available_configs:
            return None

        target_hidden_size = model_config.get_hidden_size()
        splits = getattr(self, "splits_per_rank", 4)

        # Try exact match first
        exact_key = f"h{target_hidden_size}_s{splits}"
        if exact_key in available_configs:
            return available_configs[exact_key]

        # Fallback: try different splits for same hidden_size
        for fallback_splits in [4, 8]:
            if fallback_splits != splits:
                fallback_key = f"h{target_hidden_size}_s{fallback_splits}"
                if fallback_key in available_configs:
                    logger.warning(
                        f"No config for splits={splits}, using splits={fallback_splits}"
                    )
                    return available_configs[fallback_key]

        # Fallback: try closest hidden_size from available configs
        try:
            # Parse available hidden sizes from config keys (format: h{size}_s{splits})
            available_sizes = []
            for key in available_configs:
                try:
                    if key.startswith("h") and "_s" in key:
                        size_str = key.split("_s")[0][1:]  # Remove 'h' prefix
                        size = int(size_str)
                        available_sizes.append((size, key))
                except (ValueError, IndexError):
                    continue  # Skip malformed keys

            if not available_sizes:
                # If no parseable keys, just return first available config
                return next(iter(available_configs.values()))

            # Find closest hidden size, preferring exact splits match
            best_match = None
            best_distance = float("inf")

            for size, key in available_sizes:
                distance = abs(size - target_hidden_size)

                # Prefer configs with matching splits, then by distance
                key_splits = int(key.split("_s")[1]) if "_s" in key else 4
                splits_match = key_splits == splits

                if distance < best_distance or (
                    distance == best_distance
                    and splits_match
                    and (best_match is None or not best_match[2])
                ):
                    best_match = (size, key, splits_match)
                    best_distance = distance

            if best_match:
                closest_size, best_key, _ = best_match
                if closest_size != target_hidden_size:
                    logger.warning(
                        f"No config for hidden_size={target_hidden_size}, "
                        f"using closest match: {closest_size}"
                    )
                return available_configs[best_key]

        except Exception:
            # If parsing fails, just return the first available config
            return next(iter(available_configs.values()))

        return None

    @property
    def helion_kernel(self):
        """The Helion kernel function for autotuning."""
        if HELION_AVAILABLE:
            return allreduce_add_rmsnorm._helion_kernel
        return None


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

        # Initialize the CustomOp
        self.op = AllReduceAddRMSNormHelion(splits_per_rank=4)

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

        # Create op with the correct splits_per_rank for this test
        op = AllReduceAddRMSNormHelion(splits_per_rank=splits_per_rank)
        norm_out, residual_out = op.forward_helion(
            input_helion,
            residual_helion,
            gamma,
            eps,
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
