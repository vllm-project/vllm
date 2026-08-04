# SPDX-License-Identifier: Apache-2.0
"""Non-GPU context abstractions and utilities for multiprocess transport.

This module provides:
- ``EngineDrivenContextMetadata``: layout metadata dataclass for non-CUDA workers.
- ``EngineDrivenContext``: abstract base class with a two-phase prepare/commit
  interface for CPU-side KV data transfer. Concrete implementations (e.g.
  ``EngineDrivenContextPickle``) each decide *how* data is serialised and transported.
- ``create_engine_driven_context()``: factory that returns the appropriate
  ``EngineDrivenContext`` subclass.
- ``compute_kv_layout``, ``gather_paged_kv_to_cpu``, ``scatter_cpu_to_paged_kv``:
  shared gather/scatter utilities used by all concrete implementations.
"""

# Future
from __future__ import annotations

# Standard
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING
import inspect

# Third Party
import numpy as np
import torch

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.mq import MessageQueueClient

if TYPE_CHECKING:
    # First Party
    import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Global capability flag: does lmc_ops.multi_layer_block_kv_transfer accept
# list[torch.Tensor] directly for lmcache_objects_ptrs, or only list[int]?
#
# We inspect the function signature once at import time. If the annotation
# for ``lmcache_objects_ptrs`` includes ``Tensor``, the op can handle tensors
# natively and we pass them through. Otherwise (annotation is list[int], or
# inspect fails entirely) we must convert tensors to data pointers before
# calling.
# ---------------------------------------------------------------------------
def _detect_block_transfer_accepts_tensor() -> bool:
    """Return True if lmc_ops.multi_layer_block_kv_transfer accepts
    list[torch.Tensor] for its lmcache_objects_ptrs parameter."""
    try:
        # First Party
        import lmcache.c_ops as _lmc_ops

        fn = _lmc_ops.multi_layer_block_kv_transfer

        # Attempt: use inspect.signature (works on newer pybind11 builds)
        # Assumptions: if lmcache_objects_ptrs accepts tensors,
        # it's fallback path, and we do not convert tensors to ptrs explicitly.
        # TODO: String matching on annotations is fragile. Wait for lmc_ops to
        # expose a direct version flag (e.g., lmc_ops.__version__) or
        # an explicit capability boolean.
        try:
            sig = inspect.signature(fn)
            param = sig.parameters.get("lmcache_objects_ptrs")
            if param is not None and param.annotation is not inspect.Parameter.empty:
                ann_str = str(param.annotation)
                if "Tensor" in ann_str:
                    return True
                # Annotation exists but no Tensor mention → ptr-only
                return False
        except (ValueError, TypeError):
            pass

    except Exception:
        # Import failed or any other error → conservative: assume ptr-only
        pass

    # Default: inspect failed or lmc_ops not available → assume ptr-only
    return False


_LMC_OPS_BLOCK_TRANSFER_ACCEPTS_TENSOR: bool = _detect_block_transfer_accepts_tensor()
"""If True, ``lmc_ops.multi_layer_block_kv_transfer`` accepts
``list[torch.Tensor]`` directly for ``lmcache_objects_ptrs``.
If False, callers must convert tensors to ``list[int]`` data pointers."""

logger.info(
    "multi_layer_block_kv_transfer mode: %s",
    "tensor" if _LMC_OPS_BLOCK_TRANSFER_ACCEPTS_TENSOR else "ptr",
)


def _tensors_to_ptrs(tensors: list[torch.Tensor]) -> list[int]:
    """Convert a list of tensors to a list of their data_ptr() values."""
    return [t.data_ptr() for t in tensors]


# ---------------------------------------------------------------------------


@dataclass
class EngineDrivenContextMetadata:
    """Non-GPU context layout metadata for non-CUDA workers.

    Attributes:
        layout_desc: Memory layout descriptor used to interpret chunk payloads.
        block_size: Number of tokens per paged block.
        use_mla: Whether the worker KV format is MLA.
    """

    layout_desc: MemoryLayoutDesc
    block_size: int
    use_mla: bool


class EngineDrivenContext(ABC):
    """Abstract base class for CPU-side KV data transfer contexts.

    All concrete implementations share a common message-queue client and
    expose a uniform two-phase ``prepare/commit`` interface so that the
    worker adapter is implementation-agnostic.

    Args:
        metadata: Layout metadata describing the chunk format.
        mq_client: Message-queue client used for server communication.
        mq_timeout: Timeout in seconds for blocking MQ requests.
    """

    def __init__(
        self,
        metadata: EngineDrivenContextMetadata,
        mq_client: MessageQueueClient,
        mq_timeout: float,
    ) -> None:
        self.metadata = metadata
        self.mq_client = mq_client
        self.mq_timeout = mq_timeout

    @property
    def layout_desc(self) -> MemoryLayoutDesc:
        """The memory layout descriptor for this context."""
        return self.metadata.layout_desc

    @abstractmethod
    def prepare_store(
        self, key: IPCCacheServerKey, instance_id: int
    ) -> tuple[list[torch.Tensor], list[int]] | None:
        """Prepare SHM buffers for a store operation.

        Returns:
            None: pickle mode — no pre-allocated buffers. Caller gathers all
                chunks to CPU itself and sends the serialized data via
                commit_store.
            ([], []): SHM mode but all chunks already cached. Caller should
                skip gather and commit entirely.
            (tensors, chunk_indices): SHM mode with new chunks to write.
                - tensors[i] is a writable SHM-backed buffer for one chunk.
                - chunk_indices[i] is the position of that chunk in the full
                  block_ids sequence (e.g. [0, 2] means only chunks 0 and 2
                  need writing; chunk 1 is already cached).
                Caller gathers only these chunks into the provided tensors,
                then calls commit_store with empty payload.
        """
        ...

    @abstractmethod
    def commit_store(
        self, key: IPCCacheServerKey, instance_id: int, chunks: list[torch.Tensor]
    ) -> bool:
        """Commit store. Pickle: serialize and send. Shm: notify server."""
        ...

    @abstractmethod
    def prepare_retrieve(
        self, key: IPCCacheServerKey, instance_id: int
    ) -> list[torch.Tensor] | None:
        """Prepare retrieve. Returns chunks or shm views, or None on miss."""
        ...

    @abstractmethod
    def commit_retrieve(self, key: IPCCacheServerKey, instance_id: int) -> bool:
        """Commit retrieve. Pickle: no-op. Shm: release read locks."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by this context."""
        ...


def create_engine_driven_context(
    metadata: EngineDrivenContextMetadata,
    mq_client: MessageQueueClient,
    mq_timeout: float,
    shm_name: str,
    pool_size: int,
    *,
    use_pickle: bool = False,
) -> EngineDrivenContext:
    """Factory that returns the appropriate :class:`EngineDrivenContext` implementation.

    Returns SHM-based implementation when shared-memory pool information is
    available; otherwise falls back to the pickle-based implementation.
    If SHM initialization fails for any reason (e.g. segment not found,
    permission error), gracefully falls back to pickle transport.

    Args:
        metadata: Layout metadata for the non-GPU context.
        mq_client: Message-queue client for server communication.
        mq_timeout: Timeout in seconds for blocking MQ requests.
        shm_name: Shared-memory segment name. Empty values force pickle mode.
        pool_size: Shared-memory pool size in bytes. Non-positive values force
            pickle mode.
        use_pickle: Explicitly use pickle transport even when SHM info is
            available.

    Returns:
        A concrete :class:`EngineDrivenContext` instance.
    """
    if not shm_name or pool_size <= 0:
        use_pickle = True

    if not use_pickle:
        # Local
        from .shm import EngineDrivenContextShm

        try:
            logger.info(
                "Creating EngineDrivenContextShm (shm_name=%s, pool_size=%d)",
                shm_name,
                pool_size,
            )
            return EngineDrivenContextShm(
                metadata, mq_client, mq_timeout, shm_name, pool_size
            )
        except Exception:
            logger.warning(
                "Failed to initialize SHM context (shm_name=%s), "
                "falling back to pickle transport",
                shm_name,
                exc_info=True,
            )

    # Local
    from .pickle import EngineDrivenContextPickle

    logger.info("Creating EngineDrivenContextPickle (pickle transport)")
    return EngineDrivenContextPickle(metadata, mq_client, mq_timeout)


# ---------------------------------------------------------------------------
# Shared gather / scatter utilities
# ---------------------------------------------------------------------------


def compute_kv_layout(
    kv_caches: dict[str, torch.Tensor],
    layout_hints: LayoutHints | None = None,
) -> tuple[int, int, int, str, "lmc_ops.EngineKVFormat", int]:
    """Compute KV layout metadata from KV tensors.

    Args:
        kv_caches: Per-layer KV tensor mapping.
        layout_hints: Optional engine layout hints.

    Returns:
        Tuple of ``(block_size, num_layers, hidden_dim_size, dtype_str,``
        ``engine_kv_format, kv_size)``. ``hidden_dim_size`` is the per-token
        object width (``num_heads * head_size``; for fused-K/V formats the
        head size is the packed ``2 * head_size``) and ``kv_size`` is the
        object plane count (1 for MLA and fused-K/V, else 2).

    Raises:
        ValueError: If ``kv_caches`` is empty.
    """
    # First Party
    from lmcache.v1.gpu_connector.utils import (
        get_block_size,
        get_head_size,
        get_kv_size,
        get_num_heads,
        get_num_layers,
        normalize_kv_and_discover_format,
    )

    tensors = list(kv_caches.values())
    if not tensors:
        raise ValueError("kv_caches is empty. Cannot compute KV layout.")

    engine_kv_format, normalized = normalize_kv_and_discover_format(
        tensors, EngineType.VLLM, layout_hints=layout_hints
    )
    block_size = get_block_size(normalized, engine_kv_format)
    num_layers = get_num_layers(normalized, engine_kv_format)
    hidden_dim_size = get_num_heads(normalized, engine_kv_format) * get_head_size(
        normalized, engine_kv_format
    )
    kv_size = get_kv_size(normalized, engine_kv_format)
    dtype_str = str(tensors[0].dtype).replace("torch.", "")
    return (
        block_size,
        num_layers,
        hidden_dim_size,
        dtype_str,
        engine_kv_format,
        kv_size,
    )


def gather_paged_kv_to_cpu(
    kv_caches: dict[str, torch.Tensor],
    block_ids: list[int],
    blocks_per_chunk: int,
    layout_hints: LayoutHints | None = None,
    engine_kv_format: "lmc_ops.EngineKVFormat" | None = None,
    out: list[torch.Tensor] | None = None,
    chunk_indices: list[int] | None = None,
) -> list[torch.Tensor]:
    """Gather paged KV blocks into CPU chunk tensors.

    Args:
        kv_caches: Per-layer KV tensor mapping.
        block_ids: Flattened block IDs for all chunks.
        blocks_per_chunk: Number of paged blocks in one LMCache chunk.
        layout_hints: Optional engine layout hints.
        engine_kv_format: Optional pre-detected KV format.
        out: Optional pre-allocated output tensors.  If provided, length
            must be at least ``len(chunk_indices)`` when ``chunk_indices``
            is given, or the total number of chunks otherwise.  Any extra
            buffers beyond the number of gathered chunks are ignored.
        chunk_indices: Optional list of chunk positions (into the full
            ``block_ids`` sequence) to gather.  When provided together with
            ``out``, only those chunks are gathered and written into
            ``out[i]`` in order.  When ``None``, all chunks are gathered
            (backward-compatible behaviour).

    Returns:
        List of CPU tensors, one per chunk. For split-K/V formats each chunk
        has shape ``[2, num_layers, chunk_tokens, hidden_dim]`` where
        dimension ``0`` stores ``(K, V)``. For single-plane formats
        (``kv_size == 1``: MLA and fused-K/V) each chunk has shape
        ``[num_layers, chunk_tokens, num_heads * head_size]`` — for fused
        formats ``head_size`` is the packed ``2 * head_size``.

    Raises:
        ValueError: If ``out`` is provided with fewer buffers than the number
            of gathered chunks.
    """
    # First Party
    from lmcache.v1.gpu_connector.utils import (
        get_block_size,
        get_head_size,
        get_kv_size,
        get_num_blocks,
        get_num_heads,
        get_num_layers,
        make_page_buffer_shape_desc,
        normalize_kv_and_discover_format,
    )
    import lmcache.c_ops as lmc_ops

    tensors = list(kv_caches.values())
    fmt, normalized = normalize_kv_and_discover_format(
        tensors, EngineType.VLLM, layout_hints=layout_hints
    )
    if engine_kv_format is None:
        engine_kv_format = fmt

    block_size = get_block_size(normalized, engine_kv_format)
    num_layers = get_num_layers(normalized, engine_kv_format)
    content_size = get_num_heads(normalized, engine_kv_format) * get_head_size(
        normalized, engine_kv_format
    )
    num_blocks = get_num_blocks(normalized, engine_kv_format)
    num_chunks = len(block_ids) // blocks_per_chunk
    chunk_tokens = blocks_per_chunk * block_size

    shape_desc = make_page_buffer_shape_desc(
        normalized,
        engine_kv_format,
        layer_idx=0,
        num_layers_in_group=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
    )

    iter_indices = (
        list(chunk_indices) if chunk_indices is not None else list(range(num_chunks))
    )
    # Require at least one output buffer per gathered chunk. Extra trailing
    # buffers are ignored (see ``chunks = out[: len(iter_indices)]`` below),
    # mirroring the scatter-side length check for consistency.
    if out is not None and len(out) < len(iter_indices):
        raise ValueError(
            f"out length ({len(out)}) must be at least the number of "
            f"gathered chunks ({len(iter_indices)})"
        )

    # Determine if pinned memory is strictly required
    # (only for the compiled C++ path which does not accept tensor)
    requires_pinned = not _LMC_OPS_BLOCK_TRANSFER_ACCEPTS_TENSOR
    needs_staging = False
    staged_chunks = []

    if out is None:
        # One object plane per K/V entry: MLA and fused-K/V formats
        # (kv_size == 1) store a single plane, split formats store K and V.
        if get_kv_size(normalized, engine_kv_format) == 1:
            chunks = [
                torch.empty(
                    (num_layers, chunk_tokens, content_size),
                    dtype=tensors[0].dtype,
                    device=torch.device("cpu"),
                    pin_memory=requires_pinned,
                )
                for _ in iter_indices
            ]
        else:
            chunks = [
                torch.empty(
                    (2, num_layers, chunk_tokens, content_size),
                    dtype=tensors[0].dtype,
                    device=torch.device("cpu"),
                    pin_memory=requires_pinned,
                )
                for _ in iter_indices
            ]
    else:
        _target_out = out[: len(iter_indices)]

        if requires_pinned and not all(t.is_pinned() for t in _target_out):
            # Core fallback: Unpinned memory (e.g., IPC Shared Memory) detected.
            # We cannot dynamically call `.pin_memory()` on `out` because it
            # would allocate new tensors, breaking the caller's expectation
            # of an in-place update. Instead, we allocate a temporary pinned
            # staging buffer for the C++ kernel to write to safely.
            logger.warning(
                "Unpinned memory detected in 'out' during "
                "gather_paged_kv_to_cpu (likely Shared Memory). "
                "Using an internal pinned staging buffer, which "
                "adds a CPU memory copy overhead."
            )
            needs_staging = True
            staged_chunks = [torch.empty_like(t, pin_memory=True) for t in _target_out]
            chunks = (
                staged_chunks  # Point to the safe staging buffer for the H2D transfer
            )
        else:
            # Ideal case: Memory is pinned, or we are using Python fallback.
            # Ignore any extra trailing buffers beyond what we actually gather so
            # the kernel's ``total_blocks % num_objects`` invariant still holds.
            # Return ``out`` unchanged when no trimming is needed so the in-place
            # fill contract (result is out) is preserved.
            if len(out) == len(iter_indices):
                chunks = out
            else:
                chunks = out[: len(iter_indices)]

    selected_block_ids: list[int] = []
    for chunk_idx in iter_indices:
        selected_block_ids.extend(
            block_ids[chunk_idx * blocks_per_chunk : (chunk_idx + 1) * blocks_per_chunk]
        )

    if selected_block_ids:
        if _LMC_OPS_BLOCK_TRANSFER_ACCEPTS_TENSOR:
            # Python fallback: accepts tensor list directly for all params.
            paged_arg = normalized
            objs_arg = chunks
            block_ids_arg = selected_block_ids

            # call kernel in one shot
            lmc_ops.multi_layer_block_kv_transfer(
                paged_arg,
                objs_arg,
                block_ids_arg,
                tensors[0].device,
                lmc_ops.TransferDirection.D2H,
                shape_desc,
                chunk_tokens,
                engine_kv_format,
                0,
            )

        else:
            # Compiled C++/CUDA/XPU: requires int64 pointer tensor and list[int].
            _ptrs_np = np.array(
                [t.data_ptr() for t in normalized],  # type: ignore[union-attr]
                dtype=np.uint64,
            ).view(np.int64)
            paged_arg = torch.from_numpy(_ptrs_np).to(device=tensors[0].device)

            # This safely points to either the pre-pinned chunks
            # OR the temporary staged_chunks
            objs_arg = _tensors_to_ptrs(chunks)

            block_ids_arg = torch.tensor(
                selected_block_ids, dtype=torch.int64, device=tensors[0].device
            )

            # Split transfer to respect CUDA kernel's object count limitation
            MAX_OBJECTS = 4
            req_blocks_per_obj = blocks_per_chunk
            total_objects = len(objs_arg)

            for i in range(0, total_objects, MAX_OBJECTS):
                # Slice object pointers and corresponding block IDs
                batch_objs_ptrs = objs_arg[i : i + MAX_OBJECTS]

                start_block = i * req_blocks_per_obj
                end_block = min(
                    (i + MAX_OBJECTS) * req_blocks_per_obj, len(selected_block_ids)
                )
                batch_blocks = block_ids_arg[start_block:end_block]

                # Execute batched transfer
                lmc_ops.multi_layer_block_kv_transfer(
                    paged_arg,
                    batch_objs_ptrs,
                    batch_blocks,
                    tensors[0].device,
                    lmc_ops.TransferDirection.D2H,
                    shape_desc,
                    chunk_tokens,
                    engine_kv_format,
                    0,
                )

    # --- Final reconciliation ---
    # If we used a staging buffer to protect unpinned shared memory,
    # we now copy the gathered data back into the caller's original tensors.
    if needs_staging:
        assert out is not None
        # The CPU MUST block and wait for the GPU ONLY when a temporary
        # staging buffer is used. This is because the CPU needs to immediately
        # read this data for the memory copy below.
        torch_dev.synchronize()

        for dst, src in zip(_target_out, staged_chunks, strict=False):
            dst.copy_(src)  # High-speed CPU-to-CPU memory copy

        if len(out) == len(iter_indices):
            chunks = out
        else:
            chunks = _target_out

    # Fast path: The async GPU copy might still be in progress.
    # We intentionally omit synchronization here for performance.
    # WARNING: The caller MUST explicitly call `torch_dev.synchronize()`
    # before consuming these chunks to ensure data validity.

    return chunks


def scatter_cpu_to_paged_kv(
    kv_caches: dict[str, torch.Tensor],
    block_ids: list[int],
    chunks: list[torch.Tensor],
    blocks_per_chunk: int,
    skip_first_n_tokens: int = 0,
    layout_hints: LayoutHints | None = None,
    engine_kv_format: "lmc_ops.EngineKVFormat" | None = None,
) -> None:
    """Scatter CPU chunk tensors back into paged KV tensors.

    Args:
        kv_caches: Per-layer KV tensor mapping to write into.
        block_ids: Flattened destination block IDs for all chunks.  Length
            must be at least ``len(chunks) * blocks_per_chunk``; any extra
            trailing block IDs are ignored.
        chunks: List of CPU chunk tensors (as returned by
            :func:`gather_paged_kv_to_cpu`).
        blocks_per_chunk: Number of paged blocks in one LMCache chunk.
        skip_first_n_tokens: Token prefix to skip when scattering.  Must be a
            multiple of ``block_size``; non-aligned values are rounded down
            to the nearest whole block and an error is logged (matching the
            GPU transfer path).
        layout_hints: Optional engine layout hints.
        engine_kv_format: Optional pre-detected KV format.

    Raises:
        ValueError: If ``block_ids`` is shorter than
            ``len(chunks) * blocks_per_chunk``.
    """
    # First Party
    from lmcache.v1.gpu_connector.utils import (
        get_block_size,
        get_num_blocks,
        get_num_layers,
        make_page_buffer_shape_desc,
        normalize_kv_and_discover_format,
    )
    import lmcache.c_ops as lmc_ops

    if not chunks:
        return
    # Require enough block IDs to cover every chunk. Extra trailing block IDs
    # are ignored by the per-chunk slicing below, mirroring the gather-side
    # ``out`` length check for consistency.
    if len(block_ids) < len(chunks) * blocks_per_chunk:
        raise ValueError(
            f"block_ids length ({len(block_ids)}) must be at least "
            f"len(chunks) ({len(chunks)}) * blocks_per_chunk "
            f"({blocks_per_chunk})"
        )

    tensors = list(kv_caches.values())
    fmt, normalized = normalize_kv_and_discover_format(
        tensors, EngineType.VLLM, layout_hints=layout_hints
    )
    if engine_kv_format is None:
        engine_kv_format = fmt

    block_size = get_block_size(normalized, engine_kv_format)
    num_layers = get_num_layers(normalized, engine_kv_format)
    num_blocks = get_num_blocks(normalized, engine_kv_format)
    chunk_tokens = blocks_per_chunk * block_size

    # Block-level transfer can only skip whole blocks. A non-aligned prefix is
    # rounded down to the nearest block (matching the lmcache-driven path in
    # lmcache_driven_transfer.py) rather than raising, so a slightly misaligned skip
    # degrades gracefully instead of failing the whole retrieve.
    if skip_first_n_tokens % block_size != 0:
        logger.error(
            "skip_first_n_tokens (%d) is not block-aligned (block_size=%d); "
            "rounding down to %d blocks",
            skip_first_n_tokens,
            block_size,
            skip_first_n_tokens // block_size,
        )
    skip_prefix_n_blocks = skip_first_n_tokens // block_size

    shape_desc = make_page_buffer_shape_desc(
        normalized,
        engine_kv_format,
        layer_idx=0,
        num_layers_in_group=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
    )

    selected_block_ids: list[int] = []
    for chunk_idx in range(len(chunks)):
        selected_block_ids.extend(
            block_ids[chunk_idx * blocks_per_chunk : (chunk_idx + 1) * blocks_per_chunk]
        )

    if not selected_block_ids:
        return

    if _LMC_OPS_BLOCK_TRANSFER_ACCEPTS_TENSOR:
        # Python fallback: accepts tensor list directly for all params.
        paged_arg = normalized
        objs_arg = chunks
        block_ids_arg = selected_block_ids

        lmc_ops.multi_layer_block_kv_transfer(
            paged_arg,
            objs_arg,
            block_ids_arg,
            tensors[0].device,
            lmc_ops.TransferDirection.H2D,
            shape_desc,
            chunk_tokens,
            engine_kv_format,
            skip_prefix_n_blocks,
        )
    else:
        # assuming this is c ops path which requires pin memory
        # TODO: may have a better approach here
        # Defensive check: Ensure all incoming CPU chunks are pinned memory.
        # Otherwise, the underlying CUDA kernel may throw an Illegal
        # Memory Access error during H2D transfer.
        if not all(chunk.is_pinned() for chunk in chunks):
            logger.warning(
                "Received unpinned CPU tensors in scatter_cpu_to_paged_kv. "
                "Dynamically pinning memory now, which may incur additional"
                "synchronization overhead."
            )
            chunks = [
                chunk.pin_memory() if not chunk.is_pinned() else chunk
                for chunk in chunks
            ]

        # Compiled C++/CUDA/XPU: requires int64 pointer tensor and list[int].
        _ptrs_np = np.array(
            [t.data_ptr() for t in normalized],  # type: ignore[union-attr]
            dtype=np.uint64,
        ).view(np.int64)
        paged_arg = torch.from_numpy(_ptrs_np).to(device=tensors[0].device)
        objs_arg = _tensors_to_ptrs(chunks)
        block_ids_arg = torch.tensor(
            selected_block_ids, dtype=torch.int64, device=tensors[0].device
        )

        # Batched transfer to satisfy cuda's limitation (max 4 objects)
        MAX_OBJECTS = 4
        req_blocks_per_obj = (
            blocks_per_chunk  # Each chunk corresponds to one object's blocks
        )
        total_chunks = len(chunks)

        for i in range(0, total_chunks, MAX_OBJECTS):
            # Slice objects and block IDs for this batch
            batch_objs_ptrs = objs_arg[i : i + MAX_OBJECTS]

            start_block = i * req_blocks_per_obj
            end_block = min(
                (i + MAX_OBJECTS) * req_blocks_per_obj, len(selected_block_ids)
            )
            batch_blocks = block_ids_arg[start_block:end_block]

            # Execute transfer for this batch
            lmc_ops.multi_layer_block_kv_transfer(
                paged_arg,
                batch_objs_ptrs,
                batch_blocks,
                tensors[0].device,
                lmc_ops.TransferDirection.H2D,
                shape_desc,
                chunk_tokens,
                engine_kv_format,
                skip_prefix_n_blocks if i == 0 else 0,
            )
    # Fast path: The async GPU copy might still be in progress.
    # We intentionally omit synchronization here for performance.
    # WARNING: The caller MUST explicitly call `torch_dev.synchronize()`
    # before consuming these chunks to ensure data validity.
