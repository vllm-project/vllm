# SPDX-License-Identifier: Apache-2.0
"""CPU-only cache context for platforms without CUDA GPUs.

This module sits next to :mod:`lmcache.v1.platform.cuda.cache_context`
under :mod:`lmcache.v1.platform` and provides the same public API as
:class:`~lmcache.v1.platform.cuda.cache_context.GPUCacheContext` but
keeps all tensors on CPU. Stream / Event objects are provided by
:class:`~lmcache.v1.platform.cpu.stub_cpu_device.StubStream` so
CPU-only hosts never import ``cupy`` or instantiate a real CUDA
stream object.

The platform-agnostic dispatcher ``create_cache_context`` lives in
:mod:`lmcache.v1.platform.cache_context`.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Sequence
from typing import TYPE_CHECKING
import os

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.utils import (
    LayoutHints,
    get_group_data_ptrs,
    normalize_and_discover_per_layer_formats,
)
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
from lmcache.v1.multiprocess.custom_types import KVCache
from lmcache.v1.multiprocess.group_view import engine_group_layer_indices
from lmcache.v1.platform.base.cache_context import BaseCacheContext
from lmcache.v1.platform.cpu.stub_cpu_device import StubStream

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.multiprocess.group_view import EngineGroupInfo

logger = init_logger(__name__)


class CPUCacheContext(BaseCacheContext):
    """CPU-only cache context with the same public API as
    :class:`GPUCacheContext`.

    All tensors live on CPU. CUDA streams and cupy streams are
    replaced by :class:`StubStream` no-op objects so callers can keep
    using ``stream.synchronize()`` / ``wait_event(...)`` etc. without
    branching on the active backend.

    KV cache tensors are reconstructed from the
    :class:`CpuShmTensorWrapper` instances sent by the client over
    POSIX shared memory -- the server does **not** allocate the KV
    cache itself. This mirrors the GPU-mode CUDA-IPC flow where the
    client owns the buffers and the server only maps them.
    """

    device_type = "cpu"

    def __init__(
        self,
        kv_caches: KVCache,
        lmcache_tokens_per_chunk: int = 256,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: "Sequence[EngineGroupInfo]" = (),
        engine_type: EngineType = EngineType.VLLM,
        separate_object_groups: bool = True,
        full_sw_kv: bool = False,
    ) -> None:
        if not kv_caches:
            raise ValueError(
                "CPUCacheContext requires a non-empty list of "
                "CpuShmTensorWrapper; the legacy server-side "
                "self-allocation path has been removed."
            )

        # First Party
        from lmcache.v1.platform.cuda.cache_context import (
            unwrap_kv_cache_tensors,
        )

        unwrapped = unwrap_kv_cache_tensors(kv_caches)
        self.device_ = torch.device("cpu")

        # Discover layout & build KV layer groups via the same path
        # GPUCacheContext uses, so we don't need to hand-roll any
        # PageBufferShapeDesc here. ``layout_hints`` / ``engine_type``
        # are forwarded so the signature matches GPUCacheContext.
        (
            kv_caches_normalized,
            engine_kv_formats,
        ) = normalize_and_discover_per_layer_formats(
            unwrapped,
            engine_group_layer_indices(engine_group_infos),
            engine_type,
            layout_hints,
        )
        kv_caches_list: list[torch.Tensor] = list(kv_caches_normalized)
        num_layers_val = len(engine_kv_formats)
        kv_layer_groups_manager = KVLayerGroupsManager(
            kv_caches_list,
            engine_kv_formats=engine_kv_formats,
            engine_group_infos=engine_group_infos,
            lmcache_tokens_per_chunk=lmcache_tokens_per_chunk,
            separate_object_groups=separate_object_groups,
        )
        # Set full_sw_kv before the temp buffer / object groups are sized so
        # both use the full (un-windowed) per-chunk geometry (mirrors
        # GPUCacheContext).
        if full_sw_kv:
            kv_layer_groups_manager.enable_full_sw_kv()

        # Pre-allocated block IDs buffer (CPU).
        _MAX_BLOCK_IDS = 1_000_000
        block_ids_buffer = torch.empty(_MAX_BLOCK_IDS, dtype=torch.long)

        super().__init__(
            kv_caches=kv_caches_list,
            device=self.device_,
            num_layers=num_layers_val,
            kv_layer_groups_manager=kv_layer_groups_manager,
            block_ids_buffer=block_ids_buffer,
            lmcache_tokens_per_chunk=lmcache_tokens_per_chunk,
        )

        # Per-group KV pointer tensors (CPU). Reuse the same helper
        # GPUCacheContext relies on so the layout matches exactly.
        self.group_kv_pointers_: list[torch.Tensor] = [
            torch.tensor(
                get_group_data_ptrs(
                    self.kv_caches_,
                    self.get_engine_kv_format(idx),
                    group.layer_indices,
                ),
                dtype=torch.long,
            )
            for idx, group in enumerate(self.kv_layer_groups_manager_.kernel_groups)
        ]

        self.kv_cache_pointers_ = torch.tensor(
            [t.data_ptr() for t in self.kv_caches_], dtype=torch.long
        )

        # Temporary buffer for transfers (same layout as
        # GPUCacheContext but on CPU).
        self._max_batch_size = 4
        self.tmp_chunk_group_offsets_: list[int] = [0]
        for group_idx, group in enumerate(self.kv_layer_groups_manager_.kernel_groups):
            shape = self.get_kv_buffer_shape(lmcache_tokens_per_chunk, group_idx)
            byte_size = shape.numel() * group.dtype.itemsize
            self.tmp_chunk_group_offsets_.append(
                self.tmp_chunk_group_offsets_[-1] + byte_size
            )
        self.tmp_chunk_bytes_ = self.tmp_chunk_group_offsets_[-1]
        # Buffer lives on CPU; keep the attribute name aligned with the
        # context to avoid GPU-prefixed naming bleeding into a CPU-only
        # class. The public ``get_tmp_gpu_buffer_flat`` method name is
        # preserved so ``server.py`` can duck-type across backends.
        self.tmp_cpu_buffer_ = torch.empty(
            self.tmp_chunk_bytes_ * self.max_batch_size,
            dtype=torch.uint8,
        )

        # Mock streams. ``StubStream`` already implements the small
        # subset of the API server-side code uses (``synchronize``,
        # ``wait_event``, ``record_event`` ...), so we never import
        # cupy or instantiate a real CUDA stream object here.
        self.cuda_stream_: StubStream = StubStream(device="cpu")
        self.cupy_stream_: StubStream = self.cuda_stream_
        self.high_priority_cuda_stream_: StubStream = StubStream(
            device="cpu", priority=0
        )
        self.high_priority_cupy_stream_: StubStream = self.high_priority_cuda_stream_

        # Sanity-check: warn if /dev/shm looks too small for the
        # registered KV cache. Only meaningful on Linux where
        # /dev/shm is the default tmpfs backing POSIX SHM.
        self._check_shm_capacity()

        logger.info(
            "CPUCacheContext: %d layers, %d blocks, dtype=%s (shm-backed)",
            self.num_layers_,
            self.num_blocks,
            self.kv_caches_[0].dtype,
        )

    # -- Internal helpers --

    _SHM_PATH = "/dev/shm"

    def _check_shm_capacity(self) -> None:
        """Warn if /dev/shm free space is smaller than the KV cache."""
        if not os.path.isdir(self._SHM_PATH):
            return
        try:
            st = os.statvfs(self._SHM_PATH)
        except OSError:
            return
        free_bytes = st.f_bavail * st.f_frsize
        kv_bytes = sum(t.numel() * t.element_size() for t in self.kv_caches_)
        if kv_bytes > free_bytes:
            logger.warning(
                "Insufficient /dev/shm space for CPU KV cache: "
                "need %d bytes but only %d bytes available. "
                "Consider increasing the size of /dev/shm "
                "(e.g. mount -o remount,size=<N>G /dev/shm).",
                kv_bytes,
                free_bytes,
            )

    def close(self) -> None:
        """Release resources. No-op for CPU context (no GDS staging buffer)."""
        pass

    # -- Properties (same API as GPUCacheContext) --

    @property
    def max_batch_size(self) -> int:
        """Returns the maximum number of concurrent batches."""
        return self._max_batch_size

    @property
    def kv_pointers(self) -> torch.Tensor:
        """Returns a tensor of KV cache data pointers."""
        return self.kv_cache_pointers_

    @property
    def stream(self) -> StubStream:
        """Returns the (mock) CUDA stream."""
        return self.cuda_stream_

    @property
    def cupy_stream(self) -> StubStream:
        """Returns the (mock) external stream."""
        return self.cupy_stream_

    @property
    def high_priority_stream(self) -> StubStream:
        """Returns the (mock) high-priority CUDA stream."""
        return self.high_priority_cuda_stream_

    @property
    def high_priority_cupy_stream(self) -> StubStream:
        """Returns the (mock) high-priority external stream."""
        return self.high_priority_cupy_stream_

    @property
    def block_size(self) -> int:
        """Returns the block size (tokens per block)."""
        return self.kv_layer_groups_manager_.kernel_groups[0].shape_desc.bs

    @property
    def group_slots_per_blocks(self) -> list[int]:
        """Per-group physical slot count (``shape_desc.bs``) in group
        order."""
        return [
            group.shape_desc.bs for group in self.kv_layer_groups_manager_.kernel_groups
        ]

    def blocks_for_tokens(self, num_logical_tokens: int, group_idx: int) -> int:
        """Number of blocks that span *num_logical_tokens* for a group.

        Mirrors :meth:`GPUCacheContext.blocks_for_tokens`.
        """
        group = self.kv_layer_groups_manager_.kernel_groups[group_idx]
        physical_slots = (
            num_logical_tokens * group.slots_per_block // group.tokens_per_block
        )
        return physical_slots // group.shape_desc.bs

    def get_group_kv_pointers(self, group_idx: int) -> torch.Tensor:
        """Returns the KV cache pointer tensor for the given group."""
        return self.group_kv_pointers_[group_idx]

    def get_kernel_group_kv_pointers(self, kernel_group_idx: int) -> torch.Tensor:
        """Returns the KV pointer tensor for the given kernel group.

        Mirrors :meth:`GPUCacheContext.get_kernel_group_kv_pointers`.
        """
        return self.group_kv_pointers_[kernel_group_idx]

    def get_kernel_group_shape_dtype(
        self,
        num_tokens: int,
        kernel_group_idx: int,
    ) -> tuple[torch.Size, torch.dtype]:
        """Returns the shape and dtype for the given kernel group index and
        number of tokens.

        Mirrors :meth:`GPUCacheContext.get_kernel_group_shape_dtype` so
        callers such as ``lmcache_driven_transfer.get_layout_desc`` can duck-type
        across GPU and CPU backends.

        Args:
            num_tokens: Number of tokens.
            kernel_group_idx: Index of the kernel group.

        Returns:
            A ``(shape, dtype)`` tuple for the given kernel group.
        """
        group = self.kv_layer_groups_manager_.kernel_groups[kernel_group_idx]
        compress_ratio = group.tokens_per_block // group.slots_per_block
        if num_tokens % compress_ratio != 0:
            raise ValueError(
                "num_tokens (%d) is not a multiple of compress_ratio (%d) "
                "for kernel_group_idx %d"
                % (num_tokens, compress_ratio, kernel_group_idx)
            )
        num_slots = num_tokens // compress_ratio
        sd = group.shape_desc
        shape = torch.Size(
            (sd.kv_size, group.num_layers, num_slots, group.hidden_dim_size)
        )
        return shape, group.dtype

    def get_tmp_gpu_buffer_flat(self, chunk_idx: int) -> torch.Tensor:
        """Returns the flat uint8 temp buffer for the given chunk."""
        if chunk_idx >= self.max_batch_size:
            raise ValueError(
                "chunk_idx %d >= max_batch_size %d" % (chunk_idx, self.max_batch_size)
            )
        start = chunk_idx * self.tmp_chunk_bytes_
        return self.tmp_cpu_buffer_[start : start + self.tmp_chunk_bytes_]

    def get_temp_kernel_group_buffer(
        self, batch_idx: int, kernel_group_idx: int
    ) -> torch.Tensor:
        """Returns the typed temp buffer for the given batch and kernel group.

        Mirrors :meth:`GPUCacheContext.get_temp_kernel_group_buffer`.

        Args:
            batch_idx: Batch slot index (0 <= batch_idx < max_batch_size).
            kernel_group_idx: Index of the kernel group.

        Returns:
            A typed tensor view with the correct shape and dtype.
        """
        if batch_idx >= self.max_batch_size:
            raise ValueError(
                "batch_idx %d >= max_batch_size %d" % (batch_idx, self.max_batch_size)
            )
        group = self.kv_layer_groups_manager_.kernel_groups[kernel_group_idx]
        shape = self.get_kv_buffer_shape(
            self.lmcache_tokens_per_chunk, kernel_group_idx
        )
        g_start = self.tmp_chunk_group_offsets_[kernel_group_idx]
        g_end = self.tmp_chunk_group_offsets_[kernel_group_idx + 1]
        chunk = self.tmp_chunk_bytes_
        return (
            self.tmp_cpu_buffer_[
                batch_idx * chunk + g_start : batch_idx * chunk + g_end
            ]
            .view(group.dtype)
            .view(shape)
        )

    def get_temp_object_group_buffer(
        self, batch_idx: int, object_group_idx: int
    ) -> torch.Tensor:
        """Returns the flat uint8 temp buffer for the given batch and object
        group.

        Mirrors :meth:`GPUCacheContext.get_temp_object_group_buffer`.

        Args:
            batch_idx: Batch slot index (0 <= batch_idx < max_batch_size).
            object_group_idx: Index of the object group.

        Returns:
            A flat uint8 tensor view covering the object group's byte range.
        """
        if batch_idx >= self.max_batch_size:
            raise ValueError(
                "batch_idx %d >= max_batch_size %d" % (batch_idx, self.max_batch_size)
            )
        manager = self.kv_layer_groups_manager_
        object_group = manager.object_groups[object_group_idx]
        kg_indices = object_group.kernel_group_indices
        # Object group spans from the first to the last kernel group's range.
        g_start = self.tmp_chunk_group_offsets_[kg_indices[0]]
        g_end = self.tmp_chunk_group_offsets_[kg_indices[-1] + 1]
        chunk = self.tmp_chunk_bytes_
        return self.tmp_cpu_buffer_[
            batch_idx * chunk + g_start : batch_idx * chunk + g_end
        ]

    def get_tmp_chunk_gpu_buffer(self, group_idx: int = 0) -> torch.Tensor:
        """Returns a typed view of the temp buffer for one chunk."""
        group = self.kv_layer_groups_manager_.kernel_groups[group_idx]
        shape = self.get_kv_buffer_shape(self.lmcache_tokens_per_chunk, group_idx)
        start = self.tmp_chunk_group_offsets_[group_idx]
        end = self.tmp_chunk_group_offsets_[group_idx + 1]
        return self.tmp_cpu_buffer_[start:end].view(group.dtype).view(shape)

    def get_tmp_chunk_gpu_buffer_batched(
        self, batch_size: int, group_idx: int = 0
    ) -> list[torch.Tensor]:
        """Returns a list of non-overlapping temp buffer views."""
        if batch_size > self.max_batch_size:
            raise ValueError(
                "batch_size %d > max_batch_size %d" % (batch_size, self.max_batch_size)
            )
        group = self.kv_layer_groups_manager_.kernel_groups[group_idx]
        shape = self.get_kv_buffer_shape(self.lmcache_tokens_per_chunk, group_idx)
        g_start = self.tmp_chunk_group_offsets_[group_idx]
        g_end = self.tmp_chunk_group_offsets_[group_idx + 1]
        chunk = self.tmp_chunk_bytes_
        return [
            self.tmp_cpu_buffer_[i * chunk + g_start : i * chunk + g_end]
            .view(group.dtype)
            .view(shape)
            for i in range(batch_size)
        ]

    def cache_size_per_token(self) -> int:
        """Returns cache size per *logical* token in bytes,
        summed across all groups.

        Mirrors :meth:`GPUCacheContext.cache_size_per_token`.
        """
        total = 0
        for group_idx, group in enumerate(self.kv_layer_groups_manager_.kernel_groups):
            compress_ratio = group.tokens_per_block // group.slots_per_block
            numels = self.get_kv_buffer_shape(compress_ratio, group_idx).numel()
            slot_bytes = numels * group.dtype.itemsize
            total += slot_bytes // compress_ratio
        return total
