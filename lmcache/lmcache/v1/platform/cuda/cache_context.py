# SPDX-License-Identifier: Apache-2.0
"""
GPU Cache Context management for LMCache multiprocessing.

This module provides GPU-side KV cache management functionality, including:
- GPUCacheContext: Manages shape and pointers to vLLM GPU KV cache tensors
- Helper functions for tensor operations and key resolution
"""

# Standard
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any
import array

# Third Party
import torch

if TYPE_CHECKING:
    # Third Party
    import cupy

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.gds_context import get_gds_context
from lmcache.v1.gpu_connector.utils import (
    LayoutHints,
    get_device,
    get_group_data_ptrs,
    normalize_and_discover_per_layer_formats,
)
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
from lmcache.v1.multiprocess.custom_types import KVCache
from lmcache.v1.multiprocess.group_view import (
    EngineGroupInfo,
    engine_group_layer_indices,
)
from lmcache.v1.platform.base.cache_context import BaseCacheContext

logger = init_logger(__name__)


def unwrap_kv_cache_tensors(kv_caches: KVCache) -> list[torch.Tensor]:
    unwrapped_tensors = []
    for ipc_wrapper in kv_caches:
        tensor = ipc_wrapper.to_tensor()
        unwrapped_tensors.append(tensor)
    return unwrapped_tensors


def list_to_gpu_tensor(lis: list[int], device: torch.device) -> torch.Tensor:
    return torch.frombuffer(array.array("q", lis), dtype=torch.long).to(
        device, non_blocking=True
    )


class _TempGPUBuffer:
    """
    Manages the temporary GPU buffer for GPUCacheContext

    The logical layout of the temp GPU buffer is (batch size,
    object group, kernel group).

    Here is an example of batch size = 4, with 2 object groups,
    and 2 kernel groups per object group:
    [
        batch 0:
            - object group 0: kernel group 0 | kernel group 1 | ...
            - object group 1: kernel group 2 | kernel group 3 | ...

        batch 1:
            - object group 0: kernel group 0 | kernel group 1 | ...
            - object group 1: kernel group 2 | kernel group 3 | ...

        batch 2:
            - object group 0: kernel group 0 | kernel group 1 | ...
            - object group 1: kernel group 2 | kernel group 3 | ...

        batch 3:
            - object group 0: kernel group 0 | kernel group 1 | ...
            - object group 1: kernel group 2 | kernel group 3 | ...
    ]

    During the multi-layer copy kernel launch, we will do it at kernel
    group level, which means we will have:
    ```
    gpu_buffers = [
        get_temp_kernel_group_buffer(batch_idx, kernel_group_idx)
        for batch_idx in range(batch_size)
    ]
    ```

    During the lmcache_memcpy_async launch, we will do it at the object group
    level, which will be:
    ```
    for i in range(batch_size):
        gpu_buffer = get_temp_object_group_buffer(batch_idx, object_group_idx)
        lmcache_memcpy_async(...)
    ```
    """

    def __init__(
        self,
        kv_layer_groups_manager: KVLayerGroupsManager,
        lmcache_tokens_per_chunk: int,
        device: torch.device,
        max_batch_size: int = 4,
    ) -> None:
        self._kv_groups_manager = kv_layer_groups_manager
        self._lmcache_tokens_per_chunk = lmcache_tokens_per_chunk
        self._max_batch_size = max_batch_size

        self._temp_buffer = torch.empty(
            self._get_size_for_single_batch() * max_batch_size,
            dtype=torch.uint8,
            device=device,
        )

        # Offset map: (batch_idx, object_group_idx, kernel_group_idx) ->
        # (byte offset in the temp buffer, size of the buffer in bytes)
        self._offset_map: dict[tuple[int, int, int], tuple[int, int]] = {}

        # (batch_idx, kernel_group_idx) -> (byte offset for the kernel group,
        # size of the buffer in bytes).
        self._offset_map_kernel_group_only: dict[tuple[int, int], tuple[int, int]] = {}

        # (batch_idx, object_group_idx) -> (byte offset for the object group,
        # size of the buffer in bytes)
        self._offset_map_object_group_only: dict[tuple[int, int], tuple[int, int]] = {}

        offset = 0
        for batch_idx in range(max_batch_size):
            for object_group_idx in range(self._kv_groups_manager.num_object_groups):
                object_group_size = 0
                object_group_start_offset = offset

                for kernel_group_idx in self._kv_groups_manager.object_groups[
                    object_group_idx
                ].kernel_group_indices:
                    key = (batch_idx, object_group_idx, kernel_group_idx)
                    key2 = (batch_idx, kernel_group_idx)

                    size = self._get_size_for_kernel_group(kernel_group_idx)
                    self._offset_map[key] = (offset, size)
                    self._offset_map_kernel_group_only[key2] = (offset, size)

                    offset += size
                    object_group_size += size

                key3 = (batch_idx, object_group_idx)
                self._offset_map_object_group_only[key3] = (
                    object_group_start_offset,
                    object_group_size,
                )

        # Shape/dtype cache for kernel groups
        self._shape_cache_kernel_group: dict[int, tuple[torch.Size, torch.dtype]] = {}
        for kernel_group_idx in range(self._kv_groups_manager.num_kernel_groups):
            shape = self._get_shape_for_kernel_group(
                self._lmcache_tokens_per_chunk, kernel_group_idx
            )
            group = self._kv_groups_manager.kernel_groups[kernel_group_idx]
            dtype = group.dtype
            self._shape_cache_kernel_group[kernel_group_idx] = (shape, dtype)

    # Public APIs
    @property
    def max_batch_size(self) -> int:
        """Maximum number of chunks (batch slots) the buffer holds."""
        return self._max_batch_size

    @property
    def buffer(self) -> torch.Tensor:
        """The flat staging tensor (for GDS cuFile registration)."""
        return self._temp_buffer

    def get_temp_kernel_group_buffer(
        self, batch_idx: int, kernel_group_idx: int
    ) -> torch.Tensor:
        """
        Returns the temp GPU buffer for the given batch index and kernel group index.
        The returned buffer is with the correct shape and dtype for the kernel group.

        Args:
            batch_idx: Index of the batch (0 <= batch_idx < max_batch_size)
            kernel_group_idx: Index of the kernel group.

        Returns:
            The temp GPU buffer for the given batch index and kernel group index.

        Raises:
            ValueError: If the batch_idx or kernel_group_idx is out of range.
        """
        key = (batch_idx, kernel_group_idx)
        if key not in self._offset_map_kernel_group_only:
            raise ValueError(
                f"Invalid batch_idx {batch_idx} or kernel_group_idx {kernel_group_idx}"
            )

        offset, size = self._offset_map_kernel_group_only[key]
        shape, dtype = self._shape_cache_kernel_group[kernel_group_idx]
        return self._temp_buffer[offset : offset + size].view(dtype).view(shape)

    def get_temp_object_group_buffer(
        self, batch_idx: int, object_group_idx: int
    ) -> torch.Tensor:
        """
        Returns the temp GPU buffer for the given batch index and object group index
        The returned buffer is a flat uint8 raw tensor.

        Args:
            batch_idx: Index of the batch (0 <= batch_idx < max_batch_size)
            object_group_idx: Index of the object group.

        Returns:
            The temp GPU buffer for the given batch index and object group index.
        """
        key = (batch_idx, object_group_idx)
        if key not in self._offset_map_object_group_only:
            raise ValueError(
                f"Invalid batch_idx {batch_idx} or object_group_idx {object_group_idx}"
            )

        offset, size = self._offset_map_object_group_only[key]
        return self._temp_buffer[offset : offset + size]

    def get_kernel_group_shape_dtype(
        self,
        num_tokens: int,
        kernel_group_idx: int,
    ) -> tuple[torch.Size, torch.dtype]:
        """
        Returns the shape and dtype for the given kernel group index and
        number of tokens.

        Will be exported by GPUCacheContext and used to construct the
        MemoryLayoutDesc

        Args:
            num_tokens: Number of tokens. Must be a whole number of lmcache
                chunk size.
            kernel_group_idx: Index of the kernel group.

        Returns:
            The shape and dtype for the given kernel group index and
            number of tokens.
        """
        _, dtype = self._shape_cache_kernel_group[kernel_group_idx]
        shape = self._get_shape_for_kernel_group(num_tokens, kernel_group_idx)

        return shape, dtype

    def get_cache_size_per_token(self) -> int:
        """
        Returns the cache size per token (in bytes), summed across all kernel groups.
        """
        return self._get_size_for_single_batch() // self._lmcache_tokens_per_chunk

    # Helper functions
    def _get_shape_for_kernel_group(
        self,
        num_tokens: int,
        kernel_group_idx: int,
    ) -> torch.Size:
        """
        Returns the shape of the temp GPU buffer for the given kernel group index

        Args:
            num_tokens: Number of tokens
            kernel_group_idx: Index of the kernel group.

        Returns:
            The shape of the temp GPU buffer for the given kernel group index.

        Raises:
            ValueError: If ``num_tokens`` is not a whole number of LMCache
                chunks.
        """
        if num_tokens % self._lmcache_tokens_per_chunk != 0:
            raise ValueError(
                f"num_tokens ({num_tokens}) must be a multiple of "
                f"lmcache_tokens_per_chunk ({self._lmcache_tokens_per_chunk})"
            )

        group = self._kv_groups_manager.kernel_groups[kernel_group_idx]
        sd = group.shape_desc

        num_chunks = num_tokens // self._lmcache_tokens_per_chunk
        num_slots = (
            self._kv_groups_manager.get_slots_per_chunk_in_sw(kernel_group_idx)
            * num_chunks
        )

        return torch.Size(
            (sd.kv_size, group.num_layers, num_slots, group.hidden_dim_size)
        )

    def _get_size_for_kernel_group(self, kernel_group_idx: int) -> int:
        """
        Returns the size in bytes of the temp GPU buffer for the given kernel group
        index

        **Assumes the size is lmcache_tokens_per_chunk

        Will only be called during initialization
        """
        shape = self._get_shape_for_kernel_group(
            self._lmcache_tokens_per_chunk, kernel_group_idx
        )
        kernel_group = self._kv_groups_manager.kernel_groups[kernel_group_idx]
        dtype = kernel_group.dtype
        return shape.numel() * dtype.itemsize

    def _get_size_for_object_group(self, object_group_idx: int) -> int:
        """
        Returns the size in bytes of the temp GPU buffer for the given object group

        **Assumes the size is lmcache_tokens_per_chunk

        Will only be called during initialization
        """
        object_group = self._kv_groups_manager.object_groups[object_group_idx]
        return sum(
            self._get_size_for_kernel_group(kernel_group_idx)
            for kernel_group_idx in object_group.kernel_group_indices
        )

    def _get_size_for_single_batch(self) -> int:
        """
        Returns the size in bytes of the temp GPU buffer for a single batch
        (i.e., a single chunk)

        **Assumes the size is lmcache_tokens_per_chunk
        """
        return sum(
            self._get_size_for_object_group(object_group_idx)
            for object_group_idx in range(self._kv_groups_manager.num_object_groups)
        )


class GPUCacheContext(BaseCacheContext):
    """
    Manages the shape and pointers to vLLM GPU KV cache tensors.
    """

    device_type = "cuda"

    def __init__(
        self,
        kv_caches: KVCache,
        lmcache_tokens_per_chunk: int = 256,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
        engine_type: EngineType = EngineType.VLLM,
        separate_object_groups: bool = True,
        full_sw_kv: bool = False,
    ):
        unwrapped = unwrap_kv_cache_tensors(kv_caches)
        kv_caches_norm, engine_kv_formats = normalize_and_discover_per_layer_formats(
            unwrapped,
            engine_group_layer_indices(engine_group_infos),
            engine_type,
            layout_hints,
        )
        self.device_ = get_device(kv_caches_norm)
        num_layers_val = len(engine_kv_formats)

        kv_layer_groups_manager = KVLayerGroupsManager(
            kv_caches_norm,
            engine_kv_formats=engine_kv_formats,
            engine_group_infos=engine_group_infos,
            lmcache_tokens_per_chunk=lmcache_tokens_per_chunk,
            separate_object_groups=separate_object_groups,
        )
        # Set full_sw_kv before the temp buffer / object groups are built so
        # both are sized for the full (un-windowed) per-chunk geometry.
        if full_sw_kv:
            kv_layer_groups_manager.enable_full_sw_kv()

        # Pre-allocated GPU buffer for block IDs (up to 1M elements).
        # The caller copies block_ids into this buffer before launching the
        # block-level kernel. Single-thread assumption: no lock needed.
        _MAX_BLOCK_IDS = 1 << 20
        block_ids_buffer = torch.empty(
            _MAX_BLOCK_IDS, dtype=torch.long, device=self.device_
        )

        super().__init__(
            kv_caches=kv_caches_norm,
            device=self.device_,
            num_layers=num_layers_val,
            kv_layer_groups_manager=kv_layer_groups_manager,
            block_ids_buffer=block_ids_buffer,
            lmcache_tokens_per_chunk=lmcache_tokens_per_chunk,
        )

        self.group_kv_pointers_: list[torch.Tensor] = []
        for idx, group in enumerate(self.kv_layer_groups_manager_.kernel_groups):
            ptrs = get_group_data_ptrs(
                self.kv_caches_, self.get_engine_kv_format(idx), group.layer_indices
            )
            self.group_kv_pointers_.append(list_to_gpu_tensor(ptrs, self.device_))

        # Temporary GPU buffer for transfers — a single flat uint8 buffer
        self._temp_buffer = _TempGPUBuffer(
            kv_layer_groups_manager=self.kv_layer_groups_manager_,
            lmcache_tokens_per_chunk=lmcache_tokens_per_chunk,
            device=self.device_,
            max_batch_size=4,
        )

        # GPU streams
        self.cuda_stream_ = torch_dev.Stream(device=self.device_)

        # Register the staging buffer with the GDS cuFile context on the
        # context's CUDA stream.
        with torch_dev.stream(self.cuda_stream_):
            get_gds_context().register_gpu_buffer(self._temp_buffer.buffer)

        # Third Party
        import cupy

        self.cupy_stream_: "cupy.cuda.Stream" = cupy.cuda.ExternalStream(
            self.cuda_stream_.cuda_stream, self.device_.index
        )

        # Extra initialization
        self.cupy_stream_.launch_host_func(
            lambda logger: logger.info(
                "Initialized cuda stream on device %s", str(self.device_)
            ),
            logger,
        )

    def close(self) -> None:
        """
        Deregister this context's GDS staging buffer (reverse of __init__).
        """
        with torch_dev.stream(self.cuda_stream_):
            get_gds_context().deregister_gpu_buffer(self._temp_buffer.buffer)

    @property
    def stream(self) -> Any:
        """
        Returns the GPU stream for KV cache operations
        """
        return self.cuda_stream_

    @property
    def cupy_stream(self) -> "cupy.cuda.Stream":
        return self.cupy_stream_

    def get_kernel_group_kv_pointers(self, kernel_group_idx: int) -> torch.Tensor:
        """Returns the pre-computed GPU tensor of KV cache pointers for the
        given kernel group index.
        """
        return self.group_kv_pointers_[kernel_group_idx]

    def get_temp_kernel_group_buffer(
        self, batch_idx: int, kernel_group_idx: int
    ) -> torch.Tensor:
        """Returns the temporary GPU buffer for the given batch index and kernel
        group index, with the correct shape and dtype for the kernel group.

        Args:
            batch_idx: Index of the batch (0 <= batch_idx < max_batch_size)
            kernel_group_idx: Index of the kernel group.

        Returns:
            The temp GPU buffer for the given batch index and kernel group index.
        """
        return self._temp_buffer.get_temp_kernel_group_buffer(
            batch_idx, kernel_group_idx
        )

    @property
    def max_batch_size(self) -> int:
        """Maximum number of chunks processed concurrently in one batch."""
        return self._temp_buffer.max_batch_size

    def get_temp_object_group_buffer(
        self, batch_idx: int, object_group_idx: int
    ) -> torch.Tensor:
        """Returns the temporary GPU buffer for the given batch index and object
        group index, as a flat uint8 tensor.

        Args:
            batch_idx: Index of the batch (0 <= batch_idx < max_batch_size)
            object_group_idx: Index of the object group.

        Returns:
            The temp GPU buffer for the given batch index and object group index.
        """
        return self._temp_buffer.get_temp_object_group_buffer(
            batch_idx, object_group_idx
        )

    def get_kernel_group_shape_dtype(
        self,
        num_tokens: int,
        kernel_group_idx: int,
    ) -> tuple[torch.Size, torch.dtype]:
        """Returns the shape and dtype for the given kernel group index and number
        of tokens.
        Will be exported by GPUCacheContext and used to construct the MemoryLayoutDesc

        Args:
            num_tokens: Number of tokens. Must be a whole number of lmcache
                chunk size.
            kernel_group_idx: Index of the kernel group.

        Returns:
            The shape and dtype for the given kernel group index and number of tokens.
        """
        return self._temp_buffer.get_kernel_group_shape_dtype(
            num_tokens, kernel_group_idx
        )

    def cache_size_per_token(self) -> int:
        """
        Returns the cache size per *logical* token (in bytes), summed
        across all groups. For a compressed group, one physical slot
        stores ``compress_ratio`` logical tokens, so the per-logical-token
        contribution is ``physical_slot_bytes // compress_ratio``.

        Reporting-only metric (surfaced via the ``/api/status`` HTTP
        endpoint and the ``lmcache describe`` CLI); sub-byte truncation
        from integer division is acceptable.
        """
        return self._temp_buffer.get_cache_size_per_token()


class PlainGPUCacheContext:
    """
    A plain GPU cache context that have a single contiguous 2LTD buffer
    """

    def __init__(self, kv_caches: KVCache, lmcache_tokens_per_chunk: int = 256):
        assert len(kv_caches) == 1, (
            "PlainGPUCacheContext only supports a single KV cache tensor"
        )

        # KV cache basics
        self._kv_cache = unwrap_kv_cache_tensors(kv_caches)[0]
        self._device = self._kv_cache.device

        # Shape related
        shape = self._kv_cache.shape
        assert len(shape) == 4, "Expected [2, L, T, D] for plain GPU cache"

        self._num_layers = shape[1]
        self._num_tokens = shape[2]
        self._hidden_dim_size = shape[3]

        # Temporary buffer
        tmp_buffer_shape = self.get_kv_buffer_shape(lmcache_tokens_per_chunk)
        self._tmp_gpu_buffer = torch.empty(
            tmp_buffer_shape, dtype=self.dtype, device=self.device
        )

        # GPU streams
        self._cuda_stream = torch_dev.Stream(device=self._device)
        # Third Party
        import cupy

        self._cupy_stream: "cupy.cuda.Stream" = cupy.cuda.ExternalStream(
            self._cuda_stream.cuda_stream, self._device.index
        )

        # Extra initialization
        self._cupy_stream.launch_host_func(
            lambda logger: logger.info(
                "Initialized cuda stream on device %s", str(self._device)
            ),
            logger,
        )

    def get_kv_buffer_shape(self, num_tokens: int) -> torch.Size:
        """
        Returns the shape of the KV buffer for the given number of tokens
        """
        return torch.Size((2, self._num_layers, num_tokens, self._hidden_dim_size))

    def get_tmp_gpu_buffer(self, num_tokens: int) -> torch.Tensor:
        """
        Returns the temporary GPU buffer for transfers
        """
        return self._tmp_gpu_buffer[:, :, :num_tokens, :]

    def slice_kv_cache_on_tokens(self, start: int, end: int) -> torch.Tensor:
        """
        Slices the KV cache tensor on the token dimension
        """
        return self._kv_cache[:, :, start:end, :]

    @property
    def dtype(self) -> torch.dtype:
        return self._kv_cache.dtype

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def stream(self) -> Any:
        """Returns the device-specific GPU stream (e.g., torch_dev.Stream)."""
        return self._cuda_stream

    @property
    def cupy_stream(self) -> "cupy.cuda.Stream":
        return self._cupy_stream

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def num_tokens(self) -> int:
        return self._num_tokens

    @property
    def hidden_dim_size(self) -> int:
        return self._hidden_dim_size

    @property
    def kv_cache_tensor(self) -> torch.Tensor:
        return self._kv_cache
