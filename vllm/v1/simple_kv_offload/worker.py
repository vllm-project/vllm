# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side handler for SimpleCPUOffloadConnector."""

from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.simple_kv_offload.copy_backend import CopyBackend, get_copy_backend
from vllm.v1.simple_kv_offload.metadata import SimpleCPUOffloadMetadata

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class SimpleCPUOffloadWorker:
    """Worker-side handler for CPU offloading transfers."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: "KVCacheConfig | None",
        cpu_capacity_bytes: int,
        copy_backend: str = "kernel",
    ):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.cpu_capacity_bytes = cpu_capacity_bytes

        self.gpu_kv_caches: dict[str, torch.Tensor] | None = None
        self.cpu_kv_caches: dict[str, torch.Tensor] | None = None
        self.device: torch.device | None = None
        self.num_cpu_blocks: int = 0

        # CUDA streams for the async transfers
        self.load_stream: torch.cuda.Stream | None = None
        self.store_stream: torch.cuda.Stream | None = None

        self._backend: CopyBackend = get_copy_backend(copy_backend)

        # Ordered (event_idx, Event). Events pre-allocated on main thread.
        self._load_events: list[tuple[int, torch.Event]] = []
        self._store_events: list[tuple[int, torch.Event]] = []
        # High-water marks: highest event_idx completed per stream.
        # When the event list is empty, the hwm covers all prior events.
        self._load_hwm: int = -1
        self._store_hwm: int = -1

        # Metadata for the current step
        self._connector_metadata: SimpleCPUOffloadMetadata | None = None

        # Pending event index sets, populated in bind_connector_metadata
        self._pending_load_event_indices: set[int] = set()
        self._pending_store_event_indices: set[int] = set()

    @property
    def _is_initialized(self) -> bool:
        return self.gpu_kv_caches is not None and self.cpu_kv_caches is not None

    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor],
    ) -> None:
        """Register GPU KV caches and allocate pinned CPU tensors.
        The worker will infer the underlying raw storage from the kv_caches.

        Args:
            kv_caches: Per-layer GPU KV caches. Values are either a single
                tensor (attention layers) or a list of tensors (Mamba layers
                in hybrid models). All values are included for offloading
                by resolving to their underlying raw storage.
        """
        if not kv_caches:
            logger.warning("No KV caches to offload.")
            return

        # Resolve each entry to a representative tensor for storage
        # deduplication. For attention layers the value is already a tensor;
        # for Mamba layers it is a list of tensors that all share the same
        # underlying raw storage, so we take the first one.
        def _representative_tensor(
            v: torch.Tensor | list[torch.Tensor],
        ) -> torch.Tensor:
            if isinstance(v, torch.Tensor):
                return v
            elif isinstance(v, list):
                return v[0]
            else:
                raise ValueError(f"Unsupported type: {type(v)}")

        first_tensor = _representative_tensor(next(iter(kv_caches.values())))
        self.device = first_tensor.device

        assert self.kv_cache_config is not None
        num_blocks = self.kv_cache_config.num_blocks

        # Deduplicate: multiple layers may share the same backing storage.
        seen_ptrs: dict[int, tuple[str, torch.Tensor]] = {}
        for name, value in kv_caches.items():
            tensor = _representative_tensor(value)
            ptr = tensor.untyped_storage().data_ptr()
            if ptr not in seen_ptrs:
                seen_ptrs[ptr] = (name, tensor)

        # Build [num_blocks, block_bytes] int8 views from each unique
        # storage so that stride(0) gives block_bytes for the copy op.
        #
        # The physical layout varies across attention backends:
        #   FlashAttn/ROCm:  (2, num_blocks, ...) -> K/V outermost, 2 segments
        #   FlashInfer/MLA:  (num_blocks, ...)    -> blocks outermost, 1 segment
        # We detect this from the tensor's strides, mirroring KVBlockZeroer.
        unique_gpu_caches: dict[str, torch.Tensor] = {}
        for name, tensor in seen_ptrs.values():
            storage = tensor.untyped_storage()
            raw = torch.empty(0, dtype=torch.int8, device=self.device).set_(
                storage, 0, (storage.nbytes(),)
            )
            # Find the num_blocks dimension
            block_dim = next(
                d for d in range(tensor.ndim) if tensor.shape[d] == num_blocks
            )
            el = tensor.element_size()
            block_stride_bytes = tensor.stride(block_dim) * el
            # Dims before block_dim with larger stride are "outer" (e.g. K/V).
            outer_dims = [
                d
                for d in range(block_dim)
                if tensor.stride(d) * el > block_stride_bytes
            ]
            if not outer_dims:  # no outer dimensions, so all blocks are contiguous
                unique_gpu_caches[name] = raw.view(num_blocks, -1)
            else:  # outer dimensions exist, so blocks are segmented
                for idx in range(tensor.shape[outer_dims[0]]):
                    offset = idx * tensor.stride(outer_dims[0]) * el
                    chunk = raw[offset : offset + num_blocks * block_stride_bytes]
                    unique_gpu_caches[f"{name}.{idx}"] = chunk.view(num_blocks, -1)

        # Compute per-tensor bytes_per_block. Tensors may have different
        # page_size_bytes (e.g., UniformTypeKVCacheSpecs with varying head_size).
        per_tensor_bpb = [
            t.stride(0) * t.element_size() for t in unique_gpu_caches.values()
        ]
        total_bytes_per_block = sum(per_tensor_bpb)

        self.num_cpu_blocks = max(1, self.cpu_capacity_bytes // total_bytes_per_block)

        logger.info(
            "SimpleCPUOffloadWorker: %d unique GPU KV tensors, "
            "allocating %d CPU blocks (%.2f GB)",
            len(unique_gpu_caches),
            self.num_cpu_blocks,
            (self.num_cpu_blocks * total_bytes_per_block) / (1024**3),
        )

        pin_memory = is_pin_memory_available()
        if not pin_memory:
            logger.warning(
                "Pinned memory not available. CPU offload performance may be degraded."
            )

        self.gpu_kv_caches = unique_gpu_caches
        self.cpu_kv_caches = {}
        for name, gpu_tensor in unique_gpu_caches.items():
            cpu_shape = (self.num_cpu_blocks,) + gpu_tensor.shape[1:]
            self.cpu_kv_caches[name] = torch.zeros(
                cpu_shape, dtype=gpu_tensor.dtype, device="cpu", pin_memory=pin_memory
            )

        # Use lowest priority so KV cache I/O yields to compute streams.
        low_pri, _ = torch.cuda.Stream.priority_range()
        self.load_stream = torch.cuda.Stream(priority=low_pri)
        self.store_stream = torch.cuda.Stream(priority=low_pri)

        # Initialize copy backend with caches and streams.
        self._backend.init(
            self.gpu_kv_caches,
            self.cpu_kv_caches,
            self.device,
            self.load_stream,
            self.store_stream,
        )

    def bind_connector_metadata(self, metadata: SimpleCPUOffloadMetadata) -> None:
        self._connector_metadata = metadata
        if metadata.load_event >= 0:
            self._pending_load_event_indices.add(metadata.load_event)
        if metadata.store_event >= 0:
            self._pending_store_event_indices.add(metadata.store_event)

    def clear_connector_metadata(self) -> None:
        self._connector_metadata = None

    def start_load_kv(self) -> None:
        # NOTE: we defer launching both load and store to get_finished(),
        # which runs after model execution. This hides the CPU-side
        # block copy op overhead (~5ms) behind GPU compute.
        pass

    def wait_for_save(self) -> None:
        pass

    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[set[str] | None, set[str] | None]:
        """Updates from worker to scheduler on completed transfer events.

        Additionally, it will submit deferred load and store transfers after model
         execution to hide the CPU-side block copy op overhead behind GPU compute.

        Returns:
            tuple of (finished_sending, finished_recving).
            - finished_sending is only used by connector scheduler, and we use
            it to store the finished store event ids rather than req_ids.
            - finished_recving still tracks the req_ids that have finished loading.
        """
        # (1) Submit deferred transfers (if any)
        metadata = self._connector_metadata
        if metadata is not None and self._is_initialized:
            self._launch_copy_kernel(
                metadata.load_cpu_blocks,
                metadata.load_gpu_blocks,
                metadata.load_event,
                is_store=False,
            )
            self._launch_copy_kernel(
                metadata.store_gpu_blocks,
                metadata.store_cpu_blocks,
                metadata.store_event,
                is_store=True,
            )

        # (2) Track completed transfer events
        finished_recving: set[str] = set()
        finished_sending: set[str] = set()

        if self._pending_load_event_indices:
            load_wm = self._drain_stream_events(is_store=False)
            for j in [j for j in self._pending_load_event_indices if j <= load_wm]:
                req_ids = (
                    metadata.load_event_to_reqs.get(j) if metadata is not None else None
                )
                if req_ids:
                    finished_recving.update(req_ids)
                    self._pending_load_event_indices.discard(j)

        if self._pending_store_event_indices:
            store_wm = self._drain_stream_events(is_store=True)
            for j in [j for j in self._pending_store_event_indices if j <= store_wm]:
                self._pending_store_event_indices.discard(j)
                finished_sending.add(f"__store_done_{j}")

        return finished_sending, finished_recving

    def handle_preemptions(self, preempted_req_ids: set[str]) -> None:
        """Sync all in-flight transfers before preempted blocks are reused."""
        for event_idx, event in self._load_events:
            event.synchronize()
            self._load_hwm = event_idx
        self._load_events.clear()

        for event_idx, event in self._store_events:
            event.synchronize()
            self._store_hwm = event_idx
        self._store_events.clear()

    def _launch_copy_kernel(
        self,
        src_blocks: list[int],
        dst_blocks: list[int],
        event_idx: int,
        is_store: bool,
    ) -> None:
        if not src_blocks:
            return

        if is_store:
            assert self.store_stream is not None
            # Await for GPU blocks to be populated.
            self.store_stream.wait_stream(torch.cuda.current_stream(self.device))

        events = self._store_events if is_store else self._load_events
        self._backend.launch_copy(
            src_blocks,
            dst_blocks,
            is_store,
            event_idx,
            events,
        )

    def _drain_stream_events(self, is_store: bool) -> int:
        """Drain completed events and return the high-water mark."""
        events = self._store_events if is_store else self._load_events
        hwm = self._store_hwm if is_store else self._load_hwm
        while events:
            event_idx, event = events[0]
            if not event.query():
                break
            hwm = event_idx
            events.pop(0)
        if is_store:
            self._store_hwm = hwm
        else:
            self._load_hwm = hwm
        return hwm
