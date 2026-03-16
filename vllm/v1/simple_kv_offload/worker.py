# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side handler for SimpleCPUOffloadConnector."""

from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.simple_kv_offload import (
    copy_ops,
)
from vllm.v1.simple_kv_offload.metadata import (
    SimpleCPUOffloadMetadata,
)

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.simple_kv_offload.copy_ops import (  # noqa: E501
        LaunchParams,
    )

logger = init_logger(__name__)


class SimpleCPUOffloadWorker:
    """Worker-side handler for CPU offloading transfers."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: "KVCacheConfig | None",
        cpu_capacity_bytes: int,
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

        # Cached launch params for the Triton kernel
        self._store_launch_params: LaunchParams | None = None
        self._load_launch_params: LaunchParams | None = None

        # Ordered (event_idx, event) — stream ordering lets us break early
        self._load_events: list[tuple[int, torch.cuda.Event]] = []
        self._store_events: list[tuple[int, torch.cuda.Event]] = []
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
        """Whether KV caches are registered and ready for transfers."""
        return (
            self.gpu_kv_caches is not None
            and self.cpu_kv_caches is not None
            and self.load_stream is not None
            and self.store_stream is not None
        )

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
            return v[0]

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

        # Reconstruct [num_blocks, page_size_bytes] int8 views from storage
        # so stride(0) gives page_size_bytes for the Triton copy kernel.
        unique_gpu_caches: dict[str, torch.Tensor] = {}
        for name, tensor in seen_ptrs.values():
            storage = tensor.untyped_storage()
            raw = torch.empty(0, dtype=torch.int8, device=self.device).set_(
                storage, 0, (storage.nbytes(),)
            )
            unique_gpu_caches[name] = raw.view(num_blocks, -1)

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

        self.gpu_kv_caches = unique_gpu_caches
        self.cpu_kv_caches = {}
        for name, gpu_tensor in unique_gpu_caches.items():
            cpu_shape = (self.num_cpu_blocks,) + gpu_tensor.shape[1:]
            self.cpu_kv_caches[name] = torch.zeros(
                cpu_shape,
                dtype=gpu_tensor.dtype,
                device="cpu",
                pin_memory=pin_memory,
            )

        if not pin_memory:
            logger.warning(
                "Pinned memory not available. CPU offload performance may be degraded."
            )

        self._store_launch_params = copy_ops.build_launch_params(
            self.gpu_kv_caches, self.cpu_kv_caches
        )
        self._load_launch_params = copy_ops.build_launch_params(
            self.cpu_kv_caches, self.gpu_kv_caches
        )

        # Use lowest priority so KV cache I/O yields to compute streams.
        low_pri, _ = torch.cuda.Stream.priority_range()
        self.load_stream = torch.cuda.Stream(priority=low_pri)
        self.store_stream = torch.cuda.Stream(priority=low_pri)

    def bind_connector_metadata(self, metadata: SimpleCPUOffloadMetadata) -> None:
        self._connector_metadata = metadata
        if metadata.load_event >= 0:
            self._pending_load_event_indices.add(metadata.load_event)
        if metadata.store_event >= 0:
            self._pending_store_event_indices.add(metadata.store_event)

    def clear_connector_metadata(self) -> None:
        self._connector_metadata = None

    def start_load_kv(self) -> None:
        # NOTE: we defer launching both load and store kernels to
        # get_finished(), which runs after model execution in both the
        # normal forward path and the no_forward path. This hides the
        # CPU-side Triton kernel launch overhead behind GPU compute.
        pass

    def wait_for_save(self) -> None:
        # Transfers are submitted in get_finished() instead, so that
        # they work correctly in both the forward and no_forward paths.
        pass

    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[set[str] | None, set[str] | None]:
        """Updates from worker to scheduler on completed transfer events.

        Additionally, it will submit deferred load and store transfers, after
        model execution, to hide CPU-side Triton launch overhead behind
        GPU compute.

        Returns:
            tuple of (finished_sending, finished_recving).
            - finish_sending is only used by connector scheduler, and we use it to
            store the finished store event ids rather than req_ids.
            - finished_recving still tracks the req_ids that have finished loading.
        """
        # (1) Optionally submit deferred transfers
        if self._is_initialized and self._connector_metadata is not None:
            metadata = self._connector_metadata
            self._launch_copy_kernel(  # Launch the load kernel
                src_blocks=metadata.load_cpu_blocks,
                dst_blocks=metadata.load_gpu_blocks,
                event_idx=metadata.load_event,
                is_store=False,
            )
            self._launch_copy_kernel(  # Launch the store kernel
                src_blocks=metadata.store_gpu_blocks,
                dst_blocks=metadata.store_cpu_blocks,
                event_idx=metadata.store_event,
                is_store=True,
            )

        # (2) Track completed transfer events
        finished_recving: set[str] = set()
        finished_sending: set[str] = set()

        load_wm = self._drain_stream_events(is_store=False)
        meta = self._connector_metadata
        for j in [j for j in self._pending_load_event_indices if j <= load_wm]:
            req_ids = meta.load_event_to_reqs.get(j) if meta is not None else None
            if req_ids:
                finished_recving.update(req_ids)
                self._pending_load_event_indices.discard(j)

        store_wm = self._drain_stream_events(is_store=True)
        for j in [j for j in self._pending_store_event_indices if j <= store_wm]:
            self._pending_store_event_indices.discard(j)
            finished_sending.add(f"__store_done_{j}")

        return finished_sending or None, finished_recving or None

    def _launch_copy_kernel(
        self,
        src_blocks: list[int],
        dst_blocks: list[int],
        event_idx: int,
        is_store: bool,
    ) -> None:
        """Launch an async block copy on *stream* and record a tracking event."""
        if not src_blocks:
            return

        assert self.gpu_kv_caches is not None
        assert self.cpu_kv_caches is not None

        if is_store:
            src_caches, dst_caches = self.gpu_kv_caches, self.cpu_kv_caches
            stream = self.store_stream
            events = self._store_events
            launch_params = self._store_launch_params
        else:
            src_caches, dst_caches = self.cpu_kv_caches, self.gpu_kv_caches
            stream = self.load_stream
            events = self._load_events
            launch_params = self._load_launch_params

        assert stream is not None

        first_src = next(iter(src_caches.values()))
        first_dst = next(iter(dst_caches.values()))
        self._validate_block_ids(src_blocks, first_src.shape[0], "Source")
        self._validate_block_ids(dst_blocks, first_dst.shape[0], "Dest")

        with torch.cuda.stream(stream):
            block_mapping = torch.tensor(
                list(zip(src_blocks, dst_blocks)),
                dtype=torch.int64,
                device=self.device,
            )

            copy_ops.copy_blocks(
                src_caches,
                dst_caches,
                block_mapping,
                launch_params=launch_params,
            )
            # record_stream here after kernel launch to prevent the caching allocator
            # from recycling block_mapping before the kernel finishes reading it.
            block_mapping.record_stream(stream)
            event = torch.cuda.Event()
            event.record(stream)

        events.append((event_idx, event))
        logger.debug(
            "Submitted %s of %d blocks (event_idx=%d) store" if is_store else "load",
            len(src_blocks),
            event_idx,
        )

    def _drain_stream_events(self, is_store: bool) -> int:
        """Drain completed events and return the high-water mark.

        Returns the highest event_idx known to have completed. The hwm
        is persisted so that when the event list is empty, all prior
        events are still recognized as finished.
        """
        events = self._store_events if is_store else self._load_events
        hwm = self._store_hwm if is_store else self._load_hwm
        while events:
            event_idx, event = events[0]
            if event.query():
                hwm = event_idx
                events.pop(0)
            else:
                break  # Stream ordering: nothing after this can have fired.
        if is_store:
            self._store_hwm = hwm
        else:
            self._load_hwm = hwm
        return hwm

    @staticmethod
    def _validate_block_ids(block_ids: list[int], num_blocks: int, label: str) -> None:
        if not block_ids:
            return
        lo, hi = min(block_ids), max(block_ids)
        if lo < 0 or hi >= num_blocks:
            bad = lo if lo < 0 else hi
            raise ValueError(f"{label} block ID {bad} out of bounds [0, {num_blocks})")
