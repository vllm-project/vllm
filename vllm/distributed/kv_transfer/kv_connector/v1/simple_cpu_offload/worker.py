# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side handler for SimpleCPUOffloadConnector."""

from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload import (
    copy_ops,
)
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload.metadata import (
    SimpleCPUOffloadMetadata,
)
from vllm.logger import init_logger
from vllm.utils.platform_utils import is_pin_memory_available

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload.copy_ops import (  # noqa: E501
        LaunchParams,
    )
    from vllm.v1.kv_cache_interface import KVCacheConfig

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

        # Cached launch params for the Triton kernel
        self._store_launch_params: LaunchParams | None = None
        self._load_launch_params: LaunchParams | None = None

        # CUDA streams for the async transfers
        self.load_stream: torch.cuda.Stream | None = None
        self.store_stream: torch.cuda.Stream | None = None

        # Ordered (event_idx, event) — stream ordering lets us break early
        self._load_events: list[tuple[int, torch.cuda.Event]] = []
        self._store_events: list[tuple[int, torch.cuda.Event]] = []

        # Deferred stores: queued in wait_for_save(), flushed in start_load_kv()
        self._pending_store_events: list[tuple[int, list[int], list[int]]] = []
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
        kv_cache_raw_tensors: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Register GPU KV caches and allocate pinned CPU tensors.

        Args:
            kv_caches: Reshaped per-layer GPU KV caches.
            kv_cache_raw_tensors: Raw int8 tensors before reshape. If provided,
                used for transfers (HMA-safe). Falls back to kv_caches if None.
        """
        raw = kv_cache_raw_tensors if kv_cache_raw_tensors is not None else kv_caches
        self.device = next(iter(raw.values())).device

        # Deduplicate shared tensors (multiple layers may share the same tensor)
        seen_ptrs: dict[int, tuple[str, torch.Tensor]] = {}
        for name, tensor in raw.items():
            ptr = tensor.data_ptr()
            if ptr not in seen_ptrs:
                seen_ptrs[ptr] = (name, tensor)

        # Build ordered dict of unique raw tensors for the Triton kernel
        unique_gpu_caches: dict[str, torch.Tensor] = {}
        for name, tensor in seen_ptrs.values():
            # Raw tensors are 1D int8. Reshape to [num_blocks, page_size_bytes]
            # so stride(0) gives page_size_bytes for the Triton kernel.
            if tensor.dim() == 1:
                assert self.kv_cache_config is not None
                num_blocks = self.kv_cache_config.num_blocks
                tensor = tensor.view(num_blocks, -1)
            unique_gpu_caches[name] = tensor

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
        """Flush deferred stores, then start async loads from CPU to GPU."""
        if not self._is_initialized:
            logger.warning("KV caches not registered, skipping load")
            return

        self._submit_pending_stores()

        if self._connector_metadata is None:
            return

        metadata = self._connector_metadata
        if not metadata.load_gpu_blocks:
            return

        assert self.load_stream is not None
        assert self.cpu_kv_caches is not None
        assert self.gpu_kv_caches is not None

        with torch.cuda.stream(self.load_stream):
            self._copy_blocks(
                src_caches=self.cpu_kv_caches,
                dst_caches=self.gpu_kv_caches,
                src_block_ids=metadata.load_cpu_blocks,
                dst_block_ids=metadata.load_gpu_blocks,
                is_store=False,
            )
            event = torch.cuda.Event()
            event.record(self.load_stream)

        self._load_events.append((metadata.load_event, event))
        logger.debug(
            "Started loading %d blocks from CPU (event_idx=%d)",
            len(metadata.load_gpu_blocks),
            metadata.load_event,
        )

    def wait_for_save(self) -> None:
        """Queue store events; actual submission deferred to start_load_kv()."""
        if self._connector_metadata is None:
            return

        if not self._is_initialized:
            return

        metadata = self._connector_metadata
        if not metadata.store_gpu_blocks:
            return

        self._pending_store_events.append(
            (metadata.store_event, metadata.store_gpu_blocks, metadata.store_cpu_blocks)
        )
        logger.debug(
            "Queued storing %d blocks to CPU (event_idx=%d)",
            len(metadata.store_gpu_blocks),
            metadata.store_event,
        )

    def _submit_pending_stores(self) -> None:
        if not self._pending_store_events:
            return
        if not self._is_initialized:
            return

        assert self.store_stream is not None
        assert self.gpu_kv_caches is not None
        assert self.cpu_kv_caches is not None

        all_src: list[int] = []
        all_dst: list[int] = []
        for _, src, dst in self._pending_store_events:
            all_src.extend(src)
            all_dst.extend(dst)

        with torch.cuda.stream(self.store_stream):
            if all_src:
                self._copy_blocks(
                    src_caches=self.gpu_kv_caches,
                    dst_caches=self.cpu_kv_caches,
                    src_block_ids=all_src,
                    dst_block_ids=all_dst,
                    is_store=True,
                )
            # One event covers all batched jobs; they share the same completion point.
            event = torch.cuda.Event()
            event.record(self.store_stream)

        for event_idx, _, _ in self._pending_store_events:
            self._store_events.append((event_idx, event))
            logger.debug("Submitted deferred store to CPU (event_idx=%d)", event_idx)

        self._pending_store_events.clear()

    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[set[str] | None, set[str] | None]:
        """Updates from worker to scheduler on completed transfer events.

        Returns:
            tuple of (finished_sending, finished_recving).
            - finish_sending is only used by connector scheduler, and we use it to
            store the finished store event ids rather than req_ids.
            - finished_recving still tracks the req_ids that have finished loading.
        """
        finished_recving: set[str] = set()
        finished_sending: set[str] = set()

        load_wm = self._drain_stream_events(self._load_events)
        meta = self._connector_metadata
        for j in [j for j in self._pending_load_event_indices if j <= load_wm]:
            req_ids = meta.load_event_to_reqs.get(j) if meta is not None else None
            if req_ids:
                finished_recving.update(req_ids)
                self._pending_load_event_indices.discard(j)

        store_wm = self._drain_stream_events(self._store_events)

        for j in [j for j in self._pending_store_event_indices if j <= store_wm]:
            self._pending_store_event_indices.discard(j)
            finished_sending.add(f"__store_done_{j}")

        return finished_sending or None, finished_recving or None

    @staticmethod
    def _drain_stream_events(events: list[tuple[int, torch.cuda.Event]]) -> int:
        watermark = -1
        while events:
            event_idx, event = events[0]
            if event.query():
                watermark = event_idx
                events.pop(0)
            else:
                break  # Stream ordering: nothing after this can have fired.
        return watermark

    @staticmethod
    def _validate_block_ids(block_ids: list[int], num_blocks: int, label: str) -> None:
        if not block_ids:
            return
        lo, hi = min(block_ids), max(block_ids)
        if lo < 0 or hi >= num_blocks:
            bad = lo if lo < 0 else hi
            raise ValueError(f"{label} block ID {bad} out of bounds [0, {num_blocks})")

    def _copy_blocks(
        self,
        src_caches: dict[str, torch.Tensor],
        dst_caches: dict[str, torch.Tensor],
        src_block_ids: list[int],
        dst_block_ids: list[int],
        is_store: bool = True,
    ) -> None:
        """Execute multi-layer Triton kernel block transfer."""
        first_src = next(iter(src_caches.values()))
        first_dst = next(iter(dst_caches.values()))
        self._validate_block_ids(src_block_ids, first_src.shape[0], "Source")
        self._validate_block_ids(dst_block_ids, first_dst.shape[0], "Dest")

        block_mapping = torch.tensor(
            list(zip(src_block_ids, dst_block_ids)),
            dtype=torch.int64,
            device=self.device,
        )

        launch_params = (
            self._store_launch_params if is_store else self._load_launch_params
        )

        copy_ops.copy_blocks(
            src_caches,
            dst_caches,
            block_mapping,
            launch_params=launch_params,
        )

    def handle_preemptions(self) -> None:
        """Sync all in-flight transfers before preempted blocks are reused."""
        self._submit_pending_stores()

        for _, event in self._load_events:
            event.synchronize()
        self._load_events.clear()

        for _, event in self._store_events:
            event.synchronize()
        self._store_events.clear()
