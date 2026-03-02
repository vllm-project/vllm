# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side manager for SimpleCPUOffloadConnector."""

import contextlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vllm.config import VllmConfig
from vllm.distributed.kv_events import KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload.metadata import (
    SimpleCPUOffloadMetadata,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import (
    KVCacheCoordinator,
    get_kv_cache_coordinator,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class TransferMeta:
    gpu_block_ids: list[int]
    cpu_block_ids: list[int]


@dataclass
class RequestState:
    """Consolidated per-request state for CPU offloading."""

    request: "Request"
    gpu_block_ids: tuple[list[int], ...]

    # Set when request_finished is called but transfers are still in-flight.
    # Defers block cleanup to the completion handler.
    finished: bool = False

    # Load tracking
    load_transfer: TransferMeta | None = None
    load_event: int | None = None

    # Store tracking (eager mode only)
    store_events: set[int] = field(default_factory=set)
    # Per-group cursors tracking how many blocks have been stored/skipped.
    num_stored_blocks: list[int] = field(default_factory=list)


class SimpleCPUOffloadScheduler:
    """Scheduler-side manager for CPU offloading."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: "KVCacheConfig | None",
        cpu_capacity_bytes: int,
        lazy_offload: bool = False,
        min_lookahead_blocks: int = 8,
    ):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        # NOTE: We use the same block size for both GPU and CPU.
        self.block_size = vllm_config.cache_config.block_size
        # Derive a CPU KVCacheConfig from the GPU config and build a coordinator
        self.cpu_kv_cache_config = self._derive_cpu_config(
            kv_cache_config, cpu_capacity_bytes
        )
        self.num_cpu_blocks = self.cpu_kv_cache_config.num_blocks

        logger.info(
            "SimpleCPUOffloadScheduler: Allocating %d CPU blocks "
            "(%.2f GB capacity, mode=%s)",
            self.num_cpu_blocks,
            cpu_capacity_bytes / (1024**3),
            "lazy" if lazy_offload else "eager",
        )

        self.cpu_coordinator: KVCacheCoordinator = get_kv_cache_coordinator(
            kv_cache_config=self.cpu_kv_cache_config,
            max_model_len=vllm_config.model_config.max_model_len,
            use_eagle=False,
            enable_caching=True,
            enable_kv_cache_events=False,
            dcp_world_size=1,
            pcp_world_size=1,
            hash_block_size=self.block_size,
        )
        self.cpu_block_pool = self.cpu_coordinator.block_pool

        # GPU block pool reference - injected after scheduler builds kv_cache_manager
        self._gpu_block_pool: BlockPool | None = None

        # Load metadata
        self._reqs_to_load: dict[str, RequestState] = {}
        # Inverse maps: event_idx -> req_ids. Keyed by event index because the
        # worker reports completions by event index, not request id.
        self._load_event_to_reqs: dict[int, list[str]] = {}
        # FIXME (yifan): This is a hack to get num_local_computed_tokens without
        # modifying the connector API. But this will cause potential memory leaks.
        # Temporarily stores num_computed_tokens per request between
        # get_num_new_matched_tokens() and update_state_after_alloc().
        self._req_local_computed: dict[str, int] = {}

        # Store metadata
        self._lazy_mode = lazy_offload
        # Lazy store mode only
        self._min_lookahead_blocks = min_lookahead_blocks
        self._store_event_to_blocks: dict[int, TransferMeta] = {}
        # Eager mode only
        self._reqs_to_store: dict[str, RequestState] = {}
        self._store_event_to_reqs: dict[int, list[str]] = {}

        # Event counters
        self._load_event_counter: int = 0
        self._store_event_counter: int = 0

    @staticmethod
    def _derive_cpu_config(
        gpu_config: "KVCacheConfig", cpu_capacity_bytes: int
    ) -> "KVCacheConfig":
        """Derive a CPU KVCacheConfig from the GPU config.
        Same kv_cache_groups, num_blocks scaled by CPU/GPU memory ratio."""
        # Import here to avoid potential circular imports
        from vllm.v1.kv_cache_interface import KVCacheConfig as KVCacheConfigCls
        from vllm.v1.kv_cache_interface import KVCacheTensor

        page_sizes = {
            g.kv_cache_spec.page_size_bytes for g in gpu_config.kv_cache_groups
        }
        assert len(page_sizes) == 1, (
            f"Expected uniform page_size_bytes, got {page_sizes}"
        )
        page_size_bytes = next(iter(page_sizes))
        num_tensors = len(gpu_config.kv_cache_tensors)
        assert num_tensors > 0

        num_cpu_blocks = max(1, cpu_capacity_bytes // num_tensors // page_size_bytes)
        cpu_tensors = [
            KVCacheTensor(
                size=page_size_bytes * num_cpu_blocks,
                shared_by=list(t.shared_by),
            )
            for t in gpu_config.kv_cache_tensors
        ]

        return KVCacheConfigCls(
            num_blocks=num_cpu_blocks,
            kv_cache_tensors=cpu_tensors,
            kv_cache_groups=gpu_config.kv_cache_groups,
        )

    def bind_gpu_block_pool(self, gpu_block_pool: BlockPool) -> None:
        """Bind GPU block pool so that we can touch blocks during stores.
        Called by Scheduler after kv_cache_manager is ready."""
        self._gpu_block_pool = gpu_block_pool

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """Return (num_new_tokens, is_async) from consecutive CPU cache hits."""
        skipped = num_computed_tokens // self.block_size
        remaining_hashes = request.block_hashes[skipped:]

        if not remaining_hashes:
            return 0, False

        max_hit_len = len(remaining_hashes) * self.block_size
        _, hit_length = self.cpu_coordinator.find_longest_cache_hit(
            remaining_hashes, max_hit_len
        )

        if hit_length > 0:
            # Save for update_state_after_alloc to avoid recomputing skipped.
            self._req_local_computed[request.request_id] = num_computed_tokens
            logger.debug(
                "Request %s: CPU cache hit, %d external tokens can be loaded",
                request.request_id,
                hit_length,
            )
            return hit_length, True

        return 0, False

    # TODO (yifan): this function now assumes eager offloading and only matches
    # the suffix part of the prefix cache. Another interface is needed for lazy
    # offloading, which should check prefix cache hits in GPU block pool and CPU
    # block pool in a single pass.
    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        """Prepare load metadata after GPU block allocation."""
        req_id = request.request_id
        block_ids_by_group = blocks.get_block_ids()

        # Store tracking (eager mode only). Track here because this is the only
        # place we can see all scheduled requests. With chunked prefill, this
        # method is called multiple times for the same request across scheduling
        # steps, so update gpu_block_ids if the entry already exists.
        if not self._lazy_mode:
            existing = self._reqs_to_store.get(req_id)
            if existing is not None:
                existing.gpu_block_ids = block_ids_by_group
            else:
                self._reqs_to_store[req_id] = RequestState(
                    request=request,
                    gpu_block_ids=block_ids_by_group,
                    num_stored_blocks=[0] * len(block_ids_by_group),
                )

        if num_external_tokens == 0:
            return

        num_blocks_to_load = num_external_tokens // self.block_size
        assert num_blocks_to_load > 0

        num_computed_tokens = self._req_local_computed.pop(req_id, 0)
        skipped = num_computed_tokens // self.block_size
        hashes_to_load = request.block_hashes[skipped : skipped + num_blocks_to_load]

        # Find CPU cached blocks across all groups.
        max_hit_len = len(hashes_to_load) * self.block_size
        cpu_hit_blocks, hit_length = self.cpu_coordinator.find_longest_cache_hit(
            hashes_to_load, max_hit_len
        )
        assert hit_length == num_external_tokens, (
            f"Expected {num_external_tokens} hit tokens, got {hit_length}"
        )

        # Build transfer pairs across all groups.
        total_computed_tokens = num_computed_tokens + num_external_tokens
        kv_cache_groups = self.cpu_kv_cache_config.kv_cache_groups
        num_groups = len(kv_cache_groups)

        gpu_block_ids: list[int] = []
        cpu_block_ids: list[int] = []
        cpu_blocks_to_touch: list[KVCacheBlock] = []

        for g in range(num_groups):
            cpu_blocks_g = cpu_hit_blocks[g]
            n_ext_g = len(cpu_blocks_g)
            if n_ext_g == 0:
                continue

            # Number of blocks in the computed range for this group.
            g_block_size = kv_cache_groups[g].kv_cache_spec.block_size
            n_computed_g = cdiv(total_computed_tokens, g_block_size)

            # Back-trace: ext blocks sit at the tail of the computed range.
            gpu_ext_start = n_computed_g - n_ext_g
            group_gpu_ids = block_ids_by_group[g]

            for i, cpu_blk in enumerate(cpu_blocks_g):
                # Skip null blocks (e.g. sliding window or mamba padding).
                if cpu_blk.is_null:
                    continue
                gpu_block_ids.append(group_gpu_ids[gpu_ext_start + i])
                cpu_block_ids.append(cpu_blk.block_id)
                cpu_blocks_to_touch.append(cpu_blk)

        # Touch CPU blocks to prevent eviction during async load.
        self.cpu_block_pool.touch(cpu_blocks_to_touch)

        # Touch GPU blocks to prevent freeing during async load
        if self._gpu_block_pool is not None:
            self._gpu_block_pool.touch(
                [self._gpu_block_pool.blocks[bid] for bid in gpu_block_ids]
            )

        assert req_id not in self._reqs_to_load
        self._reqs_to_load[req_id] = RequestState(
            request=request,
            gpu_block_ids=block_ids_by_group,
            load_transfer=TransferMeta(gpu_block_ids, cpu_block_ids),
        )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> SimpleCPUOffloadMetadata:
        """Build metadata for worker to execute transfers this step."""
        # --- Stores ---
        store_event = -1
        store_gpu, store_cpu, store_req_ids = self.prepare_store_specs(scheduler_output)
        if store_gpu:
            store_event = self._store_event_counter
            self._store_event_counter += 1
            self._store_event_to_blocks[store_event] = TransferMeta(
                store_gpu,
                store_cpu,
            )
            if store_req_ids:  # For eager mode only, track req->blocks mapping
                self._store_event_to_reqs[store_event] = store_req_ids
                for req_id in store_req_ids:
                    state = self._reqs_to_store.get(req_id)
                    if state is not None:
                        state.store_events.add(store_event)

        # --- Loads ---
        load_event = -1
        load_gpu: list[int] = []
        load_cpu: list[int] = []
        load_req_ids: list[str] = []
        for req_id, state in self._reqs_to_load.items():
            if state.load_event is not None:
                continue
            transfer = state.load_transfer
            assert transfer is not None
            load_gpu.extend(transfer.gpu_block_ids)
            load_cpu.extend(transfer.cpu_block_ids)
            load_req_ids.append(req_id)
        if load_req_ids:
            load_event = self._load_event_counter
            self._load_event_counter += 1
            for req_id in load_req_ids:
                self._reqs_to_load[req_id].load_event = load_event
            self._load_event_to_reqs[load_event] = load_req_ids

        return SimpleCPUOffloadMetadata(
            load_event=load_event,
            load_gpu_blocks=load_gpu,
            load_cpu_blocks=load_cpu,
            load_event_to_reqs=self._load_event_to_reqs,
            store_event=store_event,
            store_gpu_blocks=store_gpu,
            store_cpu_blocks=store_cpu,
        )

    def prepare_store_specs(
        self, scheduler_output: SchedulerOutput
    ) -> tuple[list[int], list[int], list[str]]:
        """Prepare store specs for the store event."""
        if self._lazy_mode:
            return self._prepare_lazy_store_specs()
        else:
            return self._prepare_eager_store_specs(scheduler_output)

    def _prepare_lazy_store_specs(
        self,
    ) -> tuple[list[int], list[int], list[str]]:
        """Pick LRU-front GPU eviction candidates, allocate CPU slots.

        Touches GPU blocks (ref_cnt 0->1) to prevent eviction during async copy.
        On completion, update_connector_output decrements back to 0.

        Returns:
            (gpu_block_ids, cpu_block_ids, req_ids) for the store event.
        """
        total_tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
        n_lookahead = max(total_tokens // self.block_size, self._min_lookahead_blocks)
        if self._gpu_block_pool is None or n_lookahead <= 0:
            return [], [], []

        gpu_ids: list[int] = []
        block_hashes: list[bytes] = []

        num_free = self.cpu_block_pool.get_num_free_blocks()

        candidates = self._gpu_block_pool.get_eviction_candidates(n_lookahead)
        for gpu_block in candidates:
            bhash_with_group = gpu_block.block_hash
            if bhash_with_group is None:
                continue
            if (
                self.cpu_block_pool.cached_block_hash_to_block.get_one_block(
                    bhash_with_group
                )
                is not None
            ):
                continue
            if num_free <= 0:
                break
            num_free -= 1
            gpu_ids.append(gpu_block.block_id)
            block_hashes.append(bhash_with_group)

        # Batch allocate CPU blocks and stamp their hashes ahead of time.
        if gpu_ids:
            cpu_blocks_alloc = self.cpu_block_pool.get_new_blocks(len(gpu_ids))
            cpu_ids = [blk.block_id for blk in cpu_blocks_alloc]
            for cpu_blk, bhash in zip(cpu_blocks_alloc, block_hashes):
                cpu_blk._block_hash = bhash
            # Touch GPU blocks to prevent freeing during async copy
            self._gpu_block_pool.touch(
                [self._gpu_block_pool.blocks[bid] for bid in gpu_ids]
            )
        else:
            cpu_ids = []

        return gpu_ids, cpu_ids, []

    def _prepare_eager_store_specs(
        self,
        scheduler_output: SchedulerOutput,
    ) -> tuple[list[int], list[int], list[str]]:
        """Identify newly computed blocks to offload from scheduler requests.

        Iterates over all KV cache groups for each request, retrieving block
        hashes directly from GPU blocks so that hash_block_size vs group
        block_size mismatches are handled correctly.

        Returns:
            (gpu_block_ids, cpu_block_ids, req_ids) for the store event.
        """
        merged_gpu_block_ids: list[int] = []
        merged_cpu_block_ids: list[int] = []
        req_ids: list[str] = []

        gpu_block_pool = self._gpu_block_pool
        cpu_block_pool = self.cpu_block_pool
        num_free = cpu_block_pool.get_num_free_blocks()
        num_groups = len(self.cpu_kv_cache_config.kv_cache_groups)
        if gpu_block_pool is None:
            return [], [], []

        for req_id, num_new_tokens in scheduler_output.num_scheduled_tokens.items():
            if num_new_tokens == 0:
                continue

            state = self._reqs_to_store.get(req_id)
            if state is None or state.finished:
                continue

            # --- Phase 1: Scan blocks, classify as cached vs to-store ---
            gpu_block_ids: list[int] = []
            block_hashes_to_store: list[bytes] = []
            advanced_per_group: list[int] = [0] * num_groups
            out_of_space = False

            for g in range(num_groups):
                # FIXME (yifan): handle CPU cache eviction, where
                # num_stored_blocks can be stale and omit evicted blocks in
                # the middle of the request.
                already_stored_g = state.num_stored_blocks[g]
                group_gpu_ids = state.gpu_block_ids[g]

                for gpu_block_id in group_gpu_ids[already_stored_g:]:
                    gpu_block = gpu_block_pool.blocks[gpu_block_id]
                    bhash_with_group = gpu_block.block_hash
                    if bhash_with_group is None:
                        break

                    # Check if this group's data is already cached in CPU.
                    if (
                        cpu_block_pool.cached_block_hash_to_block.get_one_block(
                            bhash_with_group
                        )
                        is not None
                    ):
                        advanced_per_group[g] += 1
                        continue

                    if num_free <= 0:
                        out_of_space = True
                        break
                    num_free -= 1

                    gpu_block_ids.append(gpu_block_id)
                    block_hashes_to_store.append(bhash_with_group)
                    advanced_per_group[g] += 1

                if out_of_space:
                    break

            # --- Phase 2: Batch allocate CPU blocks and stamp hashes ---
            n_to_alloc = len(gpu_block_ids)
            if n_to_alloc > 0:
                cpu_blocks_alloc = cpu_block_pool.get_new_blocks(n_to_alloc)
                cpu_block_ids = [blk.block_id for blk in cpu_blocks_alloc]
                for cpu_blk, bhash in zip(cpu_blocks_alloc, block_hashes_to_store):
                    cpu_blk._block_hash = bhash
            else:
                cpu_block_ids = []

            if cpu_block_ids:
                req_ids.append(req_id)
                merged_gpu_block_ids.extend(gpu_block_ids)
                merged_cpu_block_ids.extend(cpu_block_ids)

                # Touch GPU blocks to prevent freeing during async copy
                gpu_block_pool.touch(
                    [gpu_block_pool.blocks[bid] for bid in gpu_block_ids]
                )

                logger.debug(
                    "Request %s: Scheduling store of %d blocks to CPU (%d groups)",
                    req_id,
                    len(cpu_block_ids),
                    num_groups,
                )

            # Advance per-group cursors (includes cached hits + newly stored)
            for g in range(num_groups):
                state.num_stored_blocks[g] += advanced_per_group[g]

        return (
            merged_gpu_block_ids,
            merged_cpu_block_ids,
            req_ids,
        )

    def update_connector_output(
        self,
        connector_output: KVConnectorOutput,
    ) -> None:
        """Handle async transfer completions from worker.

        The worker treats load and store differently:
        - For load which blocks are tightly coupled with requests, the worker reports
            finished_recving with the request ID.
        - For store which blocks are not tightly coupled with requests, the worker
            reports finished_sending with the event index, and the scheduler should
            update request metadata accordingly.

        The connector emits event-index sentinels (__load_done_N,
        __store_done_N). We translate those back to req_ids using our
        inverse maps, process completions, and mutate
        connector_output.finished_recving with real req_ids for the
        scheduler.
        """
        # --- Load completions ---
        # Unlike stores, a request has at most one load event, so completion means done
        for req_id in list(connector_output.finished_recving or []):
            self._cleanup_load_request(req_id)

        # --- Store completions ---
        for sentinel in connector_output.finished_sending or []:
            event_idx = int(sentinel[len("__store_done_") :])

            # Both lazy and eager: process event-level blocks
            transfer = self._store_event_to_blocks.pop(event_idx)
            self._process_store_completion(
                transfer.gpu_block_ids, transfer.cpu_block_ids
            )
            logger.debug(
                "Store event %d completed: cached %d blocks to CPU",
                event_idx,
                len(transfer.cpu_block_ids),
            )

            # Eager only: update per-req state
            if not self._lazy_mode:
                for req_id in self._store_event_to_reqs.pop(event_idx, []):
                    state = self._reqs_to_store.get(req_id)
                    if state is None:
                        continue
                    state.store_events.discard(event_idx)
                    if state.finished and not state.store_events:
                        self._cleanup_store_request(req_id)

        # Scheduler doesn't need finished_sending since we protect blocks with ref_cnt.
        connector_output.finished_sending = None

    def _process_store_completion(
        self, gpu_block_ids: list[int], cpu_block_ids: list[int]
    ) -> None:
        """Cache CPU blocks per-group and release GPU refs.

        Block hashes were stamped on CPU blocks at allocation time (in
        ``_prepare_*_store_specs``).  Here we just register them in the
        cache map so they become discoverable by the load path.
        """
        assert len(cpu_block_ids) == len(gpu_block_ids)

        cpu_blocks = [self.cpu_block_pool.blocks[bid] for bid in cpu_block_ids]

        for cpu_block in cpu_blocks:
            bhash = cpu_block.block_hash
            assert bhash is not None
            self.cpu_block_pool.cached_block_hash_to_block.insert(bhash, cpu_block)

        self.cpu_block_pool.free_blocks(cpu_blocks)
        if self._gpu_block_pool is not None:
            self._gpu_block_pool.free_blocks(
                self._gpu_block_pool.blocks[bid] for bid in gpu_block_ids
            )

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Always returns (False, None). GPU blocks are protected by ref_cnt,
        so the scheduler can free blocks immediately."""
        req_id = request.request_id

        # Handle load: defer cleanup if load is in-flight
        load_state = self._reqs_to_load.get(req_id)
        if load_state is not None:
            if load_state.load_event is not None:
                load_state.finished = True  # Defer: load in-flight
            else:
                self._cleanup_load_request(req_id)

        # Handle store (eager mode only): defer cleanup if stores in-flight
        if not self._lazy_mode:
            store_state = self._reqs_to_store.get(req_id)
            if store_state is not None:
                if store_state.store_events:
                    store_state.finished = True  # Defer: stores in-flight
                else:
                    self._cleanup_store_request(req_id)

        # FIXME (yifan): remove this after the connector API is modified.
        if req_id in self._req_local_computed:
            del self._req_local_computed[req_id]

        return False, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        return self.request_finished(request, block_ids=[])

    def _cleanup_load_request(self, req_id: str) -> None:
        """Release all load resources for a request.

        Shared between request_finished() and update_connector_output() paths.
        Removes the request from _reqs_to_load, cleans up event mappings,
        and frees CPU/GPU touch refs.
        """
        state = self._reqs_to_load.pop(req_id, None)
        if state is None:
            return
        # Remove from load event mapping (only this req, not whole event)
        if state.load_event is not None:
            reqs = self._load_event_to_reqs.get(state.load_event)
            if reqs is not None:
                with contextlib.suppress(ValueError):
                    reqs.remove(req_id)
                if not reqs:
                    self._load_event_to_reqs.pop(state.load_event, None)
        # Free CPU touch refs
        if state.load_transfer is not None:
            self.cpu_block_pool.free_blocks(
                self.cpu_block_pool.blocks[bid]
                for bid in state.load_transfer.cpu_block_ids
            )
        # Free GPU touch refs
        if state.load_transfer is not None and self._gpu_block_pool is not None:
            self._gpu_block_pool.free_blocks(
                self._gpu_block_pool.blocks[bid]
                for bid in state.load_transfer.gpu_block_ids
            )

    def _cleanup_store_request(self, req_id: str) -> None:
        """Release store metadata for a request.

        Metadata-only cleanup — no block freeing. Job completion handles
        block caching and GPU ref freeing via _process_store_completion().
        """
        state = self._reqs_to_store.pop(req_id, None)
        if state is None:
            return
        for event_idx in list(state.store_events):
            reqs = self._store_event_to_reqs.get(event_idx)
            if reqs is not None:
                with contextlib.suppress(ValueError):
                    reqs.remove(req_id)
                if not reqs:
                    self._store_event_to_reqs.pop(event_idx, None)
        state.store_events.clear()

    def take_events(self) -> Iterable[KVCacheEvent]:
        """Return KV cache events for telemetry."""
        return self.cpu_block_pool.take_events()
