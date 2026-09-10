# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-type KV cache managers for the HiSparse host, resident and hot groups.

The host group owns a private host block pool and publishes prefix hashes only
once its pages are durable; the resident and hot groups are per-request GPU
state whose lifecycle is driven by ``HiSparseCoordinator`` through the
standard manager hooks. ``HiSparseCoordinator`` binds itself to every HiSparse
manager when the scheduler binds the KV cache manager to the connector.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.distributed.kv_events import MEDIUM_CPU, KVCacheEvent
from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHashList, KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    SingleTypeKVCacheManager,
)
from vllm.v1.kv_cache_interface import HiSparseHotSpec, KVCacheSpec
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.v1.hisparse.coordinator import HiSparseCoordinator


class _SharedEventQueueBlockPool(BlockPool):
    """A pool that publishes into another pool's live KV event queue.

    The owner rebinds its queue on every drain, so this reads it through the
    owner rather than holding a reference.
    """

    def __init__(self, *args, event_owner: BlockPool, **kwargs) -> None:
        self._event_owner = event_owner
        super().__init__(*args, **kwargs)

    @property
    def kv_event_queue(self) -> list[KVCacheEvent]:
        return self._event_owner.kv_event_queue

    @kv_event_queue.setter
    def kv_event_queue(self, events: list[KVCacheEvent]) -> None:
        # ``BlockPool.__init__`` seeds an empty queue and ``take_events``
        # swaps in a fresh one; both belong to the owner, which drains it.
        assert not events


class HiSparseSourceManager(FullAttentionManager):
    """Host-tier manager with a private pool; publishes hashes once durable.

    Host capacity is best effort: a page that cannot get a host block keeps a
    null host entry, so its GPU copy is never written back and stays pinned.
    """

    coordinator: "HiSparseCoordinator | None" = None
    # The host tier keeps prefixes the device groups have already lost.
    retains_longer_hit = True

    @property
    def records_new_block_ids(self) -> bool:
        return False

    def take_new_block_ids(self) -> list[int]:
        self.new_block_ids = []
        return []

    def bind_host_pool(self, num_blocks: int) -> None:
        """Replace the device pool this group was built with by a host one."""
        device_pool = self.block_pool
        self.block_pool = _SharedEventQueueBlockPool(
            num_gpu_blocks=num_blocks,
            enable_caching=self.enable_caching and self.kv_cache_spec.prefix_cacheable,
            hash_block_size=device_pool.hash_block_size,
            enable_kv_cache_events=device_pool.enable_kv_cache_events,
            metrics_collector=device_pool.metrics_collector,
            medium=MEDIUM_CPU,
            event_owner=device_pool,
        )
        self._null_block = self.block_pool.null_block

    def take_pending_cow_copies(self) -> list[tuple[KVCacheBlock, KVCacheBlock]]:
        """Host copies never reach the worker's generic block-copy path."""
        return []

    def take_host_cow_copies(self) -> list[tuple[KVCacheBlock, KVCacheBlock]]:
        """Drain host copies for the coordinator to hand to the connector."""
        copies = self._pending_cow_copies
        self._pending_cow_copies = []
        return copies

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        # Host blocks come from the private pool and are never a device cost.
        return 0

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list[KVCacheBlock]:
        req_blocks = self.req_to_blocks[request_id]
        num_new_blocks = cdiv(num_tokens, self.block_size) - len(req_blocks)
        num_free = self.block_pool.get_num_free_blocks()
        if num_new_blocks <= num_free:
            return super().allocate_new_blocks(
                request_id, num_tokens, num_tokens_main_model
            )
        # Host exhaustion: take what fits and leave the rest without a host
        # page, so those GPU pages are never written back and stay pinned.
        new_blocks: list[KVCacheBlock] = []
        if num_free:
            fit_tokens = (len(req_blocks) + num_free) * self.block_size
            new_blocks = super().allocate_new_blocks(
                request_id, fit_tokens, min(num_tokens_main_model, fit_tokens)
            )
        req_blocks.extend([self._null_block] * (num_new_blocks - len(new_blocks)))
        return new_blocks

    def allocate_external_computed_blocks(
        self,
        request_id: str,
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        # The connector writes these host pages directly; they only become
        # readable once the request is committed at that prefix length.
        assert self.coordinator is not None
        self.coordinator.record_pending_host_import(
            request_id, num_local_computed_tokens + num_external_computed_tokens
        )
        super().allocate_external_computed_blocks(
            request_id, num_local_computed_tokens, num_external_computed_tokens
        )

    def cache_blocks(
        self,
        request: Request,
        num_tokens: int,
        retention_interval: int | None = None,
        *,
        replay_boundaries: Sequence[int],
    ) -> None:
        assert self.coordinator is not None
        self.coordinator.complete_pending_host_import(request.request_id, num_tokens)
        self.coordinator.publish_when_ready(
            request,
            num_tokens,
            retention_interval,
            replay_boundaries=replay_boundaries,
        )

    def publish_blocks(
        self,
        request: Request,
        num_tokens: int,
        retention_interval: int | None = None,
        *,
        replay_boundaries: Sequence[int],
    ) -> None:
        super().cache_blocks(
            request,
            num_tokens,
            retention_interval=retention_interval,
            replay_boundaries=replay_boundaries,
        )

    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
        assert self.coordinator is not None
        self.coordinator.free(request_id)
        return super().pop_blocks_for_free(request_id)


class _HiSparseAuxiliaryManager(SingleTypeKVCacheManager):
    """Base for ephemeral groups whose host source owns prefix caching."""

    coordinator: "HiSparseCoordinator | None" = None

    def __init__(self, kv_cache_spec: KVCacheSpec, **kwargs) -> None:
        # Never prefix-cached, but the per-step ``cache_blocks`` hook is where
        # residency work runs, so stay opted in regardless of prefix caching.
        kwargs["enable_caching"] = True
        super().__init__(kv_cache_spec, **kwargs)

    def cache_blocks(
        self,
        request: Request,
        num_tokens: int,
        retention_interval: int | None = None,
        *,
        replay_boundaries: Sequence[int],
    ) -> None:
        return None

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        return 0

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        drop_eagle_block: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        return tuple([] for _ in kv_cache_group_ids), 0


class HiSparseHotManager(_HiSparseAuxiliaryManager):
    """Allocate a hot region only after a request acquires CPU-only history."""

    def __init__(self, kv_cache_spec: HiSparseHotSpec, **kwargs) -> None:
        super().__init__(kv_cache_spec, **kwargs)
        self.blocks_per_request = kv_cache_spec.blocks_per_request
        self.hot_required: set[str] = set()

    def require_hot(self, request_id: str) -> None:
        self.hot_required.add(request_id)

    def has_hot(self, request_id: str) -> bool:
        return len(self.req_to_blocks.get(request_id, ())) == self.blocks_per_request

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        assert not new_computed_blocks
        # A hot region is needed to read host-backed history: an external
        # import, a new request resuming a host prefix, or one already asked
        # to transition. Running requests keep their earlier answer.
        host_import = total_computed_tokens > num_local_computed_tokens
        resumes_host_prefix = (
            num_local_computed_tokens > 0 and request_id not in self.num_cached_block
        )
        if host_import or resumes_host_prefix or request_id in self.hot_required:
            return self.get_num_required_blocks(request_id)
        return 0

    def get_num_required_blocks(self, request_id: str) -> int:
        return max(
            self.blocks_per_request - len(self.req_to_blocks.get(request_id, ())),
            0,
        )

    def add_local_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        assert not new_computed_blocks
        self.num_cached_block[request_id] = 0
        assert self.coordinator is not None
        if num_local_computed_tokens > 0 or not self.coordinator.resident_managers:
            self.require_hot(request_id)

    def allocate_external_computed_blocks(
        self,
        request_id: str,
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        self.require_hot(request_id)

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list[KVCacheBlock]:
        # Cold admissions can bypass add_local_computed_blocks.
        self.num_cached_block[request_id] = 0
        if request_id not in self.hot_required:
            return []
        req_blocks = self.req_to_blocks[request_id]
        num_new_blocks = self.blocks_per_request - len(req_blocks)
        if num_new_blocks <= 0:
            return []
        new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
        req_blocks.extend(new_blocks)
        return new_blocks

    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
        self.hot_required.discard(request_id)
        return super().pop_blocks_for_free(request_id)


class HiSparseResidentManager(_HiSparseAuxiliaryManager):
    """Track GPU-resident pages for otherwise host-backed KV."""

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        del num_tokens_main_model
        assert not new_computed_blocks
        if total_computed_tokens > num_local_computed_tokens:
            if total_computed_tokens % self.block_size != 0:
                raise ValueError(
                    "A host-only HiSparse import must end on a cache-block boundary."
                )
            imported_pages = total_computed_tokens // self.block_size
            return max(cdiv(num_tokens, self.block_size) - imported_pages, 0)
        existing = len(self.req_to_blocks.get(request_id, ()))
        host_pages = cdiv(num_local_computed_tokens, self.block_size)
        required = cdiv(num_tokens, self.block_size)
        if apply_admission_cap:
            assert self._max_admission_blocks_per_request is not None
            required = min(required, self._max_admission_blocks_per_request)
        return max(required - max(existing, host_pages), 0)

    def allocate_external_computed_blocks(
        self,
        request_id: str,
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        """Represent imported host history with null resident pages."""
        assert num_external_computed_tokens > 0
        num_tokens = num_local_computed_tokens + num_external_computed_tokens
        blocks = self.req_to_blocks[request_id]
        tail_page = cdiv(num_tokens, self.block_size) - 1
        if len(blocks) <= tail_page:
            blocks.extend([self._null_block] * (tail_page + 1 - len(blocks)))

    def add_local_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        assert not new_computed_blocks
        req_blocks = self.req_to_blocks[request_id]
        assert not req_blocks
        num_host_pages = cdiv(num_local_computed_tokens, self.block_size)
        req_blocks.extend([self._null_block] * num_host_pages)
        self.num_cached_block[request_id] = 0
        assert self.coordinator is not None
        self.coordinator.commit_computed_blocks(request_id, num_host_pages)

    def cache_blocks(
        self,
        request: Request,
        num_tokens: int,
        retention_interval: int | None = None,
        *,
        replay_boundaries: Sequence[int],
    ) -> None:
        assert self.coordinator is not None
        self.coordinator.plan_prefix_materialization(request.request_id, num_tokens)
        self.coordinator.update_residency(request.request_id)

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list[KVCacheBlock]:
        del num_tokens_main_model
        req_blocks = self.req_to_blocks[request_id]
        num_new_blocks = cdiv(num_tokens, self.block_size) - len(req_blocks)
        if num_new_blocks <= 0:
            return []
        new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
        req_blocks.extend(new_blocks)
        return new_blocks

    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
        assert self.coordinator is not None
        self.coordinator.free(request_id)
        return super().pop_blocks_for_free(request_id)

    def adopt_resident_page(
        self, request_id: str, block_idx: int, block: KVCacheBlock
    ) -> bool:
        """Point a null prefix page at a pinned GPU copy of its contents."""
        blocks = self.req_to_blocks.get(request_id)
        if blocks is None or block_idx >= len(blocks) or not blocks[block_idx].is_null:
            return False
        blocks[block_idx] = block
        return True

    def get_resident_page(self, request_id: str, block_idx: int) -> KVCacheBlock | None:
        blocks = self.req_to_blocks.get(request_id)
        if blocks is None or block_idx >= len(blocks):
            return None
        block = blocks[block_idx]
        return None if block.is_null else block
