# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHashWithGroupId,
    KVCacheBlock,
    KVCacheBlockCopy,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    HiSparseHotManager,
    HiSparseResidentManager,
    HiSparseSourceManager,
    SingleTypeKVCacheManager,
)
from vllm.v1.hisparse.types import (
    SparseKVOffloadCommand,
    SparseKVPageTransfer,
    SparseKVRowMirror,
)
from vllm.v1.kv_cache_interface import (
    HiSparseResidentSpec,
    KVCacheConfig,
)
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheManager

# Sealed pages this many positions behind the block-table tail stay pinned so
# a page written by an in-flight step is never handed out under it.
_ACTIVE_TAIL_PAGES = 2


@dataclass
class _PendingPublication:
    request: Request
    num_computed_tokens: int
    num_pages: int
    retention_interval: int | None
    replay_boundaries: Sequence[int]


@dataclass
class _HiSparseRequestState:
    """Residency bookkeeping for one request.

    A resident page moves through three states: dirty (only the GPU copy
    exists), clean (``valid_pages``: the host copy is complete) and unpinned
    (``unpinned_pages``: clean, and its allocation reference was released so
    the pool may reuse it). ``pinned_clean`` tracks clean pages still holding
    their reference, which only happens before the request can read from host.
    """

    valid_pages: set[int] = field(default_factory=set)
    ready_prefix_pages: int = 0
    pending_pages: dict[int, int] = field(default_factory=dict)
    publication: _PendingPublication | None = None
    copies_recorded_blocks: int = 0
    pinned_clean: set[int] = field(default_factory=set)
    unpinned_pages: set[int] = field(default_factory=set)


@dataclass
class _PendingSpill:
    transfer_id: int
    page: tuple[str, int]
    request_state: _HiSparseRequestState
    host_block: KVCacheBlock
    resident_blocks: tuple[KVCacheBlock, ...]
    expected_worker_completions: int = 0
    worker_completions: int = 0
    enqueue_applied: bool = False


class HiSparseCoordinator:
    """Own HiSparse host allocation, host publication, spills, and GPU residency.

    GPU-resident pages are a write-back cache of the host tier. Once a page's
    host copy is durable and its owner can read from host (it has, or has
    asked for, a hot region), the page's allocation reference is released with
    ``BlockPool.unpin_blocks``: the block stays in the block table and keeps
    being read, but the pool counts it as free and may hand it out, at which
    point the owner's page is nulled and its block table republished.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        managers: tuple[SingleTypeKVCacheManager, ...],
        max_model_len: int,
    ) -> None:
        self.managers = managers
        self.max_model_len = max_model_len
        groups = kv_cache_config.kv_cache_groups

        resident_managers: list[HiSparseResidentManager] = []
        hot_managers: list[HiSparseHotManager] = []
        resident_group_ids: list[int] = []
        for group_id, (group, manager) in enumerate(zip(groups, managers)):
            if isinstance(group.kv_cache_spec, HiSparseResidentSpec):
                if not isinstance(manager, HiSparseResidentManager):
                    raise TypeError(
                        "HiSparse resident specs require resident cache managers."
                    )
                resident_managers.append(manager)
                resident_group_ids.append(group_id)
                assert not group.host_resident
            if isinstance(manager, HiSparseHotManager):
                hot_managers.append(manager)
                assert not group.host_resident
            if isinstance(manager, (HiSparseHotManager, HiSparseResidentManager)):
                manager.coordinator = self
        self.resident_managers = tuple(resident_managers)
        self.hot_managers = tuple(hot_managers)
        self.host_manager: HiSparseSourceManager | None = None
        self.host_group_id: int | None = None
        self.max_spill_pages = 0
        for group_id, manager in enumerate(managers):
            if isinstance(manager, HiSparseSourceManager):
                if self.host_manager is not None:
                    raise ValueError("Only one HiSparse host group is supported.")
                num_host_blocks = kv_cache_config.hisparse_host_num_blocks
                if num_host_blocks is None:
                    raise ValueError("HiSparse host group needs host capacity.")
                self.host_group_id = group_id
                self.host_manager = manager
                manager.coordinator = self
                manager.bind_host_pool(num_host_blocks)
        self.has_host_cache = self.host_manager is not None
        self.gpu_pool: BlockPool | None = None
        self.transition_watermark = 0
        if self.resident_managers:
            assert self.host_manager is not None
            assert self.host_group_id is not None
            if self.host_group_id > min(resident_group_ids):
                raise ValueError("HiSparse host group must precede resident groups.")
            resident_block_sizes = {
                manager.block_size for manager in self.resident_managers
            }
            if len(resident_block_sizes) != 1:
                raise ValueError(
                    "HiSparse resident cache groups must use one block size."
                )
            resident_block_size = resident_block_sizes.pop()
            if self.host_manager.block_size != resident_block_size:
                raise ValueError("HiSparse host and resident block sizes must match.")
            self.max_spill_pages = max_model_len // resident_block_size
            self.gpu_pool = self.resident_managers[0].block_pool
            # Requests start reading from host once the shared pool runs this
            # low, so admissions find evictable pages rather than pinned ones.
            hot_cost = sum(manager.blocks_per_request for manager in hot_managers)
            self.transition_watermark = max(hot_cost, kv_cache_config.num_blocks // 10)

        # Host copies handed to the worker, by destination block id, until it
        # reports having run them.
        self._retained_copies: dict[int, tuple[KVCacheBlock, KVCacheBlock]] = {}
        # request -> prefix length an external load is still filling in.
        self._pending_imports: dict[str, int] = {}
        self.block_table_updates: set[str] = set()
        self.spills_to_send: list[SparseKVPageTransfer] = []
        self.pending_spills: dict[int, _PendingSpill] = {}
        self.request_states: dict[str, _HiSparseRequestState] = {}
        self.next_spill_id = 0
        # Published host block hash -> GPU copies of that page, readable until
        # the pool reuses them. Lets a host prefix hit come back GPU-resident.
        self.copies: dict[BlockHashWithGroupId, tuple[KVCacheBlock, ...]] = {}
        self._copy_by_block: dict[int, BlockHashWithGroupId] = {}
        # Unpinned GPU block id -> (request, page) pairs still reading it.
        self._owners: dict[int, set[tuple[str, int]]] = {}

    def _get_request_state(self, request_id: str) -> _HiSparseRequestState:
        state = self.request_states.get(request_id)
        if state is None:
            state = _HiSparseRequestState()
            self.request_states[request_id] = state
        return state

    def commit_computed_blocks(self, request_id: str, num_host_pages: int) -> None:
        """Account for a prefix hit on host pages the request now references."""
        if not self.resident_managers or self.host_manager is None:
            return
        host_blocks = self.host_manager.req_to_blocks.get(request_id, ())
        num_host_pages = min(num_host_pages, len(host_blocks))
        if num_host_pages == 0:
            return
        state = self._get_request_state(request_id)
        state.valid_pages.update(range(num_host_pages))
        state.ready_prefix_pages = max(num_host_pages, state.ready_prefix_pages)
        self._adopt_copies(request_id, state, host_blocks[:num_host_pages])
        state.copies_recorded_blocks = max(state.copies_recorded_blocks, num_host_pages)

    def take_host_block_copies(self) -> tuple[KVCacheBlockCopy, ...]:
        """Drain host copy-on-write work, retaining both endpoints.

        The worker runs the copies in the step that carries them, so the
        endpoints are released when that step's output comes back.
        """
        if self.host_manager is None:
            return ()
        pairs = self.host_manager.take_host_cow_copies()
        for source_block, cow_block in pairs:
            self._retained_copies[cow_block.block_id] = (source_block, cow_block)
        return tuple(
            KVCacheBlockCopy(
                src_block_id=source_block.block_id,
                dst_block_id=cow_block.block_id,
            )
            for source_block, cow_block in pairs
        )

    def release_completed_host_copies(self, dst_block_ids: Iterable[int]) -> None:
        """Release the endpoints of host copies the worker reports having run."""
        blocks: list[KVCacheBlock] = []
        for dst_block_id in dst_block_ids:
            pair = self._retained_copies.pop(dst_block_id, None)
            if pair is not None:
                blocks.extend(pair)
        if not blocks:
            return
        assert self.host_manager is not None
        self.host_manager.block_pool.free_blocks(reversed(blocks))

    def get_host_block_pool(self) -> BlockPool | None:
        manager = self.host_manager
        return manager.block_pool if manager is not None else None

    # ------------------------------------------------------------------
    # GPU copies of published host pages
    # ------------------------------------------------------------------

    def _adopt_copies(
        self,
        request_id: str,
        state: _HiSparseRequestState,
        host_blocks: Sequence[KVCacheBlock],
    ) -> None:
        """Point null prefix pages at readable GPU copies of their contents.

        Runs after the hit length is fully reconciled, so copy hits and misses
        can be scattered per page without affecting the prefix hit.
        """
        if not self.copies:
            return
        for host_idx, host_block in enumerate(host_blocks):
            if host_block.is_null or host_block.block_hash is None:
                continue
            blocks = self.copies.get(host_block.block_hash)
            if blocks is None:
                continue
            for manager, block in zip(self.resident_managers, blocks):
                if manager.adopt_resident_page(request_id, host_idx, block):
                    manager.block_pool.touch([block])
                    state.pinned_clean.add(host_idx)

    def _record_copies(self, request_id: str, num_computed_tokens: int) -> None:
        """Index the GPU copies of just-published host blocks for later hits."""
        if not self.resident_managers:
            return
        assert self.host_manager is not None
        host_blocks = self.host_manager.req_to_blocks.get(request_id)
        if not host_blocks:
            return
        state = self._get_request_state(request_id)
        num_host_blocks = min(
            num_computed_tokens // self.host_manager.block_size, len(host_blocks)
        )
        for host_idx in range(state.copies_recorded_blocks, num_host_blocks):
            host_block = host_blocks[host_idx]
            if host_block.is_null or host_block.block_hash is None:
                continue
            blocks: list[KVCacheBlock] = []
            for manager in self.resident_managers:
                block = manager.get_resident_page(request_id, host_idx)
                if block is None:
                    break
                blocks.append(block)
            if len(blocks) != len(self.resident_managers):
                continue
            self._drop_copy(host_block.block_hash)
            self.copies[host_block.block_hash] = tuple(blocks)
            for block in blocks:
                self._copy_by_block[block.block_id] = host_block.block_hash
        state.copies_recorded_blocks = max(
            state.copies_recorded_blocks, num_host_blocks
        )

    def _drop_copy(self, host_hash: BlockHashWithGroupId) -> None:
        blocks = self.copies.pop(host_hash, None)
        if blocks is None:
            return
        for block in blocks:
            self._copy_by_block.pop(block.block_id, None)

    # ------------------------------------------------------------------
    # Residency: pinning, unpinning and losing pages
    # ------------------------------------------------------------------

    def _can_read_from_host(self, request_id: str) -> bool:
        """Whether resident pages may be dropped under the request."""
        return bool(self.hot_managers) and all(
            manager.has_hot(request_id) for manager in self.hot_managers
        )

    def _resident_page_blocks(
        self, request_id: str, page_idx: int
    ) -> list[KVCacheBlock] | None:
        blocks: list[KVCacheBlock] = []
        for manager in self.resident_managers:
            req_blocks = manager.req_to_blocks.get(request_id)
            if (
                req_blocks is None
                or page_idx >= len(req_blocks) - _ACTIVE_TAIL_PAGES
                or req_blocks[page_idx].is_null
            ):
                return None
            blocks.append(req_blocks[page_idx])
        return blocks

    def _unpin_page(
        self, request_id: str, state: _HiSparseRequestState, page_idx: int
    ) -> bool:
        if page_idx in state.pending_pages or page_idx not in state.valid_pages:
            return False
        blocks = self._resident_page_blocks(request_id, page_idx)
        if blocks is None:
            return False
        for manager, block in zip(self.resident_managers, blocks):
            self._owners.setdefault(block.block_id, set()).add((request_id, page_idx))
            manager.block_pool.unpin_blocks([block], self._on_block_reused)
        state.pinned_clean.discard(page_idx)
        state.unpinned_pages.add(page_idx)
        return True

    def _unpin_clean_pages(self, request_id: str, state: _HiSparseRequestState) -> None:
        for page_idx in sorted(state.pinned_clean):
            self._unpin_page(request_id, state, page_idx)

    def _page_became_clean(
        self, request_id: str, state: _HiSparseRequestState, page_idx: int
    ) -> None:
        state.pinned_clean.add(page_idx)
        if self._can_read_from_host(request_id):
            self._unpin_page(request_id, state, page_idx)

    def _on_block_reused(self, block: KVCacheBlock) -> None:
        host_hash = self._copy_by_block.get(block.block_id)
        if host_hash is not None:
            self._drop_copy(host_hash)
        for request_id, page_idx in self._owners.pop(block.block_id, ()):
            self._lose_page(request_id, page_idx)

    def _lose_page(self, request_id: str, page_idx: int) -> None:
        state = self.request_states.get(request_id)
        if state is None:
            return
        for manager in self.resident_managers:
            blocks = manager.req_to_blocks.get(request_id)
            if blocks is None or page_idx >= len(blocks) or blocks[page_idx].is_null:
                continue
            block = blocks[page_idx]
            blocks[page_idx] = manager._null_block
            owners = self._owners.get(block.block_id)
            if owners is not None:
                owners.discard((request_id, page_idx))
        state.unpinned_pages.discard(page_idx)
        self.block_table_updates.add(request_id)
        for hot_manager in self.hot_managers:
            hot_manager.require_hot(request_id)

    def update_residency(self, request_id: str) -> None:
        """Per-step residency policy for a scheduled request.

        A request that can read from host releases every clean sealed page to
        the pool. One that cannot keeps its pages pinned until the shared pool
        runs low, then asks for a hot region. Pages remain pinned until that region
        is allocated on a subsequent scheduling pass.
        """
        if not self.resident_managers:
            return
        state = self._get_request_state(request_id)
        if not self._can_read_from_host(request_id):
            assert self.gpu_pool is not None
            if self.gpu_pool.get_num_free_blocks() >= self.transition_watermark:
                return
            for manager in self.hot_managers:
                manager.require_hot(request_id)
            return
        self._unpin_clean_pages(request_id, state)

    # ------------------------------------------------------------------
    # Host publication and spills
    # ------------------------------------------------------------------

    def plan_prefix_materialization(
        self, request_id: str, num_computed_tokens: int
    ) -> None:
        """Eagerly write back every sealed page that has no host copy yet."""
        if not self.resident_managers:
            return
        assert self.host_manager is not None
        host_block_size = self.host_manager.block_size
        num_pages = num_computed_tokens // host_block_size
        state = self._get_request_state(request_id)
        budget = max(self.max_spill_pages - len(self.spills_to_send), 0)
        for page_idx in range(num_pages):
            if budget == 0:
                break
            if page_idx in state.pending_pages:
                continue
            if page_idx in state.valid_pages:
                continue
            if not all(
                manager.get_resident_page(request_id, page_idx) is not None
                for manager in self.resident_managers
            ):
                continue
            if self._plan_spill(request_id, page_idx, after_forward=True):
                budget -= 1

    def publish_when_ready(
        self,
        request: Request,
        num_computed_tokens: int,
        retention_interval: int | None,
        *,
        replay_boundaries: Sequence[int],
    ) -> None:
        """Publish host-source hashes only after their pages are durable."""
        manager = self.host_manager
        if manager is None:
            return
        num_pages = num_computed_tokens // manager.block_size
        request_id = request.request_id
        state = self._get_request_state(request_id)
        if state.ready_prefix_pages >= num_pages:
            manager.publish_blocks(
                request,
                num_computed_tokens,
                retention_interval=retention_interval,
                replay_boundaries=replay_boundaries,
            )
            self._record_copies(request_id, num_computed_tokens)
            state.publication = None
            return
        state.publication = _PendingPublication(
            request=request,
            num_computed_tokens=num_computed_tokens,
            num_pages=num_pages,
            retention_interval=retention_interval,
            replay_boundaries=replay_boundaries,
        )

    def _publish_host_blocks_if_ready(self, request_id: str) -> None:
        state = self.request_states.get(request_id)
        if state is None:
            return
        publication = state.publication
        if publication is None or state.ready_prefix_pages < publication.num_pages:
            return
        assert self.host_manager is not None
        self.host_manager.publish_blocks(
            publication.request,
            publication.num_computed_tokens,
            retention_interval=publication.retention_interval,
            replay_boundaries=publication.replay_boundaries,
        )
        self._record_copies(request_id, publication.num_computed_tokens)
        state.publication = None

    def record_pending_host_import(self, request_id: str, num_tokens: int) -> None:
        """Note a prefix an external load is populating in host pages."""
        self._pending_imports[request_id] = num_tokens

    def finish_host_import(self, request_id: str, *, failed: bool) -> None:
        num_tokens = self._pending_imports.pop(request_id, None)
        if num_tokens is not None and not failed:
            self.complete_host_import(request_id, num_tokens)

    def complete_host_import(self, request_id: str, num_computed_tokens: int) -> None:
        """Publish externally populated host pages after connector completion."""
        if not self.resident_managers:
            return
        block_size = self.resident_managers[0].block_size
        num_pages = num_computed_tokens // block_size
        state = self._get_request_state(request_id)
        state.valid_pages = set(range(num_pages))
        state.ready_prefix_pages = num_pages
        self._publish_host_blocks_if_ready(request_id)

    def _plan_spill(
        self,
        request_id: str,
        page_idx: int,
        *,
        after_forward: bool,
    ) -> bool:
        assert self.host_manager is not None
        state = self._get_request_state(request_id)
        host_blocks = self.host_manager.req_to_blocks.get(request_id)
        if host_blocks is None or page_idx >= len(host_blocks):
            return False
        host_block = host_blocks[page_idx]
        if host_block.is_null:
            return False

        blocks: list[KVCacheBlock] = []
        for manager in self.resident_managers:
            block = manager.get_resident_page(request_id, page_idx)
            if block is None:
                return False
            blocks.append(block)

        self.host_manager.block_pool.touch([host_block])
        for manager, block in zip(self.resident_managers, blocks):
            manager.block_pool.touch([block])
        spill_id = self.next_spill_id
        self.next_spill_id += 1
        plan = SparseKVPageTransfer(
            transfer_id=spill_id,
            destination_block_id=host_block.block_id,
            source_block_ids=tuple(block.block_id for block in blocks),
            after_forward=after_forward,
        )
        self.pending_spills[spill_id] = _PendingSpill(
            transfer_id=spill_id,
            page=(request_id, page_idx),
            request_state=state,
            host_block=host_block,
            resident_blocks=tuple(blocks),
        )
        state.pending_pages[page_idx] = spill_id
        self.spills_to_send.append(plan)
        return True

    # ------------------------------------------------------------------
    # Worker-facing views
    # ------------------------------------------------------------------

    def build_row_mirrors(
        self,
        requests: Iterable[tuple[str, int, int]],
    ) -> tuple[SparseKVRowMirror, ...]:
        """Map scheduled rows to scheduler-owned resident and host blocks."""
        if not self.resident_managers or self.host_manager is None:
            return ()
        block_size = self.resident_managers[0].block_size
        mirrors: list[SparseKVRowMirror] = []
        for request_id, num_computed_tokens, num_scheduled_tokens in requests:
            host_blocks = self.host_manager.req_to_blocks.get(request_id)
            resident_blocks = [
                manager.req_to_blocks.get(request_id)
                for manager in self.resident_managers
            ]
            if host_blocks is None or any(blocks is None for blocks in resident_blocks):
                continue
            token_position = num_computed_tokens
            end_position = token_position + num_scheduled_tokens
            while token_position < end_position:
                page_idx, row_offset = divmod(token_position, block_size)
                num_rows = min(block_size - row_offset, end_position - token_position)
                # A page without a GPU copy has no mirror source, and a page
                # without a host block has no destination. Skip just that page:
                # its neighbours in the window are still mirrorable, and a
                # window only ever moves forward, so dropping them here would
                # leave their host rows permanently stale.
                source_starts = []
                for blocks in resident_blocks:
                    assert blocks is not None
                    if page_idx >= len(blocks) or blocks[page_idx].is_null:
                        break
                    source_starts.append(
                        blocks[page_idx].block_id * block_size + row_offset
                    )
                if len(source_starts) != len(resident_blocks):
                    token_position += num_rows
                    continue
                host_block_idx = page_idx
                if host_block_idx >= len(host_blocks):
                    token_position += num_rows
                    continue
                host_block = host_blocks[host_block_idx]
                if host_block.is_null:
                    token_position += num_rows
                    continue
                mirrors.append(
                    SparseKVRowMirror(
                        source_starts=tuple(source_starts),
                        destination_start=(
                            host_block.block_id * block_size + row_offset
                        ),
                        num_rows=num_rows,
                    )
                )
                token_position += num_rows
        return tuple(mirrors)

    def all_context_pages_resident(
        self,
        requests: Iterable[tuple[str, int, int]],
    ) -> bool:
        """Return whether every scheduled request can read only resident KV."""
        if not self.resident_managers:
            return False
        block_size = self.resident_managers[0].block_size
        for request_id, num_computed_tokens, num_scheduled_tokens in requests:
            num_pages = cdiv(num_computed_tokens + num_scheduled_tokens, block_size)
            for page_idx in range(num_pages):
                if any(
                    manager.get_resident_page(request_id, page_idx) is None
                    for manager in self.resident_managers
                ):
                    return False
        return True

    def take_block_table_updates(self) -> dict[str, tuple[list[int], ...]]:
        updates = {
            request_id: tuple(
                [block.block_id for block in manager.req_to_blocks.get(request_id, [])]
                for manager in self.managers
            )
            for request_id in self.block_table_updates
        }
        self.block_table_updates.clear()
        return updates

    def build_offload_command(self) -> SparseKVOffloadCommand | None:
        """Package residency decisions for the worker connector."""
        if not self.resident_managers:
            return None
        command = SparseKVOffloadCommand(page_transfers=self.spills_to_send)
        self.spills_to_send = []
        return command

    def has_pending_reclamation(self) -> bool:
        """Whether an in-flight spill will free GPU blocks on completion."""
        return any(
            pending.page[0] in self.request_states
            and self._can_read_from_host(pending.page[0])
            for pending in self.pending_spills.values()
        )

    def has_pending_work(self) -> bool:
        return bool(self.spills_to_send or self.pending_spills or self._retained_copies)

    def update_spills(
        self,
        enqueued_counts: Mapping[int, int],
        completed_counts: Mapping[int, int],
    ) -> None:
        for spill_id, count in enqueued_counts.items():
            pending = self.pending_spills.get(spill_id)
            if pending is not None and pending.expected_worker_completions == 0:
                pending.expected_worker_completions = count
        for spill_id, count in completed_counts.items():
            pending = self.pending_spills.get(spill_id)
            if pending is not None:
                pending.worker_completions += count

        self._apply_enqueued_spills()
        self._complete_host_writes()

    def _apply_enqueued_spills(self) -> None:
        """Drop the DMA pins once the worker has enqueued a spill."""
        for pending in self.pending_spills.values():
            if (
                pending.enqueue_applied
                or not pending.expected_worker_completions
                or pending.worker_completions < pending.expected_worker_completions
            ):
                continue
            for manager, block in zip(self.resident_managers, pending.resident_blocks):
                manager.block_pool.free_blocks([block])
            pending.enqueue_applied = True

    def _complete_host_writes(self) -> None:
        completed = [
            pending
            for pending in self.pending_spills.values()
            if pending.enqueue_applied
            and pending.expected_worker_completions
            and pending.worker_completions >= pending.expected_worker_completions
        ]
        completed_request_ids: set[str] = set()
        for pending in completed:
            self.pending_spills.pop(pending.transfer_id, None)
            request_id, page_idx = pending.page
            pending.request_state.pending_pages.pop(page_idx, None)
            if self.request_states.get(request_id) is pending.request_state:
                pending.request_state.valid_pages.add(page_idx)
                self._page_became_clean(request_id, pending.request_state, page_idx)
                completed_request_ids.add(request_id)
            assert self.host_manager is not None
            self.host_manager.block_pool.free_blocks([pending.host_block])
        for request_id in completed_request_ids:
            state = self.request_states[request_id]
            while state.ready_prefix_pages in state.valid_pages:
                state.ready_prefix_pages += 1
            self._publish_host_blocks_if_ready(request_id)

    def free(self, request_id: str) -> None:
        """Detach the request; its clean pages stay readable copies in the pool."""
        self._pending_imports.pop(request_id, None)
        state = self.request_states.pop(request_id, None)
        if state is None:
            return
        for manager in self.resident_managers:
            blocks = manager.req_to_blocks.get(request_id)
            if not blocks:
                continue
            for page_idx, block in enumerate(blocks):
                if block.is_null:
                    continue
                if page_idx in state.unpinned_pages:
                    owners = self._owners.get(block.block_id)
                    if owners is not None:
                        owners.discard((request_id, page_idx))
                elif (
                    page_idx in state.valid_pages
                    and page_idx not in state.pending_pages
                ):
                    manager.block_pool.unpin_blocks([block], self._on_block_reused)
                else:
                    continue
                blocks[page_idx] = manager._null_block


def get_hisparse_coordinator(
    kv_cache_manager: "KVCacheManager",
) -> HiSparseCoordinator:
    """Return the coordinator shared by a KV cache manager's HiSparse groups.

    Built on first call and memoised on the HiSparse managers it binds itself
    to. Must run before the first request is admitted.
    """
    managers = tuple(kv_cache_manager.coordinator.single_type_managers)
    for manager in managers:
        coordinator = getattr(manager, "coordinator", None)
        if coordinator is not None:
            assert isinstance(coordinator, HiSparseCoordinator)
            return coordinator
    coordinator = HiSparseCoordinator(
        kv_cache_manager.kv_cache_config, managers, kv_cache_manager.max_model_len
    )
    if coordinator.host_manager is None:
        raise ValueError("No HiSparse cache group is configured.")
    return coordinator
