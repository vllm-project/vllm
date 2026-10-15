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
    HiSparseDeviceManager,
    HiSparseHostManager,
    HiSparseHotManager,
    SingleTypeKVCacheManager,
)
from vllm.v1.hisparse.types import (
    ACTIVE_TAIL_PAGES,
    HiSparsePageTransfer,
    HiSparseResidencyUpdate,
    HiSparseRowMirror,
    HiSparseTransferCommand,
)
from vllm.v1.kv_cache_interface import (
    HiSparseDeviceSpec,
    KVCacheConfig,
)
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheManager


@dataclass
class _PendingPublication:
    request: Request
    num_computed_tokens: int
    num_pages: int
    retention_interval: int | None
    replay_boundaries: Sequence[int]
    leased_host_blocks: list[KVCacheBlock] | None = None
    num_cached_blocks: int = 0


@dataclass
class _HiSparseRequestState:
    """Residency bookkeeping for one request.

    A device page moves through three states: not yet durable (only the GPU
    copy exists), durable (``durable_pages``: its host copy is complete) and
    released (``released_pages``: durable, and its allocation reference was
    released so the pool may evict it). ``pinned_durable_pages`` tracks durable pages
    still holding their reference, which only happens before the request can
    read from host.
    """

    durable_pages: set[int] = field(default_factory=set)
    num_durable_prefix_pages: int = 0
    in_flight_transfers: dict[int, int] = field(default_factory=dict)
    pending_publication: _PendingPublication | None = None
    gpu_copies_recorded_up_to: int = 0
    pinned_durable_pages: set[int] = field(default_factory=set)
    released_pages: set[int] = field(default_factory=set)
    # Prefix pages whose GPU copies are revived after the admitting allocation.
    pages_to_revive: int = 0
    # The worker holds this request's residency for every page below
    # synced_pages, except the stale pages changed since it was sent.
    synced_pages: int = 0
    stale_pages: set[int] = field(default_factory=set)


@dataclass
class _PendingTransfer:
    transfer_id: int
    page: tuple[str, int]
    request_state: _HiSparseRequestState
    host_block: KVCacheBlock
    device_blocks: tuple[KVCacheBlock, ...]
    is_restore: bool = False
    expected_worker_completions: int = 0
    worker_completions: int = 0
    enqueue_applied: bool = False


class HiSparseCoordinator:
    """Own HiSparse host allocation, publication, page transfers, and GPU residency.

    device pages are a write-back cache of the host tier. Once a page's
    host copy is durable and its request can read from host (it has, or has
    asked for, a hot buffer), the page's allocation reference is released with
    ``BlockPool.unpin_blocks``: the block keeps being read, but the pool counts
    it as free and may hand it out, at which point the request stops reading it.
    Residency changes reach the worker through ``take_residency_updates``, not
    the block table, which stays append-only.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        managers: tuple[SingleTypeKVCacheManager, ...],
        max_model_len: int,
        *,
        num_reprefillable_tokens: int,
    ) -> None:
        self.managers = managers
        self.max_model_len = max_model_len
        self.num_reprefillable_tokens = num_reprefillable_tokens
        groups = kv_cache_config.kv_cache_groups

        device_managers: list[HiSparseDeviceManager] = []
        hot_managers: list[HiSparseHotManager] = []
        device_group_ids: list[int] = []
        for group_id, (group, manager) in enumerate(zip(groups, managers)):
            if isinstance(group.kv_cache_spec, HiSparseDeviceSpec):
                if not isinstance(manager, HiSparseDeviceManager):
                    raise TypeError(
                        "HiSparse device specs require device cache managers."
                    )
                device_managers.append(manager)
                device_group_ids.append(group_id)
                assert not group.host_resident
            if isinstance(manager, HiSparseHotManager):
                hot_managers.append(manager)
                assert not group.host_resident
            if isinstance(manager, (HiSparseHotManager, HiSparseDeviceManager)):
                manager.coordinator = self
        self.device_managers = tuple(device_managers)
        self.hot_managers = tuple(hot_managers)
        self.host_manager: HiSparseHostManager | None = None
        self.host_group_id: int | None = None
        self.max_transfers_per_step = 0
        for group_id, manager in enumerate(managers):
            if isinstance(manager, HiSparseHostManager):
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
        self.read_from_host_watermark = 0
        if self.device_managers:
            assert self.host_manager is not None
            assert self.host_group_id is not None
            if self.host_group_id > min(device_group_ids):
                raise ValueError("HiSparse host group must precede device groups.")
            device_block_sizes = {
                manager.block_size for manager in self.device_managers
            }
            if len(device_block_sizes) != 1:
                raise ValueError(
                    "HiSparse device cache groups must use one block size."
                )
            device_block_size = device_block_sizes.pop()
            if self.host_manager.block_size != device_block_size:
                raise ValueError("HiSparse host and device block sizes must match.")
            self.max_transfers_per_step = max_model_len // device_block_size
            self.gpu_pool = self.device_managers[0].block_pool
            # Requests start reading from host once the shared pool runs this
            # low, so admissions find evictable pages rather than pinned ones.
            hot_cost = sum(manager.blocks_per_request for manager in hot_managers)
            self.read_from_host_watermark = max(
                hot_cost, kv_cache_config.num_blocks // 10
            )

        # Host copies handed to the worker, by destination block id, until it
        # reports having run them.
        self._retained_cow_copies: dict[int, tuple[KVCacheBlock, KVCacheBlock]] = {}
        # request -> prefix length an external load is still filling in.
        self._pending_import_tokens: dict[str, int] = {}
        # Requests whose device pages were all non-null when last scanned.
        # Pages only become null at admission or import (before the first
        # scan) and when the pool evicts a released page (_unmap_evicted_page).
        self._on_device_requests: set[str] = set()
        self.transfers_to_send: list[HiSparsePageTransfer] = []
        self.pending_transfers: dict[int, _PendingTransfer] = {}
        self.request_states: dict[str, _HiSparseRequestState] = {}
        self.next_transfer_id = 0
        # Published host block hash -> GPU copies of that page, readable until
        # the pool evicts them. Lets a host prefix hit come back on device.
        self.gpu_copies: dict[BlockHashWithGroupId, tuple[KVCacheBlock, ...]] = {}
        self._gpu_copy_by_block: dict[int, BlockHashWithGroupId] = {}
        # Released GPU block id -> (request, page) pairs still reading it.
        self._gpu_copy_readers: dict[int, set[tuple[str, int]]] = {}

    def _get_request_state(self, request_id: str) -> _HiSparseRequestState:
        state = self.request_states.get(request_id)
        if state is None:
            state = _HiSparseRequestState()
            self.request_states[request_id] = state
        return state

    def record_host_prefix_hit(self, request_id: str, num_host_pages: int) -> None:
        """Account for a prefix hit on host pages the request now references."""
        self._on_device_requests.discard(request_id)
        if not self.device_managers or self.host_manager is None:
            return
        host_blocks = self.host_manager.req_to_blocks.get(request_id, ())
        num_host_pages = min(num_host_pages, len(host_blocks))
        if num_host_pages == 0:
            return
        state = self._get_request_state(request_id)
        state.durable_pages.update(range(num_host_pages))
        state.num_durable_prefix_pages = max(
            num_host_pages, state.num_durable_prefix_pages
        )
        # Reviving pins copies out of the free pool, so it must wait until the
        # allocation admission counted on has run (see update_residency).
        state.pages_to_revive = max(state.pages_to_revive, num_host_pages)
        state.gpu_copies_recorded_up_to = max(
            state.gpu_copies_recorded_up_to, num_host_pages
        )

    def take_host_cow_copies(self) -> tuple[KVCacheBlockCopy, ...]:
        """Drain host copy-on-write work, retaining both endpoints.

        The worker runs the copies in the step that carries them, so the
        endpoints are freed when that step's output comes back.
        """
        if self.host_manager is None:
            return ()
        pairs = self.host_manager.take_host_cow_copies()
        for source_block, cow_block in pairs:
            self._retained_cow_copies[cow_block.block_id] = (source_block, cow_block)
        return tuple(
            KVCacheBlockCopy(
                src_block_id=source_block.block_id,
                dst_block_id=cow_block.block_id,
            )
            for source_block, cow_block in pairs
        )

    def free_completed_host_cow_copies(self, dst_block_ids: Iterable[int]) -> None:
        """Free the endpoints of host copies the worker reports having run."""
        blocks: list[KVCacheBlock] = []
        for dst_block_id in dst_block_ids:
            pair = self._retained_cow_copies.pop(dst_block_id, None)
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

    def _revive_gpu_copies(
        self,
        request_id: str,
        state: _HiSparseRequestState,
        host_blocks: Sequence[KVCacheBlock],
    ) -> None:
        """Point null prefix pages at readable GPU copies of their contents.

        Runs after the hit length is fully reconciled, so copy hits and misses
        can be scattered per page without affecting the prefix hit.
        """
        if not self.gpu_copies:
            return
        for page_idx, host_block in enumerate(host_blocks):
            if host_block.is_null or host_block.block_hash is None:
                continue
            blocks = self.gpu_copies.get(host_block.block_hash)
            if blocks is None:
                continue
            for manager, block in zip(self.device_managers, blocks):
                if manager.revive_device_page(request_id, page_idx, block):
                    manager.block_pool.touch([block])
                    state.pinned_durable_pages.add(page_idx)
                    if page_idx < state.synced_pages:
                        state.stale_pages.add(page_idx)

    def _record_gpu_copies(self, request_id: str, num_computed_tokens: int) -> None:
        """Index the GPU copies of just-published host pages for later hits."""
        if not self.device_managers:
            return
        assert self.host_manager is not None
        host_blocks = self.host_manager.req_to_blocks.get(request_id)
        if not host_blocks:
            return
        state = self._get_request_state(request_id)
        num_host_pages = min(
            num_computed_tokens // self.host_manager.block_size, len(host_blocks)
        )
        for page_idx in range(state.gpu_copies_recorded_up_to, num_host_pages):
            if page_idx in state.in_flight_transfers:
                continue
            self._record_gpu_copy(request_id, page_idx, host_blocks[page_idx])
        state.gpu_copies_recorded_up_to = max(
            state.gpu_copies_recorded_up_to, num_host_pages
        )

    def _record_gpu_copy(
        self, request_id: str, page_idx: int, host_block: KVCacheBlock
    ) -> None:
        if host_block.is_null or host_block.block_hash is None:
            return
        blocks: list[KVCacheBlock] = []
        for manager in self.device_managers:
            block = manager.get_device_page(request_id, page_idx)
            if block is None:
                return
            blocks.append(block)
        self._drop_gpu_copy(host_block.block_hash)
        self.gpu_copies[host_block.block_hash] = tuple(blocks)
        for block in blocks:
            self._gpu_copy_by_block[block.block_id] = host_block.block_hash

    def _drop_gpu_copy(self, host_hash: BlockHashWithGroupId) -> None:
        blocks = self.gpu_copies.pop(host_hash, None)
        if blocks is None:
            return
        for block in blocks:
            self._gpu_copy_by_block.pop(block.block_id, None)

    # ------------------------------------------------------------------
    # Residency: pinning, releasing and evicting pages
    # ------------------------------------------------------------------

    def _can_read_from_host(self, request_id: str) -> bool:
        """Whether device pages may be dropped under the request."""
        return bool(self.hot_managers) and all(
            manager.has_hot_buffer(request_id) for manager in self.hot_managers
        )

    def _fills_admission_window(self, request_id: str) -> bool:
        """Whether the request holds the device pages admission reserved.

        Admission caps a request's device pages at the in-flight window,
        assuming older pages move to host; keeping them pinned past it lets a
        chunked prefill outgrow the pool it was admitted into.
        """
        return any(
            len(manager.req_to_blocks.get(request_id, ()))
            >= manager.max_admission_blocks_per_request
            for manager in self.device_managers
        )

    def _device_page_blocks(
        self, request_id: str, page_idx: int
    ) -> list[KVCacheBlock] | None:
        blocks: list[KVCacheBlock] = []
        for manager in self.device_managers:
            num_pages = len(manager.req_to_blocks.get(request_id, ()))
            block = manager.get_device_page(request_id, page_idx)
            if page_idx >= num_pages - ACTIVE_TAIL_PAGES or block is None:
                return None
            blocks.append(block)
        return blocks

    def _release_page(
        self, request_id: str, state: _HiSparseRequestState, page_idx: int
    ) -> bool:
        if page_idx in state.in_flight_transfers or page_idx not in state.durable_pages:
            return False
        blocks = self._device_page_blocks(request_id, page_idx)
        if blocks is None:
            return False
        for manager, block in zip(self.device_managers, blocks):
            self._gpu_copy_readers.setdefault(block.block_id, set()).add(
                (request_id, page_idx)
            )
            manager.block_pool.unpin_blocks([block], self._on_block_evicted)
        state.pinned_durable_pages.discard(page_idx)
        state.released_pages.add(page_idx)
        return True

    def _release_durable_pages(
        self, request_id: str, state: _HiSparseRequestState
    ) -> None:
        # Tail first, so a prefix loses its tail pages before its head.
        for page_idx in sorted(state.pinned_durable_pages, reverse=True):
            self._release_page(request_id, state, page_idx)

    def _page_became_durable(
        self, request_id: str, state: _HiSparseRequestState, page_idx: int
    ) -> None:
        state.pinned_durable_pages.add(page_idx)
        if self._can_read_from_host(request_id):
            self._release_page(request_id, state, page_idx)

    def _on_block_evicted(self, block: KVCacheBlock) -> None:
        host_hash = self._gpu_copy_by_block.get(block.block_id)
        if host_hash is not None:
            self._drop_gpu_copy(host_hash)
        for request_id, page_idx in self._gpu_copy_readers.pop(block.block_id, ()):
            self._unmap_evicted_page(request_id, page_idx)

    def _unmap_evicted_page(self, request_id: str, page_idx: int) -> None:
        self._on_device_requests.discard(request_id)
        state = self.request_states.get(request_id)
        if state is None:
            return
        for manager in self.device_managers:
            block = manager.drop_device_page(request_id, page_idx)
            if block is None:
                continue
            readers = self._gpu_copy_readers.get(block.block_id)
            if readers is not None:
                readers.discard((request_id, page_idx))
        state.released_pages.discard(page_idx)
        if page_idx < state.synced_pages:
            state.stale_pages.add(page_idx)
        for hot_manager in self.hot_managers:
            hot_manager.request_hot_buffer(request_id)

    def update_residency(self, request_id: str) -> None:
        """Per-step residency policy for a scheduled request.

        A request that can read from host releases every durable sealed page to
        the pool. One that cannot keeps its pages pinned until the shared pool
        runs low, or until it holds the device pages admission reserved for
        it, then asks for a hot buffer. Pages remain pinned until that buffer is
        allocated on a subsequent scheduling pass.
        """
        if not self.device_managers:
            return
        state = self._get_request_state(request_id)
        if state.pages_to_revive:
            assert self.host_manager is not None
            host_blocks = self.host_manager.req_to_blocks.get(request_id, ())
            self._revive_gpu_copies(
                request_id, state, host_blocks[: state.pages_to_revive]
            )
            state.pages_to_revive = 0
        if not self._can_read_from_host(request_id):
            assert self.gpu_pool is not None
            if (
                self.gpu_pool.get_num_free_blocks() >= self.read_from_host_watermark
                and not self._fills_admission_window(request_id)
            ):
                return
            for manager in self.hot_managers:
                manager.request_hot_buffer(request_id)
            return
        self._release_durable_pages(request_id, state)

    # ------------------------------------------------------------------
    # Host publication and page transfers
    # ------------------------------------------------------------------

    def advance_scheduled(self, requests: Iterable[tuple[str, int]]) -> None:
        """Run the per-step residency work for each scheduled request.

        ``requests`` pairs a request id with the number of tokens computed
        once the step completes. As in ``KVCacheCoordinator.cache_blocks``, the
        last ``num_reprefillable_tokens`` are not final yet and are not written
        back.
        """
        if not self.device_managers:
            return
        for request_id, num_tokens in requests:
            self.plan_write_backs(
                request_id, max(0, num_tokens - self.num_reprefillable_tokens)
            )
            self.update_residency(request_id)

    def plan_write_backs(self, request_id: str, num_computed_tokens: int) -> None:
        """Eagerly write back every sealed page that has no host copy yet."""
        if not self.device_managers:
            return
        assert self.host_manager is not None
        host_block_size = self.host_manager.block_size
        num_pages = num_computed_tokens // host_block_size
        importing_pages = cdiv(
            self._pending_import_tokens.get(request_id, 0), host_block_size
        )
        state = self._get_request_state(request_id)
        budget = max(self.max_transfers_per_step - len(self.transfers_to_send), 0)
        first_page = max(state.num_durable_prefix_pages, importing_pages)
        for page_idx in range(first_page, num_pages):
            if budget == 0:
                break
            if page_idx in state.in_flight_transfers:
                continue
            if page_idx in state.durable_pages:
                continue
            if not all(
                manager.get_device_page(request_id, page_idx) is not None
                for manager in self.device_managers
            ):
                continue
            if self._plan_page_transfer(request_id, page_idx, runs_after_forward=True):
                budget -= 1

    def publish_when_durable(
        self,
        request: Request,
        num_computed_tokens: int,
        retention_interval: int | None,
        *,
        replay_boundaries: Sequence[int],
    ) -> None:
        """Publish host hashes of the pages that are already durable."""
        manager = self.host_manager
        if manager is None:
            return
        request_id = request.request_id
        self._get_request_state(request_id).pending_publication = _PendingPublication(
            request=request,
            num_computed_tokens=num_computed_tokens,
            num_pages=num_computed_tokens // manager.block_size,
            retention_interval=retention_interval,
            replay_boundaries=replay_boundaries,
        )
        self._publish_if_durable(request_id)

    def _publish_if_durable(self, request_id: str) -> None:
        """Publish the durable prefix of the pending publication.

        Host writes trail a long prefill by about a chunk, so waiting for the
        whole computed prefix would publish nothing until the prefill ends, and
        a prefill preempted before then would recompute from the start.
        """
        state = self.request_states.get(request_id)
        if state is None:
            return
        publication = state.pending_publication
        if publication is None:
            return
        assert self.host_manager is not None
        num_tokens = min(
            publication.num_computed_tokens,
            state.num_durable_prefix_pages * self.host_manager.block_size,
        )
        self.host_manager.publish_blocks(
            publication.request,
            num_tokens,
            retention_interval=publication.retention_interval,
            replay_boundaries=publication.replay_boundaries,
        )
        self._record_gpu_copies(request_id, num_tokens)
        if state.num_durable_prefix_pages >= publication.num_pages:
            state.pending_publication = None

    def record_pending_host_import(self, request_id: str, num_tokens: int) -> None:
        """Note a prefix an external load is populating in host pages."""
        self._on_device_requests.discard(request_id)
        self._pending_import_tokens[request_id] = num_tokens

    def finish_host_import(self, request_id: str, *, failed: bool) -> None:
        """Publish externally populated host pages after connector completion."""
        num_computed_tokens = self._pending_import_tokens.pop(request_id, None)
        if num_computed_tokens is None or failed or not self.device_managers:
            return
        block_size = self.device_managers[0].block_size
        num_pages = num_computed_tokens // block_size
        state = self._get_request_state(request_id)
        state.durable_pages = set(range(num_pages))
        state.num_durable_prefix_pages = num_pages
        if num_computed_tokens:
            self._plan_page_transfer(
                request_id,
                (num_computed_tokens - 1) // block_size,
                runs_after_forward=False,
                is_restore=True,
            )
        self._publish_if_durable(request_id)

    def _plan_page_transfer(
        self,
        request_id: str,
        page_idx: int,
        *,
        runs_after_forward: bool,
        is_restore: bool = False,
    ) -> bool:
        assert self.host_manager is not None
        state = self._get_request_state(request_id)
        if page_idx in state.in_flight_transfers:
            return False
        host_blocks = self.host_manager.req_to_blocks.get(request_id)
        if host_blocks is None or page_idx >= len(host_blocks):
            return False
        host_block = host_blocks[page_idx]
        if host_block.is_null:
            return False

        blocks: list[KVCacheBlock] = []
        for manager in self.device_managers:
            block = manager.get_device_page(request_id, page_idx)
            if block is None:
                return False
            blocks.append(block)

        self.host_manager.block_pool.touch([host_block])
        for manager, block in zip(self.device_managers, blocks):
            manager.block_pool.touch([block])
        transfer_id = self.next_transfer_id
        self.next_transfer_id += 1
        plan = HiSparsePageTransfer(
            transfer_id=transfer_id,
            host_block_id=host_block.block_id,
            device_block_ids=tuple(block.block_id for block in blocks),
            runs_after_forward=runs_after_forward,
            is_restore=is_restore,
        )
        self.pending_transfers[transfer_id] = _PendingTransfer(
            transfer_id=transfer_id,
            page=(request_id, page_idx),
            request_state=state,
            host_block=host_block,
            device_blocks=tuple(blocks),
            is_restore=is_restore,
        )
        state.in_flight_transfers[page_idx] = transfer_id
        self.transfers_to_send.append(plan)
        return True

    # ------------------------------------------------------------------
    # Worker-facing views
    # ------------------------------------------------------------------

    def build_row_mirrors(
        self,
        requests: Iterable[tuple[str, int, int]],
    ) -> tuple[HiSparseRowMirror, ...]:
        """Map scheduled rows to scheduler-owned device and host blocks."""
        if not self.device_managers or self.host_manager is None:
            return ()
        block_size = self.device_managers[0].block_size
        mirrors: list[HiSparseRowMirror] = []
        for request_id, num_computed_tokens, num_scheduled_tokens in requests:
            host_blocks = self.host_manager.req_to_blocks.get(request_id)
            if host_blocks is None or any(
                request_id not in manager.req_to_blocks
                for manager in self.device_managers
            ):
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
                for manager in self.device_managers:
                    block = manager.get_device_page(request_id, page_idx)
                    if block is None:
                        break
                    source_starts.append(block.block_id * block_size + row_offset)
                if len(source_starts) != len(self.device_managers):
                    token_position += num_rows
                    continue
                if page_idx >= len(host_blocks):
                    token_position += num_rows
                    continue
                host_block = host_blocks[page_idx]
                if host_block.is_null:
                    token_position += num_rows
                    continue
                mirrors.append(
                    HiSparseRowMirror(
                        source_starts=tuple(source_starts),
                        destination_start=(
                            host_block.block_id * block_size + row_offset
                        ),
                        num_rows=num_rows,
                    )
                )
                token_position += num_rows
        return tuple(mirrors)

    def all_context_pages_on_device(
        self,
        requests: Iterable[tuple[str, int, int]],
    ) -> bool:
        """Return whether every scheduled request can read only device KV."""
        if not self.device_managers:
            return False
        first = self.device_managers[0]
        for request_id, num_computed_tokens, num_scheduled_tokens in requests:
            num_pages = cdiv(
                num_computed_tokens + num_scheduled_tokens, first.block_size
            )
            if len(first.req_to_blocks.get(request_id, ())) < num_pages:
                return False
            if request_id in self._on_device_requests:
                continue
            if any(
                block.is_null
                for manager in self.device_managers
                for block in manager.get_residency_row(request_id)
            ):
                return False
            self._on_device_requests.add(request_id)
        return True

    def take_residency_updates(
        self, request_ids: Iterable[str]
    ) -> dict[str, HiSparseResidencyUpdate]:
        """Device block ids the worker lacks for the scheduled requests:
        stale pages plus pages appended since the last update.

        Requests that are not scheduled keep their changes until they are:
        only a scheduled request's residency is read.
        """
        updates: dict[str, HiSparseResidencyUpdate] = {}
        if not self.device_managers:
            return updates
        first = self.device_managers[0]
        for request_id in request_ids:
            state = self._get_request_state(request_id)
            num_pages = len(first.req_to_blocks.get(request_id, ()))
            start = state.synced_pages
            if not state.stale_pages and start >= num_pages:
                continue
            stale_pages = sorted(state.stale_pages)
            block_ids: list[list[int]] = []
            for manager in self.device_managers:
                null_block_id = manager.block_pool.null_block.block_id
                group_block_ids = [
                    null_block_id if block is None else block.block_id
                    for block in (
                        manager.get_device_page(request_id, page_idx)
                        for page_idx in stale_pages
                    )
                ]
                group_block_ids.extend(
                    block.block_id
                    for block in manager.get_residency_row(request_id, start)
                )
                block_ids.append(group_block_ids)
            updates[request_id] = HiSparseResidencyUpdate(
                pages=[*stale_pages, *range(start, num_pages)],
                block_ids=tuple(block_ids),
            )
            state.stale_pages.clear()
            state.synced_pages = num_pages
        return updates

    def build_transfer_command(self) -> HiSparseTransferCommand | None:
        """Package residency decisions for the worker connector."""
        if not self.device_managers:
            return None
        command = HiSparseTransferCommand(page_transfers=self.transfers_to_send)
        self.transfers_to_send = []
        return command

    def has_pending_block_frees(self) -> bool:
        """Whether an in-flight transfer will free GPU or host blocks on completion."""
        return any(
            pending.host_block.ref_cnt == 1
            or (
                pending.page[0] in self.request_states
                and self._can_read_from_host(pending.page[0])
            )
            for pending in self.pending_transfers.values()
        )

    def has_pending_work(self) -> bool:
        return bool(
            self.transfers_to_send
            or self.pending_transfers
            or self._retained_cow_copies
        )

    def update_transfers(
        self,
        enqueued_counts: Mapping[int, int],
        completed_counts: Mapping[int, int],
    ) -> None:
        for transfer_id, count in enqueued_counts.items():
            pending = self.pending_transfers.get(transfer_id)
            if pending is not None and pending.expected_worker_completions == 0:
                pending.expected_worker_completions = count
        for transfer_id, count in completed_counts.items():
            pending = self.pending_transfers.get(transfer_id)
            if pending is not None:
                pending.worker_completions += count

        self._apply_enqueued_transfers()
        self._complete_transfers()

    def _apply_enqueued_transfers(self) -> None:
        """Free the GPU leases once every worker has completed the transfer."""
        for pending in self.pending_transfers.values():
            if (
                pending.enqueue_applied
                or not pending.expected_worker_completions
                or pending.worker_completions < pending.expected_worker_completions
            ):
                continue
            for manager, block in zip(self.device_managers, pending.device_blocks):
                manager.block_pool.free_blocks([block])
            pending.enqueue_applied = True

    def _complete_transfers(self) -> None:
        completed = [
            pending
            for pending in self.pending_transfers.values()
            if pending.enqueue_applied
            and pending.expected_worker_completions
            and pending.worker_completions >= pending.expected_worker_completions
        ]
        completed_request_ids: set[str] = set()
        for pending in completed:
            self.pending_transfers.pop(pending.transfer_id, None)
            request_id, page_idx = pending.page
            pending.request_state.in_flight_transfers.pop(page_idx, None)
            if self.request_states.get(request_id) is pending.request_state:
                if not pending.is_restore:
                    pending.request_state.durable_pages.add(page_idx)
                if page_idx in pending.request_state.durable_pages:
                    self._page_became_durable(
                        request_id, pending.request_state, page_idx
                    )
                    if pending.is_restore:
                        self._record_gpu_copy(request_id, page_idx, pending.host_block)
                completed_request_ids.add(request_id)
            elif pending.request_state.pending_publication is not None:
                if not pending.is_restore:
                    pending.request_state.durable_pages.add(page_idx)
                self._publish_finished_prefix(pending.request_state)
            assert self.host_manager is not None
            self.host_manager.block_pool.free_blocks([pending.host_block])
        for request_id in completed_request_ids:
            state = self.request_states[request_id]
            while state.num_durable_prefix_pages in state.durable_pages:
                state.num_durable_prefix_pages += 1
            self._publish_if_durable(request_id)

    def _publish_finished_prefix(self, state: _HiSparseRequestState) -> None:
        publication = state.pending_publication
        if publication is None or state.in_flight_transfers:
            return
        blocks = publication.leased_host_blocks
        if blocks is None:
            return
        assert self.host_manager is not None
        while state.num_durable_prefix_pages in state.durable_pages:
            state.num_durable_prefix_pages += 1
        # The host group uses full-attention retention. Only sealed pages with
        # completed copies can outlive the request, including on cancellation.
        self.host_manager.block_pool.cache_full_blocks(
            request=publication.request,
            blocks=blocks,
            num_cached_blocks=publication.num_cached_blocks,
            num_full_blocks=min(state.num_durable_prefix_pages, publication.num_pages),
            block_size=self.host_manager.block_size,
            kv_cache_group_id=self.host_manager.kv_cache_group_id,
        )
        self.host_manager.block_pool.free_blocks(reversed(blocks))
        state.pending_publication = None

    def free(self, request_id: str) -> None:
        """Detach the request; its durable pages stay readable GPU copies."""
        self._on_device_requests.discard(request_id)
        self._pending_import_tokens.pop(request_id, None)
        state = self.request_states.pop(request_id, None)
        if state is None:
            return
        publication = state.pending_publication
        if (
            publication is not None
            and publication.request.is_finished()
            and state.in_flight_transfers
        ):
            assert self.host_manager is not None
            publication.leased_host_blocks = self.host_manager.req_to_blocks[
                request_id
            ][: publication.num_pages]
            publication.num_cached_blocks = self.host_manager.num_cached_block.get(
                request_id, 0
            )
            # Completed pages need leases too: another allocation may evict
            # them before the remaining copies finish and the prefix publishes.
            self.host_manager.block_pool.touch(publication.leased_host_blocks)
        else:
            # Preemption can cancel a scheduled forward after its write-backs were
            # planned, so its copies do not prove that the KV was computed.
            state.pending_publication = None
        for manager in self.device_managers:
            blocks = manager.get_residency_row(request_id)
            # Tail first, so a prefix loses its tail pages before its head.
            for page_idx in reversed(range(len(blocks))):
                block = blocks[page_idx]
                if block.is_null:
                    continue
                if page_idx in state.released_pages:
                    readers = self._gpu_copy_readers.get(block.block_id)
                    if readers is not None:
                        readers.discard((request_id, page_idx))
                elif (
                    page_idx in state.durable_pages
                    and page_idx not in state.in_flight_transfers
                ):
                    manager.block_pool.unpin_blocks([block], self._on_block_evicted)
                else:
                    continue
                manager.drop_device_page(request_id, page_idx)


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
        kv_cache_manager.kv_cache_config,
        managers,
        kv_cache_manager.max_model_len,
        num_reprefillable_tokens=kv_cache_manager.coordinator.num_reprefillable_tokens,
    )
    if coordinator.host_manager is None:
        raise ValueError("No HiSparse cache group is configured.")
    return coordinator
