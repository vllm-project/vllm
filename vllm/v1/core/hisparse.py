# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from dataclasses import dataclass

from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import (
    HiSparseHotManager,
    HiSparseResidentManager,
    SingleTypeKVCacheManager,
)
from vllm.v1.kv_cache_interface import (
    HiSparseResidentSpec,
    HiSparseSpill,
    KVCacheConfig,
)


@dataclass
class _PendingSpill:
    plan: HiSparseSpill
    host_manager: SingleTypeKVCacheManager
    host_block: KVCacheBlock
    resident_entries: tuple[tuple[int, HiSparseResidentManager, KVCacheBlock], ...]
    release_after: bool
    copy_enqueued: bool = False


class HiSparseKVCacheController:
    """Own HiSparse residency transitions for a KV-cache coordinator."""

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        managers: tuple[SingleTypeKVCacheManager, ...],
        max_model_len: int,
    ) -> None:
        self.groups = kv_cache_config.kv_cache_groups
        self.managers = managers
        self.max_model_len = max_model_len

        resident_entries_by_pool: dict[
            int, list[tuple[int, HiSparseResidentManager]]
        ] = {}
        hot_managers_by_pool: dict[int, list[HiSparseHotManager]] = {}
        growing_managers_by_pool: dict[int, int] = {}
        for group_id, (group, manager) in enumerate(zip(self.groups, managers)):
            pool_id = group.block_pool_id
            if isinstance(group.kv_cache_spec, HiSparseResidentSpec):
                if not isinstance(manager, HiSparseResidentManager):
                    raise TypeError(
                        "HiSparse resident specs require resident cache managers."
                    )
                resident_entries_by_pool.setdefault(pool_id, []).append(
                    (group_id, manager)
                )
            if isinstance(manager, HiSparseHotManager):
                hot_managers_by_pool.setdefault(pool_id, []).append(manager)
            else:
                growing_managers_by_pool[pool_id] = (
                    growing_managers_by_pool.get(pool_id, 0) + 1
                )

        self.resident_entries_by_pool = {
            pool_id: tuple(entries)
            for pool_id, entries in resident_entries_by_pool.items()
        }
        self.resident_entries = tuple(
            entry
            for entries in self.resident_entries_by_pool.values()
            for entry in entries
        )
        self.host_manager = managers[0] if self.resident_entries else None
        self.hot_managers_by_pool = {
            pool_id: tuple(managers)
            for pool_id, managers in hot_managers_by_pool.items()
        }
        self.hot_managers = tuple(
            manager
            for managers in self.hot_managers_by_pool.values()
            for manager in managers
        )
        self.growing_managers_by_pool = growing_managers_by_pool

        self.block_table_updates: set[str] = set()
        self.spills_to_send: list[HiSparseSpill] = []
        self.pending_spills: dict[int, _PendingSpill] = {}
        self.pending_pages: dict[tuple[str, int], int] = {}
        self.transition_target_pages: dict[str, int] = {}
        self.next_spill_id = 0

    def require_hot_if_needed(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_local_computed_tokens: int,
    ) -> None:
        has_cpu_history = bool(new_computed_blocks) or (
            total_computed_tokens > num_local_computed_tokens
        )
        if not self.resident_entries or has_cpu_history:
            for manager in self.hot_managers:
                manager.require_hot(request_id)

    def reclaim_resident_blocks(self, block_pool_id: int, num_blocks: int) -> int:
        """Reclaim host-valid pages and enqueue copies for GPU-only pages."""
        resident_entries = self.resident_entries_by_pool.get(block_pool_id, ())
        if not resident_entries or num_blocks <= 0:
            return 0
        resident_managers = tuple(manager for _, manager in resident_entries)
        spill_plan_budget = max(
            self.max_model_len // resident_managers[0].block_size
            - len(self.spills_to_send),
            0,
        )
        candidates = resident_managers[0].reclaimable_pages()
        for manager in resident_managers[1:]:
            candidates.intersection_update(manager.reclaimable_pages())

        hot_managers = self.hot_managers_by_pool.get(block_pool_id, ())
        by_request: dict[str, list[int]] = {}
        for request_id, block_idx in candidates:
            by_request.setdefault(request_id, []).append(block_idx)
        num_blocks = max(
            num_blocks,
            len(by_request) * self.growing_managers_by_pool[block_pool_id],
        )

        reclaimed = 0
        eventual = sum(
            len(pending.resident_entries)
            for pending in self.pending_spills.values()
            if pending.release_after
            and pending.resident_entries[0][1].block_pool
            is resident_managers[0].block_pool
        )
        for request_id, pages in sorted(
            by_request.items(), key=lambda item: len(item[1]), reverse=True
        ):
            pages.sort()
            has_hot = all(manager.has_hot(request_id) for manager in hot_managers)
            if has_hot:
                for block_idx in pages:
                    if eventual + reclaimed >= num_blocks:
                        break
                    if all(
                        block_idx in manager.host_valid_pages.get(request_id, set())
                        for manager in resident_managers
                    ):
                        for manager in resident_managers:
                            block = manager.release_resident_page(request_id, block_idx)
                            assert block is not None
                            reclaimed += 1
                        self.block_table_updates.add(request_id)
                    else:
                        pending = (request_id, block_idx) in self.pending_pages
                        if not pending and spill_plan_budget == 0:
                            continue
                        if self._mark_or_plan_release(
                            request_id, block_idx, resident_entries
                        ):
                            eventual += len(resident_entries)
                            if not pending:
                                spill_plan_budget -= 1
                continue

            if request_id in self.transition_target_pages:
                continue
            hot_cost = sum(manager.blocks_per_request for manager in hot_managers)
            remaining = max(num_blocks - reclaimed - eventual, 1)
            pages_needed = cdiv(remaining + hot_cost, len(resident_entries))
            pages_needed = min(len(pages), pages_needed, spill_plan_budget)
            if pages_needed * len(resident_entries) <= hot_cost:
                continue
            for hot_manager in hot_managers:
                hot_manager.require_hot(request_id)
            self.transition_target_pages[request_id] = pages_needed
            for block_idx in pages[:pages_needed]:
                pending = (request_id, block_idx) in self.pending_pages
                if self._mark_or_plan_release(request_id, block_idx, resident_entries):
                    eventual += len(resident_entries)
                    if not pending:
                        spill_plan_budget -= 1
            eventual -= hot_cost
            if reclaimed + eventual >= num_blocks:
                break
        return reclaimed

    def _mark_or_plan_release(
        self,
        request_id: str,
        block_idx: int,
        resident_entries: Sequence[tuple[int, HiSparseResidentManager]],
    ) -> bool:
        pending_id = self.pending_pages.get((request_id, block_idx))
        if pending_id is not None:
            pending = self.pending_spills[pending_id]
            if pending.release_after:
                return False
            pending.release_after = True
            return True
        return (
            self._plan_spill(
                request_id,
                block_idx,
                resident_entries,
                release_after=True,
                after_forward=False,
            )
            is not None
        )

    def plan_prefix_materialization(
        self, request_id: str, num_computed_tokens: int
    ) -> None:
        if not self.resident_entries:
            return
        assert self.host_manager is not None
        host_block_size = self.host_manager.block_size
        resident_block_size = self.resident_entries[0][1].block_size
        if host_block_size % resident_block_size != 0:
            raise RuntimeError(
                "HiSparse host and resident block sizes are incompatible."
            )
        num_pages = (num_computed_tokens // host_block_size) * (
            host_block_size // resident_block_size
        )
        for page_idx in range(num_pages):
            key = (request_id, page_idx)
            if key in self.pending_pages:
                continue
            if all(
                page_idx in manager.host_valid_pages.get(request_id, set())
                for _, manager in self.resident_entries
            ):
                continue
            if not all(
                manager.get_resident_page(request_id, page_idx) is not None
                for _, manager in self.resident_entries
            ):
                continue
            self._plan_spill(
                request_id,
                page_idx,
                self.resident_entries,
                release_after=False,
                after_forward=True,
            )

    def _plan_spill(
        self,
        request_id: str,
        page_idx: int,
        resident_entries: Sequence[tuple[int, HiSparseResidentManager]],
        *,
        release_after: bool,
        after_forward: bool,
    ) -> HiSparseSpill | None:
        assert self.host_manager is not None
        resident_block_size = resident_entries[0][1].block_size
        if self.host_manager.block_size % resident_block_size != 0:
            raise RuntimeError(
                "HiSparse host and resident block sizes are incompatible."
            )
        pages_per_host_block = self.host_manager.block_size // resident_block_size
        host_block_idx = page_idx // pages_per_host_block
        host_blocks = self.host_manager.req_to_blocks.get(request_id)
        if host_blocks is None or host_block_idx >= len(host_blocks):
            return None
        host_block = host_blocks[host_block_idx]
        if host_block.is_null:
            return None

        blocks: list[tuple[int, HiSparseResidentManager, KVCacheBlock]] = []
        for group_id, manager in resident_entries:
            block = manager.get_resident_page(request_id, page_idx)
            if block is None:
                return None
            blocks.append((group_id, manager, block))

        self.host_manager.block_pool.touch([host_block])
        for _, manager, block in blocks:
            manager.block_pool.touch([block])
        spill_id = self.next_spill_id
        self.next_spill_id += 1
        plan = HiSparseSpill(
            spill_id=spill_id,
            request_id=request_id,
            page_index=page_idx,
            host_block_id=host_block.block_id,
            host_page_offset=page_idx % pages_per_host_block,
            resident_block_ids=tuple(
                (group_id, block.block_id) for group_id, _, block in blocks
            ),
            after_forward=after_forward,
        )
        self.pending_spills[spill_id] = _PendingSpill(
            plan=plan,
            host_manager=self.host_manager,
            host_block=host_block,
            resident_entries=tuple(blocks),
            release_after=release_after,
        )
        self.pending_pages[(request_id, page_idx)] = spill_id
        self.spills_to_send.append(plan)
        return plan

    def take_spills(self) -> list[HiSparseSpill] | None:
        if not self.spills_to_send:
            return None
        plans = self.spills_to_send
        self.spills_to_send = []
        return plans

    def has_pending_reclamation(self) -> bool:
        return bool(self.transition_target_pages) or any(
            pending.release_after for pending in self.pending_spills.values()
        )

    def are_requests_fully_resident(self, request_ids: Sequence[str]) -> bool:
        return (
            bool(request_ids)
            and bool(self.resident_entries)
            and all(
                manager.is_fully_resident(request_id)
                for request_id in request_ids
                for _, manager in self.resident_entries
            )
        )

    def complete_spills(self, spill_ids: Sequence[int] | None) -> None:
        if not spill_ids:
            return
        for spill_id in spill_ids:
            pending = self.pending_spills.get(spill_id)
            if pending is not None:
                pending.copy_enqueued = True

        request_ids = {
            pending.plan.request_id
            for pending in self.pending_spills.values()
            if pending.copy_enqueued
        }
        for request_id in request_ids:
            ready = [
                pending
                for pending in self.pending_spills.values()
                if pending.copy_enqueued and pending.plan.request_id == request_id
            ]
            transition_target = self.transition_target_pages.get(request_id)
            release_ready = [pending for pending in ready if pending.release_after]
            if transition_target is not None and len(release_ready) < transition_target:
                for pending in ready:
                    if not pending.release_after:
                        self._finalize_spill(pending, release=False)
                continue

            if transition_target is not None:
                for pending in release_ready:
                    self._finalize_spill(pending, release=True)
                for pending in ready:
                    if not pending.release_after:
                        self._finalize_spill(pending, release=False)
                self.transition_target_pages.pop(request_id, None)
                if any(
                    request_id in manager.req_to_blocks
                    for _, manager, _ in release_ready[0].resident_entries
                ):
                    for manager in self.hot_managers:
                        manager.activate_hot(request_id)
                    self.block_table_updates.add(request_id)
            else:
                for pending in ready:
                    self._finalize_spill(pending, release=pending.release_after)

    def _finalize_spill(self, pending: _PendingSpill, *, release: bool) -> None:
        plan = pending.plan
        self.pending_spills.pop(plan.spill_id, None)
        self.pending_pages.pop((plan.request_id, plan.page_index), None)
        table_changed = False
        for _, manager, block in pending.resident_entries:
            current = manager.get_resident_page(plan.request_id, plan.page_index)
            if current is block:
                manager.mark_host_valid(plan.request_id, plan.page_index)
                if release:
                    released = manager.release_resident_page(
                        plan.request_id,
                        plan.page_index,
                        expected_block=block,
                    )
                    assert released is block
                    table_changed = True
            manager.block_pool.free_blocks([block])
        pending.host_manager.block_pool.free_blocks([pending.host_block])
        if table_changed:
            self.block_table_updates.add(plan.request_id)

    def take_block_table_update_requests(self) -> set[str]:
        requests = self.block_table_updates
        self.block_table_updates = set()
        return requests
