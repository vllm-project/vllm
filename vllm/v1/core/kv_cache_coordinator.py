# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

from vllm import envs
from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    KVCacheBlock,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    CrossAttentionManager,
    HiSparseHotManager,
    SingleTypeKVCacheManager,
    get_manager_for_kv_cache_spec,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    HiSparseResidentSpec,
    HiSparseSpill,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request


@dataclass
class _PendingHiSparseSpill:
    plan: HiSparseSpill
    host_manager: SingleTypeKVCacheManager
    host_block: KVCacheBlock
    resident_entries: tuple[tuple[int, SingleTypeKVCacheManager, KVCacheBlock], ...]
    release_after: bool
    copy_enqueued: bool = False


def _validate_prefix_cache_retention_interval(
    retention_interval: int | None,
    scheduler_block_size: int,
    kv_cache_config: KVCacheConfig,
) -> None:
    if retention_interval is None:
        return

    # Retention sparsifies sliding-window and Mamba (linear-attention)
    # checkpoints; full-attention and chunked-local groups cache densely and
    # ignore it (their hit granularity must stay fine).
    if not any(
        isinstance(g.kv_cache_spec, (SlidingWindowSpec, MambaSpec))
        for g in kv_cache_config.kv_cache_groups
    ):
        raise ValueError(
            "VLLM_PREFIX_CACHE_RETENTION_INTERVAL is set but this model has "
            "no sliding-window or Mamba KV cache group, so retention has no "
            "effect. Unset it (it only applies to sliding-window and Mamba "
            "attention)."
        )

    if retention_interval < 0 or retention_interval % scheduler_block_size != 0:
        raise ValueError(
            f"VLLM_PREFIX_CACHE_RETENTION_INTERVAL ({retention_interval}) "
            "must be non-negative and a multiple of scheduler_block_size "
            f"({scheduler_block_size})."
        )


class KVCacheCoordinator(ABC):
    """
    Coordinate the KV cache of different KV cache groups.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        max_in_flight_tokens: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        scheduler_block_size: int,
        hash_block_size: int,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        self.kv_cache_config = kv_cache_config
        self.max_model_len = max_model_len
        self.enable_caching = enable_caching
        # The scheduling granularity (LCM of all group block sizes), must be a multiple
        # of the hash_block_size and the block size of each group.
        assert scheduler_block_size % hash_block_size == 0 and all(
            scheduler_block_size % g.kv_cache_spec.block_size == 0
            for g in kv_cache_config.kv_cache_groups
        )
        self.scheduler_block_size = scheduler_block_size

        assert kv_cache_config.num_blocks_by_pool is not None
        pool_enable_caching = [False] * len(kv_cache_config.num_blocks_by_pool)
        for group in kv_cache_config.kv_cache_groups:
            pool_enable_caching[group.block_pool_id] |= (
                enable_caching and group.enable_prefix_caching
            )
        self.block_pools = tuple(
            BlockPool(
                num_gpu_blocks=num_blocks,
                enable_caching=pool_enable_caching[pool_id],
                hash_block_size=hash_block_size,
                enable_kv_cache_events=enable_kv_cache_events,
                metrics_collector=metrics_collector,
            )
            for pool_id, num_blocks in enumerate(kv_cache_config.num_blocks_by_pool)
        )
        # Compatibility alias for callers that only support the traditional
        # single-domain layout.
        self.block_pool = self.block_pools[0]

        # KV cache group indices that get the EAGLE last-block drop.
        self.eagle_group_ids: set[int] = {
            i for i, g in enumerate(kv_cache_config.kv_cache_groups) if g.is_eagle_group
        }
        # Conservatively fall back to flag all groups when no group is flagged.
        if use_eagle and not self.eagle_group_ids:
            self.eagle_group_ids = set(range(len(kv_cache_config.kv_cache_groups)))

        self.single_type_managers = tuple(
            get_manager_for_kv_cache_spec(
                kv_cache_spec=kv_cache_group.kv_cache_spec,
                max_in_flight_tokens=max_in_flight_tokens,
                max_model_len=max_model_len,
                block_pool=self.block_pools[kv_cache_group.block_pool_id],
                enable_caching=(
                    enable_caching and kv_cache_group.enable_prefix_caching
                ),
                kv_cache_group_id=i,
                dcp_world_size=dcp_world_size,
                pcp_world_size=pcp_world_size,
                scheduler_block_size=self.scheduler_block_size,
                needs_kv_cache_zeroing=(
                    kv_cache_group.block_pool_id
                    in self.kv_cache_config.zeroing_block_pool_ids
                ),
            )
            for i, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups)
        )
        self._hisparse_block_table_updates: set[str] = set()
        self._hisparse_spills_to_send: list[HiSparseSpill] = []
        self._pending_hisparse_spills: dict[int, _PendingHiSparseSpill] = {}
        self._pending_hisparse_pages: dict[tuple[str, int], int] = {}
        self._hisparse_transition_target_pages: dict[str, int] = {}
        self._next_hisparse_spill_id = 0

        # A positive retention interval must be a multiple of the base hit granularity
        # (``scheduler_block_size``) to land on real cache-hit boundaries.
        # 0 = keep only the latest replay boundary; None = dense;
        self.retention_interval = envs.VLLM_PREFIX_CACHE_RETENTION_INTERVAL
        _validate_prefix_cache_retention_interval(
            self.retention_interval, self.scheduler_block_size, kv_cache_config
        )

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_encoder_tokens: int,
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        return sum(
            self.get_num_blocks_to_allocate_by_pool(
                request_id,
                num_tokens,
                new_computed_blocks,
                num_encoder_tokens,
                total_computed_tokens,
                num_local_computed_tokens,
                num_tokens_main_model,
                apply_admission_cap=apply_admission_cap,
            )
        )

    def get_num_blocks_to_allocate_by_pool(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_encoder_tokens: int,
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> tuple[int, ...]:
        """Get allocation requirements independently for each block pool."""
        has_hisparse_resident = any(
            isinstance(group.kv_cache_spec, HiSparseResidentSpec)
            for group in self.kv_cache_config.kv_cache_groups
        )
        has_cpu_history = bool(new_computed_blocks[0]) or (
            total_computed_tokens > num_local_computed_tokens
        )
        if not has_hisparse_resident or has_cpu_history:
            for manager in self.single_type_managers:
                if isinstance(manager, HiSparseHotManager):
                    manager.require_hot(request_id)
        required = [0] * len(self.block_pools)
        for i, manager in enumerate(self.single_type_managers):
            group = self.kv_cache_config.kv_cache_groups[i]
            if isinstance(manager, CrossAttentionManager):
                num_blocks = manager.get_num_blocks_to_allocate(
                    request_id,
                    num_encoder_tokens,
                    [],
                    0,
                    0,
                    num_encoder_tokens,
                    apply_admission_cap=apply_admission_cap,
                )
            else:
                num_blocks = manager.get_num_blocks_to_allocate(
                    request_id,
                    num_tokens,
                    new_computed_blocks[i],
                    total_computed_tokens,
                    num_local_computed_tokens,
                    num_tokens_main_model,
                    apply_admission_cap=apply_admission_cap,
                )
            required[group.block_pool_id] += num_blocks
        return tuple(required)

    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        """
        Add the new computed blocks to the request. Optionally allocate new
            blocks for external computed tokens (if any).

        Args:
            request_id: The request ID.
            new_computed_blocks: The new computed blocks just hitting the
                prefix cache.
            num_local_computed_tokens: The number of local computed tokens.
            num_external_computed_tokens: The number of external computed tokens.
        """
        # A running request is already tracked in num_cached_block and won't
        # have new prefix-cache hits, so this is a no-op for it.
        if any(
            request_id in manager.num_cached_block
            for manager in self.single_type_managers
        ):
            assert all(len(blocks) == 0 for blocks in new_computed_blocks)
            return

        # Two-phase allocation (issue #33775): first touch every group's local
        # cache-hit blocks, then allocate external blocks for every group. This
        # ensures an earlier group's external `get_new_blocks` cannot evict a
        # later group's not-yet-touched cache-hit blocks.
        for i, manager in enumerate(self.single_type_managers):
            manager.add_local_computed_blocks(
                request_id,
                new_computed_blocks[i],
                num_local_computed_tokens,
                num_external_computed_tokens,
            )
        if num_external_computed_tokens > 0:
            for manager in self.single_type_managers:
                manager.allocate_external_computed_blocks(
                    request_id,
                    num_local_computed_tokens,
                    num_external_computed_tokens,
                )

    def allocate_new_blocks(
        self,
        request_id: str,
        num_tokens: int,
        num_tokens_main_model: int,
        num_encoder_tokens: int = 0,
    ) -> tuple[list[KVCacheBlock], ...]:
        """
        Allocate new blocks for the request to give it at least `num_tokens`
        token slots.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including
                tokens that are already allocated).
            num_tokens_main_model: The number of tokens for the main model (aka target
                model in spec decode). w/o spec decode, it is num_tokens;
                with spec decode, it is num_tokens - num_lookahead_tokens.
            num_encoder_tokens: The number of encoder tokens for allocating
                blocks for cross-attention.

        Returns:
            The new allocated blocks.
        """
        return tuple(
            manager.allocate_new_blocks(
                request_id,
                num_encoder_tokens
                if isinstance(manager, CrossAttentionManager)
                else num_tokens,
                num_tokens_main_model,
            )
            for manager in self.single_type_managers
        )

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """
        Cache the blocks for the request.

        Args:
            request: The request.
            num_computed_tokens: The total number of tokens
                that need to be cached
                (including tokens that are already cached).
        """
        for group, manager in zip(
            self.kv_cache_config.kv_cache_groups, self.single_type_managers
        ):
            if not group.enable_prefix_caching:
                continue
            manager.cache_blocks(
                request,
                num_computed_tokens,
                retention_interval=self.retention_interval,
            )
        self._plan_hisparse_prefix_materialization(
            request.request_id, num_computed_tokens
        )

    def free(self, request_id: str) -> None:
        """
        Free the blocks for the request.

        Args:
            request_id: The request ID.
        """
        for manager in self.single_type_managers:
            manager.free(request_id)

    def reclaim_hisparse_resident_blocks(
        self, block_pool_id: int, num_blocks: int
    ) -> int:
        """Reclaim host-valid pages and enqueue copies for GPU-only pages."""
        resident_entries = [
            (group_id, manager)
            for group_id, (group, manager) in enumerate(
                zip(
                    self.kv_cache_config.kv_cache_groups,
                    self.single_type_managers,
                )
            )
            if group.block_pool_id == block_pool_id
            and isinstance(group.kv_cache_spec, HiSparseResidentSpec)
        ]
        if not resident_entries or num_blocks <= 0:
            return 0
        resident_managers = [manager for _, manager in resident_entries]
        spill_plan_budget = max(
            self.max_model_len // resident_managers[0].block_size
            - len(self._hisparse_spills_to_send),
            0,
        )
        growing_managers = sum(
            1
            for group, manager in zip(
                self.kv_cache_config.kv_cache_groups,
                self.single_type_managers,
            )
            if group.block_pool_id == block_pool_id
            and not isinstance(manager, HiSparseHotManager)
        )
        candidates = resident_managers[0].reclaimable_pages()  # type: ignore[attr-defined]
        for manager in resident_managers[1:]:
            candidates.intersection_update(
                manager.reclaimable_pages()  # type: ignore[attr-defined]
            )
        hot_managers = [
            manager
            for group, manager in zip(
                self.kv_cache_config.kv_cache_groups,
                self.single_type_managers,
            )
            if group.block_pool_id == block_pool_id
            and isinstance(manager, HiSparseHotManager)
        ]
        by_request: dict[str, list[int]] = {}
        for request_id, block_idx in candidates:
            by_request.setdefault(request_id, []).append(block_idx)
        num_blocks = max(num_blocks, len(by_request) * growing_managers)

        reclaimed = 0
        eventual = sum(
            len(pending.resident_entries)
            for pending in self._pending_hisparse_spills.values()
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
                    host_valid = all(
                        block_idx in manager.host_valid_pages.get(request_id, set())  # type: ignore[attr-defined]
                        for manager in resident_managers
                    )
                    if host_valid:
                        for manager in resident_managers:
                            block = manager.release_resident_page(  # type: ignore[attr-defined]
                                request_id, block_idx
                            )
                            assert block is not None
                            reclaimed += 1
                        self._hisparse_block_table_updates.add(request_id)
                    else:
                        pending = (
                            request_id,
                            block_idx,
                        ) in self._pending_hisparse_pages
                        if not pending and spill_plan_budget == 0:
                            continue
                        if self._mark_or_plan_hisparse_release(
                            request_id, block_idx, resident_entries
                        ):
                            eventual += len(resident_entries)
                            if not pending:
                                spill_plan_budget -= 1
                continue

            if request_id in self._hisparse_transition_target_pages:
                continue
            hot_cost = sum(manager.blocks_per_request for manager in hot_managers)
            remaining = max(num_blocks - reclaimed - eventual, 1)
            pages_needed = cdiv(remaining + hot_cost, len(resident_entries))
            pages_needed = min(len(pages), pages_needed, spill_plan_budget)
            if pages_needed * len(resident_entries) <= hot_cost:
                continue
            for manager in hot_managers:
                manager.require_hot(request_id)
            self._hisparse_transition_target_pages[request_id] = pages_needed
            for block_idx in pages[:pages_needed]:
                pending = (request_id, block_idx) in self._pending_hisparse_pages
                if self._mark_or_plan_hisparse_release(
                    request_id, block_idx, resident_entries
                ):
                    eventual += len(resident_entries)
                    if not pending:
                        spill_plan_budget -= 1
            eventual -= hot_cost
            if reclaimed + eventual >= num_blocks:
                break
        return reclaimed

    def _mark_or_plan_hisparse_release(
        self,
        request_id: str,
        block_idx: int,
        resident_entries: list[tuple[int, SingleTypeKVCacheManager]],
    ) -> bool:
        pending_id = self._pending_hisparse_pages.get((request_id, block_idx))
        if pending_id is not None:
            pending = self._pending_hisparse_spills[pending_id]
            if pending.release_after:
                return False
            pending.release_after = True
            return True
        return (
            self._plan_hisparse_spill(
                request_id,
                block_idx,
                resident_entries,
                release_after=True,
                after_forward=False,
            )
            is not None
        )

    def _plan_hisparse_prefix_materialization(
        self, request_id: str, num_computed_tokens: int
    ) -> None:
        resident_entries = [
            (group_id, manager)
            for group_id, (group, manager) in enumerate(
                zip(
                    self.kv_cache_config.kv_cache_groups,
                    self.single_type_managers,
                )
            )
            if isinstance(group.kv_cache_spec, HiSparseResidentSpec)
        ]
        if not resident_entries:
            return
        host_block_size = self.single_type_managers[0].block_size
        resident_block_size = resident_entries[0][1].block_size
        if host_block_size % resident_block_size != 0:
            raise RuntimeError(
                "HiSparse host and resident block sizes are incompatible."
            )
        num_pages = (num_computed_tokens // host_block_size) * (
            host_block_size // resident_block_size
        )
        for page_idx in range(num_pages):
            key = (request_id, page_idx)
            if key in self._pending_hisparse_pages:
                continue
            if all(
                page_idx in manager.host_valid_pages.get(request_id, set())  # type: ignore[attr-defined]
                for _, manager in resident_entries
            ):
                continue
            if not all(
                manager.get_resident_page(request_id, page_idx) is not None  # type: ignore[attr-defined]
                for _, manager in resident_entries
            ):
                continue
            self._plan_hisparse_spill(
                request_id,
                page_idx,
                resident_entries,
                release_after=False,
                after_forward=True,
            )

    def _plan_hisparse_spill(
        self,
        request_id: str,
        page_idx: int,
        resident_entries: list[tuple[int, SingleTypeKVCacheManager]],
        *,
        release_after: bool,
        after_forward: bool,
    ) -> HiSparseSpill | None:
        host_manager = self.single_type_managers[0]
        resident_block_size = resident_entries[0][1].block_size
        if host_manager.block_size % resident_block_size != 0:
            raise RuntimeError(
                "HiSparse host and resident block sizes are incompatible."
            )
        pages_per_host_block = host_manager.block_size // resident_block_size
        host_block_idx = page_idx // pages_per_host_block
        host_blocks = host_manager.req_to_blocks.get(request_id)
        if host_blocks is None or host_block_idx >= len(host_blocks):
            return None
        host_block = host_blocks[host_block_idx]
        if host_block.is_null:
            return None
        blocks: list[tuple[int, SingleTypeKVCacheManager, KVCacheBlock]] = []
        for group_id, manager in resident_entries:
            block = manager.get_resident_page(request_id, page_idx)  # type: ignore[attr-defined]
            if block is None:
                return None
            blocks.append((group_id, manager, block))

        host_manager.block_pool.touch([host_block])
        for _, manager, block in blocks:
            manager.block_pool.touch([block])
        spill_id = self._next_hisparse_spill_id
        self._next_hisparse_spill_id += 1
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
        self._pending_hisparse_spills[spill_id] = _PendingHiSparseSpill(
            plan=plan,
            host_manager=host_manager,
            host_block=host_block,
            resident_entries=tuple(blocks),
            release_after=release_after,
        )
        self._pending_hisparse_pages[(request_id, page_idx)] = spill_id
        self._hisparse_spills_to_send.append(plan)
        return plan

    def take_hisparse_spills(self) -> list[HiSparseSpill] | None:
        if not self._hisparse_spills_to_send:
            return None
        plans = self._hisparse_spills_to_send
        self._hisparse_spills_to_send = []
        return plans

    def has_pending_hisparse_reclamation(self) -> bool:
        return bool(self._hisparse_transition_target_pages) or any(
            pending.release_after for pending in self._pending_hisparse_spills.values()
        )

    def are_hisparse_requests_fully_resident(self, request_ids: Sequence[str]) -> bool:
        resident_managers = [
            manager
            for group, manager in zip(
                self.kv_cache_config.kv_cache_groups,
                self.single_type_managers,
            )
            if isinstance(group.kv_cache_spec, HiSparseResidentSpec)
        ]
        return (
            bool(request_ids)
            and bool(resident_managers)
            and all(
                manager.is_fully_resident(request_id)  # type: ignore[attr-defined]
                for request_id in request_ids
                for manager in resident_managers
            )
        )

    def complete_hisparse_spills(self, spill_ids: list[int]) -> None:
        for spill_id in spill_ids:
            pending = self._pending_hisparse_spills.get(spill_id)
            if pending is not None:
                pending.copy_enqueued = True

        request_ids = {
            pending.plan.request_id
            for pending in self._pending_hisparse_spills.values()
            if pending.copy_enqueued
        }
        for request_id in request_ids:
            ready = [
                pending
                for pending in self._pending_hisparse_spills.values()
                if pending.copy_enqueued and pending.plan.request_id == request_id
            ]
            transition_target = self._hisparse_transition_target_pages.get(request_id)
            release_ready = [pending for pending in ready if pending.release_after]
            if transition_target is not None and len(release_ready) < transition_target:
                for pending in ready:
                    if not pending.release_after:
                        self._finalize_hisparse_spill(pending, release=False)
                continue

            if transition_target is not None:
                for pending in release_ready:
                    self._finalize_hisparse_spill(pending, release=True)
                for pending in ready:
                    if not pending.release_after:
                        self._finalize_hisparse_spill(pending, release=False)
                self._hisparse_transition_target_pages.pop(request_id, None)
                if any(
                    request_id in manager.req_to_blocks
                    for _, manager, _ in release_ready[0].resident_entries
                ):
                    for manager in self.single_type_managers:
                        if isinstance(manager, HiSparseHotManager):
                            manager.activate_hot(request_id)
                    self._hisparse_block_table_updates.add(request_id)
            else:
                for pending in ready:
                    self._finalize_hisparse_spill(
                        pending, release=pending.release_after
                    )

    def _finalize_hisparse_spill(
        self, pending: _PendingHiSparseSpill, *, release: bool
    ) -> None:
        plan = pending.plan
        self._pending_hisparse_spills.pop(plan.spill_id, None)
        self._pending_hisparse_pages.pop((plan.request_id, plan.page_index), None)
        table_changed = False
        for _, manager, block in pending.resident_entries:
            current = manager.get_resident_page(  # type: ignore[attr-defined]
                plan.request_id, plan.page_index
            )
            if current is block:
                manager.mark_host_valid(plan.request_id, plan.page_index)  # type: ignore[attr-defined]
                if release:
                    released = manager.release_resident_page(  # type: ignore[attr-defined]
                        plan.request_id,
                        plan.page_index,
                        expected_block=block,
                    )
                    assert released is block
                    table_changed = True
            manager.block_pool.free_blocks([block])
        pending.host_manager.block_pool.free_blocks([pending.host_block])
        if table_changed:
            self._hisparse_block_table_updates.add(plan.request_id)

    def take_hisparse_block_table_update_requests(self) -> set[str]:
        requests = self._hisparse_block_table_updates
        self._hisparse_block_table_updates = set()
        return requests

    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
        """
        Pop the request's bookkeeping from all single-type managers and
        return its blocks without returning them to the block pool. The
        caller must eventually pass the returned blocks to
        `block_pool.free_blocks`, freeing them in reverse order (so that
        tail blocks are evicted first).

        Args:
            request_id: The request ID.

        Returns:
            The request's blocks in allocation order.
        """
        blocks: list[KVCacheBlock] = []
        for manager in self.single_type_managers:
            blocks.extend(manager.pop_blocks_for_free(request_id))
        return blocks

    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
        """
        Get the number of common prefix blocks for all requests with allocated
        KV cache for each kv cache group.

        Args:
            running_request_id: The request ID of any running request, used to
                identify the common prefix blocks.

        Returns:
            list[int]: The number of common prefix blocks for each kv cache group.
        """
        return [
            manager.get_num_common_prefix_blocks(running_request_id)
            for manager in self.single_type_managers
        ]

    def remove_skipped_blocks(
        self,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        """
        Remove the blocks that are no longer needed from `blocks` and replace
        the removed blocks with null_block.

        Args:
            request_id: The request ID.
            processed_computed_tokens: Computed-token prefix length covering
                fully processed and committed tokens only (safe to free).
            num_prompt_tokens: Optional prompt length. R-SWA managers use this to
                free gap blocks between the prefill tail and decode window; other
                manager types ignore it.
        """
        for manager in self.single_type_managers:
            manager.remove_skipped_blocks(
                request_id, processed_computed_tokens, num_prompt_tokens
            )

    def get_blocks(self, request_id: str) -> tuple[list[KVCacheBlock], ...]:
        """
        Get the blocks for the request.
        """
        return tuple(
            manager.req_to_blocks.get(request_id) or []
            for manager in self.single_type_managers
        )

    @abstractmethod
    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int, int]:
        """Returns the per-group hit blocks, the hit length, and the number of
        ``num_uncached_common_prefix_tokens`` (a shared prefix that a
        sparse-retention group has not cached yet; 0 unless hybrid)."""
        pass

    def new_step_starts(self) -> None:
        """Notify each manager that a new step is starting."""
        for manager in self.single_type_managers:
            manager.new_step_starts()


class KVCacheCoordinatorNoPrefixCache(KVCacheCoordinator):
    """
    KV cache coordinator to use if prefix caching is disabled or unsupported.
    In contrast to UnitaryKVCacheCoordinator and HybridKVCacheCoordinator,
    supports arbitrary numbers of KV cache groups (including 0 groups).
    Does not implement any features related to prefix caching.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        max_in_flight_tokens: int,
        use_eagle: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        scheduler_block_size: int,
        hash_block_size: int,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        super().__init__(
            kv_cache_config,
            max_model_len,
            max_in_flight_tokens,
            use_eagle,
            False,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
        )
        self.num_single_type_manager = len(self.single_type_managers)

    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
        return [0] * self.num_single_type_manager

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int, int]:
        blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in range(self.num_single_type_manager)
        )
        return blocks, 0, 0


class UnitaryKVCacheCoordinator(KVCacheCoordinator):
    """
    KV cache coordinator for models with only one KV cache group. This is the
    case for models with only one KV cache type, e.g., all attention layers use
    full attention or all attention layers use sliding window attention.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        max_in_flight_tokens: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        scheduler_block_size: int,
        hash_block_size: int,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        super().__init__(
            kv_cache_config,
            max_model_len,
            max_in_flight_tokens,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
        )
        self.kv_cache_spec = self.kv_cache_config.kv_cache_groups[0].kv_cache_spec
        self.block_size = self.kv_cache_spec.block_size
        self.dcp_world_size = dcp_world_size
        self.pcp_world_size = pcp_world_size
        if dcp_world_size > 1:
            self.block_size *= dcp_world_size
        # For models using only Mamba, block_size is set to max_model_len when
        # prefix caching is disabled, and hash_block_size validation is skipped.
        assert not enable_caching or (hash_block_size == self.block_size), (
            "UnitaryKVCacheCoordinator assumes hash_block_size == block_size"
        )
        assert len(self.kv_cache_config.kv_cache_groups) == 1, (
            "UnitaryKVCacheCoordinator assumes only one kv cache group"
        )
        # Single group; useless but just set ``use_eagle`` for consistency regardless.
        self.single_type_managers[0].use_eagle = 0 in self.eagle_group_ids

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int, int]:
        hit_blocks, hit_length = self.single_type_managers[0].find_longest_cache_hit(
            block_hashes=block_hashes,
            max_length=max_cache_hit_length,
            kv_cache_group_ids=[0],
            block_pool=self.single_type_managers[0].block_pool,
            kv_cache_spec=self.kv_cache_spec,
            drop_eagle_block=0 in self.eagle_group_ids,
            alignment_tokens=self.block_size,
            dcp_world_size=self.dcp_world_size,
            pcp_world_size=self.pcp_world_size,
        )
        # Single group: nothing "uncached common" -- no other group to lag it.
        return hit_blocks, hit_length, 0


class SpecGroup(NamedTuple):
    """KV cache groups that share one spec, batched together for a single
    cache-hit lookup.

    ``use_eagle`` is True iff any member group is an EAGLE/MTP group. Members
    sharing a spec are cached and looked up jointly, so the EAGLE last-block drop
    is necessarily decided for the whole spec group.
    """

    spec: KVCacheSpec
    group_ids: list[int]
    manager_cls: type[SingleTypeKVCacheManager]
    use_eagle: bool
    block_pool_id: int


class HybridKVCacheCoordinator(KVCacheCoordinator):
    """
    KV cache coordinator for hybrid models with multiple KV cache types, and
    thus multiple kv cache groups.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        max_in_flight_tokens: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        scheduler_block_size: int,
        hash_block_size: int,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        super().__init__(
            kv_cache_config,
            max_model_len,
            max_in_flight_tokens,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
        )
        # hash_block_size: the block size used to compute block hashes.
        # The actual block size usually equals hash_block_size, but in cases where
        # different KV cache groups have different block sizes, the actual block size
        # can be a multiple of hash_block_size.
        self.hash_block_size = hash_block_size
        self.dcp_world_size = dcp_world_size
        group_block_sizes = [
            manager.block_size for manager in self.single_type_managers
        ]
        assert all(
            block_size % hash_block_size == 0 for block_size in group_block_sizes
        ), (
            "Each KV cache group's real block_size must be divisible by "
            f"hash_block_size. block_sizes={group_block_sizes}, "
            f"hash_block_size={hash_block_size}"
        )
        assert pcp_world_size == 1, "PCP not support hybrid attn now."
        if dcp_world_size > 1:
            # DCP shards full-attention KV across ranks and replicates Mamba
            # state; other spec types (e.g. sliding window) have no DCP-aware
            # handling yet, so reject them explicitly.
            for g in kv_cache_config.kv_cache_groups:
                assert isinstance(g.kv_cache_spec, (FullAttentionSpec, MambaSpec)), (
                    "DCP with hybrid KV cache layouts only supports "
                    "full-attention and Mamba groups, got: "
                    f"{type(g.kv_cache_spec).__name__}."
                )
        # Partial hash hits are limited to full-attention + mamba ("align")
        # without context parallelism.
        self.enable_partial_hash_hits = dcp_world_size == 1 and any(
            isinstance(g.kv_cache_spec, MambaSpec)
            and g.kv_cache_spec.mamba_cache_mode == "align"
            and g.kv_cache_spec.block_size > hash_block_size
            for g in kv_cache_config.kv_cache_groups
        )
        self.verify_and_split_kv_cache_groups()

    @property
    def _cache_hit_alignment_tokens(self) -> int:
        # Fine-grained partial hits may return hash-block-aligned lengths;
        # otherwise it must stay scheduler-block-aligned.
        return (
            self.hash_block_size
            if self.enable_partial_hash_hits
            else self.scheduler_block_size
        )

    def verify_and_split_kv_cache_groups(self) -> None:
        """Group prefix-cache groups by spec type for efficient hit lookup.

        Despite the coordinator name, this may leave one group: hybrid layouts
        can contain auxiliary groups, such as HiSparse hot caches, that do not
        participate in prefix caching.
        """
        self.attention_groups: list[SpecGroup] = []
        for i, g in enumerate(self.kv_cache_config.kv_cache_groups):
            if not g.enable_prefix_caching:
                continue
            manager_cls = self.single_type_managers[i].__class__
            spec = g.kv_cache_spec
            use_eagle = i in self.eagle_group_ids

            # Try to find an existing group with the same spec
            for idx, group in enumerate(self.attention_groups):
                if group.spec == spec and group.block_pool_id == g.block_pool_id:
                    assert manager_cls is group.manager_cls, (
                        "Expected same manager class for identical KV cache specs."
                    )
                    group.group_ids.append(i)
                    if use_eagle and not group.use_eagle:
                        self.attention_groups[idx] = group._replace(use_eagle=True)
                    break
            else:
                self.attention_groups.append(
                    SpecGroup(spec, [i], manager_cls, use_eagle, g.block_pool_id)
                )

        assert self.attention_groups, (
            "Prefix caching requires at least one persistent KV cache group."
        )

        # Put full attention first: its efficient left-to-right scan provides
        # a tighter initial bound, reducing work for subsequent groups.
        self.attention_groups.sort(
            key=lambda g: not isinstance(g.spec, FullAttentionSpec)
        )

        # Dense reference group for per-group lookups (None when the model
        # has no full-attention layers): full attention is downward-closed,
        # so any group reporting a longer per-group hit implies the union of
        # per-group hits is not consistent at a single boundary (#46453).
        first = self.attention_groups[0]
        self.full_attention_group_id: int | None = (
            first.group_ids[0] if isinstance(first.spec, FullAttentionSpec) else None
        )

        # Propagate the eagle bit to each manager (default to ``use_eagle=False``).
        for group in self.attention_groups:
            if group.use_eagle:
                for gid in group.group_ids:
                    self.single_type_managers[gid].use_eagle = True

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        if self.enable_partial_hash_hits:
            aligned_num_computed_tokens = num_computed_tokens
        else:
            # Cache hits in this coordinator are always a multiple of
            # ``scheduler_block_size`` tokens (see ``find_longest_cache_hit``).
            # Within an aligned region, SWA groups may only consult a subset of
            # blocks per ``scheduler_block_size``-segment so the unused blocks
            # also stay out of the prefix-cache hash map.
            aligned_num_computed_tokens = (
                num_computed_tokens
                // self.scheduler_block_size
                * self.scheduler_block_size
            )
        for group, manager in zip(
            self.kv_cache_config.kv_cache_groups, self.single_type_managers
        ):
            if not group.enable_prefix_caching:
                continue
            num_tokens_to_cache = aligned_num_computed_tokens
            # EAGLE groups match one block past each aligned boundary and drop
            # it, so make that lookahead block eligible to be cached.
            if manager.use_eagle and aligned_num_computed_tokens > 0:
                num_tokens_to_cache = min(
                    num_computed_tokens,
                    aligned_num_computed_tokens + manager.block_size,
                )
            # The manager already knows the fine hit granularity
            # (``scheduler_block_size``); retention is passed separately so it
            # can keep both the coarse segment tails and the fine replay
            # boundary (which needs the fine value).
            manager.cache_blocks(
                request,
                num_tokens_to_cache,
                retention_interval=self.retention_interval,
            )
        self._plan_hisparse_prefix_materialization(
            request.request_id, aligned_num_computed_tokens
        )

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int, int]:
        """
        Find the longest cache hit using an iterative fixed-point algorithm.

        Each attention type either accepts the current candidate length or
        reduces it. If any type reduces the length, restart checks over all
        types. This converges because length monotonically decreases and is
        bounded below by 0.

        Args:
            block_hashes: The block hashes of the request.
            max_cache_hit_length: The maximum length of the cache hit.

        Returns:
            A tuple containing:
                - A tuple of the cache hit blocks for each single type manager.
                - The number of tokens of the reconciled (combined) cache hit.
                - ``num_uncached_common_prefix_tokens``: a shared prefix that a
                  sparse-retention group has not cached yet (0 unless hybrid).
        """

        num_groups = len(self.kv_cache_config.kv_cache_groups)
        hit_length = max_cache_hit_length
        longest_hit_length = 0
        hit_blocks_by_group: list[list[KVCacheBlock] | None] = [None] * num_groups
        hit_length_by_group: list[int] = [0] * num_groups

        # Simple hybrid (1 full attn + 1 other): one iteration suffices.
        # Full attn is always first if it exists.
        is_simple_hybrid = len(self.attention_groups) == 2 and isinstance(
            self.attention_groups[0].spec, FullAttentionSpec
        )

        # Attention-group indices whose EAGLE drop is verified at the current
        # ``curr_hit_length``. Each eagle group applies the drop at most once
        # per candidate length (see issue #32802).
        eagle_verified: set[int] = set()

        while True:
            curr_hit_length = hit_length

            for idx, (
                spec,
                group_ids,
                manager_cls,
                use_eagle,
                block_pool_id,
            ) in enumerate(self.attention_groups):
                first_group_id = group_ids[0]
                # DCP/PCP shard each block's KV across ranks, so the manager's
                # effective block size may exceed the spec's.
                group_block_size = self.single_type_managers[first_group_id].block_size
                cached_blocks = hit_blocks_by_group[first_group_id]
                if isinstance(spec, FullAttentionSpec) and cached_blocks is not None:
                    # Full attention is downward-closed: we only need to look
                    # up cached blocks once; on subsequent iterations just trim
                    # to the (reduced) current hit length.
                    curr_hit_length = min(
                        curr_hit_length, hit_length_by_group[first_group_id]
                    )
                    continue

                drop_eagle_block = use_eagle and idx not in eagle_verified

                _max_length = curr_hit_length
                # Eagle matches one extra drop unit (one hash unit for
                # fine-grained managers, else one cache block) and then drops
                # it, landing back at the candidate length. No margin for
                # mamba: its finder never drops (draft models have no mamba
                # layers), so the hit would grow past the candidate.
                if drop_eagle_block and not isinstance(spec, MambaSpec):
                    eagle_margin = (
                        self.hash_block_size
                        if self.enable_partial_hash_hits
                        and manager_cls.supports_fine_grained_hash_lookup
                        and group_block_size > self.hash_block_size
                        else group_block_size
                    )
                    _max_length = min(
                        curr_hit_length + eagle_margin, max_cache_hit_length
                    )
                hit_blocks, _new_hit_length = manager_cls.find_longest_cache_hit(
                    block_hashes=block_hashes,
                    max_length=_max_length,
                    kv_cache_group_ids=group_ids,
                    block_pool=self.block_pools[block_pool_id],
                    kv_cache_spec=spec,
                    drop_eagle_block=drop_eagle_block,
                    alignment_tokens=self._cache_hit_alignment_tokens,
                    dcp_world_size=(
                        self.dcp_world_size
                        if isinstance(spec, FullAttentionSpec)
                        else 1
                    ),
                )
                if drop_eagle_block:
                    eagle_verified.add(idx)
                elif _new_hit_length < curr_hit_length:
                    # length shrunk; invalidate previous eagle verifications
                    eagle_verified.clear()
                curr_hit_length = _new_hit_length
                for group_id, blocks in zip(group_ids, hit_blocks):
                    hit_blocks_by_group[group_id] = blocks
                    hit_length_by_group[group_id] = _new_hit_length

                longest_hit_length = max(longest_hit_length, curr_hit_length)

            if curr_hit_length >= hit_length:
                break
            hit_length = curr_hit_length
            if is_simple_hybrid:
                break

        # Truncate full attention blocks to final hit_length (if present)
        first_group = self.attention_groups[0]
        if isinstance(first_group.spec, FullAttentionSpec):
            group_block_size = self.single_type_managers[
                first_group.group_ids[0]
            ].block_size
            num_blocks = cdiv(hit_length, group_block_size)
            for group_id in first_group.group_ids:
                if (blks := hit_blocks_by_group[group_id]) is not None:
                    del blks[num_blocks:]
                    hit_length_by_group[group_id] = hit_length

        # Uncached shared prefix detection: if any attn. group cached a longer
        # prefix than the reconciled hit, it is an uncached common prefix across
        # requests that a sparse-retention group hasn't cached yet.
        num_uncached_common_prefix_tokens = longest_hit_length - hit_length
        cache_hit_blocks = tuple(
            blocks if blocks is not None else [] for blocks in hit_blocks_by_group
        )
        return cache_hit_blocks, hit_length, num_uncached_common_prefix_tokens

    def find_longest_cache_hit_per_group(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], tuple[int, ...]]:
        """Like find_longest_cache_hit but evaluates each group independently.

        Returns:
            (blocks_per_group, hit_lengths_per_group)
        """

        num_groups = len(self.kv_cache_config.kv_cache_groups)
        hit_blocks: list[list[KVCacheBlock]] = [[] for _ in range(num_groups)]
        hit_lengths: list[int] = [0] * num_groups

        for (
            spec,
            group_ids,
            manager_cls,
            use_eagle,
            block_pool_id,
        ) in self.attention_groups:
            blocks, group_hit = manager_cls.find_longest_cache_hit(
                block_hashes=block_hashes,
                max_length=max_cache_hit_length,
                kv_cache_group_ids=group_ids,
                block_pool=self.block_pools[block_pool_id],
                kv_cache_spec=spec,
                drop_eagle_block=use_eagle,
                alignment_tokens=self._cache_hit_alignment_tokens,
            )
            for gid, blks in zip(group_ids, blocks):
                hit_blocks[gid] = blks
                hit_lengths[gid] = group_hit

        return tuple(hit_blocks), tuple(hit_lengths)


def get_kv_cache_coordinator(
    kv_cache_config: KVCacheConfig,
    max_model_len: int,
    max_in_flight_tokens: int,
    use_eagle: bool,
    enable_caching: bool,
    enable_kv_cache_events: bool,
    dcp_world_size: int,
    pcp_world_size: int,
    scheduler_block_size: int,
    hash_block_size: int,
    metrics_collector: KVCacheMetricsCollector | None = None,
) -> KVCacheCoordinator:
    if not enable_caching:
        return KVCacheCoordinatorNoPrefixCache(
            kv_cache_config,
            max_model_len,
            max_in_flight_tokens,
            use_eagle,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
        )
    if len(kv_cache_config.kv_cache_groups) == 1:
        return UnitaryKVCacheCoordinator(
            kv_cache_config,
            max_model_len,
            max_in_flight_tokens,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
        )
    return HybridKVCacheCoordinator(
        kv_cache_config,
        max_model_len,
        max_in_flight_tokens,
        use_eagle,
        enable_caching,
        enable_kv_cache_events,
        dcp_world_size=dcp_world_size,
        pcp_world_size=pcp_world_size,
        scheduler_block_size=scheduler_block_size,
        hash_block_size=hash_block_size,
        metrics_collector=metrics_collector,
    )
