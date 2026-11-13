# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resumable recurrent-state checkpoints for hybrid prefix caching.

`HybridCheckpointManager` is what the scheduler and the KV cache manager
call. It keeps three kinds of recovery units resumable:

- block-boundary states that align mode caches itself (the prompt's replay
  boundary and shared-prefix junctions under retention interval 0);
- a decode-boundary state per finished request: a scratch block holds the
  running state at the last block boundary the output crossed, published
  when the request finishes;
- a frontier per finished request (`frontier.FrontierIndex`), at its exact
  end inside a block.

Under speculative decoding `speculative.SpeculativeFrontiers` says where a
finished request's exact state is, and decode-boundary states come from the
block a verify step left the state at a crossed block boundary in.

A request's finish also releases the session leaf it resumed from, unless
another continuation needs it (`_replace_leaf`). With GreedyDual-Size
eviction (`eviction.GreedyDualSizeEviction`), block-boundary states are
never released this way; the eviction order decides when they go.
"""

import math
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

from vllm.v1.core.hybrid_checkpoint.cost import PrefillCostModel
from vllm.v1.core.hybrid_checkpoint.eviction import GreedyDualSizeEviction
from vllm.v1.core.hybrid_checkpoint.frontier import FrontierIndex, HitBlocks
from vllm.v1.core.hybrid_checkpoint.speculative import (
    SpeculativeFrontiers,
    check_speculative_config,
)
from vllm.v1.core.hybrid_checkpoint.units import CheckpointGraph
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashWithGroupId,
    KVCacheBlock,
    MambaStateGather,
    get_block_hash,
    get_group_id,
    make_block_hash_with_group_id,
)
from vllm.v1.kv_cache_interface import CircularBufferSpec
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.single_type_kv_cache_manager import MambaManager
    from vllm.v1.request import Request


def layout_groups(
    coordinator: "KVCacheCoordinator", speculative_decoding: bool = False
) -> tuple[list[int], list[int], list[int]]:
    """Full-attention, Mamba and indexer-ring group ids.

    Raises:
        ValueError: if a group has another type, a ring declares no way to
            resume inside its open group (under speculative decoding: no
            decode-path catch-up), or a kind is missing.

    """
    from vllm.v1.core.single_type_kv_cache_manager import (
        CircularBufferManager,
        FullAttentionManager,
        MambaManager,
    )

    pages: list[int] = []
    states: list[int] = []
    rings: list[int] = []
    for group_id, manager in enumerate(coordinator.single_type_managers):
        if type(manager) is FullAttentionManager:
            pages.append(group_id)
        elif type(manager) is MambaManager:
            states.append(group_id)
        elif type(manager) is CircularBufferManager:
            spec = manager.kv_cache_spec
            assert isinstance(spec, CircularBufferSpec)
            if spec.resume_catchup is None:
                raise ValueError(
                    "mamba_frontier_checkpoint needs every indexer ring to "
                    f"declare resume_catchup; group {group_id} does not"
                )
            if speculative_decoding and spec.resume_catchup != "decode":
                # The prefill path would read the open group from a ring that
                # also holds draft rows.
                raise ValueError(
                    "mamba_frontier_checkpoint under speculative decoding needs "
                    "indexer rings that only the decode path reads; group "
                    f"{group_id} declares resume_catchup {spec.resume_catchup!r}"
                )
            rings.append(group_id)
        else:
            raise ValueError(
                "mamba_frontier_checkpoint supports full-attention, Mamba and "
                f"indexer-ring groups only; group {group_id} is "
                f"{type(manager).__name__}"
            )
    if not pages or not states:
        raise ValueError(
            "mamba_frontier_checkpoint needs a full-attention and a Mamba group"
        )
    return pages, states, rings


def check_frontier_checkpoint_config(
    vllm_config: "VllmConfig",
    *,
    block_size: int,
    enable_kv_cache_events: bool,
) -> None:
    """Raise unless frontier checkpoints can run with this configuration."""
    cache = vllm_config.cache_config
    parallel = vllm_config.parallel_config
    threshold = vllm_config.scheduler_config.long_prefill_token_threshold
    refusal = None
    if not cache.enable_prefix_caching or cache.mamba_cache_mode != "align":
        refusal = "prefix caching with mamba_cache_mode 'align'"
    elif cache.prefix_cache_retention_interval != 0:
        refusal = "prefix_cache_retention_interval 0"
    elif 0 < threshold < block_size:
        refusal = f"no long_prefill_token_threshold below the block size ({block_size})"
    elif vllm_config.speculative_config is not None and (
        speculative := check_speculative_config(vllm_config)
    ):
        refusal = speculative
    elif (
        vllm_config.speculative_config is None
        and vllm_config.num_speculative_tokens > 0
    ):
        refusal = "no diffusion decoding (diffusion_config.canvas_length)"
    elif cache.use_replayssm:
        refusal = "no ReplaySSM"
    elif (transfer := vllm_config.kv_transfer_config) is not None and (
        transfer.kv_role != "kv_both"
        # Under "fail" a request whose load failed finishes with its computed
        # length over blocks the load did not write: no frontier may be cut
        # there.
        or transfer.kv_load_failure_policy != "recompute"
    ):
        refusal = (
            "a KV connector, if any, with kv_role 'kv_both' and "
            "kv_load_failure_policy 'recompute'"
        )
    elif enable_kv_cache_events:
        refusal = "no KV cache events (enable_kv_cache_events)"
    elif (
        parallel.decode_context_parallel_size != 1
        or parallel.prefill_context_parallel_size != 1
    ):
        refusal = "decode and prefill context parallel size 1"
    if refusal is not None:
        raise ValueError(f"mamba_frontier_checkpoint needs {refusal}")


class HybridCheckpointManager:
    """Keeps recurrent states resumable past the block grid and, given a
    prefill cost model, orders evictions by GreedyDual-Size.

    Args:
        coordinator: The KV cache coordinator whose groups hold the units.
        hash_fn: The prefix-cache hash function.
        waiting: The scheduler's waiting requests, first to be scheduled
            first; GreedyDual-Size eviction keeps what their hits read.
        eviction_cost: The prefill cost model of GreedyDual-Size eviction, or
            None to keep the block pool's LRU order.
        eviction_leaf_last: GreedyDual-Size evicts inner units before leaves.
        eviction_leaf_idle_s: With `eviction_leaf_last`, a leaf unused this
            long ranks as inner.
        speculative_decoding: Whether requests verify draft tokens.

    """

    def __init__(
        self,
        coordinator: "KVCacheCoordinator",
        hash_fn: Callable[[Any], bytes],
        waiting: Callable[[], Iterable["Request"]],
        eviction_cost: PrefillCostModel | None,
        eviction_leaf_last: bool = False,
        eviction_leaf_idle_s: float | None = None,
        speculative_decoding: bool = False,
    ) -> None:
        pages, states, rings = layout_groups(coordinator, speculative_decoding)
        managers = coordinator.single_type_managers
        self.pool = coordinator.block_pool
        self.block_size = coordinator.scheduler_block_size
        self.mamba_managers: list[MambaManager] = [
            managers[g]  # type: ignore[misc]
            for g in states
        ]
        self.mamba_groups = frozenset(states)
        decode_rings = [
            spec.group_size
            for g in rings
            if isinstance(spec := managers[g].kv_cache_spec, CircularBufferSpec)
            and spec.resume_catchup == "decode"
        ]
        # A request resumed inside an open indexer group of this many tokens
        # takes one token per step until the group closes.
        self.catchup_pool_size = math.lcm(1, *decode_rings)
        self.graph = CheckpointGraph(self.block_size)
        self.frontiers = FrontierIndex(coordinator, hash_fn, pages, states, rings)
        # Under speculative decoding: where a finished request's state is.
        self.speculative: SpeculativeFrontiers | None = None
        if speculative_decoding:
            self.speculative = SpeculativeFrontiers()
        self.eviction: GreedyDualSizeEviction | None = None
        if eviction_cost is not None:
            self.eviction = GreedyDualSizeEviction(
                coordinator,
                self.graph,
                self.frontiers,
                eviction_cost,
                waiting,
                leaf_last=eviction_leaf_last,
                leaf_idle_s=eviction_leaf_idle_s,
            )
            # Blocks of a dropped frontier are reused before any cached block.
            self.frontiers.reclaim = True
        # Request -> the unit its latest lookup resumes it at, until admitted.
        self._resume: dict[str, BlockHash] = {}
        # Request -> units its admissions already counted as hits.
        self._touched: dict[str, set[BlockHash]] = {}
        # Requests resumed at a frontier inside an open indexer group, until it
        # closes.
        self._catching_up: set[str] = set()
        self.num_scratch_skipped = 0
        self.num_scratch_dropped = 0
        self.pool.key_removal_listener = self.keys_removed
        self.pool.eviction_policy = self.eviction
        for manager in self.mamba_managers:
            manager.state_cached_listener = self.state_cached
            manager.track_unwritten_blocks = True

    # Units.
    def complete(self, key: BlockHash) -> bool:
        cached = self.pool.cached_block_hash_to_block
        return all(
            cached.get_one_block(
                make_block_hash_with_group_id(key, m.kv_cache_group_id)
            )
            is not None
            for m in self.mamba_managers
        )

    def held(self, key: BlockHash) -> bool:
        cached = self.pool.cached_block_hash_to_block
        return any(
            block.ref_cnt
            for m in self.mamba_managers
            for block in cached.get_all_blocks(
                make_block_hash_with_group_id(key, m.kv_cache_group_id)
            )
        )

    def state_cached(
        self, key: BlockHash, num_tokens: int, request: "Request | None"
    ) -> None:
        """A Mamba group cached `key`; the graph takes it once every group
        has."""
        if self.complete(key):
            self.graph.inserted(key, num_tokens)
        if self.eviction is not None and request is not None:
            self.eviction.record(key, num_tokens, request)

    def keys_removed(self, keys: list[BlockHashWithGroupId]) -> None:
        """The block pool dropped `keys`. A unit no longer cached in some
        Mamba group cannot be hit."""
        self.frontiers.removed(keys)
        cached = self.pool.cached_block_hash_to_block
        for grouped in keys:
            if get_group_id(grouped) not in self.mamba_groups:
                continue
            key = get_block_hash(grouped)
            if key in self.graph.units and cached.get_one_block(grouped) is None:
                self.graph.removed(key)
        if self.eviction is not None:
            self.eviction.keys_removed(keys)

    def release(self, key: BlockHash) -> None:
        """Release a unit by a persistence rule: its keys go in every group.
        Under GreedyDual-Size a block-boundary unit stays cached, no longer a
        session leaf; the eviction order decides when it goes."""
        unit = self.graph.units.get(key)
        if self.eviction is not None and unit is not None and not unit.frontier:
            self.graph.leaves.discard(key)
            return
        self.graph.removed(key)
        if self.frontiers.drop(key):
            return
        cached = self.pool.cached_block_hash_to_block
        for manager in self.mamba_managers:
            grouped = make_block_hash_with_group_id(key, manager.kv_cache_group_id)
            while (block := cached.get_one_block(grouped)) is not None:
                if block.ref_cnt or not self.pool._maybe_evict_cached_block(block):
                    break
                self.frontiers.to_front(block)

    # Scheduler hooks.
    def begin_step(self) -> None:
        if self.eviction is not None:
            self.eviction.new_step()

    def step_completed(
        self,
        request: "Request",
        start: int,
        num_scheduled: int,
        verified: bool,
        num_sampled: int,
        num_kept: int,
    ) -> None:
        """Under speculative decoding, a step of `request` completed: it
        started at device position `start`, computed `num_scheduled`
        tokens, verified drafts or not, and the device sampled
        `num_sampled` tokens, of which the scheduler kept `num_kept`.
        When the kept tokens crossed a block boundary past the prompt,
        the state there, which the step left in the block before the
        boundary, is snapshotted as the decode-boundary state."""
        assert self.speculative is not None
        self.speculative.step_completed(
            request.request_id, start, num_scheduled, verified, num_sampled, num_kept
        )
        if not verified:
            return
        boundary = (start + num_kept) // self.block_size * self.block_size
        if boundary <= start or boundary <= request.num_prompt_tokens:
            return
        index = boundary // self.block_size - 1
        blocks = []
        for manager in self.mamba_managers:
            table = manager.req_to_blocks.get(request.request_id) or []
            if index >= len(table) or table[index].is_null:
                return
            blocks.append(table[index])
        self.snapshot_decode_boundary(request, blocks, boundary)

    def take_state_gathers(
        self,
    ) -> tuple[list[MambaStateGather], list[KVCacheBlock]]:
        """The gathers that complete states published since the last call,
        and the blocks they hold until the step that runs them."""
        gathers = []
        retained: list[KVCacheBlock] = []
        for gather in self.frontiers.pending_gathers:
            group_ids = tuple(gather.blocks)
            pairs = [gather.blocks[g] for g in group_ids]
            gathers.append(
                MambaStateGather(
                    column=gather.column,
                    group_ids=group_ids,
                    dst_block_ids=tuple(dst.block_id for dst, _ in pairs),
                    src_block_ids=tuple(src.block_id for _, src in pairs),
                )
            )
            retained.extend(block for pair in pairs for block in pair)
        self.frontiers.pending_gathers.clear()
        return gathers, retained

    def drop_state_gathers(self) -> None:
        """No step runs the pending gathers (the prefix cache is reset first):
        the frontiers they complete go, and the blocks they hold are freed."""
        retained: list[KVCacheBlock] = []
        for gather in self.frontiers.pending_gathers:
            assert gather.key is not None
            self.release(gather.key)
            retained.extend(block for pair in gather.blocks.values() for block in pair)
        self.frontiers.pending_gathers.clear()
        self.pool.free_blocks(retained)

    def extend_hit(
        self,
        request: "Request",
        blocks: HitBlocks,
        hit: int,
        max_length: int,
    ) -> tuple[HitBlocks, int]:
        return self.frontiers.extend(request, blocks, hit, max_length)

    def frontier_hit_tokens(self, request: "Request", num_hits: int) -> int:
        """The tokens past the block-aligned hit that a frontier serves the
        request if it is admitted `num_hits` tokens in, else 0."""
        return self.frontiers.hit_tokens(request.request_id, num_hits)

    def admitted(
        self,
        request: "Request",
        num_computed_tokens: int,
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        """The request's lookup resumes it at `num_computed_tokens`, the
        last `num_external_tokens` of which a KV connector loads; it is
        admitted once `allocated` follows."""
        resume: BlockHash | None = None
        if num_external_tokens:
            # It resumes at the state the load ends with, cached under the
            # request's own hash at that hash unit, not at its last local
            # block.
            num_units, rest = divmod(num_computed_tokens, self.pool.hash_block_size)
            if not rest and 0 < num_units <= len(request.block_hashes):
                resume = request.block_hashes[num_units - 1]
        else:
            for group in blocks.blocks:
                if group and (block_hash := group[-1].block_hash) is not None:
                    resume = get_block_hash(block_hash)
                    break
        frontier = self.frontiers.resumed_at(request.request_id, num_computed_tokens)
        resume = frontier or resume
        if resume is None:
            self._resume.pop(request.request_id, None)
        else:
            self._resume[request.request_id] = resume
        self.graph.admitted(request.request_id, resume)

    def catching_up(self, request_id: str, start: int) -> bool:
        """Whether the request resumed at a frontier inside an indexer group
        that is still open at `start`: it then takes one token per step,
        through the decode path, until the group closes."""
        if not start % self.catchup_pool_size:
            self._catching_up.discard(request_id)
            return False
        return request_id in self._catching_up or bool(
            self.frontiers.hit_tokens(request_id, start)
        )

    def allocated(self, request: "Request", num_hits: int) -> None:
        """The request got its blocks `num_hits` tokens in: each unit it
        resumes at counts one hit per request, however often a refused
        allocation repeats the lookup."""
        if (
            self.frontiers.hit_tokens(request.request_id, num_hits)
            and num_hits % self.catchup_pool_size
        ):
            self._catching_up.add(request.request_id)
        else:
            self._catching_up.discard(request.request_id)
        self.frontiers.admitted(request, num_hits)
        resume = self._resume.pop(request.request_id, None)
        if resume is None:
            return
        counted = self._touched.setdefault(request.request_id, set())
        if resume not in counted:
            counted.add(resume)
            self.graph.touched(resume)

    def request_finished(self, request: "Request") -> None:
        """Publish the request's decode-boundary state and its frontier before
        its blocks return to the pool, and release what they replace."""
        request_id = request.request_id
        exact = None
        if self.speculative is not None:
            if request.status == RequestStatus.FINISHED_ABORTED:
                # An aborted request publishes no frontier.
                self.speculative.forget(request_id)
            else:
                exact = self.speculative.resolve(request)
        leaf = self._publish_decode_boundary(request)
        self._resume.pop(request_id, None)
        self._touched.pop(request_id, None)
        self._catching_up.discard(request_id)
        superseded = self.frontiers.superseded.pop(request_id, ())
        if superseded:
            waiting = set(self._resume.values())
            superseded = tuple(k for k in superseded if not self._needed(k, waiting))
        for key in superseded:
            unit = self.graph.units.get(key)
            if (
                unit is not None
                and unit.frontier
                and key in self.graph.leaves
                and not unit.children
                and not unit.hits
                and not self.held(key)
            ):
                self.release(key)
        graph = self.graph
        if leaf is not None and leaf in graph.units:
            graph.link(request_id, leaf)
            self._replace_leaf(request_id, leaf, boundary=True)
            if self.eviction is not None:
                # Cached from a scratch block, with no request: its path.
                self.eviction.record(leaf, graph.units[leaf].num_tokens, request)
        else:
            leaf = None
        frontier = None
        if self.speculative is None or exact is not None:
            frontier = self.frontiers.publish(request, exact)
        if frontier is not None:
            num_tokens = (
                request.num_computed_tokens if exact is None else exact.num_tokens
            )
            self.state_cached(frontier, num_tokens, request)
            graph.link(request_id, frontier)
            if leaf is None:
                self._replace_leaf(request_id, frontier)
            else:
                graph.leaves.add(frontier)
        graph.forget(request_id)
        self.frontiers.forget(request_id)

    def _needed(self, key: BlockHash, waiting: set[BlockHash]) -> bool:
        """Whether a waiting request resumes at the unit `key`: its lookup did
        (`waiting`), or under GreedyDual-Size its hit pins the unit."""
        if key in waiting:
            return True
        unit = self.graph.units.get(key)
        return (
            unit is not None
            and self.eviction is not None
            and self.eviction.pinned(unit)
        )

    def _replace_leaf(
        self, request_id: str, key: BlockHash, boundary: bool = False
    ) -> None:
        """`key` becomes its session's leaf; release the leaf the request
        resumed from unless it is a branch point, held, or where a waiting
        request resumes (`_needed`). When that was a frontier and `key` a
        later boundary state, the boundary state the frontier extended goes
        too, unless another continuation reached it."""
        graph = self.graph
        graph.leaves.add(key)
        old = graph.resumed.get(request_id)
        if old is None or old == key or old not in graph.leaves:
            return
        unit = graph.units.get(old)
        waiting = set(self._resume.values())
        if (
            unit is None
            or unit.is_branch_point
            or self.held(old)
            or self._needed(old, waiting)
        ):
            return
        shadowed = None
        if boundary and unit.frontier and unit.parent is not None:
            parent = graph.units.get(unit.parent)
            if (
                parent is not None
                and not parent.frontier
                and not parent.children - {old}
                and not parent.hits
                and unit.parent in graph.leaves
                and not self.held(unit.parent)
                and not self._needed(unit.parent, waiting)
            ):
                shadowed = unit.parent
        self.release(old)
        if shadowed is not None and shadowed in graph.units:
            self.release(shadowed)

    def reset(self) -> None:
        """The prefix cache was reset while no request held a block."""
        self.graph.clear()
        self._resume.clear()
        self._touched.clear()
        self._catching_up.clear()
        if self.eviction is not None:
            self.eviction.reset()
        self.frontiers.reset()

    # Decode-boundary states.
    def decode_boundary_sources(self, request: "Request") -> list[KVCacheBlock] | None:
        """The block holding the request's running state in every Mamba
        group, or None unless every group has one."""
        running = []
        for manager in self.mamba_managers:
            block = _running_block(manager, request.request_id)
            if block is None:
                return None
            running.append(block)
        return running

    def snapshot_decode_boundary(
        self,
        request: "Request",
        running: list[KVCacheBlock],
        position: int | None = None,
    ) -> None:
        """Copy the state at a block boundary (`running`, taken before this
        step's allocation) into the request's scratch block in every Mamba
        group, or in none. `position`: the boundary (default the request's
        computed tokens). A snapshot leaves a block free."""
        request_id = request.request_id
        managers = self.mamba_managers
        new = sum(request_id not in m.decode_boundary_scratch for m in managers)
        if new and self.pool.get_num_free_blocks() <= new:
            self.num_scratch_skipped += 1
            return
        for manager, block in zip(managers, running):
            entry = manager.decode_boundary_scratch.get(request_id)
            scratch = self.pool.get_new_blocks(1)[0] if entry is None else entry[0]
            manager.queue_block_copy(block, scratch)
            manager.decode_boundary_scratch[request_id] = (
                scratch,
                request.num_computed_tokens if position is None else position,
            )

    def drop_decode_boundary_scratch(self, request_id: str) -> bool:
        """Free the request's scratch blocks if no copy into them is in
        flight. Returns whether it did."""
        managers = [
            m for m in self.mamba_managers if request_id in m.decode_boundary_scratch
        ]
        blocks = [m.decode_boundary_scratch[request_id][0] for m in managers]
        if not blocks or any(block.ref_cnt != 1 for block in blocks):
            return False
        for manager in managers:
            del manager.decode_boundary_scratch[request_id]
        self.pool.free_blocks(blocks)
        self.num_scratch_dropped += 1
        return True

    def _publish_decode_boundary(self, request: "Request") -> BlockHash | None:
        """Cache the state at the last block boundary the output crossed in
        every Mamba group, or in none, and free the scratch blocks."""
        # Under speculative decoding the running block is not the state at
        # the computed tokens; only snapshots are.
        from_running = self.speculative is None
        sources = [
            _decode_boundary(m, request, from_running) for m in self.mamba_managers
        ]
        complete = (
            bool(sources)
            and all(source is not None for source in sources)
            and len({source[1] for source in sources if source is not None}) == 1
        )
        leaf = None
        for manager, source in zip(self.mamba_managers, sources):
            if complete and source is not None:
                block, boundary = source
                published = self.pool.cache_partial_block(
                    request=request,
                    block=block,
                    num_tokens=boundary,
                    kv_cache_group_id=manager.kv_cache_group_id,
                    block_size=manager.block_size,
                    replace_existing_hashes=True,
                )
                if published is not None:
                    self.state_cached(get_block_hash(published), boundary, None)
                    if leaf is None:
                        leaf = get_block_hash(published)
            entry = manager.decode_boundary_scratch.pop(request.request_id, None)
            if entry is not None:
                self.pool.free_blocks([entry[0]])
        return leaf


def _running_block(manager: "MambaManager", request_id: str) -> KVCacheBlock | None:
    blocks = manager.req_to_blocks.get(request_id)
    index = (len(blocks) if blocks else 0) - 1 - manager.num_speculative_blocks
    if blocks is None or index < 0 or blocks[index].is_null:
        return None
    return blocks[index]


def _decode_boundary(
    manager: "MambaManager", request: "Request", from_running: bool
) -> tuple[KVCacheBlock, int] | None:
    """The block holding the state at the last block boundary the output
    crossed, and that boundary; None when there is none past the prompt.
    `from_running`: a running state at a block boundary counts too."""
    unit = manager.block_pool.hash_block_size
    block_size = manager.block_size
    num_computed = request.num_computed_tokens
    prompt_tail = request.num_prompt_tokens // unit * unit
    source = None
    boundary = 0
    if (
        from_running
        and num_computed % block_size == 0
        and num_computed > request.num_prompt_tokens
    ):
        running = _running_block(manager, request.request_id)
        if running is not None:
            source, boundary = running, num_computed
    entry = manager.decode_boundary_scratch.get(request.request_id)
    if source is None and entry is not None:
        snapshot_boundary = entry[1] // block_size * block_size
        if snapshot_boundary > prompt_tail:
            source, boundary = entry[0], snapshot_boundary
    if (
        source is None
        or boundary <= prompt_tail
        or boundary // unit > len(request.block_hashes)
    ):
        return None
    return source, boundary
