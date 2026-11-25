# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Physical side of session KV reuse: a tree of immutable committed states.

A node holds the blocks for path positions `[start, end)` of one computed
sequence; its parent holds the positions before `start`. Concatenating the
blocks along a root-to-node path gives a block table. The store never compares
tokens or hashes blocks: the resolver tells it which node a request matches,
and the store only checks that the node still holds that position.

Invariants:
  1. Only the store publishes nodes, from the blocks of the request that
     computed them.
  2. A published node never changes except by truncating its tail. Only the
     publishing request may still write its tail block, and only past `end`.
     A request resuming inside a block gets a private copy of it.
  3. A block returns to the pool only when its `ref_cnt` reaches 0; each node
     holds one reference on each of its blocks.
"""

import math
import secrets
from collections import Counter, OrderedDict
from collections.abc import Callable
from typing import Any

from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_utils import KVCacheBlock, KVCacheBlockCopy
from vllm.v1.core.single_type_kv_cache_manager import (
    CircularBufferManager,
    FullAttentionManager,
    MambaManager,
    SlidingWindowManager,
)
from vllm.v1.request import Request, RequestStatus
from vllm.v1.session.protocol import (
    NodeExtended,
    NodeId,
    NodePublished,
    NodeTruncated,
    RequestFinished,
    SessionBinding,
    StoreEvent,
    StoreReset,
)
from vllm.v1.session.resolver import pack_tokens

_COMMITTED = (RequestStatus.FINISHED_STOPPED, RequestStatus.FINISHED_LENGTH_CAPPED)
# Aborted requests (including stop strings, which the frontend aborts) still
# computed valid KV; they publish it without advancing their session.
_PUBLISHED = (*_COMMITTED, RequestStatus.FINISHED_ABORTED)
# Provisional: the longest shared span a hybrid divergence must have before
# it is saved, for models whose attention block is larger than this.
_MAX_CHECKPOINT_GAP = 256


def validate_session_cache_config(vllm_config: VllmConfig) -> None:
    unsupported = {
        "KV connectors": vllm_config.kv_transfer_config is not None,
        "EC connectors": vllm_config.ec_transfer_config is not None,
        "KV cache events": (
            vllm_config.kv_events_config is not None
            and vllm_config.kv_events_config.enable_kv_cache_events
        ),
        "LoRA": vllm_config.lora_config is not None,
        "encoder-decoder models": vllm_config.model_config.is_encoder_decoder,
        "pipeline parallelism": (
            vllm_config.parallel_config.pipeline_parallel_size > 1
        ),
    }
    found = [name for name, present in unsupported.items() if present]
    if found:
        raise ValueError(f"Session KV cache does not support {', '.join(found)} yet")


def validate_session_request(request: Request) -> None:
    if request.prompt_embeds is not None or request.mm_features:
        raise ValueError("Session KV cache only supports text token prompts")
    if request.resumable:
        raise ValueError("Session KV cache does not support streaming input yet")


class _Node:
    __slots__ = (
        "node_id",
        "parent",
        "start",
        "end",
        "blocks",
        "children",
        "commit_points",
    )

    def __init__(
        self,
        node_id: NodeId,
        parent: "_Node | None",
        start: int,
        end: int,
        blocks: tuple[list[KVCacheBlock], ...],
        commit_points: tuple[int, ...] | None,
    ) -> None:
        self.node_id = node_id
        self.parent = parent
        self.start = start
        self.end = end
        # Per KV cache group: blocks for block indices
        # `[start // block_size, cdiv(end, block_size))`.
        self.blocks = blocks
        self.children: list[_Node] = []
        self.commit_points = commit_points

    def resume_length(self, length: int, alignment: int) -> int:
        """Longest position <= `length` a request can resume this node at."""
        if self.commit_points is None:
            length = min(length, self.end)
            length -= length % alignment
            return length if length > self.start else 0
        return max(
            (c for c in self.commit_points if self.start < c <= length), default=0
        )


class SessionKVCacheManager(KVCacheManager):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if kwargs.get("enable_caching"):
            raise ValueError("Session KV cache replaces prefix caching")
        super().__init__(*args, **kwargs)
        managers = self.coordinator.single_type_managers
        # Groups that opt out of prefix caching (windows rebuilt by bounded
        # replay, and ring buffers outside hybrid models) are per-request
        # scratch that nodes never hold, as with prefix caching.
        supported = (
            FullAttentionManager,
            SlidingWindowManager,
            MambaManager,
            CircularBufferManager,
        )
        if any(type(m) not in supported for m in managers):
            raise ValueError(
                "Session KV cache supports full, sliding-window, ring-buffer and "
                "Mamba layers only; got "
                f"{sorted({type(m).__name__ for m in managers})}"
            )
        # Mamba state and a sliding window are both state that moves on as a
        # request computes: a node keeps it only at commit points (a copy of
        # the Mamba state; references to the window's blocks), so these models
        # resume at commit points, and full-attention-only models anywhere.
        self.commit_points_only = any(
            isinstance(m, MambaManager) or self._windowed(m) for m in managers
        )
        # In such models a ring buffer (the open group of a compressed cache,
        # e.g. GLM-5.3-Flash's indexer tail) is state too: saved at commit
        # points like Mamba state. Attention-only models resume at block
        # boundaries, where the ring is empty, so it stays per-request scratch.
        self._ring_groups = [
            g
            for g, m in enumerate(managers)
            if self.commit_points_only and isinstance(m, CircularBufferManager)
        ]
        # With speculative decoding, the state after the last verified token
        # lives in whichever speculative block the acceptance count selects,
        # so a hybrid request saves its state where its prefill ends instead
        # of where it stops.
        self.speculative_state = any(
            isinstance(m, MambaManager) and m.num_speculative_blocks for m in managers
        )
        # EAGLE-style draft KV at a position also depends on the next
        # `draft_lookahead` tokens. Resuming without them costs draft
        # acceptance, never output correctness (the target model verifies);
        # attention-only models give up the few tokens, as prefix caching does,
        # while hybrid models resume exactly at their saved states.
        self.draft_lookahead = (
            0 if self.commit_points_only else kwargs.get("num_hash_lookahead_tokens", 0)
        )
        self.block_sizes = [m.block_size for m in managers]
        self._window_groups = [g for g, m in enumerate(managers) if self._windowed(m)]
        positional = [
            m.block_size
            for m in managers
            if self._positional(m) and not self._stateful(m)
        ] or [m.block_size for m in managers if self._positional(m)]
        # A divergence checkpoint costs one recurrent state block. Require the
        # shared span to cover an attention block (the same memory), capped at
        # _MAX_CHECKPOINT_GAP so hybrid models with very large blocks still
        # share common system prompts; shorter spans (such as a chat
        # template's header) are recomputed.
        self.min_checkpoint_gap = min(*positional, _MAX_CHECKPOINT_GAP)
        # Attention-only models resume at block boundaries and recompute the
        # rest of the block, which is cheaper than copying it. Hybrid models
        # must resume exactly at a commit point, copying the boundary blocks.
        self.resume_alignment = math.lcm(*positional)
        # Compressed caches (`tokens_per_state > 1`, e.g. DeepSeek V4.1 and
        # GLM-5.3 indexers) finish a state only at a multiple of the ratio;
        # the open group lives in per-request scratch. A hybrid state saved
        # elsewhere could not be resumed, so commit points are multiples of
        # every ratio. Attention-only resumes are block aligned, which blocks
        # (whole numbers of states) already satisfy.
        all_ratios = [
            spec.tokens_per_state
            for group, m in zip(self.kv_cache_config.kv_cache_groups, managers)
            if self._positional(m) and not self._stateful(m)
            for spec in group.kv_cache_specs.values()
        ]
        ratios = [r for r in all_ratios if isinstance(r, int)]
        if len(ratios) != len(all_ratios):
            raise ValueError("Session KV cache needs whole-token compression ratios")
        # A saved ring holds the open group, so states then need no alignment.
        self.commit_alignment = 1 if self._ring_groups else math.lcm(1, *ratios)
        assert all(
            m.block_size % self.commit_alignment == 0
            for m in managers
            if self._positional(m) and not self._stateful(m)
        )
        # A hybrid request's own final state is unusable when it sits in a
        # speculative block or off the commit alignment, so it also saves a
        # state where its prefill ends (rounded down to the alignment).
        self.save_at_prefill_end = self.commit_points_only and (
            self.speculative_state or self.commit_alignment > 1
        )

        self.epoch = 0
        self.seq = 0
        self._publish_seq = 0
        self._origin = secrets.randbits(63)
        self._next_node_id = 1
        # Least recently used first.
        self.nodes: OrderedDict[NodeId, _Node] = OrderedDict()
        self._listeners: list[Callable[[StoreEvent], None]] = []
        self._rebind: Callable[[Request], SessionBinding] | None = None
        # The attach just handed to the scheduler, protected from eviction
        # only while that request allocates. Like prefix caching, a waiting
        # request's match never keeps running requests from growing.
        self._pending: tuple[str, list[_Node], int] | None = None
        # Deepest position on each running request's path held by a node.
        self._tip: dict[str, tuple[_Node | None, int]] = {}
        # The node each running request publishes and extends as it computes.
        self._own: dict[str, _Node] = {}
        # Checkpoints whose recurrent state is copied with this step's
        # batched block copies: not a copy source until the step is sent.
        self._unsettled: set[NodeId] = set()
        # States of released sessions still in use (by a running request or
        # another session's state below them), freed once nothing uses them.
        self._orphans: set[NodeId] = set()
        # Tokens each request had computed before the chunk scheduled this
        # step, in case it is preempted before the step is sent.
        self._scheduled_from: dict[str, int] = {}

        self.num_attached_requests = 0
        self.num_attached_tokens = 0
        self.num_stale_bindings = 0
        self.num_evicted_blocks = 0

    def connect(
        self,
        listener: Callable[[StoreEvent], None],
        rebind: Callable[[Request], SessionBinding] | None = None,
    ) -> None:
        self._listeners.append(listener)
        self._rebind = rebind

    def _emit(self, make_event: Callable[[int], StoreEvent]) -> None:
        self.seq += 1
        event = make_event(self.seq)
        for listener in self._listeners:
            listener(event)

    # ------------------------------------------------------------------ attach
    def prefix_cache_lookup_enabled(self, request: Request) -> bool:
        return request.prefix_cache_hit_limit != 0

    def get_num_cached_tokens(self, request: Request) -> int:
        return 0

    def get_computed_blocks(
        self, request: Request, max_cache_hit_length: int | None = None
    ) -> tuple[KVCacheBlocks, int, int]:
        self._pending = None
        binding = request.session_binding
        if not self.prefix_cache_lookup_enabled(request) or binding is None:
            return self.empty_kv_cache_blocks, 0, 0
        if binding.seq < self._publish_seq and self._rebind is not None:
            # Only a new node can offer a longer match.
            binding = request.session_binding = self._rebind(request)
        if binding.epoch != self.epoch:
            self.num_stale_bindings += 1
            return self.empty_kv_cache_blocks, 0, 0

        limit = request.get_max_cache_hit_length(max_cache_hit_length)
        found = None
        for candidate in binding.candidates:
            found = self._resume_point(
                self.nodes.get(candidate.node_id),
                min(candidate.length - self.draft_lookahead, limit),
            )
            if found is not None:
                break
        else:
            if binding.candidates:
                self.num_stale_bindings += 1
        checkpoint = self._checkpoint_position(
            request, min(binding.match_length, limit), found[1] if found else 0
        )
        if found is None:
            return self.empty_kv_cache_blocks, 0, checkpoint

        node, length = found
        path = self._path(node)
        self._pending = (request.request_id, path, length)
        blocks = tuple(
            self._assemble(path, length, group)
            for group in range(len(self.block_sizes))
        )
        return self.create_kv_cache_blocks(blocks), length, checkpoint

    def take_kv_cache_block_copies(
        self,
    ) -> tuple[list[KVCacheBlockCopy], list[KVCacheBlock]]:
        # Called once at the end of every scheduling step.
        self._unsettled.clear()
        self._scheduled_from.clear()
        return super().take_kv_cache_block_copies()

    def _checkpoint_position(self, request: Request, shared: int, resume: int) -> int:
        """Where a hybrid request should first stop a prefill chunk and save
        its recurrent state: the end of the prefix it shares with other
        requests, when that is far enough past where it resumes to be worth a
        state, else where its prefill state must be saved. Attention-only
        models publish every chunk and need no stop."""
        if not self.commit_points_only:
            return 0
        shared -= shared % self.commit_alignment
        if shared - resume >= self.min_checkpoint_gap:
            return shared
        prefill_end = self._prefill_save_point(request)
        return prefill_end if prefill_end > resume else 0

    def _prefill_save_point(self, request: Request) -> int:
        """Where a hybrid request saves its state at the end of its prefill,
        when its final state will not be usable; 0 otherwise."""
        if not self.save_at_prefill_end:
            return 0
        end = request.num_prompt_tokens
        return end - end % self.commit_alignment

    def _resume_point(
        self, node: _Node | None, length: int
    ) -> tuple[_Node, int] | None:
        # A truncated or partially matching node falls back to its ancestors.
        while node is not None and length > 0:
            resume = 0
            if node.node_id not in self._unsettled:
                resume = node.resume_length(length, self.resume_alignment)
            if resume:
                return node, resume
            length = min(length, node.start)
            node = node.parent
        return None

    @staticmethod
    def _path(node: _Node) -> list[_Node]:
        path = []
        current: _Node | None = node
        while current is not None:
            path.append(current)
            current = current.parent
        return path

    @staticmethod
    def _windowed(manager: Any) -> bool:
        """Sliding-window groups whose blocks nodes hold only for the window
        before each commit point (scratch windows are not positional)."""
        return isinstance(manager, SlidingWindowManager) and manager.prefix_cacheable(
            manager.manager_config
        )

    def _is_ring(self, manager: Any) -> bool:
        return isinstance(manager, CircularBufferManager) and bool(self._ring_groups)

    def _stateful(self, manager: Any) -> bool:
        """Groups whose content moves on as a request computes, so nodes keep
        it only at commit points."""
        return (
            isinstance(manager, MambaManager)
            or self._windowed(manager)
            or self._is_ring(manager)
        )

    def _positional(self, manager: Any) -> bool:
        """Whether nodes hold this group's blocks at all: not per-request
        scratch. Mamba groups are positional for one block that holds the
        state at a node's commit point."""
        return manager.prefix_cacheable(manager.manager_config) or self._is_ring(
            manager
        )

    def _assemble(
        self, path: list[_Node], length: int, group: int
    ) -> list[KVCacheBlock]:
        """Block table for positions `[0, length)`; deeper nodes win where a
        child's first block re-covers its parent's last one."""
        manager = self.coordinator.single_type_managers[group]
        if not self._positional(manager):
            return []
        if self._is_ring(manager):
            # One ring per request: the resumed state's saved ring.
            return list(path[0].blocks[group][:1])
        block_size = self.block_sizes[group]
        table: list[KVCacheBlock | None] = [None] * cdiv(length, block_size)
        for node in path:
            for index, block in enumerate(
                node.blocks[group], start=node.start // block_size
            ):
                if index < len(table) and table[index] is None and not block.is_null:
                    table[index] = block
        if self._windowed(manager):
            # Positions before the window are never read.
            null = self.block_pool.null_block
            return [null if block is None else block for block in table]
        assert all(block is not None for block in table), (
            "path does not cover the resumed prefix"
        )
        return table  # type: ignore[return-value]

    # ------------------------------------------------------------- allocation
    def allocate_slots(
        self, request: Request, num_new_tokens: int, *args: Any, **kwargs: Any
    ) -> KVCacheBlocks | None:
        if self._pending is not None and self._pending[0] != request.request_id:
            self._pending = None
        new_blocks = super().allocate_slots(request, num_new_tokens, *args, **kwargs)
        pending, self._pending = self._pending, None
        if new_blocks is None:
            return None
        num_scheduled_end = request.num_computed_tokens + num_new_tokens
        if pending is not None:
            _, path, length = pending
            num_scheduled_end += length
        self._scheduled_from[request.request_id] = num_scheduled_end - num_new_tokens
        if pending is not None:
            self._tip[request.request_id] = (path[0], length)
            for node in path:
                self.nodes.move_to_end(node.node_id)
            self.num_attached_requests += 1
            self.num_attached_tokens += length
        boundary = request.num_computed_tokens
        prefill_end = self._prefill_save_point(request)
        if (
            self.commit_points_only
            and boundary
            and boundary in (request.shared_prefix_boundary, prefill_end)
            and not self._resumable_at(request, boundary)
        ):
            # The previous chunk stopped where this request diverges from
            # others, or where its prefill state must be saved; save a copy
            # of its state before this step advances it.
            checkpoint = self._publish(request, boundary)
            if checkpoint is not None:
                self._seal(request, checkpoint, boundary, copy=True)
            if prefill_end > boundary == request.shared_prefix_boundary:
                # The scheduler stops each prefill chunk at this boundary;
                # after the divergence, stop next where the prefill ends.
                request.shared_prefix_boundary = prefill_end
        # Like prefix caching, publish each prompt chunk from the step that
        # computes it: every attention layer writes all tokens' KV before any
        # request in the batch reads it, and a preempted request can resume
        # from its own chunks. Recurrent state is updated in place, so a
        # hybrid chunk only lets other requests find where they diverge; it
        # becomes resumable where the request saves or stops its state.
        end = min(num_scheduled_end, request.num_prompt_tokens)
        if end > self._tip.get(request.request_id, (None, 0))[1]:
            self._publish(request, end)
        return new_blocks

    def _resumable_at(self, request: Request, position: int) -> bool:
        node, tip = self._tip.get(request.request_id, (None, 0))
        return (
            node is not None
            and tip == position
            and node.resume_length(position, self.resume_alignment) == position
        )

    def _publish(self, request: Request, end: int) -> _Node | None:
        """Publish the request's KV up to `end` in its own node."""
        own = self._own.get(request.request_id)
        if own is not None and self._can_extend(request, own):
            return self._extend(request, own, end)
        parent, start = self._tip.get(request.request_id, (None, 0))
        if parent is not None and (
            parent.node_id not in self.nodes or parent.end < start
        ):
            # The request still holds every block of its own path.
            parent, start = None, 0
        if end <= start:
            return parent if end == start else None
        managers = self.coordinator.single_type_managers
        blocks = tuple(
            []
            if self._stateful(m) or not self._positional(m)
            else list(
                m.req_to_blocks[request.request_id][
                    start // m.block_size : cdiv(end, m.block_size)
                ]
            )
            for m in managers
        )
        if any(block.is_null for group in blocks for block in group):
            return None
        self.block_pool.touch([b for group in blocks for b in group])
        # A hybrid node is resumable only once sealed with a recurrent state.
        commit_points = () if self.commit_points_only else None
        node_id = NodeId(self._origin, self._next_node_id)
        node = _Node(node_id, parent, start, end, blocks, commit_points)
        self._next_node_id += 1
        self.nodes[node.node_id] = node
        if parent is not None:
            parent.children.append(node)
        self._tip[request.request_id] = (node, end)
        self._own[request.request_id] = node
        self._publish_seq = self.seq + 1
        tokens = pack_tokens(request.all_token_ids[start:end])
        self._emit(
            lambda seq: NodePublished(
                seq,
                node.node_id,
                parent.node_id if parent is not None else None,
                start,
                end,
                tokens,
                commit_points,
                request.request_id,
            )
        )
        return node

    def _seal(self, request: Request, node: _Node, end: int, copy: bool) -> bool:
        """Make a hybrid request's own node resumable at its end `end` by
        giving it the recurrent state there: a copy if the request computes
        past it, else the request's own state. A sealed node never changes."""
        request_id = request.request_id
        if (
            node is not self._own.get(request_id)
            or node.end != end
            or end % self.commit_alignment
        ):
            return False
        managers = self.coordinator.single_type_managers
        recurrent = [
            g
            for g, m in enumerate(managers)
            if isinstance(m, MambaManager) or self._is_ring(m)
        ]
        states = [
            managers[g].req_to_blocks[request_id][:1]
            if self._is_ring(managers[g])
            else managers[g].req_to_blocks[request_id][
                node.start // managers[g].block_size : cdiv(end, managers[g].block_size)
            ]
            for g in recurrent
        ]
        if any(block.is_null for group in states for block in group):
            return False
        windows: list[list[KVCacheBlock]] = []
        for g in self._window_groups:
            window = self._window(request, node, end, g)
            if window is None:
                return False
            windows.append(window)
        if copy:
            if not self._ensure_free_blocks(sum(map(len, states))):
                return False
            states = [
                [managers[g].copy_block(block) for block in group]
                for g, group in zip(recurrent, states)
            ]
            self._unsettled.add(node.node_id)
        self.block_pool.touch([b for group in states for b in group])
        for g, group in zip(recurrent, states):
            node.blocks[g][:] = group
        for g, window in zip(self._window_groups, windows):
            self.block_pool.touch(window)
            node.blocks[g][:] = window
        node.commit_points = (end,)
        del self._own[request_id]
        self._publish_seq = self.seq + 1
        self._emit(lambda seq: NodeExtended(seq, node.node_id, end, end, b"", (end,)))
        return True

    def _window(
        self, request: Request, node: _Node, end: int, group: int
    ) -> list[KVCacheBlock] | None:
        """The node's blocks of a sliding-window group for a commit point at
        `end`: the request's blocks for the window before `end`, null before
        it. None if the request no longer holds the window."""
        manager = self.coordinator.single_type_managers[group]
        size = manager.block_size
        first = node.start // size
        lower = max(first, manager.get_num_skipped_tokens(end) // size)
        held = manager.req_to_blocks[request.request_id][lower : cdiv(end, size)]
        if len(held) < cdiv(end, size) - lower or any(b.is_null for b in held):
            return None
        return [self.block_pool.null_block] * (lower - first) + list(held)

    def _can_extend(self, request: Request, node: _Node) -> bool:
        """A request may append to its own node only while it is still the
        node's tail writer: same position, and still writing the node's last
        (partial) block rather than a copy made after preemption."""
        if (
            node.commit_points
            or node.node_id not in self.nodes
            or self._tip.get(request.request_id, (None, 0)) != (node, node.end)
        ):
            return False
        for manager, blocks in zip(self.coordinator.single_type_managers, node.blocks):
            if self._stateful(manager) or not self._positional(manager):
                continue
            last = (node.end - 1) // manager.block_size
            req_blocks = manager.req_to_blocks[request.request_id]
            if node.end % manager.block_size and (
                last >= len(req_blocks) or req_blocks[last] is not blocks[-1]
            ):
                return False
        return True

    def _extend(self, request: Request, node: _Node, end: int) -> _Node:
        if end <= node.end:
            return node
        new_blocks = [
            []
            if self._stateful(manager) or not self._positional(manager)
            else manager.req_to_blocks[request.request_id][
                cdiv(node.end, manager.block_size) : cdiv(end, manager.block_size)
            ]
            for manager in self.coordinator.single_type_managers
        ]
        self.block_pool.touch([b for group in new_blocks for b in group])
        for blocks, added in zip(node.blocks, new_blocks):
            blocks.extend(added)
        start, node.end = node.end, end
        self._tip[request.request_id] = (node, end)
        self._publish_seq = self.seq + 1
        tokens = pack_tokens(request.all_token_ids[start:end])
        self._emit(lambda seq: NodeExtended(seq, node.node_id, start, end, tokens))
        return node

    def free(self, request: Request) -> None:
        request_id = request.request_id
        if self._pending is not None and self._pending[0] == request_id:
            self._pending = None
        if request.is_finished():
            committed = None
            # Speculative tokens still in flight may be rejected.
            end = min(request.num_computed_tokens, request.num_tokens)
            if request.status in _PUBLISHED and end > 0:
                committed = self._publish(request, end)
                if committed is not None and self.commit_points_only:
                    if not self.speculative_state:
                        self._seal(request, committed, end, copy=False)
                    if end not in (committed.commit_points or ()):
                        committed = None
            node_id = (
                committed.node_id
                if committed is not None and request.status in _COMMITTED
                else None
            )
            self._emit(lambda seq: RequestFinished(seq, request_id, node_id))
        else:
            own = self._own.get(request_id)
            computed = self._scheduled_from.get(request_id)
            if own is not None and computed is not None and own.end > computed:
                # Preempted in the step that scheduled it (priority policy):
                # the chunk it just published will never be computed.
                self._cut(own, computed)
        # A preempted request's chunks stay published but unprotected, as
        # with prefix caching; it re-attaches to them like any new request.
        self._tip.pop(request_id, None)
        self._own.pop(request_id, None)
        super().free(request)
        if self._orphans:
            self._free_orphans()

    # ----------------------------------------------------------------- release
    def release(self, node_ids: list[NodeId]) -> int:
        """Free the given states of ended sessions as far as nothing else uses
        them, and the rest as soon as nothing does. Returns the number of
        blocks freed now."""
        num_free = self.block_pool.get_num_free_blocks()
        self._orphans.update(n for n in node_ids if n in self.nodes)
        self._free_orphans()
        return self.block_pool.get_num_free_blocks() - num_free

    def _free_orphans(self) -> None:
        """Remove released states nothing uses, deepest first, and cut the
        rest down to what other states below them still read."""
        protected = self._protected()
        changed = True
        while changed:
            changed = False
            for node_id in list(self._orphans):
                node = self.nodes.get(node_id)
                if node is None:
                    self._orphans.discard(node_id)
                elif node_id in protected:
                    continue
                elif not node.children:
                    self._remove(node)
                    changed = True
                elif not self._trims_whole_nodes() and node.end > max(
                    child.start for child in node.children
                ):
                    self._cut(node, node.start)

    # ---------------------------------------------------------------- eviction
    def _ensure_free_blocks(self, num_blocks: int) -> bool:
        """Evict idle states only if that makes room for `num_blocks`."""
        deficit = num_blocks - self.block_pool.get_num_free_blocks()
        if deficit <= 0:
            return True
        if self._num_reclaimable_blocks(deficit) < deficit:
            return False
        while (
            deficit := num_blocks - self.block_pool.get_num_free_blocks()
        ) > 0 and self._evict(deficit):
            pass
        return deficit <= 0

    def _protected(self) -> set[NodeId]:
        protected = (
            {node.node_id for node in self._pending[1]} if self._pending else set()
        )
        protected.update(
            node.node_id for node, _ in self._tip.values() if node is not None
        )
        return protected

    def _num_reclaimable_blocks(self, limit: int) -> int:
        """Count blocks repeated eviction can free, stopping at `limit`:
        every block whose references all come from idle states, except below
        where a running or attaching request reads through an ancestor."""
        protected = self._protected()
        # Per ancestor of a protected node: the highest start it must keep.
        floors: dict[NodeId, int] = {}
        for node_id in protected:
            child = self.nodes.get(node_id)
            parent = child.parent if child is not None else None
            while child is not None and parent is not None:
                seen = parent.node_id in floors
                floors[parent.node_id] = max(floors.get(parent.node_id, 0), child.start)
                if seen:
                    break
                child, parent = parent, parent.parent
        held: Counter[int] = Counter()
        count = 0
        for node in self.nodes.values():
            if node.node_id in protected:
                continue
            floor = floors.get(node.node_id)
            if floor is None:
                candidates = [b for group in node.blocks for b in group]
            elif not self._trims_whole_nodes():
                first = floor // self.block_sizes[0] - node.start // self.block_sizes[0]
                candidates = node.blocks[0][first:]
            else:
                continue
            for block in candidates:
                held[block.block_id] += 1
                if held[block.block_id] == block.ref_cnt:
                    count += 1
                    if count >= limit:
                        return count
        return count

    def _evict(self, num_blocks: int) -> bool:
        """Free up to `num_blocks` from the tail of the least recently used
        idle state that has blocks of its own, like prefix caching's LRU."""
        protected = self._protected()
        idle = [node for node in self.nodes.values() if node.node_id not in protected]
        for node in idle:
            if self._trim(node, num_blocks):
                return True
        idle_leaves = [node for node in idle if not node.children]
        if idle_leaves:
            # Every idle leaf only shares blocks; dropping one exposes its
            # parent's tail.
            self._remove(idle_leaves[0])
            return True
        return False

    def _trims_whole_nodes(self) -> bool:
        return self.commit_points_only or len(self.block_sizes) > 1

    def _num_kept_blocks(self, node: _Node) -> int:
        """Blocks of `node` that its children's block tables read. Attention
        resumes are block aligned, so nothing reads a partial last block
        that a child re-covers."""
        block_size = self.block_sizes[0]
        floor = max((child.start for child in node.children), default=node.start)
        return floor // block_size - node.start // block_size

    def _num_trimmable_blocks(self, node: _Node) -> int:
        if self._trims_whole_nodes():
            if node.children:
                return 0
            return sum(b.ref_cnt == 1 for group in node.blocks for b in group)
        blocks = node.blocks[0]
        keep = len(blocks)
        kept = self._num_kept_blocks(node)
        while keep > kept and blocks[keep - 1].ref_cnt == 1:
            keep -= 1
        return len(blocks) - keep

    def _trim(self, node: _Node, num_blocks: int) -> int:
        freed = self._num_trimmable_blocks(node)
        if not freed:
            return 0
        if self._trims_whole_nodes():
            self._remove(node)
            return freed
        freed = min(freed, num_blocks)
        blocks = node.blocks[0]
        keep = len(blocks) - freed
        if not keep and not node.children:
            self._remove(node)
            return freed
        block_size = self.block_sizes[0]
        self._cut(node, (node.start // block_size + keep) * block_size)
        self.num_evicted_blocks += freed
        return freed

    def _cut(self, node: _Node, end: int) -> None:
        """Shrink `node` to `[start, end)`, never below a child's start."""
        if end <= node.start and not node.children:
            self._remove(node)
            return
        floor = max((child.start for child in node.children), default=node.start)
        end = max(floor, min(node.end, end))
        for manager, blocks in zip(self.coordinator.single_type_managers, node.blocks):
            if self._stateful(manager) or not self._positional(manager):
                continue
            keep = cdiv(end, manager.block_size) - node.start // manager.block_size
            self.block_pool.free_blocks(reversed(blocks[keep:]))
            del blocks[keep:]
        node.end = end
        commit_points = node.commit_points
        self._emit(lambda seq: NodeTruncated(seq, node.node_id, end, commit_points))

    def _remove(self, node: _Node) -> None:
        del self.nodes[node.node_id]
        self._orphans.discard(node.node_id)
        if node.parent is not None:
            node.parent.children.remove(node)
        freed = sum(b.ref_cnt == 1 for group in node.blocks for b in group)
        self.num_evicted_blocks += freed
        for group in node.blocks:
            self.block_pool.free_blocks(reversed(group))
        self._emit(lambda seq: NodeTruncated(seq, node.node_id, node.start, None))

    def reset_prefix_cache(self) -> bool:
        if any(m.req_to_blocks for m in self.coordinator.single_type_managers):
            # Like prefix caching, never reset under running requests.
            return False
        for node in reversed(self.nodes.values()):
            for group in node.blocks:
                self.block_pool.free_blocks(reversed(group))
        self.nodes.clear()
        self._pending = None
        self._unsettled.clear()
        self._orphans.clear()
        self._tip.clear()
        self._own.clear()
        self.epoch += 1
        # Ids of dropped states are never handed out again.
        self._origin = secrets.randbits(63)
        self._next_node_id = 1
        self._emit(lambda seq: StoreReset(seq, self.epoch))
        return super().reset_prefix_cache()

    def session_stats(self) -> dict[str, int]:
        held = {b.block_id for n in self.nodes.values() for g in n.blocks for b in g}
        return {
            "nodes": len(self.nodes),
            "node_blocks": len(held),
            "attached_requests": self.num_attached_requests,
            "attached_tokens": self.num_attached_tokens,
            "stale_bindings": self.num_stale_bindings,
            "evicted_blocks": self.num_evicted_blocks,
        }
