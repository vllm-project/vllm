# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Adapted from vllm-project/vllm-ascend
# (vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/).
"""Scheduler-side logic for MooncakeStoreConnector."""

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.coordinator import (  # noqa: E501
    partial_hash_hits_enabled,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (  # noqa: E501
    LoadSpec,
    MooncakeStoreConnectorMetadata,
    MooncakeStoreWorkerMetadata,
    ReqMeta,
    RequestTracker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker import (  # noqa: E501
    LookupKeyClient,
)
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

logger = init_logger(__name__)


def _new_req_prefill_tokens(request: NewRequestData) -> list[int]:
    """Tokens this prefill will compute KV for.

    Under the v2 model runner, resumed-from-preemption requests appear in
    ``scheduled_new_reqs`` with ``prefill_token_ids`` set to the request's full
    token list (prompt + previously-generated). For all other cases this falls
    back to the original prompt.
    """
    if request.prefill_token_ids is not None:
        return request.prefill_token_ids
    assert request.prompt_token_ids is not None
    return request.prompt_token_ids


class MooncakeStoreScheduler:
    """Scheduler-side component for MooncakeStoreConnector."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
    ):
        assert vllm_config.kv_transfer_config is not None
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        kvc_extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
        self.load_async = kvc_extra_config.get("load_async", True)
        self.lookup_async = kvc_extra_config.get("lookup_async", False)
        # Skips lookup CPU cost on instances that never load KV from the store.
        self.enable_lookup = kvc_extra_config.get("enable_lookup", True)
        self.save_decode_cache = kvc_extra_config.get("save_decode_cache", False)
        kv_event_config = vllm_config.kv_events_config
        self.enable_kv_events = bool(
            kv_event_config and kv_event_config.enable_kv_cache_events
        )
        self.client = LookupKeyClient(vllm_config)
        self.kv_cache_config = kv_cache_config

        # Align with the engine's own scheduler_block_size and hash_block_size.
        self._block_size, self._hash_block_size = resolve_kv_cache_block_sizes(
            kv_cache_config, vllm_config
        )
        self.enable_partial_hash_hits = partial_hash_hits_enabled(
            kv_cache_config.kv_cache_groups,
            self._hash_block_size,
            vllm_config.parallel_config.decode_context_parallel_size,
        )
        mamba_groups = {
            group_id: group.kv_cache_spec
            for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
            if isinstance(group.kv_cache_spec, MambaSpec)
        }
        assert all(
            spec.mamba_cache_mode == "align" for spec in mamba_groups.values()
        ), "MooncakeStoreScheduler requires mamba_cache_mode='align'"
        self._boundary_state_group_ids = frozenset(mamba_groups)

        self._gpu_block_pool: BlockPool | None = None
        self._num_workers = vllm_config.parallel_config.world_size
        self._next_store_job_id = 0
        # store_job_id -> (referenced block ids, ranks yet to report completion)
        self._pinned_saves: dict[int, tuple[list[int], int]] = {}

        # Per-request state
        self.load_specs: dict[str, LoadSpec] = {}  # to be loaded
        self._request_trackers: dict[str, RequestTracker] = {}  # scheduled new requests
        self._unfinished_requests: dict[str, tuple[Request, tuple[list[int], ...]]] = {}
        self._unfinished_request_ids: set[str] = set()
        self._finished_partial_tail_metas: dict[str, ReqMeta] = {}

    def bind_gpu_block_pool(self, gpu_block_pool: BlockPool) -> None:
        self._gpu_block_pool = gpu_block_pool

    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """Check for external KV cache hit.

        Returns ``(None, False)`` when an async lookup is still in flight,
        signaling the scheduler to retry this request on a later step.
        """
        if not self.enable_lookup:
            return 0, False

        # Fine-grained hits may land on a hash boundary inside a block; without
        # partial hits, prefixes shorter than one physical block are skipped.
        align = (
            self._hash_block_size if self.enable_partial_hash_hits else self._block_size
        )
        if request.num_tokens < align:
            return 0, False

        lookup_result = self.client.lookup(
            request.request_id,
            request.num_tokens,
            request.block_hashes,
            non_block=self.lookup_async,
        )
        if lookup_result is None:
            # Lookup not ready yet; scheduler will retry on a later step.
            return None, False
        num_external_hit_tokens = lookup_result.hit_length

        if num_external_hit_tokens < num_computed_tokens:
            need_to_allocate = 0
        else:
            need_to_allocate = num_external_hit_tokens - num_computed_tokens

        logger.debug(
            "Reqid: %s, Total tokens %d, kvpool hit tokens: %d, need to load: %d",
            request.request_id,
            request.num_tokens,
            num_external_hit_tokens,
            need_to_allocate,
        )

        if need_to_allocate <= 0:
            return 0, False

        self.load_specs[request.request_id] = LoadSpec(
            vllm_cached_tokens=num_computed_tokens,
            kvpool_cached_tokens=num_external_hit_tokens,
            can_load=False,
            tail_key_boundaries=lookup_result.tail_key_boundaries,
        )

        return need_to_allocate, self.load_async

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ):
        """Update state after block allocation."""
        local_block_ids: tuple[list[int], ...] = ()
        if num_external_tokens > 0:
            local_block_ids = self.kv_cache_config.select_transfer_block_ids(
                blocks.get_block_ids()
            )

        self._unfinished_requests[request.request_id] = (request, local_block_ids)
        self._unfinished_request_ids.add(request.request_id)

        if request.request_id not in self.load_specs:
            return

        if num_external_tokens == 0:
            self.load_specs[request.request_id].can_load = False
            return

        assert (
            num_external_tokens > 0
            and num_external_tokens
            == self.load_specs[request.request_id].kvpool_cached_tokens
            - self.load_specs[request.request_id].vllm_cached_tokens
        ), (
            f"Mismatch in number of tokens: {num_external_tokens} vs "
            f"{self.load_specs[request.request_id].kvpool_cached_tokens} - "
            f"{self.load_specs[request.request_id].vllm_cached_tokens}"
            f" for request {request.request_id}"
        )

        self.load_specs[request.request_id].can_load = True

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """Build connector metadata for this scheduler step."""
        is_consumer = self.kv_role == "kv_consumer"
        can_process_cached = not is_consumer or self.save_decode_cache

        for finished_req_id in scheduler_output.finished_req_ids:
            self.client.discard(finished_req_id)
            self.load_specs.pop(finished_req_id, None)
            self._request_trackers.pop(finished_req_id, None)
            self._unfinished_requests.pop(finished_req_id, None)
            self._unfinished_request_ids.discard(finished_req_id)

        preempted_ids = scheduler_output.preempted_req_ids or set()
        for req_id in preempted_ids:
            self.load_specs.pop(req_id, None)
            if request_tracker := self._request_trackers.get(req_id):
                request_tracker.reset()
            self._unfinished_requests.pop(req_id, None)

        meta = MooncakeStoreConnectorMetadata(
            self._unfinished_request_ids,
            preempted_ids,
        )

        # Handle new requests
        for request in scheduler_output.scheduled_new_reqs:
            load_spec = self.load_specs.pop(request.req_id, None)
            num_tokens_to_compute = (
                request.num_computed_tokens
                + scheduler_output.num_scheduled_tokens[request.req_id]
            )
            assert request.req_id in self._unfinished_requests
            request_tuple = self._unfinished_requests.get(request.req_id)
            request_real = request_tuple[0]  # type: ignore[index]

            unfolded_block_ids = tuple(
                blocks.copy()
                for blocks in self.kv_cache_config.select_transfer_block_ids(
                    request.block_ids
                )
            )

            prefill_tokens = _new_req_prefill_tokens(request)
            request_tracker = RequestTracker(
                req_id=request.req_id,
                token_len=num_tokens_to_compute,
                allocated_block_ids=unfolded_block_ids,
                num_saved_tokens=0,
                token_ids=(
                    prefill_tokens[:num_tokens_to_compute]
                    if self.enable_kv_events
                    else None
                ),
                prefill_end_tokens=len(prefill_tokens),
            )
            self._request_trackers[request.req_id] = request_tracker

            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self._block_size,
                load_spec=load_spec,
                # A consumer may write decode KV without becoming a prefill
                # producer. Loads are still carried by the same metadata.
                skip_save=is_consumer,
                block_hashes=request_real.block_hashes,
            )
            if req_meta is not None:
                meta.add_request(req_meta)

        # Handle cached (running, or MRV1 resumed-from-preemption) requests
        cached_reqs = scheduler_output.scheduled_cached_reqs
        if can_process_cached:
            for i, req_id in enumerate(cached_reqs.req_ids):
                new_block_ids = cached_reqs.new_block_ids[i]
                if new_block_ids:
                    new_block_ids = self.kv_cache_config.select_transfer_block_ids(
                        new_block_ids
                    )

                req_meta = None
                if req_id in cached_reqs.resumed_req_ids:
                    # Resumed after preemption
                    if not new_block_ids:
                        continue
                    new_block_ids = tuple(b.copy() for b in new_block_ids)
                    load_spec = self.load_specs.pop(req_id, None)
                    request_tuple = self._unfinished_requests.get(req_id)
                    request_real = request_tuple[0]  # type: ignore[index]
                    num_tokens_to_compute = (
                        request_real.num_computed_tokens
                        + scheduler_output.num_scheduled_tokens[req_id]
                    )
                    # On resume, the request re-prefills prompt + previously
                    # generated tokens (all_token_ids).
                    prefill_tokens = list(request_real.all_token_ids)
                    request_tracker = RequestTracker(
                        req_id=req_id,
                        token_len=num_tokens_to_compute,
                        allocated_block_ids=new_block_ids,
                        num_saved_tokens=0,
                        token_ids=(
                            prefill_tokens[:num_tokens_to_compute]
                            if self.enable_kv_events
                            else None
                        ),
                        prefill_end_tokens=len(prefill_tokens),
                    )
                    self._request_trackers[req_id] = request_tracker

                    req_meta = ReqMeta.from_request_tracker(
                        request_tracker,
                        self._block_size,
                        load_spec=load_spec,
                        skip_save=is_consumer,
                        block_hashes=request_real.block_hashes,
                    )
                else:
                    # Decode/chunked request
                    request_tracker = self._request_trackers[req_id]
                    num_computed_token = cached_reqs.num_computed_tokens[i]
                    # Use the tracker's snapshot of the prefill range so resumed
                    # requests keep saving past the original prompt boundary.
                    prefill_end = request_tracker.prefill_end_tokens
                    is_decode = num_computed_token >= prefill_end
                    if is_decode and not self.save_decode_cache:
                        continue

                    num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
                    req_tuple = self._unfinished_requests.get(req_id)
                    if req_tuple:
                        unfinished_req = req_tuple[0]
                        num_current_tokens = request_tracker.token_len
                        new_token_ids = unfinished_req.all_token_ids[
                            num_current_tokens : num_current_tokens + num_new_tokens
                        ]
                        request_tracker.token_len += len(new_token_ids)
                        if request_tracker.token_ids is not None:
                            request_tracker.token_ids.extend(new_token_ids)
                    else:
                        raise ValueError(
                            f"Request {req_id} is not in _unfinished_requests"
                        )
                    # A block is usually allocated before the step that fills
                    # it, so reaching a save boundary does not imply that this
                    # step has new block ids.
                    if new_block_ids:
                        request_tracker.update(new_block_ids)
                    if is_consumer and not is_decode:
                        continue

                    req_meta = ReqMeta.from_request_tracker(
                        request_tracker,
                        self._block_size,
                        load_spec=None,
                        skip_save=False,
                        block_hashes=unfinished_req.block_hashes,
                    )

                if req_meta is not None:
                    meta.add_request(req_meta)

        # Handle requests with pending load specs not yet scheduled
        request_ids = [req.req_id for req in scheduler_output.scheduled_new_reqs]
        for request_id, (
            unfinished_req,
            block_ids,
        ) in self._unfinished_requests.items():
            if request_id not in request_ids and request_id not in cached_reqs.req_ids:
                load_spec = self.load_specs.pop(request_id, None)
                if not load_spec:
                    continue
                num_tokens_to_compute = load_spec.kvpool_cached_tokens
                request_tracker = RequestTracker(
                    req_id=request_id,
                    token_len=num_tokens_to_compute,
                    allocated_block_ids=block_ids,
                    num_saved_tokens=0,
                )
                self._request_trackers[request_id] = request_tracker
                req_meta = ReqMeta.from_request_tracker(
                    request_tracker,
                    self._block_size,
                    load_spec=load_spec,
                    skip_save=None,
                    block_hashes=unfinished_req.block_hashes,
                )
                if req_meta is not None:
                    meta.add_request(req_meta)

        block_state = getattr(scheduler_output, "kv_connector_block_state", None)
        if (
            block_state is not None
            and block_state.boundary_state_offloads
            and not is_consumer
        ):
            self._handle_boundary_state_offloads(
                block_state.boundary_state_offloads, meta
            )

        self._apply_current_save_block_ids(meta, scheduler_output)

        # Finish-time handoffs arrive after the producing step's metadata was
        # built. Their exact blocks were pinned when they were registered, so
        # they remain valid after request cleanup and need no current snapshot.
        for req_meta in self._finished_partial_tail_metas.values():
            meta.add_request(req_meta)
        self._finished_partial_tail_metas.clear()

        self._reference_save_blocks(meta)
        return meta

    def _apply_current_save_block_ids(
        self,
        meta: MooncakeStoreConnectorMetadata,
        scheduler_output: SchedulerOutput,
    ) -> None:
        """Replace append-only mirrors with the core's current block tables."""
        save_metas = [req_meta for req_meta in meta.requests if req_meta.can_save]
        if not save_metas:
            return

        block_state = scheduler_output.kv_connector_block_state
        assert block_state is not None, (
            "Current block tables are required for Mooncake store jobs"
        )
        for req_meta in save_metas:
            block_ids = block_state.block_ids.get(req_meta.req_id)
            assert block_ids is not None, (
                f"Missing current block table for store request {req_meta.req_id}"
            )
            req_meta.block_ids = block_ids

    def _reference_save_blocks(self, meta: MooncakeStoreConnectorMetadata) -> None:
        """Take a GPU block reference for every store job this step emits.

        The worker DMAs out of these blocks after the step that scheduled them,
        so a reference keeps them out of the free queue even once the request
        itself is freed, until every rank reports the job done.
        """
        pool = self._gpu_block_pool
        for req_meta in meta.requests:
            if not req_meta.can_save:
                continue
            assert pool is not None, (
                "GPU block pool must be bound before any store job is emitted"
            )
            if req_meta.store_job_id is not None:
                assert req_meta.store_job_id in self._pinned_saves
                continue
            req_meta.store_job_id = store_job_id = self._next_store_job_id
            self._next_store_job_id += 1
            block_ids: list[int] = []
            if req_meta.boundary_state_offloads:
                block_ids.extend(
                    block_id for _, block_id, _ in req_meta.boundary_state_offloads
                )
            assert NULL_BLOCK_ID not in block_ids, (
                "A null block cannot back a boundary-state offload"
            )
            # Every allocated block is referenced, not just the ones covering
            # this job's token range: a rank resumes from its own last
            # successful offset, which lags the scheduler's whenever a save was
            # skipped or failed, so it may read anywhere below the range.
            block_ids.extend(
                block_id
                for group_id, group in enumerate(req_meta.block_ids)
                if group_id not in self._boundary_state_group_ids
                for block_id in group
                if block_id != NULL_BLOCK_ID
            )
            # An aligned boundary block may also be present in the request's
            # block table. Take and release exactly one reference per block.
            block_ids = list(dict.fromkeys(block_ids))
            assert NULL_BLOCK_ID not in block_ids
            if not block_ids:
                continue
            self._pinned_saves[store_job_id] = (block_ids, self._num_workers)
            pool.touch([pool.blocks[block_id] for block_id in block_ids])

    def register_finished_partial_tail(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
        partial_tail_offloads: list[tuple[int, int, int]],
    ) -> bool:
        """Queue and pin a finish-time tail for the next connector step."""
        if self.kv_role == "kv_consumer" or not partial_tail_offloads:
            return False
        tracker = self._request_trackers.get(request.request_id)
        if tracker is None or not any(block_ids):
            return False
        boundaries = {boundary for _, _, boundary in partial_tail_offloads}
        if len(boundaries) != 1:
            raise ValueError(
                "Partial-tail offloads for one request must share a boundary"
            )
        boundary_tokens = next(iter(boundaries))
        if boundary_tokens > tracker.prefill_end_tokens:
            return False

        pinned_block_ids: list[int] = []
        for group_id, block_id, _ in partial_tail_offloads:
            if group_id not in self._boundary_state_group_ids:
                return False
            if block_id == NULL_BLOCK_ID:
                return False
            pinned_block_ids.append(block_id)
        pinned_block_ids = list(dict.fromkeys(pinned_block_ids))

        pool = self._gpu_block_pool
        assert pool is not None, (
            "GPU block pool must be bound before a finish-time handoff"
        )
        assert request.request_id not in self._finished_partial_tail_metas
        store_job_id = self._next_store_job_id
        self._next_store_job_id += 1
        self._pinned_saves[store_job_id] = (pinned_block_ids, self._num_workers)
        pool.touch([pool.blocks[block_id] for block_id in pinned_block_ids])

        self._finished_partial_tail_metas[request.request_id] = ReqMeta(
            req_id=request.request_id,
            token_len_chunk=0,
            block_ids=tuple(group.copy() for group in block_ids),
            block_hashes=list(request.block_hashes),
            can_save=True,
            num_prompt_tokens=tracker.prefill_end_tokens,
            store_job_id=store_job_id,
            boundary_state_offloads=partial_tail_offloads,
        )
        tracker.has_pending_offload = True
        # The store job owns exact block refs, so request cleanup need not wait.
        return False

    def _handle_boundary_state_offloads(
        self,
        offloads: dict[str, list[tuple[int, int, int]]],
        meta: MooncakeStoreConnectorMetadata,
    ) -> None:
        """Attach exact boundary-state blocks to store jobs for this step.

        Flushed in the step they arrive: the CoW copy is enqueued before the
        connector event records, so this step's event fences the exact block.
        Entries ride the request's save meta when present, else an
        offload-only ReqMeta (``token_len_chunk=0`` skips the normal save,
        ``can_save=True`` takes the normal enqueue and store-job pinning path).
        """
        save_metas = {m.req_id: m for m in meta.requests if m.can_save}
        for req_id, entries in offloads.items():
            tracker = self._request_trackers.get(req_id)
            req_tuple = self._unfinished_requests.get(req_id)
            if tracker is None or req_tuple is None:
                # Request finished/preempted within this step; its blocks are
                # going away, so the offload is conservatively dropped.
                logger.debug("Dropping boundary-state offload for request %s", req_id)
                continue
            accepted: list[tuple[int, int, int]] = []
            for group_id, block_id, boundary_tokens in entries:
                # Every other group stops saving at the end of this prefill, so
                # a mamba-only key past it can never complete a joint hybrid
                # hit. `prefill_end_tokens` — not the original prompt length —
                # is the boundary: a resumed request re-prefills and re-saves
                # its previously generated tokens for every group.
                if boundary_tokens > tracker.prefill_end_tokens:
                    continue
                if block_id == NULL_BLOCK_ID:
                    continue
                if group_id not in self._boundary_state_group_ids:
                    continue
                accepted.append((group_id, block_id, boundary_tokens))
            if not accepted:
                continue
            tracker.has_pending_offload = True
            if (req_meta := save_metas.get(req_id)) is not None:
                req_meta.boundary_state_offloads = accepted
                continue
            meta.add_request(
                ReqMeta(
                    req_id=req_id,
                    token_len_chunk=0,
                    block_ids=tracker.allocated_block_ids,
                    block_hashes=req_tuple[0].block_hashes,
                    can_save=True,
                    num_prompt_tokens=tracker.prefill_end_tokens,
                    boundary_state_offloads=accepted,
                )
            )

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        """Drop the block references of store jobs every rank has finished."""
        meta = connector_output.kv_connector_worker_meta
        if not isinstance(meta, MooncakeStoreWorkerMetadata):
            return
        pool = self._gpu_block_pool
        assert pool is not None
        for store_job_id, count in meta.completed_saves.items():
            pinned = self._pinned_saves.get(store_job_id)
            if pinned is None:
                # The job referenced no blocks, so nothing was recorded for it.
                continue
            block_ids, remaining = pinned
            remaining -= count
            if remaining > 0:
                self._pinned_saves[store_job_id] = (block_ids, remaining)
                continue
            assert remaining == 0, (
                f"store job {store_job_id} reported by too many ranks"
            )
            del self._pinned_saves[store_job_id]
            # Tail-first, as elsewhere, so the shared prefix is evicted last.
            pool.free_blocks(pool.blocks[bid] for bid in reversed(block_ids))

    def has_pending_push_work(self) -> bool:
        """Keep the engine stepping while any store job still holds block refs.

        Completions only reach the scheduler as worker metadata on a step, so an
        engine that quiesced with jobs in flight would leave those references
        held indefinitely. Nothing else keeps it alive now that a finishing
        request no longer defers its own free.
        """
        return bool(self._pinned_saves)

    def reset_store(self) -> bool:
        """Trigger a global ``remove_all(force=True)`` on the Mooncake master.

        Routes through the existing LookupKey ZMQ admin channel to worker
        rank 0, which owns the ``MooncakeDistributedStore`` handle.

        Ordering assumption: caller (typically
        ``Scheduler.reset_connector_cache``, invoked via
        ``reset_prefix_cache(reset_connector=True)``) MUST ensure no
        in-flight Mooncake lookups or transfers. For RL workflows this is
        satisfied at the step boundary after weight updates and rollout
        drain. Violating this can allow stale KV to be served on the next
        request, defeating the hard-reset guarantee.

        Returns True on ACK from worker, False on NACK or RPC error.
        """
        try:
            ok = self.client.reset()
            if ok:
                logger.info("Mooncake store reset via remove_all succeeded.")
            else:
                logger.warning("Mooncake store reset returned NACK from worker.")
            return ok
        except Exception as e:
            logger.error("Mooncake reset_store RPC failed: %s", e)
            return False
