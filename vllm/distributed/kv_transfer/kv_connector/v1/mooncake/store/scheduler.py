# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Adapted from vllm-project/vllm-ascend
# (vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/).
"""Scheduler-side logic for MooncakeStoreConnector."""

from typing import Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (  # noqa: E501
    LoadSpec,
    MooncakeStoreConnectorMetadata,
    ReqMeta,
    RequestTracker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker import (  # noqa: E501
    LookupKeyClient,
)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request

logger = init_logger(__name__)


def _session_id_from_request(request: Request) -> str | None:
    """Read the session id from ``sampling_params.extra_args``; None if unset."""
    sampling_params = getattr(request, "sampling_params", None)
    extra_args = getattr(sampling_params, "extra_args", None)
    if not extra_args:
        return None
    session_id = extra_args.get("session_id")
    return session_id if isinstance(session_id, str) else None


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
        self.client = LookupKeyClient(vllm_config)

        # Align with the engine's own scheduler_block_size and hash_block_size.
        self._block_size, self._hash_block_size = resolve_kv_cache_block_sizes(
            kv_cache_config, vllm_config
        )

        # Per-request state
        self.load_specs: dict[str, LoadSpec] = {}  # to be loaded
        self._request_trackers: dict[str, RequestTracker] = {}  # scheduled new requests
        self._unfinished_requests: dict[str, tuple[Request, tuple[list[int], ...]]] = {}
        self._unfinished_request_ids: set[str] = set()

    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """Check for external KV cache hit.

        Returns ``(None, False)`` when an async lookup is still in flight,
        signaling the scheduler to retry this request on a later step.
        """
        # Look up against the full prefill range, not just the prompt.
        token_len = request.num_tokens // self._block_size * self._block_size
        if token_len < self._block_size:
            return 0, False

        num_external_hit_tokens = self.client.lookup(
            request.request_id,
            token_len,
            request.block_hashes,
            non_block=self.lookup_async,
        )
        if num_external_hit_tokens is None:
            # Lookup not ready yet; scheduler will retry on a later step.
            return None, False

        if num_external_hit_tokens == request.num_tokens:
            # Leave a sub-block tail uncomputed for sampling, on a block
            # boundary so the recv-side load mask covers every yielded chunk.
            num_external_hit_tokens = max(
                0,
                (request.num_tokens - 1) // self._block_size * self._block_size,
            )

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
            local_block_ids = blocks.get_block_ids()

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
        force_skip_save = self.kv_role == "kv_consumer"

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

            if isinstance(request.block_ids, tuple):
                # Multi-group: preserve per-group structure.
                unfolded_block_ids = tuple(b.copy() for b in request.block_ids)
            else:
                # Single-group legacy: list[int] -> 1-tuple.
                unfolded_block_ids = (request.block_ids.copy(),)

            prefill_tokens = _new_req_prefill_tokens(request)
            request_tracker = RequestTracker(
                req_id=request.req_id,
                token_len=num_tokens_to_compute,
                allocated_block_ids=unfolded_block_ids,
                num_saved_tokens=0,
                token_ids=prefill_tokens[:num_tokens_to_compute],
                prefill_end_tokens=len(prefill_tokens),
                session_id=_session_id_from_request(request_real),
            )
            self._request_trackers[request.req_id] = request_tracker

            last_chunk_tokens_num = (
                len(prefill_tokens) // self._block_size * self._block_size
            )

            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self._block_size,
                load_spec=load_spec,
                skip_save=force_skip_save,
                block_hashes=request_real.block_hashes,
                is_last_chunk=(request_tracker.token_len >= last_chunk_tokens_num),
            )
            if req_meta is not None:
                meta.add_request(req_meta)

        # Handle cached (running, or MRV1 resumed-from-preemption) requests
        cached_reqs = scheduler_output.scheduled_cached_reqs
        if not force_skip_save:
            for i, req_id in enumerate(cached_reqs.req_ids):
                new_block_ids = cached_reqs.new_block_ids[i]
                if not new_block_ids:
                    continue

                req_meta = None
                if req_id in cached_reqs.resumed_req_ids:
                    # Resumed after preemption
                    if isinstance(new_block_ids, tuple):
                        new_block_ids = tuple(b.copy() for b in new_block_ids)
                    else:
                        new_block_ids = (new_block_ids.copy(),)
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
                        token_ids=prefill_tokens[:num_tokens_to_compute].copy(),
                        prefill_end_tokens=len(prefill_tokens),
                        session_id=_session_id_from_request(request_real),
                    )
                    self._request_trackers[req_id] = request_tracker

                    last_chunk_tokens_num = (
                        len(prefill_tokens) // self._block_size * self._block_size
                    )
                    req_meta = ReqMeta.from_request_tracker(
                        request_tracker,
                        self._block_size,
                        load_spec=load_spec,
                        skip_save=force_skip_save,
                        block_hashes=request_real.block_hashes,
                        is_last_chunk=(
                            request_tracker.token_len >= last_chunk_tokens_num
                        ),
                    )
                else:
                    # Decode/chunked request
                    request_tracker = self._request_trackers[req_id]
                    num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
                    req_tuple = self._unfinished_requests.get(req_id)
                    if req_tuple:
                        unfinished_req = req_tuple[0]
                        num_current_tokens = request_tracker.token_len
                        new_token_ids = unfinished_req.all_token_ids[
                            num_current_tokens : num_current_tokens + num_new_tokens
                        ]
                        request_tracker.token_len += len(new_token_ids)
                    else:
                        raise ValueError(
                            f"Request {req_id} is not in _unfinished_requests"
                        )
                    num_computed_token = cached_reqs.num_computed_tokens[i]
                    # Use the tracker's snapshot of the prefill range so resumed
                    # requests keep saving past the original prompt boundary.
                    prefill_end = request_tracker.prefill_end_tokens
                    if num_computed_token >= prefill_end:
                        continue
                    request_tracker.update(new_block_ids)

                    last_chunk_tokens_num = (
                        prefill_end // self._block_size * self._block_size
                    )
                    req_meta = ReqMeta.from_request_tracker(
                        request_tracker,
                        self._block_size,
                        load_spec=None,
                        skip_save=force_skip_save,
                        block_hashes=unfinished_req.block_hashes,
                        is_last_chunk=(
                            request_tracker.token_len >= last_chunk_tokens_num
                        ),
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
                    session_id=_session_id_from_request(unfinished_req),
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

        return meta

    def request_finished(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Determine whether to delay freeing blocks for async save."""
        if self.kv_role == "kv_consumer":
            return False, None
        tracker = self._request_trackers.get(request.request_id)
        # Missing tracker can happen when the request is aborted before the
        # connector observes the normal finished lifecycle or is preempted
        # before finishing.
        if tracker is None or tracker.num_saved_tokens <= 0:
            return False, None
        total_blocks = sum(len(g) for g in block_ids)
        delay_free_blocks = total_blocks > 0
        if delay_free_blocks:
            logger.debug(
                "Delaying free of %d blocks for request %s",
                total_blocks,
                request.request_id,
            )
        return delay_free_blocks, None

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
