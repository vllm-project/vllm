# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Push-specific scheduler-side logic for the NIXL connector.

In push mode, scheduler-side responsibilities are:

* D side (decode): on ``update_state_after_alloc``, stash registration data
  (D's identity + locally allocated block IDs) into
  ``_push_pending_registrations``. The D worker drains it from
  ``meta.push_registrations`` next step and sends a NIXL notification to the
  P worker (no scheduler-level networking).
* P side (prefill): on ``request_finished``, stash the finished block IDs
  into ``_finished_request_blocks`` for the lease, and into
  ``_newly_finished_push_blocks`` so the P worker picks them up via
  ``meta.push_finished_blocks`` and matches against any D registrations
  it already received via NIXL notifications.
* Both sides: ``has_pending_push_work`` keeps the engine main loop stepping
  while pushes are in flight. ``update_connector_output`` cleans up
  ``_finished_request_blocks`` once the WRITE completes.

A soft per-registration watchdog on the D scheduler fails requests that have
been registered but not fulfilled within a configurable timeout.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from vllm import envs
from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler import (
    NixlBaseConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlConnectorMetadata,
    ReqId,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)


class NixlPushConnectorScheduler(NixlBaseConnectorScheduler):
    """Push-specific scheduler logic (WRITE-based KV transfer).

    All P2P communication is deferred to the worker level via NIXL
    notifications. The scheduler communicates with workers only through
    the standard ``build_connector_meta`` / ``update_connector_output``
    hooks.
    """

    _TRANSFER_MODE: str = "push"

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_id: str,
        kv_cache_config: KVCacheConfig,
    ):
        super().__init__(vllm_config, engine_id, kv_cache_config)
        if self.is_bidirectional_kv_xfer_enabled:
            raise NotImplementedError(
                "Bidirectional KV transfer is not supported for NIXL push connector."
            )

        # D-side: registration data to pass to D workers via metadata on
        # the next ``build_connector_meta`` call.
        self._push_pending_registrations: dict[ReqId, dict[str, Any]] = {}

        # D-side: track the wall-clock deadline for each registered request
        # to detect "registered but never fulfilled" failures (e.g. the P
        # node disappeared after registration). Keyed by D request_id.
        self._push_registration_deadlines: dict[ReqId, float] = {}

        # P-side: block IDs for finished requests, kept for the lease and
        # used to drive ``has_pending_push_work``.
        self._finished_request_blocks: dict[ReqId, BlockIds] = {}
        # P-side: newly finished blocks to ship to P workers on next step.
        self._newly_finished_push_blocks: dict[ReqId, BlockIds] = {}

        # Layer-wise overlapped WRITE-push (opt-in). P-side: producer
        # requests whose per-layer KV should be pushed during the forward.
        # ``_lw_need_save`` is seeded at alloc; multi-chunk prefills
        # accumulate their block ids in ``_lw_pending_save`` until the last
        # chunk, at which point the full block list is emitted into
        # ``reqs_to_save`` for the worker's ``save_kv_layer`` hook.
        self._layerwise = envs.VLLM_NIXL_LAYERWISE_PUSH
        self._lw_need_save: dict[ReqId, tuple[Request, BlockIds]] = {}
        self._lw_pending_save: dict[ReqId, tuple[Request, BlockIds]] = {}
        self._lw_emitted: set[ReqId] = set()

        # Soft watchdog timeout (seconds) for D-side registrations that
        # never receive a push completion. Defaults to the existing
        # decoder KV blocks TTL so behaviour matches the lease.
        assert vllm_config.kv_transfer_config is not None
        self._push_registration_timeout: float = float(
            vllm_config.kv_transfer_config.get_from_extra_config(
                "push_registration_timeout",
                self.decoder_kv_blocks_ttl,
            )
        )

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int, bool]:
        """In push mode, D doesn't pull — it registers blocks and waits.

        However, we still need to handle the do_remote_prefill case where D
        needs to know how many tokens will be pushed.
        """
        params = request.kv_transfer_params
        logger.debug(
            "NixlPushConnector get_num_new_matched_tokens: "
            "num_computed_tokens=%s, kv_transfer_params=%s",
            num_computed_tokens,
            params,
        )

        if params is not None and params.get("do_remote_prefill"):
            token_ids = request.prompt_token_ids or []
            actual = self._get_remote_prefill_token_count(len(token_ids))
            count = actual - num_computed_tokens
            if count > 0:
                return count, True

        return 0, False

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
        """In push mode, D stores registration data for the worker to send
        to P via NIXL notification (deferred to ``build_connector_meta``).
        """
        params = request.kv_transfer_params
        logger.debug(
            "NixlPushConnector update_state_after_alloc: "
            "num_external_tokens=%s, kv_transfer_params=%s",
            num_external_tokens,
            params,
        )

        if not params:
            return

        # P side: track the request as in-batch so the lease accounting
        # matches what the worker expects on the next step.
        if params.get("do_remote_decode"):
            self._reqs_in_batch.add(request.request_id)
            # Layer-wise push (P side): seed the producer request so its
            # per-layer KV can be pushed during the forward. Block ids are
            # finalized (across chunked prefill) in build_connector_meta.
            if self._layerwise:
                self._lw_need_save[request.request_id] = (
                    request,
                    blocks.get_block_ids(),
                )

        # P side with host-buffer offload: defer save to the worker.
        if self.use_host_buffer and params.get("do_remote_decode"):
            self._reqs_need_save[request.request_id] = request
            return

        # D side: only act on the first call (``do_remote_prefill`` is
        # unset on re-entry by the marker below).
        if not params.get("do_remote_prefill"):
            return

        if num_external_tokens <= 0:
            # Nothing to receive: full prefix-cache hit on D (or the scheduler
            # allocated no external tokens this pass), so there is no
            # registration to stage. Flip ``do_remote_prefill`` off so the
            # terminal ``request_finished`` does NOT take the abort branch and
            # enqueue an unseeded recv (which used to KeyError the engine).
            logger.debug(
                "NixlPushConnector D alloc: req %s num_external_tokens=%d "
                "<=0 (no remote recv staged; prefix-hit/defer)",
                request.request_id,
                num_external_tokens,
            )
            params["do_remote_prefill"] = False
            return

        # First-pass D path: stash registration data the worker will
        # ship to P on the next ``build_connector_meta`` cycle.
        logger.debug(
            "KV PUSH mode: D node storing registration for request %s",
            request.request_id,
        )
        local_block_ids: BlockIds = blocks.get_unhashed_block_ids_all_groups()
        local_block_ids = self.get_exchange_clipped_blocks(local_block_ids)

        # ``remote_*`` fields are P's coordinates (from D's perspective).
        # ``decode_*`` fields are D's own info that P needs for the
        # reverse handshake before WRITE-ing.
        self._push_pending_registrations[request.request_id] = {
            "request_id": request.request_id,
            "decode_engine_id": self.engine_id,
            "decode_host": self.side_channel_host,
            "decode_port": self.side_channel_port,
            "decode_tp_size": (self.vllm_config.parallel_config.tensor_parallel_size),
            "local_block_ids": local_block_ids,
            "remote_engine_id": params["remote_engine_id"],
            "remote_host": params["remote_host"],
            "remote_port": params["remote_port"],
            "remote_tp_size": params["tp_size"],
            "remote_pp_size": params.get("pp_size", 1),
        }
        self._push_registration_deadlines[request.request_id] = (
            time.perf_counter() + self._push_registration_timeout
        )
        # In push mode D doesn't know P's blocks; P determines them
        # from the registration. We still track the request as
        # needing recv so the engine waits for P's WRITE completion.
        # ``remote_block_ids`` is also seeded to an empty tuple so the
        # base scheduler's ``add_new_req_to_recv`` can build the
        # ReqMeta without a KeyError — the actual remote block IDs are
        # learned by P over the NIXL handshake at WRITE time.
        params["remote_block_ids"] = ()
        # main uses a 4-tuple (request, block_ids, cached, awaiting_kvs);
        # push mode never awaits a pull (P drives the WRITE), so awaiting_kvs
        # is False, matching base_scheduler's ``add_new_req_to_recv`` unpack.
        self._reqs_need_recv[request.request_id] = (
            request,
            local_block_ids,
            (),
            False,
        )
        logger.debug(
            "NixlPushConnector D alloc: req %s staged recv "
            "num_external_tokens=%d local_blocks=%d remote_eng=%s",
            request.request_id,
            num_external_tokens,
            sum(len(g) for g in local_block_ids) if local_block_ids else 0,
            params.get("remote_engine_id"),
        )

        # Mark as processed so a re-entry (e.g. preemption + reschedule)
        # doesn't re-stage the registration.
        params["do_remote_prefill"] = False

    def request_finished(
        self,
        request: Request,
        block_ids: BlockIds,
    ) -> tuple[bool, dict[str, Any] | None]:
        """Push-mode request_finished: stores blocks for workers."""
        from vllm.v1.request import RequestStatus

        params = request.kv_transfer_params
        logger.debug(
            "NixlPushConnector request_finished(%s), request_status=%s, "
            "kv_transfer_params=%s",
            request.request_id,
            request.status,
            params,
        )
        if not params:
            return False, None

        is_p_node = bool(params.get("do_remote_decode"))

        self._stop_heartbeat(request.request_id)
        # Drop any pending registration deadline; the request either
        # completed or was cancelled.
        self._push_registration_deadlines.pop(request.request_id, None)

        if params.get("do_remote_prefill"):
            # ``do_remote_prefill`` is still set, which means
            # ``update_state_after_alloc`` never ran (it would have
            # flipped this flag to False). The request was aborted
            # before it could be scheduled — e.g. rejected at the D
            # serving layer via abort_immediately. To keep P from
            # stranding the prefill blocks, we still register an empty
            # recv so the worker emits a notif that lets P free them.
            # Seed remote_block_ids so add_new_req_to_recv won't KeyError.
            params["remote_block_ids"] = ()
            self._reqs_need_recv[request.request_id] = (request, [], (), False)
            params["do_remote_prefill"] = False
            return False, None

        # Push connector only acts on the P-side terminal path; D-side
        # finishing without a remote prefill is a no-op.
        if not is_p_node:
            return False, None

        if request.status not in (
            RequestStatus.FINISHED_LENGTH_CAPPED,
            RequestStatus.FINISHED_STOPPED,
        ):
            self._reqs_not_processed.add(request.request_id)
            self._reqs_need_save.pop(request.request_id, None)
            return False, None

        delay_free_blocks = any(len(group) > 0 for group in block_ids)
        remote_num_tokens = 0
        if delay_free_blocks:
            logger.debug(
                "NixlPushConnector request_finished(%s) waiting for %d seconds "
                "before releasing blocks",
                request.request_id,
                self._kv_lease_duration,
            )
            self._reqs_need_send[request.request_id] = (
                time.perf_counter() + self._kv_lease_duration
            )

            block_ids = self.get_exchange_clipped_blocks(block_ids)
            remote_num_tokens = request.num_computed_tokens

            # Store finished blocks for worker-level matching with D
            # registrations (via NIXL notifications). Keep the request in
            # ``_finished_request_blocks`` so ``has_pending_push_work`` holds
            # the engine loop open until the WRITE completes.
            self._finished_request_blocks[request.request_id] = block_ids
            # In layer-wise mode the per-layer WRITEs were already issued from
            # the forward hook; do NOT trigger the monolithic worker WRITE.
            if not self._layerwise:
                self._newly_finished_push_blocks[request.request_id] = block_ids
            else:
                self._lw_need_save.pop(request.request_id, None)
                self._lw_pending_save.pop(request.request_id, None)
                self._lw_emitted.discard(request.request_id)

        return delay_free_blocks, dict(
            do_remote_prefill=True,
            do_remote_decode=False,
            remote_block_ids=block_ids,
            remote_engine_id=self.engine_id,
            remote_request_id=request.request_id,
            remote_host=self.side_channel_host,
            remote_port=self.side_channel_port,
            tp_size=self.transfer_tp_size,
            dcp_size=self.vllm_config.parallel_config.decode_context_parallel_size,
            pp_size=self.vllm_config.parallel_config.pipeline_parallel_size,
            remote_num_tokens=remote_num_tokens,
            transfer_mode=self._TRANSFER_MODE,
        )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        # Defensive seed: the base scheduler turns every ``_reqs_need_recv``
        # entry into a recv ReqMeta via ``add_new_req_to_recv``, which hard
        # indexes ``remote_block_ids`` and the other ``remote_*`` keys
        # (metadata.py). In push mode D legitimately has no ``remote_block_ids``
        # (P assigns physical blocks at WRITE time) and, critically, the
        # abort/rejection path (``request_finished`` with ``do_remote_prefill``
        # still set, e.g. a request that got an ErrorResponse at the D serving
        # layer and never ran ``update_state_after_alloc``) enqueues a recv
        # WITHOUT seeding these keys -> ``KeyError: 'remote_block_ids'`` kills
        # the whole EngineCore. Seed safe defaults so a stale/aborted recv is
        # reaped later by the registration watchdog + KV lease instead of
        # crashing the engine. Normal registrations already carry every key
        # (setdefault is a no-op for them).
        for req_id, (req, _, _, _) in self._reqs_need_recv.items():
            params = req.kv_transfer_params
            if params is None:
                continue
            if "remote_block_ids" not in params:
                logger.debug(
                    "NixlPushConnector: seeding missing remote_* for recv "
                    "req %s (abort/reject path; do_remote_prefill=%s)",
                    req_id,
                    params.get("do_remote_prefill"),
                )
            params.setdefault("remote_block_ids", ())
            params.setdefault("remote_engine_id", "")
            params.setdefault("remote_request_id", req_id)
            params.setdefault("remote_host", "")
            params.setdefault("remote_port", 0)

        meta = super().build_connector_meta(scheduler_output)
        assert isinstance(meta, NixlConnectorMetadata)

        # Layer-wise push (P side): identify producer requests whose prefill
        # completes this step and emit their FULL block list into
        # ``reqs_to_save`` so the worker's ``save_kv_layer`` hook can push
        # each layer's KV during this final forward. Chunked prefills
        # accumulate ``new_block_ids`` across steps until the last chunk.
        if self._layerwise:
            self._lw_build_reqs_to_save(scheduler_output, meta)

        # Watchdog: any D-side registration whose deadline has passed without
        # a corresponding push completion is treated as failed and cleaned up.
        # The corresponding request is already tracked via _reqs_need_recv;
        # the engine layer will eventually time it out via the lease, but we
        # at least drop the stale registration so we don't keep retrying.
        now = time.perf_counter()
        # Deadlines are inserted in non-decreasing order (monotonic clock +
        # constant timeout, armed once per request), and dict insertion order
        # is preserved across key deletions, so we can stop at the first
        # not-yet-expired entry instead of scanning the whole dict.
        expired = []
        for rid, deadline in self._push_registration_deadlines.items():
            if deadline > now:
                break
            expired.append(rid)
        for rid in expired:
            self._push_registration_deadlines.pop(rid, None)
            # Avoid resending a registration that already timed out.
            self._push_pending_registrations.pop(rid, None)
            logger.warning(
                "NixlPushConnector: registration for request %s timed out "
                "after %.1fs without a push completion",
                rid,
                self._push_registration_timeout,
            )

        # D side: package pending registrations for D workers to send out.
        if self._push_pending_registrations:
            meta.push_registrations = dict(self._push_pending_registrations)
            self._push_pending_registrations.clear()

        # P side: package newly finished blocks for P workers to match against
        # any D registrations they have received via NIXL notifications.
        if self._newly_finished_push_blocks:
            meta.push_finished_blocks = dict(self._newly_finished_push_blocks)
            self._newly_finished_push_blocks.clear()

        return meta

    def _lw_tokens(self, grouped: BlockIds) -> int:
        if not grouped:
            return 0
        return max((len(g) for g in grouped), default=0) * self.block_size

    @staticmethod
    def _lw_concat(a: BlockIds, b: BlockIds) -> BlockIds:
        """Concatenate two grouped block-id lists group-by-group."""
        if not a:
            return tuple(list(g) for g in b)
        return tuple(
            (list(a[i]) if i < len(a) else []) + (list(b[i]) if i < len(b) else [])
            for i in range(max(len(a), len(b)))
        )

    def _lw_emit(
        self, req_id: ReqId, req: Request, blocks: BlockIds, meta: NixlConnectorMetadata
    ) -> None:
        clipped = self.get_exchange_clipped_blocks(tuple(list(g) for g in blocks))
        meta.add_new_req_to_save(req_id, clipped, req.kv_transfer_params or {})
        self._lw_emitted.add(req_id)
        logger.debug(
            "NIXL lw[P-sched] emit reqs_to_save req=%s blocks=%d",
            req_id,
            self._lw_tokens(clipped) // max(self.block_size, 1),
        )

    def _lw_build_reqs_to_save(
        self, scheduler_output: SchedulerOutput, meta: NixlConnectorMetadata
    ) -> None:
        # 1. Seeded producers: emit immediately if prefill fits in one chunk,
        # otherwise start accumulating across chunks.
        for req_id, (req, blocks) in list(self._lw_need_save.items()):
            del self._lw_need_save[req_id]
            if req_id in self._lw_emitted:
                continue
            if self._lw_tokens(blocks) >= req.num_prompt_tokens:
                self._lw_emit(req_id, req, blocks, meta)
            else:
                self._lw_pending_save[req_id] = (req, blocks)

        # 2. Chunked continuations: accumulate new block ids until the last
        # chunk, then emit the full block list.
        cached_reqs = scheduler_output.scheduled_cached_reqs
        for req_id, new_blocks in zip(
            cached_reqs.req_ids, cached_reqs.new_block_ids, strict=True
        ):
            if req_id not in self._lw_pending_save:
                continue
            req, acc = self._lw_pending_save[req_id]
            if new_blocks is not None:
                acc = self._lw_concat(acc, new_blocks)
                self._lw_pending_save[req_id] = (req, acc)
            if self._lw_tokens(acc) >= req.num_prompt_tokens:
                self._lw_emit(req_id, req, acc, meta)
                del self._lw_pending_save[req_id]

    def has_pending_push_work(self) -> bool:
        # Keep the engine main loop alive while we have:
        # - finished P blocks awaiting WRITE completion, or
        # - pending D registrations the worker has not yet shipped, or
        # - newly finished blocks not yet shipped to P workers, or
        # - layer-wise producer requests mid-prefill awaiting their push.
        return bool(
            self._finished_request_blocks
            or self._push_pending_registrations
            or self._lw_need_save
            or self._lw_pending_save
        )

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        """Clean up finished request blocks after push completes."""
        super().update_connector_output(connector_output)
        for req_id in connector_output.finished_sending or ():
            self._finished_request_blocks.pop(req_id, None)
        # On D side, finished_recving means the push completed; clear the
        # watchdog so we don't trip an expiration on a fulfilled request.
        for req_id in connector_output.finished_recving or ():
            self._push_registration_deadlines.pop(req_id, None)
