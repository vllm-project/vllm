# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Push-specific scheduler-side logic for the NIXL connector.

In push mode, scheduler-side responsibilities are:

* D side (decode): on `update_state_after_alloc`, stash registration data
  (D's identity + locally allocated block IDs) into
  `_push_pending_registrations`. The D worker drains it from
  `meta.push_registrations` next step and sends a NIXL notification to the
  P worker (no scheduler-level networking).
* P side (prefill): on `request_finished`, stash the finished block IDs
  into `_finished_request_blocks` for the lease, and into
  `_newly_finished_push_blocks` so the P worker picks them up via
  `meta.push_finished_blocks` and matches against any D registrations
  it already received via NIXL notifications.
* Both sides: `has_pending_push_work` keeps the engine main loop stepping
  while pushes are in flight. `update_connector_output` cleans up
  `_finished_request_blocks` once the WRITE completes.

A soft per-registration watchdog on the D scheduler fails requests that have
been registered but not fulfilled within a configurable timeout.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler import (
    NixlBaseConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlConnectorMetadata,
    NixlPushChunkJob,
    ReqId,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import is_full_attention_spec

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class _ChunkPushTracker:
    request: Request
    next_block: int = 0
    final_job_enqueued: bool = False
    fallback: bool = False


class NixlPushConnectorScheduler(NixlBaseConnectorScheduler):
    """Push-specific scheduler logic (WRITE-based KV transfer).

    All P2P communication is deferred to the worker level via NIXL
    notifications. The scheduler communicates with workers only through
    the standard `build_connector_meta` / `update_connector_output`
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

        extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
        self._chunked_push = bool(extra_config.get("enable_chunked_push", False))
        self._chunk_trackers: dict[ReqId, _ChunkPushTracker] = {}
        self._chunk_fallbacks: set[ReqId] = set()
        self._chunk_cancellations: set[ReqId] = set()
        self._pinned_chunk_blocks: dict[ReqId, set[int]] = {}
        self._gpu_block_pool: BlockPool | None = None

        if self._chunked_push:
            transfer_groups = kv_cache_config.transfer_groups
            if vllm_config.kv_transfer_config.kv_role != "kv_producer":
                self._chunked_push = False
            elif self.use_host_buffer:
                raise ValueError("NIXL chunked push requires a device KV buffer")
            elif vllm_config.parallel_config.pipeline_parallel_size != 1:
                raise ValueError("NIXL chunked push currently requires PP=1")
            elif (
                vllm_config.parallel_config.prefill_context_parallel_size != 1
                or vllm_config.parallel_config.decode_context_parallel_size != 1
            ):
                raise ValueError("NIXL chunked push currently requires PCP=DCP=1")
            elif len(transfer_groups) != 1 or not is_full_attention_spec(
                transfer_groups[0].kv_cache_spec
            ):
                raise ValueError(
                    "NIXL chunked push currently requires one full-attention "
                    "transfer group"
                )

    def bind_gpu_block_pool(self, gpu_block_pool: BlockPool) -> None:
        self._gpu_block_pool = gpu_block_pool

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
        to P via NIXL notification (deferred to `build_connector_meta`).
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
            if self._chunked_push:
                tracker = self._chunk_trackers.get(request.request_id)
                if tracker is None:
                    self._chunk_trackers[request.request_id] = _ChunkPushTracker(
                        request
                    )
                else:
                    tracker.request = request

        # P side with host-buffer offload: defer save to the worker.
        if self.use_host_buffer and params.get("do_remote_decode"):
            self._reqs_need_save[request.request_id] = request
            return

        # D side: only act on the first call (``do_remote_prefill`` is
        # unset on re-entry by the marker below).
        if not params.get("do_remote_prefill"):
            return

        if num_external_tokens <= 0:
            # Nothing to receive: full prefix-cache hit on D, no
            # registration to stage.
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
        self._reqs_need_recv[request.request_id] = (request, local_block_ids, (), False)

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
            if self._chunked_push:
                self._chunk_trackers.pop(request.request_id, None)
                if request.request_id in self._pinned_chunk_blocks:
                    self._chunk_cancellations.add(request.request_id)
            self._reqs_not_processed.add(request.request_id)
            self._reqs_need_save.pop(request.request_id, None)
            delay_free_blocks = bool(
                self._chunked_push and request.request_id in self._pinned_chunk_blocks
            )
            return delay_free_blocks, None

        delay_free_blocks = any(len(group) > 0 for group in block_ids)
        remote_num_tokens = 0
        tracker = (
            self._chunk_trackers.get(request.request_id) if self._chunked_push else None
        )
        chunk_handoff = bool(tracker and tracker.final_job_enqueued)
        if delay_free_blocks:
            block_ids = self.get_exchange_clipped_blocks(block_ids)
            remote_num_tokens = request.num_computed_tokens

            if not chunk_handoff:
                logger.debug(
                    "NixlPushConnector request_finished(%s) waiting for %d "
                    "seconds before releasing blocks",
                    request.request_id,
                    self._kv_lease_duration,
                )
                self._reqs_need_send[request.request_id] = (
                    time.perf_counter() + self._kv_lease_duration
                )

                # Store finished blocks for worker-level matching with D
                # registrations (via NIXL notifications).
                self._finished_request_blocks[request.request_id] = block_ids
                self._newly_finished_push_blocks[request.request_id] = block_ids

        if chunk_handoff:
            # Keep the finished request registered until every worker reports
            # finished_sending; the explicit pins also protect blocks across
            # preemption before the request reaches this terminal path.
            delay_free_blocks = request.request_id in self._pinned_chunk_blocks
        if self._chunked_push:
            self._chunk_trackers.pop(request.request_id, None)

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
        meta = super().build_connector_meta(scheduler_output)
        assert isinstance(meta, NixlConnectorMetadata)

        if self._chunked_push:
            self._build_chunk_jobs(meta, scheduler_output)

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

    def _build_chunk_jobs(
        self,
        meta: NixlConnectorMetadata,
        scheduler_output: SchedulerOutput,
    ) -> None:
        for req_id in scheduler_output.preempted_req_ids or ():
            if tracker := self._chunk_trackers.get(req_id):
                tracker.next_block = 0
                tracker.final_job_enqueued = False
                tracker.fallback = True
                self._chunk_fallbacks.add(req_id)

        if self._chunk_fallbacks:
            meta.push_chunk_fallbacks = set(self._chunk_fallbacks)
            self._chunk_fallbacks.clear()
        if self._chunk_cancellations:
            meta.push_chunk_cancellations = set(self._chunk_cancellations)
            self._chunk_cancellations.clear()

        block_state = scheduler_output.kv_connector_block_state
        assert block_state is not None
        group = self.kv_cache_config.transfer_groups[0]
        block_size = group.kv_cache_spec.block_size
        pool = self._gpu_block_pool
        assert pool is not None, (
            "GPU block pool must be bound before NIXL chunk jobs are emitted"
        )

        for (
            req_id,
            num_scheduled_tokens,
        ) in scheduler_output.num_scheduled_tokens.items():
            tracker = self._chunk_trackers.get(req_id)
            if tracker is None or tracker.final_job_enqueued:
                continue
            request = tracker.request
            computed_after = min(
                request.num_computed_tokens + num_scheduled_tokens,
                request.num_prompt_tokens,
            )
            is_final = computed_after >= request.num_prompt_tokens
            if tracker.fallback and not is_final:
                continue
            end_block = (
                cdiv(computed_after, block_size)
                if is_final
                else computed_after // block_size
            )
            start_block = tracker.next_block
            if not is_final and end_block <= start_block:
                continue

            all_block_ids = block_state.get_block_ids(req_id)
            assert all_block_ids is not None
            transfer_block_ids = self.kv_cache_config.select_transfer_block_ids(
                all_block_ids
            )
            assert len(transfer_block_ids) == 1
            block_table = list(transfer_block_ids[0])
            total_blocks = cdiv(request.num_prompt_tokens, block_size)
            assert end_block <= len(block_table), (
                f"NIXL chunk job for {req_id} needs {end_block} blocks, "
                f"but the current table has {len(block_table)}"
            )

            job = NixlPushChunkJob(
                request_id=req_id,
                local_block_ids=(block_table[:end_block],),
                start_block=0 if tracker.fallback else start_block,
                end_block=end_block,
                total_blocks=total_blocks,
                is_final=is_final,
            )
            meta.push_chunk_jobs.append(job)

            # The final job also owns the full table so the worker can safely
            # fall back to the stock monolithic WRITE after a late registration
            # or an incompatible peer handshake.
            referenced = (
                block_table[:end_block]
                if is_final
                else block_table[start_block:end_block]
            )
            pinned = self._pinned_chunk_blocks.setdefault(req_id, set())
            new_blocks = set(referenced) - pinned
            if new_blocks:
                pool.touch([pool.blocks[block_id] for block_id in new_blocks])
                pinned.update(new_blocks)

            tracker.next_block = end_block
            tracker.final_job_enqueued = is_final

    def has_pending_push_work(self) -> bool:
        # Keep the engine main loop alive while we have:
        # - finished P blocks awaiting WRITE completion, or
        # - pending D registrations the worker has not yet shipped, or
        # - newly finished blocks not yet shipped to P workers, or
        # - chunk jobs still pinning source blocks.
        return bool(
            self._finished_request_blocks
            or self._push_pending_registrations
            or (self._chunked_push and self._pinned_chunk_blocks)
        )

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        """Clean up finished request blocks after push completes."""
        super().update_connector_output(connector_output)
        for req_id in connector_output.finished_sending or ():
            block_ids = (
                self._pinned_chunk_blocks.pop(req_id, None)
                if self._chunked_push
                else None
            )
            if block_ids:
                pool = self._gpu_block_pool
                assert pool is not None
                pool.free_blocks(
                    pool.blocks[bid] for bid in sorted(block_ids, reverse=True)
                )
            self._finished_request_blocks.pop(req_id, None)
        # On D side, finished_recving means the push completed; clear the
        # watchdog so we don't trip an expiration on a fulfilled request.
        for req_id in connector_output.finished_recving or ():
            self._push_registration_deadlines.pop(req_id, None)
