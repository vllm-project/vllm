# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Push-specific (WRITE) worker-side logic for the NIXL connector.

A dedicated `nixl-push-writer` thread owns all push-related NIXL ops:
calls `get_new_notifs` (routing PUSH_REG internally; HB / completion
notifs are forwarded to the engine main thread), sends PUSH_REG via
`send_notif`, matches D registrations with P finished blocks, and
issues WRITE transfers via `make_prepped_xfer` / `transfer`.

The engine main thread feeds the writer through queues:
`_reg_send_inbox` (D-side regs to send), `_prefill_inbox` (P-side finished
requests, push jobs and retirements, in the order the engine thread queued
them) and `_pending_completion_notifs`
(non-PUSH_REG notifs forwarded back for HB / completion accounting).
The handshake-completion callback feeds `_deferred_push_inbox` with
matched pushes whose P→D handshake has finished so the writer can
(re-)issue the WRITE without ever blocking on the network.

Wake model: the writer self-polls every
`_PUSH_WRITER_POLL_INTERVAL_MS` only while it has unmatched
`_push_finished_blocks` (i.e. P-side blocks waiting for a D PUSH_REG
notif that has no other wake source). All other progress is
event-driven: the engine main thread sets `_push_writer_wake` from
`start_load_kv` (when handing it new work) and from `get_finished`
(so each engine step gives the writer a chance to drain NIXL notifs);
the handshake-completion callback sets the same event after a deferred
PUSH_REG send or a deferred push WRITE has been queued. When a request's
lease expires (the base worker reports it via `done_sending`) or the WRITE completes,
`get_finished` enqueues an eviction onto `_evict_finished_inbox` so
the writer drops any leftover `_push_finished_blocks` /
`_pending_d_registrations` and stops self-polling.
"""

import queue
import threading
import time
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import msgspec
import torch

from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorTransferResults,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    NixlBaseConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    PUSH_REG_NOTIF_PREFIX,
    NixlAgentMetadata,
    NixlConnectorMetadata,
    NixlPushJob,
    RemoteMeta,
    ReqId,
    ReqMeta,
    TransferHandle,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    ReadSpec,
    _is_attention_spec,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import get_base_request_id
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)

# Writer-thread poll cadence while there is in-flight push state. When
# fully idle, the writer blocks on a wake event signalled by the engine
# main thread (start_load_kv / get_finished). Smaller -> lower latency
# while active, slightly more CPU.
_PUSH_WRITER_POLL_INTERVAL_MS = 1.0


@dataclass
class _PushJob:
    """One batch of KV blocks: a push job and the step event of its blocks."""

    push_job: NixlPushJob
    event: torch.Event
    handles: list[TransferHandle] = field(default_factory=list)
    submitted: bool = False
    # Preempted or aborted: poll the posted WRITEs, never notify D.
    cancelled: bool = False


@dataclass
class _PushRequest:
    """The writer's early-push state for one request."""

    # D's registration; kept across a preemption because D registers once.
    reg_data: dict[str, Any] | None = None
    # Blocks of the request when it finishes; the hand-off waits for the job.
    finished_blocks: BlockIds | None = None
    # D was told to decode; sent once, after the job's WRITEs complete.
    notified: bool = False
    # The job was not written or the notification failed.
    failed: bool = False
    # Aborted or evicted: poll the posted WRITEs to completion, then drop the
    # record; never notify D or push the request again.
    dropped: bool = False
    job: _PushJob | None = None


@dataclass
class _PushRetirement:
    """Preempted and aborted requests whose push jobs are cancelled. `done`
    is set once none of their early-push WRITEs is in flight; a whole-request
    push started at finish is protected by the KV lease instead."""

    preempted: set[ReqId]
    aborted: set[ReqId]
    done: threading.Event = field(default_factory=threading.Event)


class NixlPushConnectorWorker(NixlBaseConnectorWorker):
    """Push-specific (WRITE) worker logic. See module docstring."""

    # Distinguishes push from pull in the NIXL compatibility hash.
    _TRANSFER_MODE: str = "push"

    _supports_pp_hma = True

    def __init__(
        self,
        vllm_config: "VllmConfig",
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, engine_id, kv_cache_config)

        # Heartbeat handshakes to a PP-sharded producer must be notif-only,
        # like the PUSH_REG path.
        self._hb_handshake_notif_only = True

        # Push-specific state.
        # P-side: outgoing WRITE handles awaiting completion, keyed by
        # request_id. Mutated by writer (submit) and main thread
        # (``_pop_done_transfers``); guarded by
        # ``_sending_transfers_lock``.
        self._sending_transfers = defaultdict[ReqId, list[TransferHandle]](list)
        self._send_failures: set[ReqId] = set()
        self._sending_transfers_lock = threading.Lock()

        # Writer-thread owned matching state.
        # P-side: finished request blocks received from scheduler metadata
        # that have not yet been matched with an incoming D registration.
        self._push_finished_blocks: dict[ReqId, BlockIds] = {}
        # P-side: D registrations received via NIXL notification that have
        # not yet been matched with a finished P request.
        self._pending_d_registrations: dict[ReqId, dict[str, Any]] = {}

        # Cross-thread channels.
        self._reg_send_inbox: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue()
        # One queue, so that the writer sees a request's push job before it
        # finishes and a retirement after the job it drops.
        self._prefill_inbox: queue.Queue[
            tuple[str, BlockIds] | _PushJob | _PushRetirement
        ] = queue.Queue()
        self._pending_completion_notifs: queue.Queue[bytes] = queue.Queue()
        # Main thread → writer: req_ids whose lease has expired or whose
        # WRITE has completed. Writer drops them from
        # ``_push_finished_blocks`` so an unmatched entry doesn't keep the
        # writer busy-polling forever.
        self._evict_finished_inbox: queue.Queue[str] = queue.Queue()
        # Handshakes that have just completed and are ready for the WRITE on wthread
        self._deferred_push_inbox = queue.Queue[tuple[str, BlockIds, dict[str, Any]]]()

        # Wake signal from engine main thread (start_load_kv / get_finished).
        # Writer self-polls at _PUSH_WRITER_POLL_INTERVAL_MS while it has
        # active in-flight state; otherwise it blocks until signalled.
        self._push_writer_wake = threading.Event()

        self._push_writer_stop = threading.Event()
        self._push_writer_thread: threading.Thread | None = None

        extra_config = self.kv_transfer_config.kv_connector_extra_config
        self._early_push = bool(
            extra_config.get("enable_early_push", False)
            and self.kv_transfer_config.kv_role == "kv_producer"
        )
        # Push job state belongs to the writer; one record per request.
        self._push_requests: dict[ReqId, _PushRequest] = {}
        self._push_retirements: list[_PushRetirement] = []

    def _validate_remote_parallel_config(
        self, agent_metadata: NixlAgentMetadata
    ) -> None:
        super()._validate_remote_parallel_config(agent_metadata)
        local = (self.block_size, self._physical_blocks_per_logical_kv_block)
        remote = (
            agent_metadata.block_size,
            agent_metadata.physical_blocks_per_logical_kv_block,
        )
        if local != remote:
            raise NotImplementedError(
                "NixlPushConnector requires identical P/D block sizes. Kernel "
                "block size and kernel blocks per logical block: local "
                f"{local}, remote engine {agent_metadata.engine_id} {remote}."
            )

    # --- Lifecycle ----------------------------------------------------- #

    def register_kv_caches(self, kv_caches: dict[str, "torch.Tensor"]):
        super().register_kv_caches(kv_caches)
        if self._mixed_mem_types:
            raise NotImplementedError(
                "NixlPushConnector does not support mixed-memory KV caches"
            )
        if self._push_writer_thread is None:
            self._push_writer_thread = threading.Thread(
                target=self._push_writer_loop,
                daemon=True,
                name="nixl-push-writer",
            )
            self._push_writer_thread.start()
            logger.info("nixl-push-writer thread started (rank=%d)", self.tp_rank)

    def _validate_remote_agent_handshake(
        self,
        nixl_agent_meta: NixlAgentMetadata,
        remote_tp_size: int,
        remote_dcp_size: int = 1,
        local_region_ids: Sequence[int] | None = None,
    ) -> None:
        super()._validate_remote_agent_handshake(
            nixl_agent_meta, remote_tp_size, remote_dcp_size, local_region_ids
        )
        assert self.transfer_topo is not None
        if (
            self._head_sharded_draft_regions
            and self.transfer_topo.tp_size != remote_tp_size
        ):
            raise NotImplementedError(
                "NIXL push mode does not support a head-sharded draft's KV "
                "under an MLA target with heterogeneous TP. Use pull mode."
            )

    def shutdown(self) -> None:
        self._push_writer_stop.set()
        # Unblock the writer if it's waiting in the no-active-state branch.
        self._push_writer_wake.set()
        if self._push_writer_thread is not None:
            self._push_writer_thread.join(timeout=2)
            self._push_writer_thread = None
        with self._sending_transfers_lock:
            for handles in self._sending_transfers.values():
                for handle in handles:
                    self.nixl_wrapper.release_xfer_handle(handle)
            self._sending_transfers.clear()
            self._send_failures.clear()
        for req in self._push_requests.values():
            if req.job is not None:
                for handle in req.job.handles:
                    self.nixl_wrapper.release_xfer_handle(handle)
        self._push_requests.clear()
        self._push_retirements.clear()
        super().shutdown()

    # --- Engine-main-thread entry point -------------------------------- #

    def start_load_kv(self, metadata: NixlConnectorMetadata):
        """Pre-process metadata; defer NIXL ops to the writer thread."""
        if self.pcp_rank > 0 and not self.pcp_dcp_sharded:
            return

        # D-side: track reqs waiting for P to push.
        for req_id, meta in metadata.reqs_to_recv.items():
            meta.local_physical_block_ids = self._logical_to_kernel_block_ids(
                meta.local_block_ids, self._physical_blocks_per_logical_kv_block
            )
            assert meta.remote is not None
            remote_engine_id = meta.remote.engine_id
            # Update last activity from this remote (same as pull), but only for
            # an already-connected engine so we never leave an _engine_last_active
            # entry without a _remote_agents entry (breaks _cleanup_remote_engine).
            if remote_engine_id in self._remote_agents:
                self._engine_last_active[remote_engine_id] = time.perf_counter()

            logger.debug(
                "start_load_kv (push) for request %s from remote engine %s. "
                "Num local_block_ids: %s. Num remote_block_ids: %s. ",
                req_id,
                remote_engine_id,
                len(meta.local_physical_block_ids),
                len(meta.remote.block_ids),
            )
            self._recving_metadata[req_id] = meta

        # --- D-side: registrations to send to P via NIXL ---
        if metadata.push_registrations:
            for req_id, reg_data in metadata.push_registrations.items():
                self._reg_send_inbox.put((req_id, reg_data))
            self._push_writer_wake.set()

        # --- P-side: newly finished blocks awaiting a D registration match ---
        if metadata.push_finished_blocks:
            for req_id, block_ids in metadata.push_finished_blocks.items():
                self._prefill_inbox.put((req_id, block_ids))
            self._push_writer_wake.set()

        # Batch + lease tracking (same as pull).
        for req_id in metadata.reqs_in_batch:
            self._reqs_to_process.add(req_id)
        for req_id in metadata.reqs_not_processed:
            self._reqs_to_process.discard(req_id)
            assert req_id not in self._reqs_to_send
        # Rebase scheduler-clock deadlines onto this worker's clock — see the
        # equivalent block in pull_worker.start_load_kv for the rationale.
        now_local = time.perf_counter()
        for req_id, expiration_time in metadata.reqs_to_send.items():
            if req_id in self._reqs_to_process:
                if metadata.scheduler_clock:
                    expiration_time = now_local + (
                        expiration_time - metadata.scheduler_clock
                    )
                self._reqs_to_send[req_id] = expiration_time

        # Heartbeats still leave from the main thread (base worker behaviour).
        self._send_heartbeats(metadata)

    def finish_push_step(self, metadata: NixlConnectorMetadata) -> None:
        """Hand this step's push jobs to the writer, which writes their blocks
        once the step's event shows the forward has computed them."""
        if not self._early_push or not metadata.push_jobs:
            return
        event = torch.Event()
        event.record()
        for job in metadata.push_jobs:
            self._prefill_inbox.put(_PushJob(job, event))
        self._push_writer_wake.set()

    def handle_preemptions(self, metadata: NixlConnectorMetadata) -> None:
        """Have the writer cancel the push jobs of preempted and aborted
        requests, and wait until none of their WRITEs is in flight before the
        blocks are reused. A request whose job was handed to the worker is
        pushed whole at finish; one preempted before that stays eligible."""
        if not self._early_push or not (
            metadata.push_job_preemptions or metadata.push_job_aborts
        ):
            return
        retirement = _PushRetirement(
            set(metadata.push_job_preemptions), set(metadata.push_job_aborts)
        )
        self._prefill_inbox.put(retirement)
        self._push_writer_wake.set()
        if not retirement.done.wait(self._lease_extension):
            logger.error(
                "NIXL push WRITEs of requests %s did not complete within %ds; "
                "their blocks may be reused while still being read.",
                retirement.preempted | retirement.aborted,
                self._lease_extension,
            )

    # --- Writer thread ------------------------------------------------- #

    def _push_writer_loop(self) -> None:
        sleep_s = _PUSH_WRITER_POLL_INTERVAL_MS / 1000.0

        while not self._push_writer_stop.is_set():
            try:
                # 1. D registrations to send.
                while True:
                    try:
                        rid, rd = self._reg_send_inbox.get_nowait()
                    except queue.Empty:
                        break
                    self._send_registration_to_p(rid, rd)

                # 2. Deferred P→D pushes whose handshake just completed; do xfer now
                while True:
                    try:
                        rid, blocks, rd = self._deferred_push_inbox.get_nowait()
                    except queue.Empty:
                        break
                    self._do_start_push_kv(rid, blocks, rd)

                # 3. P-side finished requests, push jobs and retirements.
                self._take_prefill_inbox()

                # 3b. Evict finished blocks for requests that have either
                # completed (WRITE acknowledged) or whose lease expired
                # without a D registration.  Drop pending registrations
                # for the same reason so we don't leak state.
                while True:
                    try:
                        rid = self._evict_finished_inbox.get_nowait()
                    except queue.Empty:
                        break
                    self._push_finished_blocks.pop(rid, None)
                    self._pending_d_registrations.pop(rid, None)
                    if self._early_push:
                        self._cancel_push_request(rid, drop=True)

                # 4. NIXL notifs: route PUSH_REG; forward the rest.
                for notifs in self.nixl_wrapper.get_new_notifs().values():
                    for notif in notifs:
                        if notif.startswith(PUSH_REG_NOTIF_PREFIX):
                            self._handle_push_reg_notif(notif)
                        else:
                            self._pending_completion_notifs.put(notif)

                if self._early_push:
                    self._advance_push_requests()
            except Exception:
                logger.exception("nixl-push-writer error; continuing")

            # Self-poll only while there is no other wake source: P-side
            # finished blocks waiting for a D PUSH_REG match. All other
            # progress is event-driven (see module docstring).
            if self._push_finished_blocks or (
                self._early_push
                and (
                    self._push_retirements
                    or any(req.job is not None for req in self._push_requests.values())
                )
            ):
                self._push_writer_stop.wait(timeout=sleep_s)
            else:
                self._push_writer_wake.wait()
                self._push_writer_wake.clear()

    def _handle_push_reg_notif(self, notif: bytes) -> None:
        try:
            reg_data = msgspec.msgpack.decode(notif[len(PUSH_REG_NOTIF_PREFIX) :])
        except Exception:
            logger.exception("Failed to decode PUSH_REG notification payload")
            return
        rid = reg_data.get("request_id") if isinstance(reg_data, dict) else None
        if not isinstance(rid, str):
            logger.warning("PUSH_REG notif missing request_id; dropping")
            return

        match = self._pop_matching_finished_blocks(rid)
        if match is not None:
            fin_id, blocks = match
            self._do_start_push_kv(fin_id, blocks, reg_data)
        else:
            self._pending_d_registrations[rid] = reg_data

    def _push_path_supported(self, reg_data: dict[str, Any]) -> bool:
        engine_id = reg_data["decode_engine_id"]
        assert self.transfer_topo is not None
        remote_info = self.transfer_topo.get_engine_info(engine_id)
        if (
            remote_info.remote_tp_size != self.world_size
            or remote_info.remote_dcp_size != 1
        ):
            logger.warning_once(
                "NIXL early push needs a decode engine with the prefill TP "
                "size %d and DCP=1, but decode engine %s has TP=%d, DCP=%d. "
                "Pushing requests when they finish.",
                self.world_size,
                engine_id,
                remote_info.remote_tp_size,
                remote_info.remote_dcp_size,
            )
            return False
        return True

    def _cancel_push_request(self, req_id: ReqId, *, drop: bool) -> None:
        """Cancel the request's job after a preemption, abort or eviction."""
        req = self._push_requests.get(req_id)
        if req is None or (req.notified and not drop):
            # Nothing was written, or D was already told to decode.
            return
        req.dropped |= drop
        if drop:
            req.finished_blocks = None
        if req.job is not None:
            # Keep polling the posted handles; never notify D for this batch.
            req.job.cancelled = True
        elif drop:
            self._push_requests.pop(req_id, None)

    def _take_prefill_inbox(self) -> None:
        while True:
            try:
                item = self._prefill_inbox.get_nowait()
            except queue.Empty:
                return
            if isinstance(item, tuple):
                rid, blocks = item
                req = self._push_requests.get(rid) if self._early_push else None
                if req is not None:
                    # A request with a push job hands off through the job.
                    req.finished_blocks = blocks
                    continue
                # No early-push record: push the request whole once D's
                # registration is available.
                matched = self._pop_matching_registration(rid)
                if matched is not None:
                    self._do_start_push_kv(rid, blocks, matched)
                else:
                    self._push_finished_blocks[rid] = blocks
                continue
            if isinstance(item, _PushRetirement):
                for req_id in item.aborted:
                    self._cancel_push_request(req_id, drop=True)
                for req_id in item.preempted:
                    self._cancel_push_request(req_id, drop=False)
                self._push_retirements.append(item)
                continue
            # Queued in order, so a retirement cancels only earlier jobs.
            req_id = item.push_job.request_id
            req = self._push_requests.setdefault(req_id, _PushRequest())
            if req.dropped or req.notified or req.job is not None:
                # A request is pushed at most once; a request with a batch
                # still in flight cannot start another one.
                logger.warning(
                    "Dropping early push job for %s (dropped=%s, notified=%s, "
                    "job in flight=%s)",
                    req_id,
                    req.dropped,
                    req.notified,
                    req.job is not None,
                )
                continue
            req.job = item

    def _advance_push_requests(self) -> None:
        """Advance every request's push job by one step of its lifecycle."""
        try:
            for req_id, req in list(self._push_requests.items()):
                self._advance_push_request(req_id, req)
        finally:
            # An exception from one request escapes to the writer loop, which
            # logs it and retries; still release retirements whose WRITEs settled.
            self._release_push_retirements()

    def _advance_push_request(self, req_id: ReqId, req: _PushRequest) -> None:
        job = req.job
        if job is not None:
            if not job.submitted and not job.cancelled:
                # A submission that raises fails only its request; a poll or
                # hand-off that raises is retried on the next pass.
                try:
                    self._submit_push_job(req, job)
                except Exception:
                    logger.exception("NIXL early push submission failed for %s", req_id)
                    req.failed = True
                    if job.handles:
                        # Keep the posted WRITEs polled; never notify D.
                        job.submitted = True
                        job.cancelled = True
                    else:
                        req.job = None
            if job.submitted:
                # A failed WRITE may still have sibling handles in flight.
                transfers = {req_id: job.handles}
                _, failed = self._pop_done_transfers(transfers)
                req.failed |= req_id in failed
                job.handles = transfers.get(req_id, [])
            if (
                job.submitted
                and not job.cancelled
                and not job.handles
                and not req.failed
                and not req.notified
            ):
                if req.reg_data is None:
                    # A submitted job always found D's registration; fail the
                    # request instead of leaving the job pending forever.
                    logger.error("NIXL early push lost the registration of %s", req_id)
                    req.failed = True
                else:
                    # Set first: a send that raises may still have reached D.
                    req.notified = True
                    try:
                        self._send_completion_notif(req.reg_data)
                    except Exception:
                        logger.exception(
                            "NIXL early push completion notification failed for %s",
                            req_id,
                        )
                        req.failed = True
            settled = (job.submitted and not job.handles) or (
                not job.submitted and (job.cancelled or req.failed)
            )
            if settled:
                req.job = None
        if req.dropped:
            if req.job is None:
                self._push_requests.pop(req_id, None)
            return
        if req.finished_blocks is not None:
            self._handoff_pushed_request(req_id, req)

    def _submit_push_job(self, req: _PushRequest, job: _PushJob) -> None:
        """Post the job's WRITEs once D's registration, the handshake, the
        step event and TP/DCP compatibility are available."""
        if req.reg_data is None:
            req.reg_data = self._pop_matching_registration(job.push_job.request_id)
        reg_data = req.reg_data
        if reg_data is None:
            # D has not registered yet; retry on the next pass.
            return
        if (
            self._ensure_handshake(
                reg_data["decode_engine_id"],
                reg_data["decode_host"],
                reg_data["decode_port"],
                reg_data["decode_tp_size"],
            )
            is not None
        ):
            # D is not handshaked yet; the call has started the handshake and
            # the request is pushed when it finishes.
            req.failed = True
            return
        if not job.event.query():
            # The step's forward has not completed yet.
            return
        if not self._push_path_supported(reg_data):
            req.failed = True
            return

        job.handles = []
        failed = False
        try:
            self._push_blocks(
                job.push_job.request_id,
                job.push_job.block_ids,
                self._as_grouped_block_ids(reg_data["local_block_ids"]),
                reg_data,
                notify=False,
                handles=job.handles,
            )
        except Exception:
            logger.exception(
                "NIXL early push submission failed for %s", job.push_job.request_id
            )
            failed = True
        with self._sending_transfers_lock:
            if job.push_job.request_id in self._send_failures:
                self._send_failures.discard(job.push_job.request_id)
                failed = True
        if not job.handles:
            # No WRITE was posted: the request is pushed when it finishes.
            req.failed = True
            return
        # Posted WRITEs are polled to completion even when the submission
        # failed: they still read the request's blocks.
        job.submitted = True
        req.failed |= failed

    def _release_push_retirements(self) -> None:
        """Release the `handle_preemptions` wait once a retirement's requests
        have no early-push WRITEs in flight."""
        for retirement in list(self._push_retirements):
            retired = retirement.preempted | retirement.aborted
            if any(
                (r := self._push_requests.get(rid)) is not None
                and r.job is not None
                and r.job.handles
                for rid in retired
            ):
                continue
            self._push_retirements.remove(retirement)
            retirement.done.set()

    def _send_completion_notif(self, reg_data: dict[str, Any]) -> None:
        """Notify each D rank written to that the request's KV cache is
        complete, as the notification attached to a whole-request push
        WRITE does."""
        engine_id = reg_data["decode_engine_id"]
        agents = self._remote_agents[engine_id]
        pp_rank = self._single_remote_stage(engine_id).pp_rank
        notif = f"{reg_data['request_id']}:{self.world_size}".encode()
        for rank in self.tp_mappings[engine_id].all_source_ranks:
            self.nixl_wrapper.send_notif(agents[(pp_rank, rank)], notif_msg=notif)

    def _handoff_pushed_request(self, req_id: ReqId, req: _PushRequest) -> None:
        """Hand a finished request off to D once its push job is done.

        The fallback whole-request push is posted before the record is
        dropped, so a submission that raises is retried on the next pass."""
        blocks = req.finished_blocks
        assert blocks is not None
        if req.reg_data is None:
            req.reg_data = self._pop_matching_registration(req_id)
        if req.reg_data is None:
            # D never registered: no WRITE was posted, so push the request
            # whole when D registers.
            req.job = None
            self._push_finished_blocks[req_id] = blocks
            self._push_requests.pop(req_id, None)
            return
        if req.job is not None:
            # Waiting the job out before pushing the request whole keeps D
            # from being notified twice.
            return
        if req.notified:
            self._push_requests.pop(req_id, None)
            # No handle is left in `_sending_transfers`, so
            # `get_transfer_results` reports the request as a completed push.
            # A failed send may still have reached D: leave it to the lease.
            if not req.failed:
                with self._sending_transfers_lock:
                    self._sending_transfers[req_id] = []
            return
        # A previous attempt may have posted some WRITEs before raising; the
        # main thread reports their completion, and posting them again would
        # duplicate the WRITEs and the notifications attached to them.
        with self._sending_transfers_lock:
            push_started = bool(self._sending_transfers.get(req_id))
        if push_started:
            self._push_requests.pop(req_id, None)
            return
        # The job was not written and D was not notified: push the request as
        # it finished.
        self._do_start_push_kv(req_id, blocks, req.reg_data)
        self._push_requests.pop(req_id, None)

    # --- D-side registration send (writer thread) ---------------------- #

    def _send_registration_to_p(
        self,
        req_id: str,
        reg_data: dict[str, Any],
    ) -> None:
        """Handshake (if needed) then send PUSH_REG. `send_notif` always
        executes on the writer; the handshake runs on the background executor
        and the request is re-queued onto `_reg_send_inbox` once it
        completes (at which point `_ensure_handshake` returns `None` and we
        send directly)."""
        remote_pp_size = reg_data.get("remote_pp_size", 1)
        fut = self._ensure_handshake(
            reg_data["remote_engine_id"],
            reg_data["remote_host"],
            reg_data["remote_port"],
            reg_data["remote_tp_size"],
            pp_size=remote_pp_size,
            # D only ever sends PUSH_REG notifs to P and never reads or writes
            # P's memory in push mode, so it never needs the transfer
            # descriptors set up by the full add_remote_agent path.
            notif_agents_only=True,
        )
        if fut is None:
            self._do_send_reg_notif(req_id, reg_data)
            return

        def _on_handshake(
            f: Future[tuple[dict[tuple[int, int], str], float]],
            rid: str = req_id,
            rd: dict[str, Any] = reg_data,
        ) -> None:
            try:
                f.result()
            except Exception as e:
                self._log_failure(
                    failure_type="push_reg_handshake_failed", req_id=rid, error=e
                )
                self._failed_recv_reqs.put(rid)
                return
            # Re-queue for the writer to send now that the handshake is done.
            self._reg_send_inbox.put((rid, rd))
            # Wake the writer so it sends the PUSH_REG promptly even if
            # otherwise parked.
            self._push_writer_wake.set()

        fut.add_done_callback(_on_handshake)

    def _do_send_reg_notif(self, req_id: str, reg_data: dict[str, Any]) -> None:
        engine_id = reg_data["remote_engine_id"]
        notif_msg = PUSH_REG_NOTIF_PREFIX + msgspec.msgpack.encode(reg_data)
        # _remote_agents is mutated on other threads; snapshot under the lock.
        with self._handshake_lock:
            agents = dict(self._remote_agents.get(engine_id) or {})
        if not agents:
            logger.error(
                "No remote agents for engine %s; cannot send registration for %s",
                engine_id,
                req_id,
            )
            self.xfer_stats.record_failed_notification()
            self._failed_recv_reqs.put(req_id)
            return
        for rank, agent_name in agents.items():
            try:
                self.nixl_wrapper.send_notif(agent_name, notif_msg=notif_msg)
            except Exception as e:
                self._log_failure(
                    failure_type="push_reg_notif_failed",
                    req_id=req_id,
                    error=e,
                    remote_rank=rank,
                )
                self.xfer_stats.record_failed_notification()
                # Earlier registrations may still trigger WRITEs into D's blocks.
                # Keep the receive pending until those writes are finished.
        logger.debug(
            "Sent PUSH_REG for %s to engine %s (%dB)", req_id, engine_id, len(notif_msg)
        )

    # --- Matching helpers --------------------------------------------- #

    def _pop_matching_registration(self, request_id: str) -> dict[str, Any] | None:
        """Pop the D-side registration matching *request_id*.

        Exact key first, then a match after stripping the random suffix from
        both sides. No match leaves the request unmatched (push not started).
        """
        data = self._pending_d_registrations.pop(request_id, None)
        if data is not None:
            return data
        base_id = get_base_request_id(request_id)
        for reg_id in list(self._pending_d_registrations):
            if get_base_request_id(reg_id) == base_id:
                return self._pending_d_registrations.pop(reg_id)
        return None

    def _pop_matching_finished_blocks(
        self, request_id: str
    ) -> tuple[str, BlockIds] | None:
        """Pop the P-side finished blocks matching *request_id*.

        Same lookup as `_pop_matching_registration`: exact key, then a
        match after stripping the random suffix from both sides.
        """
        blocks = self._push_finished_blocks.pop(request_id, None)
        if blocks is not None:
            return request_id, blocks
        base_id = get_base_request_id(request_id)
        for fin_id in list(self._push_finished_blocks):
            if get_base_request_id(fin_id) == base_id:
                return fin_id, self._push_finished_blocks.pop(fin_id)
        return None

    # --- WRITE transfer logic (writer thread) ------------------------- #

    def _do_start_push_kv(
        self,
        request_id: str,
        local_block_ids: BlockIds,
        registration_data: dict[str, Any],
    ) -> None:
        """Start push-based KV transfer from P worker to D node.

        The P→D handshake runs on the base worker's background executor.
        If it isn't ready yet we register a completion callback, defer the
        WRITE, and re-drive this request via `_deferred_push_inbox` once
        the handshake resolves -- so the writer thread never blocks on the
        network (mirrors `_send_registration_to_p`).
        """
        if not local_block_ids:
            logger.warning("No local blocks to push for request %s", request_id)
            return

        # ``local_block_ids`` are P's logical block IDs; ``remote_block_ids``
        # (D's, from the PUSH_REG notif) are also logical.
        decode_engine_id = registration_data["decode_engine_id"]

        # Runs on the background executor; defer the WRITE until it's ready.
        fut = self._ensure_handshake(
            decode_engine_id,
            registration_data["decode_host"],
            registration_data["decode_port"],
            registration_data["decode_tp_size"],
        )
        if fut is not None:

            def _on_handshake(
                f: Future[tuple[dict[tuple[int, int], str], float]],
                rid: str = request_id,
                blocks: BlockIds = local_block_ids,
                rd: dict[str, Any] = registration_data,
            ) -> None:
                if (e := f.exception()) is not None:
                    # The engine reclaims the blocks via the TTL so we dont free here
                    self._log_failure(
                        failure_type="push_handshake_failed", req_id=rid, error=e
                    )
                    return
                self._deferred_push_inbox.put((rid, blocks, rd))
                self._push_writer_wake.set()

            fut.add_done_callback(_on_handshake)
            return

        t0 = time.perf_counter()
        handles: list[TransferHandle] = []
        try:
            self._push_blocks(
                request_id,
                self._as_grouped_block_ids(local_block_ids),
                self._as_grouped_block_ids(registration_data["local_block_ids"]),
                registration_data,
                handles=handles,
            )
        except Exception:
            # A WRITE posted before the failure is still reading P's blocks;
            # mark the push failed so the main thread does not report it as
            # completed, and let the handle drain through `_sending_transfers`.
            with self._sending_transfers_lock:
                self._send_failures.add(request_id)
            raise
        finally:
            # Publish every handle that was posted, including a partial set:
            # otherwise a failed submission leaks the handles and the blocks
            # they read.
            if handles:
                with self._sending_transfers_lock:
                    self._sending_transfers[request_id].extend(handles)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if elapsed_ms > 200.0:
            logger.warning(
                "_do_start_push_kv for %s took %.1fms (slow NIXL submission)",
                request_id,
                elapsed_ms,
            )

    def _push_blocks(
        self,
        request_id: str,
        local_block_ids: BlockIds,
        remote_block_ids: BlockIds,
        registration_data: dict[str, Any],
        *,
        notify: bool = True,
        handles: list[TransferHandle] | None = None,
    ) -> list[TransferHandle]:
        """WRITE P's logical `local_block_ids` into D's logical
        `remote_block_ids` and return the transfer handles.

        The handles are appended to `handles` as each WRITE is posted, so a
        submission that raises still leaves its posted handles tracked."""
        decode_engine_id = registration_data["decode_engine_id"]
        # Keep the engine alive while it is actively receiving pushes, mirroring
        # how pull-mode transfers touch _engine_last_active in start_load_kv.
        self._engine_last_active[decode_engine_id] = time.perf_counter()

        physical_local = self._logical_to_kernel_block_ids(
            local_block_ids, self._physical_blocks_per_logical_kv_block
        )

        push_meta = ReqMeta(
            local_block_ids=local_block_ids,
            local_physical_block_ids=physical_local,
            tp_size=self.world_size,
            remote=RemoteMeta(
                block_ids=remote_block_ids,
                host="",
                port=0,
                engine_id=decode_engine_id,
                request_id=registration_data["request_id"],
            ),
        )
        return self._xfer_blocks_for_req(
            req_id=request_id, meta=push_meta, notify=notify, handles=handles
        )

    @staticmethod
    def _as_grouped_block_ids(block_ids: BlockIds) -> BlockIds:
        """Normalise a sequence of block IDs to a tuple-of-groups shape.

        `BlockIds` is canonically a tuple of per-group lists, but some
        registration payloads collapse a single-group case to a flat
        list. Re-wrap that case so downstream group-aware helpers see a
        consistent shape."""
        if block_ids and not isinstance(block_ids[0], (list, tuple)):
            return (list(block_ids),)
        return block_ids

    def _xfer_blocks_for_req(
        self,
        req_id: str,
        meta: ReqMeta,
        *,
        notify: bool = True,
        handles: list[TransferHandle] | None = None,
    ) -> list[TransferHandle]:
        """Issue WRITE transfers to one or more remote TP ranks and return their
        handles. The WRITEs carry the request's completion notification if
        `notify`."""
        assert meta.remote is not None and self.transfer_topo is not None
        engine_id = meta.remote.engine_id
        plan = self.tp_mappings[engine_id]
        remote_info = self.transfer_topo.get_engine_info(engine_id)
        tp_ratio = self.transfer_topo.tp_ratio(remote_info.remote_tp_size)

        # Expand D's logical IDs using the ratio learned during the
        # NIXL handshake. ``meta`` is freshly built by
        # ``_push_blocks`` so mutating it here is safe.
        meta.remote.block_ids = self._logical_to_kernel_block_ids(
            meta.remote.block_ids,
            remote_info.remote_physical_blocks_per_logical,
        )
        remote_block_ids = meta.remote.block_ids
        local_block_ids = meta.local_physical_block_ids
        if not self._single_remote_stage(engine_id).pairs.groups_match:
            raise NotImplementedError(
                "NixlPushConnector does not support different producer and "
                "consumer cache-group layouts"
            )

        # MLA latent is replicated across D's TP ranks: the tp-mapping
        # collapses it to one rank (fine for reads), but push must WRITE every
        # D rank or the rest decode stale KV. For hybrid MLA+SSM the sharded
        # SSM state already targets every covered D rank, so only the
        # attention groups need widening; pure MLA writes to all handshaked
        # ranks (only the dst differs per rank).
        replicate_attn = self.use_mla and tp_ratio < 0
        if replicate_attn and not self._has_mamba:
            assert len(plan.all_source_ranks) == 1
            write_ranks = sorted(self._single_remote_stage(engine_id).xfer_side_handles)
        else:
            write_ranks = list(plan.all_source_ranks)

        num_groups = len(local_block_ids)

        def group_ids(block_ids: BlockIds, rank: int) -> BlockIds:
            return [
                list(block_ids[g])
                if (self._is_csa_linear and tp_ratio < 0)
                or (replicate_attn and _is_attention_spec(self._group_spec_types[g]))
                or rank in plan.source_ranks_per_group[g]
                else []
                for g in range(num_groups)
            ]

        read_specs = [
            ReadSpec(
                remote_rank=rank,
                local_block_ids=group_ids(local_block_ids, rank),
                remote_block_ids=group_ids(remote_block_ids, rank),
            )
            for rank in write_ranks
        ]

        out = handles if handles is not None else []
        for i, spec in enumerate(read_specs):
            remote_block_size = remote_info.remote_block_size
            logger.debug(
                "Remote agent %s available, calling _xfer_blocks"
                " on remote rank %s with remote block size %s for req %s",
                meta.remote.engine_id,
                spec.remote_rank,
                remote_block_size,
                req_id,
            )
            if tp_ratio < 0 and (not self.use_mla or len(plan.all_source_ranks) > 1):
                # Multiple targets: write each rank its chunk of local memory.
                # Hybrid MLA+SSM also lands here: its split handles replicate
                # the attention descriptors and chunk only the SSM state.
                split_key = (tp_ratio, remote_block_size)
                local_xfer_side_handle = self.src_xfer_handles_by_tp_ratio[split_key][i]
            else:
                local_xfer_side_handle = self.src_xfer_handles_by_block_size[
                    remote_block_size
                ]

            remote_xfer_side_handle = self._single_remote_stage(
                meta.remote.engine_id
            ).xfer_side_handles[spec.remote_rank]

            handle = self._xfer_blocks(
                read_spec=spec,
                request_id=req_id,
                dst_engine_id=meta.remote.engine_id,
                remote_request_id=meta.remote.request_id,
                local_xfer_side_handle=local_xfer_side_handle,
                remote_xfer_side_handle=remote_xfer_side_handle,
                notify=notify,
            )
            if handle is not None:
                out.append(handle)
        return out

    def _xfer_blocks(
        self,
        read_spec: ReadSpec,
        dst_engine_id: str,
        request_id: str,
        remote_request_id: str,
        local_xfer_side_handle: int,
        remote_xfer_side_handle: int,
        notify: bool = True,
    ) -> int | None:
        """Post a WRITE point-to-point xfer request.

        Returns the in-flight transfer handle (so the caller can track all of
        a request's handles atomically), or `None` if nothing was submitted.
        """
        assert self.transfer_topo is not None
        remote_rank = read_spec.remote_rank
        local_block_ids = read_spec.local_block_ids
        remote_block_ids = read_spec.remote_block_ids

        remote_info = self.transfer_topo.get_engine_info(dst_engine_id)

        notif_id = f"{remote_request_id}:{self.world_size}".encode() if notify else b""

        if len(local_block_ids) == 0:
            logger.warning("No blocks to push for request %s", request_id)
            return None

        # Prefix caching: D allocated only uncached blocks, so on a partial hit it
        # sends fewer than P's. End-trim P's blocks to that same suffix so we WRITE only
        # the uncomputed tail into D's slots. Runs on kernel ids, post-expansion.
        remote_block_ids, local_block_ids = self._apply_prefix_caching(
            decode_block_ids=remote_block_ids,
            prefill_block_ids=local_block_ids,
            decode_physical_per_logical=remote_info.remote_physical_blocks_per_logical,
            prefill_physical_per_logical=self._physical_blocks_per_logical_kv_block,
        )

        local_block_ids = list(local_block_ids)
        remote_block_ids = list(remote_block_ids)
        assert len(local_block_ids) == len(remote_block_ids), (
            f"push group-count mismatch for {request_id}: {len(local_block_ids)} "
            f"local vs {len(remote_block_ids)} remote groups"
        )
        for i in range(len(local_block_ids)):
            assert len(local_block_ids[i]) == len(remote_block_ids[i]), (
                f"push block-count mismatch for {request_id} group {i}: "
                f"{len(local_block_ids[i])} local vs "
                f"{len(remote_block_ids[i])} remote blocks"
            )

        # Get descs ids. Pair k writes local region pairs.local[k] into remote
        # region pairs.remote[k] at its cache group's blocks.
        stage = self._single_remote_stage(dst_engine_id)
        pairs = stage.pairs
        remote_block_descs_ids = self._compute_desc_ids(
            block_ids=remote_block_ids,
            dst_num_blocks=self.dst_num_blocks[dst_engine_id],
            block_size_ratio=None,
            physical_blocks_per_logical=remote_info.remote_physical_blocks_per_logical,
            region_num_blocks=stage.region_num_blocks,
            pairs=(pairs.remote, pairs.group),
            ssm_regions=pairs.remote_ssm_regions,
        )
        local_block_descs_ids = self._compute_desc_ids(
            block_ids=local_block_ids,
            dst_num_blocks=self.dst_num_blocks[self.engine_id],
            block_size_ratio=None,
            physical_blocks_per_logical=self._physical_blocks_per_logical_kv_block,
            region_num_blocks=self.region_num_blocks,
            pairs=(pairs.local, pairs.group),
        )

        assert len(local_block_descs_ids) == len(remote_block_descs_ids)

        handle = None
        try:
            handle = self.nixl_wrapper.make_prepped_xfer(
                "WRITE",
                local_xfer_side_handle,
                local_block_descs_ids,
                remote_xfer_side_handle,
                remote_block_descs_ids,
                notif_msg=notif_id,
            )
            self.nixl_wrapper.transfer(handle)
            # Caller tracks the handle (atomically with the request's other
            # writes) so P can free blocks once all of them are done.
            return handle
        except Exception as e:
            self._log_failure(
                failure_type="transfer_setup_failed",
                req_id=request_id,
                msg="Push WRITE submission failed; releasing handle",
                error=e,
                dst_engine_id=dst_engine_id,
                remote_rank=remote_rank,
            )
            # On the P side this WRITE failure is purely outbound; we
            # don't have a ``_recving_metadata`` entry to invalidate, so
            # we just release the handle and let the engine reschedule
            # via the lease / watchdog.
            with self._sending_transfers_lock:
                self._send_failures.add(request_id)
            if not self._handle_failed_transfer(request_id, handle):
                return handle
            return None

    # --- Notification handling on engine main thread ------------------ #

    def _get_new_notifs(self) -> set[str]:
        """Drain HB / completion notifs forwarded by the writer thread.

        The writer owns `nixl_wrapper.get_new_notifs` for push; PUSH_REG
        notifs are handled there. P learns of finished pushes from its WRITE
        handles, so this never reports finished sends.
        """
        while True:
            try:
                notif = self._pending_completion_notifs.get_nowait()
            except queue.Empty:
                break

            msg = notif.decode("utf-8")
            if msg.startswith("HB:"):
                self._handle_heartbeat(msg[3:])
                continue

            req_id, tp_size = msg.rsplit(":", 1)

            if (meta := self._recving_metadata.get(req_id)) is None:
                # The lease may have expired before the notif arrived.
                self.xfer_stats.record_notification_after_expiry()
                logger.error(
                    "Unrecognized request %s notif (may have expired).", req_id
                )
                continue
            # Consumer waits for one notif per producer rank writing
            # here: pp_size stages * producers-per-consumer (>1 when
            # producer TP > consumer TP; tp_size is the producer TP).
            producers_per_consumer = max(1, int(tp_size) // self.world_size)
            expected_notifs = meta.pp_size * producers_per_consumer
            self.consumer_notification_counts_by_req[req_id] += 1
            if self.consumer_notification_counts_by_req[req_id] < expected_notifs:
                continue
            del self.consumer_notification_counts_by_req[req_id]
            # P drove the transfer (we own no NIXL handle), so
            # materialise an empty ``_recving_transfers`` entry for
            # ``_pop_done_transfers`` to report done.
            self._recving_transfers.setdefault(req_id, [])
        return set()

    def get_transfer_results(self) -> KVConnectorTransferResults:
        # Engine main thread asking for completions: also wake the writer
        # so it gets a chance to drain NIXL notifs (heartbeats, completion
        # notifs, late PUSH_REGs) even if it had been parked.
        self._push_writer_wake.set()

        results = super().get_transfer_results()
        done_sending = results.finished_sending

        # ``_pop_done_transfers`` mutates ``_sending_transfers``; the
        # writer thread also appends to it, so guard the pop.
        with self._sending_transfers_lock:
            done_pushing, failed_pushing = self._pop_done_transfers(
                self._sending_transfers
            )
            # Remember failures until the final sibling WRITE completes.
            self._send_failures.update(failed_pushing)
            successful = {
                req_id
                for req_id in done_pushing - self._send_failures
                if req_id in self._reqs_to_send or req_id in self._reqs_to_process
            }
            self._send_failures.difference_update(done_pushing | done_sending)
        # Expired requests were already reported, even if their WRITEs finish later.
        done_pushing = successful
        for req_id in done_pushing:
            self._reqs_to_send.pop(req_id, None)
            self._reqs_to_process.discard(req_id)
            done_sending.add(req_id)

        # Tell the writer to drop any state it still holds for any
        # request that just finished (push completed) or expired
        # (lease ran out without a D registration ever arriving).
        for req_id in done_sending:
            self._evict_finished_inbox.put(req_id)
        if done_sending:
            self._push_writer_wake.set()

        return results
