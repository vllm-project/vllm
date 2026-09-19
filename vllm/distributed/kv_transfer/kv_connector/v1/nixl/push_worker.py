# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Push-specific (WRITE) worker-side logic for the NIXL connector.

A dedicated ``nixl-push-writer`` thread owns all push-related NIXL ops:
calls ``get_new_notifs`` (routing PUSH_REG internally; HB / completion
notifs are forwarded to the engine main thread), sends PUSH_REG via
``send_notif``, matches D registrations with P finished blocks, and
issues WRITE transfers via ``make_prepped_xfer`` / ``transfer``.

The engine main thread feeds the writer through queues:
``_reg_send_inbox`` (D-side regs to send), ``_finished_blocks_inbox``
(P-side blocks from metadata) and ``_pending_completion_notifs``
(non-PUSH_REG notifs forwarded back for HB / completion accounting).
The handshake-completion callback feeds ``_deferred_push_inbox`` with
matched pushes whose P→D handshake has finished so the writer can
(re-)issue the WRITE without ever blocking on the network.

Wake model: the writer self-polls every
``_PUSH_WRITER_POLL_INTERVAL_MS`` only while it has unmatched
``_push_finished_blocks`` (i.e. P-side blocks waiting for a D PUSH_REG
notif that has no other wake source). All other progress is
event-driven: the engine main thread sets ``_push_writer_wake`` from
``start_load_kv`` (when handing it new work) and from ``get_finished``
(so each engine step gives the writer a chance to drain NIXL notifs);
the handshake-completion callback sets the same event after a deferred
PUSH_REG send or a deferred push WRITE has been queued. When a request's
lease expires (the base worker reports it via ``done_sending``) or the WRITE completes,
``get_finished`` enqueues an eviction onto ``_evict_finished_inbox`` so
the writer drops any leftover ``_push_finished_blocks`` /
``_pending_d_registrations`` and stops self-polling.
"""

import contextlib
import queue
import threading
import time
from collections import defaultdict
from concurrent.futures import Future
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

import msgspec
import torch

from vllm import envs
from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorTransferResults,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    NixlBaseConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    PUSH_REG_NOTIF_PREFIX,
    NixlConnectorMetadata,
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
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)

# Writer-thread poll cadence while there is in-flight push state. When
# fully idle, the writer blocks on a wake event signalled by the engine
# main thread (start_load_kv / get_finished). Smaller -> lower latency
# while active, slightly more CPU.
_PUSH_WRITER_POLL_INTERVAL_MS = 1.0

# Per-layer write task handed from the forward thread to the writer:
# (req_id, layer_idx, cuda_event, enqueue_time). ``cuda_event`` is ``None``
# for completeness-sweep tasks, which are enqueued post-forward when all KV
# is already resident and therefore need no gating.
LayerWriteTask = tuple[ReqId, int, torch.cuda.Event | None, float]


@dataclass
class _LayerPushPlan:
    """Cached per-request WRITE geometry for the layer-wise push path.

    The descriptor lists cover every region of every layer in region-major
    order, so layer ``i``'s slice is ``[i * span, (i + 1) * span)`` with
    ``span = regions_per_layer * num_blocks``.
    """

    local_xfer_handle: int
    remote_xfer_handle: int
    local_descs: list[int]
    remote_descs: list[int]
    num_blocks: int
    regions_per_layer: int
    decode_engine_id: str
    notif_id: bytes


class _PlanFallback(Enum):
    """Sentinel for a transfer shape the layer-wise path cannot serve; the
    request is failed and its blocks are freed by the KV lease."""

    UNSUPPORTED = auto()


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
        self._finished_blocks_inbox: queue.Queue[tuple[str, BlockIds]] = queue.Queue()
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

        # --- Layer-wise overlapped WRITE-push state (opt-in) ------------- #
        # When disabled (the default) the push connector behaves
        # byte-identically to a single monolithic WRITE fired from
        # ``request_finished``. When enabled, the producer posts one WRITE per
        # layer as each layer's KV becomes ready (gated on a per-layer CUDA
        # event), overlapping the transfer with the tail of prefill compute,
        # then sends a single completion notif once all layers have landed.
        # The union of per-layer descriptor slices is byte-identical to the
        # monolithic descriptor list (region-major layout), so KV correctness
        # is preserved as long as every layer is written and the consumer
        # waits for the single completion notif.
        self._layerwise = envs.VLLM_NIXL_LAYERWISE_PUSH
        # Seconds a per-layer write task may wait for the D registration /
        # handshake before the request is failed (blocks freed via lease; the
        # D watchdog fails it).
        self._lw_defer_timeout = envs.VLLM_NIXL_LAYERWISE_DEFER_TIMEOUT
        # Ordered layer names as registered (index == region group order).
        self._lw_layer_names: list[str] = []
        self._lw_layer_index: dict[str, int] = {}
        # Forward-thread -> writer: per-layer write tasks.
        self._lw_task_q: queue.Queue[LayerWriteTask] = queue.Queue()
        # Writer-owned: tasks awaiting a D registration / handshake.
        self._lw_deferred: list[LayerWriteTask] = []
        # Per-req cached matched registration + built transfer plan.
        self._lw_reg: dict[ReqId, dict[str, Any]] = {}
        self._lw_plan: dict[ReqId, _LayerPushPlan | _PlanFallback] = {}
        # Guards the accounting dicts shared between forward + writer + main.
        self._lw_lock = threading.Lock()
        # Logical local (prefill) block ids per req (grouped).
        self._lw_local_blocks: dict[ReqId, BlockIds] = {}
        # Layers scheduled (by forward thread) per req.
        self._lw_scheduled_layers: dict[ReqId, set[int]] = defaultdict(set)
        # In-flight WRITE handles per req (appended by writer, drained by main).
        self._lw_handles: dict[ReqId, list[TransferHandle]] = defaultdict(list)
        # Completed WRITE count per req (incremented by main on DONE).
        self._lw_done: dict[ReqId, int] = defaultdict(int)
        # Expected WRITE count per req, sealed after the forward pass.
        self._lw_expected: dict[ReqId, int] = {}
        self._lw_sealed: set[ReqId] = set()
        self._lw_notified: set[ReqId] = set()
        self._lw_failed: set[ReqId] = set()

    # --- Lifecycle ----------------------------------------------------- #

    def register_kv_caches(self, kv_caches: dict[str, "torch.Tensor"]):
        super().register_kv_caches(kv_caches)
        if self._mixed_mem_types:
            raise NotImplementedError(
                "NixlPushConnector does not support mixed-memory KV caches"
            )
        if self._layerwise:
            self._lw_layer_names = list(kv_caches.keys())
            self._lw_layer_index = {
                name: i for i, name in enumerate(self._lw_layer_names)
            }
            logger.info(
                "NIXL layer-wise push ENABLED: %d layers, %d regions",
                len(self._lw_layer_names),
                self.num_regions,
            )
        if self._push_writer_thread is None:
            self._push_writer_thread = threading.Thread(
                target=self._push_writer_loop,
                daemon=True,
                name="nixl-push-writer",
            )
            self._push_writer_thread.start()
            logger.info("nixl-push-writer thread started (rank=%d)", self.tp_rank)

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
                self._finished_blocks_inbox.put((req_id, block_ids))
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

                # 3. P-side finished blocks; match against pending regs.
                while True:
                    try:
                        rid, blocks = self._finished_blocks_inbox.get_nowait()
                    except queue.Empty:
                        break
                    matched = self._pop_matching_registration(rid)
                    if matched is not None:
                        self._do_start_push_kv(rid, blocks, matched)
                    else:
                        self._push_finished_blocks[rid] = blocks

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

                # 4. NIXL notifs: route PUSH_REG; forward the rest.
                for notifs in self.nixl_wrapper.get_new_notifs().values():
                    for notif in notifs:
                        if notif.startswith(PUSH_REG_NOTIF_PREFIX):
                            self._handle_push_reg_notif(notif)
                        else:
                            self._pending_completion_notifs.put(notif)

                # 4. Layer-wise WRITE-push: drain new per-layer tasks and
                # retry deferred ones (awaiting a D registration/handshake).
                if self._layerwise:
                    self._lw_drain_tasks()
                    self._lw_process_deferred()
            except Exception:
                logger.exception("nixl-push-writer error; continuing")

            # Self-poll while there is in-flight state with no other wake
            # source: P-side finished blocks awaiting a PUSH_REG match, or
            # layer-wise tasks deferred until a registration arrives.
            if self._push_finished_blocks or (self._layerwise and self._lw_deferred):
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

        logger.debug(
            "NIXL lw[P] PUSH_REG recv rid=%s layerwise=%s deferred=%d sched_reqs=%d",
            rid,
            self._layerwise,
            len(self._lw_deferred),
            len(self._lw_scheduled_layers),
        )

        match = self._pop_matching_finished_blocks(rid)
        if match is not None:
            fin_id, blocks = match
            self._do_start_push_kv(fin_id, blocks, reg_data)
        else:
            self._pending_d_registrations[rid] = reg_data

    # --- D-side registration send (writer thread) ---------------------- #

    def _send_registration_to_p(
        self,
        req_id: str,
        reg_data: dict[str, Any],
    ) -> None:
        """Handshake (if needed) then send PUSH_REG. ``send_notif`` always
        executes on the writer; the handshake runs on the background executor
        and the request is re-queued onto ``_reg_send_inbox`` once it
        completes (at which point ``_ensure_handshake`` returns ``None`` and we
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

        Same lookup as ``_pop_matching_registration``: exact key, then a
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
        WRITE, and re-drive this request via ``_deferred_push_inbox`` once
        the handshake resolves -- so the writer thread never blocks on the
        network (mirrors ``_send_registration_to_p``).
        """
        if not local_block_ids:
            logger.warning("No local blocks to push for request %s", request_id)
            return

        # ``local_block_ids`` are P's logical block IDs; ``remote_block_ids``
        # (D's, from the PUSH_REG notif) are also logical.
        decode_engine_id = registration_data["decode_engine_id"]
        remote_block_ids = registration_data["local_block_ids"]
        decode_request_id = registration_data["request_id"]

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
        # Keep the engine alive while it is actively receiving pushes, mirroring
        # how pull-mode transfers touch _engine_last_active in start_load_kv.
        self._engine_last_active[decode_engine_id] = time.perf_counter()

        logical_local = self._as_grouped_block_ids(local_block_ids)
        logical_remote = self._as_grouped_block_ids(remote_block_ids)
        physical_local = self._logical_to_kernel_block_ids(
            logical_local, self._physical_blocks_per_logical_kv_block
        )

        push_meta = ReqMeta(
            local_block_ids=logical_local,
            local_physical_block_ids=physical_local,
            tp_size=self.world_size,
            remote=RemoteMeta(
                block_ids=logical_remote,
                host="",
                port=0,
                engine_id=decode_engine_id,
                request_id=decode_request_id,
            ),
        )

        t0 = time.perf_counter()
        self._xfer_blocks_for_req(req_id=request_id, meta=push_meta)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if elapsed_ms > 200.0:
            logger.warning(
                "_do_start_push_kv for %s took %.1fms (slow NIXL submission)",
                request_id,
                elapsed_ms,
            )

    @staticmethod
    def _as_grouped_block_ids(block_ids: BlockIds) -> BlockIds:
        """Normalise a sequence of block IDs to a tuple-of-groups shape.

        ``BlockIds`` is canonically a tuple of per-group lists, but some
        registration payloads collapse a single-group case to a flat
        list. Re-wrap that case so downstream group-aware helpers see a
        consistent shape."""
        if block_ids and not isinstance(block_ids[0], (list, tuple)):
            return (list(block_ids),)
        return block_ids

    def _xfer_blocks_for_req(self, req_id: str, meta: ReqMeta):
        """Issue WRITE transfers to one or more remote TP ranks."""
        assert meta.remote is not None and self.transfer_topo is not None
        engine_id = meta.remote.engine_id
        plan = self.tp_mappings[engine_id]
        remote_info = self.transfer_topo.get_engine_info(engine_id)
        tp_ratio = self.transfer_topo.tp_ratio(remote_info.remote_tp_size)

        # Expand D's logical IDs using the ratio learned during the
        # NIXL handshake. ``meta`` is freshly built by
        # ``_do_start_push_kv`` so mutating it here is safe.
        meta.remote.block_ids = self._logical_to_kernel_block_ids(
            meta.remote.block_ids,
            remote_info.remote_physical_blocks_per_logical,
        )
        remote_block_ids = meta.remote.block_ids
        local_block_ids = meta.local_physical_block_ids
        local_region_groups = self.region_group_ids
        remote_region_groups = self.dst_region_group_ids[engine_id]
        groups_differ = local_region_groups != remote_region_groups
        if groups_differ and not self._transfer_layer_group_ids:
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
            write_ranks = sorted(self.dst_xfer_side_handles[engine_id])
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

        handles: list[int] = []
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

            remote_xfer_side_handle = self.dst_xfer_side_handles[meta.remote.engine_id][
                spec.remote_rank
            ]

            handle = self._xfer_blocks(
                read_spec=spec,
                request_id=req_id,
                dst_engine_id=meta.remote.engine_id,
                remote_request_id=meta.remote.request_id,
                local_xfer_side_handle=local_xfer_side_handle,
                remote_xfer_side_handle=remote_xfer_side_handle,
            )
            if handle is not None:
                handles.append(handle)

        # Publish all the request's WRITE handles in one locked update: a
        # partial set would let ``_pop_done_transfers`` finish the request
        # early, then double-report it as the remaining writes land.
        if handles:
            with self._sending_transfers_lock:
                self._sending_transfers[req_id].extend(handles)

    def _xfer_blocks(
        self,
        read_spec: ReadSpec,
        dst_engine_id: str,
        request_id: str,
        remote_request_id: str,
        local_xfer_side_handle: int,
        remote_xfer_side_handle: int,
    ) -> int | None:
        """Post a WRITE point-to-point xfer request.

        Returns the in-flight transfer handle (so the caller can track all of
        a request's handles atomically), or ``None`` if nothing was submitted.
        """
        assert self.transfer_topo is not None
        remote_rank = read_spec.remote_rank
        local_block_ids = read_spec.local_block_ids
        remote_block_ids = read_spec.remote_block_ids

        remote_info = self.transfer_topo.get_engine_info(dst_engine_id)
        block_size_ratio = self.transfer_topo.block_size_ratio(
            remote_info.remote_block_size
        )
        if block_size_ratio > 1:
            local_block_ids, remote_block_ids = (
                self._map_block_ids_for_block_size_ratio(
                    local_block_ids, remote_block_ids, block_size_ratio
                )
            )

        notif_id = f"{remote_request_id}:{self.world_size}".encode()

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

        # Get descs ids.
        remote_block_descs_ids = self._compute_desc_ids(
            block_ids=remote_block_ids,
            dst_num_blocks=self.dst_num_blocks[dst_engine_id],
            block_size_ratio=None,
            physical_blocks_per_logical=remote_info.remote_physical_blocks_per_logical,
            region_num_blocks=self.dst_region_num_blocks[dst_engine_id],
            region_group_ids=self.dst_region_group_ids[dst_engine_id],
            uses_region_group_mapping=self.dst_uses_region_group_mapping[dst_engine_id],
        )
        local_block_descs_ids = self._compute_desc_ids(
            block_ids=local_block_ids,
            dst_num_blocks=self.dst_num_blocks[self.engine_id],
            block_size_ratio=block_size_ratio,
            physical_blocks_per_logical=self._physical_blocks_per_logical_kv_block,
            region_num_blocks=self.dst_region_num_blocks[self.engine_id],
            region_group_ids=self.region_group_ids,
            uses_region_group_mapping=self._uses_region_group_mapping,
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
            if not self._handle_failed_transfer(request_id, handle):
                return handle
            return None

    # --- Layer-wise overlapped WRITE-push (opt-in) -------------------- #

    def save_kv_layer_push(
        self,
        metadata: NixlConnectorMetadata,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
    ) -> None:
        """Forward-thread hook: schedule one WRITE per (req, layer).

        Called by the connector facade for every attention layer during the
        producer forward. For each request the scheduler flagged as
        finishing prefill this step (``metadata.reqs_to_save``), record a
        CUDA event marking this layer's KV as ready and hand a per-layer
        write task to the ``nixl-push-writer`` thread. Cheap on the main
        thread: only an event record + enqueue; the transfer and the
        ``event.synchronize()`` happen off-thread on the writer.
        """
        if not self._layerwise or not metadata.reqs_to_save:
            return
        layer_idx = self._lw_layer_index.get(layer_name)
        if layer_idx is None:
            return
        event = torch.cuda.Event()
        event.record()
        now = time.perf_counter()
        woke = False
        for req_id, meta in metadata.reqs_to_save.items():
            with self._lw_lock:
                if req_id in self._lw_failed:
                    continue
                if layer_idx in self._lw_scheduled_layers[req_id]:
                    continue
                self._lw_scheduled_layers[req_id].add(layer_idx)
                # Cache the logical local block ids (same every layer/step).
                if req_id not in self._lw_reg and req_id not in self._lw_plan:
                    self._lw_local_blocks[req_id] = meta.local_block_ids
            self._lw_task_q.put((req_id, layer_idx, event, now))
            woke = True
        if woke:
            self._push_writer_wake.set()

    def seal_layer_writes_push(self, metadata: NixlConnectorMetadata) -> None:
        """Main thread, after the forward: seal expected per-layer counts.

        ``save_kv_layer_push`` has now fired for every layer of this forward,
        so the scheduled-layer set is complete. Freeze the expected WRITE
        count so ``get_finished`` can detect completion.
        """
        if not self._layerwise or not metadata.reqs_to_save:
            return
        # Completeness sweep: any region NOT covered by a matching per-layer
        # save_kv_layer call (e.g. GLM DSA indexer caches whose layer_name is
        # not in _lw_layer_index) would otherwise NEVER be written, leaving
        # stale KV on D -> garbage output. Post-forward all KV is resident, so
        # enqueue the missing region indices with event=None (no gating). This
        # guarantees byte-complete KV; covered layers still overlap during the
        # forward. If coverage is already full, missing is empty (no-op).
        num_layers = len(self._lw_layer_names)
        now = time.perf_counter()
        with self._lw_lock:
            for req_id in metadata.reqs_to_save:
                if req_id in self._lw_failed:
                    continue
                covered = set(self._lw_scheduled_layers[req_id])
                missing = [i for i in range(num_layers) if i not in covered]
                for idx in missing:
                    self._lw_scheduled_layers[req_id].add(idx)
                    self._lw_task_q.put((req_id, idx, None, now))
                self._lw_expected[req_id] = len(self._lw_scheduled_layers[req_id])
                self._lw_sealed.add(req_id)
                logger.debug(
                    "NIXL layerwise seal: req %s covered=%d swept=%d total=%d",
                    req_id,
                    len(covered),
                    len(missing),
                    num_layers,
                )
        self._push_writer_wake.set()

    def _lw_drain_tasks(self) -> None:
        while True:
            try:
                task = self._lw_task_q.get_nowait()
            except queue.Empty:
                break
            self._lw_run_task(task)

    def _lw_process_deferred(self) -> None:
        if not self._lw_deferred:
            return
        now = time.perf_counter()
        still: list[LayerWriteTask] = []
        for task in self._lw_deferred:
            req_id, _, _, enqueue_time = task
            if req_id in self._lw_failed:
                continue
            if now - enqueue_time > self._lw_defer_timeout:
                logger.error(
                    "NIXL layer-wise push: req %s deferred >%.0fs waiting for "
                    "D registration; failing (blocks freed via lease).",
                    req_id,
                    self._lw_defer_timeout,
                )
                self._lw_mark_failed(req_id)
                continue
            if not self._lw_run_task(task, allow_defer=False):
                still.append(task)
        self._lw_deferred = still

    def _lw_run_task(self, task: LayerWriteTask, allow_defer: bool = True) -> bool:
        """Execute (or defer) a single per-layer write task on the writer.

        Returns True if the task was handled (executed or the req is
        failed/terminal), False if it must remain deferred.
        """
        req_id, layer_idx, event, _ = task
        if req_id in self._lw_failed:
            return True

        plan = self._lw_plan.get(req_id)
        if plan is None:
            # Need a matched D registration + built transfer plan first.
            reg = self._lw_reg.get(req_id)
            if reg is None:
                reg = self._pop_matching_registration(req_id)
                if reg is None:
                    if allow_defer:
                        self._lw_deferred.append(task)
                    return False
                self._lw_reg[req_id] = reg
            plan = self._lw_build_plan(req_id, reg)
            if plan is None:
                # Handshake not ready yet (or failed inside). Retry later.
                if allow_defer:
                    self._lw_deferred.append(task)
                return False
            self._lw_plan[req_id] = plan
            logger.debug(
                "NIXL lw[P] plan ready req=%s unsupported=%s",
                req_id,
                plan is _PlanFallback.UNSUPPORTED,
            )

        if plan is _PlanFallback.UNSUPPORTED:
            logger.error(
                "NIXL layer-wise push: unsupported transfer shape for req %s; "
                "failing (blocks freed via lease).",
                req_id,
            )
            self._lw_mark_failed(req_id)
            return True

        try:
            self._lw_write_layer(req_id, plan, layer_idx, event)
        except Exception:
            logger.exception(
                "NIXL layer-wise push: WRITE failed for req %s layer %d",
                req_id,
                layer_idx,
            )
            self._lw_mark_failed(req_id)
        return True

    def _lw_build_plan(
        self, req_id: str, registration_data: dict[str, Any]
    ) -> _LayerPushPlan | _PlanFallback | None:
        """Build a cached per-request transfer plan (simple TP/MLA case).

        Returns a plan, ``_PlanFallback.UNSUPPORTED`` if the transfer shape is
        not supported by the layer-wise path, or ``None`` if the P->D
        handshake is not ready yet (retry later)."""
        decode_engine_id = registration_data["decode_engine_id"]
        logical_local = self._as_grouped_block_ids(
            self._lw_local_blocks.get(req_id, ())
        )
        remote_logical = self._as_grouped_block_ids(
            registration_data["local_block_ids"]
        )
        if not logical_local or not remote_logical:
            logger.debug(
                "NIXL lw[P] build_plan FALLBACK req=%s reason=empty_blocks "
                "local=%s remote=%s",
                req_id,
                bool(logical_local),
                bool(remote_logical),
            )
            return _PlanFallback.UNSUPPORTED

        # ``_ensure_handshake`` returns ``None`` once the P->D handshake has
        # completed (transfer descriptors ready) and a ``Future`` while it is
        # still pending. Mirror the stock ``_do_start_push_kv`` contract: a
        # non-None future means "not ready yet" -> defer and retry (the writer
        # re-drives deferred tasks every poll until the future resolves). The
        # handshake is initiated on the base worker's background executor by
        # this call; repeated calls are idempotent (guarded by
        # ``_handshake_futures`` / ``_remote_agents``).
        fut = self._ensure_handshake(
            decode_engine_id,
            registration_data["decode_host"],
            registration_data["decode_port"],
            registration_data["decode_tp_size"],
        )
        if fut is not None:
            logger.debug(
                "NIXL lw[P] build_plan DEFER req=%s reason=handshake_not_ready "
                "dec_eng=%s",
                req_id,
                decode_engine_id,
            )
            return None
        if (
            self.transfer_topo is None
            or decode_engine_id not in self.tp_mappings
            or decode_engine_id not in self.dst_xfer_side_handles
        ):
            logger.debug(
                "NIXL lw[P] build_plan DEFER req=%s reason=topo_not_ready "
                "topo=%s in_tpmap=%s in_dstxfer=%s",
                req_id,
                self.transfer_topo is not None,
                decode_engine_id in self.tp_mappings,
                decode_engine_id in self.dst_xfer_side_handles,
            )
            return None

        remote_info = self.transfer_topo.get_engine_info(decode_engine_id)
        tp_ratio = self.transfer_topo.tp_ratio(remote_info.remote_tp_size)
        plan_map = self.tp_mappings[decode_engine_id]
        # Layer-wise path supports the homogeneous single-group case
        # (our TP=1 MLA deployment). Anything else falls back.
        if (
            len(logical_local) != 1
            or len(remote_logical) != 1
            or tp_ratio < 1
            or len(plan_map.all_source_ranks) != 1
        ):
            logger.debug(
                "NIXL lw[P] build_plan FALLBACK req=%s reason=shape "
                "n_local=%d n_remote=%d tp_ratio=%s src_ranks=%d",
                req_id,
                len(logical_local),
                len(remote_logical),
                tp_ratio,
                len(plan_map.all_source_ranks),
            )
            return _PlanFallback.UNSUPPORTED
        remote_block_size = remote_info.remote_block_size
        block_size_ratio = self.transfer_topo.block_size_ratio(remote_block_size)
        if block_size_ratio != 1:
            logger.debug(
                "NIXL lw[P] build_plan FALLBACK req=%s reason=block_size_ratio=%s",
                req_id,
                block_size_ratio,
            )
            return _PlanFallback.UNSUPPORTED

        # Expand logical block ids to kernel (physical) ids using the same
        # helper the stock WRITE path uses (``_do_start_push_kv`` /
        # ``_xfer_blocks_for_req``): local uses this worker's ratio, remote
        # uses the ratio learned for the decode engine over the handshake.
        physical_local = self._logical_to_kernel_block_ids(
            logical_local, self._physical_blocks_per_logical_kv_block
        )
        remote_physical = self._logical_to_kernel_block_ids(
            remote_logical, remote_info.remote_physical_blocks_per_logical
        )
        local0 = list(physical_local[0])
        remote0 = list(remote_physical[0])
        n = min(len(local0), len(remote0))
        if n == 0:
            return _PlanFallback.UNSUPPORTED
        local0 = local0[:n]
        remote0 = remote0[:n]

        remote_rank = plan_map.all_source_ranks[0]
        local_xfer_side_handle = self.src_xfer_handles_by_block_size[remote_block_size]
        remote_xfer_side_handle = self.dst_xfer_side_handles[decode_engine_id][
            remote_rank
        ]

        local_descs = self._compute_desc_ids(
            block_ids=[local0],
            dst_num_blocks=self.dst_num_blocks[self.engine_id],
            block_size_ratio=block_size_ratio,
            physical_blocks_per_logical=self._physical_blocks_per_logical_kv_block,
        ).tolist()
        remote_descs = self._compute_desc_ids(
            block_ids=[remote0],
            dst_num_blocks=self.dst_num_blocks[decode_engine_id],
            block_size_ratio=None,
            physical_blocks_per_logical=remote_info.remote_physical_blocks_per_logical,
        ).tolist()
        num_regions = self.num_regions
        num_blocks = len(local0)
        num_layers = len(self._lw_layer_names)
        if (
            len(local_descs) != len(remote_descs)
            or num_regions == 0
            or num_layers == 0
            or num_regions % num_layers != 0
            or len(local_descs) != num_regions * num_blocks
        ):
            logger.debug(
                "NIXL lw[P] build_plan FALLBACK req=%s reason=desc_geometry "
                "n_local_descs=%d n_remote_descs=%d num_regions=%d num_layers=%d "
                "num_blocks=%d",
                req_id,
                len(local_descs),
                len(remote_descs),
                num_regions,
                num_layers,
                num_blocks,
            )
            return _PlanFallback.UNSUPPORTED
        regions_per_layer = num_regions // num_layers
        logger.debug(
            "NIXL lw[P] build_plan OK req=%s num_regions=%d num_layers=%d "
            "num_blocks=%d regions_per_layer=%d n_descs=%d",
            req_id,
            num_regions,
            num_layers,
            num_blocks,
            regions_per_layer,
            len(local_descs),
        )

        decode_request_id = registration_data["request_id"]
        return _LayerPushPlan(
            local_xfer_handle=local_xfer_side_handle,
            remote_xfer_handle=remote_xfer_side_handle,
            local_descs=local_descs,
            remote_descs=remote_descs,
            num_blocks=num_blocks,
            regions_per_layer=regions_per_layer,
            decode_engine_id=decode_engine_id,
            notif_id=f"{decode_request_id}:{self.world_size}".encode(),
        )

    def _lw_write_layer(
        self,
        req_id: str,
        plan: _LayerPushPlan,
        layer_idx: int,
        event: torch.cuda.Event | None,
    ) -> None:
        """Post one WRITE for a single layer's region slice (no notif).

        The layer's KV must be resident before the NIC reads it, so we block
        on its CUDA event first (same guard MoRI-IO uses to avoid a
        compute/transfer race). Sweep tasks pass event=None: they are enqueued
        post-forward when all KV is already resident, so no gating is needed."""
        if event is not None:
            event.synchronize()
        span = plan.regions_per_layer * plan.num_blocks
        start = layer_idx * span
        end = start + span
        local_descs = plan.local_descs[start:end]
        remote_descs = plan.remote_descs[start:end]
        if not local_descs:
            return
        handle = self.nixl_wrapper.make_prepped_xfer(
            "WRITE",
            plan.local_xfer_handle,
            local_descs,
            plan.remote_xfer_handle,
            remote_descs,
            notif_msg=b"",
        )
        self.nixl_wrapper.transfer(handle)
        with self._lw_lock:
            self._lw_handles[req_id].append(handle)

    def _lw_mark_failed(self, req_id: str) -> None:
        with self._lw_lock:
            self._lw_failed.add(req_id)
            handles = self._lw_handles.pop(req_id, [])
        for h in handles:
            with contextlib.suppress(Exception):
                self.nixl_wrapper.release_xfer_handle(h)
        self._lw_reg.pop(req_id, None)
        self._lw_plan.pop(req_id, None)

    def _lw_cleanup(self, req_id: str) -> None:
        with self._lw_lock:
            self._lw_scheduled_layers.pop(req_id, None)
            self._lw_handles.pop(req_id, None)
            self._lw_done.pop(req_id, None)
            self._lw_expected.pop(req_id, None)
            self._lw_sealed.discard(req_id)
        self._lw_reg.pop(req_id, None)
        self._lw_plan.pop(req_id, None)
        self._lw_local_blocks.pop(req_id, None)

    def _lw_get_finished(self, done_sending: set[str]) -> None:
        """Main thread: poll per-layer WRITE handles; on completion of all
        sealed layers, send the single completion notif to D and report the
        request as done-sending (freeing P blocks)."""
        with self._lw_lock:
            req_ids = set(self._lw_handles) | set(self._lw_expected)
        for req_id in req_ids:
            if req_id in self._lw_notified or req_id in self._lw_failed:
                continue
            with self._lw_lock:
                handles = self._lw_handles.get(req_id, [])
            still: list[TransferHandle] = []
            failed = False
            for handle in handles:
                try:
                    state = self.nixl_wrapper.check_xfer_state(handle)
                except Exception:
                    failed = True
                    continue
                if state == "DONE":
                    with contextlib.suppress(Exception):
                        res = self.nixl_wrapper.get_xfer_telemetry(handle)
                        self.xfer_stats.record_transfer(res)
                    self.nixl_wrapper.release_xfer_handle(handle)
                    with self._lw_lock:
                        self._lw_done[req_id] += 1
                elif state == "PROC":
                    still.append(handle)
                else:
                    failed = True
                    self.nixl_wrapper.release_xfer_handle(handle)
            with self._lw_lock:
                self._lw_handles[req_id] = still
                expected = self._lw_expected.get(req_id)
                done = self._lw_done.get(req_id, 0)
                sealed = req_id in self._lw_sealed
            if failed:
                logger.error(
                    "NIXL layer-wise push: WRITE state error for req %s; "
                    "failing (blocks freed via lease).",
                    req_id,
                )
                self._lw_mark_failed(req_id)
                continue
            if sealed and expected is not None and done >= expected and not still:
                logger.debug(
                    "NIXL lw[P] all writes DONE req=%s done=%d expected=%d "
                    "-> completion",
                    req_id,
                    done,
                    expected,
                )
                self._lw_send_completion(req_id)
                self._lw_notified.add(req_id)
                done_sending.add(req_id)
                self._lw_cleanup(req_id)
            elif sealed and expected is not None and (handles or done):
                logger.debug(
                    "NIXL lw[P] writes progress req=%s done=%d expected=%d inflight=%d",
                    req_id,
                    done,
                    expected,
                    len(still),
                )

    def _lw_send_completion(self, req_id: str) -> None:
        reg = self._lw_reg.get(req_id)
        plan = self._lw_plan.get(req_id)
        if reg is None or not isinstance(plan, _LayerPushPlan):
            return
        engine_id = plan.decode_engine_id
        notif_id = plan.notif_id
        agents = self._remote_agents.get(engine_id, {})
        logger.debug(
            "NIXL lw[P] send_completion req=%s engine=%s agents=%d notif=%s",
            req_id,
            engine_id,
            len(agents),
            notif_id,
        )
        for agent_name in agents.values():
            try:
                self.nixl_wrapper.send_notif(agent_name, notif_msg=notif_id)
            except Exception as e:
                self._log_failure(
                    failure_type="layerwise_completion_notif_failed",
                    req_id=req_id,
                    error=e,
                    dst_engine_id=engine_id,
                )

    # --- Notification handling on engine main thread ------------------ #

    def _get_new_notifs(self) -> set[str]:
        """Drain HB / completion notifs forwarded by the writer thread.

        The writer owns ``nixl_wrapper.get_new_notifs`` for push; PUSH_REG
        notifs are handled there. Everything else is forwarded here for
        existing accounting.
        """
        assert self.transfer_topo is not None
        notified_req_ids: set[str] = set()
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

            # Not tracked as a P-side send/process for this notif.
            if req_id not in self._reqs_to_send and req_id not in self._reqs_to_process:
                if (meta := self._recving_metadata.get(req_id)) is not None:
                    # Consumer waits for one notif per producer rank writing
                    # here: pp_size stages * producers-per-consumer (>1 when
                    # producer TP > consumer TP; tp_size is the producer TP).
                    producers_per_consumer = max(1, int(tp_size) // self.world_size)
                    expected_notifs = meta.pp_size * producers_per_consumer
                    self.consumer_notification_counts_by_req[req_id] += 1
                    notifs = self.consumer_notification_counts_by_req[req_id]
                    if notifs < expected_notifs:
                        continue
                    del self.consumer_notification_counts_by_req[req_id]
                    # P drove the transfer (we own no NIXL handle), so
                    # materialise an empty ``_recving_transfers`` entry for
                    # ``_pop_done_transfers`` to report done.
                    logger.debug(
                        "NIXL lw[D] completion notif recv req=%s -> mark recving done",
                        req_id,
                    )
                    self._recving_transfers.setdefault(req_id, [])
                else:
                    # Not tracked on either side (lease may have expired
                    # before the notif arrived). Log and skip.
                    logger.error(
                        "Unrecognized request %s notif (may have expired).",
                        req_id,
                    )
                continue

            n_consumers = int(tp_size)
            tp_ratio = self.transfer_topo.tp_ratio(n_consumers)
            consumers_per_producer = -tp_ratio if n_consumers > self.world_size else 1
            self.consumer_notification_counts_by_req[req_id] += 1
            if (
                self.consumer_notification_counts_by_req[req_id]
                == consumers_per_producer
            ):
                notified_req_ids.add(req_id)
                del self.consumer_notification_counts_by_req[req_id]
                self._reqs_to_process.remove(req_id)
                self._reqs_to_send.pop(req_id, None)
        return notified_req_ids

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
        # A failed send must never be reported as done: its blocks
        # are freed via the lease / watchdog instead.
        done_pushing = {
            req_id
            for req_id in done_pushing - failed_pushing
            if req_id in self._recving_metadata
        }
        for req_id in done_pushing:
            self._reqs_to_send.pop(req_id, None)
            self._reqs_to_process.discard(req_id)
            self.consumer_notification_counts_by_req.pop(req_id, None)
            done_sending.add(req_id)

        # Layer-wise push: detect per-layer WRITE completion and emit the
        # single completion notif to D once all sealed layers have landed.
        if self._layerwise:
            self._lw_get_finished(done_sending)
            for req_id in done_sending:
                self._reqs_to_send.pop(req_id, None)
                self._reqs_to_process.discard(req_id)

        # Tell the writer to drop any state it still holds for any
        # request that just finished (push completed) or expired
        # (lease ran out without a D registration ever arriving).
        for req_id in done_sending:
            self._evict_finished_inbox.put(req_id)
        if done_sending:
            self._push_writer_wake.set()

        return results
