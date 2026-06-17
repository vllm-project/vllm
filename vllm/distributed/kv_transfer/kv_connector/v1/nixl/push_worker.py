# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Push-specific (WRITE) worker-side logic for the NIXL connector.

A dedicated ``nixl-push-writer`` thread owns all push-related NIXL ops:
calls ``get_new_notifs`` (routing PUSH_REG internally; HB / completion
notifs are forwarded to the engine main thread), sends PUSH_REG via
``send_notif``, matches D registrations with P finished blocks, and
issues WRITE transfers via ``make_prepped_xfer`` / ``transfer``.

The engine main thread feeds the writer through three queues:
``_reg_send_inbox`` (D-side regs to send), ``_finished_blocks_inbox``
(P-side blocks from metadata) and ``_pending_completion_notifs``
(non-PUSH_REG notifs forwarded back for HB / completion accounting).

Wake model: the writer self-polls every
``_PUSH_WRITER_POLL_INTERVAL_MS`` only while it has unmatched
``_push_finished_blocks`` (i.e. P-side blocks waiting for a D PUSH_REG
notif that has no other wake source). All other progress is
event-driven: the engine main thread sets ``_push_writer_wake`` from
``start_load_kv`` (when handing it new work) and from ``get_finished``
(so each engine step gives the writer a chance to drain NIXL notifs);
the handshake-completion callback sets the same event after a deferred
PUSH_REG send has been queued. When a request's lease expires (the base
worker reports it via ``done_sending``) or the WRITE completes,
``get_finished`` enqueues an eviction onto ``_evict_finished_inbox`` so
the writer drops any leftover ``_push_finished_blocks`` /
``_pending_d_registrations`` and stops self-polling.
"""

import queue
import threading
import time
from collections import defaultdict
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any

import msgspec
import numpy as np

from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
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
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import ReadSpec
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import get_base_request_id
from vllm.logger import init_logger

if TYPE_CHECKING:
    import torch

    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)

# Writer-thread poll cadence while there is in-flight push state. When
# fully idle, the writer blocks on a wake event signalled by the engine
# main thread (start_load_kv / get_finished). Smaller -> lower latency
# while active, slightly more CPU.
_PUSH_WRITER_POLL_INTERVAL_MS = 1.0


class NixlPushConnectorWorker(NixlBaseConnectorWorker):
    """Push-specific (WRITE) worker logic. See module docstring."""

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

        # Wake signal from engine main thread (start_load_kv / get_finished).
        # Writer self-polls at _PUSH_WRITER_POLL_INTERVAL_MS while it has
        # active in-flight state; otherwise it blocks until signalled.
        self._push_writer_wake = threading.Event()

        self._push_writer_stop = threading.Event()
        self._push_writer_thread: threading.Thread | None = None

    # --- Lifecycle ----------------------------------------------------- #

    def register_kv_caches(self, kv_caches: dict[str, "torch.Tensor"]):
        super().register_kv_caches(kv_caches)
        if self._push_writer_thread is None:
            self._push_writer_thread = threading.Thread(
                target=self._push_writer_loop,
                daemon=True,
                name="nixl-push-writer",
            )
            self._push_writer_thread.start()
            logger.info("nixl-push-writer thread started (rank=%d)", self.tp_rank)

    def shutdown(self):
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
        # D-side: track reqs waiting for P to push.
        for req_id, meta in metadata.reqs_to_recv.items():
            meta.local_physical_block_ids = self._logical_to_kernel_block_ids(
                meta.local_block_ids
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
        for req_id, expiration_time in metadata.reqs_to_send.items():
            if req_id in self._reqs_to_process:
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

                # 2. P-side finished blocks; match against pending regs.
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

                # 2b. Evict finished blocks for requests that have either
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

                # 3. NIXL notifs: route PUSH_REG; forward the rest.
                for notifs in self.nixl_wrapper.get_new_notifs().values():
                    for notif in notifs:
                        if notif.startswith(PUSH_REG_NOTIF_PREFIX):
                            self._handle_push_reg_notif(notif)
                        else:
                            self._pending_completion_notifs.put(notif)
            except Exception:
                logger.exception("nixl-push-writer error; continuing")

            # Self-poll only while there is no other wake source: P-side
            # finished blocks waiting for a D PUSH_REG match. All other
            # progress is event-driven (see module docstring).
            if self._push_finished_blocks:
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
            # D never addresses P memory in push mode; just load P's agents.
            notif_agents_only=remote_pp_size > 1,
        )
        if fut is None:
            self._do_send_reg_notif(req_id, reg_data)
            return

        def _on_handshake(
            f: Future[dict[int | tuple[int, int], str]],
            rid: str = req_id,
            rd: dict[str, Any] = reg_data,
        ) -> None:
            try:
                f.result()
            except Exception as e:
                self._log_failure(
                    failure_type="push_reg_handshake_failed", req_id=rid, error=e
                )
                self._handle_failed_transfer(rid, None)
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
        agents = self._remote_agents.get(engine_id)
        if not agents:
            logger.error(
                "No remote agents for engine %s; cannot send registration for %s",
                engine_id,
                req_id,
            )
            self._handle_failed_transfer(req_id, None)
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

        ``local_block_ids`` are P's *logical* block IDs (from the P
        scheduler's metadata). ``registration_data["local_block_ids"]``
        are D's *logical* block IDs (from D's scheduler, sent over the
        PUSH_REG notif). All conversion to physical block IDs is
        deferred to ``_xfer_blocks_for_req`` so each side uses its own
        physical-blocks-per-logical ratio (P uses
        ``self._physical_blocks_per_logical_kv_block``; D's ratio is
        learned during the NIXL handshake)."""
        decode_engine_id = registration_data["decode_engine_id"]
        remote_block_ids = registration_data["local_block_ids"]
        decode_host = registration_data["decode_host"]
        decode_port = registration_data["decode_port"]
        decode_request_id = registration_data["request_id"]
        if not local_block_ids:
            logger.warning("No local blocks to push for request %s", request_id)
            return

        if not self._ensure_d_handshake(
            decode_engine_id,
            decode_host,
            decode_port,
            registration_data["decode_tp_size"],
            request_id,
        ):
            return

        # Both sides are kept in logical form here; ``_xfer_blocks_for_req``
        # expands each side using the appropriate ratio.
        logical_local = self._as_grouped_block_ids(local_block_ids)
        logical_remote = self._as_grouped_block_ids(remote_block_ids)
        physical_local = self._logical_to_kernel_block_ids(logical_local)

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

    def _ensure_d_handshake(
        self,
        decode_engine_id: str,
        decode_host: str,
        decode_port: int,
        decode_tp_size: int,
        request_id: str,
    ) -> bool:
        """First-time P→D handshake. Blocking call on the writer thread.

        Returns True iff the handshake succeeded (or had already been
        completed). Returns False if the handshake raised; the request is
        skipped in that case (the engine layer will reschedule or fail it
        via the standard lease/timeout path)."""
        if decode_engine_id in self._remote_agents:
            return True
        try:
            remote_agents = self._nixl_handshake(
                decode_host,
                decode_port,
                decode_tp_size,
                decode_engine_id,
            )
        except Exception:
            logger.exception(
                "Failed handshake to D %s for push %s",
                decode_engine_id,
                request_id,
            )
            return False
        with self._handshake_lock:
            self._remote_agents[decode_engine_id] = remote_agents
        logger.info(
            "Push handshake to D %s done (%d agents)",
            decode_engine_id,
            len(remote_agents),
        )
        return True

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
        meta.remote.block_ids = self._logical_to_remote_kernel_block_ids(
            meta.remote.block_ids,
            remote_info.remote_physical_blocks_per_logical,
        )
        remote_block_ids = meta.remote.block_ids
        local_block_ids = meta.local_physical_block_ids
        num_groups = len(local_block_ids)
        read_specs = [
            ReadSpec(
                remote_rank=rank,
                local_block_ids=[
                    list(local_block_ids[g])
                    if rank in plan.source_ranks_per_group[g]
                    else []
                    for g in range(num_groups)
                ],
                remote_block_ids=[
                    list(remote_block_ids[g])
                    if rank in plan.source_ranks_per_group[g]
                    else []
                    for g in range(num_groups)
                ],
            )
            for rank in plan.all_source_ranks
        ]

        if self.use_mla and tp_ratio < 0:
            # MLA latent is replicated across D's TP ranks: the tp-mapping
            # collapses to one rank (fine for reads), but push must WRITE every
            # D rank or the rest decode stale KV; only the dst differs per rank.
            assert len(read_specs) == 1
            base_spec = read_specs[0]
            read_specs = [
                ReadSpec(
                    remote_rank=rank,
                    local_block_ids=base_spec.local_block_ids,
                    remote_block_ids=base_spec.remote_block_ids,
                )
                for rank in self.dst_xfer_side_handles[meta.remote.engine_id]
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
            if tp_ratio < 0 and not self.use_mla:
                assert remote_block_size == self.block_size
                local_xfer_side_handle = self.src_xfer_handles_by_tp_ratio[tp_ratio][i]
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
            assert not self._is_hma_required
            local_block_ids0 = local_block_ids[0] if local_block_ids else []
            remote_block_ids0 = remote_block_ids[0]
            local_block_ids_mapped = self.get_mapped_blocks(
                np.asarray(local_block_ids0), block_size_ratio
            ).tolist()
            if len(local_block_ids_mapped) > len(remote_block_ids0):
                local_block_ids_mapped = local_block_ids_mapped[
                    : len(remote_block_ids0)
                ]
            local_block_ids = [local_block_ids_mapped] if local_block_ids_mapped else []
            remote_block_ids = [remote_block_ids0]

        notif_id = f"{remote_request_id}:{self.world_size}".encode()

        if len(local_block_ids) == 0:
            logger.warning("No blocks to push for request %s", request_id)
            return None

        # Align per-group block counts for push.
        local_block_ids = list(local_block_ids)
        remote_block_ids = list(remote_block_ids)
        for i in range(min(len(local_block_ids), len(remote_block_ids))):
            num_local = len(local_block_ids[i])
            num_remote = len(remote_block_ids[i])
            if num_local > num_remote:
                local_block_ids[i] = local_block_ids[i][:num_remote]
            elif num_local < num_remote:
                remote_block_ids[i] = remote_block_ids[i][:num_local]

        # Get descs ids.
        remote_block_descs_ids = self._compute_desc_ids(
            block_ids=remote_block_ids,
            dst_num_blocks=self.dst_num_blocks[dst_engine_id],
            block_size_ratio=None,
            physical_blocks_per_logical=remote_info.remote_physical_blocks_per_logical,
        )
        local_block_descs_ids = self._compute_desc_ids(
            block_ids=local_block_ids,
            dst_num_blocks=self.dst_num_blocks[self.engine_id],
            block_size_ratio=block_size_ratio,
            physical_blocks_per_logical=self._physical_blocks_per_logical_kv_block,
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
            if handle is not None:
                self.nixl_wrapper.release_xfer_handle(handle)
            self.xfer_stats.record_failed_transfer()
            return None

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
                    # D-side: P signalled push completion. Each of the
                    # producer's pp_size stages sends one notif; wait for all.
                    self.consumer_notification_counts_by_req[req_id] += 1
                    if self.consumer_notification_counts_by_req[req_id] < meta.pp_size:
                        continue
                    del self.consumer_notification_counts_by_req[req_id]
                    # P drove the transfer (we own no NIXL handle), so
                    # materialise an empty ``_recving_transfers`` entry for
                    # ``_pop_done_transfers`` to report done.
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

    def get_finished(self) -> tuple[set[str], set[str]]:
        # Engine main thread asking for completions: also wake the writer
        # so it gets a chance to drain NIXL notifs (heartbeats, completion
        # notifs, late PUSH_REGs) even if it had been parked.
        self._push_writer_wake.set()

        done_sending, done_recving = super().get_finished()

        # ``_pop_done_transfers`` mutates ``_sending_transfers``; the
        # writer thread also appends to it, so guard the pop.
        with self._sending_transfers_lock:
            done_pushing = self._pop_done_transfers(self._sending_transfers)
        for req_id in done_pushing:
            self._reqs_to_send.pop(req_id, None)
            self._reqs_to_process.discard(req_id)
            self.consumer_notification_counts_by_req.pop(req_id, None)
            done_sending.add(req_id)

        # Tell the writer to drop any state it still holds for any
        # request that just finished (push completed) or expired
        # (lease ran out without a D registration ever arriving).
        for req_id in done_sending:
            self._evict_finished_inbox.put(req_id)
        if done_sending:
            self._push_writer_wake.set()

        return done_sending, done_recving
