# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
P2PSecondaryTierManager: Secondary tier for P2P KV cache sharing.

Owns transports and a single bidirectional P2PSession per remote peer.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import TYPE_CHECKING

from typing_extensions import override

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    OffloadKey,
    ReqContext,
    RequestOffloadingContext,
)
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.base import (
    JobMetadata,
    JobResult,
    SecondaryTierManager,
)
from vllm.v1.kv_offload.tiering.p2p.control import ControlTransport, ZmqTransport
from vllm.v1.kv_offload.tiering.p2p.data import DataTransport, NixlTransport
from vllm.v1.kv_offload.tiering.p2p.session import P2PSession

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec

logger = init_logger(__name__)

# Reap pending (not-yet-connected) sessions older than this. Protects
# against the prefiller buffering blocks for a decoder that never
# connects (decoder died, network partition, lost kv_request_id).
# Must be longer than the per-store deadline so the store-timeout path
# fires first for individual jobs.
_PENDING_SESSION_TIMEOUT_S = 60.0

# Time we wait during shutdown for inflight transfers to drain via
# cancel(mode="wait") before falling back to mode="immediate". Bounded
# so a wedged peer can't hang shutdown.
_SHUTDOWN_DRAIN_TIMEOUT_S = 3.0

# Sleep between iterations of the bounded drain loops in drain_jobs() and
# _drain_inflight_for_shutdown(). Short enough to keep latency low, long
# enough to avoid busy-spinning the scheduler thread.
_DRAIN_SLEEP_S = 0.001


class P2PSecondaryTierManager(SecondaryTierManager):
    """Secondary tier for P2P KV cache sharing.

    A single P2PSession per remote peer handles both client-role (loading
    blocks from the peer) and server-role (serving blocks to the peer)
    over the same control connection.

    Single-threaded: every public method runs on the scheduler thread, and
    the engine drives polling via ``get_finished_jobs()`` once per step.
    ``has_pending_work()`` keeps the engine ticking so the control transport
    and existing sessions are polled even when no requests are scheduled.
    """

    def __init__(
        self,
        offloading_spec: OffloadingSpec,
        primary_kv_view: memoryview,
        tier_type: str = "p2p",
        host: str = "0.0.0.0",
        port: int = 7777,
        backends: list[str] | None = None,
        num_threads: int = 4,
        **kwargs,
    ) -> None:
        """Initialize the P2P secondary tier manager.

        All keyword arguments after ``primary_kv_view`` come from the
        ``secondary_tiers`` entry in ``kv_connector_extra_config``. See
        ``docs/features/kv_offloading_usage.md`` for the user-facing
        configuration reference.

        Args:
            offloading_spec: Owning ``OffloadingSpec`` (provides
                ``vllm_config`` and the offloaded block layout).
            primary_kv_view: Memoryview over the CPU primary tier; the
                NIXL agent registers this region for RDMA transfers.
            tier_type: Tier identifier (defaults to ``"p2p"``).
            host: Address the ZMQ control socket binds to.
            port: Port for the ZMQ control socket. Must be reachable
                from peers.
            backends: NIXL transport backends (e.g. ``["UCX"]``,
                ``["MOONCAKE"]``, ``["LIBFABRIC"]``). Defaults to
                ``["UCX"]``. When any non-UCX backend is requested, the
                NIXL agent is initialized with ``backends=...``;
                otherwise it falls back to a UCX-only agent with
                ``num_threads`` threads.
            num_threads: NIXL agent worker threads for the UCX-only
                branch. Ignored when ``backends`` contains a non-UCX
                entry.
            **kwargs: Reserved for future tier-specific options.
        """
        super().__init__(offloading_spec, primary_kv_view, tier_type)
        port = int(port)
        self._local_id = f"{host}:{port}"

        config_fields = FileMapper.from_offloading_spec(
            root_dir="",
            offloading_spec=offloading_spec,
            gpu_blocks_per_file=offloading_spec.block_size_factor,
            parallel_agnostic=True,
        ).get_run_config()
        self._data: DataTransport = NixlTransport(
            self._local_id,
            primary_kv_view,
            config_fields=config_fields,
            backends=backends,
            num_threads=int(num_threads),
        )
        self._control: ControlTransport = ZmqTransport(self._local_id, host, port)

        self._sessions: dict[str, P2PSession] = {}
        # peer_id → time.monotonic() at which a pending (not-yet-connected)
        # session was created. Cleared when the peer's connection arrives
        # (in _accept_new_peers) or when the session is reaped. Used by
        # _reap_dead_sessions to time out stranded pending sessions.
        self._pending_session_created_at: dict[str, float] = {}

        self._finished_jobs: list[JobResult] = []
        # kv_request_ids that hit a transport/session failure; On load lookup()
        # rejects them so the request falls back to local prefill.
        self._failed_req_ids: set[str] = set()

    # ------------------------------------------------------------------
    # SecondaryTierManager interface
    # ------------------------------------------------------------------

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        kv_params = req_context.kv_transfer_params
        if (
            not kv_params
            or not kv_params.get("do_remote_prefill")
            or not kv_params.get("remote_host")
            or not kv_params.get("remote_port")
            or not kv_params.get("kv_request_id")
        ):
            return False

        kv_request_id = kv_params["kv_request_id"]
        return kv_request_id not in self._failed_req_ids

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        """Ensure a session exists for the remote peer.

        On the decoder side (do_remote_prefill), create a session and
        initiate the connection if one doesn't already exist. On the
        prefiller side, the session is created lazily by submit_store
        in pending mode and promoted to connected when the decoder's
        inbound connection arrives.
        """
        kv_params = req_context.kv_transfer_params
        if kv_params and kv_params.get("do_remote_prefill"):
            peer_id = self._remote_id_from_params(kv_params)
            if peer_id:
                self._get_or_create_session(peer_id, initiate_connect=True)
        return RequestOffloadingContext()

    @override
    def on_request_finished(self, req_context: ReqContext) -> None:
        """Cancels pending loads and prunes state."""
        kv_params = req_context.kv_transfer_params
        if not kv_params:
            return
        kv_request_id = kv_params.get("kv_request_id")
        if not kv_request_id:
            return
        self._failed_req_ids.discard(kv_request_id)
        peer_id = self._remote_id_from_params(kv_params)
        if peer_id:
            session = self._sessions.get(peer_id)
            if session is not None:
                session.finish_request(kv_request_id)

    @override
    def submit_store(self, job_metadata: JobMetadata) -> None:
        job_id = job_metadata.job_id
        keys = list(job_metadata.keys)
        block_ids = job_metadata.block_ids

        assert len(keys) == len(block_ids)

        kv_params = job_metadata.req_context.kv_transfer_params
        logger.debug(
            "P2P %s: submit_store ENTRY job_id=%d blocks=%d "
            "do_remote_decode=%s kv_request_id=%s peer=%s",
            self._local_id,
            job_id,
            len(block_ids),
            bool(kv_params and kv_params.get("do_remote_decode")),
            (kv_params or {}).get("kv_request_id"),
            self._remote_id_from_params(kv_params or {}),
        )
        if not kv_params or not kv_params.get("do_remote_decode"):
            self._finished_jobs.append(JobResult(job_id=job_id, success=True))
            return

        kv_request_id = kv_params.get("kv_request_id")
        peer_id = self._remote_id_from_params(kv_params)
        if not kv_request_id or not peer_id:
            logger.warning(
                "P2P %s: submit_store missing kv_request_id or peer",
                self._local_id,
            )
            self._finished_jobs.append(JobResult(job_id=job_id, success=False))
            return

        # Lazy session creation: prefiller-first mode finishes prefill
        # (and submit_store) before the decoder connects. Create a
        # pending session here so blocks can be buffered; the inbound
        # connect from the decoder will be attached in _accept_new_peers.
        session = self._sessions.get(peer_id)
        if session is None:
            session = self._make_pending_session(peer_id)
            self._sessions[peer_id] = session
            self._pending_session_created_at[peer_id] = time.monotonic()
            logger.info(
                "P2P %s: created pending session for peer %s "
                "(kv_request_id=%s, %d blocks)",
                self._local_id,
                peer_id,
                kv_request_id,
                len(block_ids),
            )
        session.add_stored_blocks(kv_request_id, keys, block_ids, job_id)

    @override
    def submit_load(self, job_metadata: JobMetadata) -> None:
        job_id = job_metadata.job_id
        keys = list(job_metadata.keys)
        block_ids = job_metadata.block_ids

        kv_params = job_metadata.req_context.kv_transfer_params
        logger.debug(
            "P2P %s: submit_load ENTRY job_id=%d blocks=%d kv_request_id=%s peer=%s",
            self._local_id,
            job_id,
            len(block_ids),
            (kv_params or {}).get("kv_request_id"),
            self._remote_id_from_params(kv_params or {}),
        )
        if (
            not kv_params
            or not kv_params.get("remote_host")
            or not kv_params.get("remote_port")
            or not kv_params.get("kv_request_id")
        ):
            logger.debug(
                "P2P %s: submit_load job_id=%d FAILED missing kv_params",
                self._local_id,
                job_id,
            )
            self._finished_jobs.append(JobResult(job_id=job_id, success=False))
            return

        kv_request_id = kv_params["kv_request_id"]
        peer_id = self._remote_id_from_params(kv_params)
        assert peer_id is not None  # guaranteed by kv_params checks above

        if not keys:
            logger.debug(
                "P2P %s: submit_load job_id=%d short-circuit success (no keys)",
                self._local_id,
                job_id,
            )
            self._finished_jobs.append(JobResult(job_id=job_id, success=True))
            return

        session = self._sessions.get(peer_id)
        if session is None:
            logger.warning(
                "P2P %s: submit_load job_id=%d NO SESSION for peer=%s",
                self._local_id,
                job_id,
                peer_id,
            )
            self._finished_jobs.append(JobResult(job_id=job_id, success=False))
            self._failed_req_ids.add(kv_request_id)
            return
        logger.debug(
            "P2P %s: submit_load job_id=%d -> request_blocks peer=%s "
            "kv_request_id=%s blocks=%d session_ready=%s",
            self._local_id,
            job_id,
            peer_id,
            kv_request_id,
            len(block_ids),
            session.ready,
        )
        session.request_blocks(job_id, kv_request_id, keys, block_ids)

    @override
    def get_finished_jobs(self) -> Iterable[JobResult]:
        # Drive one polling sweep on the scheduler thread, then hand off
        # whatever has accumulated. The engine calls this once per step
        # (and keeps stepping while has_pending_work() is True).
        self._poll_once()
        result = self._finished_jobs
        self._finished_jobs = []
        return result

    @override
    def has_pending_work(self) -> bool:
        # The engine tick is the only driver of _control.poll() and
        # session.poll(); without it we miss new peer connects and
        # inbound fetch messages on existing sessions. Keep the engine
        # ticking for the lifetime of this manager.
        return True

    @override
    def drain_jobs(self) -> None:
        """Block until every submitted load/store job has completed or failed.

        Loops calling ``_poll_once()`` until no session has outstanding
        inbound loads or in-flight outbound stores. Mid-flight transfers
        are NOT cancelled — the caller (``TieringOffloadingManager.reset_cache``)
        needs the primary memoryview to be quiescent, not aborted. Results
        accumulate in ``_finished_jobs`` and are surfaced by the next
        ``get_finished_jobs()`` call.
        """
        start = time.monotonic()
        warned = False
        while True:
            self._poll_once()
            pending = any(
                s._client._inbound or s._server._inflight
                for s in self._sessions.values()
            )
            if not pending:
                return
            if not warned and time.monotonic() - start > 5.0:
                logger.warning(
                    "P2PSecondaryTierManager.drain_jobs: still draining "
                    "after 5s; a stuck transfer will block the engine.",
                )
                warned = True
            time.sleep(_DRAIN_SLEEP_S)

    @override
    def on_schedule_end(self) -> None:
        return

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _remote_id_from_params(kv_params: dict) -> str | None:
        host = kv_params.get("remote_host")
        port = kv_params.get("remote_port")
        if host and port:
            return f"{host}:{port}"
        return None

    def _make_pending_session(self, peer_id: str) -> P2PSession:
        return P2PSession(
            peer_id=peer_id,
            local_id=self._local_id,
            transport=self._data,
            local_block_len=self._data.block_len,
        )

    def _get_or_create_session(
        self, peer_id: str, *, initiate_connect: bool
    ) -> P2PSession:
        """Return the existing session for peer_id, or create a new one.

        If initiate_connect is True and no session exists, opens an
        outbound ControlConnection, builds a connected session (which
        sends our ConnectMsg immediately), and stores it.

        If initiate_connect is False and no session exists, returns a
        pending session (no connection yet).
        """
        session = self._sessions.get(peer_id)
        if session is not None:
            return session
        if initiate_connect:
            conn = self._control.connect(peer_id)
            session = P2PSession(
                peer_id=peer_id,
                local_id=self._local_id,
                transport=self._data,
                local_block_len=self._data.block_len,
                conn=conn,
            )
        else:
            session = self._make_pending_session(peer_id)
            self._pending_session_created_at[peer_id] = time.monotonic()
        self._sessions[peer_id] = session
        return session

    def _accept_new_peers(self, new_connections: list) -> None:
        for conn in new_connections:
            logger.info(
                "P2P %s: accepting incoming connection from %s",
                self._local_id,
                conn.peer_id,
            )
            try:
                existing = self._sessions.get(conn.peer_id)
                if existing is not None:
                    if existing.connected:
                        raise ValueError(f"duplicate connection from {conn.peer_id}")
                    existing.attach_connection(conn)
                    self._pending_session_created_at.pop(conn.peer_id, None)
                    logger.info(
                        "P2P %s: attached connection to pending session for %s",
                        self._local_id,
                        conn.peer_id,
                    )
                else:
                    self._sessions[conn.peer_id] = P2PSession(
                        peer_id=conn.peer_id,
                        local_id=self._local_id,
                        transport=self._data,
                        local_block_len=self._data.block_len,
                        conn=conn,
                    )
                    logger.info(
                        "P2P %s: created connected session for %s",
                        self._local_id,
                        conn.peer_id,
                    )
            except (ValueError, KeyError, TypeError, AssertionError) as exc:
                logger.error("P2P %s: rejecting peer: %s", self._local_id, exc)
                conn.close()

    def _reap_dead_sessions(self) -> None:
        # Two reasons to reap:
        #   1. Connected session whose connection died — peer is gone.
        #   2. Pending session that has been waiting longer than
        #      _PENDING_SESSION_TIMEOUT_S for a connection that never
        #      arrived (decoder died, partition, lost kv_request_id).
        # If a late connect arrives after a pending reap, _accept_new_peers
        # will create a fresh connected session for it; the buffered work
        # is gone (already surfaced as failures via session.close()) and
        # _failed_req_ids steers any pending lookup to local prefill.
        dead: list[str] | None = None
        deadline = time.monotonic() - _PENDING_SESSION_TIMEOUT_S
        for pid, s in self._sessions.items():
            if s.connected and not s.alive:
                if dead is None:
                    dead = []
                dead.append(pid)
            elif not s.connected:
                created_at = self._pending_session_created_at.get(pid)
                if created_at is not None and created_at <= deadline:
                    logger.warning(
                        "P2P %s: pending session for peer %s timed out "
                        "after %.0fs without connection — reaping",
                        self._local_id,
                        pid,
                        _PENDING_SESSION_TIMEOUT_S,
                    )
                    if dead is None:
                        dead = []
                    dead.append(pid)
        if dead is None:
            return
        for pid in dead:
            session = self._sessions.pop(pid)
            self._pending_session_created_at.pop(pid, None)
            failed_loads, failed_stores = session.close()
            for job_id, kv_request_id in failed_loads:
                self._finished_jobs.append(JobResult(job_id=job_id, success=False))
                self._failed_req_ids.add(kv_request_id)
            for job_id in failed_stores:
                self._finished_jobs.append(JobResult(job_id=job_id, success=False))
            self._data.remove_remote_peer(pid)
            logger.warning("P2P %s: peer %s down", self._local_id, pid)

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    def _poll_once(self) -> None:
        """One sweep of the polling work.

        Drains the control transport, polls every session, accumulates
        their results into ``_finished_jobs``, and reaps any dead sessions.
        Runs on the scheduler thread.
        """
        new_connections = self._control.poll()
        if new_connections:
            logger.info(
                "P2P %s: _poll_once got %d new connection(s): %s",
                self._local_id,
                len(new_connections),
                [c.peer_id for c in new_connections],
            )

        self._accept_new_peers(new_connections)

        for session in self._sessions.values():
            loads, stores = session.poll()
            for lr in loads:
                self._finished_jobs.append(
                    JobResult(job_id=lr.job_id, success=lr.success)
                )
                if not lr.success:
                    self._failed_req_ids.add(lr.kv_request_id)
            for sr in stores:
                self._finished_jobs.append(
                    JobResult(job_id=sr.job_id, success=sr.success)
                )

        self._reap_dead_sessions()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @override
    def shutdown(self) -> None:
        self._drain_inflight_for_shutdown()
        for session in self._sessions.values():
            session.close()
        self._sessions.clear()
        self._pending_session_created_at.clear()
        self._control.close()
        self._data.close()

    def _drain_inflight_for_shutdown(self) -> None:
        """Best-effort drain of inflight transfers before closing _data.

        Mirrors session._drain_abort but as a single bounded loop. Collects
        inflight transfer_ids from each session, repeatedly calls
        _data.cancel(..., mode="wait") and _data.poll() so handles can
        surface as done/failed, and falls back to mode="immediate" once
        _SHUTDOWN_DRAIN_TIMEOUT_S elapses so a wedged peer can't hang us.
        """
        ids = [tid for s in self._sessions.values() for tid in s._server._inflight]
        if not ids:
            return
        deadline = time.monotonic() + _SHUTDOWN_DRAIN_TIMEOUT_S
        still: list[int] = ids
        while still and time.monotonic() < deadline:
            still = list(self._data.cancel(still, mode="wait"))
            if not still:
                break
            # poll() advances NIXL handle state so the next wait-cancel
            # has a chance to release the handles.
            self._data.poll()
            time.sleep(_DRAIN_SLEEP_S)
        if still:
            logger.warning(
                "P2P %s: shutdown drain timed out after %.1fs with %d "
                "transfers still inflight — force-cancelling",
                self._local_id,
                _SHUTDOWN_DRAIN_TIMEOUT_S,
                len(still),
            )
            self._data.cancel(still, mode="immediate")
