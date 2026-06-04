# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
P2PSecondaryTierManager: Secondary tier for P2P KV cache sharing.

Owns transports, sessions, and the SecondaryTierManager interface.
"""

from __future__ import annotations

import threading
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
from vllm.v1.kv_offload.tiering.base import (
    JobMetadata,
    JobResult,
    SecondaryTierManager,
)
from vllm.v1.kv_offload.tiering.p2p.control import ControlTransport, ZmqTransport
from vllm.v1.kv_offload.tiering.p2p.data import DataTransport, NixlTransport
from vllm.v1.kv_offload.tiering.p2p.session import (
    P2PClientSession,
    P2PServerSession,
)

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec

logger = init_logger(__name__)


class P2PSecondaryTierManager(SecondaryTierManager):
    """Secondary tier for P2P KV cache sharing."""

    def __init__(
        self,
        offloading_spec: OffloadingSpec,
        primary_kv_view: memoryview,
        tier_type: str = "p2p",
        host: str = "0.0.0.0",
        port: int = 7777,
        **kwargs,
    ) -> None:
        super().__init__(offloading_spec, primary_kv_view, tier_type)
        port = int(port)
        self._local_id = f"{host}:{port}"

        # Pull threading-related kwargs out before constructing transports.
        self._poller_enabled: bool = bool(kwargs.pop("start_poller", True))
        self._poll_interval: float = float(kwargs.pop("poll_interval", 0.001))

        config_fields = self._build_config_fields(offloading_spec)
        self._data: DataTransport = NixlTransport(
            self._local_id, primary_kv_view, config_fields=config_fields
        )
        self._control: ControlTransport = ZmqTransport(self._local_id, host, port)

        self._server_sessions: dict[str, P2PServerSession] = {}
        self._client_sessions: dict[str, P2PClientSession] = {}

        self._finished_jobs: list[JobResult] = []
        self._failed_req_ids: set[str] = set()

        # Concurrency: a re-entrant lock serialises the poller thread
        # against the scheduler thread. Each public API method acquires the
        # lock around its body and releases it on return. RLock allows safe
        # re-entry when one method calls another (e.g. when get_finished_jobs
        # drives _poll_once synchronously in test mode).
        self._lock = threading.RLock()

        # Poller stats for diagnostics (poller-thread-only fields).
        self._poll_acquire_count: int = 0
        self._poll_wait_warned_at: float = 0.0
        self._poll_max_wait_s: float = 0.0
        self._poll_max_held_s: float = 0.0

        self._stop_event = threading.Event()
        self._poller_thread: threading.Thread | None = None
        if self._poller_enabled:
            self._poller_thread = threading.Thread(
                target=self._poll_loop,
                name=f"p2p-poller-{self._local_id}",
                daemon=True,
            )
            self._poller_thread.start()

    # ------------------------------------------------------------------
    # SecondaryTierManager interface
    # ------------------------------------------------------------------

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        with self._lock:
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
        """Ensures client session exists for the remote peer.

        Only opens a client session on the decoder side (do_remote_prefill).
        On the prefiller side, the remote peer in kv_transfer_params is the
        decoder's address, used by submit_store to look up the server session
        — we must not connect to it as a client.
        """
        with self._lock:
            kv_params = req_context.kv_transfer_params
            if kv_params and kv_params.get("do_remote_prefill"):
                peer_id = self._remote_id_from_params(kv_params)
                if peer_id and peer_id not in self._client_sessions:
                    conn = self._control.connect(peer_id)
                    self._client_sessions[peer_id] = P2PClientSession(
                        peer_id,
                        conn,
                        local_id=self._local_id,
                        transport=self._data,
                    )
            return RequestOffloadingContext()

    @override
    def on_request_finished(self, req_context: ReqContext) -> None:
        """Cancels pending loads and prunes state."""
        with self._lock:
            kv_params = req_context.kv_transfer_params
            if not kv_params:
                return
            kv_request_id = kv_params.get("kv_request_id")
            if not kv_request_id:
                return
            self._failed_req_ids.discard(kv_request_id)
            peer_id = self._remote_id_from_params(kv_params)
            if peer_id:
                session = self._client_sessions.get(peer_id)
                if session is not None:
                    session.cancel_request(kv_request_id)

    @override
    def submit_store(self, job_metadata: JobMetadata) -> None:
        with self._lock:
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
            # (and submit_store) before the decoder connects. Create a pending
            # session here so blocks can be buffered; _accept_new_peers will
            # attach the connection once it arrives.
            session = self._server_sessions.get(peer_id)
            if session is None:
                session = P2PServerSession(
                    peer_id=peer_id,
                    local_id=self._local_id,
                    transport=self._data,
                    local_block_len=self._data.block_len,
                )
                self._server_sessions[peer_id] = session
                logger.info(
                    "P2P %s: created pending server session for peer %s "
                    "(kv_request_id=%s, %d blocks)",
                    self._local_id,
                    peer_id,
                    kv_request_id,
                    len(block_ids),
                )
            session.add_stored_blocks(kv_request_id, keys, block_ids, job_id)

    @override
    def submit_load(self, job_metadata: JobMetadata) -> None:
        with self._lock:
            job_id = job_metadata.job_id
            keys = list(job_metadata.keys)
            block_ids = job_metadata.block_ids

            kv_params = job_metadata.req_context.kv_transfer_params
            logger.debug(
                "P2P %s: submit_load ENTRY job_id=%d blocks=%d "
                "kv_request_id=%s peer=%s",
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

            session = self._client_sessions.get(peer_id)
            if session is None:
                logger.warning(
                    "P2P %s: submit_load job_id=%d NO CLIENT SESSION for peer=%s",
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
        with self._lock:
            # When the background poller is disabled (tests), drive a single
            # poll iteration synchronously so callers see the same behaviour
            # as before this change. RLock allows recursive entry from the
            # same thread.
            if not self._poller_enabled:
                self._poll_once()
            result = self._finished_jobs
            self._finished_jobs = []
            return result

    @override
    def on_schedule_end(self) -> None:
        """No-op. Each API method now manages its own lock acquire/release.

        Kept for SecondaryTierManager interface compatibility.
        """
        return

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _build_config_fields(offloading_spec: OffloadingSpec) -> dict | None:
        """Extract config fields for fingerprint from the offloading spec."""
        try:
            model_config = offloading_spec.vllm_config.model_config
            parallel_config = offloading_spec.vllm_config.parallel_config
            gpu_block_sizes = list(offloading_spec.gpu_block_size)
            fields: dict = {
                "model": model_config.model,
                "dtype": str(model_config.dtype),
                "hash_block_size": offloading_spec.hash_block_size,
                "block_size_factor": offloading_spec.block_size_factor,
                "gpu_block_size": gpu_block_sizes,
            }
            # TP size only matters with multiple kv_cache_groups
            if len(gpu_block_sizes) > 1:
                fields["tp_size"] = parallel_config.tensor_parallel_size
            return fields
        except (AttributeError, TypeError):
            return None

    @staticmethod
    def _remote_id_from_params(kv_params: dict) -> str | None:
        host = kv_params.get("remote_host")
        port = kv_params.get("remote_port")
        if host and port:
            return f"{host}:{port}"
        return None

    def _accept_new_peers(self, new_connections: list) -> None:
        for conn in new_connections:
            logger.info(
                "P2P %s: accepting incoming connection from %s",
                self._local_id,
                conn.peer_id,
            )
            try:
                existing = self._server_sessions.get(conn.peer_id)
                if existing is not None:
                    if existing.connected:
                        raise ValueError(f"duplicate connection from {conn.peer_id}")
                    existing.attach_connection(conn)
                    logger.info(
                        "P2P %s: attached connection to pending session for %s",
                        self._local_id,
                        conn.peer_id,
                    )
                else:
                    self._server_sessions[conn.peer_id] = P2PServerSession(
                        peer_id=conn.peer_id,
                        local_id=self._local_id,
                        transport=self._data,
                        local_block_len=self._data.block_len,
                        conn=conn,
                    )
                    logger.info(
                        "P2P %s: created connected server session for %s",
                        self._local_id,
                        conn.peer_id,
                    )
            except (ValueError, KeyError, TypeError, AssertionError) as exc:
                logger.error("P2P %s: rejecting peer: %s", self._local_id, exc)
                conn.close()

    def _reap_dead_servers(self) -> None:
        # Pending sessions (no connection yet) are kept untouched —
        # they're awaiting the decoder's connect handshake, not dead.
        dead = [
            pid
            for pid, s in self._server_sessions.items()
            if s.connected and not s.alive
        ]
        for pid in dead:
            session = self._server_sessions.pop(pid)
            for job_id in session.close():
                self._finished_jobs.append(JobResult(job_id=job_id, success=False))
            self._data.remove_remote_peer(pid)
            logger.warning("P2P %s: server peer %s down", self._local_id, pid)

    def _reap_dead_clients(self) -> None:
        dead = [pid for pid, s in self._client_sessions.items() if not s.alive]
        for pid in dead:
            session = self._client_sessions.pop(pid)
            for job_id, kv_request_id in session.close():
                self._finished_jobs.append(JobResult(job_id=job_id, success=False))
                self._failed_req_ids.add(kv_request_id)
            logger.warning("P2P %s: client peer %s down", self._local_id, pid)

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    def _poll_once(self) -> None:
        """One sweep of the polling work.

        Drains the control transport, polls every server and client
        session, accumulates their results into ``_finished_jobs``, and
        reaps any dead sessions. Holds ``self._lock`` for the duration of
        the per-session polling; the scheduler thread cannot interleave
        API calls until this returns.
        """
        new_connections = self._control.poll()
        if new_connections:
            logger.info(
                "P2P %s: _poll_once got %d new connection(s): %s",
                self._local_id,
                len(new_connections),
                [c.peer_id for c in new_connections],
            )
        wait_t0 = time.monotonic()
        with self._lock:
            wait_s = time.monotonic() - wait_t0
            held_t0 = time.monotonic()
            self._poll_acquire_count += 1
            if wait_s > self._poll_max_wait_s:
                self._poll_max_wait_s = wait_s
            if wait_s > 0.05:
                logger.debug(
                    "P2P %s: POLLER waited %.3fs for lock",
                    self._local_id,
                    wait_s,
                )

            self._accept_new_peers(new_connections)

            for srv in self._server_sessions.values():
                for sr in srv.poll():
                    self._finished_jobs.append(
                        JobResult(job_id=sr.job_id, success=sr.success)
                    )

            for cli in self._client_sessions.values():
                for lr in cli.poll():
                    self._finished_jobs.append(
                        JobResult(job_id=lr.job_id, success=lr.success)
                    )
                    if not lr.success:
                        self._failed_req_ids.add(lr.kv_request_id)

            self._reap_dead_servers()
            self._reap_dead_clients()

            held_s = time.monotonic() - held_t0
            if held_s > self._poll_max_held_s:
                self._poll_max_held_s = held_s
            if held_s > 0.1:
                logger.debug(
                    "P2P %s: _poll_once held lock %.3fs (sessions server:%d client:%d)",
                    self._local_id,
                    held_s,
                    len(self._server_sessions),
                    len(self._client_sessions),
                )

    def _poll_loop(self) -> None:
        """Background thread target: drive _poll_once on an interval.

        Exits when ``_stop_event`` is set. Each iteration's exception is
        logged and swallowed so the poller survives transient failures.
        """
        logger.info("P2P %s: poller thread started", self._local_id)
        ticks = 0
        while not self._stop_event.is_set():
            try:
                self._poll_once()
            except Exception:
                logger.exception("P2P %s: poller iteration failed", self._local_id)
            ticks += 1
            if ticks % 5000 == 0:
                logger.debug(
                    "P2P %s: poller tick=%d sessions=server:%d client:%d "
                    "[poll_acq=%d max_wait=%.3fs max_held=%.3fs]",
                    self._local_id,
                    ticks,
                    len(self._server_sessions),
                    len(self._client_sessions),
                    self._poll_acquire_count,
                    self._poll_max_wait_s,
                    self._poll_max_held_s,
                )
            self._stop_event.wait(self._poll_interval)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @override
    def shutdown(self) -> None:
        self._stop_event.set()
        if self._poller_thread is not None:
            self._poller_thread.join(timeout=5.0)
            self._poller_thread = None
        with self._lock:
            for srv in self._server_sessions.values():
                srv.close()
            for cli in self._client_sessions.values():
                cli.close()
            self._server_sessions.clear()
            self._client_sessions.clear()
            self._control.close()
            self._data.close()
