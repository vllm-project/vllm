# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
P2PSecondaryTierManager: Secondary tier for P2P KV cache sharing.

Owns transports and a single bidirectional P2PSession per remote peer.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from typing_extensions import override

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    LookupResult,
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
from vllm.v1.kv_offload.tiering.p2p.tiering_callbacks import (
    TieringCallbacks,
    _AllMissCallbacks,
)

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec
    from vllm.v1.kv_offload.tiering.p2p.control.base import ControlConnection

logger = init_logger(__name__)

# Reap unbound store batches that have been parked without a FetchMsg
# binding them to a session for longer than this. Protects against the
# prefiller buffering blocks for a decoder that never asks (decoder died,
# network partition, lost kv_request_id). Must be longer than the per-store
# deadline so the store-timeout path fires first for individual jobs.
_UNBOUND_STORE_TIMEOUT_S = 60.0

# Time we wait during shutdown for inflight transfers to drain via
# cancel(mode="wait") before falling back to mode="immediate". Bounded
# so a wedged peer can't hang shutdown.
_SHUTDOWN_DRAIN_TIMEOUT_S = 3.0

# Sleep between iterations of the bounded drain loops in drain_jobs() and
# _drain_inflight_for_shutdown(). Short enough to keep latency low, long
# enough to avoid busy-spinning the scheduler thread.
_DRAIN_SLEEP_S = 0.001


def _prefill_params(kv_params: dict | None) -> dict | None:
    """Return the ``prefill`` sub-dict, or None if absent.

    Set on decoder requests; carries kv_request_id, remote_host, remote_port.
    """
    if not kv_params:
        return None
    return kv_params.get("prefill")


def _decode_params(kv_params: dict | None) -> dict | None:
    """Return the ``decode`` sub-dict, or None if absent.

    Set on prefiller requests; carries kv_request_id.
    """
    if not kv_params:
        return None
    return kv_params.get("decode")


def _p2p_params(kv_params: dict | None) -> dict | None:
    """Return the ``p2p`` sub-dict, or None if absent.

    Set on symmetric-P2P consumer requests; carries kv_request_id,
    remote_host, remote_port.
    """
    if not kv_params:
        return None
    return kv_params.get("p2p")


def _consumer_params(kv_params: dict | None) -> dict | None:
    """Return the consumer sub-dict for either PD or symmetric-P2P.

    Decoder (PD) requests carry ``prefill``; symmetric-P2P consumers
    carry ``p2p``. Both have the same shape (kv_request_id, remote_host,
    remote_port) so callers can use whichever is set.
    """
    return _prefill_params(kv_params) or _p2p_params(kv_params)


@dataclass
class _UnboundStoreBatch:
    """A submit_store batch parked at the manager before any peer has fetched.

    Indexed by kv_request_id only — the prefiller no longer learns the peer
    identity at store time. When a FetchMsg(kv_request_id) arrives on some
    session, the manager binds the kv_request_id to that session and replays
    every parked batch into ServerRole via session.add_stored_blocks.
    """

    job_id: int
    keys: list[OffloadKey]
    block_ids: Sequence[int]
    submitted_at: float = field(default_factory=time.monotonic)


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
        tiering_callbacks: TieringCallbacks | None = None,
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
            tiering_callbacks: TieringManager-facing callbacks invoked
                by the producer's server role to answer inbound
                ``LookupMsg`` traffic. Defaults to
                :class:`_AllMissCallbacks`, which preserves today's
                all-miss behaviour until the real adapter is wired.
            **kwargs: Reserved for future tier-specific options.
        """
        super().__init__(offloading_spec, primary_kv_view, tier_type)
        self._tiering_callbacks: TieringCallbacks = (
            tiering_callbacks if tiering_callbacks is not None else _AllMissCallbacks()
        )
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
        # kv_request_id → session, set when the bound session has received
        # FetchMsg for that id. submit_store after binding routes directly
        # to the session; before binding, batches are parked in
        # _unbound_stores below. Stays in sync with _sessions: entries
        # pointing to a reaped session are purged in _reap_dead_sessions.
        self._kv_to_session: dict[str, P2PSession] = {}
        # kv_request_id → list of batches submit_store'd before any peer
        # asked for that id. Drained into a session by _on_session_fetch
        # when the corresponding FetchMsg arrives, or surfaced as failures
        # by _reap_unbound_stores after _UNBOUND_STORE_TIMEOUT_S.
        self._unbound_stores: dict[str, list[_UnboundStoreBatch]] = {}

        self._finished_jobs: list[JobResult] = []
        # kv_request_ids that hit a transport/session failure; On load lookup()
        # rejects them so the request falls back to local prefill.
        self._failed_req_ids: set[str] = set()

    # ------------------------------------------------------------------
    # SecondaryTierManager interface
    # ------------------------------------------------------------------

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        consumer = _consumer_params(req_context.kv_transfer_params)
        if (
            not consumer
            or not consumer.get("remote_host")
            or not consumer.get("remote_port")
            or not consumer.get("kv_request_id")
        ):
            return LookupResult.MISS

        kv_request_id = consumer["kv_request_id"]
        if kv_request_id in self._failed_req_ids:
            return LookupResult.MISS

        # Symmetric-P2P consumer (``p2p`` sub-dict): probe the peer
        # asynchronously. First call registers the (kv_request_id,
        # block_hash) entry and returns RETRY; flush_pending_lookups()
        # in on_schedule_end batches the LookupMsg; a later step's
        # lookup() returns HIT/MISS once LookupRespMsg has arrived.
        # PD path (``prefill`` sub-dict only) keeps the eager HIT today.
        if _p2p_params(req_context.kv_transfer_params):
            peer_id = self._remote_id_from_params(consumer)
            session = self._sessions.get(peer_id) if peer_id else None
            if session is None:
                return LookupResult.MISS
            result = session.register_lookup(kv_request_id, key)
            if result is True:
                return LookupResult.HIT
            if result is False:
                return LookupResult.MISS
            return LookupResult.RETRY

        # PD consumer (we are the decoder): all kv blocks should be on the
        # prefiller side. Return HIT immediately.
        return LookupResult.HIT

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        """Open the outbound session toward the producer if needed.

        On the consumer side (``prefill`` for PD or ``p2p`` for symmetric
        P2P), open a session toward the producer at remote_host:remote_port
        so submit_load can issue FetchMsg as soon as it fires. On the
        prefiller side, sessions are created when the consumer's inbound
        connection arrives in _accept_new_peers — submit_store no longer
        pre-creates anything.
        """
        consumer = _consumer_params(req_context.kv_transfer_params)
        if consumer:
            peer_id = self._remote_id_from_params(consumer)
            if peer_id:
                self._get_or_create_session(peer_id)
        return RequestOffloadingContext()

    @override
    def on_request_finished(self, req_context: ReqContext) -> None:
        """Cancels pending loads and prunes session-scoped state.

        Decoder side (``prefill`` set): looks up the session by peer_id
        because the producer's address is what addresses the client-role
        load to cancel. Prefiller side (``decode`` set): looks up via
        kv_request_id because peer_id is no longer carried on store-time
        kv_transfer_params; if a session has bound the id, finish it. If
        no session has bound the id yet, this is a no-op: parked batches
        in `_unbound_stores` are left in place and cleaned up only by
        `_reap_unbound_stores` after `_UNBOUND_STORE_TIMEOUT_S`.
        """
        kv_params = req_context.kv_transfer_params
        if not kv_params:
            return
        prefill = _prefill_params(kv_params)
        decode = _decode_params(kv_params)
        kv_request_id = (prefill or decode or {}).get("kv_request_id")
        if not kv_request_id:
            return
        self._failed_req_ids.discard(kv_request_id)

        if prefill:
            peer_id = self._remote_id_from_params(prefill)
            if peer_id:
                session = self._sessions.get(peer_id)
                if session is not None:
                    session.finish_request(kv_request_id)
            return

        # Prefiller-side finish: identify the session via kv_request_id.
        session = self._kv_to_session.pop(kv_request_id, None)
        if session is not None:
            session.finish_request(kv_request_id)
            return

    @override
    def submit_store(self, job_metadata: JobMetadata) -> None:
        job_id = job_metadata.job_id
        keys = list(job_metadata.keys)
        block_ids = job_metadata.block_ids

        assert len(keys) == len(block_ids)

        kv_params = job_metadata.req_context.kv_transfer_params
        decode = _decode_params(kv_params)
        logger.debug(
            "P2P %s: submit_store ENTRY job_id=%d blocks=%d decode=%s kv_request_id=%s",
            self._local_id,
            job_id,
            len(block_ids),
            decode is not None,
            (decode or {}).get("kv_request_id"),
        )
        # Absent ``decode`` block => not a remote-decode request: succeed
        # locally without parking. An empty/malformed dict is still a
        # remote-decode signal and must fail the missing-id check below.
        if decode is None:
            self._finished_jobs.append(JobResult(job_id=job_id, success=True))
            return

        kv_request_id = decode.get("kv_request_id")
        if not kv_request_id:
            logger.warning(
                "P2P %s: submit_store missing kv_request_id",
                self._local_id,
            )
            self._finished_jobs.append(JobResult(job_id=job_id, success=False))
            return

        # Fast path: a session has already received FetchMsg for this id,
        # so we can route the batch straight into its ServerRole.
        session = self._kv_to_session.get(kv_request_id)
        if session is not None:
            session.add_stored_blocks(kv_request_id, keys, block_ids, job_id)
            return

        # No session bound yet — park the batch keyed by kv_request_id.
        # _on_session_fetch drains it on the first FetchMsg; if no peer
        # ever asks, _reap_unbound_stores surfaces the job as failed.
        self._unbound_stores.setdefault(kv_request_id, []).append(
            _UnboundStoreBatch(
                job_id=job_id,
                keys=keys,
                block_ids=block_ids,
            )
        )
        logger.info(
            "P2P %s: parked submit_store kv_request_id=%s job_id=%d blocks=%d",
            self._local_id,
            kv_request_id,
            job_id,
            len(block_ids),
        )

    @override
    def submit_load(self, job_metadata: JobMetadata) -> None:
        job_id = job_metadata.job_id
        keys = list(job_metadata.keys)
        block_ids = job_metadata.block_ids

        consumer = _consumer_params(job_metadata.req_context.kv_transfer_params)
        logger.debug(
            "P2P %s: submit_load ENTRY job_id=%d blocks=%d kv_request_id=%s peer=%s",
            self._local_id,
            job_id,
            len(block_ids),
            (consumer or {}).get("kv_request_id"),
            self._remote_id_from_params(consumer or {}),
        )
        if (
            not consumer
            or not consumer.get("remote_host")
            or not consumer.get("remote_port")
            or not consumer.get("kv_request_id")
        ):
            logger.debug(
                "P2P %s: submit_load job_id=%d FAILED missing consumer params",
                self._local_id,
                job_id,
            )
            self._finished_jobs.append(JobResult(job_id=job_id, success=False))
            return

        kv_request_id = consumer["kv_request_id"]
        peer_id = self._remote_id_from_params(consumer)
        assert peer_id is not None  # guaranteed by consumer checks above

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
        # Flush any p2p lookups aggregated during this step.
        # One LookupMsg per (peer, kv_request_id) with unsent entries;
        # send-gating happens inside the session if not yet ready.
        for session in self._sessions.values():
            session.flush_pending_lookups()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _remote_id_from_params(role_params: dict) -> str | None:
        """Build peer_id from a role-scoped sub-dict (``prefill``/``p2p``)."""
        host = role_params.get("remote_host")
        port = role_params.get("remote_port")
        if host and port:
            return f"{host}:{port}"
        return None

    def _get_or_create_session(self, peer_id: str) -> P2PSession:
        """Return the existing session for peer_id, or open one outbound.

        Consumer-side helper for on_new_request: when ``prefill`` (PD)
        or ``p2p`` (symmetric P2P) is set, the consumer must reach the
        producer at peer_id. If we already have a session toward that
        peer (from a prior load or a peer-initiated inbound), reuse it;
        otherwise open an outbound ControlConnection and build a
        connected session.
        """
        session = self._sessions.get(peer_id)
        if session is not None:
            return session
        conn = self._control.connect(peer_id)
        session = P2PSession(
            peer_id=peer_id,
            local_id=self._local_id,
            transport=self._data,
            local_block_len=self._data.block_len,
            tiering_callbacks=self._tiering_callbacks,
            conn=conn,
        )
        self._sessions[peer_id] = session
        return session

    def _accept_new_peers(self, new_connections: Sequence[ControlConnection]) -> None:
        for conn in new_connections:
            logger.info(
                "P2P %s: accepting incoming connection from %s",
                self._local_id,
                conn.peer_id,
            )
            try:
                existing = self._sessions.get(conn.peer_id)
                if existing is not None:
                    raise ValueError(f"duplicate connection from {conn.peer_id}")
                self._sessions[conn.peer_id] = P2PSession(
                    peer_id=conn.peer_id,
                    local_id=self._local_id,
                    transport=self._data,
                    local_block_len=self._data.block_len,
                    tiering_callbacks=self._tiering_callbacks,
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
        # Reap connected sessions whose connection died — peer is gone.
        # Stranded prefiller-side stores are no longer tracked through a
        # session (they live in _unbound_stores keyed by kv_request_id);
        # _reap_unbound_stores handles their timeout independently.
        dead: list[str] | None = None
        for pid, s in self._sessions.items():
            if s.connected and not s.alive:
                if dead is None:
                    dead = []
                dead.append(pid)
        if dead is None:
            return
        for pid in dead:
            session = self._sessions.pop(pid)
            # Purge any kv_request_id → session entries pointing at this
            # session so subsequent submit_stores fall back to the unbound
            # path (which will time out into failure if no peer rebinds).
            stale_kv_ids = [
                kid for kid, s in self._kv_to_session.items() if s is session
            ]
            for kid in stale_kv_ids:
                del self._kv_to_session[kid]
            failed_loads, failed_stores = session.close()
            for job_id, kv_request_id in failed_loads:
                self._finished_jobs.append(JobResult(job_id=job_id, success=False))
                self._failed_req_ids.add(kv_request_id)
            for job_id in failed_stores:
                self._finished_jobs.append(JobResult(job_id=job_id, success=False))
            self._data.remove_remote_peer(pid)
            logger.warning("P2P %s: peer %s down", self._local_id, pid)

    def _reap_unbound_stores(self) -> None:
        """Time out submit_store batches that no peer has ever fetched.

        Walks `_unbound_stores` for entries whose oldest batch is older
        than `_UNBOUND_STORE_TIMEOUT_S`. Drops the kv_request_id, surfaces
        every batched job as failed, and adds the id to `_failed_req_ids`
        so a late inbound FetchMsg short-circuits to a clean rejection.
        """
        if not self._unbound_stores:
            return
        deadline = time.monotonic() - _UNBOUND_STORE_TIMEOUT_S
        expired: list[str] | None = None
        for kid, batches in self._unbound_stores.items():
            # Batches are appended in arrival order, so the head is oldest.
            if batches and batches[0].submitted_at <= deadline:
                if expired is None:
                    expired = []
                expired.append(kid)
        if expired is None:
            return
        for kid in expired:
            batches = self._unbound_stores.pop(kid)
            self._failed_req_ids.add(kid)
            for batch in batches:
                self._finished_jobs.append(
                    JobResult(job_id=batch.job_id, success=False)
                )
            logger.warning(
                "P2P %s: unbound store kv_request_id=%s timed out after %.0fs "
                "without a fetch — failing %d job(s)",
                self._local_id,
                kid,
                _UNBOUND_STORE_TIMEOUT_S,
                len(batches),
            )

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
            result = session.poll()
            for lr in result.loads:
                self._finished_jobs.append(
                    JobResult(job_id=lr.job_id, success=lr.success)
                )
                if not lr.success:
                    self._failed_req_ids.add(lr.kv_request_id)
            for sr in result.stores:
                self._finished_jobs.append(
                    JobResult(job_id=sr.job_id, success=sr.success)
                )
            # Bind kv_request_id → session for any FetchMsg this tick and
            # replay any submit_store batches parked while no peer was
            # asking. ServerRole.on_fetch already recorded the demand
            # inline in dispatch, so the replayed add_stored_blocks calls
            # match that demand and submit transfers immediately.
            for kv_request_id in result.new_fetch_ids:
                self._kv_to_session[kv_request_id] = session
                for batch in self._unbound_stores.pop(kv_request_id, ()):
                    session.add_stored_blocks(
                        kv_request_id, batch.keys, batch.block_ids, batch.job_id
                    )

        self._reap_dead_sessions()
        self._reap_unbound_stores()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @override
    def shutdown(self) -> None:
        self._drain_inflight_for_shutdown()
        for session in self._sessions.values():
            session.close()
        self._sessions.clear()
        self._kv_to_session.clear()
        # Surface buffered store jobs as failed so the engine doesn't
        # leak them; the manager is going away after this call.
        for batches in self._unbound_stores.values():
            for batch in batches:
                self._finished_jobs.append(
                    JobResult(job_id=batch.job_id, success=False)
                )
        self._unbound_stores.clear()
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
