# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NixlTransport: Data-plane transport for RDMA-based KV block transfers via NIXL.

Threading: the agent is reached from the scheduler thread and, when the tiering
manager runs one, from its tier-poll thread. The manager's lock serializes
them strictly -- one Python thread inside any agent call at a time, with a full
memory barrier between.

That lock is the only protection, not a belt over NIXL's own. nixl_agent picks
NIXL_THREAD_SYNC_STRICT only when ``enable_listen`` is set, which this transport
does not set, so the agent runs with NIXL_THREAD_SYNC_NONE -- under which NIXL's
internal agent lock compiles down to no-ops, leaving every agent entry point
unguarded. So the manager lock must cover all of them, and the lifetime of every
handle passed between calls. The exception is peer registration moved onto a
worker thread, as NixlConnector's handshake executor does: it runs outside the
lock and relies, like that connector, on NIXL tolerating it beside transfers.

Serialized use from a second thread is supported (checked against NIXL 1.4.1
sources): NIXL never creates its UCX worker in UCS_THREAD_MODE_SINGLE, the only
creator-thread-affine mode. The weakest mode it ever requests is
UCS_THREAD_MODE_SERIALIZED, documented as "multiple threads can access, but only
one at a time", and because ``enable_prog_thread`` defaults on here it actually
requests UCS_THREAD_MODE_MULTI and fails construction if the UCX build cannot
provide it.

Two caveats. NIXL binds a shared UCX worker per calling thread
(``tlsSharedWorkerMap``), which is harmless only while there is exactly one
shared worker -- true here, but it would stop being true if a ``num_workers``
UCX backend param were ever set. And NIXL 1.4.1 does accept an explicit
``sync_mode`` (including NIXL_THREAD_SYNC_RW), which would make its own locking
real; that parameter does not exist before 1.x, so passing it would need a
version gate, the way ``capture_telemetry`` does.
"""

from __future__ import annotations

import itertools
import threading
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, NamedTuple

import numpy as np

from vllm.distributed.nixl_utils import NixlWrapper as _NixlAgent
from vllm.distributed.nixl_utils import nixl_agent_config as _NixlAgentConfig
from vllm.logger import init_logger
from vllm.v1.kv_offload.tiering.p2p.data.base import (
    CancelMode,
    DataTransport,
    PollResult,
)

logger = init_logger(__name__)

# Shared sentinel returned by poll() in the steady state (no inflight, or
# no transfer changed state since the last poll). Tuples make it immutable;
# callers only iterate / membership-test / equality-check.
_EMPTY_POLL_RESULT: PollResult = PollResult(done=(), failed=())

# Consecutive transfer-scope failures, with no success between, that fault
# a peer (see "Failure model" in data/base.py).
_MAX_CONSECUTIVE_TRANSFER_FAILURES = 3

# NIXL errors that fault the peer's registration at once, even when raised
# at submission, by exception name → status: the remote endpoint is gone,
# or the remote agent is unknown.
_PEER_FAULT_ERRORS = {
    "nixlRemoteDisconnectError": "NIXL_ERR_REMOTE_DISCONNECT",
    "nixlNotFoundError": "NIXL_ERR_NOT_FOUND",
}


def _peer_fault_status(exc: Exception) -> str | None:
    """The NIXL status of *exc* if it is a peer-scope failure, else None.

    Matched by name and status string so the bindings need not be imported.
    """
    status = _PEER_FAULT_ERRORS.get(type(exc).__name__)
    if status is not None:
        return status
    text = str(exc)
    return next((s for s in _PEER_FAULT_ERRORS.values() if s in text), None)


def _is_terminal_state_error(exc: Exception) -> bool:
    """True if a failed NIXL state check ends the transfer.

    Only ``nixlRemoteDisconnectError`` does: NIXL raises it once the target
    endpoint is gone, so nothing more can land. Any other raise says nothing
    about the WRITE itself; #59790 measured a WRITE that kept landing after
    ``remove_remote_agent`` while its handle raised ``nixlNotFoundError``.
    Such a transfer stays inflight, so its pins and the peer's destination
    blocks are not reused under it. The raise still faults its peer (see
    ``_record_state_check_failure``), which bounds how long it stays.
    """
    return _peer_fault_status(exc) == "NIXL_ERR_REMOTE_DISCONNECT"


class _Inflight(NamedTuple):
    """A submitted-but-not-yet-drained transfer.

    ``peer_id`` lets poll() scope to a single owning session, since the
    transport is shared across all peer sessions of the engine.
    ``struck`` is set once the transfer has counted against its peer, so a
    state check that keeps raising charges it once, not once per poll.
    """

    peer_id: str
    handle: object
    struck: bool = False


class NixlTransport(DataTransport):
    """Manages a NIXL agent, memory registration, and block transfers.

    Wraps the NIXL C library behind a Python interface so the rest of the
    P2P tier code never touches NIXL types directly. Tracks inflight
    handles internally and returns completed/failed tags on poll.
    """

    def __init__(
        self,
        agent_name: str,
        view: memoryview,
        config_fields: dict | None = None,
        backends: list[str] | None = None,
        num_threads: int = 4,
    ) -> None:
        super().__init__(view, config_fields=config_fields)
        self._agent_name = agent_name
        self._backends = list(backends) if backends else ["UCX"]
        self._num_threads = num_threads
        self._agent: Any = None
        self._reg: Any = None
        self._local_dlist: Any = None
        self._remote_dlists: dict[str, object] = {}
        self._peer_nixl_names: dict[str, str] = {}
        # transfer_id → _Inflight(peer_id, handle).
        self._inflight: dict[int, _Inflight] = {}
        self._next_id = itertools.count()
        # peer_id → transfer-scope failures since its last success.
        self._peer_strikes: dict[str, int] = {}
        # peer_id → fault cause; reported by poll(peer_id) until removal.
        self._peer_faults: dict[str, str] = {}
        # Registration is O(num_blocks) NIXL work — add_remote_agent plus a
        # prep_xfer_dlist over every block of the peer's region — so it runs
        # off the scheduler thread, mirroring the NIXL connector's
        # _handshake_initiation_executor. One worker only: NIXL is not
        # guaranteed to be thread-safe, so registrations stay serialized
        # against each other.
        self._reg_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="vllm-p2p-nixl-reg"
        )
        # peer_id → registration generation, bumped on every add and remove.
        # A worker that finishes building handles under a stale generation
        # releases them instead of publishing, so a late completion cannot
        # revive a peer that was reaped or superseded meanwhile.
        self._peer_gen: dict[str, int] = {}
        # Guards _peer_gen together with _peer_nixl_names/_remote_dlists so
        # the generation check and the publish are one atomic step.
        self._peer_lock = threading.Lock()

        self._init(view)

    @property
    def available(self) -> bool:
        return self._agent is not None

    def _init(self, view: memoryview) -> None:
        if _NixlAgent is None:
            return

        non_ucx_backends = [b for b in self._backends if b != "UCX"]
        if non_ucx_backends:
            cfg = _NixlAgentConfig(backends=self._backends, capture_telemetry=True)
            logger.info(
                "NixlTransport %s: NIXL backends=%s",
                self._agent_name,
                self._backends,
            )
        else:
            cfg = _NixlAgentConfig(backends=[], capture_telemetry=True)
            logger.info(
                "NixlTransport %s: NIXL backends=[UCX] num_threads=%d",
                self._agent_name,
                self._num_threads,
            )
        self._agent = _NixlAgent(self._agent_name, cfg)
        if not non_ucx_backends:
            # The tier moves host memory only. UCX otherwise also wires up
            # rc_gda GPU lanes to every peer, and their connect fails after a
            # peer stalls or is replaced, leaving that peer unreachable until
            # restart. UCX_* env vars win over engine_config.
            params = {"engine_config": "TLS=^rc_gda"}
            if self._num_threads > 0:
                params["num_threads"] = str(self._num_threads)
            self._agent.create_backend("UCX", params)

        total_size = self._num_blocks * self._block_len
        reg_descs = [(self._base_addr, total_size, 0, "")]
        self._reg = self._agent.register_memory(reg_descs, mem_type="DRAM")

        block_tuples = [
            (self._base_addr + i * self._block_len, self._block_len, 0)
            for i in range(self._num_blocks)
        ]
        xfer_dlist = self._agent.get_xfer_descs(block_tuples, mem_type="DRAM")
        self._local_dlist = self._agent.prep_xfer_dlist("NIXL_INIT_AGENT", xfer_dlist)
        logger.info(
            "NixlTransport %s: registered %d blocks", self._agent_name, self._num_blocks
        )

    def get_agent_metadata(self) -> bytes:
        assert self._agent is not None
        return self._agent.get_agent_metadata()

    # ------------------------------------------------------------------
    # Peer management
    # ------------------------------------------------------------------

    def add_remote_peer(
        self,
        peer_id: str,
        agent_metadata: bytes,
        base_addr: int,
        num_blocks: int,
        block_len: int,
    ) -> None:
        gen = self._begin_registration(peer_id)
        nixl_name, remote_dlist = self._build_peer_handles(
            agent_metadata, base_addr, num_blocks, block_len
        )
        self._publish_peer(peer_id, gen, nixl_name, remote_dlist)

    def add_remote_peer_async(
        self,
        peer_id: str,
        agent_metadata: bytes,
        base_addr: int,
        num_blocks: int,
        block_len: int,
    ) -> Future[None]:
        gen = self._begin_registration(peer_id)

        def register() -> None:
            nixl_name, remote_dlist = self._build_peer_handles(
                agent_metadata, base_addr, num_blocks, block_len
            )
            if not self._publish_peer(peer_id, gen, nixl_name, remote_dlist):
                raise RuntimeError(
                    f"registration for peer {peer_id} was superseded "
                    "(peer removed or re-registered while registering)"
                )

        return self._reg_executor.submit(register)

    def remove_remote_peer(self, peer_id: str) -> None:
        with self._peer_lock:
            # Invalidate any registration still being built for this peer so
            # its worker releases the handles instead of publishing them.
            self._peer_gen[peer_id] = self._peer_gen.get(peer_id, 0) + 1
            nixl_name = self._peer_nixl_names.pop(peer_id, None)
            dlist = self._remote_dlists.pop(peer_id, None)
        self._peer_strikes.pop(peer_id, None)
        self._peer_faults.pop(peer_id, None)
        if self._agent is not None:
            self._release_remote_handles(peer_id, dlist, nixl_name)

    def _begin_registration(self, peer_id: str) -> int:
        """Claim the next registration generation for *peer_id*."""
        with self._peer_lock:
            gen = self._peer_gen.get(peer_id, 0) + 1
            self._peer_gen[peer_id] = gen
            return gen

    def _build_peer_handles(
        self,
        agent_metadata: bytes,
        base_addr: int,
        num_blocks: int,
        block_len: int,
    ) -> tuple[str, Any]:
        """The expensive half: agent wireup plus one descriptor per block.

        Touches no shared state, so it runs unlocked — holding the peer
        lock across it would stall the scheduler thread for exactly as
        long as the registration it is meant to move off that thread.
        """
        nixl_name = self._agent.add_remote_agent(agent_metadata)
        block_descs = [
            (base_addr + i * block_len, block_len, 0) for i in range(num_blocks)
        ]
        xfer_dlist = self._agent.get_xfer_descs(block_descs, mem_type="DRAM")
        remote_dlist = self._agent.prep_xfer_dlist(nixl_name, xfer_dlist)
        return nixl_name, remote_dlist

    def _publish_peer(
        self,
        peer_id: str,
        gen: int,
        nixl_name: str,
        remote_dlist: Any,
    ) -> bool:
        """Install freshly built handles, unless *gen* is stale.

        Returns False when the peer was removed or re-registered while the
        handles were being built; the handles are released here since
        nothing else ever saw them.
        """
        with self._peer_lock:
            fresh = self._peer_gen.get(peer_id) == gen
            if fresh:
                self._peer_nixl_names[peer_id] = nixl_name
                self._remote_dlists[peer_id] = remote_dlist
        if fresh:
            return True
        logger.info(
            "NixlTransport %s: discarding superseded registration for peer=%s",
            self._agent_name,
            peer_id,
        )
        if self._agent is not None:
            self._release_remote_handles(peer_id, remote_dlist, nixl_name)
        return False

    def _release_remote_handles(
        self, peer_id: str, dlist: object | None, nixl_name: str | None
    ) -> None:
        """Release a peer's descriptor list and remote agent independently.

        The two teardown operations are independent: NIXL may reject one
        after a transport failure, and a ``release_dlist_handle`` that
        raises must not skip ``remove_remote_agent``. Skipping it would
        leave a stale remote registration behind after the peer is gone
        (peer removal) or after its registration was superseded.
        """
        if dlist is not None:
            try:
                self._agent.release_dlist_handle(dlist)
            except Exception as exc:
                logger.warning(
                    "NixlTransport %s: releasing dlist handle for peer=%s failed: %s",
                    self._agent_name,
                    peer_id,
                    exc,
                )
        if nixl_name:
            try:
                self._agent.remove_remote_agent(nixl_name)
            except Exception as exc:
                logger.warning(
                    "NixlTransport %s: removing remote agent for peer=%s failed: %s",
                    self._agent_name,
                    peer_id,
                    exc,
                )

    # ------------------------------------------------------------------
    # Transfer submission and polling
    # ------------------------------------------------------------------

    def write_blocks(
        self,
        peer_id: str,
        local_idxs: list[int],
        remote_idxs: list[int],
    ) -> int | None:
        """Submit a WRITE transfer to *peer_id*.

        Returns a transfer ID, or None if the peer is not registered or
        submission fails. The ID is returned via poll() when the transfer
        completes or fails.
        """
        remote_dlist = self._remote_dlists.get(peer_id)
        if remote_dlist is None:
            logger.warning(
                "NixlTransport %s: write_blocks NO REMOTE DLIST for peer=%s "
                "(known peers=%s)",
                self._agent_name,
                peer_id,
                list(self._remote_dlists.keys()),
            )
            return None
        logger.debug(
            "NixlTransport %s: write_blocks NIXL.transfer peer=%s blocks=%d",
            self._agent_name,
            peer_id,
            len(local_idxs),
        )
        local_arr = np.asarray(local_idxs, dtype=np.int32)
        remote_arr = np.asarray(remote_idxs, dtype=np.int32)
        handle = None
        try:
            handle = self._agent.make_prepped_xfer(
                "WRITE", self._local_dlist, local_arr, remote_dlist, remote_arr
            )
            # transfer() reports a failed post as "ERR" instead of raising.
            if self._agent.transfer(handle) == "ERR":
                raise RuntimeError("transfer() posted with status ERR")
        except Exception as exc:
            logger.warning(
                "NixlTransport %s: write_blocks failed for peer=%s: %s",
                self._agent_name,
                peer_id,
                exc,
            )
            if handle is not None:
                self._release_handles([handle])
            self._record_failure(peer_id, exc)
            return None
        transfer_id = next(self._next_id)
        self._inflight[transfer_id] = _Inflight(peer_id, handle)
        return transfer_id

    def poll(self, peer_id: str | None = None) -> PollResult:
        """Poll inflight transfers.

        When *peer_id* is given, only transfers submitted for that peer_id are
        checked and drained — the transport is shared across peer sessions, so
        an unscoped poll by one session would consume and discard siblings'
        completions. *peer_id* None polls every peer (shutdown drain only).

        Returns PollResult(done=..., failed=..., peer_fault=...) with
        transfer IDs. Completed handles are released automatically.
        ``peer_fault`` is set on a scoped poll once *peer_id* is faulted.
        """
        if not self._inflight:
            return self._poll_result(peer_id, None, None)

        done_ids: list[int] | None = None
        failed_ids: list[int] | None = None
        # Strikes are applied once the batch is classified: poll() sees
        # completions, not their order, so a DONE anywhere in the batch is
        # the success "between" its peer's earlier failures.
        succeeded: set[str] | None = None
        failures: list[tuple[str, str]] | None = None

        for transfer_id, entry in self._inflight.items():
            if peer_id is not None and entry.peer_id != peer_id:
                continue
            try:
                state = self._agent.check_xfer_state(entry.handle)
            except Exception as exc:
                # A check that keeps raising warns once, then at debug.
                (logger.debug if entry.struck else logger.warning)(
                    "NixlTransport %s: check_xfer_state failed for transfer_id=%d: %s",
                    self._agent_name,
                    transfer_id,
                    exc,
                )
                if not entry.struck:
                    self._record_state_check_failure(entry.peer_id, exc)
                    entry = self._inflight[transfer_id] = entry._replace(struck=True)
                if not _is_terminal_state_error(exc):
                    # Kept: the WRITE may still be landing. Its peer is now
                    # faulted, so the session disconnects once this drains
                    # or at its drain cap, and the teardown reclaims it.
                    continue
                state = "ERR"
            if state == "DONE":
                if done_ids is None:
                    done_ids = []
                done_ids.append(transfer_id)
                if succeeded is None:
                    succeeded = set()
                succeeded.add(entry.peer_id)
            elif state not in ("PROC", "PEND"):
                if failed_ids is None:
                    failed_ids = []
                failed_ids.append(transfer_id)
                if not entry.struck:
                    if failures is None:
                        failures = []
                    failures.append((entry.peer_id, f"state {state}"))

        for pid in succeeded or ():
            self._peer_strikes.pop(pid, None)
        for pid, cause in failures or ():
            self._record_transfer_failure(pid, cause)

        if done_ids is None and failed_ids is None:
            return self._poll_result(peer_id, None, None)

        handles_to_release = []
        for tid in done_ids or ():
            handles_to_release.append(self._inflight.pop(tid).handle)
        for tid in failed_ids or ():
            handles_to_release.append(self._inflight.pop(tid).handle)
        self._release_handles(handles_to_release)

        return self._poll_result(peer_id, done_ids, failed_ids)

    def _poll_result(
        self,
        peer_id: str | None,
        done_ids: list[int] | None,
        failed_ids: list[int] | None,
    ) -> PollResult:
        fault = self._peer_faults.get(peer_id) if peer_id is not None else None
        if done_ids is None and failed_ids is None and fault is None:
            return _EMPTY_POLL_RESULT
        return PollResult(
            done=done_ids if done_ids is not None else _EMPTY_POLL_RESULT.done,
            failed=failed_ids if failed_ids is not None else _EMPTY_POLL_RESULT.failed,
            peer_fault=fault,
        )

    def cancel(
        self,
        transfer_ids: Iterable[int],
        mode: CancelMode = "immediate",
    ) -> list[int]:
        """Cancel inflight transfers by their IDs.

        See ``DataTransport.cancel`` for the contract. In "wait" mode, a
        transfer is released only once NIXL reports it terminal. Releasing
        a PROC/PEND handle does not stop the WRITE: the UCX backend frees
        the request without cancelling RMA, and the bytes keep landing. So
        PROC/PEND transfers, and those whose state check or
        ``release_xfer_handle`` raises, stay in ``self._inflight`` for a
        later ``poll()`` to observe. A state check that raises faults the
        peer; only a remote disconnect also counts as terminal (see
        ``_is_terminal_state_error``).
        """
        if mode == "immediate":
            handles = [
                self._inflight.pop(tid).handle
                for tid in transfer_ids
                if tid in self._inflight
            ]
            self._release_handles(handles)
            return []

        still_inflight: list[int] = []
        for tid in transfer_ids:
            entry = self._inflight.get(tid)
            if entry is None:
                continue
            try:
                state = self._agent.check_xfer_state(entry.handle)
            except Exception as exc:
                logger.debug(
                    "NixlTransport %s: cancel state check failed for "
                    "transfer_id=%d: %s",
                    self._agent_name,
                    tid,
                    exc,
                )
                self._record_state_check_failure(entry.peer_id, exc)
                if not _is_terminal_state_error(exc):
                    still_inflight.append(tid)
                    continue
                # The handle may no longer be releasable; it is terminal
                # either way, so drop it instead of retrying forever.
                self._release_handles([entry.handle])
                del self._inflight[tid]
                continue
            if state in ("PROC", "PEND"):
                still_inflight.append(tid)
                continue
            try:
                self._agent.release_xfer_handle(entry.handle)
            except Exception as exc:
                logger.debug(
                    "NixlTransport %s: cancel pending for transfer_id=%d: %s",
                    self._agent_name,
                    tid,
                    exc,
                )
                still_inflight.append(tid)
                continue
            del self._inflight[tid]
        return still_inflight

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        # Stop accepting registrations and let any in-flight one finish
        # before the agent goes away underneath it.
        self._reg_executor.shutdown(wait=True)
        if self._agent is None:
            return
        self._release_handles([entry.handle for entry in self._inflight.values()])
        self._inflight.clear()
        for peer_id in list(self._remote_dlists):
            self.remove_remote_peer(peer_id)
        if self._local_dlist is not None:
            self._agent.release_dlist_handle(self._local_dlist)
            self._local_dlist = None
        if self._reg is not None:
            self._agent.deregister_memory(self._reg)
            self._reg = None
        self._agent = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _record_failure(self, peer_id: str, exc: Exception) -> None:
        """Classify a NIXL exception: a peer fault, or one strike."""
        status = _peer_fault_status(exc)
        if status is not None:
            self._set_peer_fault(peer_id, status)
        else:
            self._record_transfer_failure(peer_id, f"{type(exc).__name__}: {exc}")

    def _record_state_check_failure(self, peer_id: str, exc: Exception) -> None:
        """Fault *peer_id* for a ``check_xfer_state`` that raised.

        The bindings raise for every NIXL error status (only success and
        in-progress are returned), so a raise is a failure NIXL reported,
        not a flaky check. A peer whose transfer state can't be trusted is
        torn down rather than given strikes: with strikes, successes to the
        same peer could keep resetting them while the transfer stays
        inflight, holding its pins, for good.
        """
        self._set_peer_fault(
            peer_id,
            _peer_fault_status(exc)
            or f"check_xfer_state raised {type(exc).__name__}: {exc}",
        )

    def _record_transfer_failure(self, peer_id: str, cause: str) -> None:
        strikes = self._peer_strikes.get(peer_id, 0) + 1
        self._peer_strikes[peer_id] = strikes
        if strikes >= _MAX_CONSECUTIVE_TRANSFER_FAILURES:
            self._set_peer_fault(
                peer_id, f"{strikes} consecutive transfer failures (last: {cause})"
            )

    def _set_peer_fault(self, peer_id: str, cause: str) -> None:
        if peer_id in self._peer_faults:
            return
        self._peer_faults[peer_id] = cause
        logger.warning(
            "NixlTransport %s: registration for peer=%s judged unusable: %s",
            self._agent_name,
            peer_id,
            cause,
        )

    def _release_handles(self, handles: list[object]) -> None:
        if self._agent is None:
            return
        for handle in handles:
            try:
                self._agent.release_xfer_handle(handle)
            except Exception as exc:
                logger.warning(
                    "NixlTransport %s: release_xfer_handle failed: %s",
                    self._agent_name,
                    exc,
                )
