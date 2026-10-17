# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NixlTransport: Data-plane transport for RDMA-based KV block transfers via NIXL."""

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


class _Inflight(NamedTuple):
    """A submitted-but-not-yet-drained transfer.

    ``peer_id`` lets poll() scope to a single owning session, since the
    transport is shared across all peer sessions of the engine.
    """

    peer_id: str
    handle: object


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
            cfg = _NixlAgentConfig(
                num_threads=self._num_threads, capture_telemetry=True
            )
            logger.info(
                "NixlTransport %s: NIXL backends=[UCX] num_threads=%d",
                self._agent_name,
                self._num_threads,
            )
        self._agent = _NixlAgent(self._agent_name, cfg)

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
        if self._agent is not None:
            if dlist is not None:
                self._agent.release_dlist_handle(dlist)
            if nixl_name:
                self._agent.remove_remote_agent(nixl_name)

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
            try:
                self._agent.release_dlist_handle(remote_dlist)
                self._agent.remove_remote_agent(nixl_name)
            except Exception as exc:
                logger.warning(
                    "NixlTransport %s: releasing superseded registration "
                    "for peer=%s failed: %s",
                    self._agent_name,
                    peer_id,
                    exc,
                )
        return False

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

        Returns a transfer ID, or None if the peer is not registered.
        The ID is returned via poll() when the transfer completes or fails.
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
        handle = self._agent.make_prepped_xfer(
            "WRITE",
            self._local_dlist,
            np.asarray(local_idxs, dtype=np.int32),
            remote_dlist,
            np.asarray(remote_idxs, dtype=np.int32),
        )
        self._agent.transfer(handle)
        transfer_id = next(self._next_id)
        self._inflight[transfer_id] = _Inflight(peer_id, handle)
        return transfer_id

    def poll(self, peer_id: str | None = None) -> PollResult:
        """Poll inflight transfers.

        When *peer_id* is given, only transfers submitted for that peer_id are
        checked and drained — the transport is shared across peer sessions, so
        an unscoped poll by one session would consume and discard siblings'
        completions. *peer_id* None polls every peer (shutdown drain only).

        Returns PollResult(done=..., failed=...) with transfer IDs.
        Completed handles are released automatically.
        """
        if not self._inflight:
            return _EMPTY_POLL_RESULT

        done_ids: list[int] | None = None
        failed_ids: list[int] | None = None

        for transfer_id, entry in self._inflight.items():
            if peer_id is not None and entry.peer_id != peer_id:
                continue
            try:
                state = self._agent.check_xfer_state(entry.handle)
            except Exception as exc:
                logger.warning(
                    "NixlTransport %s: check_xfer_state failed for transfer_id=%d: %s",
                    self._agent_name,
                    transfer_id,
                    exc,
                )
                continue
            if state == "DONE":
                if done_ids is None:
                    done_ids = []
                done_ids.append(transfer_id)
            elif state not in ("PROC", "PEND"):
                if failed_ids is None:
                    failed_ids = []
                failed_ids.append(transfer_id)

        if done_ids is None and failed_ids is None:
            return _EMPTY_POLL_RESULT

        handles_to_release = []
        for tid in done_ids or ():
            handles_to_release.append(self._inflight.pop(tid).handle)
        for tid in failed_ids or ():
            handles_to_release.append(self._inflight.pop(tid).handle)
        self._release_handles(handles_to_release)

        return PollResult(
            done=done_ids if done_ids is not None else _EMPTY_POLL_RESULT.done,
            failed=failed_ids if failed_ids is not None else _EMPTY_POLL_RESULT.failed,
        )

    def cancel(
        self,
        transfer_ids: Iterable[int],
        mode: CancelMode = "immediate",
    ) -> list[int]:
        """Cancel inflight transfers by their IDs.

        See ``DataTransport.cancel`` for the contract. In "wait" mode,
        transfers whose ``release_xfer_handle`` raises (NIXL could not
        complete the abort because the backend is still draining) stay
        in ``self._inflight`` so a later ``poll()`` will observe them.
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
