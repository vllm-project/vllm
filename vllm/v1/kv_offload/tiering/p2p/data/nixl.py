# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
NixlTransport: Data-plane transport for RDMA-based KV block transfers via NIXL.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from typing import Any

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


class NixlTransport(DataTransport):
    """Manages a NIXL agent, memory registration, and block transfers.

    Wraps the NIXL C library behind a Python interface so the rest of the
    P2P tier code never touches NIXL types directly. Tracks inflight
    handles internally and returns completed/failed tags on poll.
    """

    def __init__(
        self,
        local_id: str,
        view: memoryview,
        config_fields: dict | None = None,
        backends: list[str] | None = None,
        num_threads: int = 4,
    ) -> None:
        super().__init__(view, config_fields=config_fields)
        self._local_id = local_id
        self._backends = list(backends) if backends else ["UCX"]
        self._num_threads = num_threads
        self._agent: Any = None
        self._reg: Any = None
        self._local_dlist: Any = None
        self._remote_dlists: dict[str, object] = {}
        self._peer_nixl_names: dict[str, str] = {}
        # transfer_id → (peer_id, handle). The peer_id lets poll() scope to a
        # single owning session, since this transport is shared across all
        # peer sessions of the engine.
        self._inflight: dict[int, tuple[str, object]] = {}
        self._next_id = itertools.count()

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
                self._local_id,
                self._backends,
            )
        else:
            cfg = _NixlAgentConfig(
                num_threads=self._num_threads, capture_telemetry=True
            )
            logger.info(
                "NixlTransport %s: NIXL backends=[UCX] num_threads=%d",
                self._local_id,
                self._num_threads,
            )
        self._agent = _NixlAgent(self._local_id, cfg)

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
            "NixlTransport %s: registered %d blocks", self._local_id, self._num_blocks
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
        nixl_name = self._agent.add_remote_agent(agent_metadata)
        block_descs = [
            (base_addr + i * block_len, block_len, 0) for i in range(num_blocks)
        ]
        xfer_dlist = self._agent.get_xfer_descs(block_descs, mem_type="DRAM")
        remote_dlist = self._agent.prep_xfer_dlist(nixl_name, xfer_dlist)
        self._peer_nixl_names[peer_id] = nixl_name
        self._remote_dlists[peer_id] = remote_dlist

    def remove_remote_peer(self, peer_id: str) -> None:
        nixl_name = self._peer_nixl_names.pop(peer_id, None)
        dlist = self._remote_dlists.pop(peer_id, None)
        if self._agent is not None:
            if dlist is not None:
                self._agent.release_dlist_handle(dlist)
            if nixl_name:
                self._agent.remove_remote_agent(nixl_name)

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
                self._local_id,
                peer_id,
                list(self._remote_dlists.keys()),
            )
            return None
        logger.debug(
            "NixlTransport %s: write_blocks NIXL.transfer peer=%s blocks=%d",
            self._local_id,
            peer_id,
            len(local_idxs),
        )
        handle = self._agent.make_prepped_xfer(
            "WRITE",
            self._local_dlist,
            local_idxs,
            remote_dlist,
            remote_idxs,
        )
        self._agent.transfer(handle)
        transfer_id = next(self._next_id)
        self._inflight[transfer_id] = (peer_id, handle)
        return transfer_id

    def poll(self, owner: str | None = None) -> PollResult:
        """Poll inflight transfers.

        When *owner* is given, only transfers submitted for that peer_id are
        checked and drained — the transport is shared across peer sessions, so
        an unscoped poll by one session would consume and discard siblings'
        completions. *owner* None polls every peer (shutdown drain only).

        Returns PollResult(done=..., failed=...) with transfer IDs.
        Completed handles are released automatically.
        """
        if not self._inflight:
            return _EMPTY_POLL_RESULT

        done_ids: list[int] | None = None
        failed_ids: list[int] | None = None

        for transfer_id, (peer_id, handle) in self._inflight.items():
            if owner is not None and peer_id != owner:
                continue
            try:
                state = self._agent.check_xfer_state(handle)
            except Exception as exc:
                logger.warning(
                    "NixlTransport %s: check_xfer_state failed for transfer_id=%d: %s",
                    self._local_id,
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
            handles_to_release.append(self._inflight.pop(tid)[1])
        for tid in failed_ids or ():
            handles_to_release.append(self._inflight.pop(tid)[1])
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
                self._inflight.pop(tid)[1]
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
            handle = entry[1]
            try:
                self._agent.release_xfer_handle(handle)
            except Exception as exc:
                logger.debug(
                    "NixlTransport %s: cancel pending for transfer_id=%d: %s",
                    self._local_id,
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
        if self._agent is None:
            return
        self._release_handles([handle for _, handle in self._inflight.values()])
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
                    self._local_id,
                    exc,
                )
