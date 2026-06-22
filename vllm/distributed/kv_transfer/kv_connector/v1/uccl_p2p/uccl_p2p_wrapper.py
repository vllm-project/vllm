# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thin wrapper around uccl.p2p.Endpoint for vLLM KV connector use."""

import queue
import threading
import time
from collections import defaultdict
from typing import Any

import torch

from vllm.distributed.uccl_p2p_utils import Endpoint as UcclP2pEndpoint
from vllm.logger import init_logger

logger = init_logger(__name__)


class UcclP2pWrapper:
    """Wraps a single uccl.p2p.Endpoint instance and exposes a NIXL-like API.

    A single endpoint safely supports a background blocking ``recv()`` thread
    concurrent with the main thread calling ``register_memory()`` /
    ``transfer()`` / ``poll_async()`` (only fine-grained ``shared_mutex`` on
    conn/mr tables; ``poll`` is lock-free).

    The wrapper intentionally mimics the NixlWrapper interface used by the
    vLLM NIXL connector so that the higher-level connector logic can be reused
    with minimal changes.
    """

    def __init__(self, local_gpu_idx: int):
        if UcclP2pEndpoint is None:
            raise RuntimeError(
                "uccl.p2p is not available. Please install the UCCL P2P package."
            )

        self._ep = UcclP2pEndpoint(local_gpu_idx)
        self._ep.start_passive_accept()

        self._metadata = self._ep.get_metadata()

        # Agent name -> conn_id (shared by data transfers and notifications).
        self._conn_ids: dict[str, int] = {}

        # Notification receive state.
        self._notif_queue: queue.Queue[tuple[str, bytes]] = queue.Queue()
        self._notif_stop = threading.Event()
        self._notif_recv_thread = threading.Thread(
            target=self._notif_recv_loop, daemon=True, name="uccl_p2p_notif_recv"
        )
        self._notif_recv_thread.start()

        # Small pinned CPU buffers used for notifications.
        self._notif_buf_size = 256
        self._notif_send_buf = torch.zeros(
            self._notif_buf_size, dtype=torch.uint8, pin_memory=True
        )
        self._notif_recv_buf = torch.zeros(
            self._notif_buf_size, dtype=torch.uint8, pin_memory=True
        )
        send_descs = self._ep.register_memory([self._notif_send_buf])
        if not send_descs:
            raise RuntimeError("UCCL P2P notification send buffer registration failed")
        self._notif_send_mr = send_descs[0].mr_id
        recv_descs = self._ep.register_memory([self._notif_recv_buf])
        if not recv_descs:
            raise RuntimeError("UCCL P2P notification recv buffer registration failed")
        self._notif_recv_mr = recv_descs[0].mr_id

    # ------------------------------------------------------------------
    # Endpoint metadata
    # ------------------------------------------------------------------
    def get_agent_metadata(self) -> bytes:
        """Return the endpoint metadata bytes."""
        return self._metadata

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------
    def add_remote_agent(
        self,
        agent_name: str,
        endpoint_metadata: bytes,
    ) -> str:
        """Add a remote endpoint and return the agent name."""
        if agent_name in self._conn_ids:
            return agent_name

        ok, conn_id = self._ep.add_remote_endpoint(endpoint_metadata)
        if not ok:
            raise RuntimeError(f"Failed to add UCCL P2P endpoint for {agent_name}")
        self._conn_ids[agent_name] = conn_id

        return agent_name

    def remove_remote_agent(self, agent_name: str) -> None:
        conn_id = self._conn_ids.pop(agent_name, None)
        if conn_id is not None:
            self._ep.remove_remote_endpoint(conn_id)

    # ------------------------------------------------------------------
    # Memory registration
    # ------------------------------------------------------------------
    def register_memory(self, tensor_list: list[torch.Tensor]) -> list[Any]:
        return self._ep.register_memory(tensor_list)

    def deregister_memory(self, desc_list: list[Any]) -> None:
        self._ep.deregister_memory(desc_list)

    def get_serialized_descs(self, desc_list: list[Any]) -> bytes:
        return self._ep.get_serialized_descs(desc_list)

    def deserialize_descs(self, serialized: bytes) -> list[Any]:
        return self._ep.deserialize_descs(serialized)

    # ------------------------------------------------------------------
    # Transfer
    # ------------------------------------------------------------------
    def make_prepped_xfer(
        self,
        op_name: str,
        local_descs: list[Any],
        remote_descs: list[Any],
        notif_msg: bytes | None = None,
        agent_name: str | None = None,
    ) -> int:
        """Start a transfer and return a transfer_id.

        The transfer begins immediately.  Callers send any notification
        separately via ``send_notif()``.
        """
        if agent_name is None:
            raise ValueError("agent_name is required for UCCL P2P transfer")
        conn_id = self._conn_ids[agent_name]
        ok, transfer_id = self._ep.transfer(
            conn_id, op_name.lower(), local_descs, remote_descs
        )
        if not ok:
            raise RuntimeError(f"Failed to start UCCL P2P {op_name} transfer")
        return transfer_id

    def transfer(self, transfer_id: int) -> None:
        # UCCL P2P starts transfers immediately in ``transfer()``.
        pass

    def check_xfer_state(self, transfer_id: int) -> str:
        ok, is_done = self._ep.poll_async(transfer_id)
        if not ok:
            return "ERR"
        return "DONE" if is_done else "PROC"

    def release_xfer_handle(self, transfer_id: int) -> None:
        pass

    def release_dlist_handle(self, handle: Any) -> None:
        pass

    def get_xfer_telemetry(self, transfer_id: int) -> dict[str, Any]:
        return {}

    # ------------------------------------------------------------------
    # Notifications (emulated with UCCL P2P send/recv)
    # ------------------------------------------------------------------
    def send_notif(self, agent_name: str, notif_msg: bytes) -> None:
        conn_id = self._conn_ids[agent_name]
        n = min(len(notif_msg), self._notif_buf_size)
        self._notif_send_buf[:n].copy_(
            torch.frombuffer(notif_msg[:n], dtype=torch.uint8)
        )
        ok = self._ep.send(
            conn_id, self._notif_send_mr, self._notif_send_buf.data_ptr(), n
        )
        if not ok:
            raise RuntimeError(f"Failed to send UCCL P2P notification to {agent_name}")

    def get_new_notifs(self) -> dict[str, list[bytes]]:
        """Drain notification queue and return {agent_name: [messages]}."""
        result: dict[str, list[bytes]] = defaultdict(list)
        while not self._notif_queue.empty():
            try:
                agent_name, msg = self._notif_queue.get_nowait()
                result[agent_name].append(msg)
            except queue.Empty:
                break
        return dict(result)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _notif_recv_loop(self) -> None:
        while not self._notif_stop.is_set():
            received = False
            for agent_name, conn_id in list(self._conn_ids.items()):
                try:
                    ok = self._ep.recv(
                        conn_id,
                        self._notif_recv_mr,
                        self._notif_recv_buf.data_ptr(),
                        self._notif_buf_size,
                    )
                    if ok:
                        arr = self._notif_recv_buf.numpy()
                        msg = arr.tobytes().rstrip(b"\x00")
                        if msg:
                            self._notif_queue.put((agent_name, msg))
                        received = True
                except Exception as e:
                    logger.debug("UCCL P2P recv error on %s: %s", agent_name, e)
            if not received:
                time.sleep(0.001)

    def shutdown(self) -> None:
        self._notif_stop.set()
        self._notif_recv_thread.join(timeout=1.0)
