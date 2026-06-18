# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NIXL-backed data-plane transport for ECCPUConnector."""

from typing import Any

from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.data.base import (
    DataTransport,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.utils import (
    build_block_descs,
    deserialize_mem_descriptor,
    serialize_mem_descriptor,
)
from vllm.distributed.nixl_utils import NixlWrapper, nixl_agent_config
from vllm.logger import init_logger

logger = init_logger(__name__)
_NIXL_DRAM = "DRAM"


class NixlDataTransport(DataTransport):
    """Adapts NixlWrapper to the DataTransport interface.

    Dlist handles (prepped xfer descriptor lists for remote peers) are stored
    internally keyed by agent_name. Callers never touch raw NIXL handles —
    post_read() resolves the handle from agent_name internally.

    Thread-safety: NixlWrapper is not thread-safe. In single-role pods this is
    safe by construction (producer touches it only from the router thread;
    consumer only from the scheduler thread). In ec_both pods serialize behind
    a lock before enabling ec_both.
    """

    def __init__(
        self,
        agent_name: str,
        base_ptr: int,
        num_blocks: int,
        block_size_bytes: int,
        total_size_bytes: int,
        num_threads: int = 1,
    ) -> None:
        if NixlWrapper is None or nixl_agent_config is None:
            raise RuntimeError(
                "ECCPUConnector requires NIXL; "
                "install the `nixl` package or set a different ec_connector."
            )
        self._nixl = NixlWrapper(
            agent_name,
            nixl_agent_config(num_threads=num_threads, capture_telemetry=True),
        )
        block_descs = build_block_descs(base_ptr, num_blocks, block_size_bytes)
        reg_descs = self._nixl.get_reg_descs(
            [(base_ptr, total_size_bytes, 0, "")], _NIXL_DRAM
        )
        self._nixl.register_memory(reg_descs, backends=["UCX"])
        self._reg_descs = reg_descs
        xfer_descs = self._nixl.get_xfer_descs(block_descs, _NIXL_DRAM)
        self._local_xfer_handle = self._nixl.prep_xfer_dlist(
            "NIXL_INIT_AGENT", xfer_descs
        )
        self._agent_metadata = self._nixl.get_agent_metadata()
        self._mem_descriptor_bytes = serialize_mem_descriptor(block_descs)
        # agent_name → remote dlist handle (from add_remote_peer, used by post_read)
        self._peer_handles: dict[str, int] = {}

    def get_agent_metadata(self) -> bytes:
        return self._agent_metadata

    def get_mem_descriptor(self) -> bytes:
        return self._mem_descriptor_bytes

    def add_remote_peer(self, metadata: bytes, mem_descriptor: bytes) -> str:
        """Register remote peer, prep its dlist, store handle internally.

        Returns agent_name for use in post_read() and remove_remote_peer().
        """
        agent_name = self._nixl.add_remote_agent(metadata)
        remote_blocks = deserialize_mem_descriptor(mem_descriptor)
        remote_xfer_descs = self._nixl.get_xfer_descs(remote_blocks, _NIXL_DRAM)
        self._peer_handles[agent_name] = self._nixl.prep_xfer_dlist(
            agent_name, remote_xfer_descs
        )
        return agent_name

    def remove_remote_peer(self, agent_name: str) -> None:
        self._peer_handles.pop(agent_name, None)
        try:
            self._nixl.remove_remote_agent(agent_name)
        except Exception:
            logger.warning(
                "EC: remove_remote_peer failed for %s", agent_name, exc_info=True
            )

    def post_read(
        self,
        local_indices: list[int],
        agent_name: str,
        remote_indices: list[int],
        notif_msg: bytes,
    ) -> Any:
        if len(local_indices) != len(remote_indices):
            raise ValueError(
                f"EC: local/remote block count mismatch "
                f"({len(local_indices)} vs {len(remote_indices)})"
            )
        remote_handle = self._peer_handles[agent_name]
        handle = self._nixl.make_prepped_xfer(
            "READ",
            self._local_xfer_handle,
            local_indices,
            remote_handle,
            remote_indices,
            notif_msg=notif_msg,
        )
        self._nixl.transfer(handle)
        return handle

    def get_new_notifs(self) -> dict[str, list[bytes]]:
        return self._nixl.get_new_notifs()

    def check_xfer_state(self, handle: Any) -> str:
        return self._nixl.check_xfer_state(handle)

    def release_xfer_handle(self, handle: Any) -> None:
        try:
            self._nixl.release_xfer_handle(handle)
        except Exception:
            logger.warning("EC: release_xfer_handle failed", exc_info=True)

    def deregister(self) -> None:
        try:
            self._nixl.deregister_memory(self._reg_descs)
        except Exception:
            logger.warning("EC: deregister failed", exc_info=True)
