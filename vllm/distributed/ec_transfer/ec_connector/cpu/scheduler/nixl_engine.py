# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NIXL-backed TransferEngine."""

from typing import Any

from vllm.distributed.ec_transfer.ec_connector.cpu.utils import (
    ProducerPeer,
    deserialize_mem_descriptor,
)
from vllm.distributed.nixl_utils import NixlWrapper, nixl_agent_config
from vllm.logger import init_logger

logger = init_logger(__name__)
_NIXL_DRAM = "DRAM"


class NixlEngine:
    """Adapts NixlWrapper to the TransferEngine protocol."""

    def __init__(self, agent_name: str, num_threads: int = 1) -> None:
        if NixlWrapper is None or nixl_agent_config is None:
            raise RuntimeError(
                "ECCPUConnector requires NIXL; "
                "install the `nixl` package or set a different ec_connector."
            )
        self._nixl = NixlWrapper(
            agent_name,
            nixl_agent_config(num_threads=num_threads, capture_telemetry=True),
        )

    def register_region(
        self,
        block_descs: list,
        base_ptr: int,
        total_size_bytes: int,
    ) -> tuple[Any, int]:
        """Register mmap region; return (reg_descs, local_xfer_handle)."""
        reg_descs = self._nixl.get_reg_descs(
            [(base_ptr, total_size_bytes, 0, "")], _NIXL_DRAM
        )
        self._nixl.register_memory(reg_descs, backends=["UCX"])
        xfer_descs = self._nixl.get_xfer_descs(block_descs, _NIXL_DRAM)
        local_xfer_handle = self._nixl.prep_xfer_dlist("NIXL_INIT_AGENT", xfer_descs)
        return reg_descs, local_xfer_handle

    def deregister_memory(self, reg_descs: Any) -> None:
        try:
            self._nixl.deregister_memory(reg_descs)
        except Exception:
            logger.debug("ec: deregister_memory failed", exc_info=True)

    def get_agent_metadata(self) -> bytes:
        return self._nixl.get_agent_metadata()

    def add_remote_peer(self, metadata: bytes, mem_descriptor: bytes) -> ProducerPeer:
        """Register a consumer as a remote NIXL peer; return ProducerPeer."""
        agent_name = self._nixl.add_remote_agent(metadata)
        remote_blocks = deserialize_mem_descriptor(mem_descriptor)
        remote_xfer_descs = self._nixl.get_xfer_descs(remote_blocks, _NIXL_DRAM)
        remote_xfer_handle = self._nixl.prep_xfer_dlist(agent_name, remote_xfer_descs)
        return ProducerPeer(
            nixl_agent_name=agent_name,
            nixl_metadata_bytes=metadata,
            nixl_xfer_handle=remote_xfer_handle,
        )

    def add_remote_agent(self, metadata: bytes) -> str:
        """Register a remote NIXL agent; return agent_name."""
        return self._nixl.add_remote_agent(metadata)

    def remove_remote_agent(self, agent_name: str) -> None:
        try:
            self._nixl.remove_remote_agent(agent_name)
        except Exception:
            logger.warning(
                "ec: remove_remote_agent failed for %s", agent_name, exc_info=True
            )

    def post_write(
        self,
        local_xfer_handle: int,
        local_indices: list[int],
        peer: ProducerPeer,
        remote_indices: list[int],
    ) -> Any:
        if len(local_indices) != len(remote_indices):
            raise ValueError(
                f"ec: local/remote block count mismatch "
                f"({len(local_indices)} vs {len(remote_indices)})"
            )
        handle = self._nixl.make_prepped_xfer(
            "WRITE",
            local_xfer_handle,
            local_indices,
            peer.nixl_xfer_handle,
            remote_indices,
            notif_msg=b"",
        )
        self._nixl.transfer(handle)
        return handle

    def check_xfer_state(self, handle: Any) -> str:
        """Return raw NIXL state string; propagates exceptions to caller."""
        return self._nixl.check_xfer_state(handle)

    def release_xfer_handle(self, handle: Any) -> None:
        try:
            self._nixl.release_xfer_handle(handle)
        except Exception:
            logger.exception("ec: release_xfer_handle failed")
