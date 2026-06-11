# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NIXL-backed transfer engine for the ECCPUConnector scheduler."""

from typing import Any

from vllm.distributed.ec_transfer.ec_connector.cpu.utils import (
    build_block_descs,
    deserialize_mem_descriptor,
    serialize_mem_descriptor,
)
from vllm.distributed.nixl_utils import NixlWrapper, nixl_agent_config
from vllm.logger import init_logger

logger = init_logger(__name__)
_NIXL_DRAM = "DRAM"


class NixlEngine:
    """Adapts NixlWrapper to the duck-typed engine the scheduler delegates use.

    Thread-safety: the underlying NixlWrapper is not thread-safe, and one
    NixlEngine instance is shared by every role on a node. In single-role pods
    this is safe by construction — a producer touches it only from the
    transport's router thread (``get_new_notifs``), a consumer only from the
    scheduler thread (``add_remote_source``/``post_read``/``check_xfer_state``/
    ``release_xfer_handle``). An ``ec_both`` pod runs both, so those two threads
    would call into the same agent concurrently. Before enabling ``ec_both``
    (e.g. for peer-to-peer transfer), serialize every method here behind a lock
    (or give each role its own agent); today nothing enforces it.
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

    def deregister_memory(self) -> None:
        try:
            self._nixl.deregister_memory(self._reg_descs)
        except Exception:
            logger.warning("EC: deregister_memory failed", exc_info=True)

    def add_remote_source(
        self, metadata: bytes, mem_descriptor: bytes
    ) -> tuple[str, int]:
        """Register a remote producer and prep a READ-source dlist over it.

        `metadata` is the producer's `get_agent_metadata()` blob carried on the
        live `XferAck`, so its rkeys reference the current producer process.
        `mem_descriptor` is the producer's msgpack-encoded block descs. Returns
        `(agent_name, remote_read_handle)`; the handle is the prepared dlist
        passed as the READ source to `post_read`.
        """
        agent_name = self._nixl.add_remote_agent(metadata)
        remote_blocks = deserialize_mem_descriptor(mem_descriptor)
        remote_xfer_descs = self._nixl.get_xfer_descs(remote_blocks, _NIXL_DRAM)
        remote_read_handle = self._nixl.prep_xfer_dlist(agent_name, remote_xfer_descs)
        return agent_name, remote_read_handle

    def remove_remote_agent(self, agent_name: str) -> None:
        try:
            self._nixl.remove_remote_agent(agent_name)
        except Exception:
            logger.warning(
                "EC: remove_remote_agent failed for %s", agent_name, exc_info=True
            )

    def post_read(
        self,
        local_indices: list[int],
        remote_read_handle: int,
        remote_indices: list[int],
        notif_msg: bytes,
    ) -> Any:
        """Issue a consumer-initiated READ: pull `remote_indices` from the
        producer's region into the consumer's `local_indices`.

        `notif_msg` is delivered to the producer (the read's remote/target)
        when the transfer completes, signalling it may unpin the source.
        """
        if len(local_indices) != len(remote_indices):
            raise ValueError(
                f"EC: local/remote block count mismatch "
                f"({len(local_indices)} vs {len(remote_indices)})"
            )
        handle = self._nixl.make_prepped_xfer(
            "READ",
            self._local_xfer_handle,
            local_indices,
            remote_read_handle,
            remote_indices,
            notif_msg=notif_msg,
        )
        self._nixl.transfer(handle)
        return handle

    def get_new_notifs(self) -> dict[str, list[bytes]]:
        """Drain completion notifications addressed to this agent.

        On the producer this reports which `mm_hash`es remote consumers have
        finished reading, so their source blocks can be unpinned.
        """
        return self._nixl.get_new_notifs()

    def check_xfer_state(self, handle: Any) -> str:
        """Return raw NIXL state string; propagates exceptions to caller."""
        return self._nixl.check_xfer_state(handle)

    def release_xfer_handle(self, handle: Any) -> None:
        try:
            self._nixl.release_xfer_handle(handle)
        except Exception:
            logger.warning("EC: release_xfer_handle failed", exc_info=True)
