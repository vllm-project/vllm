# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional, Union
import asyncio

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.transfer_channel.py_socket_channel import PySocketChannel

logger = init_logger(__name__)


# Data plane: Global shared data store for simulating data transfer
# within a single process. This is a process-local Python dict,
# NOT shared across processes or machines.
_GLOBAL_DATA_STORE: dict[str, dict[int, torch.Tensor]] = {}


class MockMemoryChannel(PySocketChannel):
    """
    Mock memory-based transfer channel for single-process testing.

    Control plane: Inherits ZMQ-based handshake from PySocketChannel.
    Data plane: Uses global dict to simulate tensor transfer.

    Only works within a single process. For production use NixlChannel.
    """

    def __init__(
        self,
        async_mode: bool = False,
        **kwargs,
    ):
        super().__init__(async_mode=async_mode, **kwargs)

    ############################################################
    # Data plane: Override hook to initialize global data store
    ############################################################
    def _on_peer_connected(self, peer_url: str):
        """Initialize data store when a peer connects"""
        if peer_url not in _GLOBAL_DATA_STORE:
            _GLOBAL_DATA_STORE[peer_url] = {}

    ############################################################
    # Data plane: Utility functions
    ############################################################
    def get_local_mem_indices(
        self, objects: Union[list[bytes], list[MemoryObj]]
    ) -> list[int]:
        local_indices = []
        if isinstance(objects[0], MemoryObj):
            for mem_obj in objects:
                assert isinstance(mem_obj, MemoryObj)
                local_indices.append(mem_obj.meta.address)
        elif isinstance(objects[0], bytes):
            raise NotImplementedError(
                "Sending raw bytes is not supported in MockMemoryChannel"
            )
        return local_indices

    ############################################################
    # Data plane: Send/Recv functions (not implemented)
    ############################################################
    def batched_send(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        raise NotImplementedError

    def batched_recv(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        raise NotImplementedError

    async def async_batched_send(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        raise NotImplementedError

    async def async_batched_recv(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        raise NotImplementedError

    ############################################################
    # Data plane: Read/Write functions (global dict simulation)
    ############################################################
    def remote_xfer_handler_exists(self, receiver_or_sender_id: str) -> bool:
        return True

    def batched_write(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        raise NotImplementedError("Sync mode not supported in MockMemoryChannel")

    def batched_read(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        raise NotImplementedError("Sync mode not supported in MockMemoryChannel")

    async def async_batched_write(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """Write tensors to global dict to simulate data transfer"""
        assert transfer_spec is not None

        receiver_id = transfer_spec["receiver_id"]
        remote_indexes = transfer_spec["remote_indexes"]

        for obj, remote_idx in zip(objects, remote_indexes, strict=True):
            if isinstance(obj, MemoryObj) and obj.tensor is not None:
                if receiver_id not in _GLOBAL_DATA_STORE:
                    _GLOBAL_DATA_STORE[receiver_id] = {}
                _GLOBAL_DATA_STORE[receiver_id][remote_idx] = obj.tensor.clone()

        await asyncio.sleep(0.001)
        return len(objects)

    async def async_batched_read(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """Read tensors from global dict to simulate data transfer"""
        assert transfer_spec is not None

        sender_id = transfer_spec["sender_id"]
        remote_indexes = transfer_spec["remote_indexes"]

        for buf, remote_idx in zip(buffers, remote_indexes, strict=True):
            if isinstance(buf, MemoryObj) and buf.tensor is not None:
                if (
                    sender_id in _GLOBAL_DATA_STORE
                    and remote_idx in _GLOBAL_DATA_STORE[sender_id]
                ):
                    buf.tensor.copy_(_GLOBAL_DATA_STORE[sender_id][remote_idx])

        await asyncio.sleep(0.001)
        return len(buffers)
