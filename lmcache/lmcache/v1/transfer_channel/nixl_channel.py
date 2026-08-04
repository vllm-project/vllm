# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union
import asyncio
import threading
import time
import uuid

# Third Party
import msgspec
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.v1.memory_management import (
    MemoryObj,
)

if TYPE_CHECKING:
    # Third Party
    from nixl._api import NixlAgent

# First Party
from lmcache.v1.rpc_utils import get_zmq_context, get_zmq_socket
from lmcache.v1.transfer_channel.abstract import BaseTransferChannel
from lmcache.v1.transfer_channel.transfer_utils import (
    InitSideMsgBase,
    InitSideRetMsgBase,
    SideMsg,
)

logger = init_logger(__name__)


class NixlMsgBase(msgspec.Struct, tag=True):
    """Base class for all nixl-related messages"""

    pass


class NixlInitRequest(NixlMsgBase):
    local_meta_bytes: bytes  # Metadata from the sender nixl agent


class NixlMemRegRequest(NixlMsgBase):
    remote_agent_name: bytes
    local_id: str
    local_xfer_dlist_bytes: bytes


class NixlInitResponse(NixlMsgBase):
    remote_agent_name: bytes
    remote_meta_bytes: bytes  # Metadata from the receiver nixl agent


class NixlMemRegResponse(NixlMsgBase):
    remote_xfer_dlist_bytes: bytes  # Serialized transfer descriptors for the receiver


NixlMsg = Union[
    NixlInitRequest, NixlInitResponse, NixlMemRegRequest, NixlMemRegResponse
]


class NixlChannel(BaseTransferChannel):
    def __init__(
        self,
        async_mode: bool = False,
        device: Optional[str] = None,
        **kwargs,
    ):
        assert "role" in kwargs
        assert "buffer_ptr" in kwargs
        assert "buffer_size" in kwargs
        assert "align_bytes" in kwargs
        assert "tp_rank" in kwargs
        assert "peer_init_url" in kwargs

        if "backends" in kwargs:
            backends = kwargs["backends"]
        else:
            backends = ["UCX"]

        self.role = kwargs["role"]

        self.nixl_wrapper = NixlAgentWrapper(
            buffer_ptr=kwargs["buffer_ptr"],
            buffer_size=kwargs["buffer_size"],
            page_size=kwargs["align_bytes"],
            tp_rank=kwargs["tp_rank"],
            backends=backends,
            device=device,
        )
        self.nixl_agent = self.nixl_wrapper.agent

        # Used for P2P
        self.peer_lookup_url = kwargs.get("peer_lookup_url", None)

        self.running = True
        self.remote_xfer_handlers_dict: dict[
            str, NixlAgent.nixl_prepped_dlist_handle
        ] = {}

        self.side_channels: list[zmq.Socket] = []
        self.running_threads: list[threading.Thread] = []

        self.async_mode = async_mode
        if self.async_mode:
            self.zmq_context = get_zmq_context(use_asyncio=True)
        else:
            self.zmq_context = get_zmq_context(use_asyncio=False)
        self.peer_init_url = kwargs["peer_init_url"]
        self.event_loop = kwargs.get("event_loop", None)

        self._init_side_channels()

    ############################################################
    # Initialization functions
    ############################################################
    def lazy_init_peer_connection(
        self,
        local_id: str,
        peer_id: str,
        peer_init_url: str,
        init_side_msg: Optional[InitSideMsgBase] = None,
    ) -> Optional[InitSideRetMsgBase]:
        # Initialize temporary socket for nixl initialization
        init_tmp_socket = get_zmq_socket(
            self.zmq_context,
            peer_init_url,
            "tcp",
            zmq.REQ,
            "connect",
        )

        # Build and send init request
        nixl_init_req = NixlInitRequest(
            local_meta_bytes=self.nixl_agent.get_agent_metadata(),
        )
        init_tmp_socket.send(msgspec.msgpack.encode(nixl_init_req))

        # Wait remote agent metadata and register remote agent
        nixl_init_resp_bytes = init_tmp_socket.recv()
        nixl_init_resp = msgspec.msgpack.decode(nixl_init_resp_bytes, type=NixlMsg)
        remote_meta_bytes = nixl_init_resp.remote_meta_bytes
        remote_agent_name = self.nixl_agent.add_remote_agent(remote_meta_bytes)

        # Register remote memory
        local_xfer_dlist_bytes = self.nixl_agent.get_serialized_descs(
            self.nixl_wrapper.xfer_descs
        )
        nixl_mem_reg_req = NixlMemRegRequest(
            remote_agent_name=nixl_init_resp.remote_agent_name,
            local_id=local_id,
            local_xfer_dlist_bytes=local_xfer_dlist_bytes,
        )
        init_tmp_socket.send(msgspec.msgpack.encode(nixl_mem_reg_req))
        nixl_mem_reg_resp_bytes = init_tmp_socket.recv()
        nixl_mem_reg_resp = msgspec.msgpack.decode(
            nixl_mem_reg_resp_bytes, type=NixlMsg
        )

        remote_xfer_dlist_bytes = nixl_mem_reg_resp.remote_xfer_dlist_bytes
        remote_xfer_dlist = self.nixl_agent.deserialize_descs(remote_xfer_dlist_bytes)
        remote_xfer_handlers = self.nixl_agent.prep_xfer_dlist(
            remote_agent_name, remote_xfer_dlist
        )
        self.remote_xfer_handlers_dict[peer_id] = remote_xfer_handlers

        # Send side message if any
        init_ret_msg: Optional[InitSideRetMsgBase] = None
        if init_side_msg is not None:
            init_ret_msg = self.send_init_side_msg(
                init_tmp_socket,
                init_side_msg,
            )

        init_tmp_socket.close()
        return init_ret_msg

    async def async_lazy_init_peer_connection(
        self,
        local_id: str,
        peer_id: str,
        peer_init_url: str,
        init_side_msg: Optional[InitSideMsgBase] = None,
    ) -> Optional[InitSideRetMsgBase]:
        # Initialize temporary socket for nixl initialization
        init_tmp_socket = get_zmq_socket(
            self.zmq_context,
            peer_init_url,
            "tcp",
            zmq.REQ,
            "connect",
        )
        # Build and send init request
        nixl_init_req = NixlInitRequest(
            local_meta_bytes=self.nixl_agent.get_agent_metadata(),
        )
        await init_tmp_socket.send(msgspec.msgpack.encode(nixl_init_req))
        # Wait remote agent metadata and register remote agent
        nixl_init_resp_bytes = await init_tmp_socket.recv()
        nixl_init_resp = msgspec.msgpack.decode(nixl_init_resp_bytes, type=NixlMsg)
        remote_meta_bytes = nixl_init_resp.remote_meta_bytes
        remote_agent_name = self.nixl_agent.add_remote_agent(remote_meta_bytes)

        # Register remote memory
        local_xfer_dlist_bytes = self.nixl_agent.get_serialized_descs(
            self.nixl_wrapper.xfer_descs
        )
        nixl_mem_reg_req = NixlMemRegRequest(
            remote_agent_name=nixl_init_resp.remote_agent_name,
            local_id=local_id,
            local_xfer_dlist_bytes=local_xfer_dlist_bytes,
        )

        await init_tmp_socket.send(msgspec.msgpack.encode(nixl_mem_reg_req))
        nixl_mem_reg_resp_bytes = await init_tmp_socket.recv()
        nixl_mem_reg_resp = msgspec.msgpack.decode(
            nixl_mem_reg_resp_bytes, type=NixlMsg
        )

        remote_xfer_dlist_bytes = nixl_mem_reg_resp.remote_xfer_dlist_bytes
        remote_xfer_dlist = self.nixl_agent.deserialize_descs(remote_xfer_dlist_bytes)
        remote_xfer_handlers = self.nixl_agent.prep_xfer_dlist(
            remote_agent_name, remote_xfer_dlist
        )
        self.remote_xfer_handlers_dict[peer_id] = remote_xfer_handlers

        # Send side message if any
        init_ret_msg: Optional[InitSideRetMsgBase] = None
        if init_side_msg is not None:
            init_ret_msg = await self.async_send_init_side_msg(
                init_tmp_socket,
                init_side_msg,
            )

        init_tmp_socket.close()
        return init_ret_msg

    def remote_xfer_handler_exists(self, receiver_or_sender_id: str) -> bool:
        return receiver_or_sender_id in self.remote_xfer_handlers_dict

    def _init_side_channels(self):
        if self.peer_init_url is None:
            return

        if self.async_mode:
            # Start listening coroutine for initialization side channel
            asyncio.run_coroutine_threadsafe(self._async_init_loop(), self.event_loop)
        else:
            # Start listening thread for initialization side channel
            self.init_thread = threading.Thread(target=self._init_loop, daemon=True)
            self.init_thread.start()
            self.running_threads.append(self.init_thread)

    def _handle_init_msg(
        self, req: Union[NixlMsg, InitSideMsgBase]
    ) -> Union[NixlMsg, InitSideRetMsgBase]:
        resp: Union[NixlMsg, InitSideRetMsgBase]
        if isinstance(req, NixlInitRequest):
            agent_name = self.nixl_agent.add_remote_agent(req.local_meta_bytes)

            resp = NixlInitResponse(
                remote_agent_name=agent_name,
                remote_meta_bytes=self.nixl_agent.get_agent_metadata(),
            )

            logger.info("Replying initialization response")

        elif isinstance(req, NixlMemRegRequest):
            local_xfer_descs = self.nixl_agent.get_serialized_descs(
                self.nixl_wrapper.xfer_descs
            )

            remote_xfer_dlist_bytes = req.local_xfer_dlist_bytes
            remote_xfer_dlist = self.nixl_agent.deserialize_descs(
                remote_xfer_dlist_bytes
            )
            remote_xfer_handlers = self.nixl_agent.prep_xfer_dlist(
                req.remote_agent_name, remote_xfer_dlist
            )
            self.remote_xfer_handlers_dict[req.local_id] = remote_xfer_handlers

            resp = NixlMemRegResponse(
                remote_xfer_dlist_bytes=local_xfer_descs,
            )

            logger.info("Replying mem register response")
        elif isinstance(req, InitSideMsgBase):
            resp = self.handle_init_side_msg(req)
            logger.info("Replying P2P init side response")
        else:
            raise ValueError(f"Unsupported InitMsg type: {type(req)}")

        return resp

    def _init_loop(self):
        # Initialize initialization side channels
        self.init_side_channel = get_zmq_socket(
            self.zmq_context,
            self.peer_init_url,
            "tcp",
            zmq.REP,
            "bind",
        )
        self.side_channels.append(self.init_side_channel)

        # NOTE: Initialization has to be two stages:
        # (1) Exchanging the metadata.
        # (2) Registering the memory descriptors.
        # Otherwise, there's a chance that nixl got stuck
        # (handle always give "PROC" status) during the first request.
        # (3) Exchanging side messages if any. This depends on the backend
        # that uses the channel.
        while self.running:
            try:
                req_bytes = self.init_side_channel.recv()

                logger.info("Received initialization request")

                req = msgspec.msgpack.decode(req_bytes, type=Union[NixlMsg, SideMsg])

                resp = self._handle_init_msg(req)

                self.init_side_channel.send(msgspec.msgpack.encode(resp))

            except Exception as e:
                logger.error("Failed to process initialization loop: %s", str(e))
                if self.running:
                    time.sleep(0.01)

    async def _async_init_loop(self):
        # Initialize initialization side channels
        self.init_side_channel = get_zmq_socket(
            self.zmq_context,
            self.peer_init_url,
            "tcp",
            zmq.REP,
            "bind",
        )
        self.side_channels.append(self.init_side_channel)
        logger.info("Starting async initialization loop")

        while self.running:
            try:
                req_bytes = await self.init_side_channel.recv()

                logger.info("Received initialization request")

                req = msgspec.msgpack.decode(req_bytes, type=Union[NixlMsg, SideMsg])

                resp = self._handle_init_msg(req)

                await self.init_side_channel.send(msgspec.msgpack.encode(resp))

            except Exception as e:
                logger.error("Failed to process initialization loop: %s", str(e))
                if self.running:
                    time.sleep(0.01)

    ############################################################
    # Utility functions
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
                "Sending raw bytes is not supported in NIXL channel"
            )
        return local_indices

    ############################################################
    # Send/Recv functions
    ############################################################

    ### Send and Recv must be called in pair ###
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
    # Read/Write functions
    ############################################################

    ### Read and Write only need to be called on one side ###
    def batched_write(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """
        Write a batch of data through the nixl channel.

        :param objects: A list of bytes or MemoryObj to be written.
        :param transfer_spec: Additional specifications for the transfer.

        :return: Number of successfully transferred objects.
        """
        assert transfer_spec is not None

        handle = self.nixl_agent.make_prepped_xfer(
            "WRITE",
            self.nixl_wrapper.xfer_handler,
            self.get_local_mem_indices(objects),
            self.remote_xfer_handlers_dict[transfer_spec["receiver_id"]],
            transfer_spec["remote_indexes"],
        )

        self.nixl_agent.transfer(handle)

        # TODO(Jiayi) tune hyperparameters
        wait_time = 0.001
        while True:
            status = self.nixl_agent.check_xfer_state(handle)
            logger.debug(f"Transfer status: {status}")

            if status == "ERR":
                logger.error("Error in send operation")
                raise RuntimeError("Failed to send objects to remote peer")
            elif status == "PROC":
                time.sleep(wait_time)  # Avoid busy waiting
                continue
            assert status == "DONE", f"Transfer status is {status}, expected DONE"
            # self._proxy_side_channel.send(notif_msg_bytes)
            break

        return len(objects)

    def batched_read(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """
        Read a batch of data from a remote peer through the nixl channel.

        This is the reverse of batched_write: the local side initiates a READ
        from the remote peer's memory into local buffers.

        :param buffers: A list of MemoryObj to store the read data.
        :param transfer_spec: Must contain 'sender_id' and 'remote_indexes'.

        :return: Number of successfully transferred objects.
        """
        assert transfer_spec is not None

        handle = self.nixl_agent.make_prepped_xfer(
            "READ",
            self.nixl_wrapper.xfer_handler,
            self.get_local_mem_indices(buffers),
            self.remote_xfer_handlers_dict[transfer_spec["sender_id"]],
            transfer_spec["remote_indexes"],
        )

        self.nixl_agent.transfer(handle)

        # Poll for completion
        wait_time = 0.001
        while True:
            status = self.nixl_agent.check_xfer_state(handle)
            logger.debug(f"Read transfer status: {status}")

            if status == "ERR":
                logger.error("Error in read operation")
                raise RuntimeError("Failed to read objects from remote peer")
            elif status == "PROC":
                time.sleep(wait_time)
                continue
            assert status == "DONE", f"Transfer status is {status}, expected DONE"
            break

        return len(buffers)

    async def async_batched_write(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """
        Write a batch of data through the channel.

        :param objects: A list of bytes or MemoryObj to be written.
        :param transfer_spec: Additional specifications for the transfer.
            Should contain 'receiver_id' and 'remote_indexes'.

        :return: Number of successfully transferred objects.
        """

        assert transfer_spec is not None

        handle = self.nixl_agent.make_prepped_xfer(
            "WRITE",
            self.nixl_wrapper.xfer_handler,
            self.get_local_mem_indices(objects),
            self.remote_xfer_handlers_dict[transfer_spec["receiver_id"]],
            transfer_spec["remote_indexes"],
        )
        self.nixl_agent.transfer(handle)

        # TODO(Jiayi) tune hyperparameters
        wait_time = 0.001
        while True:
            status = self.nixl_agent.check_xfer_state(handle)
            logger.debug(f"Transfer status: {status}")

            if status == "ERR":
                logger.error("Error in send operation")
                raise RuntimeError("Failed to send objects to remote peer")
            elif status == "PROC":
                await asyncio.sleep(wait_time)  # Avoid busy waiting
                continue
            assert status == "DONE", f"Transfer status is {status}, expected DONE"
            # self._proxy_side_channel.send(notif_msg_bytes)
            break
        return len(objects)

    async def async_batched_read(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """
        Read a batch of data through the channel.

        :param buffers: A list of bytes or MemoryObj to store the read data.
        :param transfer_spec: Additional specifications for the transfer.

        :return: True if the send operation is successful.
        """

        assert transfer_spec is not None

        handle = self.nixl_agent.make_prepped_xfer(
            "READ",
            self.nixl_wrapper.xfer_handler,
            self.get_local_mem_indices(buffers),
            self.remote_xfer_handlers_dict[transfer_spec["sender_id"]],
            transfer_spec["remote_indexes"],
        )
        self.nixl_agent.transfer(handle)

        # TODO(Jiayi) tune hyperparameters
        wait_time = 0.001
        while True:
            status = self.nixl_agent.check_xfer_state(handle)
            logger.debug(f"Transfer status: {status}")

            if status == "ERR":
                logger.error("Error in send operation")
                raise RuntimeError("Failed to send objects to remote peer")
            elif status == "PROC":
                await asyncio.sleep(wait_time)  # Avoid busy waiting
                continue
            assert status == "DONE", f"Transfer status is {status}, expected DONE"
            # self._proxy_side_channel.send(notif_msg_bytes)
            break
        return len(buffers)

    ############################################################
    # Cleanup-related functions
    ############################################################

    def close(self):
        self.running = False
        for thread in self.running_threads:
            thread.join()
        self.zmq_context.term()
        self.nixl_agent.deregister_memory(self.nixl_wrapper.reg_descs)
        self.nixl_agent.release_dlist_handle(self.nixl_wrapper.xfer_handler)

        for remote_xfer_handler in self.remote_xfer_handlers_dict.values():
            self.nixl_agent.release_dlist_handle(remote_xfer_handler)


@dataclass
class NixlAgentWrapper:
    agent: "NixlAgent"
    reg_descs: Any
    xfer_descs: Any
    xfer_handler: Any

    def __init__(
        self,
        buffer_ptr: int,
        buffer_size: int,
        page_size: int,
        tp_rank: int,
        backends: list[str],
        device: Optional[str] = None,
    ):
        """
        Initialize the NIXL agent.

        Args:
            buffer_size (int): The size of the buffer.
            buffer_ptr (int): The pointer to the buffer.
            page_size (int): The page size of NIXL and
                the lmcache memory allocator.
            tp_rank (int): The tensor parallel rank.
            backends (list[str]): The list of backends to use.

        Returns:
            NixlWrapper: The NIXL agent.
            reg_dlist: the registered memory descriptor list.
            xfer_dlist: the local transfer descriptor list.
            prepped_xfer_handler: the prepped transfer handler.
        """
        try:
            # Third Party
            from nixl._api import nixl_agent as NixlAgent
            from nixl._api import nixl_agent_config
        except ImportError as err:
            raise RuntimeError("NIXL is not available") from err

        # Handle None backends by setting default to ["UCX"]
        if backends is None:
            backends = ["UCX"]

        # Create a NIXL agent
        nixl_agent = NixlAgent(
            str(uuid.uuid4()),
            nixl_agent_config(backends=backends),
        )

        # Register the memory
        # The four fields are (base_addr, length, dev_id, meta_info)
        # https://github.com/ai-dynamo/nixl/blob/main/src/api/cpp/nixl_descriptors.h#L152
        memory_desc = [(buffer_ptr, buffer_size, tp_rank, "")]
        _VRAM_DEVICE_TYPES = {"cuda", "xpu", "hpu"}

        device_type = str(device).split(":")[0]
        if device_type == "cpu":
            mem_type = "cpu"
        elif device_type in _VRAM_DEVICE_TYPES:
            mem_type = "VRAM"
        else:
            raise ValueError(
                f"Unsupported device type '{device_type}' for NIXL. "
                f"Supported accelerators: {_VRAM_DEVICE_TYPES}"
            )

        reg_descs = nixl_agent.get_reg_descs(memory_desc, mem_type=mem_type)
        nixl_agent.register_memory(reg_descs)

        # Create xfer handlers
        xfer_desc = []
        for base_addr in range(buffer_ptr, buffer_ptr + buffer_size, page_size):
            xfer_desc.append((base_addr, page_size, tp_rank))

        xfer_descs = nixl_agent.get_xfer_descs(xfer_desc, mem_type=mem_type)
        xfer_handler = nixl_agent.prep_xfer_dlist("", xfer_descs, mem_type=mem_type)
        self.agent = nixl_agent
        self.reg_descs = reg_descs
        self.xfer_descs = xfer_descs
        self.xfer_handler = xfer_handler
