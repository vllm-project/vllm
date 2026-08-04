# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional, Union
import abc

# Third Party
import msgspec
import zmq

# First Party
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.transfer_channel.transfer_utils import (
    InitSideMsgBase,
    InitSideRetMsgBase,
    P2PInitSideMsg,
    P2PInitSideRetMsg,
    SideMsg,
)


class BaseTransferChannel(metaclass=abc.ABCMeta):
    ### Initialization-related functions ###
    @abc.abstractmethod
    def lazy_init_peer_connection(
        self,
        local_id: str,
        peer_id: str,
        peer_init_url: str,
        init_side_msg: Optional[InitSideMsgBase] = None,
    ) -> Optional[InitSideRetMsgBase]:
        """
        Lazily initialize the connection to a peer.

        :param local_id: The ID of itself.
        peer_id: The ID of the peer to connect to.
        peer_init_url: The URL used to initialize the connection.
        init_side_msg: An optional side message to be sent to the peer
        during initialization.

        :return: An optional side message received from the peer.
        """

        raise NotImplementedError

    @abc.abstractmethod
    async def async_lazy_init_peer_connection(
        self,
        local_id: str,
        peer_id: str,
        peer_init_url: str,
        init_side_msg: Optional[InitSideMsgBase] = None,
    ) -> Optional[InitSideRetMsgBase]:
        """
        Async version of `lazy_init_peer_connection`.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def remote_xfer_handler_exists(self, receiver_or_sender_id: str) -> bool:
        """
        Check if the remote transfer handler exists.

        :param receiver_or_sender_id: The ID of the receiver or sender.

        :return: True if the remote transfer handler exists, False otherwise.
        """
        raise NotImplementedError

    def handle_init_side_msg(
        self,
        req: InitSideMsgBase,
    ) -> InitSideRetMsgBase:
        """
        Handle side messages during initialization.

        :param req: The initialization-related side message
        received from the peer.

        :return: A side message to be sent back to the peer.
        """
        if isinstance(req, P2PInitSideMsg):
            assert hasattr(self, "peer_lookup_url"), (
                "P2PInitSideMsg requires `peer_lookup_url` attribute."
            )
            return P2PInitSideRetMsg(
                peer_lookup_url=self.peer_lookup_url,
            )
        else:
            raise ValueError(f"Unsupported InitSideMsg type: {type(req)}")

    def send_init_side_msg(
        self,
        init_tmp_socket: zmq.Socket,
        init_side_msg: InitSideMsgBase,
    ) -> InitSideRetMsgBase:
        """
        Send side messages during initialization.

        :param socket: The ZMQ socket used for sending the message.
        :param init_side_msg: The initialization-related side message
        to be sent to the peer.

        :return: A side message received from the peer.
        """
        init_msg_bytes = msgspec.msgpack.encode(init_side_msg)
        init_tmp_socket.send(init_msg_bytes)

        init_ret_msg_bytes = init_tmp_socket.recv()
        init_ret_msg = msgspec.msgpack.decode(
            init_ret_msg_bytes,
            type=SideMsg,
        )

        return init_ret_msg

    async def async_send_init_side_msg(
        self,
        init_tmp_socket: zmq.Socket,
        init_side_msg: InitSideMsgBase,
    ) -> InitSideRetMsgBase:
        """
        Async version of send_init_side_msg.
        """
        init_msg_bytes = msgspec.msgpack.encode(init_side_msg)
        await init_tmp_socket.send(init_msg_bytes)

        init_ret_msg_bytes = await init_tmp_socket.recv()
        init_ret_msg = msgspec.msgpack.decode(
            init_ret_msg_bytes,
            type=SideMsg,
        )

        return init_ret_msg

    ### Utility functions ###
    @abc.abstractmethod
    def get_local_mem_indices(
        self, objects: Union[list[bytes], list[MemoryObj]]
    ) -> list[int]:
        """
        Get the memory indices of objects.

        :param objects: A list of bytes or MemoryObj to be checked.

        :return: The memory indices of the objects.
        """
        raise NotImplementedError

    ### Send and Recv must be called in pair ###
    @abc.abstractmethod
    def batched_send(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """
        Send a batch of data through the channel.

        :param objects: A list of bytes or MemoryObj to be sent.
        :param transfer_spec: Additional specifications for the transfer.

        :return: Number of successfully transferred objects.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def batched_recv(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """
        Receive a batch of data through the channel.

        :param buffer: A list of bytes or MemoryObj to store the received data.
        :param transfer_spec: Additional specifications for the transfer.

        :return: Number of successfully transferred objects.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def async_batched_send(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """
        Async send a batch of data through the channel.

        :param objects: A list of bytes or MemoryObj to be sent.
        :param transfer_spec: Additional specifications for the transfer.

        :return: Number of successfully transferred objects.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def async_batched_recv(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """
        Async receive a batch of data through the channel.

        :param buffers: A list of bytes or MemoryObj to store the received data.
        :param transfer_spec: Additional specifications for the transfer.

        :return: Number of successfully transferred objects.
        """
        raise NotImplementedError

    ### Read and Write only need to be called on one side ###
    @abc.abstractmethod
    def batched_write(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """
        Write a batch of data through the channel.

        :param objects: A list of bytes or MemoryObj to be written.
        :param transfer_spec: Additional specifications for the transfer.

        :return: Number of successfully transferred objects.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def batched_read(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """
        Read a batch of data through the channel.

        :param buffers: A list of bytes or MemoryObj to store the read data.
        :param transfer_spec: Additional specifications for the transfer.

        :return: Number of successfully transferred objects.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def async_batched_write(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """
        Async write a batch of data through the channel.

        :param objects: A list of bytes or MemoryObj to be written.
        :param transfer_spec: Additional specifications for the transfer.

        :return: Number of successfully transferred objects.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def async_batched_read(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """
        Async read a batch of data through the channel.

        :param buffers: A list of bytes or MemoryObj to store the read data.
        :param transfer_spec: Additional specifications for the transfer.

        :return: Number of successfully transferred objects.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def close(self) -> None:
        """
        Close the transfer channel and release any resources.
        """
        raise NotImplementedError
