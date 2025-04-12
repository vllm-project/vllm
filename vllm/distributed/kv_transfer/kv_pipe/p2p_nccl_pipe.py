# SPDX-License-Identifier: Apache-2.0

import logging
import threading
import time
import typing
from collections import deque
from typing import Any, Deque, Dict, List, Optional

import msgpack
import torch
import zmq

from vllm.config import KVTransferConfig
from vllm.distributed.device_communicators.pynccl_wrapper import (
    NCCLLibrary, buffer_type, cudaStream_t, ncclComm_t, ncclDataTypeEnum)
from vllm.utils import current_stream, get_ip

logger = logging.getLogger(__name__)


class P2pNcclPipe:

    def __init__(self,
                 local_rank: int,
                 config: KVTransferConfig,
                 hostname: str = "",
                 port_offset: int = 0,
                 library_path: Optional[str] = None) -> None:
        self.config = config
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{self.local_rank}")
        self.nccl = NCCLLibrary(library_path)

        if not hostname:
            hostname = get_ip()
        port = self.config.kv_port + port_offset
        if port == 0:
            raise ValueError("Port cannot be 0")
        self._hostname = hostname
        self._port = port

        self.zmq_address = f"{self._hostname}:{self._port}"
        self.http_address = (
            f"{self._hostname}:"
            f"{self.config.kv_connector_extra_config['http_port']}")

        proxy_ip = self.config.get_from_extra_config("proxy_ip", "")
        proxy_port = self.config.get_from_extra_config("proxy_port", "")
        if proxy_ip == "" or proxy_port == "":
            self.proxy_address = ""
        else:
            self.proxy_address = proxy_ip + ":" + proxy_port

        self.context = zmq.Context()
        self.router_socket = self.context.socket(zmq.ROUTER)
        self.router_socket.bind(f"tcp://{self.zmq_address}")

        self.poller = zmq.Poller()
        self.poller.register(self.router_socket, zmq.POLLIN)

        self.send_store: Deque[List[Any]] = deque()  # tensor_id: torch.Tensor
        self.recv_store: Dict[str,
                              torch.Tensor] = {}  # tensor_id: torch.Tensor
        self.socks: Dict[str, Any] = {}  # remote_address: client socket
        self.comms: Dict[str, Any] = {}  # remote_address: (ncclComm_t, rank)

        self.buffer_size = 0
        self.buffer_size_threshold = self.config.kv_buffer_size

        self.send_store_cv = threading.Condition()
        self.recv_store_cv = threading.Condition()
        self.comm_cv = threading.Condition()

        self._listener_thread = threading.Thread(
            target=self._listen_for_requests, daemon=True)
        self._listener_thread.start()

        self._send_thread = threading.Thread(target=self._send_async,
                                             daemon=True)
        self._send_thread.start()

        self._ping_thread = None
        if port_offset == 0 and self.proxy_address != "":
            self._ping_thread = threading.Thread(target=self._ping,
                                                 daemon=True)
            self._ping_thread.start()

    def _create_connect(self, remote_address: typing.Optional[str] = None):
        assert remote_address is not None
        if remote_address not in self.socks:
            sock = self.context.socket(zmq.DEALER)
            sock.setsockopt_string(zmq.IDENTITY, self.zmq_address)
            sock.connect(f"tcp://{remote_address}")
            self.socks[remote_address] = sock
            if remote_address in self.comms:
                logger.info("comm exists, remote_address: %s, comms: %s",
                            remote_address, self.comms)
                return sock, self.comms[remote_address]

            unique_id = self.nccl.ncclGetUniqueId()
            data = {"cmd": "NEW", "unique_id": bytes(unique_id.internal)}
            sock.send(msgpack.dumps(data))

            with torch.cuda.device(self.device):
                rank = 0
                comm: ncclComm_t = self.nccl.ncclCommInitRank(
                    2, unique_id, rank)
                self.comms[remote_address] = (comm, rank)
                logger.info("ncclCommInitRank Success, %s 👉 %s, MyRank: %s",
                            self.zmq_address, remote_address, rank)

        return self.socks[remote_address], self.comms[remote_address]

    def send_tensor(
        self,
        tensor_id: str,
        tensor: torch.Tensor,
        remote_address: typing.Optional[str] = None,
    ):
        # tensor = tensor.clone()
        if remote_address is None:
            with self.recv_store_cv:
                self.recv_store[tensor_id] = tensor
                self.recv_store_cv.notify()
        else:
            self._send_sync(tensor_id, tensor, remote_address)
            # with self.send_store_cv:
            #     self.send_store.append([tensor_id, remote_address, tensor])
            #     self.send_store_cv.notify()

    def recv_tensor(
        self,
        tensor_id: str,
        remote_address: typing.Optional[str] = None,
    ) -> torch.Tensor:
        logger.info("Recv From %s, tensor_id: %s", remote_address, tensor_id)

        if remote_address is None:
            start_time = time.time()
            with self.recv_store_cv:
                if tensor_id not in self.recv_store:
                    self.recv_store_cv.wait(timeout=0.001)
                tensor = self.recv_store.pop(tensor_id, None)
            duration = time.time() - start_time
            if tensor is not None:
                self.buffer_size -= (tensor.element_size() * tensor.numel())
                logger.info(
                    "🚧🚧🚧 Recv From %s, tensor_id: %s, shape: %s, "
                    "duration: %.3fms, size: %.3fGB", remote_address,
                    tensor_id, tensor.shape, duration * 1000,
                    tensor.element_size() * tensor.numel() / 1024**3)
            else:
                logger.warning(
                    "🚧🚧🚧 Recv From %s, tensor_id: %s, "
                    "duration: %.3fms,", remote_address, tensor_id,
                    duration * 1000)
            return tensor

        if remote_address not in self.socks:
            self._create_connect(remote_address)

        sock = self.socks[remote_address]
        comm, rank = self.comms[remote_address]

        data = {"cmd": "GET", "tensor_id": tensor_id}
        sock.send(msgpack.dumps(data))

        message = sock.recv()
        data = msgpack.loads(message)
        if data["ret"] == 0:
            tensor = torch.empty(data["shape"],
                                 dtype=getattr(torch, data["dtype"]),
                                 device=self.device)
            self._recv(comm, tensor, rank ^ 1)
            return tensor

        return None

    def _listen_for_requests(self):
        while True:
            socks = dict(self.poller.poll())
            if self.router_socket in socks:
                remote_address, message = self.router_socket.recv_multipart()
                data = msgpack.loads(message)
                logger.debug("Received message from %s, data: %s",
                             remote_address.decode(), data)
                if data["cmd"] == "NEW":
                    unique_id = self.nccl.unique_id_from_bytes(
                        bytes(data["unique_id"]))
                    with torch.cuda.device(self.device):
                        rank = 1
                        comm: ncclComm_t = self.nccl.ncclCommInitRank(
                            2, unique_id, rank)
                        self.comms[remote_address.decode()] = (comm, rank)
                        logger.info(
                            "ncclCommInitRank Success, %s 👈 %s, MyRank: %s",
                            self.zmq_address, remote_address.decode(), rank)
                elif data["cmd"] == "PUT":
                    try:
                        tensor = torch.empty(data["shape"],
                                             dtype=getattr(
                                                 torch, data["dtype"]),
                                             device=self.device)

                        tensor_size = tensor.element_size() * tensor.numel()
                        if (self.buffer_size + tensor_size
                                > self.buffer_size_threshold):
                            self.router_socket.send_multipart(
                                [remote_address, b"2"])
                            logger.warning(
                                "Recv Tensor, Out Of Threshold, "
                                "%s 👈 %s, data: %s", self.zmq_address,
                                remote_address.decode(), data)
                            continue

                        self.buffer_size += tensor_size
                        self.router_socket.send_multipart(
                            [remote_address, b"0"])
                        comm, rank = self.comms[remote_address.decode()]
                        self._recv(comm, tensor, rank ^ 1)

                        tensor_id = data["tensor_id"]
                        with self.recv_store_cv:
                            self.recv_store[tensor_id] = tensor
                            self.recv_store_cv.notify()
                        logger.info(
                            "Recv Tensor, %s 👈 %s, rank: %s, data: %s, "
                            "tensor: %s", self.zmq_address,
                            remote_address.decode(), rank, data, tensor.shape)

                    except torch.cuda.OutOfMemoryError:
                        self.router_socket.send_multipart(
                            [remote_address, b"1"])
                        logger.warning(
                            "Recv Tensor, Out Of Memory, %s 👈 %s, data: %s",
                            self.zmq_address, remote_address.decode(), data)

                elif data["cmd"] == "GET":
                    tensor_id = data["tensor_id"]
                    with self.send_store_cv:
                        for item in self.send_store:
                            _tensor_id, _remote_address, tensor = item
                            if tensor_id == _tensor_id:
                                data = {
                                    "ret":
                                    0,
                                    "shape":
                                    tensor.shape,
                                    "dtype":
                                    str(tensor.dtype).replace("torch.", "")
                                }
                            else:
                                data = {"ret": 1}
                            self.router_socket.send_multipart(
                                [remote_address,
                                 msgpack.dumps(data)])
                            if data["ret"] == 0:
                                self._send(comm, tensor.to(self.device),
                                           rank ^ 1)
                            break
                else:
                    logger.info(
                        "Unexpected, Received message from %s, data: %s",
                        remote_address, data)

    def _send_async(self):
        while True:
            with self.send_store_cv:
                while not self.send_store:
                    self.send_store_cv.wait()
                tensor_id, remote_address, tensor = self.send_store.popleft()
            self._send_sync(tensor_id, tensor, remote_address)

    def _send_sync(
        self,
        tensor_id: str,
        tensor: torch.Tensor,
        remote_address: typing.Optional[str] = None,
    ):
        if remote_address is None:
            return
        if remote_address not in self.socks:
            self._create_connect(remote_address)

        sock = self.socks[remote_address]
        comm, rank = self.comms[remote_address]
        data = {
            "cmd": "PUT",
            "tensor_id": tensor_id,
            "shape": tensor.shape,
            "dtype": str(tensor.dtype).replace("torch.", "")
        }
        sock.send(msgpack.dumps(data))

        response = sock.recv()
        if response != b"0":
            # with self.send_store_cv:
            #     self.send_store.append([tensor_id, remote_address, tensor])
            #     self.send_store_cv.notify()
            logger.warning(
                "Send Tensor, Peer Out Of Memory/Threshold, %s 👉 %s, "
                "MyRank: %s, data: %s, tensor: %s, size: %fGB, response: %s",
                self.zmq_address, remote_address, rank, data, tensor.shape,
                tensor.element_size() * tensor.numel() / 1024**3,
                response.decode())
            return

        self._send(comm, tensor.to(self.device), rank ^ 1)
        logger.info("Send Tensor, %s 👉 %s, MyRank: %s, data: %s, tensor: %s",
                    self.zmq_address, remote_address, rank, data, tensor.shape)

    def _ping(self):
        sock = self.context.socket(zmq.DEALER)
        sock.setsockopt_string(zmq.IDENTITY, self.zmq_address)
        logger.info("ping start, zmq_address: %s", self.zmq_address)
        sock.connect(f"tcp://{self.proxy_address}")
        data = {
            "type": "P" if self.config.is_kv_producer else "D",
            "http_address": self.http_address,
            "zmq_address": self.zmq_address
        }
        while True:
            sock.send(msgpack.dumps(data))
            # logger.info("ping, zmq_address: %s", self.zmq_address)
            time.sleep(3)

    def _send(self, comm, tensor: torch.Tensor, dst: int, stream=None):
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}")
        if stream is None:
            stream = current_stream()

        with self.comm_cv:
            self.nccl.ncclSend(buffer_type(tensor.data_ptr()), tensor.numel(),
                               ncclDataTypeEnum.from_torch(tensor.dtype), dst,
                               comm, cudaStream_t(stream.cuda_stream))

    def _recv(self, comm, tensor: torch.Tensor, src: int, stream=None):
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}")
        if stream is None:
            stream = current_stream()

        with self.comm_cv:
            self.nccl.ncclRecv(buffer_type(tensor.data_ptr()), tensor.numel(),
                               ncclDataTypeEnum.from_torch(tensor.dtype), src,
                               comm, cudaStream_t(stream.cuda_stream))

    def close(self) -> None:
        self._listener_thread.join()
        if self._ping_thread is not None:
            self._ping_thread.join()
